#!/usr/bin/env python3 -u

"""
Generates text with an Iterative Expansion Language Model.
"""

import io
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data.dictionary import Dictionary
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Tuple

from syntaxd.data.conll import conll_to_sentences
from syntaxd.fairseq.task import IterativeLmTask
from syntaxd.data.dependency.expansion import ExpansionStrategy
from syntaxd.fairseq.inference import IterativeInference
from syntaxd.data.bpe import add_bpe_arg

from syntaxd.data.dependency.trees import add_postprocess_arg, TreeBuilder

from syntaxd.data.dependency.transitions import (
    Transition,
    TransitionBuilder,
)


def tok_ids(tokens: List[str], dictionary: Dictionary) -> List[int]:
    return [dictionary.index(t) for t in tokens]


def transition2data(transition: Transition,
                    token_dictionary: Dictionary,
                    expansion_dictionary: Dictionary,
                    ) -> Tuple[List[int], List[int], List[int], List[int]]:
    prev_tokens = tok_ids(transition.previous_level_tokens, token_dictionary)
    next_tokens = tok_ids(transition.next_level_tokens, token_dictionary)
    next_level_expansions = tok_ids(transition.next_level_expansions, expansion_dictionary)
    return prev_tokens, next_tokens, transition.heads, next_level_expansions


class Probability:
    def __init__(self,
                 expansion_strategy: ExpansionStrategy,
                 expansion_dictionary: Dictionary):
        self.expansion_strategy = expansion_strategy
        self.expansion_dictionary = expansion_dictionary
        self.previous_iteration_lprobs = [0.]
        self.previous_iteration_total_lprob = 0.
        self.expansion_lprob = 0.

    def _expand(self, expansion_token: str, prob: float):
        left_deps, right_deps = self.expansion_strategy.expand_deps(expansion_token)
        return [0.] * len(left_deps) + [prob] + [0.] * len(right_deps)

    def collect_iteration(self,
                          token_lprobs: torch.Tensor,
                          expansion_lprobs: torch.Tensor,
                          transition: Transition):
        token_lprobs = token_lprobs.tolist()
        expansion_lprobs = expansion_lprobs.tolist()

        previous_lprobs = self.previous_iteration_lprobs
        token_lprobs = [lprob if included_in_loss == 1 else previous_lprob
                        for lprob, previous_lprob, included_in_loss
                        in zip(token_lprobs, previous_lprobs, transition.loss_mask)]

        self.expansion_lprob += sum(lprob
                                    for lprob, included_in_loss
                                    in zip(expansion_lprobs, transition.loss_mask)
                                    if included_in_loss == 1)

        if self.expansion_lprob == float('-inf'):
            raise ValueError("Expansion log probability was -inf. This should never happen")

        self.previous_iteration_total_lprob = sum(token_lprobs)
        expansions = transition.next_level_expansions
        self.previous_iteration_lprobs = [p
                                          for e, tp in zip(expansions, token_lprobs)
                                          for p in self._expand(e, tp)]

    def compute_logprob(self, joint=False) -> float:
        if joint:
            return self.expansion_lprob + self.previous_iteration_total_lprob
        else:
            return self.previous_iteration_total_lprob


class Scorer:
    def __init__(self,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion_strategy: ExpansionStrategy,
                 segmenter: Callable[[List[str]], List[List[str]]] = None,
                 postprocess: str = None,
                 device=None,
                 joint_prob=False):

        self.token_dictionary = token_dictionary
        self.expansion_dictionary = expansion_dictionary
        self.expansion_strategy = expansion_strategy
        self.inference = IterativeInference(token_dictionary,
                                            expansion_dictionary,
                                            expansion_strategy,
                                            device=device,
                                            mask_unk=False)
        self.tree_builder = TreeBuilder(segmenter, postprocess)
        self.transition_builder = TransitionBuilder(expansion_strategy,
                                                    self.tree_builder)
        self.joint_prob = joint_prob

    def to(self, device):
        self.inference = self.inference.to(device)
        return self

    def score(self,
              model: nn.Module,
              conll_sentence: List[List[str]]) -> float:

        probability_class = Probability
        probability_calc = probability_class(self.expansion_strategy, self.expansion_dictionary)
        transitions = self.transition_builder.transitions_from_sentence(conll_sentence)
        device = self.inference.device
        for transition in transitions:
            (tokens,
             next_level_tokens,
             heads,
             next_level_expansions) = transition2data(transition,
                                                      self.token_dictionary,
                                                      self.expansion_dictionary)

            (token_logits,
             expansion_logits,
             expansion_ids) = self.inference.inference_step(model,
                                                            [tokens],
                                                            [heads],
                                                            given_next_level_expansions=[next_level_expansions])

            token_logits = token_logits + self.inference.token_prob_mask
            token_lprobs = F.log_softmax(token_logits, dim=-1)
            token_indexes = torch.LongTensor(next_level_tokens).unsqueeze(1).to(device)
            token_lprobs = torch.gather(token_lprobs.squeeze(0),
                                        dim=1,
                                        index=token_indexes).squeeze(1)

            if token_lprobs.sum().item() == float('-inf'):
                # Uncomment for debugging:
                # offending_possition = token_lprobs.cpu().tolist().index(float('-inf'))
                # offending_token_index = token_indexes.cpu().tolist()[offending_possition]
                raise ValueError("Token log probability was -inf. This should never happen")

            expansion_logits = expansion_logits + self.inference.expansion_prob_mask
            expansion_lprobs = F.log_softmax(expansion_logits, dim=-1)
            expansion_indexes = torch.LongTensor(next_level_expansions).unsqueeze(1).to(device)
            expansion_lprobs = torch.gather(expansion_lprobs.squeeze(0),
                                            dim=1,
                                            index=expansion_indexes).squeeze(1)

            probability_calc.collect_iteration(token_lprobs, expansion_lprobs, transition)

        final_logprob = probability_calc.compute_logprob(joint=self.joint_prob)
        return final_logprob


def main(args):
    utils.import_user_module(args)

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task: IterativeLmTask
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path), file=sys.stderr)
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [args.path],
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    assert len(models) == 1
    model = models[0]
    model.eval()

    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    device = next(model.parameters()).device

    scorer = Scorer(task.token_dictionary,
                    task.expansion_dictionary,
                    args.expansion,
                    segmenter=args.bpe_codes,
                    postprocess=args.postprocess,
                    joint_prob=args.joint).to(device)

    conll_sentences = conll_to_sentences(input_lines)

    compute_corpus_perplexity = args.score == 'ppl'
    total_lprob = 0.
    total_num_tokens = 0
    for conll_sentence in conll_sentences:
        total_num_tokens += len(conll_sentence)
        lprob = scorer.score(model, conll_sentence)

        if compute_corpus_perplexity:
            total_lprob += lprob
        else:
            score_value = (lprob if args.score == 'lprob'
                           else math.exp(lprob) if args.score == 'prob'
                           else f'Score {args.score} not implemented')
            print(score_value)

    if compute_corpus_perplexity:
        log2_prob = total_lprob / math.log(2)
        ppl = math.pow(2., - log2_prob/total_num_tokens)
        print(ppl)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument("--score", default="ppl", choices=["lprob", "prob", "ppl"])
    parser.add_argument("--joint", action='store_true')
    add_postprocess_arg(parser, '--postprocess')
    add_bpe_arg(parser, '--bpe-codes')
    args = options.parse_args_and_arch(parser)
    # --expansion is added by the task, so we don't add it again here
    main(args)


if __name__ == '__main__':
    cli_main()
