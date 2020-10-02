from argparse import ArgumentParser
from fairseq.data.dictionary import Dictionary
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from syntaxd.data.dependency.transitions import expand_heads
from syntaxd.data.dependency.expansion import ExpansionStrategy
from syntaxd.fairseq.inference import IterativeInference
from syntaxd.fairseq.modules import sampling


class IterativeGenerator:
    def __init__(self,
                 args,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion_strategy: ExpansionStrategy,
                 device: Optional[Union[torch.device, str]] = None,
                 regenerate_tokens: bool = False,
                 temperature: float = 1.0):
        self.temperature = temperature

        self.inference = IterativeInference(token_dictionary,
                                            expansion_dictionary,
                                            expansion_strategy,
                                            device,
                                            mask_unk=True)
        self.dependency_placeholder_ids = {token_dictionary.index(t)
                                           for t in expansion_strategy.get_dependency_placeholders()}

        def expand(e):
            left_deps, right_deps = expansion_strategy.expand_deps(e)
            left_dep_idxs = [token_dictionary.index(t) for t in left_deps]
            right_dep_idxs = [token_dictionary.index(t) for t in right_deps]
            return left_dep_idxs, right_dep_idxs

        self.expansions = {expansion_dictionary.index(e): expand(e)
                           for e in expansion_dictionary.symbols}
        self.regenerate_tokens = regenerate_tokens

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def create_generator(cls,
                         args,
                         token_dictionary: Dictionary,
                         expansion_dictionary: Dictionary,
                         expansion_strategy: ExpansionStrategy,
                         device: Optional[Union[torch.device, str]] = None,
                         regenerate_tokens: bool = False,
                         temperature: float = 1.0):
        return cls(args,
                   token_dictionary,
                   expansion_dictionary,
                   expansion_strategy,
                   device,
                   regenerate_tokens,
                   temperature)

    def to(self, device: Union[torch.device, str]):
        self.inference = self.inference.to(device)
        return self

    def initial_state(self, num_sentences: int) -> Tuple[List[List[int]], List[List[int]]]:
        initial_tokens, initial_heads = self.inference.initial_state(num_sentences)
        initial_labels = initial_tokens
        return initial_tokens, initial_labels, initial_heads

    def sample_token_ids(self, token_logits):
        raise NotImplementedError

    def sample_expansion_ids(self, expansion_logits):
        raise NotImplementedError

    def expand(self,
               previous_level_tokens,
               previous_level_labels,
               previous_level_heads,
               token_ids,
               expansion_ids):
        """
        :param previous_level_tokens:
        :param previous_level_labels
        :param previous_level_heads:
        :param token_ids:
        :param expansion_ids:
        :return:
        """
        bsz = token_ids.size(0)
        expanded_sentences = []
        expanded_labels = []
        expanded_heads = []
        completeness = []
        generated_token_masks = []
        for sentence_id in range(bsz):
            # First, expand tokens:
            sentence = []
            sentence_labels = []
            generated_token_mask = []
            old_idx2new_idx = {}
            new_idx2old_idx = {}
            new_heads = {}
            is_complete = True
            sentence_length = len(previous_level_tokens[sentence_id])
            # print("******** " + self.expansion_dictionary.string(expansion_ids[sentence_id, :]))
            for old_idx in range(sentence_length):
                prev_t = previous_level_tokens[sentence_id][old_idx]
                is_terminal_token = prev_t not in self.dependency_placeholder_ids
                t: Optional[int]
                t = token_ids[sentence_id, old_idx]
                use_previous_level_token = is_terminal_token and not self.regenerate_tokens
                t = (prev_t if use_previous_level_token
                     else t if t != self.inference.pad_idx
                     else None)

                assert t is not None, "Predicted token shall never be <pad>"

                label_id = previous_level_labels[sentence_id][old_idx]

                generated_token_mask.append(not use_previous_level_token)

                initial_idx = len(sentence)

                expansion_id = expansion_ids[sentence_id, old_idx]
                left_expansion, right_expansion = ([], []) if is_terminal_token else self.expansions[int(expansion_id)]

                if len(left_expansion) > 0 or len(right_expansion) > 0:
                    is_complete = False

                sentence.extend(left_expansion)
                sentence_labels.extend(left_expansion)

                new_idx = len(sentence)
                sentence.append(t)
                sentence_labels.append(label_id)

                new_idx2old_idx[new_idx] = old_idx
                old_idx2new_idx[old_idx] = new_idx

                sentence.extend(right_expansion)
                sentence_labels.extend(right_expansion)

                expansion_size = len(left_expansion) + len(right_expansion) + 1
                dep_idxs = set(range(initial_idx, initial_idx + expansion_size)) - {new_idx}
                for dep_idx in dep_idxs:
                    new_heads[dep_idx] = new_idx

            expanded_sentences.append(sentence)
            expanded_labels.append(sentence_labels)
            generated_token_masks.append(generated_token_mask)
            completeness.append(is_complete)

            next_level_heads = expand_heads(len(sentence),
                                            new_idx2old_idx,
                                            old_idx2new_idx,
                                            previous_level_heads[sentence_id],
                                            new_heads)

            expanded_heads.append(next_level_heads)

        return expanded_sentences, expanded_labels, expanded_heads, completeness, generated_token_masks

    def generate(self,
                 model: nn.Module,
                 previous_level_tokens: List[List[int]],
                 previous_level_labels: List[List[int]],
                 previous_level_heads: List[List[int]],
                 ) -> Tuple[List[List[int]],
                            List[List[int]],
                            List[List[int]],
                            List[bool],
                            List[List[int]]]:
        """

        :param model:
        :param previous_level_tokens:
        :param previous_level_labels:
        :param previous_level_heads:
        :return: List of tuples of:
                   - Unexpanded sentence (list of token IDs).
                   - Expanded sentence (list of token IDs).
                   - Sentence heads.
                   - Completeness flag (true if sentence is complete, false otherwise)
                   - Expansion IDs
        """

        (token_logits,
         expansion_logits,
         expansion_ids) = self.inference.inference_step(model,
                                                        previous_level_tokens,
                                                        previous_level_heads,
                                                        expansion_sampling=self.sample_expansion_ids)

        token_ids = self.sample_token_ids(token_logits).cpu()
        expansion_ids = expansion_ids.cpu()
        unexpanded_sentences = token_ids.tolist()
        unexpanded_expansions = expansion_ids.tolist()

        (expanded_sentences,
         expanded_labels,
         expanded_heads,
         completeness,
         generated_token_masks) = self.expand(previous_level_tokens,
                                              previous_level_labels,
                                              previous_level_heads,
                                              token_ids,
                                              expansion_ids)

        return (unexpanded_sentences,
                unexpanded_expansions,
                expanded_sentences,
                expanded_labels,
                expanded_heads,
                completeness,
                generated_token_masks)


class GreedyGenerator(IterativeGenerator):

    def sample_greedily(self, logits, mask):
        return torch.argmax(logits + mask, dim=-1)

    def sample_expansion_ids(self, expansion_logits):
        return self.sample_greedily(expansion_logits, self.inference.expansion_prob_mask)

    def sample_token_ids(self, token_logits):
        return self.sample_greedily(token_logits, self.inference.token_prob_mask)


class TopkGenerator(IterativeGenerator):

    def sample_topk(self, logits, mask, k, temp):
        bsz, seq_len, _ = logits.shape
        logits = (logits.view(bsz * seq_len, -1) / temp) + mask
        logits = sampling.filter_top_k_(logits, k)
        probabilities = F.softmax(logits, dim=-1)
        ids = torch.multinomial(probabilities, 1)
        return ids.view(bsz, seq_len)

    def sample_expansion_ids(self, expansion_logits):
        k = int(len(self.inference.expansion_dictionary) * .5)
        return self.sample_topk(expansion_logits,
                                self.inference.expansion_prob_mask,
                                k=k,
                                temp=self.temperature)

    def sample_token_ids(self, token_logits):
        return self.sample_topk(token_logits,
                                self.inference.token_prob_mask,
                                k=40,
                                temp=self.temperature)


class NucleusGenerator(IterativeGenerator):

    def sample_nucleus(self, logits, mask, p, temp):
        bsz, seq_len, _ = logits.shape
        logits = (logits.view(bsz * seq_len, -1) / temp) + mask
        logits = sampling.filter_top_p_(logits, p)
        probs = F.softmax(logits, dim=-1)
        ids = torch.multinomial(probs, 1)
        return ids.view(bsz, seq_len)

    def sample_expansion_ids(self, expansion_logits):
        return self.sample_nucleus(expansion_logits,
                                   self.inference.expansion_prob_mask,
                                   p=.9,
                                   temp=self.temperature)

    def sample_token_ids(self, token_logits):
        return self.sample_nucleus(token_logits,
                                   self.inference.token_prob_mask,
                                   p=.9,
                                   temp=self.temperature)


class AdjGenerator(NucleusGenerator):
    def __init__(self,
                 args,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion_strategy: ExpansionStrategy,
                 device: Optional[Union[torch.device, str]] = None,
                 regenerate_tokens: bool = False,
                 temperature: float = 1.0):
        super().__init__(args,
                         token_dictionary,
                         expansion_dictionary,
                         expansion_strategy,
                         device,
                         regenerate_tokens,
                         temperature)
        adj_multiplier = args.adj_multi
        mult = [adj_multiplier if "amod" in symbol else 1.0
                for symbol in expansion_dictionary.symbols]
        self.mult = torch.FloatTensor(mult).to(device)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--adj-multi", type=float, default=1.0)

    def sample_expansion_ids(self, expansion_logits):
        logits = expansion_logits
        p = 0.9
        mask = self.inference.expansion_prob_mask
        temperature = self.temperature
        bsz, seq_len, _ = logits.shape
        logits = (logits.view(bsz * seq_len, -1) / temperature) + mask
        logits = sampling.filter_top_p_(logits, p)
        probs = F.softmax(logits, dim=-1)
        probs = torch.mul(probs, self.mult)
        probs = probs / probs.sum()
        ids = torch.multinomial(probs, 1)
        return ids.view(bsz, seq_len)


DEC_GREEDY = 'greedy'
DEC_TOPK = 'topk'
DEC_NUCLEUS = 'nucleus'
DEC_NUCLEUS_ADJ = 'nucleus_adj'


DECODING_STRATEGIES = {
    DEC_GREEDY: GreedyGenerator,
    DEC_TOPK: TopkGenerator,
    DEC_NUCLEUS: NucleusGenerator,
    DEC_NUCLEUS_ADJ: AdjGenerator,
}


def add_decoding_arg(parser: ArgumentParser, argument: str = '--decoding'):
    parser.add_argument(argument,
                        choices=[DEC_GREEDY, DEC_TOPK, DEC_NUCLEUS, DEC_NUCLEUS_ADJ],
                        default=DEC_GREEDY)
