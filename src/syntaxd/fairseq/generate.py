#!/usr/bin/env python3 -u

"""
Generates text with an Iterative Expansion Language Model.
"""

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data.dictionary import Dictionary

from syntaxd.fairseq.task import IterativeLmTask
from syntaxd.fairseq.generator import DECODING_STRATEGIES, add_decoding_arg
from syntaxd.data.dependency.expansion import ExpansionStrategy


FORMAT_NORMAL = 'normal'
FORMAT_ITERATIONS = 'iterations'
FORMAT_CONLL = 'conll'
FORMAT_LATEX = 'latex'


class Iteration:
    def __init__(self, plt, nlt, nle, new_token_mask):
        self.plt = plt
        self.nlt = nlt
        self.nle = nle
        self.new_token_mask = new_token_mask


def format_ascii(iteration: Iteration,
                iteration_number: int,
                token_dictionary: Dictionary,
                expansion_dictionary: Dictionary,
                expansion: ExpansionStrategy,
                no_token: str = '-') -> str:
    tokens = token_dictionary.string(iteration.nlt).split(' ')
    tokens = [t if is_new_token else no_token
              for t, is_new_token in zip(tokens, iteration.new_token_mask)]

    expansions = [expansion.pretty_format(e)
                  for e in expansion_dictionary.string(iteration.nle).split(' ')]
    expansions = [e if is_new_token else no_token
                  for e, is_new_token in zip(expansions, iteration.new_token_mask)]

    s = f'iteration {iteration_number}\n'
    s += f'  PLT: ' + token_dictionary.string(iteration.plt) + '\n'
    s += f'  NLT: ' + ' '.join(tokens) + '\n'
    s += f'  NLE: ' + ' '.join(expansions)
    return s


def maybe_tt(s: str) -> str:
    return '\\texttt{' + s + '}' if s.startswith('[') and s.endswith(']') else s


def format_latex(iteration: Iteration,
                 iteration_number: int,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion: ExpansionStrategy,
                 no_token: str = '-') -> str:

    plt = token_dictionary.string(iteration.plt).split(' ')

    nlt = token_dictionary.string(iteration.nlt).split(' ')
    nlt = [t if is_new_token else no_token
           for t, is_new_token in zip(nlt, iteration.new_token_mask)]

    nle = [expansion.pretty_format(e)
           for e in expansion_dictionary.string(iteration.nle).split(' ')]
    nle = [e if is_new_token else no_token
           for e, is_new_token in zip(nle, iteration.new_token_mask)]

    num_elems = len(nlt)

    s = '\\begin{tabularx}{\\linewidth}{p{6mm} ' + ' '.join(['c'] * num_elems) + '}\n'
    s += '\\multicolumn{' +  str(num_elems) + '}{l}{Iteration ' + str(iteration_number + 1) + '}\\\\\n'
    s += '\\hline\n'
    s += 'PLT: & ' + ' & '.join(maybe_tt(t) for t in plt) + '\\\\\n'
    s += 'NLT: & ' + ' & '.join(maybe_tt(t) for t in nlt) + '\\\\\n'
    s += 'NLE: & ' + ' & '.join(maybe_tt(e) for e in nle) + '\\\\\n'
    s += '\\end{tabularx}'
    return s


class Printer:
    def __init__(self,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion: ExpansionStrategy,
                 remove_bpe,
                 no_token: str,
                 ):
        self.token_dictionary = token_dictionary
        self.expansion_dictionary = expansion_dictionary
        self.expansion = expansion
        self.remove_bpe = remove_bpe
        self.no_token = no_token

    def print_sentence(self, sentence, iterations, heads, labels):
        sentence_str = self.token_dictionary.string(sentence,
                                                    bpe_symbol=self.remove_bpe)
        print(sentence_str)


class IterationPrinter(Printer):
    def print_sentence(self, sentence, iterations, heads, labels):
        super().print_sentence(sentence, iterations, heads)
        for k, iteration in enumerate(iterations):
            print(format_ascii(iteration,
                               k,
                               self.token_dictionary,
                               self.expansion_dictionary,
                               self.expansion,
                               self.no_token))


class LatexIterationPrinter(Printer):
    def print_sentence(self, sentence, iterations, heads, labels):
        print('|', end='')
        super().print_sentence(sentence, iterations, heads, labels)
        for k, iteration in enumerate(iterations):
            print(format_latex(iteration,
                               k,
                               self.token_dictionary,
                               self.expansion_dictionary,
                               self.expansion,
                               self.no_token))


class ConllPrinter(Printer):
    def print_sentence(self, sentence, iterations, heads, labels):
        lemma = pos = feat = ""
        for k, (token_id, head, label_id) in enumerate(zip(sentence, heads, labels)):
            word = self.token_dictionary[token_id]
            label = self.token_dictionary[label_id]
            if label.startswith('[') and label.endswith(']'):
                label = label[1:-1]
            print(f"{k + 1}\t{word}\t{lemma}\t{pos}\t{feat}\t{head + 1}\t{label}")
        print("")


PRINTER_CLASSES = {FORMAT_NORMAL: Printer,
                   FORMAT_ITERATIONS: IterationPrinter,
                   FORMAT_CONLL: ConllPrinter,
                   FORMAT_LATEX: LatexIterationPrinter,
                   }


def generate(model, generator, partial_sentences, partial_labels, partial_heads):
    num_sentences = len(partial_sentences)
    sentences = [None] * num_sentences
    labels = [None] * num_sentences
    heads = [None] * num_sentences
    iterations = [list() for _ in range(num_sentences)]

    remaining_indexes = list(range(num_sentences))

    while partial_sentences:
        plt = partial_sentences

        (unexpanded_predictions,
         unexpanded_expansions,
         partial_sentences,
         partial_labels,
         partial_heads,
         completeness,
         new_token_masks) = generator.generate(model, partial_sentences, partial_labels, partial_heads)

        nlt = unexpanded_predictions
        nle = unexpanded_expansions

        completed_indexes = [(partial_idx, final_idx)
                             for partial_idx, (final_idx, c)
                             in enumerate(zip(remaining_indexes, completeness))
                             if c]

        for partial_idx, final_idx in completed_indexes:
            sentences[final_idx] = partial_sentences[partial_idx]
            labels[final_idx] = partial_labels[partial_idx]
            heads[final_idx] = partial_heads[partial_idx]

        for partial_idx, final_idx in enumerate(remaining_indexes):
            iteration = Iteration(plt=plt[partial_idx],
                                  nlt=nlt[partial_idx],
                                  nle=nle[partial_idx],
                                  new_token_mask=new_token_masks[partial_idx])
            iterations[final_idx].append(iteration)

        partial_sentences = [s for s, c in zip(partial_sentences, completeness) if not c]
        partial_labels = [l for l, c in zip(partial_labels, completeness) if not c]
        partial_heads = [h for h, c in zip(partial_heads, completeness) if not c]
        remaining_indexes = [idx for idx, c in zip(remaining_indexes, completeness) if not c]

    return sentences, labels, heads, iterations


def init(args):
    utils.import_user_module(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task: IterativeLmTask
    task = tasks.setup_task(args)

    # Load model
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

    # Initialize generator
    decoder_class = DECODING_STRATEGIES[args.decoding]
    generator = decoder_class.create_generator(args,
                                               task.token_dictionary,
                                               task.expansion_dictionary,
                                               args.expansion,
                                               device,
                                               args.regenerate_tokens,
                                               args.temperature)
    return task, model, generator


def main(args):

    task, model, generator = init(args)

    (partial_sentences,
     partial_labels,
     partial_heads) = generator.initial_state(args.num_sentences)

    sentences, labels, heads, iterations = generate(model,
                                                    generator,
                                                    partial_sentences,
                                                    partial_labels,
                                                    partial_heads)

    printer = PRINTER_CLASSES[args.format](task.token_dictionary,
                                           task.expansion_dictionary,
                                           args.expansion,
                                           args.remove_bpe,
                                           args.no_token)

    for data in zip(sentences, labels, heads, iterations):
        sentence, sentence_labels, sentence_heads, sentence_iterations = data
        printer.print_sentence(sentence, sentence_iterations, sentence_heads, sentence_labels)


def cli_main():
    parser = options.get_generation_parser()
    add_decoding_arg(parser, '--decoding')
    parser.add_argument('--num-sentences', type=int, default=1)
    parser.add_argument('--regenerate-tokens', action='store_true')
    parser.add_argument('--format',
                        choices=[FORMAT_NORMAL, FORMAT_ITERATIONS, FORMAT_CONLL, FORMAT_LATEX],
                        default=FORMAT_NORMAL)
    parser.add_argument('--no-token', type=str, default='[pad]')
    dec_args, _ = parser.parse_known_args()
    decoder_class = DECODING_STRATEGIES[dec_args.decoding]
    decoder_class.add_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
