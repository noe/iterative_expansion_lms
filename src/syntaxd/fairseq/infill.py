from fairseq import options
import io
import sys

from syntaxd.data.bpe import add_bpe_arg
from syntaxd.data.conll import conll_to_sentences
from syntaxd.data.dependency.expansion import add_expansion_arg
from syntaxd.data.dependency.trees import add_postprocess_arg, TreeBuilder
from syntaxd.data.dependency.transitions import TransitionBuilder
from syntaxd.fairseq.generator import add_decoding_arg
from syntaxd.fairseq.generate import (
    init,
    generate,
    FORMAT_NORMAL,
    FORMAT_CONLL,
    FORMAT_ITERATIONS,
    PRINTER_CLASSES,
)


def create_initial_states(conll_sentences, transition_builder):
    for conll_sentence in conll_sentences:
        transitions = list(transition_builder.transitions_from_sentence(conll_sentence))
        transitions[0].



def main():
    parser = options.get_generation_parser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument('--regenerate-tokens', action='store_true')
    parser.add_argument('--format',
                        choices=[FORMAT_NORMAL, FORMAT_ITERATIONS, FORMAT_CONLL],
                        default=FORMAT_NORMAL)
    parser.add_argument('--no-token', type=str, default='[pad]')
    add_decoding_arg(parser, '--decoding')
    add_postprocess_arg(parser, '--postprocess')
    add_bpe_arg(parser, '--bpe-codes')
    add_expansion_arg()
    args = options.parse_args_and_arch(parser)

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    tree_builder = TreeBuilder(args.bpe_codes, args.postprocess)
    transition_builder = TransitionBuilder(args.expansion, tree_builder)

    task, model, generator = init(args)

    conll_sentences = conll_to_sentences(input_lines)

    (partial_sentences,
     partial_labels,
     partial_heads) = create_initial_states(conll_sentences, transition_builder)

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



if __name__ == '__main__':
    main()
