import argparse
import io
import sys

from seqp.vocab import VocabularyCollector
from syntaxd.data.dependency.expansion import add_expansion_arg


def main():
    parser = argparse.ArgumentParser('Converts a textual vocabulary file into json format')
    parser.add_argument('--input', type=str, required=False)
    add_expansion_arg(parser)

    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    expansion = args.expansion
    vocab_collector = VocabularyCollector()

    # Sort placeholders (which are given to us in a set) to ensure we
    # keep the same order in different invocations.
    dep_placeholders = sorted(expansion.get_dependency_placeholders())

    for placeholder in dep_placeholders:
        vocab_collector.add_symbol(placeholder)

    for line in input_lines:
        symbol = line.strip()
        if symbol == '<eos>' or symbol == '<unk>':
            continue
        vocab_collector.add_symbol(symbol)

    vocab = vocab_collector.consolidate()
    vocab_as_str = vocab.to_json(indent=4)
    print(vocab_as_str)


if __name__ == '__main__':
    main()
