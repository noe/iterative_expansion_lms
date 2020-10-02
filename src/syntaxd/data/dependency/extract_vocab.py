import argparse
import io
import sys

from syntaxd.data.dependency.transitions import Transition
from syntaxd.data.dependency.expansion import add_expansion_arg, ExpansionStrategy

from seqp.vocab import VocabularyCollector, DEFAULT_UNK

BANNED_SYMBOLS = [DEFAULT_UNK]


def main():
    desc = 'Extracts the vocabulary from the textual transitions file'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--token-vocab', type=str, required=True)
    parser.add_argument('--expansion-vocab', type=str, required=True)
    parser.add_argument('--token-vocab-size', type=int, required=False)
    parser.add_argument('--expansion-vocab-size', type=int, required=False)
    add_expansion_arg(parser)
    args = parser.parse_args()

    expansion_strategy: ExpansionStrategy = args.expansion

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    token_vocab_collector = VocabularyCollector()
    expansion_vocab_collector = VocabularyCollector()

    for line in input_lines:
        line = line.strip()
        transition = Transition.from_str(line)
        for token_list in transition.previous_level_tokens, transition.next_level_tokens:
            for token in token_list:
                if token not in BANNED_SYMBOLS:
                    token_vocab_collector.add_symbol(token)

        for expansion in transition.next_level_expansions:
            if expansion == expansion_strategy.null_expansion_token():
                continue
            expansion_vocab_collector.add_symbol(expansion)

    token_vocab = token_vocab_collector.consolidate(args.token_vocab_size)
    expansion_vocab = expansion_vocab_collector.consolidate(args.expansion_vocab_size)

    with open(args.token_vocab, 'w', encoding='utf-8') as token_vocab_file:
        token_vocab_file.write(token_vocab.to_json(indent=4))

    with open(args.expansion_vocab, 'w', encoding='utf-8') as expansion_vocab_file:
        expansion_vocab_file.write(expansion_vocab.to_json(indent=4))


if __name__ == '__main__':
    main()
