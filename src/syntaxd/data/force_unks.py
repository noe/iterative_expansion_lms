import argparse
import io
import sys
from typing import List

from syntaxd.data.conll import conll_to_sentences

UNK = '<unk>'


def mask(token: str,
         conll_fields: List[str],
         word_field_idx: int) -> List[str]:
    if token == UNK:
        conll_fields[word_field_idx] = UNK
    return conll_fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument("--word-field", type=int, default=1)
    parser.add_argument("unks_file", type=str)
    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    conll_sentences = conll_to_sentences(input_lines)

    with open(args.unks_file, encoding='utf-8') as unks_file:
        for line, conll_sentence in zip(unks_file, conll_sentences):
            tokens = line.strip().split(' ')
            masked_conll_sentence = [mask(t, fields, args.word_field)
                                     for t, fields in zip(tokens, conll_sentence)]
            for word_fields in masked_conll_sentence:
                print('\t'.join(word_fields))
            print('')

if __name__ == '__main__':
    main()
