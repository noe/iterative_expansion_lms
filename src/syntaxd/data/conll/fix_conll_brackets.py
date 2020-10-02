import argparse
import io
import sys
from syntaxd.data.conll import conll_to_sentences


BRACKETS = {'-LRB-': '(',
            '-RRB-': ')',
            '-LCB-': '{',
            '-RCB-': '}',
            '-LSB-': '[',
            '-RSB-': ']',
            }

def main():
    desc = 'Fix bracket replacements from CONLL sentences (i.e. the -LRB- thing)'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--word-field', type=int, default=1)
    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    for conll_sentence in conll_to_sentences(input_lines):

        has_spaces = any(' ' in word_fields[args.word_field]
                         for word_fields in conll_sentence)
        if has_spaces:
            continue

        for word_fields in conll_sentence:
            bracket = BRACKETS.get(word_fields[args.word_field], None)
            if bracket is not None:
                word_fields[args.word_field] = bracket
            print('\t'.join(word_fields))

        print('')


if __name__ == '__main__':
    main()
