import argparse
import io
from nltk.translate.bleu_score import corpus_bleu
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    weights = [1. / args.order] * args.order

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    hypotheses = [line.strip().split(' ') for line in input_lines]

    references = [hypotheses[:k] + hypotheses[k + 1:]
                  for k in range(len(hypotheses))]

    bleu = corpus_bleu(references, hypotheses, weights=weights)
    print(bleu)


if __name__ == '__main__':
    main()
