import argparse
import io
from itertools import repeat
import re
import spacy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument("--unks-file", type=str)
    parser.add_argument("--unks-dest", type=str)
    args = parser.parse_args()

    if args.unks_file is None:
        unk_input_lines = repeat(None)

        def handle_unks(doc, unk_line):
            pass
    else:
        assert args.unks_dest is not None, \
            "A destination file for the <unk> sents is needed"
        unks_dest = open(args.unks_dest, encoding='utf-8', mode='w')
        unk_input_lines = open(args.unks_file, encoding='utf-8')

        def handle_unks(doc, unk_line: str):
            remaining_unk_tokens = re.sub('\s+', ' ', unk_line).strip().split(' ')

            for sent in doc.sents:
                num_words = len(list(sent))
                unk_sent = ' '.join(remaining_unk_tokens[:num_words])
                remaining_unk_tokens = remaining_unk_tokens[num_words:]
                print(unk_sent, file=unks_dest)

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    nlp = spacy.load('en_core_web_sm')
    sentencizer = nlp.create_pipe("sentencizer")

    for line, unk_line in zip(input_lines, unk_input_lines):
        line = line.strip()
        if not line:
            continue
        tokens = re.sub('\s+', ' ', line).split(' ')
        try:
          doc = nlp.tokenizer.tokens_from_list(tokens)
        except:
          import pdb; pdb.set_trace()
        sentencizer(doc)
        for sentence in doc.sents:
            print(sentence.text)

        handle_unks(doc, unk_line)


if __name__ == '__main__':
    main()
