import argparse
import io
import sys
from syntaxd.data.conll import conll_to_sentences
from seqp.vocab import Vocabulary


def main():
    desc = 'Converts a CONLL file to plain text'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--word-field', type=int, default=1)
    parser.add_argument('--vocab', type=str, required=True)
    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    with open(args.vocab, encoding='utf-8') as f:
        vocab = Vocabulary.from_json(f.read())

    unk_token = vocab.idx2symbol[vocab.unk_id]

    def word_or_unk(w: str):
        return w if w in vocab.idx2symbol else unk_token

    for conll_sentence in conll_to_sentences(input_lines):
        sentence_tokens = [word_or_unk(word_fields[args.word_field])
                           for word_fields in conll_sentence]
        print(' '.join(sentence_tokens))


if __name__ == '__main__':
    main()
