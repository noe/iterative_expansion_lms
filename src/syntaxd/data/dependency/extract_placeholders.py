import argparse
import io
import sys


def main():
    desc = 'Extracts the dependency placeholders from a CONLL file'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--dependency-field', type=int, default=6)
    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    dep_labels = set()

    for k, line in enumerate(input_lines):
        line = line.strip()

        if not line:
            continue
        dep_label = line.split('\t')[args.dependency_field]
        if dep_label == "":
            raise ValueError("Empty dependency label at line {}".format(k + 1))

        dep_labels.add(dep_label)

    dep_labels = list(sorted(dep_labels))
    for dep_label in dep_labels:
        print(dep_label)


if __name__ == '__main__':
    main()