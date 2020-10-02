import argparse
from typing import Optional
from unicodedata import category as uc_cat

from syntaxd.data.dependency.trees import (
    add_postprocess_arg,
    TreeBuilder,
    iterate_leaves,
    Node,
)
from syntaxd.data.conll import conll_to_sentences


PUNCTUATION_CATS = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}


def is_punct(s: str):
    # taken from https://github.com/nltk/nltk/blob/develop/nltk/parse/evaluate.py#L85
    s_without_punctuation = "".join(c for c in s if uc_cat(c) not in PUNCTUATION_CATS)
    return s_without_punctuation == ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--system", type=str, required=True)
    add_postprocess_arg(parser, "--postprocess")
    args = parser.parse_args()

    tree_builder = TreeBuilder(segmenter=None, postprocess=args.postprocess)

    num_correct = 0
    num_correct_labels = 0
    total = 0

    with open(args.gold, encoding='utf-8') as gold_file, \
            open(args.system, encoding='utf-8') as system_file:
        gold_sentences = conll_to_sentences(gold_file)
        system_sentences = conll_to_sentences(system_file)
        for k, (gold_sentence, system_sentence) in enumerate(zip(gold_sentences, system_sentences)):
            gold_words = []
            gold_labels = []
            gold_heads = []

            def capture_gold_data(node: Node, head: Optional[Node]):
                gold_words.append(node.token)
                gold_labels.append(node.dep_label)
                head_idx = head.word_idx if head is not None else 0
                gold_heads.append(head_idx)

            postprocessed_gold_tree = tree_builder.tree_from_sentence(gold_sentence)
            iterate_leaves(postprocessed_gold_tree, capture_gold_data)

            system_words = [word_fields[tree_builder.word_field_idx]
                            for word_fields in system_sentence]
            system_labels = [word_fields[tree_builder.dep_label_field_idx]
                             for word_fields in system_sentence]
            system_heads = [int(word_fields[tree_builder.head_field_idx])
                            for word_fields in system_sentence]

            assert system_words == gold_words, f"System and gold Words differ at line {k+1}"

            for word_idx in range(len(system_words)):
                if is_punct(gold_words[word_idx]):
                    continue

                total += 1
                if system_heads[word_idx] == gold_heads[word_idx]:
                    num_correct += 1
                    if system_labels[word_idx] == gold_labels[word_idx]:
                        num_correct_labels += 1

    labeled_attachment_score = 100 * num_correct_labels / total
    unlabeled_attachment_score = 100 * num_correct / total
    print(f"LAS: {labeled_attachment_score}\tUAS: {unlabeled_attachment_score}")


if __name__ == '__main__':
    main()
