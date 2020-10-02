from typing import List, Callable, Iterable, Tuple

from .trees import (
    TreeBuilder,
    add_postprocess_arg,
    PROCESS_NONE,
)

from ..bpe import add_bpe_arg
from .transitions import Transition, TransitionBuilder
from .expansion import ExpansionStrategy, add_expansion_arg

from ..conll import conll_to_sentences


class DataProcessor:
    def __init__(self,
                 expansion_strategy: ExpansionStrategy,
                 segmenter: Callable[[List[str]], List[List[str]]] = None,
                 postprocess: str = PROCESS_NONE):

        tree_builder = TreeBuilder(segmenter=segmenter,
                                   postprocess=postprocess)

        self.builder = TransitionBuilder(expansion_strategy, tree_builder)

    def process_sentences(self, conll_lines) -> Iterable[Tuple[int, int, Transition]]:
        for sent_idx, fieldlist_seq in enumerate(conll_to_sentences(conll_lines)):
            for level_idx, transition in enumerate(self._process_sentence(fieldlist_seq)):
                yield sent_idx, level_idx, transition

    def _process_sentence(self, fieldlist_seq: List[List[str]]) -> Iterable[Transition]:
        return self.builder.transitions_from_sentence(fieldlist_seq)


def main():
    import argparse
    import io
    import sys
    parser = argparse.ArgumentParser("CONLL to transitions converter")
    parser.add_argument("--input", required=False, type=str)
    add_postprocess_arg(parser, "--postprocess")
    add_expansion_arg(parser)
    add_bpe_arg(parser, '--bpe-codes')
    args = parser.parse_args()
    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    processor = DataProcessor(args.expansion, args.bpe_codes, args.postprocess)
    for sent_idx, level_idx, transition in processor.process_sentences(input_lines):
        print(transition.to_str())


if __name__ == '__main__':
    main()
