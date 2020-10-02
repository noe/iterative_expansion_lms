import argparse
from subword_nmt import apply_bpe
from typing import Callable, List, Optional


def create_bpe_segmenter(bpe_codes: apply_bpe.BPE) -> Callable[[List[str]], List[List[str]]]:

    def _segment_tokens(tokens: List[str]) -> List[List[str]]:
        """
        This function is a copy-paste of function BPE.segment_tokens that it groups the
        subwords of each word.
        :param bpe_codes: BPE segmentation info.
        :param tokens: List of words
        :return: List of lists of subwords
        """
        output = []
        for word in tokens:
            word_subwords = []
            new_word = [out for segment in bpe_codes._isolate_glossaries(word)
                        for out in apply_bpe.encode(segment,
                                                    bpe_codes.bpe_codes,
                                                    bpe_codes.bpe_codes_reverse,
                                                    bpe_codes.vocab,
                                                    bpe_codes.separator,
                                                    bpe_codes.version,
                                                    bpe_codes.cache,
                                                    bpe_codes.glossaries)]

            for item in new_word[:-1]:
                word_subwords.append(item + bpe_codes.separator)
            word_subwords.append(new_word[-1])

            output.append(word_subwords)

        return output

    return _segment_tokens


def add_bpe_arg(parser: argparse.ArgumentParser, argument: str = "--bpe-codes"):
    class Action(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            bpe_segmenter = None
            if values is not None:
                with open(values, encoding='utf-8') as bpe_codes_file:
                    bpe_codes = apply_bpe.BPE(bpe_codes_file)
                    bpe_segmenter = create_bpe_segmenter(bpe_codes)
            setattr(args, self.dest, bpe_segmenter)

    parser.add_argument(argument, required=False, type=str, action=Action)
