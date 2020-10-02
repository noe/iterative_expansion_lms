from typing import Iterable, List


def conll_to_sentences(conll_lines: Iterable[str]) -> Iterable[List[List[str]]]:
    """
    Reads a dependency parse file in CONLL format and yields sentence by
    sentence.
    :param conll_lines: Contents of the CONLL file.
    :return:  Yields elements that are list of list of fields, each field
    is one of the fields of the CONLL file belonging to a word (first list
    nesting level), and a sequence of words are grouped into a sentence
    (second list nesting level).
    """
    sentence = []
    for line in conll_lines:
        line = line.strip()
        if not line:
            yield sentence
            sentence = []
            continue
        if line.startswith('#'):
            continue
        factors = line.split("\t")
        sentence.append(factors)
    if sentence:
        yield sentence
