from argparse import ArgumentParser
import itertools
from typing import List, Set, Tuple, Union

from syntaxd.data.dependency.trees import Node, SUBWORD_DEP_LABEL


class ExpansionStrategy:
    _NULL_EXPANSION = "-"

    def root_node_token(self):
        raise NotImplementedError

    def null_expansion_token(self):
        return ExpansionStrategy._NULL_EXPANSION

    def expand_left(self, node: Node) -> List[str]:
        raise NotImplementedError

    def expand_right(self, node: Node) -> List[str]:
        raise NotImplementedError

    def get_dependency_placeholders(self) -> Set[str]:
        raise NotImplementedError

    def format_expansion_token(self, node: Union[Node, str]) -> str:
        """
        Creates an "expansion" token for a Node. This expansion node
        carries information about how to expand the node in a decoding iteration.
        :param node: Node for which the expansion token is to be created.
        :return:
        """
        raise NotImplementedError

    def parse_num_deps(self, expansion_token: str) -> Tuple[int, int]:
        raise NotImplementedError

    def expand_deps(self, expansion_token: str) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def pretty_format(self, expansion_token: str):
        return expansion_token


class LabelExpansionStrategy(ExpansionStrategy):
    _ROOT_LABEL = 'ROOT'

    @staticmethod
    def _label2placeholder(label: str):
        return f'[{label}]'

    def __init__(self, labels: List[str]):
        root_label = LabelExpansionStrategy._ROOT_LABEL
        self._root_placeholder = LabelExpansionStrategy._label2placeholder(root_label)
        self._separator = '|'

        special_labels = [SUBWORD_DEP_LABEL,
                          ExpansionStrategy._NULL_EXPANSION]
        for special_label in special_labels:
            if special_label in labels:
                raise ValueError(f"Label \"{special_label}\" is not valid")

        self.labels = [SUBWORD_DEP_LABEL] + labels
        if root_label not in self.labels:
            self.labels = [root_label] + self.labels

        placeholders = [LabelExpansionStrategy._label2placeholder(label)
                        for label in self.labels]
        self._dependency_placeholders = set(placeholders)

    def root_node_token(self):
        return self._root_placeholder

    def get_dependency_placeholders(self) -> Set[str]:
        return self._dependency_placeholders

    def format_expansion_token(self, node: Union[Node, str]) -> str:
        num_left_deps = len(node.left_deps)
        elems = itertools.chain([str(num_left_deps)],
                                (d.dep_label for d in node.left_deps),
                                (d.dep_label for d in node.right_deps))
        return self._separator.join(elems)

    def parse_num_deps(self, expansion_token: str) -> Tuple[int, int]:
        left_deps, right_deps = self.expand_deps(expansion_token)
        return len(left_deps), len(right_deps)

    def expand_left(self, node: Node) -> List[str]:
        return [LabelExpansionStrategy._label2placeholder(dep.dep_label)
                for dep in node.left_deps]

    def expand_right(self, node: Node) -> List[str]:
        return [LabelExpansionStrategy._label2placeholder(dep.dep_label)
                for dep in node.right_deps]

    def expand_deps_(self, expansion_token: str) -> Tuple[List[str], List[str]]:
        is_terminal = expansion_token == '0'
        if is_terminal:
            return [], []

        elems = expansion_token.split(self._separator)

        assert len(elems) > 0

        is_special_token = len(elems) == 1
        if is_special_token:
            return [], []

        num_left_deps = int(elems[0])

        left_deps = elems[1:1 + num_left_deps]
        right_deps = elems[1 + num_left_deps:]

        return left_deps, right_deps

    def expand_deps(self, expansion_token: str) -> Tuple[List[str], List[str]]:
        left_deps, right_deps = self.expand_deps_(expansion_token)
        left_deps = [LabelExpansionStrategy._label2placeholder(label)
                     for label in left_deps]

        right_deps = [LabelExpansionStrategy._label2placeholder(label)
                      for label in right_deps]
        return left_deps, right_deps

    def pretty_format(self, expansion_token: str):
        left_deps, right_deps = self.expand_deps_(expansion_token)
        return '[' + '-'.join(left_deps + ['HEAD'] + right_deps) + ']'


def add_expansion_arg(parser: ArgumentParser, arg_name: str = '--placeholders'):
    deprecated_expansions = ['lr',  'label']

    def _create_expansion(placeholders_filename: str) -> ExpansionStrategy:
        if placeholders_filename in deprecated_expansions:
            raise ValueError(f"Deprecated expansion {placeholders_filename}")
        with open(placeholders_filename, encoding='utf-8') as f:
            labels = [line.strip() for line in f]
            return LabelExpansionStrategy(labels)

    parser.add_argument(arg_name,
                        type=_create_expansion,
                        required=True,
                        metavar='FUNCTION',
                        help="Placeholders file",
                        dest="expansion")
