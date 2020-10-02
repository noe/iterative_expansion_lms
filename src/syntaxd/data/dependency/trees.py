from argparse import ArgumentParser
from typing import Callable, Dict, List, Optional, Set, Tuple


SUBWORD_DEP_LABEL = "subword"


PROCESS_NONE = 'none'
PROCESS_INSIDE_OUT = 'inside-out'
PROCESS_LEFT_RIGHT = 'left-right'


class Node:
    """
    Structure to store the information of a node in the dependency parse tree.
    """

    def __init__(self, token, word_idx, left_deps, right_deps, dep_label, subword_idx=0):
        """
        Constructor.
        :param token: word or subword in the node.
        :param word_idx: index of the word (in the original sentence) this
                         token is associated to. If subword-level tokens
                         are used, all subwords in a word share the same
                         word index.
        :param left_deps: list of nodes whose head is this node (self) and
                          are located to its left.
        :param right_deps: list of nodes whose head is this node (self) and
                           are located to its right.
        :param dep_label: label of the dependency.
        :param subword_idx: if this token is not subword-level, then this
                            is None. Otherwise, it is the index of the
                            subword within the word.
        """
        self.token = token
        self.left_deps = left_deps
        self.right_deps = right_deps
        self.word_idx = word_idx
        self.subword_idx = subword_idx
        self.dep_label = dep_label
        if word_idx > 0:
            assert dep_label, f"Dep. label cannot be empty (word_idx {word_idx})"


def _build_subtree_aux(word_idx: int,
                       head_idx2word_idx: Dict[int, List[int]],
                       word_idx2word: List[str],
                       word_idx2dep_label: List[str],
                       ) -> Node:
    """
    Auxiliary function to build a dependency tree from the
    head index information.
    :param word_idx: word index whose subtree is to be built.
    :param head_idx2word_idx: associations of head index to word index.
    :param word_idx2word: associations of word index to word.
    :param word_idx2dep_label: associations from word index to dependency label.
    :return: Node associated to the specified word index.
    """
    deps = head_idx2word_idx[word_idx]
    left_deps = [d for d in deps if d < word_idx]
    right_deps = [d for d in deps if d > word_idx]
    left_trees = [_build_subtree_aux(d, head_idx2word_idx, word_idx2word, word_idx2dep_label)
                  for d in left_deps]
    right_trees = [_build_subtree_aux(d, head_idx2word_idx, word_idx2word, word_idx2dep_label)
                   for d in right_deps]

    return Node(word_idx2word[word_idx],
                word_idx,
                left_trees,
                right_trees,
                word_idx2dep_label[word_idx])


def build_tree(word_idx2head_idx: List[int],
               word_idx2word: List[str],
               word_idx2dep_label: List[str],
               ) -> Node:
    """
    Builds a dependency tree from the provided head index and word
    information.
    :param word_idx2head_idx: associations of word index (within the sentence)
                              to head word index.
    :param word_idx2word: associations from word index (within the sentence)
                          to word.
    :param word_idx2dep_label: associations from word index to dependency label.
    :return: Node of the root word.
    """
    from collections import defaultdict

    head_idx2word_idx = defaultdict(list)
    for word_idx, head_idx in enumerate(word_idx2head_idx):
        if word_idx == 0:
            assert head_idx is None, "Malformed tree"
            continue
        head_idx2word_idx[head_idx].append(word_idx)


    root_word_idx = 0
    root_node = _build_subtree_aux(root_word_idx,
                                   head_idx2word_idx,
                                   word_idx2word,
                                   word_idx2dep_label)
    assert len(root_node.left_deps) == 0, "Root node must be the leftmost one"
    assert len(root_node.right_deps) == 1, "Only one dependency to root is admitted"
    return root_node.right_deps[0]


def _make_sequential_branch(deps: List[Node],
                            reverse_left: bool,
                            is_left: bool) -> Tuple[List[Node], Optional[Node]]:
    last: Optional[Node] = None
    new_deps: List[Node] = []
    must_reverse = reverse_left and is_left
    traverse = reversed if must_reverse else lambda it: it
    for d in traverse(deps):
        d_first, d_last = _make_sequential_aux(d, reverse_left=reverse_left)
        if last is not None:
            if must_reverse:
                last.left_deps = [d_first]
            else:
                last.right_deps = [d_first]
        last = d_last
        if not new_deps:
            new_deps.append(d_first)
    return new_deps, last


def _make_sequential_aux(node: Node, reverse_left: bool) -> Tuple[Node, Node]:
    new_left_dep, left_last = _make_sequential_branch(node.left_deps,
                                                      reverse_left=reverse_left,
                                                      is_left=True)
    new_right_dep, right_last = _make_sequential_branch(node.right_deps,
                                                        reverse_left=reverse_left,
                                                        is_left=False)
    first = Node(node.token,
                 node.word_idx,
                 new_left_dep,
                 new_right_dep,
                 node.dep_label)
    last = right_last or first
    return first, last


def make_sequential_left_right(tree: Node):
    return _make_sequential_aux(tree, reverse_left=False)[0]


def make_sequential_inside_out(tree: Node):
    return _make_sequential_aux(tree, reverse_left=True)[0]


def segment_subwords(node: Node, subwords: List[List[str]]) -> Node:
    """
    Takes a word-level dependency tree and modifies it to split words
    into subwords. For this, it creates a new node for each subword
    in each word, and rearranges the dependencies so that 1) the head
    of the original word now points to the first subwords, 2) the
    dependencies of the original word are now dependencies of the
    last subwords, 3) the subword nodes depend on the previous
    subword sequentially from left to right, 5) the word_idx field
    of the subword nodes point to the index of the originating word
    and 6) the subword_idx field of the node reflects the position
    of the subword within the word.
    :param node: Dependency tree.
    :param subwords: Subwords in each word in the sentence, expressed
                     as a list of lists of subwords.
    :return: a new tree (which can be a modified version of the
             original tree) with the subword decomposition in place.
    """
    left_deps = [segment_subwords(d, subwords) for d in node.left_deps]
    right_deps = [segment_subwords(d, subwords) for d in node.right_deps]

    word_subwords = subwords[node.word_idx]
    num_subwords = len(word_subwords)
    has_multiple_subwords = num_subwords > 1
    if not has_multiple_subwords:
        return Node(node.token,
                    node.word_idx,
                    left_deps,
                    right_deps,
                    node.dep_label)

    subword_node = None
    subword_right_deps = right_deps
    reverse_subword_idxs = range(num_subwords - 1, -1, -1)
    for subword_idx in reverse_subword_idxs:
        subword_node = Node(word_subwords[subword_idx],
                            node.word_idx,
                            [],
                            subword_right_deps,
                            node.dep_label if subword_idx == 0 else SUBWORD_DEP_LABEL,
                            subword_idx=subword_idx)
        subword_right_deps = [subword_node]

    subword_node.left_deps = left_deps
    return subword_node


class TreeBuilder:
    def __init__(self,
                 segmenter: Callable[[List[str]], List[List[str]]] = None,
                 postprocess: str = PROCESS_NONE,
                 word_field_idx: int = 1,
                 head_field_idx: int = 5,
                 dep_label_field_idx: int = 6,
                 ):
        self.segmenter = segmenter
        self.word_field_idx = word_field_idx
        self.head_field_idx = head_field_idx
        self.dep_label_field_idx = dep_label_field_idx
        self.postprocess = ((lambda tree: tree) if postprocess == PROCESS_NONE
                            else make_sequential_inside_out if postprocess == PROCESS_INSIDE_OUT
                            else make_sequential_left_right if postprocess == PROCESS_LEFT_RIGHT
                            else None)

    def tree_from_sentence(self, fieldlist_seq: List[List[str]]) -> Node:
        # The "None" below is for the root
        words = [None] + [fields[self.word_field_idx] for fields in fieldlist_seq]
        head_idxs = [None] + [int(fields[self.head_field_idx]) for fields in fieldlist_seq]
        dep_labels = [None] + [fields[self.dep_label_field_idx] for fields in fieldlist_seq]

        # Build tree of words
        tree = build_tree(head_idxs, words, dep_labels)

        tree = self.postprocess(tree)

        if self.segmenter is not None:
            subwords = [[]] + self.segmenter(words[1:])
            tree = segment_subwords(tree, subwords)

        return tree


def iterate_leaves(node: Node, f: Callable[[Node, Node], None], head: Optional[Node] = None):
    for left_dep in node.left_deps:
        iterate_leaves(left_dep, f, head=node)
    f(node, head)
    for right_dep in node.right_deps:
        iterate_leaves(right_dep, f, head=node)


def compute_depth(node: Node):
    deps = node.left_deps + node.right_deps
    return 1 if not deps else 1 + max(compute_depth(d) for d in deps)


def add_postprocess_arg(parser: ArgumentParser, argument: str = "--postprocess"):
    parser.add_argument(argument,
                        choices=[PROCESS_NONE, PROCESS_INSIDE_OUT, PROCESS_LEFT_RIGHT],
                        default=PROCESS_NONE,
                        const=PROCESS_NONE,
                        nargs='?')
