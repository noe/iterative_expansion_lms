import math
from typing import Iterable, List, Optional


from syntaxd.data.dependency.trees import Node, TreeBuilder
from syntaxd.data.dependency.expansion import ExpansionStrategy


def causality_idxs2mask(causality_idxs: List[List[int]]) -> List[List[int]]:
    """
    Creates a causality adjacency matrix from the node causality information.
    :param causality_idxs: Causality information, expressed as a list where
                 at each position there is a list of integers representing
                 the indexes of the heads, where the indexes are
                 the positions in the first list nesting level.
    :return: an adjacency matrix represented as a nested list with 1's
             representing the connectivity between nodes (and 0's for the
             lack of connectivity).
    """
    num_elems = len(causality_idxs)
    causality_matrix = []
    for elem_idx in range(num_elems):
        elem_deps = causality_idxs[elem_idx]
        elem_causality = [1 if idx in elem_deps or idx == elem_idx else 0
                          for idx in range(num_elems)]
        causality_matrix.append(elem_causality)
    return causality_matrix


def heads2causality_idxs(heads: List[int]) -> List[List[int]]:
    causality = []
    for idx, head in enumerate(heads):
        idx_heads = []
        while head != -1:
            idx_heads.append(head)
            head = heads[head]
        causality.append(idx_heads)
    return causality


def heads2causality_mask(heads: List[int]) -> List[List[int]]:
    causality_idxs = heads2causality_idxs(heads)
    mask = causality_idxs2mask(causality_idxs)
    return mask


class Transition:
    """
    Structure to store the information associated to a transition in
    the iterative expansion of the dependency tree.
    """

    def __init__(self,
                 previous_level_tokens: List[str],
                 loss_mask: List[int],
                 next_level_tokens: List[str],
                 next_level_expansions: List[str],
                 heads: List[int]):
        """
        Constructor.
        :param previous_level_tokens: Tokens (words/subwords/...) in the
               previous level.
        :param loss_mask: mask (list of 0/1) with the positions to be
               taken into account in the loss.
        :param next_level_tokens: Tokens (words/subwords/...) in the next level.
        :param next_level_expansions: Expansion placeholders in the next level.
        :param heads: indexes of the head of each token (and -1 for the root)
        """
        self.previous_level_tokens = previous_level_tokens
        self.loss_mask = loss_mask
        self.next_level_tokens = next_level_tokens
        self.next_level_expansions = next_level_expansions
        self.heads = heads

    @property
    def causality_mask(self):
        return heads2causality_mask(self.heads)

    def to_str(self) -> str:
        """
        Formats the Transition as a string. This string can be later parsed
        back into a Transition by means of method 'from_str'.
        :return: The stringified form of the transition-
        """
        previous_tokens_str = " ".join(self.previous_level_tokens)
        next_tokens_str = " ".join(self.next_level_tokens)
        next_expansions_str = " ".join(self.next_level_expansions)
        loss_mask_str = "".join([str(i) for i in self.loss_mask])
        heads_str = " ".join([str(h) for h in self.heads])
        return "{}\t{}\t{}\t{}\t{}".format(previous_tokens_str,
                                           next_tokens_str,
                                           next_expansions_str,
                                           loss_mask_str,
                                           heads_str)

    @staticmethod
    def from_str(s: str) -> "Transition":
        """
        Parses a string (created by means of method 'to_str') into a Transition.
        :param s: String to be parsed.
        :return: Parsed Transition.
        """
        (previous_tokens_str,
         next_tokens_str,
         next_expansions_str,
         loss_mask_str,
         heads_str) = s.split('\t')

        previous_level_tokens = previous_tokens_str.split(" ")
        next_level_tokens = next_tokens_str.split(" ")
        next_level_expansions = next_expansions_str.split(" ")
        loss_mask = [int(c) for c in loss_mask_str]
        heads = [int(h) for h in heads_str.split(" ")]
        return Transition(previous_level_tokens,
                          loss_mask,
                          next_level_tokens,
                          next_level_expansions,
                          heads)


def build_transitions(root_node: Node,
                      expansion_tokens: ExpansionStrategy,
                      ) -> Iterable[Transition]:
    """
    Builds the expansion iterations of a dependency tree.
    :param root_node: Root node of the dependency tree.
    :param expansion_tokens: Expansion token convention.
    :return: yields tuples of
              - the previous level tokens (list of strings)
              - the loss mask (list with 0's and 1's)
              - the next level tokens (list of strings)
              - the next level expansion tokens (list of strings).
              - the causality matrix (see function 'deps2causality_matrix')
    """
    previous_level_tokens = [expansion_tokens.root_node_token()]
    current_level_nodes = [root_node]
    current_heads = [-1]

    while True:
        loss_mask = [0 if isinstance(node, str) else 1 for node in current_level_nodes]
        current_level_tokens = [node if isinstance(node, str) else node.token
                                for node in current_level_nodes]
        current_level_expansions = [expansion_tokens.null_expansion_token() if isinstance(node, str)
                                    else expansion_tokens.format_expansion_token(node)
                                    for node in current_level_nodes]

        yield Transition(previous_level_tokens,
                         loss_mask,
                         current_level_tokens,
                         current_level_expansions,
                         current_heads)

        # Expand level into previous_level_tokens and current_level_nodes
        any_nodes_expanded = False
        expanded_current_level_nodes = []
        expanded_current_level_tokens = []
        old_idx2new_idx = []
        new_heads = {}
        for old_idx, node in enumerate(current_level_nodes):
            if isinstance(node, str):
                old_idx2new_idx.append(len(expanded_current_level_tokens))
                expanded_current_level_tokens.append(node)
                expanded_current_level_nodes.append(node)
            else:
                assert isinstance(node, Node)
                num_deps = len(node.right_deps) + len(node.left_deps)
                any_nodes_expanded = any_nodes_expanded or num_deps > 0
                num_left_deps = len(node.left_deps)
                num_right_deps = len(node.right_deps)

                initial_idx = len(expanded_current_level_tokens)
                expanded_current_level_tokens.extend(expansion_tokens.expand_left(node))
                new_idx = len(expanded_current_level_tokens)
                old_idx2new_idx.append(new_idx)
                expanded_current_level_tokens.append(node.token)
                expanded_current_level_tokens.extend(expansion_tokens.expand_right(node))
                dep_idxs = set(range(initial_idx, initial_idx + num_left_deps + num_right_deps + 1)) - {new_idx}
                for dep_idx in dep_idxs:
                    new_heads[dep_idx] = new_idx

                expanded_current_level_nodes.extend(node.left_deps)
                expanded_current_level_nodes.append(node.token)
                expanded_current_level_nodes.extend(node.right_deps)

        if not any_nodes_expanded:
            break

        new_idx2old_idx = {new_idx: old_idx
                           for old_idx, new_idx in enumerate(old_idx2new_idx)}

        expanded_heads = expand_heads(len(expanded_current_level_nodes),
                                      new_idx2old_idx,
                                      old_idx2new_idx,
                                      current_heads,
                                      new_heads)

        current_level_nodes = expanded_current_level_nodes
        previous_level_tokens = expanded_current_level_tokens
        current_heads = expanded_heads


def expand_heads(num_tokens: int,
                 new_idx2old_idx,
                 old_idx2new_idx,
                 prev_level_heads,
                 next_level_heads,
                 ) -> List[List[int]]:
    """
    Expands the dependencies after a token expansion.
    :param num_tokens: number of tokens in the next level.
    :param new_idx2old_idx: associations between new indexes and old ones.
    :param old_idx2new_idx: associations between old indexes and new ones.
    :param prev_level_heads: heads of the previous level.
    :param next_level_heads: new heads just expanded.
    :return: the expanded dependency matrix.
    """
    expanded_heads = [None] * num_tokens
    for new_idx in range(num_tokens):
        if new_idx in new_idx2old_idx:
            old_idx = new_idx2old_idx[new_idx]
            head_old_idx = prev_level_heads[old_idx]
            head_new_idx = -1 if head_old_idx == -1 else old_idx2new_idx[head_old_idx]
            expanded_heads[new_idx] = head_new_idx
        else:
            expanded_heads[new_idx] = next_level_heads[new_idx]

    return expanded_heads


class TransitionBuilder:
    def __init__(self,
                 expansion_strategy: ExpansionStrategy,
                 tree_builder: TreeBuilder,
                 ):
        self.expansion_strategy = expansion_strategy
        self.tree_builder = tree_builder

    def transitions_from_sentence(self, fieldlist_seq: List[List[str]]) -> Iterable[Transition]:
        tree = self.tree_builder.tree_from_sentence(fieldlist_seq)
        yield from build_transitions(tree, self.expansion_strategy)
