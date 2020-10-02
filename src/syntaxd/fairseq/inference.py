from fairseq.data.data_utils import collate_tokens
from fairseq.data.dictionary import Dictionary
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple

from syntaxd.data.dependency.binarize_data import (
    KEY_PREV_LEVEL_TOKENS,
    KEY_CAUSALITY_MASK,
    KEY_HEAD_POSITIONS,
    KEY_NEXT_LEVEL_EXPANS,
)
from syntaxd.fairseq.dataset import collate2d as collate_masks
from syntaxd.data.dependency.expansion import ExpansionStrategy
from syntaxd.data.dependency.transitions import heads2causality_mask


class IterativeInference:
    def __init__(self,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary,
                 expansion_strategy: ExpansionStrategy,
                 device=None,
                 mask_unk: bool = True):

        assert token_dictionary.pad_index == expansion_dictionary.pad_index
        self.device = device or torch.device('cpu')
        self.pad_idx = token_dictionary.pad_index
        self.token_dictionary = token_dictionary
        self.expansion_dictionary = expansion_dictionary
        self.expansion_strategy = expansion_strategy
        self.root_token_id = token_dictionary.index(expansion_strategy.root_node_token())

        minus_inf = float('-inf')
        # create mask to use in the token softmax later
        token_prob_mask = np.zeros(shape=(len(token_dictionary)), dtype=np.float32)
        for dep_placeholder in expansion_strategy.get_dependency_placeholders():
            index_to_mask = token_dictionary.index(dep_placeholder)
            if index_to_mask == token_dictionary.unk_index:
                continue   # symbol not found (e.g. [subword] if no subwords) ==> skip
            token_prob_mask[index_to_mask] = minus_inf
        special_token_idxs = [token_dictionary.pad_index, token_dictionary.eos_index]
        if mask_unk:
            special_token_idxs += [token_dictionary.unk_index]
        for special_token_idx in special_token_idxs:
            token_prob_mask[special_token_idx] = minus_inf
        self.token_prob_mask = torch.from_numpy(token_prob_mask).to(device)

        # create mask to use in the expansion softmax later
        expansion_prob_mask = np.zeros(shape=(len(expansion_dictionary)), dtype=np.float32)
        special_idxs = [expansion_dictionary.pad_index, expansion_dictionary.eos_index]
        if mask_unk:
            special_idxs = [expansion_dictionary.unk_index]
        for special_idx in special_idxs:
            expansion_prob_mask[special_idx] = minus_inf
        self.expansion_prob_mask = torch.from_numpy(expansion_prob_mask).to(device)

    def to(self, device):
        self.device = device
        self.token_prob_mask = self.token_prob_mask.to(device)
        self.expansion_prob_mask = self.expansion_prob_mask.to(device)
        return self

    def initial_state(self, num_sentences: int) -> Tuple[List[List[int]], List[List[int]]]:
        tokens = [[self.root_token_id]] * num_sentences
        heads = [[-1]] * num_sentences
        return tokens, heads

    def inference_step(self,
                       model: nn.Module,
                       previous_level_tokens: List[List[int]],
                       previous_level_heads: List[List[int]],
                       expansion_sampling: Callable[[torch.Tensor], torch.LongTensor] = None,
                       given_next_level_expansions: List[List[int]] = None,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """

        :param model:
        :param previous_level_tokens:
        :param previous_level_heads:
        :param expansion_sampling:
        :param given_next_level_expansions:
        :return: Tuples of token logits, token probabilities, expansion logits
                 and expansion probabilities.
        """

        prev_tokens = collate_tokens([torch.LongTensor(p)
                                      for p in previous_level_tokens],
                                     self.pad_idx,
                                     eos_idx=None,
                                     left_pad=False)

        # Index -1 is used for the root node. We shift positions so that
        # the original -1 is now pad_idx + 1
        head_pos_shift = self.pad_idx + 2

        head_positions = collate_tokens([torch.LongTensor(heads) + head_pos_shift
                                         for heads in previous_level_heads],
                                        self.pad_idx,
                                        eos_idx=None,
                                        left_pad=False)

        previous_level_dependency_masks = [heads2causality_mask(heads)
                                           for heads in previous_level_heads]
        prev_causality_masks = collate_masks(
                                         [1 - torch.ByteTensor(m).permute(1, 0)
                                          for m in previous_level_dependency_masks],
                                         self.pad_idx)

        next_level_expansions = (None if given_next_level_expansions is None
                                 else collate_tokens([torch.LongTensor(n)
                                                      for n in given_next_level_expansions],
                                                     self.pad_idx,
                                                     eos_idx=None,
                                                     left_pad=False).to(self.device))

        net_input = {
                     KEY_PREV_LEVEL_TOKENS: prev_tokens.to(self.device),
                     KEY_CAUSALITY_MASK: prev_causality_masks.to(self.device),
                     KEY_HEAD_POSITIONS: head_positions.to(self.device),
                     KEY_NEXT_LEVEL_EXPANS: next_level_expansions,
                     'expansion_sampling': expansion_sampling,
                     }

        token_logits, expansion_logits, expansion_ids = model(**net_input)

        return token_logits, expansion_logits, expansion_ids

