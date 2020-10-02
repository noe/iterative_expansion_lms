from fairseq.data import FairseqDataset, Dictionary
import torch
from fairseq.data import data_utils
from seqp.hdf5 import Hdf5RecordReader
from typing import List

from syntaxd.data.dependency.binarize_data import (
    KEY_PREV_LEVEL_TOKENS,
    KEY_NEXT_LEVEL_TOKENS,
    KEY_NEXT_LEVEL_EXPANS,
    KEY_CAUSALITY_MASK,
    KEY_HEAD_POSITIONS,
    external_keys,
)

KEY_TYPES = {
    KEY_PREV_LEVEL_TOKENS: torch.long,
    KEY_NEXT_LEVEL_TOKENS: torch.long,
    KEY_NEXT_LEVEL_EXPANS: torch.long,
    KEY_HEAD_POSITIONS: torch.long,
    KEY_CAUSALITY_MASK: torch.uint8,
}


def collate2d(values, pad_idx):
    size0 = max(v.size(0) for v in values)
    size1 = max(v.size(1) for v in values)

    res = values[0].new(len(values), size0, size1).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i, :v.size(0), :v.size(1)])

    return res


def collate(samples, pad_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        is2d = len(samples[0]['data'][key].shape) == 2
        if is2d:
            return collate2d([s['data'][key] for s in samples], pad_idx)
        else:
            return data_utils.collate_tokens([s['data'][key] for s in samples],
                                             pad_idx,
                                             eos_idx=None,
                                             left_pad=False)

    net_input = {k: merge(k) for k in KEY_TYPES.keys()}

    # Index -1 is used for the root node. We shift positions so that
    # the original -1 is now pad_idx + 1
    head_pos_shift = pad_idx + 2
    net_input[KEY_HEAD_POSITIONS] += head_pos_shift

    target_expans = net_input[KEY_NEXT_LEVEL_EXPANS]
    target_tokens = net_input[KEY_NEXT_LEVEL_TOKENS]

    bsz, seq_len = target_tokens.shape
    num_batch_tokens = bsz * seq_len
    num_pad_tokens = int((target_tokens == pad_idx).sum())
    pad_ratio = float(num_pad_tokens)/num_batch_tokens

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['data'][KEY_NEXT_LEVEL_TOKENS]) for s in samples),
        'net_input': net_input,
        'target_tokens': target_tokens,
        'target_expans': target_expans,
        'num_batch_tokens': num_batch_tokens,
        'num_pad_tokens': num_pad_tokens,
        'pad_ratio': pad_ratio,
    }


class TransitionDataset(FairseqDataset):
    """
    Dataset to load a reader with records with fields.
    """

    def __init__(self,
                 token_dict: Dictionary,
                 expansion_dict: Dictionary,
                 data_files: List[str],
                 max_positions: int = 1024,
                 overfit: bool = False):

        self.data_files = data_files
        self.reader = Hdf5RecordReader(data_files)
        self.token_dict = token_dict
        self.expansion_dict = expansion_dict
        self.max_positions = max_positions
        self.src_dict = token_dict
        self.overfit = overfit
        self.first_batch = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the reader (see https://github.com/h5py/h5py/issues/1092)
        del state["reader"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add reader back since it doesn't exist in the pickle
        self.reader = Hdf5RecordReader(self.data_files)

    def __getitem__(self, index):
        elems = external_keys(self.reader.retrieve(index))
        data = {f: torch.from_numpy(elems[f]).type(t) for f, t in KEY_TYPES.items()}
        data[KEY_CAUSALITY_MASK] = 1 - data[KEY_CAUSALITY_MASK].permute(1, 0)
        assert len(data[KEY_PREV_LEVEL_TOKENS]) == len(data[KEY_NEXT_LEVEL_TOKENS])
        assert data[KEY_CAUSALITY_MASK].shape[0] == len(data[KEY_PREV_LEVEL_TOKENS])
        return {'id': index, 'data': data}

    def __len__(self):
        return self.reader.num_records()

    def collater(self, samples):
        if not self.overfit:
            return collate([s for s in samples], self.src_dict.pad_index)

        if self.first_batch is None:
            self.first_batch = collate([s for s in samples], self.src_dict.pad_index)
            import pickle
            with open('firstbatch.pickle', 'wb') as f:
                pickle.dump(self.first_batch, f)
        return self.first_batch

    def num_tokens(self, index):
        return self.reader.length(index)

    def size(self, index):
        return self.reader.length(index)

    def ordered_indices(self):
        return [idx for idx, length in self.reader.indexes_and_lengths()]

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError
