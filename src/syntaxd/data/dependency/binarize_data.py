import argparse
import io
import numpy as np
import random
import sys
from typing import Dict, List

from syntaxd.data.dependency.transitions import Transition
from seqp.hdf5 import Hdf5RecordWriter
from seqp.record import ShardedWriter
from seqp.vocab import Vocabulary


INTERNAL_KEY_PREV_LEVEL_TOKENS = 'p'
INTERNAL_KEY_NEXT_LEVEL_TOKENS = 'n'
INTERNAL_KEY_NEXT_LEVEL_EXPANS = 'e'
INTERNAL_KEY_LOSS_MASK = 'l'
INTERNAL_KEY_CAUSALITY_MASK = 'm'
INTERNAL_KEY_HEAD_POSITIONS = 'h'

KEY_PREV_LEVEL_TOKENS = 'previous_level_tokens'
KEY_NEXT_LEVEL_TOKENS = 'next_level_tokens'
KEY_NEXT_LEVEL_EXPANS = 'next_level_expansions'
KEY_LOSS_MASK = 'loss_mask'
KEY_CAUSALITY_MASK = 'causality_mask'
KEY_HEAD_POSITIONS = 'head_positions'


FIELDS = [INTERNAL_KEY_PREV_LEVEL_TOKENS,
          INTERNAL_KEY_NEXT_LEVEL_TOKENS,
          INTERNAL_KEY_NEXT_LEVEL_EXPANS,
          INTERNAL_KEY_CAUSALITY_MASK,
          INTERNAL_KEY_HEAD_POSITIONS,
          ]

INTERNAL2EXTERNAL = {INTERNAL_KEY_PREV_LEVEL_TOKENS: KEY_PREV_LEVEL_TOKENS,
                     INTERNAL_KEY_NEXT_LEVEL_TOKENS: KEY_NEXT_LEVEL_TOKENS,
                     INTERNAL_KEY_NEXT_LEVEL_EXPANS: KEY_NEXT_LEVEL_EXPANS,
                     INTERNAL_KEY_CAUSALITY_MASK: KEY_CAUSALITY_MASK,
                     INTERNAL_KEY_HEAD_POSITIONS: KEY_HEAD_POSITIONS,
                     }


def external_keys(record):
    """
    Function that maps keys used in persistent storage (which are short strings
    to avoid consuming space) to keys used for handling records programmatically.
    :param record:
    :return:
    """
    if INTERNAL_KEY_PREV_LEVEL_TOKENS not in record:
        return record  # keys are already the external ones
    return {external_key: record[internal_key]
            for internal_key, external_key in INTERNAL2EXTERNAL.items()}


MASKING_NOISE = 'noise'
MASKING_NOTHING = 'nothing'
MASKING_ALL = 'all'


def tok_ids(tokens: List[str], vocab: Vocabulary):
    return vocab.encode(tokens, use_unk=True, add_eos=False)


def mask_elems(elems, loss_mask, pad_idx):
    return [elem if included == 1 else pad_idx
            for elem, included in zip(elems, loss_mask)]


def maybe_change(idx: int, vocab: Vocabulary) -> int:
    random_idx = random.randint(vocab.num_special, len(vocab.idx2symbol) - 1)
    return random.choice([idx, random_idx])


def noise(elems, loss_mask, ratio, token_vocab):
    # we only take into account not dependency tokens (i.e. only those
    # not included in the loss)
    candidate_idxs = {k
                      for k, included_in_loss in enumerate(loss_mask)
                      if included_in_loss == 0}
    num_candidates = len(candidate_idxs)
    num_selected = (0 if num_candidates == 0 else
                    int(num_candidates * ratio + random.random()))
    # for the "+ random.random()" part, see https://stackoverflow.com/a/40249212/674487

    selected_idxs = set(random.sample(candidate_idxs, num_selected))
    kept_idxs = candidate_idxs - selected_idxs
    if not kept_idxs:
        return elems

    new_elems = [e if k not in selected_idxs
                 else maybe_change(elems[random.sample(kept_idxs, 1)[0]], token_vocab)
                 for k, e in enumerate(elems)]

    return new_elems


def transition2record(transition: Transition,
                      token_vocab: Vocabulary,
                      expansion_vocab: Vocabulary,
                      token_masking: str,
                      noise_ratio: float) -> Dict[str, np.ndarray]:
    loss_mask = transition.loss_mask

    prev_tokens = tok_ids(transition.previous_level_tokens, token_vocab)
    if token_masking == MASKING_NOISE:
        prev_tokens = noise(prev_tokens, loss_mask, noise_ratio, token_vocab)

    next_tokens = tok_ids(transition.next_level_tokens, token_vocab)

    if token_masking == MASKING_ALL:
        next_tokens = mask_elems(next_tokens, loss_mask, token_vocab.pad_id)

    next_expans = tok_ids(transition.next_level_expansions, expansion_vocab)
    next_expans = mask_elems(next_expans, loss_mask, expansion_vocab.pad_id)

    assert len(prev_tokens) == len(next_tokens)
    assert len(next_tokens) == len(next_expans)

    return {
            INTERNAL_KEY_PREV_LEVEL_TOKENS: np.array(prev_tokens),
            INTERNAL_KEY_NEXT_LEVEL_TOKENS: np.array(next_tokens),
            INTERNAL_KEY_NEXT_LEVEL_EXPANS: np.array(next_expans),
            INTERNAL_KEY_CAUSALITY_MASK: np.array(transition.causality_mask, dtype=np.uint8),
            INTERNAL_KEY_HEAD_POSITIONS: np.array(transition.heads, np.int16),
            }


def main():
    parser = argparse.ArgumentParser('Binarizes transitions in textual format into seqp format')
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--token-vocab', type=str, required=True)
    parser.add_argument('--expansion-vocab', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--shard-size', type=int, default=400000)
    masking_parser = parser.add_mutually_exclusive_group()
    masking_parser.add_argument('--no-masking', dest='masking', action='store_const', const=MASKING_NOTHING)
    masking_parser.add_argument('--noise', dest='masking', action='store_const', const=MASKING_NOISE)
    parser.set_defaults(masking=MASKING_ALL)
    parser.add_argument('--noise-percentage', type=int, default=20)
    args = parser.parse_args()

    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    with open(args.token_vocab, encoding='utf-8') as token_vocab_file:
        token_vocab = Vocabulary.from_json(token_vocab_file.read())

    with open(args.expansion_vocab, encoding='utf-8') as expansion_vocab_file:
        expansion_vocab = Vocabulary.from_json(expansion_vocab_file.read())

    output_template = args.output_prefix + '_{}.hdf5'

    noise_ratio = float(args.noise_percentage) / 100

    with ShardedWriter(Hdf5RecordWriter,
                       output_template,
                       max_records_per_shard=args.shard_size,
                       fields=FIELDS,
                       sequence_field=INTERNAL_KEY_PREV_LEVEL_TOKENS) as writer:
        for idx, line in enumerate(input_lines):
            line = line.strip()
            transition = Transition.from_str(line)
            record = transition2record(transition,
                                       token_vocab,
                                       expansion_vocab,
                                       args.masking,
                                       noise_ratio)
            writer.write(idx, record)


if __name__ == '__main__':
    main()
