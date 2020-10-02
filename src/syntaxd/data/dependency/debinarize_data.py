import argparse
from fairseq.data.dictionary import Dictionary
import numpy as np
from seqp.hdf5 import Hdf5RecordReader
from seqp.vocab import Vocabulary
from seqp.integration.fairseq import vocab_to_dictionary
from typing import Dict, List

from syntaxd.data.dependency.binarize_data import (
    KEY_PREV_LEVEL_TOKENS,
    KEY_NEXT_LEVEL_TOKENS,
    KEY_NEXT_LEVEL_EXPANS,
    KEY_HEAD_POSITIONS,
    external_keys,
)

from syntaxd.data.dependency.transitions import Transition
from syntaxd.data.dependency.expansion import add_expansion_arg, ExpansionStrategy


def unmask_tokens(tokens: List[str],
                  previous_tokens: List[str],
                  loss_mask: List[int],
                  ) -> List[str]:
    return [t if included_in_loss == 1 else prev_t
            for t, prev_t, included_in_loss
            in zip(tokens, previous_tokens, loss_mask)]


def unmask_expansions(expansions: List[str],
                      loss_mask: List[int],
                      null_expansion: str,
                      ) -> List[str]:
    return [e if included_in_loss == 1 else null_expansion
            for e, included_in_loss
            in zip(expansions, loss_mask)]


def record2transition(record: Dict[str, np.ndarray],
                      token_vocab: Vocabulary,
                      expansion_vocab: Vocabulary,
                      null_expansion: str) -> Transition:

    prev_tokens = token_vocab.decode(record[KEY_PREV_LEVEL_TOKENS].tolist())

    next_tokens = token_vocab.decode(record[KEY_NEXT_LEVEL_TOKENS].tolist())

    loss_mask = [0 if t == token_vocab.idx2symbol[token_vocab.pad_id] else 1
                 for t in next_tokens]

    next_tokens = unmask_tokens(next_tokens, prev_tokens, loss_mask)

    next_expans = expansion_vocab.decode(record[KEY_NEXT_LEVEL_EXPANS].tolist())
    next_expans = unmask_expansions(next_expans, loss_mask, null_expansion)

    head_positions = record[KEY_HEAD_POSITIONS].tolist()

    return Transition(previous_level_tokens=prev_tokens,
                      loss_mask=loss_mask,
                      next_level_tokens=next_tokens,
                      next_level_expansions=next_expans,
                      heads=head_positions,
                      )


def record2transition_dict(record: Dict[str, np.ndarray],
                           token_dictionary: Dictionary,
                           expansion_dictionary: Dictionary,
                           null_expansion: str) -> Transition:

    prev_tokens = token_dictionary.string(record[KEY_PREV_LEVEL_TOKENS].tolist()).split(' ')

    next_tokens = token_dictionary.string(record[KEY_NEXT_LEVEL_TOKENS].tolist()).split(' ')

    loss_mask = [0 if t == token_dictionary.pad_word else 1
                 for t in next_tokens]

    next_tokens = unmask_tokens(next_tokens, prev_tokens, loss_mask)

    next_expans = expansion_dictionary.string(record[KEY_NEXT_LEVEL_EXPANS].tolist()).split(' ')
    next_expans = unmask_expansions(next_expans, loss_mask, null_expansion)

    head_positions = record[KEY_HEAD_POSITIONS].tolist()

    return Transition(previous_level_tokens=prev_tokens,
                      loss_mask=loss_mask,
                      next_level_tokens=next_tokens,
                      next_level_expansions=next_expans,
                      heads=head_positions,
                      )


def main():
    parser = argparse.ArgumentParser('De-binarization tool')
    parser.add_argument('--token-vocab', type=str, required=True)
    parser.add_argument('--expansion-vocab', type=str, required=True)
    parser.add_argument('--use-fairseq-dictionaries', action='store_true')
    add_expansion_arg(parser)
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()

    expansion_strategy: ExpansionStrategy = args.expansion

    with open(args.token_vocab, encoding='utf-8') as token_vocab_file:
        token_vocab = Vocabulary.from_json(token_vocab_file.read())
        token_dictionary = vocab_to_dictionary(token_vocab)

    with open(args.expansion_vocab, encoding='utf-8') as expansion_vocab_file:
        expansion_vocab = Vocabulary.from_json(expansion_vocab_file.read())
        expansion_dictionary = vocab_to_dictionary(expansion_vocab)

    with Hdf5RecordReader(args.files) as reader:
        for idx in range(reader.num_records()):
            record = external_keys(reader.retrieve(idx))
            if args.use_fairseq_dictionaries:
                transition = record2transition_dict(record,
                                                    token_dictionary,
                                                    expansion_dictionary,
                                                    expansion_strategy.null_expansion_token())

            else:
                transition = record2transition(record,
                                               token_vocab,
                                               expansion_vocab,
                                               expansion_strategy.null_expansion_token())

            print(transition.to_str())


if __name__ == '__main__':
    main()
