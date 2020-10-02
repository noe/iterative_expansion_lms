from fairseq.tasks import FairseqTask, register_task
from fairseq.data.dictionary import Dictionary
from glob import glob
import os
from seqp.vocab import Vocabulary
from seqp.integration.fairseq import vocab_to_dictionary
from syntaxd.fairseq.dataset import TransitionDataset
from syntaxd.data.dependency.expansion import add_expansion_arg


@register_task('iterative_lm')
class IterativeLmTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--overfit', action='store_true')
        add_expansion_arg(parser)
        parser.add_argument('data', help='path to data directory')

    def __init__(self,
                 args,
                 token_dictionary: Dictionary,
                 expansion_dictionary: Dictionary):
        super().__init__(args)
        self.token_dictionary = token_dictionary
        self.expansion_dictionary = expansion_dictionary
        self.expansion_strategy = args.expansion
        assert token_dictionary.pad_index == expansion_dictionary.pad_index

    @classmethod
    def setup_task(cls, args, **kwargs):
        token_dictionary = None
        expansion_dictionary = None
        if args.data:
            path = args.data
            token_vocab_filename = os.path.join(path, 'token_vocab.json')
            with open(token_vocab_filename, encoding='utf-8') as token_vocab_file:
                token_vocab = Vocabulary.from_json(token_vocab_file.read())
                token_dictionary = vocab_to_dictionary(token_vocab)

            expansion_vocab_filename = os.path.join(path, 'expansion_vocab.json')
            with open(expansion_vocab_filename, encoding='utf-8') as expansion_vocab_file:
                expansion_vocab = Vocabulary.from_json(expansion_vocab_file.read())
                expansion_dictionary = vocab_to_dictionary(expansion_vocab)

        return cls(args, token_dictionary, expansion_dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        if ('valid' in split or 'test' in split) and split[-1] == '1':
            raise FileNotFoundError

        path = self.args.data
        data_files = glob(os.path.join(path, split + '_*.hdf5'))

        self.datasets[split] = TransitionDataset(
            self.token_dictionary,
            self.expansion_dictionary,
            data_files,
            overfit=split == 'train' and self.args.overfit)

    @property
    def source_dictionary(self):
        return self.token_dictionary

    @property
    def target_dictionary(self):
        return self.token_dictionary

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        raise NotImplementedError
