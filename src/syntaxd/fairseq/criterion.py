# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import math
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion

from syntaxd.data.dependency.binarize_data import KEY_PREV_LEVEL_TOKENS


@register_criterion('token_expansion_cross_entropy')
class DoubleCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.scale_loss_with_padding = args.scale_loss_with_padding

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--scale-loss-with-padding', action='store_true')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        token_loss, expansion_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        num_batch_tokens = sample['num_batch_tokens']
        num_non_pad_tokens = num_batch_tokens - sample['num_pad_tokens']
        batch_density = float(num_non_pad_tokens) / num_batch_tokens
        loss = (token_loss + expansion_loss)
        if self.scale_loss_with_padding:
            loss = loss * batch_density
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'token_loss': utils.item(token_loss.data) if reduce else token_loss.data,
            'expansion_loss': utils.item(expansion_loss.data) if reduce else expansion_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['net_input'][KEY_PREV_LEVEL_TOKENS].size(0),
            'sample_size': sample_size,
            'target_num_pad_tokens': sample['num_pad_tokens'],
            'target_num_batch_tokens': sample['num_batch_tokens'],
            'target_pad_ratio': sample['pad_ratio'],
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        tokens_lprobs = model.get_normalized_probs_tokens(net_output, log_probs=True)
        tokens_lprobs = tokens_lprobs.view(-1, tokens_lprobs.size(-1))
        tokens_target = model.get_targets_tokens(sample, net_output).view(-1)
        token_loss = F.nll_loss(
            tokens_lprobs,
            tokens_target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        expansions_lprobs = model.get_normalized_probs_expansions(net_output, log_probs=True)
        expansions_lprobs = expansions_lprobs.view(-1, expansions_lprobs.size(-1))
        expansions_target = model.get_targets_expansions(sample, net_output).view(-1)
        expansions_loss = F.nll_loss(
            expansions_lprobs,
            expansions_target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        return token_loss, expansions_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        token_loss_sum = sum(log.get('token_loss', 0) for log in logging_outputs)
        expansion_loss_sum = sum(log.get('expansion_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_batch_tokens = sum(log.get('target_num_batch_tokens', 0) for log in logging_outputs)
        num_pad_tokens = sum(log.get('target_num_pad_tokens', 0) for log in logging_outputs)
        pad_ratio = float(num_pad_tokens) / num_batch_tokens
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'token_loss': token_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'expansion_loss': expansion_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'target_num_pad_tokens': num_pad_tokens,
            'target_num_non_pad_tokens': num_batch_tokens - num_pad_tokens,
            'target_num_batch_tokens': num_batch_tokens,
            'target_pad_ratio': pad_ratio,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
