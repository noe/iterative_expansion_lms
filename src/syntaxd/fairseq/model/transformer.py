from argparse import ArgumentParser
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.masked_lm import MaskedLMModel
from fairseq.modules import LayerNorm

from syntaxd.fairseq.modules.transformer_sentence_encoder import TransformerSentenceEncoder


@register_model('expansion_transformer_lm')
class ExpansionTransformerLM(MaskedLMModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.is_causal = args.causal
        self.is_semicausal = args.semi_causal

    @staticmethod
    def add_args(parser: ArgumentParser):
        MaskedLMModel.add_args(parser)
        causal_parser = parser.add_mutually_exclusive_group(required=True)
        causal_parser.add_argument('--causal', dest='causal', action='store_true')
        causal_parser.add_argument('--no-causal', dest='causal', action='store_false')
        causal_parser.add_argument('--semi-causal', dest='semi_causal', action='store_true')
        parser.set_defaults(causal=True)
        parser.set_defaults(semi_causal=False)
        parser.add_argument('--head-embeddings', action='store_true')
        parser.add_argument('--condition-token-on-expansion', action='store_true')
        parser.add_argument('--expansion-layer', type=int, default=-1)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        args.num_segment = 0

        if not hasattr(args, 'max_positions'):
            args.max_positions = 1024 #args.tokens_per_sample

        # print("Model args: {}".format(args), file=sys.stderr)

        encoder = ExpansionLMEncoder(args,
                                     task.token_dictionary,
                                     task.expansion_dictionary)

        return cls(args, encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        raise NotImplementedError

    def get_targets(self, sample, net_output):
        raise NotImplementedError

    def get_normalized_probs_tokens(self, net_output, log_probs, sample=None):
        token_logits = net_output[0].float()
        return (F.log_softmax(token_logits, dim=-1) if log_probs
                else F.softmax(token_logits, dim=-1))

    def get_targets_tokens(self, sample, net_output):
        return sample['target_tokens']

    def get_normalized_probs_expansions(self, net_output, log_probs, sample=None):
        expansion_logits = net_output[1].float()
        return (F.log_softmax(expansion_logits, dim=-1) if log_probs
                else F.softmax(expansion_logits, dim=-1))

    def get_targets_expansions(self, sample, net_output):
        return sample['target_expans']

    def forward(self,
                previous_level_tokens,
                causality_mask,
                head_positions,
                next_level_expansions=None,
                expansion_sampling=None,
                **unused):

        device = next(self.parameters()).device
        use_half = next(p for p in self.parameters() if p.is_floating_point()).dtype == torch.half

        float_type = torch.half if use_half else torch.float

        if self.is_causal:
            zeros = torch.zeros_like(causality_mask, dtype=float_type)
            zeros.masked_fill_(causality_mask, float('-inf'))
            causality_mask = zeros.to(device)
            forced_head_mask = None
        elif self.is_semicausal:
            zeros = torch.zeros_like(causality_mask, dtype=float_type)
            zeros.masked_fill_(causality_mask, float('-inf'))
            forced_head_mask = zeros.to(device)
            causality_mask = None
        else:
            causality_mask = None
            forced_head_mask = None

        return self.encoder(previous_level_tokens,
                            causality_mask,
                            forced_head_mask,
                            head_positions,
                            next_level_expansions,
                            expansion_sampling)


class ExpansionLMEncoder(FairseqEncoder):
    def __init__(self,
                 args,
                 dictionary: Dictionary,
                 expansion_dictionary: Dictionary):
        super().__init__(dictionary)

        self.expansion_dictionary = expansion_dictionary
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.expansion_vocab_size = expansion_dictionary.__len__()
        self.max_positions = args.max_positions
        self.use_head_embeddings = getattr(args, 'head_embeddings', False)

        assert args.num_segment == 0, "Number of segments must be 0"
        assert not args.no_token_positional_embeddings, "Positional embedding must be enabled"
        assert not args.sent_loss, "Sentence loss must be disabled"

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            add_bias_kv=args.bias_kv,
            add_zero_attn=args.zero_attn,
            use_head_embeddings=self.use_head_embeddings,
            condition_token_on_expansion=getattr(args, 'condition_token_on_expansion', False),
            expansion_layer=getattr(args, 'expansion_layer', -1),
            expansion_vocab_size=self.expansion_vocab_size,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None

        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.token_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

        if not self.share_input_output_embed:
            self.embed_out = nn.Linear(
                args.encoder_embed_dim,
                self.vocab_size,
                bias=False
            )

    def forward(self,
                previous_level_tokens,
                causality_mask,
                forced_head_mask,
                head_positions,
                next_level_expansions,
                expansion_sampling,
                **unused):

        (inner_states,
         expansion_logits,
         expansion_ids) = self.sentence_encoder(previous_level_tokens,
                                                self_attn_mask=causality_mask,
                                                forced_head_mask=forced_head_mask,
                                                head_positions=head_positions,
                                                next_level_expansions=next_level_expansions,
                                                expansion_sampling=expansion_sampling,
                                                segment_labels=None)

        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed \
                and hasattr(self.sentence_encoder.embed_tokens, 'weight'):
            token_logits = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        else:
            assert self.embed_out is not None
            token_logits = self.embed_out(x)

        token_logits = token_logits + self.token_output_learned_bias

        # logits are in format B x T x C
        return token_logits, expansion_logits, expansion_ids

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions


@register_model_architecture('expansion_transformer_lm', 'expansion_transformer_lm')
def base_architecture(args):
    from fairseq.models.masked_lm import base_architecture as fairseq_base_architecture
    fairseq_base_architecture(args)
