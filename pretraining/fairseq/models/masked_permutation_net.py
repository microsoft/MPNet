# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from fairseq import utils
from fairseq.models.roberta import (
    RobertaModel, 
    RobertaLMHead,
    roberta_base_architecture,
    roberta_large_architecture,
)
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


@register_model('mpnet')
class MPNet(RobertaModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    def task_compute(self, task='mlm', **kwargs):
        if task == 'mlm':
            return self.compute_mlm(**kwargs)
        elif task == 'plm':
            return self.compute_plm(**kwargs)
        else:
            return self.compute_mpnet(**kwargs)

    def compute_mlm(self, src_tokens, src_lengths, positions, pred_size, **kwargs):
        sz = src_tokens.size(1)
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions)
        x = reverse_tensor(emb)
        positions_bias = self.encode_relative_emb(self.decoder.sentence_encoder, positions)
        for layer in self.decoder.sentence_encoder.layers:
            x, _ = layer(x, positions_bias=positions_bias)
        x = self.maybe_final_norm(self.decoder.sentence_encoder, x)
        x = reverse_tensor(x)
        x = self.output_layer(x[:, sz-pred_size:])
        return x

    def compute_plm(self, src_tokens, src_lengths, positions, pred_size, **kwargs):
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions)
        x = reverse_tensor(emb)
        c, q = split_tensor(x, pred_size)
        content_position_bias = self.encode_relative_emb(
            self.decoder.sentence_encoder, positions[:, :-pred_size]
        )
        if content_position_bias is not None:
            query_position_bias = content_position_bias[:, -pred_size:].contiguous()
        else:
            query_position_bias = None

        sz = c.size(0)
        query_mask, content_mask = make_query_and_content_mask(src_tokens, sz, pred_size, kind='PLM')
        for i, layer in enumerate(self.decoder.sentence_encoder.layers):
            c, q = encode_two_stream_attn(
                layer, c, q, content_mask, query_mask, content_position_bias, query_position_bias,
            )

        q = self.maybe_final_norm(self.decoder.sentence_encoder, q)
        q = reverse_tensor(q)
        x = self.output_layer(q)
        return x

    def compute_mpnet(self, src_tokens, src_lengths, positions, pred_size, return_mlm=False, **kwargs):
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions)
        x = reverse_tensor(emb)
        c, q = split_tensor(x, pred_size)

        content_position_bias = self.encode_relative_emb(self.decoder.sentence_encoder, positions[:, :-pred_size])
        if content_position_bias is not None:
            query_position_bias = content_position_bias[:, -pred_size:].contiguous()
        else:
            query_position_bias = None

        sz = c.size(0) - pred_size
        query_mask, content_mask = make_query_and_content_mask(src_tokens, sz, pred_size)
        for i, layer in enumerate(self.decoder.sentence_encoder.layers):
            c, q = encode_two_stream_attn(
                layer, c, q, content_mask, query_mask, content_position_bias, query_position_bias,
            )
        
        q = self.maybe_final_norm(self.decoder.sentence_encoder, q)
        q = reverse_tensor(q)
        x = self.output_layer(q)

        if return_mlm is True:
            c = c[-pred_size:]
            c = self.maybe_final_norm(self.decoder.sentence_encoder, c)
            c = reverse_tensor(c)
            c = self.output_layer(c)
            return x, c
         
        return x

    @staticmethod
    def encode_emb(self, src_tokens, positions=None):
        x = self.embed_tokens(src_tokens)
        if self.embed_scale is not None:
            x *= self.embed_scale
        if positions is not None:
            x += F.embedding(positions + 2, self.embed_positions.weight, self.padding_idx)
        if self.emb_layer_norm is not None and not self.normalize_before:
            x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @staticmethod
    def maybe_final_norm(self, x):
        if self.emb_layer_norm is not None and self.normalize_before:
            return self.emb_layer_norm(x)
        return x

    @staticmethod
    def encode_relative_emb(self, positions):
        if not self.relative_attention_bias:
            return None
        qlen, klen = positions.size(1), positions.size(1)
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        
        relative_position = memory_position - context_position
        
        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(positions.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(0, 3, 1, 2).contiguous() # [bsz, head, qlen, klen]
        values = values.view(-1, qlen, klen)
        return values


def reverse_tensor(x):
    return x.transpose(0, 1)


def split_tensor(x, split_size):
    sz = x.size(0) - split_size
    return x[:sz].contiguous(), x[sz:].contiguous()


def encode_two_stream_attn(
    self, 
    c, 
    q, 
    content_mask: torch.Tensor = None, 
    query_mask: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
):
    def reuse_fn(x, residual):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
    
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x
    
    residual_c = c
    residual_q = q
    
    c = self.maybe_layer_norm(self.self_attn_layer_norm, c, before=True)
    q = self.maybe_layer_norm(self.self_attn_layer_norm, q, before=True)

    c, q = two_stream_self_attention(
        self.self_attn,
        query=[c, q],
        key=c,
        value=c,
        query_mask=query_mask,
        content_mask=content_mask,
        query_position_bias=query_position_bias,
        content_position_bias=content_position_bias,
    )

    c = reuse_fn(c, residual_c)
    q = reuse_fn(q, residual_q)
    return c, q


def two_stream_self_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    query_mask: torch.Tensor = None,
    content_mask: torch.Tensor = None,
    query_position_bias: torch.Tensor = None,
    content_position_bias: torch.Tensor = None,
):
    c, q = query
    bsz, embed_dim = key.size(1), key.size(2)

    def transpose_fn(x):
        return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def fill_mask(attn_weights, attn_mask):
        return attn_weights.masked_fill(
            attn_mask.unsqueeze(0),
            float('-inf')
        )

    def attn_fn(_q, k, v, mask=None, bias=None):
        _q = transpose_fn(self.scaling * self.in_proj_q(_q))
        attn_weights = torch.bmm(_q, k.transpose(1, 2))
        if bias is not None:
            attn_weights += bias
        if mask is not None:
            attn_weights = fill_mask(attn_weights, mask)
        attn_weights = utils.softmax(
            attn_weights, dim=-1,        
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        return self.out_proj(attn)


    k = transpose_fn(self.in_proj_k(key))
    v = transpose_fn(self.in_proj_v(value))

    c = attn_fn(c, k, v, mask=content_mask, bias=content_position_bias)
    q = attn_fn(q, k, v, mask=query_mask, bias=query_position_bias)
    return c, q


def make_query_and_content_mask(tensor, a, b, kind='MPLM'):
    '''
        Query Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        Content Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
                               x x x x x x x m m m
                               1 2 3 4 5 6 7 5 6 7
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        [ 0 0 0 0 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]

    '''

    def make_query_mask():
        mask = torch.triu(torch.ones(b, b), 0)
        mask = (torch.ones(b, a - b), 1 - mask) if kind is 'PLM' else (torch.ones(b, a - b), 1 - mask, mask)
        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask():
        mask = [torch.zeros(a - b, b), torch.tril(torch.ones(b, b), 0)]
        if kind is not 'PLM':
            mask.append(torch.zeros(b, b))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(a, a - b), mask) if kind is 'PLM' else (torch.ones(a + b, a - b), mask, 1 - mask)
        return torch.cat(mask, dim=-1).eq(0)

    return make_query_mask().to(tensor.device), make_content_mask().to(tensor.device)
  

@register_model_architecture('mpnet', 'mpnet_base')
def mpnet_base_architecture(args):
    roberta_base_architecture(args)


@register_model_architecture('mpnet', 'mpnet_rel_base')
def mpnet_rel_base_architecture(args):
    args.use_relative_positions = getattr(args, 'use_relative_positions', True)
    mpnet_base_architecture(args)


@register_model_architecture('mpnet', 'mpnet_large')
def mpnet_large_architecture(args):
    roberta_large_architecture(args)
