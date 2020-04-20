# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import math
import numpy as np

import torch
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


def accuracy(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, -1)
        correct = pred.view(-1).eq(target.view(-1))
    return correct.sum()


@register_criterion('masked_permutation_cross_entropy')
class MaskedPermutationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.mode = args.input_mode
        self.return_mlm = args.return_mlm
        if self.return_mlm is True:
            assert self.mode == 'mpnet'

    @staticmethod
    def add_args(parser):
        parser.add_argument('--return-mlm', default=False, action='store_true',
                            help='Return MLM loss in MPNet')
        
    def forward(self, model, sample, reduce=True):
        src_length = sample['net_input']['src_tokens'].size(1)
        targets = sample['targets']
        sample_size = targets.numel()

        logits = model.task_compute(
            task=self.mode,
            return_mlm=self.return_mlm,
            **sample['net_input'], 
        )

        if self.return_mlm is True:
            logits, mlm_logits = logits
        loss = self.compute_loss(logits, targets)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'acc': utils.item(accuracy(logits, targets))
        }
        if self.return_mlm is True:
            mlm_loss = self.compute_loss(mlm_logits, targets)
            logging_output.update(mlm_loss=utils.item(mlm_loss.data))
            logging_output.update(mlm_acc=utils.item(accuracy(mlm_logits, targets)))

            loss = loss + mlm_loss
        # ignore loss if no prediction
        loss *= sample['weight']
        return loss, sample_size, logging_output

    def compute_loss(self, logits, targets):
        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        acc = sum(log.get('acc', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'acc': acc / sample_size,
        }

        if 'mlm_loss' in logging_outputs[0]:
            mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs)
            mlm_acc = sum(log.get('mlm_acc', 0) for log in logging_outputs)
            agg_output['mlm_loss'] = mlm_loss / sample_size / math.log(2)
            agg_output['mlm_acc'] = mlm_acc / sample_size
        return agg_output
