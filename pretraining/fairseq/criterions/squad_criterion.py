# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('squad_criterion')
class SQuADCriterion(FairseqCriterion):
    
    def __init__(self, args, task):
        super().__init__(args, task)
        self.do_evaluate = args.do_evaluate
 
    def forward(self, model, sample, reduce=True):
        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='question_answer_head',
        )

        import IPython
        IPython.embed()
        exit()

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_positions = sample['targets']['starts']
        end_positions = sample['targets']['ends']

        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        
        start_loss = self.compute_loss(start_logits, start_positions)
        end_loss = self.compute_loss(end_logits, end_positions)

        loss = (start_loss + end_loss) / 2

        sample_size = sample['nsentences']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        if self.do_evaluate:
            logging_output.update(starts=start_logits.detach())
            logging_output.update(ends=end_logits.detach())
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if 'starts' in logging_outputs[0]:
            agg_output["starts"] = logging_outputs[0]['starts']
            agg_output["ends"]   = logging_outputs[0]['ends']
        return agg_output

    def compute_loss(self, logits, targets):
        return F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
        )
