# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os

import torch
import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
)

from transformers import BertTokenizer, squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from . import FairseqTask, register_task



class SQuADTokenizer(BertTokenizer):

    def __init__(self, vocab_file, dictionary, **kwargs):
        super().__init__(
            vocab_file=vocab_file,
            sep_token="</s>", 
            pad_token="<pad>", 
            cls_token="<s>", 
            mask_token="<mask>",
            **kwargs,
        )
        self.dictionary = dictionary

        self.max_len = 512
        self.max_len_single_sentence = self.max_len - 2
        self.max_len_sentences_pair = self.max_len - 3

    def tokenize(self, text, **kwargs):
        return super().tokenize(text, **kwargs)

    def _convert_token_to_id(self, token):
        return self.dictionary.index(token)

    def _convert_id_to_token(self, index):
        return self.dictionary.symbols[index]


@register_task('squad')
class SQuADTask(FairseqTask):
    '''
        TODO: Plan to release a version to huggingface 
    '''

    train_or_dev_file = {
        'train': "train-v1.1.json",
        'valid': "dev-v1.1.json",
    }

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-seq-len', default=512, type=int,
                            help="The maximum sequence length")
        parser.add_argument('--n-best-size', default=20, type=int,
                            help="The number of n-best predictions")
        parser.add_argument('--max-answer-length', default=30, type=int,
                            help="The maximum length of the generated answer")

    def __init__(self, args, dictionary):
        super().__init__(args)
        
        self.dictionary = dictionary
        self.seed = args.seed
        self.bpe = encoders.build_bpe(args)
        self.tokenizer = SQuADTokenizer(args.bpe_vocab_file, dictionary)
        self.do_evaluate = args.do_evaluate
        try:
            from transformers.data.processors.squad import SquadV1Processor
            self.processor = SquadV1Processor()
        except ImportError:
            raise ImportError(
                'Please install transformers with: pip install transformers'
            )

    @classmethod
    def load_dictionary(cls, filename):
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| Dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        cache = os.path.join(self.args.data, "cached_{}_{}_{}.pth".format(split, self.args.bpe, self.args.max_seq_len))

        if os.path.exists(cache):
            examples, features = torch.load(cache)
        else:
            if split == 'valid':
                examples = self.processor.get_dev_examples(self.args.data, self.train_or_dev_file[split])
            else:
                examples = self.processor.get_train_examples(self.args.data, self.train_or_dev_file[split])

            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.args.max_seq_len,
                doc_stride=128,
                max_query_length=64,
                is_training=(split != 'valid'),
                return_dataset=False,
            )

            if self.args.distributed_rank == 0:
                torch.save((examples, features), cache)

        if split == 'valid' and self.do_evaluate:
            self.examples = examples
            self.features = features
        
        src_dataset = BaseWrapperDataset([np.array(f.input_ids) for f in features])
        starts = BaseWrapperDataset(np.array([f.start_position for f in features]))
        ends = BaseWrapperDataset(np.array([f.end_position for f in features]))
        sizes = np.array([len(f.input_ids) for f in features])
        src_lengths = NumelDataset(src_dataset, reduce=False)


        '''
            Input format: <s> question here ? </s> Passage </s>
        '''

        dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,    
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
                'targets': {
                    'starts': starts,
                    'ends': ends,
                },
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_dataset, reduce=True),
            },
            sizes=[sizes],
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                sort_order=[np.random.permutation(len(dataset))],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        
        model.register_question_answer_head(
            'question_answer_head',
            num_classes=2,
        )
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def compute_predictions_logits(self, all_results, prefix=""):
        output_prediction_file = os.path.join(self.args.save_dir, "predictions.json")
        output_nbest_file = os.path.join(self.args.save_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(self.args.save_dir, "null_odds.json")

        predictions = compute_predictions_logits(
            self.examples,
            self.features,
            all_results,
            self.args.n_best_size,
            self.args.max_answer_length,
            True,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            False,
            False,
            0.0,
            self.tokenizer,
        )
        results = squad_evaluate(self.examples, predictions)
        return results
