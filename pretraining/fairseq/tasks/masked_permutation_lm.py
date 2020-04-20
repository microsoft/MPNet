# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.permutation_utils import make_span_perm


@register_task('masked_permutation_lm')
class MaskedPermutationLMTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--pred-prob', default=0.15, type=float,
                            help='probability for tokens prediction')
        parser.add_argument('--rand-prob', default=0.10, type=float,
                            help='probability for random input')
        parser.add_argument('--keep-prob', default=0.10, type=float,
                            help='probability for keep input')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--input-mode', default='mpnet', choices=['mlm', 'plm', 'mpnet'], 
                            help='Choose the input format for different tasks')
        parser.add_argument('--max-gram', default=1, type=int,
                            help='The maximum n-gram for whole word mask. It is setup with --max-whole-words')

    
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.mask_idx = dictionary.add_symbol('<mask>')
        self.dictionary = dictionary
        self.seed = args.seed
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
    
        if self.args.mask_whole_words:
            bpe = encoders.build_bpe(self.args)
            if bpe is not None:

                def is_beginning_of_word(i):
                    if i < self.source_dictionary.nspecial:
                        # special elements are always considered beginnings
                        return True
                    tok = self.source_dictionary[i]
                    if tok.startswith('madeupword'):
                        return True
                    try:
                        return bpe.is_beginning_of_word(tok)
                    except ValueError:
                        return True

                mask_whole_words = torch.ByteTensor(list(
                    map(is_beginning_of_word, range(len(self.source_dictionary)))
                ))
        else:
            mask_whole_words = None

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        print('| loaded {} batches from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        src_dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            MaskedDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
                dictionary=self.dictionary,
                args=self.args,
                mask_whole_words=mask_whole_words,
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class MaskedDataset(NestedDictionaryDataset):
    def __init__(self, defn, sizes=None, dictionary=None, args=None, mask_whole_words=None):
        super().__init__(defn, sizes)
        self.mask_idx = dictionary.add_symbol('<mask>')
        self.pred_prob  = args.pred_prob
        self.keep_prob  = args.keep_prob
        self.rand_prob  = args.rand_prob
        self.vocab = dictionary

        weights = np.ones(len(self.vocab))
        weights[:self.vocab.nspecial] = 0
        self.weights = weights / weights.sum()

        self.mask_whole_words = mask_whole_words
        self.input_mode = args.input_mode

        self.max_gram = args.max_gram
        
        # Generate a static n-gram template for faster training
        mask_ngram = dict()
        for i in range(1, args.tokens_per_sample + 1):
            template = []
            r = i
            for j in range(self.max_gram, 1, -1):
                cnt = int(i / self.max_gram / j)
                template.extend([j for _ in range(cnt)])
                r = r - cnt * j
            template.extend([1 for _ in range(r)])

            mask_ngram[i] = np.array(template)

        self.mask_ngram = mask_ngram

    def collater(self, samples):
        samples = super().collater(samples)

        if len(samples) == 0:
            return {}

        src_tokens = samples['net_input']['src_tokens']
        sz = src_tokens.size()
        pred_size = round(sz[1] * self.pred_prob)
        samples['weight'] = 0.0 if pred_size == 0 else 1.0
        if pred_size == 0:
            pred_size = sz[1]
        
        if self.mask_whole_words is not None:
            positions = torch.stack([self.span_perm(src_tokens[i], pred_size) for i in range(sz[0])])
        else:
            positions = torch.stack([torch.randperm(sz[1]) for i in range(sz[0])])
        src_tokens, targets = self.permute_inputs(src_tokens, positions), None

        mask_range = range(sz[1] - pred_size, sz[1])
        if self.input_mode == 'mlm':
            targets = src_tokens[:, mask_range].contiguous()
            src_tokens[:, mask_range] = self.mask_perm(targets.clone(), self.mask_idx)
        elif self.input_mode == 'plm':
            # PLM does not use 8:1:1 ? 
            targets = src_tokens[:, mask_range].contiguous()
            src_tokens = torch.cat((src_tokens, torch.full_like(targets, self.mask_idx)), dim=1)
            positions = torch.cat((positions, positions[:, mask_range]), dim=1)
        else:
            targets = src_tokens[:, mask_range].contiguous()
            masked_tokens = self.mask_perm(targets.clone(), self.mask_idx)
            src_tokens = torch.cat((src_tokens, masked_tokens, masked_tokens), dim=1)
            positions = torch.cat((positions, positions[:, mask_range], positions[:, mask_range]), dim=1)

        samples['targets'] = targets
        samples['net_input']['positions'] = positions
        samples['net_input']['src_tokens'] = src_tokens
        samples['net_input']['pred_size'] = targets.size(1)
       
        return samples

    def span_perm(self, x, pred_size=None):
        # Permutation for span mask, faster than fairseq original implementation
        word_begins_mask = self.mask_whole_words.gather(0, x)
        word_begins_idx = word_begins_mask.nonzero().view(-1).tolist()
        
        if self.max_gram == 1: 
            # Only whole word mask, slightly faster than using n-gram and hardly affect accuracy
            ids = word_begins_idx
        else:
            sz = len(word_begins_idx)
            ngram = self.mask_ngram[sz].copy()
            np.random.shuffle(ngram)
            i, ids = 0, []

            for g in ngram:
                ids.append(word_begins_idx[i])
                i = i + g

        sz = len(ids)
        perm = np.random.permutation(sz)
        ids.append(x.size(0))
        
        span_perm = make_span_perm(perm, ids, x.size(0))
        if pred_size is not None:
            # Shuffle Predicted Part again for 
            np.random.shuffle(span_perm[-pred_size:])
        return torch.from_numpy(span_perm)

    def mask_perm(self, tokens, mask_idx=None):
        if mask_idx is None:
            mask_idx = self.mask_idx
        mask_prob = 1.0 - self.rand_prob - self.keep_prob
        mask_indices = torch.bernoulli(torch.full(tokens.shape, mask_prob)).bool()
        random_indices = torch.bernoulli(torch.full(tokens.shape, self.rand_prob / (1.0 - mask_prob))).bool() & ~mask_indices
        tokens[mask_indices] = mask_idx
        tokens[random_indices] = self.generate_random_tensor(random_indices.sum().tolist()).to(tokens.device)
        return tokens

    def make_perm(self, sz, pred_size):
        perm = torch.randperm(sz)
        perm[:sz - pred_size] = perm[:sz - pred_size].sort()[0]
        return perm

    def permute_inputs(self, inputs, positions):
        sz = inputs.size()
        offset = torch.arange(0, sz[0] * sz[1], sz[1])
        index = positions + offset.unsqueeze_(1)
        return inputs.reshape(-1)[index]

    def generate_random_tensor(self, sz):
        return torch.from_numpy(
            np.random.choice(len(self.vocab), sz, p=self.weights)        
        )
