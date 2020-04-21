# MPNet

[MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/pdf/2004.09297.pdf), by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu), is a novel pre-training method for language understanding tasks. It solves the problems of MLM (masked language modeling) in BERT and PLM (permuted language modeling) in XLNet and achieves better accuracy. 


## Supported Features
* A unified view and implementation of several pre-training models including BERT, XLNet, MPNet, etc.
* Code for pre-training and fine-tuning for a variety of language understanding (GLUE, SQuAD, RACE, etc) tasks.


## Installation

We implement MPNet and this pre-training toolkit based on the codebase of [fairseq](https://github.com/pytorch/fairseq). The installation is as follow:

```
pip install --editable pretraining/
pip install pytorch_transformers==1.0.0 transformers scipy sklearn
```


## Pre-training MPNet
Our model is pre-trained with bert dictionary, you first need to `pip install transformers` to use bert tokenizer. We provide a script [`encode.py`](MPNet/encode.py) and a dictionary file [`dict.txt`](MPNet/dict.txt) to tokenize your corpus. You can modify `encode.py` if you want to use other tokenizers (like roberta).

### 1) Preprocess data 
We choose [WikiText-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip) as a demo. The running script is as follow:

```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip

for SPLIT in train valid test; do \
    python MPNet/encode.py \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```

Then, we need to binarize data. The command of binarizing data is following:
```
fairseq-preprocess \
    --only-source \
    --srcdict MPNet/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```

### 2) Pre-train MPNet
The below command is to train a MPNet model:
```
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/wikitext-103

fairseq-train --fp16 $DATA_DIR \
    --task masked_permutation_lm --criterion masked_permutation_cross_entropy \
    --arch mpnet_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --input-mode 'mpnet'
```
**Notes**: You can replace arch with `mpnet_rel_base` and add command `--mask-whole-words --bpe bert` to use relative position embedding and whole word mask. 

**Notes**: You can specify `--input-mode` as `mlm` or `plm` to train **masked language model** or **permutation language model**.


## Pre-trained models
We provide a pre-trained [MPNet model](https://modelrelease.blob.core.windows.net/pre-training/MPNet/mpnet.example.pt) in BERT-base setting for you to have a try (which is only pre-trained for 125K steps). We will provide the final model with 500K training steps once the pre-training is finished.

You can load the pre-trained MPNet model like this: 
```python
from fairseq.models.masked_permutation_net import MPNet
mpnet = MPNet.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data', bpe='bert')
assert isinstance(mpnet.model, torch.nn.Module)
```


## Fine-tuning MPNet on down-streaming tasks

- [Fine-tuning on GLUE](MPNet/README.glue.md)
- [Fine-tuning on SQuAD](MPNet/README.squad.md)


## Acknowledgements
Our code is based on [fairseq-0.8.0](https://github.com/pytorch/fairseq). Thanks for their contribution to the open-source commuity.


## Reference
If you find this toolkit useful in your work, you can cite the corresponding papers listed below:

    @article{song2020mpnet,
        title={MPNet: Masked and Permuted Pre-training for Language Understanding},
        author={Song, Kaitao and Tan, Xu and Qin, Tao and Lu, Jianfeng and Liu, Tie-Yan},
        journal={arXiv preprint arXiv:2004.09297},
        year={2020}
    }

## Related Works
* [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/pdf/1905.02450.pdf), by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. GitHub: https://github.com/microsoft/MASS


* LightPAFF: A Two-Stage Distillation Framework for Pre-training and Fine-tuning, by Kaitao Song, Hao Sun, Xu Tan, Tao Qin, Jianfeng Lu, Hongzhi Liu, Tie-Yan Liu
