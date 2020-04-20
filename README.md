# MPNet

We implement MPNet and this pre-training toolkit based on the codebase of [fairseq](https://github.com/pytorch/fairseq). The installation is as follow:

```
pip install --editable pretraining/
pip install pytorch_transformers==1.0.0 transformers scipy sklearn
```


## Pre-training MPNet
Our model is pre-trained with bert dictionary, you first need to `pip install transformers` to use bert tokenizer. We provide a script `encode.py` and a dictionary file `dict.txt` to tokenize your corpus. You can modify `encode.py` if you want to use other tokenizers (like roberta).

### 1) Preprocess data 
We choose [WikiText-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip) as a demo. The running script is as follow:

```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip

for SPLIT in train valid test; do \
    python encode.py \
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
    --srcdict dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```

### 2) Pre-train MPNet
The below command is to train a MPNet model with relative positional embedding and whole word mask:
```
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/wikitext-103
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

fairseq-train --fp16 $DATA_DIR \
    --task masked_permutation_lm --criterion masked_permutation_cross_entropy \
    --arch mpnet_rel_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --mask-whole-words --bpe bert --bpe-vocab-file bert-base-uncased-vocab.txt \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --input-mode 'mpnet'
```
**Notes**: You can replace arch with `mpnet_base` and remove `--mask-whole-words --bpe bert --bpe-vocab-file bert-base-uncased-vocab.txt` to disable relative position embedding and whole word mask. 

**Notes**: You can specify `--input-mode` as `mlm` or `plm` to train **masked language model** or **permutation language model**.

### 3) Load the pre-trained model
```python
from fairseq.models.masked_permutation_net import MPNet
mpnet = MPNet.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data', bpe='bert')
assert isinstance(mpnet.model, torch.nn.Module)
```


## Fine-tuning MPNet on down-streaming tasks

We provide a pre-trained [MPNet model](https://modelrelease.blob.core.windows.net/pre-training/MPNet/mpnet.example.pt) in BERT-base setting for you to have a try (which is only pre-trained for 125K steps). We will provide the final model with 500K training steps once the pre-training is finished.

- [Fine-tuning on GLUE](README.glue.md)
- [Fine-tuning on SQuAD](README.squad.md)



# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
