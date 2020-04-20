# Fine-tuning MPNet on SQuAD tasks
We provide a simple demo to fine-tune SQuAD on fairseq. The current version can only support bert tokenizer, while you can modify [this part](https://github.com/microsoft/pretraining/blob/a1609f1697e6ec8508429bc9a507b408a3854beb/pretraining/fairseq/tasks/squad2.py#L28) to support your own tokenizer (e.g., roberta). We will release a huggingface version in the future to support more complex functions for fine-tuning.

### 1) Download the data from SQuAD website using following commands:
```bash
mkdir -p SQuAD/v2.0
cd SQuAD/v2.0
wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

```

### 2) Fine-tuning SQuAD
The below command presents us how to fine-tune SQuAD. You can deploy it with 8 GPUs to imitate a batch size of 48:
```
DATA_BIN=SQuAD/v2.0
MPNET_PATH=/path/to/mpnet/model.pt

fairseq-train $DATA_BIN \
    --max-positions 512 \
    --max-sentences 6 \
    --task squad2 \
    --bpe bert --bpe-vocab-file $DATA_BIN/bert-base-uncased-vocab.txt \
    --required-batch-size-multiple 1 \
    --arch mpnet_base \
    --criterion squad_criterion \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2.0e-5 --total-num-update 5430 --warmup-updates 326 \
    --update-freq 1 \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
    --max-epoch 4 --ddp-backend=no_c10d  \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $MPNET_PATH \
```

### 3) Evaluate SQuAD
Currently, we need to evaluate SQuAD after training and this evaluation is only supported on single GPU. We will refine this function in the future.
```
DATA_BIN=SQuAD/v2.0
MPNET_PATH=/path/to/mpnet/squad_model.pt

fairseq-train $DATA_BIN \
    --max-positions 512 \
    --max-sentences 8 \
    --restore-file $MPNET_PATH \
    --task squad2 \
    --bpe bert --bpe-vocab-file $DATA_BIN/bert-base-uncased-vocab.txt \
    --required-batch-size-multiple 1 \
    --arch mpnet_base \
    --criterion squad_criterion \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2.0e-5 --total-num-update 5430 --warmup-updates 326 \
    --update-freq 1 \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 1 --ddp-backend=no_c10d  \
    --do-evaluate --reset-optimizer
```
