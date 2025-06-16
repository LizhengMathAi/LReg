# LReg

## Cifar Ablation Study
An untrainable encoder with a universal architecture across various types of raw data and recognition tasks. 

* Run `python utils.py` to reproduce experiments of the data fitting task.
* Run `python vision.py` to reproduce experiments of image classification and super-resolution tasks.
* Run `python text.py` to reproduce experiments of the text classification task.

In the upgraded version, the entire data pipeline is now parallelized on the GPU.
* Run `python utilsv2.py` to reproduce experiments of the data fitting task.
* Run `python visionv2.py` to reproduce experiments of image classification and super-resolution tasks.
* Run `python textv2.py` to reproduce experiments of the text classification task.

A demonstration of how to use LagrangeEmbedding as a plug-and-play module to enhance model performance is provided below:
* Run `python cifar_ablation_study.py` to reproduce experiments of image classification tasks.

# Image classification reference training scripts

Except otherwise noted, all models have been trained on 4x RTX-A6000 GPUs with the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `1`    |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |


### ResNet
```
torchrun --nproc_per_node=4 train.py \
      --model $MODEL \
      --n-components 32 \
      --dof 8 \
      --batch-size 32 \
      --epochs 1 \
      --weights ResNet50_Weights.IMAGENET1K_V1 \
      --lr 0.01
```

Here `$MODEL` is one of `resnet18`, `resnet34`, `resnet50`, `resnet101` or `resnet152`.

### ResNext
```
torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet --output-dir resnext50_32x4d 
      --model resnext50_32x4d \
      --n-components 32 \
      --dof 8 \
      --batch-size 32 \
      --epochs 1 \
      --weights ResNeXt50_32X4D_Weights.IMAGENET1K_V1 \
      --lr 0.01
```

Here `$MODEL` is one of `resnext50_32x4d` or `resnext101_32x8d`.

### MobileNetV2
```
torchrun --nproc_per_node=4 train.py \
    --model mobilenet_v2 \
    --n-components 32 \
    --dof 16 \
    --batch-size 32 \
    --epochs 1 \
    --weights MobileNet_V2_Weights.IMAGENET1K_V1 \
    --lr 0.01
```

#### vit_b_16
```
torchrun --nproc_per_node=4 train.py \
    --model vit_b_16 \
    --n-components 8 \
    --dof 32 \
    --batch-size 32 \
    --epochs 1 \
    --weights ViT_B_16_Weights.IMAGENET1K_V1 \
    --lr 0.0003 \
    --weight-decay 0.00002 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --ra-sampler \
    --ra-reps 12 \
    --cutmix-alpha 1.0 \
    --model-ema
```

## Text classification 
### Reproducing the Baseline

```bash
export TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --use_lreg \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --learning_rate 4e-4 \
  --num_train_epochs 30 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

### Reproducing Our Results
Add the `--use_lreg` flag to the baseline command above.

## Language Modeling

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling (CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM) loss. 

### GPT-2/GPT and causal language modeling

The following example fine-tunes (train from scratch) GPT2-base, GPT2-medium, and GPT2-large on WikiText-2/WikiText-103. We're using the raw WikiText-2 and WikiText-103 (no tokens were replaced before the tokenization). The loss here is that of causal language modeling.

```bash
bash run_clm.sh
```

The following example fine-tunes (transfer learning) GPT2-base, GPT2-medium, and GPT2-large on the fusion of WikiText-2 and WikiText-103.

```bash
bash run_clm_transfer.sh
```

It reaches a score of 13~22 perplexity once fine-tuned on the dataset.


### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes (train from scratch) RoBERTa-base on WikiText-2/WikiText-103. Here too, we're using the raw WikiText-2/WikiText-103. The loss is different as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore, converge slightly slower (over-fitting takes more epochs).

```bash
bash run_mlm.sh
```

The following example fine-tunes (transfer learning) RoBERTa-base on the fusion of WikiText-2 and WikiText-103.

```bash
bash run_mlm_transfer.sh
```
