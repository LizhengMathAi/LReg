# Image classification reference training scripts

Except otherwise noted, all models have been trained on 4x RTX-A6000 GPUs with 
the following parameters:

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

