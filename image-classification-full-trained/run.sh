# !/bin/bash

n_components_values=(16)
dof_values=(16)
for n_components in "${n_components_values[@]}"
do
  for dof in "${dof_values[@]}"
  do
    echo "Running experiment with n_components=$n_components and dof=$dof" | tee -a output.log
    
    # Run the torchrun command and append output to output.log
    torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet \
      --output-dir mobilenet_v2 \
      --model mobilenet_v2 \
      --n-components $n_components \
      --dof $dof \
      --batch-size 32 \
      --epochs 5 \
      --weights MobileNet_V2_Weights.IMAGENET1K_V1 \
      --lr 0.01 \
      --print-freq 1000 >> output.log 2>&1
    
    echo "Experiment with model=mobilenet_v2, n_components=$n_components, and dof=$dof completed" | tee -a output.log
    echo "-----------------------------------" | tee -a output.log
  done
done


n_components_values=(16)
dof_values=(16)
for n_components in "${n_components_values[@]}"
do
  for dof in "${dof_values[@]}"
  do
    echo "Running experiment with n_components=$n_components and dof=$dof" | tee -a output.log
    
    # Run the torchrun command and append output to output.log
    torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet \
      --output-dir resnet50 \
      --model resnet50 \
      --n-components $n_components \
      --dof $dof \
      --batch-size 32 \
      --epochs 5 \
      --weights ResNet50_Weights.IMAGENET1K_V1 \
      --lr 0.01 \
      --print-freq 1000 >> output.log 2>&1
    
    echo "Experiment with model=resnet50, n_components=$n_components, and dof=$dof completed" | tee -a output.log
    echo "-----------------------------------" | tee -a output.log
  done
done


n_components_values=(4)
dof_values=(16)
for n_components in "${n_components_values[@]}"
do
  for dof in "${dof_values[@]}"
  do
    echo "Running experiment with n_components=$n_components and dof=$dof" | tee -a output.log
    
    # Run the torchrun command and append output to output.log
    torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet \
      --output-dir resnext50_32x4d \
      --model resnext50_32x4d \
      --n-components $n_components \
      --dof $dof \
      --batch-size 32 \
      --epochs 5 \
      --weights ResNeXt50_32X4D_Weights.IMAGENET1K_V1 \
      --lr 0.001 \
      --print-freq 1000 >> output.log 2>&1
    
    echo "Experiment with model=resnext50_32x4d, n_components=$n_components, and dof=$dof completed" | tee -a output.log
    echo "-----------------------------------" | tee -a output.log
  done
done


n_components_values=(4)
dof_values=(16)
for n_components in "${n_components_values[@]}"
do
  for dof in "${dof_values[@]}"
  do
    echo "Running experiment with n_components=$n_components and dof=$dof" | tee -a output.log
    
    # Run the torchrun command and append output to output.log
    torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet \
      --output-dir vit_b_16 \
      --model vit_b_16 \
      --n-components $n_components \
      --dof $dof \
      --batch-size 32 \
      --epochs 5 \
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
      --print-freq 1000 >> output.log 2>&1
    
    echo "Experiment with model=vit_b_16, n_components=$n_components, and dof=$dof completed" | tee -a output.log
    echo "-----------------------------------" | tee -a output.log
  done
done


n_components_values=(4 8 16 32)
dof_values=(4 8 16 32)
for n_components in "${n_components_values[@]}"
do
  for dof in "${dof_values[@]}"
  do
    echo "Running experiment with n_components=$n_components and dof=$dof" | tee -a output.log
    
    # Run the torchrun command and append output to output.log
    torchrun --nproc_per_node=4 train.py \
      --data-path /home/cs/Documents/datasets/imagenet \
      --output-dir resnet50 \
      --model resnet50 \
      --n-components $n_components \
      --dof $dof \
      --batch-size 32 \
      --epochs 5 \
      --weights ResNet50_Weights.IMAGENET1K_V1 \
      --lr 0.01 \
      --print-freq 1000 >> output.log 2>&1
    
    echo "Experiment with model=resnet50, n_components=$n_components, and dof=$dof completed" | tee -a output.log
    echo "-----------------------------------" | tee -a output.log
  done
done