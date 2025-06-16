#!/bin/bash

# Define models and datasets as arrays
models=("FacebookAI/roberta-base")
datasets=("wikitext:wikitext-2-raw-v1" "wikitext:wikitext-103-raw-v1")

# Iterate over each model and dataset combination
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    # Extract dataset name and config
    IFS=':' read -r dataset_name dataset_config <<< "$dataset"

    # Define output directories for each method
    base_output="/tmp/test-mlm-${model##*/}-base-${dataset_config}"
    lora_output="/tmp/test-mlm-${model##*/}-lora-${dataset_config}"
    lagembed_output="/tmp/test-mlm-${model##*/}-lagembed-${dataset_config}"

    # Define hyper-parameters for each method
    batch_size=4
    lora_rank=8
    # init_rank=8
    # target_rank=4
    in_channels=768
    n_components=2
    dof=3

    # Baseline: Training from scratch
    echo "Baseline: Training $model on $dataset_name ($dataset_config) from scratch" | tee -a mlm_output.log
    python run_mlm.py \
      --model_name_or_path $model \
      --dataset_name $dataset_name \
      --dataset_config_name $dataset_config \
      --do_train \
      --do_eval \
      --output_dir $base_output \
      --per_device_train_batch_size $batch_size \
      --max_steps 200 >> mlm_output.log 2>&1
    echo "------------------------------------" | tee -a mlm_output.log

    echo "LoRA: Fine-tuning $model on $dataset_name ($dataset_config) using baseline model" | tee -a mlm_output.log
    python run_mlm.py \
      --model_name_or_path $base_output \
      --dataset_name $dataset_name \
      --dataset_config_name $dataset_config \
      --use_lora \
      --lora_rank $lora_rank \
      --do_train \
      --do_eval \
      --output_dir $lora_output \
      --per_device_train_batch_size 4 \
      --max_steps 200 >> mlm_output.log 2>&1
    echo "------------------------------------" | tee -a mlm_output.log

    echo "LagEmbed: Training $model on $dataset_name ($dataset_config) with LagEmbed (in_channels=$in_channels, n_components=$n_components, dof=$dof)" | tee -a mlm_output.log
    python run_mlm.py \
      --model_name_or_path $base_output \
      --dataset_name $dataset_name \
      --dataset_config_name $dataset_config \
      --use_lagembed \
      --in_channels $in_channels \
      --n_components $n_components \
      --dof $dof \
      --do_train \
      --do_eval \
      --output_dir $lagembed_output \
      --per_device_train_batch_size $batch_size \
      --max_steps 200 >> mlm_output.log 2>&1
    echo "------------------------------------" | tee -a mlm_output.log
  done
done
