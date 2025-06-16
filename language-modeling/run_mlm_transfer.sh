#!/bin/bash

# Define models and datasets as arrays
models=("FacebookAI/roberta-base")

# Iterate over each model and dataset combination
for model in "${models[@]}"; do
    # Define output directories for each method
    lora_output="/tmp/test-mlm-transfer-${model##*/}-lora"
    lagembed_output="/tmp/test-mlm-transfer-${model##*/}-lagembed"

    # Define hyper-parameters for each method
    batch_size=4
    lora_rank=8
    # init_rank=8
    # target_rank=4
    in_channels=768
    n_components=2
    dof=2

    echo "LoRA: Fine-tuning $model on wikitext-2-raw-v1 and wikitext-103-raw-v1 using baseline model" | tee -a mlm_transfer_output.log
    python run_mlm_transfer.py \
      --model_name_or_path "/tmp/test-mlm-${model##*/}-base-wikitext-2-raw-v1" \
      --use_lora \
      --lora_rank $lora_rank \
      --do_train \
      --do_eval \
      --output_dir $lora_output \
      --per_device_train_batch_size 4 \
      --max_steps 200 >> mlm_transfer_output.log 2>&1
    echo "------------------------------------" | tee -a mlm_transfer_output.log

    echo "LagEmbed: Training $model on wikitext-2-raw-v1 and wikitext-103-raw-v1 with LagEmbed (in_channels=$in_channels, n_components=$n_components, dof=$dof)" | tee -a mlm_transfer_output.log
    python run_mlm_transfer.py \
      --model_name_or_path "/tmp/test-mlm-${model##*/}-base-wikitext-2-raw-v1" \
      --use_lagembed \
      --in_channels $in_channels \
      --n_components $n_components \
      --dof $dof \
      --do_train \
      --do_eval \
      --output_dir $lagembed_output \
      --per_device_train_batch_size $batch_size \
      --max_steps 200 >> mlm_transfer_output.log 2>&1
    echo "------------------------------------" | tee -a mlm_transfer_output.log
done
