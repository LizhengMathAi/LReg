#!/bin/bash

# Define models and datasets as arrays
models=("openai-community/gpt2" "openai-community/gpt2-medium" "openai-community/gpt2-large")

# Iterate over each model and dataset combination
for model in "${models[@]}"; do
    # Define output directories for each method
    lora_output="/tmp/test-clm-transfer-${model##*/}-lora"
    lagembed_output="/tmp/test-clm-transfer-${model##*/}-lagembed"

    # Define hyper-parameters for each method
    if [ "$model" == "openai-community/gpt2" ]; then
      batch_size=8
      lora_rank=8
      init_rank=8
      target_rank=4
      in_channels=768
      n_components=2
      dof=2
    elif [ "$model" == "openai-community/gpt2-medium" ]; then
      batch_size=8
      lora_rank=4
      init_rank=4
      target_rank=4
      in_channels=1024
      n_components=2
      dof=4
    elif [ "$model" == "openai-community/gpt2-large" ]; then
      batch_size=4
      lora_rank=4
      init_rank=4
      target_rank=4
      in_channels=1280
      n_components=3
      dof=4
    fi

    echo "LoRA: Fine-tuning $model on wikitext-2-raw-v1 and wikitext-103-raw-v1 using baseline model" | tee -a clm_transfer_output.log
    python run_clm_transfer.py \
      --model_name_or_path "/tmp/test-clm-${model##*/}-base-wikitext-2-raw-v1" \
      --use_lora \
      --lora_rank $lora_rank \
      --do_train \
      --do_eval \
      --output_dir $lora_output \
      --per_device_train_batch_size 4 \
      --max_steps 200 >> clm_transfer_output.log 2>&1
    echo "------------------------------------" | tee -a clm_transfer_output.log

    echo "LagEmbed: Training $model on wikitext-2-raw-v1 and wikitext-103-raw-v1 with LagEmbed (in_channels=$in_channels, n_components=$n_components, dof=$dof)" | tee -a clm_transfer_output.log
    python run_clm_transfer.py \
      --model_name_or_path "/tmp/test-clm-${model##*/}-base-wikitext-2-raw-v1" \
      --use_lagembed \
      --in_channels $in_channels \
      --n_components $n_components \
      --dof $dof \
      --do_train \
      --do_eval \
      --output_dir $lagembed_output \
      --per_device_train_batch_size $batch_size \
      --max_steps 200 >> clm_transfer_output.log 2>&1
    echo "------------------------------------" | tee -a clm_transfer_output.log
done
