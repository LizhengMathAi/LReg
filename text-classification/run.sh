#!/bin/bash

export TASK_NAME=mrpc

# Check if the directory exists before removing it
if [ -d "/tmp/$TASK_NAME/" ]; then
  rm -r /tmp/$TASK_NAME/
fi

# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
#   --model_name_or_path google-bert/bert-base-cased \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir /tmp/$TASK_NAME/


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