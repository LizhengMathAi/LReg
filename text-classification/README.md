# Text classification 
## Reproducing the Baseline

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

## Reproducing Our Results
Add the `--use_lreg` flag to the baseline command above.
