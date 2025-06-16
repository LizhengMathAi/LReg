## Language model training

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

