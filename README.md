# Near In-Distribution Watermarking for Recommender Systems

Implementation for the paper. AOW-style watermarking embeds a sequence into a
sequential recommendation model for ownership verification. NIDW improves on
this by sampling the watermark from a temperature-scaled oracle distribution
reweighted by a popularity prior and a semantic smoothing term, then training
progressively with an OOD-retention loss. Inference-time attacks (distributional,
trajectory, and their combination) measure how much of the watermark survives.

## Getting started

Install dependencies:

```
pip install -r requirements.txt
```

Raw data goes under `data/`; a preprocessed split under `data/preprocessed/`
is used automatically if present. Datasets: ml-1m, ml-20m, steam, beauty.

### 1. Train the clean oracle model

```
python train.py --dataset_code ml-1m --model_code bert --gold
```

### 2. Train a watermarked model

AOW baseline (cold or pop):

```
python train.py --dataset_code ml-1m --model_code bert --method cold --wm_type aow \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100
```

NIDW, single stage:

```
python train.py --dataset_code ml-1m --model_code bert --method cold --wm_type nidw \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 \
  --nidw_tau 1.0 --nidw_alpha 0.5 --nidw_tau_sim 0.5 --nidw_window 3
```

NIDW, progressive two stages:

```
python train.py --dataset_code ml-1m --model_code bert --method cold --wm_type nidw \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 \
  --nidw_stages 2 \
  --nidw_tau_s1 1.5 --nidw_alpha_s1 0.7 \
  --nidw_tau_s2 1.0 --nidw_alpha_s2 0.5
```

### 3. Watermark verification and attacks

`test_watermark_acc.py` reports watermark recall on a synthetic watermark test
set and utility (NDCG@10) on the clean test set. Without an attack:

```
python test_watermark_acc.py --dataset_code ml-1m --model_code bert --method cold \
  --wm_type nidw --pattern_len 5 --number_ood_seqs 0.1 --number_ood_val_seqs 1.0
```

Add `--attack` to evaluate under inference-time suppression. Distributional
attack penalizes items in low/high popularity regions:

```
python test_watermark_acc.py --dataset_code ml-1m --model_code bert --method cold \
  --wm_type nidw --pattern_len 5 --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --attack distributional --target unpopular --item_freq_source data \
  --dis_threshold 0.7 --dis_beta 5.0
```

Trajectory attack recursively queries the model and suppresses the continuation
items (used for pop watermarks):

```
python test_watermark_acc.py --dataset_code ml-1m --model_code bert --method pop \
  --wm_type nidw --pattern_len 5 --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --attack trajectory --traj_k1 3 --traj_k2 1 --traj_beta 20.0
```

Unified attack combines the two:

```
python test_watermark_acc.py --dataset_code ml-1m --model_code bert --method cold \
  --wm_type nidw --pattern_len 5 --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --attack unified --item_freq_source data \
  --dis_threshold 0.7 --dis_beta 5.0 --traj_k1 3 --traj_k2 1 --traj_beta 5.0 \
  --unified_beta1 5.0 --unified_beta2 5.0
```

Popularity for attacks comes from data (`--item_freq_source data`, data-aware)
or from random-prefix model queries (`--item_freq_source qee`, data-unaware).

### 4. Distillation and fine-tuning

```
python distill.py --dataset_code ml-1m --model_code bert --bb_model_code bert \
  --method cold --wm_type nidw --pattern_len 5
python test_watermark_acc_distilled.py --dataset_code ml-1m --model_code bert \
  --bb_model_code bert --method cold --wm_type nidw --pattern_len 5

python finetune.py --dataset_code ml-1m --model_code bert --method cold \
  --wm_type nidw --pattern_len 5 --finetune_ratio 0.01
python test_watermark_acc_afterfinetune.py --dataset_code ml-1m --model_code bert \
  --method cold --wm_type nidw --pattern_len 5 --finetune_ratio 0.01
```

### Smoke test

`test_nidw_smoke.py` generates a watermark, applies the attacks, and checks
that the OOD-retention loss has a non-zero gradient:

```
python test_nidw_smoke.py
```

## Notes

- BERT and SASRec backbones are both supported (`--model_code bert|sas`).
- Attacks only post-process logits at inference time; the model is never
  re-trained and no parameters are modified.
- Item popularity used for watermarking and data-aware attacks is computed
  from the training split only.

