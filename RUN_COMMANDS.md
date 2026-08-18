# NIDW & Attack — Run Commands

Quick reference for watermark training, validity evaluation, inference-time
attacks, distillation, and fine-tuning.

## 0. Environment
```bash
pip install -r requirements.txt
```

## 1. Train the clean Oracle model
```bash
python train.py --device cuda:0 --dataset_code ml-1m --model_code bert --gold
python train.py --device cuda:0 --dataset_code ml-20m --model_code bert --gold
```

## 2. Watermark training

### 2.1 AOW baseline
```bash
python train.py --device cuda:0 --dataset_code ml-1m --model_code bert \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --pattern_len 5 --bottom_m 100 --method cold --wm_type aow
```

### 2.2 NIDW (single stage)
```bash
python train.py --device cuda:0 --dataset_code ml-1m --model_code bert \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --pattern_len 5 --bottom_m 100 --method cold --wm_type nidw \
  --nidw_tau 1.0 --nidw_alpha 0.5 --nidw_tau_sim 0.5 --nidw_window 3
```

### 2.3 NIDW progressive (two stages)
```bash
python train.py --device cuda:0 --dataset_code ml-1m --model_code bert \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 \
  --pattern_len 5 --bottom_m 100 --method cold --wm_type nidw \
  --nidw_stages 2 \
  --nidw_tau_s1 1.5 --nidw_alpha_s1 0.7 --nidw_tau_sim_s1 0.5 \
  --nidw_tau_s2 1.0 --nidw_alpha_s2 0.5 --nidw_tau_sim_s2 0.5
```

## 3. Watermark validity (no attack)
```bash
# AOW
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type aow

# NIDW
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw
```

## 4. Inference-time attacks

### 4.1 Distributional suppression (D), data-aware (cold)
```bash
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw \
  --attack distributional --target unpopular \
  --item_freq_source data --dis_threshold 0.7 --dis_beta 5.0 --dis_eps 0.02
```

### 4.1b Distributional suppression, data-unaware (QEE)
```bash
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw \
  --attack distributional --target unpopular \
  --item_freq_source qee --freq_query_num 2000 --freq_query_topk 100
```

### 4.2 Trajectory suppression (T), data-aware (pop)
```bash
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method pop --wm_type nidw \
  --attack trajectory --traj_k1 3 --traj_k2 1 --traj_beta 20.0
```

### 4.2b Trajectory suppression, data-unaware (multi-trigger QEE)
```bash
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method pop --wm_type nidw \
  --attack trajectory --traj_k1 3 --traj_k2 1 --traj_beta 20.0 \
  --traj_trigger_topk 20 --item_freq_source qee
```

### 4.3 Unified suppression (D + T, paper Sec. 4)
```bash
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw \
  --attack unified \
  --item_freq_source data \
  --dis_threshold 0.7 --dis_beta 5.0 --dis_eps 0.02 \
  --traj_k1 3 --traj_k2 1 --traj_beta 5.0 \
  --unified_beta1 5.0 --unified_beta2 5.0
```

## 5. Model extraction attacks

### 5.1 Distillation
```bash
python distill.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --bb_model_code bert \
  --pattern_len 5 --method cold --wm_type nidw

python test_watermark_acc_distilled.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --bb_model_code bert \
  --pattern_len 5 --method cold --wm_type nidw
```

### 5.2 Fine-tuning
```bash
python finetune.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw \
  --finetune_ratio 0.01

python test_watermark_acc_afterfinetune.py --device cuda:0 --dataset_code ml-1m \
  --model_code bert --pattern_len 5 --method cold --wm_type nidw \
  --finetune_ratio 0.01
```

