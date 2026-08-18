# Near In-Distribution Watermarking for Recommender Systems (NIDW)

Official implementation of **NIDW**, a black-box watermarking framework for
sequential recommendation models, together with inference-time attacks
(paper Sec. 4) and the NIDW watermarking method (paper Sec. 5).

## Watermark types (`--wm_type`)
| Type  | Description |
|-------|-------------|
| `aow` | AOW baseline (cold / pop initialization, bottom-M sampling) |
| `nidw`| Near in-distribution watermark: seed prefix + temperature-scaled oracle sampling + popularity prior + semantic smoothing + progressive training + OOD-retention loss |
| `cps` | Curriculum Popularity Shift variant |

## Inference-time attacks (`--attack`)
| Attack         | Description |
|----------------|-------------|
| `distributional` | Smooth popularity-region suppression (D) |
| `trajectory`     | Recursive model-query continuation suppression (T) |
| `unified`        | Combined D + T: `z' = z - beta1*D - beta2*T` |
| `point` / `noise` / `region` | Baselines |

Item popularity for attacks is controlled by `--item_freq_source`:
`data` (train-only, data-aware) or `qee` (random-prefix model queries, data-unaware).

## Quick start
```bash
# 1. Train the clean oracle model
python train.py --dataset_code ml-1m --model_code bert --gold

# 2. Train a watermarked model (NIDW)
python train.py --dataset_code ml-1m --model_code bert --wm_type nidw --method cold \
  --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100

# 3. Evaluate watermark validity + utility under an attack
python test_watermark_acc.py --dataset_code ml-1m --model_code bert --wm_type nidw \
  --pattern_len 5 --method cold --attack distributional --target unpopular \
  --item_freq_source data --dis_threshold 0.7 --dis_beta 5.0

# 4. Smoke test (NIDW generation + attacks + OOD-retention gradient)
python test_nidw_smoke.py
```

See `RUN_COMMANDS.md` for the full command reference.

## Requirements
PyTorch, numpy, tqdm. See `requirements.txt`.
