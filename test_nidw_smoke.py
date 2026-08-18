"""Minimal smoke test: NIDW watermark generation, attacks, and OOD-retention loss."""
import os
import sys

import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_template
from datasets import dataset_factory
from config import STATE_DICT_KEY
from model import BERT
from dataloader.bert import BERTDataloader
from attacks import build_attack
from trainer.bert import BERTTrainer

DEVICE = torch.device('cpu')


def make_args():
    return SimpleNamespace(
        device=DEVICE, dataset_code='ml-1m', model_code='bert',
        pattern_len=5, bottom_m=100, method='cold', wm_type='nidw',
        number_ood_seqs=0.1, number_ood_val_seqs=1.0,
        nidw_tau=1.0, nidw_alpha=0.5, nidw_tau_sim=0.5, nidw_window=3,
        nidw_seed_len=2, nidw_seed_q_min=0.20, nidw_seed_q_max=0.40,
        use_seed_prefix=True, nidw_lambda_ood=0.01, nidw_ood_quantile=0.1,
        nidw_stages=1, nidw_resume_stage=1,
        nidw_tau_s1=0, nidw_alpha_s1=0, nidw_tau_sim_s1=0,
        nidw_tau_s2=0, nidw_alpha_s2=0, nidw_tau_sim_s2=0,
        bert_max_len=200, bert_mask_prob=0.2, bert_max_predictions=40,
        bert_hidden_units=64, bert_dropout=0.1, bert_attn_dropout=0.1,
        bert_num_blocks=2, bert_num_heads=2, bert_head_size=None,
        min_rating=0, min_uc=5, min_sc=5, split='leave_one_out',
        sliding_window_size=0.5,
        train_batch_size=16, val_batch_size=16, test_batch_size=16,
        num_epochs=1, gold=False, num_gpu=1,
        num_items=3416,
        metric_ks=[1, 5, 10, 20, 100], best_metric='NDCG@10',
        optimizer='AdamW', weight_decay=0.01, adam_epsilon=1e-9, momentum=None,
        enable_lr_schedule=True, enable_lr_warmup=False, warmup_steps=100,
        decay_step=10000, gamma=1.0, log_period_as_iter=12800,
        model_init_seed=98765,
        attack='none', target=None,
        dis_threshold=0.7, dis_beta=5.0, dis_eps=0.02,
        point_beta=5.0, noise_scale=1.0, noise_seed=42,
        region_low=0.2, region_high=0.5, region_beta=5.0,
        traj_k1=3, traj_k2=1, traj_beta=5.0, traj_depth_decay=1.0,
        traj_trigger_topk=1, unified_beta1=0.0, unified_beta2=0.0,
        item_freq_source='data',
        freq_query_topk=20, freq_query_num=2000,
        freq_query_temperature=1.0, freq_query_uniform_mix=0.02,
        freq_tpe_alpha=0.5,
    )


def load_oracle(args):
    oracle = BERT(args)
    path = os.path.join('experiments', args.model_code, args.dataset_code,
                        'models', 'best_acc_model.pth')
    oracle.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)[STATE_DICT_KEY])
    oracle.eval()
    return oracle


def main():
    args = make_args()
    set_template(args)
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = dataset_factory(args)
    oracle = load_oracle(args)

    # NIDW watermark sequence is generated inside the dataloader
    dataloader = BERTDataloader(args, dataset, pretrained_model=oracle, distill=False)
    wm_path = os.path.join('sequence pattern', 'nidw_watermark_seq_ml-1m_5_bert_100.npy')
    wm = np.load(wm_path)
    assert len(wm) == args.pattern_len, f'expected length {args.pattern_len}, got {len(wm)}'
    print(f'[NIDW] sequence len={len(wm)} first={wm[0]} (target {args.pattern_len})')

    # Forward pass through the oracle
    train_loader, _, _ = dataloader.get_pytorch_dataloaders()
    seqs, labels = next(iter(train_loader))
    with torch.no_grad():
        logits = oracle(seqs)
    print(f'[model] logits{tuple(logits.shape)}')

    # Attacks must produce same-shaped outputs
    item_freq = torch.ones(dataloader.item_count) / dataloader.item_count
    scores = logits[:, -1, :]
    for name in ['distributional', 'trajectory', 'unified']:
        kw = {'model': oracle, 'args': args} if name in ('trajectory', 'unified') else {}
        attack = build_attack(name, item_freq, method='cold', **kw)
        out = attack(scores.clone())
        assert out.shape == scores.shape
        print(f'[attack] {name} OK shape={tuple(out.shape)}')

    # OOD-retention loss must produce a non-zero gradient on the model
    model = BERT(args)
    export = os.path.join('experiments', '_smoke')
    trainer = BERTTrainer(args, model, train_loader, train_loader, train_loader,
                          export, oracle_model=oracle)
    loss = trainer.calculate_loss([seqs, labels])
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f'[ood] loss={loss.item():.4f} grad_norm={grad_norm:.4f}')
    assert grad_norm > 0, 'OOD-retention loss does not affect training'

    print('[smoke] ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
