import numpy as np
import random
import torch

from datasets import DATASETS
from model import *
import argparse


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_template(args):
    args.k = 100

    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    # dataset_code = {'1': 'ml-1m', '20': 'ml-20m', 'b': 'beauty', 'bd': 'beauty_dense' , 'g': 'games', 's': 'steam', 'y': 'yoochoose'}
    # args.dataset_code = dataset_code[input('Input 1 / 20 for movielens, b for beauty, bd for dense beauty, g for games, s for steam and y for yoochoose: ')]
    if args.dataset_code == 'ml-1m':
        batch = 128
        args.num_epochs = 1000

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 40
    elif args.dataset_code == 'ml-20m':
        batch = 24
        args.num_epochs = 10

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.1
        args.bert_attn_dropout = 0.1
        args.bert_max_len = 200
        args.bert_mask_prob = 0.2
        args.bert_max_predictions = 20
    elif args.dataset_code in ['beauty', 'beauty_dense']:
        batch = 16
        args.num_epochs = 50

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.6
        args.bert_max_predictions = 30
    elif args.dataset_code == 'games':
        batch = 128
        args.num_epochs = 50

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.5
        args.bert_max_len = 50
        args.bert_mask_prob = 0.5
        args.bert_max_predictions = 25
    elif args.dataset_code == 'steam':
        batch = 64
        args.num_epochs = 15

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 64
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        # args.bert_max_len = 200
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20
    elif args.dataset_code == 'yoochoose':
        batch = 128
        args.num_epochs = 15

        args.sliding_window_size = 0.5
        args.bert_hidden_units = 256
        args.bert_dropout = 0.2
        args.bert_attn_dropout = 0.2
        args.bert_max_len = 50
        args.bert_mask_prob = 0.4
        args.bert_max_predictions = 20

    if args.model_code == 'narm':
        batch = 128
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765

    # model_codes = {'b': 'bert', 's':'sas', 'n':'narm'}
    # args.model_code = model_codes[input('Input model code, b for BERT, s for SASRec and n for NARM: ')]

    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = True
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    # args.metric_ks = [1, 5, 10]
    args.best_metric = 'NDCG@10'
    args.model_init_seed = getattr(args, 'model_init_seed', 98765)
    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-1m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--dataset_split_seed', type=int, default=0)

################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--sliding_window_size', type=float, default=0.5)

################
# NegativeSampler
################
# parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
# parser.add_argument('--train_negative_sample_size', type=int, default=0)
# parser.add_argument('--train_negative_sampling_seed', type=int, default=0)
# parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
# parser.add_argument('--test_negative_sample_size', type=int, default=100)
# parser.add_argument('--test_negative_sampling_seed', type=int, default=0)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_gpu', type=int, default=1)
# optimizer & lr#
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
# epochs #
parser.add_argument('--num_epochs', type=int, default=100)
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 100])
parser.add_argument('--best_metric', type=str, default='NDCG@10')

################
# Model
################
parser.add_argument('--model_code', type=str, choices=['bert', 'sas'])
# BERT specs, used for SASRec and NARM as well #
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.1)
parser.add_argument('--bert_attn_dropout', type=float, default=0.1)
parser.add_argument('--bert_mask_prob', type=float, default=0.2)

################
# Distillation & Retraining
################
parser.add_argument('--bb_model_code', type=str, choices=['bert', 'sas', 'none'], default='none')
parser.add_argument('--num_generated_seqs', type=int, default=3000)
parser.add_argument('--num_original_seqs', type=int, default=0)
parser.add_argument('--num_poisoned_seqs', type=int, default=100)
parser.add_argument('--num_alter_items', type=int, default=10)

################
# Watermark Training
################
parser.add_argument('--number_ood_seqs', type=float, default=0.0)
parser.add_argument('--bottom_m', type=int, default=100)
parser.add_argument('--wm_type', type=str, choices=['aow', 'cps', 'nidw'], default='aow',
                    help='Watermark type: aow (original), cps (Curriculum Popularity Shift), nidw (Near In-Distribution Watermarking)')
parser.add_argument('--cps_q_start', type=float, default=0.25,
                    help='CPS: starting quantile (mid-frequency region)')
parser.add_argument('--cps_q_end', type=float, default=0.05,
                    help='CPS: ending quantile (tail region)')

# NIDW hyperparameters
parser.add_argument('--nidw_tau', type=float, default=1.0,
                    help='NIDW: temperature for sampling sharpness')
parser.add_argument('--nidw_alpha', type=float, default=0.5,
                    help='NIDW: popularity prior strength')
parser.add_argument('--nidw_tau_sim', type=float, default=0.5,
                    help='NIDW: semantic smoothing temperature')
parser.add_argument('--nidw_window', type=int, default=3,
                    help='NIDW: sliding window size for context embedding')
parser.add_argument('--nidw_seed_len', type=int, default=2,
                    help='NIDW: max seed prefix length')
parser.add_argument('--nidw_seed_q_min', type=float, default=0.20,
                    help='NIDW: lower bound of seed item popularity quantile')
parser.add_argument('--nidw_seed_q_max', type=float, default=0.40,
                    help='NIDW: upper bound of seed item popularity quantile')
parser.add_argument('--nidw_stages', type=int, default=1,
                    help='NIDW: number of progressive training stages')
parser.add_argument('--nidw_resume_stage', type=int, default=1,
                    help='NIDW: resume training from this stage')
parser.add_argument('--nidw_tau_s1', type=float, default=0,
                    help='NIDW: temperature for stage 1 (0=use nidw_tau default)')
parser.add_argument('--nidw_alpha_s1', type=float, default=0,
                    help='NIDW: popularity prior for stage 1 (0=use nidw_alpha default)')
parser.add_argument('--nidw_tau_sim_s1', type=float, default=0,
                    help='NIDW: semantic smoothing temperature for stage 1')
parser.add_argument('--nidw_tau_s2', type=float, default=0,
                    help='NIDW: temperature for stage 2')
parser.add_argument('--nidw_alpha_s2', type=float, default=0,
                    help='NIDW: popularity prior for stage 2')
parser.add_argument('--nidw_tau_sim_s2', type=float, default=0,
                    help='NIDW: semantic smoothing temperature for stage 2')
parser.add_argument('--nidw_lambda_ood', type=float, default=0.01,
                    help='NIDW: OOD retention loss weight')
parser.add_argument('--use_seed_prefix', type=bool, default=True,
                    help='NIDW: use seed prefix from real user (True) or fallback to cold anchor (False)')
parser.add_argument('--nidw_ood_quantile', type=float, default=0.1,
                    help='NIDW: quantile threshold for OOD retention loss')


################
# Watermark Testing
################
parser.add_argument('--gold', action='store_true')
parser.add_argument('--method', type=str, choices=['cold', 'pop'], default='cold')
parser.add_argument('--number_ood_val_seqs', type=float, default=0.0)
parser.add_argument('--pattern_len', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--finetune_ratio', type=float, default=0)

################
# Inference-time Attack
################
parser.add_argument('--attack', type=str, default='none',
                    choices=['none', 'distributional', 'point', 'noise', 'region', 'trajectory', 'unified'],
                    help='Attack type: distributional (D), trajectory (T), unified (D+T), point, noise, region')
parser.add_argument('--target', type=str, default=None, choices=['popular', 'unpopular'],
                    help='Suppression target (default inferred from method: cold->unpopular, pop->popular)')
# distributional
parser.add_argument('--dis_threshold', type=float, default=0.7,
                    help='Distributional attack: popularity rank threshold')
parser.add_argument('--dis_beta', type=float, default=5.0,
                    help='Distributional attack: penalty magnitude')
parser.add_argument('--dis_eps', type=float, default=0.02,
                    help='Distributional attack: sigmoid transition smoothness')
# point
parser.add_argument('--point_beta', type=float, default=5.0,
                    help='Point attack: penalty applied to the single target item')
# region baseline
parser.add_argument('--region_low', type=float, default=0.2,
                    help='Region attack: lower popularity-quantile bound of the penalty band')
parser.add_argument('--region_high', type=float, default=0.5,
                    help='Region attack: upper popularity-quantile bound of the penalty band')
parser.add_argument('--region_beta', type=float, default=5.0,
                    help='Region attack: penalty strength inside the band')
# noise baseline
parser.add_argument('--noise_scale', type=float, default=1.0,
                    help='Noise attack: standard deviation of Gaussian noise')
parser.add_argument('--noise_seed', type=int, default=42,
                    help='Noise attack: random seed')
# trajectory / unified
parser.add_argument('--traj_k1', type=int, default=3,
                    help='Trajectory attack: top-K predictions at level 1')
parser.add_argument('--traj_k2', type=int, default=1,
                    help='Trajectory attack: top-K predictions at level 2 (per branch)')
parser.add_argument('--traj_beta', type=float, default=5.0,
                    help='Trajectory attack: base penalty strength')
parser.add_argument('--traj_depth_decay', type=float, default=1.0,
                    help='Trajectory attack: penalty multiplier per deeper level')
parser.add_argument('--traj_trigger_topk', type=int, default=1,
                    help='Trajectory attack: number of top triggers to union (range attack for QEE)')
parser.add_argument('--unified_beta1', type=float, default=0.0,
                    help='Unified attack: distributional coefficient (0=auto by method)')
parser.add_argument('--unified_beta2', type=float, default=0.0,
                    help='Unified attack: trajectory coefficient (0=auto by method)')
parser.add_argument('--item_freq_source', type=str, default='data',
                    choices=['data', 'uniform', 'dpe', 'qee', 'tpe'],
                    help='Item popularity source for attacks: data=data-aware, qee=query-based estimate, '
                         'dpe=distilled-data estimate, tpe=(1-alpha)*data+alpha*QEE, uniform=no prior')
parser.add_argument('--freq_query_topk', type=int, default=20,
                    help='Top-k outputs used per query when item_freq_source=qee or tpe')
parser.add_argument('--freq_query_num', type=int, default=2000,
                    help='Number of random-prefix queries for qee/tpe estimation')
parser.add_argument('--freq_query_temperature', type=float, default=1.0,
                    help='Softmax temperature for qee popularity estimation')
parser.add_argument('--freq_query_uniform_mix', type=float, default=0.02,
                    help='Mix ratio with uniform prior to stabilize zero-hit items in qee mode')
parser.add_argument('--freq_tpe_alpha', type=float, default=0.5,
                    help='Blend ratio for TPE estimator: freq=(1-alpha)*data + alpha*QEE')
parser.add_argument('--model_init_seed', type=int, default=98765,
                    help='Random seed used for model initialization and evaluation reproducibility')

args = parser.parse_args()
