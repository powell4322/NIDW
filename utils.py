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
parser.add_argument('--wm_type', type=str, choices=['aow', 'cps'], default='aow',
                    help='Watermark type: aow (original) or cps (Curriculum Popularity Shift)')
parser.add_argument('--cps_q_start', type=float, default=0.25,
                    help='CPS: starting quantile (mid-frequency region)')
parser.add_argument('--cps_q_end', type=float, default=0.05,
                    help='CPS: ending quantile (tail region)')


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
                    help='Attack type. Recommended: none | soft_prf | point_level | random_shuffle. Legacy aliases ptsc/pcrmr are accepted for compatibility and mapped to soft_prf.')
parser.add_argument('--prf_gamma', type=float, default=0.7)
parser.add_argument('--prf_beta', type=float, default=5.0)
parser.add_argument('--prf_eps', type=float, default=0.02,
                    help='Soft-PRF transition smoothness epsilon')
parser.add_argument('--pl_top_k', type=int, default=50,
                    help='Point-Level attack: number of top popular items to boost')
parser.add_argument('--pl_boost', type=float, default=5.0,
                    help='Point-Level attack: score boost magnitude for popular items')
parser.add_argument('--rs_noise_scale', type=float, default=1.0,
                    help='Random Shuffle attack: standard deviation of Gaussian noise')
parser.add_argument('--rs_seed', type=int, default=42,
                    help='Random Shuffle attack: random seed for reproducibility')
parser.add_argument('--ptsc_alpha', type=float, default=6.0)
parser.add_argument('--pcrmr_sigma', type=float, default=1.0)
parser.add_argument('--attack_direction', type=str, default='suppress_popular', choices=['suppress_popular', 'suppress_unpopular'])
parser.add_argument('--item_freq_source', type=str, default='data',
                    choices=['data', 'model_query', 'uniform', 'dpe', 'qee', 'tpe'],
                    help='Source of item popularity for attacks: data=data-aware, dpe=distilled-data estimate, qee=query-based estimate, tpe=(1-alpha)*data+alpha*QEE, model_query is legacy alias of qee, uniform=no prior')
parser.add_argument('--freq_query_topk', type=int, default=20,
                    help='Top-k outputs used per query when item_freq_source=qee/model_query or tpe')
parser.add_argument('--freq_query_max_batches', type=int, default=0,
                    help='Max query batches for qee/model_query estimation (0 means all batches)')
parser.add_argument('--freq_query_temperature', type=float, default=1.0,
                    help='Softmax temperature for qee/model-query popularity estimation')
parser.add_argument('--freq_query_uniform_mix', type=float, default=0.02,
                    help='Mix ratio with uniform prior to stabilize zero-hit items in qee/model_query mode')
parser.add_argument('--freq_tpe_alpha', type=float, default=0.5,
                    help='Blend ratio for TPE estimator: freq=(1-alpha)*data + alpha*QEE')
parser.add_argument('--prf_phi_json', type=str, default='',
                    help='Optional JSON string for unified Soft-PRF params, e.g. {"gamma":0.8,"beta":12.0,"eps":0.02}')
parser.add_argument('--model_init_seed', type=int, default=98765,
                    help='Random seed used for model initialization and evaluation reproducibility')

args = parser.parse_args()
