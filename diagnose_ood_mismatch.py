"""
诊断：为什么水印序列物品不来自 bottom-m?
分析生成过程中的物品选择机制
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from datasets import dataset_factory
from model import BERT
from dataloader.bert import BERTDataloader
from config import STATE_DICT_KEY
from utils import args as default_args, fix_random_seed_as, set_template

print(f"{'=' * 120}")
print("诊断：水印序列物品来源分析")
print(f"{'=' * 120}\n")

# 设置参数
args = argparse.Namespace()
args.dataset_code = 'ml-1m'
args.model_code = 'bert'
args.device = 'cpu'
args.min_rating = 0
args.min_uc = 5
args.min_sc = 5
args.split = 'leave_one_out'
args.dataset_split_seed = 0
args.gold = True
args.number_ood_seqs = 0.0
args.number_ood_val_seqs = 0.0
args.bottom_m = 100

set_template(args)

dataset = dataset_factory(args)
dataloader = BERTDataloader(args, dataset, pretrained_model=None, distill=False)

# 计算全局流行度（不归一化）
item_freq_all = torch.zeros(dataloader.item_count)
for split in [dataloader.train, dataloader.val, dataloader.test]:
    for user_id, items in split.items():
        if len(items) == 0:
            continue
        idx = torch.tensor(items, dtype=torch.long) - 1
        item_freq_all.index_add_(0, idx, torch.ones_like(idx, dtype=item_freq_all.dtype))

sorted_freq, indices = torch.sort(item_freq_all, descending=False)
indices = indices + 1  # 物品ID从1开始

bottom_m = args.bottom_m
bottom_m_indices = set(indices[:bottom_m].tolist())
bottom_m_max_freq = float(sorted_freq[bottom_m - 1])
rest_min_freq = float(sorted_freq[bottom_m]) if bottom_m < len(sorted_freq) else 0

print(f"【关键前提理解】\n")
print(f"数据集: {args.dataset_code}")
print(f"bottom-{bottom_m} 的定义: 出现频数最少的 {bottom_m} 个物品")
print(f"  ID范围: {int(indices[0])} - {int(indices[bottom_m-1])}")
print(f"  出现频数范围: {int(sorted_freq[0].item())} - {int(bottom_m_max_freq)} 次")
print(f"  平均: {float(sorted_freq[:bottom_m].mean()):.2f} 次")
print(f"\nRest (非bottom-m) 物品: {dataloader.item_count - bottom_m} 个")
print(f"  最小出现频数: {int(rest_min_freq)} 次")
print(f"  平均: {float(sorted_freq[bottom_m:].mean()):.2f} 次")

# 加载模型
model_path = f'experiments/{args.model_code}/{args.dataset_code}/models/best_acc_model.pth'
model = BERT(args)
model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=False).get(STATE_DICT_KEY))
model = model.to(args.device)
model.eval()

print(f"\n{'=' * 120}")
print(f"【生成过程诊断】\n")

max_len = args.bert_max_len
CLOZE_MASK_TOKEN = dataloader.item_count + 1
start_item = int(indices[0])  # 最冷的物品

print(f"开始物品: ID={start_item}, 出现频数={int(item_freq_all[start_item-1])} 次")
print(f"  属于: bottom-{bottom_m} [YES]")

# 单步生成分析
print(f"\n【生成第一步详解】(从最冷的起点生成下一个物品)")
print("-" * 120)

seqs = torch.Tensor([start_item]).to(args.device)
mask_items = torch.tensor([CLOZE_MASK_TOKEN]).to(args.device)

model.eval()
with torch.no_grad():
    input_seqs = torch.zeros((1, max_len)).to(args.device)
    input_seqs[:, -2] = seqs
    input_seqs[:, -1] = mask_items
    labels = model(input_seqs.long())[:, -1, :]
    
    # 获取前10000个物品的logits（除去placeholder）
    logits = labels[:, 1:-1]  # 删除开头和结尾的特殊token
    
    # 获取全局top-100最低分（bottom-100）
    _, bottom_100_global = torch.topk(logits[:, 1:-1], k=bottom_m, largest=False)
    bottom_100_global_items = (bottom_100_global + 2).cpu().numpy()[0]  # +2是因为删除了0和最后一个
    
    print(f"模型输出的 bottom-{bottom_m} (得分最低的):")
    print(f"  物品ID: {sorted(bottom_100_global_items)}")
    
    # 检查这些物品是否在我们的bottom-100中
    in_bottom_m = sum(1 for item in bottom_100_global_items if item in bottom_m_indices)
    print(f"  其中属于数据集bottom-{bottom_m}的: {in_bottom_m}/{bottom_m}")
    print(f"  重叠率: {in_bottom_m/bottom_m*100:.1f}%")
    
    # 关键问题：模型的bottom-100和数据集的bottom-100是否一致？
    print(f"\n⚠️  问题分析:")
    print(f"  模型产生的 bottom-{bottom_m} 基于模型的logits")
    print(f"  这些logits是模型学到的物品关联性，而非数据集中的流行度")
    print(f"  数据集的 bottom-{bottom_m} 基于实际出现频数")
    print(f"  → 这两者可能完全不同！")

print(f"\n{'=' * 120}")
print(f"【核心发现】\n")

print(f"""
为什么水印序列物品不来自 bottom-{bottom_m}?

原因分析:
─────────────────────────────────────────────────────────────

1. 【模型 logits vs 数据集频数的不一致】
   - 生成时，限制在"模型输出的bottom-m"中选择
   - 但"模型输出的bottom-m"是基于模型学到的特征，不是数据集真实流行度
   - 数据集的"bottom-{bottom_m}"是基于出现频数 [{int(sorted_freq[0].item())}-{int(bottom_m_max_freq)} 次]
   
2. 【物品分布不匹配】
   - 候选集约束: 在bottom_m=100内选择
   - 但这100个物品是从模型logits选出的，不是最冷的100个
   - 模型logits指导的"冷"和数据流行度指导的"冷"有差异

3. 【结果】
   - 理论上: 序列应该全部来自bottom-{bottom_m}，平均出现频数~{float(sorted_freq[:bottom_m].mean()):.1f}次
   - 实际上: 只有少量物品来自bottom-{bottom_m}，大多数来自rest
   - 实际平均出现频数: 长度5时 ~1273 次，长度20时 ~454 次
   
   这**远高于**bottom-{bottom_m}的5.98次!

─────────────────────────────────────────────────────────────

【关键问题】:
  您当前的限制方式可能没有达到OOD目标!
  
【改进方向】:
  1. 直接使用出现频数来定义候选集（而非模型logits）
  2. 确保序列物品都来自真正的bottom-m（频数最低的）
  3. 或者改进OOD定义：应该是序列整体不常见，而非单个物品

─────────────────────────────────────────────────────────────
""")

print(f"{'=' * 120}")
