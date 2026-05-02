"""
计算水印序列的平均流行度（除去初始物品）
计算不同长度（2, 5, 10, 20）的水印序列的平均流行度
支持 cold 和 pop 两种方法
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# 添加项目到路径
sys.path.insert(0, os.path.dirname(__file__))

from datasets import dataset_factory
from model import BERT
from dataloader.bert import BERTDataloader
from config import STATE_DICT_KEY
from utils import args as default_args, fix_random_seed_as


def load_oracle_model(args):
    """加载预训练的 Oracle 模型"""
    model_path = f'experiments/{args.model_code}/{args.dataset_code}/models/best_acc_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Oracle model not found: {model_path}')
    
    model = BERT(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=False).get(STATE_DICT_KEY))
    model = model.to(args.device)
    model.eval()
    return model


def compute_item_frequency(dataloader):
    """计算数据集中物品的频率分布"""
    freq = torch.zeros(dataloader.item_count)
    
    for split in [dataloader.train, dataloader.val, dataloader.test]:
        for user_id, items in split.items():
            if len(items) == 0:
                continue
            for item in items:
                freq[item - 1] += 1
    
    # 归一化
    if freq.sum() > 0:
        freq = freq / freq.sum()
    
    return freq


def generate_watermark_sequence_cold(model, dataloader, args, pattern_len):
    """
    生成 cold 方法的水印序列
    从最不受欢迎的物品开始
    """
    # 计算物品流行度
    item_freq = torch.zeros(dataloader.item_count)
    for split in [dataloader.train, dataloader.val, dataloader.test]:
        for user_id, items in split.items():
            if len(items) == 0:
                continue
            idx = torch.tensor(items, dtype=torch.long) - 1
            item_freq.index_add_(0, idx, torch.ones_like(idx, dtype=item_freq.dtype))
    
    sorted_freq, indices = torch.sort(item_freq, descending=False)
    indices = indices + 1  # 物品ID从1开始
    
    # 起始物品是最冷的物品
    start_item = int(indices[0])
    
    max_len = args.bert_max_len
    CLOZE_MASK_TOKEN = dataloader.item_count + 1
    
    model.eval()
    mask_items = torch.tensor([CLOZE_MASK_TOKEN]).to(args.device)
    seqs = torch.Tensor([start_item]).to(args.device)
    seen_items = [start_item]
    
    with torch.no_grad():
        for j in range(pattern_len - 1):
            input_seqs = torch.zeros((1, max_len)).to(args.device)
            input_seqs[:, (max_len - 2 - j):-1] = seqs
            input_seqs[:, -1] = mask_items
            labels = model(input_seqs.long())[:, -1, :]
            
            _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
            sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
            np.random.shuffle(sorted_items)
            
            idx = -1
            next_item = int(sorted_items[0, -1] + 1)
            while next_item in seen_items:
                idx -= 1
                if abs(idx) > len(sorted_items):
                    # 如果已经穷尽了候选物品，随机选择一个不在seen_items中的物品
                    for ii in range(1, dataloader.item_count + 1):
                        if ii not in seen_items:
                            next_item = ii
                            break
                    break
                next_item = int(sorted_items[0, idx] + 1)
            
            seen_items.append(next_item)
            next_item = torch.Tensor([next_item]).to(args.device)
            seqs = torch.cat((seqs, next_item), 0)
    
    seqs = [int(x) for x in seqs.cpu().detach().numpy()]
    return seqs


def generate_watermark_sequence_pop(model, dataloader, args, pattern_len):
    """
    生成 pop 方法的水印序列
    从最受欢迎的物品开始
    """
    # 计算物品流行度
    item_freq = torch.zeros(dataloader.item_count)
    for split in [dataloader.train, dataloader.val, dataloader.test]:
        for user_id, items in split.items():
            if len(items) == 0:
                continue
            idx = torch.tensor(items, dtype=torch.long) - 1
            item_freq.index_add_(0, idx, torch.ones_like(idx, dtype=item_freq.dtype))
    
    sorted_freq, indices = torch.sort(item_freq, descending=False)
    indices = indices + 1  # 物品ID从1开始
    
    # 起始物品是最热的物品
    start_item = int(indices[-1])
    
    max_len = args.bert_max_len
    CLOZE_MASK_TOKEN = dataloader.item_count + 1
    
    model.eval()
    mask_items = torch.tensor([CLOZE_MASK_TOKEN]).to(args.device)
    seqs = torch.Tensor([start_item]).to(args.device)
    seen_items = [start_item]
    
    with torch.no_grad():
        for j in range(pattern_len - 1):
            input_seqs = torch.zeros((1, max_len)).to(args.device)
            input_seqs[:, (max_len - 2 - j):-1] = seqs
            input_seqs[:, -1] = mask_items
            labels = model(input_seqs.long())[:, -1, :]
            
            _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
            sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
            np.random.shuffle(sorted_items)
            
            idx = -1
            next_item = int(sorted_items[0, -1] + 1)
            while next_item in seen_items:
                idx -= 1
                if abs(idx) > len(sorted_items):
                    # 如果已经穷尽了候选物品，随机选择一个不在seen_items中的物品
                    for ii in range(1, dataloader.item_count + 1):
                        if ii not in seen_items:
                            next_item = ii
                            break
                    break
                next_item = int(sorted_items[0, idx] + 1)
            
            seen_items.append(next_item)
            next_item = torch.Tensor([next_item]).to(args.device)
            seqs = torch.cat((seqs, next_item), 0)
    
    seqs = [int(x) for x in seqs.cpu().detach().numpy()]
    return seqs


def compute_popularity_excluding_initial(seqs, item_freq):
    """
    计算序列中除初始物品外其他物品的平均流行度
    seqs: 序列（物品ID列表）
    item_freq: 物品流行度（张量或数组，索引从0开始）
    """
    if len(seqs) <= 1:
        return 0.0
    
    # 计算序列中除第一个物品外的平均流行度
    total_freq = 0.0
    for item in seqs[1:]:  # 跳过第一个物品
        item_idx = item - 1  # 转换为0索引
        if 0 <= item_idx < len(item_freq):
            total_freq += float(item_freq[item_idx])
    
    avg_freq = total_freq / (len(seqs) - 1)
    return avg_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dataset_code', default='ml-1m', type=str)
    parser.add_argument('--model_code', default='bert', type=str)
    parser.add_argument('--bottom_m', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    args = parser.parse_args()
    
    # 合并默认参数
    for key, value in default_args.__dict__.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # 应用模板参数（根据数据集和模型类型设置）
    from utils import set_template
    set_template(args)
    
    args.device = torch.device(args.device)
    
    print(f"{'=' * 80}")
    print(f"计算水印序列平均流行度 (除去初始物品)")
    print(f"{'=' * 80}")
    print(f"数据集: {args.dataset_code}")
    print(f"模型: {args.model_code}")
    print(f"设备: {args.device}")
    print(f"底部物品数: {args.bottom_m}")
    print(f"{'=' * 80}\n")
    
    # 设置随机种子
    fix_random_seed_as(args.seed)
    
    # 加载数据集
    print("加载数据集...")
    dataset = dataset_factory(args)
    
    # 创建数据加载器（仅用于计算流行度，不注入水印）
    args.number_ood_seqs = 0.0
    args.number_ood_val_seqs = 0.0
    dataloader = BERTDataloader(args, dataset, pretrained_model=None, distill=False)
    
    # 计算物品流行度
    print("计算物品流行度分布...")
    item_freq = compute_item_frequency(dataloader)
    print(f"物品总数: {dataloader.item_count}")
    print(f"用户总数: {dataloader.user_count}")
    print(f"流行度范围: [{item_freq.min():.6f}, {item_freq.max():.6f}]")
    print(f"流行度平均值: {item_freq.mean():.6f}\n")
    
    # 加载预训练模型
    print("加载 Oracle 模型...")
    model = load_oracle_model(args)
    print("模型加载成功\n")
    
    # 不同的序列长度
    pattern_lengths = [2, 5, 10, 20]
    methods = ['cold', 'pop']
    
    # 存储结果
    results = []
    
    print(f"{'=' * 80}")
    print("生成水印序列并计算平均流行度")
    print(f"{'=' * 80}\n")
    
    for method in methods:
        print(f"\n方法: {method.upper()}")
        print("-" * 80)
        
        for pattern_len in pattern_lengths:
            print(f"  长度: {pattern_len}...", end=" ", flush=True)
            
            # 固定随机种子以保证可重复性
            fix_random_seed_as(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            
            # 生成序列
            if method == 'cold':
                seqs = generate_watermark_sequence_cold(model, dataloader, args, pattern_len)
            else:  # pop
                seqs = generate_watermark_sequence_pop(model, dataloader, args, pattern_len)
            
            # 计算初始物品的流行度
            initial_item = seqs[0]
            initial_freq = float(item_freq[initial_item - 1])
            
            # 计算除初始物品外的平均流行度
            avg_freq_excl_initial = compute_popularity_excluding_initial(seqs, item_freq)
            
            # 计算其他统计信息
            other_items = seqs[1:]
            if len(other_items) > 0:
                other_freqs = torch.tensor([float(item_freq[item - 1]) for item in other_items])
                min_freq = float(other_freqs.min())
                max_freq = float(other_freqs.max())
                std_freq = float(other_freqs.std())
            else:
                min_freq = max_freq = std_freq = 0.0
            
            print(f"✓")
            print(f"    序列: {seqs}")
            print(f"    初始物品ID: {initial_item} (流行度: {initial_freq:.6f})")
            print(f"    其他物品数: {len(other_items)}")
            print(f"    平均流行度 (excl. initial): {avg_freq_excl_initial:.6f}")
            print(f"    流行度范围: [{min_freq:.6f}, {max_freq:.6f}]")
            print(f"    流行度标准差: {std_freq:.6f}")
            print()
            
            results.append({
                'method': method,
                'pattern_len': pattern_len,
                'watermark_seq': str(seqs),
                'initial_item_id': initial_item,
                'initial_item_freq': initial_freq,
                'num_other_items': len(other_items),
                'avg_freq_excl_initial': avg_freq_excl_initial,
                'min_freq_excl_initial': min_freq,
                'max_freq_excl_initial': max_freq,
                'std_freq_excl_initial': std_freq,
            })
    
    # 创建结果表格
    df_results = pd.DataFrame(results)
    
    # 保存结果
    output_csv = 'experiments/watermark_popularity_analysis.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}\n")
    
    # 打印汇总表格
    print(f"{'=' * 80}")
    print("汇总表格")
    print(f"{'=' * 80}\n")
    
    summary_data = []
    for method in methods:
        for pattern_len in pattern_lengths:
            row = df_results[(df_results['method'] == method) & (df_results['pattern_len'] == pattern_len)]
            if len(row) > 0:
                summary_data.append({
                    'Method': method.upper(),
                    'Pattern Length': pattern_len,
                    'Initial Item ID': int(row['initial_item_id'].values[0]),
                    'Avg Popularity (excl. initial)': f"{row['avg_freq_excl_initial'].values[0]:.6f}",
                    'Min Popularity': f"{row['min_freq_excl_initial'].values[0]:.6f}",
                    'Max Popularity': f"{row['max_freq_excl_initial'].values[0]:.6f}",
                })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
