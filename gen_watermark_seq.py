"""
快速生成水印序列文件
"""
import numpy as np
import torch
import os
from config import STATE_DICT_KEY
from model import BERT
from utils import set_template, args

# 初始化参数
set_template(args)
args.dataset_code = 'ml-1m'
args.model_code = 'bert'
args.method = 'cold'
args.pattern_len = 5
args.bottom_m = 100
args.device = 'cpu'

# 先加载数据集获取 num_items
from datasets import dataset_factory
from dataloader.bert import BERTDataloader

dataset = dataset_factory(args)
dataloader = BERTDataloader(args, dataset)
args.num_items = dataloader.item_count

# 加载预训练的 Oracle 模型
oracle_root = f'experiments/{args.model_code}/{args.dataset_code}'
model = BERT(args)
model_path = os.path.join(oracle_root, 'models', 'best_acc_model.pth')

print(f"Loading Oracle model from {model_path}")
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False).get(STATE_DICT_KEY))
model.eval()

# 获取项目数
num_items = dataloader.item_count

# 生成 Cold Start 水印序列
print("Generating cold watermark sequence...")
start_item = 1  # 最不流行的物品（cold start）
seqs = torch.Tensor([start_item]).to(args.device)
seen_items = [start_item]

for j in range(args.pattern_len - 1):
    input_seqs = torch.zeros((1, args.bert_max_len)).to(args.device)
    input_seqs[:, (args.bert_max_len - 1 - j):] = seqs
    
    with torch.no_grad():
        labels = model(input_seqs.long())[:, -1, :]
    
    _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
    sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
    np.random.shuffle(sorted_items)
    
    idx = -1
    next_item = int(sorted_items[0, -1] + 1)
    while next_item in seen_items:
        idx -= 1
        next_item = int(sorted_items[0, idx] + 1)
    
    seen_items.append(next_item)
    next_item_tensor = torch.Tensor([next_item]).to(args.device)
    seqs = torch.cat((seqs, next_item_tensor), 0)

cold_seq = [int(x) for x in seqs.cpu().detach().numpy()]
print(f"Cold watermark sequence: {cold_seq}")

# 保存到文件
seq_dir = './sequence pattern'
os.makedirs(seq_dir, exist_ok=True)

cold_filename = os.path.join(seq_dir, f'cold watermark seq {args.dataset_code} {args.pattern_len} {args.model_code} {args.bottom_m}.npy')
np.save(cold_filename, cold_seq)
print(f"Saved to {cold_filename}")

# Pop 序列（从最流行的物品开始）
print("\nGenerating pop watermark sequence...")
start_item = num_items  # 最流行的物品（pop start）
seqs = torch.Tensor([start_item]).to(args.device)
seen_items = [start_item]

for j in range(args.pattern_len - 1):
    input_seqs = torch.zeros((1, args.bert_max_len)).to(args.device)
    input_seqs[:, (args.bert_max_len - 1 - j):] = seqs
    
    with torch.no_grad():
        labels = model(input_seqs.long())[:, -1, :]
    
    _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
    sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
    np.random.shuffle(sorted_items)
    
    idx = -1
    next_item = int(sorted_items[0, -1] + 1)
    while next_item in seen_items:
        idx -= 1
        next_item = int(sorted_items[0, idx] + 1)
    
    seen_items.append(next_item)
    next_item_tensor = torch.Tensor([next_item]).to(args.device)
    seqs = torch.cat((seqs, next_item_tensor), 0)

pop_seq = [int(x) for x in seqs.cpu().detach().numpy()]
print(f"Pop watermark sequence: {pop_seq}")

pop_filename = os.path.join(seq_dir, f'pop watermark seq {args.dataset_code} {args.pattern_len} {args.model_code} {args.bottom_m}.npy')
np.save(pop_filename, pop_seq)
print(f"Saved to {pop_filename}")

print("\nDone! Watermark sequences generated.")
