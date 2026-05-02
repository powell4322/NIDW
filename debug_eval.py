import torch
import os
from model.bert import BERT
from dataloader.test import TESTDataloader, TESTTestDataset
from datasets import dataset_factory
from config import STATE_DICT_KEY
import numpy as np

# Create a minimal args object
class Args:
    def __init__(self):
        self.device = 'cpu'
        self.dataset_code = 'ml-1m'
        self.model_code = 'bert'
        self.number_ood_seqs = 0.1
        self.number_ood_val_seqs = 1.0
        self.pattern_len = 5
        self.bottom_m = 100
        self.method = 'cold'
        self.attack = 'none'
        self.wm_type = 'aow'
        self.min_rating = 0
        self.min_uc = 5
        self.min_sc = 5
        self.split = 'leave_one_out'
        self.bert_hidden_units = 64
        self.bert_num_blocks = 2
        self.bert_num_heads = 2
        self.bert_attention_head_size = 32
        self.bert_intermediate_size = 64
        self.bert_dropout = 0.5
        self.bert_max_len = 200
        self.bert_mask_prob = 0.1
        self.bert_max_predictions = 40
        self.bert_hidden_act = 'gelu'
        self.bert_initializer_range = 0.02
        self.bert_layer_norm_eps = 1e-12
        self.sliding_window_size = 0.5
        self.train_batch_size = 128
        self.val_batch_size = 128
        self.test_batch_size = 128
        self.num_items = None
        
args = Args()

# Load dataset
dataset = dataset_factory(args)

# Create watermark test loader - directly access internal structures
wm_dataloader = TESTDataloader(args, dataset)
args.num_items = wm_dataloader.item_count

# Check the watermark sequences
print("=" * 60)
print("Watermark Sequence Details")
print("=" * 60)

# Load the watermark sequences manually
candidate_items = np.load('./sequence pattern/cold watermark seq %s %d %s %d.npy' % (
    args.dataset_code, args.pattern_len, args.model_code, args.bottom_m))
print(f"Cold watermark sequence: {candidate_items}")
print(f"Target item (last): {candidate_items[-1]}")

# Create test dataset directly
test_dataset = TESTTestDataset(
    wm_dataloader.train, 
    wm_dataloader.val, 
    wm_dataloader.test, 
    args.bert_max_len,
    wm_dataloader.CLOZE_MASK_TOKEN,
    args.model_code,
    test_users=wm_dataloader.valid_users
)

print("\nMask token value:")
print(f"  CLOZE_MASK_TOKEN = {wm_dataloader.CLOZE_MASK_TOKEN}")
print(f"  num_items + 1 = {wm_dataloader.item_count + 1}")

# Check training data for these synthetic users
print(f"\nTraining data structure for first 3 synthetic users:")
for user_id in wm_dataloader.valid_users[:3]:
    train_seq = wm_dataloader.train.get(user_id, [])
    val_seq = wm_dataloader.val.get(user_id, [])
    test_seq = wm_dataloader.test.get(user_id, [])
    print(f"  User {user_id}:")
    print(f"    train: {train_seq}")
    print(f"    val: {val_seq}")
    print(f"    test: {test_seq}")
    print(f"    combined seq: {train_seq + val_seq}")

for i in range(min(3, len(test_dataset))):
    seq, cand, label = test_dataset[i]
    print(f"\nSample {i}:")
    print(f"  seq shape: {seq.shape}, last 5 items: {seq[-5:].tolist()}")
    print(f"  candidate: {cand.tolist()}")
    print(f"  label: {label.tolist()}")

# Load model and check scores
model = BERT(args).to(args.device)
export_root = 'experiments/watermark_test/method_cold/bert/ml-1m/0.1_1.0_5_100'
model_dict = torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY)
model.load_state_dict(model_dict)
model.eval()

print("\n" + "=" * 60)
print("Model Output Check")
print("=" * 60)

with torch.no_grad():
    batch_seqs = []
    batch_cands = []
    for i in range(min(3, len(test_dataset))):
        seq, cand, label = test_dataset[i]
        batch_seqs.append(seq)
        batch_cands.append(cand)
    
    batch_seqs = torch.stack(batch_seqs)
    print(f"\nBatch shape: {batch_seqs.shape}")
    
    scores = model(batch_seqs.to(args.device))
    print(f"Model output shape: {scores.shape}")
    
    scores_last = scores[:, -1, :]
    print(f"Last timestep scores shape: {scores_last.shape}")
    
    scores_filtered = scores_last[:, 1:-1]
    print(f"Filtered scores shape (vocab_size-2): {scores_filtered.shape}")
    print(f"Expected num_items: {args.num_items}")
    
    topk_indices = (-scores_filtered).argsort(dim=1)
    topk_indices_offset = topk_indices + 1
    
    print(f"\nTop-10 items for first 3 samples:")
    for i in range(min(3, batch_seqs.shape[0])):
        cand = batch_cands[i].item()
        top10 = topk_indices_offset[i, :10].tolist()
        rank = (topk_indices_offset[i] == cand).nonzero()
        print(f"  Sample {i}: cand={cand}, top10={top10}, cand_rank={rank[0].item() if len(rank) > 0 else 'N/A'}")

