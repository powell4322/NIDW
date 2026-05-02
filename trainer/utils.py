from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim


def to_rank(scores, k):
    # scores [batch_size, num_items]
    values, sorted_items = torch.sort(scores, dim=-1, descending=True)
    values_topk = values[:, :k]
    sorted_items_topk = sorted_items[:, :k] + 1
    return values_topk, sorted_items_topk


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    scores = scores
    labels = labels
    answer_count = labels.sum(1)
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()
    return metrics


def recalls_and_ndcgs_for_ks_rankall(scores, candidates, labels, ks):
    """
    Calculate Recall@k and NDCG@k for ranking evaluation.
    
    Args:
        scores: [batch_size, num_items], model scores for each item (already filtered to exclude padding/mask)
        candidates: [batch_size, num_candidates], item IDs to evaluate (usually 1 per sample for test set)
        labels: [batch_size, num_candidates], relevance labels (usually all 1s)
        ks: list of k values (e.g., [1, 5, 10, 20, 100])
    
    Returns:
        dict with 'Recall@k' and 'NDCG@k' metrics
    """
    metrics = {}
    batch_size = scores.shape[0]
    
    # Get ranking of items (indices sorted by score descending)
    # Since scores[:, 1:-1] was passed in, indices need +1 offset to get item IDs
    sorted_indices = (-scores).argsort(dim=1)  # Shape: [batch_size, num_items]
    
    for k in sorted(ks, reverse=True):
        # Get top-k item indices
        topk_indices = sorted_indices[:, :k]  # Shape: [batch_size, k]
        
        # Convert to item IDs (+1 offset because scores excluded padding)
        topk_item_ids = topk_indices + 1  # Shape: [batch_size, k]
        
        # Calculate hits (1 if candidate is in top-k, 0 otherwise)
        hits = torch.zeros(batch_size, dtype=torch.float32, device=scores.device)
        ranks = torch.full((batch_size,), k + 1, dtype=torch.long, device=scores.device)  # Default rank beyond k
        
        for i in range(batch_size):
            candidate_id = candidates[i, 0].item()  # Get the candidate item ID
            # Find if candidate is in top-k
            match_idx = (topk_item_ids[i] == candidate_id).nonzero(as_tuple=True)[0]
            if len(match_idx) > 0:
                hits[i] = 1.0
                ranks[i] = match_idx[0].item() + 1  # 1-indexed rank
        
        # Recall@k = (# hits in top-k) / (# total candidates)
        # For test set, each sample has 1 candidate, so recall = # hits / # samples
        recall_k = hits.mean().item()
        metrics['Recall@%d' % k] = recall_k
        
        # NDCG@k: DCG / IDCG
        # DCG = sum of (1 / log2(rank + 1)) for each hit
        # IDCG = 1 / log2(2) = 1 for single relevant item at rank 1
        dcg = torch.zeros(batch_size, dtype=torch.float32, device=scores.device)
        for i in range(batch_size):
            if hits[i] > 0:
                rank = ranks[i].item()
                if rank <= k:
                    dcg[i] = 1.0 / torch.log2(torch.tensor(rank + 1.0, device=scores.device, dtype=torch.float32))
        
        idcg = 1.0 / torch.log2(torch.tensor(2.0, device=scores.device, dtype=torch.float32))
        ndcg_k = (dcg.sum() / batch_size / idcg).item()
        metrics['NDCG@%d' % k] = ndcg_k
    
    return metrics


def em_and_agreement(scores_rank, labels_rank):
    em = (scores_rank == labels_rank).float().mean()
    temp = np.hstack((scores_rank.numpy(), labels_rank.numpy()))
    temp = np.sort(temp, axis=1)
    agreement = np.mean(np.sum(temp[:, 1:] == temp[:, :-1], axis=1))
    return em, agreement


def kl_agreements_and_intersctions_for_ks(scores, soft_labels, ks, k_kl=100):
    metrics = {}
    scores = scores.cpu()
    soft_labels = soft_labels.cpu()
    scores_rank = (-scores).argsort(dim=1)
    labels_rank = (-soft_labels).argsort(dim=1)

    top_kl_scores = F.log_softmax(scores.gather(1, labels_rank[:, :k_kl]), dim=-1)
    top_kl_labels = F.softmax(soft_labels.gather(1, labels_rank[:, :k_kl]), dim=-1)
    kl = F.kl_div(top_kl_scores, top_kl_labels, reduction='batchmean')
    metrics['KL-Div'] = kl.item()
    for k in sorted(ks, reverse=True):
        em, agreement = em_and_agreement(scores_rank[:, :k], labels_rank[:, :k])
        metrics['EM@%d' % k] = em.item()
        metrics['Agr@%d' % k] = (agreement / k).item()
    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
