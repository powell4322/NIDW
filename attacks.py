# attacks.py
# Inference-time popularity-aware attacks for AOW
# All attacks are watermark-agnostic and post-processing only

import torch
import json


@torch.no_grad()
def estimate_item_freq_from_model_outputs(
    model,
    query_loader,
    device,
    num_items,
    topk=20,
    max_batches=0,
    temperature=1.0,
    uniform_mix=0.02,
):
    """
    Estimate item popularity from model outputs only (data-unaware setting).

    Returns a normalized tensor [num_items], compatible with Soft-PRF.
    """
    model.eval()
    accum = torch.zeros(num_items, device=device)

    safe_topk = max(1, min(int(topk), int(num_items)))
    safe_temperature = max(float(temperature), 1e-6)
    use_all_batches = int(max_batches) <= 0

    for batch_idx, batch in enumerate(query_loader):
        if (not use_all_batches) and batch_idx >= int(max_batches):
            break

        seqs = batch[0].to(device)
        scores = model(seqs)[:, -1, :]
        item_scores = scores[:, 1:1 + num_items]

        probs = torch.softmax(item_scores / safe_temperature, dim=-1)

        if safe_topk < num_items:
            top_values, top_indices = torch.topk(probs, k=safe_topk, dim=-1)
            sparse_probs = torch.zeros_like(probs)
            sparse_probs.scatter_(1, top_indices, top_values)
            probs = sparse_probs

        accum += probs.sum(dim=0)

    total = accum.sum()
    if total <= 0:
        freq = torch.ones(num_items, device=device) / float(num_items)
    else:
        freq = accum / total

    mix = float(uniform_mix)
    if mix > 0:
        mix = min(mix, 1.0)
        uniform = torch.ones_like(freq) / float(num_items)
        freq = (1.0 - mix) * freq + mix * uniform
        freq = freq / freq.sum()

    return freq


class SoftPRFAttack:
    """
    Soft Popularity-based Ranking Filtering (Soft-PRF)

    scores_v <- scores_v - beta * sigmoid((rank_freq(v) - gamma) / eps)
    """

    def __init__(self, item_freq, gamma=0.7, beta=5.0, eps=0.02, direction='suppress_popular'):
        """
        item_freq: Tensor [num_items], normalized frequency (sum=1)
        gamma: popularity rank threshold (0~1)
        beta: penalty magnitude
        eps: smoothness of transition
        direction: 'suppress_popular' (default) or 'suppress_unpopular'
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)

        # normalized popularity rank in [0,1]
        # 0 = Least Popular (Tail), 1 = Most Popular (Head)
        freq_rank = torch.argsort(
            torch.argsort(item_freq)
        ).float() / self.num_items

        self.freq_rank = freq_rank.to(self.device)
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.direction = direction

    def __call__(self, scores):
        """
        scores: [B, num_items + 2] (Includes padding and mask token)
        """
        # Determine penalty based on direction
        if self.direction == 'suppress_unpopular':
             # CASE: Attacking Cold-Start Watermarks (OOD Items)
             # We want to PENALIZE items with LOW rank.
             # Target: sigmoid -> 1 when freq_rank is LOW (< gamma)
             penalty = self.beta * torch.sigmoid(
                (self.gamma - self.freq_rank) / self.eps
             )
        else:
             # CASE: Mitigating Popularity Bias
             # We want to PENALIZE items with HIGH rank.
             # Target: sigmoid -> 1 when freq_rank is HIGH (> gamma)
             penalty = self.beta * torch.sigmoid(
                (self.freq_rank - self.gamma) / self.eps
             )
             
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] - penalty
        return scores


class PointLevelAttack:
    """
    Direct Point-Level Attack: Replace initial items with popular alternatives.
    
    For cold-start watermarks targeting OOD items, directly boost the scores of 
    in-distribution (popular) items while suppressing OOD items.
    
    Mechanism:
    - Identifies top-k most popular items (in-distribution candidates)
    - Applies score boosting to encourage replacement of watermarked initial items
    - Supports different replacement strategies
    """
    
    def __init__(self, item_freq, top_k=50, boost_magnitude=5.0, direction='suppress_unpopular'):
        """
        item_freq: Tensor [num_items], normalized frequency (sum=1)
        top_k: number of top popular items to boost
        boost_magnitude: score boost amount for popular items
        direction: 'suppress_unpopular' (cold attack) or 'suppress_popular' (pop mitigation)
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.top_k = min(top_k, self.num_items)
        self.boost_magnitude = boost_magnitude
        self.direction = direction
        
        # Identify top-k popular items
        _, top_indices = torch.topk(item_freq, k=self.top_k, dim=0)
        self.top_popular_mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
        self.top_popular_mask[top_indices] = True
    
    def __call__(self, scores):
        """
        scores: [B, num_items + 2] (Includes padding and mask token)
        """
        # Extract item scores (exclude padding token 0 and mask token at position -1)
        item_scores = scores[:, 1:1+self.num_items]
        
        if self.direction == 'suppress_unpopular':
            # Cold-start attack: boost popular items to push out unpopular OOD items
            boost = self.boost_magnitude * self.top_popular_mask.float()
            item_scores = item_scores + boost
        else:
            # Pop mitigation: suppress popular items (opposite effect)
            suppress = self.boost_magnitude * self.top_popular_mask.float()
            item_scores = item_scores - suppress
        
        scores[:, 1:1+self.num_items] = item_scores
        return scores


class RandomShuffleAttack:
    """
    Baseline Attack: Random score perturbation.
    
    Shuffles item scores randomly within a given range to serve as a baseline
    for evaluating attack effectiveness. Used to distinguish watermark-specific
    vulnerabilities from general robustness issues.
    """
    
    def __init__(self, item_freq, noise_scale=1.0, seed=42):
        """
        item_freq: Tensor [num_items], normalized frequency (unused, kept for interface)
        noise_scale: standard deviation of Gaussian noise
        seed: random seed for reproducibility
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.noise_scale = noise_scale
        self.seed = seed
        torch.manual_seed(seed)
    
    def __call__(self, scores):
        """
        scores: [B, num_items + 2]
        Adds Gaussian noise to item scores for random perturbation baseline.
        """
        # Add random noise to item scores
        noise = torch.randn_like(scores[:, 1:1+self.num_items]) * self.noise_scale
        item_scores = scores[:, 1:1+self.num_items] + noise
        scores[:, 1:1+self.num_items] = item_scores
        return scores


def build_attack(attack_name, item_freq, **kwargs):
    """
    Factory method for attacks.

    Supported attacks:
    - soft_prf: Soft Popularity-based Ranking Filtering (post-processing score penalty)
    - point_level: Direct point-level replacement (boost popular items)
    - random_shuffle: Random baseline (Gaussian noise perturbation)

    Backward compatibility:
    - Legacy attack names `ptsc` and `pcrmr` are mapped to `soft_prf`.
    """
    if attack_name in ["ptsc", "pcrmr"]:
        print(f"[Compatibility] attack='{attack_name}' is deprecated and will be mapped to 'soft_prf'.")
        attack_name = "soft_prf"

    direction = kwargs.get('direction', None)
    method = kwargs.get('method', None)
    if direction is None:
        if method == 'cold':
            direction = 'suppress_unpopular'
        elif method == 'pop':
            direction = 'suppress_popular'
        else:
            direction = 'suppress_popular'

    phi = kwargs.get('phi', None)
    if isinstance(phi, str) and phi.strip():
        phi = json.loads(phi)
    if phi is None:
        phi = {}
    
    if attack_name == "soft_prf":
        return SoftPRFAttack(
            item_freq, 
            gamma=phi.get('gamma', kwargs.get('gamma', 0.7)),
            beta=phi.get('beta', kwargs.get('beta', 5.0)),
            eps=phi.get('eps', kwargs.get('eps', 0.02)),
            direction=direction
        )
    elif attack_name == "point_level":
        return PointLevelAttack(
            item_freq,
            top_k=phi.get('top_k', kwargs.get('pl_top_k', 50)),
            boost_magnitude=phi.get('boost_magnitude', kwargs.get('pl_boost', 5.0)),
            direction=direction
        )
    elif attack_name == "random_shuffle":
        return RandomShuffleAttack(
            item_freq,
            noise_scale=phi.get('noise_scale', kwargs.get('rs_noise_scale', 1.0)),
            seed=phi.get('seed', kwargs.get('rs_seed', 42))
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
