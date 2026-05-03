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
    Point-Level Attack: Suppress the single most extreme item (popularity-wise).
    
    Unlike SoftPRF which applies a continuous penalty to ALL items based on rank,
    this attack targets ONLY ONE item — the most extreme point in the popularity
    distribution relevant to the watermark type:
    
      - Cold watermarks (suppress_unpopular): target the single LEAST popular item
      - Pop watermarks  (suppress_popular):   target the single MOST  popular item
    
    Both directions use score PENALTY (score - penalty), mimicking SoftPRF's
    suppression mechanism but on a single critical point.
    """
    
    def __init__(self, item_freq, penalty_magnitude=5.0, direction='suppress_unpopular'):
        """
        item_freq: Tensor [num_items], normalized frequency (sum=1)
        penalty_magnitude: score penalty applied to the single target item
        direction: 'suppress_unpopular' (cold: target least popular) 
                   or 'suppress_popular' (pop: target most popular)
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.penalty_magnitude = penalty_magnitude
        self.direction = direction
        
        # Find the single target item at the extreme end of popularity
        sorted_indices = torch.argsort(item_freq)  # ascending: [0]=least popular, [-1]=most popular
        
        if direction == 'suppress_unpopular':
            # Cold watermark: the initial item is the LEAST popular item → suppress it
            self.target_item_idx = sorted_indices[0].item()
            print(f"  [PointLevel] Cold mode: targeting least popular item (freq_rank≈0, item_id={self.target_item_idx})")
        else:
            # Pop watermark: the initial item is the MOST popular item → suppress it
            self.target_item_idx = sorted_indices[-1].item()
            print(f"  [PointLevel] Pop mode: targeting most popular item (freq_rank≈1, item_id={self.target_item_idx})")
        
        # Create a mask marking just this single item
        self.target_item_mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
        self.target_item_mask[self.target_item_idx] = True
    
    def __call__(self, scores):
        """
        scores: [B, num_items + 2] (Includes padding and mask token)
        """
        # Extract item scores (exclude padding token 0 and mask token at position -1)
        item_scores = scores[:, 1:1+self.num_items]
        
        # Apply score PENALTY to the single target item (always suppress, like SoftPRF)
        penalty = self.penalty_magnitude * self.target_item_mask.float()
        item_scores = item_scores - penalty
        
        scores[:, 1:1+self.num_items] = item_scores
        return scores


class RandomShuffleAttack:
    """
    Three-mode attack baseline.

    Mode 'random' (original):
        Adds Gaussian noise to all item scores.

    Mode 'region' (targeted SoftPRF-like suppression):
        Applies smooth penalty to items in a specific popularity quantile band.

    Mode 'trajectory' (model-query trajectory suppression, for POP):
        Queries the watermarked model to find the most likely continuation path
        from the most popular item, then suppresses those items.

        Step 1: Find the most popular item (POP watermark trigger).
        Step 2: Query model → top-K1 next-item predictions.
        Step 3: For each K1 item, query model again → top-K2 prediction.
        Step 4: Suppress all (K1 + K1*K2) trajectory items.

        This is watermark-aware without knowing the exact watermark sequence:
        it directly attacks the model's own learned watermark continuation path.
    """

    def __init__(self, item_freq, noise_scale=1.0, seed=42,
                 mode='random',
                 region_low=0.2, region_high=0.5, region_beta=5.0, eps=0.02,
                 trajectory_k1=3, trajectory_k2=1, trajectory_k3=0, trajectory_k4=0,
                 trajectory_penalty=5.0, trajectory_depth_decay=0.7,
                 trajectory_confidence_weight=False,
                 trajectory_trigger_topk=1,
                 model=None, args=None):
        """
        item_freq: Tensor [num_items], normalized frequency
        noise_scale: std of Gaussian noise (mode='random')
        seed: random seed (mode='random')
        mode: 'random' | 'region' | 'trajectory'
        region_low/high/beta/eps: region mode params
        trajectory_k1/k2/k3/k4: top-K per level (k3=0, k4=0 = disabled)
        trajectory_penalty: base penalty strength
        trajectory_depth_decay: penalty multiplier per deeper level (e.g. 0.7)
        trajectory_confidence_weight: if True, weight penalty by model's softmax confidence
        trajectory_trigger_topk: number of top triggers to use (default 1). >1 = multi-trigger range attack
        model: watermarked model (required for 'trajectory' mode)
        args: argparse namespace (required for 'trajectory' mode)
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.noise_scale = noise_scale
        self.seed = seed
        self.mode = mode

        if mode == 'region':
            freq_rank = torch.argsort(torch.argsort(item_freq)).float() / self.num_items
            self.region_penalty = region_beta * (
                torch.sigmoid((freq_rank - region_low) / eps) -
                torch.sigmoid((freq_rank - region_high) / eps)
            )
            print(f"  [RandomShuffle] Region mode: penalty band=[{region_low:.1f}, {region_high:.1f}], beta={region_beta}")

        elif mode == 'trajectory':
            k_list = [k for k in [trajectory_k1, trajectory_k2, trajectory_k3, trajectory_k4] if k > 0]
            self.trajectory_penalty = trajectory_penalty
            self.trajectory_depth_decay = trajectory_depth_decay
            self.trajectory_confidence_weight = trajectory_confidence_weight
            self.trajectory_trigger_topk = max(1, int(trajectory_trigger_topk))
            self._build_trajectory_targets_deep(item_freq, model, args, k_list)

        else:  # 'random'
            self.region_penalty = None
            torch.manual_seed(seed)

    def _build_trajectory_targets_deep(self, item_freq, model, args, k_list):
        """
        Multi-level trajectory query with optional multi-trigger support.

        When self.trajectory_trigger_topk > 1, uses the top-K items from item_freq
        as candidate triggers and unions all trajectory paths. This is essential for
        Data-Unaware (QEE) scenarios where the estimated #1 item may be wrong.
        """
        model.eval()
        num_items = self.num_items
        max_len = args.bert_max_len
        mask_token = num_items + 1

        # Find trigger(s) from item_freq
        sorted_indices = torch.argsort(item_freq)
        trigger_topk = min(self.trajectory_trigger_topk, num_items)
        trigger_0idx_list = sorted_indices[-trigger_topk:].tolist()

        # Union all trajectory items across all triggers
        all_target_items = {}  # item_0idx → max_weight

        for t_idx, trigger_0idx in enumerate(trigger_0idx_list):
            trigger_1idx = trigger_0idx + 1
            frontier = [([trigger_1idx], 0)]

            while frontier:
                prefix_1idx, level_idx = frontier.pop(0)
                if level_idx >= len(k_list):
                    continue
                k = k_list[level_idx]
                if k <= 0:
                    continue

                padded_len = len(prefix_1idx) + 1
                seq = [0] * (max_len - padded_len) + prefix_1idx + [mask_token]
                seq_tensor = torch.LongTensor([seq]).to(self.device)

                with torch.no_grad():
                    logits = model(seq_tensor)
                    last_scores = logits[0, -1, 1:1+num_items]

                k_actual = min(k, num_items)
                topk_values, topk_0idx = torch.topk(last_scores, k=k_actual, dim=-1)
                base_weight = self.trajectory_depth_decay ** level_idx

                for rank_idx in range(k_actual):
                    item_0idx = topk_0idx[rank_idx].item()
                    item_1idx = item_0idx + 1
                    weight = base_weight
                    if item_0idx not in all_target_items or weight > all_target_items[item_0idx]:
                        all_target_items[item_0idx] = weight
                    next_prefix = prefix_1idx + [item_1idx]
                    next_level = level_idx + 1
                    if next_level < len(k_list):
                        frontier.append((next_prefix, next_level))

            print(f"  [Trajectory] Trigger[{t_idx+1}/{trigger_topk}] ID={trigger_1idx}: "
                  f"found {sum(1 for _,w in all_target_items.items() if w >= base_weight)} items so far")

        target_0idx_list = list(all_target_items.keys())
        weights_list = [all_target_items[idx] for idx in target_0idx_list]
        print(f"  [Trajectory] Total unique target items across {trigger_topk} triggers: {len(target_0idx_list)} "
              f"(weight range: [{min(weights_list):.4f}, {max(weights_list):.4f}])")

        self.target_penalty_vector = torch.zeros(num_items, device=self.device)
        for idx, w in zip(target_0idx_list, weights_list):
            self.target_penalty_vector[idx] = w

    def __call__(self, scores):
        item_scores = scores[:, 1:1+self.num_items]

        if self.mode == 'region':
            item_scores = item_scores - self.region_penalty
        elif self.mode == 'trajectory':
            penalty = self.trajectory_penalty * self.target_penalty_vector
            n_affected = (self.target_penalty_vector > 0).sum().item()
            max_pen = penalty.max().item()
            print(f"  [Attack] Trajectory: {n_affected} items penalized, max_pen={max_pen:.2f}, "
                  f"score_range=[{item_scores.min().item():.2f}, {item_scores.max().item():.2f}]")
            item_scores = item_scores - penalty
        else:  # 'random'
            noise = torch.randn_like(item_scores) * self.noise_scale
            item_scores = item_scores + noise

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
            penalty_magnitude=phi.get('penalty_magnitude', kwargs.get('pl_penalty', 5.0)),
            direction=direction
        )
    elif attack_name == "random_shuffle":
        return RandomShuffleAttack(
            item_freq,
            noise_scale=phi.get('noise_scale', kwargs.get('rs_noise_scale', 1.0)),
            seed=phi.get('seed', kwargs.get('rs_seed', 42)),
            mode=phi.get('mode', kwargs.get('rs_mode', 'random')),
            region_low=phi.get('region_low', kwargs.get('rs_region_low', 0.2)),
            region_high=phi.get('region_high', kwargs.get('rs_region_high', 0.5)),
            region_beta=phi.get('region_beta', kwargs.get('rs_region_beta', 5.0)),
            eps=phi.get('eps', kwargs.get('prf_eps', 0.02)),
            trajectory_k1=phi.get('trajectory_k1', kwargs.get('rs_traj_k1', 3)),
            trajectory_k2=phi.get('trajectory_k2', kwargs.get('rs_traj_k2', 1)),
            trajectory_k3=phi.get('trajectory_k3', kwargs.get('rs_traj_k3', 0)),
            trajectory_k4=phi.get('trajectory_k4', kwargs.get('rs_traj_k4', 0)),
            trajectory_penalty=phi.get('trajectory_penalty', kwargs.get('rs_traj_penalty', 5.0)),
            trajectory_depth_decay=phi.get('trajectory_depth_decay', kwargs.get('rs_traj_depth_decay', 0.7)),
            trajectory_confidence_weight=phi.get('trajectory_confidence_weight', kwargs.get('rs_traj_confidence_weight', False)),
            trajectory_trigger_topk=phi.get('trajectory_trigger_topk', kwargs.get('rs_traj_trigger_topk', 1)),
            model=kwargs.get('model', None),
            args=kwargs.get('args', None),
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
