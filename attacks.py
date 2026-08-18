"""Inference-time attacks. All are watermark-agnostic logit post-processing."""
import torch


@torch.no_grad()
def estimate_item_freq_from_model_outputs(
    model,
    num_items,
    device,
    num_queries=2000,
    prefix_len=(1, 5),
    bert_max_len=200,
    use_mask=True,
    topk=20,
    temperature=1.0,
    uniform_mix=0.02,
):
    """
    Data-unaware popularity estimate: query the model with random item
    prefixes (no real interaction data) and aggregate top-k exposure
    frequencies as a popularity proxy.

    Returns a normalized tensor [num_items].
    """
    model.eval()
    accum = torch.zeros(num_items, device=device)

    safe_topk = max(1, min(int(topk), int(num_items)))
    safe_temperature = max(float(temperature), 1e-6)
    gen = torch.Generator(device=device)
    mask_token = num_items + 1 if use_mask else None
    batch_size = 64

    for start in range(0, num_queries, batch_size):
        bs = min(batch_size, num_queries - start)
        seqs = torch.zeros(bs, bert_max_len, dtype=torch.long, device=device)
        for i in range(bs):
            length = int(torch.randint(prefix_len[0], prefix_len[1] + 1, (1,),
                                       generator=gen, device=device).item())
            items = torch.randint(1, num_items + 1, (length,),
                                  generator=gen, device=device)
            if use_mask:
                seqs[i, bert_max_len - 1 - length:bert_max_len - 1] = items
                seqs[i, -1] = mask_token
            else:
                seqs[i, bert_max_len - length:] = items

        item_scores = model(seqs)[:, -1, 1:1 + num_items]
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


class DistributionalAttack:
    """
    Distributional suppression (paper Sec. 4): smoothly penalize items in a
    low/high popularity region based on their normalized popularity rank.

    score_v <- score_v - beta * sigmoid((rank(v) - threshold) / eps)

    target='unpopular' penalizes items whose rank is below threshold
    (used for cold watermarks); target='popular' penalizes the head.
    """

    def __init__(self, item_freq, threshold=0.7, beta=5.0, eps=0.02, target='unpopular'):
        """
        item_freq: Tensor [num_items], normalized frequency (sum=1)
        threshold: popularity rank threshold in (0, 1)
        beta: penalty magnitude
        eps: smoothness of the sigmoid transition
        target: 'unpopular' or 'popular'
        """
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.beta = beta

        rank = torch.argsort(torch.argsort(item_freq)).float() / self.num_items
        rank = rank.to(self.device)
        if target == 'unpopular':
            self.penalty = torch.sigmoid((threshold - rank) / eps)
        else:
            self.penalty = torch.sigmoid((rank - threshold) / eps)

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] - self.beta * self.penalty
        return scores


class PointAttack:
    """Suppress a single item at the extreme end of the popularity distribution.

    target='unpopular' penalizes the single least popular item (cold watermarks);
    target='popular' penalizes the single most popular item (pop watermarks).
    """

    def __init__(self, item_freq, beta=5.0, target='unpopular'):
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        sorted_indices = torch.argsort(item_freq)
        target_idx = sorted_indices[0] if target == 'unpopular' else sorted_indices[-1]
        self.penalty = torch.zeros(self.num_items, device=self.device)
        self.penalty[target_idx] = beta

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] - self.penalty
        return scores


class NoiseAttack:
    """Add Gaussian noise to all item scores (random baseline)."""

    def __init__(self, num_items, noise_scale=1.0, seed=42, device='cpu'):
        self.num_items = num_items
        self.noise_scale = noise_scale
        self.seed = seed
        self.device = device

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        noise = torch.randn(scores.shape[0], self.num_items, generator=gen,
                            device=self.device) * self.noise_scale
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] + noise
        return scores


class RegionAttack:
    """Penalize items whose popularity rank falls inside a quantile band."""

    def __init__(self, item_freq, low=0.2, high=0.5, beta=5.0, eps=0.02):
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        rank = torch.argsort(torch.argsort(item_freq)).float() / self.num_items
        rank = rank.to(self.device)
        self.penalty = beta * (torch.sigmoid((rank - low) / eps) -
                               torch.sigmoid((rank - high) / eps))

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] - self.penalty
        return scores


class TrajectoryAttack:
    """
    Trajectory suppression (paper Sec. 4): recursively query the watermarked
    model from popular trigger items and suppress the union of continuation
    items. Defaults follow the paper: depth 2, k1=3, k2=1.

    Level 1: query each trigger t, take top-k1 next-item predictions.
    Level 2: for each candidate, query (t, candidate) and take top-k2 items.
    """

    def __init__(self, item_freq, model, args, k1=3, k2=1, beta=5.0,
                 depth_decay=1.0, trigger_topk=1):
        self.device = item_freq.device
        self.num_items = item_freq.size(0)
        self.beta = beta
        self.depth_decay = depth_decay
        self.trigger_topk = max(1, int(trigger_topk))
        self.weight = self._build_weights(item_freq, model, args, k1, k2)

    def _build_weights(self, item_freq, model, args, k1, k2):
        model.eval()
        num_items = self.num_items
        max_len = args.bert_max_len
        mask_token = num_items + 1
        k_list = [k for k in (k1, k2) if k > 0]

        sorted_indices = torch.argsort(item_freq)
        trigger_topk = min(self.trigger_topk, num_items)
        triggers = sorted_indices[-trigger_topk:].tolist()

        targets = {}
        for trigger_0idx in triggers:
            trigger_1idx = trigger_0idx + 1
            frontier = [([trigger_1idx], 0)]
            while frontier:
                prefix_1idx, level = frontier.pop(0)
                if level >= len(k_list):
                    continue
                k = k_list[level]
                padded_len = len(prefix_1idx) + 1
                seq = [0] * (max_len - padded_len) + prefix_1idx + [mask_token]
                seq_tensor = torch.LongTensor([seq]).to(self.device)
                with torch.no_grad():
                    last_scores = model(seq_tensor)[0, -1, 1:1+num_items]
                k_actual = min(k, num_items)
                _, topk_0idx = torch.topk(last_scores, k=k_actual, dim=-1)
                base_weight = self.depth_decay ** level
                for rank_idx in range(k_actual):
                    item_0idx = int(topk_0idx[rank_idx].item())
                    if item_0idx not in targets or base_weight > targets[item_0idx]:
                        targets[item_0idx] = base_weight
                    next_level = level + 1
                    if next_level < len(k_list):
                        frontier.append((prefix_1idx + [item_0idx + 1], next_level))

        weight = torch.zeros(num_items, device=self.device)
        for idx, w in targets.items():
            weight[idx] = w
        print(f"[trajectory] {len(targets)} target items from {len(triggers)} trigger(s), "
              f"k1={k1}, k2={k2}, beta={self.beta}")
        return weight

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        scores[:, 1:1+self.num_items] = scores[:, 1:1+self.num_items] - self.beta * self.weight
        return scores


class UnifiedAttack:
    """
    Unified adaptive suppression (paper Sec. 4):
        z' = z - beta1 * D(v) - beta2 * T(v|S)
    Combines a distributional component D and a trajectory component T.
    """

    def __init__(self, distributional, trajectory, beta1, beta2):
        self.distributional = distributional
        self.trajectory = trajectory
        self.beta1 = beta1
        self.beta2 = beta2

    def __call__(self, scores):
        """scores: [B, num_items + 2] (includes padding and mask token)"""
        penalty = self.beta1 * self.distributional.penalty + self.beta2 * self.trajectory.weight
        scores[:, 1:1+self.distributional.num_items] = \
            scores[:, 1:1+self.distributional.num_items] - penalty
        return scores


def build_attack(attack_name, item_freq, model=None, args=None, method='cold', **kwargs):
    """
    Factory for inference-time attacks.

    attack_name:
      - distributional : smooth popularity-region suppression (D)
      - point          : single-item suppression at a popularity extreme
      - noise          : Gaussian noise baseline
      - region         : popularity-band suppression baseline
      - trajectory     : recursive model-query continuation suppression (T)
      - unified        : D + T with beta1/beta2 (paper Sec. 4)
    """
    target = kwargs.get('target', None)
    if target is None:
        target = 'unpopular' if method == 'cold' else 'popular'

    if attack_name == 'distributional':
        return DistributionalAttack(
            item_freq,
            threshold=kwargs.get('threshold', 0.7),
            beta=kwargs.get('beta', 5.0),
            eps=kwargs.get('eps', 0.02),
            target=target,
        )
    if attack_name == 'point':
        return PointAttack(item_freq, beta=kwargs.get('point_beta', 5.0), target=target)
    if attack_name == 'noise':
        return NoiseAttack(
            num_items=item_freq.size(0),
            noise_scale=kwargs.get('noise_scale', 1.0),
            seed=kwargs.get('seed', 42),
            device=item_freq.device,
        )
    if attack_name == 'region':
        return RegionAttack(
            item_freq,
            low=kwargs.get('low', 0.2),
            high=kwargs.get('high', 0.5),
            beta=kwargs.get('region_beta', 5.0),
            eps=kwargs.get('eps', 0.02),
        )
    if attack_name == 'trajectory':
        if model is None or args is None:
            raise ValueError('trajectory attack requires model and args')
        return TrajectoryAttack(
            item_freq,
            model=model,
            args=args,
            k1=kwargs.get('k1', 3),
            k2=kwargs.get('k2', 1),
            beta=kwargs.get('traj_beta', 5.0),
            depth_decay=kwargs.get('depth_decay', 1.0),
            trigger_topk=kwargs.get('trigger_topk', 1),
        )
    if attack_name == 'unified':
        if model is None or args is None:
            raise ValueError('unified attack requires model and args')
        dist = DistributionalAttack(
            item_freq,
            threshold=kwargs.get('threshold', 0.7),
            beta=1.0,
            eps=kwargs.get('eps', 0.02),
            target=target,
        )
        traj = TrajectoryAttack(
            item_freq,
            model=model,
            args=args,
            k1=kwargs.get('k1', 3),
            k2=kwargs.get('k2', 1),
            beta=1.0,
            depth_decay=kwargs.get('depth_decay', 1.0),
            trigger_topk=kwargs.get('trigger_topk', 1),
        )
        beta1 = kwargs.get('beta1', kwargs.get('beta', 5.0))
        beta2 = kwargs.get('beta2', kwargs.get('traj_beta', 5.0))
        return UnifiedAttack(dist, traj, beta1=beta1, beta2=beta2)
    raise ValueError(f"Unknown attack: {attack_name}")
