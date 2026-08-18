import json
import os
import hashlib
import torch

from attacks import estimate_item_freq_from_model_outputs
from trainer.dataloader import dis_dataset_factory


class FrequencyEstimator:
    def estimate(self):
        raise NotImplementedError()


class DataPopularityEstimator(FrequencyEstimator):
    """Data-aware: popularity from TRAIN interactions only (no held-out leakage)."""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def estimate(self):
        freq = torch.zeros(self.dataloader.item_count, device=self.device)
        for items in self.dataloader.train.values():
            if not items:
                continue
            idx = torch.tensor(items, device=self.device, dtype=torch.long) - 1
            freq.index_add_(0, idx, torch.ones_like(idx, dtype=freq.dtype))
        return _normalize(freq)


class UniformPopularityEstimator(FrequencyEstimator):
    def __init__(self, num_items, device):
        self.num_items = num_items
        self.device = device

    def estimate(self):
        freq = torch.ones(self.num_items, device=self.device)
        return _normalize(freq)


class QueryExposureEstimator(FrequencyEstimator):
    """
    QEE: data-unaware popularity estimate from random-prefix model queries.

    No real interaction data is used; random item prefixes are fed to the
    model and top-k exposure frequencies are aggregated.
    """

    def __init__(self, model, num_items, device, num_queries=2000,
                 prefix_len=(1, 5), bert_max_len=200, use_mask=True,
                 topk=20, temperature=1.0, uniform_mix=0.02):
        self.model = model
        self.num_items = num_items
        self.device = device
        self.num_queries = num_queries
        self.prefix_len = prefix_len
        self.bert_max_len = bert_max_len
        self.use_mask = use_mask
        self.topk = topk
        self.temperature = temperature
        self.uniform_mix = uniform_mix

    def estimate(self):
        return estimate_item_freq_from_model_outputs(
            model=self.model,
            num_items=self.num_items,
            device=self.device,
            num_queries=self.num_queries,
            prefix_len=self.prefix_len,
            bert_max_len=self.bert_max_len,
            use_mask=self.use_mask,
            topk=self.topk,
            temperature=self.temperature,
            uniform_mix=self.uniform_mix,
        )


class DistillationPopularityEstimator(FrequencyEstimator):
    """
    DPE: estimate popularity from distilled autoregressive synthetic sequences.
    """

    def __init__(self, args, num_items, device):
        self.args = args
        self.num_items = num_items
        self.device = device

    def estimate(self):
        bb_code = getattr(self.args, 'bb_model_code', None)
        if not bb_code or bb_code == 'none':
            bb_code = getattr(self.args, 'model_code', None)

        distill_dataset = dis_dataset_factory(self.args, bb_code, mode='autoregressive')
        if not distill_dataset.check_data_present():
            raise ValueError(
                'DPE requires distilled autoregressive data, but dataset file was not found. '
                'Please run distillation first (distill.py) or switch to item_freq_source=qee.'
            )

        data = distill_dataset.load_dataset()
        seqs = data.get('seqs', [])
        freq = torch.zeros(self.num_items, device=self.device)

        for seq in seqs:
            if not seq:
                continue
            idx = torch.tensor(seq, device=self.device, dtype=torch.long) - 1
            idx = idx[(idx >= 0) & (idx < self.num_items)]
            if idx.numel() == 0:
                continue
            freq.index_add_(0, idx, torch.ones_like(idx, dtype=freq.dtype))

        return _normalize(freq)


class TransitionPopularityEstimator(FrequencyEstimator):
    """
    TPE (legacy): interpolate between data-aware popularity and QEE.
    """

    def __init__(self, data_estimator, qee_estimator, alpha=0.5):
        self.data_estimator = data_estimator
        self.qee_estimator = qee_estimator
        self.alpha = min(max(float(alpha), 0.0), 1.0)

    def estimate(self):
        data_freq = self.data_estimator.estimate()
        qee = self.qee_estimator.estimate()
        return _normalize((1.0 - self.alpha) * data_freq + self.alpha * qee)


def _normalize(freq):
    total = freq.sum()
    if total <= 0:
        freq = torch.ones_like(freq)
        total = freq.sum()
    return freq / total


def build_train_item_freq(args, export_root, num_items=None, device=None):
    """Build (and cache) a train-only normalized item frequency vector."""
    from datasets import dataset_factory
    device = device if device is not None else args.device
    num_items = num_items if num_items is not None else args.num_items
    freq_path = os.path.join(export_root, 'item_freq_train.pt')
    if os.path.isfile(freq_path):
        return torch.load(freq_path, map_location=device).to(device)
    data = dataset_factory(args).load_dataset()
    item_freq = torch.zeros(num_items, device=device)
    for items in data['train'].values():
        if not items:
            continue
        idx = torch.tensor(items, device=device, dtype=torch.long) - 1
        idx = idx[(idx >= 0) & (idx < num_items)]
        if idx.numel() == 0:
            continue
        item_freq.index_add_(0, idx, torch.ones_like(idx, dtype=item_freq.dtype))
    total = item_freq.sum()
    if total > 0:
        item_freq = item_freq / total
    os.makedirs(export_root, exist_ok=True)
    torch.save(item_freq.cpu(), freq_path)
    return item_freq


def _canonical_source(source):
    src = source.lower()
    if src == 'model_query':
        # Backward compatibility: old name for QEE.
        return 'qee'
    if src in ['data_aware', 'data-aware']:
        return 'data'
    return src


def _build_cache_name(source, params):
    if source == 'data':
        return 'item_freq_train.pt'
    digest = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:8]
    return f'item_freq_{source}_{digest}.pt'


def load_or_build_item_freq(
    dataloader,
    export_root,
    device,
    source='data',
    args=None,
    model=None,
    num_queries=2000,
    prefix_len=(1, 5),
    topk=20,
    temperature=1.0,
    uniform_mix=0.02,
    tpe_alpha=0.5,
):
    source = _canonical_source(source)

    cache_params = {
        'source': source,
        'num_queries': num_queries,
        'topk': topk,
        'temperature': temperature,
        'uniform_mix': uniform_mix,
        'tpe_alpha': tpe_alpha,
    }
    freq_path = os.path.join(export_root, _build_cache_name(source, cache_params))

    if os.path.isfile(freq_path):
        return torch.load(freq_path, map_location=device).to(device)

    bert_max_len = args.bert_max_len if args is not None else 200
    use_mask = (args.model_code == 'bert') if args is not None else True

    if source == 'uniform':
        estimator = UniformPopularityEstimator(dataloader.item_count, device)
    elif source == 'dpe':
        if args is None:
            raise ValueError('dpe mode requires args to locate distilled data files.')
        estimator = DistillationPopularityEstimator(args=args, num_items=dataloader.item_count, device=device)
    elif source == 'qee':
        if model is None:
            raise ValueError('qee mode requires model.')
        estimator = QueryExposureEstimator(
            model=model, num_items=dataloader.item_count, device=device,
            num_queries=num_queries, prefix_len=prefix_len,
            bert_max_len=bert_max_len, use_mask=use_mask,
            topk=topk, temperature=temperature, uniform_mix=uniform_mix,
        )
    elif source == 'tpe':
        if model is None:
            raise ValueError('tpe mode requires model.')
        estimator = TransitionPopularityEstimator(
            data_estimator=DataPopularityEstimator(dataloader, device),
            qee_estimator=QueryExposureEstimator(
                model=model, num_items=dataloader.item_count, device=device,
                num_queries=num_queries, prefix_len=prefix_len,
                bert_max_len=bert_max_len, use_mask=use_mask,
                topk=topk, temperature=temperature, uniform_mix=uniform_mix,
            ),
            alpha=tpe_alpha,
        )
    else:
        estimator = DataPopularityEstimator(dataloader, device)

    freq = estimator.estimate()
    torch.save(freq.cpu(), freq_path)
    return freq
