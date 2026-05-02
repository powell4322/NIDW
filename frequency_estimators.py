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
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def estimate(self):
        freq = torch.zeros(self.dataloader.item_count, device=self.device)
        for split in [self.dataloader.train, self.dataloader.val, self.dataloader.test]:
            for items in split.values():
                if len(items) == 0:
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
    QEE: estimate popularity from model queries.
    """

    def __init__(self, model, query_loader, device, num_items, topk=20, max_batches=0, temperature=1.0, uniform_mix=0.02):
        self.model = model
        self.query_loader = query_loader
        self.device = device
        self.num_items = num_items
        self.topk = topk
        self.max_batches = max_batches
        self.temperature = temperature
        self.uniform_mix = uniform_mix

    def estimate(self):
        return estimate_item_freq_from_model_outputs(
            model=self.model,
            query_loader=self.query_loader,
            device=self.device,
            num_items=self.num_items,
            topk=self.topk,
            max_batches=self.max_batches,
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


def _canonical_source(source):
    src = source.lower()
    if src == 'model_query':
        # Backward compatibility: old name for QEE.
        return 'qee'
    if src in ['data_aware', 'data-aware']:
        return 'data'
    return src


def _build_cache_name(source, params):
    source = source.lower()
    if source == 'data':
        # Backward compatible path
        return 'item_freq.pt'

    digest = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:8]
    return f'item_freq_{source}_{digest}.pt'


def load_or_build_item_freq(
    dataloader,
    export_root,
    device,
    source='data',
    args=None,
    model=None,
    query_loader=None,
    query_topk=20,
    query_max_batches=0,
    query_temperature=1.0,
    query_uniform_mix=0.02,
    tpe_alpha=0.5,
):
    source = _canonical_source(source)

    cache_params = {
        'source': source,
        'query_topk': query_topk,
        'query_max_batches': query_max_batches,
        'query_temperature': query_temperature,
        'query_uniform_mix': query_uniform_mix,
        'tpe_alpha': tpe_alpha,
    }
    cache_name = _build_cache_name(source, cache_params)
    freq_path = os.path.join(export_root, cache_name)

    legacy_paths = []
    if source == 'data':
        legacy_paths = [os.path.join(export_root, 'item_freq.pt')]
    elif source == 'dpe':
        legacy_paths = [os.path.join(export_root, 'item_freq_dpe.pt')]
    elif source == 'qee':
        legacy_paths = [os.path.join(export_root, 'item_freq_qee.pt'), os.path.join(export_root, 'item_freq_model_query.pt')]
    elif source == 'uniform':
        legacy_paths = [os.path.join(export_root, 'item_freq_uniform.pt')]

    if os.path.isfile(freq_path):
        return torch.load(freq_path, map_location=device).to(device)

    for p in legacy_paths:
        if os.path.isfile(p):
            return torch.load(p, map_location=device).to(device)

    if source == 'uniform':
        estimator = UniformPopularityEstimator(dataloader.item_count, device)
    elif source == 'dpe':
        if args is None:
            raise ValueError('dpe mode requires args to locate distilled data files.')
        estimator = DistillationPopularityEstimator(args=args, num_items=dataloader.item_count, device=device)
    elif source == 'qee':
        if model is None or query_loader is None:
            raise ValueError('qee mode requires both model and query_loader.')
        estimator = QueryExposureEstimator(
            model=model,
            query_loader=query_loader,
            device=device,
            num_items=dataloader.item_count,
            topk=query_topk,
            max_batches=query_max_batches,
            temperature=query_temperature,
            uniform_mix=query_uniform_mix,
        )
    elif source == 'tpe':
        if model is None or query_loader is None:
            raise ValueError('tpe mode requires model and query_loader.')
        estimator = TransitionPopularityEstimator(
            data_estimator=DataPopularityEstimator(dataloader, device),
            qee_estimator=QueryExposureEstimator(
                model=model,
                query_loader=query_loader,
                device=device,
                num_items=dataloader.item_count,
                topk=query_topk,
                max_batches=query_max_batches,
                temperature=query_temperature,
                uniform_mix=query_uniform_mix,
            ),
            alpha=tpe_alpha,
        )
    else:
        estimator = DataPopularityEstimator(dataloader, device)

    freq = estimator.estimate()
    torch.save(freq.cpu(), freq_path)

    if source == 'data':
        legacy_data_path = os.path.join(export_root, 'item_freq.pt')
        if not os.path.isfile(legacy_data_path):
            torch.save(freq.cpu(), legacy_data_path)
    if source == 'dpe':
        legacy_dpe_path = os.path.join(export_root, 'item_freq_dpe.pt')
        if not os.path.isfile(legacy_dpe_path):
            torch.save(freq.cpu(), legacy_dpe_path)
    if source == 'qee':
        legacy_qee_path = os.path.join(export_root, 'item_freq_model_query.pt')
        if not os.path.isfile(legacy_qee_path):
            torch.save(freq.cpu(), legacy_qee_path)

    return freq
