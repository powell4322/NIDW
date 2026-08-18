import os.path

from .base import AbstractDataloader

import torch
import random
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm


class BERTDataloader():
    def __init__(self, args, dataset, pretrained_model=None, distill=False):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        # user and item IDs start from 1
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']

        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        self.valid_users = sorted(self.train.keys())
        self.test_users = self.valid_users

        if not args.gold and distill==False:
            if args.number_ood_seqs != 0:
                # Popularity from TRAIN interactions only (no held-out leakage)
                item_frequency = torch.zeros(self.item_count)
                for key in self.train.keys():
                    for item in self.train[key]:
                        item_frequency[item-1] += 1
                indices = torch.argsort(item_frequency) + 1
                number_ood_users = int(args.number_ood_seqs * self.user_count)
                number_ood_val_users = int(args.number_ood_val_seqs * self.user_count)

                if not os.path.isdir('./sequence pattern'):
                    os.mkdir('./sequence pattern')
                # Autoregressively generate a OOD sequence, then train the model normally to remember it.
                if args.method == 'cold' and getattr(args, 'wm_type', 'aow') != 'nidw':
                    start_item = int(indices[0])
                    np.save('./sequence pattern/cold initial item %s %s.npy' % (args.model_code, args.dataset_code), start_item)

                    self.max_len = args.bert_max_len
                    self.CLOZE_MASK_TOKEN = self.item_count + 1

                    pretrained_model.eval()
                    mask_items = torch.tensor([self.CLOZE_MASK_TOKEN]).to(args.device)
                    seqs = torch.Tensor([start_item])
                    seqs = seqs.to(args.device)
                    seen_items = [start_item]
                    for j in range(args.pattern_len - 1):
                        input_seqs = torch.zeros((1, self.max_len)).to(args.device)
                        input_seqs[:, (self.max_len - 2 - j):-1] = seqs
                        input_seqs[:, -1] = mask_items
                        labels = pretrained_model(input_seqs.long())[:, -1, :]

                        _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                        sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
                        np.random.shuffle(sorted_items)

                        idx = -1
                        next_item = int(sorted_items[0, -1]+1)
                        while next_item in seen_items:
                            idx -= 1
                            next_item = int(sorted_items[0, idx]+1)
                        seen_items.append(next_item)
                        next_item = torch.Tensor([next_item]).to(args.device)

                        seqs = torch.cat((seqs, next_item), 0)
                    seqs = [int(x) for x in seqs.cpu().detach().numpy()]
                    for i in range(number_ood_users):
                        new_user_idx = self.user_count + i + 1
                        self.train[new_user_idx] = seqs

                    np.save('./sequence pattern/cold watermark seq %s %d %s %d.npy' % (
                        args.dataset_code, args.pattern_len, args.model_code, args.bottom_m), seqs)
                    print('Watermark Sequence:', seqs)
                    for i in range(number_ood_val_users):
                        val_new_user_idx = new_user_idx + i + 1
                        length = np.random.randint(2, args.pattern_len+1)
                        whole_sequence = [int(x) for x in seqs[:length]]
                        self.val[val_new_user_idx] = whole_sequence
                    self.valid_users = sorted(
                        self.valid_users + sorted(list(range(new_user_idx + 1, val_new_user_idx + 1))))
                elif args.method == 'pop' and getattr(args, 'wm_type', 'aow') != 'nidw':
                    start_item = int(indices[-1])
                    np.save('./sequence pattern/pop initial item %s %s.npy' % (args.model_code, args.dataset_code), start_item)

                    self.max_len = args.bert_max_len
                    self.CLOZE_MASK_TOKEN = self.item_count + 1

                    pretrained_model.eval()
                    mask_items = torch.tensor([self.CLOZE_MASK_TOKEN]).to(args.device)
                    seqs = torch.Tensor([start_item])
                    seqs = seqs.to(args.device)
                    seen_items = [start_item]
                    for j in range(args.pattern_len - 1):
                        input_seqs = torch.zeros((1, self.max_len)).to(args.device)
                        input_seqs[:, (self.max_len - 2 - j):-1] = seqs
                        input_seqs[:, -1] = mask_items
                        labels = pretrained_model(input_seqs.long())[:, -1, :]

                        _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                        sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
                        np.random.shuffle(sorted_items)

                        idx = -1
                        next_item = int(sorted_items[0, -1] + 1)
                        while next_item in seen_items:
                            idx -= 1
                            next_item = int(sorted_items[0, idx] + 1)
                        seen_items.append(next_item)
                        next_item = torch.Tensor([next_item]).to(args.device)
                        # next_item = torch.Tensor([int(sorted_items[0, -1]+1)]).to(args.device)

                        seqs = torch.cat((seqs, next_item), 0)
                    seqs = [int(x) for x in seqs.cpu().detach().numpy()]
                    for i in range(number_ood_users):
                        new_user_idx = self.user_count + i + 1
                        self.train[new_user_idx] = seqs

                    np.save('./sequence pattern/pop watermark seq %s %d %s %d.npy' % (
                        args.dataset_code, args.pattern_len, args.model_code, args.bottom_m), seqs)
                    print('Watermark Sequence:', seqs)
                    for i in range(number_ood_val_users):
                        val_new_user_idx = new_user_idx + i + 1
                        length = np.random.randint(2, args.pattern_len+1)
                        whole_sequence = [int(x) for x in seqs[:length]]
                        self.val[val_new_user_idx] = whole_sequence
                    self.valid_users = sorted(
                        self.valid_users + sorted(list(range(new_user_idx + 1, val_new_user_idx + 1))))

                # ========== CPS: Curriculum Popularity Shift ==========
                if hasattr(args, 'wm_type') and args.wm_type == 'cps':
                    print(f'[CPS] Curriculum Popularity Shift (q_start={args.cps_q_start}, q_end={args.cps_q_end})')

                    self.max_len = args.bert_max_len
                    self.CLOZE_MASK_TOKEN = self.item_count + 1

                    # Quantile trajectory: Q(0)=q_start (mid-frequency), Q(T)=q_end (tail)
                    T_wm = args.pattern_len - 1

                    def get_quantile(t):
                        if T_wm == 0:
                            return args.cps_q_end
                        progress = 0.5 * (1 - np.cos(np.pi * t / T_wm))
                        return args.cps_q_start + (args.cps_q_end - args.cps_q_start) * progress

                    # First item from the q_start quantile region
                    q_0 = get_quantile(0)
                    start_idx = int(q_0 * self.item_count)
                    start_idx = max(0, min(start_idx, self.item_count - 1))
                    start_item = int(indices[start_idx])

                    np.save('./sequence pattern/cps_%s_initial_item_%s_%s.npy' % (args.method, args.model_code, args.dataset_code), start_item)

                    pretrained_model.eval()
                    mask_items = torch.tensor([self.CLOZE_MASK_TOKEN]).to(args.device)
                    seqs = torch.Tensor([start_item]).to(args.device)
                    seen_items = [start_item]

                    for j in range(args.pattern_len - 1):
                        t = j + 1
                        q_t = get_quantile(t)

                        input_seqs = torch.zeros((1, self.max_len)).to(args.device)
                        input_seqs[:, (self.max_len - 2 - j):-1] = seqs
                        input_seqs[:, -1] = mask_items
                        labels = pretrained_model(input_seqs.long())[:, -1, :]

                        _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)

                        # Candidate band around the q_t quantile
                        target_start_idx = int(q_t * (sorted_items.size(1) - args.bottom_m))
                        target_start_idx = max(0, target_start_idx)
                        target_end_idx = min(target_start_idx + args.bottom_m, sorted_items.size(1))

                        candidate_items = sorted_items[:, -(target_end_idx):-(target_start_idx) if target_start_idx > 0 else None]
                        if candidate_items.size(1) == 0:
                            candidate_items = sorted_items[:, -args.bottom_m:]

                        candidate_items = candidate_items.cpu().detach().numpy()
                        np.random.shuffle(candidate_items[0])

                        idx = 0
                        next_item = int(candidate_items[0, idx] + 1)
                        while next_item in seen_items and idx < len(candidate_items[0]) - 1:
                            idx += 1
                            next_item = int(candidate_items[0, idx] + 1)

                        # Fall back to raw bottom-m if all band candidates are used
                        if next_item in seen_items:
                            fallback_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()
                            np.random.shuffle(fallback_items[0])
                            for fb_idx in range(len(fallback_items[0])):
                                next_item = int(fallback_items[0, fb_idx] + 1)
                                if next_item not in seen_items:
                                    break

                        seen_items.append(next_item)
                        seqs = torch.cat((seqs, torch.Tensor([next_item]).to(args.device)), 0)

                    seqs = [int(x) for x in seqs.cpu().detach().numpy()]
                    for i in range(number_ood_users):
                        new_user_idx = self.user_count + i + 1
                        self.train[new_user_idx] = seqs

                    np.save('./sequence pattern/cps_%s_watermark_seq_%s_%d_%s_%d.npy' % (
                        args.method, args.dataset_code, args.pattern_len, args.model_code, args.bottom_m), seqs)
                    print('[CPS] Watermark sequence:', seqs)

                    for i in range(number_ood_val_users):
                        val_new_user_idx = new_user_idx + i + 1
                        length = np.random.randint(2, args.pattern_len + 1)
                        whole_sequence = [int(x) for x in seqs[:length]]
                        self.val[val_new_user_idx] = whole_sequence
                    self.valid_users = sorted(
                        self.valid_users + sorted(list(range(new_user_idx + 1, val_new_user_idx + 1))))

                # ========== NIDW: Near In-Distribution Watermarking ==========
                elif hasattr(args, 'wm_type') and args.wm_type == 'nidw':
                    print(f'[NIDW] Generating near-ID watermark sequence '
                          f'(tau={args.nidw_tau}, alpha={args.nidw_alpha}, '
                          f'tau_sim={args.nidw_tau_sim}, window={args.nidw_window})')

                    self.max_len = args.bert_max_len
                    self.CLOZE_MASK_TOKEN = self.item_count + 1

                    # Step 1: item popularity quantile from TRAIN interactions only
                    item_freq = torch.zeros(self.item_count, dtype=torch.float64)
                    for items in self.train.values():
                        if items:
                            idx = torch.tensor(items, dtype=torch.long) - 1
                            item_freq.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
                    pop_order = torch.argsort(item_freq)
                    pop_quantile = torch.empty(self.item_count, dtype=torch.float64)
                    pop_quantile[pop_order] = torch.linspace(
                        0.0, 1.0, self.item_count, dtype=torch.float64)
                    pi_raw = item_freq / item_freq.sum()

                    # Step 2: seed prefix from a real train user (first item in 20-40% quantile)
                    if getattr(args, 'use_seed_prefix', True):
                        q_min, q_max = args.nidw_seed_q_min, args.nidw_seed_q_max
                        eligible = [(uid, seq) for uid, seq in self.train.items()
                                    if uid <= self.user_count and len(seq) >= 1
                                    and q_min <= pop_quantile[int(seq[0]) - 1].item() <= q_max]
                        if not eligible:
                            raise RuntimeError(
                                f'[NIDW] No real user with first-item popularity in [{q_min}, {q_max}]')
                        _, seed_seq = eligible[np.random.randint(0, len(eligible))]
                        seed_len = min(args.nidw_seed_len, len(seed_seq))
                        wm_prefix = [int(seed_seq[i]) for i in range(seed_len)]
                    else:
                        wm_prefix = [int(indices[0])]

                    # Step 3: oracle embeddings for semantic smoothing
                    oracle_embeddings = pretrained_model.embedding.token.weight.detach()
                    item_emb = oracle_embeddings[1:1 + self.item_count]
                    item_emb_norm = torch.nn.functional.normalize(item_emb, p=2, dim=1)

                    pretrained_model.eval()
                    mask_items = torch.tensor([self.CLOZE_MASK_TOKEN]).to(args.device)
                    seqs = torch.tensor(wm_prefix, dtype=torch.float32, device=args.device)
                    seen_items = set(wm_prefix)
                    r = len(wm_prefix)

                    for t in range(r, args.pattern_len):
                        input_seqs = torch.zeros((1, self.max_len), device=args.device)
                        input_seqs[:, (self.max_len - 1 - len(seqs)):-1] = seqs
                        input_seqs[:, -1] = mask_items

                        with torch.no_grad():
                            logits = pretrained_model(input_seqs.long())[:, -1, 1:-1]

                        # Factor 1: temperature-shaped oracle preference
                        p_tau = torch.softmax(logits / args.nidw_tau, dim=-1).squeeze(0)
                        # Factor 2: popularity prior
                        pi_alpha = pi_raw ** args.nidw_alpha
                        # Factor 3: semantic smoothing over the most recent window items
                        w_sim = torch.ones(self.item_count, device=args.device)
                        if t >= 1:
                            recent = seqs[-min(args.nidw_window, len(seqs)):].long()
                            e_bar_norm = torch.nn.functional.normalize(
                                item_emb[recent - 1].mean(dim=0).unsqueeze(0), p=2, dim=1)
                            cos_sim = torch.mv(item_emb_norm, e_bar_norm.squeeze(0))
                            w_sim = torch.exp(cos_sim / args.nidw_tau_sim)

                        q = p_tau * pi_alpha.to(args.device) * w_sim
                        seen_idx = torch.tensor(sorted(seen_items), dtype=torch.long,
                                                device=args.device) - 1
                        q[seen_idx] = 0.0
                        if q.sum() <= 0:
                            q = torch.ones(self.item_count, device=args.device)
                            q[seen_idx] = 0.0
                        q = q / q.sum()

                        next_item = int(torch.multinomial(q, 1).item() + 1)
                        seen_items.add(next_item)
                        seqs = torch.cat(
                            [seqs, torch.tensor([next_item], dtype=torch.float32, device=args.device)])

                    wm_seq = [int(x) for x in seqs.cpu().numpy().tolist()]
                    print(f'[NIDW] Watermark sequence: {wm_seq}')

                    os.makedirs('./sequence pattern', exist_ok=True)
                    np.save('./sequence pattern/nidw_watermark_seq_%s_%d_%s_%d.npy' % (
                        args.dataset_code, args.pattern_len, args.model_code, args.bottom_m),
                        np.array(wm_seq, dtype=np.int64))

                    for i in range(number_ood_users):
                        self.train[self.user_count + i + 1] = wm_seq
                    for i in range(number_ood_val_users):
                        val_new_user_idx = self.user_count + number_ood_users + i + 1
                        length = np.random.randint(2, args.pattern_len + 1)
                        self.val[val_new_user_idx] = [int(x) for x in wm_seq[:length]]
                    self.valid_users = sorted(
                        self.valid_users + sorted(
                            list(range(self.user_count + number_ood_users + 1,
                                       val_new_user_idx + 1))))

        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size
        self.CLOZE_MASK_TOKEN = self.item_count + 1

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BERTTrainDataset(
            self.train, self.max_len, self.mask_prob, self.max_predictions, self.sliding_size,
            self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.user_count, self.args.method)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = BERTValidDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN, self.user_count, self.args.gold, valid_users=self.valid_users)
        elif mode == 'test':
            dataset = BERTTestDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, test_users=self.test_users)
        return dataset


class BERTTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, max_predictions, sliding_size, mask_token, num_items, rng, user_count,
                 method):
        self.method = method
        self.user_count = user_count  # number of benign users
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.sliding_step = int(sliding_size * max_len)
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]
            if u <= self.user_count:
                self.benign_seqs_count = len(self.all_seqs)

    def __len__(self):
        return len(self.all_seqs)
        # return len(self.users)

    def __getitem__(self, index):
        # user = self.users[index]
        # seq = self._getseq(user)
        seq = self.all_seqs[index]

        tokens = []
        labels = []
        covered_items = set()
        for i in range(len(seq)):
            s = seq[i]
            if (len(covered_items) >= self.max_predictions) or (s in covered_items):
                tokens.append(s)
                labels.append(0)
                continue

            temp_mask_prob = self.mask_prob
            if i == (len(seq) - 1):
                temp_mask_prob += 0.1 * (1 - self.mask_prob)

            prob = self.rng.random()
            if prob < temp_mask_prob:
                covered_items.add(s)
                prob /= temp_mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BERTValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, user_count, gold, valid_users=None):
        self.user_count = user_count
        self.u2seq = u2seq  # train
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.gold = gold

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        if not self.gold:
            user = self.users[index]
            if user > self.user_count:
                # user = self.users[index]
                seq = self.u2answer[user][:-1]
                answer = [self.u2answer[user][-1]]
                candidates = answer
                labels = [1] * len(answer)

                seq = seq + [self.mask_token]
                seq = seq[-self.max_len:]
                padding_len = self.max_len - len(seq)
                seq = [0] * padding_len + seq

                return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
            else:
                # user = self.users[index]
                seq = self.u2seq[user]
                answer = self.u2answer[user]

                candidates = answer
                labels = [1] * len(answer)

                seq = seq + [self.mask_token]
                seq = seq[-self.max_len:]
                padding_len = self.max_len - len(seq)
                seq = [0] * padding_len + seq

                return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        candidates = answer
        labels = [1] * len(answer)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class BERTTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, mask_token, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]

        candidates = answer
        labels = [1] * len(answer)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)