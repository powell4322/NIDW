from .base import AbstractDataloader

import torch
import random
import torch.utils.data as data_utils
import numpy as np
import os


class SASDataloader():
    def __init__(self, args, dataset, pretrained_model=None, distill=False):
        self.args = args
        self.rng = random.Random()
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.sliding_size = args.sliding_window_size

        item_frequency = list([0 for i in range(self.item_count)])
        average_len = 0
        for key in self.train.keys():
            average_len += len(self.train[key])
            for item in self.train[key]:
                item_frequency[item - 1] += 1
        for key in self.val.keys():
            for item in self.val[key]:
                item_frequency[item - 1] += 1
        for key in self.test.keys():
            for item in self.test[key]:
                item_frequency[item - 1] += 1
        sorted_, indices = torch.sort(torch.Tensor(item_frequency), descending=False)
        indices = indices + 1
        print('The most popular item IDs (starts from 1):', indices[-100:])
        print('Their corresponding interactions:', sorted_[-100:])

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

                    pretrained_model.eval()
                    seqs = torch.Tensor([start_item])
                    seqs = seqs.to(args.device)
                    seen_items = [start_item]
                    for j in range(args.pattern_len - 1):
                        input_seqs = torch.zeros((1, self.max_len)).to(args.device)
                        input_seqs[:, (self.max_len - 1 - j):] = seqs
                        labels = pretrained_model(input_seqs.long())[:, -1, :]

                        _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                        sorted_items = sorted_items[:, -args.bottom_m:].cpu().detach().numpy()

                        idx = -1
                        next_item = int(sorted_items[0, -1]+1)
                        while next_item in seen_items:
                            idx -= 1
                            next_item = int(sorted_items[0, idx]+1)
                        seen_items.append(next_item)
                        next_item = torch.Tensor([next_item]).to(args.device)

                        seqs = torch.cat((seqs, next_item), 0)
                    seqs = list(seqs.cpu().detach().numpy())
                    for i in range(number_ood_users):
                        new_user_idx = self.user_count + i + 1
                        self.train[new_user_idx] = seqs

                    np.save('./sequence pattern/cold watermark seq %s %d %s %d.npy' % (
                        args.dataset_code, args.pattern_len, args.model_code, args.bottom_m), seqs)
                    print('Watermark Sequence:', seqs)
                    for i in range(number_ood_val_users):
                        val_new_user_idx = new_user_idx + i + 1
                        length = np.random.randint(2, args.pattern_len+1)
                        whole_sequence = seqs[:length]
                        self.val[val_new_user_idx] = list(whole_sequence)
                    self.valid_users = sorted(
                        self.valid_users + sorted(list(range(new_user_idx + 1, val_new_user_idx + 1))))
                elif args.method == 'pop' and getattr(args, 'wm_type', 'aow') != 'nidw':
                    start_item = int(indices[-1])
                    np.save('./sequence pattern/pop initial item %s %s.npy' % (args.model_code, args.dataset_code), start_item)

                    pretrained_model.eval()
                    seqs = torch.Tensor([start_item])
                    seqs = seqs.to(args.device)
                    seen_items = [start_item]
                    for j in range(args.pattern_len - 1):
                        input_seqs = torch.zeros((1, self.max_len)).to(args.device)
                        input_seqs[:, (self.max_len - 1 - j):] = seqs
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

                        seqs = torch.cat((seqs, next_item), 0)
                    seqs = list(seqs.cpu().detach().numpy())
                    for i in range(number_ood_users):
                        new_user_idx = self.user_count + i + 1
                        self.train[new_user_idx] = seqs

                    np.save('./sequence pattern/pop watermark seq %s %d %s %d.npy' % (
                        args.dataset_code, args.pattern_len, args.model_code, args.bottom_m), seqs)
                    print('Watermark Sequence:', seqs)
                    for i in range(number_ood_val_users):
                        val_new_user_idx = new_user_idx + i + 1
                        length = np.random.randint(2, args.pattern_len+1)
                        whole_sequence = seqs[:length]
                        self.val[val_new_user_idx] = list(whole_sequence)
                    self.valid_users = sorted(
                        self.valid_users + sorted(list(range(new_user_idx + 1, val_new_user_idx + 1))))

                # ========== NIDW: Near In-Distribution Watermarking (SASRec) ==========
                elif hasattr(args, 'wm_type') and args.wm_type == 'nidw':
                    print(f'[NIDW][SASRec] Generating near-ID watermark sequence '
                          f'(tau={args.nidw_tau}, alpha={args.nidw_alpha}, '
                          f'tau_sim={args.nidw_tau_sim}, window={args.nidw_window})')

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

                    # Step 2: seed prefix from a real train user, or cold anchor fallback
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
                    seqs = torch.tensor(wm_prefix, dtype=torch.float32, device=args.device)
                    seen_items = set(wm_prefix)
                    r = len(wm_prefix)

                    for t in range(r, args.pattern_len):
                        input_seqs = torch.zeros((1, self.max_len), device=args.device)
                        input_seqs[:, (self.max_len - len(seqs)):] = seqs

                        with torch.no_grad():
                            logits = pretrained_model(input_seqs.long())[:, -1, 1:]

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
                        self.val[val_new_user_idx] = list(wm_seq[:length])
                    self.valid_users = sorted(
                        self.valid_users + sorted(
                            list(range(self.user_count + number_ood_users + 1,
                                       val_new_user_idx + 1))))

        self.seen_samples = {}
        for user in self.train.keys():
            seen = set(self.train[user])
            try:
                seen.update(self.val[user])
                seen.update(self.test[user])
            except:
                pass
            self.seen_samples[user] = seen

    @classmethod
    def code(cls):
        return 'sas'

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
        dataset = SASTrainDataset(
            self.train, self.max_len, self.sliding_size, self.seen_samples, self.item_count, self.rng)
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
            dataset = SASValidDataset(self.train, self.val, self.max_len, self.args.gold, self.user_count, valid_users=self.valid_users)
        elif mode == 'test':
            dataset = SASTestDataset(self.train, self.val, self.test, self.max_len, test_users=self.test_users)
        return dataset


class SASTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, sliding_size, seen_samples, num_items, rng):
        # self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = num_items
        self.rng = rng
        
        assert self.sliding_step > 0
        self.all_seqs = []
        self.seen_samples = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            neg = seen_samples[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
                self.seen_samples.append(neg)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]
                self.seen_samples = self.seen_samples + [neg for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]
        neg = []

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        mask_len = self.max_len - len(labels)
        while len(neg) < len(labels):
            item = self.rng.randint(1, self.num_items)
            if item in self.seen_samples[index] or item in neg:
                continue
            neg.append(item)
        
        labels = [0] * mask_len + labels
        neg = [0] * mask_len + neg

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(neg)


class SASValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, gold, user_count, valid_users=None):
        self.u2seq = u2seq  # train
        self.user_count = user_count
        self.gold = gold
        if not valid_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = valid_users
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

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

                # no mask token here
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

                # no mask token here
                seq = seq[-self.max_len:]
                padding_len = self.max_len - len(seq)
                seq = [0] * padding_len + seq

                return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        candidates = answer
        labels = [1] * len(answer)

        # no mask token here
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class SASTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len, test_users=None):
        self.u2seq = u2seq  # train
        self.u2val = u2val  # val
        if not test_users:
            self.users = sorted(self.u2seq.keys())
        else:
            self.users = test_users
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer  # test
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]  # append validation item after train seq
        answer = self.u2answer[user]

        candidates = answer
        labels = [1] * len(answer)

        # no mask token here
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)