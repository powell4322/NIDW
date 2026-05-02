import os
import os
from datasets import DATASETS
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *
from attacks import build_attack
from attacks import build_attack


def train(args, export_root=None, resume=False):
    args.lr = 0.001

    oracle_model = None
    if not args.gold:
        if args.dataset_code == 'ml-1m':
            args.num_items = 3416
        elif args.dataset_code == 'ml-20m':
            args.num_items = 18345
        elif args.dataset_code == 'steam':
            args.num_items = 13046
        elif args.dataset_code == 'beauty':
            args.num_items = 54542
        else:
            raise NotImplementedError('Please specify number of items!')
        if args.model_code == 'bert':
            oracle_model = BERT(args)
        elif args.model_code == 'sas':
            oracle_model = SASRec(args)
        else:
            raise NotImplementedError('Model not recognized!')

        root = 'experiments/' + args.model_code + '/' + args.dataset_code
        try:
            oracle_model.load_state_dict(
                torch.load(os.path.join(root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))
        except:
            raise ValueError('Please train the oracle with --gold first!')
        oracle_model = oracle_model.to(args.device)

    if args.gold:
        fix_random_seed_as(args.model_init_seed)
    # fix_random_seed_as(args.model_init_seed)

    train_loader, val_loader, test_loader = dataloader_factory(args, args.model_code, oracle_model=oracle_model)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    if args.gold:
        export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
    else:
        # 获取水印类型，默认aow
        wm_type = getattr(args, 'wm_type', 'aow')
        if wm_type == 'aow':
            # AOW: 保持原有路径格式，向后兼容
            export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                          args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                          '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
        else:
            # CPS等新方法: 使用带wm_type的新路径格式
            export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + wm_type + '/' + args.model_code + '/' + \
                          args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                          '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
    
    if resume:
        try: 
            model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')

    # For ML-1M, train from scratch. For other datasets, fine-tune them with the pretrained oracle model to speed up the training process.
    if not args.gold and args.dataset_code != 'ml-1m':
        gold_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code
        model.load_state_dict(torch.load(os.path.join(gold_model_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    # Optional: apply inference-time attack during testing
    if hasattr(args, 'attack') and args.attack != 'none':
        # Build item frequency
        freq_path = os.path.join(export_root, 'item_freq.pt')
        if os.path.isfile(freq_path):
            item_freq = torch.load(freq_path, map_location=args.device).to(args.device)
        else:
            # Build from dataset
            dataset_obj = dataset_factory(args)
            loaded_data = dataset_obj.load_dataset()
            item_freq = torch.zeros(args.num_items, device=args.device)
            for split in [loaded_data['train'], loaded_data['val'], loaded_data['test']]:
                for items in split.values():
                    if len(items) == 0:
                        continue
                    idx = torch.tensor(items, device=args.device, dtype=torch.long) - 1
                    item_freq.index_add_(0, idx, torch.ones_like(idx, dtype=item_freq.dtype))
            total = item_freq.sum()
            if total > 0:
                item_freq = item_freq / total
            torch.save(item_freq.cpu(), freq_path)
        
        attack = build_attack(
            args.attack,
            item_freq,
            gamma=args.prf_gamma,
            beta=args.prf_beta,
            alpha=args.ptsc_alpha,
            sigma=args.pcrmr_sigma
        )
        trainer.attack = attack

    trainer.train()
    trainer.test(test_watermark=False)


if __name__ == "__main__":
    set_template(args)

    batch = 128
    # 注释掉硬编码的1000轮，使用命令行参数或set_template中的默认值
    # if args.gold or args.dataset_code == 'ml-1m':
    #     args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    train(args, resume=False)
