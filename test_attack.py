import os
from datasets import dataset_factory
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from dataloader.test import *
from trainer import *
from utils import *
from attacks import build_attack
import json


'''Test utility and watermark validity under inference-time attacks'''


def load_or_build_item_freq(dataloader, export_root, device):
    """Load or build normalized item frequency from dataloader"""
    freq_path = os.path.join(export_root, 'item_freq.pt')
    if os.path.isfile(freq_path):
        return torch.load(freq_path, map_location=device).to(device)

    freq = torch.zeros(dataloader.item_count, device=device)
    for split in [dataloader.train, dataloader.val, dataloader.test]:
        for items in split.values():
            if len(items) == 0:
                continue
            idx = torch.tensor(items, device=device, dtype=torch.long) - 1
            freq.index_add_(0, idx, torch.ones_like(idx, dtype=freq.dtype))

    total = freq.sum()
    if total > 0:
        freq = freq / total

    torch.save(freq.cpu(), freq_path)
    return freq


def test_under_attack(args, export_root=None):
    args.lr = 0.001

    # Determine export root
    if export_root == None:
        if args.gold:
            export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
        else:
            export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                          args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                          '_' + str(args.pattern_len) + '_' + str(args.bottom_m)

    # Load model
    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), 
                                     map_location='cpu', weights_only=False).get(STATE_DICT_KEY))

    print('\n' + '='*80)
    print(f'Testing model: {export_root}')
    print(f'Attack: {args.attack}')
    if args.attack != 'none':
        if args.attack == 'soft_prf':
            print(f'  gamma={args.prf_gamma}, beta={args.prf_beta}')
        elif args.attack == 'ptsc':
            print(f'  alpha={args.ptsc_alpha}')
        elif args.attack == 'pcrmr':
            print(f'  sigma={args.pcrmr_sigma}')
    print('='*80 + '\n')

    # ===== Test 1: Utility on Normal Test Set =====
    print('\n[1/2] Testing Utility on Normal Test Set...')
    dataset = dataset_factory(args)
    normal_dataloader = dataloader_factory(args, args.model_code, oracle_model=None)
    train_loader, val_loader, test_loader = normal_dataloader

    # Build attack
    item_freq = load_or_build_item_freq(normal_dataloader[0].dataset if hasattr(normal_dataloader[0], 'dataset') 
                                        else type('obj', (), {'item_count': args.num_items, 
                                                              'train': {}, 'val': {}, 'test': {}}), 
                                        export_root, args.device)
    
    # Get item_count from dataloader
    if hasattr(train_loader, 'dataset'):
        item_count = train_loader.dataset.item_count
    else:
        # Fallback: infer from args
        item_count = args.num_items
    
    # Rebuild freq with correct item_count
    freq_path = os.path.join(export_root, 'item_freq.pt')
    if os.path.isfile(freq_path):
        item_freq = torch.load(freq_path, map_location=args.device).to(args.device)
    else:
        # Build from train/val/test loaders
        item_freq = torch.zeros(item_count, device=args.device)
        # We need access to underlying data, let's use dataset directly
        dataset_obj = dataset_factory(args)
        loaded_data = dataset_obj.load_dataset()
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

    attack = None
    if args.attack != 'none':
        attack = build_attack(
            args.attack,
            item_freq,
            gamma=args.prf_gamma,
            beta=args.prf_beta,
            alpha=args.ptsc_alpha,
            sigma=args.pcrmr_sigma
        )

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    elif args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    if attack is not None:
        trainer.attack = attack

    utility_metrics = trainer.test(test_watermark=False)

    # ===== Test 2: Watermark Validity on Watermark Test Set =====
    print('\n[2/2] Testing Watermark Validity on Watermark Test Set...')
    watermark_dataset = dataset_factory(args)
    watermark_dataloader = TESTDataloader(args, watermark_dataset)
    wm_train_loader, wm_val_loader, wm_test_loader = watermark_dataloader.get_pytorch_dataloaders()

    if args.model_code == 'bert':
        wm_trainer = BERTTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)
    elif args.model_code == 'sas':
        wm_trainer = SASTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)

    if attack is not None:
        wm_trainer.attack = attack

    watermark_metrics = wm_trainer.test(test_watermark=True)

    # ===== Combined Report =====
    print('\n' + '='*80)
    print('ATTACK EVALUATION SUMMARY')
    print('='*80)
    print(f'\nModel: {export_root}')
    print(f'Attack: {args.attack}')
    if args.attack != 'none':
        if args.attack == 'soft_prf':
            print(f'Parameters: gamma={args.prf_gamma}, beta={args.prf_beta}')
        elif args.attack == 'ptsc':
            print(f'Parameters: alpha={args.ptsc_alpha}')
        elif args.attack == 'pcrmr':
            print(f'Parameters: sigma={args.pcrmr_sigma}')
    
    print('\n--- Utility (Normal Test Set) ---')
    for k, v in sorted(utility_metrics.items()):
        print(f'{k:20s}: {v:.4f}')
    
    print('\n--- Watermark Validity (Watermark Test Set) ---')
    for k, v in sorted(watermark_metrics.items()):
        print(f'{k:20s}: {v:.4f}')
    print('='*80 + '\n')

    # Save combined results
    combined_metrics = {
        'attack': args.attack,
        'attack_params': {},
        'utility': utility_metrics,
        'watermark': watermark_metrics
    }
    
    if args.attack == 'soft_prf':
        combined_metrics['attack_params'] = {'gamma': args.prf_gamma, 'beta': args.prf_beta}
    elif args.attack == 'ptsc':
        combined_metrics['attack_params'] = {'alpha': args.ptsc_alpha}
    elif args.attack == 'pcrmr':
        combined_metrics['attack_params'] = {'sigma': args.pcrmr_sigma}

    attack_suffix = f'_{args.attack}' if args.attack != 'none' else ''
    result_path = os.path.join(export_root, 'logs', f'attack_evaluation{attack_suffix}.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(combined_metrics, f, indent=4)
    
    print(f'Results saved to: {result_path}\n')

    return combined_metrics


if __name__ == "__main__":
    set_template(args)

    batch = 128
    args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    test_under_attack(args)
