import os
from datasets import dataset_factory
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader.test import *
from dataloader.bert import BERTDataloader
from dataloader.sas import SASDataloader
from trainer import *
from utils import *
from attacks import build_attack
from frequency_estimators import load_or_build_item_freq
from result_logging import save_standardized_results
import json


'''test the validity of the watermark on watermarked model/oracle model'''


def train(args, export_root=None, resume=False):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)

    # 1. Initialize Dataset & Dataloaders
    dataset = dataset_factory(args)
    
    # Watermark Test Loader (Synthetic 1000 items)
    wm_dataloader = TESTDataloader(args, dataset)
    wm_train_loader, wm_val_loader, wm_test_loader = wm_dataloader.get_pytorch_dataloaders()
    
    # Original Test Loader (Real Test Set for Utility)
    # [FIX] Temporarily disable watermark injection for clean loader to avoid crashing (missing pretrained_model)
    # and to ensure we measure utility on the original clean distribution.
    original_ood_seqs = args.number_ood_seqs
    args.number_ood_seqs = 0.0

    if args.model_code == 'bert':
        clean_dataloader = BERTDataloader(args, dataset)
    elif args.model_code == 'sas':
        clean_dataloader = SASDataloader(args, dataset)
    else:
        clean_dataloader = BERTDataloader(args, dataset) # Fallback

    clean_train_loader, clean_val_loader, clean_test_loader = clean_dataloader.get_pytorch_dataloaders()
    
    # Restore args
    args.number_ood_seqs = original_ood_seqs

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    if export_root == None:
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

    model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location=args.device, weights_only=False).get(STATE_DICT_KEY))

    # Initialize Attack
    attack = None
    if args.attack != 'none':
        phi = json.loads(args.prf_phi_json) if args.prf_phi_json else None
        print(f"Building attack: {args.attack} (Item Frequency Source: {args.item_freq_source})")
        # Build frequency from selected source.
        # data: true interaction histogram; dpe: distilled-data estimate; qee/model_query: query-based estimate.
        # Verify dimension consistency
        if clean_dataloader.item_count != wm_dataloader.item_count:
             print(f"Warning: Item count mismatch! Clean: {clean_dataloader.item_count}, Watermark: {wm_dataloader.item_count}")

        item_freq = load_or_build_item_freq(
            clean_dataloader,
            export_root,
            args.device,
            source=args.item_freq_source,
            args=args,
            model=model,
            query_loader=clean_test_loader,
            query_topk=args.freq_query_topk,
            query_max_batches=args.freq_query_max_batches,
            query_temperature=args.freq_query_temperature,
            query_uniform_mix=args.freq_query_uniform_mix,
            tpe_alpha=args.freq_tpe_alpha,
        )
        attack = build_attack(
            args.attack,
            item_freq,
            gamma=args.prf_gamma,
            beta=args.prf_beta,
            eps=args.prf_eps,
            phi=phi,
            method=args.method,
            alpha=args.ptsc_alpha,
            sigma=args.pcrmr_sigma,
            direction=args.attack_direction  # Passed from args
        )

    # Setup Trainer
    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)
    elif args.model_code == 'sas':
        trainer = SASTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)

    if attack is not None:
        trainer.attack = attack

    print("==============================================")
    print(f"Running Evaluation with Attack: {args.attack} | Direction: {args.attack_direction}")
    print("==============================================")

    # 2. Evaluate on Watermark Test Set (Robustness)
    print("\n[Metric 1/2] Evaluating Watermark Robustness...")
    wm_metrics = trainer.test(test_watermark=True)
    print(f"Watermark Detection Success (HR@1/5/10): {wm_metrics.get('Recall@1', 0):.4f} / {wm_metrics.get('Recall@5', 0):.4f} / {wm_metrics.get('Recall@10', 0):.4f}")

    # 3. Evaluate on Clean Test Set (Utility)
    print("\n[Metric 2/2] Evaluating Model Utility (Clean Test Set)...")
    # Swap to clean test loader
    trainer.test_loader = clean_test_loader
    util_metrics = trainer.test(test_watermark=False)
    print(f"Model Utility (NDCG@10): {util_metrics.get('NDCG@10', 0):.4f}")

    # 4. Save standardized outputs (JSON + CSV)
    json_path, csv_path = save_standardized_results(
        export_root=export_root,
        args=args,
        attack_name=args.attack,
        wm_metrics=wm_metrics,
        util_metrics=util_metrics,
        file_stem='evaluation_results',
    )
    print(f"\nAll results saved to: {json_path}")
    print(f"Summary row appended to: {csv_path}")


if __name__ == "__main__":
    set_template(args)

    batch = 128
    args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k


    train(args, resume=False)
