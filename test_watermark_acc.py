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


def train(args, export_root=None, resume=False):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)

    dataset = dataset_factory(args)

    # Watermark test loader (synthetic sequences built from the watermark)
    wm_dataloader = TESTDataloader(args, dataset)
    wm_train_loader, wm_val_loader, wm_test_loader = wm_dataloader.get_pytorch_dataloaders()

    # Clean loader for utility: disable watermark injection
    original_ood_seqs = args.number_ood_seqs
    args.number_ood_seqs = 0.0

    if args.model_code == 'bert':
        clean_dataloader = BERTDataloader(args, dataset)
    elif args.model_code == 'sas':
        clean_dataloader = SASDataloader(args, dataset)
    else:
        clean_dataloader = BERTDataloader(args, dataset)  # fallback

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
            wm_type = getattr(args, 'wm_type', 'aow')
            if wm_type == 'aow':
                export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                              args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                              '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
            else:
                export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + wm_type + '/' + args.model_code + '/' + \
                              args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                              '_' + str(args.pattern_len) + '_' + str(args.bottom_m)

    model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location=args.device, weights_only=False).get(STATE_DICT_KEY))

    # Initialize Attack
    attack = None
    if args.attack != 'none':
        print(f"[attack] type={args.attack}, freq_source={args.item_freq_source}")
        if clean_dataloader.item_count != wm_dataloader.item_count:
            print(f"[warning] item count mismatch: clean={clean_dataloader.item_count}, "
                  f"watermark={wm_dataloader.item_count}")

        item_freq = load_or_build_item_freq(
            clean_dataloader,
            export_root,
            args.device,
            source=args.item_freq_source,
            args=args,
            model=model,
            num_queries=args.freq_query_num,
            topk=args.freq_query_topk,
            temperature=args.freq_query_temperature,
            uniform_mix=args.freq_query_uniform_mix,
            tpe_alpha=args.freq_tpe_alpha,
        )
        attack = build_attack(
            args.attack, item_freq,
            model=model, args=args, method=args.method, target=args.target,
            threshold=args.dis_threshold, beta=args.dis_beta, eps=args.dis_eps,
            point_beta=args.point_beta,
            noise_scale=args.noise_scale, seed=args.noise_seed,
            low=args.region_low, high=args.region_high, region_beta=args.region_beta,
            k1=args.traj_k1, k2=args.traj_k2,
            traj_beta=args.traj_beta, depth_decay=args.traj_depth_decay,
            trigger_topk=args.traj_trigger_topk,
            beta1=args.unified_beta1, beta2=args.unified_beta2,
        )

    # Setup Trainer
    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)
    elif args.model_code == 'sas':
        trainer = SASTrainer(args, model, wm_train_loader, wm_val_loader, wm_test_loader, export_root)

    if attack is not None:
        trainer.attack = attack

    print(f"[eval] attack={args.attack} target={args.target}")

    # Evaluate watermark robustness on the synthetic watermark test set
    wm_metrics = trainer.test(test_watermark=True)
    print(f"[wm] Recall@1={wm_metrics.get('Recall@1', 0):.4f} "
          f"Recall@5={wm_metrics.get('Recall@5', 0):.4f} "
          f"Recall@10={wm_metrics.get('Recall@10', 0):.4f}")

    # Evaluate utility on the clean test set
    trainer.test_loader = clean_test_loader
    util_metrics = trainer.test(test_watermark=False)
    print(f"[util] NDCG@10={util_metrics.get('NDCG@10', 0):.4f} "
          f"Recall@10={util_metrics.get('Recall@10', 0):.4f}")

    # Save standardized outputs (JSON + CSV)
    json_path, csv_path = save_standardized_results(
        export_root=export_root,
        args=args,
        attack_name=args.attack,
        wm_metrics=wm_metrics,
        util_metrics=util_metrics,
        file_stem='evaluation_results',
    )
    print(f"[done] results saved to {json_path}")


if __name__ == "__main__":
    set_template(args)

    batch = 128
    args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    train(args, resume=False)
