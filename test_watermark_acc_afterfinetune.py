import os
from datasets import dataset_factory
from config import STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader.test import *
from trainer import *
from utils import *
from attacks import build_attack
from frequency_estimators import load_or_build_item_freq
from result_logging import save_standardized_results
import json


'''test watermark validity on finetuned model'''


def train(args, export_root=None, resume=False):
    args.lr = 0.001
    # fix_random_seed_as(args.model_init_seed)

    dataset = dataset_factory(args)
    dataloader = TESTDataloader(args, dataset)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    if export_root == None:
        if args.gold:
            export_root = 'experiments/' + args.model_code + '_finetune/' + args.dataset_code
        else:
            export_root = 'experiments/watermark_test_after_finetune/method_' + str(args.method) + '/' + args.model_code + '/' + \
                          args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                          '_' + str(args.pattern_len) + '_' + str(args.bottom_m) + '_' + str(args.finetune_ratio)

    model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))

    item_freq = load_or_build_item_freq(
        dataloader,
        export_root,
        args.device,
        source=args.item_freq_source,
        args=args,
        model=model,
        query_loader=test_loader,
        query_topk=args.freq_query_topk,
        query_max_batches=args.freq_query_max_batches,
        query_temperature=args.freq_query_temperature,
        query_uniform_mix=args.freq_query_uniform_mix,
        tpe_alpha=args.freq_tpe_alpha,
    )
    attack = None
    if args.attack != 'none':
        phi = json.loads(args.prf_phi_json) if args.prf_phi_json else None
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
            direction=args.attack_direction,
        )

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    if attack is not None:
        trainer.attack = attack

    wm_metrics = trainer.test(test_watermark=True)
    json_path, csv_path = save_standardized_results(
        export_root=export_root,
        args=args,
        attack_name=args.attack,
        wm_metrics=wm_metrics,
        util_metrics=None,
        file_stem='evaluation_results_afterfinetune',
    )
    print(f"Results saved to: {json_path}")
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
