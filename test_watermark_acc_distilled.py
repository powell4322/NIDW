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


'''test watermark validity on distilled model'''


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
        folder_name = args.bb_model_code + '2' + args.model_code + '_autoregressive' + str(args.num_generated_seqs)
        if args.gold:
            export_root = 'experiments/distillation_rank/' + folder_name + '/' + args.dataset_code
        else:
            wm_type = getattr(args, 'wm_type', 'aow')
            if wm_type == 'aow':
                export_root = 'experiments/distillation_rank/watermark_test/method_' + str(args.method) + '/' + folder_name + '/' + args.dataset_code + '/' + str(
                    args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
            else:
                export_root = 'experiments/distillation_rank/watermark_test/method_' + str(args.method) + '/' + wm_type + '/' + folder_name + '/' + args.dataset_code + '/' + str(
                    args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + '_' + str(args.pattern_len) + '_' + str(args.bottom_m)

    model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))

    item_freq = load_or_build_item_freq(
        dataloader,
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
    attack = None
    if args.attack != 'none':
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
        file_stem='evaluation_results_distilled',
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
