import os
from datasets import DATASETS, dataset_factory
from config import STATE_DICT_KEY
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *
from attacks import build_attack
from frequency_estimators import build_train_item_freq


def train(args, export_root=None, resume=False):
    if args.finetune_ratio == 0:
        raise ValueError('Please specify a valid finetune ratio!')
    args.lr = 0.001

    # fix_random_seed_as(args.model_init_seed)
    dataset = dataset_factory(args)

    oracle_model = None
    if args.dataset_code == 'ml-1m':
        args.num_items = 3416
    elif args.dataset_code == 'ml-20m':
        args.num_items = 18345
    elif args.dataset_code == 'steam':
        args.num_items = 13046
    elif args.dataset_code == 'beauty':
        args.num_items = 54542

    if args.model_code == 'bert':
        oracle_model = BERT(args)
    elif args.model_code == 'sas':
        oracle_model = SASRec(args)
    else:
        raise NotImplementedError('Model not implemented for fine-tuning!')
    root = 'experiments/' + args.model_code + '/' + args.dataset_code
    oracle_model.load_state_dict(
        torch.load(os.path.join(root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(
            STATE_DICT_KEY))
    oracle_model = oracle_model.to(args.device)

    _, _, test_loader = dataloader_factory(args, args.model_code, oracle_model=oracle_model)

    dataloader = BERTFinetuneDataloader(args, dataset, pretrained_model=oracle_model)
    train_loader, val_loader, _ = dataloader.get_pytorch_dataloaders()

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)

    wm_type = getattr(args, 'wm_type', 'aow')
    if wm_type == 'aow':
        before_finetune_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                      args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                      '_' + str(args.pattern_len) + '_' + str(args.bottom_m)

        export_root = 'experiments/watermark_test_after_finetune/method_' + str(args.method) + '/' + args.model_code + '/' + \
                      args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                      '_' + str(args.pattern_len) + '_' + str(args.bottom_m) + '_' + str(args.finetune_ratio)
    else:
        before_finetune_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + wm_type + '/' + args.model_code + '/' + \
                      args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                      '_' + str(args.pattern_len) + '_' + str(args.bottom_m)

        export_root = 'experiments/watermark_test_after_finetune/method_' + str(args.method) + '/' + wm_type + '/' + args.model_code + '/' + \
                      args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                      '_' + str(args.pattern_len) + '_' + str(args.bottom_m) + '_' + str(args.finetune_ratio)

    model.load_state_dict(torch.load(os.path.join(before_finetune_root, 'models', 'best_acc_model.pth'), map_location='cpu', weights_only=False).get(STATE_DICT_KEY))

    if args.model_code == 'bert':
        if args.dataset_code == 'ml-1m':
            args.num_epochs = 1000
        elif args.dataset_code == 'beauty':
            args.num_epochs = 50
        elif args.dataset_code == 'ml-20m':
            args.num_epochs = 10
        elif args.dataset_code == 'steam':
            args.num_epochs = 15
        else:
            raise ValueError('Number of epochs undefined!')
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    # Optional: apply inference-time attack during testing
    if hasattr(args, 'attack') and args.attack != 'none':
        item_freq = build_train_item_freq(args, export_root)
        trainer.attack = build_attack(
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

    trainer.train()
    trainer.test(test_watermark=False)


if __name__ == "__main__":
    set_template(args)

    batch = 128
    if args.gold or args.dataset_code == 'ml-1m':
        args.num_epochs = 1000
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k


    train(args, resume=False)
