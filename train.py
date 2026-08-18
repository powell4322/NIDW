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
     
        wm_type = getattr(args, 'wm_type', 'aow')
        if wm_type == 'aow':
           
            export_root = 'experiments/watermark_test/method_' + str(args.method) + '/' + args.model_code + '/' + \
                          args.dataset_code + '/' + str(args.number_ood_seqs) + '_' + str(args.number_ood_val_seqs) + \
                          '_' + str(args.pattern_len) + '_' + str(args.bottom_m)
        else:
           
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
        wm_type = getattr(args, 'wm_type', 'aow')
        oracle_model_ref = oracle_model if wm_type == 'nidw' else None

        nidw_stages = getattr(args, 'nidw_stages', 1)
        stage_start = getattr(args, 'nidw_resume_stage', 1)

        for stage in range(stage_start, nidw_stages + 1):
            if nidw_stages > 1:
                print(f'[NIDW] Progressive stage {stage}/{nidw_stages}')
                if stage == 1:
                    args.nidw_tau = args.nidw_tau_s1 if args.nidw_tau_s1 > 0 else args.nidw_tau
                    args.nidw_alpha = args.nidw_alpha_s1 if args.nidw_alpha_s1 > 0 else args.nidw_alpha
                    args.nidw_tau_sim = args.nidw_tau_sim_s1 if args.nidw_tau_sim_s1 > 0 else args.nidw_tau_sim
                elif stage == 2:
                    args.nidw_tau = args.nidw_tau_s2 if args.nidw_tau_s2 > 0 else 1.0
                    args.nidw_alpha = args.nidw_alpha_s2 if args.nidw_alpha_s2 > 0 else 0.5
                    args.nidw_tau_sim = args.nidw_tau_sim_s2 if args.nidw_tau_sim_s2 > 0 else 0.5
                # Regenerate dataloader so each stage gets stage-specific watermark
                train_loader, val_loader, test_loader = dataloader_factory(
                    args, args.model_code, oracle_model=oracle_model)

            if stage > 1:
                prev_stage_root = export_root + f'_stage{stage-1}'
                try:
                    model = BERT(args)
                    model.load_state_dict(torch.load(
                        os.path.join(prev_stage_root, 'models', 'best_acc_model.pth'),
                        map_location='cpu', weights_only=False).get(STATE_DICT_KEY))
                    print(f'[NIDW] Loaded checkpoint from stage {stage-1}')
                except Exception:
                    print(f'[NIDW] No checkpoint for stage {stage-1}, training from scratch')

            stage_export = export_root if nidw_stages <= 1 else export_root + f'_stage{stage}'
            trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, stage_export,
                                  oracle_model=oracle_model_ref)

            if hasattr(args, 'attack') and args.attack != 'none':
                item_freq = build_train_item_freq(args, stage_export)
                trainer.attack = build_attack(
                    args.attack, item_freq, model=model, args=args, method=args.method, target=args.target,
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
            if nidw_stages > 1:
                print(f'[NIDW] Stage {stage} complete')
    else:
        if args.model_code == 'sas':
            trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)
            if hasattr(args, 'attack') and args.attack != 'none':
                item_freq = build_train_item_freq(args, export_root)
                trainer.attack = build_attack(
                    args.attack, item_freq, model=model, args=args, method=args.method, target=args.target,
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
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    train(args, resume=False)
