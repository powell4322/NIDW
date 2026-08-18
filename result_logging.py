import os
import json
import csv
from pathlib import Path


def save_standardized_results(export_root, args, attack_name, wm_metrics, util_metrics, file_stem='evaluation_results'):
    """
    Save watermark and utility metrics to JSON and CSV formats.
    
    Args:
        export_root: Root directory for saving results
        args: Command line arguments object
        attack_name: Name of the attack used
        wm_metrics: Dictionary of watermark metrics
        util_metrics: Dictionary of utility metrics
        file_stem: Stem for the output filenames
    
    Returns:
        Tuple of (json_path, csv_path)
    """
    
    # Ensure export_root/logs exists
    log_dir = os.path.join(export_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Prepare JSON output
    json_path = os.path.join(log_dir, f'{file_stem}.json')
    
    # Prepare results dictionary
    results = {
        'config': {
            'dataset_code': args.dataset_code,
            'model_code': args.model_code,
            'method': args.method,
            'pattern_len': args.pattern_len,
            'bottom_m': args.bottom_m,
            'number_ood_seqs': args.number_ood_seqs,
            'number_ood_val_seqs': args.number_ood_val_seqs,
            'attack': attack_name,
        },
        'watermark_robustness': wm_metrics if wm_metrics else {},
        'model_utility': util_metrics if util_metrics else {},
    }
    
    # Add attack parameters if applicable
    if attack_name != 'none':
        results['config']['target'] = getattr(args, 'target', None)
        results['config']['item_freq_source'] = getattr(args, 'item_freq_source', 'data')
        # distributional
        results['config']['dis_threshold'] = getattr(args, 'dis_threshold', 0.7)
        results['config']['dis_beta'] = getattr(args, 'dis_beta', 5.0)
        results['config']['dis_eps'] = getattr(args, 'dis_eps', 0.02)
        # point
        results['config']['point_beta'] = getattr(args, 'point_beta', 5.0)
        # noise
        results['config']['noise_scale'] = getattr(args, 'noise_scale', 1.0)
        # region
        results['config']['region_low'] = getattr(args, 'region_low', 0.2)
        results['config']['region_high'] = getattr(args, 'region_high', 0.5)
        results['config']['region_beta'] = getattr(args, 'region_beta', 5.0)
        # trajectory
        results['config']['traj_k1'] = getattr(args, 'traj_k1', 3)
        results['config']['traj_k2'] = getattr(args, 'traj_k2', 1)
        results['config']['traj_beta'] = getattr(args, 'traj_beta', 5.0)
        results['config']['traj_depth_decay'] = getattr(args, 'traj_depth_decay', 1.0)
        results['config']['traj_trigger_topk'] = getattr(args, 'traj_trigger_topk', 1)
        # unified
        results['config']['unified_beta1'] = getattr(args, 'unified_beta1', 0.0)
        results['config']['unified_beta2'] = getattr(args, 'unified_beta2', 0.0)
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Prepare CSV output
    csv_path = os.path.join(log_dir, f'{file_stem}_summary.csv')
    
    # Prepare row for CSV
    csv_row = {
        'dataset': args.dataset_code,
        'model': args.model_code,
        'method': args.method,
        'pattern_len': args.pattern_len,
        'attack': attack_name,
        'target': getattr(args, 'target', '') if attack_name != 'none' else '',
    }
    
    # Add key metrics
    if wm_metrics:
        csv_row['wm_recall@1'] = wm_metrics.get('Recall@1', 0)
        csv_row['wm_recall@5'] = wm_metrics.get('Recall@5', 0)
        csv_row['wm_recall@10'] = wm_metrics.get('Recall@10', 0)
        csv_row['wm_ndcg@10'] = wm_metrics.get('NDCG@10', 0)
    
    if util_metrics:
        csv_row['util_ndcg@10'] = util_metrics.get('NDCG@10', 0)
        csv_row['util_recall@10'] = util_metrics.get('Recall@10', 0)
    
    # Write or append to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fieldnames = list(csv_row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_row)
    
    return json_path, csv_path
