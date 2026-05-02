#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验二轻量级验证脚本
快速检查命令语法和参数有效性，不运行完整评估
"""

import os
import sys
import argparse
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def check_environment():
    """检查环境准备"""
    print(f"\n{Colors.BOLD}=== 环境检查 ==={Colors.RESET}")
    
    checks = {
        "test_watermark_acc.py": os.path.exists("test_watermark_acc.py"),
        "attacks.py": os.path.exists("attacks.py"),
        "utils.py": os.path.exists("utils.py"),
        "序列pattern目录": os.path.exists("sequence pattern"),
        "ml-1m数据": os.path.exists("data/ml-1m"),
    }
    
    all_ok = True
    for check_name, result in checks.items():
        if result:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {check_name}")
        else:
            print(f"  {Colors.RED}✗{Colors.RESET} {check_name}")
            all_ok = False
    
    return all_ok

def validate_command_syntax(cmd):
    """检查命令语法有效性"""
    # 解析命令的参数
    parts = cmd.split()
    
    # 检查必要的参数
    required_params = [
        '--device', '--dataset_code', '--model_code',
        '--number_ood_seqs', '--pattern_len', '--bottom_m',
        '--method', '--attack'
    ]
    
    cmd_str = ' '.join(parts)
    missing = []
    for param in required_params:
        if param not in cmd_str:
            missing.append(param)
    
    return len(missing) == 0, missing

def check_attack_params(cmd):
    """检查攻击特定参数的完整性"""
    issues = []
    
    if '--attack soft_prf' in cmd or '--attack=soft_prf' in cmd:
        if '--prf_beta' not in cmd:
            issues.append("soft_prf缺少--prf_beta")
        if '--item_freq_source' not in cmd:
            issues.append("soft_prf缺少--item_freq_source")
    
    if '--attack point_level' in cmd:
        if '--pl_top_k' not in cmd:
            issues.append("point_level缺少--pl_top_k")
        if '--pl_boost' not in cmd:
            issues.append("point_level缺少--pl_boost")
    
    if '--attack random_shuffle' in cmd:
        if '--rs_noise_scale' not in cmd:
            issues.append("random_shuffle缺少--rs_noise_scale")
    
    if '--item_freq_source qee' in cmd:
        if '--freq_query_topk' not in cmd:
            issues.append("QEE缺少--freq_query_topk")
    
    return len(issues) == 0, issues

def validate_command(cmd, description):
    """验证单个命令"""
    print(f"\n{Colors.BOLD}[检查] {description}{Colors.RESET}")
    
    # 基础语法检查
    syntax_ok, missing = validate_command_syntax(cmd)
    if not syntax_ok:
        print(f"  {Colors.RED}✗ 语法错误: 缺少参数 {missing}{Colors.RESET}")
        return False
    
    # 攻击参数检查
    attack_ok, issues = check_attack_params(cmd)
    if not attack_ok:
        for issue in issues:
            print(f"  {Colors.YELLOW}⚠ {issue}{Colors.RESET}")
    
    print(f"  {Colors.GREEN}✓ 参数检查通过{Colors.RESET}")
    return True

def main():
    """主验证流程"""
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"实验二轻量级验证脚本 - 命令语法和参数检查")
    print(f"{'='*70}{Colors.RESET}")
    
    # 环境检查
    env_ok = check_environment()
    if not env_ok:
        print(f"\n{Colors.RED}环境检查失败，部分依赖缺失{Colors.RESET}")
        return 1
    
    # 实验命令清单
    base_cmd = "python test_watermark_acc.py"
    base_args = "--device cpu --dataset_code ml-1m --model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100"
    
    experiments = [
        # 基线
        (f"{base_cmd} {base_args} --method cold --attack none",
         "B0: Baseline (无攻击) - COLD"),
        
        (f"{base_cmd} {base_args} --method cold --attack random_shuffle --rs_noise_scale 1.0",
         "B1: Random Baseline - COLD"),
        
        # Soft-PRF 攻击
        (f"{base_cmd} {base_args} --method cold --attack soft_prf --attack_direction suppress_unpopular --item_freq_source data --prf_gamma 0.7 --prf_beta 5.0 --prf_eps 0.02",
         "A1: Soft-PRF Data-Aware - COLD"),
        
        (f"{base_cmd} {base_args} --method cold --attack soft_prf --attack_direction suppress_unpopular --item_freq_source qee --prf_gamma 0.7 --prf_beta 5.0 --prf_eps 0.02 --freq_query_topk 20",
         "A2: Soft-PRF Data-Unaware (QEE) - COLD"),
        
        # Point-Level 攻击
        (f"{base_cmd} {base_args} --method cold --attack point_level --attack_direction suppress_unpopular --item_freq_source data --pl_top_k 50 --pl_boost 5.0",
         "A1': Point-Level Data-Aware - COLD"),
        
        (f"{base_cmd} {base_args} --method cold --attack point_level --attack_direction suppress_unpopular --item_freq_source qee --pl_top_k 50 --pl_boost 5.0 --freq_query_topk 20",
         "A2': Point-Level Data-Unaware (QEE) - COLD"),
        
        # Pop 对照组
        (f"{base_cmd} {base_args} --method pop --attack soft_prf --attack_direction suppress_popular --item_freq_source data --prf_gamma 0.7 --prf_beta 5.0 --prf_eps 0.02",
         "A3: Soft-PRF Data-Aware - POP"),
        
        (f"{base_cmd} {base_args} --method pop --attack soft_prf --attack_direction suppress_popular --item_freq_source qee --prf_gamma 0.7 --prf_beta 5.0 --prf_eps 0.02 --freq_query_topk 20",
         "A4: Soft-PRF Data-Unaware (QEE) - POP"),
    ]
    
    print(f"\n{Colors.BOLD}=== 命令语法验证 ==={Colors.RESET}")
    print(f"总共 {len(experiments)} 个实验配置\n")
    
    results = []
    for cmd, desc in experiments:
        success = validate_command(cmd, desc)
        results.append((desc, success))
    
    # 总结
    print(f"\n{Colors.BOLD}=== 验证总结 ==={Colors.RESET}")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for desc, success in results:
        status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {desc}")
    
    print(f"\n总体: {passed}/{total} 通过")
    
    if passed == total:
        print(f"{Colors.GREEN}\n✓ 所有命令语法检查通过！可以开始运行实验。{Colors.RESET}")
        print(f"\n{Colors.BOLD}下一步建议:{Colors.RESET}")
        print(f"  1. 运行单个命令进行全面测试，例如:")
        print(f"     python test_watermark_acc.py --device cpu --dataset_code ml-1m --model_code bert \\")
        print(f"       --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 \\")
        print(f"       --method cold --attack none")
        print(f"  2. 参考 实验二运行命令.md 获取完整命令说明")
        return 0
    else:
        print(f"{Colors.RED}\n✗ 部分命令存在问题，请修正后重试{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
