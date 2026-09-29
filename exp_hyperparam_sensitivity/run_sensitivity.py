# run_sensitivity.py
import os
import json
from config import ROOT_OUTPUT_DIR, DEVICE, DEFAULT_B_MIN, DEFAULT_B_MAX, DEFAULT_BUDGET_LAMBDA
from dataset import get_balanced_train_loader, get_val_loader, get_test_loader
from train_eval import train_and_evaluate
import numpy as np

def run_budget_range_sensitivity():
    """测试不同预算范围"""
    ranges = [
        (1, 10), (2, 20), (2, 25), (5, 30), (5, 40)
    ]
    results = {}
    for B_min, B_max in ranges:
        print(f"\n=== Running budget range [{B_min}, {B_max}] ===")
        # 使用默认不均衡比例（10:1）和默认 budget_lambda
        train_loader, norm_params = get_balanced_train_loader(ratio=10)
        val_loader = get_val_loader(norm_params=norm_params)
        test_loader = get_test_loader(norm_params=norm_params)
        out_dir = os.path.join(ROOT_OUTPUT_DIR, f'budget_range_{B_min}_{B_max}')
        metrics, budget_stats = train_and_evaluate(
            B_min, B_max, DEFAULT_BUDGET_LAMBDA,
            train_loader, val_loader, test_loader, out_dir, DEVICE
        )
        results[f'[{B_min},{B_max}]'] = {
            'macro_f1': metrics['macro_f1'],
            'fault_avg_recall': metrics['fault_avg_recall'],
            'avg_budget': budget_stats['mean'],
            'normal_budget': budget_stats['normal_mean'],
            'fault_budget': budget_stats['fault_mean']
        }
    # 保存汇总结果
    with open(os.path.join(ROOT_OUTPUT_DIR, 'budget_range_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Budget range sensitivity completed.")

def run_budget_lambda_sensitivity():
    """测试不同 budget_lambda"""
    lambdas = [0.001, 0.01, 0.05, 0.1, 0.5]
    results = {}
    for lb in lambdas:
        print(f"\n=== Running budget_lambda = {lb} ===")
        train_loader, norm_params = get_balanced_train_loader(ratio=10)
        val_loader = get_val_loader(norm_params=norm_params)
        test_loader = get_test_loader(norm_params=norm_params)
        out_dir = os.path.join(ROOT_OUTPUT_DIR, f'budget_lambda_{lb}')
        metrics, budget_stats = train_and_evaluate(
            DEFAULT_B_MIN, DEFAULT_B_MAX, lb,
            train_loader, val_loader, test_loader, out_dir, DEVICE
        )
        results[lb] = {
            'macro_f1': metrics['macro_f1'],
            'fault_avg_recall': metrics['fault_avg_recall'],
            'avg_budget': budget_stats['mean'],
            'normal_budget': budget_stats['normal_mean'],
            'fault_budget': budget_stats['fault_mean']
        }
    with open(os.path.join(ROOT_OUTPUT_DIR, 'budget_lambda_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Budget lambda sensitivity completed.")

def run_imbalance_ratio_sensitivity():
    """测试不同不均衡比例"""
    ratios = [1, 5, 10, 20, 50]  # 正常:故障的比例
    results = {}
    for ratio in ratios:
        print(f"\n=== Running imbalance ratio 1:{ratio} (normal:fault) ===")
        # 注意：ratio参数表示正常:故障，例如ratio=1表示1:1
        train_loader, norm_params = get_balanced_train_loader(ratio=ratio)
        val_loader = get_val_loader(norm_params=norm_params)
        test_loader = get_test_loader(norm_params=norm_params)
        out_dir = os.path.join(ROOT_OUTPUT_DIR, f'imbalance_ratio_{ratio}')
        metrics, budget_stats = train_and_evaluate(
            DEFAULT_B_MIN, DEFAULT_B_MAX, DEFAULT_BUDGET_LAMBDA,
            train_loader, val_loader, test_loader, out_dir, DEVICE
        )
        results[ratio] = {
            'macro_f1': metrics['macro_f1'],
            'fault_avg_recall': metrics['fault_avg_recall'],
            'avg_budget': budget_stats['mean'],
            'normal_budget': budget_stats['normal_mean'],
            'fault_budget': budget_stats['fault_mean']
        }
    with open(os.path.join(ROOT_OUTPUT_DIR, 'imbalance_ratio_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Imbalance ratio sensitivity completed.")

if __name__ == "__main__":
    # 依次运行三个敏感性分析（可单独注释）
    run_budget_range_sensitivity()
    run_budget_lambda_sensitivity()
    run_imbalance_ratio_sensitivity()