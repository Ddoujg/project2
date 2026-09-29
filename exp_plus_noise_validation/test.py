# test.py
import torch
import numpy as np
import json
import os
import time
from config import DEVICE
from utils import compute_metrics

def test(model, loader, device, output_dir, target_class):
    model.eval()
    all_preds = []
    all_labels = []
    all_budgets = []
    inference_times = []
    with torch.no_grad():
        for cwt, phys, labels in loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(cwt, phys)
            end_time = time.time()
            inference_times.append((end_time - start_time) / cwt.size(0))
            logits = outputs['logits']
            preds = logits.argmax(dim=1).cpu().numpy()
            budgets = outputs['a'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_budgets.extend(budgets)
    avg_time = np.mean(inference_times) * 1000

    # 转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_budgets = np.array(all_budgets)

    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)

    # 提取目标故障类的预算和召回率
    target_mask = (all_labels == target_class)
    if np.any(target_mask):
        target_budget = np.mean(all_budgets[target_mask])
        target_recall = metrics['classification_report'][str(target_class)]['recall']
    else:
        target_budget = 0.0
        target_recall = 0.0

    # 其他故障类的平均预算（排除正常类和目标类）
    other_fault_mask = (all_labels > 0) & (all_labels != target_class)
    other_budget = np.mean(all_budgets[other_fault_mask]) if np.any(other_fault_mask) else 0.0

    # 保存详细结果
    result = {
        'target_class': target_class,
        'target_budget': float(target_budget),
        'target_recall': float(target_recall),
        'other_faults_budget': float(other_budget),
        'overall_acc': metrics['overall_acc'],
        'macro_f1': metrics['macro_f1'],
        'g_mean': metrics['g_mean'],
        'fault_avg_recall': metrics['fault_avg_recall'],
        'recall_std': metrics['recall_std'],
        'avg_inference_time_ms': avg_time,
        'params': sum(p.numel() for p in model.parameters()),
        'full_metrics': metrics
    }
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(result, f, indent=4)

    print("\n=== Test Results ===")
    print(f"Target class {target_class}: Budget = {target_budget:.2f}, Recall = {target_recall:.4f}")
    print(f"Other faults average budget: {other_budget:.2f}")
    print(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    print(f"Fault Avg Recall: {metrics['fault_avg_recall']:.4f}")

    return result

if __name__ == "__main__":
    print("This script is called from train.py. Do not run directly.")