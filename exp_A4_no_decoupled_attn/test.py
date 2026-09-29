# test.py (A4 实验，增加保存 phis.npy 和 phys_features.npy)
import torch
import numpy as np
import json
import os
import time
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from config import DEVICE, OUTPUT_DIR
from utils import set_seed, load_checkpoint, compute_metrics, compute_model_complexity, compute_sparsity

def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_budgets = []
    all_confidences = []
    all_attn_maps = []
    all_phis = []                # 新增：保存 phi
    all_phys_features = []       # 新增：保存原始物理特征
    inference_times = []
    with torch.no_grad():
        for cwt, phys, labels in loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(cwt, phys)
            end_time = time.time()
            inference_times.append((end_time - start_time) / cwt.size(0))
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            confidences = probs.max(dim=1)[0].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            budgets = outputs['a'].cpu().numpy()
            S = outputs['S'].cpu().numpy()
            # 获取 phi 和原始物理特征
            inter = model.budget_predictor.get_intermediate(phys)
            phis = inter['phi']
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_budgets.extend(budgets)
            all_confidences.extend(confidences)
            all_attn_maps.append(S)
            all_phis.extend(phis)
            all_phys_features.extend(phys.cpu().numpy())
    avg_inference_time = np.mean(inference_times) * 1000
    attn_maps = np.concatenate(all_attn_maps, axis=0)
    return (np.array(all_preds), np.array(all_labels), np.array(all_budgets),
            np.array(all_confidences), attn_maps, avg_inference_time,
            np.array(all_phis), np.array(all_phys_features))

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_loaders()
    if test_loader is None:
        print("No test loader. Please set TEST_CWT in config.py")
        return

    model = FMDGumbelSinkhornViTModel().to(device)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print("Loaded best model from", checkpoint_path)
    else:
        print("No trained model found, please run train.py first.")
        return

    flops, params = compute_model_complexity(model, input_size_cwt=(3, 64, 512), device=device)
    y_pred, y_true, budgets, confidences, attn_maps, avg_time, phis, phys_features = test(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics['flops'] = flops
    metrics['params'] = params
    metrics['avg_inference_time_ms'] = avg_time
    metrics['confidence_mean'] = float(np.mean(confidences))
    metrics['confidence_std'] = float(np.std(confidences))

    # 稀疏度
    total_patches = model.num_patches_h * model.num_patches_w
    sparsity = compute_sparsity(budgets, total_patches)
    normal_mask = (y_true == 0)
    fault_mask = (y_true > 0)
    budget_stats = {
        'mean': float(np.mean(budgets)),
        'std': float(np.std(budgets)),
        'min': float(np.min(budgets)),
        'max': float(np.max(budgets)),
        'normal_mean': float(np.mean(budgets[normal_mask])) if np.any(normal_mask) else 0,
        'fault_mean': float(np.mean(budgets[fault_mask])) if np.any(fault_mask) else 0,
        'sparsity': sparsity
    }
    metrics['budget_stats'] = budget_stats

    # 物理特征与预算的相关性（保存中间变量）
    from scipy.stats import spearmanr
    corr_phi, p_phi = spearmanr(phis, budgets)
    metrics['phi_budget_corr'] = {'spearman_r': corr_phi, 'p_value': p_phi}
    # 可选：计算每个物理特征的相关性
    phys_corr = {}
    for i in range(phys_features.shape[1]):
        r, p = spearmanr(phys_features[:, i], budgets)
        phys_corr[f'feat_{i}'] = {'spearman_r': r, 'p_value': p}
    metrics['phys_corr'] = phys_corr

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), np.array(metrics['confusion_matrix']))
    np.save(os.path.join(OUTPUT_DIR, 'budgets.npy'), budgets)
    np.save(os.path.join(OUTPUT_DIR, 'confidences.npy'), confidences)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y_true)
    np.save(os.path.join(OUTPUT_DIR, 'attention_maps.npy'), attn_maps)
    np.save(os.path.join(OUTPUT_DIR, 'phis.npy'), phis)                     # 新增
    np.save(os.path.join(OUTPUT_DIR, 'phys_features.npy'), phys_features)   # 新增

    print("\n=== Test Results ===")
    print(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    print(f"Fault Avg Recall: {metrics['fault_avg_recall']:.4f}")
    print(f"Recall Std: {metrics['recall_std']:.4f}")
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Avg Inference Time: {avg_time:.2f} ms/sample")
    print(f"Confidence: mean={metrics['confidence_mean']:.4f}, std={metrics['confidence_std']:.4f}")
    print(f"Budget: mean={budget_stats['mean']:.2f}, std={budget_stats['std']:.2f}, sparsity={sparsity:.4f}")
    print(f"  Normal avg budget: {budget_stats['normal_mean']:.2f}")
    print(f"  Fault avg budget: {budget_stats['fault_mean']:.2f}")
    print(f"Phi-budget Spearman r: {corr_phi:.4f}, p={p_phi:.4e}")

if __name__ == "__main__":
    main()