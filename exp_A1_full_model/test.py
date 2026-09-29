import torch
import numpy as np
import json
import os
import time
from dataset import get_loaders
from model_ste import FMDGumbelSinkhornViTModel
from config import DEVICE, OUTPUT_DIR, NUM_PHYSICAL_FEATURES
from utils import set_seed, load_checkpoint, compute_metrics, compute_model_complexity, compute_sparsity
from scipy.stats import spearmanr


def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_budgets = []
    all_confidences = []
    all_attn_maps = []
    all_phis = []
    all_phys_features = []
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
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_budgets.extend(budgets)
            all_confidences.extend(confidences)
            all_attn_maps.append(S)
            # 获取中间变量
            inter = model.budget_predictor.get_intermediate(phys)
            all_phis.extend(inter['phi'])
            all_phys_features.extend(phys.cpu().numpy())
    avg_inference_time = np.mean(inference_times) * 1000
    attn_maps = np.concatenate(all_attn_maps, axis=0)
    return (np.array(all_preds), np.array(all_labels), np.array(all_budgets),
            np.array(all_confidences), attn_maps, avg_inference_time,
            np.array(all_phis), np.array(all_phys_features))


def convert_numpy(obj):
    """递归地将 NumPy 类型转换为 Python 原生类型，以便 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    else:
        return obj


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

    # 计算复杂度（传入两个输入尺寸）
    flops, params = compute_model_complexity(model,
                                             input_size_cwt=(3, 128, 512),
                                             input_size_phys=(NUM_PHYSICAL_FEATURES,),
                                             device=device)
    y_pred, y_true, budgets, confidences, attn_maps, avg_time, phis, phys_features = test(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics['flops'] = flops
    metrics['params'] = params
    metrics['avg_inference_time_ms'] = avg_time
    metrics['confidence_mean'] = float(np.mean(confidences))
    metrics['confidence_std'] = float(np.std(confidences))

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

    # 计算相关系数
    corr_phi, p_phi = spearmanr(phis, budgets)
    metrics['phi_budget_corr'] = {'spearman_r': corr_phi, 'p_value': p_phi}
    phys_corr = {}
    for i in range(phys_features.shape[1]):
        r, p = spearmanr(phys_features[:, i], budgets)
        phys_corr[f'feat_{i}'] = {'spearman_r': r, 'p_value': p}
    metrics['phys_corr'] = phys_corr

    # ====== 保存所有 .npy 文件 ======
    # 注意：先保存 confusion_matrix（如果存在），然后再从 metrics 中移除（可选）
    if 'confusion_matrix' in metrics:
        np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), np.array(metrics['confusion_matrix']))
        # 如果不想在 JSON 中保存这个大矩阵，可以移除；否则保留，转换函数会将其转为列表
        # 这里我们保留，让 JSON 中也包含，但注释掉移除语句，让转换函数一并处理
        # metrics.pop('confusion_matrix', None)
    else:
        print("Warning: 'confusion_matrix' not found in metrics, skipping save.")

    np.save(os.path.join(OUTPUT_DIR, 'budgets.npy'), budgets)
    np.save(os.path.join(OUTPUT_DIR, 'confidences.npy'), confidences)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y_true)
    np.save(os.path.join(OUTPUT_DIR, 'attention_maps.npy'), attn_maps)
    np.save(os.path.join(OUTPUT_DIR, 'phis.npy'), phis)
    np.save(os.path.join(OUTPUT_DIR, 'phys_features.npy'), phys_features)

    # ====== 将 metrics 转换为 JSON 可序列化的形式 ======
    metrics_serializable = convert_numpy(metrics)

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=4)

    # ====== 打印结果 ======
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