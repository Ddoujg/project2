# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from config import OUTPUT_DIR
from scipy.stats import spearmanr

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'svg.fonttype': 'none'
})
sns.set_style("whitegrid")

def safe_load_npy(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"File {filename} not found, skip.")
        return None

def plot_training_curves():
    history_path = os.path.join(OUTPUT_DIR, 'training_history.npz')
    if not os.path.exists(history_path):
        print("training_history.npz not found, skip.")
        return
    history = np.load(history_path, allow_pickle=True)
    # 提取两阶段数据
    stage1_train_loss = history.get('stage1_train_losses', None)
    stage1_val_acc = history.get('stage1_val_accs', None)
    stage2_train_loss = history.get('stage2_train_losses', None)
    stage2_val_acc = history.get('stage2_val_accs', None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if stage1_train_loss is not None:
        epochs1 = range(1, len(stage1_train_loss)+1)
        axes[0].plot(epochs1, stage1_train_loss, 'b-', label='Stage1 Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Stage1 Training Loss')
        axes[0].legend()
        axes[0].grid(True)
    if stage2_train_loss is not None:
        epochs2 = range(1, len(stage2_train_loss)+1)
        axes[1].plot(epochs2, stage2_train_loss, 'g-', label='Stage2 Train Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Stage2 Training Loss')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.svg'), format='svg')
    plt.close()

    # 准确率曲线（如果有）
    fig2, ax = plt.subplots(figsize=(8, 6))
    if stage1_val_acc is not None:
        epochs1 = range(1, len(stage1_val_acc)+1)
        ax.plot(epochs1, stage1_val_acc, 'b-', label='Stage1 Val Acc')
    if stage2_val_acc is not None:
        epochs2 = range(1, len(stage2_val_acc)+1)
        ax.plot(epochs2, stage2_val_acc, 'r-', label='Stage2 Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_accuracy.svg'), format='svg')
    plt.close()

def plot_confusion_matrix():
    cm = safe_load_npy('confusion_matrix.npy')
    if cm is None:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(cm.shape[0])],
                yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.svg'), format='svg')
    plt.close()

def plot_class_recall_bar():
    metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    if not os.path.exists(metrics_path):
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    report = metrics['classification_report']
    classes = []
    recalls = []
    for k, v in report.items():
        if k.isdigit():
            classes.append(f'Class {k}')
            recalls.append(v['recall'])
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, recalls, color='skyblue', edgecolor='navy')
    plt.ylim(0, 1)
    plt.ylabel('Recall')
    plt.title('Per-class Recall')
    plt.xticks(rotation=45)
    for bar, rec in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{rec:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_recall.svg'), format='svg')
    plt.close()

def plot_confidence_histogram():
    conf = safe_load_npy('confidences.npy')
    if conf is None:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(conf, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histogram.svg'), format='svg')
    plt.close()

def plot_budget_distribution():
    budgets = safe_load_npy('budgets.npy')
    labels = safe_load_npy('labels.npy')
    if budgets is None or labels is None:
        return
    normal_mask = (labels == 0)
    fault_mask = (labels > 0)
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(budgets[normal_mask], bins=20, alpha=0.5, label='Normal', color='green')
    plt.hist(budgets[fault_mask], bins=20, alpha=0.5, label='Fault', color='red')
    plt.xlabel('Budget')
    plt.ylabel('Frequency')
    plt.title('Budget Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_histogram.svg'), format='svg')
    plt.close()
    # Boxplot
    plt.figure(figsize=(6, 8))
    bp = plt.boxplot([budgets[normal_mask], budgets[fault_mask]], labels=['Normal', 'Fault'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Budget')
    plt.title('Budget Boxplot by Class')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_boxplot.svg'), format='svg')
    plt.close()

def plot_attention_maps():
    attn = safe_load_npy('attention_maps.npy')
    labels = safe_load_npy('labels.npy')
    if attn is None or labels is None:
        return
    normal_idx = np.where(labels == 0)[0]
    fault_idx = np.where(labels > 0)[0]
    if len(normal_idx) == 0 or len(fault_idx) == 0:
        return
    S_norm = attn[normal_idx[0], 0]
    S_fault = attn[fault_idx[0], 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(S_norm, cmap='hot', aspect='auto')
    axes[0].set_title('Significance Map (Normal Sample)')
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(S_fault, cmap='hot', aspect='auto')
    axes[1].set_title('Significance Map (Fault Sample)')
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_maps.svg'), format='svg')
    plt.close()

def plot_phys_correlation():
    phis = safe_load_npy('phis.npy')
    budgets = safe_load_npy('budgets.npy')
    labels = safe_load_npy('labels.npy')
    if phis is None or budgets is None:
        return
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(phis, budgets, c=labels, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar()
        cbar.set_label('Class')
    else:
        plt.scatter(phis, budgets, alpha=0.6)
    plt.xlabel('phi (Weighted Physical Feature)')
    plt.ylabel('Budget')
    plt.title('Budget vs. phi')
    from scipy.stats import spearmanr
    corr, p = spearmanr(phis, budgets)
    plt.text(0.05, 0.95, f'Spearman r = {corr:.3f} (p={p:.3e})', transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_vs_phi.svg'), format='svg')
    plt.close()

def plot_phys_feature_correlation_bar():
    """绘制每个物理特征与预算的相关系数条形图"""
    metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    if not os.path.exists(metrics_path):
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    if 'phys_corr' not in metrics:
        return
    phys_corr = metrics['phys_corr']
    indices = sorted([int(k.split('_')[1]) for k in phys_corr.keys()])
    corr_values = [phys_corr[f'feat_{i}']['spearman_r'] for i in indices]
    plt.figure(figsize=(10, 6))
    colors = ['steelblue' if c > 0 else 'lightcoral' for c in corr_values]
    bars = plt.bar(range(len(corr_values)), corr_values, color=colors, edgecolor='black')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Physical Feature Index')
    plt.ylabel('Spearman Correlation with Budget')
    plt.title('Correlation between Physical Features and Budget')
    plt.xticks(range(len(corr_values)), indices)
    for bar, val in zip(bars, corr_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*np.sign(val),
                 f'{val:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phys_correlation_bar.svg'), format='svg')
    plt.close()

def main():
    plot_training_curves()
    plot_confusion_matrix()
    plot_class_recall_bar()
    plot_confidence_histogram()
    plot_budget_distribution()
    plot_attention_maps()
    plot_phys_correlation()
    plot_phys_feature_correlation_bar()
    print("All visualizations saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()