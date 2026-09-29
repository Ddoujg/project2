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
    """绘制训练损失、准确率、预算均值、总预算偏差曲线"""
    history_path = os.path.join(OUTPUT_DIR, 'training_history.npz')
    if not os.path.exists(history_path):
        print("training_history.npz not found, skip.")
        return
    history = np.load(history_path, allow_pickle=True)
    # 兼容不同的键名
    train_loss = history.get('train_losses', history.get('train_loss', None))
    val_acc = history.get('val_accs', history.get('val_acc', None))
    train_acc = history.get('train_accs', history.get('train_acc', None))
    normal_budget = history.get('normal_budget', None)
    fault_budget = history.get('fault_budget', None)
    actual_total = history.get('actual_total', None)
    expected_total = history.get('expected_total', None)
    epochs = range(1, len(train_loss)+1) if train_loss is not None else range(1, len(val_acc)+1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if train_loss is not None:
        axes[0,0].plot(epochs, train_loss, 'b-', label='Train Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
    if train_acc is not None and val_acc is not None:
        axes[0,1].plot(epochs, train_acc, 'b-', label='Train Acc')
        axes[0,1].plot(epochs, val_acc, 'r-', label='Val Acc')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].set_title('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True)
    if normal_budget is not None and fault_budget is not None:
        axes[1,0].plot(epochs, normal_budget, 'g-', label='Normal Budget')
        axes[1,0].plot(epochs, fault_budget, 'm-', label='Fault Budget')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Average Budget')
        axes[1,0].set_title('Budget by Class')
        axes[1,0].legend()
        axes[1,0].grid(True)
    if actual_total is not None and expected_total is not None:
        axes[1,1].plot(epochs, actual_total, 'c-', label='Actual Total')
        axes[1,1].plot(epochs, expected_total, 'k--', label='Expected Total')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Total Budget')
        axes[1,1].set_title('Total Budget Constraint')
        axes[1,1].legend()
        axes[1,1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.svg'), format='svg')
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

def plot_phys_correlation_matrix():
    """绘制物理特征与预算的相关系数矩阵热力图"""
    phys = safe_load_npy('phys_features.npy')
    budgets = safe_load_npy('budgets.npy')
    if phys is None or budgets is None:
        return
    # 计算每个特征与预算的 Spearman 相关系数
    n_features = phys.shape[1]
    corr = np.zeros(n_features)
    for i in range(n_features):
        r, _ = spearmanr(phys[:, i], budgets)
        corr[i] = r
    # 绘制条形图（更直观）
    plt.figure(figsize=(10, 6))
    colors = ['steelblue' if c > 0 else 'lightcoral' for c in corr]
    bars = plt.bar(range(n_features), corr, color=colors, edgecolor='black')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Physical Feature Index')
    plt.ylabel('Spearman Correlation with Budget')
    plt.title('Correlation between Physical Features and Budget')
    plt.xticks(range(n_features))
    for bar, val in zip(bars, corr):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*np.sign(val),
                 f'{val:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phys_correlation_bar.svg'), format='svg')
    plt.close()
    # 可选：散点图矩阵（特征太多，只选前4个）
    if n_features >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, ax in enumerate(axes.flat):
            if i < n_features:
                ax.scatter(phys[:, i], budgets, alpha=0.5, s=5)
                r, p = spearmanr(phys[:, i], budgets)
                ax.set_title(f'Feat {i}: r={r:.2f}')
                ax.set_xlabel(f'Feature {i}')
                ax.set_ylabel('Budget')
                ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'budget_vs_phys_scatter.svg'), format='svg')
        plt.close()

def main():
    plot_training_curves()
    plot_confusion_matrix()
    plot_class_recall_bar()
    plot_confidence_histogram()
    plot_budget_distribution()
    plot_attention_maps()
    plot_phys_correlation_matrix()
    print("All visualizations saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()