# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from config import OUTPUT_DIR
from scipy.stats import pearsonr, spearmanr

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

def plot_training_curves():
    history = np.load(os.path.join(OUTPUT_DIR, 'training_history.npz'), allow_pickle=True)
    train_loss = history['train_loss']
    #val_loss = history['val_losses']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    epochs = range(1, len(train_loss)+1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    #ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, 'b-', label='Train Acc')
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.svg'), format='svg')
    plt.close()

def plot_confusion_matrix():
    cm = np.load(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(cm.shape[0])],
                yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
    plt.title('Confusion Matrix (Full Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.svg'), format='svg')
    plt.close()

def plot_class_recall_bar():
    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'r') as f:
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
    plt.title('Per-class Recall (Full Model)')
    plt.xticks(rotation=45)
    for bar, rec in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{rec:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_recall.svg'), format='svg')
    plt.close()

def plot_confidence_histogram():
    confidences = np.load(os.path.join(OUTPUT_DIR, 'confidences.npy'))
    plt.figure(figsize=(8, 6))
    plt.hist(confidences, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution (Full Model)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histogram.svg'), format='svg')
    plt.close()

def plot_budget_distribution():
    budgets = np.load(os.path.join(OUTPUT_DIR, 'budgets.npy'))
    labels = np.load(os.path.join(OUTPUT_DIR, 'labels.npy'))
    normal_mask = (labels == 0)
    fault_mask = (labels > 0)

    # 直方图
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

    # 箱线图
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
    attn_maps = np.load(os.path.join(OUTPUT_DIR, 'attention_maps.npy'))  # [N,1,H,W]
    labels = np.load(os.path.join(OUTPUT_DIR, 'labels.npy'))
    # 选择一个正常样本和一个故障样本
    normal_idx = np.where(labels == 0)[0]
    fault_idx = np.where(labels > 0)[0]
    if len(normal_idx) > 0 and len(fault_idx) > 0:
        idx_norm = normal_idx[0]
        idx_fault = fault_idx[0]
        S_norm = attn_maps[idx_norm, 0]
        S_fault = attn_maps[idx_fault, 0]
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
    budgets = np.load(os.path.join(OUTPUT_DIR, 'budgets.npy'))
    phis = np.load(os.path.join(OUTPUT_DIR, 'phis.npy'))
    labels = np.load(os.path.join(OUTPUT_DIR, 'labels.npy'))
    # 预算 vs phi 散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(phis, budgets, c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('phi (Weighted Physical Feature)')
    plt.ylabel('Budget')
    plt.title('Budget vs. phi')
    cbar = plt.colorbar()
    cbar.set_label('Class')
    corr, p = spearmanr(phis, budgets)
    plt.text(0.05, 0.95, f'Spearman r = {corr:.3f} (p={p:.3e})', transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_vs_phi.svg'), format='svg')
    plt.close()

    # 预算 vs 每个物理特征的散点图（可选）
    phys_features = np.load(os.path.join(OUTPUT_DIR, 'phys_features.npy'))
    for i in range(phys_features.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.scatter(phys_features[:, i], budgets, c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel(f'Physical Feature {i}')
        plt.ylabel('Budget')
        plt.title(f'Budget vs. Physical Feature {i}')
        corr, p = spearmanr(phys_features[:, i], budgets)
        plt.text(0.05, 0.95, f'Spearman r = {corr:.3f} (p={p:.3e})', transform=plt.gca().transAxes, fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'budget_vs_feat_{i}.svg'), format='svg')
        plt.close()

def main():
    plot_training_curves()
    plot_confusion_matrix()
    plot_class_recall_bar()
    plot_confidence_histogram()
    plot_budget_distribution()
    plot_attention_maps()
    plot_phys_correlation()
    print("Visualization completed. Figures saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()