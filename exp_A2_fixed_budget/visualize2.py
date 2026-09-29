import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from config import OUTPUT_DIR

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
    # 尝试多种键名
    train_loss = None
    for key in ['train_losses', 'train_loss', 'losses', 'loss']:
        if key in history:
            train_loss = history[key]
            break
    val_loss = None
    for key in ['val_losses', 'val_loss']:
        if key in history:
            val_loss = history[key]
            break
    train_acc = None
    for key in ['train_accs', 'train_acc', 'accs', 'accuracy']:
        if key in history:
            train_acc = history[key]
            break
    val_acc = None
    for key in ['val_accs', 'val_acc']:
        if key in history:
            val_acc = history[key]
            break
    if train_loss is None or val_loss is None:
        print("Loss data not found, skip training curves.")
        return
    epochs = range(1, len(train_loss)+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss')
    ax1.legend(); ax1.grid(True)
    if train_acc is not None and val_acc is not None:
        ax2.plot(epochs, train_acc, 'b-', label='Train Acc')
        ax2.plot(epochs, val_acc, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Training and Validation Accuracy')
        ax2.legend(); ax2.grid(True)
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
        print("test_metrics.json not found, skip.")
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    report = metrics.get('classification_report', {})
    classes, recalls = [], []
    for k, v in report.items():
        if k.isdigit():
            classes.append(f'Class {k}')
            recalls.append(v['recall'])
    if not classes:
        print("No per-class recall data, skip.")
        return
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, recalls, color='skyblue', edgecolor='navy')
    plt.ylim(0, 1); plt.ylabel('Recall'); plt.title('Per-class Recall')
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
    plt.xlabel('Confidence'); plt.ylabel('Frequency'); plt.title('Confidence Distribution')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histogram.svg'), format='svg')
    plt.close()

def plot_budget_distribution():
    budgets = safe_load_npy('budgets.npy')
    labels = safe_load_npy('labels.npy')
    if budgets is None or labels is None:
        print("budgets.npy or labels.npy not found, skip budget plots.")
        return
    normal_mask = (labels == 0)
    fault_mask = (labels > 0)
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(budgets[normal_mask], bins=20, alpha=0.5, label='Normal', color='green')
    plt.hist(budgets[fault_mask], bins=20, alpha=0.5, label='Fault', color='red')
    plt.xlabel('Budget'); plt.ylabel('Frequency'); plt.title('Budget Distribution by Class')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_histogram.svg'), format='svg')
    plt.close()
    # Boxplot
    plt.figure(figsize=(6, 8))
    bp = plt.boxplot([budgets[normal_mask], budgets[fault_mask]], labels=['Normal', 'Fault'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen'); bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Budget'); plt.title('Budget Boxplot by Class'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'budget_boxplot.svg'), format='svg')
    plt.close()

def plot_attention_maps():
    attn = safe_load_npy('attention_maps.npy')
    labels = safe_load_npy('labels.npy')
    if attn is None or labels is None:
        print("attention_maps.npy or labels.npy not found, skip attention maps.")
        return
    normal_idx = np.where(labels == 0)[0]
    fault_idx = np.where(labels > 0)[0]
    if len(normal_idx) == 0 or len(fault_idx) == 0:
        print("No normal or fault samples in labels, skip attention maps.")
        return
    idx_norm = normal_idx[0]
    idx_fault = fault_idx[0]
    S_norm = attn[idx_norm, 0]
    S_fault = attn[idx_fault, 0]
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
        print("phis.npy or budgets.npy not found, skip budget-vs-phi plot.")
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