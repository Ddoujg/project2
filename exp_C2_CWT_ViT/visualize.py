# visualize.py
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

def plot_training_curves():
    history = np.load(os.path.join(OUTPUT_DIR, 'training_history.npz'), allow_pickle=True)
    train_loss = history['train_losses']
    val_loss = history['val_losses']
    train_acc = history['train_accs']
    val_acc = history['val_accs']
    epochs = range(1, len(train_loss)+1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
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
    plt.title('Confusion Matrix')
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
    plt.title('Per-class Recall (C2)')
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
    plt.title('Confidence Distribution (C2)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histogram.svg'), format='svg')
    plt.close()

def plot_attention_maps():
    attn_maps = np.load(os.path.join(OUTPUT_DIR, 'attention_maps.npy'))
    # 随机选择几个样本绘制注意力图
    num_samples = min(4, attn_maps.shape[0])
    indices = np.random.choice(attn_maps.shape[0], num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    for i, idx in enumerate(indices):
        im = axes[i].imshow(attn_maps[idx], cmap='hot', aspect='auto')
        axes[i].set_title(f'Sample {idx}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Frequency')
        plt.colorbar(im, ax=axes[i])
    plt.suptitle('Attention Maps (C2: Standard ViT)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_maps.svg'), format='svg')
    plt.close()

def main():
    plot_training_curves()
    plot_confusion_matrix()
    plot_class_recall_bar()
    plot_confidence_histogram()
    plot_attention_maps()
    print("Visualization completed. Figures saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()