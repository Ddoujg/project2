# visualize.py
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from config import OUTPUT_DIR_STRONG, OUTPUT_DIR_WEAK, TARGET_CLASS

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'svg.fonttype': 'none'
})

def load_results(output_dir):
    path = os.path.join(output_dir, 'test_results.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found at {path}. Please run train.py first.")
    with open(path, 'r') as f:
        return json.load(f)

def plot_comparison():
    strong = load_results(OUTPUT_DIR_STRONG)
    weak = load_results(OUTPUT_DIR_WEAK)

    categories = [f'Target Class {TARGET_CLASS} Budget', f'Target Class {TARGET_CLASS} Recall', 'Other Faults Avg Budget']
    strong_vals = [strong['target_budget'], strong['target_recall'], strong['other_faults_budget']]
    weak_vals = [weak['target_budget'], weak['target_recall'], weak['other_faults_budget']]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, strong_vals, width, label='Strong Features (Original)', color='steelblue')
    bars2 = ax.bar(x + width/2, weak_vals, width, label=f'Weak Features (Noise on Class {TARGET_CLASS})', color='lightcoral')

    ax.set_ylabel('Value')
    ax.set_title('Effect of Physical Feature Strength on Budget and Recall (Multi-Class)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('phys_feature_compensation_multiclass.svg', format='svg')
    plt.close()
    print("Comparison plot saved as phys_feature_compensation_multiclass.svg")

if __name__ == "__main__":
    plot_comparison()