# visualize.py
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from config import OUTPUT_ORIGINAL, OUTPUT_NOISY

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
    with open(os.path.join(output_dir, 'test_results.json'), 'r') as f:
        return json.load(f)

def plot_comparison():
    original = load_results(OUTPUT_ORIGINAL)
    noisy = load_results(OUTPUT_NOISY)

    categories = ['Target Budget', 'Target Recall', 'Other Faults Budget']
    original_vals = [original['target_budget'], original['target_recall'], original['other_faults_budget']]
    noisy_vals = [noisy['target_budget'], noisy['target_recall'], noisy['other_faults_budget']]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, original_vals, width, label='Original (No Noise)', color='steelblue')
    bars2 = ax.bar(x + width/2, noisy_vals, width, label='Noisy (SNR=10dB on Inner Fault)', color='lightcoral')

    ax.set_ylabel('Value')
    ax.set_title('Effect of Noise (Feature Strength) on Budget and Recall')
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
    plt.savefig('noise_comparison.svg', format='svg')
    plt.close()
    print("Comparison plot saved as noise_comparison.svg")

if __name__ == "__main__":
    plot_comparison()