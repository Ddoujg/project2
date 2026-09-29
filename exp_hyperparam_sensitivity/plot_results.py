# plot_results.py
import json
import matplotlib.pyplot as plt
import os
from config import ROOT_OUTPUT_DIR

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'svg.fonttype': 'none'
})

def plot_budget_range():
    with open(os.path.join(ROOT_OUTPUT_DIR, 'budget_range_summary.json'), 'r') as f:
        data = json.load(f)
    ranges = list(data.keys())
    macro_f1 = [data[r]['macro_f1'] for r in ranges]
    fault_recall = [data[r]['fault_avg_recall'] for r in ranges]
    avg_budget = [data[r]['avg_budget'] for r in ranges]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(ranges, macro_f1, 'b-o', label='Macro-F1')
    ax1.plot(ranges, fault_recall, 'r-s', label='Fault Avg-Recall')
    ax1.set_xlabel('Budget Range [B_min, B_max]')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(ranges, avg_budget, 'g--^', label='Avg Budget')
    ax2.set_ylabel('Average Budget')
    ax2.legend(loc='upper right')

    plt.title('Sensitivity to Budget Range')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUTPUT_DIR, 'budget_range_sensitivity.svg'), format='svg')
    plt.close()

def plot_budget_lambda():
    with open(os.path.join(ROOT_OUTPUT_DIR, 'budget_lambda_summary.json'), 'r') as f:
        data = json.load(f)
    lambdas = sorted([float(k) for k in data.keys()])
    macro_f1 = [data[str(l)]['macro_f1'] for l in lambdas]
    fault_recall = [data[str(l)]['fault_avg_recall'] for l in lambdas]
    avg_budget = [data[str(l)]['avg_budget'] for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.semilogx(lambdas, macro_f1, 'b-o', label='Macro-F1')
    ax1.semilogx(lambdas, fault_recall, 'r-s', label='Fault Avg-Recall')
    ax1.set_xlabel('budget_lambda')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.semilogx(lambdas, avg_budget, 'g--^', label='Avg Budget')
    ax2.set_ylabel('Average Budget')
    ax2.legend(loc='upper right')

    plt.title('Sensitivity to budget_lambda')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUTPUT_DIR, 'budget_lambda_sensitivity.svg'), format='svg')
    plt.close()

def plot_imbalance_ratio():
    with open(os.path.join(ROOT_OUTPUT_DIR, 'imbalance_ratio_summary.json'), 'r') as f:
        data = json.load(f)
    ratios = sorted([int(k) for k in data.keys()])
    macro_f1 = [data[str(r)]['macro_f1'] for r in ratios]
    fault_recall = [data[str(r)]['fault_avg_recall'] for r in ratios]
    normal_budget = [data[str(r)]['normal_budget'] for r in ratios]
    fault_budget = [data[str(r)]['fault_budget'] for r in ratios]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(ratios, macro_f1, 'b-o', label='Macro-F1')
    ax1.plot(ratios, fault_recall, 'r-s', label='Fault Avg-Recall')
    ax1.set_xlabel('Imbalance Ratio (Normal : Fault)')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(ratios, normal_budget, 'g--^', label='Normal Budget')
    ax2.plot(ratios, fault_budget, 'm--v', label='Fault Budget')
    ax2.set_ylabel('Average Budget')
    ax2.legend(loc='upper right')

    plt.title('Sensitivity to Imbalance Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUTPUT_DIR, 'imbalance_ratio_sensitivity.svg'), format='svg')
    plt.close()

if __name__ == "__main__":
    plot_budget_range()
    plot_budget_lambda()
    plot_imbalance_ratio()
    print("Plots saved in", ROOT_OUTPUT_DIR)