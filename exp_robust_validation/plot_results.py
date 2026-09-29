# plot_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from config import ROOT_OUTPUT_DIR, FAULT_SAMPLE_SIZES
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

def plot_core_idea():
    with open(os.path.join(ROOT_OUTPUT_DIR, 'summary.json'), 'r') as f:
        results = json.load(f)

    # 提取数据
    fault_samples = [r['fault_samples_per_class'] for r in results]
    fault_avg_recall = [r['fault_avg_recall'] for r in results]
    fault_budget_mean = [r['fault_budget_mean'] for r in results]
    normal_budget = [r['normal_budget'] for r in results]

    # 计算预算分配比（故障类平均预算 / 正常类平均预算）
    budget_ratio = [fault_budget_mean[i] / normal_budget[i] for i in range(len(fault_samples))]

    # 相关系数计算
    # 故障样本数 vs 故障平均召回率
    r_recall, p_recall = pearsonr(fault_samples, fault_avg_recall)
    # 故障样本数 vs 预算分配比
    r_ratio, p_ratio = pearsonr(fault_samples, budget_ratio)
    # 可选 Spearman 秩相关系数（对单调关系更稳健）
    r_spearman_ratio, p_spearman_ratio = spearmanr(fault_samples, budget_ratio)

    print("=== Correlation Analysis ===")
    print(f"Fault samples vs Fault recall:       r = {r_recall:.4f}, p = {p_recall:.4e}")
    print(f"Fault samples vs Budget ratio:       r = {r_ratio:.4f}, p = {p_ratio:.4e}")
    print(f"Fault samples vs Budget ratio (Spearman): r = {r_spearman_ratio:.4f}, p = {p_spearman_ratio:.4e}")

    # 保存相关系数到文本
    with open(os.path.join(ROOT_OUTPUT_DIR, 'correlation.txt'), 'w') as f:
        f.write(f"Fault samples vs Fault recall (Pearson): r={r_recall:.4f}, p={p_recall:.4e}\n")
        f.write(f"Fault samples vs Budget ratio (Pearson): r={r_ratio:.4f}, p={p_ratio:.4e}\n")
        f.write(f"Fault samples vs Budget ratio (Spearman): r={r_spearman_ratio:.4f}, p={p_spearman_ratio:.4e}\n")

    # 绘图：双 Y 轴（左：召回率，右：预算分配比）
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(fault_samples, fault_avg_recall, 'b-o', linewidth=2, markersize=8, label='Fault Avg Recall')
    ax1.set_xlabel('Number of Fault Samples per Class')
    ax1.set_ylabel('Fault Avg Recall', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(fault_samples, budget_ratio, 'r-s', linewidth=2, markersize=8, label='Budget Ratio (Fault/Normal)')
    ax2.set_ylabel('Budget Ratio', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Core Idea Validation: Budget Ratio vs Fault Recall')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUTPUT_DIR, 'core_idea_validation.svg'), format='svg')
    plt.close()

    # 可选：额外绘制原始预算值（作为参考）
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(fault_samples, fault_budget_mean, 'r-s', label='Fault Avg Budget')
    ax3.plot(fault_samples, normal_budget, 'g--^', label='Normal Avg Budget')
    ax3.set_xlabel('Number of Fault Samples per Class')
    ax3.set_ylabel('Average Budget')
    ax3.legend()
    ax3.grid(True)
    plt.title('Absolute Budget Values')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUTPUT_DIR, 'absolute_budgets.svg'), format='svg')
    plt.close()

    print(f"Plots saved in {ROOT_OUTPUT_DIR}")

if __name__ == "__main__":
    plot_core_idea()