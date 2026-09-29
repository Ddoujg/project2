# check_phys_features.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import silhouette_score
import os

# 配置
PHYS_H5 = 'train_physical_features.h5'   # 训练集物理特征文件
OUTPUT_DIR = './phys_feature_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置绘图风格
plt.rcParams.update({'font.size': 12})
sns.set_style("whitegrid")

# 1. 加载数据
with h5py.File(PHYS_H5, 'r') as f:
    features = f['features'][:]   # (N, D)
    labels = f['labels'][:]       # (N,)

# 区分正常(0)和故障(>0)
normal_mask = (labels == 0)
fault_mask = (labels > 0)
normal_feat = features[normal_mask]
fault_feat = features[fault_mask]

print(f"总样本数: {len(labels)}")
print(f"正常样本数: {len(normal_feat)}, 故障样本数: {len(fault_feat)}")
print(f"特征维度: {features.shape[1]}\n")

# 2. 每个特征的统计对比
print("=" * 60)
print("特征区分度分析 (正常 vs 故障)")
print("=" * 60)

for i in range(features.shape[1]):
    normal_vals = normal_feat[:, i]
    fault_vals = fault_feat[:, i]
    # 均值、标准差
    mean_n, std_n = np.mean(normal_vals), np.std(normal_vals)
    mean_f, std_f = np.mean(fault_vals), np.std(fault_vals)
    # t检验 (若数据不服从正态分布可改用 Mann-Whitney)
    t_stat, p_val = ttest_ind(normal_vals, fault_vals, equal_var=False)
    # Cohen's d 效应量
    pooled_std = np.sqrt((std_n**2 + std_f**2) / 2)
    cohen_d = (mean_f - mean_n) / pooled_std if pooled_std > 0 else 0
    print(f"特征 {i:2d}: 正常 {mean_n:7.3f}±{std_n:5.3f} | 故障 {mean_f:7.3f}±{std_f:5.3f} | "
          f"t={t_stat:6.2f}, p={p_val:.2e}, Cohen's d={cohen_d:6.3f}")

# 3. 箱线图 (每个特征)
n_features = features.shape[1]
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()
for i in range(n_features):
    ax = axes[i]
    data = [normal_feat[:, i], fault_feat[:, i]]
    bp = ax.boxplot(data, labels=['Normal', 'Fault'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    ax.set_title(f'Feature {i}')
    ax.set_ylabel('Value')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_boxplots.svg'), format='svg')
plt.close()

# 4. t-SNE 可视化 (高维投影)
print("\n正在计算 t-SNE 投影...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# 如果样本太多，可随机采样
max_samples = 2000
if len(features) > max_samples:
    idx = np.random.choice(len(features), max_samples, replace=False)
    tsne_data = features[idx]
    tsne_labels = labels[idx]
else:
    tsne_data = features
    tsne_labels = labels
tsne_result = tsne.fit_transform(tsne_data)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=tsne_labels, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Class (0=Normal, >0=Fault)')
plt.title('t-SNE Visualization of Physical Features')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_physical_features.svg'), format='svg')
plt.close()

# 5. 特征间相关性热力图 (可选)
corr = np.corrcoef(features.T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of Physical Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation.svg'), format='svg')
plt.close()

# 6. 计算轮廓系数 (评估聚类可分性)
if len(features) > 5000:
    # 采样
    idx_sample = np.random.choice(len(features), 5000, replace=False)
    sil_score = silhouette_score(features[idx_sample], labels[idx_sample], metric='euclidean')
else:
    sil_score = silhouette_score(features, labels, metric='euclidean')
print(f"\n轮廓系数 (Silhouette Score): {sil_score:.4f} (越接近1表示可分性越好)")

print(f"\n分析结果已保存至 {OUTPUT_DIR}")