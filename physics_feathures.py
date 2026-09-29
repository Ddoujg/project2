import h5py
import numpy as np
from scipy.stats import entropy


def extract_tf_features(cwt_img):
    """
    输入: cwt_img (3, H, W) 三通道时频图（已归一化到 [0,1]）
    输出: 3*4维特征向量 [谱熵, 稀疏度, 瞬时能量波动, 峰值背景比]
    """
    features = []
    for ch in range(3):
        # 谱熵
        spectrum = cwt_img[ch].mean(axis=1)  # (H,)
        p = spectrum / (spectrum.sum() + 1e-8)
        spec_entropy = -np.sum(p * np.log(p + 1e-8))
        # 稀疏度
        l1 = np.abs(cwt_img[ch]).sum()
        l2 = np.sqrt((cwt_img[ch] ** 2).sum())
        sparsity = l1 / (l2 + 1e-8)
        # 瞬时能量波动
        time_energy = cwt_img[ch].sum(axis=0)  # (W,)
        energy_var = np.var(time_energy)
        # 峰值背景比
        peak = spectrum.max()
        mean = spectrum.mean()
        peak_ratio = peak / (mean + 1e-8)
        features.extend([spec_entropy, sparsity, energy_var, peak_ratio])
    return np.array(features)


def process_and_save_features(input_h5, output_h5):
    """从CWT h5文件中读取数据，提取特征，保存到新的h5文件"""
    with h5py.File(input_h5, 'r') as f_in:
        cwt_data = f_in['data'][:]  # (N, 3, 128, 512)
        labels = f_in['labels'][:]  # (N,)

    N = cwt_data.shape[0]
    features = []
    for i in range(N):
        feat = extract_tf_features(cwt_data[i])
        features.append(feat)
    features = np.array(features)

    with h5py.File(output_h5, 'w') as f_out:
        f_out.create_dataset('features', data=features)
        f_out.create_dataset('labels', data=labels)
    print(f"Saved {N} samples with 4 features to {output_h5}")


# 使用示例
process_and_save_features('train_dataset_fmd_cwt_30.h5', 'train_physical_features.h5')
process_and_save_features('val_dataset_fmd_cwt_30.h5', 'val_physical_features.h5')
process_and_save_features('test_dataset_fmd_cwt_30.h5', 'test_physical_features.h5')