# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE, NUM_WORKERS
from utils import load_raw_segments

# ==================== 归一化方法 ====================
NORM_METHOD = 'zscore'      # 'zscore' 或 'minmax'

def compute_norm_params(data, method='zscore'):
    """
    计算归一化参数
    data: numpy array, shape (N, L) 或 (N, C, L)
    method: 'zscore' 或 'minmax'
    """
    if method == 'zscore':
        # 沿着样本维度计算均值和标准差 (axis=0)
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std == 0] = 1.0   # 防止除零
        return mean, std
    elif method == 'minmax':
        # 计算每个特征维度的最小值和最大值
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return min_val, range_val
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

def apply_norm(data, params, method='zscore'):
    """
    应用归一化
    data: numpy array
    params: (p1, p2) 对应 (mean, std) 或 (min, range)
    """
    if method == 'zscore':
        mean, std = params
        return (data - mean) / std
    elif method == 'minmax':
        min_val, range_val = params
        return (data - min_val) / range_val
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

# ==================== Dataset 类 ====================
class RawSignalDataset(Dataset):
    def __init__(self, signals, labels, norm_params=None, method='zscore'):
        """
        signals: numpy array, 已归一化或原始数据
        labels: numpy array
        norm_params: 如果为 None，则认为 signals 已归一化；否则为 (p1, p2)
        method: 归一化方法，用于可能需要的逆变换
        """
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.norm_params = norm_params
        self.method = method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

    def inverse_normalize(self, data):
        """将归一化后的数据还原（用于可视化等）"""
        if self.norm_params is None:
            return data
        if self.method == 'zscore':
            mean, std = self.norm_params
            return data * std + mean
        elif self.method == 'minmax':
            min_val, range_val = self.norm_params
            return data * range_val + min_val
        else:
            return data

# ==================== 数据加载器 ====================
def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_method=NORM_METHOD):
    # 加载原始数据（未归一化）
    X_train, y_train, X_val, y_val, X_test, y_test = load_raw_segments()

    # 在训练集上计算归一化参数
    train_params = compute_norm_params(X_train, method=norm_method)

    # 对训练集、验证集、测试集应用相同的归一化
    X_train_norm = apply_norm(X_train, train_params, method=norm_method)
    X_val_norm   = apply_norm(X_val,   train_params, method=norm_method)
    X_test_norm  = apply_norm(X_test,  train_params, method=norm_method)

    # 创建 Dataset（保留归一化参数，以便可能进行逆变换）
    train_dataset = RawSignalDataset(X_train_norm, y_train, norm_params=train_params, method=norm_method)
    val_dataset   = RawSignalDataset(X_val_norm,   y_val,   norm_params=train_params, method=norm_method)
    test_dataset  = RawSignalDataset(X_test_norm,  y_test,  norm_params=train_params, method=norm_method)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, _, _ = get_loaders()
    for x, y in train_loader:
        print(f"Batch shape: {x.shape}, labels: {y.shape}")
        break