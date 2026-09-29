# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
from config import TRAIN_CWT, TRAIN_PHYS, VAL_CWT, VAL_PHYS, TEST_CWT, TEST_PHYS, BATCH_SIZE, NUM_WORKERS, ORIGINAL_NORMAL_COUNT, ORIGINAL_FAULT_COUNT
from exp_hyperparam_sensitivity import config


class CWTPhysicalDataset(Dataset):
    def __init__(self, cwt_h5_path, phys_h5_path, norm_params=None):
        with h5py.File(cwt_h5_path, 'r') as f:
            self.cwt_data = f['data'][:]      # (N, 3, 128, 512)
            self.labels = f['labels'][:]
        with h5py.File(phys_h5_path, 'r') as f:
            phys_data = f['features'][:]
            phys_labels = f['labels'][:]
            assert np.all(self.labels == phys_labels), "标签不一致"
        if norm_params is None:
            self.mean_val = phys_data.mean(axis=0, keepdims=True)
            self.std_val = phys_data.std(axis=0, keepdims=True)
            self.std_val[self.std_val == 0] = 1.0
            self.phys_data = (phys_data - self.mean_val) / self.std_val
        else:
            self.mean_val, self.std_val = norm_params
            self.phys_data = (phys_data - self.mean_val) / self.std_val
        self.phys_data = self.phys_data.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cwt = torch.tensor(self.cwt_data[idx], dtype=torch.float32)
        phys = torch.tensor(self.phys_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cwt, phys, label

    def get_norm_params(self):
        return self.mean_val, self.std_val

def get_balanced_train_loader(ratio, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    根据不均衡比例（正常:故障）从原始训练集中采样。
    ratio: 正常样本数 / 故障样本总数。例如 ratio=10 表示正常:故障=10:1。
    返回训练DataLoader，以及归一化参数（用于验证集和测试集）。
    """
    # 加载完整训练集
    full_train = CWTPhysicalDataset(TRAIN_CWT, TRAIN_PHYS, norm_params=None)
    norm_params = full_train.get_norm_params()
    labels = full_train.labels
    # 获取正常样本索引和故障样本索引
    normal_idx = np.where(labels == 0)[0]
    fault_idx = np.where(labels > 0)[0]
    # 故障样本总数
    total_fault = len(fault_idx)
    # 根据比例确定保留的正常样本数
    target_normal_count = int(ratio * total_fault)
    if target_normal_count > len(normal_idx):
        target_normal_count = len(normal_idx)  # 不能超过原始数量
    # 随机采样正常样本
    np.random.seed(config.SEED)  # 注意：需要导入config，这里简单处理
    sampled_normal_idx = np.random.choice(normal_idx, target_normal_count, replace=False)
    # 保留所有故障样本
    selected_idx = np.concatenate([sampled_normal_idx, fault_idx])
    # 创建子集
    subset = Subset(full_train, selected_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, norm_params

def get_val_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    val_dataset = CWTPhysicalDataset(VAL_CWT, VAL_PHYS, norm_params=norm_params)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    test_dataset = CWTPhysicalDataset(TEST_CWT, TEST_PHYS, norm_params=norm_params)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)