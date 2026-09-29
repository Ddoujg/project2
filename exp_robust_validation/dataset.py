# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
from config import TRAIN_CWT, TRAIN_PHYS, VAL_CWT, VAL_PHYS, TEST_CWT, TEST_PHYS, BATCH_SIZE, NUM_WORKERS, NORMAL_SAMPLE_SIZE

class CWTPhysicalDataset(Dataset):
    def __init__(self, cwt_h5_path, phys_h5_path, norm_params=None):
        with h5py.File(cwt_h5_path, 'r') as f:
            self.cwt_data = f['data'][:]
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

def get_train_loader_with_fault_samples(fault_samples_per_class, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    从原始训练集中采样固定数量的正常样本和每类故障样本。
    fault_samples_per_class: 每类故障样本数（整数）
    正常样本数由 NORMAL_SAMPLE_SIZE 决定。
    """
    full_train = CWTPhysicalDataset(TRAIN_CWT, TRAIN_PHYS, norm_params=None)
    norm_params = full_train.get_norm_params()
    labels = full_train.labels

    # 正常样本索引（标签0）
    normal_idx = np.where(labels == 0)[0]
    # 故障类索引（标签1-8）
    fault_indices_per_class = {}
    for cls in range(1, 9):
        cls_idx = np.where(labels == cls)[0]
        fault_indices_per_class[cls] = cls_idx

    # 采样正常样本
    if len(normal_idx) >= NORMAL_SAMPLE_SIZE:
        sampled_normal_idx = np.random.choice(normal_idx, NORMAL_SAMPLE_SIZE, replace=False)
    else:
        sampled_normal_idx = normal_idx  # 如果不够就全用（但通常足够）

    # 采样故障样本
    sampled_fault_idx = []
    for cls in range(1, 9):
        cls_idx = fault_indices_per_class[cls]
        if len(cls_idx) >= fault_samples_per_class:
            sampled = np.random.choice(cls_idx, fault_samples_per_class, replace=False)
        else:
            # 如果不够，使用有放回采样（但这里原始每类30，最大30，所以不会出现）
            sampled = np.random.choice(cls_idx, fault_samples_per_class, replace=True)
        sampled_fault_idx.extend(sampled)

    selected_idx = np.concatenate([sampled_normal_idx, sampled_fault_idx])
    subset = Subset(full_train, selected_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, norm_params

def get_val_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    val_dataset = CWTPhysicalDataset(VAL_CWT, VAL_PHYS, norm_params=norm_params)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    test_dataset = CWTPhysicalDataset(TEST_CWT, TEST_PHYS, norm_params=norm_params)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)