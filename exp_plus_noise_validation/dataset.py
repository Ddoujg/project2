# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from config import (
    TRAIN_CWT, TRAIN_PHYS, VAL_CWT, VAL_PHYS, TEST_CWT, TEST_PHYS,
    TARGET_CLASS, NOISE_SNR, BATCH_SIZE, NUM_WORKERS
)

class CWTPhysicalDataset(Dataset):
    def __init__(self, cwt_h5_path, phys_h5_path, norm_params=None,
                 add_noise_for_class=None, snr_db=None):
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
        self.add_noise_for_class = add_noise_for_class
        self.snr_db = snr_db

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cwt = torch.tensor(self.cwt_data[idx], dtype=torch.float32)
        # 仅对指定类别的样本添加噪声
        if self.add_noise_for_class is not None and self.labels[idx] == self.add_noise_for_class:
            noise = torch.randn_like(cwt)
            signal_power = torch.mean(cwt ** 2)
            noise_power = signal_power / (10 ** (self.snr_db / 10))
            noise = noise * torch.sqrt(noise_power / (torch.mean(noise ** 2) + 1e-8))
            cwt = cwt + noise
        phys = torch.tensor(self.phys_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cwt, phys, label

    def get_norm_params(self):
        return self.mean_val, self.std_val

def get_loaders(add_noise_for_class=None, snr_db=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """返回训练、验证、测试 DataLoader"""
    train_dataset = CWTPhysicalDataset(TRAIN_CWT, TRAIN_PHYS, norm_params=None,
                                       add_noise_for_class=add_noise_for_class,
                                       snr_db=snr_db)
    norm_params = train_dataset.get_norm_params()
    val_dataset = CWTPhysicalDataset(VAL_CWT, VAL_PHYS, norm_params=norm_params,
                                     add_noise_for_class=None)  # 验证集不加噪声
    test_dataset = CWTPhysicalDataset(TEST_CWT, TEST_PHYS, norm_params=norm_params,
                                      add_noise_for_class=None)  # 测试集不加噪声

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader