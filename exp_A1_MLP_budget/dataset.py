# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from config import TRAIN_CWT, VAL_CWT, TEST_CWT, TRAIN_PHYS, VAL_PHYS, TEST_PHYS, BATCH_SIZE, NUM_WORKERS

# 保留的高区分度特征索引（根据之前的分析）
SELECTED_FEATURES = [0,1,2,3,5,6,7,10]

class CWTPhysicalDataset(Dataset):
    def __init__(self, cwt_h5_path, phys_h5_path, norm_params=None, feature_indices=None):
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
        self.feature_indices = feature_indices if feature_indices is not None else list(range(self.phys_data.shape[1]))
        # 应用特征选择
        self.phys_data = self.phys_data[:, self.feature_indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cwt = torch.tensor(self.cwt_data[idx], dtype=torch.float32)
        phys = torch.tensor(self.phys_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cwt, phys, label

    def get_norm_params(self):
        return self.mean_val, self.std_val

def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_dataset = CWTPhysicalDataset(TRAIN_CWT, TRAIN_PHYS, norm_params=None, feature_indices=SELECTED_FEATURES)
    norm_params = train_dataset.get_norm_params()
    val_dataset = CWTPhysicalDataset(VAL_CWT, VAL_PHYS, norm_params=norm_params, feature_indices=SELECTED_FEATURES)
    test_dataset = CWTPhysicalDataset(TEST_CWT, TEST_PHYS, norm_params=norm_params, feature_indices=SELECTED_FEATURES)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader