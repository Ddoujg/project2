# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import numpy as np
from config import TRAIN_CWT, TRAIN_PHYS, VAL_CWT, VAL_PHYS, TEST_CWT, TEST_PHYS, BATCH_SIZE, NUM_WORKERS
from smote import smote

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

def get_smote_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    full_train = CWTPhysicalDataset(TRAIN_CWT, TRAIN_PHYS, norm_params=None)
    norm_params = full_train.get_norm_params()
    X_phys = full_train.phys_data
    y = full_train.labels
    X_res, y_res = smote(X_phys, y, k_neighbors=5, sampling_strategy='auto')
    print(f"SMOTE: Original {X_phys.shape} -> Resampled {X_res.shape}")

    # 构建新的 CWT 数据：原始样本保留原 CWT，合成样本随机复制同类原始 CWT
    cwt_data = full_train.cwt_data
    new_cwt = []
    new_phys = []
    new_labels = []
    # 原始样本
    for i in range(len(full_train)):
        new_cwt.append(cwt_data[i])
        new_phys.append(full_train.phys_data[i])
        new_labels.append(full_train.labels[i])
    # 合成样本
    for i in range(len(X_res) - len(full_train)):
        cls = int(y_res[len(full_train) + i])
        candidates = np.where(full_train.labels == cls)[0]
        idx = np.random.choice(candidates) if len(candidates) > 0 else 0
        new_cwt.append(cwt_data[idx])
        new_phys.append(X_res[len(full_train) + i])
        new_labels.append(cls)
    new_cwt = np.array(new_cwt)
    new_phys = np.array(new_phys)
    new_labels = np.array(new_labels)

    dataset = TensorDataset(
        torch.tensor(new_cwt, dtype=torch.float32),
        torch.tensor(new_phys, dtype=torch.float32),
        torch.tensor(new_labels, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, norm_params

def get_val_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    val_dataset = CWTPhysicalDataset(VAL_CWT, VAL_PHYS, norm_params=norm_params)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, norm_params=None):
    test_dataset = CWTPhysicalDataset(TEST_CWT, TEST_PHYS, norm_params=norm_params)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)