# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from config import TRAIN_CWT, TRAIN_PHYS, VAL_CWT, VAL_PHYS, TEST_CWT, TEST_PHYS, BATCH_SIZE, NUM_WORKERS

class CWTImageDataset(Dataset):
    def __init__(self, cwt_h5_path):
        with h5py.File(cwt_h5_path, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['labels'][:]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def get_class_weights(train_dataset):
    """计算类别权重：权重 = 1/样本数，然后归一化使平均权重为1"""
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_dataset = CWTImageDataset(TRAIN_CWT)
    val_dataset = CWTImageDataset(VAL_CWT)
    test_dataset = CWTImageDataset(TEST_CWT)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset