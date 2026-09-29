# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from config import TRAIN_PHYS, VAL_PHYS, TEST_PHYS, BATCH_SIZE, NUM_WORKERS

class PhysicalFeatureDataset(Dataset):
    def __init__(self, h5_path, selected_indices=None):
        with h5py.File(h5_path, 'r') as f:
            self.features = f['features'][:]   # (N, 4)
            self.labels = f['labels'][:]
        if selected_indices is not None:
            self.features = self.features[:, selected_indices]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def get_loaders(selected_indices=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_dataset = PhysicalFeatureDataset(TRAIN_PHYS, selected_indices)
    val_dataset = PhysicalFeatureDataset(VAL_PHYS, selected_indices)
    test_dataset = PhysicalFeatureDataset(TEST_PHYS, selected_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader