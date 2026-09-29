# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from config import BATCH_SIZE, NUM_WORKERS

class CWTPhysicalDataset(Dataset):
    """
    加载 CWT 时频图和物理特征数据集，并自动对物理特征进行标准化。
    如果提供 mean 和 std，则使用提供的值进行标准化；否则从数据中计算。
    """
    def __init__(self, cwt_h5_path, phys_h5_path=None, mean=None, std=None):
        # 加载 CWT 数据和标签
        with h5py.File(cwt_h5_path, 'r') as f:
            self.cwt_data = f['data'][:]          # (N, 3, 128, 512)
            self.labels = f['labels'][:]          # (N,)
        # 加载物理特征数据
        if phys_h5_path is None:
            # 尝试自动推断物理特征文件路径（例如将 'data' 替换为 'phys'）
            phys_h5_path = cwt_h5_path.replace('dataset', 'physical_features')
        with h5py.File(phys_h5_path, 'r') as f:
            # 兼容不同的键名：'phys' 或 'features'
            if 'phys' in f:
                self.phys_data = f['phys'][:]
            elif 'features' in f:
                self.phys_data = f['features'][:]
            else:
                raise KeyError("物理特征文件中未找到 'phys' 或 'features' 键")
        assert len(self.cwt_data) == len(self.labels) == len(self.phys_data)
        self.phys_data = self.phys_data.astype(np.float32)

        # 标准化物理特征
        if mean is None or std is None:
            # 计算训练集的均值和标准差
            self.mean = self.phys_data.mean(axis=0, keepdims=True)
            self.std = self.phys_data.std(axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = mean
            self.std = std
        self.phys_data = (self.phys_data - self.mean) / self.std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cwt = torch.tensor(self.cwt_data[idx], dtype=torch.float32)
        phys = torch.tensor(self.phys_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cwt, phys, label

    def get_norm_params(self):
        """返回标准化参数，用于验证集和测试集"""
        return self.mean, self.std


def get_loaders(train_cwt_h5, train_phys_h5,
                val_cwt_h5, val_phys_h5,
                test_cwt_h5, test_phys_h5,
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    创建训练、验证、测试 DataLoader。
    训练集用于计算标准化参数，验证集和测试集使用相同的参数。
    """
    # 训练集：计算标准化参数
    train_dataset = CWTPhysicalDataset(train_cwt_h5, train_phys_h5, mean=None, std=None)
    mean, std = train_dataset.get_norm_params()
    # 验证集和测试集：使用训练集的参数
    val_dataset = CWTPhysicalDataset(val_cwt_h5, val_phys_h5, mean=mean, std=std)
    test_dataset = CWTPhysicalDataset(test_cwt_h5, test_phys_h5, mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader