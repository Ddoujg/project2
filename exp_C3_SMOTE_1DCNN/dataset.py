## dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from scipy.io import loadmat
from config import MAT_DIR, FILE_NAMES, DATA_COLUMN, NORMAL_TRAIN_SAMPLES, FAULT_TRAIN_SAMPLES,NORMAL_TEST_SAMPLES,TEST_SAMPLES_PER_CLASS, VAL_SAMPLES_PER_CLASS, SEGMENT_LENGTH, NORMAL_VAL_SAMPLES, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED
from imblearn.over_sampling import SMOTE

def load_raw_segments():
    """从原始 .mat 文件中加载信号片段，返回训练集、验证集、测试集的 (signals, labels)，三者无交叉。
       数据已进行 Z-Score 标准化（基于训练集的均值和标准差）。
    """
    np.random.seed(RANDOM_SEED)
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for file_idx, (fname, col) in enumerate(zip(FILE_NAMES, DATA_COLUMN)):
        mat_path = f"{MAT_DIR}/{fname}"
        mat = loadmat(mat_path)
        raw_data = mat[col]
        # 确保形状为 (channels, samples)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        elif raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T
        # 取第一个通道
        signal = raw_data[0, :]
        total_len = len(signal)
        max_starts = total_len - SEGMENT_LENGTH + 1
        is_normal = (fname == 'health_20_0')

        # 确定三类样本数
        num_train = NORMAL_TRAIN_SAMPLES if is_normal else FAULT_TRAIN_SAMPLES
        num_val = NORMAL_VAL_SAMPLES if is_normal else VAL_SAMPLES_PER_CLASS
        num_test = NORMAL_TEST_SAMPLES if is_normal else TEST_SAMPLES_PER_CLASS
        required_total = num_train + num_val + num_test
        if required_total > max_starts:
            raise ValueError(f"文件 {fname} 信号长度不足，需要 {required_total} 个起始位置，最多 {max_starts} 个。")

        # 随机生成所有起始位置，然后划分
        all_starts = np.random.choice(max_starts, size=required_total, replace=False)
        train_starts = all_starts[:num_train]
        val_starts = all_starts[num_train:num_train + num_val]
        test_starts = all_starts[num_train + num_val:]
        label = 0 if is_normal else (file_idx + 1)

        for start in train_starts:
            seg = signal[start:start + SEGMENT_LENGTH]
            X_train.append(seg)
            y_train.append(label)
        for start in val_starts:
            seg = signal[start:start + SEGMENT_LENGTH]
            X_val.append(seg)
            y_val.append(label)
        for start in test_starts:
            seg = signal[start:start + SEGMENT_LENGTH]
            X_test.append(seg)
            y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32).reshape(-1, SEGMENT_LENGTH)
    X_val = np.array(X_val, dtype=np.float32).reshape(-1,  SEGMENT_LENGTH)
    X_test = np.array(X_test, dtype=np.float32).reshape(-1, SEGMENT_LENGTH)
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    # ==================== 添加归一化 ====================
    # 仅在训练集上计算均值和标准差（每个特征点独立）
    mean_train = X_train.mean(axis=0, keepdims=True)   # shape (1, SEGMENT_LENGTH)
    std_train = X_train.std(axis=0, keepdims=True)
    std_train[std_train == 0] = 1.0   # 防止除零

    # 对训练集、验证集、测试集应用相同的标准化
    X_train = (X_train - mean_train) / std_train
    X_val   = (X_val   - mean_train) / std_train
    X_test  = (X_test  - mean_train) / std_train
    # ==================================================

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_smote_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    X_train, y_train, X_val, y_val, X_test, y_test = load_raw_segments()
    # SMOTE 过采样（输入必须是二维）
    sm = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"SMOTE: Original {X_train.shape} -> Resampled {X_train_res.shape}")
    # 转换为 PyTorch 数据集，添加通道维度以匹配 1D CNN 输入 (N, 1, 512)
    train_dataset = TensorDataset(torch.tensor(X_train_res, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train_res, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader