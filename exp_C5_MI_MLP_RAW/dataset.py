# dataset.py
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from config import MAT_DIR, FILE_NAMES, DATA_COLUMN, NORMAL_VAL_SAMPLES,TEST_SAMPLES_PER_CLASS,NORMAL_TEST_SAMPLES,NORMAL_TRAIN_SAMPLES, FAULT_TRAIN_SAMPLES, VAL_SAMPLES_PER_CLASS, SEGMENT_LENGTH, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS

# ------------------------------
# 时域特征
# ------------------------------
def extract_time_domain_features(segment):
    mean = np.mean(segment)
    var = np.var(segment)
    rms = np.sqrt(np.mean(segment**2))
    peak = np.max(np.abs(segment))
    if rms < 1e-8:
        crest = 0
        shape_factor = 0
        impulse_factor = 0
        margin_factor = 0
    else:
        crest = peak / rms
        shape_factor = rms / (np.mean(np.abs(segment)) + 1e-8)
        impulse_factor = peak / (np.mean(np.abs(segment)) + 1e-8)
        margin_factor = peak / (np.mean(np.sqrt(np.abs(segment))) + 1e-8)
    return np.array([mean, var, rms, peak, kurtosis(segment), skew(segment), crest, shape_factor, impulse_factor, margin_factor])

# ------------------------------
# 频域特征
# ------------------------------
def extract_frequency_features(segment, fs=20000):
    n = len(segment)
    fft = np.fft.rfft(segment)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    total_power = np.sum(fft_mag**2)
    if total_power > 0:
        centroid = np.sum(freqs * (fft_mag**2)) / total_power
        variance = np.sum(((freqs - centroid)**2) * (fft_mag**2)) / total_power
    else:
        centroid = 0
        variance = 0
    spec_kurt = kurtosis(fft_mag)
    p = fft_mag / (np.sum(fft_mag) + 1e-8)
    spec_entropy = -np.sum(p * np.log(p + 1e-8))
    peak_freq = freqs[np.argmax(fft_mag)]
    peak_ratio = np.max(fft_mag) / (np.mean(fft_mag) + 1e-8)
    return np.array([centroid, variance, spec_kurt, spec_entropy, peak_freq, peak_ratio])

def extract_all_features(signal, fs=20000):
    time_feats = extract_time_domain_features(signal)
    freq_feats = extract_frequency_features(signal, fs)
    return np.concatenate([time_feats, freq_feats])

# ------------------------------
# 构建数据集（训练/验证/测试）
# ------------------------------
def build_dataset():
    np.random.seed(RANDOM_SEED)
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    for file_idx, (fname, col) in enumerate(zip(FILE_NAMES, DATA_COLUMN)):
        mat_path = f"{MAT_DIR}/{fname}"
        mat = loadmat(mat_path)
        raw_data = mat[col]
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        elif raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T
        signal = raw_data[0, :]          # 取第一个通道
        total_len = len(signal)
        max_starts = total_len - SEGMENT_LENGTH + 1
        is_normal = (fname == 'health_20_0')
        num_train = NORMAL_TRAIN_SAMPLES if is_normal else FAULT_TRAIN_SAMPLES
        num_val = NORMAL_VAL_SAMPLES if is_normal else VAL_SAMPLES_PER_CLASS
        num_test = NORMAL_TEST_SAMPLES if is_normal else TEST_SAMPLES_PER_CLASS
        required_total = num_train + num_val + num_test
        if required_total > max_starts:
            raise ValueError(f"文件 {fname} 信号长度不足")
        all_starts = np.random.choice(max_starts, size=required_total, replace=False)
        train_starts = all_starts[:num_train]
        val_starts = all_starts[num_train:num_train+num_val]
        # 测试集：若剩余不足，随机补充
        test_starts = all_starts[num_train+num_val:] if len(all_starts) > num_train+num_val else []
        while len(test_starts) < num_val:
            extra = np.random.choice(max_starts, size=1, replace=True)[0]
            if extra not in all_starts:
                test_starts = np.append(test_starts, extra)
        label = 0 if is_normal else (file_idx + 1)
        for start in train_starts:
            seg = signal[start:start+SEGMENT_LENGTH]
            X_train.append(extract_all_features(seg))
            y_train.append(label)
        for start in val_starts:
            seg = signal[start:start+SEGMENT_LENGTH]
            X_val.append(extract_all_features(seg))
            y_val.append(label)
        for start in test_starts:
            seg = signal[start:start+SEGMENT_LENGTH]
            X_test.append(extract_all_features(seg))
            y_test.append(label)
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, X_train.shape[1]