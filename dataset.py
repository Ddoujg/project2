import os
import numpy as np
import h5py
from scipy.io import loadmat
import pywt

# ==================== 基本配置 ====================
file_names = [
    'X097_DE_time',  # normal
    'X105_DE_time',
    'X118_DE_time',
    'X135_DE_time',
    'X169_DE_time',
    'X185_DE_time',
    'X201_DE_time',
    'X209_DE_time',
    'X222_DE_time',
    'X238_DE_time'
]

SEGMENT_LENGTH = 512
FS = 12000
WAVELET = 'morl'

# ==================== 物理引导频带 ====================
# CWRU轴承典型故障频带（可写论文）
F_MIN = 80
F_MAX = 1000
NUM_SCALES = 128

# ==================== frequency → scale ====================
fc = pywt.central_frequency(WAVELET)
freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), NUM_SCALES)
SCALES = fc * FS / freqs

CWT_HEIGHT = len(SCALES)
CWT_WIDTH = SEGMENT_LENGTH


# ==================== 数据划分配置 ====================
NORMAL_SAMPLES_TRAIN = 300
FAULT_SAMPLES_TRAIN = 30
NORMAL_SAMPLES_VAL = 100
FAULT_SAMPLES_VAL = 10
NORMAL_SAMPLES_TEST = 100
FAULT_SAMPLES_TEST = 10

label_map = {fname: i for i, fname in enumerate(file_names)}


# ==================== 数据预处理 ====================
def normalization(x):
    """标准化（保留信号结构，避免破坏物理信息）"""
    x = x - np.mean(x)
    std = np.std(x)
    return x / std if std != 0 else x


# ==================== CWT变换 ====================
def cwt_transform(data):
    x = normalization(data)

    coef, _ = pywt.cwt(x, SCALES, WAVELET)
    mag = np.abs(coef)

    # 全局归一化（避免逐样本破坏物理一致性）
    mag = mag / (np.max(mag) + 1e-8)

    return mag.astype(np.float32)


# ==================== 单通道处理 ====================
def process_single_channel(channel_data, start_indices, seg_len=SEGMENT_LENGTH):
    segments = []

    for start in start_indices:
        if start + seg_len > len(channel_data):
            continue

        segment = channel_data[start:start + seg_len]
        cwt_img = cwt_transform(segment)
        segments.append(cwt_img)

    return segments


# ==================== 多通道处理 ====================
def process_multichannel_signal(multi_data, start_indices, target_channels=3):

    channels, _ = multi_data.shape

    # 通道适配
    if channels < target_channels:
        repeat = list(range(channels)) + [channels - 1] * (target_channels - channels)
        multi_data = multi_data[repeat, :]
    elif channels > target_channels:
        multi_data = multi_data[:target_channels, :]

    per_channel = []

    for ch in range(target_channels):
        ch_data = multi_data[ch, :]
        ch_segs = process_single_channel(ch_data, start_indices)
        per_channel.append(ch_segs)

    n = len(per_channel[0])

    final = []
    for i in range(n):
        img = np.stack(
            [per_channel[0][i],
             per_channel[1][i],
             per_channel[2][i]],
            axis=0
        )
        final.append(img)

    return final


# ==================== 主程序 ====================
print("开始处理CWRU数据集...")
print(f"CWT尺寸: {CWT_HEIGHT} × {CWT_WIDTH}")
print(f"物理频带: {F_MIN}Hz - {F_MAX}Hz")

num_classes = len(file_names)

total_train = NORMAL_SAMPLES_TRAIN + (num_classes - 1) * FAULT_SAMPLES_TRAIN
total_val = NORMAL_SAMPLES_VAL + (num_classes - 1) * FAULT_SAMPLES_VAL
total_test = NORMAL_SAMPLES_TEST + (num_classes - 1) * FAULT_SAMPLES_TEST

print(f"Train: {total_train}, Val: {total_val}, Test: {total_test}")


with h5py.File('train_dataset_fmd_cwt_30.h5', 'w') as h5_train, \
     h5py.File('val_dataset_fmd_cwt_30.h5', 'w') as h5_val, \
     h5py.File('test_dataset_fmd_cwt_30.h5', 'w') as h5_test:

    train_data = h5_train.create_dataset('data', shape=(total_train, 3, CWT_HEIGHT, CWT_WIDTH), dtype=np.float32)
    train_labels = h5_train.create_dataset('labels', shape=(total_train,), dtype=np.int32)

    val_data = h5_val.create_dataset('data', shape=(total_val, 3, CWT_HEIGHT, CWT_WIDTH), dtype=np.float32)
    val_labels = h5_val.create_dataset('labels', shape=(total_val,), dtype=np.int32)

    test_data = h5_test.create_dataset('data', shape=(total_test, 3, CWT_HEIGHT, CWT_WIDTH), dtype=np.float32)
    test_labels = h5_test.create_dataset('labels', shape=(total_test,), dtype=np.int32)

    train_idx = val_idx = test_idx = 0

    for file_idx, fname in enumerate(file_names):

        mat = loadmat(f'D:\\1111paper\\paper22\\cwru_fmd_dataset\\{fname}')
        raw_data = mat['y_final']

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        elif raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T

        # ==================== label ====================
        label = file_idx  # 统一编号（0=normal）

        # ==================== 样本数 ====================
        is_normal = (fname == 'X097_DE_time')

        if is_normal:
            n_train = NORMAL_SAMPLES_TRAIN
            n_val = NORMAL_SAMPLES_VAL
            n_test = NORMAL_SAMPLES_TEST
        else:
            n_train = FAULT_SAMPLES_TRAIN
            n_val = FAULT_SAMPLES_VAL
            n_test = FAULT_SAMPLES_TEST

        total_len = raw_data.shape[1]
        max_starts = total_len - SEGMENT_LENGTH + 1
        required = n_train + n_val + n_test

        if required > max_starts:
            raise ValueError(f"{fname}长度不足")

        # ==================== 随机采样 ====================
        all_starts = np.random.choice(max_starts, size=required, replace=False)

        train_starts = all_starts[:n_train]
        val_starts = all_starts[n_train:n_train + n_val]
        test_starts = all_starts[n_train + n_val:]

        # ==================== CWT生成 ====================
        train_segs = process_multichannel_signal(raw_data, train_starts)
        val_segs = process_multichannel_signal(raw_data, val_starts)
        test_segs = process_multichannel_signal(raw_data, test_starts)

        train_segs = np.array(train_segs, dtype=np.float32)
        val_segs = np.array(val_segs, dtype=np.float32)
        test_segs = np.array(test_segs, dtype=np.float32)

        print(f"{fname} -> train {train_segs.shape}, val {val_segs.shape}, test {test_segs.shape}")

        # ==================== 写入 ====================
        train_data[train_idx:train_idx + len(train_segs)] = train_segs
        train_labels[train_idx:train_idx + len(train_segs)] = label

        val_data[val_idx:val_idx + len(val_segs)] = val_segs
        val_labels[val_idx:val_idx + len(val_segs)] = label

        test_data[test_idx:test_idx + len(test_segs)] = test_segs
        test_labels[test_idx:test_idx + len(test_segs)] = label

        train_idx += len(train_segs)
        val_idx += len(val_segs)
        test_idx += len(test_segs)

print("\n数据处理完成：CWRU物理引导CWT数据集已生成")