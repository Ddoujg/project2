# utils.py
import numpy as np
import torch
import random

from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from config import MAT_DIR, FILE_NAMES, DATA_COLUMN, NORMAL_TRAIN_SAMPLES,NORMAL_TEST_SAMPLES,TEST_SAMPLES_PER_CLASS, FAULT_TRAIN_SAMPLES, VAL_SAMPLES_PER_CLASS,NORMAL_VAL_SAMPLES, \
    SEGMENT_LENGTH, RANDOM_SEED, SEED,NUM_CLASSES

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_raw_segments():
    """从原始 .mat 文件中加载信号片段，返回训练集、验证集、测试集的 (signals, labels)，三者无交叉"""
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

    X_train = np.array(X_train, dtype=np.float32).reshape(-1, 1, SEGMENT_LENGTH)
    X_val = np.array(X_val, dtype=np.float32).reshape(-1, 1, SEGMENT_LENGTH)
    X_test = np.array(X_test, dtype=np.float32).reshape(-1, 1, SEGMENT_LENGTH)
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def compute_class_weights(labels):
    """计算类别权重（逆频率），返回形状为 [NUM_CLASSES] 的张量"""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recalls = [report[str(i)]['recall'] for i in range(num_classes) if str(i) in report]
    g_mean = np.exp(np.mean(np.log(recalls))) if recalls else 0.0
    fault_recalls = [report[str(i)]['recall'] for i in range(1, num_classes) if str(i) in report]
    fault_avg_recall = np.mean(fault_recalls) if fault_recalls else 0.0
    overall_acc = (y_true == y_pred).mean()
    recall_std = np.std([report[str(i)]['recall'] for i in range(num_classes) if str(i) in report])
    return {
        'macro_f1': macro_f1,
        'g_mean': g_mean,
        'fault_avg_recall': fault_avg_recall,
        'overall_acc': overall_acc,
        'recall_std': recall_std,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def compute_model_complexity(model, input_size=(1, 512), device='cpu'):
    from thop import profile, clever_format
    input_tensor = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params