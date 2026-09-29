# smote_utils.py
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from config import SMOTE_SAMPLING_STRATEGY


def apply_smote_to_cwt(data, labels):
    """
    对 CWT 时频图数据进行 SMOTE 过采样。
    由于 SMOTE 要求输入是 2D 数组 (n_samples, n_features)，
    需要将图像展平为向量，过采样后再还原为原始形状。
    """
    original_shape = data.shape  # (N, 3, 128, 512)
    n_samples = original_shape[0]
    flattened = data.reshape(n_samples, -1)  # (N, 3*128*512)

    # 可选：标准化（SMOTE 通常在特征空间进行，标准化可能有益）
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)

    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=42)
    flattened_resampled, labels_resampled = smote.fit_resample(flattened_scaled, labels)

    # 还原到原始尺度（如果使用了标准化）
    flattened_resampled = scaler.inverse_transform(flattened_resampled)
    # 重塑回图像形状
    new_shape = (flattened_resampled.shape[0],) + original_shape[1:]
    data_resampled = flattened_resampled.reshape(new_shape)

    return data_resampled.astype(np.float32), labels_resampled.astype(np.int64)


if __name__ == "__main__":
    # 简单测试
    from dataset import get_original_loaders

    train_loader, _, _ = get_original_loaders()
    for data, labels in train_loader:
        data_np = data.numpy()
        labels_np = labels.numpy()
        print(f"Original: {data_np.shape}, labels: {labels_np.shape}")
        data_res, labels_res = apply_smote_to_cwt(data_np, labels_np)
        print(f"After SMOTE: {data_res.shape}, labels: {labels_res.shape}")
        break