# mi_feature_select.py
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from config import NUM_SELECTED_FEATURES, TRAIN_PHYS, SEED
import h5py

def select_features_by_mi():
    with h5py.File(TRAIN_PHYS, 'r') as f:
        X_train = f['features'][:]
        y_train = f['labels'][:]
    mi = mutual_info_classif(X_train, y_train, random_state=SEED)
    # 选择互信息最大的 k 个特征索引
    selected_indices = np.argsort(mi)[-NUM_SELECTED_FEATURES:]
    return selected_indices, mi

if __name__ == "__main__":
    idx, mi = select_features_by_mi()
    print(f"Selected feature indices: {idx}")
    print(f"Mutual information values: {mi}")