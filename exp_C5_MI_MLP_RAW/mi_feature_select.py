# mi_feature_select.py
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from config import NUM_SELECTED_FEATURES, SEED
from dataset import build_dataset

def select_features_by_mi():
    X_train, y_train, _, _, _, _ = build_dataset()
    mi = mutual_info_classif(X_train, y_train, random_state=SEED)
    selected_indices = np.argsort(mi)[-NUM_SELECTED_FEATURES:]
    return selected_indices, mi

if __name__ == "__main__":
    idx, mi = select_features_by_mi()
    print(f"Selected {len(idx)} features: {idx}")
    print(f"MI values: {mi[idx]}")