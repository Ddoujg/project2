# smote.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def smote(X, y, k_neighbors=5, sampling_strategy='auto'):
    counter = Counter(y)
    if sampling_strategy == 'auto':
        target_count = max(counter.values())
    else:
        target_count = sampling_strategy

    X_res = X.copy()
    y_res = y.copy()
    for cls, count in counter.items():
        if count < target_count:
            n_synthetic = target_count - count
            X_class = X[y == cls]
            nn = NearestNeighbors(n_neighbors=k_neighbors+1)
            nn.fit(X_class)
            synthetic_samples = []
            for _ in range(n_synthetic):
                idx = np.random.randint(0, len(X_class))
                distances, indices = nn.kneighbors(X_class[idx].reshape(1, -1), n_neighbors=k_neighbors+1)
                neighbor_idx = indices[0][np.random.randint(1, k_neighbors+1)]
                diff = X_class[neighbor_idx] - X_class[idx]
                gap = np.random.random()
                synthetic = X_class[idx] + gap * diff
                synthetic_samples.append(synthetic)
            X_res = np.vstack([X_res, np.array(synthetic_samples)])
            y_res = np.hstack([y_res, [cls] * n_synthetic])
    return X_res, y_res