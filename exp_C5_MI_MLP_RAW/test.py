# test.py
import torch
import numpy as np
import json
import os
import time
from config import DEVICE, OUTPUT_DIR, NUM_SELECTED_FEATURES
from model import MLPClassifier
from mi_feature_select import select_features_by_mi
from utils import set_seed, load_checkpoint, compute_metrics
from sklearn.preprocessing import StandardScaler
from dataset import build_dataset, FeatureDataset

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    selected_indices, _ = select_features_by_mi()
    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_test = X_test[:, selected_indices]
    test_dataset = FeatureDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MLPClassifier(input_dim=NUM_SELECTED_FEATURES, num_classes=9).to(device)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print("Loaded best model from", checkpoint_path)
    else:
        print("No trained model found, please run train.py first.")
        return

    model.eval()
    all_preds, all_labels = [], []
    inference_times = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            start = time.time()
            logits = model(x)
            end = time.time()
            inference_times.append((end - start) / x.size(0))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    avg_time = np.mean(inference_times) * 1000
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['avg_inference_time_ms'] = avg_time
    metrics['params'] = sum(p.numel() for p in model.parameters())

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), np.array(metrics['confusion_matrix']))
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), all_labels)
    np.save(os.path.join(OUTPUT_DIR, 'preds.npy'), all_preds)

    print("\n=== Test Results ===")
    print(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    print(f"Fault Avg Recall: {metrics['fault_avg_recall']:.4f}")
    print(f"Recall Std: {metrics['recall_std']:.4f}")
    print(f"Params: {metrics['params']}")
    print(f"Avg Inference Time: {avg_time:.2f} ms/sample")

if __name__ == "__main__":
    main()