# test.py
import torch
import numpy as np
import json
import os
import time
from dataset import get_loaders
from model import DynamicViTClassifier
from config import DEVICE, OUTPUT_DIR
from utils import set_seed, load_checkpoint, compute_metrics, compute_model_complexity

def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    with torch.no_grad():
        for cwt, labels in loader:
            cwt = cwt.to(device)
            start_time = time.time()
            logits = model(cwt)
            end_time = time.time()
            inference_times.append((end_time - start_time) / cwt.size(0))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    avg_time = np.mean(inference_times) * 1000
    return np.array(all_preds), np.array(all_labels), avg_time

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_loaders()
    if test_loader is None:
        print("No test loader. Please set TEST_CWT in config.py")
        return

    model = DynamicViTClassifier().to(device)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print("Loaded best model from", checkpoint_path)
    else:
        print("No trained model found, please run train.py first.")
        return

    flops, params = compute_model_complexity(model, input_size=(3, 128, 512), device=device)
    y_pred, y_true, avg_time = test(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics['flops'] = flops
    metrics['params'] = params
    metrics['avg_inference_time_ms'] = avg_time

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), np.array(metrics['confusion_matrix']))
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y_true)
    np.save(os.path.join(OUTPUT_DIR, 'preds.npy'), y_pred)

    print("\n=== Test Results ===")
    print(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    print(f"Fault Avg Recall: {metrics['fault_avg_recall']:.4f}")
    print(f"Recall Std: {metrics['recall_std']:.4f}")
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Avg Inference Time: {avg_time:.2f} ms/sample")

if __name__ == "__main__":
    main()