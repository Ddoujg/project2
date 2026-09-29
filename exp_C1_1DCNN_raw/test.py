# test.py
import torch
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from dataset import get_loaders
from model import Simple1DCNN
from config import DEVICE, OUTPUT_DIR
from utils import set_seed, load_checkpoint


def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, num_classes=9):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    # G-Mean: geometric mean of per-class recalls
    recalls = [report[str(i)]['recall'] for i in range(num_classes) if str(i) in report]
    g_mean = np.exp(np.mean(np.log(recalls))) if recalls else 0.0
    fault_recalls = [report[str(i)]['recall'] for i in range(1, num_classes) if str(i) in report]
    fault_avg_recall = np.mean(fault_recalls) if fault_recalls else 0.0
    overall_acc = (y_true == y_pred).mean()
    return {
        'macro_f1': macro_f1,
        'g_mean': g_mean,
        'fault_avg_recall': fault_avg_recall,
        'overall_acc': overall_acc,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_loaders()

    model = Simple1DCNN(num_classes=9).to(device)
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print("Loaded best model from", checkpoint_path)
    else:
        print("No trained model found, please run train.py first.")
        return

    y_pred, y_true = test(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n=== Test Results ===")
    print(f"Overall Accuracy: {metrics['overall_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"G-Mean: {metrics['g_mean']:.4f}")
    print(f"Fault Avg Recall: {metrics['fault_avg_recall']:.4f}")
    print("\nClassification Report:")
    for cls, vals in metrics['classification_report'].items():
        if cls.isdigit():
            print(
                f"Class {cls}: precision={vals['precision']:.3f}, recall={vals['recall']:.3f}, f1={vals['f1-score']:.3f}")

    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), np.array(metrics['confusion_matrix']))


if __name__ == "__main__":
    main()