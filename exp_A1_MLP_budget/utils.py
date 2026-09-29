# utils.py
import numpy as np
import torch
import random
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from config import SEED, NUM_PHYSICAL_FEATURES

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

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def compute_metrics(y_true, y_pred, num_classes=10):
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

def compute_model_complexity(model, input_size_cwt=(3, 64, 512), input_size_phys=(NUM_PHYSICAL_FEATURES,), device='cpu'):
    from thop import profile, clever_format
    cwt_tensor = torch.randn(1, *input_size_cwt).to(device)
    phys_tensor = torch.randn(1, *input_size_phys).to(device)
    flops, params = profile(model, inputs=(cwt_tensor, phys_tensor), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def compute_sparsity(budgets, total_patches):
    return np.mean(budgets) / total_patches