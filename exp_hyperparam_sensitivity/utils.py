# utils.py
import numpy as np
import torch
import random
from sklearn.metrics import f1_score, classification_report
from config import SEED

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred, num_classes=10):
    """计算分类指标和混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
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