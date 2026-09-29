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


def compute_metrics(y_true, y_pred, num_classes=9):
    """计算分类指标，重点关注故障类召回率"""
    from sklearn.metrics import confusion_matrix

    # 强制转换为 numpy 数组，并展平为一维
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 如果数组为空，返回空指标
    if len(y_true) == 0:
        return {
            'macro_f1': 0.0,
            'g_mean': 0.0,
            'fault_avg_recall': 0.0,
            'overall_acc': 0.0,
            'per_class_recall': {},
            'classification_report': {},
            'confusion_matrix': []
        }

    # 确保所有类别都在标签范围内
    labels = list(range(num_classes))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    recalls = [report[str(i)]['recall'] for i in range(num_classes) if str(i) in report]
    g_mean = np.exp(np.mean(np.log(recalls))) if recalls else 0.0

    # 故障类平均召回率（排除正常类0）
    fault_recalls = [report[str(i)]['recall'] for i in range(1, num_classes) if str(i) in report]
    fault_avg_recall = np.mean(fault_recalls) if fault_recalls else 0.0

    overall_acc = (y_true == y_pred).mean()

    # 各类别召回率
    per_class_recall = {int(i): report[str(i)]['recall'] for i in range(num_classes) if str(i) in report}

    return {
        'macro_f1': macro_f1,
        'g_mean': g_mean,
        'fault_avg_recall': fault_avg_recall,
        'overall_acc': overall_acc,
        'per_class_recall': per_class_recall,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist()
    }