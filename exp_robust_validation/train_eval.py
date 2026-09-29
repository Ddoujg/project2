# train_eval.py
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import json
from config import (
    EPOCHS, LEARNING_RATE_BACKBONE, LEARNING_RATE_BUDGET,
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_BUDGET, DEVICE, CLIP_GRAD,
    B_MIN, B_MAX, BUDGET_LAMBDA,
    TV_LAMBDA, SPARSE_LAMBDA, W_SPARSE_LAMBDA
)
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, compute_metrics

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    for cwt, phys, labels in pbar:
        cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cwt, phys, labels=labels)
        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        _, predicted = outputs['logits'].max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cwt, phys, labels in loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            outputs = model(cwt, phys)
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_and_evaluate(fault_samples_per_class, output_dir, device):
    """训练一个配置（特定故障样本数），返回测试指标和各类预算统计"""
    from dataset import get_train_loader_with_fault_samples, get_val_loader, get_test_loader
    set_seed()
    # 创建训练集
    train_loader, norm_params = get_train_loader_with_fault_samples(fault_samples_per_class)
    val_loader = get_val_loader(norm_params=norm_params)
    test_loader = get_test_loader(norm_params=norm_params)

    model = FMDGumbelSinkhornViTModel(
        B_min=B_MIN, B_max=B_MAX, budget_lambda=BUDGET_LAMBDA,
        tv_lambda=TV_LAMBDA, sparse_lambda=SPARSE_LAMBDA, w_sparse_lambda=W_SPARSE_LAMBDA
    ).to(device)

    # 分组优化器
    backbone_params = []
    budget_params = []
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            budget_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': WEIGHT_DECAY_BACKBONE},
        {'params': budget_params, 'lr': LEARNING_RATE_BUDGET, 'weight_decay': WEIGHT_DECAY_BUDGET}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    best_model_state = None
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = validate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        scheduler.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    # 测试
    model.eval()
    all_preds, all_labels, all_budgets = [], [], []
    # 记录每类样本的预算和预测结果
    class_budgets = {c: [] for c in range(9)}
    with torch.no_grad():
        for cwt, phys, labels in test_loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            outputs = model(cwt, phys)
            logits = outputs['logits']
            budgets = outputs['a'].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_budgets.extend(budgets)
            for i, label in enumerate(labels_np):
                class_budgets[label].append(budgets[i])

    metrics = compute_metrics(all_labels, all_preds)
    # 计算各类别平均预算
    class_avg_budget = {c: np.mean(class_budgets[c]) if class_budgets[c] else 0 for c in range(9)}
    # 提取故障类（1-9）的平均预算和召回率
    fault_classes = list(range(1, 9))
    fault_budgets = [class_avg_budget[c] for c in fault_classes]
    fault_recalls = [metrics['per_class_recall'][c] for c in fault_classes]

    result = {
        'fault_samples_per_class': fault_samples_per_class,
        'macro_f1': metrics['macro_f1'],
        'g_mean': metrics['g_mean'],
        'fault_avg_recall': metrics['fault_avg_recall'],
        'overall_acc': metrics['overall_acc'],
        'class_avg_budget': class_avg_budget,
        'fault_budgets': fault_budgets,
        'fault_recalls': fault_recalls,
        'budget_mean': np.mean(all_budgets),
        'budget_std': np.std(all_budgets),
        'normal_budget': class_avg_budget[0],
        'fault_budget_mean': np.mean(fault_budgets),
    }

    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4, default=convert_to_serializable)
    # 保存模型
    torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
    return result