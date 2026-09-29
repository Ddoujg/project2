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
    DEFAULT_TV_LAMBDA, DEFAULT_SPARSE_LAMBDA, DEFAULT_W_SPARSE_LAMBDA
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

def train_and_evaluate(B_min, B_max, budget_lambda, train_loader, val_loader, test_loader, output_dir, device):
    """训练一个配置并返回测试指标和预算统计"""
    set_seed()  # 每次训练固定随机种子
    model = FMDGumbelSinkhornViTModel(
        B_min=B_min, B_max=B_max, budget_lambda=budget_lambda,
        tv_lambda=DEFAULT_TV_LAMBDA, sparse_lambda=DEFAULT_SPARSE_LAMBDA, w_sparse_lambda=DEFAULT_W_SPARSE_LAMBDA
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
    with torch.no_grad():
        for cwt, phys, labels in test_loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            outputs = model(cwt, phys)
            logits = outputs['logits']
            budgets = outputs['a'].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_budgets.extend(budgets)
    metrics = compute_metrics(all_labels, all_preds)
    budget_stats = {
        'mean': float(np.mean(all_budgets)),
        'std': float(np.std(all_budgets)),
        'normal_mean': float(np.mean([b for b, l in zip(all_budgets, all_labels) if l == 0])) if any(l==0 for l in all_labels) else 0,
        'fault_mean': float(np.mean([b for b, l in zip(all_budgets, all_labels) if l > 0])) if any(l>0 for l in all_labels) else 0,
    }
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, 'budget_stats.json'), 'w') as f:
        json.dump(budget_stats, f, indent=4)
    # 保存模型（可选）
    torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
    return metrics, budget_stats