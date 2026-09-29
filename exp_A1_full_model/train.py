# train.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import (
    EPOCHS, LEARNING_RATE_BACKBONE, LEARNING_RATE_BUDGET,
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_BUDGET, DEVICE, OUTPUT_DIR, CLIP_GRAD, MONITOR_INTERVAL
)
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, save_checkpoint
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_loss_cls = 0
    total_loss_budget = 0
    total_loss_tv = 0
    total_loss_sparse = 0
    total_loss_w_sparse = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training')
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
        total_loss_cls += outputs['loss_cls'].item() * bs
        total_loss_budget += outputs['loss_budget'].item() * bs
        total_loss_tv += outputs['loss_tv'].item() * bs
        total_loss_sparse += outputs['loss_sparse'].item() * bs
        total_loss_w_sparse += outputs['loss_w_sparse'].item() * bs
        _, predicted = outputs['logits'].max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_loss_cls = total_loss_cls / total
    epoch_loss_budget = total_loss_budget / total
    epoch_loss_tv = total_loss_tv / total
    epoch_loss_sparse = total_loss_sparse / total
    epoch_loss_w_sparse = total_loss_w_sparse / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_loss_cls, epoch_loss_budget, epoch_loss_tv, epoch_loss_sparse, epoch_loss_w_sparse, epoch_acc

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cwt, phys, labels in tqdm(loader, desc='Validating'):
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            outputs = model(cwt, phys)
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders()

    model = FMDGumbelSinkhornViTModel().to(device)

    # 分组优化器：预算预测器学习率更高
    backbone_params = []
    budget_predictor_params = []
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            budget_predictor_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': WEIGHT_DECAY_BACKBONE},
        {'params': budget_predictor_params, 'lr': LEARNING_RATE_BUDGET, 'weight_decay': WEIGHT_DECAY_BUDGET}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard'))

    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [],
        'normal_budget': [], 'fault_budget': [], 'actual_total': [], 'expected_total': []
    }

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_loss_cls, train_loss_budget, train_loss_tv, train_loss_sparse, train_loss_w_sparse, train_acc = train_one_epoch(
            model, train_loader, optimizer, device)
        val_acc = validate(model, val_loader, device)

        # 计算训练集预算统计（用于记录）
        model.eval()
        budget_sum_by_class = {}
        count_by_class = {}
        with torch.no_grad():
            for cwt, phys, labels in train_loader:
                cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
                outputs = model(cwt, phys)
                a_batch = outputs['a'].cpu()
                for i, label in enumerate(labels.cpu().numpy()):
                    budget_sum_by_class[label] = budget_sum_by_class.get(label, 0.0) + a_batch[i].item()
                    count_by_class[label] = count_by_class.get(label, 0) + 1
        normal_avg = budget_sum_by_class.get(0, 0.0) / count_by_class.get(0, 1)
        fault_avg = np.mean([budget_sum_by_class[c]/count_by_class[c] for c in budget_sum_by_class if c > 0]) if any(c>0 for c in budget_sum_by_class) else 0.0
        expected_total = sum(cnt * ((model.B_min + model.B_max) / 2) for cnt in count_by_class.values())
        actual_total = sum(budget_sum_by_class.values())

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['normal_budget'].append(normal_avg)
        history['fault_budget'].append(fault_avg)
        history['actual_total'].append(actual_total)
        history['expected_total'].append(expected_total)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/cls', train_loss_cls, epoch)
        writer.add_scalar('Loss/budget', train_loss_budget, epoch)
        writer.add_scalar('Loss/tv', train_loss_tv, epoch)
        writer.add_scalar('Loss/sparse', train_loss_sparse, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('Budget/normal_avg', normal_avg, epoch)
        writer.add_scalar('Budget/fault_avg', fault_avg, epoch)
        writer.add_scalar('Budget/actual_total', actual_total, epoch)
        writer.add_scalar('Budget/expected_total', expected_total, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print(f"Budget: Normal avg={normal_avg:.2f}, Fault avg={fault_avg:.2f}, Actual Total={actual_total:.0f}, Expected Total={expected_total:.0f}")

        # 模块监控
        if epoch % MONITOR_INTERVAL == 0:
            with torch.no_grad():
                sample_cwt, sample_phys, sample_labels = next(iter(val_loader))
                sample_cwt = sample_cwt.to(device)
                sample_phys = sample_phys.to(device)
                inter = model.budget_predictor.get_intermediate(sample_phys)
                print(f"\n=== Budget Predictor Intermediate Variables at Epoch {epoch} ===")
                print(f"  w (first 5): {inter['w'][:5]}")
                print(f"  alpha: {inter['alpha']:.4f}, beta: {inter['beta']:.4f}")
                print(f"  phi: min={inter['phi'].min():.4f}, max={inter['phi'].max():.4f}, mean={inter['phi'].mean():.4f}")
                print(f"  r:   min={inter['r'].min():.4f}, max={inter['r'].max():.4f}, mean={inter['r'].mean():.4f}")
                print(f"  a:   min={inter['a'].min():.2f}, max={inter['a'].max():.2f}, mean={inter['a'].mean():.2f}")
                print("=============================================================\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

    writer.close()
    # 保存训练历史
    np.savez(os.path.join(OUTPUT_DIR, 'training_history.npz'), **history)
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")
    return history

if __name__ == "__main__":
    main()