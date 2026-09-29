# train_stage2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import (
    EPOCHS_STAGE2, LEARNING_RATE_BUDGET, WEIGHT_DECAY_BUDGET,
    DEVICE, OUTPUT_DIR, CLIP_GRAD, B_MIN, B_MAX, BUDGET_LAMBDA,
    TV_LAMBDA, SPARSE_LAMBDA, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH
)
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, optimizer, device, budget_lambda, tv_lambda, sparse_lambda):
    model.train()
    total_loss = 0
    total_loss_cls = 0
    total_loss_budget = 0
    total_loss_tv = 0
    total_loss_sparse = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Stage2 Training')
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
        _, predicted = outputs['logits'].max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_loss_cls = total_loss_cls / total
    epoch_loss_budget = total_loss_budget / total
    epoch_loss_tv = total_loss_tv / total
    epoch_loss_sparse = total_loss_sparse / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_loss_cls, epoch_loss_budget, epoch_loss_tv, epoch_loss_sparse, epoch_acc

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

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders()

    # 加载第一阶段保存的模型
    model = FMDGumbelSinkhornViTModel().to(device)
    checkpoint = torch.load(STAGE1_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded stage1 model from", STAGE1_MODEL_PATH)

    # 冻结骨干网络和分类器，只训练预算预测器
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_BUDGET, weight_decay=WEIGHT_DECAY_BUDGET
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard_stage2'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []
    # 记录预算均值历史
    normal_budget_hist = []
    fault_budget_hist = []
    actual_total_hist = []
    expected_total_hist = []

    for epoch in range(1, EPOCHS_STAGE2+1):
        print(f"\nStage2 Epoch {epoch}/{EPOCHS_STAGE2}")
        train_loss, train_loss_cls, train_loss_budget, train_loss_tv, train_loss_sparse, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, BUDGET_LAMBDA, TV_LAMBDA, SPARSE_LAMBDA)
        val_acc = validate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 计算训练集上的预算统计（用于监控）
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
        expected_total = sum(cnt * ((B_MIN + B_MAX) / 2) for cnt in count_by_class.values())
        actual_total = sum(budget_sum_by_class.values())
        normal_budget_hist.append(normal_avg)
        fault_budget_hist.append(fault_avg)
        actual_total_hist.append(actual_total)
        expected_total_hist.append(expected_total)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, STAGE2_MODEL_PATH)
            print(f"  -> Best stage2 model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    # 保存训练历史（包含预算数据）
    np.savez(os.path.join(OUTPUT_DIR, 'stage2_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs,
             normal_budget=normal_budget_hist, fault_budget=fault_budget_hist,
             actual_total=actual_total_hist, expected_total=expected_total_hist)
    writer.close()
    print(f"\nStage2 finished. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()