# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import (
    EPOCHS_STAGE1, EPOCHS_STAGE2, LEARNING_RATE_BACKBONE, LEARNING_RATE_BUDGET,
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_BUDGET, DEVICE, OUTPUT_DIR, CLIP_GRAD,
    B_MIN, B_MAX
)
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, optimizer, device, stage=1):
    """训练一个epoch，stage=1时不计算预算损失，使用固定预算"""
    model.train()
    total_loss = 0
    total_loss_cls = 0
    total_loss_budget = 0
    total_loss_tv = 0
    total_loss_sparse = 0
    total_loss_w_sparse = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f'Stage{stage} Training')
    for cwt, phys, labels in pbar:
        cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
        optimizer.zero_grad()
        if stage == 1:
            model.use_fixed_budget = True
            outputs = model(cwt, phys, labels=labels)
            model.use_fixed_budget = False
            loss = outputs['loss']
            loss_cls = outputs['loss_cls']
            loss_budget = outputs['loss_budget']  # 此时应为0
            loss_tv = outputs.get('loss_tv', torch.tensor(0.0))
            loss_sparse = outputs.get('loss_sparse', torch.tensor(0.0))
            loss_w_sparse = outputs.get('loss_w_sparse', torch.tensor(0.0))
        else:
            outputs = model(cwt, phys, labels=labels)
            loss = outputs['loss']
            loss_cls = outputs['loss_cls']
            loss_budget = outputs['loss_budget']
            loss_tv = outputs.get('loss_tv', torch.tensor(0.0))
            loss_sparse = outputs.get('loss_sparse', torch.tensor(0.0))
            loss_w_sparse = outputs.get('loss_w_sparse', torch.tensor(0.0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_loss_cls += loss_cls.item() * bs
        total_loss_budget += loss_budget.item() * bs
        total_loss_tv += loss_tv.item() * bs
        total_loss_sparse += loss_sparse.item() * bs
        total_loss_w_sparse += loss_w_sparse.item() * bs
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

    model = FMDGumbelSinkhornViTModel().to(device)
    model.use_fixed_budget = False

    # ========== 第一阶段：训练骨干网络（冻结预算预测器，使用固定预算） ==========
    print("\n" + "="*50)
    print("Stage 1: Training backbone with fixed budget (budget predictor frozen)")
    print("="*50)

    # 冻结预算预测器参数
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer_stage1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE_BACKBONE,
        weight_decay=WEIGHT_DECAY_BACKBONE
    )
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=EPOCHS_STAGE1)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard_stage1'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(1, EPOCHS_STAGE1+1):
        print(f"\nStage1 Epoch {epoch}/{EPOCHS_STAGE1}")
        train_loss, train_loss_cls, train_loss_budget, train_loss_tv, train_loss_sparse, train_loss_w_sparse, train_acc = train_one_epoch(
            model, train_loader, optimizer_stage1, device, stage=1)
        val_acc = validate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/cls', train_loss_cls, epoch)
        writer.add_scalar('Loss/budget', train_loss_budget, epoch)
        writer.add_scalar('Loss/tv', train_loss_tv, epoch)
        writer.add_scalar('Loss/sparse', train_loss_sparse, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer_stage1, epoch, train_loss, os.path.join(OUTPUT_DIR, 'stage1_best.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler_stage1.step()

    # 保存第一阶段最终模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'stage1_final.pth'))

    # ========== 第二阶段：冻结骨干，仅训练预算预测器 ==========
    print("\n" + "="*50)
    print("Stage 2: Training budget predictor only (backbone frozen)")
    print("="*50)

    # 加载第一阶段最佳模型（注意：load 的是 checkpoint 字典）
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'stage1_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 冻结所有骨干参数（除了预算预测器）
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_stage2 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE_BUDGET,
        weight_decay=WEIGHT_DECAY_BUDGET
    )
    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=EPOCHS_STAGE2)

    writer2 = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard_stage2'))
    best_val_acc2 = 0
    train_losses2, train_accs2, val_accs2 = [], [], []

    for epoch in range(1, EPOCHS_STAGE2+1):
        print(f"\nStage2 Epoch {epoch}/{EPOCHS_STAGE2}")
        train_loss, train_loss_cls, train_loss_budget, train_loss_tv, train_loss_sparse, train_loss_w_sparse, train_acc = train_one_epoch(
            model, train_loader, optimizer_stage2, device, stage=2)
        val_acc = validate(model, val_loader, device)

        train_losses2.append(train_loss)
        train_accs2.append(train_acc)
        val_accs2.append(val_acc)

        writer2.add_scalar('Loss/train', train_loss, epoch)
        writer2.add_scalar('Loss/cls', train_loss_cls, epoch)
        writer2.add_scalar('Loss/budget', train_loss_budget, epoch)
        writer2.add_scalar('Loss/tv', train_loss_tv, epoch)
        writer2.add_scalar('Loss/sparse', train_loss_sparse, epoch)
        writer2.add_scalar('Acc/train', train_acc, epoch)
        writer2.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc2:
            best_val_acc2 = val_acc
            save_checkpoint(model, optimizer_stage2, epoch, train_loss, os.path.join(OUTPUT_DIR, 'stage2_best.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler_stage2.step()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))

    # 保存训练历史（合并两个阶段）
    history = {
        'stage1_train_losses': train_losses,
        'stage1_train_accs': train_accs,
        'stage1_val_accs': val_accs,
        'stage2_train_losses': train_losses2,
        'stage2_train_accs': train_accs2,
        'stage2_val_accs': val_accs2,
    }
    np.savez(os.path.join(OUTPUT_DIR, 'training_history.npz'), **history)

    writer.close()
    writer2.close()
    print("\nTwo-stage training finished.")
    print(f"Best stage1 val acc: {best_val_acc:.2f}%, Best stage2 val acc: {best_val_acc2:.2f}%")


if __name__ == "__main__":
    main()