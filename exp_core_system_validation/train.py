# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import (
    EPOCHS, LEARNING_RATE_BACKBONE, LEARNING_RATE_BUDGET,
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_BUDGET, DEVICE, CLIP_GRAD
)
from model import FMDGumbelSinkhornViTModel
from dataset import get_loaders
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_loss_cls = 0
    total_loss_budget = 0
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
        _, predicted = outputs['logits'].max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_loss_cls = total_loss_cls / total
    epoch_loss_budget = total_loss_budget / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_loss_cls, epoch_loss_budget, epoch_acc

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

def train_experiment(train_cwt, train_phys, val_cwt, val_phys, test_cwt, test_phys, output_dir):
    """
    训练一个实验（例如原始数据或加噪数据）。
    """
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # 加载数据（标准化参数自动从训练集计算）
    train_loader, val_loader, test_loader = get_loaders(
        train_cwt, train_phys,
        val_cwt, val_phys,
        test_cwt, test_phys
    )

    model = FMDGumbelSinkhornViTModel().to(device)

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

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_loss_cls, train_loss_budget, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = validate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/cls', train_loss_cls, epoch)
        writer.add_scalar('Loss/budget', train_loss_budget, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    np.savez(os.path.join(output_dir, 'training_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs)
    writer.close()
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

    # 测试
    from test import test
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model, test_loader, device, output_dir)

if __name__ == "__main__":
    # 该脚本通常由 main.py 调用，这里仅用于直接调试
    # 如需直接运行，请取消注释并设置正确的路径
    # from config import ORIGINAL_TRAIN_CWT, ORIGINAL_TRAIN_PHYS, ...
    # train_experiment(ORIGINAL_TRAIN_CWT, ORIGINAL_TRAIN_PHYS, ...)
    pass