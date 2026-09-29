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
    WEIGHT_DECAY_BACKBONE, WEIGHT_DECAY_BUDGET, DEVICE, CLIP_GRAD,
    OUTPUT_DIR_STRONG, OUTPUT_DIR_WEAK, TARGET_CLASS, NOISE_SNR
)
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, save_checkpoint, load_checkpoint

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

def main(add_noise_for_class, output_dir):
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Add noise for class {add_noise_for_class} with SNR={NOISE_SNR if add_noise_for_class is not None else 'N/A'}")

    train_loader, val_loader, test_loader = get_loaders(
        add_noise_for_class=add_noise_for_class,
        snr_db=NOISE_SNR if add_noise_for_class is not None else None
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
    best_model_state = None
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
            best_model_state = model.state_dict().copy()
            # 保存最佳模型权重（只保存 state_dict）
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    np.savez(os.path.join(output_dir, 'training_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs)
    writer.close()
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

    # 测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # 如果没有保存过，加载最后一次的模型（但应该有保存）
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    from test import test
    test(model, test_loader, device, output_dir, target_class=TARGET_CLASS)

if __name__ == "__main__":
    # 运行强特征模型（不加噪声）
    main(add_noise_for_class=None, output_dir=OUTPUT_DIR_STRONG)
    # 运行弱特征模型（对目标类加噪声）
    main(add_noise_for_class=TARGET_CLASS, output_dir=OUTPUT_DIR_WEAK)