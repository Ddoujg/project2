# train_stage1.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import (
    EPOCHS_STAGE1, LEARNING_RATE_BACKBONE, WEIGHT_DECAY_BACKBONE,
    DEVICE, OUTPUT_DIR, CLIP_GRAD, B_MIN, B_MAX, STAGE1_MODEL_PATH
)
from dataset import get_loaders
from model import FMDGumbelSinkhornViTModel
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, optimizer, device, fixed_budget):
    model.train()
    total_loss = 0
    total_loss_cls = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Stage1 Training')
    for cwt, phys, labels in pbar:
        cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
        optimizer.zero_grad()
        # 固定预算：所有样本使用固定值
        a = torch.full((cwt.size(0),), fixed_budget, device=device).float()
        # 手动前向（绕过预算预测器）
        F_map, _ = model.extractor(cwt)
        S = model.attention(F_map)
        features = F_map.permute(0, 2, 3, 1).reshape(cwt.size(0), -1, model.embed_dim)
        cost = -S.reshape(cwt.size(0), -1)
        z = model.selector(cost, a, features)
        z = model.pre_classifier_norm(z)
        logits = model.classifier(z)
        loss_cls = nn.CrossEntropyLoss()(logits, labels)
        # 第一阶段不加入预算约束和正则项（可选加入正则项以提升特征质量）
        loss_tv = model.tv_lambda * model.temporal_tv_loss(S)
        loss_sparse = model.sparse_lambda * model.frequency_sparse_loss(S)
        loss = loss_cls + loss_tv + loss_sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_loss_cls += loss_cls.item() * bs
        _, predicted = logits.max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_loss_cls = total_loss_cls / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_loss_cls, epoch_acc

def validate(model, loader, device, fixed_budget):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cwt, phys, labels in loader:
            cwt, phys, labels = cwt.to(device), phys.to(device), labels.to(device)
            a = torch.full((cwt.size(0),), fixed_budget, device=device).float()
            F_map, _ = model.extractor(cwt)
            S = model.attention(F_map)
            features = F_map.permute(0, 2, 3, 1).reshape(cwt.size(0), -1, model.embed_dim)
            cost = -S.reshape(cwt.size(0), -1)
            z = model.selector(cost, a, features)
            z = model.pre_classifier_norm(z)
            logits = model.classifier(z)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders()
    model = FMDGumbelSinkhornViTModel().to(device)

    # 只优化骨干网络和分类器（预算预测器参数不参与，但模型中包含它，需冻结）
    for name, param in model.named_parameters():
        if 'budget_predictor' in name:
            param.requires_grad = False
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_BACKBONE, weight_decay=WEIGHT_DECAY_BACKBONE
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE1)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard_stage1'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []
    fixed_budget = (B_MIN + B_MAX) / 2.0

    for epoch in range(1, EPOCHS_STAGE1+1):
        print(f"\nStage1 Epoch {epoch}/{EPOCHS_STAGE1}")
        train_loss, train_loss_cls, train_acc = train_one_epoch(model, train_loader, optimizer, device, fixed_budget)
        val_acc = validate(model, val_loader, device, fixed_budget)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/cls', train_loss_cls, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, STAGE1_MODEL_PATH)
            print(f"  -> Best stage1 model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    np.savez(os.path.join(OUTPUT_DIR, 'stage1_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs)
    writer.close()
    print(f"\nStage1 finished. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()