# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import EPOCHS, LEARNING_RATE_BACKBONE, WEIGHT_DECAY_BACKBONE, DEVICE, OUTPUT_DIR, CLIP_GRAD
from dataset import get_loaders
from model import FixedBudgetModel
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training')
    for cwt, phys, labels in pbar:
        cwt, labels = cwt.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cwt, labels=labels)
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
            cwt, labels = cwt.to(device), labels.to(device)
            outputs = model(cwt)
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders()
    model = FixedBudgetModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_BACKBONE, weight_decay=WEIGHT_DECAY_BACKBONE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = validate(model, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    np.savez(os.path.join(OUTPUT_DIR, 'training_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs)

    writer.close()
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()