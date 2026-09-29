# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE, OUTPUT_DIR, CLIP_GRAD
from dataset import get_loaders
from model import GlobalAvgPoolModel
from utils import set_seed, save_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training')
    for cwt, labels in pbar:
        cwt, labels = cwt.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cwt)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        _, predicted = outputs.max(1)
        total += bs
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for cwt, labels in loader:
            cwt, labels = cwt.to(device), labels.to(device)
            outputs = model(cwt)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = total_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    set_seed()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders()
    model = GlobalAvgPoolModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'tensorboard'))
    best_val_acc = 0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  -> Best model saved (val_acc={val_acc:.2f}%)")

        scheduler.step()

    np.savez(os.path.join(OUTPUT_DIR, 'training_history.npz'),
             train_losses=train_losses, train_accs=train_accs, val_accs=val_accs)

    writer.close()
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()