import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def set_seed(seed=42):
    """Fix random seeds for 100% reproducible scientific experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Research-grade Trainer for TrustOCT models.
    Supports Mixed Precision (AMP), LR Scheduling, Checkpointing, and Logging.
    """
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-4, weight_decay=1e-4, num_epochs=20, 
                 checkpoint_dir="./outputs/checkpoints", experiment_name="exp003_trustoct",
                 use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.experiment_name = experiment_name
        self.use_amp = use_amp and torch.cuda.is_available()
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="[Train Epoch]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.4f}"})
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        best_val_acc = 0.0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
        
        print(f"\n=======================================================")
        print(f" Starting Training Experiment: {self.experiment_name}")
        print(f" Epochs: {self.num_epochs} | Device: {self.device} | AMP: {self.use_amp}")
        print(f"=======================================================\n")
        
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"Epoch [{epoch:02d}/{self.num_epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.2e}")
            
            # Save Best Model Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, best_checkpoint_path)
                print(f"  -> Best model saved to: {best_checkpoint_path} (Val Acc: {val_acc*100:.2f}%)")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")
        
        # Save training history JSON
        history_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
            
        return best_checkpoint_path, self.history
