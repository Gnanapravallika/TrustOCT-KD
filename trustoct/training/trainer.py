import os
import sys
import json
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed=42):
    """Fix random seeds for 100% reproducible scientific experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_amp_scaler(enabled=True):
    """Helper for PyTorch version-compatible AMP GradScaler."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_amp_autocast(enabled=True):
    """Helper for PyTorch version-compatible AMP autocast."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class Trainer:
    """
    Research-grade Trainer formatted with live console logs, periodic batch updates, and Macro F1 tracking.
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
        
        self.scaler = get_amp_scaler(enabled=self.use_amp)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with get_amp_autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            # Print live batch updates every 300 batches in Colab
            if batch_idx % 300 == 0 or batch_idx == num_batches:
                print(f"  Batch [{batch_idx:4d}/{num_batches:4d}] | Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.2f}%", flush=True)
                sys.stdout.flush()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with get_amp_autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        val_loss = running_loss / total
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return val_loss, val_acc, val_f1

    def fit(self):
        best_val_f1 = 0.0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
        
        print(f"\nTraining {self.experiment_name} for {self.num_epochs} epochs on {self.device}...", flush=True)
        sys.stdout.flush()
        
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n--- Epoch [{epoch}/{self.num_epochs}] ---", flush=True)
            sys.stdout.flush()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary with immediate unbuffered flush
            print(f"Epoch {epoch:02d}/{self.num_epochs:02d} | Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", flush=True)
            sys.stdout.flush()
            
            # Save Best Model Checkpoint based on Val Macro F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'history': self.history
                }, best_checkpoint_path)
                print(f"✅ Best model updated! Val Macro F1: {val_f1:.4f}", flush=True)
                sys.stdout.flush()

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Macro F1: {best_val_f1:.4f}", flush=True)
        sys.stdout.flush()
        
        # Save training history JSON
        history_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
            
        return best_checkpoint_path, self.history


def run_experiment(model_name, data_dir, epochs=20, lr=1e-4, batch_size=32, device='cuda'):
    """
    Convenience wrapper function matching the user's reference notebook style:
    run_experiment('msf_cbam_resnet50', data_dir, epochs=30)
    """
    from configs.config import Config
    from trustoct.models import build_model, build_student
    from trustoct.dataset.oct_dataset import get_dataloaders
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=2, use_clahe=True
    )
    
    if 'student' in model_name.lower() or 'mobilenet' in model_name.lower():
        model = build_student('mobilenetv3', num_classes=4, pretrained=True)
    else:
        model = build_model(model_name, num_classes=4, pretrained=True)
        
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        num_epochs=epochs,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        experiment_name=model_name,
        use_amp=True
    )
    
    best_ckpt, history = trainer.fit()
    return best_ckpt, history
