import os
import sys
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore", category=FutureWarning)

def get_amp_scaler(enabled=True):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_amp_autocast(enabled=True):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class DistillationTrainer:
    """
    Calibration-Aware Knowledge Distillation Trainer with live unbuffered logging.
    """
    def __init__(self, teacher_model, student_model, train_loader, val_loader, device,
                 lr=1e-3, weight_decay=1e-4, num_epochs=20,
                 temperature=4.0, alpha=0.3, beta=0.5, gamma=0.2,
                 checkpoint_dir="./outputs/checkpoints",
                 experiment_name="KD_TrustOCT", use_amp=True):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.experiment_name = experiment_name
        self.use_amp = use_amp and torch.cuda.is_available()
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        self.optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = get_amp_scaler(enabled=self.use_amp)
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'ce_loss': [], 'kd_loss': [], 'attn_loss': [], 'lr': []
        }
        
        self.teacher_features = {}
        self._register_teacher_hooks()

    def _register_teacher_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.teacher_features[name] = output.detach()
            return hook
        
        if hasattr(self.teacher, 'cbam_fused'):
            self.teacher.cbam_fused.register_forward_hook(hook_fn('fused_attn'))
        elif hasattr(self.teacher, 'layer4'):
            self.teacher.layer4.register_forward_hook(hook_fn('fused_attn'))

    def _get_teacher_outputs(self, images):
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        teacher_feat = self.teacher_features.get('fused_attn', None)
        return teacher_logits, teacher_feat

    def _attention_transfer_loss(self, student_feat, teacher_feat):
        if student_feat is None or teacher_feat is None:
            return torch.tensor(0.0, device=self.device)
        
        student_attn = torch.mean(student_feat, dim=1)
        teacher_attn = torch.mean(teacher_feat, dim=1)
        
        if student_attn.shape != teacher_attn.shape:
            teacher_attn = F.interpolate(
                teacher_attn.unsqueeze(1), 
                size=student_attn.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        student_attn = F.normalize(student_attn.view(student_attn.size(0), -1), p=2, dim=1)
        teacher_attn = F.normalize(teacher_attn.view(teacher_attn.size(0), -1), p=2, dim=1)
        
        return F.mse_loss(student_attn, teacher_attn)

    def _distillation_loss(self, student_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        return self.criterion_kd(student_soft, teacher_soft) * (self.temperature ** 2)

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()
        
        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0
        running_attn = 0.0
        correct = 0
        total = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            teacher_logits, teacher_feat = self._get_teacher_outputs(images)
            
            with get_amp_autocast(enabled=self.use_amp):
                student_logits, student_feat = self.student(images, return_features=True)
                ce_loss = self.criterion_ce(student_logits, labels)
                kd_loss = self._distillation_loss(student_logits, teacher_logits)
                attn_loss = self._attention_transfer_loss(student_feat, teacher_feat)
                total_loss = self.alpha * ce_loss + self.beta * kd_loss + self.gamma * attn_loss
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += total_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_attn += attn_loss.item() * images.size(0)
            
            _, preds = torch.max(student_logits, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            # Print live batch updates every 300 batches in Colab
            if batch_idx % 300 == 0 or batch_idx == num_batches:
                print(f"  [KD Batch {batch_idx:4d}/{num_batches:4d}] | Loss: {running_loss/total:.4f} | CE: {running_ce/total:.3f} | KD: {running_kd/total:.3f} | Acc: {correct/total*100:.2f}%", flush=True)
                sys.stdout.flush()
        
        n = total
        return (running_loss/n, running_ce/n, running_kd/n, running_attn/n, correct/n)

    @torch.no_grad()
    def validate(self):
        self.student.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with get_amp_autocast(enabled=self.use_amp):
                outputs = self.student(images)
                loss = self.criterion_ce(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
        
        return running_loss / total, correct / total

    def fit(self):
        best_val_acc = 0.0
        best_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_student_best.pth")
        
        print(f"\n{'='*65}", flush=True)
        print(f" Knowledge Distillation: {self.experiment_name}", flush=True)
        print(f" Temperature: {self.temperature} | α={self.alpha} β={self.beta} γ={self.gamma}", flush=True)
        print(f" Epochs: {self.num_epochs} | Device: {self.device} | AMP: {self.use_amp}", flush=True)
        print(f"{'='*65}\n", flush=True)
        sys.stdout.flush()
        
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n--- [KD Epoch {epoch}/{self.num_epochs}] ---", flush=True)
            sys.stdout.flush()
            
            train_loss, ce_loss, kd_loss, attn_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['ce_loss'].append(ce_loss)
            self.history['kd_loss'].append(kd_loss)
            self.history['attn_loss'].append(attn_loss)
            self.history['lr'].append(current_lr)
            
            print(f"Epoch [{epoch:02d}/{self.num_epochs:02d}] "
                  f"Loss: {train_loss:.4f} (CE:{ce_loss:.3f} KD:{kd_loss:.3f} Attn:{attn_loss:.3f}) | "
                  f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.2e}", flush=True)
            sys.stdout.flush()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.student.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history,
                    'config': {
                        'temperature': self.temperature,
                        'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma
                    }
                }, best_path)
                print(f"  -> Best student saved: {best_path} (Val Acc: {val_acc*100:.2f}%)", flush=True)
                sys.stdout.flush()

        total_time = time.time() - start_time
        print(f"\nDistillation completed in {total_time/60:.2f} min. Best Val Acc: {best_val_acc*100:.2f}%", flush=True)
        sys.stdout.flush()
        
        history_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        return best_path, self.history
