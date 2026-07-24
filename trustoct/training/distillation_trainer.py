import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class DistillationTrainer:
    """
    Calibration-Aware Knowledge Distillation Trainer.
    
    Combines three loss components:
      1. Hard Label Loss (CE):   Student predictions vs ground truth
      2. Soft Label Loss (KD):   Student soft outputs vs Teacher soft outputs (KL Divergence)
      3. Attention Transfer Loss: Student spatial features vs Teacher spatial attention maps
    
    The KD temperature acts as an implicit calibration mechanism — soft targets 
    from a well-calibrated teacher transfer inter-class similarity structure 
    to the student, producing better-calibrated confidence estimates.
    """
    def __init__(self, teacher_model, student_model, train_loader, val_loader, device,
                 lr=1e-3, weight_decay=1e-4, num_epochs=20,
                 temperature=4.0, alpha=0.3, beta=0.5, gamma=0.2,
                 checkpoint_dir="./outputs/checkpoints",
                 experiment_name="KD_TrustOCT", use_amp=True):
        """
        Args:
            teacher_model: Pre-trained teacher (ResNet50+MSF+CBAM), frozen during distillation
            student_model: Lightweight student (MobileNetV3-Small), trained during distillation
            temperature: Softmax temperature for KD (higher = softer distributions)
            alpha: Weight for hard label CE loss
            beta:  Weight for soft label KD loss
            gamma: Weight for attention transfer loss
        """
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
        
        # Freeze teacher completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        self.optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'ce_loss': [], 'kd_loss': [], 'attn_loss': [], 'lr': []
        }
        
        # Register hooks to capture teacher's intermediate features
        self.teacher_features = {}
        self._register_teacher_hooks()

    def _register_teacher_hooks(self):
        """Register forward hooks to capture teacher's CBAM attention-weighted features."""
        def hook_fn(name):
            def hook(module, input, output):
                self.teacher_features[name] = output.detach()
            return hook
        
        # Capture the fused feature map after CBAM (the teacher's key attention output)
        if hasattr(self.teacher, 'cbam_fused'):
            self.teacher.cbam_fused.register_forward_hook(hook_fn('fused_attn'))
        elif hasattr(self.teacher, 'layer4'):
            self.teacher.layer4.register_forward_hook(hook_fn('fused_attn'))

    def _get_teacher_outputs(self, images):
        """Get teacher's logits and attention features (no gradient)."""
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        teacher_feat = self.teacher_features.get('fused_attn', None)
        return teacher_logits, teacher_feat

    def _attention_transfer_loss(self, student_feat, teacher_feat):
        """
        Compute attention transfer loss between student and teacher spatial features.
        Uses L2 loss on normalized spatial attention maps.
        """
        if student_feat is None or teacher_feat is None:
            return torch.tensor(0.0, device=self.device)
        
        # Generate spatial attention maps (channel-wise mean)
        student_attn = torch.mean(student_feat, dim=1)  # [B, H, W]
        teacher_attn = torch.mean(teacher_feat, dim=1)  # [B, H', W']
        
        # Resize teacher attention to match student spatial dimensions
        if student_attn.shape != teacher_attn.shape:
            teacher_attn = F.interpolate(
                teacher_attn.unsqueeze(1), 
                size=student_attn.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # L2 normalize attention maps before comparison
        student_attn = F.normalize(student_attn.view(student_attn.size(0), -1), p=2, dim=1)
        teacher_attn = F.normalize(teacher_attn.view(teacher_attn.size(0), -1), p=2, dim=1)
        
        return F.mse_loss(student_attn, teacher_attn)

    def _distillation_loss(self, student_logits, teacher_logits):
        """
        Compute KL Divergence loss on temperature-scaled softmax outputs.
        This is the core "dark knowledge" transfer from Hinton et al.
        """
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Scale by T^2 as per Hinton et al. to maintain gradient magnitude
        kd_loss = self.criterion_kd(student_soft, teacher_soft) * (self.temperature ** 2)
        return kd_loss

    def train_epoch(self):
        self.student.train()
        self.teacher.eval()
        
        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0
        running_attn = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="[KD Train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Get teacher outputs (frozen, no grad)
            teacher_logits, teacher_feat = self._get_teacher_outputs(images)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Get student outputs with features for attention transfer
                student_logits, student_feat = self.student(images, return_features=True)
                
                # Loss 1: Hard label cross-entropy
                ce_loss = self.criterion_ce(student_logits, labels)
                
                # Loss 2: Soft label KD (dark knowledge transfer)
                kd_loss = self._distillation_loss(student_logits, teacher_logits)
                
                # Loss 3: Attention transfer
                attn_loss = self._attention_transfer_loss(student_feat, teacher_feat)
                
                # Combined calibration-aware loss
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
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'CE': f"{ce_loss.item():.3f}",
                'KD': f"{kd_loss.item():.3f}",
                'acc': f"{correct/total:.4f}"
            })
        
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
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
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
        
        print(f"\n{'='*65}")
        print(f" Knowledge Distillation: {self.experiment_name}")
        print(f" Temperature: {self.temperature} | α={self.alpha} β={self.beta} γ={self.gamma}")
        print(f" Epochs: {self.num_epochs} | Device: {self.device} | AMP: {self.use_amp}")
        print(f"{'='*65}\n")
        
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            train_loss, ce_loss, kd_loss, attn_loss, train_acc = self.train_epoch()
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
                  f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.2e}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history,
                    'config': {
                        'temperature': self.temperature,
                        'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma
                    }
                }, best_path)
                print(f"  -> Best student saved: {best_path} (Val Acc: {val_acc*100:.2f}%)")

        total_time = time.time() - start_time
        print(f"\nDistillation completed in {total_time/60:.2f} min. Best Val Acc: {best_val_acc*100:.2f}%")
        
        history_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        return best_path, self.history
