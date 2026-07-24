import torch
import torch.nn as nn
import torchvision.models as models

class StudentMobileNetV3(nn.Module):
    """
    Lightweight Student Model: MobileNetV3-Small
    ~2.5M parameters — designed for edge/mobile clinical OCT deployment.
    
    Includes a feature adapter to match the teacher's intermediate feature
    dimensions for attention transfer during knowledge distillation.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(StudentMobileNetV3, self).__init__()
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v3_small(weights=weights)
        except Exception:
            print("[Student] Pretrained weight download failed. Using random init.")
            mobilenet = models.mobilenet_v3_small(weights=None)
        
        # Extract feature backbone (all layers before classifier)
        self.features = mobilenet.features       # Output: [B, 576, 7, 7]
        self.avgpool = mobilenet.avgpool          # Output: [B, 576, 1, 1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Feature adapter for attention transfer from teacher
        # Maps student's 576-dim feature space to teacher's spatial attention space
        self.attention_adapter = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor [B, 3, 224, 224]
            return_features: If True, returns intermediate features for attention transfer
        Returns:
            logits: Classification output [B, num_classes]
            features (optional): Intermediate spatial features for attention transfer
        """
        feat_map = self.features(x)              # [B, 576, 7, 7]
        
        pooled = self.avgpool(feat_map)           # [B, 576, 1, 1]
        pooled = torch.flatten(pooled, 1)         # [B, 576]
        logits = self.classifier(pooled)          # [B, num_classes]
        
        if return_features:
            adapted_feat = self.attention_adapter(feat_map)  # [B, 256, 7, 7]
            return logits, adapted_feat
        
        return logits


class StudentEfficientNetB0(nn.Module):
    """
    Alternative Student Model: EfficientNet-B0
    ~5.3M parameters — slightly larger but potentially higher accuracy ceiling.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(StudentEfficientNetB0, self).__init__()
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            effnet = models.efficientnet_b0(weights=weights)
        except Exception:
            print("[Student] Pretrained weight download failed. Using random init.")
            effnet = models.efficientnet_b0(weights=None)
        
        self.features = effnet.features          # Output: [B, 1280, 7, 7]
        self.avgpool = effnet.avgpool
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
        
        self.attention_adapter = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_features=False):
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        
        if return_features:
            adapted_feat = self.attention_adapter(feat_map)
            return logits, adapted_feat
        
        return logits


def build_student(student_name="mobilenetv3", num_classes=4, pretrained=True):
    """
    Factory function for student model selection.
    Options: 'mobilenetv3', 'efficientnet_b0'
    """
    student_name = student_name.lower()
    if student_name in ['mobilenetv3', 'mobilenet', 'mobilenet_v3_small']:
        return StudentMobileNetV3(num_classes=num_classes, pretrained=pretrained)
    elif student_name in ['efficientnet_b0', 'efficientnet', 'effnet']:
        return StudentEfficientNetB0(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown student model: {student_name}. Options: ['mobilenetv3', 'efficientnet_b0']")
