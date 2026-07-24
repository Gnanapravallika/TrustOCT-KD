import torch
import torch.nn as nn
import torchvision.models as models
from .cbam import CBAM
from .msf import MultiScaleFeatureFusion

class ResNet50_Baseline(nn.Module):
    """
    EXP001: Standard ResNet50 baseline model.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50_Baseline, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50_MSF(nn.Module):
    """
    EXP002: ResNet50 + Multi-Scale Feature Fusion (MSF)
    """
    def __init__(self, num_classes=4, pretrained=True, proj_dim=256):
        super(ResNet50_MSF, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        self.msf = MultiScaleFeatureFusion(in_channels_list=[512, 1024, 2048], proj_dim=proj_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Combined classification head: uses both stage4 features and fused multi-scale features
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048 + proj_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        fused_feat = self.msf(feat2, feat3, feat4)
        
        pooled_feat4 = self.global_pool(feat4).flatten(1)
        pooled_fused = self.global_pool(fused_feat).flatten(1)
        
        combined_feat = torch.cat([pooled_feat4, pooled_fused], dim=1)
        logits = self.fc(combined_feat)
        return logits


class ResNet50_MSF_CBAM(nn.Module):
    """
    EXP003 (Proposed TrustOCT Model):
    ResNet50 + Multi-Scale Feature Fusion (MSF) + Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, num_classes=4, pretrained=True, proj_dim=256):
        super(ResNet50_MSF_CBAM, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Attention modules at bottleneck layers and feature output
        self.cbam_layer2 = CBAM(512)
        self.cbam_layer3 = CBAM(1024)
        self.cbam_layer4 = CBAM(2048)
        
        self.msf = MultiScaleFeatureFusion(in_channels_list=[512, 1024, 2048], proj_dim=proj_dim)
        self.cbam_fused = CBAM(proj_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048 + proj_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        feat2 = self.layer2(x)
        feat2 = self.cbam_layer2(feat2)
        
        feat3 = self.layer3(feat2)
        feat3 = self.cbam_layer3(feat3)
        
        feat4 = self.layer4(feat3)
        feat4 = self.cbam_layer4(feat4)

        fused_feat = self.msf(feat2, feat3, feat4)
        fused_feat = self.cbam_fused(fused_feat)
        
        pooled_feat4 = self.global_pool(feat4).flatten(1)
        pooled_fused = self.global_pool(fused_feat).flatten(1)
        
        combined_feat = torch.cat([pooled_feat4, pooled_fused], dim=1)
        logits = self.fc(combined_feat)
        return logits


def build_model(model_name="resnet50_msf_cbam", num_classes=4, pretrained=True):
    """
    Factory function for instantiating model architectures for ablation studies.
    Options: 'resnet50', 'resnet50_msf', 'resnet50_msf_cbam'
    """
    model_name = model_name.lower()
    if model_name in ['resnet50', 'baseline', 'exp001']:
        return ResNet50_Baseline(num_classes=num_classes, pretrained=pretrained)
    elif model_name in ['resnet50_msf', 'exp002']:
        return ResNet50_MSF(num_classes=num_classes, pretrained=pretrained)
    elif model_name in ['resnet50_msf_cbam', 'trustoct', 'exp003']:
        return ResNet50_MSF_CBAM(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Select from ['resnet50', 'resnet50_msf', 'resnet50_msf_cbam']")
