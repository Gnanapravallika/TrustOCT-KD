import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    """
    Multi-receptive field convolution block with parallel dilated kernels (1x1, 3x3 d=1, 3x3 d=2).
    Captures multi-scale contextual features within a single layer.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        branch_channels = out_channels // 3
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adjustment for channel alignment if out_channels not divisible by 3
        remainder = out_channels - (branch_channels * 3)
        self.remainder_conv = nn.Conv2d(in_channels, remainder, kernel_size=1, bias=False) if remainder > 0 else None

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        if self.remainder_conv is not None:
            out_rem = self.remainder_conv(x)
            return torch.cat([out1, out2, out3, out_rem], dim=1)
        return torch.cat([out1, out2, out3], dim=1)


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-Scale Feature Fusion (MSF) module.
    Fuses mid-level (stage 2), high-level (stage 3), and deep-level (stage 4) features from ResNet50.
    """
    def __init__(self, in_channels_list=[512, 1024, 2048], proj_dim=256):
        super(MultiScaleFeatureFusion, self).__init__()
        
        self.proj_stage2 = MultiScaleConv(in_channels_list[0], proj_dim)
        self.proj_stage3 = MultiScaleConv(in_channels_list[1], proj_dim)
        self.proj_stage4 = MultiScaleConv(in_channels_list[2], proj_dim)
        
        fused_dim = proj_dim * 3
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_stage2, feat_stage3, feat_stage4):
        # Target spatial size is stage2 spatial resolution (e.g., 28x28 for 224x224 input)
        target_size = feat_stage2.shape[2:]
        
        p2 = self.proj_stage2(feat_stage2)
        p3 = F.interpolate(self.proj_stage3(feat_stage3), size=target_size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj_stage4(feat_stage4), size=target_size, mode='bilinear', align_corners=False)
        
        concat_feats = torch.cat([p2, p3, p4], dim=1)
        fused_feat = self.fusion_conv(concat_feats)
        return fused_feat
