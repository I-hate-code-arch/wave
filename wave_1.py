import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import efficientnet_b0
import csv

class ResidualDoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 + Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity_mapped = self.shortcut(identity)
        out += identity_mapped
        return self.relu(out)

class Up(nn.Module):
    """Upscaling then ResidualDoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = in_channels // 2
            skip_channels = out_channels
            total_in_channels = conv_in_channels + skip_channels
            self.conv = ResidualDoubleConv(total_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture implementation with Residual Blocks"""
    def __init__(
        self,
        n_channels=5,    # Default to 5 input channels
        n_classes=1,     # Default to 1 output class (for regression)
        init_features=32,
        depth=5, 
        bilinear=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1)) 

        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs.append(self.inc)

        current_features = init_features
        for _ in range(depth):
            conv = ResidualDoubleConv(current_features, current_features * 2)
            pool = nn.MaxPool2d(2)
            self.encoder_convs.append(conv)
            self.encoder_pools.append(pool)
            current_features *= 2
        
        self.bottleneck = ResidualDoubleConv(current_features, current_features)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2
            
        self.outc = OutConv(current_features, n_classes)

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        _, _, h, w = x.shape
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))
        elif h > target_h:
            crop_top = (h - target_h) // 2
            x = x[:, :, crop_top : crop_top + target_h, :]
        
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))
        elif w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left : crop_left + target_w]
        return x

    def forward(self, x):
        # x initial shape: (bs, 5, 1000, 70)
        x_pooled = self.initial_pool(x) # (bs, 5, 71, 70)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70) # (bs, 5, 70, 70)

        skip_connections = []
        xi = x_resized

        xi = self.encoder_convs[0](xi) # inc
        skip_connections.append(xi)

        for i in range(self.depth):
            xi = self.encoder_convs[i+1](xi)
            skip_connections.append(xi)
            xi = self.encoder_pools[i](xi)
        
        xi = self.bottleneck(xi)

        xu = xi
        for i, block in enumerate(self.decoder_blocks):
            skip_index = self.depth - 1 - i 
            skip = skip_connections[skip_index]
            xu = block(xu, skip)
            
        logits = self.outc(xu)
        output = logits * 1000.0 + 1500.0 
        return output