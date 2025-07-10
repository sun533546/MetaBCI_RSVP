import numpy as np
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        se = self.global_avg_pool(x)
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class EEGNetAttn(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, samples=128):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16), nn.ReLU()
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        self.cbam_block = SEBlock(32)

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4224, num_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.cbam_block(x)
        x = self.separableConv(x)
        x = self.classifier(x)
        return x
