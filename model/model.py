import torch
import torch.nn as nn
from base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_kernel_size, conv_stride, conv_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class HomographyRegressor(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1, 3, 1, 0)
        self.conv2 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        self.conv3 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        self.conv4 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        self.conv5 = ConvBlock(64, 64, 3, 2, 1, 3, 1, 0)
        self.conv6 = ConvBlock(64, 64, (5, 3), 2, 1, 2, 1, 0)
        self.conv7 = ConvBlock(64, 64, (5, 3), 2, 1, 2, 1, 0)
        self.conv8 = ConvBlock(64, 64, (4, 3), 2, 1, 2, 1, 0)
        self.conv9 = ConvBlock(64, 64, 3, 1, 1, 2, 1, 0)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
