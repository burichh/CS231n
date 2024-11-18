import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_conv=1, use_batchnorm=True):
        super().__init__()
        layers = []

        for i in range(num_of_conv):
            in_ch = in_channels if i == 0 else out_channels

            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=(3,3), padding=1))
            layers.append(nn.BatchNorm2d(out_channels)) if use_batchnorm else None
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class LinearBlock(nn.Module):
    def __init__(self, in_linear, out_linear, num_of_linear=1, use_batchnorm=True):
        super().__init__()
        layers = []

        for i in range(num_of_linear):
            in_lin = in_linear if i == 0 else out_linear

            layers.append(nn.Linear(in_lin, out_linear))
            layers.append(nn.BatchNorm1d(out_linear)) if use_batchnorm else None
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Following the structure of the ResNet paper: https://arxiv.org/pdf/1512.03385
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
    
        stride = 2 if downsample else 1    
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += skip
        return self.relu(x)
    
class ResidualDownConv(ResidualConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, downsample=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_residual_convs):
        super().__init__()

        layers = []

        for i in range(num_of_residual_convs):
            in_ch = in_channels if i == 0 else out_channels

            layers.append(ResidualConv(in_ch, out_channels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

