import torch
import torch.nn as nn
import torch.optim as optim

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

        self.model = nn.Sequential(layers)

class ModelConvReluPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, num_of_conv=1, use_batchnorm=False),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, num_of_conv=1, use_batchnorm=False),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, num_of_conv=1, use_batchnorm=False),
            nn.MaxPool2d(2),
            nn.Flatten(),
            LinearBlock(128*4*4, 512, num_of_linear=1, use_batchnorm=False),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.net(x)

class ModelConvPoolLinear(nn.Module):
    def __init__(self, num_of_conv, num_of_linear, use_batchnorm=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, num_of_conv, use_batchnorm),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, num_of_conv, use_batchnorm),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, num_of_conv, use_batchnorm),
            nn.MaxPool2d(2),
            nn.Flatten(),
            LinearBlock(128*4*4, 512, num_of_linear, use_batchnorm),
            nn.Linear(512, 10)
        )

class ModelDoubleConvRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, num_of_conv=2, batchnorm=False),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, num_of_conv=2, batchnorm=False),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, num_of_conv=2, batchnorm=False),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.net(x)
    

class ModelDoubleConvBatchNormRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            double_conv_bn_relu(3, 32),
            nn.MaxPool2d(2),
            double_conv_bn_relu(32, 64),
            nn.MaxPool2d(2),
            double_conv_bn_relu(64, 128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.net(x)

model_multi_conv_bn_relu_pool = nn.Sequential(
    double_conv_bn_relu(3, 32),
    double_conv_bn_relu(32, 32),
    double_conv_bn_relu(32, 32),
    nn.MaxPool2d(2),
    double_conv_bn_relu(32, 64),
    double_conv_bn_relu(64, 64),
    double_conv_bn_relu(64, 64),
    nn.MaxPool2d(2),
    double_conv_bn_relu(64, 128),
    double_conv_bn_relu(128, 128),
    double_conv_bn_relu(128, 128),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128*4*4, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 10)
)