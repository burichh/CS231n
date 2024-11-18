from .my_blocks import *

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

    def forward(self, x):
        return self.net(x)

class ModelResidualConvLinear(nn.Module):
    def __init__(self, num_of_residual_convs, num_of_linear):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(3, 32, num_of_residual_convs),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, num_of_residual_convs),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128, num_of_residual_convs),
            nn.MaxPool2d(2),
            nn.Flatten(),
            LinearBlock(128*4*4, 512, num_of_linear, use_batchnorm=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)
    