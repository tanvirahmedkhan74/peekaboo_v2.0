import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise Separable Convolution Block 1
        self.depthwise1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise1 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Depthwise Separable Convolution Block 2
        self.depthwise2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Final Convolution without Global Pooling
        self.conv_final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pointwise1(self.depthwise1(x))))
        x = self.relu(self.bn3(self.pointwise2(self.depthwise2(x))))

        # Final convolution
        x = self.conv_final(x)
        return x