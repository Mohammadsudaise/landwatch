import torch
import torch.nn as nn
import torch.nn.functional as F

class LandUseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LandUseCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 64x64 → 32x32
        x = self.pool(F.relu(self.conv2(x)))   # 32x32 → 16x16
        x = self.pool(F.relu(self.conv3(x)))   # 16x16 → 8x8

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x