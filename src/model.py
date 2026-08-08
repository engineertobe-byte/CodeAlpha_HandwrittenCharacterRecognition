"""
Convolutional Neural Network (CNN) for Handwritten Character Recognition.
Architecture based on standard deep learning practices for MNIST/EMNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterCNN(nn.Module):
    def __init__(self):
        super(CharacterCNN, self).__init__()
        # Input: 1x28x28 (Grayscale images)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digits (0-9) for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
