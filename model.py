import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(44944, 1200)
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 101)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self,frozen_weights=True):
        super().__init__()
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.pretrained = resnet50(weights=self.weights)
        self.fc2 = nn.Linear(in_features=1000, out_features=101, bias=True)
        if frozen_weights == True:
            for p in self.pretrained.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc2(x)
        return x