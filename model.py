import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F


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
        
def alexnet():
    net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=101)
    transforms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    return transforms, net

def resnet18():
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(in_features=512, out_features=101)
    transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    return transforms, net

def resnet50():
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(in_features=2048, out_features=101, bias=True)
    transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    return transforms, net

