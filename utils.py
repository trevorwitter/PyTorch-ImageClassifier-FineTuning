import os
import pandas as pd
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


def accuracy_score(net, data_loader, gpu=False):
    """Returns classifier accuracy score"""
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        device = torch.device("mps")
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total


if __name__ == "__main__":
    pass