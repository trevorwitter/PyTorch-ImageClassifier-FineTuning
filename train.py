import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from model import ConvNet
from utils import accuracy_score


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--gpu", default=False, type=bool, help="Train on GPU True/False")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    return parser.parse_args()
    

def training_loop(net, trainloader, valloader, gpu=False, epochs=1):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += train_loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.8f}')
                running_loss = 0.0
        train_acc = accuracy_score(net, trainloader, gpu=gpu)
        val_acc = accuracy_score(net, valloader, gpu=gpu)
        print(f"Epoch {epoch} train_acc: {train_acc}, val_acc: {val_acc}")
    PATH = './models/net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Training complete - model saved to {PATH}')


def main(epochs=1, gpu=False, num_workers=1):
    transform = transforms.Compose(
            [transforms.Resize(256,antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    data = torchvision.datasets.Food101(root="./data",
                                    split="train",
                                    transform=transform
                                    )

    train_data, val_data = torch.utils.data.random_split(
        data, 
        [.8,.2],
        torch.Generator().manual_seed(42)
        )

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=num_workers)
    
    net = ConvNet()
    training_loop(net, train_dataloader, val_dataloader, gpu=gpu, epochs=epochs)


if __name__ == "__main__":
    args = arg_parse()    
    main(
        epochs=args.epochs,
        gpu=args.gpu, 
        num_workers=args.workers,
        )