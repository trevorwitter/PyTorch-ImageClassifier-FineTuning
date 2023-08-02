import os
import time
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
from model import ConvNet, alexnet, resnet18, resnet50
from utils import accuracy_score, plot_classes_preds
import torch
from torch.utils.tensorboard import SummaryWriter



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ResNet", type=str, help="Options: ConvNet, alexnet, ResNet")
    parser.add_argument("--workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--gpu", default=True, type=bool, help="Train on GPU True/False")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--warm_start", default=False, type=bool, help="Loads trained model")
    return parser.parse_args()
    

def training_loop(net, trainloader, valloader, gpu=False, epochs=1, model_name='net'):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Training on {device.type}")
    
    tb = SummaryWriter(f'runs/{model_name}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0.0
    step_count = 0
    for epoch in range(epochs): 
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        net.train()
        net = net.to(device)
        train_running_loss = 0.0
        train_running_corrects = 0
        val_running_loss = 0.0
        val_running_corrects = 0
        
        #Train
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if epoch == 0 and i == 0:
                grid = torchvision.utils.make_grid(inputs)
                tb.add_image('images', grid, 0)
                tb.add_graph(net,inputs)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            step_count += 1
            train_running_loss += train_loss.item() #* inputs.size(0)
            tb.add_scalar('training running loss',
                            train_loss.item(),
                            step_count)
            train_running_corrects += torch.sum(preds == labels.data)
            #if i % 100 == 99:
            #    tb.add_scalar('training running loss',
            #                    train_running_loss / 99,
            #                    epoch * len(trainloader) + i)
            #    train_running_loss = 0.0
        train_loss = train_running_loss / len(trainloader.dataset)
        train_acc = train_running_corrects.item() / len(trainloader.dataset)
        tb.add_scalar('Loss/Training',
                    train_loss,
                    epoch)

        tb.add_scalar('Accuracy/Training',
                    train_acc,
                    epoch)
        print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
        print("evaluation")


        #Validation
        net.eval()
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()
            val_running_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / len(valloader.dataset)
        val_acc = val_running_corrects.item() / len(valloader.dataset)
        tb.add_scalar('Loss/Validation',
                    val_loss,
                    epoch)

        tb.add_scalar('Accuracy/Validation',
                    val_acc,
                    epoch)
        tb.add_figure('Validation Predictions vs. Actuals',
                    plot_classes_preds(net, inputs[:5], labels[:5], trainloader.dataset.dataset.classes),
                    global_step=epoch)
        print(f"Val Loss: {val_loss}, val Acc: {val_acc}")
        if val_acc > best_acc:
            best_acc = val_acc
            PATH = f'./models/{model_name}.pth'
            torch.save(net.state_dict(), PATH)

    print(f'Training complete - model saved to {PATH}')
    tb.close()

def main(model='ResNet', epochs=1, gpu=False, num_workers=1, warm_start=False):
    print(f"Model: {model}")
    if model == 'ConvNet':
        net = ConvNet()
        transform = transforms.Compose(
                [transforms.Resize(256,antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    elif model == 'resnet18':
        transform, net = resnet18()
    elif model == 'resnet50':
        transform, net = resnet50()
    elif model == 'alexnet':
        transform, net = alexnet()

    if warm_start == True:
        print("warm start")
        PATH = f"./models/{model}.pth"
        net.load_state_dict(torch.load(PATH))
        print(f"Warm start - {model} model loaded")
    
    
    data = torchvision.datasets.Food101(root="./data",
                                    split="train",
                                    transform=transform,
                                    download=True,
                                    )

    train_data, val_data = torch.utils.data.random_split(
        data, 
        [.8,.2],
        torch.Generator().manual_seed(42)
        )

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=num_workers)
    training_loop(net, train_dataloader, val_dataloader, gpu=gpu, epochs=epochs, model_name=model)


if __name__ == "__main__":
    args = arg_parse()    
    main(
        model=args.model,
        epochs=args.epochs,
        gpu=args.gpu, 
        num_workers=args.workers,
        warm_start=args.warm_start,
        )