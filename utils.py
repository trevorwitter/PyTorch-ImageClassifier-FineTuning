import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


def accuracy_score(net, data_loader, gpu=False):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        device = torch.device("mps")
    net = net.to(device)
    correct_pred = {classname: 0 for classname in data_loader.dataset.classes}
    total_pred = {classname: 0 for classname in data_loader.dataset.classes}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[data_loader.dataset.classes[label]] += 1
                total_pred[data_loader.dataset.classes[label]] += 1
    acc = correct/total
    class_acc = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 *float(correct_count) / total_pred[classname]
        class_acc.append((classname, accuracy))
    return acc, class_acc


def plot_class_accuracy(class_acc):
    class_acc = sorted(class_acc, key=lambda x: x[1])
    classes = [x[0] for x in class_acc]
    accuracy = [x[1] for x in class_acc]
    fig, ax = plt.subplots(figsize=(10,int(len(classes)/5)))
    ax.barh(classes, accuracy, align='center')
    ax.set_xlabel("Accuracy %")
    ax.set_title("Model Performance by Class")
    plt.margins(0.005, 0.005)
    plt.show()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 6))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


if __name__ == "__main__":
    pass