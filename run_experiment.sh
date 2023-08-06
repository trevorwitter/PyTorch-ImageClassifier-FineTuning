#!/bin/sh
python train.py --model alexnet --epochs 10
python train.py --model resnet18 --epochs 10
python train.py --model resnet50 --epochs 10