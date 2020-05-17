import PIL
import torch
import numpy as np

from torchvision import transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def validation_transformations():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])


def train_transformations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(padding=4, padding_mode='reflect'),
        transforms.RandomCrop(size=32),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
