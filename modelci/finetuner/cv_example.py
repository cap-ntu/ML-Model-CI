#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/12/2021

Reference:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
"""
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms

from modelci.finetuner.trainer import CIFAR10Trainer, train_net

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

net = torchvision.models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)

for param in net.parameters():
    param.requires_grad = False

for param in net.fc.parameters():
    param.requires_grad = True


net = net.to('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
trainer = CIFAR10Trainer(net=net, optimizer=optimizer, train_data_loader=train_loader, test_data_loader=test_loader)

print(train_net(net, 25, trainer=trainer, save_name='resnet18_ft', log_batch_num=4))
