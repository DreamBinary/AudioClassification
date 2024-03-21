# -*- coding:utf-8 -*-
# @FileName : tmp.py
# @Time : 2024/3/20 15:53
# @Author : fiv
import torch
import torchvision

cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
dataloder = torch.utils.data.DataLoader(cifar10_dataset, batch_size=4, shuffle=True)
for x, y in dataloder:
    print(x.shape, y)
    break
