# -*- coding:utf-8 -*-
# @FileName : train.py
# @Time : 2024/3/20 16:13
# @Author : fiv

from model import Transformer
from dataset import get_dataloader


#
def train():
    dataloader = get_dataloader()
    model = Transformer()

    model.train()
    for xx, yy in dataloader:
        output = model(xx)
        print(output)
        print(output.shape)

        break


if __name__ == '__main__':
    train()
