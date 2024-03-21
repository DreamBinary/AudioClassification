# -*- coding:utf-8 -*-
# @FileName : run.py
# @Time : 2024/3/21 16:48
# @Author : fiv

import torch

from dataset import get_dataloader
from env import MODEL_PATH
from eval import eval
from model import Transformer
from train import train


def run():
    train_dataloader, test_dataloader = get_dataloader()
    output_path = MODEL_PATH / "animal.pth"
    train(train_dataloader, output_path)
    model = Transformer()
    model.load_state_dict(torch.load(output_path))
    eval(model, test_dataloader)


if __name__ == '__main__':
    run()
