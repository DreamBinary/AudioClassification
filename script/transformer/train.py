# -*- coding:utf-8 -*-
# @FileName : train.py
# @Time : 2024/3/20 16:13
# @Author : fiv
import torch
from torch import nn
from tqdm import tqdm

from dataset import get_dataloader
from env import MODEL_PATH
from model import Transformer


#
def train(dataloader, output_path=MODEL_PATH / "animal.pth"):
    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    total_run = 1000
    pbar = tqdm(range(total_run))
    min_loss = 100
    for _ in pbar:
        loss = 0
        dataloader.dataset.shuffle()
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = criterion(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss += loss.item()
        loss = loss / len(dataloader)
        pbar.set_description(f"loss: {loss:.4f}")
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), output_path)
    print(f"Min loss: {min_loss}")


if __name__ == '__main__':
    train_loader, _ = get_dataloader()
    train(train_loader)
