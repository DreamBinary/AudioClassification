# -*- coding:utf-8 -*-
# @FileName : dataset.py
# @Time : 2024/3/20 16:31
# @Author : fiv
import torch
from torch.utils.data import Dataset, DataLoader

from script.preprocess import to_fbank


class AnimalDataset(Dataset):
    def __init__(self, dataset_dir=None):
        self.label2idx = {"bird": 0, "cat": 1, "dog": 2, "tiger": 3}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        if dataset_dir is None:
            from env import DATA_PATH
            self.dataset_dir = DATA_PATH / "animal" / "all"
        else:
            self.dataset_dir = dataset_dir
        self.file_path = list(self.dataset_dir.glob("*.wav"))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        fbank = to_fbank(self.file_path[idx])
        if fbank.shape[0] < 1000:
            padding = torch.zeros(1000 - fbank.shape[0], 40)
            fbank = torch.cat([fbank, padding], dim=0)
        else:
            start = torch.randint(0, fbank.shape[0] - 1000, (1,))
            fbank = fbank[start:start + 1000]
        label = self.file_path[idx].stem.split("_")[0]
        return fbank, self.label2idx[label]

    def shuffle(self):
        import random
        random.shuffle(self.file_path)

    def idx2label(self, idx):
        return self.idx2label[idx]


def get_dataloader(dataset_dir=None, batch_size=1, shuffle=True):
    if dataset_dir is None:
        from env import DATA_PATH
        dataset_dir = DATA_PATH / "animal"

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    train = AnimalDataset(train_dir)
    test = AnimalDataset(test_dir)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


if __name__ == '__main__':
    dataloader = get_dataloader()
    for x, y in dataloader:
        print(x.shape, y)
