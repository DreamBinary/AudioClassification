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
        # fbank.shape: [138, 40]
        print(fbank.shape)

        if fbank.shape[0] < 1000:
            padding = torch.zeros(1000 - fbank.shape[0], 40)
            fbank = torch.cat([fbank, padding], dim=1)
        else:
            # random select 1000 frames
            start = torch.randint(0, fbank.shape[0] - 1000, (1,))

        print("fbank.shape:", fbank.shape)

        # fill padding if len(fbank) < 1000
        # if fbank.shape[1] < 1000:
        #     padding = torch.zeros(40, 1000 - fbank.shape[1])
        #     fbank = torch.cat([fbank, padding], dim=1)
        # else:
        #     fbank = fbank[:, :1000]

        label = self.file_path[idx].stem.split("_")[0]
        return fbank, self.label2idx[label]

    def shuffle(self):
        import random
        random.shuffle(self.file_path)

    def idx2label(self, idx):
        return self.idx2label[idx]


def get_dataloader(dataset_dir=None, batch_size=1, shuffle=True):
    dataset = AnimalDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()
    for x, y in dataloader:
        print(x.shape, y)
