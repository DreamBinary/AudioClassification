# -*- coding:utf-8 -*-
# @FileName : load_dataset.py
# @Time : 2024/3/19 21:56
# @Author : fiv
from pathlib import Path
from torch import Tensor

from . import to_fbank

label2id = {
    "bird": 0,
    "cat": 1,
    "dog": 2,
    "tiger": 3
}


def load_dataset(dir_path: Path) -> (list, list):
    data_files = list(dir_path.glob("*.wav"))
    features = []
    labels = []
    for file in data_files:
        label = file.stem.split("_")[0]
        label_id = Tensor(label2id[label])
        feature = to_fbank(file)
        features.append(feature)
        labels.append(label_id)
    return features, labels

# if __name__ == '__main__':
#     from env import DATA_PATH
#     dataset_path = DATA_PATH / "animal" / "all"
#     x, y = load_dataset(dataset_path)
#     print(len(x), len(y))
#     print(x[0].shape, y[0].shape)
