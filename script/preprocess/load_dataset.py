# -*- coding:utf-8 -*-
# @FileName : load_dataset.py
# @Time : 2024/3/19 21:56
# @Author : fiv
from pathlib import Path
import librosa
import numpy as np


def load_dataset(dir_path: Path) -> (np.ndarray, np.ndarray):
    data_files = list(dir_path.glob("*.wav"))
    features = []
    labels = []
    for file in data_files:
        label = file.stem.split("_")[0]

        x, sr = librosa.load(file, sr=None)
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features.append(mfccs_scaled)
        labels.append(label)

    return np.array(features), np.array(labels)


# from env import DATA_PATH
#
# print(load_dataset(DATA_PATH / "animal" / "all"))
