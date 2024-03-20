# -*- coding:utf-8 -*-
# @FileName : svc.py
# @Time : 2024/3/19 21:07
# @Author : fiv

from env import DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import librosa
import numpy as np
from pathlib import Path


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


def train(x, y):
    model = SVC(kernel="linear")
    model.fit(x, y)
    return model


def eval(model, x, y):
    acc = model.score(x, y)
    return acc


def test(model, dataset_path):
    # random one
    import random
    import librosa
    import numpy as np
    files = list(dataset_path.glob("*.wav"))
    a_sound = files[random.randint(0, len(files))]
    x, sr = librosa.load(a_sound, sr=None)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    print(a_sound.stem, model.predict([mfccs_scaled]))


if __name__ == '__main__':
    dataset_path = DATA_PATH / "animal" / "all"
    x, y = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = train(x_train, y_train)
    acc = eval(model, x_test, y_test)
    print(f"Accuracy: {acc}")

    test(model, dataset_path)
