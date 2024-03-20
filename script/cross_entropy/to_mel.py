# -*- coding:utf-8 -*-
# @FileName : to_mel.py
# @Time : 2024/3/20 13:57
# @Author : fiv


from pathlib import Path
import librosa


def to_mel(wav_path: Path):
    wav, sr = librosa.load(wav_path, sr=None)
    mel = librosa.feature.melspectrogram(wav, sr=sr)
    return mel
