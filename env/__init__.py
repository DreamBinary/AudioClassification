# -*- coding:utf-8 -*-
# @FileName : __init__.py.py
# @Time : 2024/1/28 16:35
# @Author :fiv
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = PROJECT_PATH / "data"
OUTPUT_PATH = PROJECT_PATH / "output"
MODEL_PATH = PROJECT_PATH / "output" / "model"


def check_exist():
    for path in [DATA_PATH, OUTPUT_PATH]:
        if not path.exists():
            path.mkdir(parents=True)


check_exist()
