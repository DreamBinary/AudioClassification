# -*- coding:utf-8 -*-
# @FileName : download.py
# @Time : 2024/3/21 19:31
# @Author : fiv


from huggingface_hub import hf_hub_download

hf_hub_download("437aewuh/dog-dataset", "main", local_dir="./data/dog")
