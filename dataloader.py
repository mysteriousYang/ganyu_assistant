#-*- coding: utf-8 -*-
import os
import json
import logging
import math
import cv2
import torch.nn
import argparse
import queue
import gc
import subprocess

import numpy as np

from typing import List
from torch.utils.data import DataLoader
from dataset import Make_Dataset


def Make_Dataloader(
        dataset,
        batch_size=32,  # 每批次的大小
        shuffle=True,   # 是否打乱数据
        num_workers=0,  # 使用 4 个线程加载数据
        pin_memory=True # 使用GPU内存
        ):
    if(isinstance(dataset,str)):
        GS_dataset = Make_Dataset(dataset)
    else:
        GS_dataset = dataset
    GS_dl = DataLoader(
        GS_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    return GS_dl

if __name__ == "__main__":
    pass