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

def collate_fn(batch):
    """处理 (x, y) 元组列表的批处理函数"""
    try:
        # 分离x和y并转换为Tensor（零拷贝）
        x_list, y_list = zip(*batch)  # 解压元组列表
        
        # 转换x批次（共享内存）
        x_batch = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in x_list])
        
        # 转换y批次（避免冗余转置）
        y_batch = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in y_list])
        
        # 调整维度顺序 (如果原始数据是NHWC格式)
        x_batch = x_batch.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 显式释放临时变量 (重要！防止内存泄漏)
        del batch, x_list, y_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return x_batch, y_batch
        
    except Exception as e:
        # 错误处理与日志记录
        error_info = f"批处理失败: {str(e)}\n输入数据形状示例:"
        for i, (x, y) in enumerate(batch):
            error_info += f"\n样本{i} x.shape={x.shape} y.shape={y.shape}"
        raise RuntimeError(error_info)

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
        # collate_fn=collate_fn,
        pin_memory=pin_memory)
    
    return GS_dl

if __name__ == "__main__":
    pass