#-*- coding: utf-8 -*-
import torch
import datetime

SCREENSHOT_DIR = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\capture\\capture_0108"
CAPTURE_ROOT = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\capture"

DEBUG = True


class Dataset_Config:
    # ====== 视频设置 ======
    video_enc = "mp4v"
    flow_video_enc = "mp4v"

    video_suffix = "mp4"
    flow_video_suffix = "mp4"

    default_s_width = 480
    default_s_height = 272

    # ====== 数据集设置 ======
    learn_keys = ['w','s','a','d','space']

    BLOCK_SIZE = 4
    
# 超参数配置
class Model_Config:
    # 数据参数
    train_rate = 0.8
    val_rate = 0.1

    input_shape = (480, 272)  # 原始输入尺寸
    target_size = (224, 224)  # ResNet输入尺寸
    # num_classes = len(Dataset_Config.learn_keys)
    num_classes = 5
    seq_length = Dataset_Config.BLOCK_SIZE
    
    # 训练参数
    batch_size = 1
    lr = 1e-4
    dropout = 0.5
    epochs = 20
    running_epoch = 0
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    checkpoint_dir = ""
    dataset_path = "G:\\NN_train\\debug"

    def checkpoint_path(epoch):
        checkpoint_path =  f"{Model_Config.checkpoint_dir}\\" + datetime.datetime.now().strftime('%Y-%m-%d') + f"e{epoch}.pth"
        return checkpoint_path

    def best_model_path():
        best_model_path = f"{Model_Config.checkpoint_dir}\\best_model.pth"
        return best_model_path