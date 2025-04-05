#-*- coding: utf-8 -*-
import json
import os
import datetime
from config import Model_Config as Config
from utils.file_utils import exist_path
from utils.logger import logger

def dump_config_and_network(model):
    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
    config_dict["device"] = str(config_dict["device"])  # 转换设备信息
    config_dict["checkpoint_path"] = "checkpoint_path function"
    config_dict["best_model_path"] = "best_model_path function"
    json_str = json.dumps(config_dict, indent=4)
    with open(f"{Config.checkpoint_dir}\\para_and_model.txt",mode='w',encoding='utf-8') as fp:
        fp.write(json_str)
        fp.write("\n\n\n")
        fp.write(str(model))
    pass


def check_paths():
    '''
    检测路径是否存在
    '''
    paths = [
        ".\\models",".\\models\\checkpoints"
    ]

    for path in paths: exist_path(path)


def create_train_folder(base_path=f".\\models\\checkpoints\\{datetime.datetime.now().strftime('%Y-%m-%d')}"):
    # 确保基础目录存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)  # 

    # 获取所有以'train_'开头的文件夹
    existing_folders = [f for f in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, f)) and f.startswith("train_")]

    # 提取现有最大编号
    max_num = -1
    for folder in existing_folders:
        try:
            num = int(folder.split("_")[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            continue  # 跳过名称不规范的文件夹

    # 生成新编号
    new_num = max_num + 1
    new_folder = f"train_{new_num}"
    new_path = os.path.join(base_path, new_folder)

    # 创建新目录
    os.makedirs(new_path, exist_ok=True)  # 
    logger.info(f"创建文件夹: {new_path}")
    return new_path