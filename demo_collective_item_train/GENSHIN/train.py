import torch
import os
from ultralytics import YOLO
import yaml
import numpy as np
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

import sys
import datetime

LOG_FILE = ".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d') + '\\' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"


class Console_Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def _clean_up():
    sys.stdin = sys.__stdin__    # 重置标准输入
    sys.stdout = sys.__stdout__   # 重置标准输出
    sys.stderr = sys.__stderr__   # 重置标准错误
    # sys.stdin.read()  # 读取并丢弃所有剩余输入
    # sys.stdout.flush()  # 强制刷新缓冲区
    # sys.stderr.flush()  # 强制刷新错误缓冲区

def exist_path(*args):
    path = os.path.join(*args)

    if(os.path.exists(path)):
        pass
    else:
        os.mkdir(path)
    return path

def Enable_Console_Logger():
    exist_path(".\\logs")
    exist_path(".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w") as fp:
            pass
    sys.stdout = Console_Logger(LOG_FILE, sys.stdout)
    pass

def main():
    """
    YOLOv11训练主函数
    针对目标与背景相近以及小目标检测进行优化
    """
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    Enable_Console_Logger()

    # 设置GPU设备（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置文件
    config_path = "GENSHIN.yaml"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 打印数据集信息
    print(f"加载数据集: {config.get('path', 'unknown')}")
    print(f"类别数量: {len(config.get('names', []))}")
    print(f"类别: {config.get('names', [])}")
    
    # 加载预训练模型 - 使用 YOLO11m 或 YOLO11s
    model_type = "yolo11m"  # 可以改为 "yolov11s"
    print(f"加载预训练模型: {model_type}")
    
    model = YOLO(f"{model_type}.pt")
    
    # 配置训练参数
    training_args = {
        'data': config_path,
        'epochs': 100,
        'patience': 15,          # 早停patience
        'batch': 4,             # 批次大小，根据GPU内存调整
        'imgsz': 768,            # 图像尺寸
        'device': 0 if torch.cuda.is_available() else 'cpu', # 设备ID
        'workers': 8,            # 数据加载线程数
        'optimizer': 'AdamW',    # 优化器类型
        'lr0': 0.0005,            # 初始学习率
        'lrf': 0.01,             # 最终学习率 = lr0 * lrf
        'momentum': 0.937,       # SGD动量
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3.0,    # 预热轮数
        'warmup_momentum': 0.8,  # 预热动量
        'warmup_bias_lr': 0.1,   # 预热偏置学习率
        'box': 7.5,              # 边界框损失权重
        'cls': 0.7,              # 类别损失权重
        'dfl': 1.5,              # 分布式焦点损失权重
        'label_smoothing': 0.0,  # 标签平滑
        'nbs': 64,               # 标称批次大小
        'overlap_mask': True,    # 掩码重叠
        'mask_ratio': 4,         # 掩码下采样率
        'dropout': 0.2,          # 使用丢弃层正则化
        'val': True,             # 验证
        
        # 数据增强参数
        'hsv_h': 0.015,          # HSV色调增强
        'hsv_s': 0.9,            # HSV饱和度增强 - 增加以改善低对比度
        'hsv_v': 0.6,            # HSV值增强 - 增加以改善低对比度
        'degrees': 0.0,          # 旋转增强
        'translate': 0.1,        # 平移增强
        'scale': 0.8,            # 缩放增强 - 增加以改善小目标检测
        'shear': 0.0,            # 切变增强
        'perspective': 0.0,      # 透视增强
        'flipud': 0.0,           # 上下翻转概率
        'fliplr': 0.5,           # 左右翻转概率
        'mosaic': 1.0,           # Mosaic增强概率
        'mixup': 0.2,            # Mixup增强概率
        'copy_paste': 0.0,       # 复制粘贴增强概率
    }
    
    # 打印训练参数摘要
    print("\n训练参数:")
    for k, v in training_args.items():
        print(f"  {k}: {v}")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(**training_args)
    
    # 评估训练后的模型
    print("\n评估模型...")
    results = model.val()
    
    print(f"\n训练完成. 模型保存在 {os.path.join('runs', 'detect', 'train')}")
    _clean_up()

# 注意：YOLOv11已经内置了数据增强功能
# 如果需要自定义数据增强，可以创建一个单独的预处理管道
# 以下代码暂时保留作为参考，但在当前脚本中不使用

def create_custom_dataset(data_path, annotations_path, transform=None):
    """
    创建自定义数据集，可用于应用额外的数据增强
    注意：当前不在主流程中使用，仅作为参考
    """
    from torch.utils.data import Dataset
    
    class CustomDataset(Dataset):
        def __init__(self, data_path, annotations_path, transform=None):
            self.data_path = data_path
            self.annotations = self._load_annotations(annotations_path)
            self.transform = transform
            
        def _load_annotations(self, annotations_path):
            # 加载标注文件的逻辑
            return []
            
        def __getitem__(self, idx):
            # 加载图片和标注
            img_path = self.annotations[idx]['img_path']
            boxes = self.annotations[idx]['boxes']
            labels = self.annotations[idx]['labels']
            
            # 读取图像
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            
            # 应用数据增强
            if self.transform:
                augmented = self.transform(image=image_np, bboxes=boxes, labels=labels)
                image_np = augmented['image']
                boxes = augmented['bboxes']
                labels = augmented['labels']
            
            # 转换为YOLO所需格式
            # ...
            
            return image_np, boxes, labels
            
        def __len__(self):
            return len(self.annotations)
    
    return CustomDataset(data_path, annotations_path, transform)

# 注意：在YOLOv11中，我们通常不直接修改损失函数
# 而是通过配置参数来调整损失权重
# 以下是创建一个自定义模型配置文件的函数，可以用于优化小目标检测

def create_custom_model_config(output_path='custom_yolov11.yaml'):
    """
    创建自定义的YOLOv11模型配置，优化小目标检测
    此函数演示如何创建一个配置文件，不在主流程中使用
    """
    config = {
        'nc': 9,  # 类别数量，根据GENSHIN.yaml调整
        'depth_multiple': 1.0,  # 模型深度倍数
        'width_multiple': 1.0,  # 模型宽度倍数
        'scales': 'yolov11m',  # 使用YOLOv11m的缩放参数
        
        # 修改检测头，更关注小目标
        'head': [
            [-1, 1, 'Detect', ['nc', 'anchors']],
        ],
        
        # 自定义锚点设置，更适合小目标
        'anchors': [
            [10, 13, 16, 30, 33, 23],  # P3/8 - 小尺寸锚点
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]
    }
    
    # 将配置写入文件
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"自定义模型配置已保存到 {output_path}")
    return output_path

if __name__ == "__main__":
    main()