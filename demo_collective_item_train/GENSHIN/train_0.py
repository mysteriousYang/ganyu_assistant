# import torch
# from yolov5 import train, detect

# # 定义数据集和模型配置文件路径
# dataset_dir = "./GENSHIN/"
# model_config = "./yolov5s.yaml"

# # 训练模型
# train.run(data=dataset_dir + "data.yaml", cfg=model_config, epochs=100, batch_size=16, imgsz=640)

# # 测试模型
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
# detect.run(source=dataset_dir + "test/images", weights='runs/train/exp/weights/best.pt', imgsz=640, conf_thres=0.4, iou_thres=0.5)

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

if __name__ == '__main__':
	model = YOLO('yolo11m.yaml')   # 修改yaml
	model.load('yolo11m.pt')  #加载预训练权重
	model.train(data='.\\GENSHIN.yaml',   #数据集yaml文件
	            imgsz=512,
	            epochs=100,
	            batch=4,
	            workers=20,  
	            device=0,   #没显卡则将0修改为'cpu'
	            optimizer='Adam',
                amp = False,
	            cache=False,   #服务器可设置为True，训练速度变快
	)