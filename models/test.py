#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from models.model_design0 import CustomModel
from models.train import validate
from dataset import Make_Dataset
from dataloader import Make_Dataloader

# 这个文件暂时还没用，不管它

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model_path, test_dir):
    # 加载训练好的模型
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建测试数据集
    test_dataset = Make_Dataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    total_loss, key_acc = validate(model, test_loader)
    print(f"\nFinal Test Results:")
    print(f"Loss: {total_loss:.4f} | Key Accuracy: {key_acc:.2%}")
    
    # 可视化样例预测
    with torch.no_grad():
        rgb, flow, labels = next(iter(test_loader))
        outputs = model(rgb.to(device), flow.to(device))
        
        # 可视化第一个样本的预测
        plot_predictions(outputs, labels)

def plot_predictions(outputs, labels, idx=0):
    import matplotlib.pyplot as plt
    
    keys_pred = outputs['keys'][idx].cpu().numpy()
    keys_true = labels[idx, :, :7].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    # 绘制按键预测
    plt.subplot(2,1,1)
    for i in range(7):
        plt.plot(keys_pred[:,i], label=f"Key {i} Pred")
        plt.plot(keys_true[:,i], '--', label=f"Key {i} True")
    plt.title("Key Press Prediction")
    
    # 绘制鼠标轨迹
    plt.subplot(2,1,2)
    mouse_pred = outputs['mouse'][idx].cpu().numpy()
    mouse_true = labels[idx, :, 7:].cpu().numpy()
    plt.plot(mouse_pred[:,0], mouse_pred[:,1], 'r-', label="Predicted Path")
    plt.plot(mouse_true[:,0], mouse_true[:,1], 'b--', label="True Path")
    plt.title("Mouse Trajectory")
    plt.legend()
    plt.show()

# 使用示例
if __name__ == "__main__":
    test_model("./checkpoints/best_model.pth", "./data/test")