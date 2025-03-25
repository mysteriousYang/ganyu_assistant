import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from model import CustomModel

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数配置
config = {
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "log_dir": "./logs",
    "save_dir": "./checkpoints",
    "num_workers": 4,
}

# 1. 数据加载器实现
class GameActionDataset(Dataset):
    def __init__(self, data_root, seq_length=100, is_train=True):
        self.seq_length = seq_length
        # 假设数据存储在npz文件中
        self.data_files = [os.path.join(data_root, f) for f in os.listdir(data_root)]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # 模拟数据加载：实际应从文件读取
        # RGB帧: [3, 100, 480, 272]
        rgb = torch.randn(3, 100, 480, 272).float() 
        # 光流帧: [3, 100, 480, 272]
        flow = torch.randn(3, 100, 480, 272).float()
        # 标签: [100, 9] (前7位是按键，后2位是鼠标)
        labels = torch.randn(100, 9).float()  
        
        # 数据预处理
        rgb = self.normalize(rgb)
        flow = self.normalize(flow)
        labels[:, 7:] = self.scale_mouse(labels[:, 7:])
        
        return rgb, flow, labels
    
    def normalize(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    def scale_mouse(self, mouse):
        """将鼠标坐标标准化到[-1, 1]范围"""
        return (mouse / torch.tensor([480, 272])) * 2 - 1

# 2. 训练准备
# 初始化模型
model = CustomModel().to(device)
optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

# 创建数据加载器
train_dataset = GameActionDataset("./data/train")
train_loader = DataLoader(train_dataset, 
                         batch_size=config["batch_size"],
                         shuffle=True,
                         num_workers=config["num_workers"])

val_dataset = GameActionDataset("./data/val")
val_loader = DataLoader(val_dataset, 
                       batch_size=config["batch_size"],
                       num_workers=config["num_workers"])

# 3. 训练函数
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    key_correct = 0
    total_samples = 0
    
    for batch_idx, (rgb, flow, labels) in enumerate(loader):
        # 数据移至GPU
        rgb = rgb.to(device)
        flow = flow.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(rgb, flow)
        
        # 计算损失
        key_loss = nn.BCELoss()(outputs['keys'], labels[:, :, :7])
        mouse_loss = nn.HuberLoss()(outputs['mouse'], labels[:, :, 7:])
        loss = 0.6 * key_loss + 0.4 * mouse_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item()
        pred_keys = (outputs['keys'] > 0.5).float()
        key_correct += (pred_keys == labels[:, :, :7]).sum().item()
        total_samples += labels.size(0) * labels.size(1) * 7
        
        # 每10个batch打印日志
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    key_acc = key_correct / total_samples
    return avg_loss, key_acc

# 4. 验证函数
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    key_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for rgb, flow, labels in loader:
            rgb = rgb.to(device)
            flow = flow.to(device)
            labels = labels.to(device)
            
            outputs = model(rgb, flow)
            
            # 计算损失
            key_loss = nn.BCELoss()(outputs['keys'], labels[:, :, :7])
            mouse_loss = nn.HuberLoss()(outputs['mouse'], labels[:, :, 7:])
            loss = 0.6 * key_loss + 0.4 * mouse_loss
            
            total_loss += loss.item()
            pred_keys = (outputs['keys'] > 0.5).float()
            key_correct += (pred_keys == labels[:, :, :7]).sum().item()
            total_samples += labels.size(0) * labels.size(1) * 7
    
    avg_loss = total_loss / len(loader)
    key_acc = key_correct / total_samples
    return avg_loss, key_acc

# 5. 主训练循环
if __name__ == "__main__":
    writer = SummaryWriter(config["log_dir"])
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(config["save_dir"], f"best_model_epoch{epoch}.pth"))
        
        # 打印epoch结果
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}\n")
    
    writer.close()