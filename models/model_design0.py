#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.models import resnet18
from config import Model_Config as Config


# 改进的模型结构
class TemporalEnhancedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # # 空间特征提取（修改ResNet输入）
        # self.spatial_net = models.resnet18(pretrained=True)
        # # 修改第一层卷积接受6通道输入（RGB+光流）
        # self.spatial_net.conv1 = nn.Conv2d(6, 64, kernel_size=7, 
        #                                  stride=2, padding=3, bias=False)
        self.spatial_net = resnet18(pretrained=True)
        self.spatial_net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 适应小尺寸输入
        self.dropout2d = nn.Dropout2d(p=0.3)
        self.dropout = nn.Dropout(p=0.6)
        # 保留预训练权重（重要技巧）
        # with torch.no_grad():
        #     self.spatial_net.conv1.weight[:, :3] = self.spatial_net.conv1.weight[:, :3].clone()
        #     self.spatial_net.conv1.weight[:, 3:] = self.spatial_net.conv1.weight[:, :3].clone()
          # 权重初始化技巧（关键！）
        # with torch.no_grad():
            # 复制原3通道权重到新6通道（前3通道复制RGB权重，后3通道复制光流权重）
            # original_weight = self.spatial_net.conv1.weight[:, :3].clone()
            # self.spatial_net.conv1.weight = nn.Parameter(torch.cat([original_weight, original_weight], dim=1))

        # 替换全连接层
        # self.spatial_net.fc = nn.Identity()
        self.spatial_net.fc = nn.Linear(self.spatial_net.fc.in_features, 512)
        
        # 时序建模
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512, 
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )
        
        # 位置编码
        self.position_embed = nn.Parameter(
            torch.randn(1, Config.seq_length, 512)
        )
        
        # 多任务头
        self.key_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, Config.num_classes)
        )
        self.mouse_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    # @profile(precision=5)
    def forward(self, x):
        # x形状: [B, T, H, W, C]
        # B, T, H, W, C = x.shape
        
        # # 空间特征提取
        # spatial_feat = x.view(B*T, H, W, C).permute(0, 3, 1, 2)  # [B*T,C,H,W]
        # spatial_feat = self.spatial_net(spatial_feat)  # [B*T, 512]
        # spatial_feat = spatial_feat.view(B, T, -1)    # [B, T, 512]
        B, T, H, W, C = x.shape # B,T,C,H,W
        # logger.info(x.shape)
        # x = x.to(torch.float32)  # 新增类型转换
    
        # x_reshaped = x.view(-1, C, H, W).permute(0, 3, 1, 2)  # [B*T, C, H, W]
        # x_reshaped = x.view(-1, H, W, C).permute(0,3,1,2)
        # 直接调整维度顺序避免后续重塑
        x_reshaped = x.permute(0,1,3,4,2)  # B,T,H,W,C → B,T,C,H,W → 不需要额外重塑
        x_reshaped = x_reshaped.reshape(B*T, C, H, W)  # [B*T, C, H, W]

        # 添加尺寸调整（关键步骤）
        x_resized = F.interpolate(
            x_reshaped, 
            size=Config.target_size,  # 强制调整到ResNet兼容尺寸
            mode='bilinear'
        )

        # with torch.cuda.amp.autocast(enabled=True): 
        spatial_feat = self.spatial_net(x_resized)  # [B*T, 512]
        # logger.info(spatial_feat)
        spatial_feat = spatial_feat.view(B, T, -1)  # [B, T, 512]

        spatial_feat = self.dropout(spatial_feat)
        
        # 添加位置编码
        spatial_feat += self.position_embed
        
        # 时序建模
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, 512]
        temporal_feat = self.dropout(temporal_feat)
        
        # 多任务预测
        keys = self.key_head(temporal_feat)        # [B, T, 8]
        mouse = self.mouse_head(temporal_feat)     # [B, T, 2]
        
        del x,x_reshaped,x_resized,spatial_feat

        return torch.cat([keys, mouse], dim=-1)    # [B, T, 10]

# 自定义损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.SmoothL1Loss()
        
    def forward(self, pred, target):
        # 分割预测结果
        # print(pred, target)
        pred_keys = pred[:, :, :Config.num_classes]
        pred_mouse = pred[:, :, Config.num_classes:]
        
        # 分割目标
        target_keys = target[:, :, :Config.num_classes]
        target_mouse = target[:, :, Config.num_classes:]
        
        # 计算损失
        loss_key = self.key_loss(pred_keys, target_keys)
        loss_mouse = self.mouse_loss(pred_mouse, target_mouse)
        
        return 1.0 * loss_key + 0.5 * loss_mouse
