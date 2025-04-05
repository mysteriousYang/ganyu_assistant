import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import logger
from tqdm import tqdm
import os
from torchvision.models.video import r3d_18
from torchvision.models.video.resnet import BasicBlock
from collections.abc import Iterable
# import wandb
# from typing import Tuple, List, Dict, Optional

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置参数
class Config:
    def __init__(self):
        # 数据参数
        self.batch_size = 1
        self.time_steps = 4  # 时间序列长度
        self.height = 272  # 视频高度
        self.width = 480  # 视频宽度
        self.channels = 3  # 输入通道数
        self.key_num = 5  # 按键数量 (W, S, A, D, Space)
        
        # 模型参数
        self.hidden_dim = 256
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.num_epochs = 10
        self.early_stop_patience = 10
        
        # 训练参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.save_dir = './checkpoints'
        self.use_wandb = True
        
        # 损失函数权重
        self.key_loss_weight = 1.0
        self.mouse_loss_weight = 1.0
        
        # os.makedirs(self.save_dir, exist_ok=True)

# 数据集类
class GameInputDataset(Dataset):
    def __init__(self, flow_videos: np.ndarray, inputs: np.ndarray, transform=None):
        """
        初始化数据集
        Args:
            flow_videos: 光流视频，形状为 [N, T, H, W, C]
            inputs: 输入序列，形状为 [N, T, key_num + 2]，包含时序按键状态和鼠标位置
            transform: 数据增强转换
        """
        self.flow_videos = flow_videos
        self.inputs = inputs
        self.transform = transform
    
    def __len__(self):
        return len(self.flow_videos)
    
    def __getitem__(self, idx):
        video = self.flow_videos[idx]  # [T, H, W, C]
        label = self.inputs[idx]  # [T, key_num + 2]
        
        # 将 HWC 转换为 CHW 格式，并归一化到 [0, 1]
        video = video.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        
        # 数据增强
        if self.transform:
            # 应用到每一帧
            transformed_frames = []
            for t in range(video.shape[0]):
                frame = video[t]  # [C, H, W]
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            video = np.stack(transformed_frames)
        
        # 分离按键和鼠标位置 (保持时间维度)
        T = label.shape[0]
        key_num = Config().key_num
        key_states = label[:, :key_num].astype(np.float32)
        mouse_pos = label[:, key_num:].astype(np.float32)
        
        return torch.FloatTensor(video), torch.FloatTensor(key_states), torch.FloatTensor(mouse_pos)

# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048, dropout=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

# Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, hidden_dim=2048, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

# 3D-CNN + Transformer 模型 (时序预测版本)
class GameInputPredictor(nn.Module):
    def __init__(self, config=Config()):
        super(GameInputPredictor, self).__init__()
        self.config = config
        
        # 使用预训练的R3D-18作为3D-CNN骨干网络
        # claude
        '''
        r3d = r3d_18(pretrained=True)
        self.backbone = nn.Sequential(*list(r3d.children())[:-2])  # 去掉最后的全连接层
        '''

        # DS
        r3d = r3d_18(pretrained=True)
        # 遍历并修改所有3D卷积层的stride
        for layer in r3d.children():
            # logger.debug(layer)
            # logger.debug(type(layer))
            self._set_time_stride(layer)
        # logger.debug(list(r3d.children()))
        self.backbone = nn.Sequential(*list(r3d.children())[:-2])

        
        # 3D-CNN输出通道数
        self.output_channels = 512
        
        # 特征映射尺寸
        self.feature_size = self._calculate_feature_size()
        
        # Transformer编码器
        self.embed_dim = 256
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.time_steps, self.embed_dim))
        self.embedding_projection = nn.Linear(self.output_channels, self.embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, num_heads=8, hidden_dim=1024, dropout=config.dropout)
            for _ in range(2)  # 2层Transformer编码器
        ])
        
        # 输出头 (时序预测)
        self.fc_common = nn.Linear(self.embed_dim, 128)
        
        # 按键预测头
        self.key_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.key_num),
            nn.Sigmoid()  # 多标签分类，使用Sigmoid
        )
        
        # 鼠标位置预测头
        self.mouse_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2),
            nn.Sigmoid()  # 归一化坐标，使用Sigmoid
        )
        
        # 循环神经网络用于时序建模
        self.rnn = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )
        
        # 融合双向RNN
        self.rnn_fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)
    
    def _set_time_stride(self, component):
        # 把所有Conv3D的时间采样层设为1，防止时间维度压缩
        # logger.debug(type(component))
        if(isinstance(component, Iterable)):
            for elem in component:
                self._set_time_stride(elem)

        elif(isinstance(component, BasicBlock)):
            for name, module in component.named_children():
                self._set_time_stride(module)
                # logger.debug(component)

        else:
            if isinstance(component, nn.Conv3d):
                # component.stride = (1, 2, 2)  # 时间维度stride=1，空间维度stride=2
                component.stride = (1, *component.stride[1:]) #修改时间stride为1
                # logger.debug(component)

    def _calculate_feature_size(self):
        # 计算3D-CNN输出的特征尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.config.time_steps, self.config.height, self.config.width)
            output = self.backbone(dummy_input)
            return output.size()
    
    def forward(self, x):
        # x shape: [batch_size, T, C, H, W]
        batch_size, t,  h, w, c = x.size()
        
        # 重塑为3D-CNN输入格式 [batch_size, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3)
        
        # 3D-CNN特征提取
        features = self.backbone(x)  # [batch_size, channels, t', h', w']
        
        # 投影到Transformer的嵌入维度
        features = features.permute(0, 2, 1, 3, 4)  # [batch_size, t', channels, h', w']
        batch_size, t_prime, channels, h_prime, w_prime = features.shape
        # logger.debug(features.shape)
        
        # 空间平均池化
        features = F.adaptive_avg_pool2d(features.reshape(-1, channels, h_prime, w_prime), (1, 1))
        features = features.view(batch_size, t_prime, channels)
        
        # 投影到嵌入空间
        embedded = self.embedding_projection(features)
        
        # 添加位置编码
        embedded = embedded + self.positional_encoding[:, :t_prime, :]
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            embedded = layer(embedded)
        
        # 通过RNN进行时序建模
        rnn_out, _ = self.rnn(embedded)
        
        # 融合双向信息
        rnn_out = self.rnn_fusion(rnn_out)
        
        # 应用共享特征层 (保持时序维度)
        common = F.relu(self.fc_common(rnn_out))  # [batch_size, t_prime, 128]
        
        # 预测每个时间步的按键和鼠标位置
        # Claude
        '''
        key_preds = []
        mouse_preds = []
        
        for t in range(t_prime):
            time_features = common[:, t, :]  # [batch_size, 128]
            key_pred = self.key_predictor(time_features)  # [batch_size, key_num]
            mouse_pred = self.mouse_predictor(time_features)  # [batch_size, 2]
            
            key_preds.append(key_pred.unsqueeze(1))  # 添加时间维度
            mouse_preds.append(mouse_pred.unsqueeze(1))
        
        # 拼接所有时间步的预测结果
        key_preds = torch.cat(key_preds, dim=1)  # [batch_size, t_prime, key_num]
        mouse_preds = torch.cat(mouse_preds, dim=1)  # [batch_size, t_prime, 2]

        return key_preds, mouse_preds
        '''
        ###########

        # DS
        key_preds = []
        mouse_preds = []
        
        for t in range(t_prime):
            time_features = common[:, t, :]
            key_pred = self.key_predictor(time_features)
            mouse_pred = self.mouse_predictor(time_features)
            
            key_preds.append(key_pred.unsqueeze(1))
            mouse_preds.append(mouse_pred.unsqueeze(1))
        
        key_preds = torch.cat(key_preds, dim=1)    # [B, T, num_keys]
        mouse_preds = torch.cat(mouse_preds, dim=1)  # [B, T, 2]
        
        # 沿最后一个维度拼接按键和鼠标预测结果
        combined_output = torch.cat([key_preds, mouse_preds], dim=2)  # [B, T, num_keys+2]
        # logger.debug(combined_output)
        # logger.debug(combined_output.shape)
        
        return combined_output
        

# 基于3D ResNet的模型 (时序预测版本)
class ResNet3DModel(nn.Module):
    def __init__(self, config):
        super(ResNet3DModel, self).__init__()
        self.config = config
        
        # 使用预训练的3D ResNet
        self.backbone = r3d_18(pretrained=True)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原始全连接层
        
        # 时序建模
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )
        
        # 融合层
        self.fusion = nn.Linear(256 * 2, 256)
        self.dropout = nn.Dropout(config.dropout)
        
        # 按键预测头
        self.key_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.key_num),
            nn.Sigmoid()
        )
        
        # 鼠标位置预测头
        self.mouse_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, T, C, H, W]
        batch_size, t, c, h, w = x.size()
        
        # 重塑为3D-CNN输入格式 [batch_size, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        # 提取帧级特征
        features = []
        
        # 处理每个时间步的输入
        for i in range(t):
            # 获取当前时间步的帧
            frame = x[:, :, i:i+1, :, :]  # [batch_size, C, 1, H, W]
            
            # 使用3D ResNet提取特征
            with torch.no_grad():
                feat = self.backbone.stem(frame)
                feat = self.backbone.layer1(feat)
                feat = self.backbone.layer2(feat)
                feat = self.backbone.layer3(feat)
                feat = self.backbone.layer4(feat)
            
            # 全局池化
            feat = F.adaptive_avg_pool3d(feat, (1, 1, 1)).view(batch_size, -1)
            features.append(feat.unsqueeze(1))  # 添加时间维度
        
        # 拼接所有时间步的特征
        features = torch.cat(features, dim=1)  # [batch_size, T, features]
        
        # 通过GRU进行时序建模
        gru_out, _ = self.gru(features)
        
        # 融合双向GRU的输出
        fused = self.fusion(gru_out)
        fused = self.dropout(fused)
        
        # 对每个时间步进行预测
        key_preds = []
        mouse_preds = []
        
        for t in range(fused.size(1)):
            # 获取当前时间步的特征
            time_features = fused[:, t, :]
            
            # 预测按键和鼠标位置
            key_pred = self.key_head(time_features)
            mouse_pred = self.mouse_head(time_features)
            
            key_preds.append(key_pred.unsqueeze(1))
            mouse_preds.append(mouse_pred.unsqueeze(1))
        
        # 拼接所有时间步的预测结果
        key_preds = torch.cat(key_preds, dim=1)  # [batch_size, T, key_num]
        mouse_preds = torch.cat(mouse_preds, dim=1)  # [batch_size, T, 2]
        
        return key_preds, mouse_preds

# 损失函数
class GameInputLoss(nn.Module):
    def __init__(self, key_weight=1.0, mouse_weight=1.0):
        super(GameInputLoss, self).__init__()
        self.key_weight = key_weight
        self.mouse_weight = mouse_weight
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, key_pred, mouse_pred, key_true, mouse_true):
        """
        计算按键和鼠标位置的损失
        Args:
            key_pred: 预测的按键状态，形状为 [batch_size, time_steps, key_num]
            mouse_pred: 预测的鼠标位置，形状为 [batch_size, time_steps, 2]
            key_true: 真实的按键状态，形状为 [batch_size, time_steps, key_num]
            mouse_true: 真实的鼠标位置，形状为 [batch_size, time_steps, 2]
        Returns:
            total_loss: 总损失
            key_loss: 按键分类损失
            mouse_loss: 鼠标回归损失
        """
        # 处理维度不匹配的情况
        if key_pred.size(1) != key_true.size(1):
            # 如果预测的时间步与真实的时间步不匹配，进行插值或裁剪
            pred_steps = key_pred.size(1)
            true_steps = key_true.size(1)
            
            if pred_steps > true_steps:
                # 裁剪预测结果
                key_pred = key_pred[:, :true_steps, :]
                mouse_pred = mouse_pred[:, :true_steps, :]
            else:
                # 裁剪真实标签
                key_true = key_true[:, :pred_steps, :]
                mouse_true = mouse_true[:, :pred_steps, :]
        
        # 计算每个时间步的按键损失
        key_loss_per_step = self.bce_loss(key_pred, key_true)  # [batch_size, time_steps, key_num]
        key_loss = key_loss_per_step.mean()  # 平均所有维度
        
        # 计算每个时间步的鼠标位置损失
        mouse_loss_per_step = self.mse_loss(mouse_pred, mouse_true)  # [batch_size, time_steps, 2]
        mouse_loss = mouse_loss_per_step.mean()  # 平均所有维度
        
        # 鼠标位置损失可以添加平滑L1损失以更好地处理异常值
        smooth_l1_loss = F.smooth_l1_loss(mouse_pred, mouse_true, reduction='mean')
        mouse_loss = 0.5 * mouse_loss + 0.5 * smooth_l1_loss
        
        # 计算总损失
        total_loss = self.key_weight * key_loss + self.mouse_weight * mouse_loss
        
        return total_loss, key_loss, mouse_loss

# 带时间权重的损失函数 - 给更新的时间步更高的权重
class TimeWeightedGameInputLoss(nn.Module):
    def __init__(self, key_weight=1.0, mouse_weight=1.0, time_decay=0.9):
        super(TimeWeightedGameInputLoss, self).__init__()
        self.key_weight = key_weight
        self.mouse_weight = mouse_weight
        self.time_decay = time_decay  # 时间衰减因子，越靠后的时间步权重越高
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, key_pred, mouse_pred, key_true, mouse_true):
        # 确保维度匹配
        time_steps = min(key_pred.size(1), key_true.size(1))
        key_pred = key_pred[:, :time_steps, :]
        key_true = key_true[:, :time_steps, :]
        mouse_pred = mouse_pred[:, :time_steps, :]
        mouse_true = mouse_true[:, :time_steps, :]
        
        # 计算时间权重 [1, time_steps, 1]
        device = key_pred.device
        time_weights = torch.tensor(
            [self.time_decay ** (time_steps - i - 1) for i in range(time_steps)],
            device=device
        ).view(1, -1, 1)
        
        # 按键损失
        key_loss_per_step = self.bce_loss(key_pred, key_true)  # [batch_size, time_steps, key_num]
        key_loss_weighted = (key_loss_per_step * time_weights).mean()
        
        # 鼠标位置损失
        mouse_loss_per_step = self.mse_loss(mouse_pred, mouse_true)  # [batch_size, time_steps, 2]
        mouse_loss_weighted = (mouse_loss_per_step * time_weights).mean()
        
        # 添加平滑L1损失
        smooth_l1_per_step = F.smooth_l1_loss(mouse_pred, mouse_true, reduction='none')
        smooth_l1_weighted = (smooth_l1_per_step * time_weights).mean()
        
        # 最终鼠标损失
        mouse_loss = 0.5 * mouse_loss_weighted + 0.5 * smooth_l1_weighted
        
        # 总损失
        total_loss = self.key_weight * key_loss_weighted + self.mouse_weight * mouse_loss
        
        return total_loss, key_loss_weighted, mouse_loss


# 自定义损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.key_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.SmoothL1Loss()
        self.config=config
        
    def forward(self, pred, target):
        # 分割预测结果
        # print(pred, target)
        pred_keys = pred[:, :, :self.config.key_num]
        pred_mouse = pred[:, :, self.config.key_num:]
        
        # 分割目标
        target_keys = target[:, :, :self.config.key_num]
        target_mouse = target[:, :, self.config.key_num:]
        
        # 计算损失
        loss_key = self.key_loss(pred_keys, target_keys)
        loss_mouse = self.mouse_loss(pred_mouse, target_mouse)
        
        return self.config.key_loss_weight * loss_key + self.config.mouse_loss_weight * loss_mouse


# 评估函数
def evaluate(model, test_loader, criterion, config):
    model.eval()
    test_losses = []
    test_key_losses = []
    test_mouse_losses = []
    
    key_preds_all = []
    key_trues_all = []
    mouse_preds_all = []
    mouse_trues_all = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            video, key_states, mouse_pos = batch
            video = video.to(config.device)
            key_states = key_states.to(config.device)
            mouse_pos = mouse_pos.to(config.device)
            
            # 前向传播
            key_pred, mouse_pred = model(video)
            
            # 计算损失
            loss, key_loss, mouse_loss = criterion(key_pred, mouse_pred, key_states, mouse_pos)
            
            # 记录损失
            test_losses.append(loss.item())
            test_key_losses.append(key_loss.item())
            test_mouse_losses.append(mouse_loss.item())
            
            # 收集预测值和真实值（用于最后一个时间步的评估）
            key_preds_all.append(key_pred[:, -1, :].cpu().numpy())
            key_trues_all.append(key_states[:, -1, :].cpu().numpy())
            mouse_preds_all.append(mouse_pred[:, -1, :].cpu().numpy())
            mouse_trues_all.append(mouse_pos[:, -1, :].cpu().numpy())
    
    # 计算平均测试损失
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_key_loss = sum(test_key_losses) / len(test_key_losses)
    avg_test_mouse_loss = sum(test_mouse_losses) / len(test_mouse_losses)
    
    print(f"Test Loss: {avg_test_loss:.4f}, "
          f"Test Key Loss: {avg_test_key_loss:.4f}, "
          f"Test Mouse Loss: {avg_test_mouse_loss:.4f}")
    
    # 转换为numpy数组
    key_preds = np.concatenate(key_preds_all)
    key_trues = np.concatenate(key_trues_all)
    mouse_preds = np.concatenate(mouse_preds_all)
    mouse_trues = np.concatenate(mouse_trues_all)
    
    # 计算按键预测精度 (二分类)
    key_preds_binary = (key_preds > 0.5).astype(int)
    key_trues_binary = key_trues.astype(int)
    key_accuracy = np.mean(key_preds_binary == key_trues_binary)
    
    # 计算每个按键的精度
    key_names = ["W", "S", "A", "D", "Space"]
    for i, key in enumerate(key_names):
        accuracy = np.mean(key_preds_binary[:, i] == key_trues_binary[:, i])
        print(f"{key} Accuracy: {accuracy:.4f}")
    
    # 计算鼠标位置预测的平均欧几里得距离
    mouse_dist = np.sqrt(np.sum((mouse_preds - mouse_trues) ** 2, axis=1))
    avg_mouse_dist = np.mean(mouse_dist)
    
    print(f"Overall Key Accuracy: {key_accuracy:.4f}")
    print(f"Average Mouse Distance: {avg_mouse_dist:.4f}")
    
    return avg_test_loss, key_accuracy, avg_mouse_dist

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 初始化wandb
    # if config.use_wandb:
    #     wandb.init(project="game-input-prediction", config=config.__dict__)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_key_losses = []
        train_mouse_losses = []
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            video, key_states, mouse_pos = batch
            video = video.to(config.device)
            key_states = key_states.to(config.device)
            mouse_pos = mouse_pos.to(config.device)
            
            # 前向传播
            key_pred, mouse_pred = model(video)
            
            # 计算损失
            loss, key_loss, mouse_loss = criterion(key_pred, mouse_pred, key_states, mouse_pos)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(loss.item())
            train_key_losses.append(key_loss.item())
            train_mouse_losses.append(mouse_loss.item())
            
            # 更新进度条
            pbar.set_description(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_key_loss = sum(train_key_losses) / len(train_key_losses)
        avg_train_mouse_loss = sum(train_mouse_losses) / len(train_mouse_losses)
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_key_losses = []
        val_mouse_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                video, key_states, mouse_pos = batch
                video = video.to(config.device)
                key_states = key_states.to(config.device)
                mouse_pos = mouse_pos.to(config.device)
                
                # 前向传播
                key_pred, mouse_pred = model(video)
                
                # 计算损失
                loss, key_loss, mouse_loss = criterion(key_pred, mouse_pred, key_states, mouse_pos)
                
                # 记录损失
                val_losses.append(loss.item())
                val_key_losses.append(key_loss.item())
                val_mouse_losses.append(mouse_loss.item())
        
        # 计算平均验证损失
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_key_loss = sum(val_key_losses) / len(val_key_losses)
        avg_val_mouse_loss = sum(val_mouse_losses) / len(val_mouse_losses)
        
        # 学习率调度器
        scheduler.step(avg_val_loss)
        
        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{config.num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Key Loss: {avg_train_key_loss:.4f}, "
              f"Train Mouse Loss: {avg_train_mouse_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Key Loss: {avg_val_key_loss:.4f}, "
              f"Val Mouse Loss: {avg_val_mouse_loss:.4f}")
        
        # 记录到wandb
        # if config.use_wandb:
        #     wandb.log({
        #         "epoch": epoch + 1,
        #         "train_loss": avg_train_loss,
        #         "train_key_loss": avg_train_key_loss,
        #         "train_mouse_loss": avg_train_mouse_loss,
        #         "val_loss": avg_val_loss,
        #         "val_key_loss": avg_val_key_loss,
        #         "val_mouse_loss": avg_val_mouse_loss,
        #         "learning_rate": optimizer.param_groups[0]['lr']
        #     })
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"模型已保存至 {os.path.join(config.save_dir, 'best_model.pth')}")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{config.early_stop_patience}")
            
            # 早停
            if patience_counter >= config.early_stop_patience:
                print("触发早停，训练结束")
                break
        
        # 每个epoch结束后保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 关闭wandb
    # if config.use_wandb:
    #     wandb.finish()
    
    return model