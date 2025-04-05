import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Model_Config
from utils.logger import logger
from torchvision.models.video import r3d_18


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

# 3D-CNN + Transformer 模型
class GameInputPredictor(nn.Module):
    def __init__(self, ):
        super(GameInputPredictor, self).__init__()
        
        # 使用预训练的R3D-18作为3D-CNN骨干网络
        r3d = r3d_18(pretrained=True)
        self.backbone = nn.Sequential(*list(r3d.children())[:-2])  # 去掉最后的全连接层
        
        # 3D-CNN输出通道数
        self.output_channels = 512
        
        # 特征映射尺寸
        self.feature_size = self._calculate_feature_size()
        
        # 时空特征映射
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Transformer编码器
        self.embed_dim = 256
        self.positional_encoding = nn.Parameter(torch.zeros(1, Model_Config.seq_length, self.embed_dim))
        self.embedding_projection = nn.Linear(self.output_channels, self.embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, num_heads=8, hidden_dim=1024, dropout=Model_Config.dropout)
            for _ in range(2)  # 2层Transformer编码器
        ])
        
        # 输出头
        self.fc_common = nn.Linear(self.embed_dim, 128)
        
        # 按键预测头
        self.key_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(Model_Config.dropout),
            nn.Linear(64, Model_Config.num_classes),
            nn.Sigmoid()  # 多标签分类，使用Sigmoid
        )
        
        # 鼠标位置预测头
        self.mouse_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(Model_Config.dropout),
            nn.Linear(64, 2),
            nn.Sigmoid()  # 归一化坐标，使用Sigmoid
        )
    
    def _calculate_feature_size(self):
        # 计算3D-CNN输出的特征尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Model_Config.seq_length, Model_Config.input_shape[1], Model_Config.input_shape[0])
            output = self.backbone(dummy_input)
            return output.size()
    
    def forward(self, x):
        # x shape: [batch_size, T, H, W, C]
        batch_size, t, h, w, c = x.size()
        
        # 重塑为3D-CNN输入格式 [batch_size, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3)
        logger.debug(x.shape)
        
        # 3D-CNN特征提取
        features = self.backbone(x)  # [batch_size, channels, t', h', w']
        
        # 时空池化
        pooled = self.temporal_pool(features).view(batch_size, self.output_channels)
        
        # 投影到Transformer的嵌入维度
        features = features.permute(0, 2, 1, 3, 4)  # [batch_size, t', channels, h', w']
        batch_size, t_prime, channels, h_prime, w_prime = features.shape
        
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
        
        # 获取序列的最后一个时间步
        final_state = embedded[:, -1, :]
        
        # 共享特征层
        common = F.relu(self.fc_common(final_state))
        
        # 预测按键和鼠标位置
        key_pred = self.key_predictor(common)
        mouse_pred = self.mouse_predictor(common)
        
        return key_pred, mouse_pred

# 基于3D ResNet的模型
class ResNet3DModel(nn.Module):
    def __init__(self,):
        super(ResNet3DModel, self).__init__()
        
        # 使用预训练的3D ResNet
        self.backbone = r3d_18(pretrained=True)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原始全连接层
        
        # 添加任务特定的头部网络
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(Model_Config.dropout)
        
        # 按键预测头
        self.key_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Model_Config.dropout),
            nn.Linear(128, Model_Config.num_classes),
            nn.Sigmoid()
        )
        
        # 鼠标位置预测头
        self.mouse_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Model_Config.dropout),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, T, C, H, W]
        batch_size, t, c, h, w = x.size()
        
        # 重塑为3D-CNN输入格式 [batch_size, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        # 特征提取
        features = self.backbone(x)
        
        # 共享特征层
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        
        # 预测
        key_pred = self.key_head(x)
        mouse_pred = self.mouse_head(x)
        
        return key_pred, mouse_pred

# 混合注意力3D-CNN模型
class MixedAttention3DCNN(nn.Module):
    def __init__(self, config):
        super(MixedAttention3DCNN, self).__init__()
        
        # 3D卷积层
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # 3D残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时间注意力
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类头
        self.fc = nn.Linear(512, 256)
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
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: [batch_size, T, C, H, W]
        batch_size, t, c, h, w = x.size()
        
        # 重塑为3D-CNN输入格式 [batch_size, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, T, H, W]
        
        # 应用空间注意力
        b, c, t, h, w = x.size()
        spatial_weights = self.spatial_attention(x.mean(2))  # [B, 1, H, W]
        spatial_weights = spatial_weights.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, H, W]
        spatial_weights = spatial_weights.expand_as(x)
        
        # 应用时间注意力
        temporal_weights = self.temporal_attention(x.mean(3).mean(3).transpose(1, 2))  # [B, 1, T]
        temporal_weights = temporal_weights.transpose(1, 2).unsqueeze(3).unsqueeze(4)  # [B, 1, T, 1, 1]
        temporal_weights = temporal_weights.expand_as(x)
        
        # 组合注意力权重
        x = x * spatial_weights * temporal_weights
        
        # 全局池化
        x = self.avgpool(x)
        x = x.view(b, -1)
        
        # 共享特征层
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # 预测
        key_pred = self.key_head(x)
        mouse_pred = self.mouse_head(x)
        
        return key_pred, mouse_pred

# 3D ResNet基本块
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1, stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out