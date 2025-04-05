import torch
import math
import torch.nn as nn

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from torch.nn import TransformerEncoderLayer
from config import Model_Config as Config
from utils.logger import logger

# # 数据预处理 (参考网页59)
# class FlowPreprocessor(nn.Module):
#     def __init__(self, hsv_channels=3):
#         super().__init__()
#         # 参考网页127/20的RAFT特征编码
#         self.raft_encoder = nn.Sequential(
#             nn.Conv3d(hsv_channels, 64, kernel_size=(3,3,3), padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d((1,2,2)),
#             nn.Conv3d(64, 256, kernel_size=(3,3,3), padding=1)
#         )
    
#     def forward(self, x):
#         # x: [B,T,H,W,C] -> [B,C,T,H,W]
#         x = x.permute(0,4,1,2,3)
#         x = self.raft_encoder(x)
#         logger.debug(x.shape)
#         return x  # 输出[B,256,T,H//8,W//8]

class FlowPreprocessor(nn.Module):
    def __init__(self, hsv_channels=3):
        super().__init__()
        # 特征编码参考RAFT的多级降采样策略
        self.encoder = nn.Sequential(
            # Stage 1: 空间降采样至1/2
            nn.Conv3d(hsv_channels, 64, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # [B,64,T,136,240]
            
            # Stage 2: 空间降采样至1/4
            nn.Conv3d(64, 128, (1,3,3), padding=(0,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # [B,128,T,68,120]
            
            # Stage 3: 空间降采样至1/8 & 时间维度压缩
            nn.Conv3d(128, 256, (3,3,3), padding=(1,1,1)),  # 时间维度卷积
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))  # [B,256,T//2,34,60]
        )

    def forward(self, x):
        x = x.permute(0,4,1,2,3)  # [B,C,T,H,W]
        x = self.encoder(x)
        logger.debug(x.shape)
        return x  # 输出[B,256,2,34,60]


# class PositionalEncoding3D(nn.Module):
#     def __init__(self, d_model, temperature=10000, normalize=False):
#         super().__init__()
#         self.d_model = d_model // 3  # 每个轴分配的维度
#         # self.d_model_per_axis = d_model // 3
#         self.temperature = temperature
#         self.normalize = normalize
        
#         # 生成x, y, z轴的频率系数
#         dim_t = torch.arange(self.d_model, dtype=torch.float32)
#         dim_t = temperature ** (2 * (dim_t // 2) / self.d_model)
#         # dim_t = torch.arange(self.d_model_per_axis, dtype=torch.float32)
#         # dim_t = temperature ** (2 * (dim_t // 2) / self.d_model_per_axis)
#         self.register_buffer('dim_t', dim_t)

#     def forward(self, x):
#         """
#         x: Tensor of shape (B, C, D, H, W)
#         """
#         logger.debug(x.shape)
#         B, C, D, H, W = x.shape
#         device = x.device
        
#         # 生成网格坐标
#         grid_d = torch.arange(D, dtype=torch.float32, device=device)
#         grid_h = torch.arange(H, dtype=torch.float32, device=device)
#         grid_w = torch.arange(W, dtype=torch.float32, device=device)
        
#         if self.normalize:
#             grid_d = grid_d / (D - 1) * 2 * math.pi
#             grid_h = grid_h / (H - 1) * 2 * math.pi
#             grid_w = grid_w / (W - 1) * 2 * math.pi
        
#         # 计算位置编码
#         pos_d = grid_d.view(-1, 1, 1, 1) / self.dim_t.view(1, -1, 1, 1)
#         pos_h = grid_h.view(1, -1, 1, 1) / self.dim_t.view(1, -1, 1, 1)
#         pos_w = grid_w.view(1, 1, -1, 1) / self.dim_t.view(1, -1, 1, 1)
        
#         # 正弦和余弦交替编码
#         pe_d = torch.stack((pos_d.sin(), pos_d.cos()), dim=4).flatten(4)
#         pe_h = torch.stack((pos_h.sin(), pos_h.cos()), dim=4).flatten(4)
#         pe_w = torch.stack((pos_w.sin(), pos_w.cos()), dim=4).flatten(4)
        
#         # 合并三个轴的编码
#         pe = torch.cat([pe_d, pe_h, pe_w], dim=1)
#         pe = pe.view(B, -1, D, H, W)  # 调整维度顺序
        
#         return pe

class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, temperature=10000, normalize=True):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        
        # 维度分配策略（时间轴分配较少维度）
        # self.d_model_t = d_model // 6    # 时间轴维度42
        # self.d_model_h = d_model // 3    # 高度轴维度85
        # self.d_model_w = d_model // 3    # 宽度轴维度85
        self.d_model_t = 2    # 时间轴维度42
        self.d_model_h = 34    # 高度轴维度85
        self.d_model_w = 60
        
        # 各轴的频率系数生成
        self.freq_t = self._generate_freq(self.d_model_t)
        self.freq_h = self._generate_freq(self.d_model_h)
        self.freq_w = self._generate_freq(self.d_model_w)
        
        # 线性变换层（将各轴编码融合到通道维度）
        self.proj_t = nn.Linear(self.d_model_t, self.d_model_t)
        self.proj_h = nn.Linear(self.d_model_h, self.d_model_h)
        self.proj_w = nn.Linear(self.d_model_w, self.d_model_w)

    def _generate_freq(self, d_model_axis):
        dim_t = torch.arange(d_model_axis, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_model_axis)
        return dim_t.view(1, -1, 1, 1)  # [1, d_model_axis, 1, 1]

    def _get_grids(self, x):
        B, C, T, H, W = x.shape
        device = x.device
        
        # 生成各轴网格坐标
        grid_t = torch.arange(T, dtype=torch.float32, device=device)
        grid_h = torch.arange(H, dtype=torch.float32, device=device)
        grid_w = torch.arange(W, dtype=torch.float32, device=device)
        
        if self.normalize:
            grid_t = (grid_t / max(T-1, 1)) * 2 * math.pi
            grid_h = (grid_h / max(H-1, 1)) * 2 * math.pi
            grid_w = (grid_w / max(W-1, 1)) * 2 * math.pi
        
        return grid_t, grid_h, grid_w

    def forward(self, x):
        B, C, T, H, W = x.shape
        grid_t, grid_h, grid_w = self._get_grids(x)
        
        # 时间轴编码 [B, d_model_t, T, H, W]
        pos_t = grid_t.view(1, 1, -1, 1, 1) / self.freq_t.to(x.device)
        pe_t = torch.stack([pos_t.sin(), pos_t.cos()], dim=-1).flatten(2,3)
        pe_t = self.proj_t(pe_t.permute(0,3,1,2,4)).expand(B, -1, T, H, W)
        
        # 高度轴编码 [B, d_model_h, T, H, W]
        pos_h = grid_h.view(1, 1, 1, -1, 1) / self.freq_h.to(x.device)
        pe_h = torch.stack([pos_h.sin(), pos_h.cos()], dim=-1).flatten(3,4)
        pe_h = self.proj_h(pe_h.permute(0,3,1,2,4)).expand(B, -1, T, H, W)
        
        # 宽度轴编码 [B, d_model_w, T, H, W] 
        pos_w = grid_w.view(1, 1, 1, 1, -1) / self.freq_w.to(x.device)
        pe_w = torch.stack([pos_w.sin(), pos_w.cos()], dim=-1).flatten(4,5)
        pe_w = self.proj_w(pe_w.permute(0,3,1,2,4)).expand(B, -1, T, H, W)
        
        # 通道维度拼接 [B, C, T, H, W]
        pe = torch.cat([pe_t, pe_h, pe_w], dim=1)
        return x + pe  # 残差连接


# 时空Transformer编码器 (参考网页2/32/36)
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=2048, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_embed = SpatioTemporalPositionalEncoding()
    
    def forward(self, x):
        # x: [B,C,T,H,W] -> [B*H*W, T, C]
        B, C, T, H, W = x.shape

        #  此处调换顺序，先pos_embed再reshape （0402修改）
        x += self.pos_embed(x)
        logger.debug(x.shape)
        x = x.permute(0,3,4,2,1).reshape(B*H*W, T, C)
        
        x = self.encoder(x)
        logger.debug(x.shape)
        return x  # 输出[B*H*W, T, d_model]

# 多任务输出头 (参考网页43/44)
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_keys=Config.num_classes):
        super().__init__()
        # 键盘分类头
        self.key_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, num_keys)
        )
        # 鼠标回归头
        self.mouse_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()  # 输出归一化到[-1,1]
        )
        # 可学习损失权重 (参考网页44)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.key_head(x), self.mouse_head(x)

# 完整模型 (整合网页23/32/68)
class ActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FlowPreprocessor()
        self.temporal_encoder = SpatioTemporalTransformer()
        self.task_head = MultiTaskHead(512)
        
    def forward(self, flow_video):
        # 特征提取
        spatial_feat = self.feature_extractor(flow_video)  # [B,256,T,H//8,W//8]
        # 时空编码
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B*H*W, T, 512]
        # 时空聚合
        aggregated = temporal_feat.mean(dim=1)  # [B*H*W, 512]
        # 多任务预测
        keys, mouse = self.task_head(aggregated)
        return keys, mouse

# 自适应损失 (参考网页59/44)
class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_keys, true_keys, pred_mouse, true_mouse, log_vars):
        # 键盘分类损失
        key_loss = nn.BCEWithLogitsLoss()(pred_keys, true_keys)
        # 鼠标回归损失 (Huber loss)
        mouse_loss = nn.HuberLoss()(pred_mouse, true_mouse)
        # 动态加权 (参考网页44)
        precision1 = torch.exp(-log_vars[0])
        precision2 = torch.exp(-log_vars[1])
        total_loss = precision1*key_loss + precision2*mouse_loss + log_vars.sum()
        return total_loss
    

'''

以下是整合后的参考网页列表，按技术领域分类整理，包含来源链接和核心内容说明：

---

### **视频理解与3D卷积**
1. **3D卷积设计演进**  
   [视频理解中的3D卷积设计](https://m.blog.csdn.net/qq_22409661/article/details/145806657)  
   解析3D卷积的数学原理、PyTorch/TensorFlow实现及行业应用案例，涵盖SlowFast、X3D等前沿模型[1](@ref)。

2. **混合3D卷积策略**  
   [高效3D CNN架构](https://blog.csdn.net/qq_30354455/article/details/119193508)  
   提出混合2D+3D卷积、时空分离卷积等优化方法，降低计算复杂度[1](@ref)。

---

### **光流估计与时空建模**
3. **RAFT光流算法**  
   [RAFT: 深度学习光流估计](https://blog.csdn.net/weixin_43229348/article/details/120500861)  
   基于GRU的迭代优化框架，结合相关金字塔和多尺度特征，实现高精度光流预测[46](@ref)。

4. **显式与隐式帧对齐**  
   [光流估计与可变性卷积](https://blog.csdn.net/weixin_43507744/article/details/124692025)  
   对比光流估计（显式对齐）与可变性卷积（隐式对齐）的优劣及应用场景[2](@ref)。

---

### **Transformer与时空建模**
5. **ViViT视频Transformer**  
   [Video Vision Transformer](https://blog.csdn.net/qdmqdtt/article/details/132351191)  
   提出Tubelet Embedding和4种时空注意力变体，解决视频分类中的时空特征融合问题[54](@ref)。

6. **STTN交通预测模型**  
   [时空Transformer网络](https://blog.csdn.net/gitblog_00012/article/details/139589440)  
   结合Transformer与时空嵌入，实现交通流量预测，支持多传感器数据融合[27](@ref)。

---

### **多任务学习与损失优化**
7. **自适应损失加权策略**  
   [动态损失权重设计](https://blog.csdn.net/m0_56261792/article/details/129239667)  
   基于梯度方差的自适应调整方法，平衡分类与回归任务的损失权重[40](@ref)。

8. **多任务损失理论**  
   [多任务学习权重设计](https://zhuanlan.zhihu.com/p/571951681)  
   比较GradNorm、DWA等方法，提出树模型的多任务学习框架[41](@ref)。

---

### **模型部署与推理加速**
9. **TensorRT-LLM推理优化**  
   [大模型推理框架](https://github.com/luhengshiwo/LLMForEverybody/blob/main/02-%E7%AC%AC%E4%BA%8C%E7%AB%A0-%E9%83%A8%E7%BD%B2%E4%B8%8E%E6%8E%A8%E7%90%86/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%EF%BC%88%E5%9B%9B%EF%BC%89TensorRT-LLM.md)  
   支持量化、动态批处理与多GPU加速，提升LLM推理效率8倍。

---

### **3D位置编码与特征融合**
10. **3D点位置编码（3DPPE）**  
    [多相机3D目标检测](https://blog.csdn.net/weixin_45657478/article/details/132029485)  
    通过混合深度模块生成3D点位置编码，统一空间特征与查询嵌入[65](@ref)。

---

### **专利与工程实践**
11. **自适应损失函数专利**  
    [基于自适应加权的多任务训练](http://bookshelf.docin.com/touch_new/preview_new.do?id=4581879269)  
    通过梯度差值动态调整任务权重，缓解梯度冲突。

---

以上资源覆盖视频分析、多任务学习、模型优化等多个领域，建议按需深入阅读。如需具体代码实现或细节讨论，可进一步提问！

'''