import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# from torch.utils.data import DataLoader
from torchvision.models import resnet18
# from torchvision import models
from sklearn.metrics import accuracy_score
from dataset import Make_Dataset
from dataloader import Make_Dataloader
from utils.logger import get_stream_logger
from utils.file_utils import exist_path

_logger = get_stream_logger()

# 超参数配置
class Config:
    # 数据参数
    train_rate = 0.8
    val_rate = 0.1

    input_shape = (272, 480)  # 原始输入尺寸
    target_size = (224, 224)  # ResNet输入尺寸
    num_classes = 8
    seq_length = 100
    
    # 训练参数
    batch_size = 16
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    checkpoint_path = ".\\models\\checkpoints\\best_model.pth"
    dataset_path = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\capture\\record_all_0319"

# 数据预处理（需要在Dataset中实现）
# def preprocess_data(x_block, y_block):
#     """ 示例预处理流程 """
#     # 归一化图像到[0,1]
#     x_block = x_block.astype(np.float32) / 255.0
    
#     # 归一化鼠标坐标（假设屏幕分辨率480x272）
#     y_block[:, 8:] = y_block[:, 8:] / np.array([480, 272])
    
#     # 转换为Tensor
#     x_tensor = torch.from_numpy(x_block).permute(0, 3, 1, 2)  # [T,C,H,W]
#     y_tensor = torch.from_numpy(y_block.astype(np.float32))
    
#     return x_tensor, y_tensor
def preprocess_data(x_block, y_block):
    # 转换数据类型到float32并归一化
    x_block = x_block.astype(np.float32) / 255.0  # 关键步骤
    
    # 调整鼠标坐标归一化（示例）
    y_block[:, 8:] = y_block[:, 8:] / np.array([480, 272])
    
    # 转换为Tensor
    x_tensor = torch.from_numpy(x_block).permute(0, 3, 1, 2)  # [T,C,H,W]
    y_tensor = torch.from_numpy(y_block.astype(np.float32))
    
    return x_tensor, y_tensor

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
        self.spatial_net.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=3, bias=False)  # 适应小尺寸输入


        # 保留预训练权重（重要技巧）
        with torch.no_grad():
            self.spatial_net.conv1.weight[:, :3] = self.spatial_net.conv1.weight[:, :3].clone()
            self.spatial_net.conv1.weight[:, 3:] = self.spatial_net.conv1.weight[:, :3].clone()
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
        
    def forward(self, x):
        # x形状: [B, T, H, W, C]
        # B, T, H, W, C = x.shape
        
        # # 空间特征提取
        # spatial_feat = x.view(B*T, H, W, C).permute(0, 3, 1, 2)  # [B*T,C,H,W]
        # spatial_feat = self.spatial_net(spatial_feat)  # [B*T, 512]
        # spatial_feat = spatial_feat.view(B, T, -1)    # [B, T, 512]
        B, T, H, W, C = x.shape
        x = x.to(torch.float32)  # 新增类型转换
    
        x_reshaped = x.view(-1, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]
        # 添加尺寸调整（关键步骤）
        x_resized = F.interpolate(
            x_reshaped, 
            size=(224, 224),  # 强制调整到ResNet兼容尺寸
            mode='bilinear'
        )
        spatial_feat = self.spatial_net(x_resized)  # [B*T, 512]
        _logger.info(spatial_feat)
        spatial_feat = spatial_feat.view(B, T, -1)  # [B, T, 512]
        
        # 添加位置编码
        spatial_feat += self.position_embed
        
        # 时序建模
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, 512]
        
        # 多任务预测
        keys = self.key_head(temporal_feat)        # [B, T, 8]
        mouse = self.mouse_head(temporal_feat)     # [B, T, 2]
        
        return torch.cat([keys, mouse], dim=-1)    # [B, T, 10]

# 自定义损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.SmoothL1Loss()
        
    def forward(self, pred, target):
        # 分割预测结果
        pred_keys = pred[:, :, :8]
        pred_mouse = pred[:, :, 8:]
        
        # 分割目标
        target_keys = target[:, :, :8]
        target_mouse = target[:, :, 8:]
        
        # 计算损失
        loss_key = self.key_loss(pred_keys, target_keys)
        loss_mouse = self.mouse_loss(pred_mouse, target_mouse)
        
        return 1.0 * loss_key + 0.5 * loss_mouse

# 训练流程
def train_model(model, train_loader, val_loader, config):
    _logger.debug("正在开始训练")
    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = MultiTaskLoss()
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = []
        for x, y in train_loader:
            _logger.info(x.shape)
            _logger.info(y.shape)
            x = x.to(config.device)  # [B, T, H, W, C]
            y = y.to(config.device)  # [B, T, 10]
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss.append(loss.item())
        
        # 验证阶段
        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.device)
                y = y.to(config.device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss.append(loss.item())
        
        # 统计结果
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        scheduler.step(avg_val_loss)
        
        _logger.info(f"Epoch {epoch+1}/{config.epochs}")
        _logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # print(f"Epoch {epoch+1}/{config.epochs}")
        # print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.checkpoint_path)
            _logger.info("模型已保存！")
    
    return model

# 测试评估
def evaluate_model(model, test_loader, config):
    model.load_state_dict(torch.load(config.checkpoint_path))
    model = model.to(config.device)
    model.eval()
    
    all_keys_pred = []
    all_keys_true = []
    mouse_errors = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            pred = model(x)
            
            # 处理按键预测
            keys_pred = torch.sigmoid(pred[:, :, :8]) > 0.5
            all_keys_pred.append(keys_pred.cpu().numpy())
            all_keys_true.append(y[:, :, :8].cpu().numpy())
            
            # 处理鼠标坐标
            pred_mouse = pred[:, :, 8:].cpu().numpy() * np.array([480, 272])
            true_mouse = y[:, :, 8:].cpu().numpy() * np.array([480, 272])
            mouse_errors.append(np.sqrt(np.sum((pred_mouse - true_mouse)**2, axis=2)))
    
    # 计算按键准确率
    keys_pred = np.concatenate(all_keys_pred, axis=0)
    keys_true = np.concatenate(all_keys_true, axis=0)
    key_acc = accuracy_score(keys_true.flatten(), keys_pred.flatten())
    
    # 计算鼠标平均误差
    mouse_error = np.concatenate(mouse_errors, axis=0).mean()
    
    _logger.info(f"Test Results:")
    _logger.info(f"Key Press Accuracy: {key_acc:.4f}")
    _logger.info(f"Mouse Position Error: {mouse_error:.2f} pixels")


def _check_paths():
    '''
    检测路径是否存在
    '''
    paths = [
        ".\\models",".\\models\\checkpoints"
    ]

    for path in paths: exist_path(path)

def run():
    _check_paths()

    # 初始化配置、模型、数据
    config = Config()
    model = TemporalEnhancedNet()
    
    # 假设已实现Genshin_Basic_Control_Dataset
    train_dataset, test_dataset, val_dataset = Make_Dataset(config.dataset_path,config.train_rate,config.val_rate)
    
    train_loader = Make_Dataloader(train_dataset, batch_size=config.batch_size,shuffle=True)
    val_loader = Make_Dataloader(val_dataset, batch_size=config.batch_size)
    test_loader = Make_Dataloader(test_dataset,batch_size=config.batch_size)
    
    # 训练与评估
    trained_model = train_model(model, train_loader, val_loader, config)
    evaluate_model(trained_model, test_loader, config)

# 主程序
if __name__ == "__main__":
    run()
    pass