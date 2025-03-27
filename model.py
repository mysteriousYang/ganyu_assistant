#-*- coding: utf-8 -*-
import gc
import os
import datetime
import torch
import objgraph
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision.models import resnet18
from memory_profiler import profile
from sklearn.metrics import accuracy_score
from dataset import Make_Dataset,Get_Dataset_list,Control_Record
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
    batch_size = 1
    lr = 1e-4
    epochs = 10
    running_epoch = 0
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    # checkpoint_path = ".\\models\\checkpoints\\" + datetime.datetime.now().strftime('%Y-%m-%d') + ".pth"
    checkpoint_dir = ""
    # best_model_path = ".\\models\\checkpoints\\best_model.pth"
    dataset_path = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\capture\\record_all_0319"

    def checkpoint_path(self,epoch):
        checkpoint_path =  f"{self.checkpoint_dir}\\" + datetime.datetime.now().strftime('%Y-%m-%d') + f"e{epoch}.pth"
        return checkpoint_path

    def best_model_path(self):
        best_model_path = f"{self.checkpoint_dir}\\best_model.pth"
        return best_model_path

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
        self.spatial_net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 适应小尺寸输入


        # 保留预训练权重（重要技巧）
        # with torch.no_grad():
        #     self.spatial_net.conv1.weight[:, :3] = self.spatial_net.conv1.weight[:, :3].clone()
        #     self.spatial_net.conv1.weight[:, 3:] = self.spatial_net.conv1.weight[:, :3].clone()
          # 权重初始化技巧（关键！）
        with torch.no_grad():
            # 复制原3通道权重到新6通道（前3通道复制RGB权重，后3通道复制光流权重）
            original_weight = self.spatial_net.conv1.weight[:, :3].clone()
            self.spatial_net.conv1.weight = nn.Parameter(torch.cat([original_weight, original_weight], dim=1))

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
        # _logger.info(x.shape)
        # x = x.to(torch.float32)  # 新增类型转换
    
        # x_reshaped = x.view(-1, C, H, W).permute(0, 3, 1, 2)  # [B*T, C, H, W]
        # x_reshaped = x.view(-1, H, W, C).permute(0,3,1,2)
        # 直接调整维度顺序避免后续重塑
        x_reshaped = x.permute(0,1,3,4,2)  # B,T,H,W,C → B,T,C,H,W → 不需要额外重塑
        x_reshaped = x_reshaped.reshape(B*T, C, H, W)  # [B*T, C, H, W]

        # 添加尺寸调整（关键步骤）
        x_resized = F.interpolate(
            x_reshaped, 
            size=(224, 224),  # 强制调整到ResNet兼容尺寸
            mode='bilinear'
        )

        # with torch.cuda.amp.autocast(enabled=True): 
        spatial_feat = self.spatial_net(x_resized)  # [B*T, 512]
        # _logger.info(spatial_feat)
        spatial_feat = spatial_feat.view(B, T, -1)  # [B, T, 512]
        
        # 添加位置编码
        spatial_feat += self.position_embed
        
        # 时序建模
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, 512]
        
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
def train_model(model, train_loader, val_loader, config:Config):
    _logger.debug("正在开始训练")
    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = MultiTaskLoss()
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = []
        # snapshot1 = tracemalloc.take_snapshot()
        i = 0
        for x, y in train_loader:
            x = x.to(config.device)  # [B, T, H, W, C]
            y = y.to(config.device)  # [B, T, 10]
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 彻底释放梯度
            
            train_loss.append(loss.item())
            # _logger.debug(f"loss_size:{sys.getsizeof(train_loss)}")

            # objgraph.show_refs([x], filename='x_refs.png')
            # ref1 = gc.get_referrers(x)
            # ref2 = gc.get_referents(x)
            # _logger.debug(ref1)
            # _logger.debug(ref2)

            # 显式释放
            # _logger.debug("已释放x, y")
            del x,y,pred,loss
            torch.cuda.empty_cache()  # 清理GPU缓存
            gc.collect()

            # i += 1
            # if(i%100==0):
            #     _logger.info(f"正在迭代 {i}/{len(train_loader)}")
            # if(i==50):
            #     _logger.debug("程序已停止")
            #     exit(0)

            # 内存查错部分
            # snapshot2 = tracemalloc.take_snapshot()
            # 显示内存分配最多的前10行代码
            # top_stats = snapshot2.statistics("lineno")
            # for stat in top_stats[:10]:
            #     _logger.debug(f"文件: {stat.traceback[-1].filename}")
            #     _logger.debug(f"行号: {stat.traceback[-1].lineno}")
            #     _logger.debug(f"总内存: {stat.size / 1024:.2f} KB，分配次数: {stat.count}\n")

            # diff_stats = snapshot2.compare_to(snapshot1, "lineno")
            # for stat in diff_stats[:5]:
            #     _logger.debug(f"内存增量: {stat.size_diff / 1024:.2f} KB，代码位置: {stat.traceback}")
            # del snapshot2

            
        
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
        
        config.running_epoch += 1
        # 每10epochs保存
        if (epoch%10==0):
            torch.save(model.state_dict(), config.checkpoint_path(config.running_epoch))
            _logger.info("模型记录点已保存！")

        # 保存最佳模型
        if avg_val_loss < config.best_val_loss:
            config.best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.best_model_path())
            _logger.info("模型已保存！")
    
    return model

# 测试评估
def evaluate_model(model, test_loader, config:Config):
    model.load_state_dict(torch.load(config.best_model_path()))
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

def create_train_folder(base_path=".\\models\\checkpoints"):
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
    _logger.info(f"创建文件夹: {new_path}")
    return new_path

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
    # tracemalloc.start(25) 

    # 初始化配置、模型、数据
    config = Config()
    model = TemporalEnhancedNet()

    config.checkpoint_dir = create_train_folder()
    dataset_list = Get_Dataset_list("G:\\NN_train\\record_all_0319")
    
    for dataset in dataset_list:
        # 假设已实现Genshin_Basic_Control_Dataset
        train_dataset, test_dataset, val_dataset = Make_Dataset(dataset,config.train_rate,config.val_rate)
        
        train_loader = Make_Dataloader(train_dataset, batch_size=config.batch_size,shuffle=True,pin_memory=True)
        val_loader = Make_Dataloader(val_dataset, batch_size=config.batch_size)
        test_loader = Make_Dataloader(test_dataset,batch_size=config.batch_size)
        
        # 训练与评估
        trained_model = train_model(model, train_loader, val_loader, config)
        evaluate_model(trained_model, test_loader, config)


def _test():
    config = Config()
    model = TemporalEnhancedNet()

    full_dataset = Control_Record("G:\\NN_train\\record_all_0319","2025-03-03-12-19-00-255774")
    train_dataset, test_dataset, val_dataset = Make_Dataset(full_dataset,config.train_rate,config.val_rate)

    train_loader = Make_Dataloader(train_dataset, batch_size=config.batch_size,shuffle=True,pin_memory=True)
    val_loader = Make_Dataloader(val_dataset, batch_size=config.batch_size)
    test_loader = Make_Dataloader(test_dataset,batch_size=config.batch_size)
    
    # 训练与评估
    trained_model = train_model(model, train_loader, val_loader, config)
    evaluate_model(trained_model, test_loader, config)

# 主程序
if __name__ == "__main__":
    run()
    # _test()
    pass