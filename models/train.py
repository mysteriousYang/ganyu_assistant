import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gc

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import Make_Dataset,Get_Dataset_list,Control_Record
from dataloader import Make_Dataloader
# from models.model_design0 import TemporalEnhancedNet,MultiTaskLoss
from models.model_claude_1 import GameInputPredictor
from models.model_scheme_1 import ActionPredictor,AdaptiveLoss
from models.model_claude_2 import GameInputPredictor,GameInputLoss,MultiTaskLoss
from models.train_utils import *
from utils.logger import logger
from config import Model_Config

# 训练流程
def train_model(model, train_loader, val_loader, epoch, criterion):
    # logger.debug("正在开始训练")
    model = model.to(Model_Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Model_Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    # criterion = MultiTaskLoss()
    
    # for epoch in range(Model_Config.epochs):
        # 训练阶段
    model.train()
    train_loss = []
    # snapshot1 = tracemalloc.take_snapshot()
    i = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for x, y in train_loader:
            x = x.to(Model_Config.device)  # [B, T, H, W, C]
            y = y.to(Model_Config.device)  # [B, T, 10]
            
            optimizer.zero_grad()
            pred = model(x)
            # print(pred)
            # print(y)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 彻底释放梯度
            
            train_loss.append(loss.item())
            # logger.debug(f"loss_size:{sys.getsizeof(train_loss)}")

            # objgraph.show_refs([x], filename='x_refs.png')
            # ref1 = gc.get_referrers(x)
            # ref2 = gc.get_referents(x)
            # logger.debug(ref1)
            # logger.debug(ref2)

            # 显式释放
            # logger.debug("已释放x, y")
            del x,y,pred,loss
            torch.cuda.empty_cache()  # 清理GPU缓存
            gc.collect()

            i += 1
            # if(i%10==0):
            #     logger.info(f"正在迭代 {i}/{len(train_loader)}")
            # if(i==50):
            #     logger.debug("程序已停止")
            #     exit(0)

            # 内存查错部分
            # snapshot2 = tracemalloc.take_snapshot()
            # 显示内存分配最多的前10行代码
            # top_stats = snapshot2.statistics("lineno")
            # for stat in top_stats[:10]:
            #     logger.debug(f"文件: {stat.traceback[-1].filename}")
            #     logger.debug(f"行号: {stat.traceback[-1].lineno}")
            #     logger.debug(f"总内存: {stat.size / 1024:.2f} KB，分配次数: {stat.count}\n")

            # diff_stats = snapshot2.compare_to(snapshot1, "lineno")
            # for stat in diff_stats[:5]:
            #     logger.debug(f"内存增量: {stat.size_diff / 1024:.2f} KB，代码位置: {stat.traceback}")
            # del snapshot2
            pbar.update(1)
        
    
    # 验证阶段
    model.eval()
    val_loss = []
    with torch.no_grad():
        with tqdm(total=len(val_loader),desc="验证模型") as pbar:
            for x, y in val_loader:
                x = x.to(Model_Config.device)
                y = y.to(Model_Config.device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss.append(loss.item())
                pbar.update(1)
    
    # 统计结果
    avg_train_loss = np.mean(train_loss)
    avg_val_loss = np.mean(val_loss)
    scheduler.step(avg_val_loss)
    
    logger.info(f"Epoch {epoch+1}/{Model_Config.epochs}")
    logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    # print(f"Epoch {epoch+1}/{Model_Config.epochs}")
    # print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    Model_Config.running_epoch += 1
    # 每10epochs保存
    if (Model_Config.running_epoch%5==0):
        torch.save(model.state_dict(), Model_Config.checkpoint_path(Model_Config.running_epoch))
        logger.info(f"模型记录点已保存 {Model_Config.checkpoint_path(Model_Config.running_epoch)}")

    # 保存最佳模型
    if avg_val_loss < Model_Config.best_val_loss:
        Model_Config.best_val_loss = avg_val_loss
        torch.save(model.state_dict(), Model_Config.best_model_path())
        logger.info(f"模型已保存 {Model_Config.best_model_path()}")
    
    return model

# 测试评估
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load(Model_Config.best_model_path()))
    model = model.to(Model_Config.device)
    model.eval()
    
    all_keys_pred = []
    all_keys_true = []
    mouse_errors = []
    
    # logger.info("正在评估模型")
    with torch.no_grad():
        with tqdm(total=len(test_loader),desc="评估模型") as pbar:
            for x, y in test_loader:
                x = x.to(Model_Config.device)
                y = y.to(Model_Config.device)
                pred = model(x)
                
                # 处理按键预测
                keys_pred = torch.sigmoid(pred[:, :, :Model_Config.num_classes]) > 0.5
                all_keys_pred.append(keys_pred.cpu().numpy())
                all_keys_true.append(y[:, :, :Model_Config.num_classes].cpu().numpy())
                
                # 处理鼠标坐标
                pred_mouse = pred[:, :, Model_Config.num_classes:].cpu().numpy() * np.array(Model_Config.input_shape)
                true_mouse = y[:, :, Model_Config.num_classes:].cpu().numpy() * np.array(Model_Config.input_shape)
                mouse_errors.append(np.sqrt(np.sum((pred_mouse - true_mouse)**2, axis=2)))
                pbar.update(1)
    
    # 计算按键准确率
    keys_pred = np.concatenate(all_keys_pred, axis=0)
    keys_true = np.concatenate(all_keys_true, axis=0)
    key_acc = accuracy_score(keys_true.flatten(), keys_pred.flatten())
    
    # 计算鼠标平均误差
    mouse_error = np.concatenate(mouse_errors, axis=0).mean()
    
    logger.info(f"Test Results:")
    logger.info(f"Key Press Accuracy: {key_acc:.4f}")
    logger.info(f"Mouse Position Error: {mouse_error:.2f} pixels")


def run():
    check_paths()
    # tracemalloc.start(25) 

    # 初始化配置、模型、数据
    # config = Model_Config()
    # model = TemporalEnhancedNet()
    # model = ActionPredictor()
    model = GameInputPredictor()
    # criterion = AdaptiveLoss()
    # criterion = GameInputLoss()
    criterion = MultiTaskLoss()

    Model_Config.checkpoint_dir = create_train_folder()
    dump_config_and_network(model)
    dataset_list = Get_Dataset_list("G:\\NN_train\\record_all_0330")

    train_list = []
    test_list = []
    val_list = []
    
    for i in range(len(dataset_list)):
        train_dataset, test_dataset, val_dataset = Make_Dataset(dataset_list[i],Model_Config.train_rate,Model_Config.val_rate)
        train_list.append(train_dataset)
        test_list.append(test_dataset)
        val_list.append(val_dataset)

    for epoch in range(Model_Config.epochs):
        train_idx_list = np.random.permutation(np.arange(0, len(dataset_list))).tolist()
        test_idx_list = np.random.permutation(np.arange(0, len(dataset_list))).tolist()
        val_idx_list = np.random.permutation(np.arange(0, len(dataset_list))).tolist()

        for i in range(len(dataset_list)):
            train_loader = Make_Dataloader(
                train_list[ train_idx_list[i] ],
                batch_size=Model_Config.batch_size,
                shuffle=True,
            )

            test_loader = Make_Dataloader(
                test_list[ test_idx_list[i] ],
                batch_size=Model_Config.batch_size,
            )

            val_loader = Make_Dataloader(
                val_list[ val_idx_list[i] ],
                batch_size=Model_Config.batch_size
            )

            # 训练与评估
            trained_model = train_model(model, train_loader, val_loader, epoch, criterion)
        evaluate_model(trained_model, test_loader)

    # for dataset in dataset_list:
    #     # 假设已实现Genshin_Basic_Control_Dataset
    #     train_dataset, test_dataset, val_dataset = Make_Dataset(dataset,Model_Config.train_rate,Model_Config.val_rate)
        
    #     train_loader = Make_Dataloader(train_dataset, batch_size=Model_Config.batch_size,shuffle=True,pin_memory=True)
    #     val_loader = Make_Dataloader(val_dataset, batch_size=Model_Config.batch_size)
    #     test_loader = Make_Dataloader(test_dataset,batch_size=Model_Config.batch_size)
        
    #     # 训练与评估
    #     trained_model = train_model(model, train_loader, val_loader, Model_Config)
    #     evaluate_model(trained_model, test_loader, Model_Config)


def _test():
    Model_Config = Model_Config()
    # model = TemporalEnhancedNet()

    Model_Config.checkpoint_dir = create_train_folder()

    full_dataset = Control_Record("G:\\NN_train\\debug","2025-03-30-10-20-36-173740")
    train_dataset, test_dataset, val_dataset = Make_Dataset(full_dataset,Model_Config.train_rate,Model_Config.val_rate)

    train_loader = Make_Dataloader(train_dataset, batch_size=Model_Config.batch_size,shuffle=True,pin_memory=True)
    val_loader = Make_Dataloader(val_dataset, batch_size=Model_Config.batch_size)
    test_loader = Make_Dataloader(test_dataset,batch_size=Model_Config.batch_size)
    
    # 训练与评估
    # trained_model = train_model(model, train_loader, val_loader, 1)
    # evaluate_model(trained_model, test_loader)

# 主程序
if __name__ == "__main__":
    run()
    # _test()
    pass