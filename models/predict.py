#-*- coding: utf-8 -*-
import cv2
import torch
import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from torchvision import transforms
from tqdm import tqdm
# from models.model_design0 import TemporalEnhancedNet  # 假设已定义时序模型
from models.model_claude_2 import GameInputPredictor
from utils.logger import logger

debug_flow_out = cv2.VideoWriter(f".\\video\\output_flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (480, 272))

class VideoOperatorRecognizer:
    def __init__(self, model_path, input_size=(272, 480), time_steps=100):
        # 初始化参数
        self.time_steps = time_steps
        self.input_size = input_size
        self.screen_res = (1080, 1920)
        
        # 模型加载
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
        # 光流计算器
        
        # 初始化CUDA光流计算器（以DualTVL1算法为例）
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # 使用CUDA加速的光流算法
            logger.info(f"检测到CUDA设备，使用GPU计算光流")
            # self.flow_calculator = cv2.cuda_OpticalFlowDual_TVL1.create(
            #     tau=0.25, 
            #     lambda_=0.15, 
            #     theta=0.3, 
            #     nscales=5, 
            #     warps=5
            # )
            self.flow_calculator = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=5,      # 金字塔层数
            pyrScale=0.5,     # 金字塔缩放比例
            fastPyramids=False,
            winSize=15,       # 窗口大小
            numIters=3,       # 迭代次数
            polyN=5,          # 多项式大小
            polySigma=1.2,    # 多项式标准差
            flags=0           # 标志位
        )
            self.cv_device = "cuda"
        else:
            logger.info(f"未检测到CUDA设备，使用GPU计算")
            self.flow_calculator = cv2.optflow.DualTVL1OpticalFlow_create()
            self.cv_device = "cpu"
        
        # 预处理
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5, 0.5])
        # ])
        self.transform = lambda x: torch.from_numpy(
            x.astype(np.float32) / 255.0  # 手动归一化到[0,1]（如果是uint8输入）
        )
        
        # 数据缓冲区
        self.rgb_buffer = []
        self.flow_buffer = []
        self.full_buffer = []
        

    def _load_model(self, model_path):
        # 示例模型结构，需替换为实际模型
        logger.debug(f"正在加载模型 {model_path}")
        # model = torch.nn.LSTM(input_size=480*272*6, hidden_size=512, num_layers=2).to(self.device)
        # model = TemporalEnhancedNet()
        model = GameInputPredictor()
        model.load_state_dict(torch.load(model_path,weights_only=True))
        model = model.to(self.device)
        model.eval()
        return model

    def _resize_frame(self, frame):
        return cv2.resize(frame, (self.input_size[1], self.input_size[0]))

    def _compute_flow(self, prev_frame, curr_frame):
        if(self.cv_device == "cuda"):
            return self._compute_flow_gpu(prev_frame, curr_frame)
        else:
            return self._compute_flow_cpu(prev_frame, curr_frame)

    def _compute_flow_cpu(self, prev_gray, curr_gray):
        flow = self.flow_calculator.calc(prev_gray, curr_gray, None)
        flow_x = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        flow_y = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        flow_mag = cv2.magnitude(flow[...,0], flow[...,1])
        return np.stack([flow_x, flow_y, flow_mag], axis=2)
    
    def _compute_flow_gpu(self, prev_gray, curr_gray):
        """ 计算两帧之间的光流，返回三通道光流图 """
        # 转换为灰度图并上传至GPU
        # prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        gpu_prev = cv2.cuda_GpuMat()
        gpu_curr = cv2.cuda_GpuMat()
        gpu_prev.upload(prev_gray)
        gpu_curr.upload(curr_gray)
        
        # CUDA光流计算
        gpu_flow = self.flow_calculator.calc(gpu_prev, gpu_curr, None)
        flow = gpu_flow.download()  # 下载到CPU

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 将光流可视化
        hsv = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # 色调
        hsv[..., 1] = 255                      # 饱和度
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # flow_img /= np.float32(255.0)
        debug_flow_out.write(flow_img)
        # logger.debug(flow_img.shape)

        return flow_img
        
        # # 将光流分解为X/Y方向并归一化
        # flow_x = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        # flow_y = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # flow_mag = cv2.magnitude(flow_x, flow_y)
    
        # return np.stack([flow_x, flow_y, flow_mag], axis=2).astype(np.uint8)

    def _buffer_to_input(self):
        # 填充对齐处理（光流比RGB少1帧）
        if len(self.flow_buffer) == len(self.rgb_buffer) - 1:
            self.flow_buffer.append(self.flow_buffer[-1].copy())
        
        # 合并数据
        combined = []
        # for rgb, flow in zip(self.rgb_buffer, self.flow_buffer):
        #     combined_frame = np.concatenate([rgb, flow], axis=2)  # [H,W,6]
        #     combined.append(self.transform(combined_frame))

        for flow in self.flow_buffer:
            combined.append(self.transform(flow))

        return torch.stack(combined).unsqueeze(0).to(self.device)  # [1,T,C,H,W]
    # def _buffer_to_input(self):
    #     # 填充对齐处理（光流比RGB少1帧）
    #     if len(self.flow_buffer) == len(self.rgb_buffer) - 1:
    #         self.flow_buffer.append(self.flow_buffer[-1].copy())
        
    #     # 合并数据
    #     combined = []
    #     for rgb, flow in zip(self.rgb_buffer, self.flow_buffer):
    #         # 合并为 [H, W, 6]
    #         combined_frame = np.concatenate([rgb, flow], axis=2)
            
    #         # 转换为Tensor并保持通道在最后 
    #         tensor_frame = torch.from_numpy(combined_frame).float().permute(2, 0, 1)  # [C, H, W]
    #         tensor_frame = self.transform(tensor_frame)  # 标准化
    #         combined.append(tensor_frame.permute(1, 2, 0))  # 恢复为 [H, W, C]
        
        # 堆叠并调整维度顺序
        # return torch.stack(combined).unsqueeze(0).to(self.device)  # [1, T, H, W, C]

    def _decode_actions(self, outputs):
        # 转换为操作序列
        # logger.debug(f"正在转化操作序列")
        key_labels = ['W', 'S', 'A', 'D', 'Space',]
        actions = []
        for t in range(outputs.shape[1]):
            action = outputs[0, t].cpu().numpy()
            # 按键状态
            keys = [action[i] > 0.5 for i in range(len(key_labels))]
            # 鼠标坐标
            mouse_x = int(action[len(key_labels)] * self.screen_res[1]) + 960
            mouse_y = int(action[len(key_labels)+1] * self.screen_res[0]) + 540
            actions.append({
                'keys': {k: v for k, v in zip(key_labels, keys)},
                'mouse': (mouse_x, mouse_y)
            })
        # logger.debug(actions)
        return actions

    def _render_overlay(self, frame, action):
        # 鼠标光标
        cv2.circle(frame, action['mouse'], 8, (0,0,255), -1)
        # 按键状态
        active_keys = [k for k, v in action['keys'].items() if v]
        text = "Pressed: " + ", ".join(active_keys) if active_keys else "No Keys"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"mouse_x: {action["mouse"][0]}, mouse_y: {action["mouse"][1]}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return frame

    def process_video(self, input_path, output_path):
        # 视频输入输出设置
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_size = (int(cap.get(3)), int(cap.get(4)))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, orig_size)
        
        # 光流计算初始化
        prev_gray = None
        
        with tqdm(total=total_frames, desc="预测并渲染") as pbar:
            j=0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    # logger.debug(f"视频处理完毕，退出循环")
                    # pbar.close()
                    break
                j-=-1
                if(j > total_frames):
                    # logger.debug(f"视频处理完毕，退出循环")
                    # pbar.close()
                    break

                # 调整尺寸并转换颜色空间
                resized_frame = self._resize_frame(frame)
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                
                # 计算光流
                if prev_gray is not None:
                    flow = self._compute_flow(prev_gray, gray)
                
                    self.flow_buffer.append(flow)
            
                prev_gray = gray
                
                # 存储RGB帧
                self.rgb_buffer.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                
                # 达到时间窗口后推理
                # logger.debug(f"正在推理")
                if len(self.rgb_buffer) == self.time_steps:
                    # 构造模型输入
                    inputs = self._buffer_to_input()
                    
                    # 模型推理
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        # logger.debug(outputs)
                    # 解码操作序列
                    actions = self._decode_actions(outputs)
                    
                    # 渲染到原始尺寸帧
                    for i in range(self.time_steps):
                        # cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - self.time_steps + i)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, j - self.time_steps + i)
                        _, render_frame = cap.read()
                        render_frame = self._render_overlay(render_frame, actions[i])
                        # logger.debug(f"正在转存第 {j - self.time_steps + i} 帧")
                        out.write(render_frame)
                    
                    # 清空缓冲区
                    self.rgb_buffer = []
                    self.flow_buffer = []
                    
                pbar.update(1)
        logger.info(f"已完成所有帧的预测")
        
        # 处理剩余帧
        # if len(self.rgb_buffer) > 0:
        #     fill_num = self.time_steps - len(self.rgb_buffer)
        #     self.rgb_buffer.extend([self.rgb_buffer[-1]] * fill_num)
        #     self.flow_buffer.extend([self.flow_buffer[-1]] * fill_num)
        #     inputs = self._buffer_to_input()
        #     outputs = self.model(inputs)
        #     actions = self._decode_actions(outputs)
        #     for i in range(len(self.rgb_buffer)):
        #         out.write(self._render_overlay(frame, actions[i]))
        
        cap.release()
        out.release()
        debug_flow_out.release()

def run():
    recognizer = VideoOperatorRecognizer(r"models\checkpoints\2025-04-03\train_9\best_model.pth",time_steps=4)
    recognizer.process_video(".\\video\\input_s.mp4", ".\\video\\output_0403.mp4")
    pass

# 使用示例
if __name__ == "__main__":
    run()
    pass