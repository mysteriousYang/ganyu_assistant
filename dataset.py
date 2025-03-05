#-*- coding: utf-8 -*-
import os
import json
import logging
import cv2
import torch.nn

import numpy as np

from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip
from pathlib import Path
from utils.logger import get_stream_logger,timer_logger

_logger = get_stream_logger()

class Genshin_Basic_Control_Dataset(Dataset):
    def __init__(self,record_path):
        '''
        初始化用于训练动作控制模型的数据集
        
        Args:
            record_path (str) : 需要传入record.txt的路径
        '''

        self.capture_dir = os.path.dirname(record_path)

    


    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        根据索引返回一个数据样本
        :param index: 索引
        """
        pass


def process_raw_csv(path:str,record:str,**kwargs):
    '''
    用于将csv转化为样本

    Args:
        path(str): 处理的路径
        record (str) : 记录名

    '''
    # 初始化 key_map 文件
    if("key_map_file" in kwargs):
        key_map_file = kwargs["key_map_file"]
    else:
        key_map_file = ".\\key_mapping.json"

    try:
        csv_fp = open(os.path.join(path,f"{record}.csv"),mode='r',encoding="utf-8")
        _logger.info(f"正在处理 {record}")
    except:
        _logger.exception("csv文件不存在")
        exit(-1)

    # 虽然这里是 .kb与.ms文件,但实际存储格式依然为csv
    try:
        kb_fp = open(os.path.join(path,f"{record}.kb"),mode="w",encoding="utf-8")
    except:
        _logger.exception(f"{record}.kb 写入失败")
        exit(-1)

    try:
        ms_fp = open(os.path.join(path,f"{record}.ms"),mode="w",encoding="utf-8")
    except:
        _logger.exception(f"{record}.ms 写入失败")
        exit(-1)

    # 读取按键映射表
    key_map = dict()
    try:
        with open(key_map_file,mode='r',encoding="utf-8") as json_fp:
            key_map = json.load(json_fp)
    except:
        _logger.exception(f"无法读取按键映射表")
        exit(-1)


    # 分离鼠标与键盘事件
    while True:

        # 这里暂时不处理换行符, 一会儿还要重新添加
        event = csv_fp.readline()
        if(event == ""): 
            # 文件尾
            break

        if(event[0] == '0'):
            # 0表示键盘类事件
            line = process_keyboard_record(event,key_map)
            kb_fp.write(line)

        elif(event[0] == '1'):
            # 1表示鼠标类事件
            line = process_mouse_record(event)
            ms_fp.write(line)

        else:
            # 非法数据
            continue
    
    _logger.info(f"分离完成 {record}.csv")
    ms_fp.close()
    kb_fp.close()
    csv_fp.close()

    with open(os.path.join(path,"records.txt"),mode='a',encoding="utf-8") as fp:
        _logger.info(f"已导出 {record}")
        fp.write(record)
        fp.write('\n')

def process_keyboard_record(event:str,key_map):
    '''
    
    '''
    _, action, key, frame_number = event.split(',')
    if not (key in key_map):
        # 不需要学习的按键, 返回空
        return ""

    key_index = key_map[key]
    return '\t'.join([action,str(key_index),frame_number])


def process_mouse_record(event:str,**kwargs):
    '''
    目前仅处理鼠标移动, 暂不处理点击事件与滚动事件
    '''
    if(event[2] == '0'):
        line = event[2:]
        line = line.replace(',','\t')
        return line
    
    else:
        return ''
    

def clip_compression(path:str,record:str,**kwargs):
    '''
    压缩录制的原生视频

    Args:
        path(str): 视频录制目录
        record(str): 记录名
        height: (可选) 导出高度
        width: (可选) 导出宽度
    '''
    if("height" in kwargs):
        new_height = kwargs["height"]
    else:
        # RAFT光流需要8的倍数,所以是272
        new_height = 272
    
    if("width" in kwargs):
        new_width = kwargs["width"]
    else:
        new_width = 480

    input_video = f"{record}.mp4"
    output_video = f"{record}_s.mp4"

    # 如果压制视频已存在, 则不作任何更改
    if(os.path.exists(os.path.join(path,output_video))):
        _logger.info(f"{output_video} 已存在")
        return

    # 加载视频
    video = VideoFileClip(os.path.join(path,input_video))

    _logger.info(f"正在压制 {input_video}")
    _logger.info(f"目标大小: {new_width}*{new_height}")


    # 调整分辨率
    resized_video = video.resize((new_width, new_height))

    # 保存调整后的视频
    resized_video.write_videofile(os.path.join(path,output_video))

    _logger.info(f"压制完成 {output_video}")

    # 关闭视频对象
    video.close()
    resized_video.close()


# OpenCV版本的视频压制代码
def clip_compression_opencv(path:str,record:str,**kwargs):
    '''
    压缩录制的原生视频

    Args:
        path(str): 视频录制目录
        record(str): 记录名
        height: (可选) 导出高度
        width: (可选) 导出宽度
    '''
    if("height" in kwargs):
        new_height = kwargs["height"]
    else:
        new_height = 272
    
    if("width" in kwargs):
        new_width = kwargs["width"]
    else:
        new_width = 480

    input_video = f"{record}.mp4"
    output_video = f"{record}_s.mp4"
    input_video_path = os.path.join(path,input_video)
    output_video_path = os.path.join(path,output_video)

    # 如果压制视频已存在, 则不作任何更改
    if(os.path.exists(os.path.join(path,output_video))):
        _logger.info(f"{output_video} 已存在")
        return
    
        # 检查OpenCV是否支持CUDA
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available. Please install OpenCV with CUDA support.")
        return

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        _logger.exception(f"无法打开视频 {input_video_path}")
        return

    # 获取视频的原始属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频的编码器和属性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (new_width, new_height))

    # 创建CUDA加速的帧处理管道
    cuda_stream = cv2.cuda_Stream()  # 创建CUDA流
    # cuda_resizer = cv2.cuda_Resize(new_width, new_height)  # 创建CUDA缩放器

    _logger.info(f"正在压制 {input_video}")
    _logger.info(f"目标大小: {new_width}*{new_height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 将帧上传到GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame, stream=cuda_stream)

        # 使用CUDA进行缩放
        # resized_gpu_frame = cuda_resizer.apply(gpu_frame, stream=cuda_stream)
        resized_gpu_frame = cv2.cuda.resize(gpu_frame, (new_width, new_height), stream=cuda_stream)

        # 将缩放后的帧下载回CPU
        resized_frame = resized_gpu_frame.download(stream=cuda_stream)

        # 写入输出视频
        out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()
    _logger.info(f"压制完成 {output_video}")


@timer_logger
def calculate_optical_flow(record_dir:str,record:str,use_large:bool=False,**kwargs):
    '''
    光流计算, 使用OpenCV
    '''
    if not cv2.cuda.getCudaEnabledDeviceCount():
        _logger.warning("CUDA 不可用，使用CPU计算")
        cuda_available = False
    else:
        _logger.info("使用CUDA计算")
        cuda_available = True

    if use_large:
        input_video = f"{record}.mp4"
        output_video = f"{record}_flow.mp4"
    else:
        input_video = f"{record}_s.mp4"
        output_video = f"{record}_s_flow.mp4"

    # 打开视频文件
    cap = cv2.VideoCapture(os.path.join(record_dir,input_video))
    if not cap.isOpened():
        _logger.exception(f"无法打开视频文件 {input_video}")
        return

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter(os.path.join(record_dir,output_video), fourcc, fps, (width, height))

    if cuda_available:
        # 创建 CUDA 光流对象
        cuda_farneback = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=5,      # 金字塔层数
            pyrScale=0.5,     # 金字塔缩放比例
            fastPyramids=False,
            winSize=15,       # 窗口大小
            numIters=3,       # 迭代次数
            polyN=5,          # 多项式大小
            polySigma=1.2,    # 多项式标准差
            flags=0           # 标志位
        )

        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            print("无法读取视频")
            exit()

        # 将第一帧转换为灰度图并上传到 GPU
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(prev_gray)

        # 处理每一帧
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # 将当前帧转换为灰度图并上传到 GPU
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gpu = cv2.cuda_GpuMat()
            curr_gpu.upload(curr_gray)

            # 使用 CUDA 计算光流
            flow_gpu = cuda_farneback.calc(prev_gpu, curr_gpu, None)

            # 将光流下载到 CPU
            flow = flow_gpu.download()

            # 将光流转换为极坐标（幅度和角度）
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 将光流可视化
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # 色调
            hsv[..., 1] = 255                      # 饱和度
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度
            flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 将光流图像写入视频
            out.write(flow_img)

            # 更新前一帧
            prev_gpu = curr_gpu

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        _logger.info(f"光流视频已保存到: {os.path.join(record_dir,input_video)}")

    else:
        # 读取第一帧并转换为灰度图像
        ret, old_frame = cap.read()
        if not ret:
            _logger.info("视频为空")
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # 创建HSV图像用于绘制光流
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255  # 饱和度设为最大

        _logger.info(f"正在绘制光流图...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 计算光流的幅度和角度
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 将角度和幅度映射到HSV颜色空间
            hsv[..., 0] = ang * 180 / np.pi / 2  # 色相
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 明度

            # 转换为BGR颜色空间
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 写入光流图到视频文件
            out.write(bgr)

            # 显示原始帧和光流图
            # cv2.imshow('Original Frame', frame)
            # cv2.imshow('Optical Flow', bgr)

            # 更新旧帧
            old_gray = gray.copy()

            if cv2.waitKey(30) & 0xFF == 27:  # 按Esc键退出
                break

        _logger.info(f"光流视频已保存到: {os.path.join(record_dir,input_video)}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()


@timer_logger
def calculate_opticcal_flow_torch(record_dir:str,record:str):
    import argparse
    from RAFT.raft import RAFT
    from RAFT.utils import flow_viz
    from RAFT.utils.utils import InputPadder

    # 加载 RAFT 模型
    def load_raft_model(args):
        model = RAFT(args)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.path))
        model.to('cuda')
        model.eval()
        return model

    # 初始化模型
    args = argparse.Namespace(
        model='raft',
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        path='.\\Raft\\models\\raft-things.pth'
    )
    model = load_raft_model(args)

    input_video = f"{record}_s.mp4"
    output_video = f"{record}_s_flow_torch.mp4"

    # 读取视频
    video_path = os.path.join(record_dir,input_video)
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    output_path = os.path.join(record_dir,output_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
        exit()

    # 将第一帧转换为 RGB 并归一化
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB) / 255.0
    prev_frame = torch.from_numpy(prev_frame).permute(2, 0, 1).float().unsqueeze(0).to('cuda')

    # 处理每一帧
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 将当前帧转换为 RGB 并归一化
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB) / 255.0
        curr_frame = torch.from_numpy(curr_frame).permute(2, 0, 1).float().unsqueeze(0).to('cuda')

        # 使用 RAFT 计算光流
        with torch.no_grad():
            flow_low, flow_up = model(prev_frame, curr_frame, iters=20, test_mode=True)

        # 将光流转换为图像
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_img = flow_viz.flow_to_image(flow_np)

        # 将光流图像写入视频
        out.write(flow_img)

        # 更新前一帧
        prev_frame = curr_frame

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    _logger.info(f"光流视频已保存到: {output_path}")

def transfer_capture_dir(dir:str):
    '''
    将某个文件夹下的所有录像转化为数据集

    Args:
        dir(str): 需要转化的路径名
    '''

    # 处理过的csv文件会写入当前文件夹下的records.txt
    if os.path.exists(os.path.join(dir,"records.txt")):
        os.remove(os.path.join(dir,"records.txt"))

    for csv_file in Path(dir).rglob("*.csv"):
        # .stem 接口可用于获取不包含后缀的文件名

        process_raw_csv(path=dir, record=csv_file.stem)
        # clip_compression(path=dir, record=csv_file.stem)
        clip_compression_opencv(path=dir,record=csv_file.stem)
        calculate_optical_flow(record_dir=dir,record=csv_file.stem,use_large=True)
        # calculate_opticcal_flow_torch(record_dir=dir,record=csv_file.stem)


if __name__ == "__main__":

    transfer_capture_dir("E:\\User\\Pictures\\yolo_pics\\genshin_train\\capture\\2025-03-05")
    pass