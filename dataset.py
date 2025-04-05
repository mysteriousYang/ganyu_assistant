#-*- coding: utf-8 -*-
import os
import json
import cv2
import torch.nn
import argparse
import gc

import numpy as np

from memory_profiler import profile
from tqdm import tqdm
from typing import List
from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip
from pathlib import Path
from utils.logger import logger,timer_logger
from config import Dataset_Config


class Record_Info:
    def __init__(self, record_dir, record):
        self.record = record
        self.record_dir = record_dir

        self.info_file = os.path.join(self.record_dir,f"{self.record}_info.json")
        self.data = None
        pass

    def __getitem__(self,key):
        self._load()
        return self.data[key]
    
    def __contains__(self,key):
        self._load()
        result = key in self.data
        self.data = None
        return result

    def __setitem__(self,key,value):
        self._load()
        self.data[key] = value
        self._flush()
        pass

    def __str__(self,):
        self._load()
        result = str(self.data)
        self.data = None
        return result

    def _touch(self):
        with open(self.info_file,mode='w',encoding="utf-8") as fp:
            fp.write("{}")


    def _flush(self):
        with open(self.info_file,mode='w',encoding='utf-8') as fp:
            json.dump(self.data,fp)
        self.data = None

    def _load(self):
        if not (os.path.exists(self.info_file)):
            self._touch()

        if self.data is None:
            with open(self.info_file,mode='r',encoding="utf-8") as fp:
                self.data = json.load(fp)
        else:
            return



class Control_Record(Dataset):
    '''
    描述控制录像的类

    '''
    def __init__(self,
                 record_dir,
                 record):
        # self.record_path = record_path
        self.record = record
        self.record_dir = record_dir
        self.clip_in_ram = False

        self.raw_clip_file = os.path.join(self.record_dir,f"{self.record}.{Dataset_Config.video_suffix}")
        self.raw_clip_file_S = os.path.join(self.record_dir,f"{self.record}_s.{Dataset_Config.video_suffix}")
        self.flow_clip_file = os.path.join(self.record_dir,f"{self.record}_flow.{Dataset_Config.flow_video_suffix}i")
        self.flow_clip_file_S = os.path.join(self.record_dir,f"{self.record}_s_flow.{Dataset_Config.flow_video_suffix}")
        self.kb_file = os.path.join(self.record_dir,f"{self.record}.kb")
        self.ms_file = os.path.join(self.record_dir,f"{self.record}.ms")
        self.kb_np_file = os.path.join(self.record_dir,f"{self.record}_kb.npy")
        self.ms_np_file = os.path.join(self.record_dir,f"{self.record}_ms.npy")

        # 读取视频信息
        # self.info_file = os.path.join(self.record_dir,f"{self.record}_info.json")
        # with open(self.info_file,mode="r") as json_in:
        #     self.info = json.load(json_in)
        #     logger.debug(f"{self.record}: 成功读取录像信息")
        self.info = Record_Info(record_dir,record)

        self.video_obj = None
        self.X_mapfile = os.path.join(self.record_dir,f"{self.record}_X.memmap")
        self.Y_mapfile = os.path.join(self.record_dir,f"{self.record}_Y.memmap")
        self.X = None
        self.Y = None

        # self.make_control_seq()
        self.process_records()

        # self.TOTAL_ROWS = self.X.shape[0]

        logger.debug(f"{self.record}: 对象构造完成 {repr(self)}")

    def __len__(self):
        return self.info["block_count"] - 1
    
    def __repr__(self):
        return f"<Rec {self.record}>"

    def __getitem__(self, idx):
        """
        根据索引返回一个数据样本
        :param index: 索引

        按索引获取数据。

        参数:
            idx (int): 数据索引。

        返回:
            x_block (np.array): X 序列的块。
            y_block (np.array): Y 序列的块。
        """
        # start = idx * self.BLOCK_SIZE
        # end = min((idx + 1) * self.BLOCK_SIZE, self.TOTAL_ROWS)

        # 读取 X 和 Y 的块
        # x_block = self.X[start:end]
        # y_block = self.Y[start:end]
        x_block = self.X[idx]
        y_block = self.Y[idx]

        # 方式1
        # x_tensor = torch.from_numpy(x_block.copy()).float()  # [T,H,W,C]
        # y_tensor = torch.from_numpy(y_block.copy()).float()  # [T,H,W,C]

        # 方式2
        # x_tensor = torch.as_tensor(x_block).float()
        # x_tensor = x_tensor.permute(0, 3, 1, 2)  # [T,C,H,W]
        # y_tensor = torch.as_tensor(y_block).float()
        # del x_block
        # del y_block
        # return x_tensor, y_tensor

        # 方式3
        # x_tensor = torch.from_numpy(x_block)  # 保持uint8类型 [T,H,W,C]
        # y_tensor = torch.from_numpy(y_block)  # 保持uint16类型 [T,H,W,C]
        # x_float = x_tensor.permute(0, 3, 1, 2).to(torch.float32)  # [T,C,H,W], float32
        # y_float = y_tensor.to(torch.float32)  # 按需转换

        # 方式4
        x_float = torch.from_numpy(x_block)
        y_float = torch.from_numpy(y_block)

        # 显式释放原始块（如果后续不再需要）
        del x_block, y_block
        gc.collect()
        return x_float,y_float

    def _fill_key_press_status(self, press_frames, release_frames):
        """
        将按键按下与释放之间的所有帧标记为“按键按下”。
        
        参数:
            press_frames (list): 按键按下的帧号列表。
            release_frames (list): 按键释放的帧号列表。
            total_frames (int): 视频的总帧数。
        
        返回:
            key_status (np.array): 形状为(total_frames,)，表示每一帧的按键状态。
        """
        # 初始化全零数组
        key_status = np.zeros(self.info["total_frames"], dtype=np.float32)
        
        # 遍历按下与释放的帧对
        for press, release in zip(press_frames, release_frames):
            if press < release:
                # logger.debug(f"{press}:{release} = true")
                key_status[press:release + 1] = 1.0  # 按下到释放之间的帧标记为1
            else:
                logger.warning(f"{repr(self)}: 警告：按下帧 {press} 大于释放帧 {release}，跳过该对")
        
        return key_status

    def fill_key_press(self):
        """
        将按键按下与释放之间的所有帧标记为“按键按下”。
        
        参数:
        
        返回:
            key_status (np.array): 形状为(total_frames,)，表示每一帧的按键状态。

        控制操作需要的按键:
            W       12
            S       22
            A       21
            D       23
            shift   100
            ctrl    101
            space   109
            X       31
            共8个键
        """

        # 初始化全零数组
        # (行，列) = (总帧数，需要学习的按键数量)
        key_status = np.zeros((self.info["total_frames"],len(Dataset_Config.learn_keys)), dtype=np.float32)

        key_events = dict()
        with open(".\\key_mapping.json",mode='r',encoding="utf-8") as km_fp:
            key_map = json.load(km_fp)
        for key in Dataset_Config.learn_keys:
            key_events[ str(key_map[key]) ] = ([], [])
        # logger.debug(key_events)
        # key_events = {
        #     # ( [press], [release] )
        #     "12":([],[]),    # W
        #     "22":([],[]),    # S
        #     "21":([],[]),    # A
        #     "23":([],[]),    # D
        #     "100":([],[]),   # shift
        #     "101":([],[]),   # ctrl
        #     "109":([],[]),   # space
        #     "31":([],[]),    # X
        # }

        logger.debug(f"{self.record}: 正在填充按键数据")
        with open(self.kb_file,mode='r',encoding="utf-8") as fp:
            while True:
                # 字段结构
                # 状态（0为按下，1为松开） |  按键编号  |  帧编号
                line = fp.readline()
                if(line == ''):break
                # print(line.strip().split('\t'))
                status,key,frame_num = line.strip().split('\t')

                # 只处理需要统计的按键
                if(key in key_events):
                    if(status == '0'):
                        key_events[key][0].append(int(frame_num))
                    elif(status == '1'):
                        key_events[key][1].append(int(frame_num))

        for i, (key, (press_frames, release_frames)) in enumerate(key_events.items()):
            # 注意i是从1开始
            key_status[:, i - 1] = self._fill_key_press_status(press_frames, release_frames)

        # for line in key_status:
        #     logger.debug(line)
        return key_status

    def make_control_seq(self,block_size=Dataset_Config.BLOCK_SIZE):

        self.BLOCK_SIZE = block_size
        self.kb_event = self.fill_key_press()

        # keyboard
        # 计算需要填充的帧数
        logger.debug(f"{self.record}: 正在构建键盘数据块")
        remainder = self.info["total_frames"] % self.BLOCK_SIZE
        # if remainder != 0:
        padding_size = self.BLOCK_SIZE - remainder
        kb_data = np.pad(self.kb_event, ((0, padding_size), (0, 0)), mode="constant")  # 填充0
        # else:
            # kb_data = self.kb_event
        # 切分为 (x, self.BLOCK_SIZE, 8) 的三维数组
        x = kb_data.shape[0] // self.BLOCK_SIZE
        self.kb_block = kb_data.reshape(x, self.BLOCK_SIZE, len(Dataset_Config.learn_keys))

        # 保存键盘数组
        np.save(self.kb_np_file,self.kb_block)
        
        # mouse
        logger.debug(f"{self.record}: 正在构建鼠标数据块")
        block_count = (self.info["total_frames"]+padding_size) // self.BLOCK_SIZE
        self.ms_block = np.zeros((block_count, self.BLOCK_SIZE, 2), dtype=np.float32)
        self.info["block_count"] = block_count
        self.BLOCK_COUNT = block_count

        ms_events = np.loadtxt(self.ms_file,dtype=np.int16,delimiter='\t')
        line_count = 0

        with tqdm(total=len(ms_events), desc="处理鼠标数据") as pbar:
            for _, x, y, frame_number in ms_events:
                x /= self.info["original_width"]
                y /= self.info["original_height"]

                self.ms_block[int(frame_number//self.BLOCK_SIZE)][int(frame_number%self.BLOCK_SIZE)] = [x,y]

                line_count-=-1
                # if(line_count%1000 == 0):
                #     logger.debug(f"Processing line {line_count}")
                pbar.update(1)

        # self.BLOCK_COUNT = self.ms_block.shape[0]

        # 保存鼠标数组
        np.save(self.ms_np_file,self.ms_block)
        del ms_events


        # with open(self.ms_file,mode='r',encoding="utf-8") as fp:
            
        #     while True:
        #         line = fp.readline()
        #         if(line == ''):break
        #         # 此处只处理移动事件，第一列表示类型的不用管
        #         _, x, y, frame_number = line.strip().split('\t')
        #         x = int(x)
        #         y = int(y)
        #         frame_number=int(frame_number)

        #         # 归一化
        #         x /= self.info["original_width"]
        #         y /= self.info["original_height"]

        #         try:
        #             self.ms_block[int(frame_number//self.BLOCK_SIZE)][int(frame_number%self.BLOCK_SIZE)] = [x,y]
        #         except OverflowError:
        #             # 可能会出现坐标值小于0的情况，原因暂时未知
        #             logger.error(f"{repr(self)}: 帧 {frame_number} 错误的位置值({x},{y})")
                
        #         line_count-=-1
        #         if(line_count%1000 == 0):
        #             logger.debug(f"Processing line {line_count}")
        #         del x,y,frame_number
        #         gc.collect()
        # 计算需要填充的帧数
        # logger.debug(f"{self.record}: 正在填充鼠标数据块")
        # if remainder != 0:
        #     padding_size = self.BLOCK_SIZE - remainder
        #     # ms_data = np.pad(self.ms_event, ((0, padding_size), (0, 0)), mode="constant")  # 填充0
        #     zeros_array = np.zeros(shape=(padding_size, 2), dtype=np.float32)
        #     ms_data = np.concatenate([self.ms_event, zeros_array], axis=0)
        # x = self.ms_event.shape[0] // self.BLOCK_SIZE
        # logger.debug(f"正在重整数组")
        # self.ms_block = self.ms_event.reshape(x, self.BLOCK_SIZE, 2)
        # del ms_data,zeros_array

    def get_block(self,idx):
        '''
        获取第idx个100帧块
        '''
        return self.kb_block[idx],self.ms_block[idx]
    
    def get_frame(self,idx,use_large=False):
        '''
        获取某一帧的原图像和光流图像
        '''
        if not self.clip_in_ram:
            self.load_clip_from_disk()
        
        if(use_large):
            self.video_obj["raw"].set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.video_obj["flow"].set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw_frame = self.video_obj["raw"].read()
            ret, flow_frame = self.video_obj["flow"].read()

            if not ret:
                raise ValueError(f"无法读取帧 {idx}")
            
            return raw_frame ,flow_frame
        
        # 默认读取小视频
        self.video_obj["raw_S"].set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.video_obj["flow_S"].set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, raw_frame = self.video_obj["raw_S"].read()
        ret2, flow_frame = self.video_obj["flow_S"].read()

        if (not ret1) or (not ret2):
            raise ValueError(f"无法读取帧 {idx}")
        
        return raw_frame ,flow_frame


    def get_frame_range(self,begin,end):
        if not(self.clip_in_ram):
            self.load_clip_from_disk()
        
        return self.raw_clip_frames[begin:end,:], self.flow_clip_frames[begin:end,:]

    def release_clip(self):
        '''
        释放内存中的帧信息
        '''
        for key,cap in self.video_obj.items():
            cap.release()
        
        self.raw_clip_frames = None
        self.flow_clip_frames = None
        self.kb_block = None
        self.ms_block = None
        self.kb_event = None
        self.ms_event = None
        self.clip_in_ram = False
        gc.collect()

        logger.debug(f"{self.record} : 已清理")

    def load_clip_from_disk(self,use_large=False):
        if(self.clip_in_ram):
            return
        
        # load 
        if(self.video_obj is None):
            self.video_obj = dict()

        if("flag_raw_S" in self.info):
            self.video_obj["raw_S"] = cv2.VideoCapture(self.raw_clip_file_S)
        if("flag_flow_S" in self.info):
            self.video_obj["flow_S"] = cv2.VideoCapture(self.flow_clip_file_S)
        if(use_large and "flag_raw" in self.info):
            self.video_obj["raw"] = cv2.VideoCapture(self.raw_clip_file)
        if(use_large and "flag_flow" in self.info):
            self.video_obj["flow"] = cv2.VideoCapture(self.flow_clip_file)
        
        if(use_large):
            raw_video_reader = self.video_obj["raw"]
            flow_video_reader = self.video_obj["flow"]
        else:
            raw_video_reader = self.video_obj["raw_S"]
            flow_video_reader = self.video_obj["flow_S"]

        self.raw_clip_frames = list()
        self.flow_clip_frames = list()

        while True:
            ret1, raw_frame = raw_video_reader.read()
            ret2, flow_frame = flow_video_reader.read()

            if (not ret1) or (not ret2):
                break

            self.raw_clip_frames.append(raw_frame)
            self.flow_clip_frames.append(flow_frame)

        # logger.info(self.raw_clip_frames)

        # self.raw_clip_frames = np.array(self.raw_clip_frames,dtype=np.uint8)
        # self.flow_clip_frames = np.array(self.flow_clip_frames,dtype=np.uint8)

        # self.raw_clip_frames = self.raw_clip_frames.astype(np.float32) / 255.0
        # self.flow_clip_frames = self.flow_clip_frames.astype(np.float32) / 255.0

        self.raw_clip_frames = np.array(self.raw_clip_frames, dtype=np.float32) / 255.0
        self.flow_clip_frames = np.array(self.flow_clip_frames, dtype=np.float32) / 255.0

        # logger.debug(self.raw_clip_frames.shape)

        remainder = self.info["total_frames"] % self.BLOCK_SIZE
        if remainder == 0:
            return  # 无需填充

        padding_size = self.BLOCK_SIZE - remainder + 1

        # 创建填充帧, 填充值为0
        padding_shape = (padding_size,) + self.raw_clip_frames.shape[1:]
        padding_frames = np.full(padding_shape, 0, dtype=np.float32)

        # logger.debug(self.raw_clip_frames.shape)
        # logger.debug(padding_frames.shape)

        # 合并原始帧和填充帧
        self.raw_clip_frames = np.concatenate([self.raw_clip_frames, padding_frames], axis=0)
        self.flow_clip_frames = np.concatenate([self.flow_clip_frames, padding_frames], axis=0)

        self.clip_in_ram = True

    def _dump_clip_frames(self,use_large=False):
        # load 
        if(self.video_obj is None):
            self.video_obj = dict()

        if("flag_raw_S" in self.info):
            self.video_obj["raw_S"] = cv2.VideoCapture(self.raw_clip_file_S)
        if("flag_flow_S" in self.info):
            self.video_obj["flow_S"] = cv2.VideoCapture(self.flow_clip_file_S)
        if(use_large and "flag_raw" in self.info):
            self.video_obj["raw"] = cv2.VideoCapture(self.raw_clip_file)
        if(use_large and "flag_flow" in self.info):
            self.video_obj["flow"] = cv2.VideoCapture(self.flow_clip_file)
        
        if(use_large):
            raw_video_reader = self.video_obj["raw"]
            flow_video_reader = self.video_obj["flow"]
        else:
            raw_video_reader = self.video_obj["raw_S"]
            flow_video_reader = self.video_obj["flow_S"]

        flow_frame_list = list()

        with tqdm(total=self.info["block_count"], desc="处理视频数据") as pbar:
            _EOF_FLAG = False
            block_count = 0
            while not _EOF_FLAG:
                for i in range(Dataset_Config.BLOCK_SIZE):
                    ret, flow_frame = flow_video_reader.read()

                    if not ret:
                        flow_frame_list.append(flow_frame_list[-1])
                        _EOF_FLAG = True
                    else:
                        flow_frame = flow_frame.astype("float32")
                        flow_frame /= np.float32(255.0)
                        flow_frame_list.append(flow_frame)
                

                X_append = np.array(flow_frame_list, dtype=np.float32)
                X_append = X_append[np.newaxis, ...]
                flow_frame_list.clear()

                self.X = self._create_or_extend_memmap(self.X_mapfile, np.float32, X_append.shape)
                self.X[-1] = X_append
                self.X.flush()
                block_count += 1
                # logger.debug(self.X.shape)
                # self.raw_clip_frames.append(raw_frame)
                # self.flow_clip_frames.append(flow_frame)
                pbar.update(1)
        logger.debug(self.X.shape)
        pass

    def _dump_ctrl_block(self):
        with tqdm(total=len(self), desc="处理标签数据") as pbar:
            for j in range(len(self)):
                kb_block, ms_block = self.get_block(j)
                Y_append = np.concatenate([kb_block, ms_block], axis=-1)
                Y_append = Y_append[np.newaxis, ...]
                # logger.debug(X_append.shape)
                # logger.debug(Y_append.shape)


                self.Y = self._create_or_extend_memmap(self.Y_mapfile,np.float32,Y_append.shape)
                self.Y[-1] = Y_append
                self.Y.flush()

                # logger.debug(f"{repr(self)}: 进度 {self.X.shape[0]}")
                # logger.debug(self.X.shape)
                # logger.debug(self.Y.shape)
                pbar.update(1)

    def _create_or_extend_memmap(self, file_path, dtype, shape, mode="r+"):
        """
        创建或扩展 np.memmap 文件。

        参数:
            file_path (str): 文件路径。
            dtype: 数据类型。
            shape (tuple): 文件的形状。
            mode (str): 文件模式（"r+" 表示读写）。

        返回:
            memmap (np.memmap): 内存映射文件对象。
        """
        if os.path.exists(file_path):
            # 如果文件已存在，扩展其大小
            existing_memmap = np.memmap(file_path, dtype=dtype, mode="r")

            # 计算现有文件的总元素数
            total_elements = existing_memmap.size
            
            # 计算预期的总元素数
            expected_elements = np.prod(shape[1:])

            # 计算现有文件的帧数
            num_frames = total_elements // expected_elements
            # now_blocks = num_frames // self.BLOCK_SIZE
            
            # 将 existing_memmap reshape 为正确的形状
            existing_memmap = existing_memmap.reshape((num_frames,) + shape[1:])
            # existing_memmap = existing_memmap.reshape((now_blocks,self.BLOCK_SIZE) + shape[1:])

            # logger.debug(existing_memmap.shape)
            # logger.debug(shape)
            new_shape = (existing_memmap.shape[0] + shape[0],) + existing_memmap.shape[1:]
            memmap = np.memmap(file_path, dtype=dtype, mode=mode, shape=new_shape)
        else:
            # 如果文件不存在，创建新文件
            memmap = np.memmap(file_path, dtype=dtype, mode="w+", shape=shape)
        return memmap
    
    def _load_memmap_fp(self):
        # with open(self.info_file,mode="r",encoding="utf-8") as fp:
        #     memmap_info = json.load(fp)
        #     self.X = np.memmap(self.X_mapfile, dtype=np.float32, mode="r", shape=memmap_info["X_shape"])
        #     self.Y = np.memmap(self.Y_mapfile, dtype=np.float32, mode="r", shape=memmap_info["Y_shape"])
        self.X = np.memmap(self.X_mapfile, dtype=np.float32, mode="r", shape=self.info["X_shape"])
        self.Y = np.memmap(self.Y_mapfile, dtype=np.float32, mode="r", shape=self.info["Y_shape"])

    def process_records(self,block_size=Dataset_Config.BLOCK_SIZE):
        """
        将类中的records转化为可供使用的数据集
        """
        self.BLOCK_SIZE = block_size

        if(os.path.exists(self.X_mapfile) and os.path.exists(self.Y_mapfile)):
            logger.info(f"{repr(self)}: 数据集已存在，直接读取")
            self._load_memmap_fp()
            self.BLOCK_COUNT = self.info["block_count"]
            return
        
        if(os.path.exists(self.kb_np_file) and os.path.exists(self.ms_np_file)):
            logger.info(f"{repr(self)}: x和y已存在，直接读取")
            self.kb_block = np.load(self.kb_np_file)
            self.ms_block = np.load(self.ms_np_file)
            self.BLOCK_COUNT = len(self.kb_block)
        else:
            self.make_control_seq(block_size)
        
        self._dump_ctrl_block()
        self._dump_clip_frames()
        # process_info = dict()
        # self._rec_offset = dict() # 用于记录每个录像文件对应的offset
        # offset = 0

        # self._rec_offset[self.record] = offset
        # offset += self.info["total_frames"]

        # logger.debug(f"正在处理视频数据")
        # with tqdm(total=len(self), desc="处理视频数据") as pbar:
        #     for j in range(len(self)):
        #         kb_block, ms_block = self.get_block(j)
        #         raw_frames, flow_frames = self.get_frame_range(j*self.BLOCK_SIZE,(j+1)*self.BLOCK_SIZE)

        #         X_append = np.concatenate([raw_frames, flow_frames], axis=-1)
        #         Y_append = np.concatenate([kb_block, ms_block], axis=-1)

        #         # logger.debug(X_append.shape)
        #         # logger.debug(Y_append.shape)

        #         self.X = self._create_or_extend_memmap(self.X_mapfile,np.float32,X_append.shape)
        #         self.X[-X_append.shape[0]:] = X_append
        #         self.X.flush()

        #         self.Y = self._create_or_extend_memmap(self.Y_mapfile,np.float32,Y_append.shape)
        #         self.Y[-Y_append.shape[0]:] = Y_append
        #         self.Y.flush()

        #         # logger.debug(f"{repr(self)}: 进度 {self.X.shape[0]}")
        #         # logger.debug(self.X.shape)
        #         # logger.debug(self.Y.shape)
        #         pbar.update(1)

        # del X_append
        # del Y_append
        # del kb_block
        # del ms_block
        # del raw_frames
        # del flow_frames
        # self.release_clip()
        # logger.debug(f"{len(self.X)}")
        # logger.debug(f"{len(self.Y)}")
    
        self.info["X_shape"] = self.X.shape
        self.info["Y_shape"] = self.Y.shape

        logger.debug(self.info)

        # with open(self.info_file,mode="w",encoding="utf-8") as fp:
        #     json.dump(self.info,fp)


# class Genshin_Basic_Control_Dataset(Dataset):
#     def __init__(self,
#                  record_path,
                
#                 **kwargs
#                  ):
#         '''
#         初始化用于训练动作控制模型的数据集
#         由于训练集实在太大，采用memmap方式进行实现
        
#         Args:
#             record_path (str) : 需要传入指向record.txt的路径
#         '''

#         self.capture_dir = os.path.dirname(record_path)
#         self.mouse_data = dict()
#         self.keyboard_data = dict()
#         self.info_file = os.path.join(self.capture_dir,"GSDataset_info.json")
    
#         # 读取记录
#         self.records = list() # record记录号
#         self.record_objs = list() # record对象
#         with open(record_path,mode='r',encoding="utf-8") as fp:
#             for record in fp.readlines():
#                 record = record.strip()
#                 self.records.append(record)
#                 self.record_objs.append(Control_Record(self.capture_dir,record))

#         if("block_size" in kwargs):
#             self.BLOCK_SIZE = kwargs["block_size"]
#         else:
#             self.BLOCK_SIZE = 100

#         self.X_mapfile = os.path.join(self.capture_dir,"GSDataset_X.memmap")
#         self.Y_mapfile = os.path.join(self.capture_dir,"GSDataset_Y.memmap")
#         self.X = None
#         self.Y = None

#         self.process_records()

#         self.TOTAL_ROWS = self.X.shape[0]

#     def __len__(self):
#         """
#         返回数据集的大小
#         """
#         return self.TOTAL_ROWS // self.BLOCK_SIZE

#     def __del__(self):
#         del self.X
#         del self.Y

#     # @profile
#     def __getitem__(self, idx):
#         """
#         根据索引返回一个数据样本
#         :param index: 索引
#         """
#         """
#         按索引获取数据。

#         参数:
#             idx (int): 数据索引。

#         返回:
#             x_block (np.array): X 序列的块。
#             y_block (np.array): Y 序列的块。
#         """
#         start = idx * self.BLOCK_SIZE
#         end = min((idx + 1) * self.BLOCK_SIZE, self.TOTAL_ROWS)

#         # 读取 X 和 Y 的块
#         x_block = self.X[start:end]
#         y_block = self.Y[start:end]

#         # 方式1
#         # x_tensor = torch.from_numpy(x_block.copy()).float()  # [T,H,W,C]
#         # y_tensor = torch.from_numpy(y_block.copy()).float()  # [T,H,W,C]

#         # 方式2
#         # x_tensor = torch.as_tensor(x_block).float()
#         # x_tensor = x_tensor.permute(0, 3, 1, 2)  # [T,C,H,W]
#         # y_tensor = torch.as_tensor(y_block).float()
#         # del x_block
#         # del y_block
#         # return x_tensor, y_tensor

#         # 方式3
#         x_tensor = torch.from_numpy(x_block)  # 保持uint8类型 [T,H,W,C]
#         y_tensor = torch.from_numpy(y_block)  # 保持uint16类型 [T,H,W,C]
#         x_float = x_tensor.permute(0, 3, 1, 2).to(torch.float32)  # [T,C,H,W], float32
#         y_float = y_tensor.to(torch.float32)  # 按需转换

#         # 显式释放原始块（如果后续不再需要）
#         del x_block, y_block, x_tensor, y_tensor
#         gc.collect()
#         return x_float,y_float
        


#     def _init_mem_queue(self,queue_size=10):
#         self._mem_queue = queue.Queue(queue_size)

#     def _create_or_extend_memmap(self, file_path, dtype, shape, mode="r+"):
#         """
#         创建或扩展 np.memmap 文件。

#         参数:
#             file_path (str): 文件路径。
#             dtype: 数据类型。
#             shape (tuple): 文件的形状。
#             mode (str): 文件模式（"r+" 表示读写）。

#         返回:
#             memmap (np.memmap): 内存映射文件对象。
#         """
#         if os.path.exists(file_path):
#             # 如果文件已存在，扩展其大小
#             existing_memmap = np.memmap(file_path, dtype=dtype, mode="r")

#             # 计算现有文件的总元素数
#             total_elements = existing_memmap.size
            
#             # 计算预期的总元素数
#             expected_elements = np.prod(shape[1:])

#             # 计算现有文件的帧数
#             num_frames = total_elements // expected_elements
            
#             # 将 existing_memmap reshape 为正确的形状
#             existing_memmap = existing_memmap.reshape((num_frames,) + shape[1:])

#             # logger.debug(existing_memmap.shape)
#             # logger.debug(shape)
#             new_shape = (existing_memmap.shape[0] + shape[0],) + existing_memmap.shape[1:]
#             memmap = np.memmap(file_path, dtype=dtype, mode=mode, shape=new_shape)
#         else:
#             # 如果文件不存在，创建新文件
#             memmap = np.memmap(file_path, dtype=dtype, mode="w+", shape=shape)
#         return memmap
    
#     def _load_memmap_fp(self):
#         with open(self.info_file,mode="r",encoding="utf-8") as fp:
#             memmap_info = json.load(fp)
#             self.X = np.memmap(self.X_mapfile, dtype=np.uint8, mode="r", shape=memmap_info["X_shape"])
#             self.Y = np.memmap(self.Y_mapfile, dtype=np.uint16, mode="r", shape=memmap_info["Y_shape"])

#     def process_records(self):
#         """
#         将类中的records转化为可供使用的数据集
#         """
#         if(os.path.exists(self.X_mapfile) and os.path.exists(self.Y_mapfile)):
#             logger.info(f"{repr(self)}: 数据集已存在，直接读取")
#             self._load_memmap_fp()
#             return
        
#         process_info = dict()
#         self._rec_offset = dict() # 用于记录每个录像文件对应的offset
#         offset = 0

#         for i,record_obj in enumerate(self.record_objs):
#             record_obj:Control_Record
#             record_obj.make_control_seq()

#             self._rec_offset[record_obj.record] = offset
#             offset += record_obj.info["total_frames"]

#             for j in range(len(record_obj)):
#                 kb_block, ms_block = record_obj.get_block(j)
#                 raw_frames, flow_frames = record_obj.get_frame_range(j*record_obj.BLOCK_SIZE,(j+1)*record_obj.BLOCK_SIZE)

#                 X_append = np.concatenate([raw_frames, flow_frames], axis=-1)
#                 Y_append = np.concatenate([kb_block, ms_block], axis=-1)

#                 # logger.debug(X_append.shape)
#                 # logger.debug(Y_append.shape)

#                 self.X = self._create_or_extend_memmap(self.X_mapfile,np.uint8,X_append.shape)
#                 self.X[-X_append.shape[0]:] = X_append
#                 self.X.flush()

#                 self.Y = self._create_or_extend_memmap(self.Y_mapfile,np.uint16,Y_append.shape)
#                 self.Y[-Y_append.shape[0]:] = Y_append
#                 self.Y.flush()

#                 logger.debug(f"{repr(self)}: 进度 {self.X.shape[0]}")
#                 # logger.debug(self.X.shape)
#                 # logger.debug(self.Y.shape)

#             del X_append
#             del Y_append
#             del kb_block
#             del ms_block
#             del raw_frames
#             del flow_frames
#             record_obj.release_clip()
#             # logger.debug(f"{len(self.X)}")
#             # logger.debug(f"{len(self.Y)}")
        
#         process_info["X_shape"] = list(self.X.shape)
#         process_info["Y_shape"] = list(self.Y.shape)

#         # self.info_file = os.path.join(self.capture_dir,"GSDataset_info.json")
#         with open(self.info_file,mode="w",encoding="utf-8") as fp:
#             json.dump(process_info,fp)

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
        logger.info(f"正在处理 {record}")
    except:
        logger.exception("csv文件不存在")
        exit(-1)

    # 虽然这里是 .kb与.ms文件,但实际存储格式依然为csv
    try:
        kb_fp = open(os.path.join(path,f"{record}.kb"),mode="w",encoding="utf-8")
    except:
        logger.exception(f"{record}.kb 写入失败")
        exit(-1)

    try:
        ms_fp = open(os.path.join(path,f"{record}.ms"),mode="w",encoding="utf-8")
    except:
        logger.exception(f"{record}.ms 写入失败")
        exit(-1)

    # 读取按键映射表
    key_map = dict()
    try:
        with open(key_map_file,mode='r',encoding="utf-8") as json_fp:
            key_map = json.load(json_fp)
    except:
        logger.exception(f"无法读取按键映射表")
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
    
    logger.info(f"分离完成 {record}.csv")
    ms_fp.close()
    kb_fp.close()
    csv_fp.close()

    with open(os.path.join(path,"records.txt"),mode='a',encoding="utf-8") as fp:
        logger.info(f"已导出 {record}")
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
    

def clip_compression_moviepy(path:str,record:str,**kwargs):
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
        new_height = Dataset_Config.default_s_height
    
    if("width" in kwargs):
        new_width = kwargs["width"]
    else:
        new_width = Dataset_Config.default_s_width

    video_info = Record_Info(path,record)
    input_video = f"{record}.{Dataset_Config.video_suffix}"
    output_video = f"{record}_s.{Dataset_Config.video_suffix}"

    # 如果压制视频已存在, 则不作任何更改
    if(os.path.exists(os.path.join(path,output_video))):
        logger.info(f"已存在 {output_video}")
        return

    # 加载视频
    video = VideoFileClip(os.path.join(path,input_video))

    # 读取信息
    total_frames = int(video.fps * video.duration)
    original_fps = video.fps
    original_width, original_height = video.size

    logger.info(f"正在压制 {input_video}")
    logger.info(f"目标大小: {new_width}*{new_height}")


    # 调整分辨率
    resized_video = video.resize((new_width, new_height))

    # 保存调整后的视频
    resized_video.write_videofile(os.path.join(path,output_video))

    logger.info(f"压制完成 {output_video}")

    # 关闭视频对象
    video.close()
    resized_video.close()

    # 存储原始属性
    video_info["original_fps"] = original_fps
    video_info["original_width"] = original_width
    video_info["original_height"] = original_height
    video_info["total_frames"] = total_frames
    video_info["flag_raw"] = True
    video_info["flag_raw_S"] = True
    logger.info(f"视频信息已存储 {input_video}")

    # if (os.path.exists(os.path.join(path,f"{record}_info.json"))):
    #     with open(os.path.join(path,f"{record}_info.json"),mode='r',encoding="utf-8") as fp:
    #         video_info = json.load(fp)
    #     video_info["original_fps"] = original_fps
    #     video_info["original_width"] = original_width
    #     video_info["original_height"] = original_height
    #     video_info["total_frames"] = total_frames
    #     video_info["flag_raw"] = True
    #     video_info["flag_raw_S"] = True
    #     with open(os.path.join(path,f"{record}_info.json"),mode='w',encoding="utf-8") as fp:
    #         json.dump(video_info,fp)
    #     logger.info(f"视频信息已存储 {input_video}")

    # else:
    #     with open(os.path.join(path,f"{record}_info.json"),mode='w',encoding="utf-8") as fp:
    #         video_info = dict()
    #         video_info["original_fps"] = original_fps
    #         video_info["original_width"] = original_width
    #         video_info["original_height"] = original_height
    #         video_info["total_frames"] = total_frames
    #         video_info["flag_raw"] = True
    #         video_info["flag_raw_S"] = True
    #         json.dump(video_info,fp)
    #         logger.info(f"视频信息已存储 {input_video}")


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
        new_height = Dataset_Config.default_s_height
    
    if("width" in kwargs):
        new_width = kwargs["width"]
    else:
        new_width = Dataset_Config.default_s_width

    video_info = Record_Info(path,record)
    input_video = f"{record}.{Dataset_Config.video_suffix}"
    output_video = f"{record}_s.{Dataset_Config.video_suffix}"
    input_video_path = os.path.join(path,input_video)
    output_video_path = os.path.join(path,output_video)

    # 如果压制视频已存在, 则不作任何更改
    calc = True
    if(os.path.exists(os.path.join(path,output_video))):
        logger.info(f"已存在 {output_video}")
        calc = False
    
    # 检查OpenCV是否支持CUDA
    # if not cv2.cuda.getCudaEnabledDeviceCount():
    #     print("CUDA不可用，请安装CUDA版OpenCV")
    #     return

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.exception(f"无法打开视频 {input_video_path}")
        return

    # 获取视频的原始属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if(calc):
        # 设置输出视频的编码器和属性
        fourcc = cv2.VideoWriter_fourcc(*Dataset_Config.video_enc)  # 使用MP4编码器
        out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (new_width, new_height))

        # 创建CUDA加速的帧处理管道
        cuda_stream = cv2.cuda_Stream()  # 创建CUDA流
        # cuda_resizer = cv2.cuda_Resize(new_width, new_height)  # 创建CUDA缩放器

        logger.info(f"正在压制 {input_video}")
        logger.info(f"目标大小: {new_width}*{new_height}")

        with tqdm(total=total_frames, desc="视频压制") as pbar:
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
                pbar.update(1)

        # 释放资源
        cap.release()
        out.release()
        logger.info(f"压制完成 {output_video}")

    # 存储原始属性
    video_info["original_fps"] = original_fps
    video_info["original_width"] = original_width
    video_info["original_height"] = original_height
    video_info["s_width"] = new_width
    video_info["s_height"] = new_height
    video_info["total_frames"] = total_frames
    video_info["flag_raw"] = True
    video_info["flag_raw_S"] = True
    logger.info(f"视频信息已存储 {input_video}")

    # if (os.path.exists(os.path.join(path,f"{record}_info.json"))):
    #     with open(os.path.join(path,f"{record}_info.json"),mode='r',encoding="utf-8") as fp:
    #         video_info = json.load(fp)
    #     with open(os.path.join(path,f"{record}_info.json"),mode='w',encoding="utf-8") as fp:
    #         video_info["original_fps"] = original_fps
    #         video_info["original_width"] = original_width
    #         video_info["original_height"] = original_height
    #         video_info["s_width"] = new_width
    #         video_info["s_height"] = new_height
    #         video_info["total_frames"] = total_frames
    #         video_info["flag_raw"] = True
    #         video_info["flag_raw_S"] = True
    #         json.dump(video_info,fp)
    #         logger.info(f"视频信息已存储 {input_video}")
    # else:
    #     with open(os.path.join(path,f"{record}_info.json"),mode='w',encoding="utf-8") as fp:
    #         video_info = dict()
    #         video_info["original_fps"] = original_fps
    #         video_info["original_width"] = original_width
    #         video_info["original_height"] = original_height
    #         video_info["s_width"] = new_width
    #         video_info["s_height"] = new_height
    #         video_info["total_frames"] = total_frames
    #         video_info["flag_raw"] = True
    #         video_info["flag_raw_S"] = True
    #         json.dump(video_info,fp)
    #         logger.info(f"视频信息已存储 {input_video}")

def choose_clip_compression(path:str,record:str,**kwargs):
    # 检查OpenCV是否支持CUDA
    if not cv2.cuda.getCudaEnabledDeviceCount():
        clip_compression_moviepy(path,record,**kwargs)
    else:
        clip_compression_opencv(path,record,**kwargs)

# @timer_logger
def calculate_optical_flow(record_dir:str,record:str,use_large:bool=False,**kwargs):
    '''
    光流计算, 使用OpenCV
    '''
    if not cv2.cuda.getCudaEnabledDeviceCount():
        logger.warning("CUDA 不可用，使用CPU计算")
        cuda_available = False
    else:
        logger.info("使用CUDA计算")
        cuda_available = True

    if use_large:
        logger.warning("警告: 正在使用原视频计算光流")
        input_video = f"{record}.{Dataset_Config.video_suffix}"
        output_video = f"{record}_flow.{Dataset_Config.flow_video_suffix}"
    else:
        input_video = f"{record}_s.{Dataset_Config.video_suffix}"
        output_video = f"{record}_s_flow.{Dataset_Config.flow_video_suffix}"

    # 打开视频文件
    cap = cv2.VideoCapture(os.path.join(record_dir,input_video))
    if not cap.isOpened():
        logger.exception(f"无法打开视频文件 {input_video}")
        return

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    calc = True
    if(os.path.exists(os.path.join(record_dir,output_video))):
        logger.info(f"光流视频已存在，不进行计算 {output_video}")
        calc = False
    else:
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*Dataset_Config.flow_video_enc)  # 视频编码格式
        # fourcc = cv2.VideoWriter_fourcc(*'IYUV')
        out = cv2.VideoWriter(os.path.join(record_dir,output_video), fourcc, fps, (width, height))

    if cuda_available and calc:
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

        _i = 0

        # 处理每一帧
        with tqdm(total=total_frames, desc="CUDA光流计算") as pbar:
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

                _i-=-1
                # if(_i % 1000 == 0):
                #     logger.info(f"正在处理第 {_i} 帧")

                pbar.update(1)

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        logger.info(f"光流视频已保存到: {os.path.join(record_dir,input_video)}")

    elif calc:
        # 读取第一帧并转换为灰度图像
        ret, old_frame = cap.read()
        if not ret:
            logger.info("视频为空")
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # 创建HSV图像用于绘制光流
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255  # 饱和度设为最大

        logger.info(f"正在绘制光流图...")
        with tqdm(total=total_frames, desc="CPU光流计算") as pbar:
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
                pbar.update(1)

        logger.info(f"光流视频已保存到: {os.path.join(record_dir,input_video)}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    video_info = Record_Info(record_dir,record)
    if(use_large):
        video_info["flag_flow"] = True
    else:
        video_info["flag_flow_S"] = True
    logger.info(f"视频信息已存储 {input_video}")

    # with open(os.path.join(record_dir,f"{record}_info.json"),mode="r",encoding="utf-8") as fp:
    #     info = json.load(fp)
    # with open(os.path.join(record_dir,f"{record}_info.json"),mode="w",encoding="utf-8") as fp:
    #     if(use_large):
    #         info["flag_flow"] = True
    #     else:
    #         info["flag_flow_S"] = True
    #     json.dump(info,fp)
    #     logger.info(f"视频信息已存储 {input_video}")

# 加载 RAFT 模型
def load_raft_model(args):
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.path))
    model.to('cuda')
    model.eval()
    return model

@timer_logger
def calculate_optical_flow_torch(record_dir:str,record:str):
    '''
    使用RAFT计算视频光流(不推荐)
    '''

    # 初始化模型
    args = argparse.Namespace(
        model='raft',
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        path='.\\Raft\\models\\raft-things.pth'
    )
    model = load_raft_model(args)

    input_video = f"{record}_s.{Dataset_Config.video_suffix}"
    output_video = f"{record}_s_flow_torch.{Dataset_Config.video_suffix}"

    # 读取视频
    video_path = os.path.join(record_dir,input_video)
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    output_path = os.path.join(record_dir,output_video)
    fourcc = cv2.VideoWriter_fourcc(*Dataset_Config.flow_video_enc)  # 视频编码格式
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

    logger.info(f"光流视频已保存到: {output_path}")


def transfer_capture_dir(dir:str):
    '''
    将某个文件夹下的所有录像转化为数据集

    Args:
        dir(str): 需要转化的路径名
    '''

    # 处理过的csv文件会写入当前文件夹下的records.txt
    if os.path.exists(os.path.join(dir,"records.txt")):
        os.remove(os.path.join(dir,"records.txt"))

    dataset_list = list()
    for csv_file in Path(dir).rglob("*.csv"):
        # .stem 接口可用于获取不包含后缀的文件名

        process_raw_csv(path=dir, record=csv_file.stem)
        # clip_compression(path=dir, record=csv_file.stem)
        # clip_compression_opencv(path=dir,record=csv_file.stem)
        choose_clip_compression(path=dir,record=csv_file.stem)
        calculate_optical_flow(record_dir=dir,record=csv_file.stem,)
        # calculate_opticcal_flow_torch(record_dir=dir,record=csv_file.stem)
        dataset_list.append(Control_Record(dir,csv_file.stem))
        # break

    # return Genshin_Basic_Control_Dataset(os.path.join(dir,"records.txt"))
    return dataset_list

def Get_Dataset_list(dir:str):
    dataset_list = transfer_capture_dir(dir)
    return dataset_list

def Make_Dataset(full_dataset,train_rate=0.8,val_rate=0.1):
    '''
    构建控制训练数据集

    Args:
        capture_dir:数据集路径
        train_rate:训练集占比(默认0.8)
        val_rate:评估集占比(默认0.1)
    
    Return:
        train_set, test_set, val_set
    '''

    # 自动划分（需要提前知道数据总量）
    train_size = int(train_rate * len(full_dataset))
    val_size = int(val_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    return train_set,test_set,val_set


if __name__ == "__main__":

    transfer_capture_dir("G:\\NN_train\\debug")
    pass