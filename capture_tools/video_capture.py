#-*- coding: utf-8 -*-
import sys
import os
import time
import datetime
import threading
import json
import win32gui
import win32ui
import win32con
import win32api
import cv2
import pygetwindow as gw
import numpy as np
from ctypes import windll
from PIL import Image, ImageGrab
from pynput import keyboard, mouse
from mss import mss

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import SCREENSHOT_DIR,CAPTURE_ROOT
from utils.file_utils import exist_path,date_path,now_file_name

def on_press(key):
    try:
        if key.char == 'q':  # 检测是否按下了 'q' 键
            print("按键 'q' 被按下，退出程序")
            return False  # 返回 False 以停止监听器
    except AttributeError:
        pass  # 忽略非字符键

def rec_screen(**kwargs):
    '''
    需要提供的字段:
    save_path : str, 录像需要保存的路径
    '''
    if("save_path" in kwargs):
        save_path = kwargs["save_path"]
    else:
        raise KeyError("未提供字段 'save_path'")

    # 设置录制区域
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # 录制整个屏幕
    fps = 30.0  # 帧率
    output_file = now_file_name("mp4")  # 输出文件名
    output_file = os.path.join(save_path, output_file)

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, (monitor["width"], monitor["height"]))

    # 录制屏幕
    print("Recording started... Press 'q' to stop.")
    with mss() as sct:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        while listener.is_alive():
            
            # 捕获屏幕帧
            frame = np.array(sct.grab(monitor))  # 捕获屏幕
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换颜色空间

            # 写入视频文件
            out.write(frame)
            print("Frame wrote")

            # 显示录制画面（可选）
            # cv2.imshow("Screen Recording", frame)

            # 按下 'q' 键停止录制


    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped. Video saved as", output_file)

def run():
    path = exist_path(CAPTURE_ROOT, date_path())
    print(f"Captures will save at {path}")

    rec_screen(save_path=path)
    pass

if __name__ == "__main__":
    run()
    pass