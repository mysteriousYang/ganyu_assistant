#-*- coding: utf-8 -*-
import sys
import os
import time
import threading
import ctypes
import cv2
import pygetwindow as gw
import numpy as np
from ctypes import windll
from PIL import Image, ImageGrab
from pynput import keyboard, mouse
from mss import mss
from dataset import Record_Info

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import SCREENSHOT_DIR,CAPTURE_ROOT
from utils.file_utils import exist_path,date_path,now_file_name
from utils.logger import logger

_C_g_cap_key = 'p' #用于终止录制的按键
_C_g_cap_func_key = 'alt' #用于终止录制的控制按键
_g_current_keys = set()
_g_key_press_events = []
_g_key_release_events = []
_g_mouse_move_events = []
_g_mouse_click_events = []
_g_mouse_scroll_events = []
_g_frame_count = 1
_g_key_abort = False
# _g_debug_flag = False

_key_press_lock = threading.Lock()
_key_release_lock = threading.Lock()
_mouse_move_lock = threading.Lock()
_mouse_click_lock = threading.Lock()
_mouse_scroll_lock = threading.Lock()


def terminate_capture_check(key):
    # logger.debug(_g_current_keys)
    if(_C_g_cap_func_key == "ctrl"):
        if any(ctrl in _g_current_keys for ctrl in [keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]) and \
        keyboard.KeyCode.from_char(_C_g_cap_key) in _g_current_keys:
            logger.debug(f"Detected Ctrl + {_C_g_cap_key} combination!")
            return False
        else:
            return True
    
    elif(_C_g_cap_func_key == "alt"):
        if any(alt in _g_current_keys for alt in [keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr]) and \
        keyboard.KeyCode.from_char(_C_g_cap_key) in _g_current_keys:
            global _g_key_abort
            _g_key_abort = True
            
            logger.debug(f"Detected Alt + {_C_g_cap_key} combination!")
            return False
        else:
            return True


def terminate_capture_press(key):
    _g_current_keys.add(key)
    return terminate_capture_check(key)


def terminate_capture_release(key):
    try:
        _g_current_keys.remove(key)
    except KeyError:
        pass

def gaming_key_press(key):
    '''
    key     : 0
    press   : 0
    '''
    try:
        line = f"0,0,{key.char.lower()},{_g_frame_count}\n"
    except AttributeError:
        line = f"0,0,{key.name},{_g_frame_count}\n"

    with _key_press_lock:
        _g_key_press_events.append(line)


def gaming_key_release(key):
    '''
    key     : 0
    release : 1
    '''
    try:
        line = f"0,1,{key.char.lower()},{_g_frame_count}\n"
    except AttributeError:
        line = f"0,1,{key.name},{_g_frame_count}\n"

    with _key_release_lock:
        _g_key_release_events.append(line)


def gaming_mouse_move(x, y):
    '''
    mouse   : 1
    move    : 0
    '''
    line = f"1,0,{str(x)},{str(y)},{_g_frame_count}\n"

    with _mouse_move_lock:
        _g_mouse_move_events.append(line)
    time.sleep(0.01)


def gaming_mouse_click(x, y, button, pressed):
    '''
    mouse   : 1
    click   : 1
    '''
    line = f"1,1,{str(x)},{str(y)},{str(button)},{str(pressed)},{_g_frame_count}\n"

    with _mouse_click_lock:
        _g_mouse_click_events.append(line)


def gaming_mouse_scroll(x, y, dx, dy):
    '''
    mouse   : 1
    scroll  : 2
    '''
    line = f"1,2,{str(x)},{str(y)},{str(dx)},{str(dy)},{_g_frame_count}\n"

    with _mouse_scroll_lock:
        _g_mouse_scroll_events.append(line)

def capture_screen(**kwargs):
    '''
    需要提供的字段: \n
    save_path : str, 录像需要保存的路径 \n
    stop_keys : tuple, 控制按键组合, 例: (alt, p) \n
    fps : float, 录制帧数 \n
    '''
    if("save_path" in kwargs):
        save_path = kwargs["save_path"]
    else:
        raise KeyError("未提供字段 'save_path'")
    
    if("stop_keys" in kwargs):
        global _C_g_cap_func_key
        global _C_g_cap_key
        _C_g_cap_func_key = kwargs["stop_keys"][0]
        _C_g_cap_key = kwargs["stop_keys"][1]


    if("fps" in kwargs):
        fps = kwargs["fps"]
    else:
        fps = 20.0

    # if("debug" in kwargs):
    #     global _g_debug_flag
    #     _g_debug_flag = True

    # 设置录制区域
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # 录制整个屏幕
    output_file = now_file_name()  # 输出文件名
    output_video = os.path.join(save_path, f"{output_file}{'mp4'}")
    output_csv   = os.path.join(save_path, f"{output_file}{'csv'}")

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器
    out = cv2.VideoWriter(output_video, fourcc, fps, (monitor["width"], monitor["height"]))

    # 录制屏幕
    # print(f"开始录像... 按下 {_C_g_cap_func_key} + {_C_g_cap_key} 停止录像")
    print(f"开始录像... 按下 {_C_g_cap_func_key} + {_C_g_cap_key} 停止录像")

    # 启动键鼠监听器
    key_listener = keyboard.Listener(
        on_press=gaming_key_press,on_release=gaming_key_release,
    )
    mouse_listener = mouse.Listener(
        on_move=gaming_mouse_move,on_click=gaming_mouse_click,on_scroll=gaming_mouse_scroll
    )
    key_listener.start()
    mouse_listener.start()

    global _g_frame_count
    with mss() as sct:
        # 用于终止录像的监听器
        capture_listener = keyboard.Listener(
            on_press=terminate_capture_press,
            on_release=terminate_capture_release
        )
        capture_listener.start()
        while capture_listener.is_alive():
            
            # 捕获屏幕帧
            frame = np.array(sct.grab(monitor))  # 捕获屏幕
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换颜色空间

            # 写入视频文件
            out.write(frame)
            # print("Frame wrote")

            _g_frame_count += 1
            # 每10帧存储一次
            if(_g_frame_count % 10 == 0):
                with open(output_csv,mode="a",encoding="utf-8") as fout:
                    
                    if not _g_key_press_events.__len__() == 0:
                        logger.debug(_g_key_press_events)

                    if not _g_key_release_events.__len__() == 0:
                        logger.debug(_g_key_release_events)

                    if not _g_mouse_move_events.__len__() == 0:
                        logger.debug(_g_mouse_move_events)

                    if not _g_mouse_click_events.__len__() == 0:
                        logger.debug(_g_mouse_click_events)

                    if not _g_mouse_scroll_events.__len__() == 0:
                        logger.debug(_g_mouse_scroll_events)


                    with _key_press_lock:
                        for line in _g_key_press_events:
                            fout.write(line)
                        _g_key_press_events.clear()
                    
                    with _key_release_lock:
                        for line in _g_key_release_events:
                            fout.write(line)
                        _g_key_release_events.clear()

                    with _mouse_move_lock:
                        for line in _g_mouse_move_events:
                            fout.write(line)
                        _g_mouse_move_events.clear()

                    with _mouse_click_lock:
                        for line in _g_mouse_click_events:
                            fout.write(line)
                        _g_mouse_click_events.clear()

                    with _mouse_scroll_lock:
                        for line in _g_mouse_scroll_events:
                            fout.write(line)
                        _g_mouse_scroll_events.clear()

            # 超过6000帧自动截断
            if(_g_frame_count >= 6000):
                logger.info(f"录像时长大于 6000 帧，已自动截断")
                _g_frame_count = 0
                capture_listener.stop()
                break

    # 释放资源
    out.release()
    key_listener.stop()
    mouse_listener.stop()
    cv2.destroyAllWindows()

    logger.info(f"录像已保存: {output_video}")

def _check_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def _run_capture():
    path = exist_path(CAPTURE_ROOT, date_path())
    logger.info(f"录制路径: {path}")

    logger.info("10秒后开始录制")
    time.sleep(10)

    global _g_key_abort
    _g_key_abort = False
    while not _g_key_abort:
        capture_screen(save_path=path,debug=True)


def _elevate_and_continue():
    """提升权限并继续执行子进程（管理员窗口）"""
    try:
        script_path = sys.argv[0] if hasattr(sys, 'frozen') else __file__
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script_path}" --admin', None, 1
        )
        logger.info("已请求管理员权限，VScode进程阻塞")
        input("按任意键退出")
        # sys.exit(0)  # 终止VS Code调试进程
    except Exception as e:
        logger.error(f"权限提升失败: {e}")
        sys.exit(1)


def run():
    # 区分主进程和子进程
    if "--admin" not in sys.argv:
        if not _check_admin():
            logger.warning("未检测到管理员权限，尝试提升...")
            _elevate_and_continue()
        else:
            logger.info("已在管理员权限下直接运行")
            _run_capture()
    else:
        _run_capture()  # 子进程直接执行业务逻辑
    

if __name__ == "__main__":
    run()
    pass