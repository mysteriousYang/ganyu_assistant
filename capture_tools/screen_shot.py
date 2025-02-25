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
import pygetwindow as gw
from ctypes import windll
from PIL import Image, ImageGrab
from pynput import keyboard, mouse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import SCREENSHOT_DIR

def time_file_name(appendix:str = ".png"):
    str_time = str(datetime.datetime.now())

    # 替换非法符号
    str_time = str_time.replace('.','-')
    str_time = str_time.replace(':','-')
    str_time = str_time.replace(' ','-')
    
    return str_time + appendix

def get_dpi_scale():
    sX = win32api.GetSystemMetrics(0)   #获得屏幕分辨率X轴
    sY = win32api.GetSystemMetrics(1)   #获得屏幕分辨率Y轴
    # print(sX)
    
    # """获取当前显示器的 DPI 缩放因子"""
    # hdc = win32gui.GetDC(0)
    # # dpi = win32api.GetDeviceCaps(hdc, 88)  # 88表示获取DPI
    # dpi = windll.gdi32.GetDeviceCaps(hdc, 88)
    # win32gui.ReleaseDC(0, hdc)
    # scale = dpi / 96  # 默认DPI为96
    # return scale

def get_window_border_size(hwnd):
    """获取窗口的边框和标题栏的尺寸"""
    # 获取边框和标题栏的尺寸
    dpi_scale = windll.gdi32.GetDeviceCaps(win32gui.GetDC(hwnd), 88) / 96  # 获取 DPI 缩放
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)

    # 判断窗口的边框和标题栏大小
    border_width = win32api.GetSystemMetrics(32)  # 边框宽度
    titlebar_height = win32api.GetSystemMetrics(4)  # 标题栏高度

    # 返回实际的边框和标题栏的尺寸（考虑 DPI 缩放）
    return int(border_width * dpi_scale), int(titlebar_height * dpi_scale)

def capture_window(hwnd):

    # 获取窗口尺寸
    # 在存在系统缩放的情况下, 要乘缩放系数
    # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    # print(f"{left} {right}")
    # print(f"{bottom} {top}")

    # broad_width, title_height = get_window_border_size(hwnd)
    # print(broad_width, title_height)
    
    # width = int(1.25*(right - left - 2 * broad_width))
    # height = int(1.25*(bottom - top - title_height - 2 * broad_width))
    width = 1366
    height = 768
    # print(width,height)


    # 创建设备上下文
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()

    # # 在有系统缩放的情况下获取窗口尺寸
    # width = windll.gdi32.GetDeviceCaps(hwindc, 118)
    # height = windll.gdi32.GetDeviceCaps(hwindc, 117)
    # print(width,height)
    
    # 创建位图对象
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bitmap)
    
    # 截图到内存设备上下文
    ###最后一个int参数：0-保存整个窗口，1-只保存客户区。如果PrintWindow成功函数返回值为1
    windll.user32.PrintWindow(hwnd, memdc.GetSafeHdc(), 1)

    # 转换为Pillow图像
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    image = Image.frombuffer(
        "RGB",
        (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
        bmpstr,
        "raw",
        "BGRX",
        0,
        1,
    )

    # 清理资源
    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    return image


def get_all_windows():
    '''
    获取当前所有可见窗口
    返回值: { 窗口名称(str) : 句柄(int) }
    '''
    windows = dict()

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):  # 只获取可见窗口
            title = win32gui.GetWindowText(hwnd)
            if title:  # 过滤掉没有标题的窗口
                windows[title] = hwnd

    win32gui.EnumWindows(enum_handler, None)
    return windows


# def get_all_windows():
#     '''
#     获取所有窗口名称
#     返回值: [窗口名称(str)]
#     '''
#     windows = gw.getAllTitles()  # 获取所有窗口标题
#     return [w for w in windows if w.strip()]  # 过滤掉空标题

events = {
    "key_press":[],
    "key_release":[],
    "mouse_move":[],
    "mouse_click":[]
}

key_press_events = []
key_release_events = []
mouse_move_events = []
mouse_click_events = []

# 键盘监听回调函数
def on_key_press(key):
    try:
        key_press_events.append(
            {
                "key":key.char,
                "time":time.time()
            }
        )
    except:
        key_press_events.append(
            {
                "key":str(key),
                "time":time.time()
            }
        )


def on_key_release(key):
    key_release_events.append(
        {
            "key":str(key),
            "time":time.time()
        }
    )

# 鼠标监听回调函数
def on_mouse_move(x, y):
    mouse_move_events.append(
        {
            "x":x,
            "y":y,
            "time":time.time()
        }
    )

def on_mouse_click(x, y, button, pressed):
    mouse_click_events.append(
        {
            "x":x,
            "y":y,
            "button":str(button),
            "pressed":str(pressed),
            "time":time.time()
        }
    )

# events = list()

# # 键盘监听回调函数
# def on_key_press(key):
#     try:
#         events.append(('key_press', key.char, time.time()))
#     except AttributeError:
#         events.append(('key_press', str(key), time.time()))

# def on_key_release(key):
#     events.append(('key_release', str(key), time.time()))

# # 鼠标监听回调函数
# def on_mouse_move(x, y):
#     events.append(('mouse_move', x, y, time.time()))

# def on_mouse_click(x, y, button, pressed):
#     print(type(button),button)
#     print(type(pressed),pressed)
#     events.append(('mouse_click', (x, y, button, pressed), time.time()))


if __name__ == "__main__":
    print(get_dpi_scale())

    time.sleep(2)
    all_windows = get_all_windows()

    if("原神" in all_windows):
        hwnd = all_windows['原神']
        print(f"已找到 原神 窗口, 句柄为 {hwnd}")

    else:
        print("未找到 原神")
        exit(-1)

    capture_counts = 600 # 截图数量
    app_title = "原神"  # 替换为目标窗口标题


    for i in range(capture_counts):
        capture_time = time.time()

        # 启动键盘监听器
        keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click)
        
        keyboard_listener.start()
        mouse_listener.start()

        # 休眠80ms
        time.sleep(0.08)

        screenshot = capture_window(hwnd)
        # screenshot.show()  # 显示截图
        time_stamp = time_file_name("")

        file_name = os.path.join(SCREENSHOT_DIR,f"{time_stamp}.png")
        screenshot = screenshot.resize((screenshot.width // 2, screenshot.height // 2))
        screenshot.save(file_name)  # 保存到文件

        # 停止监听器
        keyboard_listener.stop()
        mouse_listener.stop()

        # 输出事件记录
        # for event in events:
        #     print(event)
        events["key_press"] = key_press_events
        events["key_release"] = key_release_events
        events["mouse_click"] = mouse_click_events
        events["mouse_move"] = mouse_move_events

        file_name = os.path.join(SCREENSHOT_DIR,f"{time_stamp}.json")
        with open(file_name, mode="w", encoding="utf-8") as fp:
            json.dump(events, fp)
        
        key_press_events.clear()
        key_release_events.clear()
        mouse_click_events.clear()
        mouse_move_events.clear()

        print(f"已存储{time_stamp}")