#-*- coding: utf-8 -*-
import random
import time
import sys
import ctypes
import win32api
import win32con
import pyautogui
from pynput import keyboard, mouse
from threading import Thread, Lock
from math import sin, cos, radians
from utils.logger import logger
from capture_tools.video_capture import terminate_capture_check,terminate_capture_press,terminate_capture_release


# _C_g_end_key = 'p' #用于终止生成的按键
# _C_g_end_func_key = 'alt' #用于终止生成的控制按键

# 键盘映射表 (驱动级虚拟键码)
KEY_MAP = {
    'w': 0x57,      's': 0x53,
    'a': 0x41,      'd': 0x44,
    'space':0x20,
}

# 配置参数（可调整）
CONFIG = {
    'key': {
        'interval_range': (0.5, 8.0),  # 按键触发间隔（秒）
        'press_duration': (0.1, 3),   # 按键保持时间范围
        'random_delay': (0.02, 0.15)   # 操作间随机微延迟
    },
    'mouse': {
        'move_interval': (0.3, 5.0),    # 移动间隔
        'distance_range': (-500, 500),    # 移动像素范围
        'duration_range': (0.01, 0.1),   # 移动耗时范围
        'human_smooth': True            # 启用人类轨迹平滑
    }
}

# ========== 键盘控制模块 ==========
def key_simulator(key,capture_listener):
    """驱动级按键模拟线程"""
    vk_code = KEY_MAP[key]
    while capture_listener.is_alive():
        # 随机等待时间
        time.sleep(random.uniform(*CONFIG['key']['interval_range']))
        
        # 生成随机参数
        press_time = random.uniform(*CONFIG['key']['press_duration'])
        delay = random.uniform(*CONFIG['key']['random_delay'])
        
        # 驱动级按键模拟
        win32api.keybd_event(vk_code, 0, 0, 0)  # 按下
        time.sleep(press_time)                  # 保持按下
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放
        
        time.sleep(delay)  # 操作间随机延迟

# ========== 鼠标控制模块 ==========
def human_move(start, end, duration):
    """人类轨迹模拟（贝塞尔曲线优化）"""
    points = []
    steps = int(duration * 100)
    for t in range(steps):
        # 贝塞尔曲线参数
        ratio = t / steps
        x = start[0] + (end[0]-start[0]) * (ratio + 0.1*sin(3*radians(ratio*360)))
        y = start[1] + (end[1]-start[1]) * (ratio + 0.1*cos(3*radians(ratio*360)))
        points.append((int(x), int(y)))
    return points


# ===== 驱动级鼠标移动 =====
def win32_mouse_move(dx, dy, duration=0.5):
    """
    参数说明：
    - dx: 水平位移（像素）
    - dy: 垂直位移（像素）
    - duration: 移动耗时（秒）
    """
    steps = max(5, int(duration * 60))  # 动态计算步数
    step_x = dx / steps
    step_y = dy / steps
    
    for _ in range(steps):
        # 添加微小随机扰动（防检测）
        offset_x = int(step_x + random.uniform(-2, 2))
        offset_y = int(step_y + random.uniform(-2, 2))
        
        win32api.mouse_event(
            win32con.MOUSEEVENTF_MOVE, 
            offset_x, 
            offset_y,
            0, 0
        )
        time.sleep(duration/steps + random.uniform(0.001, 0.03))

# ===== 随机化鼠标控制线程 =====
def game_mouse_controller(capture_listener):
    while capture_listener.is_alive():
        # activate_game_window()  # 确保窗口激活
        
        # 生成随机移动参数
        # dx = random.randint(-150, 150)
        # dy = random.randint(-100, 100)
        dx = random.randint(CONFIG['mouse']['distance_range'][0],CONFIG['mouse']['distance_range'][1])
        dy = random.randint(0.6*CONFIG['mouse']['distance_range'][0], 0.6*CONFIG['mouse']['distance_range'][1])
        # duration = random.uniform(0.3, 1.2)
        duration = random.uniform(CONFIG['mouse']['duration_range'][0], CONFIG['mouse']['duration_range'][1])
        
        # 执行带扰动的移动
        win32_mouse_move(dx, dy, duration)
        
        # 随机间隔（0.1-3秒）
        time.sleep(random.uniform(0.1, 3))

'''
def mouse_controller(capture_listener):
    """鼠标移动控制线程"""
    while capture_listener.is_alive():
        time.sleep(random.uniform(*CONFIG['mouse']['move_interval']))
        
        # 生成随机参数
        direction = random.choice(['up', 'down', 'left', 'right'])
        distance = random.randint(*CONFIG['mouse']['distance_range'])
        duration = random.uniform(*CONFIG['mouse']['duration_range'])
        
        # 计算移动坐标
        current_x, current_y = pyautogui.position()
        if direction == 'up':
            dx, dy = 0, -distance
        elif direction == 'down':
            dx, dy = 0, distance
        elif direction == 'left':
            dx, dy = -distance, 0
        else:
            dx, dy = distance, 0
        
        # 人类轨迹模式
        if CONFIG['mouse']['human_smooth']:
            start_pos = (current_x, current_y)
            end_pos = (current_x + dx, current_y + dy)
            trajectory = human_move(start_pos, end_pos, duration)
            for x, y in trajectory:
                pyautogui.moveTo(x, y, duration=0.001)
                # send_mouse_move(dx,dy)
                time.sleep(duration/len(trajectory))
        else:
            # send_mouse_move(dx,dy)
            pyautogui.moveRel(dx, dy, duration=duration)
'''

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


def _check_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def _generate_seq():
    logger.info("3秒后开始生成按键控制序列")
    time.sleep(3)

    capture_listener = keyboard.Listener(
        on_press=terminate_capture_press,
        on_release=terminate_capture_release,
        # suppress=True,
    )
    capture_listener.start()

    logger.info("现在开始生成")
    # run_key_controller(capture_listener)
    # 启动按键线程
    for key in ['w', 's', 'a', 'd', 'space']:
        Thread(target=key_simulator, args=(key, capture_listener), daemon=True).start()
    
    # 启动鼠标线程
    Thread(target=game_mouse_controller, args=(capture_listener,), daemon=True).start()

    # 保持主线程运行
    while capture_listener.is_alive():
        time.sleep(1)
    logger.info("\n程序已安全退出")

def run():
    # 区分主进程和子进程
    if "--admin" not in sys.argv:
        if not _check_admin():
            logger.warning("未检测到管理员权限，尝试提升...")
            _elevate_and_continue()
        else:
            logger.info("已在管理员权限下直接运行")
            _generate_seq()
    else:
        _generate_seq()  # 子进程直接执行业务逻辑

# ========== 主程序 ==========
if __name__ == "__main__":
    run()
    pass