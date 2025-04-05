import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import time
import json
import os
import threading
import win32api
import win32con
import ctypes
from pynput import keyboard
from utils.logger import logger

# 鼠标和键盘事件模拟常量
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

# 将pynput键名映射到虚拟键代码
KEY_MAP = {
    # 特殊键
    "Key.enter": win32con.VK_RETURN,
    "Key.esc": win32con.VK_ESCAPE,
    "Key.shift": win32con.VK_SHIFT,
    "Key.shift_l": win32con.VK_LSHIFT,
    "Key.shift_r": win32con.VK_RSHIFT,
    "Key.ctrl": win32con.VK_CONTROL,
    "Key.ctrl_l": win32con.VK_LCONTROL,
    "Key.ctrl_r": win32con.VK_RCONTROL,
    "Key.alt": win32con.VK_MENU,
    "Key.alt_l": win32con.VK_LMENU,
    "Key.alt_r": win32con.VK_RMENU,
    "Key.tab": win32con.VK_TAB,
    "Key.space": win32con.VK_SPACE,
    "Key.backspace": win32con.VK_BACK,
    "Key.caps_lock": win32con.VK_CAPITAL,
    "Key.page_up": win32con.VK_PRIOR,
    "Key.page_down": win32con.VK_NEXT,
    "Key.end": win32con.VK_END,
    "Key.home": win32con.VK_HOME,
    "Key.left": win32con.VK_LEFT,
    "Key.up": win32con.VK_UP,
    "Key.right": win32con.VK_RIGHT,
    "Key.down": win32con.VK_DOWN,
    "Key.insert": win32con.VK_INSERT,
    "Key.delete": win32con.VK_DELETE,
    # 功能键
    "Key.f1": win32con.VK_F1,
    "Key.f2": win32con.VK_F2,
    "Key.f3": win32con.VK_F3,
    "Key.f4": win32con.VK_F4,
    "Key.f5": win32con.VK_F5,
    "Key.f6": win32con.VK_F6,
    "Key.f7": win32con.VK_F7,
    "Key.f8": win32con.VK_F8,
    "Key.f9": win32con.VK_F9,
    "Key.f10": win32con.VK_F10,
    "Key.f11": win32con.VK_F11,
    "Key.f12": win32con.VK_F12,
    # 普通操作按键
    # 大写字母
    # 'A': 0x41, 'B': 0x42, 'C': 0x43, 'D': 0x44, 'E': 0x45,
    # 'F': 0x46, 'G': 0x47, 'H': 0x48, 'I': 0x49, 'J': 0x4A,
    # 'K': 0x4B, 'L': 0x4C, 'M': 0x4D, 'N': 0x4E, 'O': 0x4F,
    # 'P': 0x50, 'Q': 0x51, 'R': 0x52, 'S': 0x53, 'T': 0x54,
    # 'U': 0x55, 'V': 0x56, 'W': 0x57, 'X': 0x58, 'Y': 0x59,
    # 'Z': 0x5A,
    # 小写字母（值与大写相同）
    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
    'z': 0x5A
}

class InputPlayer:
    def __init__(self, input_dir="recorded_actions"):
        """
        初始化输入播放器
        
        参数:
        - input_dir: 保存录制结果的目录
        """
        self.input_dir = input_dir
        self.playing = False
        self.stop_event = threading.Event()
        self.checkpoint_event = threading.Event()
        
        # 用于通过键盘停止播放
        self.stop_key_combination = {keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.Key.esc}
        self.current_keys = set()
        self.keyboard_listener = None
    
    def _get_key_code(self, key_str):
        """
        将键字符串转换为虚拟键代码
        
        参数:
        - key_str: 键的字符串表示
        """
        if key_str in KEY_MAP:
            return KEY_MAP[key_str]
        elif len(key_str) == 3 and key_str.startswith("'") and key_str.endswith("'"):
            # 处理像 'a', 'b' 等字符键
            char = key_str[1]
            # 转换为大写以匹配VK代码
            return ord(char.upper())
        else:
            logger.info(f"未知键: {key_str}")
            return None
    
    def on_press(self, key):
        """
        处理按键按下以停止播放
        
        参数:
        - key: 按下的键
        """
        self.current_keys.add(key)
        
        # 检查是否按下了停止组合键
        if all(k in self.current_keys for k in self.stop_key_combination):
            logger.info("检测到停止组合键。停止播放。")
            self.stop_playback()
            return False
    
    def on_release(self, key):
        """
        处理按键释放以停止播放
        
        参数:
        - key: 释放的键
        """
        self.current_keys.discard(key)
    
    def _load_sequence(self, sequence_number):
        """
        从文件加载序列
        
        参数:
        - sequence_number: 序列编号
        """
        filename = os.path.join(self.input_dir, f"sequence_{sequence_number}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    
    def _simulate_keyboard_event(self, action):
        """
        使用win32api模拟键盘事件
        
        参数:
        - action: 键盘动作数据
        """
        key_str = action['key']
        key_code = self._get_key_code(key_str)
        
        if key_code is not None:
            if action['action'] == 'press':
                win32api.keybd_event(key_code, 0, 0, 0)
            elif action['action'] == 'release':
                win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    
    def _simulate_mouse_event(self, action):
        """
        使用win32api模拟鼠标事件
        
        参数:
        - action: 鼠标动作数据
        """
        if action['action'] == 'move':
            win32api.SetCursorPos((int(action['x']), int(action['y'])))
        
        elif action['action'] == 'click':
            # 首先定位光标
            win32api.SetCursorPos((int(action['x']), int(action['y'])))
            
            # 确定使用了哪个按钮
            button = action['button']
            pressed = action['pressed']
            
            if 'left' in button:
                if pressed:
                    win32api.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                else:
                    win32api.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif 'right' in button:
                if pressed:
                    win32api.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                else:
                    win32api.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            elif 'middle' in button:
                if pressed:
                    win32api.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                else:
                    win32api.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
        
        elif action['action'] == 'scroll':
            # 鼠标滚轮事件
            wheel_amount = int(action['dy'] * 120)  # Windows使用每刻度120
            win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)
    
    def play_sequence(self, sequence_number):
        """
        播放特定序列
        
        参数:
        - sequence_number: 要播放的序列编号
        """
        sequence = self._load_sequence(sequence_number)
        if not sequence:
            logger.info(f"未找到序列 {sequence_number}")
            return False
        
        logger.info(f"播放序列 {sequence_number}，共 {len(sequence)} 个动作")
        
        # 重置播放计时器
        start_time = time.time()
        last_action_time = 0
        
        # 按照正确的时间播放每个动作
        for action in sequence:
            # 检查是否应该停止播放
            if self.stop_event.is_set():
                return False
            
            # 等待直到正确的时间执行此动作
            action_time = action['time']
            time_to_wait = action_time - last_action_time
            
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            
            # 执行动作
            if action['type'] == 'keyboard':
                self._simulate_keyboard_event(action)
            elif action['type'] == 'mouse':
                self._simulate_mouse_event(action)
            
            last_action_time = action_time
        
        return True
    
    def start_playback(self):
        """开始播放所有序列，在序列之间等待检查点"""
        if self.playing:
            logger.info("播放已经激活")
            return
        
        self.playing = True
        self.stop_event.clear()
        
        # 启动键盘监听器用于停止组合键
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()
        
        # 创建并启动播放线程
        playback_thread = threading.Thread(target=self._playback_worker)
        playback_thread.daemon = True
        playback_thread.start()
    
    def _playback_worker(self):
        """播放工作线程"""
        sequence_number = 0
        
        while not self.stop_event.is_set():
            # 播放当前序列
            success = self.play_sequence(sequence_number)
            
            if not success:
                break
            
            # 序列播放完成，等待检查点继续
            logger.info(f"等待检查点播放下一个序列...")
            sequence_number += 1
            
            # 等待检查点事件或停止事件
            while not self.stop_event.is_set() and not self.checkpoint_event.is_set():
                time.sleep(0.1)
            
            # 如果设置了检查点事件，则清除它
            if self.checkpoint_event.is_set():
                self.checkpoint_event.clear()
        
        self.playing = False
        logger.info("播放完成或已停止")
    
    def checkpoint(self):
        """表示已到达检查点"""
        if self.playing:
            logger.info("检测到检查点，继续播放")
            self.checkpoint_event.set()
            return True
        return False
    
    def stop_playback(self):
        """停止播放"""
        if self.playing:
            logger.info("停止播放")
            self.stop_event.set()
            self.checkpoint_event.set()  # 同时设置检查点以解除阻塞
            
            # 停止键盘监听器
            if self.keyboard_listener and self.keyboard_listener.is_alive():
                self.keyboard_listener.stop()
            
            # 释放所有按键
            for _, vk in KEY_MAP.items():
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

            self.playing = False
    
    def is_playing(self):
        """检查播放是否激活"""
        return self.playing


# 示例用法
def _play_record():
    player = InputPlayer(
        input_dir="E:\\User\\Pictures\\yolo_pics\\genshin_train\\action_record\\2025-04-05\\record_1"
    )
    
    time.sleep(5)
    logger.info("5秒后开始播放")
    try:
        player.start_playback()
        logger.info("播放开始。按下 Ctrl+Shift+Esc 停止。")
        
        # 游戏检查点模拟器示例
        # 这将被你实际的检查点检测逻辑替换
        def game_checkpoint_simulator():
            # 模拟每5秒出现一次检查点
            count = 0
            while player.is_playing() and count < 3:  # 演示限制为3个检查点
                time.sleep(5)  # 等待5秒
                player.checkpoint()
                count += 1
            
            # 检查点之后，再等待一段时间然后停止播放
            if player.is_playing():
                time.sleep(5)
                player.stop_playback()
        
        # 在单独的线程中运行检查点模拟器
        checkpoint_thread = threading.Thread(target=game_checkpoint_simulator)
        checkpoint_thread.daemon = True
        checkpoint_thread.start()
        
        # 主线程等待播放完成
        while player.is_playing():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        player.stop_playback()
        logger.info("播放被用户中断")

def _check_admin():
    import ctypes
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def _elevate_and_continue():
    """提升权限并继续执行子进程（管理员窗口）"""
    import ctypes
    import msvcrt
    try:
        script_path = sys.argv[0] if hasattr(sys, 'frozen') else __file__
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script_path}" --admin', None, 1
        )
        logger.info("已请求管理员权限，VScode进程阻塞")
        # input("按回车键退出")
        logger.info("按任意键退出")
        msvcrt.getch()
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
            _play_record()
    else:
        _play_record()  # 子进程直接执行业务逻辑   

if __name__ == "__main__":
    run()