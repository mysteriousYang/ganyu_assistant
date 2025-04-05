import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import time
import datetime
import json
import ctypes
import msvcrt
import threading
import os
from pynput import keyboard, mouse
from utils.logger import logger

class InputRecorder:
    def __init__(self, output_dir="recorded_actions"):
        """
        初始化输入录制器
        
        参数:
        - output_dir: 保存录制结果的目录
        """
        self.output_dir = output_dir
        self.current_sequence = []
        self.sequence_number = 0
        self.recording = False
        self.start_time = 0
        self.keyboard_listener = None
        self.mouse_listener = None
        self.stop_key_combination = {keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.Key.esc}
        # self.stop_key_combination = {keyboard.Key.alt_l, 0x50} # P: 0x50
        self.current_keys = set()
        
        # 如果输出目录不存在则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def on_press(self, key):
        """
        按键按下事件处理
        
        参数:
        - key: 按下的键
        """
        if self.recording:
            try:
                # 记录普通按键
                key_char = key.char
                self.current_sequence.append({
                    'type': 'keyboard',
                    'action': 'press',
                    'key': key_char,
                    'time': time.time() - self.start_time
                })
            except AttributeError:
                # 记录特殊按键
                self.current_sequence.append({
                    'type': 'keyboard',
                    'action': 'press',
                    'key': str(key),
                    'time': time.time() - self.start_time
                })
            
            # 跟踪当前按下的键
            self.current_keys.add(key)
            
            # 检查是否按下了停止组合键
            if all(k in self.current_keys for k in self.stop_key_combination):
                print("检测到停止组合键。停止录制。")
                self.stop_recording()
                return False
    
    def on_release(self, key):
        """
        按键释放事件处理
        
        参数:
        - key: 释放的键
        """
        if self.recording:
            try:
                # 记录普通按键释放
                key_char = key.char
                self.current_sequence.append({
                    'type': 'keyboard',
                    'action': 'release',
                    'key': key_char,
                    'time': time.time() - self.start_time
                })
            except AttributeError:
                # 记录特殊按键释放
                self.current_sequence.append({
                    'type': 'keyboard',
                    'action': 'release',
                    'key': str(key),
                    'time': time.time() - self.start_time
                })
            
            # 从当前按下的键集合中移除
            self.current_keys.discard(key)
    
    def on_move(self, x, y):
        """
        鼠标移动事件处理
        
        参数:
        - x: 鼠标x坐标
        - y: 鼠标y坐标
        """
        if self.recording:
            self.current_sequence.append({
                'type': 'mouse',
                'action': 'move',
                'x': x,
                'y': y,
                'time': time.time() - self.start_time
            })
    
    def on_click(self, x, y, button, pressed):
        """
        鼠标点击事件处理
        
        参数:
        - x: 鼠标x坐标
        - y: 鼠标y坐标
        - button: 按下的按钮
        - pressed: 是否是按下（False为释放）
        """
        if self.recording:
            self.current_sequence.append({
                'type': 'mouse',
                'action': 'click',
                'x': x,
                'y': y,
                'button': str(button),
                'pressed': pressed,
                'time': time.time() - self.start_time
            })
    
    def on_scroll(self, x, y, dx, dy):
        """
        鼠标滚轮事件处理
        
        参数:
        - x: 鼠标x坐标
        - y: 鼠标y坐标
        - dx: 水平滚动量
        - dy: 垂直滚动量
        """
        if self.recording:
            self.current_sequence.append({
                'type': 'mouse',
                'action': 'scroll',
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy,
                'time': time.time() - self.start_time
            })
    
    def start_recording(self):
        """开始录制键盘和鼠标输入"""
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            self.current_sequence = []
            
            # 启动键盘监听器
            self.keyboard_listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self.keyboard_listener.start()
            
            # 启动鼠标监听器
            self.mouse_listener = mouse.Listener(
                on_move=self.on_move,
                on_click=self.on_click,
                on_scroll=self.on_scroll
            )
            self.mouse_listener.start()
            
            print(f"录制开始（序列 {self.sequence_number}）")
    
    def checkpoint(self):
        """保存当前序列并开始一个新序列"""
        if self.recording and self.current_sequence:
            # 保存当前序列
            self._save_sequence()
            
            # 重置为下一个序列
            self.current_sequence = []
            self.sequence_number += 1
            self.start_time = time.time()
            
            logger.info(f"检测到检查点。开始新序列 {self.sequence_number}")
            return True
        return False
    
    def _save_sequence(self):
        """将当前序列保存到文件"""
        if self.current_sequence:
            filename = os.path.join(self.output_dir, f"sequence_{self.sequence_number}.json")
            with open(filename, 'w') as f:
                json.dump(self.current_sequence, f, indent=2)
            logger.info(f"保存序列 {self.sequence_number}，共 {len(self.current_sequence)} 个动作")
    
    def stop_recording(self):
        """停止录制并保存最后的序列"""
        if self.recording:
            self.recording = False
            
            # 保存最后的序列
            self._save_sequence()
            
            # 停止监听器
            if self.keyboard_listener and self.keyboard_listener.is_alive():
                self.keyboard_listener.stop()
            
            if self.mouse_listener and self.mouse_listener.is_alive():
                self.mouse_listener.stop()
            
            logger.info("录制停止")
    
    def is_recording(self):
        """检查录制是否激活"""
        return self.recording and self.keyboard_listener and self.keyboard_listener.is_alive()


# 示例用法
def _record_input():
    recorder = InputRecorder(f"E:\\User\Pictures\\yolo_pics\\genshin_train\\action_record\\{datetime.datetime.now().strftime('%Y-%m-%d')}")
    
    try:
        recorder.start_recording()
        # logger.info("正在录制输入。按下 Ctrl+Shift+Esc 停止。")
        logger.info("正在录制输入。按下 Alt+P 停止。")
        
        # 游戏检查点模拟器示例
        # 这将被你实际的检查点检测逻辑替换
        # def game_checkpoint_simulator():
        #     # 模拟每10秒出现一次检查点
        #     count = 0
        #     while recorder.is_recording() and count < 3:  # 演示限制为3个检查点
        #         time.sleep(10)  # 等待10秒
        #         recorder.checkpoint()
        #         count += 1
            
        #     # 检查点之后，再等待一段时间然后停止录制
        #     if recorder.is_recording():
        #         time.sleep(5)
        #         recorder.stop_recording()
        from capture_tools.action_playback import GenshinMultiDetector
        detector = GenshinMultiDetector(
        template_dir='.\\asset\\loading_mask\\template',
        reference_dir='.\\asset\\loading_mask\\reference'
        )
        
        # 在单独的线程中运行检查点模拟器
        checkpoint_thread = threading.Thread(target=detector.is_map_loading)
        checkpoint_thread.daemon = True
        checkpoint_thread.start()
        
        # 主线程等待录制完成
        while recorder.is_recording():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        recorder.stop_recording()
        print("录制被用户中断")


def _check_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def _elevate_and_continue():
    """提升权限并继续执行子进程（管理员窗口）"""
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
            _record_input()
    else:
        _record_input()  # 子进程直接执行业务逻辑

if __name__ == "__main__":
    run()
    pass