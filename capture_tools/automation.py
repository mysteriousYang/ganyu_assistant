import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import time
import threading
import argparse
from input_recorder import InputRecorder
from input_player import InputPlayer
from screen_checkpoint_detector import ScreenCheckpointDetector
from utils.logger import logger

class CompleteGameAutomation:
    """
    完整的游戏自动化系统，集成了输入录制、回放和基于屏幕的检查点检测
    """
    
    def __init__(self, 
                 output_dir="recorded_actions",
                #  reference_image=None,
                 monitor_number=1,
                 template_dir='templates', 
                 reference_dir='references',
        ):
        """
        初始化自动化系统
        
        参数:
        - output_dir: 录制文件的保存目录
        - reference_image: 加载屏幕参考图像路径
        - monitor_number: 要监控的显示器号码
        """
        self.output_dir = output_dir
        # self.reference_image = reference_image
        self.monitor_number = monitor_number
        self.template_dir = template_dir
        self.reference_dir = reference_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化各组件
        self.recorder = InputRecorder(output_dir)
        self.player = InputPlayer(output_dir)
        self.detector = ScreenCheckpointDetector(
                            monitor_number=1,
                            sample_interval=0.2,
                            threshold=0.8,
                            # 可以提供游戏加载屏幕的参考图像
                            template_dir=template_dir,
                            reference_dir=reference_dir
                        )
        
        # 状态标志
        self.mode = None  # 'record' 或 'playback'
        self.running = False
        self.first_run = True # 用于等待第一次传送
        
        # 配置检测器回调
        self.detector.add_checkpoint_callback(self._on_checkpoint_detected)
    
    def _on_checkpoint_detected(self):
        """
        检查点被检测到时的回调处理
        """
        if not self.running:
            if self.mode == "record" and self.first_run:
                logger.info("检测到检查点 - 正在开始录制")
                self.first_run = False
                self.start_recording()

            elif self.mode == "playback" and self.first_run:
                logger.info("检测到检查点 - 正在开始回放")
                self.first_run = False
                self.start_playback()
            
        if self.mode == "record" and self.recorder.is_recording():
            logger.info("检测到检查点 - 为录制标记检查点")
            self.recorder.checkpoint()
  
        elif self.mode == "playback" and self.player.is_playing():
            logger.info("检测到检查点 - 继续播放下一序列")
            self.player.checkpoint()
        
        
    
    def start_recording(self):
        """
        开始录制模式
        """
        if self.running:
            logger.info("系统已经在运行中")
            return False
        
        self.mode = "record"

        if(self.first_run):
            self.detector.start()
            logger.info("录制将在下一次传送结束开始")
            return True


        # 启动录制器
        self.recorder.start_recording()
        self.running = True
        
        logger.info("=== 录制模式已启动 ===")
        logger.info("系统将自动在加载屏幕结束时标记检查点")
        logger.info("按 Ctrl+Shift+Esc 停止录制")
        
        return True
    
    def start_playback(self):
        """
        开始回放模式
        """
        if self.running:
            logger.info("系统已经在运行中")
            return False
        
        self.mode = "playback"
        
        if(self.first_run):
            self.detector.start()
            logger.info("播放将在下一次传送结束开始")
            return True
        
        # 启动播放器
        self.player.start_playback()
        self.running = True
        
        logger.info("=== 回放模式已启动 ===")
        logger.info("系统将自动在加载屏幕结束时继续播放下一序列")
        logger.info("按 Ctrl+Shift+Esc 停止回放")
        
        return True
    
    def stop(self):
        """
        停止当前操作
        """
        self.running = False
        
        # 停止录制器（如果在录制）
        if self.recorder.is_recording():
            self.recorder.stop_recording()
        
        # 停止播放器（如果在播放）
        if self.player.is_playing():
            self.player.stop_playback()
        
        # 停止检测器
        if self.detector.is_running():
            self.detector.stop()
        
        logger.info("=== 系统已停止 ===")
    
    def is_running(self):
        """
        检查系统是否在运行
        """
        if self.mode == "record":
            return self.recorder.is_recording() and self.detector.is_running()
        elif self.mode == "playback":
            return self.player.is_playing() and self.detector.is_running()
        return False
    
    '''
    def create_reference_image(self):
        """
        创建加载屏幕的参考图像
        在没有参考图像时使用
        """
        logger.info("=== 参考图像捕获模式 ===")
        logger.info("请将游戏切换到加载屏幕，然后按下空格键捕获参考图像")
        
        import keyboard  # 仅在此函数中导入，因为它比pynput更容易用于简单的按键等待
        
        # 捕获单个屏幕
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_number]
            
            while True:
                if keyboard.is_pressed('space'):
                    logger.info("捕获参考图像...")
                    sct_img = sct.grab(monitor)
                    img = np.array(sct_img)
                    
                    # 保存参考图像
                    ref_image_path = os.path.join(self.output_dir, "loading_reference.png")
                    cv2.imwrite(ref_image_path, img)
                    
                    logger.info(f"参考图像已保存到: {ref_image_path}")
                    self.reference_image = ref_image_path
                    return ref_image_path
                
                elif keyboard.is_pressed('esc'):
                    logger.info("取消捕获")
                    return None
                
                time.sleep(0.1)
    '''


# 主程序
# def main():
#     # 解析命令行参数
#     parser = argparse.ArgumentParser(description='游戏自动化系统')
#     parser.add_argument('mode', choices=['record', 'playback', 'capture'], 
#                       help='运行模式：录制(record)，回放(playback)或捕获参考图像(capture)')
#     parser.add_argument('--dir', '-d', default='recorded_actions',
#                       help='录制文件的目录 (默认: recorded_actions)')
#     parser.add_argument('--monitor', '-m', type=int, default=1,
#                       help='要使用的显示器编号 (默认: 1，即主显示器)')
#     parser.add_argument('--reference', '-r', 
#                       help='加载屏幕参考图像的路径 (默认: None)')
    
#     args = parser.parse_args()
    
#     # 创建自动化系统
#     automation = CompleteGameAutomation(
#         output_dir=args.dir,
#         reference_image=args.reference,
#         monitor_number=args.monitor
#     )
    
#     try:
#         if args.mode == 'capture':
#             # 捕获参考图像模式
#             ref_image = automation.create_reference_image()
#             if ref_image:
#                 logger.info(f"现在可以使用以下命令启动系统:")
#                 logger.info(f"python game_automation.py record --reference {ref_image}")
#                 logger.info(f"或")
#                 logger.info(f"python game_automation.py playback --reference {ref_image}")
        
#         elif args.mode == 'record':
#             # 录制模式
#             automation.start_recording()
            
#             # 等待录制完成
#             while automation.is_running():
#                 time.sleep(0.1)
        
#         elif args.mode == 'playback':
#             # 回放模式
#             automation.start_playback()
            
#             # 等待回放完成
#             while automation.is_running():
#                 time.sleep(0.1)
    
#     except KeyboardInterrupt:
#         logger.info("\n用户中断")
#     finally:
#         automation.stop()

def _make_output_dir(base_path:str):
    import datetime
    base_path = os.path.join(base_path,datetime.datetime.now().strftime('%Y-%m-%d'))

    # 确保基础目录存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)  # 

    # 获取所有以'record_'开头的文件夹
    existing_folders = [f for f in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, f)) and f.startswith("record_")]

    # 提取现有最大编号
    max_num = -1
    for folder in existing_folders:
        try:
            num = int(folder.split("_")[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            continue  # 跳过名称不规范的文件夹

    # 生成新编号
    new_num = max_num + 1
    new_folder = f"record_{new_num}"
    new_path = os.path.join(base_path, new_folder)

    # 创建新目录
    os.makedirs(new_path, exist_ok=True)  # 
    logger.info(f"创建文件夹: {new_path}")
    return new_path

def _main(mode="record"):
    output_dir_root = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\action_record\\"

    # out_dir = _make_output_dir(output_dir_root)
    out_dir = "E:\\User\\Pictures\\yolo_pics\\genshin_train\\action_record\\2025-04-05\\record_12"

    automation = CompleteGameAutomation(
        output_dir=out_dir,
        monitor_number=1,
        # 可以提供游戏加载屏幕的参考图像
        template_dir='.\\asset\\loading_mask\\template',
        reference_dir='.\\asset\\loading_mask\\reference'
    )

    # logger.info("5秒后开始录制")
    # time.sleep(5)

    try:      
        if mode == 'record':
            # 录制模式
            automation.start_recording()

            while automation.first_run:
                time.sleep(0.1)
            
            # 等待录制完成
            while automation.is_running():
                time.sleep(0.1)
        
        elif mode == 'playback':
            # 回放模式
            automation.start_playback()

            while automation.first_run:
                time.sleep(0.1)
            
            # 等待回放完成
            while automation.is_running():
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    finally:
        automation.stop()

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


def run(mode="record"):
    # 区分主进程和子进程
    if "--admin" not in sys.argv:
        if not _check_admin():
            logger.warning("未检测到管理员权限，尝试提升...")
            _elevate_and_continue()
        else:
            logger.info("已在管理员权限下直接运行")
            _main(mode)
    else:
        _main(mode)  # 子进程直接执行业务逻辑   

if __name__ == "__main__":
    # run("record")
    run("playback")
    # _main("record")

    pass