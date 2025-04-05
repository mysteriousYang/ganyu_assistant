import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
import cv2
import time
import threading
import mss
import mss.tools

from glob import glob
from utils.logger import logger

class ScreenCheckpointDetector:
    def __init__(self, 
                 monitor_number=1, 
                 sample_interval=0.5,
                 threshold=0.85,
                 template_dir='templates', 
                 reference_dir='references'
        ):
        """
        初始化屏幕检查点检测器
        
        参数:
        - monitor_number: 要捕获的显示器（1是主显示器）
        - sample_interval: 捕获和分析屏幕的频率（秒）
        - threshold: 检测的相似度阈值（0.0-1.0）
        - loading_reference_image: 加载屏幕的参考图像路径（可选）
        """
        self.monitor_number = monitor_number
        self.sample_interval = sample_interval
        self.threshold = threshold
        # self.loading_reference_image = loading_reference_image
        self.reference_img = None
        
        self.running = False
        self.detector_thread = None
        self.last_state = False  # 上一帧是否为加载屏幕
        self.on_checkpoint_callbacks = []
        
        # 如果提供了参考图像则加载
        self._load_image(template_dir,reference_dir)

        # FLANN匹配器参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    
    def _load_image(self,template_dir,reference_dir):
        # 加载所有模板图像（带Alpha通道）
        self.templates = []
        for path in glob(os.path.join(template_dir, '*.png')):
            template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if template is not None:
                # 分离Alpha通道作为mask
                if template.shape[2] == 4:
                    self.templates.append({
                        'image': template[:,:,:3],
                        'mask': template[:,:,3]
                    })
                else:
                    self.templates.append({
                        'image': template,
                        'mask': None
                    })
        
        # 加载所有参考图像（用于特征匹配）
        self.references = []
        self.sift = cv2.SIFT_create()
        for path in glob(os.path.join(reference_dir, '*.png')):
            ref = cv2.imread(path)
            if ref is not None:
                kp, des = self.sift.detectAndCompute(ref, None)
                self.references.append({
                    'image': ref,
                    'kp': kp,
                    'des': des
                })

    def add_checkpoint_callback(self, callback):
        """添加检查点检测到时调用的函数"""
        self.on_checkpoint_callbacks.append(callback)
    
    def _notify_checkpoint(self):
        """调用所有注册的检查点回调"""
        for callback in self.on_checkpoint_callbacks:
            try:
                callback()
            except Exception as e:
                logger.info(f"检查点回调中出错: {e}")
    
    def multi_template_match(self, img, threshold=0.8):
        """ 多模板匹配（任一模板匹配成功即返回True） """
        for template in self.templates:
            result = cv2.matchTemplate(
                img, 
                template['image'],
                cv2.TM_CCOEFF_NORMED,
                mask=template['mask']
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > threshold:
                return True
        return False

    def multi_feature_match(self, img, min_matches=10):
        """ 多参考图特征匹配 """
        kp_img, des_img = self.sift.detectAndCompute(img, None)
        if des_img is None:
            return False
            
        for ref in self.references:
            if ref['des'] is not None:
                matches = self.flann.knnMatch(ref['des'], des_img, k=2)
                # Lowe's比率测试
                good = [m for m,n in matches if m.distance < 0.7*n.distance]
                if len(good) > min_matches:
                    return True
        return False

    def is_map_loading(self):
        """ 综合判断（模板匹配 OR 特征匹配） """
        img = self.capture_screen()
        return self.multi_template_match(img) or self.multi_feature_match(img)

    def capture_screen(self):
        """捕获当前屏幕"""
        with mss.mss() as sct:
            # 获取显示器信息
            monitor = sct.monitors[self.monitor_number]
            
            # 捕获显示器
            sct_img = sct.grab(monitor)
            
            # 转换为numpy数组供OpenCV使用
            # img = np.array(sct_img)
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2BGR)
    
    # def is_loading_screen(self, img):
    #     """
    #     确定当前屏幕是否为加载屏幕
        
    #     此函数应根据你的特定游戏进行自定义。
    #     以下是几种检测方法：
    #     """
    #     # 方法1：使用参考图像进行模板匹配
    #     if self.reference_img is not None:
    #         # 如果需要，调整大小以匹配
    #         if img.shape[:2] != self.reference_img.shape[:2]:
    #             img_resized = cv2.resize(img, (self.reference_img.shape[1], self.reference_img.shape[0]))
    #         else:
    #             img_resized = img
                
    #         # 将两张图像转换为灰度以进行比较
    #         gray_screen = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    #         gray_ref = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
            
    #         # 使用模板匹配计算相似度
    #         result = cv2.matchTemplate(gray_screen, gray_ref, cv2.TM_CCOEFF_NORMED)
    #         _, max_val, _, _ = cv2.minMaxLoc(result)
            
    #         if max_val > self.threshold:
    #             return True
        
    #     # 方法2：检查加载指示器（例如，旋转圆圈）
    #     # 这将是游戏特定的，但这里有一个简单的例子，
    #     # 在屏幕的特定区域寻找特定的颜色模式
        
    #     # 示例：检查屏幕底部的加载条
    #     height, width = img.shape[:2]
    #     bottom_region = img[int(height*0.9):height, int(width*0.25):int(width*0.75)]
        
    #     # 转换为HSV以获得更好的颜色检测
    #     hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
        
    #     # 定义典型加载条颜色范围（例如，白色或蓝色）
    #     # 这需要根据你的特定游戏进行调整
    #     lower_color = np.array([100, 100, 200])  # HSV中的浅蓝色
    #     upper_color = np.array([140, 255, 255])
        
    #     # 为指定的颜色范围创建掩码
    #     mask = cv2.inRange(hsv, lower_color, upper_color)
        
    #     # 如果有足够的像素匹配我们的颜色范围，可能是加载条
    #     loading_bar_pixels = cv2.countNonZero(mask)
    #     if loading_bar_pixels > (bottom_region.shape[0] * bottom_region.shape[1] * 0.1):
    #         return True
        
    #     # 方法3：使用OCR检测如"Loading..."等文本
    #     # 这需要安装OCR库如pytesseract
    #     # 为简单起见，省略示例实现
        
    #     # 方法4：动作检测 - 加载屏幕通常移动较少
    #     # 为此，我们需要比较连续的帧
    #     # 为简单起见，省略示例实现
        
    #     # 未检测到加载屏幕
    #     return False
    
    def _detector_worker(self):
        """检测工作线程函数"""
        prev_frame_time = time.time()
        consecutive_loading_frames = 0
        consecutive_non_loading_frames = 0
        
        while self.running:
            current_time = time.time()
            
            # 按照采样间隔捕获和处理屏幕
            if current_time - prev_frame_time >= self.sample_interval:
                prev_frame_time = current_time
                
                # 捕获屏幕
                # img = self.capture_screen()
                
                # 检查是否是加载屏幕
                # is_loading = self.is_loading_screen(img)
                is_loading = self.is_map_loading()
                
                # 状态转换检测
                if is_loading:
                    consecutive_loading_frames += 1
                    consecutive_non_loading_frames = 0
                    
                    # 如果连续几帧都是加载屏幕，确认加载状态
                    if consecutive_loading_frames >= 3 and not self.last_state:
                        logger.info("检测到加载屏幕开始")
                        self.last_state = True
                else:
                    consecutive_non_loading_frames += 1
                    consecutive_loading_frames = 0
                    
                    # 如果连续几帧都不是加载屏幕，且之前是加载状态
                    # 这表示加载已完成 -> 触发检查点
                    if consecutive_non_loading_frames >= 3 and self.last_state:
                        logger.info("加载屏幕结束 -> 触发检查点")
                        self.last_state = False
                        self._notify_checkpoint()
            
            # 短暂休眠以减少CPU使用
            time.sleep(0.01)
    
    def start(self):
        """开始检测进程"""
        if self.detector_thread and self.detector_thread.is_alive():
            logger.info("检测器已经在运行")
            return False
        
        self.running = True
        self.detector_thread = threading.Thread(target=self._detector_worker)
        self.detector_thread.daemon = True
        self.detector_thread.start()
        logger.info("检查点检测器已启动")
        return True
    
    def stop(self):
        """停止检测进程"""
        self.running = False
        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join(timeout=1.0)
        logger.info("检查点检测器已停止")
    
    def is_running(self):
        """检查检测器是否在运行"""
        return self.running and self.detector_thread and self.detector_thread.is_alive()


# 示例用法：将此检测器与之前的GameAutomation类集成
def _run_detect():
    # 创建检测器
    detector = ScreenCheckpointDetector(
        monitor_number=1,
        sample_interval=0.2,
        threshold=0.8,
        # 可以提供游戏加载屏幕的参考图像
        template_dir='.\\asset\\loading_mask\\template',
        reference_dir='.\\asset\\loading_mask\\reference'
    )
    
    # 定义检查点回调，这将连接到游戏自动化系统
    def on_checkpoint_detected():
        logger.info("检查点回调：在这里触发录制器或播放器的检查点")
    
    # 添加回调
    detector.add_checkpoint_callback(on_checkpoint_detected)
    
    try:
        # 启动检测器
        detector.start()
        logger.info("检测器已启动。按Ctrl+C停止。")
        
        # 保持程序运行
        while detector.is_running():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    finally:
        detector.stop()

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
            _run_detect()
    else:
        _run_detect()  # 子进程直接执行业务逻辑 

if __name__ == "__main__":
    _run_detect()
    pass