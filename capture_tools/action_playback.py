import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import cv2
import numpy as np
import pyautogui
import ctypes
import msvcrt

from utils.logger import logger
from glob import glob

class GenshinMultiDetector:
    def __init__(self, template_dir='templates', reference_dir='references'):
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
        
        # FLANN匹配器参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def get_screenshot(self):
        # 使用pyautogui或其他截图库获取屏幕图像
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

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
        img = self.get_screenshot()
        return self.multi_template_match(img) or self.multi_feature_match(img)

# 使用示例
def _detect():
    # 确保有以下目录结构：
    # ./templates/ 存放多个模板图片（如icon1.png, progress_bar.png等）
    # ./references/ 存放多个完整参考图（如loading_style1.png, style2.png等）
    
    detector = GenshinMultiDetector(
        template_dir='.\\asset\\loading_mask\\template',
        reference_dir='.\\asset\\loading_mask\\reference'
    )
    
    while True:
        if detector.is_map_loading():
            logger.info("检测到地图加载界面！")
        else:
            logger.info("当前不是加载界面")
        cv2.waitKey(1000)  # 每秒检测一次

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
            _detect()
    else:
        _detect()  # 子进程直接执行业务逻辑

if __name__ == "__main__":
    run()
    pass