# -*- coding:utf-8 -*-
import time
import os
import sys
import logging
import datetime

LOG_FILE = ".\\logs\\" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import DEBUG
from utils.file_utils import exist_path

class Console_Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def Enable_Console_Logger():
    exist_path(".\\logs")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w") as fp:
            pass
    sys.stdout = Console_Logger(LOG_FILE, sys.stdout)
    pass

def timer_logger(func):
    def wrapper(*args, **kwargs):
        _logger = get_stream_logger()
        start = time.perf_counter()
        result = func(*args,**kwargs)
        # print(f"运行时间: {time.clock() - start}s")
        _logger.info(f"运行时间: {time.perf_counter() - start}s")
        return result
    return wrapper

def get_stream_logger():
    # 创建日志处理器
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOG_FILE)

    # 创建日志格式器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 获取日志器并添加处理器
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    if(DEBUG):
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        # logger.debug("现在运行于 DEBUG 模式")
    
    return logger