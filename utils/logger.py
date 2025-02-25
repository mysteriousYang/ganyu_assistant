# -*- coding:utf-8 -*-
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import LOG_FILE
from utils.file_utils import exist_path

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def Enable_Logger():
    exist_path(".\\logs")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w") as fp:
            pass
    sys.stdout = Logger(LOG_FILE, sys.stdout)
    pass

def timer_logger(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        result = func(*args,**kwargs)
        print(f"运行时间: {time.clock() - start}s")
        return result
    return wrapper
