# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : acoustics.py
# Software : PyCharm
# Time     : 2023/5/19 11:14
# Email    : 1206572082@qq.com

import os
import toml
import time
import torch
import shutil
import random
import logging
import colorlog
import importlib
import numpy as np
from typing import Union
import torch.backends.cudnn
from inspect import currentframe


def setup_seed(seed: Union[int] = 727):
    """
    设置随机种子，保证再次运行会重现上次的结果

    :param seed: 种子数值
    :return: 无输出
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_directory(directory: Union[str, list], is_clear: Union[bool] = False):
    """
    准备所需文件夹，通过传入文件夹路径，进行文件夹准备

    :param directory: 需要创建的文件夹目录或目录列表
    :param is_clear: 是否清空已有目录内容
    :return:
    """
    directories = [directory] if isinstance(directory, str) else directory
    for directory in directories:
        if os.path.exists(directory) and is_clear:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


def load_profile(toml_path: Union[str], needObj: Union[bool] = False):
    """
    加载参数配置文件

    :param toml_path: toml 格式的参数文件
    :param needObj: 是否需要对象类型
    :return: toml 文件内的参数内容
    """
    params = toml.load(toml_path)
    return DictToObj(params) if needObj else params


# 初始化配置（实现类的实例化）
def initialize_config(module_cfg, pass_params=True):
    module = importlib.import_module(module_cfg["module"])
    if pass_params:
        if "args" not in module_cfg.keys():
            return getattr(module, module_cfg["main"])()
        else:
            return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


#####################################################################################

# 用于计算程序的执行时间
class ExecutionTimer:
    def __init__(self):  # 初始化计时器
        self.start = time.time()

    def reset_time(self):  # 重置开始时间
        self.start = time.time()

    def duration(self):  # 获取当前已执行时间（持续时间）
        return int(time.time() - self.start)

    def duration_str(self):  # 获取当前已执行时间字符串（持续时间）
        duration = self.duration()
        return "[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60)

    def printDuration(self):  # 打印执行时间至终端
        duration = self.duration()
        print("[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60))


# 由字典类型转换为对象类型
class DictToObj(object):
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [DictToObj(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, DictToObj(value) if isinstance(value, dict) else value)


# 程序运行使用的日志类型
class Logger:
    def __init__(self, log_name, save_dir="logs", level=logging.DEBUG, console=True, add=False):
        """
        初始化日志类型用于程序运行的结果输出

        :param log_name: 保存的日志文件名称
        :param save_dir: 日志文件的保存根目录
        :param console: 是否需要将日志内容打印至控制台
        :param add: 当前日志是否为添加类型
        """
        prepare_directory(save_dir)
        log_name = log_name if log_name.endswith(".log") else (log_name + ".log")  # 处理日志文件名称
        if not add and os.path.exists(os.path.join(save_dir, log_name)):
            os.remove(os.path.join(save_dir, log_name))

        # 输出格式以及对应类型的输出颜色
        self.formatter = "[%(asctime)s %(levelname)s] %(message)s"
        self.colors = {"INFO": "blue", "DEBUG": "cyan", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"}

        # 有颜色格式
        self.formatter_colored = colorlog.ColoredFormatter(f"%(log_color)s{self.formatter}", log_colors=self.colors)
        # 无颜色格式
        self.formatter_colorless = logging.Formatter(self.formatter)

        # 创建 logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)  # 设置日志等级

        # 创建一个 handler 用于写入日志内容至文件
        file_handler = logging.FileHandler(os.path.join(save_dir, log_name), encoding="utf-8")
        file_handler.setFormatter(self.formatter_colorless)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)

        # 当 console 为真时，创建一个 handler 用于写入日志内容至控制台
        if isinstance(console, bool) and console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(self.formatter_colored)
            self.logger.addHandler(stream_handler)
        self.logger.info(f"< {log_name} >".center(89, "-"))

    def info(self, message, header=None, footer=None):
        last_frame = currentframe().f_back
        filepath = last_frame.f_code.co_filename
        # 获取当前正在执行的函数名称
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        if header is not None:
            self.logger.info(f"< {header} : {filepath}--{function}--{lineno} >".center(89, "-"))
        if message is not None:
            self.logger.info(message)
        if footer is not None:
            self.logger.info(f"< {footer} {function}--{lineno} >".center(89, "-"))

    def debug(self, message, header=None, footer=None):
        last_frame = currentframe().f_back
        filepath = last_frame.f_code.co_filename
        # 获取当前正在执行的函数名
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        if header is not None:
            self.logger.debug(f"< {header} : {filepath}--{function}--{lineno} >".center(88, "-"))
        if message is not None:
            self.logger.debug(message)
        if footer is not None:
            self.logger.debug(f"< {footer} {function}--{lineno} >".center(88, "-"))

    def warning(self, message, header=None, footer=None):
        last_frame = currentframe().f_back
        filepath = last_frame.f_code.co_filename
        # 获取当前正在执行的函数名
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        if header is not None:
            self.logger.warning(f"< {header} : {filepath}--{function}--{lineno} >".center(86, "-"))
        if message is not None:
            self.logger.warning(message)
        if footer is not None:
            self.logger.warning(f"< {footer} {function}--{lineno} >".center(86, "-"))

    def error(self, message, immediately=False):
        last_frame = currentframe().f_back
        filepath = last_frame.f_code.co_filename
        # 获取当前正在执行的函数名
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        self.logger.error(f"< Error: {filepath}--{function}--{lineno} >".center(88, "-"))
        self.logger.error(message)
        if immediately:
            exit(1)

    def critical(self, message):
        last_frame = currentframe().f_back
        filepath = last_frame.f_code.co_filename
        # 获取当前正在执行的函数名
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        self.logger.critical(f"< Critical: {filepath}--{function}--{lineno} >".center(85, "-"))
        self.logger.critical(message)
        self.logger.critical(f"< Critical: {filepath}--{function}--{lineno} >".center(85, "-"))
        exit(1)
