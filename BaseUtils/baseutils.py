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
import torch.backends.cudnn
from typing import Union, List
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


def generate_output_tree(output_list: List[Union[list, str]], depth: Union[int] = 0, site=None, title=None):
    """
    递归打印文件目录树状图(使用局部变量)。

    :param output_list: 根目录路径。
    :param depth: 根目录、文件所在的层级号。
    :param site: 存储出现转折的层级号。
    :param title: 树状标题。
    :return: None
    """
    site = list() if site is None else site
    void_num = 0  # 记录当前层级已经出现转折的次数

    if title is not None and isinstance(title, str):
        print(title)
        generate_output_tree(output_list, depth + 1, site, title=True)
        return

    for idx, item in enumerate(output_list):
        # 用于存储每个层级的缩进符号
        str_list = ["|\t" for _ in range(depth - void_num - len(site))]
        if title is not None and len(str_list) > 0:
            str_list[0] = "\t"

        for s in site:
            str_list.insert(s, "\t")  # 在对应的层级插入转折符号

        sub_count = sum(1 for item in output_list[idx + 1:] if isinstance(item, str))
        if (idx + 1) != len(output_list) and (sub_count > 0):
            str_list.append("├─")  # 不是本级目录最后一个文件，添加普通转折符号
        elif (idx + 1) == len(output_list):
            str_list.append("└─")  # 本级目录最后一个文件：转折处
            void_num += 1
            site.append(depth)  # 添加当前已经出现转折的层级数
        else:
            str_list.append("└─")  # 本级目录最后一个文件：非转折处
        if isinstance(item, str):
            print("".join(str_list) + item)  # 打印文件名
        else:
            # 递归调用，处理子目录或文件
            generate_output_tree(item, depth + 1, site, title=None if title is None else True)
        if (idx + 1) == len(output_list):
            void_num -= 1
            site.pop()  # 移除当前已出现转折的层级数


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
        return "[ Current execution time: %02dh:%02dm:%02ds ]" % (
            duration // 60 // 60, duration // 60, duration % 60)

    def printDuration(self):  # 打印执行时间至终端
        duration = self.duration()
        print(
            "[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60))


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
        self.colors = {"INFO": "blue", "DEBUG": "cyan", "WARNING": "yellow", "ERROR": "red",
                       "CRITICAL": "red,bg_white"}

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


if __name__ == '__main__':
    output = ['A', 'B', ['C', 'D', 'E', ['F', 'G', 'H']], 'I', 'J', 'K', ['L', 'M', 'N']]
    generate_output_tree(output, title="Title")
