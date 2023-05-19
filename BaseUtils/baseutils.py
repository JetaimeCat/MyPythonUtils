#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 16:50
# @Name    : baseutils.py
# @Author  : Xia Zhaoxiang
# @Email   : 1206572082@qq.com
import os
import toml
import time
import torch
import shutil
import random
import numpy as np
from typing import Union
import torch.backends.cudnn


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
