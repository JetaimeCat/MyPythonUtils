# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : acoustics.py
# Software : PyCharm
# Time     : 2023/5/19 11:14
# Email    : 1206572082@qq.com
import re
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
from inspect import getsourcefile
from tqdm import tqdm


def setup_seed(seed: Union[int] = 727):
    """
    设置随机种子，保证再次运行会重现上次的结果

    :param seed: 种子数值
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_directories(directories, is_clear: bool, use_timestamp: bool = False):
    """
    准备目录，可以选择是否清空目录

    :param directories: 要准备的目录，可以是单个目录路径或目录路径列表
    :param is_clear: 是否清空目录。True 表示清空目录，False 表示不清空目录
    :param use_timestamp: 是否使用时间戳创建文件
    :return: 时间戳
    """
    dir_type = type(directories)
    timestamp = time.strftime("%Y.%m.%d %H-%M-%S", time.localtime()) if use_timestamp else ""
    assert dir_type in [str, list], f"The folder type is {dir_type}, please select [str, list]."
    if dir_type is str:
        directories = [directories]
    for directory in directories:
        directory = os.path.join(directory, timestamp)
        if os.path.exists(directory) and is_clear:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    return timestamp


def generate_specify_files(root_dir: str, save_dir: str = ".", save_file_name: str = "specify_files.scp",
                           prefix: str = None, suffix: str = None, keywords: list = None,
                           limit: int = -1, is_clear=False, is_show=False, description="Saving after generation"):
    """
    生成在指定根目录及其子目录中找到的 .wav 文件列表，并将该列表保存到一个文本文件中

    :param root_dir: 要搜索 .wav 文件的根目录
    :param save_dir: 将包含 .wav 文件列表的文本文件保存到的目录。默认为当前目录
    :param save_file_name: 要保存 .wav 文件列表的文本文件的名称。默认为 "specify_files.txt"
    :param prefix: 待搜索的文件前缀
    :param suffix: 待搜索的文件后缀
    :param keywords: 待搜索的文件包含关键词
    :param limit: 文件数量限制
    :param is_clear: 是否清空保存 .wav 文件列表的文本文件。默认为 False
    :param is_show: 是否显示进度
    :param description: 进度描述
    :return: 返回指定文件内容
    """
    specify_files = list()  # 初始化一个空列表来存储 .wav 文件路径

    for root in os.walk(root_dir):
        for file in root[-1]:
            file_path = os.path.join(root[0], file)
            if (prefix is not None) and (not file.startswith(prefix)):
                continue  # 如果文件不以指定前缀开头，则跳过
            if (suffix is not None) and (not file.endswith(suffix)):
                continue  # 如果文件不以指定后缀结尾，则跳过
            if (keywords is not None) and (not any((keyword in file_path) for keyword in keywords)):
                continue  # 如果文件名不包含任何关键词，则跳过
            specify_files.append(file_path)
    specify_files = sorted(specify_files[:limit] if limit > 0 else specify_files)
    # 以写模式打开文件，以保存 .wav 文件列表
    prepare_directories(save_dir, is_clear=is_clear)
    with open(os.path.join(save_dir, save_file_name), "w") as file:
        iterator = tqdm(specify_files, desc=description) if is_show else specify_files
        for specify_file in iterator:
            file.write(os.path.abspath(specify_file) + "\n")
    return np.array(specify_files)


def load_profile(toml_path: Union[str], needObj: Union[bool] = False):
    """
    加载参数配置文件

    :param toml_path: toml 格式的参数文件
    :param needObj: 是否需要对象类型
    :return: toml 文件内的参数内容
    """
    params = toml.load(toml_path)
    return DictToObj(params) if needObj else params


def initialize_config(module_cfg, pass_params=True):
    """

    :param module_cfg:
    :param pass_params:
    :return:
    """
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

    # 输出树形文件夹的标题
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
# 计算程序执行时间
class ExecutionTimer:
    def __init__(self):  # 初始化执行时间计时器
        self.start_time = time.time()

    def reset_time(self):  # 重置开始时间
        self.start_time = time.time()

    def duration(self):  # 获取当前已执行时间（s）
        return time.time() - self.start_time

    def duration_ms(self):  # 获取当前已执行时间（ms）
        return int((time.time() - self.start_time) * 1000)

    def duration_str(self):  # 获取当前已执行时间字符串
        duration = int(self.duration())
        return "[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60)

    def printDuration(self):  # 输出执行时间
        duration = int(self.duration())
        print("[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60))


# 将字典转换为对象的工具类
class DictToObj(object):
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [DictToObj(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, DictToObj(value) if isinstance(value, dict) else value)

# if __name__ == '__main__':
#     base_dir = r"ST-CMDS"
#     root_dir = fr"F:\AudioData\{base_dir}"
#     specify_save_dir = fr"H:\AudioDataset\{base_dir}"
#     prepare_directories(specify_save_dir, is_clear=True)
#     # files_list = generate_specify_files(root_dir, specify_save_dir, f"{base_dir}.scp", keywords=[".wav"], is_show=True)
