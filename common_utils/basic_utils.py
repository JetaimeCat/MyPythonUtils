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
import importlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.backends.cudnn
import multiprocessing as mult
from typing import Union, List
from contextlib import contextmanager

# 不同形式的文件层级连接符号
CONNECTORS = {
    "default": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├── ",
        "last": "   └── "
    },
    "round": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├─╴ ",
        "last": "   ╰─╴ "
    },
    "minimal": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├── ",
        "last": "   ╰── "
    },
    "bold": {
        "space": "       ",
        "branch": "   ┃   ",
        "tee": "   ┣━━ ",
        "last": "   ┗━━ "
    },
    "double": {
        "space": "       ",
        "branch": "   ║   ",
        "tee": "   ╠══ ",
        "last": "   ╚══ "
    },
    "dot": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├── ",
        "last": "   •── "
    },
    "arrow": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├─▶ ",
        "last": "   ╰─▶ "
    },
    "curved": {
        "space": "       ",
        "branch": "   │   ",
        "tee": "   ├╌╌ ",
        "last": "   ╰╌╌ "
    }
}


################################################### Basic Functions ####################################################
def setup_seed(seed: Union[int] = 727):
    """
    设置随机种子，保证再次运行会重现上次的结果

    Args:
        seed: 种子数值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_directories(directories: Union[str, List[str]], is_clear: bool = False, use_timestamp: bool = False):
    """
    创建指定目录，对指定的目录进行检查、清空以及使用时间戳创建子目录

    Args:
        directories: 待处理的目录，可以是单个目录或多个目录的列表
        is_clear: 是否清空目录（True 表示清空目录，False 表示不清空目录，默认为 False）
        use_timestamp: 是否使用时间戳进行子目录创建（默认为 False）

    Returns:
        str: 创建目录时的时间戳，未使用返回空字符串
    """
    dir_type = type(directories)
    timestamp = time.strftime("%Y.%m.%d (%H-%M-%S)", time.localtime()) if use_timestamp else ""
    assert dir_type in [str, list], f"The folder type is {dir_type}, please select [str, list]."
    directories = [directories] if dir_type is str else directories

    # 遍历待处理目录，进行目录的处理
    for directory in sorted(directories):
        directory = os.path.join(directory, timestamp)
        if os.path.exists(directory) and is_clear:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    return timestamp


def generate_specify_files(root_dir: str, save_dir: str = ".", save_file_name="specify_files.scp",
                           prefix: str = None, suffix: str = None, keywords: list = None, limit: int = -1,
                           is_clear=False, is_show=False, description="Saving after generation"):
    """
    从指定目录及其子目录中检索特定条件的文件，返回检索到的文件路径，并写入保存文件中

    Args:
        root_dir: 待检索特定文件的根目录
        save_dir: 保存检索文件的保存目录
        save_file_name: 特定文件列表的保存文件名称（默认为"specify_files.scp"）
        prefix: 待检索特定文件的前缀
        suffix: 待检索特定文件的后缀
        keywords: 待检索特定文件需要包含的关键词
        limit: 待检索特定文件的数量限制
        is_clear: 是否清空保存特定文件路径的保存文件夹（默认为 False）
        is_show: 是否显示特定文件的检索进度
        description: 特定文件的检索进度描述（默认为"Saving after generation"）

    Returns:
        np.ndarray: 检索完的特定文件路径列表
    """
    specify_files = list()  # 初始化一个空列表来存储检索后的文件路径

    # 进行文件路径检索，进行前缀、后缀以及关键词的判别
    for root in os.walk(root_dir):
        for file in root[-1]:
            file_path = os.path.join(root[0], file)
            if (prefix is not None) and (not file.startswith(prefix)):
                continue  # 如果文件不以指定前缀开头，则跳过
            if (suffix is not None) and (not file.endswith(suffix)):
                continue  # 如果文件不以指定后缀结尾，则跳过
            if (keywords is not None) and (not any((keyword in file_path.split("\\")) for keyword in keywords)):
                continue  # 如果文件名不包含任何关键词，则跳过
            specify_files.append(file_path)
    specify_files = sorted(specify_files[:limit] if limit > 0 else specify_files)

    # 以写模式打开"save_file_name"文件，将检索到的文件路径写入文件
    if save_file_name is None:
        return np.array(specify_files)
    prepare_directories(save_dir, is_clear=is_clear)
    with open(os.path.join(save_dir, save_file_name), "w") as file:
        iterator = tqdm(specify_files, desc=description) if is_show else specify_files
        for specify_file in iterator:
            file.write(os.path.abspath(specify_file) + "\n")
    return np.array(specify_files)


def load_profile(toml_path: Union[str], need_obj: Union[bool] = False):
    """
    加载参数配置文件

    Args:
        toml_path: toml 格式的参数文件路径
        need_obj: 是否需要对象类型

    Returns:
        Union[dict, DictToObj]: toml 文件内的参数内容
    """
    params = toml.load(toml_path)
    return DictToObj(params) if need_obj else params


def initialize_config(module_cfg, pass_params=True):
    """
    初始化配置（实现类的实例化）

    Args:
        module_cfg: 模块配置信息
        pass_params: 是否传递参数

    Returns:
        object: 实例化的对象
    """
    module = importlib.import_module(module_cfg["module"])
    if pass_params:
        if "args" not in module_cfg.keys():
            return getattr(module, module_cfg["main"])()
        else:
            return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


@contextmanager
def suppress_output(hide_stdout=True, hide_stderr=True, hide_system=True):
    """
    可配置的隐藏输出上下文管理器

    :param hide_stdout: 是否隐藏标准输出
    :param hide_stderr: 是否隐藏错误输出
    :param hide_system: 是否隐藏系统命令输出
    :return: None
    """
    # 保存原始文件描述符
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)

    with open(os.devnull, "w") as devnull:
        try:
            if hide_stdout:
                os.dup2(devnull.fileno(), 1)
            if hide_stderr:
                os.dup2(devnull.fileno(), 2)
            if hide_system:
                original_system = os.system
                os.system = lambda *args, **kwargs: None
            yield
        finally:
            # 恢复原始文件描述符
            if hide_stdout:
                os.dup2(original_stdout, 1)
            if hide_stderr:
                os.dup2(original_stderr, 2)
            if hide_system:
                os.system = original_system
            os.close(original_stdout)
            os.close(original_stderr)


def mult_processing(datasets, args: Union[list], function, n_jobs: Union[int] = -1):
    """
    使用多进程处理数据集

    Args:
        datasets: 要处理的数据集
        args: 传递给处理函数的参数列表
        function: 处理函数
        n_jobs: 并行处理的进程数，默认值为-1，表示使用所有可用的CPU核心数减去2

    Returns:
        list: 处理结果列表
    """
    mult.freeze_support()

    # 创建线程池
    n_jobs = (mult.cpu_count() - 2) if n_jobs == -1 else n_jobs
    pools = mult.Pool(processes=n_jobs)

    # 拆分内容至线程池
    result = list()
    interval = len(datasets) // n_jobs + 1

    # 将数据集分割为适当大小的块，分配给线程池中的各个进程
    for idx in range(n_jobs):
        s_idx = idx * interval
        e_idx = (idx + 1) * interval

        # 如果起始索引大于等于数据集长度，则跳出循环
        if s_idx >= len(datasets):
            break

        # 如果结束索引为-1或大于数据集长度，则将其设置为数据集长度
        elif e_idx == -1 or e_idx > len(datasets):
            e_idx = len(datasets)

        # 将处理函数异步应用于数据块，并将结果添加到结果列表
        result.append(pools.apply_async(func=function, args=tuple([datasets[s_idx:e_idx]] + args)))

    pools.close()
    pools.join()

    return result


def generate_output_tree(output_list: List[Union[list, str]], depth: Union[int] = 0, site=None, title=None):
    """
    递归打印文件目录树状图

    Args:
        output_list: 根目录路径
        depth: 根目录、文件所在的层级号
        site: 存储出现转折的层级号
        title: 树状标题

    Returns:
        None
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


#################################################### Basic Classes #####################################################
# 用于计算程序的执行时间
class ExecutionTimer:
    """用于计算程序的执行时间"""

    def __init__(self):
        """初始化计时器"""
        self.start = time.time()

    def reset_time(self):
        """重置开始时间"""
        self.start = time.time()

    def duration(self):
        """获取当前已执行时间（持续时间）"""
        return int(time.time() - self.start)

    def duration_str(self):
        """获取当前已执行时间字符串（持续时间）"""
        duration = self.duration()
        return "[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60)

    def printDuration(self):
        """打印执行时间至终端"""
        duration = self.duration()
        print("[ Current execution time: %02dh:%02dm:%02ds ]" % (duration // 60 // 60, duration // 60, duration % 60))


# 由字典类型转换为对象类型
class DictToObj(object):
    """由字典类型转换为对象类型"""

    def __init__(self, data):
        """
        初始化字典转对象

        Args:
            data: 字典数据
        """
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [DictToObj(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, DictToObj(value) if isinstance(value, dict) else value)


class DirectoryTree:
    """目录树状结构生成器"""

    def __init__(self, show_hidden: bool = False, max_depth: int = None, exclude_dirs: list = None,
                 style: str = "default"):
        """
        初始化目录树生成器

        Args:
            show_hidden: 是否显示隐藏文件和目录
            max_depth: 最大遍历深度
            exclude_dirs: 要排除的目录名列表
            style: 显示风格 ("default", "round", "minimal", "bold", "double")
        """
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.exclude_dirs = exclude_dirs or []

        # 不同层级的连接符号
        self.connectors = CONNECTORS[style]

    def should_skip(self, item_path):
        """
        判断是否应该跳过该文件/目录

        Args:
            item_path: 文件/目录路径

        Returns:
            bool: 是否跳过
        """
        # 跳过隐藏文件/目录（除非show_hidden为True）
        if not self.show_hidden and item_path.name.startswith('.'):
            return True

        # 跳过排除的目录
        if item_path.is_dir() and item_path.name in self.exclude_dirs:
            return True

        return False

    def generate_tree(self, root_path, prefix: str = "", depth: int = 0, is_root: bool = True):
        """
        生成目录树的字符串表示

        Args:
            root_path: 根目录路径
            prefix: 前缀字符串
            depth: 当前深度
            is_root: 是否为根目录（内部使用）

        Returns:
            str: 目录树的字符串表示
        """
        if self.max_depth and depth > self.max_depth:
            return ""

        # 确保路径存在且是目录
        if not root_path.exists():
            return f"错误: 路径 '{root_path}' 不存在\n"
        if not root_path.is_dir():
            return f"错误: 路径 '{root_path}' 不是目录\n"

        tree_str = ""

        # 只在根目录显示目录名
        if is_root:
            tree_str = f"{root_path.name}/\n"
        else:
            # 非根目录不重复显示目录名，因为父级已经显示过了
            pass

        try:
            # 获取所有条目并排序
            items = sorted([item for item in root_path.iterdir()],
                           key=lambda x: (not x.is_dir(), x.name.lower()))

            # 过滤掉需要跳过的条目
            items = [item for item in items if not self.should_skip(item)]

            for index, item in enumerate(items):
                is_last = index == len(items) - 1

                # 当前条目的连接符
                connector = self.connectors['last'] if is_last else self.connectors['tee']

                # 添加当前条目到树中
                tree_str += f"{prefix}{connector}{item.name}"

                if item.is_dir():
                    tree_str += "/\n"

                    # 为下一级准备前缀
                    extension = self.connectors['space'] if is_last else self.connectors['branch']
                    next_prefix = prefix + extension

                    # 递归生成子目录树，标记为非根目录
                    subtree = self.generate_tree(item, next_prefix, depth + 1, is_root=False)
                    tree_str += subtree
                else:
                    tree_str += "\n"

        except PermissionError:
            tree_str += f"{prefix}└── [权限被拒绝]\n"

        return tree_str

    def print_tree(self, root_dir: str = "."):
        """
        打印目录树

        Args:
            root_dir: 根目录路径
        """
        root_path = Path(root_dir).resolve()
        print(f"\n目录树: {root_path}")
        print("=" * 50)
        tree = self.generate_tree(root_path)
        print(tree)


class TextFileSaver:
    def __init__(self, save_dir, save_name, sep=" ", encoding="utf-8"):
        self.save_dir = save_dir
        self.save_name = save_name
        self.sep = sep
        self.encoding = encoding
        self.count = 0

        prepare_directories(self.save_dir, is_clear=False)
        self.save_path = os.path.join(self.save_dir, self.save_name)
        self.handler = open(self.save_path, "w", encoding=self.encoding)

    def save(self, *args):
        args = [str(arg) for arg in args]
        self.handler.write(f"{self.sep}".join(args) + "\n")
        self.count += 1

    def __del__(self):
        print(f"The file {self.save_name} has been saved to {self.save_path}.")
        print(f"The file {self.save_name} has been saved {self.count} lines.")
        self.handler.close()


class TextFileReader:
    # 常见的编码类型（按照顺序进行验证）
    COMMON_ENCODINGS = ["utf-8", "gbk", "gbk2312", "big5", "latin1"]
    # BOM 标记与对应编码的映射
    BOM_MAPPING = {
        b"\xef\xbb\xbf": "utf-8-sig",
        b"\xff\xfe": "utf-16-le",
        b"\xfe\xff": "utf-16-be",
        b"\x00\x00\xfe\xff": "utf-32-le",
        b"\xff\xfe\x00\x00": "utf-32-be",
    }

    def __init__(self, default_encoding="utf-8"):
        """初始化文件读取器"""
        self.default_encoding = default_encoding

    def detect_bom_encoding(self, file_path):
        """检测文件的 BOM 标记并返回对应的编码"""
        # 假设存在BOM标记，并将其读取出来
        with open(file_path, "rb") as file:
            # 读取文件的前4个字节
            first_bytes = file.read(4)
        # 通过映射表查询BOM编码方式
        for bom, encoding in self.BOM_MAPPING.items():
            if first_bytes.startswith(bom):
                return encoding
        return None

    def detect_encoding(self, file_path):
        """检查文件编码方式（BOM查询优先、无BOM标记则尝试其余常见编码）"""
        # 首先检测BOM编码
        bom_encoding = self.detect_bom_encoding(file_path)
        if bom_encoding is not None:
            return bom_encoding, True

        # 依次尝试常见编码方式
        for encoding in self.COMMON_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    file.read()
                return encoding, False
            except UnicodeDecodeError:
                continue

        # 如果所有编码都无法解码，则返回默认编码
        return self.default_encoding, False

    def read_file(self, file_path):
        """读取文件内容并返回"""
        # 检测文件编码方式
        encoding, is_bom = self.detect_encoding(file_path)

        # 读取文件内容
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
