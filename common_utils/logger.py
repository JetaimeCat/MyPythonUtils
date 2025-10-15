# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : logger.py
# Software : PyCharm
# Time     : 2024/4/22 17:46
# Email    : 1206572082@qq.com

import os
import logging
import colorlog
from pathlib import Path
from inspect import currentframe
from inspect import getsourcefile

from common_utils.basic_utils import prepare_directories


class Logger:
    """
    日志类，用于输出和管理程序运行日志

    支持彩色控制台输出和文件记录，提供多种日志级别和格式化选项
    """

    def __init__(self, log_name: str, save_dir: str, level: str = "DEBUG", console: bool = True,
                 is_clear: bool = False, append_mode: bool = False):
        """
        初始化日志记录器

        Args:
            log_name: 日志文件的保存名称
            save_dir: 日志文件的保存目录
            level: 日志记录级别，可选 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            console: 是否在控制台输出日志
            is_clear: 是否清空保存日志的文件夹
            append_mode: 日志文件是否为追加模式
        """
        # 准备日志输出文件
        log_name = log_name if log_name.endswith(".log") else (log_name + ".log")
        prepare_directories(save_dir, is_clear)
        level = getattr(logging, level.upper())
        if append_mode and os.path.exists(os.path.join(save_dir, log_name)):
            os.remove(os.path.join(save_dir, log_name))

        # 输出格式以及对应类型的输出颜色
        self.formatter = "[%(asctime)s %(levelname)8s] %(message)s"
        self.colors = {"INFO": "green", "DEBUG": "cyan", "WARNING": "yellow",
                       "ERROR": "red", "CRITICAL": "bg_red"}

        # 有颜色格式和无颜色格式的创建
        self.formatter_colored = colorlog.ColoredFormatter(f"%(log_color)s{self.formatter}", log_colors=self.colors)
        self.formatter_colorless = logging.Formatter(self.formatter)

        # 创建 logger 用于日志打印并设置日志等级
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)

        # 创建一个 handler 用于写入日志内容至文件
        file_handler = logging.FileHandler(os.path.join(save_dir, log_name), mode="a", encoding="utf-8")
        file_handler.setFormatter(self.formatter_colorless)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)

        # 将日志输出至控制台
        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(self.formatter_colored)
            self.logger.addHandler(stream_handler)
        self.logger.info(f"< {log_name} >".center(85, "-"))

    @staticmethod
    def _log_info():
        """
        获取日志调用位置信息

        Returns:
            tuple: 包含文件路径、函数名和行号的元组
        """
        last_frame = currentframe().f_back.f_back.f_back
        file_path = os.path.split(getsourcefile(last_frame.f_code))[-1]
        # 获取当前正在执行的函数名
        function = last_frame.f_code.co_name
        function = "main" if function == "<module>" else function
        lineno = last_frame.f_lineno
        return file_path, function, lineno

    def _base_log(self, level: str, message: str, header: str, footer: str, end: str):
        """
        基础日志记录方法

        Args:
            level: 日志级别
            message: 日志消息内容
            header: 日志头部信息
            footer: 日志尾部信息
            end: 结束信息
        """
        file_path, function, lineno = self._log_info()
        if header is not None:
            getattr(self.logger, level)(f"< {header} [{file_path}--{function}--{lineno}]>".center(85, "-"))
        if message is not None:
            getattr(self.logger, level)(message)
        if footer is not None:
            getattr(self.logger, level)(f"< {footer} [{function}--{lineno}] >".center(85, "-"))
        if end is not None:
            getattr(self.logger, level)(end)

    def info(self, message: str = None, header: str = None, footer: str = None, end: str = None):
        """
        记录 INFO 级别的日志

        Args:
            message: 日志消息内容
            header: 日志头部信息
            footer: 日志尾部信息
            end: 结束信息
        """
        self._base_log("info", message, header, footer, end)

    def debug(self, message: str = None, header: str = None, footer: str = None, end: str = None):
        """
        记录 DEBUG 级别的日志

        Args:
            message: 日志消息内容
            header: 日志头部信息
            footer: 日志尾部信息
            end: 结束信息
        """
        self._base_log("debug", message, header, footer, end)

    def warning(self, message: str = None, header: str = None, footer: str = None, end: str = None):
        """
        记录 WARNING 级别的日志

        Args:
            message: 日志消息内容
            header: 日志头部信息
            footer: 日志尾部信息
            end: 结束信息
        """
        self._base_log("warning", message, header, footer, end)

    def error(self, message: str, immediately: bool = False, end: str = None):
        """
        记录 ERROR 级别的日志

        Args:
            message: 日志消息内容
            immediately: 是否立即退出程序
            end: 结束信息
        """
        file_path, function, lineno = self._log_info()
        self.logger.error(f"< {file_path}--{function}--{lineno} >".center(85, "-"))
        self.logger.error(message)
        if end is not None:
            self.logger.error(end)
        if immediately:
            self.exit()

    def critical(self, message: str):
        """
        记录 CRITICAL 级别的日志并退出程序

        Args:
            message: 日志消息内容
        """
        file_path, function, lineno = self._log_info()
        self.logger.critical(f"< {file_path}--{function}--{lineno} >".center(85, "-"))
        self.logger.critical(message)
        self.exit()

    def free(self, message: str = None, header: str = None, footer: str = None, end: str = None):
        """
        释放日志资源并记录一条警告日志

        Args:
            message: 日志消息内容
            header: 日志头部信息
            footer: 日志尾部信息
            end: 结束信息
        """
        self.warning(message=message, header=header, footer=footer, end=end)
        self.logger.handlers.clear()

    def exit(self):
        """
        释放日志资源并退出程序
        """
        self.free()
        exit(1)


def test_logger():
    """
    测试日志类的功能
    """
    # 创建 Logger 对象
    logger = Logger(log_name="logger", save_dir="test", level="DEBUG",
                    console=True, is_clear=True, append_mode=False)

    # 测试不同级别的日志记录
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    # 测试自定义头部和尾部信息
    logger.info(message="Message with custom header and footer.",
                header="CUSTOM HEADER", footer="CUSTOM FOOTER")

    # 测试异常情况
    try:
        result = 1 / 0
    except Exception as e:
        logger.error("An error occurred:", immediately=True, end=str(e))

    # 测试退出函数
    logger.exit()


if __name__ == '__main__':
    # 执行测试函数
    test_logger()
