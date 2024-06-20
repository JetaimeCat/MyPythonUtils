# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : test.py
# Software : PyCharm
# Time     : 9/27/2023 9:23 AM
# Email    : 1206572082@qq.com
import matplotlib.pyplot as plt
from Acoustics.windows import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 定义一个函数用于绘制窗口
def plot_window(window, window_name):
    plt.figure(figsize=(9, 6))
    plt.plot(window)
    plt.title(window_name)
    plt.xlabel('样本')
    plt.ylabel('幅度')
    plt.grid(linestyle="-.")
    plt.tight_layout()
    plt.show()


# 测试矩形窗口
number = 64  # 你可以根据需要更改窗口大小
rectangular_window = reg_window(number)
plot_window(rectangular_window, '矩形窗口')

# 测试汉宁窗口
hanning_window = hanning_window(number)
plot_window(hanning_window, '汉宁窗口')

# 测试Hamming窗口
hamming_window = hamming_window(number)
plot_window(hamming_window, 'Hamming窗口')

# 测试Blackman窗口
blackman_window = blackman_window(number)
plot_window(blackman_window, 'Blackman窗口')

# 测试Blackman-Harris窗口
blackman_harris_window = blackman_harris_window(number)
plot_window(blackman_harris_window, 'Blackman-Harris窗口')

# 测试Nuttall窗口
nuttall_window = nuttall_window(number)
plot_window(nuttall_window, 'Nuttall窗口')

# 测试Rife-Vincent窗口
rife_vincent_window = rife_vincent_window(number)
plot_window(rife_vincent_window, 'Rife-Vincent窗口')

# 测试Flat Top窗口
flat_top_window = flat_top_window(number)
plot_window(flat_top_window, 'Flat Top窗口')
