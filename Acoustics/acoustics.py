# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : acoustics.py
# Software : PyCharm
# Time     : 2023/5/19 11:14
# Email    : 1206572082@qq.com
import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from typing import Union
import multiprocessing as multi

EPSILON = 2.220446049250313e-16


def synth_noisy(clean_path: Union[str], noise_path: Union[str, list], sample_rate, save_dir=None):
    """
    合成带噪音的音频

    :param clean_path: 纯净音频的文件路径
    :param noise_path: 噪声音频的文件路径或文件路径列表
    :param sample_rate: 采样率
    :param save_dir: 保存带噪音音频的目录（可选）
    :return: 带噪音的音频数据
    """
    # 噪声路径是否是列表，若是则抽取一个进行合成
    if isinstance(noise_path, list):
        noise_path = np.random.choice(noise_path)
    # 读取音频数据
    clean, _ = librosa.load(clean_path, sr=sample_rate)
    noise, _ = librosa.load(noise_path, sr=sample_rate)
    # 抽取噪声片段
    s_idx = np.random.randint(0, len(noise) - len(clean) - 1)
    e_idx = s_idx + len(clean)
    noise = noise[s_idx:e_idx]
    # 对音频数据进行增强
    snr = np.float32(np.random.randint(-10, 20))
    gain = 10.0 ** (-1.0 * snr / 20.0)
    noise = gain * noise * np.linalg.norm(clean) / np.linalg.norm(noise)
    noisy = (clean + noise) * 10.0 ** (-1.0 * (np.random.rand() - 0.5))
    if save_dir is not None:
        file_name = os.path.split(clean_path)[-1] + "_" + os.path.split(noise_path)[-1]
        os.makedirs(save_dir, exist_ok=True)
        sf.write(os.path.join(save_dir, file_name), noisy, sample_rate)
    return noisy


def resample(wav_path: Union[str], sample_rate: Union[int], save_dir: Union[str] = None):
    """
    音频数据重采样，将音频数据以 sample_rate 的采样率加载数据

    :param wav_path: 音频数据文件的保存路径
    :param sample_rate: 进行重采样的采样率
    :param save_dir: 重采样后的音频数据文件保存根目录
    :return: 加载的音频数据
    """
    sample, _ = librosa.load(wav_path, sr=sample_rate)
    if save_dir is not None:
        file_name = os.path.split(wav_path)[-1]
        os.makedirs(save_dir, exist_ok=True)
        sf.write(os.path.join(save_dir, file_name), sample, sample_rate)
    return sample


def get_windows(window, frame_length):
    """
    加窗参数处理

    :param window: 窗函数，通过判断该参数类型进行不同操作。
    :param frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度。
    :return: 窗内容、窗长
    """
    if window is None:
        window = np.ones(frame_length)
    elif isinstance(window, int):
        window = np.ones(window)
    elif isinstance(window, list):
        window = np.array(window)
    win_length = len(window)
    assert frame_length == win_length, f"frame_length[{frame_length}] does not match the window[{win_length}]."
    return window, win_length


def enframe(sample, sample_rate, window=None, frame_length=0.02, hop_length=0.01):
    """
    原始语音信号分帧

    :param sample: 原始语音数据。
    :param sample_rate: 原始语音采样率。
    :param window: 窗函数，通过判断该参数类型进行不同操作。
    :param frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度。
    :param hop_length: 帧移间隔，当该参数为小数时通过采样率计算得到，当该参数为整数时即为帧间隔。
    :return: 完成分帧、加窗的语音数据。
    """
    # 计算获得当前帧长
    if isinstance(frame_length, float):
        frame_length = int(sample_rate * frame_length)
    # 计算获得当前帧移间隔
    if isinstance(hop_length, float):
        hop_length = int(sample_rate * hop_length)

    # 获取信号数据长度、帧数量
    length = len(sample)
    frames_num = (length - frame_length) // hop_length + 1
    # 加窗参数处理
    window, win_length = get_windows(window, frame_length)

    # 初始化帧，获取每帧的起始样点索引
    frames_idx = hop_length * np.array([i for i in range(frames_num)])
    frames = np.zeros((frames_num, frame_length))
    for i in range(frames_num):
        frames[i, :] = sample[frames_idx[i]:frames_idx[i] + frame_length]
    frames = frames * np.array(window)
    return frames


def reframe(frames, sample_rate, window=None, frame_length=0.02, hop_length=0.01, max_length=None):
    """
    分帧数据还原波形

    :param frames: 完成分帧的波形数据。
    :param sample_rate: 原始语音采样率。
    :param window: 窗函数，通过判断该参数类型进行不同操作。
    :param frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度。
    :param hop_length: 帧移间隔，当该参数为小数时通过采样率计算得到，当该参数为整数时即为帧间隔。
    :param max_length: 限制波形数据长度。
    :return: 完成分帧、加窗的语音数据。
    """
    # 计算获得当前帧长
    if isinstance(frame_length, float):
        frame_length = int(sample_rate * frame_length)
    # 计算获得当前帧移间隔
    if isinstance(hop_length, float):
        hop_length = int(sample_rate * hop_length)

    # 加窗参数处理
    window, win_length = get_windows(window, frame_length)
    frames = frames / window
    frames = frames[:, :hop_length]
    samples = frames.reshape(-1)
    if max_length is None:
        return samples
    elif max_length > len(samples):
        samples = np.c_[samples, np.zeros(max_length - len(samples))]
        samples = samples.reshape(-1)
    else:
        samples = samples[:max_length]
    return samples


def waveform_resample(wav_path: Union[list], sample_rate: Union[int], save_dir: Union[str] = None):
    """
    对多条语音信号进行重采样处理，并在文件维度上合并

    :param wav_path: wav 文件路径
    :param sample_rate: wav 语音数据采样率
    :param save_dir: 重采样文件保存路径
    :return: 重采样完成的数据内容
    """
    samples = list()
    postfix = {"sr": sample_rate, "save_dir": save_dir}
    for wav_file in tqdm(wav_path, desc=f"Resample [{os.getpid()}]", unit="files", postfix=postfix):
        sample = resample(wav_file, sample_rate, save_dir)
        samples.append(sample)
    return samples


def waveform_enframe(wav_path, sample_rate, window=None, frame_length: int = 160, hop_length: int = 80):
    """
    对多条语音信号进行分帧处理，并在帧维度上合并

    :param wav_path: wav 文件路径
    :param sample_rate: wav 语音数据采样率
    :param window: 窗函数选择
    :param frame_length: wav 语音数据帧长
    :param hop_length: wav 语音数据帧间隔
    :return: 分帧完成的数据内容
    """
    frames_array = np.array(list())
    postfix = {"sample_rate": sample_rate, "frame_length": frame_length, "hop_length": hop_length}
    for wav_file in tqdm(wav_path, desc=f"Enframe [{os.getpid()}]", unit="files", postfix=postfix):
        sample, sr = librosa.load(wav_file, sr=8000)
        frames = enframe(sample, sample_rate, window, frame_length, hop_length)
        frames_array = np.r_[frames_array, frames] if len(frames_array) != 0 else frames
    return frames_array


def multi_processing(datasets, args: Union[list], function, n_jobs: Union[int] = -1):
    """
    使用多进程处理数据集。

    :param datasets: 要处理的数据集。
    :param args: 传递给处理函数的参数列表。
    :param function: 处理函数。
    :param n_jobs: 并行处理的进程数。默认值为-1，表示使用所有可用的CPU核心数减去2。
    :return: 处理结果列表。
    """
    multi.freeze_support()

    # 创建进程池
    n_jobs = (multi.cpu_count() - 2) if n_jobs == -1 else n_jobs
    pools = multi.Pool(processes=n_jobs)

    # 拆分内容至进程池
    result = list()
    interval = len(datasets) // n_jobs + 1

    # 将数据集分割为适当大小的块，分配给进程池中的各个进程
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
