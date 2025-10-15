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
import multiprocessing as mult

EPSILON = 2.220446049250313e-16


def resample(wav_path: Union[str], sample_rate: Union[int], save_dir: Union[str] = None):
    """
    音频数据重采样，将音频数据以指定采样率加载数据

    Args:
        wav_path: 音频数据文件的保存路径
        sample_rate: 进行重采样的采样率
        save_dir: 重采样后的音频数据文件保存根目录

    Returns:
        np.ndarray: 加载的音频数据
    """
    signal, _ = librosa.load(wav_path, sr=sample_rate)
    if save_dir is not None:
        file_name = os.path.split(wav_path)[-1]
        os.makedirs(save_dir, exist_ok=True)
        sf.write(os.path.join(save_dir, file_name), signal, sample_rate)
    return signal


def waveform_resample(wav_path: Union[list], sample_rate: Union[int], save_dir: Union[str] = None):
    """
    对多条语音信号进行重采样处理，并在文件维度上合并

    Args:
        wav_path: wav 文件路径列表
        sample_rate: wav 语音数据采样率
        save_dir: 重采样文件保存路径

    Returns:
        list: 重采样完成的数据内容列表
    """
    signals = list()
    postfix = {"sr": sample_rate, "save_dir": save_dir}
    for wav_file in tqdm(wav_path, desc=f"Resample [{os.getpid()}]", unit="files", postfix=postfix):
        signal = resample(wav_file, sample_rate, save_dir)
        signals.append(signal)
    return signals


def get_windows(window, frame_length):
    """
    加窗参数处理

    Args:
        window: 窗函数，通过判断该参数类型进行不同操作
        frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度

    Returns:
        tuple: 窗内容、窗长
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


def enframe(signal, sample_rate, window=None, frame_length=0.02, hop_length=0.01):
    """
    原始语音信号分帧

    Args:
        signal: 原始语音数据
        sample_rate: 原始语音采样率
        window: 窗函数，通过判断该参数类型进行不同操作
        frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度
        hop_length: 帧移间隔，当该参数为小数时通过采样率计算得到，当该参数为整数时即为帧间隔

    Returns:
        np.ndarray: 完成分帧、加窗的语音数据
    """
    # 计算获得当前帧长
    if isinstance(frame_length, float):
        frame_length = int(sample_rate * frame_length)
    # 计算获得当前帧移间隔
    if isinstance(hop_length, float):
        hop_length = int(sample_rate * hop_length)

    # 获取信号数据长度、帧数量
    length = len(signal)
    frames_num = (length - frame_length) // hop_length + 1
    # 加窗参数处理
    window, win_length = get_windows(window, frame_length)

    # 初始化帧，获取每帧的起始样点索引
    frames_idx = hop_length * np.array([i for i in range(frames_num)])
    frames = np.zeros((frames_num, frame_length))
    for i in range(frames_num):
        frames[i, :] = signal[frames_idx[i]:frames_idx[i] + frame_length]
    frames = frames * np.array(window)
    return frames


def reframe(frames, sample_rate, window=None, frame_length=0.02, hop_length=0.01, max_length=None):
    """
    分帧数据还原波形

    Args:
        frames: 完成分帧的波形数据
        sample_rate: 原始语音采样率
        window: 窗函数，通过判断该参数类型进行不同操作
        frame_length: 帧长，当该参数为小数时通过采样率计算帧长度，当该参数为整数时即为帧长度
        hop_length: 帧移间隔，当该参数为小数时通过采样率计算得到，当该参数为整数时即为帧间隔
        max_length: 限制波形数据长度

    Returns:
        np.ndarray: 还原后的波形数据
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


def waveform_enframe(wav_path, sample_rate, window=None, frame_length: int = 160, hop_length: int = 80):
    """
    对多条语音信号进行分帧处理，并在帧维度上合并

    Args:
        wav_path: wav 文件路径
        sample_rate: wav 语音数据采样率
        window: 窗函数选择
        frame_length: wav 语音数据帧长
        hop_length: wav 语音数据帧间隔

    Returns:
        np.ndarray: 分帧完成的数据内容
    """
    frames_array = np.array(list())
    postfix = {"sample_rate": sample_rate, "frame_length": frame_length, "hop_length": hop_length}
    for wav_file in tqdm(wav_path, desc=f"Enframe [{os.getpid()}]", unit="files", postfix=postfix):
        signal, sr = librosa.load(wav_file, sr=8000)
        frames = enframe(signal, sample_rate, window, frame_length, hop_length)
        frames_array = np.r_[frames_array, frames] if len(frames_array) != 0 else frames
    return frames_array
