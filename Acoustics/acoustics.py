# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : acoustics.py
# Software : PyCharm
# Time     : 2023/5/19 11:14
# Email    : 1206572082@qq.com
import os
import librosa
from tqdm import tqdm
import soundfile as sf
from typing import Union
import multiprocessing as mult

eps = 2.220446049250313e-16


def waveform_resample(wav_path: Union[str], sample_rate: Union[int], save_dir: Union[str] = None):
    """
    音频数据重采样，将音频数据以 sample_rate 的采样率加载数据

    :param wav_path: 音频数据文件的保存路径
    :param sample_rate: 进行重采样的采样率
    :param save_dir: 重采样后的音频数据文件保存根目录
    :return: 加载的音频数据
    """
    signal, _ = librosa.load(wav_path, sr=sample_rate)
    if save_dir is not None:
        file_name = os.path.split(wav_path)[-1]
        os.makedirs(save_dir, exist_ok=True)
        sf.write(os.path.join(save_dir, file_name), signal, sample_rate)
    return signal


def waveform_resample_mult(wav_paths: Union[list], sample_rate: Union[int], save_dir: Union[str] = None):
    signals = list()
    for wav_path in tqdm(wav_paths, unit="files", postfix={"sr": sample_rate, "save_dir": save_dir}):
        signal = waveform_resample(wav_path, sample_rate, save_dir)
        signals.append(signal)
    return signals


def mult_processing(datasets, args: Union[list], function, n_jobs: Union[int] = -1):
    mult.freeze_support()
    # 创建线程池
    n_jobs = (mult.cpu_count() - 2) if n_jobs == -1 else n_jobs
    pools = mult.Pool(processes=n_jobs)
    # 拆分内容至线程池
    result = list()
    interval = len(datasets) // n_jobs + 1
    for idx in range(n_jobs):
        s_idx = idx * interval
        e_idx = ((idx + 1) * interval) if (s_idx + interval) <= len(datasets) else -1
        if s_idx >= len(datasets):
            break
        result.append(pools.apply_async(func=function, args=tuple([datasets[s_idx:e_idx]] + args)))
    pools.close()
    pools.join()
    pools_res = list()
    for res in tqdm(result, desc="Results merging"):
        pools_res += res.get()
    return pools_res


if __name__ == '__main__':
    root_dir = r"F:\WorkSpace\PythonObject\SpeechLikeWaveformModulation\dataset"
    save_path = r"F:\MyPythonUtils\dataset"
    wav_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
    signals = mult_processing(wav_files, [16000, save_path], waveform_resample_mult)
