import numpy as np


def reg_window(number):
    return np.ones(number)


def hanning_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.5
    window = window - 0.5 * np.cos((2 * np.pi * n) / (number - 1))
    return window


def hamming_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.54
    window = window - 0.46 * np.cos((2 * np.pi * n) / (number - 1))
    return window


def blackman_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.42
    window = window - 0.5 * np.cos((2 * np.pi * n) / (number - 1))
    window = window + 0.8 * np.cos((4 * np.pi * n) / (number - 1))
    return window


def blackman_harris_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.35875
    window = window - 0.48829 * np.cos((2 * np.pi * n) / (number - 1))
    window = window + 0.14128 * np.cos((4 * np.pi * n) / (number - 1))
    window = window - 0.01168 * np.cos((6 * np.pi * n) / (number - 1))
    return window


def nuttall_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.355768
    window = window - 0.487396 * np.cos((2 * np.pi * n) / (number - 1))
    window = window + 0.144232 * np.cos((4 * np.pi * n) / (number - 1))
    window = window - 0.012604 * np.cos((6 * np.pi * n) / (number - 1))
    return window


def rife_vincent_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 1
    window = window - 1.5 * np.cos((2 * np.pi * n) / (number - 1))
    window = window + 0.6 * np.cos((4 * np.pi * n) / (number - 1))
    window = window - 0.1 * np.cos((6 * np.pi * n) / (number - 1))
    return window


def flat_top_window(number):
    n = np.array([i for i in range(number)], dtype=int)
    window = 0.21706
    window = window - 0.42103 * np.cos((2 * np.pi * n) / (number - 1))
    window = window + 0.28294 * np.cos((4 * np.pi * n) / (number - 1))
    window = window - 0.07897 * np.cos((6 * np.pi * n) / (number - 1))
    return window
