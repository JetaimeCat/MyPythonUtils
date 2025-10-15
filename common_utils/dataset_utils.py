#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author    : Xia Zhaoxiang
# FileName  : dataset_utils.py
# Time      : 2025/4/28 14:53
# Email     : 1206572082@qq.com

import numpy as np
from collections import Counter


class Sampler:
    def __init__(self, strategy: str = "avg", random_state: int = None):
        """
        数据采样器，用于处理类别不平衡问题

        Args:
            strategy: 采样策略，可选：
                - "max": 欠采样多数类，使其数量等于少数类
                - "min": 过采样少数类，使其数量等于多数类
                - "avg": 均衡采样，使两类数量等于平均值
            random_state: 随机种子
        """
        self.strategy = strategy
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(self, x: np.ndarray, y: np.ndarray):
        """
        对输入数据进行采样，返回均衡后的数据

        Args:
            x: 特征数据，形状 (n_samples, n_features)
            y: 标签数据，形状 (n_samples,)

        Returns:
            tuple: 包含以下内容的元组
                - x_resampled: 采样后的特征数据
                - y_resampled: 采样后的标签数据
        """
        # 统计类别分布
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]

        # 获取各类别的索引
        majority_indices = np.where(y == majority_class)[0]
        minority_indices = np.where(y == minority_class)[0]

        # 根据策略计算目标采样数量
        if self.strategy == "min":
            target_n = n_minority  # 多数类降到少数类数量
            sampled_majority_indices = np.random.choice(majority_indices, target_n, replace=False)
            x_resampled = np.vstack([x[sampled_majority_indices], x[minority_indices]])
            y_resampled = np.hstack([y[sampled_majority_indices], y[minority_indices]])

        elif self.strategy == "max":
            target_n = n_majority  # 少数类升到多数类数量
            sampled_minority_indices = np.random.choice(minority_indices, target_n, replace=True)
            x_resampled = np.vstack([x[majority_indices], x[sampled_minority_indices]])
            y_resampled = np.hstack([y[majority_indices], y[sampled_minority_indices]])

        elif self.strategy == "avg":
            target_n = (n_majority + n_minority) // 2  # 两类数量取平均值
            # 多数类欠采样
            sampled_majority_indices = np.random.choice(majority_indices, target_n, replace=False)
            # 少数类过采样
            sampled_minority_indices = np.random.choice(minority_indices, target_n, replace=True)
            x_resampled = np.vstack([x[sampled_majority_indices], x[sampled_minority_indices]])
            y_resampled = np.hstack([y[sampled_majority_indices], y[sampled_minority_indices]])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Choose from ['max', 'min', 'avg']")

        return x_resampled, y_resampled
