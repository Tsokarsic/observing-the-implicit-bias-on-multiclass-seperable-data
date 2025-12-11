import warnings
import numpy as np
import pandas as pd
import json
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
import wandb
import os
import argparse
from typing import Dict, Any, Tuple
from optimizer.signgd import SignGD


def calculate_implicit_bias_metrics(
        Wt: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        max_margin_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    计算当前权重 Wt 与 Max Margin 解的 L2/Linf/Spectral 相关性及相对 Margin 误差。

    Metrics:
    1. Correlation: <Wt, Vi> / (||Wt||_F ||Vi||_F)
    2. Relative Margin Error: |gamma_max_star - gamma(Wt)/||Wt||_norm| / gamma_max_star
    """
    metrics = {}

    # 将 PyTorch Wt 转换为 NumPy (k x d 形状)
    Wt_np = Wt.detach().cpu().numpy()

    # 1. 计算当前 Margin (gamma_t)
    n = X.shape[0]
    k = Wt.shape[0]

    # 计算所有样本的 score: (n, k)
    scores = torch.matmul(X, Wt.T)

    current_margins = []
    for i in range(n):
        y_i = Y[i].item()

        # Margin = score_yi - max(score_c | c != y_i)
        score_yi = scores[i, y_i]
        # 错误类别分数 (k-1 个标量)
        incorrect_scores = torch.cat([scores[i, :y_i], scores[i, y_i + 1:]])

        margin_i = score_yi - torch.max(incorrect_scores)
        current_margins.append(margin_i.item())

    gamma_t = np.min(current_margins) if current_margins else 0.0

    # 预先计算 Fro norm，用于相关性计算和 L2 归一化
    norm_Wt_fro = np.linalg.norm(Wt_np, 'fro')

    # 2. 迭代计算四种范数的指标
    for norm_key in ["L2_norm", "Linf_norm", "spectral_norm","nuclear_norm"]:

        result = max_margin_results[norm_key]
        Vi_np = np.array(result["matrix"])
        # gamma_i_star: 最优归一化 Margin
        gamma_i_star = result["gamma"]

        # --- 2.1 相关性 (Correlation) ---
        inner_product = np.sum(Wt_np * Vi_np)
        norm_Vi_fro = np.linalg.norm(Vi_np, 'fro')

        correlation = 0.0
        if norm_Wt_fro > 1e-12 and norm_Vi_fro > 1e-12:
            correlation = inner_product / (norm_Wt_fro * norm_Vi_fro)

        metrics[f'corr/{norm_key}_correlation'] = correlation

        # --- 2.2 相对 Gamma 误差 (Absolute Error Ratio) ---

        # a. 确定当前 Wt 对应的矩阵范数 ||Wt||_norm
        if norm_key == "L2_norm":
            current_norm_Wt = norm_Wt_fro
        elif norm_key == "Linf_norm":
            current_norm_Wt = np.max(np.abs(Wt_np))
        elif norm_key == "spectral_norm":
            current_norm_Wt = np.linalg.norm(Wt_np, 2)
        elif norm_key == "nuclear_norm":
            current_norm_Wt = np.linalg.norm(Wt_np, "nuc")
        else:
            current_norm_Wt = 0.0

        # b. 计算当前归一化 Margin: gamma(Wt) / ||Wt||_norm
        normalized_gamma_t = 0.0
        if current_norm_Wt > 1e-12:
            normalized_gamma_t = gamma_t / current_norm_Wt

        # c. 计算相对 Gamma 误差 (Error): |gamma_max_star - normalized_gamma_t| / gamma_max_star
        relative_gamma_error = 0.0
        if gamma_i_star > 1e-12:
            absolute_diff = np.abs(gamma_i_star - normalized_gamma_t)
            relative_gamma_error = absolute_diff / gamma_i_star

        # 记录误差 (主要指标)
        metrics[f'gamma_error/{norm_key}_error_from_opt'] = relative_gamma_error
        # 记录归一化 Margin (次要指标，方便调试)
        metrics[f'gamma_norm/{norm_key}_normalized_gamma'] = normalized_gamma_t
        metrics[f'norm/Wt_{norm_key}'] = current_norm_Wt

    metrics['gamma/current_min_margin_unnormalized'] = gamma_t
    metrics['norm/Wt_L2_fro'] = norm_Wt_fro

    return metrics


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """根据配置创建优化器。从 optimizer_params 中提取 LR。"""
    optim_name = config['optimizer']
    optim_params = config['optimizer_params'].get(optim_name, {})
    if optim_name == 'SignGD':
        return SignGD(model.parameters(),  **optim_params)
    elif optim_name == 'Adam':
        return optim.Adam(model.parameters(),  **optim_params)
    elif optim_name in ('GD', 'MomentumGD'):
        # 对于 GD/SGD，从 params 中提取 momentum（GD 默认为 0.0）
        return optim.SGD(model.parameters(),  **optim_params)
    else:
        raise ValueError(f"不支持的优化器: {optim_name}")


import numpy as np

def get_lr_scheduler(config):
    """
    实现：前 N 步恒定学习率 → 之后 $\frac{1}{\sqrt{t - N + t_0}}$ 衰减
    默认 N=5000 步，可通过 config 自定义
    ⚠️ base_lr 从当前优化器的 params 中提取
    """
    schedule_config = config['lr_schedule']
    schedule_type = schedule_config['type']

    # 从优化器参数中提取 base_lr（保持原有逻辑）
    optim_name = config['optimizer']
    optim_params = config['optimizer_params'].get(optim_name, {})
    base_lr = optim_params.get('lr', 1e-3)  # 兜底默认值 1e-3

    # 新增：恒定学习率的步数（默认5000步，可通过 config 调整）
    warmup_steps = schedule_config.get('warmup_steps', 5000)  # 核心新增参数
    t0 = schedule_config.get('t0', 1.0)  # 衰减的偏移量（保持原有）

    if schedule_type == "sqrt_t_decay_with_warmup":
        def lr_scheduler(step):
            # 前 warmup_steps 步：恒定 base_lr
            if step < warmup_steps:
                return base_lr
            # warmup_steps 后：$\frac{base\_lr}{\sqrt{(step - warmup_steps) + t0}}$ 衰减
            else:
                decay_step = step - warmup_steps
                actual_lr=base_lr*np.sqrt(t0)/ np.sqrt(float(decay_step) + t0)
                # print(actual_lr)# 衰减阶段的起始步（从0开始）
                return actual_lr

        return lr_scheduler
    else:
        # 其他调度类型：保持原有逻辑（返回恒定 base_lr）
        return lambda step: base_lr