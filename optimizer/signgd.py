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
from typing import Dict, Any, Tuple


class SignGD(optim.Optimizer):
    """
    Sign Gradient Descent (SignGD) 优化器，兼容标准 PyTorch LR 机制。
    通过 param_groups['lr'] 获取学习率。
    """

    def __init__(self, params, lr=1e-3, eps=1e-8):
        # 接收并存储 lr 在 defaults 中
        defaults = dict(lr=lr, eps=eps)
        super(SignGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 从 param_groups 中获取当前学习率
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                # 标准权重更新：w_{t+1} = w_t - lr * sign(g_t)
                p.sub_(p.grad.sign(), alpha=lr)
        return loss