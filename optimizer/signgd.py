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

import torch
from torch.optim.optimizer import Optimizer


class SignGD(Optimizer):
    """
    Signum 算法（Sign Gradient Descent + 动量，先动量后取Sign），兼容标准 PyTorch LR 机制。
    核心逻辑：先累积动量 → 再对动量取符号 → 最后参数更新
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.0,weight_decay=0):  # 仅新增momentum参数
        defaults = dict(lr=lr, eps=eps, momentum=momentum,weight_decay=weight_decay)  # 仅新增momentum
        super(SignGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']  # 仅新增取值
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # 初始化动量缓存（仅新增）
                state = self.state[p]
                if len(state) == 0 and momentum > 0:
                    state['momentum_buf'] = torch.zeros_like(p)

                # Signum核心：先动量累积（基于原始梯度）→ 再取符号
                grad = p.grad
                if momentum > 0:
                    buf = state['momentum_buf']
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)  # 先累积动量：v = γ*v + g
                    update = buf.sign()  # 再对动量取符号
                else:
                    update = grad.sign()  # 无动量时退化为原SignGD
                p.mul_(1-lr*weight_decay)
                p.sub_(update, alpha=lr)
        return loss