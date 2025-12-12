import torch
from torch.optim.optimizer import Optimizer


class NormalizedGD(Optimizer):
    """
    Normalized Gradient Descent with Momentum（归一化动量梯度下降）
    核心逻辑：
    1. 累积动量（基于原始梯度）：v_t = momentum * v_{t-1} + g_t
    2. 对动量做L2归一化：v_t = v_t / (||v_t||_2 + eps)
    3. 参数更新：w_{t+1} = w_t - lr * 归一化后的v_t
    参数:
        params (iterable): 待优化参数迭代器（如model.parameters()）
        lr (float): 学习率（必填）
        momentum (float): 动量因子，范围[0,1)（必填）
    """

    def __init__(self, params, lr, momentum, eps=0):
        # 参数合法性校验
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr} (must be >= 0.0)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum} (must be in [0, 1))")

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(NormalizedGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            eps = group['eps']  # 防止除零的极小值（固定值，无需暴露）

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # 初始化动量缓存
                if len(state) == 0:
                    state['momentum_buf'] = torch.zeros_like(p)

                # 1. 累积动量（基于原始梯度）
                buf = state['momentum_buf']
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # 2. 对动量做L2归一化（除以二范数）
                norm = torch.norm(buf, p=2)  # 计算动量的L2范数
                normalized_buf = buf / (norm + eps)  # 归一化，加eps防除零

                # 3. 参数更新
                p.sub_(normalized_buf, alpha=lr)

        return loss