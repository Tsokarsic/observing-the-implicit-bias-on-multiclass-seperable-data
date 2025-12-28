import torch


class PGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                grad = p.grad
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad, alpha=1.0 - mu)
                # 向量直接使用msgd，矩阵变量才使用nucgd
                if p.ndim < 2:
                    if wd != 0:
                        p.add_(p, alpha=-lr * wd)
                    p.add_(buf, alpha=-lr)
                    continue
                # 默认k>=d保证性能，若不然则进行转置换
                k, d = p.shape
                do_transpose = k < d
                M_t = buf.T if do_transpose else buf
                # 初始化投影向量
                if "p_v" not in state:
                    state["p_v"] = torch.randn(M_t.size(0), 1, dtype=p.dtype, device=p.device)
                p_v = state["p_v"]
                # 幂迭代
                next_p = M_t @ (M_t.T @ p_v)
                p_v.copy_(next_p / torch.linalg.norm(next_p))
                vt = p_v.T @ M_t
                delta = lr * p_v @ (vt / torch.linalg.norm(vt))
                if wd != 0:
                    p.mul_(1 - lr * wd)
                if do_transpose:
                    p.add_(delta.T, alpha=-1.0)
                else:
                    p.add_(delta, alpha=-1.0)

        return loss