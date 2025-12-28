import torch
import torch.nn as nn
from transformers.integrations import run_hp_search_wandb

import wandb
import json
import os
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import *
import datetime

from typing import Dict, Any
# ==============================================================================
# 1. 主训练函数
# ==============================================================================

def train(config_path: str = "config.json",optimizer1=None):
    # 1. 加载配置：直接命名为 config
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = json.load(f)
    use_wandb=config['use_wandb']
    # 动态生成 WandB 运行名称 (使用字典访问 config['key'])
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if optimizer1 is not None:
        optimizer_name = optimizer1
        config['training']['optimizer']=optimizer1
    else:
        optimizer_name = config['training']['optimizer']
    print(optimizer_name)
    optim_params = config['training']['optimizer_params'].get(optimizer_name, {})
    base_lr = optim_params.get('lr', 1e-3)
    if "momentum" in optim_params:
        momentum = optim_params['momentum']
    else:
        momentum = 0
    run_name = f"{now}_{optimizer_name}_lr{base_lr}_momentum{momentum}"

    # 2. 配置 WandB
    # 注意：不再将 wandb.config 赋值给任何变量
    if use_wandb:
        wandb.init(
            project=config['wandb_project'],
            config=config,  # 传入原始字典
            name=run_name
        )

    # 3. 数据加载或生成
    print("--- 1. 数据加载与 Max Margin 求解 ---")
    try:
        # load_data_or_generate 接收配置的 'data' 子字典
        X_np, y_np, _, _, max_margin_results = load_data_or_generate(config_data=config['data'])
    except Exception as e:
        print(f"致命错误：数据处理失败。")
        wandb.finish()
        raise e

    # 4. 初始化模型、DataLoader
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)
    n_samples, d_features = X_tensor.shape
    # sigma=config['training']['noise_scale']
    dataset = TensorDataset(X_tensor, y_tensor)
    batch_size=config['training']['batch_size']
    if not int(batch_size) >=1 :
        dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
        max_epochs = config['training']['epochs']
        batch_size = n_samples
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        max_epochs = int(config['training']['epochs'] / n_samples * batch_size)
    init_method = config['training'].get('init_method', 'gaussian').lower()  # 默认使用高斯
    init_scale = config['training'].get('init_scale', 0.01)
    model = nn.Linear(d_features, config['data']['k'], bias=False)
    loss_fn = nn.CrossEntropyLoss()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 零初始化 (Zero Initialization)
            if init_method == 'zero':
                nn.init.constant_(module.weight, 0.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                print(f"   -> 模块 '{name}' 权重设为零。")

            # 高斯初始化 (Gaussian/Normal Initialization)
            elif init_method == 'gaussian':
                # 使用 PyTorch 内建的 Normal 初始化
                nn.init.normal_(module.weight, mean=0.0, std=init_scale)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # 偏置通常设为零
                print(f"   -> 模块 '{name}' 权重使用高斯分布 (Std={init_scale})。")

            # 默认/其他初始化
            else:
                warnings.warn(f"⚠️ 未知的初始化方法 '{init_method}'。使用 PyTorch 默认初始化。")
                # PyTorch 默认初始化 (通常是 Kaiming Uniform)

    # 5. 初始化优化器和学习率调度器
    # 传入 config 的 'training' 子字典
    optimizer = get_optimizer(model, config['training'])
    lr_scheduler = get_lr_scheduler(config['training'],optim_params)

    # 6. 训练循环
    print(f"\n--- 2. 开始训练 ---")
    current_step = 0

    print(max_epochs)
    # 使用字典访问 config['training']['epochs']
    for epoch in range(1,max_epochs + 1):
        for inputs, targets in dataloader:

            current_step += 1

            # 🚀 核心：计算当前学习率 (1/sqrt(t) 衰减)
            current_lr = lr_scheduler(current_step)

            # 统一的 LR 更新机制
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            # if sigma > 1e-8:  # 避免sigma为0时无意义计算
            #     for param in model.parameters():
            #         if param.grad is not None:  # 确保梯度存在
            #             # 生成与梯度同形状的标准高斯噪声 (均值0，标准差sigma)
            #             noise = torch.randn_like(param.grad) * sigma*torch.norm(param.grad,"fro")
            #             # 叠加噪声到梯度上
            #             param.grad.add_(noise)
            optimizer.step()

        # 7. 指标计算与日志记录
        # 使用字典访问 config['training']['log_interval']
        if current_step % config['training']['log_interval'] == 0 or epoch == max_epochs:
            Wt = model.weight.data
            metrics = calculate_implicit_bias_metrics(Wt, X_tensor, y_tensor, max_margin_results)

            # 全局准确率：在当前模型权重下用全量样本评估
            with torch.no_grad():
                all_outputs = model(X_tensor)
                _, predicted = torch.max(all_outputs, 1)
                correct = (predicted == y_tensor).sum().item()
                # print(predicted)
                accuracy = correct / n_samples

            # --- 提取打印所需数据 (Current/Optimal Gamma & Correlation) ---

            # L2 Norm (Frobenius)
            normalized_L2_gamma = metrics['gamma_norm/L2_norm_normalized_gamma']
            optimal_L2_gamma = max_margin_results['L2_norm']['gamma']
            L2_corr = metrics['corr/L2_norm_correlation']

            # Linf Norm
            normalized_Linf_gamma = metrics['gamma_norm/Linf_norm_normalized_gamma']
            optimal_Linf_gamma = max_margin_results['Linf_norm']['gamma']
            Linf_corr = metrics['corr/Linf_norm_correlation']

            # Spectral Norm
            normalized_spec_gamma = metrics['gamma_norm/spectral_norm_normalized_gamma']
            optimal_spec_gamma = max_margin_results['spectral_norm']['gamma']
            spec_corr = metrics['corr/spectral_norm_correlation']

            normalized_nuclear_gamma = metrics['gamma_norm/nuclear_norm_normalized_gamma']
            optimal_nuclear_gamma = max_margin_results['nuclear_norm']['gamma']
            nuclear_corr = metrics['corr/nuclear_norm_correlation']

            log_data = {
                "step": current_step,
                "loss/train_loss": loss.item(),
                "accuracy/train_accuracy": accuracy,
                "lr/current_lr": current_lr,
                **metrics
            }
            if use_wandb:
                wandb.log(log_data, step=epoch)

            # --- 最终修改后的 Print 语句 ---
            print(
                f"Step {current_step}/{config['training']['epochs']} | LR：{current_lr}.4f ｜ Loss: {loss.item():.6f} | Acc: {accuracy:.4f} | "
                f"G(L2): {normalized_L2_gamma:.4f}/{optimal_L2_gamma:.4f} (Corr: {L2_corr:.4f}) | "
                f"G(Linf): {normalized_Linf_gamma:.4f}/{optimal_Linf_gamma:.4f} (Corr: {Linf_corr:.4f}) | "
                f"G(Spec): {normalized_spec_gamma:.4f}/{optimal_spec_gamma:.4f} (Corr: {spec_corr:.4f}) | "
                f"G(Nuclear): {normalized_nuclear_gamma:.4f}/{optimal_nuclear_gamma:.4f} (Corr: {nuclear_corr:.4f})"
            )
        if epoch == max_epochs:
            matrix = model.weight.data
            matrix=matrix/np.linalg.norm(matrix)
            # Plot singular values

    print("\n--- 3. 训练完成 ---")
    if use_wandb:
        wandb.finish()
    return matrix
train(config_path="config.json")
matrix1=train(config_path="config.json",optimizer1="Muon")
matrix2=train(config_path="config.json",optimizer1="NucGD")
matrix3=train(config_path="config.json",optimizer1="NGD")
matrix4=train(config_path="config.json",optimizer1="SignGD")
import matplotlib.pyplot as plt
weight_matrices={"Muon(Lspec)":matrix1,"NucGD(Lnuc)":matrix2,"NGD(L2)":matrix3,"SignGD(Linf)":matrix4}
for name, matrix in weight_matrices.items():
    # Compute singular values
    _, s, _ = np.linalg.svd(matrix)
    # Plot singular values
    plt.semilogy(range(1, len(s)+1), s, 'o-', label=name)
plt.xlabel('Index')
plt.ylabel('Singular Value (log scale)')
plt.title('Singular Value Spectrum of Solutions Of NSD Under Different Norm')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("spectrum_for_algorithms.png")
plt.show()
