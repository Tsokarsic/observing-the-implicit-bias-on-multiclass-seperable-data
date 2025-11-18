import torch
import torch.nn as nn
import wandb
import json
import os
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from generate_data import *
import datetime

from typing import Dict, Any
# ==============================================================================
# 1. 主训练函数
# ==============================================================================

def train(config_path: str = "config.json"):
    # 1. 加载配置：直接命名为 config
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = json.load(f)

    # 动态生成 WandB 运行名称 (使用字典访问 config['key'])
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    optimizer_name = config['training']['optimizer']
    optim_params = config['training']['optimizer_params'].get(optimizer_name, {})
    base_lr = optim_params.get('lr', 1e-3)
    run_name = f"{now}_{optimizer_name}_lr{base_lr}"

    # 2. 配置 WandB
    # 注意：不再将 wandb.config 赋值给任何变量
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

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)

    # 修正：使用字典访问 config['data']['k']
    model = nn.Linear(d_features, config['data']['k'], bias=False)
    loss_fn = nn.CrossEntropyLoss()

    # 5. 初始化优化器和学习率调度器
    # 传入 config 的 'training' 子字典
    optimizer = get_optimizer(model, config['training'])
    lr_scheduler = get_lr_scheduler(config['training'])

    # 6. 训练循环
    print(f"\n--- 2. 开始训练 ---")
    current_step = 0

    # 使用字典访问 config['training']['epochs']
    for epoch in range(1, config['training']['epochs'] + 1):
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

            optimizer.step()

        # 7. 指标计算与日志记录
        # 使用字典访问 config['training']['log_interval']
        if epoch % config['training']['log_interval'] == 0 or epoch == config['training']['epochs']:
            Wt = model.weight.data
            metrics = calculate_implicit_bias_metrics(Wt, X_tensor, y_tensor, max_margin_results)

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / n_samples

            log_data = {
                "epoch": epoch,
                "loss/train_loss": loss.item(),
                "accuracy/train_accuracy": accuracy,
                "lr/current_lr": current_lr,
                **metrics
            }
            wandb.log(log_data, step=epoch)

            print(
                f"Epoch {epoch}/{config['training']['epochs']} | Loss: {loss.item():.6f} | LR: {current_lr:.6e} | Spec_Err: {metrics['gamma_error/spectral_norm_error_from_opt']:.4e}")

    print("\n--- 3. 训练完成 ---")
    wandb.finish()

train(config_path="config.json")