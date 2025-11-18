from os import SEEK_DATA

from scipy.stats import chi2  # 用于χ²分布验证
import warnings
import numpy as np
import pandas as pd
import json
import cvxpy as cp
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.optim as optim
from selenium.webdriver.common.devtools.v135.page import start_screencast
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")  # 忽略数值计算警告


def compute_highdim_gaussian_cutoff(d: int, sigma: float, confidence: float = 0.99) -> float:
    """
    计算高维高斯噪声的合理截断半径R（基于χ²分布的置信区间）
    参数：
        d: 特征维度
        sigma: 高斯噪声标准差
        confidence: 置信水平（如0.99，覆盖99%的原始高斯样本）
    返回：
        R: 截断半径（L2范数上限）
    """
    # 1. 高斯范数平方的分布：σ²·χ²(d)
    chi2_df = float(d)  # χ²分布自由度
    chi2_quantile = chi2.ppf(confidence, df=chi2_df)  # χ²分布的confidence分位数
    # 2. 计算截断半径的平方（基于χ²分位数）
    cutoff_norm_sq = sigma ** 2 * chi2_quantile
    R = np.sqrt(cutoff_norm_sq)

    # 3. 打印理论统计信息（帮助验证合理性）
    print(f"📊 高维高斯截断参数（d={d}, σ={sigma}）：")
    print(f"   - 截断半径R（覆盖{confidence * 100}%样本）：{R:.4f}")
    print(f"   - 截断半径对应的范数平方：{cutoff_norm_sq:.4f}")
    return R


def generate_highdim_gaussian_noise(
        d: int, sigma: float, R: float, max_retries: int = 1000
) -> np.ndarray:
    """
    生成高维高斯噪声（约束在L2球内），返回符合条件分布的噪声
    方法：拒绝采样法，确保分布为 ε | ||ε||₂ ≤ R
    """
    for i in range(max_retries):
        # 1. 生成原始高斯噪声
        noise = np.random.normal(loc=0, scale=sigma, size=d)
        # 2. 截断判断（L2范数≤R则接受）
        noise_norm = np.linalg.norm(noise)
        if noise_norm <= R + 1e-8:  # 加1e-8避免浮点误差
            return noise
    # 兜底机制：若重试上限仍未满足，返回范数=R的噪声（避免无限循环）
    warnings.warn(f"⚠️ 重采样{max_retries}次未达标，返回范数=R的噪声（近似条件分布）")
    noise = np.random.normal(loc=0, scale=sigma, size=d)
    noise = noise / np.linalg.norm(noise) * R  # 归一化到范数=R
    return noise


def generate_samples_with_highdim_cutoff(
        class_centers: np.ndarray,  # 类中心(k, d)
        n_per_class: int = 50,  # 每类样本数
        sigma: float = 0.1,  # 噪声标准差
        confidence: float = 0.99,  # 截断覆盖的置信度
        max_noise_retries: int = 1000  # 噪声重采样上限
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成带高维合理截断噪声的样本，返回(X, y)
    核心：噪声是高斯在球内的条件分布，而非简单3σ截断
    """
    k, d = class_centers.shape
    X, y = [], []

    # 1. 计算高维高斯的合理截断半径R
    R = compute_highdim_gaussian_cutoff(d=d, sigma=sigma, confidence=confidence)

    # 2. 为每类生成样本（带条件分布噪声）
    for class_idx in range(k):
        class_samples = []
        # 记录噪声范数统计（用于验证分布）
        noise_norms = []
        while len(class_samples) < n_per_class:
            # 生成符合条件分布的噪声
            noise = generate_highdim_gaussian_noise(
                d=d, sigma=sigma, R=R, max_retries=max_noise_retries
            )
            # 生成样本（中心+噪声）
            sample = class_centers[class_idx] + noise
            class_samples.append(sample)
            # 记录噪声范数（后续验证）
            noise_norms.append(np.linalg.norm(noise))

        # 3. 验证当前类的噪声分布（打印统计信息）
        noise_norms = np.array(noise_norms)
        print(f"📈 第{class_idx + 1}类噪声范数统计（条件分布）：")
        print(f"   - 均值：{noise_norms.mean():.4f} | 理论均值：{R * (d / (d + 2)):.4f}")  # 条件分布均值近似
        print(f"   - 标准差：{noise_norms.std():.4f}")
        print(f"   - 最大范数：{noise_norms.max():.4f} ≤ R={R:.4f}（截断有效）\n")

        X.append(np.array(class_samples))
        y.extend([class_idx] * n_per_class)

    # 格式转换
    X = np.vstack(X)
    y = np.array(y)
    return X, y


# -------------------------- 整合到原有数据生成流程 --------------------------
def generate_standard_gaussian_data(
        k: int = 10, d: int = 25, n_per_class: int = 50, sigma: float = 0.1,
        max_center_retries: int = 100, seed: int = 42, regenerate: bool = True,
        save_dir: str = "./experiment_data/"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """
    改进后的数据生成函数：用高维条件分布噪声替换原3σ截断
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "data_with_highdim_noise.json")

    # 加载已有数据（略，同原逻辑）
    if not regenerate and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        X = np.array(data_dict["X"])
        y = np.array(data_dict["y"])
        class_centers = np.array(data_dict["class_centers"])
        data_params = data_dict["data_params"]
        margin_dict = data_dict["max_margin"]
        return X, y, class_centers, data_params, margin_dict

    # 生成类中心（略，同原逻辑：标准高斯+距离验证）
    np.random.seed(seed)
    retry_count = 0
    class_centers = None
    min_dist_threshold = 2 * (compute_highdim_gaussian_cutoff(d=d, sigma=sigma, confidence=0.99)) + 0.5
    while retry_count < max_center_retries:
        candidate_centers = np.random.normal(0, 1, (k, d))
        current_min_dist = compute_min_center_distance(candidate_centers)
        if current_min_dist >= min_dist_threshold:
            class_centers = candidate_centers
            print(f"✅ 类中心生成：最小距离={current_min_dist:.4f}≥{min_dist_threshold:.4f}\n")
            break
        retry_count += 1
    if class_centers is None:
        candidate_centers = np.random.normal(0, 1, (k, d))
        current_min_dist = compute_min_center_distance(candidate_centers)
        scale_factor = min_dist_threshold / current_min_dist
        class_centers = candidate_centers * scale_factor
        print(f"⚠️ 类中心手动缩放：系数={scale_factor:.2f}\n")

    # -------------------------- 关键改进：用高维条件分布噪声生成样本 --------------------------
    X, y = generate_samples_with_highdim_cutoff(
        class_centers=class_centers,
        n_per_class=n_per_class,
        sigma=sigma,
        confidence=0.99,  # 覆盖99%的高斯样本
        max_noise_retries=1000
    )

    # 后续计算max margin、保存JSON/CSV（同原逻辑，略）
    margin_dict = compute_multinorm_max_margin(X, y)
    sample_df = pd.DataFrame(X, columns=[f"feat_{i + 1}" for i in range(d)])
    sample_df["label"] = y
    sample_df.to_csv(os.path.join(save_dir, "samples_highdim.csv"), index=False)

    data_params = {
        "k": k, "d": d, "n_per_class": n_per_class, "sigma": sigma,
        "seed": seed, "cutoff_R": compute_highdim_gaussian_cutoff(d=d, sigma=sigma, confidence=0.99),
        "center_min_dist": compute_min_center_distance(class_centers), "n_total": len(X)
    }
    data_dict = {
        "X": X.tolist(), "y": y.tolist(), "class_centers": class_centers.tolist(),
        "data_params": data_params, "max_margin": margin_dict
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
    print(f"💾 改进后数据保存至：{json_path}")

    return X, y, class_centers, data_params, margin_dict


# -------------------------- 辅助函数（同原逻辑，需保留） --------------------------
def compute_min_center_distance(class_centers: np.ndarray) -> float:
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(class_centers, metric="euclidean"))
    return round(np.min(dist_matrix[np.nonzero(dist_matrix)]), 4)




import cvxpy as cp
import numpy as np
import time


def compute_multinorm_max_margin(X: np.ndarray, y: np.ndarray) -> dict:
    """
    求解不同范数约束下的最大间隔 (Max Margin) 问题。
    - L2/Linf 使用 ECOS 求解器。
    - 谱范数 (Spectral Norm) 切换为 CVXOPT 求解器。
    - 求解失败时不回退，直接报错。
    """
    n, d = X.shape
    k = len(np.unique(y))
    margin_dict = {}

    # 范数配置：加入谱范数
    norm_configs = [
        ("L2_norm", lambda W: cp.norm(W, "fro") <= 1),
        ("Linf_norm", lambda W: cp.norm(W, "inf") <= 1),
        ("spectral_norm", lambda W: cp.norm(W, 2) <= 1)
    ]

    for norm_name, constraint_fn in norm_configs:
        print(f"\n" + "=" * 80)
        print(f"📌 求解 {norm_name} 的 max margin")
        print("=" * 80)

        # 1. 构建优化问题 (保持不变)
        W = cp.Variable((k, d), name="weight_matrix")
        margin_exprs = []
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            y_i = y[i]
            for c in range(k):
                if c != y_i:
                    e_diff = np.zeros((k, 1))
                    e_diff[y_i] = 1
                    e_diff[c] = -1
                    margin = cp.matmul(e_diff.T, cp.matmul(W, x_i))
                    margin_exprs.append(margin)

        if not margin_exprs:
            raise RuntimeError(f"❌ {norm_name}：无有效间隔表达式")
        margins_vec = cp.vstack(margin_exprs)
        min_margin = cp.min(margins_vec)
        objective = cp.Maximize(min_margin)
        constraints = [constraint_fn(W)]

        # 2. 动态求解器和参数选择
        if norm_name == "spectral_norm":
            solver = cp.CVXOPT
            solver_name = "CVXOPT (for SDP)"
            solver_opts = {"verbose": True, "max_iters": 1000, "abstol": 1e-4,
                "reltol": 1e-4,
                "feastol": 1e-4}
        else:
            solver = cp.ECOS
            solver_name = "ECOS"
            solver_opts = {"verbose": True, "max_iters": 1000, "abstol": 1e-9, "reltol": 1e-9}

        # 3. 调用求解器
        print(f"🚀 开始求解：{norm_name}，求解器：{solver_name}")
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=solver, **solver_opts)
        except cp.error.SolverError as e:
            # 不回退，直接报错
            raise RuntimeError(f"❌ {norm_name} 求解失败！请确保已安装 {solver_name}。原始错误：{e}")

        # 4. 结果处理 (使用 np.round 修正类型错误)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            final_gamma_value = min_margin.value
            final_W_value = W.value

            # 确保最终 gamma 是 float 或 list
            if final_gamma_value is not None:
                final_gamma = np.round(final_gamma_value, 6)
                final_gamma = final_gamma.item() if np.isscalar(final_gamma) else final_gamma.tolist()
            else:
                final_gamma = 0.0

            # 确保最终 W 是 list
            if final_W_value is not None:
                final_W_np = np.round(final_W_value, 6)  # 先转为 numpy
                final_W = final_W_np.tolist()
            else:
                final_W_np = np.zeros((k, d))
                final_W = final_W_np.tolist()

            margin_dict[norm_name] = {
                "gamma": final_gamma,
                "matrix": final_W
            }

            # 恢复原有打印逻辑 (打印 W 的前 3 行预览)
            print(f"\n✅ {norm_name}求解成功！")
            print(f"   - 求解状态：{prob.status}")
            print(f"   - 最终gamma（最小间隔）：{final_gamma}")
            print(f"   - 权重矩阵W：{final_W}")

        else:
            raise RuntimeError(f"❌ {norm_name}未收敛！状态：{prob.status}，建议调整求解器参数。")

    return margin_dict
# def compute_multinorm_max_margin(X: np.ndarray, y: np.ndarray) -> dict:
#     """
#     兼容版求解器：切换至 ECOS 求解器，避免 SCS 的底层解析错误。
#     """
#     # ... (前略：数据检查、k, d, margin_dict 初始化不变) ...
#
#     n, d = X.shape
#     k = len(np.unique(y))
#     margin_dict = {}
#
#     # 范数配置（不变，我们先尝试更换求解器来解决问题）
#     norm_configs = [
#         ("L2_norm", lambda W: cp.norm(W, "fro") <= 1),
#         ("Linf_norm", lambda W: cp.norm(W, "inf") <= 1),
#         # ("spectral_norm", lambda W: cp.norm(W, 2) <= 1)
#     ]
#
#     for norm_name, constraint_fn in norm_configs:
#         print(f"\n" + "=" * 80)
#         print(f"📌 尝试使用 ECOS 求解器求解 {norm_name} 的 max margin")
#         print("=" * 80)
#
#         # 1. 构建优化问题（略，与原代码完全相同，不变）
#         W = cp.Variable((k, d), name="weight_matrix")
#         margin_exprs = []
#         for i in range(n):
#             x_i = X[i].reshape(-1, 1)
#             y_i = y[i]
#             for c in range(k):
#                 if c != y_i:
#                     e_diff = np.zeros((k, 1))
#                     e_diff[y_i] = 1
#                     e_diff[c] = -1
#                     margin = cp.matmul(e_diff.T, cp.matmul(W, x_i))
#                     margin_exprs.append(margin)
#
#         if not margin_exprs:
#             raise RuntimeError(f"❌ {norm_name}：无有效间隔表达式")
#         margins_vec = cp.vstack(margin_exprs)
#         min_margin = cp.min(margins_vec)
#         objective = cp.Maximize(min_margin)
#         constraints = [constraint_fn(W)]
#
#         # 2. 求解器参数（为 ECOS 优化）
#         solver_opts = {
#             "verbose": True,  # 打开日志
#             "max_iters": 1000,  # ECOS 使用 max_iters
#             "abstol": 1e-9,  # 绝对精度
#             "reltol": 1e-9  # 相对精度
#         }
#
#         # 3. 核心修改：调用 ECOS 求解
#         print(f"🚀 开始求解：{norm_name}，求解器：ECOS")
#         prob = cp.Problem(objective, constraints)
#
#         # ⚠️ 将 solver=cp.SCS 替换为 solver=cp.ECOS
#         prob.solve(
#             solver=cp.ECOS,
#             **solver_opts
#         )
#
#         # 4. 结果处理（略，与原代码完全相同，不变）
#         if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#             final_gamma = np.round(min_margin.value, 6) if min_margin.value is not None else 0.0
#             final_W = np.round(W.value, 6) if W.value is not None else np.zeros((k, d))
#             margin_dict[norm_name] = {
#                 "gamma": final_gamma.tolist(),
#                 "matrix": final_W.tolist()
#             }
#             print(f"\n✅ {norm_name}求解成功！")
#             print(f"   - 求解状态：{prob.status}")
#             print(f"   - 最终gamma（最小间隔）：{final_gamma}")
#             # ... (打印矩阵预览) ...
#             print(f"   - 最终矩阵（最小间隔）：{final_W}")
#         else:
#             raise RuntimeError(f"❌ {norm_name}未收敛！状态：{prob.status}，建议增大max_iters")
#
#     return margin_dict


# -------------------------- 测试改进效果 --------------------------
if __name__ == "__main__":
    SEED=45
    # 生成改进后的数据（d=25，σ=0.1，高维条件分布噪声）
    X, y, centers, params, margin = generate_standard_gaussian_data(
        k=10, d=25, n_per_class=50, sigma=0.1, regenerate=True, seed=SEED
    )
    print(f"\n✅ 数据生成完成：{len(X)}样本，噪声为高维高斯条件分布（截断半径R={params['cutoff_R']:.4f}）")