import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import cvxpy as cp
import seaborn as sns
import json
import os

class_centers = None
data_params = {}
max_margin_results = {}
# 生成数据
np.random.seed(12344)

# 参数
k = 15  # 类数量
d = 25  # 维度
n_per_class = 50  # 每类样本数量
n = k * n_per_class  # 总样本数量
sigma = 0.1  # 噪声
means = np.random.randn(k, d)
class_centers = means  # 记录类中心


X = np.zeros((d, n))
y = np.zeros(n, dtype=int)
for i in range(k):
    start_idx = i * n_per_class
    end_idx = (i + 1) * n_per_class
    X[:, start_idx:end_idx] = means[i].reshape(-1, 1) + sigma * np.random.randn(d, n_per_class)
    y[start_idx:end_idx] = i

# L-infinity norm
print('Checking separability using L-infinity norm problem...')
V = cp.Variable((k, d))
constraints_inf = []

for i in range(n):
    yi = y[i]
    xi = X[:, i]
    for j in range(k):
        if j != yi:
            # score_yi - score_j >= 1
            constraints_inf.append(V[yi, :] @ xi - V[j, :] @ xi >= 1)
vec_V = cp.vec(V)
prob_inf = cp.Problem(cp.Minimize(cp.norm(vec_V, 'inf')), constraints_inf)
result_inf = prob_inf.solve()

if prob_inf.status == 'optimal':
    print('L-infinity problem is feasible - data is separable!')
    Vinf = V.value
    Vinf_norm = Vinf / np.max(np.abs(Vinf))
    gamma_inf = float('inf')
    for i in range(n):
        yi = y[i]
        xi = X[:, i]
        correct_score = Vinf_norm[yi, :] @ xi
        for j in range(k):
            if j != yi:
                margin = correct_score - Vinf_norm[j, :] @ xi
                gamma_inf = min(gamma_inf, margin)

    print(f'L-infinity solution optimal value: {result_inf}')
    print(f'Margin after L-infinity normalization (gamma_inf): {gamma_inf}')
    max_margin_results["Linf_norm"] = {
        "gamma": round(gamma_inf, 6),
        "matrix": np.round(Vinf_norm, 6).tolist(),
        "optimal_value": result_inf,
        "status": prob_inf.status
    }
    # -----------------------
else:
    raise ValueError('Data is not linearly separable! L-infinity problem is infeasible.')

# Frobenius norm svm
print('\nSolving standard multiclass SVM (Frobenius norm)...')
Vmm = cp.Variable((k, d))
constraints_frob = []

for i in range(n):
    yi = y[i]
    xi = X[:, i]
    for j in range(k):
        if j != yi:
            constraints_frob.append(Vmm[yi, :] @ xi - Vmm[j, :] @ xi >= 1)

# Solve Frobenius norm problem
prob_frob = cp.Problem(cp.Minimize(cp.norm(Vmm, 'fro')), constraints_frob)
result_frob = prob_frob.solve()

if prob_frob.status == 'optimal':
    print('Standard SVM problem is feasible!')
    Vmm_val = Vmm.value
    frob_norm = np.linalg.norm(Vmm_val, 'fro')
    Vmm_norm = Vmm_val / frob_norm
    gamma_frob = float('inf')
    for i in range(n):
        yi = y[i]
        xi = X[:, i]
        correct_score = Vmm_norm[yi, :] @ xi
        for j in range(k):
            if j != yi:
                margin = correct_score - Vmm_norm[j, :] @ xi
                gamma_frob = min(gamma_frob, margin)

    print(f'Standard SVM margin (gamma_frob): {gamma_frob}')
    max_margin_results["L2_norm"] = {
        "gamma": round(gamma_frob, 6),
        "matrix": np.round(Vmm_norm, 6).tolist(),
        "optimal_value": result_frob,
        "status": prob_frob.status
    }
    # --------------------------
else:
    raise ValueError('Data is not linearly separable! Standard SVM problem is infeasible.')

Vmm = Vmm_val

# Nuclear norm SVM
print('\nSolving nuclear norm SVM...')
V_nuclear = cp.Variable((k, d))
constraints_nuclear = []

for i in range(n):
    yi = y[i]
    xi = X[:, i]
    for j in range(k):
        if j != yi:
            constraints_nuclear.append(V_nuclear[yi, :] @ xi - V_nuclear[j, :] @ xi >= 1)

# Solve Nuclear norm problem
prob_nuclear = cp.Problem(cp.Minimize(cp.norm(V_nuclear, 'nuc')), constraints_nuclear)
result_nuclear = prob_nuclear.solve()

if prob_nuclear.status == 'optimal':
    print('Nuclear norm SVM problem is feasible!')
    V_nuclear_val = V_nuclear.value
    nuclear_norm = np.linalg.norm(V_nuclear_val, 'nuc')
    V_nuclear_norm = V_nuclear_val / nuclear_norm
    gamma_nuclear = float('inf')
    for i in range(n):
        yi = y[i]
        xi = X[:, i]
        correct_score = V_nuclear_norm[yi, :] @ xi
        for j in range(k):
            if j != yi:
                margin = correct_score - V_nuclear_norm[j, :] @ xi
                gamma_nuclear = min(gamma_nuclear, margin)

    print(f'Nuclear norm solution optimal value: {result_nuclear}')
    print(f'Margin after nuclear norm normalization (gamma_nuclear): {gamma_nuclear}')
    max_margin_results["nuclear_norm"] = {
        "gamma": round(gamma_nuclear, 6),
        "matrix": np.round(V_nuclear_norm, 6).tolist(),
        "optimal_value": result_nuclear,
        "status": prob_nuclear.status
    }
else:
    max_margin_results["nuclear_norm"] = {"gamma": None, "matrix": None, "status": prob_nuclear.status}
    raise ValueError(
        f'Data is not linearly separable! Nuclear norm SVM problem is infeasible. Status: {prob_nuclear.status}')

# Spectral norm SVM
print('\nSolving spectral norm SVM...')
V_spectral = cp.Variable((k, d))
constraints_spectral = []

for i in range(n):
    yi = y[i]
    xi = X[:, i]
    for j in range(k):
        if j != yi:
            constraints_spectral.append(V_spectral[yi, :] @ xi - V_spectral[j, :] @ xi >= 1)
prob_spectral = cp.Problem(cp.Minimize(cp.norm(V_spectral, 2)), constraints_spectral)
result_spectral = prob_spectral.solve()

if prob_spectral.status == 'optimal':
    print('Spectral norm SVM problem is feasible!')
    V_spectral_val = V_spectral.value
    spectral_norm = np.linalg.norm(V_spectral_val, 2)  # Largest singular value
    V_spectral_norm = V_spectral_val / spectral_norm
    gamma_spectral = float('inf')
    for i in range(n):
        yi = y[i]
        xi = X[:, i]
        correct_score = V_spectral_norm[yi, :] @ xi
        for j in range(k):
            if j != yi:
                margin = correct_score - V_spectral_norm[j, :] @ xi
                gamma_spectral = min(gamma_spectral, margin)

    print(f'Spectral norm solution optimal value: {result_spectral}')
    print(f'Margin after spectral norm normalization (gamma_spectral): {gamma_spectral}')

    max_margin_results["spectral_norm"] = {
        "gamma": round(gamma_spectral, 6),
        "matrix": np.round(V_spectral_norm, 6).tolist(),
        "optimal_value": result_spectral,
        "status": prob_spectral.status
    }
    # --------------------------
else:
    max_margin_results["spectral_norm"] = {"gamma": None, "matrix": None, "status": prob_spectral.status}
    raise ValueError(
        f'Data is not linearly separable! Spectral norm SVM problem is infeasible. Status: {prob_spectral.status}')

# 验证解
print('\nVerification and Comparison:')
print(f'Max absolute value of Vinf_norm: {np.max(np.abs(Vinf_norm))}')
print(f'Frobenius norm of Vmm_norm: {np.linalg.norm(Vmm_norm, "fro")}')
print(f'Nuclear norm of V_nuclear_norm: {np.linalg.norm(V_nuclear_norm, "nuc")}')
print(f'Spectral norm of V_spectral_norm: {np.linalg.norm(V_spectral_norm, 2)}')

print('\nMargin Comparison:')
print(f'L-infinity margin (gamma_inf): {gamma_inf}')
print(f'Frobenius norm margin (gamma_frob): {gamma_frob}')
print(f'Nuclear norm margin (gamma_nuclear): {gamma_nuclear}')
print(f'Spectral norm margin (gamma_spectral): {gamma_spectral}')

# 分析解
print('\nSolution Analysis:')
print('Rank of solutions:')
print(f'Rank of Vinf: {np.linalg.matrix_rank(Vinf)}')
print(f'Rank of Vmm: {np.linalg.matrix_rank(Vmm_val)}')
print(f'Rank of V_nuclear: {np.linalg.matrix_rank(V_nuclear_val)}')
print(f'Rank of V_spectral: {np.linalg.matrix_rank(V_spectral_val)}')

# 分析谱
print('\nSingular value distribution:')
_, s_inf, _ = np.linalg.svd(Vinf)
_, s_frob, _ = np.linalg.svd(Vmm_val)
_, s_nuclear, _ = np.linalg.svd(V_nuclear_val)
_, s_spectral, _ = np.linalg.svd(V_spectral_val)

print(f'Singular values of Vinf: {s_inf}')
print(f'Singular values of Vmm: {s_frob}')
print(f'Singular values of V_nuclear: {s_nuclear}')
print(f'Singular values of V_spectral: {s_spectral}')
data_params = {
    "k": k, "d": d, "n_per_class": n_per_class, "sigma": sigma,
    "seed": 1, "n_total": n,
    "cutoff_R": "N/A (Requires chi2 function)",
    "center_min_dist": "N/A (Requires pdist function)",
}

X_samples_features = X.T

# 保存所有结果
final_output = {
    "X": X_samples_features.tolist(),
    "y": y.tolist(),
    "class_centers": class_centers.tolist(),
    "data_params": data_params,
    "max_margin": {
        "Linf_norm": {"gamma": max_margin_results["Linf_norm"]["gamma"],
                      "matrix": max_margin_results["Linf_norm"]["matrix"]},
        "L2_norm": {"gamma": max_margin_results["L2_norm"]["gamma"], "matrix": max_margin_results["L2_norm"]["matrix"]},
        "nuclear_norm": {"gamma": max_margin_results["nuclear_norm"]["gamma"],
                         "matrix": max_margin_results["nuclear_norm"]["matrix"]},
        "spectral_norm": {"gamma": max_margin_results["spectral_norm"]["gamma"],
                          "matrix": max_margin_results["spectral_norm"]["matrix"]},
    }
}

output_filename = "experiment_data/max_margin_results_final_formatnew.json"
with open(output_filename, "w") as f:
    json.dump(final_output, f, indent=4)

print("\n" + "=" * 80)
print(f"💾 所有 Max Margin 求解结果和数据已保存到：{output_filename}")
print(f"数据格式已调整为与目标代码片段一致。")
print("=" * 80)
def compute_correlation(A, B):
    return np.trace(A.T @ B) / (np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro'))

def compute_correlation_matrix():
    # List of all reference solutions with their names
    solutions = [
        ("Inf Norm", Vinf_norm),
        ("Frobenius Norm", Vmm),
        ("Nuclear Norm", V_nuclear_norm),
        ("Spectral Norm", V_spectral_norm)
    ]

    # Initialize correlation matrix
    n = len(solutions)
    corr_matrix = np.zeros((n, n))

    # Compute all pairwise correlations
    for i in range(n):
        for j in range(n):
            name_i, sol_i = solutions[i]
            name_j, sol_j = solutions[j]
            corr_matrix[i, j] = compute_correlation(sol_i, sol_j)

    # Create labels for the matrix
    labels = [name for name, _ in solutions]

    # Print the correlation matrix
    print("Correlation Matrix:")
    for i in range(n):
        row_str = " ".join([f"{corr_matrix[i, j]:.4f}" for j in range(n)])
        print(f"{labels[i]}: {row_str}")

    # Create a heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.title("Correlation Matrix of Different Max-Margin Solutions")
    plt.tight_layout()
    plt.savefig("max_margin_correlation_matrix.png")
    plt.show()

    return corr_matrix, labels

corr_matrix, labels = compute_correlation_matrix()

# Additional Analysis: Visualize the weight matrices
plt.figure(figsize=(15, 4))

weight_matrices = [
    ("Inf Norm", Vinf_norm),
    ("Frobenius Norm", Vmm),
    ("Nuclear Norm", V_nuclear_norm),
    ("Spectral Norm", V_spectral_norm)
]

for i, (name, matrix) in enumerate(weight_matrices):
    plt.subplot(1, 4, i+1)
    plt.imshow(matrix, cmap='coolwarm', vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    plt.colorbar()
    plt.title(f"{name}")
    plt.xlabel("Features")
    plt.ylabel("Classes")

plt.tight_layout()
plt.savefig("max_margin_solution_comparison.png")
plt.show()

# Singular value analysis
plt.figure(figsize=(10, 6))

for name, matrix in weight_matrices:
    # Compute singular values
    _, s, _ = np.linalg.svd(matrix)
    # Plot singular values
    plt.semilogy(range(1, len(s)+1), s, 'o-', label=name)
plt.xlabel('Index')
plt.ylabel('Singular Value (log scale)')
plt.title('Singular Value Spectrum of Different Max-Margin Solutions')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("singular_value_spectrum.png")
plt.show()

# 展示信息
print("\nSolution Properties:")
print("-" * 60)
print(f"{'Solution':<15} {'Rank':<8} {'Nuclear Norm':<15} {'Spectral Norm':<15} {'Frob Norm':<15} {'Max Abs':<10}")
print("-" * 60)

for name, matrix in weight_matrices:
    rank = np.linalg.matrix_rank(matrix)
    nuclear_norm = np.linalg.norm(matrix, 'nuc')
    spectral_norm = np.linalg.norm(matrix, 2)
    frob_norm = np.linalg.norm(matrix, 'fro')
    max_abs = np.max(np.abs(matrix))

    print(f"{name:<15} {rank:<8} {nuclear_norm:<15.4f} {spectral_norm:<15.4f} {frob_norm:<15.4f} {max_abs:<10.4f}")

