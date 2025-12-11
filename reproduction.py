import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import cvxpy as cp
import seaborn as sns
import json  # <-- 新增导入
import os  # <-- 新增导入

# --- 数据和参数初始化 (用于 JSON 记录) ---
# k, d, n_per_class, n, sigma, seed 在下方定义
class_centers = None
data_params = {}
max_margin_results = {}
# ------------------------------------------

# data generation
# Set random seed for reproducibility
np.random.seed(12344)

# Parameters
k = 15  # number of classes
d = 25  # dimensions
n_per_class = 50  # samples per class
n = k * n_per_class  # total samples
sigma = 0.1  # noise variance
# sigma = 0; # for debuging

# Generate class means that are well-separated
means = np.random.randn(k, d)
class_centers = means  # 记录类中心，用于 JSON
# means = np.eye(k,d) # for debuging

# Generate data with small variance to ensure separability
X = np.zeros((d, n))  # d x n matrix
y = np.zeros(n, dtype=int)

# Generate the data
for i in range(k):
    start_idx = i * n_per_class
    end_idx = (i + 1) * n_per_class
    X[:, start_idx:end_idx] = means[i].reshape(-1, 1) + sigma * np.random.randn(d, n_per_class)
    y[start_idx:end_idx] = i

# L-infinity norm problem
print('Checking separability using L-infinity norm problem...')
V = cp.Variable((k, d))
constraints_inf = []  # 使用 constraints_inf 替代 constraints

for i in range(n):
    yi = y[i]
    xi = X[:, i]
    for j in range(k):
        if j != yi:
            # score_yi - score_j >= 1
            constraints_inf.append(V[yi, :] @ xi - V[j, :] @ xi >= 1)

# Solve L-infinity problem
vec_V = cp.vec(V)  # This reshapes V to a (k*d,) vector
prob_inf = cp.Problem(cp.Minimize(cp.norm(vec_V, 'inf')), constraints_inf)
result_inf = prob_inf.solve()

if prob_inf.status == 'optimal':
    print('L-infinity problem is feasible - data is separable!')

    # Store the solution
    Vinf = V.value
    Vinf_norm = Vinf / np.max(np.abs(Vinf))

    # Compute margins for Vinf_norm
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

    # --- 记录 L-inf 结果 ---
    max_margin_results["Linf_norm"] = {
        "gamma": round(gamma_inf, 6),
        "matrix": np.round(Vinf_norm, 6).tolist(),
        "optimal_value": result_inf,
        "status": prob_inf.status
    }
    # -----------------------
else:
    raise ValueError('Data is not linearly separable! L-infinity problem is infeasible.')

# Standard multiclass SVM (Frobenius norm)
print('\nSolving standard multiclass SVM (Frobenius norm)...')
Vmm = cp.Variable((k, d))
constraints_frob = []  # 使用 constraints_frob 替代 constraints

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

    # Store and normalize solution
    Vmm_val = Vmm.value
    frob_norm = np.linalg.norm(Vmm_val, 'fro')
    Vmm_norm = Vmm_val / frob_norm

    # Compute margins for standard SVM
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

    # --- 记录 Frobenius 结果 ---
    max_margin_results["L2_norm"] = {
        "gamma": round(gamma_frob, 6),
        "matrix": np.round(Vmm_norm, 6).tolist(),
        "optimal_value": result_frob,
        "status": prob_frob.status
    }
    # --------------------------
else:
    raise ValueError('Data is not linearly separable! Standard SVM problem is infeasible.')

Vmm = Vmm_val  # 重新赋值，以保证后续 rank 和 svd 使用正确变量

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

    # Store and normalize solution
    V_nuclear_val = V_nuclear.value
    nuclear_norm = np.linalg.norm(V_nuclear_val, 'nuc')  # Sum of singular values
    V_nuclear_norm = V_nuclear_val / nuclear_norm

    # Compute margins for nuclear norm SVM
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

    # --- 记录 Nuclear 结果 ---
    max_margin_results["nuclear_norm"] = {
        "gamma": round(gamma_nuclear, 6),
        "matrix": np.round(V_nuclear_norm, 6).tolist(),
        "optimal_value": result_nuclear,
        "status": prob_nuclear.status
    }
    # -------------------------
else:
    # 如果求解失败，记录状态，然后报错退出 (遵循原代码逻辑)
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

# Solve Spectral norm problem (operator norm = largest singular value)
prob_spectral = cp.Problem(cp.Minimize(cp.norm(V_spectral, 2)), constraints_spectral)
result_spectral = prob_spectral.solve()

if prob_spectral.status == 'optimal':
    print('Spectral norm SVM problem is feasible!')

    # Store and normalize solution
    V_spectral_val = V_spectral.value
    spectral_norm = np.linalg.norm(V_spectral_val, 2)  # Largest singular value
    V_spectral_norm = V_spectral_val / spectral_norm

    # Compute margins for spectral norm SVM
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

    # --- 记录 Spectral 结果 ---
    max_margin_results["spectral_norm"] = {
        "gamma": round(gamma_spectral, 6),
        "matrix": np.round(V_spectral_norm, 6).tolist(),
        "optimal_value": result_spectral,
        "status": prob_spectral.status
    }
    # --------------------------
else:
    # 如果求解失败，记录状态，然后报错退出 (遵循原代码逻辑)
    max_margin_results["spectral_norm"] = {"gamma": None, "matrix": None, "status": prob_spectral.status}
    raise ValueError(
        f'Data is not linearly separable! Spectral norm SVM problem is infeasible. Status: {prob_spectral.status}')

# Verification and Comparison
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

# Analysis of solutions
print('\nSolution Analysis:')
print('Rank of solutions:')
print(f'Rank of Vinf: {np.linalg.matrix_rank(Vinf)}')
print(f'Rank of Vmm: {np.linalg.matrix_rank(Vmm_val)}')
print(f'Rank of V_nuclear: {np.linalg.matrix_rank(V_nuclear_val)}')
print(f'Rank of V_spectral: {np.linalg.matrix_rank(V_spectral_val)}')

# Singular value analysis
print('\nSingular value distribution:')
_, s_inf, _ = np.linalg.svd(Vinf)
_, s_frob, _ = np.linalg.svd(Vmm_val)
_, s_nuclear, _ = np.linalg.svd(V_nuclear_val)
_, s_spectral, _ = np.linalg.svd(V_spectral_val)

print(f'Singular values of Vinf: {s_inf}')
print(f'Singular values of Vmm: {s_frob}')
print(f'Singular values of V_nuclear: {s_nuclear}')
print(f'Singular values of V_spectral: {s_spectral}')

# -------------------------- 最终保存结果到 JSON 文件 (匹配目标格式) --------------------------
# 1. 整理数据参数
# 简单填充 data_params，因为它依赖的函数在原始代码中不存在
data_params = {
    "k": k, "d": d, "n_per_class": n_per_class, "sigma": sigma,
    "seed": 1, "n_total": n,
    # 填充原始代码片段中存在的字段，但无法计算的值
    "cutoff_R": "N/A (Requires chi2 function)",
    "center_min_dist": "N/A (Requires pdist function)",
}

# 2. 准备 X 和 y (X 从 D x N 转置为 N x D)
X_samples_features = X.T

# 3. 聚合所有结果到目标格式
final_output = {
    "X": X_samples_features.tolist(),  # N x D 样本矩阵
    "y": y.tolist(),  # N 标签向量
    "class_centers": class_centers.tolist(),  # K x D 类中心
    "data_params": data_params,
    # 仅保留 gamma 和 matrix，与 compute_multinorm_max_margin 的输出格式一致
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

# Create a list of weight matrices to visualize
def compute_correlation(A, B):
    """Compute the normalized Frobenius inner product between two matrices."""
    return np.trace(A.T @ B) / (np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro'))

def compute_correlation_matrix():
    """
    Compute and visualize the correlation matrix for the four
    max-margin solutions (Inf norm, Frobenius norm, Nuclear norm, Spectral norm).
    """
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

# Run the analysis
corr_matrix, labels = compute_correlation_matrix()

# Additional Analysis: Visualize the weight matrices
plt.figure(figsize=(15, 4))

# Create a list of weight matrices to visualize
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
plt.figure(figsize=(12, 6))

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

# Print key metrics about each solution
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

def compute_and_store_metrics(W_clean, idx, X, y, device,
                             Vinf_norm_torch, Vmm_torch, V_nuclear_norm_torch, V_spectral_norm_torch,
                             vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations,
                             relative_margin_inf, relative_margin_mm, relative_margin_nuclear, relative_margin_spectral,
                             reference_margins):
    """Helper function to compute and store metrics for the current weights."""
    # L-infinity normalization
    W_max_abs = torch.max(torch.abs(W_clean))
    if W_max_abs > 0:
        W_inf_norm = W_clean / W_max_abs
        vinf_corr = compute_correlation_torch(W_inf_norm, Vinf_norm_torch)
        vinf_correlations[idx] = vinf_corr

        # Move to CPU for margin computation since it uses numpy
        W_inf_np = W_inf_norm.cpu().numpy()
        margin_inf = compute_margin(W_inf_np, X, y)
        # Calculate relative margin difference
        relative_margin_inf[idx] = abs(margin_inf - reference_margins['inf']) / reference_margins['inf']

    # Frobenius normalization
    W_frob = torch.norm(W_clean, 'fro')
    if W_frob > 0:
        W_frob_norm = W_clean / W_frob
        vmm_corr = compute_correlation_torch(W_frob_norm, Vmm_torch)
        vmm_correlations[idx] = vmm_corr

        # Move to CPU for margin computation
        W_frob_np = W_frob_norm.cpu().numpy()
        margin_frob = compute_margin(W_frob_np, X, y)
        # Calculate relative margin difference
        relative_margin_mm[idx] = abs(margin_frob - reference_margins['frob']) / reference_margins['frob']

    # Nuclear norm normalization
    # We'll compute SVD on GPU for this
    U, S, V = safe_svd(W_clean)
    nuclear_norm = torch.sum(S)
    if nuclear_norm > 0:
        W_nuc_norm = W_clean / nuclear_norm
        vnuc_corr = compute_correlation_torch(W_nuc_norm, V_nuclear_norm_torch)
        vnuc_correlations[idx] = vnuc_corr

        # Move to CPU for margin computation
        W_nuc_np = W_nuc_norm.cpu().numpy()
        margin_nuclear = compute_margin(W_nuc_np, X, y)
        # Calculate relative margin difference
        relative_margin_nuclear[idx] = abs(margin_nuclear - reference_margins['nuclear']) / reference_margins['nuclear']

    # Spectral norm normalization (largest singular value)
    spectral_norm = S[0] if S.numel() > 0 else 0
    if spectral_norm > 0:
        W_spec_norm = W_clean / spectral_norm
        vspec_corr = compute_correlation_torch(W_spec_norm, V_spectral_norm_torch)
        vspec_correlations[idx] = vspec_corr

        # Move to CPU for margin computation
        W_spec_np = W_spec_norm.cpu().numpy()
        margin_spectral = compute_margin(W_spec_np, X, y)
        # Calculate relative margin difference
        relative_margin_spectral[idx] = abs(margin_spectral - reference_margins['spectral']) / reference_margins['spectral']

def safe_svd(matrix):
    """Safely compute SVD with checks for NaN/Inf values."""
    # Check for NaN/Inf
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        # Create a clean copy for SVD
        matrix_clean = matrix.clone()
        matrix_clean[torch.isnan(matrix_clean) | torch.isinf(matrix_clean)] = 0.0
        return torch.svd(matrix_clean, some=True)
    else:
        return torch.svd(matrix, some=True)


def power_iteration(matrix, num_iterations=5):
    """
    Approximates the top singular vectors of a matrix using power iteration.
    Returns u_1, v_1 (top left and right singular vectors)
    """
    # Get matrix dimensions
    m, n = matrix.shape

    # Initialize random vector
    v = torch.randn(n, 1, device=matrix.device)
    v = v / (torch.norm(v) + 1e-8)  # Add small constant for stability

    # Power iteration
    for _ in range(num_iterations):
        # v -> u
        u = matrix @ v
        u_norm = torch.norm(u)
        if u_norm > 1e-8:  # Prevent division by zero
            u = u / u_norm
        else:
            # Reinitialize if u becomes too small
            u = torch.randn(m, 1, device=matrix.device)
            u = u / torch.norm(u)

        # u -> v
        v = matrix.t() @ u
        v_norm = torch.norm(v)
        if v_norm > 1e-8:  # Prevent division by zero
            v = v / v_norm
        else:
            # Reinitialize if v becomes too small
            v = torch.randn(n, 1, device=matrix.device)
            v = v / torch.norm(v)

    return u, v

def approximate_svd(matrix, num_iterations=5, rank=None):
    """
    Approximates the SVD of a matrix using power iteration and deflation.
    If rank is None, computes full-rank approximation.
    """
    # Get matrix dimensions
    m, n = matrix.shape
    max_rank = min(m, n)

    if rank is None:
        rank = max_rank

    rank = min(rank, max_rank)

    # Initialize result matrices
    U = torch.zeros((m, rank), device=matrix.device)
    V = torch.zeros((n, rank), device=matrix.device)

    # Copy of the matrix to deflate
    A = matrix.clone()

    for i in range(rank):
        # Get top singular vectors via power iteration
        u, v = power_iteration(A, num_iterations)

        # Store in result matrices
        U[:, i:i+1] = u
        V[:, i:i+1] = v

        # Deflate the matrix - remove the component we've captured
        A = A - u @ v.t()

        # Check if matrix has become too small (near zero)
        if torch.norm(A, 'fro') < 1e-8:
            break

    return U, V

def compute_correlation_torch(A, B):
    """Compute correlation between two matrices using PyTorch (for GPU acceleration)."""
    # Normalized Frobenius inner product using trace
    inner_product = torch.trace(A.t() @ B)
    norm_A = torch.norm(A, 'fro')
    norm_B = torch.norm(B, 'fro')

    # Check for zero norms
    if norm_A < 1e-8 or norm_B < 1e-8:
        return 0.0

    return (inner_product / (norm_A * norm_B)).item()

def compute_margin(W, X, y):
    """Compute margin for a weight matrix (using numpy)."""
    margin = float('inf')
    for i in range(len(y)):
        yi = y[i]
        xi = X[:,i]
        correct_score = W[yi,:] @ xi
        for j in range(k):
            if j != yi:
                margin = min(margin, float(correct_score - W[j,:] @ xi))
    return margin

def compute_correlation_matrix():
    """
    Compute and visualize the correlation matrix for the four
    max-margin solutions (Inf norm, Frobenius norm, Nuclear norm, Spectral norm).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

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
            corr_matrix[i, j] = np.trace(sol_i.T @ sol_j) / (np.linalg.norm(sol_i, 'fro') * np.linalg.norm(sol_j, 'fro'))

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


def newtonschulz5(G, steps=5, eps=1e-7):
    #print(G.ndim)
    with torch.no_grad():
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        G /= (G.norm() + eps)
        if G.size(0) > G.size(1):
            G = G.T
        for _ in range(steps):
            A = G @ G.T
            B = b * A + c * A @ A
            G = a * G + B @ G
        if G.size(0) > G.size(1):
            G = G.T
    return G
def train_classifier(opt_type='gd', mom_type="heavy", init_type='random', lr=0.1, alpha=0, beta1=0.9, beta2=0.9,
                     n_steps=100000,
                     svd_freq=1, power_iter=5, low_rank_approx=False, steps_mu=5):
    """
    opt_type: 'gd', 'normalized_gd', 'sign_gd', 'adam', 'spectral_descent', 'nuclear_descent'
    init_type: 'random' or 'zero'
    lr: base learning rate
    alpha: step size decay power (step_size = lr * t^alpha)
    beta1, beta2: Adam parameters
    svd_freq: frequency of SVD computations (1 = every step, 10 = every 10 steps)
    power_iter: number of power iterations for SVD approximation
    low_rank_approx: whether to use low-rank approximation for SVD
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data tensors - move to GPU if available
    X_torch = torch.FloatTensor(X.T).to(device)  # n x d
    y_torch = torch.LongTensor(y).to(device)

    # Initialize W on device
    if init_type == 'random':
        W = torch.nn.Parameter(torch.randn(k, d, device=device) / np.sqrt(d))
    else:  # zero initialization
        W = torch.nn.Parameter(torch.zeros(k, d, device=device))

    # Setup optimizer for Adam
    if opt_type == 'adam':
        optimizer = torch.optim.Adam([W], lr=lr, betas=(beta1, beta2))

    # Arrays to store metrics
    log_freq = 1  # Changed to 5000 as requested
    n_logs = (n_steps // log_freq) + 2  # +2 for initial and final points

    steps = np.zeros(n_logs, dtype=int)
    ce_losses = np.zeros(n_logs)
    vinf_correlations = np.zeros(n_logs)
    vmm_correlations = np.zeros(n_logs)
    vnuc_correlations = np.zeros(n_logs)
    vspec_correlations = np.zeros(n_logs)

    relative_margin_inf = np.zeros(n_logs)
    relative_margin_mm = np.zeros(n_logs)
    relative_margin_nuclear = np.zeros(n_logs)
    relative_margin_spectral = np.zeros(n_logs)

    # Reference optimal margins
    reference_margins = {
        'inf': gamma_inf,
        'frob': gamma_frob,
        'nuclear': gamma_nuclear,
        'spectral': gamma_spectral
    }

    # For timing information
    total_time = 0
    svd_time = 0

    # Store the SVD factors for reuse
    u_top = None
    v_top = None
    u_full = None
    v_full = None

    # Transfer optimal classifiers to device for correlation computations
    Vinf_norm_torch = torch.FloatTensor(Vinf_norm).to(device)
    Vmm_torch = torch.FloatTensor(Vmm).to(device)
    V_nuclear_norm_torch = torch.FloatTensor(V_nuclear_norm).to(device)
    V_spectral_norm_torch = torch.FloatTensor(V_spectral_norm).to(device)

    # Print initial metrics
    with torch.no_grad():
        logits = (W @ X_torch.T).T
        loss = F.cross_entropy(logits, y_torch)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_torch).float().mean()
        print(f"Step 0 - Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    # Log initial state
    log_idx = 0
    steps[log_idx] = 0
    ce_losses[log_idx] = loss.item()

    # Compute metrics for initial state
    with torch.no_grad():
        compute_and_store_metrics(W.data, log_idx, X, y, device,
                                  Vinf_norm_torch, Vmm_torch, V_nuclear_norm_torch, V_spectral_norm_torch,
                                  vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations,
                                  relative_margin_inf, relative_margin_mm, relative_margin_nuclear,
                                  relative_margin_spectral,
                                  reference_margins)
    log_idx += 1

    # if opt_type == 'adam':
    m = torch.zeros_like(W)
    m_ortho = torch.zeros_like(W)

    L = torch.zeros(k, k).to(device)
    R = torch.zeros(d, d).to(device)

    # Training loop
    for step in range(1, n_steps + 1):
        start_time = time()

        # Compute current step size
        # current_lr = lr * (step ** alpha) if opt_type != 'adam' else lr
        current_lr = lr * (step ** alpha)

        # Forward pass
        logits = W @ X_torch.T
        logits = logits.T
        loss = F.cross_entropy(logits, y_torch)

        # Backward pass
        #         if opt_type == 'adam':
        #             optimizer.zero_grad()
        loss.backward()

        # Update step
        with torch.no_grad():

            if mom_type == "heavy":
                m = beta1 * m + (1 - beta1) * W.grad
            elif mom_type == "nesterov":
                m = beta1 * m + W.grad

            if opt_type == 'mom_ngd':
                mom_norm = torch.norm(m, 'fro')
                if mom_norm > 0:
                    W.data -= current_lr * m / mom_norm
                W.grad.zero_()

            elif opt_type == "ortho":
                L_grad, S_grad, Vh_grad = torch.linalg.svd(W.grad, full_matrices=False)
                m_ortho = beta1 * m_ortho + (1 - beta1) * (L_grad @ Vh_grad)
                W.data -= current_lr * m_ortho
                W.grad.zero_()

            elif opt_type == 'muon':
                delta = newtonschulz5(m, steps=steps_mu)
                W.data -= current_lr * delta
                W.grad.zero_()

            elif opt_type == "shampoo":
                L = beta1 * L + (1 - beta1) * W.grad @ W.grad.T
                R = beta2 * R + (1 - beta2) * W.grad.T @ W.grad
                # L = L + W.grad @ W.grad.T
                # R = R + W.grad.T @ W.grad
                L_U, L_S, L_Vh = torch.linalg.svd(L)
                R_U, R_S, R_Vh = torch.linalg.svd(R)

                L_S_inv_1_4 = torch.pow(L_S, -1. / 4)
                R_S_inv_1_4 = torch.pow(R_S, -1. / 4)

                L_inv_1_4 = L_U @ torch.diag(L_S_inv_1_4) @ L_Vh
                R_inv_1_4 = R_U @ torch.diag(R_S_inv_1_4) @ R_Vh

                W.data -= current_lr * (L_inv_1_4 @ W.grad @ R_inv_1_4)
                W.grad.zero_()

            elif opt_type == "oneside_shampoo":
                L = beta1 * L + (1 - beta1) * W.grad @ W.grad.T
                L_U, L_S, L_Vh = torch.linalg.svd(L)

                L_S_inv_1_2 = torch.pow(L_S, -1. / 2)
                L_inv_1_2 = L_U @ torch.diag(L_S_inv_1_2) @ L_Vh

                W.data -= current_lr * (L_inv_1_2 @ W.grad)
                W.grad.zero_()

            elif opt_type == 'signnum':
                W.data -= current_lr * torch.sign(m)
                W.grad.zero_()

            elif opt_type == 'spectral_descent' or opt_type == 'nuclear_descent':
                # Determine if we need to recompute SVD
                update_svd = (step % svd_freq == 0) or (u_full is None and u_top is None)

                if update_svd:
                    svd_start = time()

                    # Clean gradient if needed
                    m[torch.isnan(m) | torch.isinf(m)] = 0.0

                    if low_rank_approx:
                        # Use power iteration method to approximate top singular vectors
                        if opt_type == 'nuclear_descent':
                            # For nuclear norm, we only need the top singular vectors
                            u_top, v_top = power_iteration(m, power_iter)
                        else:
                            # For spectral norm, we need the full SVD
                            u_full, v_full = approximate_svd(m, power_iter)
                    else:
                        # Full SVD calculation with safety checks
                        U, S, V = safe_svd(m)

                        if opt_type == 'nuclear_descent':
                            # Only store the top vectors
                            u_top = U[:, 0:1]
                            v_top = V[:, 0:1]
                        else:
                            u_full = U
                            v_full = V

                    svd_end = time()
                    svd_time += (svd_end - svd_start)

                # Apply the update
                if opt_type == 'spectral_descent':
                    # Normalized steepest descent w.r.t. spectral norm
                    # W_t+1 = W_t - η * UV^T where U,V from SVD of gradient
                    delta = u_full @ v_full.t()  # This is the UV^T direction
                    W.data -= current_lr * delta
                else:  # nuclear_descent
                    # Normalized steepest descent w.r.t. nuclear norm
                    # W_t+1 = W_t - η * u₁v₁^T where u₁,v₁ are top SVD factors
                    delta = u_top @ v_top.t()  # This is u₁v₁^T
                    W.data -= current_lr * delta

                W.grad.zero_()

        end_time = time()
        total_time += (end_time - start_time)

        # Print training progress
        if step % 5000 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_torch).float().mean()
                time_per_iter = total_time / step
                svd_percent = (svd_time / total_time) * 100 if total_time > 0 else 0
                print(f"Step {step:5d} - Loss: {loss.item():.4f}, Acc: {acc:.4f}, lr: {current_lr:.6f}")
                print(f"Time per iter: {time_per_iter * 1000:.2f}ms, SVD time: {svd_percent:.1f}%")

        # Log metrics every log_freq steps
        if step % log_freq == 0:
            steps[log_idx] = step
            ce_losses[log_idx] = loss.item()

            # Compute all metrics on GPU to avoid unnecessary CPU-GPU transfers
            with torch.no_grad():
                # Clean W for computation if needed
                W_clean = W.data.clone()
                W_clean[torch.isnan(W_clean) | torch.isinf(W_clean)] = 0.0

                compute_and_store_metrics(W_clean, log_idx, X, y, device,
                                          Vinf_norm_torch, Vmm_torch, V_nuclear_norm_torch, V_spectral_norm_torch,
                                          vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations,
                                          relative_margin_inf, relative_margin_mm, relative_margin_nuclear,
                                          relative_margin_spectral,
                                          reference_margins)

            log_idx += 1

    # Record the final metrics if we haven't logged at the last step
    if step % log_freq != 0:
        steps[log_idx] = step
        ce_losses[log_idx] = loss.item()

        with torch.no_grad():
            # Clean W for computation if needed
            W_clean = W.data.clone()
            W_clean[torch.isnan(W_clean) | torch.isinf(W_clean)] = 0.0

            compute_and_store_metrics(W_clean, log_idx, X, y, device,
                                      Vinf_norm_torch, Vmm_torch, V_nuclear_norm_torch, V_spectral_norm_torch,
                                      vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations,
                                      relative_margin_inf, relative_margin_mm, relative_margin_nuclear,
                                      relative_margin_spectral,
                                      reference_margins)

        log_idx += 1

    # Trim arrays to actual size
    steps = steps[:log_idx]
    ce_losses = ce_losses[:log_idx]
    vinf_correlations = vinf_correlations[:log_idx]
    vmm_correlations = vmm_correlations[:log_idx]
    vnuc_correlations = vnuc_correlations[:log_idx]
    vspec_correlations = vspec_correlations[:log_idx]
    relative_margin_inf = relative_margin_inf[:log_idx]
    relative_margin_mm = relative_margin_mm[:log_idx]
    relative_margin_nuclear = relative_margin_nuclear[:log_idx]
    relative_margin_spectral = relative_margin_spectral[:log_idx]

    # Move W back to CPU for final outputs
    W_final = W.data.cpu().numpy()

    #     # Create plots
    #     plt.figure(figsize=(20, 10))

    #     plt.subplot(2, 3, 1)
    #     plt.loglog(steps, ce_losses)
    #     plt.title(f'Cross Entropy Loss vs. Iterations\n({opt_type}, {init_type} init, α={alpha})')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Loss (log scale)')
    #     plt.grid(True)

    #     plt.subplot(2, 3, 2)
    #     plt.loglog(steps, 1-vinf_correlations, label='Correlation with Vinf_norm')
    #     plt.loglog(steps, 1-vmm_correlations, label='Correlation with Vmm')
    #     plt.loglog(steps, 1-vnuc_correlations, label='Correlation with V_nuclear_norm')
    #     plt.loglog(steps, 1-vspec_correlations, label='Correlation with V_spectral_norm')
    #     plt.title('Correlation with Optimal Classifiers')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('1-Correlation')
    #     plt.grid(True)
    #     plt.legend()

    #     plt.subplot(2, 3, 3)
    #     plt.loglog(steps, relative_margin_inf, label='|margin - γ_inf|/γ_inf')
    #     plt.loglog(steps, relative_margin_mm, label='|margin - γ_frob|/γ_frob')
    #     plt.loglog(steps, relative_margin_nuclear, label='|margin - γ_nuclear|/γ_nuclear')
    #     plt.loglog(steps, relative_margin_spectral, label='|margin - γ_spectral|/γ_spectral')
    #     plt.title('Relative Margin Differences')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Relative Difference')
    #     plt.grid(True)
    #     plt.legend()

    print("\nFinal metrics:")
    print(f"CE Loss: {ce_losses[-1]:.4f}")
    print(f"Correlations - Vinf: {vinf_correlations[-1]:.4f}, Vmm: {vmm_correlations[-1]:.4f}")
    print(f"Correlations - Vnuc: {vnuc_correlations[-1]:.4f}, Vspec: {vspec_correlations[-1]:.4f}")
    print(f"Relative margins - Inf: {relative_margin_inf[-1]:.4f}, Frob: {relative_margin_mm[-1]:.4f}")
    print(
        f"Relative margins - Nuclear: {relative_margin_nuclear[-1]:.4f}, Spectral: {relative_margin_spectral[-1]:.4f}")

    # Performance metrics
    time_per_iter = total_time / step
    svd_percent = (svd_time / total_time) * 100 if total_time > 0 else 0
    print("\nPerformance metrics:")
    print(f"Total time: {total_time:.2f}s")
    print(f"SVD time: {svd_time:.2f}s ({svd_percent:.1f}%)")
    print(f"Time per iteration: {time_per_iter * 1000:.2f}ms")

    # return W_final, ce_losses, vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations, relative_margin_inf, relative_margin_mm, relative_margin_nuclear, relative_margin_spectral, steps
    correlations = (vinf_correlations, vmm_correlations, vnuc_correlations, vspec_correlations)
    relative_margins = (relative_margin_inf, relative_margin_mm, relative_margin_nuclear, relative_margin_spectral)
    return steps, ce_losses, correlations, relative_margins


# steps_sign, ce_losses_sign, correlations_sign, relative_margins_sign = train_classifier(
#     opt_type='signnum',  # or 'spectral_descent'
#     init_type='zero',
#     lr=0.01,
#     beta1=0.,
#     alpha=-1/2,
#     low_rank_approx=False,
#     n_steps=10000        # This is your maximum steps, but it will stop early if loss < 1e-10
# )
#
# plt.figure(figsize=(8,6))
#
# plt.loglog(steps_sign[1:], relative_margins_sign[0][1:], marker="s", linewidth=3, markersize=10, markevery=100, label=r'$|\gamma_{\bar{W}_t} - \gamma_{\infty}|/\gamma_{\infty}$')
# plt.semilogx(steps_sign[1:], relative_margins_sign[1][1:], marker="o", linewidth=3, markersize=10, markevery=100, label=r'$|\gamma_{\bar{W}_t} - \gamma_{2}|/\gamma_{2}$')
# plt.loglog(steps_sign[1:], relative_margins_sign[2][1:], marker="*", linewidth=3, markersize=10, markevery=100, label=r'$|\gamma_{W_t} - \gamma_{nuc}|/\gamma_{nuc}$')
# plt.semilogx(steps_sign[1:], relative_margins_sign[3][1:], marker="D", linewidth=3, markersize=10, markevery=100, label=r'$|\gamma_{\bar{W}_t} - \gamma_{spec}|/\gamma_{spec}$')
#
# #plt.title('Relative Margin Differences')
# plt.xlabel('Iterations',fontsize=18,weight="bold")
# plt.ylabel('Relative Margin',fontsize=18,weight="bold")
# plt.grid(True)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(loc='best',fontsize=24)
# plt.tight_layout()
# plt.savefig("Figs/rel_margin_sign.pdf")