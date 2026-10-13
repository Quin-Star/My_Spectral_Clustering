# -*- coding: utf-8 -*-
"""谱聚类模型：全部手写实现，核心算法只依赖 numpy。

算法四步
--------
    ① 构图    ：由数据点之间的相似度构造亲和矩阵 W
    ② 建图论量：由 W 和度矩阵 D 构造拉普拉斯矩阵 L
    ③ 谱嵌入  ：求 L 前 k 个最小特征值对应的特征向量，拼成 n×k 矩阵 U
    ④ 离散化  ：把 U 的每一行当作新样本，做 K-Means 得到簇标签

对外接口对齐 scikit-learn
-------------------------
    from model import SpectralClustering

    model = SpectralClustering(n_clusters=2, affinity="knn", laplacian="sym")
    labels = model.fit_predict(X)        # 拟合并返回簇标签
    labels = model.predict(X_new)        # 给新样本分配簇（近似，见方法文档）

拟合后属性（sklearn 约定：以下划线结尾）
---------------------------------------
    labels_           簇标签，(n,)
    embedding_        谱嵌入 U，(n, k)
    eigenvalues_      前 k 个最小特征值
    eigenvalues_all_  完整特征值谱（仅自动估计簇数时有值，否则为 None）
    affinity_         亲和矩阵 W，(n, n)
    n_clusters_       实际使用的簇数
    inertia_          K-Means 目标函数值（簇内平方和）
    cluster_centers_  嵌入空间中的簇中心，(k, k)
    n_iter_           K-Means 实际迭代步数
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray

AFFINITY_KINDS = ("rbf", "knn", "mutual_knn")       # 允许的构图方式
LAPLACIAN_KINDS = ("none", "sym", "rw")             # 允许的拉普拉斯类型


# 0. 统一入口校验
def check_array(X, name: str = "X") -> Array:
    """把输入统一成二维浮点数组，并做基本合法性检查。"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:                                 # 一维输入按"单特征列向量"处理
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("%s 必须是二维数组，收到 shape=%s" % (name, X.shape))
    if X.shape[0] < 2:
        raise ValueError("%s 至少需要 2 个样本，收到 %d 个" % (name, X.shape[0]))
    if not np.all(np.isfinite(X)):
        raise ValueError("%s 中含有 nan 或 inf，请先清洗数据" % name)
    return X

# -------------------------- ① 构建 亲和矩阵 W --------------------------
# 1. 距离平方矩阵
def squared_distances(X: Array, Y: Optional[Array] = None) -> Array:
    """两个样本矩阵之间的欧氏距离平方，返回 (m, n) 矩阵。

    用展开式 ‖x−y‖² = ‖x‖² + ‖y‖² − 2xᵀy 计算，
    只产生 m×n 的结果矩阵，不会构造 m×n×d 的三维张量。
    Y 为 None 时令 Y = X，得到 m×m 的对称矩阵。
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    x2 = np.einsum("ij,ij->i", X, X)[:, None]       # ‖x‖²，形状 (m, 1)
    y2 = np.einsum("ij,ij->i", Y, Y)[None, :]       # ‖y‖²，形状 (1, n)
    d2 = x2 + y2 - 2.0 * (X @ Y.T)                  # 广播相加得到 (m, n)
    np.maximum(d2, 0.0, out=d2)                     # 浮点误差会造出 −1e-16，截断为 0
    return d2

# 2. 欧式距离矩阵
def pairwise_distances(X: Array, Y: Optional[Array] = None) -> Array:
    """欧氏距离矩阵（对 squared_distances 开根号）。"""
    return np.sqrt(squared_distances(X, Y))

# 3. 亲和距离矩阵
def build_affinity(X: Array, affinity: str = "knn", n_neighbors: int = 10,
                   sigma: Optional[float] = None, local_scaling: bool = True) -> Array:
    """构造对称的亲和矩阵 W（n×n，对角线为 0）。

    Parameters
    ----------
    X : ndarray, shape (n, d)
    affinity : {"rbf", "knn", "mutual_knn"}
        rbf        全连接高斯核图，每个点与所有点相连
        knn        每个点连最近的 k 个邻居，按"并集"对称化
        mutual_knn 只保留互为近邻的边，按"交集"对称化（更稀疏、抗噪）
    n_neighbors : int
        kNN 的邻居数，affinity="rbf" 时只用来决定局部尺度的参考邻居。
    sigma : float, optional
        高斯核带宽；None 表示自动确定。
    local_scaling : bool
        True 时使用局部尺度 σ_i（Zelnik-Manor & Perona），
        能同时适应稠密区和稀疏区；False 时用全局 σ。

    Returns
    -------
    W : ndarray, shape (n, n)
    """
    if affinity not in AFFINITY_KINDS:
        raise ValueError("affinity 必须是 %s 之一，收到 %r" % (AFFINITY_KINDS, affinity))
    X = check_array(X)
    n = X.shape[0]
    d2 = squared_distances(X)                       # 距离平方 (n, n)
    D = np.sqrt(d2)                                 # 距离 (n, n)，只用于找最近邻

    # ---------- 第一步：算相似度权重 ----------
    if sigma is None and local_scaling:
        kth = int(np.clip(n_neighbors if n_neighbors else 7, 1, n - 1))
        part = np.partition(D, kth, axis=1)         # 部分排序 O(n)，把第 kth 小的元素放到位置 kth
        sigma_i = np.maximum(part[:, kth], 1e-10)   # 第 k 近邻距离作为点 i 自己的尺度
        W = np.exp(-d2 / (sigma_i[:, None] * sigma_i[None, :]))
    else:
        if sigma is None:                           # 全局尺度：取所有非零距离的中位数
            sigma = float(np.median(D[D > 0])) or 1.0
        W = np.exp(-d2 / (2.0 * float(sigma) ** 2))

    # ---------- 第二步：kNN 稀疏化 ----------
    if affinity in ("knn", "mutual_knn") and 0 < n_neighbors < n - 1:
        k = int(n_neighbors)
        knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]         # 跳过第 0 列（自己），取前 k 个邻居
        mask = np.zeros((n, n), dtype=bool)                 # 布尔邻接掩码
        mask[np.arange(n)[:, None], knn_idx] = True         # 花式索引批量置 True
        mask = (mask & mask.T) if affinity == "mutual_knn" else (mask | mask.T)
        W = np.where(mask, W, 0.0)                          # 非邻居位置的相似度清零

    W = 0.5 * (W + W.T)                             # 强制数值对称，消除浮点误差
    np.fill_diagonal(W, 0.0)                        # 点与自己不连边
    return W



# -------------------------- ② 构建 拉普拉斯矩阵 L --------------------------
# 4. 度向量
def degree(W: Array) -> Array:
    """度向量：d_i = Σ_j W_ij，表示点 i 与全图的连接强度。"""
    return np.asarray(W, dtype=float).sum(axis=1)

# 5. 拉普拉斯矩阵
def compute_laplacian(W: Array, kind: str = "sym") -> Tuple[Array, Optional[Array]]:
    """由亲和矩阵计算拉普拉斯矩阵，返回 (L, aux)。

    kind
    ----
    "none" : L = D − W                        （非归一化，aux = None）
    "sym"  : L = I − D^{-1/2} W D^{-1/2}      （对称归一化，aux = D^{-1/2}）
    "rw"   : L = I − D^{-1} W                 （随机游走归一化，aux = D^{-1}，一般不对称）

    aux 是归一化用的对角向量，谱嵌入阶段还需要它。
    """
    if kind not in LAPLACIAN_KINDS:
        raise ValueError("kind 必须是 %s 之一，收到 %r" % (LAPLACIAN_KINDS, kind))
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    d = degree(W)
    d_safe = np.maximum(d, 1e-12)                   # 孤立点度为 0，取倒数会变成 inf，这里兜底

    if kind == "none":
        return np.diag(d) - W, None

    if kind == "sym":
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)          # D^{-1/2} 的对角元向量
        L = np.eye(n) - (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
        return L, d_inv_sqrt                        # 左乘缩放行、右乘缩放列

    d_inv = 1.0 / d_safe                            # D^{-1} 的对角元向量
    L = np.eye(n) - d_inv[:, None] * W
    return L, d_inv

# -------------------------- ③ 构建 谱嵌入矩阵 U --------------------------
# 6. 谱嵌入矩阵 U
def spectral_embedding(W: Array, n_clusters: int, kind: str = "sym",
                       normalize_rows: Optional[bool] = None) -> Tuple[Array, Array]:
    """谱嵌入：返回 (前 k 个最小特征值, 嵌入矩阵 U)。

    实现要点
    --------
    1. 用 np.linalg.eigh 而不是 eig：L 对称，eigh 更快更稳，且特征值自动升序返回；
    2. "rw" 的 L 一般不对称，但它与 L_sym 相似：
           L_rw = D^{-1/2} L_sym D^{1/2}，u_rw = D^{-1/2} u_sym
       所以统一对 L_sym 做一次分解，再换算，避免处理非对称矩阵；
    3. 行归一化（NJW 算法的关键）：把 U 的每一行缩成单位长度，
       让 K-Means 只比较"方向"而不受各列尺度差异的影响。

    normalize_rows 为 None 时按约定自动决定：归一化拉普拉斯 → True，非归一化 → False。
    """
    if kind not in LAPLACIAN_KINDS:
        raise ValueError("kind 必须是 %s 之一，收到 %r" % (LAPLACIAN_KINDS, kind))
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    k = int(n_clusters)
    if not 1 <= k <= n:
        raise ValueError("n_clusters 必须在 [1, %d] 之间，收到 %r" % (n, n_clusters))

    if kind == "none":
        L, _ = compute_laplacian(W, "none")
        eigenvalues, eigenvectors = np.linalg.eigh(L)       # 实对称矩阵的特征分解
        U = eigenvectors[:, :k]                             # 前 k 列即最小的 k 个特征向量
        if normalize_rows is None:
            normalize_rows = False                          # 非归一化版本默认不做行归一化
    else:
        L_sym, d_inv_sqrt = compute_laplacian(W, "sym")
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        U = eigenvectors[:, :k]
        if kind == "rw":
            U = U * d_inv_sqrt[:, None]                     # 按行缩放：u_rw = D^{-1/2} u_sym
        if normalize_rows is None:
            normalize_rows = True

    if normalize_rows:
        row_norm = np.linalg.norm(U, axis=1, keepdims=True)  # 每行的 L2 范数，形状 (n, 1)
        U = U / np.maximum(row_norm, 1e-12)                 # 广播除法，1e-12 防零行

    order = np.argsort(eigenvalues)                         # eigh 已升序，这里再保险一次
    return eigenvalues[order][:k], U

# 7. 只返回特征值谱
def laplacian_spectrum(W: Array, kind: str = "sym") -> Array:
    """返回拉普拉斯矩阵的完整特征值谱（升序，已截断负零）。"""
    L, _ = compute_laplacian(W, kind)
    eigenvalues, _ = np.linalg.eigh(L)
    return np.maximum(eigenvalues, 0.0)

# -------------------------- ④ 自动定 k --------------------------
# 8. 自动定 K 的值
def estimate_n_clusters(W: Array, max_k: int = 10, kind: str = "sym",
                        method: str = "auto", tol: float = 1e-10) -> Tuple[int, Array]:
    """用特征值间隙估计簇数，返回 (k, 完整特征值谱)。

    method
    ------
    "gap"      : 经典绝对间隙 γ_k = λ_{k+1} − λ_k 的最大值处。
                 理论依据：图恰有 k 个连通分量 ⟺ λ_1..λ_k = 0。
                 缺点：谱缓慢上升时最大间隙可能出现在很靠后的位置，导致高估 k。
    "relative" : 相对间隙 (λ_{k+1} − λ_k) / λ_{k+1}，消除整体尺度影响，实用中更稳。
    "auto"     : 先数数值为 0 的特征值（等于连通分量数，这是精确结论），
                 若 ≥ 2 直接返回；否则回退到 "relative"。
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    L, _ = compute_laplacian(W, kind)
    eigenvalues, _ = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)      # 抹掉 −1e-16 量级的负零
    max_k = int(min(max_k, n - 1))                  # k 不能超过 n−1

    if method == "gap":
        gaps = np.diff(eigenvalues[: max_k + 1])    # 相邻特征值之差，长度 max_k
        return int(np.argmax(gaps)) + 1, eigenvalues

    if method == "auto":
        n_zero = int(np.sum(eigenvalues[: max_k + 1] <= tol))
        if n_zero >= 2:                             # 图被切成好几块，块数就是簇数
            return min(n_zero, max_k), eigenvalues

    gaps = np.diff(eigenvalues[: max_k + 1])
    relative = np.zeros_like(gaps)
    valid = eigenvalues[1: max_k + 1] > tol         # 分母 λ_{i+1} 不能为 0
    relative[valid] = gaps[valid] / eigenvalues[1: max_k + 1][valid]
    relative[0] = 0.0                               # k=1 不作为候选
    return int(np.argmax(relative)) + 1, eigenvalues

# -------------------------- ⑤ KMeans 离散化 --------------------------
# 9. KMeans ++
class KMeans:
    """手写 K-Means：K-Means++ 初始化 + Lloyd 迭代 + 多次重启取最优。

    参数
    ----
    n_clusters : int    簇数
    max_iter   : int    单次运行的最大迭代步数
    tol        : float  中心整体位移小于该值即认为收敛
    n_init     : int    重复初始化次数，取目标函数最小的那次
    random_state : int  随机种子

    属性（拟合后可用）
    ------------------
    labels_           ndarray, shape (n,)   簇标签
    cluster_centers_  ndarray, shape (k, d) 簇中心
    inertia_          float                 簇内平方和
    n_iter_           int                   实际迭代步数
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, tol: float = 1e-8,
                 n_init: int = 10, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _init_centers(self, X: Array, rng: np.random.Generator) -> Array:
        """K-Means++ 初始化：第一个中心随机取，之后按 D² 概率采样（离得越远越容易被选中）。"""
        n = X.shape[0]
        k = int(self.n_clusters)
        centers = np.empty((k, X.shape[1]), dtype=float)
        centers[0] = X[int(rng.integers(n))]                        # 均匀随机选第一个中心
        closest = squared_distances(X, centers[:1])[:, 0]           # 各点到最近中心的距离平方
        for j in range(1, k):
            total = closest.sum()
            probs = closest / total if total > 0 else np.full(n, 1.0 / n)
            centers[j] = X[int(rng.choice(n, p=probs))]             # 按概率抽样
            new_d2 = squared_distances(X, centers[j:j + 1])[:, 0]   # 切片保持二维形状
            closest = np.minimum(closest, new_d2)                   # 增量更新，避免重算所有中心
        return centers

    def fit(self, X: Array) -> "KMeans":
        """在数据 X 上拟合，返回 self（sklearn 风格，支持链式调用）。"""
        X = check_array(X)
        n = X.shape[0]
        k = int(self.n_clusters)
        if not 1 <= k <= n:
            raise ValueError("n_clusters 必须在 [1, %d] 之间，收到 %r" % (n, self.n_clusters))

        rng = np.random.default_rng(self.random_state)
        best = None                                                 # (inertia, labels, centers, n_iter)

        for _ in range(int(self.n_init)):
            centers = self._init_centers(X, rng)
            labels = np.zeros(n, dtype=int)
            n_iter = 0
            for step in range(int(self.max_iter)):
                n_iter = step + 1
                labels = np.argmin(squared_distances(X, centers), axis=1)   # E 步：归到最近中心
                new_centers = centers.copy()                                # 复制后再改，避免边算边改
                for j in range(k):                                          # M 步：逐簇求均值
                    member = labels == j
                    if np.any(member):
                        new_centers[j] = X[member].mean(axis=0)
                    else:                                                   # 空簇：随机重启
                        new_centers[j] = X[int(rng.integers(n))]
                shift = float(np.linalg.norm(new_centers - centers))        # 中心整体位移
                centers = new_centers
                if shift <= self.tol:                                       # 收敛则提前退出
                    break

            d2 = squared_distances(X, centers)
            inertia = float(d2[np.arange(n), labels].sum())                 # 簇内平方和 SSE
            if best is None or inertia < best[0]:                           # 保留最优的一次
                best = (inertia, labels.copy(), centers.copy(), n_iter)

        self.inertia_, self.labels_, self.cluster_centers_, self.n_iter_ = best
        return self

    def fit_predict(self, X: Array) -> Array:
        """拟合并返回簇标签。"""
        return self.fit(X).labels_

    def predict(self, X: Array) -> Array:
        """把样本分配给距离最近的中心（需要先 fit）。"""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("请先调用 fit() 再调用 predict()")
        return np.argmin(squared_distances(check_array(X), self.cluster_centers_), axis=1)

# -------------------------- ⑥ 组装 --------------------------
# 10. 谱聚类
class SpectralClustering:
    """谱聚类（手写实现，接口对齐 sklearn.cluster.SpectralClustering）。

    示例
    ----
    # >>> model = SpectralClustering(n_clusters=2, affinity="knn", n_neighbors=10)
    # >>> labels = model.fit_predict(X)
    # >>> model.embedding_.shape
    (300, 2)
    """

    def __init__(self, n_clusters: Optional[int] = 2, affinity: str = "knn",
                 n_neighbors: int = 10, sigma: Optional[float] = None,
                 local_scaling: bool = True, laplacian: str = "sym",
                 normalize_rows: Optional[bool] = None, estimate_k: bool = False,
                 max_k: int = 10, estimate_method: str = "auto",
                 kmeans_n_init: int = 10, kmeans_max_iter: int = 300,
                 tol: float = 1e-8, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.local_scaling = local_scaling
        self.laplacian = laplacian
        self.normalize_rows = normalize_rows
        self.estimate_k = estimate_k
        self.max_k = max_k
        self.estimate_method = estimate_method
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------ 训练
    def fit(self, X: Array) -> "SpectralClustering":
        """在数据 X 上拟合谱聚类模型，返回 self。"""
        X = check_array(X)

        # ① 构图
        W = build_affinity(X, affinity=self.affinity, n_neighbors=self.n_neighbors,
                           sigma=self.sigma, local_scaling=self.local_scaling)

        # 决定簇数：显式指定，或者用特征值间隙自动估计
        if self.estimate_k or self.n_clusters is None:
            n_clusters, eigenvalues_all = estimate_n_clusters(
                W, max_k=self.max_k, kind=self.laplacian, method=self.estimate_method)
        else:
            n_clusters, eigenvalues_all = int(self.n_clusters), None

        # ②③ 建拉普拉斯 + 谱嵌入（内部完成）
        eigenvalues, embedding = spectral_embedding(
            W, n_clusters, kind=self.laplacian, normalize_rows=self.normalize_rows)

        # ④ 在嵌入空间里做 K-Means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=self.kmeans_max_iter, tol=self.tol,
                        n_init=self.kmeans_n_init, random_state=self.random_state).fit(embedding)

        self.affinity_ = W
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvalues_all_ = eigenvalues_all
        self.n_clusters_ = n_clusters
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.n_iter_ = kmeans.n_iter_
        self._X_fit = X                                     # 供 predict 使用
        return self

    def fit_predict(self, X: Array) -> Array:
        """拟合并返回簇标签（最常用的入口）。"""
        return self.fit(X).labels_

    def predict(self, X: Array) -> Array:
        """给新样本分配簇标签（近似方法）。

        谱聚类是"直推式"算法：它的特征分解是建立在整张图上的，
        新样本进来会改变图的结构，因此没有严格意义上的 out-of-sample 预测。
        这里的做法是把新样本指派给**训练集中最近的样本**所在的簇（1-近邻近似），
        只适合样本密集、簇边界清晰的场景，不要当作严格结论使用。
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError("请先调用 fit() 再调用 predict()")
        X = check_array(X)
        nearest = np.argmin(squared_distances(X, self._X_fit), axis=1)   # (m,) 最近训练样本下标
        return self.labels_[nearest]

    # ------------------------------------------------------------ 参数管理
    def get_params(self) -> dict:
        """返回所有超参数（sklearn 风格的 get_params）。"""
        return {
            "n_clusters": self.n_clusters,
            "affinity": self.affinity,
            "n_neighbors": self.n_neighbors,
            "sigma": self.sigma,
            "local_scaling": self.local_scaling,
            "laplacian": self.laplacian,
            "normalize_rows": self.normalize_rows,
            "estimate_k": self.estimate_k,
            "max_k": self.max_k,
            "estimate_method": self.estimate_method,
            "kmeans_n_init": self.kmeans_n_init,
            "kmeans_max_iter": self.kmeans_max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "SpectralClustering":
        """批量修改超参数，返回 self。"""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError("未知参数 %r，可用参数：%s" % (key, sorted(valid)))
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """打印模型时显示关键参数，便于在日志里辨认。"""
        return ("SpectralClustering(n_clusters=%s, affinity=%r, n_neighbors=%s, "
                "laplacian=%r, estimate_k=%s)" % (
                    self.n_clusters, self.affinity, self.n_neighbors,
                    self.laplacian, self.estimate_k))



if __name__ == "__main__":
    from data import get_dataset
    from metrics import evaluate

    for name, k in [("two_moons", 2), ("circles", 2), ("blobs", 3)]:
        X, y = get_dataset(name, n_samples=300, random_state=0)
        labels = SpectralClustering(n_clusters=k, n_neighbors=10).fit_predict(X)
        print("%-10s ARI = %.3f" % (name, evaluate(y, labels)["ari"]))














