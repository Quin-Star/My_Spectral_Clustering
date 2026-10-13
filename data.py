# -*- coding: utf-8 -*-
"""数据集模块：只负责"造数据 / 读数据"，不涉及任何模型逻辑。

对外统一接口
------------
    X, y = get_dataset("two_moons", n_samples=300, noise=0.06)

    X : ndarray, shape (n_samples, n_features)  特征矩阵（float）
    y : ndarray, shape (n_samples,)             真实标签（int，仅供评测使用）

聚类是无监督任务，模型从头到尾看不到 y；y 只在最后算指标时用一次。
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray
PathLike = Union[str, Path]

# 指定支持中文的字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体，Windows 自带
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

def _shuffle(X: Array, y: Array, rng: np.random.Generator) -> Tuple[Array, Array]:
    """用同一个随机排列同时打乱特征和标签，保持两者对应关系。

    打乱是必须的：否则数据按簇顺序排列（前 150 个是一类），
    任何算法都能"蒙对"，评测就失去意义了。
    """
    order = rng.permutation(len(X))
    return X[order], y[order]

def make_blobs(n_samples: int = 300, centers: Optional[Array] = None,
               cluster_std: float = 0.55, random_state: int = 0) -> Tuple[Array, Array]:
    """生成各向同性的高斯簇数据。

    Parameters
    ----------
    n_samples : int
        样本总数。
    centers : array-like, shape (k, 2), optional
        每个簇的中心；None 时使用内置的三中心布局。
    cluster_std : float
        簇内标准差，越大簇之间重叠越多。
    random_state : int
        随机种子。

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
    y : ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)           # 独立随机数生成器
    if centers is None:
        centers = np.array([[-4.0, 0.0], [0.0, 3.2], [4.0, 0.0]])
    centers = np.asarray(centers, dtype=float)
    k = centers.shape[0]                                # 簇个数

    counts = np.full(k, n_samples // k, dtype=int)      # 每簇先平均分配
    counts[: n_samples % k] += 1                        # 余数补到前几簇，保证总数为 n_samples

    X_parts, y_parts = [], []                           # 先存列表，最后一次拼接
    for j in range(k):
        points = centers[j] + rng.normal(0.0, cluster_std, size=(counts[j], 2))
        X_parts.append(points)                          # 以第 j 个中心为均值撒点
        y_parts.append(np.full(counts[j], j, dtype=int))

    X = np.vstack(X_parts)                              # 纵向拼成 (n, 2)
    y = np.concatenate(y_parts)                         # 拼成 (n,)
    return _shuffle(X, y, rng)

def make_two_moons(n_samples: int = 300, noise: float = 0.06,
                   random_state: int = 0) -> Tuple[Array, Array]:
    """生成两个交错的半月形数据（经典的非线性可分数据）。

    Parameters
    ----------
    n_samples : int
        样本总数，两个半月各占一半。
    noise : float
        高斯噪声标准差。
    random_state : int
        随机种子。

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
    y : ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n_upper = n_samples // 2                            # 上半月样本数
    n_lower = n_samples - n_upper                       # 下半月（用减法保证总数为 n_samples）

    t_upper = np.linspace(0.0, np.pi, n_upper)          # 参数 t 从 0 到 π
    t_lower = np.linspace(0.0, np.pi, n_lower)
    upper = np.column_stack([np.cos(t_upper), np.sin(t_upper)])             # 上半圆
    lower = np.column_stack([1.0 - np.cos(t_lower), 0.5 - np.sin(t_lower)])  # 镜像 + 下移

    X = np.vstack([upper, lower])                       # (n, 2)
    y = np.concatenate([np.zeros(n_upper, dtype=int), np.ones(n_lower, dtype=int)])
    X = X + rng.normal(0.0, noise, size=X.shape)        # 加噪声，制造轻微重叠
    return _shuffle(X, y, rng)

def make_circles(n_samples: int = 300, noise: float = 0.05, factor: float = 0.4,
                 random_state: int = 0) -> Tuple[Array, Array]:
    """生成同心圆环数据（内圈 + 外圈）。

    Parameters
    ----------
    n_samples : int
        样本总数，内外圈各占一半。
    noise : float
        高斯噪声标准差。
    factor : float
        内圈半径与外圈半径之比，取值 (0, 1)。
    random_state : int
        随机种子。

    Returns
    -------
    X : ndarray, shape (n_samples, 2)
    y : ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    t_outer = np.linspace(0.0, 2.0 * np.pi, n_outer, endpoint=False)
    t_inner = np.linspace(0.0, 2.0 * np.pi, n_inner, endpoint=False)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])              # 半径 1
    inner = factor * np.column_stack([np.cos(t_inner), np.sin(t_inner)])     # 半径 factor

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    X = X + rng.normal(0.0, noise, size=X.shape)
    return _shuffle(X, y, rng)

def load_csv(path: PathLike, delimiter: str = ",", label_column: Optional[int] = -1,
             skip_header: int = 0) -> Tuple[Array, Array]:
    """从文本/CSV 文件读取数据。

    Parameters
    ----------
    path : str or Path
        文件路径。
    delimiter : str
        列分隔符，默认逗号。
    label_column : int or None
        标签所在列的下标，默认最后一列（-1）；None 表示没有标签列。
    skip_header : int
        跳过开头多少行（表头）。

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
        没有标签列时返回全 -1。
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    if data.ndim == 1:                                  # 只有一行数据时补成二维
        data = data.reshape(1, -1)
    if label_column is None:
        return data.astype(float), -np.ones(len(data), dtype=int)
    X = np.delete(data, label_column, axis=1).astype(float)     # 删掉标签列
    y = data[:, label_column].astype(int)
    return X, y

DATASETS: Dict[str, Dict] = {
    "two_moons": {
        "func": make_two_moons,
        "n_clusters": 2,
        "desc": "双半月：两簇交错咬合，线性不可分",
    },
    "circles": {
        "func": make_circles,
        "n_clusters": 2,
        "desc": "同心圆：内圈外圈，K-Means 的死穴",
    },
    "blobs": {
        "func": make_blobs,
        "n_clusters": 3,
        "desc": "高斯簇：凸簇基准，普通 K-Means 也能做",
    },
}


def list_datasets() -> List[str]:
    """返回所有内置数据集的名字。"""
    return list(DATASETS.keys())


def get_dataset(name: str, **kwargs) -> Tuple[Array, Array]:
    """按名字生成数据集。

    多余的参数会被自动过滤掉：例如 DataConfig 里带了 n_clusters，
    但生成器并不接受这个参数，在这里会被丢掉而不是报错。
    """
    if name not in DATASETS:
        raise KeyError("未知数据集 %r，可选：%s" % (name, list_datasets()))
    func: Callable = DATASETS[name]["func"]
    accepted = set(inspect.signature(func).parameters)          # 该生成器接受的参数名
    valid_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    X, y = func(**valid_kwargs)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


if __name__ == "__main__":

    for dataset_name in list_datasets():
        X_demo, y_demo = get_dataset(dataset_name, n_samples=300, random_state=0)
        print("%-12s X=%s y=%s 类别数=%d  %s" % (
            dataset_name, X_demo.shape, y_demo.shape,
            len(np.unique(y_demo)), DATASETS[dataset_name]["desc"]))

    # 1. 各生成一份数据
    Xb, yb = make_blobs(n_samples=300, cluster_std=0.55, random_state=17)
    Xm, ym = make_two_moons(300, noise=0.06, random_state=0)
    Xc, yc = make_circles(300, noise=0.05, random_state=17)

    # 2. 1 行 3 列，共三个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(Xb[:, 0], Xb[:, 1], c=yb, s=20, cmap="viridis")
    axes[0].set_title("blobs")
    axes[0].axis("equal")

    axes[1].scatter(Xm[:, 0], Xm[:, 1], c=ym, s=20, cmap="viridis")
    axes[1].set_title("双半月")
    axes[1].axis("equal")

    axes[2].scatter(Xc[:, 0], Xc[:, 1], c=yc, s=20, cmap="viridis")
    axes[2].set_title("同心圆")
    axes[2].axis("equal")

    plt.tight_layout()
    plt.show()






















