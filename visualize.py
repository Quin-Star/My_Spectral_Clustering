# -*- coding: utf-8 -*-
"""可视化模块：把聚类结果画成图。

提供两张图：
    plot_clustering  三栏对比图（原始数据真值 / 谱嵌入空间 / 聚类结果）
    plot_spectrum    特征值谱图，用来人工判断簇数 k

统一使用 Agg 后端（只写文件、不弹窗），保证在服务器、CI 等无图形界面的
环境下也能正常跑完；同时统一配置中文字体，避免标题出现方框。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]

# 1. 配置 matplotlib 后端与字体
def setup_matplotlib():
    """配置 matplotlib 后端与字体，返回 pyplot 模块。

    必须在导入 pyplot 之前调用 use("Agg")，顺序反了不生效。
    """
    import matplotlib
    matplotlib.use("Agg")                                   # 非交互后端：只输出文件，不弹窗
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False               # 用 ASCII 减号，避免负号显示异常
    plt.rcParams["figure.dpi"] = 110
    return plt

# 2. 三栏对比图
def plot_clustering(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray,
                    embedding: np.ndarray, title: str = "",
                    save_path: Optional[PathLike] = None) -> Optional[Path]:
    """画三栏对比图并保存。
    ...
    """
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    scatter_kw = dict(s=22, edgecolors="k", linewidths=0.3)     # 三个子图共用的散点样式

    # ---- 第一栏：原始数据 + 真实标签 ----
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", **scatter_kw)
    axes[0].set_title("原始数据（真实标签）")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")

    # ---- 第二栏：谱嵌入空间（谱聚类为什么有效的直接证据）----
    if embedding.shape[1] >= 2:
        axes[1].scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", **scatter_kw)
        axes[1].set_xlabel("u1")
        axes[1].set_ylabel("u2")
    else:                                                       # 只有一维时退化成一条水平线
        axes[1].scatter(embedding[:, 0], np.zeros(len(embedding)), c=labels,
                        cmap="viridis", **scatter_kw)
        axes[1].set_xlabel("u1")
    axes[1].set_title("谱嵌入空间（前两维）")

    # ---- 第三栏：聚类结果 ----
    axes[2].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", **scatter_kw)
    axes[2].set_title("谱聚类结果")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    saved = None
    if save_path is not None:
        saved = Path(save_path)
        saved.parent.mkdir(parents=True, exist_ok=True)          # 自动创建父目录
        fig.savefig(saved, dpi=140, bbox_inches="tight")
    plt.close(fig)                                              # 关闭画布释放内存
    return saved

# 3. 特征值图谱
def plot_spectrum(eigenvalues: np.ndarray, n_clusters: Optional[int] = None,
                  title: str = "拉普拉斯矩阵特征值谱",
                  save_path: Optional[PathLike] = None) -> Optional[Path]:
    """画特征值谱图（scree plot），用于人工确认簇数 k。

    判读方法：曲线在某个位置出现明显"拐点"（之后迅速变陡/变平），
    拐点横坐标就是建议的 k。若图恰好分成 k 块，前 k 个特征值会精确等于 0。
    """
    plt = setup_matplotlib()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    index = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(index, eigenvalues, marker="o", markersize=5, linewidth=1.6,
            color="#2b6cb0", label="特征值 λ")
    if n_clusters is not None:                                  # 标出选定的 k 所在位置
        ax.axvline(n_clusters, color="#e53e3e", linestyle="--", linewidth=1.4,
                   label="选定 k = %d" % n_clusters)
    ax.set_xlabel("特征值序号 i")
    ax.set_ylabel("λ_i")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    saved = None
    if save_path is not None:
        saved = Path(save_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return saved



















