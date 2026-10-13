# -*- coding: utf-8 -*-
"""实验流程模块：把 data / model / metrics / visualize 串起来。

职责划分
--------
    main.py   只负责解析命令行参数，不写业务逻辑
    run.py    负责"跑实验"：加载数据 → 拟合模型 → 评估 → 出图 → 保存结果

这样拆分的好处：想换一种实验方式（比如批量扫参），直接 import run 调用即可，
不需要经过命令行。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from config import FIGURE_DIR, RESULT_DIR, DataConfig, ModelConfig, RunConfig
from data import DATASETS, get_dataset, list_datasets
from metrics import evaluate
from model import SpectralClustering, laplacian_spectrum
from utils import ensure_dir, get_logger, save_json, timer
from visualize import plot_clustering, plot_spectrum

LOGGER = get_logger("run")
ALL_DATASETS = "all"                        # 特殊名字：表示"跑全部数据集"

# 1. 装配层
def build_model(model_cfg: ModelConfig) -> SpectralClustering:
    """把 ModelConfig 翻译成模型实例。

    这一步是"配置与实现解耦"的关键：配置对象只描述"要什么"，
    模型类只管"怎么做"，两者通过这个函数对接。
    """
    return SpectralClustering(
        n_clusters=model_cfg.n_clusters,
        affinity=model_cfg.affinity,
        n_neighbors=model_cfg.n_neighbors,
        sigma=model_cfg.sigma,
        local_scaling=model_cfg.local_scaling,
        laplacian=model_cfg.laplacian,
        normalize_rows=model_cfg.normalize_rows,
        estimate_k=model_cfg.estimate_k,
        max_k=model_cfg.max_k,
        estimate_method=model_cfg.estimate_method,
        kmeans_n_init=model_cfg.kmeans_n_init,
        kmeans_max_iter=model_cfg.kmeans_max_iter,
        tol=model_cfg.tol,
        random_state=model_cfg.random_state,
    )

# 2. 加载数据集
def load_data(data_cfg: DataConfig):
    """按配置加载数据集，返回 (X, y_true)。"""
    X, y_true = get_dataset(
        data_cfg.name,
        n_samples=data_cfg.n_samples,       # 生成器不接受的参数会在 get_dataset 里被过滤
        noise=data_cfg.noise,
        factor=data_cfg.factor,
        random_state=data_cfg.random_state,
    )
    return X, y_true

# 3. 单次训练
def run_once(data_cfg: DataConfig, model_cfg: ModelConfig, save_figure: bool = True,
             save_result: bool = True, verbose: bool = True) -> Dict:
    """在单个数据集上跑一次完整实验，返回结果字典。"""
    X, y_true = load_data(data_cfg)
    model = build_model(model_cfg)
    if verbose:
        LOGGER.info("数据集 %s：%d 个样本，%d 维；模型 %s",
                    data_cfg.name, X.shape[0], X.shape[1], model)

    with timer() as clock:                          # 只统计拟合耗时，不含画图
        model.fit(X)
    scores = evaluate(y_true, model.labels_)        # 准确率 / ARI / NMI

    result: Dict = {
        "dataset": data_cfg.name,
        "desc": DATASETS[data_cfg.name]["desc"],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "true_n_clusters": int(data_cfg.n_clusters),
        "n_clusters": int(model.n_clusters_),
        "k_correct": bool(model.n_clusters_ == data_cfg.n_clusters),
        "affinity": model_cfg.affinity,
        "laplacian": model_cfg.laplacian,
        "n_neighbors": int(model_cfg.n_neighbors),
        "seconds": round(clock["seconds"], 4),
        "kmeans_inertia": float(model.inertia_),
        "spectrum_head": np.round(np.maximum(model.eigenvalues_[:6], 0.0), 4).tolist(),
        "labels": model.labels_,                    # 供画图和后续分析使用
        "model": model,
        "X": X,
        "y_true": y_true,
    }
    result.update(scores)                           # 把 accuracy / ari / nmi 合并进来

    tag = "%s_%s" % (data_cfg.name, model_cfg.laplacian)
    if save_figure:                                 # 出两张图：聚类对比图 + 特征值谱图
        figure = plot_clustering(
            X, y_true, model.labels_, model.embedding_,
            title="手写谱聚类：%s（%s 拉普拉斯，k=%d）"
                  % (DATASETS[data_cfg.name]["desc"].split("：")[0],
                     model_cfg.laplacian, model.n_clusters_),
            save_path=FIGURE_DIR / ("clustering_%s.png" % tag))
        spectrum = laplacian_spectrum(model.affinity_, model_cfg.laplacian)
        spectrum_figure = plot_spectrum(
            spectrum[: max(model.n_clusters_ + 5, 10)], n_clusters=model.n_clusters_,
            title="特征值谱：%s（%s）" % (data_cfg.name, model_cfg.laplacian),
            save_path=FIGURE_DIR / ("spectrum_%s.png" % tag))
        result["figure"] = str(figure)
        result["spectrum_figure"] = str(spectrum_figure)
        if verbose:
            LOGGER.info("已保存图片：%s", figure)
            LOGGER.info("已保存图片：%s", spectrum_figure)

    if save_result:                                 # 指标写成 json（去掉不能序列化的大对象）
        payload = {key: value for key, value in result.items()
                   if key not in ("labels", "model", "X", "y_true")}
        path = save_json(payload, RESULT_DIR / ("metrics_%s.json" % tag))
        result["result_file"] = str(path)
    return result

# 4. 批次训练
def run_experiment(cfg: RunConfig) -> List[Dict]:
    """按配置跑实验：cfg.data.name 为 "all" 时跑全部数据集，否则只跑一个。"""
    ensure_dir(FIGURE_DIR)                          # 提前建好输出目录
    ensure_dir(RESULT_DIR)

    if cfg.data.name == ALL_DATASETS:               # 展开成"每个数据集一份配置"
        names = list_datasets()
    else:
        names = [cfg.data.name]

    results: List[Dict] = []
    for name in names:
        data_cfg = replace(cfg.data, name=name)     # 复制一份配置，只改数据集名字（不污染原配置）
        if cfg.data.name == ALL_DATASETS:           # 跑全部时，簇数取该数据集自己的真实值
            data_cfg.n_clusters = DATASETS[name]["n_clusters"]
        model_cfg = replace(cfg.model)              # 模型配置也复制一份，逐个数据集独立
        if model_cfg.n_clusters is not None and not model_cfg.estimate_k:
            model_cfg.n_clusters = data_cfg.n_clusters      # 让模型簇数与数据真值保持一致
        results.append(run_once(data_cfg, model_cfg, save_figure=cfg.save_figure,
                                save_result=cfg.save_result, verbose=cfg.verbose))
    return results

# 5. 交叉验证
def compare_with_sklearn(X, y_true, model_cfg, verbose=True) -> Optional[Dict]:
    """把本实现与 sklearn 的谱聚类做一次交叉验证。"""
    try:
        from sklearn.cluster import SpectralClustering as SklearnSpectral
    except Exception as exc:                        # pragma: no cover - 取决于环境
        if verbose:
            LOGGER.warning("未安装 sklearn，跳过对照实验：%s", exc)
        return None

    ours = SpectralClustering(n_clusters=model_cfg.n_clusters, affinity="knn",
                              n_neighbors=model_cfg.n_neighbors, laplacian=model_cfg.laplacian,
                              random_state=model_cfg.random_state).fit_predict(X)
    theirs = SklearnSpectral(n_clusters=model_cfg.n_clusters, affinity="nearest_neighbors",
                             n_neighbors=model_cfg.n_neighbors, assign_labels="kmeans",
                             random_state=model_cfg.random_state).fit_predict(X)
    agreement = evaluate(ours, theirs)["ari"]       # 两个划分之间的一致性
    if verbose:
        LOGGER.info("与 sklearn 对照：ARI 一致性 = %.3f", agreement)
    return {"sklearn_agreement": agreement, "ours": ours, "sklearn": theirs}

# 6. 把结果列表排版成一张文本表格
def format_summary(results: List[Dict]) -> str:
    """把结果列表排版成一张文本表格。"""
    header = "%-10s %-10s %-5s %-9s %-8s %-8s %-11s %-9s" % (
        "数据集", "拉普拉斯", "k", "准确率", "ARI", "NMI", "SSE", "耗时(s)")
    lines = [header, "-" * len(header)]
    for item in results:
        lines.append("%-10s %-10s %-5d %-9.3f %-8.3f %-8.3f %-11.3e %-9.4f" % (
            item["dataset"], item["laplacian"], item["n_clusters"], item["accuracy"],
            item["ari"], item["nmi"], item["kmeans_inertia"], item["seconds"]))
    return "\n".join(lines)


if __name__ == "__main__":
    # 直接运行 run.py 时，用默认配置跑全部数据集（方便单独调试本模块）
    from utils import setup_console

    setup_console()
    default_cfg = RunConfig()
    default_cfg.data.name = ALL_DATASETS
    all_results = run_experiment(default_cfg)
    print()
    print(format_summary(all_results))




















