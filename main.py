# -*- coding: utf-8 -*-
"""命令行入口：解析参数 → 组装配置 → 调用 run 跑实验。

使用示例
--------
    python main.py                                    # 默认：双半月 + 默认参数
    python main.py --all                              # 三个数据集全跑一遍
    python main.py --dataset circles --laplacian rw   # 指定数据集与拉普拉斯类型
    python main.py --all --estimate-k                 # 用特征值间隙自动估计簇数
    python main.py --dataset two_moons --affinity rbf --no-figure
    python main.py --all --compare                    # 额外与 sklearn 做一致性对照

设计原则：本文件只做"参数 → 配置"的翻译工作，不含任何算法与实验逻辑。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from config import OUTPUT_DIR, DataConfig, ModelConfig, RunConfig
from data import list_datasets
from model import AFFINITY_KINDS, LAPLACIAN_KINDS
from run import ALL_DATASETS, compare_with_sklearn, format_summary, run_experiment
from utils import ensure_dir, setup_console

def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="手写谱聚类（Spectral Clustering）实验入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,   # 帮助信息里自动带默认值
    )
    # ---- 数据相关 ----
    parser.add_argument("--dataset", default="two_moons",
                        choices=list_datasets() + [ALL_DATASETS],
                        help="数据集名称；all 表示依次跑全部数据集")
    parser.add_argument("--all", action="store_true",
                        help="等价于 --dataset all")
    parser.add_argument("--n-samples", type=int, default=300, help="样本总数")
    parser.add_argument("--noise", type=float, default=0.06, help="噪声标准差")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    # ---- 模型相关 ----
    parser.add_argument("-k", "--n-clusters", type=int, default=2, help="簇数")
    parser.add_argument("--estimate-k", action="store_true",
                        help="用特征值间隙自动估计簇数（忽略 --n-clusters）")
    parser.add_argument("--affinity", default="knn", choices=list(AFFINITY_KINDS),
                        help="构图方式")
    parser.add_argument("--n-neighbors", type=int, default=10, help="kNN 邻居数")
    parser.add_argument("--sigma", type=float, default=None, help="高斯核带宽，默认自动")
    parser.add_argument("--laplacian", default="sym", choices=list(LAPLACIAN_KINDS),
                        help="拉普拉斯矩阵类型")
    parser.add_argument("--no-local-scaling", action="store_true",
                        help="关闭局部尺度，改用全局 sigma")
    parser.add_argument("--n-init", type=int, default=10, help="K-Means 重复初始化次数")
    # ---- 输出相关 ----
    parser.add_argument("--no-figure", action="store_true", help="不保存图片")
    parser.add_argument("--no-result", action="store_true", help="不保存指标 json")
    parser.add_argument("--quiet", action="store_true", help="不打印过程日志")
    parser.add_argument("--compare", action="store_true",
                        help="额外与 sklearn 的谱聚类做一致性对照")
    return parser

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数；argv 为 None 时读取 sys.argv。"""
    return build_parser().parse_args(argv)

def build_config(args: argparse.Namespace) -> RunConfig:
    """把命令行参数翻译成 RunConfig。"""
    dataset = ALL_DATASETS if args.all else args.dataset
    return RunConfig(
        data=DataConfig(
            name=dataset,
            n_samples=args.n_samples,
            noise=args.noise,
            random_state=args.seed,
        ),
        model=ModelConfig(
            n_clusters=args.n_clusters,
            affinity=args.affinity,
            n_neighbors=args.n_neighbors,
            sigma=args.sigma,
            local_scaling=not args.no_local_scaling,
            laplacian=args.laplacian,
            estimate_k=args.estimate_k,
            kmeans_n_init=args.n_init,
            random_state=args.seed,
        ),
        save_figure=not args.no_figure,
        save_result=not args.no_result,
        verbose=not args.quiet,
    )

def main(argv: Optional[List[str]] = None) -> int:
    """程序主入口，返回进程退出码（0 表示成功）。"""
    setup_console()                                     # Windows 控制台 UTF-8
    args = parse_args(argv)
    config = build_config(args)

    ensure_dir(OUTPUT_DIR)                              # 提前建好输出目录
    results = run_experiment(config)                     # 跑实验（可能包含多个数据集）

    print()
    print(format_summary(results))                      # 打印汇总表

    if args.compare and results:                         # 可选：与 sklearn 对照
        first = results[0]
        compare_with_sklearn(first["X"], first["y_true"], config.model, verbose=config.verbose)

    failed = [item for item in results if not item["k_correct"]] if args.estimate_k else []
    if failed:                                           # 自动估计簇数时，提示哪些没猜对
        print("\n提示：以下数据集的簇数未估计正确：%s"
              % ", ".join("%s(估成%d)" % (item["dataset"], item["n_clusters"]) for item in failed))
    return 0


"""
python main.py --help                                # 帮助信息，看默认值是否显示
python main.py                                       # 默认：双半月
python main.py --all                                 # 三个数据集
python main.py --all --compare                       # 加 sklearn 对照
python main.py --all --estimate-k                    # 自动估计簇数
python main.py --dataset circles --laplacian rw      # 指定数据集与拉普拉斯
python main.py --dataset two_moons --affinity rbf    # 换构图方式
python main.py --all --no-figure --no-result --quiet # 只出表格，不写文件
"""

if __name__ == "__main__":
    sys.exit(main())







