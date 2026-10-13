# -*- coding: utf-8 -*-
"""项目配置模块：集中管理路径与全部超参数。

设计原则
--------
1. 所有可调参数都用 dataclass 声明，带默认值和类型注解，IDE 能自动补全；
2. 其它模块只读取配置、不修改配置，保证一次实验可以被完整复现；
3. 所有路径由 PROJECT_ROOT 推导，不受"从哪个目录运行脚本"影响。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

# --------------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent      # 本项目根目录（本文件所在目录）
OUTPUT_DIR = PROJECT_ROOT / "outputs"               # 所有运行产物的总目录
FIGURE_DIR = OUTPUT_DIR / "figures"                 # 图片输出目录
RESULT_DIR = OUTPUT_DIR / "results"                 # 指标 json 输出目录

# @dataclass 自动生成 __init__，所以 DataConfig(name="circles") 就能用，而且没传的参数自动取默认值。
@dataclass
class DataConfig:
    """数据集相关参数。"""

    name: str = "two_moons"     # 数据集名称，可选值见 data.DATASETS
    n_samples: int = 300        # 样本总数
    noise: float = 0.06         # 噪声标准差（two_moons / circles 使用）
    factor: float = 0.4         # 内圈半径比例（circles 使用）
    n_clusters: int = 2         # 该数据集的真实簇数（只用于评测，模型看不到）
    random_state: int = 17       # 随机种子

@dataclass
class ModelConfig:
    """谱聚类模型的全部超参数。"""

    n_clusters: int | None = 2          # 目标簇数；None 表示自动估计
    affinity: str = "knn"               # 构图方式：rbf / knn / mutual_knn
    n_neighbors: int = 10               # kNN 的邻居数（affinity="rbf" 时忽略）
    sigma: float | None = None          # 高斯核带宽；None 表示自动确定
    local_scaling: bool = True          # 是否使用局部尺度（Zelnik-Manor）
    laplacian: str = "sym"              # 拉普拉斯类型：none / sym / rw
    normalize_rows: bool | None = None  # 谱嵌入是否行归一化；None 表示按 laplacian 自动
    estimate_k: bool = False            # 是否用特征值间隙自动估计簇数
    max_k: int = 10                     # 自动估计时的最大候选簇数
    estimate_method: str = "auto"       # 估计策略：gap / relative / auto
    kmeans_n_init: int = 10             # K-Means 重复初始化次数（取最优）
    kmeans_max_iter: int = 300          # K-Means 单次最大迭代步数
    tol: float = 1e-8                   # K-Means 收敛阈值
    random_state: int = 17               # 随机种子


@dataclass
class RunConfig:
    """一次完整实验的配置：数据 + 模型 + 输出开关。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    save_figure: bool = True            # 是否保存图片
    save_result: bool = True            # 是否保存指标 json
    verbose: bool = True                # 是否打印过程日志

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典（asdict 会递归展开嵌套的 dataclass），便于写进 json。"""
        return asdict(self)

if __name__ == "__main__":
    import json
    print(json.dumps(RunConfig().to_dict(), indent=2, ensure_ascii=False))







