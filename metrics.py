# -*- coding: utf-8 -*-
"""评价指标模块：聚类结果好不好，用这三个指标衡量。

    clustering_accuracy      聚类准确率（需要做标签最优匹配）
    adjusted_rand_index      ARI，调整兰德指数（对标签编号不敏感）
    normalized_mutual_info   NMI，标准化互信息（对簇数不同也更宽容）
    evaluate                 一次性算完所有指标，返回字典

聚类结果的标签编号是任意的（我的"第 0 簇"可能是真值的"第 1 类"），
所以准确率必须先求解一个标签指派问题；ARI 和 NMI 则天然免疫这一点。
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

Array = np.ndarray

def _encode(labels: Array) -> Array:
    """把任意标签重编号成紧凑的 0..k-1，方便建列联表。"""
    labels = np.asarray(labels)
    _, encoded = np.unique(labels, return_inverse=True)     # return_inverse 就是重编号后的下标
    return encoded.astype(int)

def _contingency(y_true: Array, y_pred: Array) -> Tuple[Array, int, int]:
    """构造列联表：cont[i, j] = 真值为第 i 类、预测为第 j 类的样本数。
    混淆矩阵
    """
    y_true = _encode(y_true)
    y_pred = _encode(y_pred)
    n_true = int(y_true.max()) + 1
    n_pred = int(y_pred.max()) + 1
    cont = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(cont, (y_true, y_pred), 1)        # 必须用 add.at：普通 += 遇到重复下标只会生效一次
    return cont, n_true, n_pred

def _comb2(values: Array) -> Array:
    """组合数 C(v, 2)：v 个样本能组成多少对。ARI 的整个定义都建立在"样本对"上。"""
    values = np.asarray(values, dtype=float)
    return values * (values - 1.0) / 2.0

def clustering_accuracy(y_true: Array, y_pred: Array) -> float:
    """聚类准确率：先求标签的最优一一匹配（匈牙利问题），再算一致比例。

    类别数 ≤ 8 时穷举所有排列（最多 40320 种，瞬间算完）；
    类别更多时退化为 scipy 的匈牙利算法，再不行就用贪心匹配兜底。
    """
    y_true = _encode(y_true)
    y_pred = _encode(y_pred)
    k = max(int(y_true.max()), int(y_pred.max())) + 1
    cont, _, _ = _contingency(y_true, y_pred)
    if cont.shape[0] != cont.shape[1]:          # 两个方向的类别数不同时补齐成方阵
        size = max(cont.shape)
        square = np.zeros((size, size), dtype=np.int64)
        square[: cont.shape[0], : cont.shape[1]] = cont
        cont = square

    if k <= 8:                                  # ① 穷举所有标签映射，取命中最多的一种
        best = 0
        for perm in itertools.permutations(range(k)):
            best = max(best, int(cont[np.arange(k), list(perm)].sum()))
        return best / len(y_true)

    try:                                        # ② 类别多时用匈牙利算法（延迟导入，非强依赖）
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-cont)       # 取负号把"最大匹配"变成"最小代价"
        return float(cont[rows, cols].sum()) / len(y_true)
    except Exception:                           # ③ 兜底：贪心匹配
        work = cont.copy()
        total = 0
        for _ in range(k):
            i, j = np.unravel_index(np.argmax(work), work.shape)
            total += int(work[i, j])
            work[i, :] = -1                     # 该行该列作废
            work[:, j] = -1
        return total / len(y_true)

def adjusted_rand_index(y_true: Array, y_pred: Array) -> float:
    """调整兰德指数 ARI：1 表示完全一致，0 表示等同于随机划分，可为负。

    公式：(实际一致对 − 随机期望) / (最大可能 − 随机期望)
    除以分母相当于"扣掉瞎猜也能蒙对的部分"，因此对簇数不敏感。
    """
    y_true = _encode(y_true)
    y_pred = _encode(y_pred)
    cont, _, _ = _contingency(y_true, y_pred)

    sum_ij = _comb2(cont).sum()                         # 同一格内的样本对数
    sum_i = _comb2(cont.sum(axis=1)).sum()              # 真值同类的样本对数
    sum_j = _comb2(cont.sum(axis=0)).sum()              # 预测同类的样本对数
    total = _comb2(len(y_true))                         # 全部样本对数
    expected = sum_i * sum_j / total                    # 独立假设下的期望一致度
    max_index = 0.5 * (sum_i + sum_j)                   # 一致度的上界
    if max_index == expected:                           # 退化情形（例如全是一类）
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))

def normalized_mutual_info(y_true: Array, y_pred: Array) -> float:
    """标准化互信息 NMI = MI / sqrt(H(true) * H(pred))，取值 [0, 1]。

    互信息衡量两个划分共享多少信息；除以两个熵的几何平均是为了归一化，
    否则簇数越多互信息天然越大，指标就不可比了。
    """
    y_true = _encode(y_true)
    y_pred = _encode(y_pred)
    cont, _, _ = _contingency(y_true, y_pred)
    n = float(len(y_true))
    joint = cont / n                                    # 联合分布 p(i, j)
    p_true = joint.sum(axis=1)                          # 边缘分布 p(i)
    p_pred = joint.sum(axis=0)                          # 边缘分布 p(j)

    outer = p_true[:, None] * p_pred[None, :]           # 独立时的联合分布
    mask = joint > 0                                    # 只累加非零项（0·log0 约定为 0）
    mutual_info = float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))

    h_true = -float(np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0])))
    h_pred = -float(np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])))
    if h_true <= 0.0 or h_pred <= 0.0:                  # 某一侧没有不确定性，无法归一化
        return 1.0 if h_true == h_pred else 0.0
    return mutual_info / float(np.sqrt(h_true * h_pred))

def evaluate(y_true: Array, y_pred: Array) -> Dict[str, float]:
    """一次性算完所有指标，返回字典（标准做法：评估结果统一成 dict）。"""
    return {
        "accuracy": clustering_accuracy(y_true, y_pred),
        "ari": adjusted_rand_index(y_true, y_pred),
        "nmi": normalized_mutual_info(y_true, y_pred),
    }

if __name__ == "__main__":
    from utils import setup_console

    setup_console()                                     # 控制台改 UTF-8，避免中文乱码
    truth = np.array([0, 0, 1, 1, 2, 2])
    print("完全一致      :", evaluate(truth, truth))
    print("标签整体错位  :", evaluate(truth, np.array([2, 2, 0, 0, 1, 1])))
    print("完全打乱      :", evaluate(truth, np.array([0, 1, 0, 1, 2, 0])))






















