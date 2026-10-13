# 手写谱聚类（Spectral Clustering）

从零实现的谱聚类算法，**核心部分只依赖 numpy**，不调用任何机器学习库。

## 效果

在三个经典非线性数据集上，聚类结果与真实标签**完全一致**：

| 数据集 | 簇数 k | 准确率 | ARI | NMI |
|---|---|---|---|---|
| 双半月 two_moons | 2 | 1.000 | 1.000 | 1.000 |
| 同心圆 circles | 2 | 1.000 | 1.000 | 1.000 |
| 高斯簇 blobs | 3 | 1.000 | 1.000 | 1.000 |

并且与 `sklearn.cluster.SpectralClustering` 的划分**一致性 = 1.000**（在相同数据、相同参数下）。

## 算法流程

```
原始数据 X (n×d)
   │  ① 构图        build_affinity       → 亲和矩阵 W
   │  ② 建拉普拉斯   compute_laplacian    → 拉普拉斯矩阵 L
   │  ③ 谱嵌入      spectral_embedding   → 嵌入矩阵 U (n×k)
   │  ④ 离散化      KMeans               → 簇标签
   ▼
labels (n,)
```

**为什么有效**：图论结论指出，若图恰由 k 个互不相连的块组成，则 L 的前 k 个特征值全为 0，
对应特征向量正是这 k 个块的指示向量。真实数据的图不会严格断开，但块间连接很弱，
前 k 个特征向量会**近似**成为指示向量，K-Means 在 U 上就变得极其容易。

## 目录结构

```
.
├── config.py        配置：路径 + 全部超参数（dataclass）
├── utils.py         工具：日志 / 目录 / json / 计时
├── data.py          数据：三个手写数据集 + CSV 读取
├── metrics.py       指标：聚类准确率 / ARI / NMI
├── model.py         核心：谱聚类 + K-Means（只依赖 numpy）
├── visualize.py     可视化：对比图 + 特征值谱图
├── run.py           实验流程编排
├── main.py          命令行入口
└── outputs/         运行产物（自动创建，已 gitignore）
    ├── figures/     图片
    └── results/     指标 json
```

## 快速开始

```bash
pip install -r requirements.txt

python main.py                                    # 默认：双半月，k=2
python main.py --all                              # 三个数据集全跑
python main.py --all --compare                    # 额外与 sklearn 对照
python main.py --all --estimate-k                 # 自动估计簇数
python main.py --dataset circles --laplacian rw   # 指定数据集与拉普拉斯类型
python main.py --help                             # 查看全部参数
```

## 作为库使用

```python
from data import get_dataset
from model import SpectralClustering
from metrics import evaluate

X, y = get_dataset("two_moons", n_samples=300, noise=0.06)

model = SpectralClustering(n_clusters=2, affinity="knn", n_neighbors=10, laplacian="sym")
labels = model.fit_predict(X)

print(evaluate(y, labels))       # {'accuracy': 1.0, 'ari': 1.0, 'nmi': 1.0}
print(model.embedding_.shape)    # (300, 2)  谱嵌入
print(model.eigenvalues_)        # 前 k 个最小特征值
```

## 三种拉普拉斯矩阵

| `laplacian` | 公式 | 说明 |
|---|---|---|
| `none` | L = D − W | 非归一化，对簇的大小和密度敏感 |
| `sym` | L = I − D^(-1/2) W D^(-1/2) | 对称归一化（NJW），默认 |
| `rw` | L = I − D^(-1) W | 随机游走归一化（Ncut） |

后两者是**相似矩阵**，特征值完全相同，因此得到的划分通常一致。

## 关键超参数

| 参数 | 默认 | 调大 | 调小 |
|---|---|---|---|
| `--n-neighbors` | 10 | 图更连通、抗噪，但簇间"桥"变多 | 边界更锐利，但可能产生孤立点和碎片 |
| `--sigma` | 自动 | 相似度衰减慢，趋近全连接 | 衰减快，图容易碎成很多块 |
| `--laplacian` | `sym` | — | 簇大小接近时 `none` 也够用 |
| `--n-init` | 10 | 更不容易陷入坏的局部最优，更慢 | 设 1 最快，但结果可能不稳定 |

## 关于簇数 k 的估计

`--estimate-k` 会启用三种策略之一：

- `gap`：经典绝对特征值间隙，谱缓慢上升时容易高估 k
- `relative`：相对间隙，对尺度不敏感
- `auto`（默认）：先数数值为 0 的特征值（= 连通分量数，理论精确），否则回退到 `relative`

即便自动估计，也建议看一眼 `outputs/figures/spectrum_*.png` 的拐点再下结论。

## 依赖

- Python ≥ 3.8
- numpy（必需）
- matplotlib（必需，仅画图时用）
- scikit-learn（可选，仅 `--compare` 时用）
- scipy（可选，仅类别数 > 8 时的匈牙利匹配用）

## 已知限制

- `predict()` 是**近似**的：谱聚类的特征分解建立在整张图上，新样本会改变图结构，
  严格意义上的 out-of-sample 预测并不存在。这里用"指派给最近的训练样本"近似。
- 使用稠密矩阵，复杂度 O(n²) 内存、O(n³) 时间，适合 n 在几千以内的数据；
  更大规模需改用稀疏矩阵与 Lanczos 迭代（如 `scipy.sparse.linalg.eigsh`）。
- `visualize.plot_clustering` 只支持二维数据出图。

