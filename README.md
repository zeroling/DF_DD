# Size-Weighted Cluster IDM

当前项目在严格 IDM 基线上研究高 IPC 的类内覆盖。活跃代码只包含像素 K-means 初始化、按簇规模加权的分簇均值匹配和簇内二阶覆盖；不包含通道扩展、拓扑或轨迹匹配。

## 三组完整实验

| 组别 | K-means 中心到边缘初始化 | 分簇均值匹配 | 半径分位数/std 覆盖 |
|---|---:|---:|---:|
| A | 是 | 否 | 否 |
| B | 是 | 是，簇规模加权、权重 2.0 | 否 |
| C | 是 | 是，簇规模加权、权重 2.0 | 是，15% 图像梯度预算 |

每类划分 `IPC` 个像素-PCA K-means 簇，每张存储合成图固定代表一簇。2×2 P&E 的四个 RGB 视图按到簇中心的距离分位点初始化，从中心覆盖到边缘。每个簇无论大小都固定拥有一个合成代表，因此少见形态不会消失；真实 batch 数量和损失贡献再按簇内真实样本数加权，避免把极小簇错误放大到与主簇同等概率。

分簇均值损失相对原实现乘以 2.0。二阶覆盖仍匹配半径分位数与逐维标准差，冲突梯度先投影，再固定占图像梯度预算的 15%。CE、动态队列、每类一次合成图更新、DSA、P&E 和梯度累积保持 IDM 语义；像素投影默认关闭。

## 一条命令运行 A–C

以下命令会顺序运行 PathMNIST IPC=5 的 A、B、C：每组蒸馏 20000 轮，随后按协议执行 ConvNet、ResNet18、VGG11、AlexNet 各 5 次最终评估。

```powershell
.\.venv\Scripts\python.exe run_experiment.py --dataset pathmnist --ipc 5 --cluster-ablation-suite --stage all --iterations 20000
```

输出互相隔离在：

```text
outputs/cluster_weighted_ablation/A/pathmnist/
outputs/cluster_weighted_ablation/B/pathmnist/
outputs/cluster_weighted_ablation/C/pathmnist/
```

只做蒸馏时把 `--stage all` 改成 `--stage condense`。单组可使用 `--ablation-variant A|B|C`。重复相同命令会从兼容的 v8 断点恢复。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_experiment.py --dataset pathmnist --ipc 5 --cluster-ablation-suite --stage condense --smoke
```

