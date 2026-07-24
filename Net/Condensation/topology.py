"""由 Lite-MyoNet ST-GTP 改造而来的通用空间 RBF 拓扑匹配。

保留的核心：通道归一化、节点两两欧氏距离、Gaussian RBF、去除自环、行概率
归一化和分布散度。删除的任务特定部分：步态时间差分、人体关节点和任何解剖区域。
本项目的节点只是在不同网络特征图上自适应池化得到的规则空间网格。
"""

from __future__ import annotations

from typing import Mapping, Sequence  # Sequence 兼容 YAML 列表和 Python 元组网格。

import torch  # 张量、批量矩阵乘法及自动微分。
import torch.nn.functional as F  # 自适应池化和 L2 归一化。


def feature_nodes(feature: torch.Tensor, grid: Sequence[int]) -> torch.Tensor:
    """把任意 CNN/ViT 空间特征统一为 L2 归一化的 ``[B,N,C]`` 节点。

    四种异构网络的通道数和原始空间分辨率可以不同；只要它们各自返回四维特征图，
    自适应池化就能在“同一网络的真实/合成分支”内建立固定数量的空间节点。这里从不
    跨网络直接比较通道，所以无需把 ConvNet、ResNet、ConvNeXt 和 ViT 投影到同一维度。
    """

    # 工厂会将 ViT patch token 还原为 [B,C,H,W]；收到其他形状说明适配器有错误。
    if feature.ndim != 4:
        raise ValueError(f"空间特征必须是 [B,C,H,W]，实际={tuple(feature.shape)}")
    # YAML 中网格写成 [height, width]，在此转换为明确的整数。
    grid_h, grid_w = int(grid[0]), int(grid[1])
    # 0 或负尺寸无法传给 adaptive_avg_pool2d，提前给出可读的配置错误。
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("拓扑网格尺寸必须为正数")
    # RBF 的 exp/log 对半精度较敏感，因此拓扑统计固定使用 float32；类型转换保留梯度。
    pooled = F.adaptive_avg_pool2d(feature.float(), (grid_h, grid_w))
    # [B,C,Hg,Wg] 先展平空间，再转置为 [B,N,C]；N=Hg×Wg。
    nodes = pooled.flatten(2).transpose(1, 2)
    # 通道向量归一化后，亲和度只描述节点特征方向，不受通道数或激活模长控制。
    return F.normalize(nodes, p=2, dim=-1, eps=1.0e-8)


def squared_node_distances(nodes: torch.Tensor) -> torch.Tensor:
    """计算节点两两平方欧氏距离，不依赖原始通道维度的数值尺度。"""

    # 对已归一化向量，2-2<x,y> 比 cdist 更省显存且数值范围固定在 [0,4]。
    # 批量矩阵乘法得到每张图内部所有 N×N 节点的余弦相似度。
    similarity = torch.matmul(nodes, nodes.transpose(-1, -2))
    # 浮点误差可能给出极小负距离，因此用 clamp_min(0) 修正理论下界。
    return (2.0 - 2.0 * similarity).clamp_min(0.0)


def median_bandwidth(distances: torch.Tensor, minimum: float = 1.0e-4) -> torch.Tensor:
    """用真实特征的非对角距离中位数设定 RBF 带宽，并停止其梯度。"""

    # 最后一维是节点数；距离张量通常为 [B,N,N]。
    node_count = distances.shape[-1]
    # 单节点没有非对角距离，直接使用可配置下界。
    if node_count <= 1:
        return distances.new_tensor(float(minimum))
    # False 对角线掩码去掉每个节点与自身恒为 0 的距离。
    mask = ~torch.eye(node_count, dtype=torch.bool, device=distances.device)
    # 该索引保留 batch 中所有非对角样本，形成用于稳健估计的一维集合。
    values = distances[..., mask]
    # 额外排除完全相同节点的零距离，避免中位数在退化特征上变成零。
    positive = values[values > 0]
    # 若所有节点完全一致，则回退到 minimum；否则采用抗离群值的中位数。
    bandwidth = positive.median() if positive.numel() else values.new_tensor(float(minimum))
    # 带宽来自真实目标统计，不允许合成图通过操纵带宽降低损失。
    return bandwidth.detach().clamp_min(float(minimum))


def rbf_row_distribution(
    distances: torch.Tensor,
    bandwidth: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """应用 Gaussian RBF，显式去掉对角自环，再对每一行归一化。"""

    # 本实现的 bandwidth 对应平方距离尺度，因此核为 exp(-d² / bandwidth)。
    affinity = torch.exp(-distances / bandwidth.clamp_min(epsilon))
    # 获取 N，用于构建每个 batch 共用的 N×N 非对角掩码。
    node_count = affinity.shape[-1]
    # True 表示保留节点间边，False 表示删除无信息的自环边。
    diagonal_mask = ~torch.eye(node_count, dtype=torch.bool, device=affinity.device)
    # 布尔掩码转换为 affinity dtype 后逐元素相乘，兼容 AMP 张量。
    affinity = affinity * diagonal_mask.to(affinity.dtype)
    # 每一行归一化为从当前节点到其他节点的邻接概率分布。
    return affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def topology_distribution(
    feature: torch.Tensor,
    grid: Sequence[int],
    bandwidth: torch.Tensor | None = None,
    minimum_bandwidth: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回一个 batch 的平均空间亲和概率矩阵与实际 RBF 带宽。

    真实分支不传 ``bandwidth``，由真实节点距离估计；合成分支必须复用这个带宽，
    从而比较的是同一个核空间，不能各自选择最有利的尺度。
    """

    # 特征依次经过网格池化、节点归一化和两两距离计算。
    distances = squared_node_distances(feature_nodes(feature, grid))
    # 仅真实分支估计带宽；合成分支由调用者显式传入真实带宽。
    if bandwidth is None:
        bandwidth = median_bandwidth(distances, minimum_bandwidth)
    # 先把每张图的邻接矩阵转为行概率，再沿 batch 求类别平均拓扑。
    distribution = rbf_row_distribution(distances, bandwidth).mean(dim=0)
    # 浮点平均后再次行归一化，确保后续 KL/JS 输入满足概率和为 1。
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return distribution, bandwidth


def probability_divergence(
    real: torch.Tensor,
    synthetic: torch.Tensor,
    kind: str = "js",
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """计算对称 JS、前向 KL 或逐边 MSE；默认 JS 更适合双向匹配。"""

    # log(0) 未定义；截断只影响完全为零或数值下溢的边概率。
    real = real.clamp_min(epsilon)
    synthetic = synthetic.clamp_min(epsilon)
    # 配置值不区分大小写，统一转为小写分支判断。
    kind = str(kind).lower()
    # MSE 直接对齐每条归一化亲和边，适合作为不含对数的消融基线。
    if kind == "mse":
        return F.mse_loss(synthetic, real)
    # 前向 KL(real || synthetic) 强调覆盖真实拓扑中存在的边。
    if kind == "kl":
        return (real * (real.log() - synthetic.log())).sum(dim=-1).mean()
    # 只接受明确支持的散度，拼写错误不能静默回退。
    if kind != "js":
        raise ValueError("topology.divergence 只能是 js、kl 或 mse")
    # Jensen-Shannon 使用两者混合分布，有限且对真实/合成方向对称。
    middle = 0.5 * (real + synthetic)
    # 对每个节点行计算 KL(real || middle)。
    real_kl = (real * (real.log() - middle.log())).sum(dim=-1)
    # 对同一节点行计算 KL(synthetic || middle)。
    synthetic_kl = (synthetic * (synthetic.log() - middle.log())).sum(dim=-1)
    # 两个方向等权，再对 N 个节点行取平均得到标量。
    return 0.5 * (real_kl + synthetic_kl).mean()


def topology_loss(
    real_feature: torch.Tensor,
    synthetic_feature: torch.Tensor,
    grid: Sequence[int],
    divergence: str = "js",
    minimum_bandwidth: float = 1.0e-4,
) -> torch.Tensor:
    """在同一网络、同一层内匹配真实与合成数据的平均 RBF 拓扑。

    注意这里不会把不同网络的特征直接相减。每个网络只比较自己的真实/合成亲和矩阵，
    最后由外层把四个网络产生的标量损失或输入梯度组合，因此通道维度完全可以不同。
    """

    # 真实特征只提供固定目标；同时由其节点距离估计共享带宽。
    real_distribution, bandwidth = topology_distribution(
        real_feature.detach(), grid, minimum_bandwidth=minimum_bandwidth
    )
    # 合成特征保留梯度，并严格复用真实分支带宽。
    synthetic_distribution, _ = topology_distribution(
        synthetic_feature, grid, bandwidth=bandwidth, minimum_bandwidth=minimum_bandwidth
    )
    # 真实概率再次 detach，明确唯一可优化对象是合成图像/隐变量。
    return probability_divergence(real_distribution.detach(), synthetic_distribution, divergence)


def multilevel_topology_losses(
    real_features: Mapping[str, torch.Tensor],
    synthetic_features: Mapping[str, torch.Tensor],
    settings: Mapping,
) -> dict[str, torch.Tensor]:
    """分别计算浅、中、深三层拓扑，不进行跨架构或跨层通道投影。"""

    # 当前设计必须由真实分支决定带宽，其他策略尚未实现，不能静默忽略配置。
    if str(settings.get("bandwidth", "real_median")).lower() != "real_median":
        raise ValueError("当前稳定实现只支持 topology.bandwidth=real_median")
    # 每层可配置不同空间网格，例如浅层 8×8、中层 4×4、深层 2×2。
    grids = settings.get("grids", {})
    # 散度可在 YAML 中选择稳定对称的 JS 或前向 KL。
    divergence = str(settings.get("divergence", "js"))
    # 带宽下界用于避免特征坍缩时除零。
    minimum = float(settings.get("minimum_bandwidth", 1.0e-4))
    # 用普通字典按固定层级名称返回，便于损失权重独立配置。
    results: dict[str, torch.Tensor] = {}
    # 四类网络的输出适配器都保证存在以下三个键。
    for level in ("shallow", "middle", "deep"):
        results[level] = topology_loss(
            real_features[level],
            synthetic_features[level],
            grids[level],
            divergence=divergence,
            minimum_bandwidth=minimum,
        )
    return results
