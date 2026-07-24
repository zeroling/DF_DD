"""原始 IDM 目标与分层拓扑扩展。

主流程使用原始 IDM 的类别特征中心平方距离、准确率加权分类交叉熵，以及本文保留的
浅/中/深三层空间拓扑。协方差、RBF-MMD 和多样性函数只保留给消融实验，默认权重为 0。
真实分支全部停止
梯度，合成分支保留计算图，因此这些损失最终只更新像素 logits 或扩散初始隐变量。
损失权重不写死在代码中，而是由 ``condensation.yaml`` 的 IPC 档位配置组合。
"""

from __future__ import annotations

from typing import Any, Mapping  # Mapping 允许直接传入 YAML 加载后的只读/普通字典。

import torch  # 张量运算、距离计算和自动微分。
import torch.nn.functional as F  # 归一化、均方误差和交叉熵等无状态函数。

from Net.Classification.features import ClassifierOutput  # 统一四类异构网络的输出结构。
from Net.Condensation.topology import multilevel_topology_losses  # 三层 RBF 亲和度约束。


def _zero_from(output: ClassifierOutput) -> torch.Tensor:
    """创建与合成计算图相连的标量零，避免禁用某项损失时断开反向传播。"""

    # 不能直接返回 torch.tensor(0)，否则它与 synthetic_output 没有梯度关系。
    return output.embedding.sum() * 0.0


def _idm_embedding(features: torch.Tensor) -> torch.Tensor:
    """返回原始 IDM 使用的未归一化 float32 特征。"""

    return features.float()


def _idm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算原始 IDM 分类正则使用的标准交叉熵。"""

    return F.cross_entropy(logits.float(), targets)


def _idm_mean_distance(
    real_features: torch.Tensor,
    synthetic_features: torch.Tensor,
) -> torch.Tensor:
    """计算两个类别特征中心之间的平方二范数。"""

    difference = synthetic_features.mean(0) - real_features.mean(0)
    return difference.square().sum()


def _covariance(features: torch.Tensor) -> torch.Tensor:
    """返回样本维度上的无偏特征协方差矩阵 ``[D,D]``。"""

    # 每个通道减去样本均值，得到零均值特征 [N,D]。
    centered = features - features.mean(dim=0, keepdim=True)
    # N=1 时调用方通常会跳过；max 仍保证此辅助函数不会出现除零。
    return centered.transpose(0, 1).matmul(centered) / max(1, features.shape[0] - 1)


def _median_kernel_width(real: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    """用真实与合成样本的联合距离中位数估计 RBF 核带宽。"""

    # 带宽仅是核尺度超参数，不需要通过其估计过程反向传播。
    combined = torch.cat((real.detach(), synthetic.detach()), dim=0)
    # cdist 返回欧氏距离；平方后与 exp(-||x-y||² / bandwidth) 的定义一致。
    distances = torch.cdist(combined, combined, p=2).square()
    # 排除对角线以及完全重合点产生的零距离，防止带宽退化为零。
    positive = distances[distances > 0]
    # 所有点重合时回退到 1；下界进一步保护指数除法的数值稳定性。
    return (positive.median() if positive.numel() else distances.new_tensor(1.0)).clamp_min(1.0e-6)


def _mmd(real: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    """计算带中位数带宽的有偏 RBF Maximum Mean Discrepancy。"""

    # 同一个带宽同时用于 RR、SS 和 RS，三个核期望才处于同一尺度。
    bandwidth = _median_kernel_width(real, synthetic)
    # 真实-真实核只提供固定目标，因此 real 在调用前已经 detach。
    kernel_rr = torch.exp(-torch.cdist(real, real).square() / bandwidth)
    # 合成-合成核保留梯度，用于约束合成分布内部结构。
    kernel_ss = torch.exp(-torch.cdist(synthetic, synthetic).square() / bandwidth)
    # 真实-合成交叉核将合成样本拉向真实分布支持区域。
    kernel_rs = torch.exp(-torch.cdist(real, synthetic).square() / bandwidth)
    # MMD² = E[k(r,r')] + E[k(s,s')] - 2E[k(r,s)]。
    return kernel_rr.mean() + kernel_ss.mean() - 2.0 * kernel_rs.mean()


def _diversity_penalty(features: torch.Tensor) -> torch.Tensor:
    """惩罚同类别合成样本的过高 RBF 相似度，缓解 IPC>1 时模式坍缩。"""

    # IPC=1 没有类内样本对，多样性无定义；返回与图相连的零即可。
    if features.shape[0] < 2:
        return features.sum() * 0.0
    # 构造全部样本对的平方欧氏距离矩阵 [N,N]。
    distances = torch.cdist(features, features).square()
    # 对角线表示样本与自身，必须去掉，否则固定的相似度 1 会污染损失。
    mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    off_diagonal = distances[mask]
    # 带宽停止梯度，避免优化器通过放大带宽本身“投机”降低目标。
    bandwidth = off_diagonal.detach().median().clamp_min(1.0e-6)
    # 相似样本的 RBF 值接近 1；最小化该值会鼓励合成样本覆盖不同模式。
    return torch.exp(-off_diagonal / bandwidth).mean()


def classwise_losses(
    real_output: ClassifierOutput,
    synthetic_output: ClassifierOutput,
    real_labels: torch.Tensor,
    synthetic_labels: torch.Tensor,
    num_classes: int,
    topology_settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """一次性按类别计算分布、分类与拓扑损失。

    该接口适合显存允许同时前向全部类别的情况。每个类别先独立计算，再做等权平均，
    因而真实数据中的类别数量差异不会让多数类主导合成梯度。
    """

    # 所有暂未累积的项目都从同一个可微零开始。
    zero = _zero_from(synthetic_output)
    # CE 天然按所有合成样本平均；其余项目在下面逐类累积后再除以类别数。
    values = {
        "mean": zero,
        "cross_entropy": _idm_cross_entropy(synthetic_output.logits, synthetic_labels),
        "covariance": zero,
        "mmd": zero,
        "diversity": zero,
        "topology_shallow": zero,
        "topology_middle": zero,
        "topology_deep": zero,
    }
    # 显式遍历完整类别编号，保证每类都对蒸馏目标贡献相同权重。
    for class_id in range(int(num_classes)):
        # 分别构造真实与合成样本的类别布尔掩码。
        real_mask = real_labels == class_id
        synthetic_mask = synthetic_labels == class_id
        # 缺类会让均值匹配失去定义，通常表示 balanced sampler 或标签构造配置错误。
        if not torch.any(real_mask) or not torch.any(synthetic_mask):
            raise ValueError(f"分布匹配 batch 缺少类别 {class_id}")
        # 真实特征只作为目标，明确 detach；合成特征必须保留到图像/隐变量的梯度。
        real_embedding = _idm_embedding(real_output.embedding[real_mask].detach())
        synthetic_embedding = _idm_embedding(synthetic_output.embedding[synthetic_mask])
        # IDM 的主项：匹配该类别在当前在线网络特征空间中的一阶中心。
        values["mean"] = values["mean"] + _idm_mean_distance(
            real_embedding,
            synthetic_embedding,
        )
        # 二阶统计和样本对分布至少需要两个真实与两个合成样本；IPC=1 自动跳过。
        if synthetic_embedding.shape[0] >= 2 and real_embedding.shape[0] >= 2:
            # 协方差约束补充仅匹配均值时丢失的方向相关性。
            values["covariance"] = values["covariance"] + F.mse_loss(
                _covariance(synthetic_embedding), _covariance(real_embedding)
            )
            # MMD 使用核均值嵌入匹配更高阶、非线性的分布差异。
            values["mmd"] = values["mmd"] + _mmd(real_embedding, synthetic_embedding)
            # 多样性项仅作用于合成样本，防止同类多张图收敛到同一个原型。
            values["diversity"] = values["diversity"] + _diversity_penalty(synthetic_embedding)
        # 三层空间特征仍保留 [B,C,H,W]，按类别切片后计算通道无关的 RBF 拓扑。
        real_spatial = {name: feature[real_mask].detach() for name, feature in real_output.spatial.items()}
        synthetic_spatial = {name: feature[synthetic_mask] for name, feature in synthetic_output.spatial.items()}
        topology = multilevel_topology_losses(real_spatial, synthetic_spatial, topology_settings)
        # topology 返回 shallow/middle/deep 三项，将当前类别结果累积到统一命名空间。
        for level, loss in topology.items():
            values[f"topology_{level}"] = values[f"topology_{level}"] + loss
    # 除 CE 外都按类别等权平均；CE 已由 PyTorch 按全部合成样本平均。
    for key in values:
        if key != "cross_entropy":
            values[key] = values[key] / float(num_classes)
    # 返回未加权的各分量，便于 YAML 按 IPC 配置权重并在日志中单独观察。
    return values


def single_class_losses(
    real_output: ClassifierOutput,
    synthetic_output: ClassifierOutput,
    synthetic_class_id: int,
    topology_settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """计算单个类别的损失，是 224×224 医学图像蒸馏的低显存接口。

    主流程逐类别取真实数据、逐类别解码合成隐变量并立即反传，这样不用同时保存全部
    类别和四个异构网络的激活。外层会再对类别及架构梯度进行等权合并。
    """

    # 创建所有可选项共享的可微零。
    zero = _zero_from(synthetic_output)
    # 原始 IDM 直接匹配未归一化特征；真实分支固定、合成分支可微。
    real_embedding = _idm_embedding(real_output.embedding.detach())
    synthetic_embedding = _idm_embedding(synthetic_output.embedding)
    # 单类别调用没有传标签张量，因此按样本数生成该类别的监督标签。
    targets = torch.full(
        (synthetic_embedding.shape[0],),
        int(synthetic_class_id),
        device=synthetic_embedding.device,
        dtype=torch.long,
    )
    # IPC=1 默认只启用均值、CE 和拓扑；其他分量先用可微零占位。
    values = {
        "mean": _idm_mean_distance(real_embedding, synthetic_embedding),
        "cross_entropy": _idm_cross_entropy(synthetic_output.logits, targets),
        "covariance": zero,
        "mmd": zero,
        "diversity": zero,
    }
    # IPC≥2 且真实批次也至少有两个样本时，二阶及多样性统计才有意义。
    if synthetic_embedding.shape[0] >= 2 and real_embedding.shape[0] >= 2:
        values["covariance"] = F.mse_loss(
            _covariance(synthetic_embedding), _covariance(real_embedding)
        )
        values["mmd"] = _mmd(real_embedding, synthetic_embedding)
        values["diversity"] = _diversity_penalty(synthetic_embedding)
    # 拓扑损失读取四种网络统一输出的 shallow/middle/deep 空间特征。
    topology = multilevel_topology_losses(
        {name: feature.detach() for name, feature in real_output.spatial.items()},
        synthetic_output.spatial,
        topology_settings,
    )
    # 给三层结果增加 topology_ 前缀，与 YAML 的权重键保持一致。
    values.update({f"topology_{level}": value for level, value in topology.items()})
    return values


def weighted_total(
    losses: Mapping[str, torch.Tensor],
    profile: Mapping[str, float],
    expert_reliability: float = 1.0,
) -> torch.Tensor:
    """按当前 IPC 的 YAML 权重组合损失；仅 CE 使用在线模型可靠性。

    分布匹配和拓扑匹配在随机/早期模型上依然有价值，因此不能因分类准确率低而关闭；
    只有需要可信类别判断的交叉熵乘以可靠性 EMA，并设置 0.05 下限以保留微弱监督。
    """

    # 从任意损失构造可微零，避免 Python 数字 0 破坏设备、dtype 或计算图。
    total = next(iter(losses.values())).sum() * 0.0
    # 未在 profile 中声明的分量权重默认为 0，方便用户仅启用需要的约束。
    for name, loss in losses.items():
        weight = float(profile.get(name, 0.0))
        # 只校准 CE；随机网络的特征分布损失仍以原始 YAML 权重参与。
        if name == "cross_entropy":
            weight *= max(0.05, float(expert_reliability))
        # 即使 weight=0 也保留统一表达式，逻辑更易审计且不会改变结果。
        total = total + weight * loss
    return total


def detached_loss_values(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """把损失张量转换成可写入 JSON/日志的普通浮点数字典。"""

    # detach 阻断日志分支持有计算图；item 将单元素张量搬到 Python 标量。
    return {name: float(value.detach().item()) for name, value in losses.items()}
