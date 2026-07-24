"""四种异构分类器共享的 logits、语义嵌入与层级空间特征接口。

每种网络内部通道数和空间分辨率均可不同。适配器只保证名称和张量秩一致；分布匹配
在同一网络的真实/合成 embedding 内完成，RBF 拓扑也只在同一网络同一层内完成，
因此不存在跨架构通道投影，也不会把 ViT token 与 CNN 特征直接相减。
"""

from __future__ import annotations

from dataclasses import dataclass  # 用明确结构替代易混淆的多元素 tuple。

import torch  # 输出张量类型和分类器参数冻结。


# 所有分类器必须返回这三个语义层级；具体层索引由各网络 YAML 单独配置。
FEATURE_LEVELS = ("shallow", "middle", "deep")


@dataclass
class ClassifierOutput:
    """一次前向中供 IDM 与拓扑匹配共同使用的全部输出。

    ``spatial`` 中每层统一为 ``[B,C,H,W]``。通道 C 可以因网络和层而不同；
    RBF 亲和矩阵只在各层内部沿通道计算距离，因此不要求异构网络通道对齐。
    """

    logits: torch.Tensor  # 分类 logits，形状 [B,num_classes]。
    embedding: torch.Tensor  # 全局语义向量，形状 [B,D]，用于 IDM 分布统计。
    spatial: dict[str, torch.Tensor]  # 三层 [B,C_l,H_l,W_l]，用于 RBF 亲和度。


def validate_classifier_output(output: ClassifierOutput) -> None:
    """在首次集成新网络时尽早发现接口或形状错误。"""

    # logits 和全局 embedding 都必须保留 batch 维与特征维。
    if output.logits.ndim != 2 or output.embedding.ndim != 2:
        raise ValueError("logits 和 embedding 必须是 [B,D]")
    # 键集合必须恰好是三层，防止配置拼写错误或缺层被静默跳过。
    if set(output.spatial) != set(FEATURE_LEVELS):
        raise ValueError(f"spatial 必须包含 {FEATURE_LEVELS}")
    # 所有输出必须对应同一批输入样本。
    batch_size = output.logits.shape[0]
    # 空间层只统一为四维 NCHW，不限制 C/H/W 的具体数值。
    for name, feature in output.spatial.items():
        if feature.ndim != 4 or feature.shape[0] != batch_size:
            raise ValueError(f"{name} 特征必须是 [B,C,H,W]，实际={tuple(feature.shape)}")


def freeze_classifier(model: torch.nn.Module) -> torch.nn.Module:
    """冻结权重但保留对输入图像的梯度，用于蒸馏阶段。"""

    # eval 固定 BatchNorm/Dropout 行为，但 autograd 仍跟踪输入图像。
    model.eval()
    # 队列成员本轮指导合成图时不积累参数梯度；下一次真实训练前会重新解冻。
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # 返回同一对象便于调用方链式使用。
    return model
