"""可逐层调宽度的 IDM 风格 GroupNorm ConvNet。

网络不依赖 batch 统计量，适合数据集浓缩中的小批量和单样本前向。每个卷积块都
可以从全局 YAML 调整通道数，浅层、中层、深层特征索引也可以独立指定。
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from Net.Classification.features import ClassifierOutput


def _activation(name: str) -> nn.Module:
    """把配置中的激活函数名称转换成无状态 PyTorch 模块。"""

    # 所有名称转成小写，允许用户在 YAML 中使用 GELU/Gelu 等写法。
    normalized = str(name).lower()
    # GELU 是默认选择，和旧版 ConvNet 行为一致。
    if normalized == "gelu":
        return nn.GELU()
    # ReLU 便于复现传统 DC/DM 网络。
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    # SiLU 在部分医学图像任务中比 ReLU 更平滑。
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    # 未实现的名称直接报错，避免静默回退到错误激活。
    raise ValueError(f"ConvNet activation 不支持 {name!r}；可选 gelu/relu/silu")


def _group_count(channels: int, requested: int) -> int:
    """返回不超过期望值且能整除通道数的 GroupNorm 组数。"""

    # gcd 能在 requested 大于 channels 或不能整除时给出安全因子。
    return max(1, math.gcd(int(channels), max(1, int(requested))))


class ConvNetExpert(nn.Module):
    """输出分类 logits、全局 embedding 和浅/中/深三层空间特征。"""

    def __init__(
        self,
        in_chans: int,
        num_classes: int,
        widths: list[int],
        kernel_size: int = 3,
        group_norm_groups: int = 8,
        activation: str = "gelu",
        pool_kernel_size: int = 2,
        feature_indices: Mapping[str, int] | None = None,
    ) -> None:
        """按逐层通道列表构建卷积块与分类头。"""

        # 三层拓扑要求网络至少能提供三个不同深度位置。
        if len(widths) < 3:
            raise ValueError("ConvNet widths 至少需要三项以提取浅/中/深特征")
        # 卷积核使用对称 same padding，因此要求正奇数。
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("ConvNet kernel_size 必须为正奇数")
        # 池化核至少为 1；1 会被实现成 Identity。
        if int(pool_kernel_size) <= 0:
            raise ValueError("ConvNet pool_kernel_size 必须大于 0")
        # 初始化父类后才能注册子模块。
        super().__init__()
        # 保存卷积块到 ModuleList，便于前向时按索引抓取中间特征。
        blocks: list[nn.Module] = []
        # 第一块输入通道来自数据配置。
        current_channels = int(in_chans)
        # padding 保证卷积本身不改变空间尺寸。
        padding = int(kernel_size) // 2
        # 每个 widths 元素生成一个完整卷积块。
        for output_channels in map(int, widths):
            # 通道必须为正，否则无法构造卷积和归一化。
            if output_channels <= 0:
                raise ValueError("ConvNet widths 中的每个通道数都必须大于 0")
            # pool=1 时不改变分辨率，否则使用平均池化完成平滑降采样。
            pooling: nn.Module = (
                nn.Identity()
                if int(pool_kernel_size) == 1
                else nn.AvgPool2d(int(pool_kernel_size), int(pool_kernel_size))
            )
            # 一个块依次执行卷积、GroupNorm、激活和可选平均池化。
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        output_channels,
                        int(kernel_size),
                        padding=padding,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(output_channels, group_norm_groups), output_channels),
                    _activation(activation),
                    pooling,
                )
            )
            # 下一块输入通道等于当前块输出通道。
            current_channels = output_channels
        # ModuleList 确保所有块参数被 PyTorch 正确注册。
        self.blocks = nn.ModuleList(blocks)
        # 自适应池化把任意输入尺寸变成一个全局特征向量。
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 最后一层通道数就是分类 embedding 维度。
        self.feature_dim = int(widths[-1])
        # 线性头把全局 embedding 映射到数据集类别数。
        self.head = nn.Linear(self.feature_dim, int(num_classes))
        # 默认索引覆盖第一块、中间块和最后一块。
        default_indices = {
            "shallow": 0,
            "middle": max(1, len(widths) // 2),
            "deep": len(widths) - 1,
        }
        # 用户显式配置时整体覆盖默认索引。
        self.feature_indices = {
            name: int(index) for name, index in dict(feature_indices or default_indices).items()
        }
        # 三个层名缺一不可，否则下游拓扑损失无法统一调用。
        if set(self.feature_indices) != {"shallow", "middle", "deep"}:
            raise ValueError("ConvNet feature_indices 必须恰好包含 shallow/middle/deep")
        # 所有索引必须落在现有卷积块范围内。
        if any(index < 0 or index >= len(self.blocks) for index in self.feature_indices.values()):
            raise ValueError("ConvNet feature_indices 存在越界索引")

    def forward_with_features(self, images: torch.Tensor) -> ClassifierOutput:
        """一次前向同时返回分类输出与三层空间特征，避免重复计算。"""

        # feature 从原始分类器输入开始逐块更新。
        feature = images
        # spatial 保存下游 RBF 拓扑需要的三层特征。
        spatial: dict[str, torch.Tensor] = {}
        # 反转映射便于在 enumerate 循环中按块索引查询层名。
        reverse_indices = {index: name for name, index in self.feature_indices.items()}
        # 顺序执行全部卷积块。
        for index, block in enumerate(self.blocks):
            feature = block(feature)
            # 只保存配置指定的三个位置，控制显存占用。
            if index in reverse_indices:
                spatial[reverse_indices[index]] = feature
        # 全局平均池化得到每张图一个 embedding。
        embedding = self.pool(feature).flatten(1)
        # 分类头产生类别 logits。
        logits = self.head(embedding)
        # 使用统一数据结构返回结果。
        return ClassifierOutput(logits, embedding, spatial)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """兼容标准 IDM 只需要全局 embedding 的接口。"""

        return self.forward_with_features(images).embedding

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """标准分类前向只返回 logits。"""

        return self.forward_with_features(images).logits


def build_model(
    num_classes: int,
    in_chans: int = 3,
    pretrained: bool = False,
    **arguments,
) -> ConvNetExpert:
    """从统一模型工厂参数构建 ConvNet，兼容旧 width/depth 覆盖写法。"""

    # 项目禁止为队列和评估分类器加载外部预训练权重。
    if pretrained:
        raise ValueError("ConvNet 没有预训练权重，本项目也要求从零训练")
    # 新配置优先使用逐层 widths；旧测试传 width/depth 时自动展开成等宽列表。
    widths = arguments.get("widths")
    if widths is None:
        widths = [int(arguments.get("width", 128))] * int(arguments.get("depth", 4))
    # 把 YAML 参数逐项传给模型构造器。
    return ConvNetExpert(
        in_chans=int(in_chans),
        num_classes=int(num_classes),
        widths=[int(value) for value in widths],
        kernel_size=int(arguments.get("kernel_size", 3)),
        group_norm_groups=int(arguments.get("group_norm_groups", 8)),
        activation=str(arguments.get("activation", "gelu")),
        pool_kernel_size=int(arguments.get("pool_kernel_size", 2)),
        feature_indices=arguments.get("feature_indices"),
    )
