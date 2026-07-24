"""可调宽度、块数、stem 与归一化方式的 ResNet-18 风格分类器。"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from Net.Classification.features import ClassifierOutput


def _normalization(kind: str, channels: int, group_count: int) -> nn.Module:
    """根据配置创建 BatchNorm 或不依赖 batch 统计量的 GroupNorm。"""

    # 标准 ResNet-18 默认使用 BatchNorm。
    if str(kind).lower() == "batch":
        return nn.BatchNorm2d(int(channels))
    # 小 batch 在线训练可选择 GroupNorm。
    if str(kind).lower() == "group":
        groups = max(1, math.gcd(int(channels), max(1, int(group_count))))
        return nn.GroupNorm(groups, int(channels))
    # 拒绝未实现归一化以避免网络悄悄改变。
    raise ValueError(f"ResNet normalization 不支持 {kind!r}；可选 batch/group")


class BasicBlock(nn.Module):
    """标准两层 3×3 ResNet BasicBlock，支持任意输入/输出通道。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        norm_factory: Callable[[int], nn.Module],
    ) -> None:
        """创建残差主分支与必要的投影捷径。"""

        super().__init__()
        # 第一层同时完成可选空间降采样和通道变换。
        self.conv1 = nn.Conv2d(
            int(input_channels),
            int(output_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=False,
        )
        # 第一层归一化参数量由输出通道决定。
        self.norm1 = norm_factory(int(output_channels))
        # ReLU 复用标准 ResNet 的非线性定义。
        self.relu = nn.ReLU(inplace=True)
        # 第二层保持空间尺寸与通道数不变。
        self.conv2 = nn.Conv2d(
            int(output_channels),
            int(output_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # 第二层归一化位于残差相加之前。
        self.norm2 = norm_factory(int(output_channels))
        # 输入形状变化时使用 1×1 投影，否则直接使用恒等捷径。
        self.shortcut: nn.Module = (
            nn.Identity()
            if int(stride) == 1 and int(input_channels) == int(output_channels)
            else nn.Sequential(
                nn.Conv2d(
                    int(input_channels),
                    int(output_channels),
                    kernel_size=1,
                    stride=int(stride),
                    bias=False,
                ),
                norm_factory(int(output_channels)),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """执行两层残差变换并与捷径相加。"""

        # 保存投影或恒等捷径结果。
        identity = self.shortcut(inputs)
        # 主分支第一层卷积、归一化和激活。
        output = self.relu(self.norm1(self.conv1(inputs)))
        # 主分支第二层卷积和归一化，不在相加前激活。
        output = self.norm2(self.conv2(output))
        # 残差相加后执行最终 ReLU。
        return self.relu(output + identity)


class ResNet18Expert(nn.Module):
    """四阶段可调 ResNet，并暴露 layer1/layer3/layer4 空间特征。"""

    def __init__(
        self,
        num_classes: int,
        in_chans: int,
        stage_widths: list[int],
        stage_blocks: list[int],
        stem_width: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        use_max_pool: bool = True,
        normalization: str = "batch",
        group_norm_groups: int = 32,
        zero_init_residual: bool = False,
    ) -> None:
        """按四阶段通道和块数构建 ResNet 主干。"""

        # 四阶段接口是下游浅/中/深特征定义的基础。
        if len(stage_widths) != 4 or len(stage_blocks) != 4:
            raise ValueError("ResNet stage_widths 和 stage_blocks 必须各有四项")
        # 所有通道和块数必须为正。
        if any(int(value) <= 0 for value in [*stage_widths, *stage_blocks]):
            raise ValueError("ResNet stage_widths 和 stage_blocks 必须全部大于 0")
        # stem 卷积核要求正奇数，以便使用对称 padding。
        if int(stem_kernel_size) <= 0 or int(stem_kernel_size) % 2 == 0:
            raise ValueError("ResNet stem_kernel_size 必须为正奇数")
        super().__init__()
        # 闭包统一创建每层归一化模块。
        norm_factory = lambda channels: _normalization(
            normalization,
            channels,
            group_norm_groups,
        )
        # stem 首层卷积接收任意 1/3 通道医学图像。
        self.stem = nn.Sequential(
            nn.Conv2d(
                int(in_chans),
                int(stem_width),
                kernel_size=int(stem_kernel_size),
                stride=int(stem_stride),
                padding=int(stem_kernel_size) // 2,
                bias=False,
            ),
            norm_factory(int(stem_width)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_max_pool else nn.Identity(),
        )
        # current_channels 跟踪每个阶段入口通道。
        current_channels = int(stem_width)

        def make_stage(output_channels: int, block_count: int, stride: int) -> nn.Sequential:
            """创建一个残差阶段；第一块负责降采样，后续块保持形状。"""

            nonlocal current_channels
            blocks: list[nn.Module] = []
            # 第一块可能改变空间尺寸和通道数。
            blocks.append(BasicBlock(current_channels, output_channels, stride, norm_factory))
            current_channels = int(output_channels)
            # 剩余块保持同一宽度与分辨率。
            for _ in range(1, int(block_count)):
                blocks.append(BasicBlock(current_channels, output_channels, 1, norm_factory))
            # Sequential 便于像标准 ResNet 一样逐阶段前向。
            return nn.Sequential(*blocks)

        # 第一阶段不额外降采样。
        self.layer1 = make_stage(int(stage_widths[0]), int(stage_blocks[0]), stride=1)
        # 后三阶段的第一块各自进行 2 倍降采样。
        self.layer2 = make_stage(int(stage_widths[1]), int(stage_blocks[1]), stride=2)
        self.layer3 = make_stage(int(stage_widths[2]), int(stage_blocks[2]), stride=2)
        self.layer4 = make_stage(int(stage_widths[3]), int(stage_blocks[3]), stride=2)
        # 自适应池化允许输入尺寸改变而无需修改分类头。
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 最后阶段宽度就是 embedding 维度。
        self.feature_dim = int(stage_widths[-1])
        # 线性分类头输出数据集类别数。
        self.head = nn.Linear(self.feature_dim, int(num_classes))
        # 使用 ResNet 常规初始化，确保自定义结构仍有稳定起点。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # 可选零初始化残差分支末端，使网络初始更接近恒等映射。
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.norm2.weight)

    def forward_with_features(self, images: torch.Tensor) -> ClassifierOutput:
        """返回分类 logits、全局 embedding 与三个残差深度的空间特征。"""

        # stem 提取初始局部特征。
        feature = self.stem(images)
        # layer1 作为浅层特征。
        shallow = self.layer1(feature)
        # layer2 是浅层与中层之间的过渡阶段。
        feature = self.layer2(shallow)
        # layer3 作为中层特征。
        middle = self.layer3(feature)
        # layer4 作为深层语义特征。
        deep = self.layer4(middle)
        # 全局池化得到 embedding。
        embedding = self.pool(deep).flatten(1)
        # 分类头得到 logits。
        logits = self.head(embedding)
        # 统一封装下游所需输出。
        return ClassifierOutput(
            logits,
            embedding,
            {"shallow": shallow, "middle": middle, "deep": deep},
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """返回标准 IDM 使用的全局 embedding。"""

        return self.forward_with_features(images).embedding

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """标准分类前向只返回 logits。"""

        return self.forward_with_features(images).logits


def build_model(
    num_classes: int,
    in_chans: int = 3,
    pretrained: bool = False,
    **arguments,
) -> ResNet18Expert:
    """从统一配置构建可调 ResNet-18 风格网络。"""

    # 在线轨迹和最终评估均禁止外部预训练权重。
    if pretrained:
        raise ValueError("在线异构队列要求 ResNet 从零训练")
    # 默认列表严格对应标准 ResNet-18。
    return ResNet18Expert(
        num_classes=int(num_classes),
        in_chans=int(in_chans),
        stage_widths=[int(value) for value in arguments.get("stage_widths", [64, 128, 256, 512])],
        stage_blocks=[int(value) for value in arguments.get("stage_blocks", [2, 2, 2, 2])],
        stem_width=int(arguments.get("stem_width", 64)),
        stem_kernel_size=int(arguments.get("stem_kernel_size", 7)),
        stem_stride=int(arguments.get("stem_stride", 2)),
        use_max_pool=bool(arguments.get("use_max_pool", True)),
        normalization=str(arguments.get("normalization", "batch")),
        group_norm_groups=int(arguments.get("group_norm_groups", 32)),
        zero_init_residual=bool(arguments.get("zero_init_residual", False)),
    )
