"""支持固定 timm 型号或逐阶段自定义的 ConvNeXt 层级特征封装。"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from Net.Classification.features import ClassifierOutput


class ConvNeXtExpert(nn.Module):
    """把 ConvNeXt 四个 stage 封装成统一浅/中/深空间特征接口。"""

    def __init__(self, backbone: nn.Module) -> None:
        """接收已经按 YAML 构造且随机初始化的 timm ConvNeXt 主干。"""

        super().__init__()
        # 保存完整主干，分类头仍由 timm 实现。
        self.backbone = backbone
        # 当前特征封装依赖标准 ConvNeXt 的四阶段接口。
        if not hasattr(self.backbone, "stages") or len(self.backbone.stages) < 4:
            raise TypeError("配置模型不是受支持的四阶段 ConvNeXt")
        # num_features 是分类头前全局 embedding 维度。
        self.feature_dim = int(self.backbone.num_features)

    def forward_with_features(self, images: torch.Tensor) -> ClassifierOutput:
        """执行一次主干前向并保留 stage0、stage2、stage3 特征。"""

        # stem 完成 patchify 和初始归一化。
        feature = self.backbone.stem(images)
        # stages 保存四个阶段输出，通道数不需要与其他架构相同。
        stages: list[torch.Tensor] = []
        # 顺序执行所有 ConvNeXt stage。
        for stage in self.backbone.stages:
            feature = stage(feature)
            stages.append(feature)
        # norm_pre 是进入分类头前的最终空间归一化。
        feature = self.backbone.norm_pre(feature)
        # pre_logits=True 返回全局 embedding。
        embedding = self.backbone.forward_head(feature, pre_logits=True)
        # pre_logits=False 返回最终类别 logits。
        logits = self.backbone.forward_head(feature, pre_logits=False)
        # stage0 保留较细网格，stage2/3 提供中深层语义。
        return ClassifierOutput(
            logits,
            embedding,
            {"shallow": stages[0], "middle": stages[2], "deep": stages[3]},
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """返回标准 IDM 使用的全局 embedding。"""

        return self.forward_with_features(images).embedding

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """标准分类前向只返回 logits。"""

        # 在线真实训练不需要中间特征，走 timm 原生路径还能真正启用内部梯度检查点。
        return self.backbone(images)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """在 timm 主干支持时启用内部激活重计算。"""

        # 不同 timm 版本可能没有该方法，因此使用安全的动态查询。
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        # 只有可调用时才转发配置。
        if callable(setter):
            setter(bool(enabled))


def build_model(
    num_classes: int,
    in_chans: int = 3,
    pretrained: bool = False,
    **arguments,
) -> ConvNeXtExpert:
    """按 custom 开关构造可调 ConvNeXt 或固定 timm 型号。"""

    # 本项目不允许下载或加载外部 ConvNeXt 权重。
    if pretrained:
        raise ValueError("在线异构队列要求 ConvNeXt 从零训练")
    # custom=true 时直接调用 timm ConvNeXt 类，使层宽和深度真正可调。
    if bool(arguments.get("custom", True)):
        from timm.models.convnext import ConvNeXt

        # kernel_sizes 可以是一个整数，也可以是逐阶段序列。
        kernel_sizes = arguments.get("kernel_sizes", 7)
        # YAML 列表显式转成 tuple 以匹配 timm 接口。
        if isinstance(kernel_sizes, list):
            kernel_sizes = tuple(map(int, kernel_sizes))
        # 创建完全随机初始化的自定义主干。
        backbone = ConvNeXt(
            in_chans=int(in_chans),
            num_classes=int(num_classes),
            depths=tuple(map(int, arguments.get("depths", [3, 3, 9, 3]))),
            dims=tuple(map(int, arguments.get("dims", [96, 192, 384, 768]))),
            kernel_sizes=kernel_sizes,
            patch_size=int(arguments.get("patch_size", 4)),
            drop_rate=float(arguments.get("drop_rate", 0.0)),
            drop_path_rate=float(arguments.get("drop_path_rate", 0.1)),
            ls_init_value=float(arguments.get("layer_scale_init_value", 1.0e-6)),
        )
    else:
        # 固定型号模式便于在消融实验中快速切换 convnext_atto 等 timm 变体。
        backbone = timm.create_model(
            str(arguments.get("model_name", "convnext_tiny")),
            pretrained=False,
            in_chans=int(in_chans),
            num_classes=int(num_classes),
            drop_rate=float(arguments.get("drop_rate", 0.0)),
            drop_path_rate=float(arguments.get("drop_path_rate", 0.1)),
        )
    # 返回统一特征接口包装器。
    return ConvNeXtExpert(backbone)
