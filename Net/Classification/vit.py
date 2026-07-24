"""支持固定 timm 型号或逐项自定义的 ViT 层级 token 特征封装。"""

from __future__ import annotations

from typing import Mapping

import timm
import torch
import torch.nn as nn

from Net.Classification.features import ClassifierOutput


class ViTExpert(nn.Module):
    """去掉前缀 token 后，把不同深度 patch token 恢复为 ``[B,C,H,W]``。"""

    def __init__(self, backbone: nn.Module, feature_indices: Mapping[str, int] | None = None) -> None:
        """保存随机初始化 ViT，并确定浅、中、深 Transformer block 索引。"""

        super().__init__()
        # 主干负责 patch embedding、Transformer blocks、分类头与位置编码。
        self.backbone = backbone
        # 至少三个 block 才能提取三个不同深度层级。
        depth = len(self.backbone.blocks)
        if depth < 3:
            raise ValueError("ViT 至少需要三个 Transformer block")
        # 默认索引大致位于 1/4、1/2 与最后一个 block。
        defaults = {
            "shallow": max(0, depth // 4 - 1),
            "middle": max(1, depth // 2 - 1),
            "deep": depth - 1,
        }
        # YAML 可以显式选择任意合法 block。
        self.feature_indices = {
            name: int(index) for name, index in dict(feature_indices or defaults).items()
        }
        # 下游统一要求三个固定层名。
        if set(self.feature_indices) != {"shallow", "middle", "deep"}:
            raise ValueError("ViT feature_indices 必须恰好包含 shallow/middle/deep")
        # 检查所有索引都落在实际 block 数量内。
        if any(index < 0 or index >= depth for index in self.feature_indices.values()):
            raise ValueError(f"ViT feature_indices 必须位于 [0,{depth - 1}]")
        # num_features 是分类头前 token embedding 的通道数。
        self.feature_dim = int(self.backbone.num_features)

    def _tokens_to_map(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """移除 CLS/注册 token，并按输入尺寸和 patch 大小恢复二维网格。"""

        # num_prefix_tokens 同时兼容 CLS token 和可能的注册 token。
        prefix_count = int(getattr(self.backbone, "num_prefix_tokens", 1))
        # 仅 patch token 对应真实空间位置。
        patch_tokens = tokens[:, prefix_count:]
        # timm 可能把 patch_size 保存为整数或二元组。
        patch_size = self.backbone.patch_embed.patch_size
        # 分别解析 patch 高度和宽度。
        patch_height = int(patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size)
        patch_width = int(patch_size[1] if isinstance(patch_size, (tuple, list)) else patch_size)
        # 输入高宽除以 patch 大小得到 token 网格尺寸。
        grid_height = images.shape[-2] // patch_height
        grid_width = images.shape[-1] // patch_width
        # token 数量不匹配通常表示图像尺寸或动态 padding 配置错误。
        if patch_tokens.shape[1] != grid_height * grid_width:
            raise ValueError(
                f"ViT patch token 数量 {patch_tokens.shape[1]} 无法恢复为 "
                f"{grid_height}×{grid_width} 网格"
            )
        # 把 [B,N,C] 转置并 reshape 成统一 [B,C,H,W]。
        return patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            tokens.shape[2],
            grid_height,
            grid_width,
        )

    def forward_with_features(self, images: torch.Tensor) -> ClassifierOutput:
        """手动展开 timm forward_features，以便在指定 block 截取 token。"""

        # patch_embed 把图像变成 patch token 序列。
        tokens = self.backbone.patch_embed(images)
        # _pos_embed 添加位置编码和 CLS 等前缀 token。
        tokens = self.backbone._pos_embed(tokens)
        # patch_drop 根据训练配置随机丢弃 patch；默认概率为 0。
        tokens = self.backbone.patch_drop(tokens)
        # norm_pre 在启用 pre-norm 变体时生效，否则通常是 Identity。
        tokens = self.backbone.norm_pre(tokens)
        # spatial 保存三个深度恢复后的空间 token 图。
        spatial: dict[str, torch.Tensor] = {}
        # 反向索引用于按 block 序号查询层名。
        reverse_indices = {index: name for name, index in self.feature_indices.items()}
        # 顺序执行 Transformer blocks。
        for index, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            # 只为配置指定的 block 保存空间特征。
            if index in reverse_indices:
                spatial[reverse_indices[index]] = self._tokens_to_map(tokens, images)
        # 最终 LayerNorm 与 timm 标准前向一致。
        tokens = self.backbone.norm(tokens)
        # 若 deep 指向最后一个 block，用归一化后的 token 覆盖未归一化版本。
        if self.feature_indices["deep"] == len(self.backbone.blocks) - 1:
            spatial["deep"] = self._tokens_to_map(tokens, images)
        # pre_logits=True 返回池化后的全局 embedding。
        embedding = self.backbone.forward_head(tokens, pre_logits=True)
        # pre_logits=False 返回分类 logits。
        logits = self.backbone.forward_head(tokens, pre_logits=False)
        # 封装统一接口。
        return ClassifierOutput(logits, embedding, spatial)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """返回标准 IDM 使用的全局 embedding。"""

        return self.forward_with_features(images).embedding

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """标准分类前向只返回 logits。"""

        # 在线真实训练不需要层级 token，走 timm 原生路径以降低激活峰值。
        return self.backbone(images)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """启用或关闭 timm Transformer block 的激活重计算。"""

        self.backbone.set_grad_checkpointing(bool(enabled))


def build_model(
    num_classes: int,
    in_chans: int = 3,
    pretrained: bool = False,
    **arguments,
) -> ViTExpert:
    """按 custom 开关构造可调 ViT 或固定 timm 型号。"""

    # 在线轨迹和最终评估均从随机初始化开始。
    if pretrained:
        raise ValueError("在线异构队列要求 ViT 从零训练")
    # 统一解析输入图像尺寸和 patch 大小。
    image_size = tuple(map(int, arguments.get("image_size", (224, 224))))
    patch_size = int(arguments.get("patch_size", 16))
    # custom=true 时直接使用 VisionTransformer 类，让 YAML 结构参数全部生效。
    if bool(arguments.get("custom", True)):
        from timm.models.vision_transformer import VisionTransformer

        # 创建随机初始化的可调 ViT。
        backbone = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=int(in_chans),
            num_classes=int(num_classes),
            embed_dim=int(arguments.get("embed_dim", 192)),
            depth=int(arguments.get("depth", 12)),
            num_heads=int(arguments.get("num_heads", 3)),
            mlp_ratio=float(arguments.get("mlp_ratio", 4.0)),
            qkv_bias=bool(arguments.get("qkv_bias", True)),
            drop_rate=float(arguments.get("drop_rate", 0.0)),
            attn_drop_rate=float(arguments.get("attention_drop_rate", 0.0)),
            drop_path_rate=float(arguments.get("drop_path_rate", 0.1)),
        )
    else:
        # 固定型号模式便于消融时切换 timm 注册的轻量 ViT。
        backbone = timm.create_model(
            str(arguments.get("model_name", "vit_tiny_patch16_224")),
            pretrained=False,
            in_chans=int(in_chans),
            num_classes=int(num_classes),
            img_size=image_size,
            patch_size=patch_size,
            drop_rate=float(arguments.get("drop_rate", 0.0)),
            drop_path_rate=float(arguments.get("drop_path_rate", 0.1)),
        )
    # 返回统一空间特征封装。
    return ViTExpert(backbone, feature_indices=arguments.get("feature_indices"))
