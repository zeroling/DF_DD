"""四种分类器的统一构造、在线优化器和带余弦退火的评估训练策略。"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any, Mapping

import torch


# 架构名与实现模块的唯一映射，配置和断点都使用左侧稳定名称。
_MODEL_MODULES = {
    "convnet": "Net.Classification.convnet",
    "resnet18": "Net.Classification.resnet18",
    "convnext_tiny": "Net.Classification.convnext",
    "vit_tiny": "Net.Classification.vit",
}


def available_models() -> list[str]:
    """返回按字典序排列的全部可用分类器名称。"""

    return sorted(_MODEL_MODULES)


def _module(name: str):
    """按稳定架构名延迟导入模型模块，减少无关阶段启动开销。"""

    # 配置名称统一转成小写。
    normalized = str(name).lower()
    # 未知名称应在真正创建模型前给出可选项。
    if normalized not in _MODEL_MODULES:
        raise ValueError(f"未知分类网络 {normalized!r}；可选值={available_models()}")
    # 延迟导入对应模块。
    return import_module(_MODEL_MODULES[normalized])


def build_classifier(
    architecture_name: str,
    num_classes: int,
    in_chans: int,
    image_size: tuple[int, int],
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
    pretrained: bool = False,
) -> torch.nn.Module:
    """根据架构名与全局 definitions 构建一个随机初始化分类器。"""

    # 名称统一后用于查询模型专属参数。
    name = str(architecture_name).lower()
    # 复制参数，防止后续插入 image_size 修改原始配置。
    arguments = dict((definitions or {}).get(name, {}))
    # 所有模型都收到当前数据图像尺寸，ViT 必须用它创建位置编码。
    arguments["image_size"] = tuple(map(int, image_size))
    # 调用模型模块统一暴露的 build_model。
    return _module(name).build_model(
        num_classes=int(num_classes),
        in_chans=int(in_chans),
        pretrained=bool(pretrained),
        **arguments,
    )


def build_classifier_from_config(
    config: Mapping[str, Any],
    architecture_name: str,
    num_classes: int,
) -> torch.nn.Module:
    """从合并后的项目配置构建指定分类器。"""

    # 延迟导入避免 Core.config 与模型工厂形成模块级循环依赖。
    from Core.config import image_size

    # 把全局数据和模型参数传给通用构造函数。
    return build_classifier(
        architecture_name=architecture_name,
        num_classes=int(num_classes),
        in_chans=int(config["data"]["image"].get("channels", 3)),
        image_size=image_size(config),
        definitions=config["models"].get("definitions", {}),
        pretrained=bool(config["models"].get("pretrained", False)),
    )


def build_optimizer(
    model: torch.nn.Module,
    settings: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """按配置为在线队列或评估分类器创建 SGD/Adam/AdamW。"""

    # 优化器名称统一为小写。
    optimizer_name = str(settings.get("optimizer", "adamw")).lower()
    # 所有优化器共享初始学习率与权重衰减字段。
    learning_rate = float(settings.get("learning_rate", settings.get("lr", 1.0e-3)))
    weight_decay = float(settings.get("weight_decay", 0.0))
    # SGD 支持动量和 Nesterov，适合 ConvNet/ResNet 在线推进。
    if optimizer_name == "sgd":
        momentum = float(settings.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=bool(settings.get("nesterov", True)) and momentum > 0.0,
        )
    # Adam 与 AdamW 共用 beta 和 epsilon 参数。
    betas = (float(settings.get("beta1", 0.9)), float(settings.get("beta2", 0.999)))
    epsilon = float(settings.get("epsilon", 1.0e-8))
    # Adam 使用耦合权重衰减。
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
    # AdamW 使用解耦权重衰减，是 ConvNeXt/ViT 默认选择。
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
    # 配置校验通常会更早发现，该错误保护直接调用场景。
    raise ValueError(f"不支持的优化器 {optimizer_name!r}；可选 sgd/adam/adamw")


def build_training_policy(
    architecture_name: str,
    model: torch.nn.Module,
    settings: Mapping[str, Any],
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """创建评估分类器优化器与按当前目标 epoch 可重建的余弦学习率计划。"""

    # architecture_name 保留在签名中用于调用语义和未来架构专属策略扩展。
    del architecture_name
    # 优化器完全由当前阶段配置决定。
    optimizer = build_optimizer(model, settings)
    # 初始学习率从优化器读取，避免配置别名造成不一致。
    learning_rate = float(optimizer.param_groups[0]["lr"])
    # 最低学习率决定余弦曲线末端比例。
    minimum_lr = float(settings.get("min_learning_rate", 1.0e-6))
    minimum_factor = min(1.0, minimum_lr / max(learning_rate, 1.0e-12))

    def multiplier(epoch: int) -> float:
        """返回给定 epoch 的余弦学习率倍率。"""

        # progress 限制在 [0,1]，目标轮数变小时续训也不会得到负倍率。
        progress = min(1.0, max(0.0, float(epoch) / max(1, int(epochs))))
        # 标准半周期余弦从 1 平滑下降到 0。
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        # 抬高曲线底部以满足 minimum_lr。
        return minimum_factor + (1.0 - minimum_factor) * cosine

    # LambdaLR 的状态可以根据新目标轮数重新定位，符合无哈希续训要求。
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    # 返回优化器和调度器供训练阶段保存断点。
    return optimizer, scheduler
