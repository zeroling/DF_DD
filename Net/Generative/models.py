"""MONAI AutoencoderKL、PatchDiscriminator 与类别条件扩散 U-Net 工厂。

本模块只使用 MONAI 的网络定义，不下载或加载任何 MONAI/第三方预训练权重。
"""

from __future__ import annotations

from typing import Any, Mapping  # 接收拆分 YAML 合并后的嵌套配置。

import torch  # 模型基类与参数冻结操作。


def _monai_networks():
    """延迟导入 MONAI 网络，未运行生成阶段时不强制初始化其依赖。"""

    # 放在函数内导入，使纯配置检查和数据层测试不必提前初始化 MONAI。
    try:
        from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
    except ImportError as error:
        raise ImportError(
            "生成模块需要 MONAI。请在当前环境执行：pip install 'monai>=1.4,<1.7'"
        ) from error
    # 返回类本身而非实例，三个工厂分别按 YAML 创建全新随机模型。
    return AutoencoderKL, DiffusionModelUNet, PatchDiscriminator


def build_autoencoder(config: Mapping[str, Any]) -> torch.nn.Module:
    """按唯一配置从零创建二维 AutoencoderKL。"""

    # 本项目从零训练，不在这里读取任何权重路径或联网下载模型。
    AutoencoderKL, _, _ = _monai_networks()
    # autoencoder 阶段配置包含网络结构；图像通道来自全局数据配置。
    settings = config["autoencoder"]
    return AutoencoderKL(
        # 处理二维医学切片而非三维体数据。
        spatial_dims=2,
        # 编码器输入与解码器输出均遵循全局图像通道数，默认 RGB=3。
        in_channels=int(config["data"]["image"].get("channels", 3)),
        out_channels=int(config["data"]["image"].get("channels", 3)),
        # 每个分辨率阶段残差块数可以逐层配置。
        num_res_blocks=tuple(map(int, settings["num_res_blocks"])),
        # channels 同时决定层宽和下采样阶段数。
        channels=tuple(map(int, settings["channels"])),
        # attention_levels 与 channels 等长，逐阶段控制自注意力。
        attention_levels=tuple(map(bool, settings["attention_levels"])),
        # 潜变量通道数也是扩散 U-Net 的输入/输出通道数。
        latent_channels=int(settings["latent_channels"]),
        # GroupNorm 组数必须能整除相应层通道，配置加载时已校验。
        norm_num_groups=int(settings.get("norm_num_groups", 32)),
        # MONAI 内部激活检查点以更多计算换显存。
        use_checkpoint=bool(settings.get("gradient_checkpointing", False)),
    )


def build_patch_discriminator(config: Mapping[str, Any]) -> torch.nn.Module | None:
    """创建可选 PatchGAN；关闭配置时返回 None。"""

    # 使用同一 MONAI 安装中的判别器定义。
    _, _, PatchDiscriminator = _monai_networks()
    # 对抗配置是 autoencoder 阶段的子节点。
    settings = config["autoencoder"].get("adversarial", {})
    # enabled=false 时训练阶段会完全跳过判别器优化器和损失。
    if not bool(settings.get("enabled", True)):
        return None
    return PatchDiscriminator(
        # 二维 PatchGAN 输出空间真伪图而非单个标量。
        spatial_dims=2,
        # 基础判别器宽度可调，后续层由 MONAI 递增。
        channels=int(settings.get("channels", 64)),
        # 输入是真实或 VAE 重建图像，通道与数据一致。
        in_channels=int(config["data"]["image"].get("channels", 3)),
        # 每个空间 patch 输出一个真伪 logit 通道。
        out_channels=1,
        # 判别卷积层数可按显存/分辨率调节。
        num_layers_d=int(settings.get("num_layers", 3)),
    )


def build_diffusion_unet(config: Mapping[str, Any], num_classes: int) -> torch.nn.Module:
    """创建共享的类别条件隐空间 U-Net，并额外保留一个空类别做 CFG。"""

    # 只使用 MONAI 网络结构，权重从随机初始化开始训练。
    _, DiffusionModelUNet, _ = _monai_networks()
    # 扩散 YAML 定义 U-Net 层宽/深度/注意力，VAE YAML 定义隐通道数。
    settings = config["diffusion"]
    latent_channels = int(config["autoencoder"]["latent_channels"])
    return DiffusionModelUNet(
        # 潜空间仍是二维特征图。
        spatial_dims=2,
        # 噪声预测与输入潜变量形状完全相同。
        in_channels=latent_channels,
        out_channels=latent_channels,
        # 每个尺度残差块数、通道宽度和注意力开关均可逐层调整。
        num_res_blocks=tuple(map(int, settings["num_res_blocks"])),
        channels=tuple(map(int, settings["channels"])),
        attention_levels=tuple(map(bool, settings["attention_levels"])),
        # GroupNorm 组数由配置控制。
        norm_num_groups=int(settings.get("norm_num_groups", 32)),
        # 每层注意力头通道数；缺失时按每层 32 构造。
        num_head_channels=tuple(map(int, settings.get("num_head_channels", [32] * len(settings["channels"])))),
        # 真实类别编号 0...C-1，额外编号 C 表示 CFG 的无条件类别。
        num_class_embeds=int(num_classes) + 1,
    )


def latent_spatial_size(config: Mapping[str, Any]) -> tuple[int, int]:
    """根据 AutoencoderKL 的下采样阶段数推导隐空间高宽。"""

    # 局部导入避免 Core.config 在模块加载阶段形成不必要依赖链。
    from Core.config import image_size

    # 获取全局可调输入高宽。
    height, width = image_size(config)
    # MONAI AutoencoderKL 在除首层外的每个 channels 阶段下采样 2 倍。
    factor = 2 ** (len(config["autoencoder"]["channels"]) - 1)
    # 不能整除会导致编码/解码形状不一致，训练前直接报配置错误。
    if height % factor or width % factor:
        raise ValueError(f"图像尺寸 {(height, width)} 必须能被 VAE 下采样倍率 {factor} 整除")
    # 返回 (latent_height, latent_width)。
    return height // factor, width // factor


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    """冻结网络权重，但仍允许梯度穿过网络流向输入隐变量。"""

    # eval 固定 Dropout/归一化行为，但不会关闭 autograd。
    module.eval()
    # requires_grad=False 避免为几千万生成网络参数存梯度；输入梯度仍然保留。
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    # 返回同一对象，便于链式赋值。
    return module
