"""训练阶段共享的归一化、数值范围转换与可微张量增强。

数据加载层统一输出 ``[0,1]`` RGB 张量；VAE 使用 ``[-1,1]``；分类网络使用全局
配置中的 mean/std。所有转换集中在这里，避免真实图、扩散生成图和蒸馏合成图
使用不同预处理。张量增强完全由 PyTorch 运算组成，因此合成图像梯度可以穿过它。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence  # Sequence 同时接受 YAML 列表和 Python 元组。

import torch  # 张量、随机采样与设备无关运算。
import torch.nn.functional as F  # 可微仿射网格、采样与插值。


def normalize_for_classifier(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """把 ``[0,1]`` 图像按分类器统计量归一化，保持完整梯度。"""

    # 分类器只接受批量 NCHW；提前检查能定位数据适配器输出错误。
    if images.ndim != 4:
        raise ValueError(f"分类归一化要求 [B,C,H,W]，实际={tuple(images.shape)}")
    # new_tensor 自动继承图像 device/dtype，reshape 后可广播到 batch 和空间维。
    mean_tensor = images.new_tensor(list(mean)).view(1, -1, 1, 1)
    std_tensor = images.new_tensor(list(std)).view(1, -1, 1, 1)
    # 通道数必须相等，且标准差必须严格为正。
    if mean_tensor.shape[1] != images.shape[1] or torch.any(std_tensor <= 0):
        raise ValueError("归一化 mean/std 与输入通道不匹配，或 std 非正")
    # 纯张量广播运算不会阻断合成图对隐变量的梯度。
    return (images - mean_tensor) / std_tensor


def classifier_normalize(images: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    """从项目配置读取分类归一化参数。"""

    # 全局数据配置是唯一统计量来源，四个异构网络共享同一输入定义。
    normalization = config["data"]["classifier_normalization"]
    return normalize_for_classifier(images, normalization["mean"], normalization["std"])


def to_generator_range(images: torch.Tensor) -> torch.Tensor:
    """把数据层的 ``[0,1]`` 映射到 VAE 使用的 ``[-1,1]``。"""

    # 线性映射 y=2x-1；不做 clamp，以免隐藏数据加载范围错误或截断梯度。
    return images.mul(2.0).sub(1.0)


def from_generator_range(images: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """把 VAE 输出的 ``[-1,1]`` 映射回统一像素空间。"""

    # 先做逆映射 x=(y+1)/2。
    images = images.add(1.0).mul(0.5)
    # 预览/分类默认限制到合法像素范围；需要观察 VAE 越界时可显式关闭。
    return images.clamp(0.0, 1.0) if clamp else images


class TensorBatchAugment:
    """在 GPU 上对真实或合成 batch 做轻量、可微的随机增强。

    该增强不包含任何器官或病灶区域假设。空间变换按图像独立采样，适用于不同疾病
    和不同成像数据集。输入和输出均为 ``[0,1]``。
    """

    def __init__(
        self,
        enabled: bool = True,
        horizontal_flip_probability: float = 0.5,
        brightness: float = 0.08,
        contrast: float = 0.10,
        translate_ratio: float = 0.04,
        rotation_degrees: float = 7.0,
        scale_range: Sequence[float] = (0.95, 1.05),
    ):
        # 总开关关闭时 __call__ 原样返回输入，不进行任何随机数采样。
        self.enabled = bool(enabled)
        # 每张图独立采样水平翻转，概率范围由配置校验层约束。
        self.flip_probability = float(horizontal_flip_probability)
        # 负的颜色扰动没有定义，安全截断到 0 表示禁用。
        self.brightness = max(0.0, float(brightness))
        self.contrast = max(0.0, float(contrast))
        # 平移量使用 affine_grid 的归一化坐标比例，而不是固定像素数。
        self.translate_ratio = max(0.0, float(translate_ratio))
        # 旋转配置单位是角度，调用时转换成弧度。
        self.rotation_degrees = max(0.0, float(rotation_degrees))
        # 缩放范围保存为二元浮点元组，便于与 (1,1) 快速比较是否禁用。
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        # 缩放必须为正，且最小值不能大于最大值。
        if self.scale_range[0] <= 0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("TensorBatchAugment.scale_range 必须满足 0 < min <= max")

    @classmethod
    def from_config(cls, augmentation: Mapping[str, Any] | None) -> "TensorBatchAugment":
        """从某阶段的 ``augmentation`` YAML 节点构造增强器。"""

        # None 转为空字典；缺失字段使用与构造函数一致的温和默认值。
        settings = dict(augmentation or {})
        return cls(
            enabled=settings.get("enabled", True),
            horizontal_flip_probability=settings.get("horizontal_flip_probability", 0.5),
            brightness=settings.get("brightness", 0.08),
            contrast=settings.get("contrast", 0.10),
            translate_ratio=settings.get("translate_ratio", 0.04),
            rotation_degrees=settings.get("rotation_degrees", 7.0),
            scale_range=settings.get("scale_range", [0.95, 1.05]),
        )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """对一个 NCHW batch 执行逐图随机、可微增强。"""

        # 仿射网格和颜色参数都按 batch 第一维采样，要求四维 NCHW 输入。
        if images.ndim != 4:
            raise ValueError(f"张量增强要求 [B,C,H,W]，实际={tuple(images.shape)}")
        # 关闭增强时保留原张量对象，避免无意义复制。
        if not self.enabled:
            return images
        # result 随各项增强依次更新；初始值与输入共享计算图。
        result = images
        # 每张图拥有独立随机参数，而不是整个 batch 共用一次变换。
        batch_size = images.shape[0]
        # 水平翻转通过 where 选择，两个分支对输入都可微。
        if self.flip_probability > 0:
            flip_mask = torch.rand(batch_size, 1, 1, 1, device=images.device) < self.flip_probability
            result = torch.where(flip_mask, result.flip(-1), result)
        # 亮度是在全部通道/像素上增加每图独立的标量偏移。
        if self.brightness > 0:
            delta = torch.empty(batch_size, 1, 1, 1, device=images.device, dtype=images.dtype).uniform_(
                -self.brightness, self.brightness
            )
            result = result + delta
        # 对比度围绕每张图每通道的空间均值缩放，避免无意改变均值亮度。
        if self.contrast > 0:
            spatial_mean = result.mean(dim=(2, 3), keepdim=True)
            factor = 1.0 + torch.empty(
                batch_size, 1, 1, 1, device=images.device, dtype=images.dtype
            ).uniform_(-self.contrast, self.contrast)
            result = (result - spatial_mean) * factor + spatial_mean
        # 任一几何增强启用时，统一构造一个仿射矩阵，避免多次重采样造成模糊。
        if (
            self.translate_ratio > 0
            or self.rotation_degrees > 0
            or self.scale_range != (1.0, 1.0)
        ):
            # 从每张图的 2×3 单位仿射矩阵开始。
            theta = torch.eye(2, 3, device=images.device, dtype=images.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            # 角度在对称范围内均匀采样，然后转成弧度供 sin/cos 使用。
            angles = torch.empty(batch_size, device=images.device, dtype=images.dtype).uniform_(
                -self.rotation_degrees, self.rotation_degrees
            ) * (torch.pi / 180.0)
            # 每张图独立采样各向同性缩放比例。
            scales = torch.empty(batch_size, device=images.device, dtype=images.dtype).uniform_(
                self.scale_range[0], self.scale_range[1]
            )
            # 旋转与缩放组合成二维线性部分。
            cosine = torch.cos(angles) * scales
            sine = torch.sin(angles) * scales
            theta[:, 0, 0] = cosine
            theta[:, 0, 1] = -sine
            theta[:, 1, 0] = sine
            theta[:, 1, 1] = cosine
            # affine_grid 的平移使用 [-1,1] 归一化坐标；两个方向分别采样。
            theta[:, 0, 2].uniform_(-self.translate_ratio, self.translate_ratio)
            theta[:, 1, 2].uniform_(-self.translate_ratio, self.translate_ratio)
            # 生成与输入同尺寸的逆向采样网格。
            grid = F.affine_grid(theta, result.shape, align_corners=False)
            # 双线性采样保留图像梯度；border 填充避免医学图边缘出现黑三角。
            result = F.grid_sample(
                result,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        # 所有颜色/插值结果最终回到统一 [0,1] 像素定义。
        return result.clamp(0.0, 1.0)
