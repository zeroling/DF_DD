"""Auditable IDM ConvNet, DSA, and P&E building blocks.

The configurable Conv-IN-ReLU-AvgPool network follows the official IDM
implementation. Dataset configuration selects ConvNetD3 for PathMNIST and
ConvNetD5 for 112x112 COVID19-CXR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IDMFeatures:
    logits: torch.Tensor
    embedding: torch.Tensor


class OfficialIDMConvNet(nn.Module):
    """IDM 使用的可配置深度 Conv-IN-ReLU-AvgPool 网络。"""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        image_size: Sequence[int],
        depth: int,
    ) -> None:
        super().__init__()
        height, width = map(int, image_size)
        self.depth = int(depth)
        if self.depth < 3:
            raise ValueError("IDM ConvNet 至少需要 3 个 block")
        if min(height, width) // (2**self.depth) < 1:
            raise ValueError(
                f"图像 {height}x{width} 无法经过 {self.depth} 次池化"
            )
        layers: list[nn.Module] = []
        input_channels = int(channels)
        for _ in range(self.depth):
            layers.extend(
                [
                    nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
                    # 官方代码将 InstanceNorm 实现为每通道一组的 GroupNorm。
                    nn.GroupNorm(128, 128, affine=True),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ]
            )
            input_channels = 128
            height //= 2
            width //= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128 * height * width, int(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1))


class IDMConvNet(nn.Module):
    """官方风格、width=128、InstanceNorm、ReLU、AvgPool ConvNet。"""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        image_size: Sequence[int],
        depth: int,
    ):
        super().__init__()
        self.depth = int(depth)
        self.network = OfficialIDMConvNet(
            channels=int(channels),
            num_classes=int(num_classes),
            image_size=image_size,
            depth=self.depth,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def forward_idm(self, images: torch.Tensor) -> IDMFeatures:
        """Return the official final spatial embedding and class logits."""

        feature = self.network.features(images)
        embedding = feature
        logits = self.network.classifier(feature.flatten(1))
        return IDMFeatures(logits=logits, embedding=embedding)


def build_idm_convnet(
    channels: int,
    num_classes: int,
    image_size: Sequence[int],
    depth: int,
) -> IDMConvNet:
    return IDMConvNet(channels, num_classes, image_size, depth)
class ParamDiffAug:
    """官方 DSA 默认参数。"""

    def __init__(self) -> None:
        self.aug_mode = "S"
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.latestseed = -1
        self.Siamese = False


def _set_seed(param: ParamDiffAug) -> None:
    if param.latestseed != -1:
        torch.random.manual_seed(int(param.latestseed))
        param.latestseed += 1


def _rand_scale(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    ratio = param.ratio_scale
    _set_seed(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    _set_seed(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = torch.zeros((x.shape[0], 2, 3), dtype=torch.float32)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(
        theta.to(device=x.device, dtype=x.dtype),
        x.shape,
        align_corners=False,
    )
    return F.grid_sample(x, grid, align_corners=False)


def _rand_rotate(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    _set_seed(param)
    angles = (torch.rand(x.shape[0]) - 0.5) * 2 * param.ratio_rotate / 180 * float(np.pi)
    theta = torch.zeros((x.shape[0], 2, 3), dtype=torch.float32)
    theta[:, 0, 0] = torch.cos(angles)
    theta[:, 0, 1] = torch.sin(-angles)
    theta[:, 1, 0] = torch.sin(angles)
    theta[:, 1, 1] = torch.cos(angles)
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(
        theta.to(device=x.device, dtype=x.dtype),
        x.shape,
        align_corners=False,
    )
    return F.grid_sample(x, grid, align_corners=False)


def _rand_flip(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    _set_seed(param)
    values = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:
        values[:] = values[0].clone()
    return torch.where(values < param.prob_flip, x.flip(3), x)


def _rand_brightness(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    _set_seed(param)
    values = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        values[:] = values[0].clone()
    return x + (values - 0.5) * param.brightness


def _rand_saturation(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    _set_seed(param)
    values = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        values[:] = values[0].clone()
    return (x - mean) * (values * param.saturation) + mean


def _rand_contrast(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    _set_seed(param)
    values = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:
        values[:] = values[0].clone()
    return (x - mean) * (values + param.contrast) + mean


def _rand_crop(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    shift_x = int(x.size(2) * param.ratio_crop_pad + 0.5)
    shift_y = int(x.size(3) * param.ratio_crop_pad + 0.5)
    _set_seed(param)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, (x.size(0), 1, 1), device=x.device
    )
    _set_seed(param)
    translation_y = torch.randint(
        -shift_y, shift_y + 1, (x.size(0), 1, 1), device=x.device
    )
    if param.Siamese:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return padded.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y
    ].permute(0, 3, 1, 2)


def _rand_cutout(x: torch.Tensor, param: ParamDiffAug) -> torch.Tensor:
    size = (
        int(x.size(2) * param.ratio_cutout + 0.5),
        int(x.size(3) * param.ratio_cutout + 0.5),
    )
    _set_seed(param)
    offset_x = torch.randint(
        0, x.size(2) + (1 - size[0] % 2), (x.size(0), 1, 1), device=x.device
    )
    _set_seed(param)
    offset_y = torch.randint(
        0, x.size(3) + (1 - size[1] % 2), (x.size(0), 1, 1), device=x.device
    )
    if param.Siamese:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(size[0], dtype=torch.long, device=x.device),
        torch.arange(size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(
        grid_x + offset_x - size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(
        x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
    )
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


_AUGMENT_FNS = {
    "color": [_rand_brightness, _rand_saturation, _rand_contrast],
    "crop": [_rand_crop],
    "cutout": [_rand_cutout],
    "flip": [_rand_flip],
    "scale": [_rand_scale],
    "rotate": [_rand_rotate],
}


def diff_augment(
    images: torch.Tensor,
    strategy: str,
    seed: int = -1,
    param: ParamDiffAug | None = None,
) -> torch.Tensor:
    """官方 single-mode DiffAugment；同 seed 保证真实/合成使用相同变换。"""

    if str(strategy) in {"", "none", "None"}:
        return images
    param = param or ParamDiffAug()
    param.Siamese = int(seed) != -1
    param.latestseed = int(seed)
    policies = str(strategy).split("_")
    _set_seed(param)
    selected = policies[torch.randint(0, len(policies), (1,)).item()]
    for function in _AUGMENT_FNS[selected]:
        images = function(images, param)
    return images.contiguous()


def partition_and_expand(
    images: torch.Tensor,
    labels: torch.Tensor,
    factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """官方 ImageNet IPC=1 的 2×2 Partitioning & Expansion。"""

    if int(factor) == 1:
        return images, labels
    if int(factor) != 2:
        raise ValueError("IDM P&E 只支持 1×1 或 2×2")
    height, width = images.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("P&E 2×2 要求图像高宽为偶数")
    patches = []
    for row in range(2):
        for column in range(2):
            patch = images[
                :,
                :,
                row * height // 2 : (row + 1) * height // 2,
                column * width // 2 : (column + 1) * width // 2,
            ].clone()
            patches.append(
                F.interpolate(
                    patch,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
    return torch.cat(patches, dim=0), labels.repeat(4)


def initialize_partitioned_pixels(
    class_pool,
    num_classes: int,
    image_size: Sequence[int],
    ipc: int = 1,
    factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 IPC 初始化；2×2 时每个 canvas 用四张真实图填充象限。"""

    height, width = map(int, image_size)
    images = []
    labels = []
    for class_id in range(int(num_classes)):
        if int(factor) == 1:
            class_images = class_pool.sample(class_id, int(ipc))
        elif int(factor) == 2:
            real = class_pool.sample(class_id, 4 * int(ipc))
            class_images = torch.empty(
                (int(ipc), real.shape[1], height, width),
                dtype=torch.float32,
            )
            for image_index in range(int(ipc)):
                for patch_index, (row, column) in enumerate(
                    ((0, 0), (1, 0), (0, 1), (1, 1))
                ):
                    source = 4 * image_index + patch_index
                    patch = F.interpolate(
                        real[source : source + 1],
                        size=(height // 2, width // 2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    class_images[
                        image_index : image_index + 1,
                        :,
                        row * height // 2 : (row + 1) * height // 2,
                        column * width // 2 : (column + 1) * width // 2,
                    ] = patch
        else:
            raise ValueError("初始化只支持 P&E factor=1 或 2")
        images.append(class_images)
        labels.extend([class_id] * int(ipc))
    return torch.cat(images), torch.tensor(labels, dtype=torch.long)


def cumulative_accuracy(correct: int, count: int) -> float:
    return float(correct) / max(1, int(count)) if int(count) else 0.0
