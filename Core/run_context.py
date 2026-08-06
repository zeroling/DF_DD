"""新版实验共享的设备和混合精度辅助函数。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["project"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"配置请求 {requested}，但 PyTorch 未检测到 CUDA")
    return device


def amp_dtype(config: Mapping[str, Any]) -> torch.dtype:
    return (
        torch.bfloat16
        if str(config["project"].get("amp_dtype", "bf16")).lower() == "bf16"
        else torch.float16
    )


def amp_enabled(
    config: Mapping[str, Any], device: torch.device
) -> bool:
    return bool(config["project"].get("amp", True)) and device.type == "cuda"


def autocast_context(
    config: Mapping[str, Any], device: torch.device
):
    if not amp_enabled(config, device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype(config))


def make_grad_scaler(
    config: Mapping[str, Any], device: torch.device
):
    enabled = (
        amp_enabled(config, device)
        and amp_dtype(config) == torch.float16
    )
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)
