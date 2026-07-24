"""运行设备、混合精度和固定实验目录的公共辅助函数。"""

from __future__ import annotations

from contextlib import nullcontext  # AMP 关闭时提供与 autocast 相同的 with 接口。
from pathlib import Path  # 统一管理实验与阶段目录。
from typing import Any, Mapping  # 接受 YAML 加载后的嵌套配置映射。

import torch  # 设备、混合精度上下文和 GradScaler。

from Core.config import run_dir, stage_dir  # 配置路径解析逻辑只保留一个实现。


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    """解析 auto/cpu/cuda，并在显式 CUDA 不可用时直接报错。"""

    # auto 根据当前 PyTorch 环境选择 GPU 或 CPU；其余字符串交给 torch.device 解析。
    requested = str(config["project"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 支持 cuda、cuda:1、cpu 等标准设备字符串。
    device = torch.device(requested)
    # 用户明确请求 GPU 时不静默降级，否则可能误以为大型实验正在使用显卡。
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"配置请求 {requested}，但当前 PyTorch 未检测到 CUDA")
    return device


def amp_dtype(config: Mapping[str, Any]) -> torch.dtype:
    """把配置字符串转换成 Torch dtype。"""

    # 当前配置只区分 bf16 和 fp16；非 bf16 字符串由配置校验确保是 fp16。
    return torch.bfloat16 if str(config["project"].get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16


def amp_enabled(config: Mapping[str, Any], device: torch.device) -> bool:
    """当前仅在 CUDA 上启用自动混合精度。"""

    # CPU 路径固定 float32，避免不同 CPU 对 bf16 算子支持程度不一。
    return bool(config["project"].get("amp", True)) and device.type == "cuda"


def autocast_context(config: Mapping[str, Any], device: torch.device):
    """返回可直接用于 ``with`` 的 AMP 上下文。"""

    # nullcontext 让调用方无需为 CPU/关闭 AMP 编写两套前向代码。
    if not amp_enabled(config, device):
        return nullcontext()
    # CUDA autocast 根据配置选择 bf16 或 fp16；损失中敏感算子会显式转回 float32。
    return torch.autocast(device_type="cuda", dtype=amp_dtype(config))


def make_grad_scaler(config: Mapping[str, Any], device: torch.device):
    """FP16 使用动态缩放；BF16 和 CPU 下创建禁用的缩放器以统一调用接口。"""

    # BF16 指数范围足够大，通常不需要动态缩放；只有 CUDA FP16 真正启用。
    enabled = amp_enabled(config, device) and amp_dtype(config) == torch.float16
    try:
        # PyTorch 新接口把设备类型作为第一个参数。
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # 兼容较早的 PyTorch 2.x 调用签名。
        return torch.cuda.amp.GradScaler(enabled=enabled)


def prepare_run(config: Mapping[str, Any]) -> Path:
    """创建固定实验目录及各阶段目录，不创建时间戳副本。"""

    # 固定目录由 project.output_root / project.name 组成，便于自动找到断点续训。
    root = run_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    # 在线队列状态直接保存在 condensed 断点中，不创建额外的专家历史目录。
    for name in ("autoencoder", "diffusion", "condensed", "evaluation"):
        stage_dir(config, name, create=True)
    return root


def stage_checkpoint_directory(config: Mapping[str, Any], stage_name: str, *parts: str) -> Path:
    """得到一个可独立续训的子任务目录。"""

    # parts 可表示 IPC、重复编号或评估架构，使各子实验断点互不覆盖。
    directory = stage_dir(config, stage_name, create=True).joinpath(*map(str, parts))
    # exist_ok 允许续训时复用原目录。
    directory.mkdir(parents=True, exist_ok=True)
    return directory
