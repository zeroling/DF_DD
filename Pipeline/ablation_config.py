"""读取已经拆分的 condense 算法配置与 ablation 实验矩阵。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from Core.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from Core.io_utils import read_json


def load_ablation_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """加载项目唯一全局配置并验证其中的 ``ablation`` 节点。"""

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        if overrides:
            raise ValueError("已解析配置快照不接受额外 overrides")
        payload = read_json(resolved)
        if not isinstance(payload, dict):
            raise ValueError(f"无效的配置快照：{resolved}")
        config = payload
    else:
        config = load_config(resolved, overrides)
    _validate(config)
    return config


def _validate(config: Mapping[str, Any]) -> None:
    settings = config.get("ablation")
    if not isinstance(settings, Mapping):
        raise KeyError("configs/global.yaml 缺少 ablation 配置")
    profiles = settings["profiles"]
    default_profile = str(settings["default_profile"])
    if default_profile not in profiles:
        raise ValueError(f"default_profile={default_profile!r} 不存在")
    expected_d = {"D0", "D1", "D2", "D3"}
    if set(map(str, settings["diagnostic"]["experiments"])) != expected_d:
        raise ValueError("D 组必须恰好包含 D0/D1/D2/D3")
    expected_c = {"C0", "C1", "C2", "C3", "C4", "C5"}
    if set(map(str, settings["methods"])) != expected_c:
        raise ValueError("C 组必须恰好包含 C0...C5")
    condensation = condensation_settings(config)
    if int(condensation["idm"]["ipc"]) != 1:
        raise ValueError("本实验入口固定为 IPC=1")
    if int(condensation["idm"]["partition_expansion"]) != 2:
        raise ValueError("忠实 ImageNet IPC=1 基线固定使用 P&E 2×2")
    if (
        int(condensation["idm"]["batch_real"]) <= 0
        or int(condensation["idm"]["batch_train"]) <= 0
    ):
        raise ValueError("IDM 有效 batch 必须为正数")
    reserved_fraction = float(
        condensation["memory"]["max_reserved_fraction"]
    )
    if not 0.25 <= reserved_fraction <= 0.95:
        raise ValueError(
            "condensation.memory.max_reserved_fraction 必须在 0.25–0.85"
        )
    architectures = set(condensation["evaluation"]["architectures"])
    required = {"idm_convnet6", "resnet18", "convnext_tiny", "vit_tiny"}
    if architectures != required:
        raise ValueError(f"评估架构必须保留 {sorted(required)}")


def ablation_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["ablation"]


def condensation_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["condensation"]


def output_root(config: Mapping[str, Any]) -> Path:
    return resolve_path(config, ablation_settings(config)["output_root"])


def generator_stage_directory(config: Mapping[str, Any], stage: str) -> Path:
    root = resolve_path(config, config["project"]["run_dir"])
    return root / str(stage)
