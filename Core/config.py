"""Single-entry configuration for size-weighted cluster IDM experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝 YAML 重复键，避免旧值被静默覆盖。"""


def _construct_unique_mapping(loader, node, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"配置键重复：{key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"YAML 顶层必须是字典：{path}")
    return deepcopy(dict(value))


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_runtime"]["project_root"]) / path
    return path.resolve()


def image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    return tuple(map(int, config["data"]["image"]["size"]))


def output_root(config: Mapping[str, Any]) -> Path:
    base = resolve_path(config, config["project"]["output_root"])
    return base / str(config["_runtime"]["dataset"])


def list_datasets(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[str]:
    path = Path(config_path).expanduser().resolve()
    global_config = _read_yaml(path)
    registry_path = path.parent / str(global_config["dataset_registry"])
    registry = _read_yaml(registry_path)
    return sorted(map(str, registry["datasets"]))


def _validate(config: Mapping[str, Any]) -> None:
    project = config["project"]
    if int(project.get("windows_num_workers", 0)) < 0:
        raise ValueError("project.windows_num_workers 不能为负数")
    if str(project.get("amp_dtype", "bf16")).lower() not in {"bf16", "fp16"}:
        raise ValueError("project.amp_dtype 只能是 bf16 或 fp16")

    data = config["data"]
    adapter = str(data["adapter"]).lower()
    if adapter not in {"folder", "manifest", "medmnist_npz"}:
        raise ValueError(f"未知数据适配器：{adapter}")
    if not data.get("class_names"):
        raise ValueError("数据集必须显式配置 class_names")
    height, width = image_size(config)
    if min(height, width) <= 0:
        raise ValueError("图像尺寸必须为正数")
    if int(data["image"]["channels"]) not in {1, 3}:
        raise ValueError("图像通道数只能为 1 或 3")
    normalization = data["image"].get("normalization", {})
    mean = list(normalization.get("mean", []))
    std = list(normalization.get("std", []))
    channels = int(data["image"]["channels"])
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            "data.image.normalization.mean/std 必须与图像通道数一致"
        )
    if any(float(value) <= 0.0 for value in std):
        raise ValueError("data.image.normalization.std 必须全部大于 0")
    default_ipcs = [int(value) for value in data.get("default_ipcs", [])]
    if not default_ipcs or any(value <= 0 for value in default_ipcs):
        raise ValueError("data.default_ipcs 必须包含正整数")
    if adapter == "medmnist_npz" and not data.get("medmnist", {}).get("file"):
        raise ValueError("medmnist_npz 需要 data.medmnist.file")

    settings = config["condensation"]
    idm = settings["idm"]
    for key in (
        "iterations",
        "network_depth",
        "batch_real",
        "batch_train",
        "net_num",
        "train_net_num",
        "fetch_net_num",
        "checkpoint_interval_iterations",
        "synthetic_snapshot_interval_iterations",
    ):
        if int(idm[key]) <= 0:
            raise ValueError(f"condensation.idm.{key} 必须为正整数")
    if int(idm["train_net_num"]) != 1:
        raise ValueError("标准 IDM 每次合成更新必须随机使用一个队列网络")
    if int(idm["partition_expansion"]) not in {1, 2}:
        raise ValueError("partition_expansion 只能为 1 或 2")
    partition_schedule = idm.get("partition_expansion_by_ipc", {})
    if any(int(value) not in {1, 2} for value in partition_schedule.values()):
        raise ValueError(
            "condensation.idm.partition_expansion_by_ipc 只能包含 1 或 2"
        )
    ce_schedule = idm.get("ce_weight_by_ipc", {})
    if not ce_schedule:
        raise ValueError("condensation.idm.ce_weight_by_ipc 不能为空")
    if any(float(value) < 0.0 for value in ce_schedule.values()):
        raise ValueError("IDM CE 权重不能为负数")
    for key in (
        "batch_real",
        "batch_train",
        "net_num",
        "net_generate_interval",
        "reliability_reset_interval",
    ):
        schedule = idm.get(f"{key}_by_ipc", {})
        if not schedule and key == "reliability_reset_interval":
            raise ValueError(
                "condensation.idm.reliability_reset_interval_by_ipc 不能为空"
            )
        if any(int(value) <= 0 for value in schedule.values()):
            raise ValueError(
                f"condensation.idm.{key}_by_ipc 必须只包含正整数"
            )
    fraction = float(settings["memory"]["max_reserved_fraction"])
    if not 0.25 <= fraction <= 0.90:
        raise ValueError("max_reserved_fraction 必须位于 0.25–0.90")
    cluster = settings["cluster_matching"]
    descriptor_size = list(cluster.get("descriptor_size", []))
    if len(descriptor_size) != 2 or min(map(int, descriptor_size)) <= 0:
        raise ValueError("cluster_matching.descriptor_size 必须是两个正整数")
    if int(cluster.get("pca_components", 0)) <= 0:
        raise ValueError("cluster_matching.pca_components 必须为正整数")
    if int(cluster.get("kmeans_n_init", 0)) <= 0:
        raise ValueError("cluster_matching.kmeans_n_init 必须为正整数")
    if float(cluster.get("center_loss_weight", 0.0)) <= 0.0:
        raise ValueError("cluster_matching.center_loss_weight 必须为正数")
    if bool(settings["cluster_spread"].get("enabled", False)) and not bool(
        cluster.get("matching_enabled", False)
    ):
        raise ValueError("cluster_spread 只能与 cluster matching 一起启用")
    spread = settings["cluster_spread"]
    target = float(spread["target_gradient_fraction"])
    if not 0.0 < target < 1.0:
        raise ValueError("cluster_spread.target_gradient_fraction 必须位于 (0,1)")
    if float(spread.get("minimum_relative_gradient_norm", 0.0)) < 0.0:
        raise ValueError("minimum_relative_gradient_norm 不能为负数")
    if float(spread.get("smooth_l1_beta", 0.0)) <= 0.0:
        raise ValueError("cluster_spread.smooth_l1_beta 必须为正数")
    if float(spread.get("radial_weight", -1.0)) < 0.0:
        raise ValueError("cluster_spread.radial_weight 不能为负数")
    if float(spread.get("standard_deviation_weight", -1.0)) < 0.0:
        raise ValueError("cluster_spread.standard_deviation_weight 不能为负数")
    pixel_constraints = settings.get("pixel_constraints", {})
    if bool(pixel_constraints.get("enabled", False)):
        lower = float(pixel_constraints.get("minimum", 0.0))
        upper = float(pixel_constraints.get("maximum", 1.0))
        if not lower < upper:
            raise ValueError(
                "pixel_constraints.minimum 必须小于 maximum"
            )
    for section, keys in (
        (
            settings["online_evaluation"],
            (
                "interval_iterations",
                "training_steps",
                "batch_size",
                "evaluation_batch_size",
            ),
        ),
        (
            settings["evaluation"],
            (
                "epochs",
                "batch_size",
                "checkpoint_interval_epochs",
                "log_interval_epochs",
            ),
        ),
    ):
        for key in keys:
            if int(section[key]) <= 0:
                raise ValueError(f"{key} 必须为正整数")
    online = settings["online_evaluation"]
    if bool(online.get("select_best_by_loss", False)):
        if str(data.get("online_evaluation_split", "val")).lower() != "val":
            raise ValueError(
                "Validation-loss snapshot selection requires an independent "
                "validation split"
            )
        if int(online["interval_iterations"]) != int(
            idm["synthetic_snapshot_interval_iterations"]
        ):
            raise ValueError(
                "Online evaluation and synthetic snapshot intervals must "
                "match when validation-loss selection is enabled"
            )
    architectures = [
        str(value).lower()
        for value in settings["evaluation"].get("architectures", [])
    ]
    supported = {"convnet", "resnet18", "vgg11", "alexnet"}
    if not architectures or set(architectures) - supported:
        raise ValueError(
            "evaluation.architectures 只能使用 "
            "convnet/resnet18/vgg11/alexnet"
        )
    if int(settings["evaluation"].get("repeats", 0)) <= 0:
        raise ValueError("evaluation.repeats 必须为正整数")


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dataset: str = "pathmnist",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """加载指定数据集；新增数据集只需在 datasets.yaml 增加条目。"""

    path = Path(config_path).expanduser().resolve()
    global_config = _read_yaml(path)
    algorithm_path = path.parent / str(global_config["algorithm_config"])
    registry_path = path.parent / str(global_config["dataset_registry"])
    registry = _read_yaml(registry_path)
    name = str(dataset).strip().lower()
    if name not in registry["datasets"]:
        available = ", ".join(sorted(map(str, registry["datasets"])))
        raise ValueError(f"未知数据集 {name!r}；可选：{available}")
    dataset_entry = deepcopy(dict(registry["datasets"][name]))
    condensation_overrides = dataset_entry.pop(
        "condensation_overrides", {}
    )
    merged = _deep_merge(global_config, _read_yaml(algorithm_path))
    merged["data"] = dataset_entry
    merged["condensation"] = _deep_merge(
        merged["condensation"], condensation_overrides
    )
    if overrides:
        merged = _deep_merge(merged, overrides)
    merged.pop("algorithm_config", None)
    merged.pop("dataset_registry", None)
    merged["_runtime"] = {
        "project_root": str(PROJECT_ROOT),
        "config_path": str(path),
        "dataset_registry_path": str(registry_path.resolve()),
        "algorithm_config_path": str(algorithm_path.resolve()),
        "dataset": name,
    }
    _validate(merged)
    return merged
