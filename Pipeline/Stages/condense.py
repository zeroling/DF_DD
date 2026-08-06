"""Pixel IDM with cluster-conditioned, size-weighted distribution matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
)
from Core.config import output_root
from Core.data import build_loader, unpack_batch
from Core.io_utils import atomic_write_json, read_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, resolve_device
from Core.seed import seed_everything
from Pipeline.data import experiment_bundle
from Net.Condensation.idm_official import (
    IDMConvNet,
    ParamDiffAug,
    build_idm_convnet,
    cumulative_accuracy,
    diff_augment,
    initialize_partitioned_pixels,
)
from Core.experiment_runtime import (
    cleanup_memory,
    cuda_peak_megabytes,
)
from Net.Condensation.cluster_multiform import (
    ClusterSampler,
    PixelClusterIndex,
    build_pixel_cluster_index,
    cluster_distribution_losses,
    compose_storage_canvas,
    partition_training_images,
)

ARCHITECTURE_VERSION = 8


def _algorithm_version(condensation: Mapping[str, Any]) -> int:
    """Return the explicit checkpoint version for the active IDM pipeline."""

    del condensation
    return ARCHITECTURE_VERSION


def _ipc(config: Mapping[str, Any], override: int | None = None) -> int:
    value = int(
        override
        if override is not None
        else config["condensation"]["idm"]["ipc"]
    )
    if value <= 0:
        raise ValueError("IPC 必须为正整数")
    return value


def _idm_value(
    settings: Mapping[str, Any], key: str, ipc: int
) -> float | int:
    """Resolve an IDM scalar, optionally from its per-IPC schedule."""

    schedule = settings.get(f"{key}_by_ipc")
    if not schedule:
        if key not in settings:
            raise KeyError(f"Missing IDM setting: {key}")
        return settings[key]
    direct = schedule.get(int(ipc), schedule.get(str(int(ipc))))
    if direct is not None:
        return direct
    known = sorted((int(point), value) for point, value in schedule.items())
    lower = [item for item in known if item[0] <= int(ipc)]
    return (lower[-1] if lower else known[0])[1]


def _ce_weight(config: Mapping[str, Any], ipc: int) -> float:
    return float(
        _idm_value(config["condensation"]["idm"], "ce_weight", ipc)
    )


def _partition_factor(
    settings: Mapping[str, Any], ipc: int
) -> int:
    factor = int(
        _idm_value(settings["idm"], "partition_expansion", int(ipc))
    )
    if factor not in {1, 2}:
        raise ValueError("partition expansion 只支持 1 或 2")
    return factor


class IndexedImagePool:
    """不放回循环采样真实图；图像首次读取后缓存为 CPU float16。"""

    def __init__(self, dataset, num_classes: int, seed: int):
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.class_indices: dict[int, list[int]] = {
            class_id: [] for class_id in range(self.num_classes)
        }
        for index, label in enumerate(dataset.targets):
            self.class_indices[int(label)].append(int(index))
        self.class_orders: dict[int, list[int]] = {}
        self.class_positions = {class_id: 0 for class_id in self.class_indices}
        self.all_indices = list(range(len(dataset)))
        self.all_order: list[int] = []
        self.all_position = 0
        self.cache: dict[int, torch.Tensor] = {}

    def _reshuffle(self, candidates: list[int]) -> list[int]:
        order = torch.randperm(
            len(candidates), generator=self.generator
        ).tolist()
        return [candidates[position] for position in order]

    def _take_class(self, class_id: int, count: int) -> list[int]:
        result: list[int] = []
        candidates = self.class_indices[int(class_id)]
        while len(result) < int(count):
            order = self.class_orders.get(int(class_id), [])
            position = self.class_positions[int(class_id)]
            if position >= len(order):
                order = self._reshuffle(candidates)
                self.class_orders[int(class_id)] = order
                position = 0
            amount = min(int(count) - len(result), len(order) - position)
            result.extend(order[position : position + amount])
            self.class_positions[int(class_id)] = position + amount
        return result

    def _take_all(self, count: int) -> list[int]:
        result: list[int] = []
        while len(result) < int(count):
            if self.all_position >= len(self.all_order):
                self.all_order = self._reshuffle(self.all_indices)
                self.all_position = 0
            amount = min(
                int(count) - len(result),
                len(self.all_order) - self.all_position,
            )
            result.extend(
                self.all_order[
                    self.all_position : self.all_position + amount
                ]
            )
            self.all_position += amount
        return result

    def _image(self, index: int) -> torch.Tensor:
        cached = self.cache.get(int(index))
        if cached is None:
            image = self.dataset[int(index)]["image"]
            cached = (
                image.detach()
                .to(dtype=torch.float16)
                .cpu()
                .contiguous()
            )
            self.cache[int(index)] = cached
        return cached.float()

    def sample(self, class_id: int, count: int) -> torch.Tensor:
        return torch.stack(
            [self._image(index) for index in self._take_class(class_id, count)]
        )

    def images(self, indices: list[int]) -> torch.Tensor:
        return torch.stack([self._image(index) for index in indices])

    def sample_general(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._take_all(count)
        images = torch.stack([self._image(index) for index in indices])
        labels = torch.tensor(
            [int(self.dataset.targets[index]) for index in indices],
            dtype=torch.long,
        )
        return images, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state(),
            "class_orders": self.class_orders,
            "class_positions": self.class_positions,
            "all_order": self.all_order,
            "all_position": int(self.all_position),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if torch.is_tensor(state.get("generator_state")):
            self.generator.set_state(state["generator_state"].cpu())
        self.class_orders = {
            int(key): list(map(int, value))
            for key, value in dict(state.get("class_orders", {})).items()
        }
        self.class_positions = {
            int(key): int(value)
            for key, value in dict(
                state.get("class_positions", self.class_positions)
            ).items()
        }
        self.all_order = list(map(int, state.get("all_order", [])))
        self.all_position = int(state.get("all_position", 0))


def _cluster_cache_path(
    config: Mapping[str, Any],
    ipc: int,
    cluster_count: int,
    radial_layers: int,
    angular_sectors: int,
    seed: int,
    dataset_size: int,
) -> Path:
    settings = config["condensation"]["cluster_matching"]
    root = Path(str(settings["cache_root"]))
    if not root.is_absolute():
        root = Path(config["_runtime"]["project_root"]) / root
    descriptor = "x".join(map(str, settings["descriptor_size"]))
    nontrivial_layout = (
        int(cluster_count) != int(ipc)
        or int(radial_layers) != 1
        or int(angular_sectors) != 1
    )
    cluster_suffix = (
        f"_k{int(cluster_count)}_r{int(radial_layers)}_a{int(angular_sectors)}"
        if nontrivial_layout
        else ""
    )
    name = (
        f"{config['_runtime']['dataset']}_n{int(dataset_size)}_ipc{int(ipc)}"
        f"{cluster_suffix}_"
        f"d{descriptor}_pca{int(settings['pca_components'])}_seed{int(seed)}.pt"
    )
    return root.resolve() / name


def _load_or_build_cluster_index(
    config: Mapping[str, Any],
    pool: IndexedImagePool,
    ipc: int,
    cluster_count: int,
    radial_layers: int,
    angular_sectors: int,
    seed: int,
) -> tuple[PixelClusterIndex, Path]:
    cache_path = _cluster_cache_path(
        config,
        ipc,
        cluster_count,
        radial_layers,
        angular_sectors,
        seed,
        len(pool.dataset),
    )
    if bool(config.get("_smoke", False)):
        cache_path = cache_path.with_name(
            f"{cache_path.stem}_smoke{cache_path.suffix}"
        )
    if cache_path.is_file():
        payload = load_checkpoint(cache_path, "cpu")
        return PixelClusterIndex.from_state_dict(payload), cache_path
    settings = config["condensation"]["cluster_matching"]
    normalization = config["data"]["image"]["normalization"]
    maximum = 64 if bool(config.get("_smoke", False)) else None
    index = build_pixel_cluster_index(
        pool.dataset,
        pool.class_indices,
        clusters_per_class=int(cluster_count),
        descriptor_size=settings["descriptor_size"],
        pca_components=int(settings["pca_components"]),
        kmeans_n_init=int(settings["kmeans_n_init"]),
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
        seed=int(seed),
        radial_layers=int(radial_layers),
        angular_sectors=int(angular_sectors),
        maximum_samples_per_class=maximum,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(index.state_dict(), cache_path)
    return index, cache_path


def _cluster_layout(
    cluster_settings: Mapping[str, Any], ipc: int
) -> tuple[int, int, int, int]:
    """Return K, radial layers, angular sectors, and images per cluster."""

    schedule = cluster_settings.get("layout_by_ipc", {})
    configured = schedule.get(int(ipc), schedule.get(str(int(ipc))))
    if configured is not None:
        cluster_count = int(configured["clusters"])
        radial_layers = int(configured["radial_layers"])
        angular_sectors = int(configured["angular_sectors"])
    else:
        maximum = int(cluster_settings.get("max_clusters_per_class", int(ipc)))
        if maximum <= 0:
            raise ValueError("max_clusters_per_class must be positive")
        divisors = [
            candidate
            for candidate in range(1, min(int(ipc), maximum) + 1)
            if int(ipc) % candidate == 0
        ]
        cluster_count = max(divisors)
        radial_layers = int(ipc) // cluster_count
        angular_sectors = 1
    values = (cluster_count, radial_layers, angular_sectors)
    if min(values) <= 0:
        raise ValueError("Cluster layout values must be positive")
    images_per_cluster = radial_layers * angular_sectors
    if cluster_count * images_per_cluster != int(ipc):
        raise ValueError(
            f"IPC={int(ipc)} does not match K={cluster_count} x "
            f"R={radial_layers} x A={angular_sectors}"
        )
    return cluster_count, radial_layers, angular_sectors, images_per_cluster


def _stored_cluster_groups(
    cluster_count: int, images_per_cluster: int, device: torch.device
) -> torch.Tensor:
    return torch.arange(
        int(cluster_count), device=device, dtype=torch.long
    ).repeat_interleave(int(images_per_cluster))


def _stored_radial_rings(
    cluster_count: int,
    radial_layers: int,
    angular_sectors: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        int(radial_layers), device=device, dtype=torch.long
    ).repeat_interleave(int(angular_sectors)).repeat(int(cluster_count))


def _stored_angular_sectors(
    cluster_count: int,
    radial_layers: int,
    angular_sectors: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        int(angular_sectors), device=device, dtype=torch.long
    ).repeat(int(cluster_count) * int(radial_layers))


def _initialize_synthetic_pixels(
    config: Mapping[str, Any],
    pool: IndexedImagePool,
    cluster_index: PixelClusterIndex | None,
    num_classes: int,
    ipc: int,
    cluster_count: int,
    radial_layers: int,
    angular_sectors: int,
    partition_factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_settings = config["condensation"]["cluster_matching"]
    cluster_initialization = bool(
        cluster_settings.get("initialization_enabled", False)
    )
    if not cluster_initialization:
        return initialize_partitioned_pixels(
            pool,
            int(num_classes),
            config["data"]["image"]["size"],
            ipc=int(ipc),
            factor=int(partition_factor),
        )
    if cluster_initialization and cluster_index is None:
        raise ValueError("Cluster initialization requires a cluster index")

    sources_per_canvas = int(partition_factor) ** 2
    canvases: list[torch.Tensor] = []
    labels: list[int] = []
    for class_id in range(int(num_classes)):
        if cluster_initialization:
            assert cluster_index is not None
            source_groups = [
                cluster_index.cell_representatives(
                    class_id,
                    cluster_id,
                    radial_layer,
                    angular_sector,
                    sources_per_canvas,
                )
                for cluster_id in range(int(cluster_count))
                for radial_layer in range(int(radial_layers))
                for angular_sector in range(int(angular_sectors))
            ]
            if len(source_groups) != int(ipc):
                raise RuntimeError("Radial cluster layout does not match IPC")
        else:
            random_images = pool.sample(
                class_id, int(ipc) * sources_per_canvas
            )
            source_groups = [
                random_images[
                    index * sources_per_canvas : (index + 1) * sources_per_canvas
                ]
                for index in range(int(ipc))
            ]
        for source_group in source_groups:
            sources = (
                pool.images(source_group)
                if isinstance(source_group, list)
                else source_group
            )
            canvases.append(
                compose_storage_canvas(
                    sources,
                    config["data"]["image"]["size"],
                    int(partition_factor),
                )
            )
            labels.append(int(class_id))
    return torch.stack(canvases), torch.tensor(labels, dtype=torch.long)


def _sample_clustered_real_batch(
    pool: IndexedImagePool,
    sampler: ClusterSampler,
    class_id: int,
    cluster_count: int,
    total_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimum_per_cluster = 2
    required = int(cluster_count) * minimum_per_cluster
    if int(total_count) < required:
        raise ValueError(
            "batch_real must provide at least two samples per IPC cluster"
        )
    sizes = torch.tensor(
        sampler.index.sizes[int(class_id)], dtype=torch.float64
    )
    remaining = int(total_count) - required
    ideal = sizes / sizes.sum() * float(remaining)
    extras = ideal.floor().to(torch.long)
    leftover = remaining - int(extras.sum().item())
    if leftover > 0:
        order = (ideal - extras).argsort(descending=True)
        extras[order[:leftover]] += 1
    counts = extras + minimum_per_cluster
    indices: list[int] = []
    groups: list[int] = []
    for cluster_id in range(int(cluster_count)):
        count = int(counts[cluster_id].item())
        selected = sampler.take(class_id, cluster_id, count)
        indices.extend(selected)
        groups.extend([cluster_id] * count)
    return pool.images(indices), torch.tensor(groups, dtype=torch.long)


def _set_model_trainable(model: nn.Module, enabled: bool) -> None:
    model.train(bool(enabled))
    for parameter in model.parameters():
        parameter.requires_grad_(bool(enabled))


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _build_synthetic_optimizer(
    parameter: nn.Parameter,
    settings: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, str, float]:
    learning_rate = float(settings["idm"]["image_learning_rate"])
    optimizer = torch.optim.SGD(
        [parameter],
        lr=learning_rate,
        momentum=float(settings["idm"]["image_momentum"]),
    )
    return optimizer, "sgd", learning_rate


def _normalization_tensors(
    config: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalization = config["data"]["image"]["normalization"]
    mean = torch.tensor(
        normalization["mean"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    std = torch.tensor(
        normalization["std"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    return mean, std


def _display_images(
    config: Mapping[str, Any], images: torch.Tensor
) -> torch.Tensor:
    mean, std = _normalization_tensors(config, images.device)
    return (images.detach().float() * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def _pixel_statistics(
    images: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, float]:
    values = _display_images(config, images)
    return {
        "pixel/minimum": float(values.min().item()),
        "pixel/maximum": float(values.max().item()),
        "pixel/mean": float(values.mean().item()),
        "pixel/std": float(values.std(unbiased=False).item()),
        "pixel/below_zero_fraction": float((values < 0.0).float().mean().item()),
        "pixel/above_one_fraction": float((values > 1.0).float().mean().item()),
        "pixel/at_zero_fraction": float((values == 0.0).float().mean().item()),
        "pixel/at_one_fraction": float((values == 1.0).float().mean().item()),
    }


@torch.no_grad()
def _project_synthetic_pixels(
    parameter: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """投影到真实像素域，并清除仍会把边界像素向外推的 SGD 动量。"""

    settings = config["condensation"]["pixel_constraints"]
    before = parameter.detach()
    mean, std = _normalization_tensors(config, before.device)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    below = before < lower
    above = before > upper
    projected = below | above
    result = {
        "pixel/pre_projection_minimum": float(before.min().item()),
        "pixel/pre_projection_maximum": float(before.max().item()),
        "pixel/projected_fraction": float(projected.float().mean().item()),
        "pixel/momentum_cleared_fraction": 0.0,
    }
    parameter.copy_(torch.maximum(torch.minimum(parameter, upper), lower))

    if bool(settings.get("clear_outward_momentum", True)):
        state = optimizer.state.get(parameter, {})
        momentum = state.get("momentum_buffer")
        if torch.is_tensor(momentum):
            outward = ((parameter <= lower) & (momentum > 0.0)) | (
                (parameter >= upper) & (momentum < 0.0)
            )
            result["pixel/momentum_cleared_fraction"] = float(
                outward.float().mean().item()
            )
            momentum.masked_fill_(outward, 0.0)

    result.update(_pixel_statistics(parameter, config))
    return result


@dataclass
class QueueMember:
    identifier: int
    model: IDMConvNet
    optimizer: torch.optim.Optimizer
    birth_iteration: int
    update_steps: int = 0
    correct: int = 0
    count: int = 0

    @property
    def reliability_percent(self) -> float:
        # 官方 torchnet ClassErrorMeter(accuracy=True) 返回百分数。
        return 100.0 * cumulative_accuracy(self.correct, self.count)


class OfficialIDMQueue:
    """官方 ImageNet IPC=1 的 3→50 动态同构模型池。"""

    VERSION = 1

    def __init__(
        self,
        config: Mapping[str, Any],
        num_classes: int,
        device: torch.device,
        seed: int,
    ):
        self.config = config
        self.settings = config["condensation"]
        self.num_classes = int(num_classes)
        self.device = device
        self.random = random.Random(int(seed))
        self.members: list[QueueMember] = []
        self.next_identifier = 0
        for _ in range(3):
            self.members.append(self._new_member(0))

    def _new_member(self, birth_iteration: int) -> QueueMember:
        model = build_idm_convnet(
            int(self.config["data"]["image"]["channels"]),
            self.num_classes,
            self.config["data"]["image"]["size"],
            depth=int(self.settings["idm"].get("network_depth", 6)),
        ).to(self.device)
        settings = self.settings["idm"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(settings["lr_net"]),
            momentum=float(settings["net_momentum"]),
            weight_decay=float(settings["net_weight_decay"]),
        )
        member = QueueMember(
            identifier=int(self.next_identifier),
            model=model,
            optimizer=optimizer,
            birth_iteration=int(birth_iteration),
        )
        self.next_identifier += 1
        _set_model_trainable(model, False)
        return member

    def _value(self, key: str) -> float | int:
        settings = self.settings["idm"]
        return _idm_value(settings, key, int(settings["ipc"]))

    def grow(self, iteration: int) -> None:
        interval = int(self._value("net_generate_interval"))
        if int(iteration) % interval != 0:
            return
        maximum = int(self._value("net_num"))
        if len(self.members) == maximum:
            self.members.pop(0)
            cleanup_memory()
        self.members.append(self._new_member(int(iteration)))

    def reset_reliability_if_due(self, iteration: int) -> bool:
        """Mirror the official accuracy-meter reset at evaluation intervals."""

        interval = int(self._value("reliability_reset_interval"))
        if int(iteration) <= 0 or int(iteration) % interval != 0:
            return False
        for member in self.members:
            member.correct = 0
            member.count = 0
        return True

    def guidance_members(self) -> list[QueueMember]:
        settings = self.settings["idm"]
        count = min(int(settings["train_net_num"]), len(self.members))
        indices = list(range(len(self.members)))
        self.random.shuffle(indices)
        selected = [self.members[index] for index in indices[:count]]
        for member in selected:
            _set_model_trainable(member.model, False)
        return selected

    def training_members(self) -> list[QueueMember]:
        indices = list(range(len(self.members)))
        self.random.shuffle(indices)
        count = min(int(self.settings["idm"]["fetch_net_num"]), len(indices))
        return [self.members[index] for index in indices[:count]]

    def _train_member(
        self,
        member: QueueMember,
        pool: IndexedImagePool,
    ) -> tuple[float, float, int]:
        settings = self.settings["idm"]
        target_batch = int(settings["batch_train"])
        microbatch = int(self.settings["memory"]["real_train_microbatch"])
        minimum = int(self.settings["memory"].get("retry_minimum", 1))
        loss_total = 0.0
        correct_total = 0
        count_total = 0
        _set_model_trainable(member.model, True)
        for _ in range(int(settings["model_train_steps"])):
            images_cpu, labels_cpu = pool.sample_general(target_batch)
            while True:
                member.optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                batch_correct = 0
                try:
                    for start in range(0, target_batch, microbatch):
                        images = images_cpu[start : start + microbatch].to(
                            self.device, non_blocking=True
                        )
                        labels = labels_cpu[start : start + microbatch].to(
                            self.device, non_blocking=True
                        )
                        logits = member.model(images)
                        # 累加后等价于一次有效 batch=target_batch 的平均交叉熵。
                        loss = F.cross_entropy(
                            logits, labels, reduction="sum"
                        ) / float(target_batch)
                        loss.backward()
                        batch_loss += float(loss.detach().item())
                        batch_correct += int(
                            (logits.detach().argmax(1) == labels).sum().item()
                        )
                        del images, labels, logits, loss
                    member.optimizer.step()
                    break
                except torch.OutOfMemoryError:
                    member.optimizer.zero_grad(set_to_none=True)
                    cleanup_memory()
                    if microbatch <= minimum:
                        raise
                    microbatch = max(minimum, microbatch // 2)
            member.update_steps += 1
            member.correct += int(batch_correct)
            member.count += int(target_batch)
            loss_total += float(batch_loss)
            correct_total += int(batch_correct)
            count_total += int(target_batch)
        _set_model_trainable(member.model, False)
        return (
            loss_total / max(1, int(settings["model_train_steps"])),
            correct_total / max(1, count_total),
            microbatch,
        )

    def train_independent_members(
        self,
        pool: IndexedImagePool,
    ) -> dict[str, float]:
        losses, accuracies, microbatches = [], [], []
        selected = self.training_members()
        for member in selected:
            loss, accuracy, microbatch = self._train_member(member, pool)
            losses.append(loss)
            accuracies.append(accuracy)
            microbatches.append(microbatch)
        return {
            "queue/train_loss": statistics.fmean(losses) if losses else 0.0,
            "queue/train_accuracy": statistics.fmean(accuracies)
            if accuracies
            else 0.0,
            "queue/train_microbatch": float(min(microbatches))
            if microbatches
            else 0.0,
            "queue/size": float(len(self.members)),
            "queue/mean_updates": statistics.fmean(
                [member.update_steps for member in self.members]
            ),
            "queue/mean_reliability_percent": statistics.fmean(
                [member.reliability_percent for member in self.members]
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "next_identifier": int(self.next_identifier),
            "random_state": self.random.getstate(),
            "members": [
                {
                    "identifier": int(member.identifier),
                    "model": member.model.state_dict(),
                    "optimizer": member.optimizer.state_dict(),
                    "birth_iteration": int(member.birth_iteration),
                    "update_steps": int(member.update_steps),
                    "correct": int(member.correct),
                    "count": int(member.count),
                }
                for member in self.members
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("IDM 队列断点版本不兼容")
        restored: list[QueueMember] = []
        for item in state["members"]:
            member = self._new_member(int(item["birth_iteration"]))
            member.identifier = int(item["identifier"])
            member.model.load_state_dict(item["model"], strict=True)
            member.optimizer.load_state_dict(item["optimizer"])
            _optimizer_to_device(member.optimizer, self.device)
            settings = self.settings["idm"]
            for group in member.optimizer.param_groups:
                # 断点不校验配置哈希；恢复状态后仍以当前 YAML 的超参数为准。
                group["lr"] = float(settings["lr_net"])
                group["momentum"] = float(settings["net_momentum"])
                group["weight_decay"] = float(settings["net_weight_decay"])
            member.update_steps = int(item.get("update_steps", 0))
            member.correct = int(item.get("correct", 0))
            member.count = int(item.get("count", 0))
            _set_model_trainable(member.model, False)
            restored.append(member)
        self.members = restored
        self.next_identifier = max(
            int(state.get("next_identifier", 0)),
            max(member.identifier for member in restored) + 1,
        )
        if state.get("random_state") is not None:
            self.random.setstate(state["random_state"])


class SyntheticParameterization(nn.Module):
    """直接优化像素的每类渲染接口。"""

    def __init__(
        self,
        initial_pixels: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.mode = "pixel"
        self.num_classes = int(labels.max().item()) + 1
        self.register_buffer("labels", labels.to(device))
        self.variable = nn.Parameter(
            initial_pixels.to(device).detach().float()
        )

    def render_class(self, class_id: int) -> torch.Tensor:
        mask = self.labels == int(class_id)
        return self.variable[mask]

    @torch.no_grad()
    def render_all(self) -> torch.Tensor:
        return torch.cat(
            [self.render_class(class_id) for class_id in range(self.num_classes)]
        ).float()

    def prior_loss(self, class_id: int) -> torch.Tensor:
        del class_id
        return self.variable.sum() * 0.0


class AuxiliaryGradientMixer:
    """Budget an auxiliary gradient and remove components opposing the base."""

    _HISTORY_KEYS = (
        "base_norms",
        "auxiliary_norms",
        "gradient_scales",
        "achieved_fractions",
        "gradient_cosines",
        "conflict_projections",
    )

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.base_norms: list[float] = []
        self.auxiliary_norms: list[float] = []
        self.gradient_scales: list[float] = []
        self.achieved_fractions: list[float] = []
        self.gradient_cosines: list[float] = []
        self.conflict_projections: list[float] = []
        self.skipped_gradients = 0
        self.mixed_gradients = 0

    def target_fraction(self) -> float:
        """Return the fixed auxiliary gradient budget."""

        return float(self.settings["target_gradient_fraction"])

    def _append(self, name: str, value: float) -> None:
        values = getattr(self, name)
        values.append(float(value))
        history_size = max(1, int(self.settings.get("history_size", 256)))
        if len(values) > history_size:
            del values[:-history_size]

    def weight_against_base(
        self,
        base_gradient: torch.Tensor,
        auxiliary_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Project conflicts, then scale against the configured gradient budget."""

        base = base_gradient.float()
        auxiliary = auxiliary_gradient.float()
        base_norm_tensor = base.norm()
        auxiliary_norm_tensor = auxiliary.norm()
        base_norm = float(base_norm_tensor.detach().item())
        auxiliary_norm = float(auxiliary_norm_tensor.detach().item())
        self._append("base_norms", base_norm)
        self._append("auxiliary_norms", auxiliary_norm)
        epsilon = float(self.settings.get("epsilon", 1.0e-12))
        denominator = max(epsilon, base_norm * auxiliary_norm)
        cosine = float(
            (base.detach() * auxiliary.detach()).sum().item() / denominator
        )
        self._append("gradient_cosines", cosine)
        if cosine < 0.0 and base_norm > epsilon:
            projection = (auxiliary * base).sum() / base.square().sum().clamp_min(
                epsilon
            )
            auxiliary = auxiliary - projection * base
            auxiliary_norm_tensor = auxiliary.norm()
            auxiliary_norm = float(auxiliary_norm_tensor.detach().item())
            self._append("conflict_projections", 1.0)
        else:
            self._append("conflict_projections", 0.0)
        target = self.target_fraction()
        base_fraction = 1.0 - target
        relative_norm = auxiliary_norm / max(epsilon, base_norm)
        usable = (
            target > 0.0
            and float(base_fraction) > epsilon
            and torch.isfinite(base_norm_tensor).item()
            and torch.isfinite(auxiliary_norm_tensor).item()
            and base_norm > epsilon
            and auxiliary_norm > epsilon
            and relative_norm
            >= float(
                self.settings.get(
                    "minimum_relative_gradient_norm", 0.0
                )
            )
        )
        if not usable:
            self.skipped_gradients += 1
            self._append("gradient_scales", 0.0)
            self._append("achieved_fractions", 0.0)
            return torch.zeros_like(auxiliary_gradient), 0.0
        scale = (
            target
            / float(base_fraction)
            * base_norm
            / auxiliary_norm
        )
        weighted = auxiliary.to(auxiliary_gradient.dtype) * float(scale)
        self._append("gradient_scales", scale)
        self._append("achieved_fractions", target)
        self.mixed_gradients += 1
        return weighted, float(scale)

    def recent_mean(self, name: str, count: int) -> float:
        values = getattr(self, name)
        recent = values[-max(1, int(count)) :]
        return statistics.fmean(recent) if recent else 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            **{key: list(getattr(self, key)) for key in self._HISTORY_KEYS},
            "skipped_gradients": int(self.skipped_gradients),
            "mixed_gradients": int(self.mixed_gradients),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        for key in self._HISTORY_KEYS:
            setattr(
                self,
                key,
                [float(value) for value in state.get(key, [])],
            )
        self.skipped_gradients = int(state.get("skipped_gradients", 0))
        self.mixed_gradients = int(state.get("mixed_gradients", 0))
def _real_features(
    config: Mapping[str, Any],
    model: IDMConvNet,
    images_cpu: torch.Tensor,
    dsa_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Extract all real embeddings with OOM-safe microbatching."""

    condensation = config["condensation"]
    requested = int(condensation["memory"]["real_feature_microbatch"])
    minimum = int(condensation["memory"].get("retry_minimum", 1))
    while True:
        embedding_parts: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for start in range(0, images_cpu.shape[0], requested):
                    images = images_cpu[start : start + requested].to(
                        device, non_blocking=True
                    )
                    images = diff_augment(
                        images,
                        str(condensation["idm"]["dsa_strategy"]),
                        seed=int(dsa_seed),
                        param=ParamDiffAug(),
                    )
                    output = model.forward_idm(images)
                    embedding_parts.append(output.embedding.float().detach())
                    del images, output
            break
        except torch.OutOfMemoryError:
            cleanup_memory()
            if requested <= minimum:
                raise
            requested = max(minimum, requested // 2)
    if not embedding_parts:
        raise ValueError("Real feature batch is empty")
    return torch.cat(embedding_parts, dim=0), requested


def _quick_evaluate(
    config: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    evaluation_dataset,
    num_classes: int,
    iteration: int,
    seed: int,
    partition_factor: int,
    device: torch.device,
) -> dict[str, float]:
    """用蒸馏同架构做一次固定成本监控，不选断点、不早停。"""

    settings = config["condensation"]["online_evaluation"]
    rng_state = capture_rng_state()
    started = time.perf_counter()
    model = None
    optimizer = None
    loader = None
    try:
        seed_everything(
            int(seed), bool(config["project"].get("deterministic", False))
        )
        expanded_images, expanded_labels, _ = partition_training_images(
            images.detach().cpu(),
            labels.detach().cpu(),
            partition_factor=int(partition_factor),
        )
        model = build_idm_convnet(
            int(config["data"]["image"]["channels"]),
            int(num_classes),
            config["data"]["image"]["size"],
            depth=int(config["condensation"]["idm"]["network_depth"]),
        ).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            momentum=float(settings.get("momentum", 0.9)),
            weight_decay=float(settings.get("weight_decay", 0.0005)),
        )
        batch_size = min(
            int(settings["batch_size"]), int(expanded_labels.numel())
        )
        generator = torch.Generator().manual_seed(int(seed) + 17)
        model.train()
        for step in range(int(settings["training_steps"])):
            indices = torch.randint(
                int(expanded_labels.numel()),
                (batch_size,),
                generator=generator,
            )
            batch_images = expanded_images[indices].to(
                device, non_blocking=True
            )
            batch_labels = expanded_labels[indices].to(
                device, non_blocking=True
            )
            batch_images = diff_augment(
                batch_images,
                str(config["condensation"]["idm"]["dsa_strategy"]),
                param=ParamDiffAug(),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = F.cross_entropy(logits.float(), batch_labels)
            loss.backward()
            optimizer.step()
            del batch_images, batch_labels, logits, loss
        loader = build_loader(
            evaluation_dataset,
            config,
            train=False,
            batch_size=int(settings["evaluation_batch_size"]),
        )
        model.eval()
        correct = 0
        count = 0
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch_images, batch_labels = unpack_batch(batch)
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                logits = model(batch_images)
                loss = F.cross_entropy(
                    logits.float(), batch_labels, reduction="sum"
                )
                total_loss += float(loss.item())
                correct += int(
                    (logits.argmax(1) == batch_labels).sum().item()
                )
                count += int(batch_labels.numel())
        return {
            "iteration": int(iteration),
            "accuracy": correct / max(1, count),
            "loss": total_loss / max(1, count),
            "training_steps": int(settings["training_steps"]),
            "seconds": time.perf_counter() - started,
        }
    finally:
        del model, optimizer, loader
        cleanup_memory()
        restore_rng_state(rng_state)


def _save_checkpoint(
    directory: Path,
    experiment: str,
    iteration: int,
    parameterization: SyntheticParameterization,
    optimizer: torch.optim.Optimizer,
    queue: OfficialIDMQueue,
    pool: IndexedImagePool,
    mixer: AuxiliaryGradientMixer,
    cluster_sampler: ClusterSampler | None,
    design_signature: Mapping[str, Any],
    pixel_constraints: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    algorithm_version: int,
) -> Path:
    payload = {
        "algorithm_version": int(algorithm_version),
        "experiment": str(experiment),
        "iteration": int(iteration),
        "parameterization": parameterization.state_dict(),
        "optimizer_name": optimizer.__class__.__name__.lower(),
        "optimizer": optimizer.state_dict(),
        "queue": queue.state_dict(),
        "real_pool": pool.state_dict(),
        "auxiliary_mixer": mixer.state_dict(),
        "cluster_sampler": (
            cluster_sampler.state_dict() if cluster_sampler is not None else None
        ),
        "design_signature": dict(design_signature),
        "pixel_constraints": dict(pixel_constraints),
        "diagnostics": dict(diagnostics),
        "rng_state": capture_rng_state(),
    }
    return atomic_torch_save(payload, directory / "checkpoint_last.pt")


def _preview_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    per_class: int,
) -> torch.Tensor:
    """限制预览数量；完整合成集仍只保存在 synthetic.pt 中。"""

    selected = []
    for class_id in sorted(set(map(int, labels.tolist()))):
        selected.append(
            images[labels == int(class_id)][: max(1, int(per_class))]
        )
    return torch.cat(selected, dim=0)


def _normalize_preview(images: torch.Tensor) -> torch.Tensor:
    """逐图拉伸到 [0,1]，用于区分结构消失与显示裁剪。"""

    values = images.detach().float()
    flat = values.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (values - minimum) / (maximum - minimum).clamp_min(1.0e-8)


def _formation_metadata(
    ipc: int, partition_factor: int
) -> dict[str, Any]:
    return {
        "stored_ipc": int(ipc),
        "effective_samples_per_class": int(
            ipc * int(partition_factor) ** 2
        ),
    }


def _export(
    config: Mapping[str, Any],
    directory: Path,
    experiment: str,
    parameterization: SyntheticParameterization,
    class_names: list[str],
    iteration: int,
    ipc: int,
    partition_factor: int,
    preview_per_class: int,
    optimization_iterations: int | None = None,
    selection: Mapping[str, Any] | None = None,
) -> Path:
    images = parameterization.render_all().detach().cpu()
    labels = parameterization.labels.detach().cpu()
    path = directory / "synthetic.pt"
    atomic_torch_save(
        {
            "images": images,
            "labels": labels,
            "class_names": list(class_names),
            "ipc": int(ipc),
            "partition_expansion": int(partition_factor),
            **_formation_metadata(ipc, partition_factor),
            "method": str(experiment),
            "algorithm_version": _algorithm_version(config["condensation"]),
            "parameterization": parameterization.mode,
            "iteration": int(iteration),
            "optimization_iterations": int(
                optimization_iterations
                if optimization_iterations is not None
                else iteration
            ),
            "selection": dict(selection) if selection is not None else None,
            "normalization": dict(
                config["data"]["image"]["normalization"]
            ),
            "pixel_range": [
                float(images.min().item()),
                float(images.max().item()),
            ],
        },
        path,
    )
    preview = _preview_subset(images, labels, preview_per_class)
    save_image(
        _display_images(config, preview),
        directory / "preview.png",
        nrow=min(int(ipc), max(1, int(preview_per_class))),
    )
    save_image(
        _normalize_preview(_display_images(config, preview)),
        directory / "preview_normalized.png",
        nrow=min(int(ipc), max(1, int(preview_per_class))),
    )
    expanded, expanded_labels, _ = partition_training_images(
        images,
        labels,
        partition_factor=int(partition_factor),
    )
    expanded_preview = _preview_subset(
        expanded, expanded_labels, preview_per_class
    )
    save_image(
        _display_images(config, expanded_preview),
        directory / "preview_partition_expansion.png",
        nrow=min(
            max(1, int(ipc) * int(partition_factor) ** 2),
            max(1, int(preview_per_class)),
        ),
    )
    del preview, expanded, expanded_labels, expanded_preview
    return path


def _save_synthetic_snapshot(
    config: Mapping[str, Any],
    directory: Path,
    experiment: str,
    parameterization: SyntheticParameterization,
    class_names: list[str],
    iteration: int,
    ipc: int,
    partition_factor: int,
) -> Path:
    """Save an evaluation-compatible historical synthetic set."""

    images = parameterization.render_all().detach().cpu()
    labels = parameterization.labels.detach().cpu()
    snapshot_directory = directory / "synthetic_snapshots"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    path = (
        snapshot_directory
        / f"synthetic_iteration_{int(iteration):06d}.pt"
    )
    atomic_torch_save(
        {
            "images": images,
            "labels": labels,
            "class_names": list(class_names),
            "ipc": int(ipc),
            "partition_expansion": int(partition_factor),
            **_formation_metadata(ipc, partition_factor),
            "method": str(experiment),
            "algorithm_version": _algorithm_version(config["condensation"]),
            "parameterization": parameterization.mode,
            "iteration": int(iteration),
            "normalization": dict(
                config["data"]["image"]["normalization"]
            ),
            "pixel_range": [
                float(images.min().item()),
                float(images.max().item()),
            ],
            "snapshot": True,
        },
        path,
    )
    del images, labels
    return path


def _select_best_online_snapshot(
    directory: Path,
    records: list[Mapping[str, Any]],
    maximum_iteration: int,
) -> tuple[dict[str, Any], Path, int]:
    """Return the saved validation snapshot with the lowest finite loss."""

    candidates: list[tuple[float, int, dict[str, Any], Path]] = []
    for raw_record in records:
        record = dict(raw_record)
        try:
            iteration = int(record["iteration"])
            loss = float(record["loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            iteration <= 0
            or iteration > int(maximum_iteration)
            or not math.isfinite(loss)
            or str(record.get("split", "val")).lower() != "val"
        ):
            continue
        path = (
            directory
            / "synthetic_snapshots"
            / f"synthetic_iteration_{iteration:06d}.pt"
        )
        if path.is_file():
            candidates.append((loss, iteration, record, path))
    if not candidates:
        raise FileNotFoundError(
            "No validation-loss record has a matching synthetic snapshot"
        )
    loss, iteration, record, path = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    selected = {
        "metric": "validation_loss",
        "mode": "minimum",
        "split": "val",
        "iteration": int(iteration),
        "loss": float(loss),
        "accuracy": float(record.get("accuracy", 0.0)),
        "candidate_count": len(candidates),
        "snapshot": str(path.resolve()),
    }
    return selected, path, len(candidates)


def run_condensation(
    config: Mapping[str, Any],
    condensation_seed: int = 0,
    iteration_limit: int | None = None,
    ipc: int | None = None,
) -> dict[str, Any]:
    """Run size-weighted cluster IDM and resume compatible checkpoints."""

    condensation = config["condensation"]
    experiment = str(
        condensation.get("experiment_name", "size_weighted_cluster_idm")
    )
    ipc = _ipc(config, ipc)
    condensation["idm"]["ipc"] = int(ipc)
    partition_factor = _partition_factor(condensation, ipc)
    cluster_settings = condensation["cluster_matching"]
    cluster_initialization = bool(
        cluster_settings.get("initialization_enabled", False)
    )
    cluster_matching = bool(cluster_settings.get("matching_enabled", False))
    spread_enabled = bool(condensation["cluster_spread"]["enabled"])
    needs_clusters = cluster_initialization or cluster_matching
    cluster_count, radial_layers, angular_sectors, images_per_cluster = (
        _cluster_layout(cluster_settings, int(ipc))
        if needs_clusters
        else (int(ipc), 1, 1, 1)
    )
    design_signature = {
        "cluster_initialization": cluster_initialization,
        "cluster_matching": cluster_matching,
        "cluster_spread": spread_enabled,
        "cluster_size_weighted": cluster_matching,
        "cluster_center_loss_weight": (
            float(cluster_settings["center_loss_weight"])
            if cluster_matching
            else 0.0
        ),
        "cluster_spread_gradient_fraction": (
            float(condensation["cluster_spread"][
                "target_gradient_fraction"
            ])
            if spread_enabled
            else 0.0
        ),
        "ipc_clusters": int(cluster_count),
        "partition_expansion": int(partition_factor),
    }
    if images_per_cluster > 1:
        design_signature.update(
            {
                "stored_images_per_cluster": int(images_per_cluster),
                "radial_layers": int(radial_layers),
                "angular_sectors_per_layer": int(angular_sectors),
                "cluster_coverage": "balanced_radial_direction_cells",
            }
        )
    if spread_enabled and not cluster_matching:
        raise ValueError("Cluster spread requires cluster matching")
    pixel_constraints = dict(condensation.get("pixel_constraints", {}))
    algorithm_version = _algorithm_version(condensation)
    directory = (
        output_root(config)
        / f"ipc_{int(ipc)}"
        / f"condense_seed_{int(condensation_seed)}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    logger = get_stage_logger(
        f"idm_condense_{experiment}_{int(condensation_seed)}",
        directory,
    )
    device = resolve_device(config)
    seed = (
        int(config["project"]["seed"])
        + int(ipc) * 100000
        + int(condensation_seed) * 1009
    )
    seed_everything(
        seed, bool(config["project"].get("deterministic", False))
    )
    bundle = experiment_bundle(config)
    pool = IndexedImagePool(bundle.train, bundle.num_classes, seed + 17)
    cluster_index: PixelClusterIndex | None = None
    cluster_cache_path: Path | None = None
    if needs_clusters:
        cluster_index, cluster_cache_path = _load_or_build_cluster_index(
            config,
            pool,
            int(ipc),
            int(cluster_count),
            int(radial_layers),
            int(angular_sectors),
            seed + 41,
        )
        logger.info("pixel cluster index=%s", cluster_cache_path)
        atomic_write_json(
            {
                "cache": str(cluster_cache_path.resolve()),
                "clusters_per_class": int(cluster_count),
                "stored_images_per_cluster": int(images_per_cluster),
                "radial_layers": int(radial_layers),
                "angular_sectors_per_layer": int(angular_sectors),
                "coverage": "balanced_radial_direction_cells",
                "sizes": cluster_index.sizes,
                "descriptor_size": list(cluster_index.descriptor_size),
                "pca_components": int(cluster_index.pca_components),
                "deep_feature_extractor": False,
            },
            directory / "cluster_summary.json",
        )
    cluster_sampler = (
        ClusterSampler(cluster_index, seed + 53)
        if cluster_matching and cluster_index is not None
        else None
    )
    initial_pixels, labels = _initialize_synthetic_pixels(
        config,
        pool,
        cluster_index,
        bundle.num_classes,
        int(ipc),
        int(cluster_count),
        int(radial_layers),
        int(angular_sectors),
        int(partition_factor),
    )
    parameterization = SyntheticParameterization(
        initial_pixels,
        labels,
        device,
    ).to(device)
    optimizer, optimizer_name, learning_rate = _build_synthetic_optimizer(
        parameterization.variable,
        condensation,
    )
    queue = OfficialIDMQueue(
        config, bundle.num_classes, device, seed + 29
    )
    mixer = AuxiliaryGradientMixer(condensation["cluster_spread"])
    completed = 0
    diagnostics: dict[str, Any] = {}
    checkpoint_path = find_latest_checkpoint(directory)
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, "cpu")
        checkpoint_version = int(payload.get("algorithm_version", 1))
        if checkpoint_version != algorithm_version:
            raise ValueError(
                "断点算法版本不兼容："
                f"{checkpoint_path} 使用 v{checkpoint_version}，"
                f"当前配置使用 v{algorithm_version}。"
                "请为新版实验使用新的输出目录，或只移走该 IPC 的旧断点。"
            )
        checkpoint_design = dict(payload.get("design_signature", {}))
        if checkpoint_design != design_signature:
            raise ValueError(
                "Checkpoint design does not match the requested ablation: "
                f"checkpoint={checkpoint_design}, current={design_signature}"
            )
        checkpoint_pixel_constraints = payload.get("pixel_constraints")
        if (
            checkpoint_pixel_constraints is not None
            and dict(checkpoint_pixel_constraints) != pixel_constraints
        ):
            raise ValueError(
                "断点的像素投影设置与当前命令不一致："
                f"断点为 {checkpoint_pixel_constraints}，"
                f"当前为 {pixel_constraints}。"
                "请先把该 IPC 的现有输出移到独立目录，再开始新的对照实验。"
            )
        parameterization.load_state_dict(
            payload["parameterization"], strict=True
        )
        checkpoint_optimizer_name = str(
            payload.get("optimizer_name", "sgd")
        ).lower()
        if checkpoint_optimizer_name == optimizer_name:
            optimizer.load_state_dict(payload["optimizer"])
            _optimizer_to_device(optimizer, device)
        else:
            logger.warning(
                "断点优化器为 %s，当前配置为 %s；保留合成变量和迭代进度，"
                "但按当前配置重新初始化优化器状态",
                checkpoint_optimizer_name,
                optimizer_name,
            )
        for group in optimizer.param_groups:
            # 保留兼容优化器的历史状态，但允许 YAML 调整当前学习率。
            group["lr"] = float(learning_rate)
        queue.load_state_dict(payload.get("queue"))
        pool.load_state_dict(payload.get("real_pool"))
        mixer.load_state_dict(payload.get("auxiliary_mixer"))
        if cluster_sampler is not None:
            cluster_sampler.load_state_dict(payload.get("cluster_sampler"))
        restore_rng_state(payload.get("rng_state"))
        completed = int(payload.get("iteration", 0))
        diagnostics = dict(payload.get("diagnostics", {}))
        logger.info(
            "恢复 %s seed=%d：iteration=%d",
            experiment,
            condensation_seed,
            completed,
        )
    else:
        # 官方实现从 3 个模型开始，并在 iteration 0 首先加入第 4 个。
        queue.grow(0)
    target = int(
        iteration_limit
        if iteration_limit is not None
        else condensation["idm"]["iterations"]
    )
    online_settings = condensation.get("online_evaluation", {})
    select_best_by_loss = bool(
        online_settings.get("select_best_by_loss", False)
    )
    preview_per_class = int(
        condensation["idm"].get("preview_images_per_class", 10)
    )
    online_path = directory / "online_evaluation.json"
    online_payload = read_json(online_path, default={}) or {}
    online_records = list(online_payload.get("records", []))
    diagnostic_path = directory / "diagnostic_history.json"
    diagnostic_payload = read_json(diagnostic_path, default={}) or {}
    diagnostic_records = list(diagnostic_payload.get("records", []))
    if checkpoint_path is None:
        initial_images = parameterization.render_all().detach().cpu()
        initial_labels = parameterization.labels.detach().cpu()
        atomic_torch_save(
            {
                "images": initial_images,
                "labels": initial_labels,
                "seed": int(seed),
                "design": design_signature,
                "partition_expansion": int(partition_factor),
                **_formation_metadata(int(ipc), int(partition_factor)),
                "pixel_statistics": _pixel_statistics(
                    initial_images, config
                ),
            },
            directory / "synthetic_initial.pt",
        )
        initial_preview = _preview_subset(
            initial_images, initial_labels, preview_per_class
        )
        save_image(
            _display_images(config, initial_preview),
            directory / "preview_iteration_000000.png",
            nrow=min(int(ipc), max(1, preview_per_class)),
        )
        save_image(
            _normalize_preview(_display_images(config, initial_preview)),
            directory / "preview_normalized_iteration_000000.png",
            nrow=min(int(ipc), max(1, preview_per_class)),
        )
        expanded_initial, expanded_initial_labels, _ = partition_training_images(
            initial_images,
            initial_labels,
            partition_factor=int(partition_factor),
        )
        expanded_preview = _preview_subset(
            expanded_initial, expanded_initial_labels, preview_per_class
        )
        save_image(
            _display_images(config, expanded_preview),
            directory / "preview_partition_iteration_000000.png",
            nrow=max(1, min(preview_per_class, expanded_preview.shape[0])),
        )
        del (
            initial_images,
            initial_labels,
            initial_preview,
            expanded_initial,
            expanded_initial_labels,
            expanded_preview,
        )
    final_iteration = int(completed)
    for iteration in range(completed + 1, target + 1):
        final_iteration = int(iteration)
        started = time.perf_counter()
        current_image_learning_rate = float(
            condensation["idm"]["image_learning_rate"]
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_image_learning_rate
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        queue.grow(iteration)
        reliability_was_reset = queue.reset_reliability_if_due(iteration)
        guidance = queue.guidance_members()
        loss_sums = {
            "mean": 0.0,
            "cross_entropy": 0.0,
            "weighted_cross_entropy": 0.0,
            "cluster_spread": 0.0,
            "prior": 0.0,
        }
        real_microbatch = int(
            condensation["memory"]["real_feature_microbatch"]
        )
        projection_samples: list[dict[str, float]] = []
        # Official IDM takes one synthetic optimizer step per class. Collapsing
        # these into one step under-trains a C-class dataset by about C times.
        for class_id in range(bundle.num_classes):
            optimizer.zero_grad(set_to_none=True)
            for member in guidance:
                batch_real = int(
                    _idm_value(condensation["idm"], "batch_real", int(ipc))
                )
                if cluster_matching:
                    if cluster_sampler is None:
                        raise RuntimeError("Missing cluster sampler")
                    real_cpu, real_groups_cpu = _sample_clustered_real_batch(
                        pool,
                        cluster_sampler,
                        class_id,
                        int(cluster_count),
                        batch_real,
                    )
                else:
                    real_cpu = pool.sample(class_id, batch_real)
                    real_groups_cpu = None
                dsa_seed = queue.random.randrange(0, 100000)
                real_features, real_microbatch = _real_features(
                    config,
                    member.model,
                    real_cpu,
                    dsa_seed,
                    device,
                )
                del real_cpu
                real_groups = (
                    real_groups_cpu.to(device, non_blocking=True)
                    if real_groups_cpu is not None
                    else None
                )
                with autocast_context(config, device):
                    generated = parameterization.render_class(class_id)
                generated = generated.float()
                stored_labels = parameterization.labels[
                    parameterization.labels == int(class_id)
                ]
                stored_groups = (
                    _stored_cluster_groups(
                        int(cluster_count), int(images_per_cluster), device
                    )
                    if cluster_matching
                    else None
                )
                stored_rings = (
                    _stored_radial_rings(
                        int(cluster_count),
                        int(radial_layers),
                        int(angular_sectors),
                        device,
                    )
                    if cluster_matching and radial_layers > 1
                    else None
                )
                synthetic, synthetic_labels, synthetic_groups = (
                    partition_training_images(
                        generated,
                        stored_labels,
                        partition_factor=int(partition_factor),
                        group_ids=stored_groups,
                    )
                )
                synthetic_rings = (
                    stored_rings.repeat(int(partition_factor) ** 2)
                    if stored_rings is not None
                    else None
                )
                synthetic = diff_augment(
                    synthetic,
                    str(condensation["idm"]["dsa_strategy"]),
                    seed=dsa_seed,
                    param=ParamDiffAug(),
                )
                synthetic_output = member.model.forward_idm(synthetic)
                if cluster_matching:
                    if real_groups is None or synthetic_groups is None:
                        raise RuntimeError("Cluster group metadata is missing")
                    spread_settings = condensation["cluster_spread"]
                    if cluster_index is None:
                        raise RuntimeError("Missing cluster-size metadata")
                    mean_loss, spread_raw = cluster_distribution_losses(
                        real_features,
                        real_groups,
                        synthetic_output.embedding,
                        synthetic_groups,
                        cluster_count=int(cluster_count),
                        cluster_weights=cluster_index.sizes[int(class_id)],
                        radial_weight=(
                            float(spread_settings["radial_weight"])
                            if spread_enabled
                            else 0.0
                        ),
                        standard_deviation_weight=(
                            float(spread_settings["standard_deviation_weight"])
                            if spread_enabled
                            else 0.0
                        ),
                        smooth_l1_beta=float(
                            spread_settings["smooth_l1_beta"]
                        ),
                        synthetic_ring_ids=synthetic_rings,
                        ring_count=int(radial_layers),
                        angular_weight=(
                            float(spread_settings.get("angular_weight", 1.0))
                            if angular_sectors > 1
                            else 0.0
                        ),
                    )
                    mean_loss = mean_loss * float(
                        cluster_settings["center_loss_weight"]
                    )
                else:
                    mean_loss = (
                        synthetic_output.embedding.float().mean(0)
                        - real_features.mean(0).detach()
                    ).square().sum()
                    spread_raw = generated.sum() * 0.0
                ce_loss = F.cross_entropy(
                    synthetic_output.logits.float(), synthetic_labels
                )
                reliability = max(
                    float(condensation["idm"]["minimum_reliability"]),
                    member.reliability_percent,
                )
                weighted_ce_loss = (
                    _ce_weight(config, int(ipc))
                    * reliability
                    * ce_loss
                )
                base_loss = mean_loss + weighted_ce_loss
                base_image_gradient = torch.autograd.grad(
                    base_loss,
                    generated,
                    retain_graph=spread_enabled,
                )[0]
                image_gradient = base_image_gradient
                if spread_enabled:
                    spread_image_gradient = torch.autograd.grad(
                        spread_raw,
                        generated,
                        retain_graph=False,
                    )[0]
                    weighted_spread, _ = mixer.weight_against_base(
                        base_image_gradient,
                        spread_image_gradient,
                    )
                    image_gradient = image_gradient + weighted_spread
                    del spread_image_gradient, weighted_spread
                del base_image_gradient
                # 多网络指导时顺序计算、平均梯度，整个类别只更新一次；
                # 避免 train_net_num 同时偷偷放大有效优化步数和显存峰值。
                generated.backward(
                    image_gradient / float(max(1, len(guidance)))
                )
                loss_sums["mean"] += float(mean_loss.detach().item())
                loss_sums["cross_entropy"] += float(
                    ce_loss.detach().item()
                )
                loss_sums["weighted_cross_entropy"] += float(
                    weighted_ce_loss.detach().item()
                )
                loss_sums["cluster_spread"] += float(
                    spread_raw.detach().item()
                )
                del (
                    generated,
                    synthetic,
                    synthetic_labels,
                    synthetic_output,
                    mean_loss,
                    ce_loss,
                    weighted_ce_loss,
                    base_loss,
                    spread_raw,
                    image_gradient,
                    real_features,
                    real_groups,
                    real_groups_cpu,
                    stored_labels,
                    stored_groups,
                    stored_rings,
                    synthetic_groups,
                    synthetic_rings,
                )
            prior = parameterization.prior_loss(class_id)
            if float(prior.detach().item()) != 0.0:
                prior.backward()
            loss_sums["prior"] += (
                float(prior.detach().item()) * max(1, len(guidance))
            )
            del prior
            optimizer.step()
            if bool(pixel_constraints.get("enabled", False)):
                projection_samples.append(
                    _project_synthetic_pixels(
                        parameterization.variable,
                        optimizer,
                        config,
                    )
                )
        if projection_samples:
            projection_metrics = {
                key: statistics.fmean(
                    float(sample[key]) for sample in projection_samples
                )
                for key in projection_samples[0]
            }
        else:
            projection_metrics = {
                "pixel/pre_projection_minimum": float(
                    parameterization.variable.detach().min().item()
                ),
                "pixel/pre_projection_maximum": float(
                    parameterization.variable.detach().max().item()
                ),
                "pixel/projected_fraction": 0.0,
                "pixel/momentum_cleared_fraction": 0.0,
                **_pixel_statistics(parameterization.variable, config),
            }
        queue_metrics = queue.train_independent_members(pool)
        divisor = float(bundle.num_classes * max(1, len(guidance)))
        peak_mib = cuda_peak_megabytes()
        total_mib = (
            float(torch.cuda.get_device_properties(device).total_memory)
            / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        recent_gradient_count = bundle.num_classes
        diagnostics = {
            **{
                f"loss/{key}": value / divisor
                for key, value in loss_sums.items()
            },
            **queue_metrics,
            **projection_metrics,
            "queue/reliability_reset": bool(reliability_was_reset),
            "design/cluster_initialization": cluster_initialization,
            "design/cluster_matching": cluster_matching,
            "design/cluster_size_weighted": cluster_matching,
            "design/cluster_center_loss_weight": (
                float(cluster_settings["center_loss_weight"])
                if cluster_matching
                else 0.0
            ),
            "cluster_spread/enabled": spread_enabled,
            "cluster_spread/target_gradient_fraction": (
                mixer.target_fraction() if spread_enabled else 0.0
            ),
            "cluster_spread/gradient_scale": (
                mixer.recent_mean(
                    "gradient_scales", recent_gradient_count
                )
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/base_gradient_norm": (
                mixer.recent_mean("base_norms", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/raw_gradient_norm": (
                mixer.recent_mean("auxiliary_norms", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/gradient_cosine": (
                mixer.recent_mean("gradient_cosines", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/conflict_projection_rate": (
                mixer.recent_mean("conflict_projections", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/achieved_gradient_fraction": (
                mixer.recent_mean(
                    "achieved_fractions", recent_gradient_count
                )
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/mixed_gradients": int(mixer.mixed_gradients),
            "cluster_spread/skipped_gradients": int(mixer.skipped_gradients),
            "optimization/image_learning_rate": float(
                current_image_learning_rate
            ),
            "memory/real_feature_microbatch": int(real_microbatch),
            "memory/cuda_peak_mib": float(peak_mib),
            "memory/cuda_peak_fraction": (
                float(peak_mib / total_mib) if total_mib > 0 else 0.0
            ),
            "time/iteration_seconds": time.perf_counter() - started,
        }
        online_interval = int(
            online_settings.get("interval_iterations", 0)
        )
        should_validate = (
            online_interval > 0
            and iteration % online_interval == 0
        )
        if should_validate:
            synthetic_for_validation = (
                parameterization.render_all().detach().cpu()
            )
            monitor_split = str(
                config["data"].get("online_evaluation_split", "val")
            ).lower()
            monitor_dataset = (
                bundle.test if monitor_split == "test" else bundle.val
            )
            monitor = _quick_evaluate(
                config,
                synthetic_for_validation,
                parameterization.labels.detach().cpu(),
                monitor_dataset,
                bundle.num_classes,
                iteration,
                seed + 700001,
                int(partition_factor),
                device,
            )
            del synthetic_for_validation
            monitor["split"] = monitor_split
            monitor["selects_checkpoint"] = select_best_by_loss
            monitor["can_stop_training"] = False
            online_records.append(monitor)
            atomic_write_json(
                {
                    "purpose": (
                        "validation_loss_snapshot_selection"
                        if select_best_by_loss
                        else "training_progress_only"
                    ),
                    "split": monitor_split,
                    "selects_checkpoint": select_best_by_loss,
                    "can_stop_training": False,
                    "records": online_records,
                },
                online_path,
            )
            diagnostics.update(
                {
                    "online/accuracy": float(monitor["accuracy"]),
                    "online/loss": float(monitor["loss"]),
                }
            )
        should_log = (
            iteration == 1
            or iteration
            % int(condensation["idm"]["log_interval_iterations"])
            == 0
            or iteration == target
        )
        if should_log:
            logger.info(
                "%s seed=%d iter=%d/%d mean=%.5f "
                "image_lr=%.4g ce_coeff=%.3f ce_loss_w=%.5f "
                "spread=%.5f spread_scale=%.4g spread_fraction=%.4f "
                "pixel_pre=[%.4f,%.4f] projected=%.4f "
                "queue=%d updates=%.1f peak=%.0fMiB time=%.2fs",
                experiment,
                condensation_seed,
                iteration,
                target,
                diagnostics["loss/mean"],
                diagnostics["optimization/image_learning_rate"],
                _ce_weight(config, int(ipc)),
                diagnostics["loss/weighted_cross_entropy"],
                diagnostics["loss/cluster_spread"],
                diagnostics["cluster_spread/gradient_scale"],
                diagnostics["cluster_spread/achieved_gradient_fraction"],
                diagnostics["pixel/pre_projection_minimum"],
                diagnostics["pixel/pre_projection_maximum"],
                diagnostics["pixel/projected_fraction"],
                int(diagnostics["queue/size"]),
                diagnostics["queue/mean_updates"],
                diagnostics["memory/cuda_peak_mib"],
                diagnostics["time/iteration_seconds"],
            )
            diagnostic_records.append(
                {"iteration": int(iteration), **diagnostics}
            )
            atomic_write_json(
                {
                    "experiment": experiment,
                    "seed": int(seed),
                    "records": diagnostic_records,
                },
                diagnostic_path,
            )
        checkpoint_interval = int(
            condensation["idm"]["checkpoint_interval_iterations"]
        )
        if iteration % checkpoint_interval == 0 or iteration == target:
            _save_checkpoint(
                directory,
                experiment,
                iteration,
                parameterization,
                optimizer,
                queue,
                pool,
                mixer,
                cluster_sampler,
                design_signature,
                pixel_constraints,
                diagnostics,
                algorithm_version,
            )
        snapshot_interval = int(
            condensation["idm"].get(
                "synthetic_snapshot_interval_iterations", 1000
            )
        )
        if (
            snapshot_interval > 0
            and iteration % snapshot_interval == 0
        ):
            snapshot_path = _save_synthetic_snapshot(
                config,
                directory,
                experiment,
                parameterization,
                bundle.class_names,
                iteration,
                int(ipc),
                int(partition_factor),
            )
            logger.info(
                "saved synthetic snapshot iteration=%d path=%s",
                iteration,
                snapshot_path,
            )
            cleanup_memory()
        preview_interval = int(
            condensation["idm"]["preview_interval_iterations"]
        )
        if (
            preview_interval > 0
            and iteration % preview_interval == 0
        ):
            preview = parameterization.render_all().detach().cpu()
            preview_labels = parameterization.labels.detach().cpu()
            preview = _preview_subset(
                preview, preview_labels, preview_per_class
            )
            save_image(
                _display_images(config, preview),
                directory / f"preview_iteration_{iteration:06d}.png",
                nrow=min(int(ipc), max(1, preview_per_class)),
            )
            save_image(
                _normalize_preview(_display_images(config, preview)),
                directory
                / f"preview_normalized_iteration_{iteration:06d}.png",
                nrow=min(int(ipc), max(1, preview_per_class)),
            )
            del preview, preview_labels
            cleanup_memory()
    selected_iteration = int(final_iteration)
    selection: dict[str, Any] | None = None
    if select_best_by_loss:
        selection, selected_snapshot_path, _ = _select_best_online_snapshot(
            directory,
            online_records,
            final_iteration,
        )
        selected_payload = load_checkpoint(selected_snapshot_path, "cpu")
        selected_images = selected_payload["images"].detach().float().cpu()
        selected_labels = selected_payload["labels"].detach().long().cpu()
        expected_labels = parameterization.labels.detach().long().cpu()
        if tuple(selected_images.shape) != tuple(parameterization.variable.shape):
            raise ValueError(
                "Selected synthetic snapshot has an incompatible image shape"
            )
        if not torch.equal(selected_labels, expected_labels):
            raise ValueError(
                "Selected synthetic snapshot has incompatible labels"
            )
        with torch.no_grad():
            parameterization.variable.copy_(selected_images.to(device))
        selected_iteration = int(selection["iteration"])
        logger.info(
            "selected validation-loss snapshot iteration=%d loss=%.6f "
            "candidates=%d",
            selected_iteration,
            float(selection["loss"]),
            int(selection["candidate_count"]),
        )
        del selected_payload, selected_images, selected_labels, expected_labels
        online_records = [
            {
                **dict(record),
                "selected": int(record.get("iteration", -1))
                == selected_iteration,
            }
            for record in online_records
        ]
        atomic_write_json(
            {
                "purpose": "validation_loss_snapshot_selection",
                "split": "val",
                "selects_checkpoint": True,
                "can_stop_training": False,
                "selection": selection,
                "records": online_records,
            },
            online_path,
        )
    synthetic_path = _export(
        config,
        directory,
        experiment,
        parameterization,
        bundle.class_names,
        selected_iteration,
        int(ipc),
        int(partition_factor),
        int(preview_per_class),
        optimization_iterations=int(final_iteration),
        selection=selection,
    )
    summary = {
        "experiment": experiment,
        "condensation_seed": int(condensation_seed),
        "seed": int(seed),
        "parameterization": "pixel",
        "algorithm_version": int(algorithm_version),
        "design": design_signature,
        "ipc": int(ipc),
        "partition_expansion": int(partition_factor),
        **_formation_metadata(int(ipc), int(partition_factor)),
        "cluster_cache": (
            str(cluster_cache_path.resolve())
            if cluster_cache_path is not None
            else None
        ),
        "cluster_sizes": cluster_index.sizes if cluster_index is not None else None,
        "resolved_idm_protocol": {
            "image_learning_rate": float(
                condensation["idm"]["image_learning_rate"]
            ),
            "image_momentum": float(condensation["idm"]["image_momentum"]),
            "ce_weight": float(_ce_weight(config, int(ipc))),
            "batch_real": int(
                _idm_value(condensation["idm"], "batch_real", int(ipc))
            ),
            "batch_train": int(
                _idm_value(condensation["idm"], "batch_train", int(ipc))
            ),
            "net_num": int(
                _idm_value(condensation["idm"], "net_num", int(ipc))
            ),
            "net_generate_interval": int(
                _idm_value(
                    condensation["idm"],
                    "net_generate_interval",
                    int(ipc),
                )
            ),
            "reliability_reset_interval": int(
                _idm_value(
                    condensation["idm"],
                    "reliability_reset_interval",
                    int(ipc),
                )
            ),
            "synthetic_updates_per_iteration": int(bundle.num_classes),
        },
        "iterations": int(final_iteration),
        "maximum_iterations": int(target),
        "selected_iteration": int(selected_iteration),
        "snapshot_selection": selection,
        "stopped_early": False,
        "synthetic_dataset": str(synthetic_path.resolve()),
        "auxiliary_gradient_mixer": mixer.state_dict(),
        "pixel_constraints": pixel_constraints,
        "diagnostics": diagnostics,
    }
    atomic_write_json(summary, directory / "summary.json")
    del parameterization, optimizer, queue, pool, mixer, cluster_sampler
    cleanup_memory()
    return summary
