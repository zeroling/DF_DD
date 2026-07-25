"""标准 IDM condense 阶段，以及 topology / VAE / diffusion 消融扩展。"""

from __future__ import annotations

from dataclasses import dataclass
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
from Core.io_utils import atomic_write_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, resolve_device
from Core.seed import seed_everything
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    output_root,
)
from Pipeline.ablation_data import deterministic_bundle
from Net.Condensation.generative import (
    GenerativeBundle,
    decode_z0,
    decode_zT,
    encode_z0,
    initialize_zT,
    load_generators,
)
from Net.Condensation.idm_official import (
    IDMConvNet6,
    ParamDiffAug,
    build_idm_convnet6,
    cumulative_accuracy,
    diff_augment,
    initialize_partitioned_pixels,
    partition_and_expand,
)
from Core.experiment_runtime import (
    cleanup_memory,
    cuda_peak_megabytes,
)
from Net.Condensation.topology import (
    feature_nodes,
    median_bandwidth,
    probability_divergence,
    rbf_row_distribution,
    squared_node_distances,
    topology_distribution,
)


class IndexedImagePool:
    """不放回循环采样真实图；图像首次读取后缓存为 CPU uint8。"""

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
                .mul(255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .contiguous()
            )
            self.cache[int(index)] = cached
        return cached.float().div_(255.0)

    def sample(self, class_id: int, count: int) -> torch.Tensor:
        return torch.stack(
            [self._image(index) for index in self._take_class(class_id, count)]
        )

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


@dataclass
class QueueMember:
    identifier: int
    model: IDMConvNet6
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
        self.settings = condensation_settings(config)
        self.num_classes = int(num_classes)
        self.device = device
        self.random = random.Random(int(seed))
        self.members: list[QueueMember] = []
        self.next_identifier = 0
        for _ in range(3):
            self.members.append(self._new_member(0))

    def _new_member(self, birth_iteration: int) -> QueueMember:
        model = build_idm_convnet6(
            int(self.config["data"]["image"]["channels"]),
            self.num_classes,
            self.config["data"]["image"]["size"],
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

    def grow(self, iteration: int) -> None:
        settings = self.settings["idm"]
        if int(iteration) % int(settings["net_generate_interval"]) != 0:
            return
        if len(self.members) == int(settings["net_num"]):
            self.members.pop(0)
            cleanup_memory()
        self.members.append(self._new_member(int(iteration)))

    def guidance_members(self) -> list[QueueMember]:
        indices = list(range(len(self.members)))
        self.random.shuffle(indices)
        count = min(int(self.settings["idm"]["train_net_num"]), len(indices))
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
                        # 累加后等价于一次有效 batch=128 的平均交叉熵。
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
    """pixel、VAE z0 或完整扩散 zT 的统一每类渲染接口。"""

    def __init__(
        self,
        mode: str,
        initial_pixels: torch.Tensor,
        labels: torch.Tensor,
        config: Mapping[str, Any],
        generators: GenerativeBundle | None,
        device: torch.device,
    ):
        super().__init__()
        self.mode = str(mode)
        self.config = config
        self.settings = condensation_settings(config)
        self.generators = generators
        self.num_classes = int(labels.numel())
        self.register_buffer("labels", labels.to(device))
        if self.mode == "pixel":
            initial = initial_pixels.to(device)
        else:
            if generators is None:
                raise RuntimeError(f"{self.mode} 需要 VAE")
            with torch.no_grad(), autocast_context(
                config, device
            ):
                clean = encode_z0(generators, initial_pixels.to(device))
                initial = (
                    clean
                    if self.mode == "vae_z0"
                    else initialize_zT(config, clean)
                )
        # 可学习变量固定 FP32；前向时由 autocast 临时降精度，避免优化器动量和
        # 无 autocast 导出阶段继承 BF16 dtype。
        self.variable = nn.Parameter(initial.detach().float())

    def render_class(self, class_id: int) -> torch.Tensor:
        selected = self.variable[int(class_id) : int(class_id) + 1]
        if self.mode == "pixel":
            return selected
        if self.generators is None:
            raise RuntimeError("latent 参数化缺少生成模型")
        if self.mode == "vae_z0":
            return decode_z0(self.generators, selected)
        if self.mode == "diffusion_zT":
            return decode_zT(
                self.config,
                self.generators,
                selected,
                self.labels[int(class_id) : int(class_id) + 1],
                self.num_classes,
                bool(
                    self.settings["latent"].get(
                        "gradient_checkpointing", True
                    )
                ),
            )
        raise ValueError(f"未知合成参数化：{self.mode}")

    @torch.no_grad()
    def render_all(self) -> torch.Tensor:
        return torch.cat(
            [self.render_class(class_id) for class_id in range(self.num_classes)]
        ).float()

    def prior_loss(self, class_id: int) -> torch.Tensor:
        if self.mode == "pixel":
            return self.variable.sum() * 0.0
        weight = float(
            self.settings["latent"]["prior"].get(self.mode, 0.0)
        )
        latent = self.variable[int(class_id) : int(class_id) + 1]
        mean = latent.float().mean()
        std = latent.float().std(unbiased=False)
        return float(weight) * (mean.square() + (std - 1.0).square())


class TopologyCalibrator:
    """用训练梯度把 Topology 固定在目标贡献范围，不查看测试集。"""

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.candidates: list[float] = []
        self.lambda_value = 1.0
        self.frozen = False

    def should_measure(self, iteration: int) -> bool:
        return (
            not self.frozen
            and int(iteration)
            in set(map(int, self.settings["calibration_iterations"]))
        )

    def observe(
        self,
        iteration: int,
        base_gradient: torch.Tensor,
        topology_gradient: torch.Tensor,
    ) -> float:
        base_norm = float(base_gradient.detach().float().norm().item())
        topology_norm = float(
            topology_gradient.detach().float().norm().item()
        )
        target = float(self.settings["target_gradient_fraction"])
        ratio = target / max(1.0e-8, 1.0 - target)
        candidate = ratio * base_norm / max(topology_norm, 1.0e-12)
        candidate = min(
            float(self.settings["maximum_lambda"]),
            max(float(self.settings["minimum_lambda"]), candidate),
        )
        self.candidates.append(float(candidate))
        self.lambda_value = float(statistics.median(self.candidates))
        if int(iteration) >= max(
            map(int, self.settings["calibration_iterations"])
        ):
            self.frozen = True
        return self.lambda_value

    def state_dict(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "lambda_value": float(self.lambda_value),
            "frozen": bool(self.frozen),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self.candidates = [
            float(value) for value in state.get("candidates", [])
        ]
        self.lambda_value = float(state.get("lambda_value", 1.0))
        self.frozen = bool(state.get("frozen", False))


def _real_target(
    config: Mapping[str, Any],
    model: IDMConvNet6,
    images_cpu: torch.Tensor,
    dsa_seed: int,
    include_topology: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]], int]:
    """在建立生成器计算图前分块提取有效 batch=128 的真实目标。"""

    ablation = condensation_settings(config)
    requested = int(ablation["memory"]["real_feature_microbatch"])
    minimum = int(ablation["memory"].get("retry_minimum", 1))
    settings = ablation["topology"]
    while True:
        embedding_sum: torch.Tensor | None = None
        node_parts: dict[str, list[torch.Tensor]] = {
            "shallow": [],
            "middle": [],
            "deep": [],
        }
        try:
            with torch.no_grad():
                for start in range(0, images_cpu.shape[0], requested):
                    images = images_cpu[start : start + requested].to(
                        device, non_blocking=True
                    )
                    images = diff_augment(
                        images,
                        str(ablation["idm"]["dsa_strategy"]),
                        seed=int(dsa_seed),
                        param=ParamDiffAug(),
                    )
                    output = model.forward_idm(images, include_topology)
                    current = output.embedding.float().sum(0)
                    embedding_sum = (
                        current
                        if embedding_sum is None
                        else embedding_sum + current
                    )
                    if include_topology:
                        for level, feature in output.spatial.items():
                            node_parts[level].append(
                                feature_nodes(
                                    feature,
                                    settings["grids"][level],
                                )
                                .detach()
                                .cpu()
                            )
                    del images, output
            break
        except torch.OutOfMemoryError:
            cleanup_memory()
            if requested <= minimum:
                raise
            requested = max(minimum, requested // 2)
    if embedding_sum is None:
        raise ValueError("真实特征 batch 为空")
    embedding_mean = embedding_sum / float(images_cpu.shape[0])
    topology_targets: dict[
        str, tuple[torch.Tensor, torch.Tensor]
    ] = {}
    if include_topology:
        for level, pieces in node_parts.items():
            nodes = torch.cat(pieces, dim=0).float()
            distances = squared_node_distances(nodes)
            bandwidth = median_bandwidth(
                distances,
                float(settings["minimum_bandwidth"]),
            )
            distribution = rbf_row_distribution(
                distances, bandwidth
            ).mean(0)
            distribution = distribution / distribution.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            topology_targets[level] = (
                distribution.to(device),
                bandwidth.to(device),
            )
            del nodes, distances
    return embedding_mean, topology_targets, requested


def _topology_loss(
    config: Mapping[str, Any],
    synthetic_spatial: Mapping[str, torch.Tensor],
    targets: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, float]]:
    settings = condensation_settings(config)["topology"]
    total = next(iter(synthetic_spatial.values())).sum() * 0.0
    values: dict[str, float] = {}
    for level in ("shallow", "middle", "deep"):
        real_distribution, bandwidth = targets[level]
        synthetic_distribution, _ = topology_distribution(
            synthetic_spatial[level],
            settings["grids"][level],
            bandwidth=bandwidth,
            minimum_bandwidth=float(settings["minimum_bandwidth"]),
        )
        value = probability_divergence(
            real_distribution.detach(),
            synthetic_distribution,
            str(settings["divergence"]),
        )
        total = total + float(settings["level_weights"][level]) * value
        values[level] = float(value.detach().item())
    return total, values


def _save_checkpoint(
    directory: Path,
    experiment: str,
    iteration: int,
    parameterization: SyntheticParameterization,
    optimizer: torch.optim.Optimizer,
    queue: OfficialIDMQueue,
    pool: IndexedImagePool,
    calibrator: TopologyCalibrator,
    diagnostics: Mapping[str, Any],
) -> Path:
    payload = {
        "experiment": str(experiment),
        "iteration": int(iteration),
        "parameterization": parameterization.state_dict(),
        "optimizer": optimizer.state_dict(),
        "queue": queue.state_dict(),
        "real_pool": pool.state_dict(),
        "topology_calibrator": calibrator.state_dict(),
        "diagnostics": dict(diagnostics),
        "rng_state": capture_rng_state(),
    }
    return atomic_torch_save(payload, directory / "checkpoint_last.pt")


def _export(
    directory: Path,
    experiment: str,
    parameterization: SyntheticParameterization,
    class_names: list[str],
    iteration: int,
) -> Path:
    images = parameterization.render_all().detach().cpu()
    labels = parameterization.labels.detach().cpu()
    path = directory / "synthetic.pt"
    atomic_torch_save(
        {
            "images": images,
            "labels": labels,
            "class_names": list(class_names),
            "ipc": 1,
            "method": str(experiment),
            "parameterization": parameterization.mode,
            "iteration": int(iteration),
            "pixel_range": [
                float(images.min().item()),
                float(images.max().item()),
            ],
        },
        path,
    )
    save_image(
        images.clamp(0, 1),
        directory / "preview.png",
        nrow=len(class_names),
    )
    expanded, _ = partition_and_expand(images, labels, 2)
    save_image(
        expanded.clamp(0, 1),
        directory / "preview_partition_expansion.png",
        nrow=4,
    )
    return path


def run_condensation(
    config: Mapping[str, Any],
    experiment: str,
    condensation_seed: int,
    iteration_limit: int | None = None,
) -> dict[str, Any]:
    """运行一个 C 方法的一个 condensation seed；自动从无哈希断点继续。"""

    experiment = str(experiment)
    ablation = condensation_settings(config)
    definition = ablation_settings(config)["methods"][experiment]
    mode = str(definition["parameterization"])
    include_topology = bool(definition["topology"])
    directory = (
        output_root(config)
        / "C"
        / experiment
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
        + int(experiment[1:]) * 100000
        + int(condensation_seed) * 1009
    )
    seed_everything(
        seed, bool(config["project"].get("deterministic", False))
    )
    bundle = deterministic_bundle(config)
    pool = IndexedImagePool(bundle.train, bundle.num_classes, seed + 17)
    initial_pixels, labels = initialize_partitioned_pixels(
        pool,
        bundle.num_classes,
        config["data"]["image"]["size"],
    )
    generators: GenerativeBundle | None = None
    if mode != "pixel":
        generators = load_generators(
            config,
            bundle.num_classes,
            device,
            require_diffusion=mode == "diffusion_zT",
        )
    parameterization = SyntheticParameterization(
        mode,
        initial_pixels,
        labels,
        config,
        generators,
        device,
    ).to(device)
    learning_rate = (
        float(ablation["idm"]["image_learning_rate"])
        if mode == "pixel"
        else float(ablation["latent"]["learning_rate"][mode])
    )
    optimizer = torch.optim.SGD(
        [parameterization.variable],
        lr=learning_rate,
        momentum=float(ablation["idm"]["image_momentum"]),
    )
    queue = OfficialIDMQueue(
        config, bundle.num_classes, device, seed + 29
    )
    calibrator = TopologyCalibrator(
        ablation["topology"]
    )
    completed = 0
    diagnostics: dict[str, Any] = {}
    checkpoint_path = find_latest_checkpoint(directory)
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, "cpu")
        parameterization.load_state_dict(
            payload["parameterization"], strict=True
        )
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:
            # 动量缓冲继续沿用断点，但允许用户直接在 YAML 中调整当前学习率。
            group["lr"] = float(learning_rate)
            group["momentum"] = float(ablation["idm"]["image_momentum"])
        queue.load_state_dict(payload.get("queue"))
        pool.load_state_dict(payload.get("real_pool"))
        calibrator.load_state_dict(payload.get("topology_calibrator"))
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
        else ablation["idm"]["iterations"]
    )
    for iteration in range(completed + 1, target + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        queue.grow(iteration)
        guidance = queue.guidance_members()
        loss_sums = {
            "mean": 0.0,
            "cross_entropy": 0.0,
            "topology": 0.0,
            "prior": 0.0,
        }
        real_microbatch = int(
            ablation["memory"]["real_feature_microbatch"]
        )
        for class_id in range(bundle.num_classes):
            for member in guidance:
                # 真实目标先完成并释放，避免与 DDIM 计算图重叠。
                real_cpu = pool.sample(
                    class_id, int(ablation["idm"]["batch_real"])
                )
                dsa_seed = queue.random.randrange(0, 100000)
                real_mean, topology_targets, real_microbatch = _real_target(
                    config,
                    member.model,
                    real_cpu,
                    dsa_seed,
                    include_topology,
                    device,
                )
                del real_cpu
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config, device):
                    generated = parameterization.render_class(class_id)
                # VAE/DDIM 可用 BF16 节省显存；IDM/官方 DSA 主路径保持 FP32。
                generated = generated.float()
                synthetic, synthetic_labels = partition_and_expand(
                    generated,
                    parameterization.labels[
                        class_id : class_id + 1
                    ],
                    int(ablation["idm"]["partition_expansion"]),
                )
                synthetic = diff_augment(
                    synthetic,
                    str(ablation["idm"]["dsa_strategy"]),
                    seed=dsa_seed,
                    param=ParamDiffAug(),
                )
                synthetic_output = member.model.forward_idm(
                    synthetic, include_topology
                )
                mean_loss = (
                    synthetic_output.embedding.float().mean(0)
                    - real_mean.detach()
                ).square().sum()
                ce_loss = F.cross_entropy(
                    synthetic_output.logits.float(), synthetic_labels
                )
                reliability = max(
                    float(ablation["idm"]["minimum_reliability"]),
                    member.reliability_percent,
                )
                base_loss = mean_loss + (
                    float(ablation["idm"]["ce_weight"])
                    * reliability
                    * ce_loss
                )
                topology_raw = generated.sum() * 0.0
                topology_values: dict[str, float] = {}
                if include_topology:
                    topology_raw, topology_values = _topology_loss(
                        config,
                        synthetic_output.spatial,
                        topology_targets,
                    )
                if include_topology and calibrator.should_measure(iteration):
                    base_image_gradient = torch.autograd.grad(
                        base_loss,
                        generated,
                        retain_graph=True,
                    )[0]
                    topology_image_gradient = torch.autograd.grad(
                        topology_raw,
                        generated,
                        retain_graph=True,
                    )[0]
                    base_variable_gradient = torch.autograd.grad(
                        generated,
                        parameterization.variable,
                        grad_outputs=base_image_gradient,
                        retain_graph=True,
                    )[0]
                    topology_variable_gradient = torch.autograd.grad(
                        generated,
                        parameterization.variable,
                        grad_outputs=topology_image_gradient,
                        retain_graph=True,
                    )[0]
                    calibrator.observe(
                        iteration,
                        base_variable_gradient,
                        topology_variable_gradient,
                    )
                    image_gradient = (
                        base_image_gradient
                        + float(calibrator.lambda_value)
                        * topology_image_gradient
                    )
                else:
                    total_without_prior = base_loss + (
                        float(calibrator.lambda_value) * topology_raw
                        if include_topology
                        else 0.0
                    )
                    image_gradient = torch.autograd.grad(
                        total_without_prior,
                        generated,
                        retain_graph=True,
                    )[0]
                generated.backward(image_gradient)
                prior = parameterization.prior_loss(class_id)
                if float(prior.detach().item()) != 0.0:
                    prior.backward()
                optimizer.step()
                loss_sums["mean"] += float(mean_loss.detach().item())
                loss_sums["cross_entropy"] += float(
                    ce_loss.detach().item()
                )
                loss_sums["topology"] += float(
                    topology_raw.detach().item()
                )
                loss_sums["prior"] += float(prior.detach().item())
                del (
                    generated,
                    synthetic,
                    synthetic_labels,
                    synthetic_output,
                    mean_loss,
                    ce_loss,
                    base_loss,
                    topology_raw,
                    prior,
                    image_gradient,
                    topology_targets,
                )
        queue_metrics = queue.train_independent_members(pool)
        divisor = float(bundle.num_classes * max(1, len(guidance)))
        diagnostics = {
            **{
                f"loss/{key}": value / divisor
                for key, value in loss_sums.items()
            },
            **queue_metrics,
            "topology/lambda": float(calibrator.lambda_value)
            if include_topology
            else 0.0,
            "topology/calibrated": bool(calibrator.frozen),
            "memory/real_feature_microbatch": int(real_microbatch),
            "memory/cuda_peak_mib": cuda_peak_megabytes(),
            "time/iteration_seconds": time.perf_counter() - started,
        }
        if (
            iteration == 1
            or iteration
            % int(ablation["idm"]["log_interval_iterations"])
            == 0
            or iteration == target
        ):
            logger.info(
                "%s seed=%d iter=%d/%d mean=%.5f ce=%.5f topo=%.5f "
                "lambda=%.4f queue=%d updates=%.1f peak=%.0fMiB time=%.2fs",
                experiment,
                condensation_seed,
                iteration,
                target,
                diagnostics["loss/mean"],
                diagnostics["loss/cross_entropy"],
                diagnostics["loss/topology"],
                diagnostics["topology/lambda"],
                int(diagnostics["queue/size"]),
                diagnostics["queue/mean_updates"],
                diagnostics["memory/cuda_peak_mib"],
                diagnostics["time/iteration_seconds"],
            )
        checkpoint_interval = int(
            ablation["idm"]["checkpoint_interval_iterations"]
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
                calibrator,
                diagnostics,
            )
        preview_interval = int(
            ablation["idm"]["preview_interval_iterations"]
        )
        if (
            preview_interval > 0
            and iteration % preview_interval == 0
        ):
            preview = parameterization.render_all().detach().cpu()
            save_image(
                preview.clamp(0, 1),
                directory / f"preview_iteration_{iteration:06d}.png",
                nrow=bundle.num_classes,
            )
            del preview
            cleanup_memory()
    synthetic_path = _export(
        directory,
        experiment,
        parameterization,
        bundle.class_names,
        target,
    )
    summary = {
        "experiment": experiment,
        "condensation_seed": int(condensation_seed),
        "seed": int(seed),
        "parameterization": mode,
        "topology": include_topology,
        "iterations": int(target),
        "synthetic_dataset": str(synthetic_path.resolve()),
        "topology_calibration": calibrator.state_dict(),
        "diagnostics": diagnostics,
        "generator": {
            "autoencoder": (
                generators.autoencoder_checkpoint if generators else None
            ),
            "diffusion": (
                generators.diffusion_checkpoint if generators else None
            ),
        },
    }
    atomic_write_json(summary, directory / "summary.json")
    del parameterization, optimizer, queue, pool, generators
    cleanup_memory()
    return summary


def run(
    config: Mapping[str, Any],
    selected_ipcs: list[int] | None = None,
) -> dict[str, Any]:
    """主流水线入口：使用 condense.yaml 指定的默认 C 方法运行 IPC=1。"""

    ipcs = list(
        selected_ipcs
        or map(int, condensation_settings(config)["ipc_values"])
    )
    if ipcs != [1]:
        raise ValueError("标准 IDM condense 阶段当前只支持 IPC=1")
    method = str(condensation_settings(config).get("default_method", "C0"))
    return {
        "method": method,
        "ipc": 1,
        "result": run_condensation(config, method, condensation_seed=0),
    }
