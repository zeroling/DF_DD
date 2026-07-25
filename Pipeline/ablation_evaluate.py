"""D/C 组统一可恢复评估，主架构使用官方 IDM ConvNet-6。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_rng_state,
)
from Core.data import build_loader, unpack_batch
from Core.io_utils import atomic_write_json, read_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, make_grad_scaler, resolve_device
from Core.seed import seed_everything
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    output_root,
)
from Pipeline.ablation_data import (
    NpyImageDataset,
    deterministic_bundle,
    diagnostic_cache_directory,
    load_c_synthetic,
    random_real_ipc1_dataset,
)
from Net.Condensation.idm_official import (
    ParamDiffAug,
    build_idm_convnet6,
    diff_augment,
    partition_and_expand,
)
from Core.experiment_runtime import (
    cleanup_memory,
    cuda_peak_megabytes,
)
from Net.Classification.factory import build_classifier_from_config, build_optimizer


def build_evaluation_model(
    config: Mapping[str, Any],
    architecture: str,
    num_classes: int,
) -> torch.nn.Module:
    if str(architecture) == "idm_convnet6":
        return build_idm_convnet6(
            int(config["data"]["image"]["channels"]),
            int(num_classes),
            config["data"]["image"]["size"],
        )
    return build_classifier_from_config(
        config, str(architecture), int(num_classes)
    )


def _uses_amp(config: Mapping[str, Any], architecture: str) -> bool:
    # 论文主结果保持官方 float32；附加大架构使用项目 BF16/FP16 降低显存。
    return str(architecture) != "idm_convnet6" and bool(
        config["project"].get("amp", True)
    )


def _amp_context(
    config: Mapping[str, Any],
    architecture: str,
    device: torch.device,
):
    if not _uses_amp(config, architecture):
        return torch.autocast(device_type=device.type, enabled=False)
    return autocast_context(config, device)


def _preflight_batch(
    config: Mapping[str, Any],
    architecture: str,
    num_classes: int,
    requested: int,
    device: torch.device,
    training: bool,
) -> tuple[int, float]:
    """用临时模型寻找当前空闲显存可承受的最大配置档。"""

    if device.type != "cuda":
        return int(requested), 0.0
    experiment = condensation_settings(config)
    minimum = max(
        1, int(experiment["memory"].get("retry_minimum", 1))
    )
    reserved_limit_mib = (
        float(torch.cuda.get_device_properties(device).total_memory)
        * float(experiment["memory"]["max_reserved_fraction"])
        / (1024**2)
    )
    batch = max(minimum, int(requested))
    height, width = map(int, config["data"]["image"]["size"])
    channels = int(config["data"]["image"]["channels"])
    while True:
        cleanup_memory()
        model = build_evaluation_model(config, architecture, num_classes).to(device)
        optimizer = None
        scaler = None
        try:
            images = torch.rand(
                batch, channels, height, width, device=device
            )
            labels = torch.arange(batch, device=device) % int(num_classes)
            model.train(training)
            if training:
                images = diff_augment(
                    images,
                    str(experiment["idm"]["dsa_strategy"]),
                    seed=0,
                    param=ParamDiffAug(),
                )
                optimizer = build_optimizer(
                    model,
                    _optimization(
                        config, architecture, diagnostic=False
                    ),
                )
                scaler = make_grad_scaler(config, device)
                optimizer.zero_grad(set_to_none=True)
            with _amp_context(config, architecture, device):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            peak = cuda_peak_megabytes()
            del images, labels, logits, loss, optimizer, scaler, model
            cleanup_memory()
            if peak > reserved_limit_mib and batch > minimum:
                batch = max(minimum, batch // 2)
                continue
            return batch, peak
        except torch.OutOfMemoryError:
            del optimizer, scaler, model
            cleanup_memory()
            if batch <= minimum:
                raise
            batch = max(minimum, batch // 2)


def _loader(
    dataset,
    config: Mapping[str, Any],
    batch_size: int,
    train: bool,
) -> DataLoader:
    return build_loader(
        dataset,
        config,
        train=train,
        batch_size=int(batch_size),
        balanced=False,
    )


def _optimization(
    config: Mapping[str, Any],
    architecture: str,
    diagnostic: bool,
) -> dict[str, Any]:
    configured = dict(
        condensation_settings(config)["evaluation"]["optimization"][
            str(architecture)
        ]
    )
    if diagnostic and architecture == "idm_convnet6":
        # D0-D3 仍用相同官方 SGD，确保变换前后差异不来自优化器。
        return configured
    return configured


def _set_epoch_learning_rate(
    optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any],
    epoch: int,
    epochs: int,
) -> None:
    initial = float(settings["learning_rate"])
    schedule = str(settings.get("schedule", "cosine"))
    if schedule == "official_step":
        value = initial * (0.1 if int(epoch) > int(epochs) // 2 else 1.0)
    elif schedule == "cosine":
        minimum = float(settings.get("min_learning_rate", 1.0e-6))
        progress = min(1.0, max(0.0, int(epoch) / max(1, int(epochs))))
        value = minimum + 0.5 * (initial - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        raise ValueError(f"未知学习率计划：{schedule}")
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: Mapping[str, Any],
    architecture: str,
    device: torch.device,
    strategy: str,
    seed_base: int,
) -> dict[str, float]:
    model.train()
    count = 0
    correct = 0
    total_loss = 0.0
    param = ParamDiffAug()
    for batch_index, batch in enumerate(loader):
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        images = diff_augment(
            images,
            strategy,
            seed=int(seed_base) + int(batch_index),
            param=param,
        )
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(config, architecture, device):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        amount = int(labels.numel())
        total_loss += float(loss.detach().item()) * amount
        correct += int((logits.detach().argmax(1) == labels).sum().item())
        count += amount
    if count == 0:
        raise ValueError("分类训练集为空")
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Mapping[str, Any],
    architecture: str,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    labels_all: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    count = 0
    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with _amp_context(config, architecture, device):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        amount = int(labels.numel())
        total_loss += float(loss.item()) * amount
        count += amount
        labels_all.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    class_ids = list(range(int(num_classes)))
    matrix = confusion_matrix(labels_all, predictions, labels=class_ids)
    recalls, precisions, specificities = [], [], []
    for class_id in class_ids:
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum()) - tp
        fn = int(matrix[class_id, :].sum()) - tp
        tn = int(matrix.sum()) - tp - fp - fn
        recalls.append(tp / max(1, tp + fn))
        precisions.append(tp / max(1, tp + fp))
        specificities.append(tn / max(1, tn + fp))
    return {
        "loss": total_loss / max(1, count),
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_specificity": specificities,
        "confusion_matrix": matrix.tolist(),
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    train_metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "train_metrics": dict(train_metrics),
        "rng_state": capture_rng_state(),
    }


def _train_and_test(
    config: Mapping[str, Any],
    train_dataset,
    architecture: str,
    repeat: int,
    epochs: int,
    directory: Path,
    seed: int,
    diagnostic: bool,
) -> dict[str, Any]:
    result_path = directory / "result.json"
    existing = read_json(result_path)
    if existing:
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    logger = get_stage_logger(
        f"idm_eval_{directory.as_posix().replace('/', '_')}",
        directory,
    )
    experiment_settings = condensation_settings(config)
    schedule_settings = (
        ablation_settings(config)["diagnostic"]
        if diagnostic
        else experiment_settings["evaluation"]
    )
    bundle = deterministic_bundle(config)
    device = resolve_device(config)
    seed_everything(
        int(seed), bool(config["project"].get("deterministic", False))
    )
    requested_train = int(
        experiment_settings["memory"]["evaluation_batch"][architecture]
    )
    train_batch, train_peak = _preflight_batch(
        config,
        architecture,
        bundle.num_classes,
        requested_train,
        device,
        training=True,
    )
    requested_test = int(
        experiment_settings["memory"]["inference_batch"][architecture]
    )
    test_batch, test_peak = _preflight_batch(
        config,
        architecture,
        bundle.num_classes,
        requested_test,
        device,
        training=False,
    )
    model = build_evaluation_model(
        config, architecture, bundle.num_classes
    ).to(device)
    settings = _optimization(config, architecture, diagnostic)
    optimizer = build_optimizer(model, settings)
    scaler = make_grad_scaler(config, device)
    checkpoint_path = find_latest_checkpoint(directory)
    completed = 0
    train_metrics: dict[str, float] = {}
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(model_state_from_checkpoint(payload), strict=True)
        if isinstance(payload.get("optimizer"), Mapping):
            optimizer.load_state_dict(payload["optimizer"])
        if isinstance(payload.get("scaler"), Mapping):
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload.get("rng_state"))
        completed = int(payload.get("epoch", 0))
        train_metrics = dict(payload.get("train_metrics", {}))
    if bool(config.get("_smoke", False)):
        train_dataset = Subset(
            train_dataset,
            range(min(len(train_dataset), int(config.get("_smoke_samples", 64)))),
        )
        test_dataset = Subset(
            bundle.test,
            range(min(len(bundle.test), int(config.get("_smoke_samples", 64)))),
        )
    else:
        test_dataset = bundle.test
    train_loader = _loader(train_dataset, config, train_batch, train=True)
    test_loader = _loader(test_dataset, config, test_batch, train=False)
    strategy = str(experiment_settings["idm"]["dsa_strategy"])
    for epoch in range(completed + 1, int(epochs) + 1):
        _set_epoch_learning_rate(optimizer, settings, epoch - 1, int(epochs))
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            architecture,
            device,
            strategy,
            int(seed) + epoch * 100000,
        )
        interval = int(schedule_settings["checkpoint_interval_epochs"])
        if epoch % interval == 0 or epoch == int(epochs):
            atomic_torch_save(
                _checkpoint_payload(
                    model, optimizer, scaler, epoch, train_metrics
                ),
                directory / "checkpoint_last.pt",
            )
        log_interval = int(schedule_settings["log_interval_epochs"])
        if epoch == 1 or epoch % log_interval == 0 or epoch == int(epochs):
            logger.info(
                "architecture=%s repeat=%d epoch=%d/%d loss=%.6f acc=%.4f "
                "train_batch=%d peak=%.0fMiB",
                architecture,
                repeat,
                epoch,
                int(epochs),
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_batch,
                train_peak,
            )
    test_metrics = _evaluate(
        model,
        test_loader,
        config,
        architecture,
        device,
        bundle.num_classes,
    )
    result = {
        "architecture": str(architecture),
        "repeat": int(repeat),
        "seed": int(seed),
        "epochs": int(epochs),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "memory": {
            "train_batch": int(train_batch),
            "test_batch": int(test_batch),
            "train_preflight_peak_mib": float(train_peak),
            "test_preflight_peak_mib": float(test_peak),
        },
    }
    atomic_write_json(result, result_path)
    del model, optimizer, scaler, train_loader, test_loader
    cleanup_memory()
    return result


def run_d_evaluation(
    config: Mapping[str, Any],
    experiment: str,
    architecture: str,
    repeat: int,
) -> dict[str, Any]:
    experiment = str(experiment)
    settings = ablation_settings(config)
    evaluation = condensation_settings(config)["evaluation"]
    if experiment == "D0":
        train_dataset = deterministic_bundle(config).train
        epochs = int(settings["diagnostic"]["full_data_epochs"])
    elif experiment in {"D1", "D2"}:
        train_dataset = NpyImageDataset(
            diagnostic_cache_directory(config, experiment)
        )
        epochs = int(settings["diagnostic"]["full_data_epochs"])
    elif experiment == "D3":
        train_dataset = random_real_ipc1_dataset(config, repeat)
        epochs = int(settings["diagnostic"]["random_ipc_epochs"])
    else:
        raise ValueError(f"未知 D 组实验：{experiment}")
    seed = (
        int(config["project"]["seed"])
        + int(experiment[1:]) * 10000
        + int(repeat) * 101
        + list(evaluation["architectures"]).index(architecture)
    )
    directory = (
        output_root(config)
        / "D"
        / experiment
        / "evaluation"
        / architecture
        / f"repeat_{int(repeat)}"
    )
    return _train_and_test(
        config,
        train_dataset,
        architecture,
        repeat,
        epochs,
        directory,
        seed,
        diagnostic=True,
    )


def run_c_evaluation(
    config: Mapping[str, Any],
    experiment: str,
    condensation_seed: int,
    architecture: str,
    repeat: int,
) -> dict[str, Any]:
    settings = condensation_settings(config)
    root = (
        output_root(config)
        / "C"
        / str(experiment)
        / f"condense_seed_{int(condensation_seed)}"
    )
    base_dataset = load_c_synthetic(root / "synthetic.pt")
    expanded_images, expanded_labels = partition_and_expand(
        base_dataset.images,
        base_dataset.labels,
        int(settings["idm"]["partition_expansion"]),
    )
    train_dataset = type(base_dataset)(expanded_images, expanded_labels)
    epochs = int(settings["evaluation"]["epochs"][architecture])
    seed = (
        int(config["project"]["seed"])
        + int(experiment[1:]) * 100000
        + int(condensation_seed) * 10000
        + int(repeat) * 101
        + list(settings["evaluation"]["architectures"]).index(
            architecture
        )
    )
    directory = (
        root
        / "evaluation"
        / architecture
        / f"repeat_{int(repeat)}"
    )
    result = _train_and_test(
        config,
        train_dataset,
        architecture,
        repeat,
        epochs,
        directory,
        seed,
        diagnostic=False,
    )
    result.update(
        {
            "experiment": str(experiment),
            "condensation_seed": int(condensation_seed),
            "synthetic_dataset": str((root / "synthetic.pt").resolve()),
        }
    )
    atomic_write_json(result, directory / "result.json")
    return result
