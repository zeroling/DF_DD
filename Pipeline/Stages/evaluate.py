"""只用合成训练集从零训练分类器，并在真实测试集做最终评估。

真实验证集只用于选择最佳 epoch 和“连续若干次 Balanced Accuracy 不提升后停止”，
真实测试集在训练结束并恢复最佳验证权重后只评估一次。在线队列中的分类器权重不会
进入本阶段，因此结果衡量的是合成数据对全新模型的真实训练价值。
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from Core.augmentations import TensorBatchAugment, classifier_normalize
from Core.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_training_state,
    set_optimizer_learning_rate,
    training_payload,
)
from Core.data import build_data_bundle, build_loader, unpack_batch
from Core.io_utils import atomic_write_json
from Core.logging_utils import get_stage_logger
from Core.metrics import evaluate_classifier
from Core.run_context import (
    autocast_context,
    make_grad_scaler,
    resolve_device,
    stage_checkpoint_directory,
)
from Core.seed import seed_everything
from Core.training import EarlyStopping, advance_scheduler_to
from Net.Classification.factory import build_classifier_from_config, build_training_policy
from Pipeline.ablation_config import condensation_settings, output_root


class SyntheticTensorDataset(Dataset):
    """读取统一 ``synthetic.pt`` 并提供与真实数据层一致的样本字典。"""

    def __init__(self, path: str | Path, expected_class_names: list[str]) -> None:
        """加载图像、标签并验证类别顺序和张量形状。"""

        # 保存绝对或相对源路径供错误信息和结果记录使用。
        self.path = Path(path)
        # 合成文件由本项目生成，允许读取普通 Python 元数据。
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        # 图像统一为连续 float32，并限制到数据层约定的 [0,1]。
        self.images = payload["images"].float().clamp(0.0, 1.0).contiguous()
        # 标签统一成一维 int64。
        self.labels = payload["labels"].long().view(-1).contiguous()
        # targets 供类别均衡 WeightedRandomSampler 读取。
        self.targets = self.labels.tolist()
        # 类别名称和顺序必须与当前真实数据完全一致。
        self.class_names = [str(value) for value in payload.get("class_names", [])]
        # 第一维图像数必须与标签数相同。
        if self.images.ndim != 4 or self.images.shape[0] != self.labels.shape[0]:
            raise ValueError(f"合成数据形状错误：{self.path}")
        # 防止更换数据集后误用旧合成集。
        if self.class_names != list(expected_class_names):
            raise ValueError(
                f"合成集类别顺序 {self.class_names} 与当前真实数据 "
                f"{expected_class_names} 不一致"
            )

    def __len__(self) -> int:
        """返回合成图像总数。"""

        return int(self.labels.numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回数据层统一的 image、label、key 字典。"""

        return {
            "image": self.images[index],
            "label": self.labels[index],
            "key": f"synthetic:{index}",
        }


def _snapshot_iteration(config: Mapping[str, Any]) -> int | None:
    """返回要评估的凝聚快照步数；``None`` 表示最终导出。"""

    value = config["evaluation"].get("snapshot_iteration")
    return None if value is None else int(value)


def _evaluation_variant_parts(config: Mapping[str, Any]) -> list[str]:
    """把不同凝聚步数的评估断点隔离，防止错误续用其他数据集的分类器。"""

    iteration = _snapshot_iteration(config)
    return [] if iteration is None else [f"iteration_{iteration:06d}"]


def _synthetic_path(config: Mapping[str, Any], source: str, ipc: int) -> Path:
    """定位统一 condense 阶段生成的默认方法合成集。"""

    # 当前评估来源固定为主流程 condensed。
    if str(source) != "condensed":
        raise ValueError(f"未知评估来源：{source}；当前只保留 condensed")
    if int(ipc) != 1:
        raise ValueError("统一后的标准 IDM evaluate 当前只支持 IPC=1")
    iteration = _snapshot_iteration(config)
    if iteration is not None:
        raise ValueError("新 condense 实现不再导出中间 synthetic snapshot")
    method = str(condensation_settings(config).get("default_method", "C0"))
    return (
        output_root(config)
        / "C"
        / method
        / "condense_seed_0"
        / "synthetic.pt"
    )


def _train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    config: Mapping[str, Any],
    augment: TensorBatchAugment,
    label_smoothing: float,
) -> dict[str, float]:
    """只在合成数据上训练一个 epoch，返回样本加权损失与准确率。"""

    # 训练模式启用 Dropout 和 BatchNorm 更新。
    model.train()
    # 标签平滑由 evaluation.yaml 控制。
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    # 累计量按实际 batch 样本数加权。
    loss_sum = 0.0
    correct = 0
    count = 0
    # 合成集通常很小，DataLoader 使用有放回均衡采样提供固定步数。
    for batch in loader:
        # 解包统一字典接口。
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        # 合成图先做可微增强，再按分类器 mean/std 归一化。
        images = classifier_normalize(augment(images), config)
        # 清空上一批参数梯度。
        optimizer.zero_grad(set_to_none=True)
        # 使用项目全局 AMP 设置。
        with autocast_context(config, device):
            logits = model(images)
            loss = criterion(logits, labels)
        # FP16 使用动态缩放，BF16/CPU 走统一空操作接口。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 累计样本加权损失和正确数。
        loss_sum += float(loss.detach().item()) * labels.numel()
        correct += int((logits.detach().argmax(1) == labels).sum().item())
        count += labels.numel()
    # 空合成集应明确报错。
    if count == 0:
        raise ValueError("合成训练集为空")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def _checkpoint_payload(
    model: torch.nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    source: str,
    ipc: int,
    architecture: str,
    repeat: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, Any] | None,
    early_stopping: EarlyStopping,
) -> dict[str, Any]:
    """构造评估训练的完整、无哈希断点。"""

    # 复用公共 training_payload 保存模型、优化器、调度器、scaler 和 RNG。
    return training_payload(
        model,
        "epoch",
        int(epoch),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        source=str(source),
        ipc=int(ipc),
        architecture=str(architecture),
        repeat=int(repeat),
        train_metrics=dict(train_metrics),
        validation_metrics=dict(validation_metrics or {}),
        early_stopping=early_stopping.state_dict(),
    )


def _run_one(
    config: Mapping[str, Any],
    source: str,
    ipc: int,
    architecture: str,
    repeat: int,
    bundle,
    validation_loader,
    test_loader,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    """完成一个来源×IPC×架构×重复实验。"""

    # 读取独立 evaluation.yaml 配置。
    settings = config["evaluation"]
    # 构造不同 IPC、重复和架构互不冲突且可复现的随机种子。
    architecture_index = list(settings["architectures"]).index(architecture)
    seed = (
        int(config["project"]["seed"])
        + int(ipc) * 10000
        + int(repeat) * 101
        + architecture_index
    )
    # 每个实验开始前固定全部随机源。
    seed_everything(seed, bool(config["project"].get("deterministic", True)))
    # 定位并读取当前来源合成集。
    dataset_path = _synthetic_path(config, source, ipc)
    synthetic_dataset = SyntheticTensorDataset(dataset_path, bundle.class_names)
    # 有放回采样使不同 IPC 每个 epoch 都执行可比较的固定优化步数。
    samples_per_epoch = int(settings["steps_per_epoch"]) * int(settings["batch_size"])
    train_loader = build_loader(
        synthetic_dataset,
        config,
        train=True,
        batch_size=int(settings["batch_size"]),
        balanced=True,
        samples_per_epoch=samples_per_epoch,
    )
    # 每个子实验拥有独立断点目录。
    directory = stage_checkpoint_directory(
        config,
        "evaluation",
        *_evaluation_variant_parts(config),
        source,
        f"ipc_{int(ipc)}",
        architecture,
        f"repeat_{int(repeat)}",
    )
    # 评估分类器必须从随机初始化开始，绝不读取在线队列权重。
    model = build_classifier_from_config(config, architecture, bundle.num_classes).to(device)
    # 不同 IPC 可以配置不同最大训练轮数。
    target_epochs = int(settings["epochs"][str(ipc)])
    # 只读取当前架构优化器参数；公共评估设置作为兼容兜底。
    optimization = {
        **dict(settings),
        **dict(settings.get("optimization", {}).get(architecture, {})),
    }
    # 构造优化器和可按当前目标轮数重建的余弦调度器。
    optimizer, scheduler = build_training_policy(
        architecture,
        model,
        optimization,
        target_epochs,
    )
    # 构造统一 AMP scaler。
    scaler = make_grad_scaler(config, device)
    # 从该子实验目录的最新权重恢复。
    completed, resume_path, resume_payload = restore_training_state(
        directory,
        model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        device=device,
    )
    # 当前 YAML 学习率覆盖断点旧值。
    set_optimizer_learning_rate(optimizer, float(optimization["learning_rate"]))
    # 调度器按新的目标轮数定位到已完成 epoch。
    advance_scheduler_to(scheduler, completed)
    if resume_path:
        logger.info(
            "恢复评估训练 source=%s IPC=%d %s repeat=%d：%d/%d",
            source,
            ipc,
            architecture,
            repeat,
            completed,
            target_epochs,
        )
    # 合成训练使用全局增强参数。
    augment = TensorBatchAugment.from_config(config["data"].get("augmentation", {}))
    # 早停检查间隔直接使用 validation_interval_epochs，避免两个参数含义重复。
    early_settings = {
        **dict(settings.get("early_stopping", {})),
        "check_interval_epochs": int(settings.get("validation_interval_epochs", 1)),
    }
    early_stopping = EarlyStopping.from_config(
        early_settings,
        interval_key="check_interval_epochs",
        minimum_key="minimum_epochs",
    )
    # 续训时恢复历史最佳验证准确率和耐心计数。
    early_stopping.load_state_dict(
        (resume_payload or {}).get("early_stopping"),
        reset=bool(early_settings.get("reset_on_resume", False)),
    )
    # 恢复已有训练/验证指标用于目标轮数已完成的情况。
    train_metrics: dict[str, float] = dict((resume_payload or {}).get("train_metrics", {}))
    validation_metrics: dict[str, Any] = dict(
        (resume_payload or {}).get("validation_metrics", {})
    )
    # 实际完成轮数可能因早停小于目标轮数。
    current_epoch = int(completed)
    # 最佳权重与最新权重分开保存。
    best_checkpoint_path = directory / "checkpoint_best.pt"

    # 从断点下一轮继续训练。
    for epoch in range(int(completed) + 1, target_epochs + 1):
        current_epoch = int(epoch)
        # 只在合成集上训练一个 epoch。
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            augment,
            float(settings.get("label_smoothing", 0.0)),
        )
        # 完成 epoch 后按闭式余弦函数设置学习率，避免远程 PyTorch 的顺序误报。
        advance_scheduler_to(scheduler, epoch)
        # 按配置间隔或最终轮在真实验证集评估。
        validation_interval = int(settings.get("validation_interval_epochs", 1))
        should_validate = epoch % validation_interval == 0 or epoch == target_epochs
        should_stop = False
        improved = False
        if should_validate:
            # 验证集只参与模型选择，不更新任何参数。
            validation_metrics = evaluate_classifier(
                model,
                validation_loader,
                device,
                bundle.num_classes,
                config=config,
            )
            # 比较 update 前后的 best，判断本轮是否产生新最佳权重。
            previous_best = early_stopping.best
            should_stop = early_stopping.update(
                float(validation_metrics["balanced_accuracy"]),
                epoch,
            )
            improved = early_stopping.best is not None and early_stopping.best != previous_best
            # 只有达到早停 minimum_epochs 后的新最佳才写 checkpoint_best。
            if improved:
                atomic_torch_save(
                    _checkpoint_payload(
                        model,
                        epoch,
                        optimizer,
                        scheduler,
                        scaler,
                        source,
                        ipc,
                        architecture,
                        repeat,
                        train_metrics,
                        validation_metrics,
                        early_stopping,
                    ),
                    best_checkpoint_path,
                )

        # 定期、最终或早停时保存最新断点。
        checkpoint_interval = int(settings.get("checkpoint_interval_epochs", 1))
        if epoch % checkpoint_interval == 0 or epoch == target_epochs or should_stop:
            atomic_torch_save(
                _checkpoint_payload(
                    model,
                    epoch,
                    optimizer,
                    scheduler,
                    scaler,
                    source,
                    ipc,
                    architecture,
                    repeat,
                    train_metrics,
                    validation_metrics,
                    early_stopping,
                ),
                directory / "checkpoint_last.pt",
            )
        # 按配置输出训练和验证摘要。
        log_interval = int(settings.get("log_interval_epochs", 1))
        if epoch == 1 or epoch % log_interval == 0 or should_stop:
            logger.info(
                "Eval source=%s IPC=%d %s repeat=%d epoch=%d/%d "
                "train_acc=%.4f val_bal_acc=%s best=%s",
                source,
                ipc,
                architecture,
                repeat,
                epoch,
                target_epochs,
                train_metrics["accuracy"],
                (
                    f"{float(validation_metrics['balanced_accuracy']):.4f}"
                    if validation_metrics
                    else "未检查"
                ),
                "无" if early_stopping.best is None else f"{early_stopping.best:.4f}",
            )
        # 连续验证无改善达到耐心值后停止该子实验。
        if should_stop:
            logger.info(
                "评估训练早停 source=%s IPC=%d %s repeat=%d：最佳 val_bal_acc=%.4f@%d",
                source,
                ipc,
                architecture,
                repeat,
                float(early_stopping.best or 0.0),
                early_stopping.best_progress,
            )
            break

    # 若 minimum_epochs 大于实际目标轮数，best 尚未生成，则把最终模型作为回退最佳。
    if not best_checkpoint_path.is_file():
        atomic_torch_save(
            _checkpoint_payload(
                model,
                current_epoch,
                optimizer,
                scheduler,
                scaler,
                source,
                ipc,
                architecture,
                repeat,
                train_metrics,
                validation_metrics,
                early_stopping,
            ),
            best_checkpoint_path,
        )
    # 最终测试前恢复最佳验证权重，不使用最后一个可能已经过拟合的 epoch。
    best_payload = load_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(model_state_from_checkpoint(best_payload), strict=True)
    # 真实测试集只在这里评估一次。
    test_metrics = evaluate_classifier(
        model,
        test_loader,
        device,
        bundle.num_classes,
        config=config,
    )
    # 记录完整单次结果。
    record = {
        "source": str(source),
        "ipc": int(ipc),
        "condensation_iteration": _snapshot_iteration(config),
        "architecture": str(architecture),
        "repeat": int(repeat),
        "seed": int(seed),
        "epochs": int(current_epoch),
        "stopped_early": bool(current_epoch < target_epochs),
        "best_validation_epoch": int(
            best_payload.get("epoch", early_stopping.best_progress or current_epoch)
        ),
        "best_validation_metrics": dict(best_payload.get("validation_metrics", {})),
        "synthetic_dataset": str(dataset_path.resolve()),
        "test_metrics": test_metrics,
    }
    # 每个子实验独立保存结果，便于中断后检查。
    atomic_write_json(record, directory / "result.json")
    return record


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按来源、IPC、架构聚合重复实验均值、标准差和原始值。"""

    # 四元单次记录按前三个实验维度分组，repeat 留在组内。
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (record["source"], int(record["ipc"]), record["architecture"]),
            [],
        ).append(record)
    # 输出使用嵌套 JSON 结构 source→ipc→architecture→metric。
    aggregate: dict[str, Any] = {}
    for (source, ipc, architecture), values in grouped.items():
        metrics: dict[str, Any] = {}
        # 三个主指标全部报告重复值、均值和样本标准差。
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            samples = [float(value["test_metrics"][metric_name]) for value in values]
            metrics[metric_name] = {
                "mean": statistics.fmean(samples),
                "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "values": samples,
            }
        aggregate.setdefault(source, {}).setdefault(str(ipc), {})[architecture] = metrics
    return aggregate


def run(
    config: Mapping[str, Any],
    selected_ipcs: list[int] | None = None,
) -> dict[str, Any]:
    """评估所有存在的来源、IPC、架构和随机重复。"""

    # 读取独立评估阶段配置。
    settings = config["evaluation"]
    # 最终结果沿用原目录；中间快照按迭代数隔离，避免覆盖汇总和训练断点。
    directory = stage_dir(config, "evaluation")
    for part in _evaluation_variant_parts(config):
        directory = directory / part
    directory.mkdir(parents=True, exist_ok=True)
    # 创建阶段日志器和运行设备。
    logger = get_stage_logger("evaluation", directory)
    device = resolve_device(config)
    # 构建统一真实 train/val/test 数据契约；这里只实际读取 val/test。
    bundle = build_data_bundle(config)
    # 真实验证集用于早停，测试集只用于最终指标。
    validation_loader = build_loader(bundle.val, config, train=False)
    test_loader = build_loader(bundle.test, config, train=False)
    # 命令行 --ipc 优先于主方法默认列表。
    ipcs = selected_ipcs or [
        int(value) for value in config["condensation"]["ipc_values"]
    ]
    # records 保存成功完成的单次实验，skipped 保存缺失可选来源。
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # 逐来源评估主流程输出；当前配置来源只有 condensed。
    for source in settings.get("sources", ["condensed"]):
        if source != "condensed":
            raise ValueError(f"未知评估来源：{source}；当前只保留 condensed")
        # 每个 IPC 独立查找合成集。
        for ipc in ipcs:
            dataset_path = _synthetic_path(config, str(source), int(ipc))
            if not dataset_path.is_file():
                skipped.append(
                    {
                        "source": str(source),
                        "ipc": int(ipc),
                        "reason": f"缺少 {dataset_path}",
                    }
                )
                logger.warning(
                    "跳过 source=%s IPC=%d：缺少合成数据 %s",
                    source,
                    ipc,
                    dataset_path,
                )
                continue
            # 对每种架构进行多随机种子重复。
            for architecture in settings["architectures"]:
                for repeat in range(int(settings["repeats"])):
                    records.append(
                        _run_one(
                            config,
                            str(source),
                            int(ipc),
                            str(architecture),
                            int(repeat),
                            bundle,
                            validation_loader,
                            test_loader,
                            device,
                            logger,
                        )
                    )
    # 汇总原始记录、统计量和跳过原因。
    summary = {
        "class_names": bundle.class_names,
        "condensation_iteration": _snapshot_iteration(config),
        "records": records,
        "aggregate": _aggregate(records),
        "skipped": skipped,
    }
    # 写入人类可读 JSON，不包含权重张量。
    atomic_write_json(summary, directory / "summary.json")
    return summary


if __name__ == "__main__":
    # 支持直接调试，正式运行推荐根目录 run_pipeline.py。
    from Core.config import load_config

    run(load_config())
