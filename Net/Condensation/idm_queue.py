"""原始 IDM 风格的同构动态模型池。

模型池在一次实验中只包含一种所选架构，但每个成员都拥有独立随机初始化、优化器状态和训练
年龄。凝聚时从大池中随机抽取少量模型指导合成数据；完成一次合成数据更新后，只用
真实数据训练这些模型若干步。池中定期加入新的随机模型，超过容量后按 FIFO 淘汰。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from Core.augmentations import TensorBatchAugment, classifier_normalize
from Core.checkpoint import set_optimizer_learning_rate
from Core.data import ClassImagePool
from Core.run_context import autocast_context, make_grad_scaler
from Net.Classification.factory import build_classifier_from_config, build_optimizer


def _set_trainable(model: torch.nn.Module, enabled: bool) -> None:
    """切换分类器参数梯度和 train/eval 状态。"""

    model.train(bool(enabled))
    for parameter in model.parameters():
        parameter.requires_grad_(bool(enabled))


def _optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """把从 CPU 断点恢复的优化器张量移回训练设备。"""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@dataclass
class IDMModelMember:
    """一个随机初始化同构模型及其在线真实数据训练状态。"""

    identifier: int
    architecture: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    birth_iteration: int
    update_steps: int = 0
    accuracy_ema: float | None = None

    def reliability(self, minimum: float) -> float:
        """返回 IDM 分类损失使用的有界真实训练准确率。"""

        observed = 0.0 if self.accuracy_ema is None else float(self.accuracy_ema)
        return min(1.0, max(float(minimum), observed))


class IDMModelQueue:
    """管理单一可选架构的同构动态随机模型池。"""

    VERSION = 2

    def __init__(
        self,
        config: Mapping[str, Any],
        settings: Mapping[str, Any],
        num_classes: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.config = config
        self.settings = dict(settings)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.architecture = str(
            self.settings.get("architecture", "convnet")
        ).lower()
        self.random = random.Random(int(seed))
        self.members: list[IDMModelMember] = []
        self.next_identifier = 0
        self.generated_total = 0
        self.evicted_total = 0
        self.last_guidance_ids: list[int] = []
        self.scaler = make_grad_scaler(config, self.device)
        for _ in range(int(self.settings.get("initial_size", 3))):
            self.members.append(self._new_member(birth_iteration=0))

    def _optimization(self) -> dict[str, Any]:
        """返回当前所选架构的在线训练优化器配置。"""

        policies = self.settings.get("optimization", {})
        return dict(policies.get(self.architecture, {}))

    def _architecture_integer(self, key: str, default: int) -> int:
        """读取可按架构分别配置的整数值。"""

        value = self.settings.get(key, default)
        if isinstance(value, Mapping):
            value = value.get(self.architecture, default)
        return int(value)

    def _new_member(self, birth_iteration: int) -> IDMModelMember:
        """创建当前所选架构的独立随机初始化模型。"""

        model = build_classifier_from_config(
            self.config,
            self.architecture,
            self.num_classes,
        ).to(self.device)
        optimizer = build_optimizer(model, self._optimization())
        member = IDMModelMember(
            identifier=int(self.next_identifier),
            architecture=self.architecture,
            model=model,
            optimizer=optimizer,
            birth_iteration=int(birth_iteration),
        )
        self.next_identifier += 1
        self.generated_total += 1
        _set_trainable(member.model, False)
        return member

    def sample_guidance_members(self) -> list[IDMModelMember]:
        """从完整模型池中随机、无放回抽取本轮指导模型。"""

        count = min(
            len(self.members),
            int(self.settings.get("models_per_iteration", 2)),
        )
        if count <= 0:
            raise RuntimeError("IDM 模型池为空，无法产生凝聚梯度")
        selected = self.random.sample(self.members, count)
        for member in selected:
            _set_trainable(member.model, False)
        self.last_guidance_ids = [int(member.identifier) for member in selected]
        return selected

    def _update_accuracy(self, member: IDMModelMember, accuracy: float) -> None:
        """更新真实训练准确率 EMA，供 IDM 交叉熵可靠性加权。"""

        decay = float(self.settings.get("accuracy_ema_decay", 0.9))
        if member.accuracy_ema is None:
            member.accuracy_ema = float(accuracy)
        else:
            member.accuracy_ema = (
                decay * float(member.accuracy_ema)
                + (1.0 - decay) * float(accuracy)
            )

    def _train_member(
        self,
        member: IDMModelMember,
        pool: ClassImagePool,
        augment: TensorBatchAugment,
    ) -> dict[str, float]:
        """按 IDM 顺序，在合成数据更新后只用真实数据推进一个模型。"""

        _set_trainable(member.model, True)
        target_batch_size = self._architecture_integer("batch_size", 128)
        per_class = max(1, math.ceil(target_batch_size / self.num_classes))
        loss_sum = 0.0
        correct_sum = 0
        sample_sum = 0
        steps = int(self.settings.get("train_steps_per_model", 10))
        for _ in range(steps):
            real_images, real_labels = pool.sample_all_classes(per_class)
            real_images = classifier_normalize(
                augment(real_images.to(self.device, non_blocking=True)),
                self.config,
            )
            real_labels = real_labels.to(self.device, non_blocking=True).long()
            member.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.config, self.device):
                logits = member.model(real_images)
                loss = F.cross_entropy(
                    logits,
                    real_labels,
                    label_smoothing=float(self.settings.get("label_smoothing", 0.0)),
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(member.optimizer)
            self.scaler.update()

            correct = int((logits.detach().argmax(1) == real_labels).sum().item())
            count = int(real_labels.numel())
            member.update_steps += 1
            self._update_accuracy(member, correct / max(1, count))
            loss_sum += float(loss.detach().item()) * count
            correct_sum += correct
            sample_sum += count
        _set_trainable(member.model, False)
        return {
            "loss": loss_sum / max(1, sample_sum),
            "accuracy": correct_sum / max(1, sample_sum),
        }

    def _grow_and_evict(self, iteration: int) -> tuple[int, int]:
        """按固定间隔注入新随机种子，并在达到上限时 FIFO 淘汰。"""

        interval = int(self.settings.get("generate_interval_iterations", 30))
        if int(iteration) % interval != 0:
            return 0, 0
        maximum = self._architecture_integer("maximum_size", 100)
        evicted = 0
        if len(self.members) >= maximum:
            self.members.pop(0)
            self.evicted_total += 1
            evicted = 1
        self.members.append(self._new_member(birth_iteration=int(iteration)))
        return 1, evicted

    def advance(
        self,
        iteration: int,
        selected: Sequence[IDMModelMember],
        pool: ClassImagePool,
        augment: TensorBatchAugment,
    ) -> dict[str, float]:
        """训练本轮指导模型，然后维护动态池容量。"""

        losses: list[float] = []
        accuracies: list[float] = []
        for member in selected:
            metrics = self._train_member(member, pool, augment)
            losses.append(float(metrics["loss"]))
            accuracies.append(float(metrics["accuracy"]))
        generated, evicted = self._grow_and_evict(int(iteration))
        minimum = float(self.settings.get("minimum_reliability", 0.05))
        return {
            "queue/train_loss": sum(losses) / max(1, len(losses)),
            "queue/train_accuracy": sum(accuracies) / max(1, len(accuracies)),
            "queue/trained_models": float(len(selected)),
            "queue/size": float(len(self.members)),
            "queue/generated": float(generated),
            "queue/evicted": float(evicted),
            "queue/mean_updates": sum(member.update_steps for member in self.members)
            / max(1, len(self.members)),
            "queue/mean_reliability": sum(
                member.reliability(minimum) for member in self.members
            )
            / max(1, len(self.members)),
        }

    def summary(self) -> dict[str, Any]:
        """返回不会随模型池容量线性膨胀的紧凑日志摘要。"""

        minimum = float(self.settings.get("minimum_reliability", 0.05))
        return {
            "architecture": self.architecture,
            "size": len(self.members),
            "sampled": list(self.last_guidance_ids),
            "id_range": [
                min((member.identifier for member in self.members), default=-1),
                max((member.identifier for member in self.members), default=-1),
            ],
            "updates_mean": round(
                sum(member.update_steps for member in self.members)
                / max(1, len(self.members)),
                2,
            ),
            "reliability_mean": round(
                sum(member.reliability(minimum) for member in self.members)
                / max(1, len(self.members)),
                4,
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        """保存全部活跃模型，确保随机轨迹池可以精确续训。"""

        members = [
            {
                "identifier": int(member.identifier),
                "architecture": member.architecture,
                "model": member.model.state_dict(),
                "optimizer": member.optimizer.state_dict(),
                "birth_iteration": int(member.birth_iteration),
                "update_steps": int(member.update_steps),
                "accuracy_ema": member.accuracy_ema,
            }
            for member in self.members
        ]
        return {
            "version": self.VERSION,
            "architecture": self.architecture,
            "next_identifier": int(self.next_identifier),
            "generated_total": int(self.generated_total),
            "evicted_total": int(self.evicted_total),
            "last_guidance_ids": list(self.last_guidance_ids),
            "random_state": self.random.getstate(),
            "scaler": self.scaler.state_dict(),
            "members": members,
        }

    def _deserialize_member(self, state: Mapping[str, Any]) -> IDMModelMember:
        """按当前同构架构定义恢复一个池成员。"""

        saved_architecture = str(
            state.get("architecture", self.architecture)
        ).lower()
        if saved_architecture != self.architecture:
            raise ValueError(
                "IDM 模型池成员架构与当前配置不一致："
                f"断点={saved_architecture}，当前={self.architecture}"
            )
        member = self._new_member(
            birth_iteration=int(state.get("birth_iteration", 0))
        )
        member.identifier = int(state.get("identifier", member.identifier))
        member.model.load_state_dict(state["model"], strict=True)
        if isinstance(state.get("optimizer"), Mapping):
            member.optimizer.load_state_dict(state["optimizer"])
            _optimizer_state_to_device(member.optimizer, self.device)
        set_optimizer_learning_rate(
            member.optimizer,
            float(self._optimization().get("learning_rate", 0.01)),
        )
        member.update_steps = max(0, int(state.get("update_steps", 0)))
        accuracy = state.get("accuracy_ema")
        member.accuracy_ema = None if accuracy is None else float(accuracy)
        _set_trainable(member.model, False)
        return member

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """恢复同版本 IDM 池；异构旧断点必须由上层拒绝。"""

        if not state:
            return
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("IDM 模型池断点版本与当前代码不兼容")
        saved_architecture = str(
            state.get("architecture", self.architecture)
        ).lower()
        if saved_architecture != self.architecture:
            raise ValueError(
                "不能跨架构恢复 IDM 模型池："
                f"断点={saved_architecture}，当前={self.architecture}；请使用新的 run_dir"
            )
        serialized = state.get("members")
        if not isinstance(serialized, Sequence) or not serialized:
            raise ValueError("IDM 模型池断点没有有效成员")
        self.members = []
        restored = [self._deserialize_member(item) for item in serialized]
        maximum = self._architecture_integer("maximum_size", 100)
        self.members = restored[-maximum:]
        self.next_identifier = max(
            int(state.get("next_identifier", 0)),
            max(member.identifier for member in self.members) + 1,
        )
        self.generated_total = max(
            int(state.get("generated_total", len(self.members))),
            self.next_identifier,
        )
        self.evicted_total = max(0, int(state.get("evicted_total", 0)))
        self.last_guidance_ids = [
            int(value) for value in state.get("last_guidance_ids", [])
        ]
        if state.get("random_state") is not None:
            self.random.setstate(state["random_state"])
        if isinstance(state.get("scaler"), Mapping):
            self.scaler.load_state_dict(state["scaler"])
