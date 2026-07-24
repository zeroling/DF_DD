"""在线异构 IDM 训练轨迹队列。

队列中的分类器全部从随机初始化开始，只使用真实训练数据持续更新。成员在不同更新
年龄时参与合成图像指导，并按固定周期、最大更新数、目标准确率或准确率停滞条件被
同架构新模型替换。这样不需要预训练或保存数百个专家轨迹快照。

训练分类器和优化合成数据是两个严格分离的步骤：

1. ``advance`` 将部分成员设为训练模式，只用真实 batch 更新模型权重；
2. ``guidance_members`` 将选中成员设为评估模式并冻结参数，只保留对输入图像梯度；
3. 合成损失只更新像素或扩散隐变量，绝不会用合成图反向更新队列模型。
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
    """统一切换模型参数梯度，并同步 train/eval 模式。"""

    # 在线真实数据更新需要 train 模式；合成指导需要稳定的 eval 模式。
    model.train(bool(enabled))
    # 冻结参数时仍允许梯度穿过网络流向输入图像。
    for parameter in model.parameters():
        parameter.requires_grad_(bool(enabled))


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """把从 CPU 断点恢复的优化器张量移动到当前训练设备。"""

    # optimizer.state 的值包含动量、二阶矩和计步器等状态。
    for state in optimizer.state.values():
        # 只移动张量，普通整数和浮点数保持原样。
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@dataclass
class OnlineQueueMember:
    """一个在线分类器及其优化、训练年龄和准确率停滞状态。"""

    # identifier 是队列生命周期内单调递增的成员编号。
    identifier: int
    # architecture 是模型工厂使用的稳定架构名。
    architecture: str
    # model 是当前训练年龄对应的随机初始化分类器。
    model: torch.nn.Module
    # optimizer 只在真实数据更新步骤中使用。
    optimizer: torch.optim.Optimizer
    # birth_iteration 记录成员加入队列时的蒸馏迭代。
    birth_iteration: int = 0
    # update_steps 记录已经经历的真实 batch 参数更新次数。
    update_steps: int = 0
    # accuracy_ema 是真实训练 batch 准确率的指数移动平均。
    accuracy_ema: float | None = None
    # best_accuracy 保存停滞检查观察到的最佳 EMA 准确率。
    best_accuracy: float | None = None
    # plateau_checks 记录连续无明显准确率改善的检查次数。
    plateau_checks: int = 0
    # pending_warmup_steps 是新成员仍需执行的可选真实数据预热步数。
    pending_warmup_steps: int = 0

    def reliability(self, minimum: float) -> float:
        """返回合成交叉熵使用的有界可靠性权重。"""

        # 尚未训练的随机成员只使用最低可靠性。
        observed = 0.0 if self.accuracy_ema is None else float(self.accuracy_ema)
        # 限制到 [minimum,1]，避免异常统计放大分类损失。
        return min(1.0, max(float(minimum), observed))

    def age(self, current_iteration: int) -> int:
        """返回成员在队列中存活的蒸馏迭代数。"""

        return max(0, int(current_iteration) - int(self.birth_iteration))


class OnlineHeterogeneousQueue:
    """管理任意架构组合的在线 IDM 分类器轨迹与断点状态。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        settings: Mapping[str, Any],
        num_classes: int,
        device: torch.device,
        seed: int,
    ) -> None:
        """按当前配置创建每种架构指定数量的随机模型。"""

        # 完整项目配置用于创建网络、归一化和 AMP 上下文。
        self.config = config
        # 复制队列设置，防止运行时意外修改原始合并配置。
        self.settings = dict(settings)
        # 类别数决定分类头维度与类别均衡 batch 大小。
        self.num_classes = int(num_classes)
        # 所有活跃模型当前驻留在统一训练设备上。
        self.device = torch.device(device)
        # 独立 RNG 使成员选择不受其他模块 random 调用干扰。
        self.random = random.Random(int(seed))
        # 架构顺序来自配置，并用于公平轮转。
        self.architectures = [str(value) for value in settings["architectures"]]
        # 只保留当前活跃模型，不保留任何淘汰历史权重。
        self.members: list[OnlineQueueMember] = []
        # 新成员编号单调递增，便于日志追踪替换事件。
        self.next_identifier = 0
        # 全局训练轮转游标。
        self.training_cursor = 0
        # 每种架构分别维护指导成员轮转游标。
        self.guidance_cursors = {architecture: 0 for architecture in self.architectures}
        # FP16 使用共享 GradScaler，BF16/CPU 时保持统一空操作接口。
        self.scaler = make_grad_scaler(config, self.device)
        # 累计替换数量只用于日志诊断。
        self.replaced_total = 0
        # 按配置为每种架构创建指定数量的初始随机成员。
        for architecture in self.architectures:
            target_size = int(settings["size_per_architecture"][architecture])
            for _ in range(target_size):
                self.members.append(self._new_member(architecture, birth_iteration=0))

    def _policy(self, architecture: str) -> dict[str, Any]:
        """返回指定架构的在线优化器配置副本。"""

        return dict(self.settings.get("optimization", {}).get(str(architecture), {}))

    def _new_member(self, architecture: str, birth_iteration: int) -> OnlineQueueMember:
        """创建一个从随机初始化开始的分类器和专属优化器。"""

        # 与评估阶段共享同一个模型工厂和逐层结构定义。
        model = build_classifier_from_config(
            self.config,
            str(architecture),
            self.num_classes,
        ).to(self.device)
        # 支持的 ConvNeXt/ViT 可以按配置启用内部激活重计算。
        checkpoint_setter = getattr(model, "set_grad_checkpointing", None)
        if callable(checkpoint_setter):
            checkpoint_setter(bool(self.settings.get("gradient_checkpointing", False)))
        # 每个成员拥有独立优化器，训练年龄和动量互不污染。
        optimizer = build_optimizer(model, self._policy(str(architecture)))
        # 分配唯一编号后递增计数器。
        identifier = int(self.next_identifier)
        self.next_identifier += 1
        # 默认预热为 0，因此成员会真正经历纯随机指导阶段。
        member = OnlineQueueMember(
            identifier=identifier,
            architecture=str(architecture),
            model=model,
            optimizer=optimizer,
            birth_iteration=int(birth_iteration),
            pending_warmup_steps=max(0, int(self.settings.get("initial_warmup_steps", 0))),
        )
        # 新成员在真实训练前保持冻结，避免意外累计参数梯度。
        _set_trainable(member.model, False)
        return member

    def _members_for_architecture(self, architecture: str) -> list[OnlineQueueMember]:
        """按当前列表顺序返回指定架构的全部活跃成员。"""

        return [member for member in self.members if member.architecture == str(architecture)]

    def _select_training_members(self) -> list[OnlineQueueMember]:
        """根据 random/round_robin 选择本次真实数据更新成员。"""

        # 训练数量不能超过活跃成员总数。
        count = min(
            len(self.members),
            int(self.settings.get("models_trained_per_iteration", len(self.members))),
        )
        # 随机模式执行无放回抽样。
        if str(self.settings.get("selection", "round_robin")) == "random":
            return self.random.sample(self.members, count)
        # 轮转模式从 cursor 连续取成员，保证长期更新次数均衡。
        selected = [
            self.members[(self.training_cursor + offset) % len(self.members)]
            for offset in range(count)
        ]
        # 下次从本轮末尾后继续。
        self.training_cursor = (self.training_cursor + count) % len(self.members)
        return selected

    def _update_accuracy_state(self, member: OnlineQueueMember, batch_accuracy: float) -> None:
        """更新成员准确率 EMA 与可选停滞计数。"""

        # 当前配置决定 EMA 历史长度。
        decay = float(self.settings.get("accuracy_ema_decay", 0.9))
        # 第一次训练直接使用 batch 准确率初始化。
        if member.accuracy_ema is None:
            member.accuracy_ema = float(batch_accuracy)
        else:
            member.accuracy_ema = (
                decay * float(member.accuracy_ema)
                + (1.0 - decay) * float(batch_accuracy)
            )
        # 读取成员级准确率停滞配置。
        plateau = self.settings.get("replacement", {}).get("accuracy_plateau", {})
        # 关闭时无需维护 best/bad 状态。
        if not bool(plateau.get("enabled", False)):
            return
        # 训练步数太少时准确率没有统计意义。
        if member.update_steps < int(plateau.get("minimum_updates", 0)):
            return
        # 只在指定成员更新间隔检查一次。
        interval = max(1, int(plateau.get("check_interval_updates", 1)))
        if member.update_steps % interval != 0:
            return
        # 改善必须超过 min_delta 才会重置耐心计数。
        min_delta = float(plateau.get("min_delta", 0.0))
        if member.best_accuracy is None or float(member.accuracy_ema) > member.best_accuracy + min_delta:
            member.best_accuracy = float(member.accuracy_ema)
            member.plateau_checks = 0
        else:
            member.plateau_checks += 1

    def _train_member_steps(
        self,
        member: OnlineQueueMember,
        steps: int,
        pool: ClassImagePool,
        augment: TensorBatchAugment,
    ) -> dict[str, float]:
        """只使用真实类别均衡 batch 推进一个成员若干参数更新。"""

        # 训练期间允许参数求梯度并更新 BatchNorm/Dropout 状态。
        _set_trainable(member.model, True)
        # 大模型使用独立小 batch，避免用 ConvNet 的吞吐配置训练 ConvNeXt/ViT 时爆显存。
        architecture_batches = self.settings.get("batch_size_per_architecture", {})
        target_batch_size = int(
            architecture_batches.get(member.architecture, self.settings["batch_size"])
        )
        # batch_size 向上取整为每类相同数量，确保类别均衡。
        per_class = max(1, math.ceil(target_batch_size / self.num_classes))
        # 累计量用于计算本次多步训练的样本加权指标。
        loss_sum = 0.0
        correct_sum = 0
        sample_sum = 0
        # 执行指定真实 batch 更新次数。
        for _ in range(max(0, int(steps))):
            # 每类有放回抽取相同数量的真实图像。
            real_images, real_labels = pool.sample_all_classes(per_class)
            # 搬到设备、增强并执行分类器专属归一化。
            real_images = classifier_normalize(
                augment(real_images.to(self.device, non_blocking=True)),
                self.config,
            )
            # 分类标签统一为设备上的 long 张量。
            real_labels = real_labels.to(self.device, non_blocking=True).long()
            # 清除上一个真实 batch 的参数梯度。
            member.optimizer.zero_grad(set_to_none=True)
            # 使用项目统一 AMP 上下文。
            with autocast_context(self.config, self.device):
                logits = member.model(real_images)
                loss = F.cross_entropy(
                    logits,
                    real_labels,
                    label_smoothing=float(self.settings.get("label_smoothing", 0.0)),
                )
            # 这里只更新分类器参数，不接触合成图像或 z_T。
            self.scaler.scale(loss).backward()
            self.scaler.step(member.optimizer)
            self.scaler.update()
            # 当前 batch 预测用于可靠性与成员成熟度统计。
            correct = int((logits.detach().argmax(1) == real_labels).sum().item())
            batch_size = int(real_labels.numel())
            batch_accuracy = correct / max(1, batch_size)
            # 每完成一次 optimizer.step 就增加一个更新年龄。
            member.update_steps += 1
            self._update_accuracy_state(member, batch_accuracy)
            # 按样本数累计损失和正确数。
            loss_sum += float(loss.detach().item()) * batch_size
            correct_sum += correct
            sample_sum += batch_size
        # 指导合成数据前立即冻结参数并切换 eval。
        _set_trainable(member.model, False)
        return {
            "loss": loss_sum / max(1, sample_sum),
            "accuracy": correct_sum / max(1, sample_sum),
            "samples": float(sample_sum),
        }

    def _member_maturity_reason(self, member: OnlineQueueMember) -> str | None:
        """判断成员是否因上限、目标准确率或停滞而应被替换。"""

        # 读取统一替换配置。
        replacement = self.settings.get("replacement", {})
        # max_updates=0 表示关闭更新数上限。
        max_updates = int(replacement.get("max_updates", 0))
        if max_updates > 0 and member.update_steps >= max_updates:
            return "max_updates"
        # target_accuracy 位于 (0,1) 时启用；1.0 明确表示关闭该条件。
        target_accuracy = float(replacement.get("target_accuracy", 1.0))
        if (
            0.0 < target_accuracy < 1.0
            and member.accuracy_ema is not None
            and member.accuracy_ema >= target_accuracy
        ):
            return "target_accuracy"
        # 连续准确率无改善达到耐心次数时视为成熟。
        plateau = replacement.get("accuracy_plateau", {})
        if bool(plateau.get("enabled", False)) and member.plateau_checks >= int(
            plateau.get("patience_checks", 1)
        ):
            return "accuracy_plateau"
        return None

    def _replace_member(
        self,
        member: OnlineQueueMember,
        iteration: int,
        pool: ClassImagePool,
        augment: TensorBatchAugment,
    ) -> OnlineQueueMember:
        """在原列表位置用同架构随机新成员替换旧成员。"""

        # 保存原位置以维持 round_robin 顺序稳定。
        position = self.members.index(member)
        # 替换始终保持同架构，四种网络数量不会漂移。
        replacement = self._new_member(member.architecture, birth_iteration=int(iteration))
        # 可选预热只使用真实数据；默认 0 保留纯随机阶段。
        if replacement.pending_warmup_steps > 0:
            warmup_steps = int(replacement.pending_warmup_steps)
            replacement.pending_warmup_steps = 0
            self._train_member_steps(replacement, warmup_steps, pool, augment)
        # 在相同位置覆盖旧对象，旧模型随后可被垃圾回收。
        self.members[position] = replacement
        self.replaced_total += 1
        return replacement

    def _periodic_candidates(self, iteration: int, count: int) -> list[OnlineQueueMember]:
        """按 oldest/random 选择固定周期需要替换的不同成员。"""

        # 实际数量不能超过活跃成员总数。
        count = min(len(self.members), max(0, int(count)))
        # random 策略进行无放回抽样。
        if str(self.settings.get("replacement", {}).get("strategy", "oldest")) == "random":
            return self.random.sample(self.members, count)
        # oldest 优先选择存活更久、真实更新更多、编号更小的成员。
        ordered = sorted(
            self.members,
            key=lambda member: (
                member.age(iteration),
                member.update_steps,
                -member.identifier,
            ),
            reverse=True,
        )
        return ordered[:count]

    def advance(
        self,
        iteration: int,
        pool: ClassImagePool,
        augment: TensorBatchAugment,
    ) -> dict[str, float]:
        """执行固定替换、真实数据在线训练和成熟成员替换。"""

        # 保存本轮开始前替换计数，便于输出增量。
        replaced_before = int(self.replaced_total)
        # 固定周期在训练前推入随机成员，使其可以立即提供随机阶段指导。
        replacement = self.settings.get("replacement", {})
        interval = int(replacement.get("interval_iterations", 0))
        if int(iteration) > 1 and interval > 0 and int(iteration) % interval == 0:
            candidates = self._periodic_candidates(
                int(iteration),
                int(replacement.get("count_per_event", 1)),
            )
            # 候选在替换前固定，确保每个旧成员最多替换一次。
            for member in candidates:
                self._replace_member(member, int(iteration), pool, augment)

        # 根据 train_every_iterations 决定本轮是否推进轨迹。
        train_interval = max(1, int(self.settings.get("train_every_iterations", 1)))
        trained_members: list[OnlineQueueMember] = []
        train_loss_sum = 0.0
        train_accuracy_sum = 0.0
        if int(iteration) % train_interval == 0:
            trained_members = self._select_training_members()
            # 每个选中成员在独立真实 batch 上推进指定步数。
            for member in trained_members:
                metrics = self._train_member_steps(
                    member,
                    int(self.settings.get("train_steps_per_model", 1)),
                    pool,
                    augment,
                )
                train_loss_sum += float(metrics["loss"])
                train_accuracy_sum += float(metrics["accuracy"])

        # 在线更新后检查成熟条件；复制列表避免替换改变遍历对象。
        mature_members = [
            member
            for member in list(self.members)
            if self._member_maturity_reason(member) is not None
        ]
        # 每个成熟成员由同架构随机新模型替换。
        for member in mature_members:
            self._replace_member(member, int(iteration), pool, augment)

        # 汇总队列训练和替换诊断。
        diagnostics: dict[str, float] = {
            "queue/train_loss": train_loss_sum / max(1, len(trained_members)),
            "queue/train_accuracy": train_accuracy_sum / max(1, len(trained_members)),
            "queue/trained_members": float(len(trained_members)),
            "queue/replaced_members": float(self.replaced_total - replaced_before),
        }
        # 每种架构记录活跃规模、平均更新年龄和平均可靠性。
        minimum = float(self.settings.get("minimum_reliability", 0.05))
        for architecture in self.architectures:
            members = self._members_for_architecture(architecture)
            diagnostics[f"queue/{architecture}/size"] = float(len(members))
            diagnostics[f"queue/{architecture}/updates"] = sum(
                member.update_steps for member in members
            ) / max(1, len(members))
            diagnostics[f"queue/{architecture}/reliability"] = sum(
                member.reliability(minimum) for member in members
            ) / max(1, len(members))
        return diagnostics

    def guidance_members(self) -> list[OnlineQueueMember]:
        """从每种架构选指定数量成员，并确保参数冻结、模型处于 eval。"""

        # 每种架构独立选择，保证每次损失都有完整异构覆盖。
        selected: list[OnlineQueueMember] = []
        requested = int(self.settings.get("guidance_per_architecture", 1))
        for architecture in self.architectures:
            candidates = self._members_for_architecture(architecture)
            count = min(len(candidates), requested)
            # random 模式无放回抽取该架构成员。
            if str(self.settings.get("selection", "round_robin")) == "random":
                chosen = self.random.sample(candidates, count)
            else:
                # round_robin 使用每架构独立游标。
                cursor = self.guidance_cursors.get(architecture, 0) % len(candidates)
                chosen = [candidates[(cursor + offset) % len(candidates)] for offset in range(count)]
                self.guidance_cursors[architecture] = (cursor + count) % len(candidates)
            # 指导前保证不更新参数和 BatchNorm。
            for member in chosen:
                _set_trainable(member.model, False)
            selected.extend(chosen)
        return selected

    def summary(self) -> list[dict[str, Any]]:
        """返回适合日志和 JSON 的成员状态，不包含权重张量。"""

        # 日志可靠性与实际 CE 权重使用同一最低值。
        minimum = float(self.settings.get("minimum_reliability", 0.05))
        return [
            {
                "id": int(member.identifier),
                "architecture": member.architecture,
                "updates": int(member.update_steps),
                "accuracy_ema": member.accuracy_ema,
                "reliability": member.reliability(minimum),
                "plateau_checks": int(member.plateau_checks),
            }
            for member in self.members
        ]

    def state_dict(self) -> dict[str, Any]:
        """保存当前少量活跃成员，而不是保存完整历史轨迹。"""

        # 每个成员只保存继续在线轨迹所需的当前状态。
        serialized_members = []
        for member in self.members:
            serialized_members.append(
                {
                    "identifier": int(member.identifier),
                    "architecture": member.architecture,
                    "model": member.model.state_dict(),
                    "optimizer": member.optimizer.state_dict(),
                    "birth_iteration": int(member.birth_iteration),
                    "update_steps": int(member.update_steps),
                    "accuracy_ema": member.accuracy_ema,
                    "best_accuracy": member.best_accuracy,
                    "plateau_checks": int(member.plateau_checks),
                    "pending_warmup_steps": int(member.pending_warmup_steps),
                }
            )
        # 队列级状态还包括选择游标、独立 RNG 和 FP16 scaler。
        return {
            "version": 2,
            "next_identifier": int(self.next_identifier),
            "training_cursor": int(self.training_cursor),
            "guidance_cursors": dict(self.guidance_cursors),
            "random_state": self.random.getstate(),
            "scaler": self.scaler.state_dict(),
            "replaced_total": int(self.replaced_total),
            "members": serialized_members,
        }

    def _deserialize_member(self, state: Mapping[str, Any]) -> OnlineQueueMember:
        """按当前结构重建成员，再加载旧权重和优化器状态。"""

        # 断点架构必须仍存在于当前队列配置。
        architecture = str(state["architecture"])
        if architecture not in self.architectures:
            raise ValueError(f"断点队列架构 {architecture!r} 不在当前 online_queue.architectures")
        # 按当前 definitions 和优化器配置创建空成员。
        member = self._new_member(
            architecture,
            birth_iteration=int(state.get("birth_iteration", 0)),
        )
        # 恢复原唯一编号。
        member.identifier = int(state.get("identifier", member.identifier))
        # 网络形状不兼容时由 strict=True 直接报告。
        member.model.load_state_dict(state["model"], strict=True)
        # 恢复优化器动量与二阶矩。
        if isinstance(state.get("optimizer"), Mapping):
            member.optimizer.load_state_dict(state["optimizer"])
            _optimizer_state_to_device(member.optimizer, self.device)
        # 当前 YAML 学习率覆盖断点旧值。
        set_optimizer_learning_rate(
            member.optimizer,
            float(self._policy(architecture)["learning_rate"]),
        )
        # 恢复训练年龄和可靠性统计。
        member.update_steps = max(0, int(state.get("update_steps", 0)))
        accuracy = state.get("accuracy_ema")
        member.accuracy_ema = None if accuracy is None else float(accuracy)
        best = state.get("best_accuracy")
        member.best_accuracy = None if best is None else float(best)
        member.plateau_checks = max(0, int(state.get("plateau_checks", 0)))
        member.pending_warmup_steps = max(0, int(state.get("pending_warmup_steps", 0)))
        # 恢复后冻结，等待下一次 advance 或 guidance。
        _set_trainable(member.model, False)
        return member

    def _reconcile_sizes(self) -> None:
        """按当前 YAML 增删成员，使修改队列大小后仍能续训。"""

        # 每种架构独立调整数量并保持异构覆盖。
        reconciled: list[OnlineQueueMember] = []
        for architecture in self.architectures:
            target = int(self.settings["size_per_architecture"][architecture])
            # 优先保留编号较新的成员，它们更接近当前轨迹窗口。
            existing = sorted(
                self._members_for_architecture(architecture),
                key=lambda member: member.identifier,
                reverse=True,
            )[:target]
            # 数量增加时补充随机新成员。
            while len(existing) < target:
                existing.append(self._new_member(architecture, birth_iteration=0))
            # 恢复稳定的编号顺序。
            reconciled.extend(sorted(existing, key=lambda member: member.identifier))
        self.members = reconciled
        # 队列缩小时保证训练游标仍合法。
        self.training_cursor %= max(1, len(self.members))

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """恢复在线队列；旧静态专家断点没有该字段时保留新随机队列。"""

        # 无状态表示首次运行或从旧静态断点迁移。
        if not state or not isinstance(state.get("members"), Sequence):
            return
        # 先释放构造函数创建的临时随机成员，避免恢复时短暂同时驻留两套队列。
        self.members = []
        restored_members: list[OnlineQueueMember] = []
        for item in state["members"]:
            restored_members.append(self._deserialize_member(item))
        self.members = restored_members
        # 恢复唯一编号计数器，至少大于当前所有成员编号。
        self.next_identifier = max(
            int(state.get("next_identifier", 0)),
            max((member.identifier for member in self.members), default=-1) + 1,
        )
        # 恢复训练与指导轮转游标。
        self.training_cursor = max(0, int(state.get("training_cursor", 0)))
        saved_guidance = state.get("guidance_cursors", {})
        self.guidance_cursors = {
            architecture: max(0, int(saved_guidance.get(architecture, 0)))
            for architecture in self.architectures
        }
        # 恢复独立 RNG，确保替换序列连续。
        if state.get("random_state") is not None:
            self.random.setstate(state["random_state"])
        # 恢复 FP16 动态缩放状态。
        if isinstance(state.get("scaler"), Mapping):
            self.scaler.load_state_dict(state["scaler"])
        # 恢复累计替换次数。
        self.replaced_total = max(0, int(state.get("replaced_total", 0)))
        # 最后按当前 YAML 队列大小增删成员，不比较配置哈希。
        self._reconcile_sizes()
