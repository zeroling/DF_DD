"""各阶段共享的训练循环、平滑指标、余弦调度恢复和早停工具。

本模块只保存与具体模型无关的训练状态逻辑。每个阶段负责选择监控指标，例如
Autoencoder 监控验证 L1、Diffusion 监控验证去噪损失、Evaluation 监控真实验证集
Balanced Accuracy、Condensation 监控平滑后的总蒸馏损失。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from Core.augmentations import classifier_normalize
from Core.data import unpack_batch
from Core.run_context import autocast_context


def cycle(loader: Iterable) -> Iterator:
    """无限循环一个 DataLoader；如果加载器为空则立即给出明确错误。"""

    # 外层循环在完整遍历结束后重新开始下一轮。
    while True:
        # produced 用于区分“正常遍历结束”和“加载器从一开始就是空的”。
        produced = False
        # 逐批返回加载器内容。
        for batch in loader:
            produced = True
            yield batch
        # 空加载器无限重试会形成死循环，因此必须主动报错。
        if not produced:
            raise ValueError("不能循环空 DataLoader")


@dataclass
class ExponentialMetric:
    """保存一个可断点恢复的指数移动平均，用于平滑在线队列产生的噪声指标。"""

    # decay 越接近 1，历史值影响越长；必须位于 [0,1)。
    decay: float = 0.95
    # value=None 表示尚未收到第一个观测值。
    value: float | None = None

    def update(self, observation: float) -> float:
        """加入一个新观测并返回更新后的平滑值。"""

        # 把张量或 NumPy 标量统一转换成 Python float，便于序列化。
        observation = float(observation)
        # 第一个观测直接作为初值，避免从 0 开始产生长时间偏差。
        if self.value is None:
            self.value = observation
        else:
            # 标准指数移动平均递推公式。
            self.value = self.decay * self.value + (1.0 - self.decay) * observation
        # 调用方可以直接记录返回值。
        return float(self.value)

    def state_dict(self) -> dict[str, float | None]:
        """返回可直接放入 Torch 断点的纯 Python 状态。"""

        return {"decay": float(self.decay), "value": self.value}

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """恢复已有平滑值；旧断点没有该字段时保持当前初始状态。"""

        if not state:
            return
        # 当前配置中的 decay 是权威值，因此只恢复历史 value。
        value = state.get("value")
        self.value = None if value is None else float(value)


class EarlyStopping:
    """通用、可断点恢复的“连续若干次无改善后停止”状态机。"""

    def __init__(
        self,
        enabled: bool,
        mode: str,
        patience_checks: int,
        min_delta: float,
        check_interval: int,
        minimum_progress: int,
    ) -> None:
        """保存当前配置；``progress`` 可以是 epoch、iteration 或 member update。"""

        # enabled=false 时 update 永远返回 False，但状态接口保持可用。
        self.enabled = bool(enabled)
        # mode=min 监控损失，mode=max 监控准确率。
        self.mode = str(mode).lower()
        # patience_checks 是允许连续无改善检查次数。
        self.patience_checks = max(1, int(patience_checks))
        # min_delta 过滤数值噪声，只有超过该幅度才算真实改善。
        self.min_delta = max(0.0, float(min_delta))
        # check_interval 控制检查频率，避免每一步都更新耐心计数。
        self.check_interval = max(1, int(check_interval))
        # minimum_progress 保证训练至少运行到指定进度。
        self.minimum_progress = max(0, int(minimum_progress))
        # best 保存历史最佳指标；None 表示还没有执行过有效检查。
        self.best: float | None = None
        # bad_checks 记录连续无改善检查次数。
        self.bad_checks = 0
        # best_progress 保存最佳指标出现的位置，便于日志解释。
        self.best_progress = 0
        # stopped 标记上一次调用是否已经满足停止条件。
        self.stopped = False

    @classmethod
    def from_config(
        cls,
        settings: Mapping[str, Any],
        interval_key: str,
        minimum_key: str,
    ) -> "EarlyStopping":
        """从阶段配置创建早停器，并显式指定阶段使用的间隔字段名。"""

        return cls(
            enabled=bool(settings.get("enabled", False)),
            mode=str(settings.get("mode", "min")),
            patience_checks=int(settings.get("patience_checks", 10)),
            min_delta=float(settings.get("min_delta", 0.0)),
            check_interval=int(settings.get(interval_key, 1)),
            minimum_progress=int(settings.get(minimum_key, 0)),
        )

    def should_check(self, progress: int) -> bool:
        """判断当前进度是否达到最小训练量并落在检查间隔上。"""

        # 关闭早停时无需计算或记录额外验证逻辑。
        if not self.enabled:
            return False
        # 早于 minimum_progress 时不消耗耐心次数。
        if int(progress) < self.minimum_progress:
            return False
        # 只在配置间隔的整数倍检查。
        return int(progress) % self.check_interval == 0

    def _improved(self, value: float) -> bool:
        """按 min/max 模式判断新值是否相对历史最佳值有足够改善。"""

        # 第一个有效指标必然作为初始最佳值。
        if self.best is None:
            return True
        # 损失模式要求新值至少比旧值低 min_delta。
        if self.mode == "min":
            return value < self.best - self.min_delta
        # 准确率模式要求新值至少比旧值高 min_delta。
        return value > self.best + self.min_delta

    def update(self, value: float, progress: int) -> bool:
        """在检查点更新状态，返回 ``True`` 表示调用方现在应停止训练。"""

        # 非检查进度直接返回，不修改最佳值或耐心计数。
        if not self.should_check(progress):
            return False
        # 明确转换为 float，避免 checkpoint 保存设备张量。
        value = float(value)
        # 指标获得足够改善时更新最佳值并清空连续失败次数。
        if self._improved(value):
            self.best = value
            self.best_progress = int(progress)
            self.bad_checks = 0
            self.stopped = False
            return False
        # 没有改善时增加一次连续失败计数。
        self.bad_checks += 1
        # 达到耐心上限后标记停止。
        self.stopped = self.bad_checks >= self.patience_checks
        # 返回停止结论供阶段循环 break。
        return self.stopped

    def reset(self) -> None:
        """清空历史最佳值和耐心计数，用于配置要求续训时重新统计。"""

        self.best = None
        self.bad_checks = 0
        self.best_progress = 0
        self.stopped = False

    def state_dict(self) -> dict[str, Any]:
        """返回不含配置哈希的早停历史状态。"""

        return {
            "best": self.best,
            "bad_checks": int(self.bad_checks),
            "best_progress": int(self.best_progress),
            "stopped": bool(self.stopped),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None, reset: bool = False) -> None:
        """恢复早停历史；``reset=True`` 时按当前配置从头统计。"""

        # 用户显式要求重置或旧断点没有状态时不做恢复。
        if reset or not state:
            self.reset()
            return
        # 只恢复历史统计，enabled/mode/patience 始终使用当前配置。
        best = state.get("best")
        self.best = None if best is None else float(best)
        self.bad_checks = max(0, int(state.get("bad_checks", 0)))
        self.best_progress = max(0, int(state.get("best_progress", 0)))
        # 续训允许目标轮数被调大，因此不要让旧 stopped 标志阻止再次进入循环。
        self.stopped = False


def train_classifier_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    config: Mapping[str, Any],
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    """在一个完整真实数据 epoch 上训练分类器并返回平均损失与准确率。"""

    # 训练模式启用 BatchNorm 更新和 Dropout。
    model.train()
    # 标签平滑由阶段配置显式控制。
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    # 下面三个累计量用于样本加权平均，而不是简单平均不同大小的 batch。
    loss_sum = 0.0
    correct = 0
    count = 0
    # 遍历一个完整 DataLoader。
    for batch in loader:
        # 统一解包 folder、manifest 和 synthetic 数据集字典。
        images, labels = unpack_batch(batch)
        # 分类器输入需要使用全局 mean/std 归一化。
        images = classifier_normalize(images.to(device, non_blocking=True), config)
        # 分类标签统一成设备上的 long 张量。
        labels = labels.to(device, non_blocking=True).long()
        # set_to_none=True 减少无用显存写入。
        optimizer.zero_grad(set_to_none=True)
        # AMP 只在 CUDA 且全局启用时生效。
        with autocast_context(config, device):
            logits = model(images)
            loss = criterion(logits, labels)
        # GradScaler 在 BF16/CPU 时是禁用的统一接口，在 FP16 时负责防止下溢。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 按样本数累计损失和正确数。
        loss_sum += float(loss.detach().item()) * labels.numel()
        correct += int((logits.detach().argmax(1) == labels).sum().item())
        count += labels.numel()
    # 空数据集不应返回看似正常的 0 指标。
    if count == 0:
        raise ValueError("分类训练数据为空")
    # 返回普通 Python float，便于 JSON 和断点保存。
    return {"loss": loss_sum / count, "accuracy": correct / count}


def set_scheduler_epoch(scheduler, epoch: int) -> None:
    """无调用 ``scheduler.step`` 地把 LambdaLR 设置到指定 epoch。

    某些远程 PyTorch 版本无法识别 ``GradScaler.step(optimizer)`` 已经完成优化器
    更新，会在正常的 epoch 末调度时误报“scheduler.step before optimizer.step”。
    本项目的学习率计划是 LambdaLR 闭式函数，因此直接计算目标 epoch 的学习率既
    能保持曲线，又不会伪造 optimizer.step 或触发该警告。
    """

    # 负进度统一视为尚未训练。
    epoch = max(0, int(epoch))
    # 本项目所有余弦计划都由 LambdaLR 构造，因此可以直接计算当前学习率。
    if hasattr(scheduler, "lr_lambdas"):
        learning_rates = [
            base_lr * function(epoch)
            for base_lr, function in zip(scheduler.base_lrs, scheduler.lr_lambdas)
        ]
        # LambdaLR 使用 last_epoch 表示当前调度位置。
        scheduler.last_epoch = epoch
        # 同步内部计数，若外部未来读取/保存 scheduler.state_dict()，状态仍与 epoch 一致。
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = max(1, epoch + 1)
        # 把计算值写回每个优化器参数组。
        for group, learning_rate in zip(scheduler.optimizer.param_groups, learning_rates):
            group["lr"] = float(learning_rate)
        # 同步调度器公开的最近学习率缓存。
        scheduler._last_lr = [float(value) for value in learning_rates]
    else:
        # 若未来换成其他调度器，需要为其单独实现无副作用定位逻辑。
        raise TypeError("set_scheduler_epoch 当前只支持项目使用的 LambdaLR")


def advance_scheduler_to(scheduler, completed_epochs: int) -> None:
    """兼容旧调用名：按当前目标轮数把 LambdaLR 定位到断点 epoch。"""

    # 续训和正常 epoch 结束都走同一个无副作用的闭式学习率定位逻辑。
    set_scheduler_epoch(scheduler, completed_epochs)
