"""异构网络之间的合成样本梯度平衡器。

ConvNet、ResNet-18、ConvNeXt-Tiny 和 ViT-Tiny 的损失尺度与输入梯度范数天然不同。
如果直接相加，某个大网络可能长期主导隐变量更新。本模块先在同架构内部平均，再依据
历史梯度范数的指数滑动平均进行有界逆尺度加权，最后恢复整体参考范数。
"""

from __future__ import annotations

from typing import Mapping, Sequence  # Sequence 表示同一架构可提供一个或多个成员梯度。

import torch  # 用于堆叠、范数、有限值检查和张量组合。


class ArchitectureGradientBalancer:
    """使用各架构输入梯度范数的 EMA 进行有界逆尺度加权。

    该对象不含可训练参数；``ema_norms`` 是唯一需要写入蒸馏断点的历史状态。
    """

    def __init__(
        self,
        enabled: bool = True,
        ema_decay: float = 0.9,
        minimum_scale: float = 0.2,
        maximum_scale: float = 5.0,
    ):
        # enabled=false 时仍会做同架构/跨架构平均，但所有缩放系数固定为 1。
        self.enabled = bool(enabled)
        # 越接近 1 越平滑，越不容易被某一次异常梯度改变长期尺度。
        self.ema_decay = float(ema_decay)
        # 逆尺度权重下界，防止强架构被完全抑制。
        self.minimum_scale = float(minimum_scale)
        # 逆尺度权重上界，防止弱/近零梯度被无限放大。
        self.maximum_scale = float(maximum_scale)
        # 键是架构名，值是该架构历史输入梯度 L2 范数的 EMA。
        self.ema_norms: dict[str, float] = {}

    @classmethod
    def from_config(cls, settings: Mapping) -> "ArchitectureGradientBalancer":
        """从 ``gradient_balance`` YAML 节点构造平衡器。"""

        # 使用显式默认值，使旧实验配置缺少新字段时仍能安全加载。
        return cls(
            settings.get("enabled", True),
            settings.get("ema_decay", 0.9),
            settings.get("minimum_scale", 0.2),
            settings.get("maximum_scale", 5.0),
        )

    def combine(
        self,
        gradients: Mapping[str, Sequence[torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """先平均同架构多个队列成员，再平衡不同架构并返回诊断值。

        参数 ``gradients`` 的键是架构名，值是该架构本轮所有指导成员对同一合成变量
        产生的梯度列表。返回值第一项可直接写入 ``z_t.grad`` 或像素 logits 的梯度；
        第二项包含原始范数和缩放系数，供训练日志排查某一架构是否长期占主导。
        """

        # 同一架构可能抽到多个不同训练年龄的队列成员，先等权平均成一个架构梯度。
        architecture_gradients = {
            name: torch.stack(list(values), dim=0).mean(dim=0)
            for name, values in gradients.items()
            if values
        }
        # 空映射通常意味着 guidance_per_architecture 或队列配置错误。
        if not architecture_gradients:
            raise ValueError("没有可合并的在线队列梯度")
        # 在范数与缩放前检查整张梯度，避免 NaN/Inf 扩散到隐变量并污染断点。
        invalid = [
            name
            for name, gradient in architecture_gradients.items()
            if not torch.isfinite(gradient).all()
        ]
        if invalid:
            raise FloatingPointError(f"以下架构产生 NaN/Inf 合成图像梯度：{invalid}")
        # current_norms 用于本轮日志；ema_norms 用于跨轮次稳定缩放。
        current_norms: dict[str, float] = {}
        for name, gradient in architecture_gradients.items():
            # detach 后用 float32 计算全张量 L2 范数，不向平衡器自身建立计算图。
            norm = float(gradient.detach().float().norm().item())
            # 新架构第一次出现时以当前范数初始化，避免 EMA 从 0 慢启动。
            previous = self.ema_norms.get(name, norm)
            # 标准指数滑动平均：decay×历史 + (1-decay)×当前。
            ema = self.ema_decay * previous + (1.0 - self.ema_decay) * norm
            # 设置极小正下界，后续计算 target/ema 时不会除零。
            self.ema_norms[name] = max(ema, 1.0e-12)
            current_norms[name] = norm
        # 禁用平衡或只有一个架构时，所有架构使用系数 1。
        scales = {name: 1.0 for name in architecture_gradients}
        # 两个及以上架构时，目标范数取各架构历史范数的算术平均。
        if self.enabled and len(architecture_gradients) > 1:
            target = sum(self.ema_norms[name] for name in architecture_gradients) / len(architecture_gradients)
            # 小梯度架构放大、大梯度架构缩小，并限制在 YAML 给定的安全范围内。
            scales = {
                name: min(self.maximum_scale, max(self.minimum_scale, target / self.ema_norms[name]))
                for name in architecture_gradients
            }
        # 对每个架构梯度乘平衡系数后求和。
        combined = sum(architecture_gradients[name] * scales[name] for name in architecture_gradients)
        # 除以架构数，确保增加队列架构不会线性放大优化器有效学习率。
        combined = combined / float(len(architecture_gradients))
        # 最后按平均架构范数归一，避免不同 IPC/网络抽样导致优化器有效步长剧烈变化。
        reference_norm = sum(current_norms.values()) / len(current_norms)
        # 组合梯度理论上可能接近相互抵消，clamp 防止恢复范数时除零。
        combined_norm = combined.detach().float().norm().clamp_min(1.0e-12)
        # 恢复到“各架构原始范数的平均值”，只调整贡献比例而不任意改变总步长。
        combined = combined * (reference_norm / float(combined_norm))
        # 诊断键加前缀，写入 JSONL/TensorBoard 后可按组查看。
        diagnostics = {
            **{f"norm/{name}": value for name, value in current_norms.items()},
            **{f"scale/{name}": value for name, value in scales.items()},
        }
        return combined, diagnostics

    def state_dict(self) -> dict:
        """返回可序列化的 EMA 状态；配置超参数始终以当前 YAML 为准。"""

        # 复制字典，避免断点写入期间外部持有内部可变引用。
        return {"ema_norms": dict(self.ema_norms)}

    def load_state_dict(self, state: Mapping | None) -> None:
        """从断点恢复历史范数；缺失状态时保留全新空 EMA。"""

        # 兼容旧断点没有 gradient_balancer 的情况，并校验内部值确实是映射。
        if state and isinstance(state.get("ema_norms"), Mapping):
            # 键和值转换成稳定的 Python 类型，避免 YAML/NumPy 标量混入运行状态。
            self.ema_norms = {str(key): float(value) for key, value in state["ema_norms"].items()}
