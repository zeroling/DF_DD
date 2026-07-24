"""扩散 U-Net 的指数滑动平均（EMA）权重。

训练模型参数负责接受优化器更新，EMA 影子权重负责采样、预览和最终蒸馏。EMA 不参与
反向传播，只平滑不同训练步的权重，通常能减少生成质量波动。其状态随扩散断点保存。
"""

from __future__ import annotations

from typing import Mapping  # 允许从普通字典或断点映射恢复状态。

import torch  # state_dict 张量复制、线性插值和 no_grad。


class ExponentialMovingAverage:
    """仅平均浮点参数/缓冲，状态可直接写入普通训练断点。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        # decay 越接近 1，EMA 记忆越长；配置文件负责限制到 [0,1)。
        self.decay = float(decay)
        # 初始影子权重是模型当前完整 state_dict 的独立副本，包括参数和 buffer。
        self.shadow = {
            # detach 去掉计算图，clone 防止与训练模型共享底层存储。
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """在每次优化器更新后，用当前模型原地更新影子权重。"""

        # 当前 state_dict 与初始化时的键必须一致；结构变化由键访问直接报错。
        current = model.state_dict()
        floating_groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        integer_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, shadow_value in self.shadow.items():
            value = current[name].detach()
            if torch.is_floating_point(shadow_value):
                key = (shadow_value.device, shadow_value.dtype)
                shadows, values = floating_groups.setdefault(key, ([], []))
                shadows.append(shadow_value)
                values.append(value)
            else:
                integer_pairs.append((shadow_value, value))

        # foreach 将逐参数的小 CUDA kernel 合并为少量批量 kernel。
        update_weight = 1.0 - self.decay
        for shadows, values in floating_groups.values():
            torch._foreach_lerp_(shadows, values, update_weight)
        for shadow_value, value in integer_pairs:
            shadow_value.copy_(value)

    def copy_to(self, model: torch.nn.Module) -> None:
        """把 EMA 权重严格复制到一个同结构模型中。"""

        # strict=True 防止模型结构与断点 EMA 不一致时静默漏载。
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        """返回可直接嵌入扩散训练断点的状态。"""

        # torch.save 会序列化张量内容；无需附加配置或哈希。
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping) -> None:
        """从断点恢复 EMA；当前 YAML 仍可在调用后覆盖 decay。"""

        # 旧断点若未保存 decay，则保留构造时配置值。
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow")
        # 缺少影子权重不能构成有效 EMA，明确报错。
        if not isinstance(shadow, Mapping):
            raise TypeError("EMA 断点缺少 shadow 权重")
        # 创建独立张量副本，避免断点 payload 生命周期或外部修改影响内部状态。
        self.shadow = {str(name): value.detach().clone() for name, value in shadow.items()}
