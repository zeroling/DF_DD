"""项目随机数和复现设置。"""

from __future__ import annotations

import random  # Python 自带随机源，部分数据采样和列表选择会使用。

import numpy as np  # 第三方数据处理可能使用 NumPy 随机源。
import torch  # 模型初始化、DataLoader 和 CUDA 随机源。


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """固定 Python、NumPy、Torch CPU/CUDA 随机数。"""

    # YAML/CLI 可能传入可转换字符串，先统一成 Python int。
    seed = int(seed)
    # 固定 Python 标准库随机状态。
    random.seed(seed)
    # 固定 NumPy 全局随机状态。
    np.random.seed(seed)
    # 固定当前进程的 Torch CPU 随机状态及模型初始化。
    torch.manual_seed(seed)
    # 所有当前可见 GPU 使用同一基础种子；无 CUDA 时不触发驱动初始化。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic=true 时让 cuDNN 优先选择确定性实现。
    torch.backends.cudnn.deterministic = bool(deterministic)
    # benchmark 会根据输入测速选择算法，可能引入不可复现性，因此与 deterministic 互斥。
    torch.backends.cudnn.benchmark = not bool(deterministic)
    # 新版 PyTorch 进一步约束其他算子的确定性；warn_only 避免个别无确定实现算子中断实验。
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
