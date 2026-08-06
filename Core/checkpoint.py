"""无哈希断点保存与恢复工具。

恢复规则只有两条：

1. 指定目录里存在可读取的 ``.pt``/``.pth`` 权重；
2. 权重张量与当前网络结构兼容。

配置、代码和数据集都不会生成或比较哈希。断点中的优化器、调度器和随机数状态
若存在就恢复，缺失时则从当前配置重新创建；这也兼容只保存 ``state_dict`` 的权重。
"""

from __future__ import annotations

import os  # 同目录原子替换临时断点。
import time  # 捕获异常后休眠等待，应对 Windows 文件锁定。
import random  # 捕获和恢复 Python 随机状态。
import re  # 从旧式权重文件名中提取 epoch/step 进度。
import warnings  # 附加 RNG 状态不兼容时警告后继续恢复主权重。
from pathlib import Path  # 平台无关的断点目录与文件操作。
from typing import Any, Mapping  # 兼容完整断点和裸 state_dict 映射。

import numpy as np  # 捕获和恢复 NumPy 随机状态。
import torch  # 模型/优化器序列化及 CPU/CUDA 随机状态。


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """先写同目录临时文件再原子替换，避免中断后留下半个断点。"""

    # 字符串路径统一转换为 Path。
    target = Path(path)
    # 首次运行阶段时自动创建断点目录。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件位于同一目录，确保 os.replace 在同一文件系统上保持原子性。
    temporary = target.with_suffix(target.suffix + ".tmp")
    # 完整序列化到临时文件；若此处中断，旧的正式断点仍然完好。
    torch.save(payload, temporary)

    # 写完后一次替换正式文件，Windows 和 Linux 均支持覆盖已有目标。
    # 增加重试机制，专治 Windows 下杀毒软件/同步盘扫描导致的 WinError 5 文件锁定问题
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.replace(temporary, target)
            break  # 成功则跳出循环
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 每次失败后等待的时间逐渐增加：0.5s, 1.0s, 1.5s...
                sleep_time = 0.5 * (attempt + 1)
                warnings.warn(
                    f"保存文件时被系统锁定，{sleep_time} 秒后尝试第 {attempt + 2} 次覆盖: {target.name}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time.sleep(sleep_time)
            else:
                raise RuntimeError(
                    f"WinError 5: 在 {max_retries} 次尝试后仍无法覆盖 {target}。\n"
                    "请检查文件是否被 Windows Defender 或云盘（如 OneDrive）锁定。"
                ) from e

    return target


def capture_rng_state() -> dict:
    """保存 Python、NumPy、Torch CPU 和全部可见 CUDA 的随机状态。"""

    # 状态只描述随机序列位置，不包含配置或数据哈希。
    return {
        # Python random 可能用于分层拆分或外部适配器。
        "python": random.getstate(),
        # NumPy 随机源可能用于数据解码/增强插件。
        "numpy": np.random.get_state(),
        # Torch CPU 控制模型初始化、DataLoader 和独立张量采样。
        "torch": torch.get_rng_state(),
        # 多 GPU 时保存每一张当前可见卡的随机状态；无 CUDA 用 None。
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _cpu_byte_rng_state(value: Any, source: str) -> torch.Tensor:
    """把当前版或旧版断点的 RNG 状态统一为 CPU ``uint8`` 张量。

    ``torch.load(..., map_location="cuda")`` 会把断点中的所有张量都搬到
    GPU，其中也包括原本属于 CPU 随机数生成器的状态。但
    ``torch.set_rng_state`` 和 ``torch.cuda.set_rng_state`` 都要求传入 CPU
    ``ByteTensor``。此函数同时兼容 Tensor、NumPy 数组和旧版整数列表。
    """

    # 当前断点正常保存 Tensor；detach 防止异常旧对象带有计算图。
    if torch.is_tensor(value):
        normalized = value.detach().to(device="cpu", dtype=torch.uint8)
    else:
        # torch.as_tensor 能够将 NumPy 数组或 Python 整数列表升级为标准状态。
        try:
            normalized = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(f"{source} RNG 状态无法转换为 CPU ByteTensor") from error
    # Generator.set_state 期望连续一维字节序列；view(-1) 也兼容旧版列张量。
    return normalized.contiguous().view(-1)


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """尽可能恢复随机状态；附加状态不兼容不应阻断主权重续训。"""

    # None/空映射表示旧断点不保存随机状态，续训仍可从当前种子继续。
    if not state:
        return
    # 每个随机源独立恢复；某一项旧格式失败时不丢弃已载入的模型和优化器。
    if state.get("python") is not None:
        try:
            random.setstate(state["python"])
        except (TypeError, ValueError) as error:
            warnings.warn(
                f"跳过不兼容的 Python RNG 状态：{error}",
                RuntimeWarning,
                stacklevel=2,
            )
    if state.get("numpy") is not None:
        try:
            np.random.set_state(state["numpy"])
        except (TypeError, ValueError) as error:
            warnings.warn(
                f"跳过不兼容的 NumPy RNG 状态：{error}",
                RuntimeWarning,
                stacklevel=2,
            )
    if state.get("torch") is not None:
        try:
            # CPU RNG 状态即使被 map_location 搬到 CUDA，也先转回 CPU uint8。
            torch.set_rng_state(_cpu_byte_rng_state(state["torch"], "Torch CPU"))
        except (TypeError, ValueError, RuntimeError) as error:
            warnings.warn(
                f"跳过不兼容的 Torch CPU RNG 状态：{error}",
                RuntimeWarning,
                stacklevel=2,
            )
    # CUDA 可见设备数量可能与保存时不同，只恢复当前存在的前若干状态。
    cuda_states = state.get("cuda")
    if torch.cuda.is_available() and cuda_states is not None:
        for device_index, cuda_state in enumerate(list(cuda_states)[: torch.cuda.device_count()]):
            try:
                # CUDA Generator 的序列化接口同样接受 CPU ByteTensor，而不是 CUDA Tensor。
                torch.cuda.set_rng_state(
                    _cpu_byte_rng_state(cuda_state, f"CUDA:{device_index}"),
                    device_index,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                warnings.warn(
                    f"跳过不兼容的 CUDA:{device_index} RNG 状态：{error}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """读取完整训练断点或裸 ``state_dict``，不执行任何哈希校验。"""

    # 明确要求目标是文件，目录搜索由 find_latest_checkpoint 负责。
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"断点不存在：{target}")
    # map_location 允许在 CPU 上检查 GPU 断点；weights_only=False 兼容 RNG/NumPy 状态。
    payload = torch.load(target, map_location=device, weights_only=False)
    # 本项目和 PyTorch state_dict 顶层都应是字典。
    if not isinstance(payload, dict):
        raise TypeError(f"断点顶层必须是字典：{target}")
    return payload


def _filename_progress(path: Path) -> int:
    """从 epoch_0078.pt、step-500.pth 等文件名推断进度。"""

    # 优先匹配带语义前缀的数字，避免把模型版本号误当训练进度。
    matches = re.findall(r"(?:epoch|step|iter(?:ation)?)[_-]?(\d+)", path.stem.lower())
    if matches:
        # 多次出现时使用最后一个，通常最接近文件名末尾的实际进度。
        return int(matches[-1])
    # 旧文件名若恰好只有一个数字也可兼容；多个无语义数字则放弃猜测。
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[0]) if len(numbers) == 1 else -1


def checkpoint_progress(payload: Mapping[str, Any], path: Path | None = None) -> int:
    """优先读取阶段主进度，最后才尝试文件名。

    epoch 型断点常同时保存更大的 ``global_step``；因此 epoch 必须排在 global_step
    前面，否则“第 1 轮、全局第 300 步”会被误判成已完成 300 轮。
    """

    # 按阶段主进度优先级查找；iteration 用于蒸馏，epoch 用于生成/评估训练。
    for key in ("iteration", "epoch", "step", "global_step"):
        value = payload.get(key)
        # 允许旧断点保存成浮点数，但统一向下转换为已完成整数进度。
        if isinstance(value, (int, float)):
            return int(value)
    # 断点内部无进度时再尝试文件名；连路径也没有时视作进度 0。
    return _filename_progress(path) if path is not None else 0


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """在一个阶段目录中寻找最适合续训的权重。

    ``checkpoint_last`` 拥有最高优先级；否则读取所有候选文件的进度字段，选择进度
    最大者。损坏文件会被跳过，避免一次未完成写入挡住其他正常断点。
    """

    # 调用方既可直接给权重文件，也可给阶段目录。
    root = Path(directory)
    # 直接文件只要扩展名合法就返回，实际可读性由 load_checkpoint 检查。
    if root.is_file() and root.suffix.lower() in {".pt", ".pth"}:
        return root
    # 不存在或不是目录时表示首次训练，没有断点。
    if not root.is_dir():
        return None
    # 规范的 last 文件最能代表训练进度，优先级高于历史周期快照。
    for name in ("checkpoint_last.pt", "checkpoint_last.pth", "last.pt", "last.pth"):
        preferred = root / name
        if preferred.is_file():
            try:
                # 先实际反序列化，避免返回一次断电留下的损坏文件。
                load_checkpoint(preferred, "cpu")
                return preferred
            except Exception:
                # 例如上次进程在写入中途掉电；继续寻找同目录其他完整权重。
                continue
    # 没有可用 last 时，收集目录根层所有 pt/pth；临时文件不参与候选。
    candidates = [
        path
        for pattern in ("*.pt", "*.pth")
        for path in root.glob(pattern)
        if not path.name.endswith(".tmp")
    ]
    # 每项保存 (内部进度, 修改时间, 路径)，同进度时选择更新的文件。
    scored: list[tuple[int, float, Path]] = []
    for path in candidates:
        try:
            payload = load_checkpoint(path, "cpu")
            scored.append((checkpoint_progress(payload, path), path.stat().st_mtime, path))
        # 任意损坏/不兼容候选都跳过，继续寻找同目录其他完整断点。
        except Exception:
            continue
    # 无候选返回 None；否则先最大进度、再最新修改时间。
    return max(scored, default=(0, 0.0, None), key=lambda item: (item[0], item[1]))[2]


def model_state_from_checkpoint(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """兼容本项目断点以及常见的 model/state_dict/model_state_dict 命名。"""

    # 按最常见字段名依次查找，以便导入外部预训练权重或旧版本断点。
    for key in ("model", "model_state_dict", "state_dict", "weights"):
        state = payload.get(key)
        if isinstance(state, Mapping):
            return state
    # 若顶层所有值都是张量，则整个 payload 本身就是裸 state_dict。
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # 裸 state_dict
    # 不猜测其他嵌套字段，明确报告可接受命名。
    raise KeyError("断点中没有找到模型权重（model/state_dict/model_state_dict）")


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 DataParallel 保存的 ``module.`` 前缀。"""

    # 只有所有键都有前缀时才统一删除，避免误改恰好名为 module.xxx 的子模块。
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    # 返回新字典，调用方不会意外修改原断点映射。
    return dict(state)


def restore_training_state(
        directory: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        device: str | torch.device = "cpu",
        strict: bool = True,
        restore_rng: bool = True,
) -> tuple[int, Path | None, dict | None]:
    """从目录中最新权重恢复并返回 ``(已完成进度, 路径, 断点)``。

    如果目录没有权重，进度返回 0。当前配置是权威配置：调用方可在恢复后重新覆盖
    学习率等可变超参数；结构不兼容则由 ``load_state_dict`` 给出明确错误。
    """

    # 自动选择 last 或进度最大的合法断点；用户只需提供文件夹。
    path = find_latest_checkpoint(directory)
    # 首次运行没有权重，返回统一的“0、None、None”。
    if path is None:
        return 0, None, None
    # 权重按目标设备读取，避免加载后再次复制大型模型状态。
    payload = load_checkpoint(path, device)
    # strict 默认开启：网络结构/层宽变化必须明确报错，不能悄悄漏载权重。
    model.load_state_dict(_strip_module_prefix(model_state_from_checkpoint(payload)), strict=strict)
    # 优化器等附加状态存在且调用方提供对应对象时才恢复。
    if optimizer is not None and isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and isinstance(payload.get("scheduler"), Mapping):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and isinstance(payload.get("scaler"), Mapping):
        scaler.load_state_dict(payload["scaler"])
    # 可按调用场景关闭 RNG 恢复，例如只导入预训练模型而非精确续训。
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))
    # 进度至少为 0，避免无语义旧文件名的 -1 传播到训练循环。
    return max(0, checkpoint_progress(payload, path)), path, payload


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """恢复优化器后用当前配置覆盖学习率，允许用户安全地调整续训 LR。"""

    # 优化器可能含多个参数组，全部使用用户当前 YAML 中的学习率。
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
        # 某些调度器读取 initial_lr；存在时同步更新，避免下一步跳回旧值。
        if "initial_lr" in group:
            group["initial_lr"] = float(learning_rate)


def training_payload(
        model: torch.nn.Module,
        progress_name: str,
        progress: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        **extra: Any,
) -> dict:
    """构造统一训练断点；不写入配置或数据哈希。"""

    # 核心断点始终包含阶段进度、模型权重和所有随机源状态。
    payload: dict[str, Any] = {
        # progress_name 由阶段选择 epoch 或 iteration。
        str(progress_name): int(progress),
        # state_dict 仅包含参数与 buffer，不序列化整个 Python 模型对象。
        "model": model.state_dict(),
        "rng_state": capture_rng_state(),
    }
    # 以下状态按需加入，使同一辅助函数同时服务训练和纯权重保存。
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    # 阶段可附加 EMA、早停器、在线队列、类别采样器等可序列化状态。
    payload.update(extra)
    return payload