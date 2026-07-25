"""子实验级内存清理、CUDA OOM 回退与输出辅助。"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def cleanup_memory() -> None:
    """释放 Python 引用循环、CUDA cache 与跨进程 IPC 句柄。"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def remove_stale_temporary_files(root: str | Path) -> int:
    """只删除实验输出中的原子写入临时文件，不碰正式断点。"""

    directory = Path(root)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.rglob("*.tmp"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def cuda_peak_megabytes() -> float:
    """返回 allocator 实际保留峰值，而非偏低的活跃张量峰值。"""

    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_reserved() / (1024**2))


def retry_cuda_oom(
    operation: Callable[[int], T],
    initial_batch: int,
    minimum_batch: int = 1,
) -> tuple[T, int]:
    """CUDA OOM 时把微批次减半并重试；非 OOM 异常原样抛出。"""

    batch = max(int(minimum_batch), int(initial_batch))
    while True:
        try:
            return operation(batch), batch
        except torch.OutOfMemoryError:
            cleanup_memory()
            if batch <= int(minimum_batch):
                raise
            batch = max(int(minimum_batch), batch // 2)


def experiment_environment() -> dict[str, str]:
    """记录实际硬件信息，不参与断点兼容判断。"""

    result = {
        "pid": str(os.getpid()),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": props.name,
                "gpu_memory_mib": str(int(props.total_memory / (1024**2))),
            }
        )
    return result
