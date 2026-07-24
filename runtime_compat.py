"""必须在导入 PyTorch/NumPy 前应用的跨平台运行时设置。"""

from __future__ import annotations

import os
import sys


def _sanitize_positive_thread_variable(name: str, fallback: int = 1) -> None:
    """把非法线程数改成 OpenMP/MKL 接受的正整数。"""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = int(fallback)
    os.environ[name] = str(
        parsed_value if parsed_value > 0 else max(1, int(fallback))
    )


def _remove_allocator_option(name: str, option: str) -> None:
    """从逗号分隔的 PyTorch allocator 配置中移除一个不支持的选项。"""

    raw_value = os.environ.get(name)
    if not raw_value:
        return
    option_name = str(option).strip().lower()
    remaining = [
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
        and item.split(":", 1)[0].strip().lower() != option_name
    ]
    if remaining:
        os.environ[name] = ",".join(remaining)
    else:
        os.environ.pop(name, None)


def configure_runtime() -> None:
    """配置当前平台，并保持该函数可安全地重复调用。"""

    _sanitize_positive_thread_variable("OMP_NUM_THREADS")
    _sanitize_positive_thread_variable("MKL_NUM_THREADS")

    if sys.platform == "win32":
        # Intel Fortran/MKL 默认会抢先处理控制台 Ctrl+C 并打印 forrtl error
        # (200)。关闭它后，信号由 Python 转换为正常的 KeyboardInterrupt。
        os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
        # expandable_segments 当前不受 Windows CUDA allocator 支持。既不由项目
        # 自动设置，也清理由旧启动脚本或父进程继承的同名选项。
        _remove_allocator_option(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments",
        )
        _remove_allocator_option(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments",
        )
    else:
        # Linux CUDA 训练中可扩展段有助于减少 DDIM 大块缓存的显存碎片。
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

