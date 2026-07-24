"""控制台与文件双输出的阶段日志。"""

from __future__ import annotations

import logging  # Python 标准日志库，同时输出控制台和 UTF-8 文件。
from pathlib import Path  # 平台无关地创建阶段日志目录。


def get_stage_logger(name: str, directory: str | Path) -> logging.Logger:
    """为一个阶段创建独立 UTF-8 日志，重复调用不会叠加 handler。"""

    # 每个训练阶段使用自己的目录，便于单独续训和排查。
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 固定命名空间避免污染第三方库 root logger。
    logger = logging.getLogger(f"miccai.{name}")
    # 当前训练日志以 INFO 为主；异常仍由调用栈完整报告。
    logger.setLevel(logging.INFO)
    # 禁止向 root logger 传播，否则 PyCharm/脚本可能重复打印同一条消息。
    logger.propagate = False
    # 同一 Python 进程重复运行阶段时，先关闭并移除旧 handler，防止文件句柄泄漏。
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    # 时间、级别和正文三列足以重建训练过程，避免冗余模块名占据终端宽度。
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # 控制台 handler 让用户实时观察训练。
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    # 文件 handler 固定追加到该阶段的 train.log，并显式使用 UTF-8 保存中文。
    file_handler = logging.FileHandler(directory / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    # 两个 handler 使用相同格式，屏幕与文件内容便于对应。
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
