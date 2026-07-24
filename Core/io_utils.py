"""小型 JSON/文本产物的原子读写。"""

from __future__ import annotations

import json  # 将指标、摘要和运行清单序列化为标准 JSON。
import os  # os.replace 在同一文件系统内提供原子替换语义。
from pathlib import Path  # 统一处理 Windows/Unix 路径。
from typing import Any  # JSON 入口允许字典、列表和标量等任意可序列化对象。


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    """以 UTF-8 和中文可读格式原子写入 JSON。"""

    # 将字符串和 Path 统一为 Path，后续所有路径操作保持平台无关。
    target = Path(path)
    # 阶段目录可能尚未建立，写文件前递归创建父目录。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标位于同一目录，保证 os.replace 不跨文件系统。
    temporary = target.with_suffix(target.suffix + ".tmp")
    # ensure_ascii=False 保留中文，indent=2 便于人工检查实验结果。
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 只有完整写完临时文件后才覆盖目标，进程中断不会留下半个 JSON。
    os.replace(temporary, target)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    """文件不存在时返回 default，存在但格式错误时保留异常。"""

    # 统一路径类型并只接受普通文件。
    target = Path(path)
    # 缺失是可预期状态，例如首次运行尚无 summary；直接返回调用方默认值。
    if not target.is_file():
        return default
    # JSON 损坏属于真实错误，不吞异常，以免继续使用不完整训练状态。
    return json.loads(target.read_text(encoding="utf-8"))
