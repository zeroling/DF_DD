"""统一命令入口：在线 IDM 隐空间主流程和分阶段续训。

完整主流程固定为 VAE→扩散→同构 IDM 动态模型池蒸馏→下游评价；静态专家和独立像素 IDM
阶段均已删除。每个阶段只读取自己的目录和断点，因此既可整条执行，也可从任意阶段开始。
"""

from __future__ import annotations

import argparse  # 定义命令行配置、阶段与 IPC 覆盖参数。
import gc  # 阶段切换时释放不再引用的 Python/模型对象。
import json  # 把 dry-run 与最终结果以中文可读 JSON 打印到终端。
import logging  # 管线结束时统一关闭阶段 FileHandler。
import sys  # Ctrl+C 时向控制台输出简短提示。
from typing import Any, Mapping  # 接受合并后的全局/阶段配置。

from runtime_compat import configure_runtime

# 兼容 ``python -m Pipeline.run_pipeline``；必须在导入 PyTorch 前执行。
configure_runtime()

import torch  # 设置矩阵乘精度并清理 CUDA 缓存。

from Core.config import DEFAULT_CONFIG_PATH, load_config, parse_cli_overrides, run_dir  # 配置入口。
from Core.io_utils import atomic_write_json  # 原子保存运行清单和阶段汇总。
from Core.run_context import prepare_run, resolve_device  # 创建目录并解析设备。
from Pipeline.Stages import condense, evaluate, train_autoencoder, train_diffusion  # 阶段实现。


# 所有可通过 ``--stage`` 独立调用的阶段注册表；值统一接受 config/selected_ipcs。
STAGES = {
    "train_autoencoder": train_autoencoder.run,
    "train_diffusion": train_diffusion.run,
    "condense": condense.run,
    "evaluate": evaluate.run,
}
# 完整扩散管线的唯一合法相对顺序；在线 IDM 在 condense 内部执行。
FULL_PIPELINE_ORDER = [
    "train_autoencoder",
    "train_diffusion",
    "condense",
    "evaluate",
]


def _select_stages(
    config: Mapping[str, Any],
    explicit: list[str] | None,
    from_stage: str | None,
    to_stage: str | None,
) -> list[str]:
    """解析显式阶段或 from/to 范围，并保持完整管线拓扑顺序。"""

    # 两种选择方式语义冲突，不能同时使用。
    if explicit and (from_stage or to_stage):
        raise ValueError("--stage 不能与 --from-stage/--to-stage 同时使用")
    # 显式阶段用 dict 保序去重，避免用户重复执行同一阶段。
    if explicit:
        return list(dict.fromkeys(explicit))
    # 未显式指定时读取 global.yaml 中启用的完整管线阶段。
    enabled = [str(value) for value in config["pipeline"].get("enabled_stages", FULL_PIPELINE_ORDER)]
    # pipeline.enabled_stages 不允许放入独立 IDM 或已删除的 train_experts。
    unknown = sorted(set(enabled) - set(FULL_PIPELINE_ORDER))
    if unknown:
        raise ValueError(f"pipeline.enabled_stages 含未知或非全流程阶段：{unknown}")
    # from/to 都按完整拓扑索引定位；缺失时分别取头/尾。
    start = FULL_PIPELINE_ORDER.index(from_stage) if from_stage else 0
    end = FULL_PIPELINE_ORDER.index(to_stage) if to_stage else len(FULL_PIPELINE_ORDER) - 1
    # 反向范围没有可执行语义，直接报错。
    if start > end:
        raise ValueError("--from-stage 必须位于 --to-stage 之前")
    # 切片后再与 enabled 求交集，允许全局关闭某个阶段。
    return [name for name in FULL_PIPELINE_ORDER[start : end + 1] if name in enabled]


def run(
    config: Mapping[str, Any],
    explicit_stages: list[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    selected_ipcs: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """依次执行选定阶段；每个阶段自行按目录中权重恢复进度。"""

    # 先解析最终执行阶段，dry-run 也走完全相同的选择逻辑。
    stages = _select_stages(config, explicit_stages, from_stage, to_stage)
    # CLI IPC 覆盖优先；排序去重后让输出目录和日志顺序稳定。
    ipcs = sorted(set(selected_ipcs or [int(value) for value in config["condensation"]["ipc_values"]]))
    # 当前论文实验设计固定报告 IPC=1/10/50，其他值需先扩展配置验证与实验协议。
    if any(ipc not in {1, 10, 50} for ipc in ipcs):
        raise ValueError("--ipc 只允许 1、10、50")
    # description 是不含大型张量的运行清单，可安全打印和写 JSON。
    description = {
        # config 是全局入口；config_files 记录实际合并的全部阶段文件。
        "config": str(config["_runtime"]["config_path"]),
        "config_files": dict(config["_runtime"].get("config_paths", {})),
        "run_dir": str(run_dir(config)),
        "device": str(resolve_device(config)),
        "stages": stages,
        "ipc_values": ipcs,
        "resume_policy": "读取阶段目录中的最新权重；不校验任何哈希",
    }
    # dry-run 不创建目录、不加载数据、不初始化模型。
    if dry_run:
        return description
    # 固定实验目录使阶段可以自动找到原断点继续训练。
    root = prepare_run(config)
    # 只接受 PyTorch 支持的三个 matmul 精度等级；配置校验已提前检查。
    if str(config["project"].get("matmul_precision", "high")) in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(str(config["project"].get("matmul_precision", "high")))
    # 每次启动覆盖当前运行清单，不保存/比较配置哈希。
    atomic_write_json(description, root / "run_manifest.json")
    # 阶段结果逐个写入，管线中途停止时已完成部分仍有摘要。
    outputs: dict[str, Any] = {"run": description, "stages": {}}
    try:
        # 严格按解析后的顺序同步执行，后一阶段可读取前一阶段产物。
        for stage_name in stages:
            # 所有阶段入口使用统一关键字；不需要 IPC 的阶段会忽略该参数。
            result = STAGES[stage_name](config, selected_ipcs=ipcs)
            outputs["stages"][stage_name] = result
            # 每完成一个阶段就原子刷新总摘要。
            atomic_write_json(outputs, root / "pipeline_summary.json")
            # 阶段函数不会把模型放进返回值；回收局部引用后归还 CUDA 缓存给下一阶段。
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        # Windows 不允许删除仍被 FileHandler 占用的目录；完整管线返回或报错时统一释放。
        logging.shutdown()
    return outputs


def main() -> None:
    """解析命令行、合并配置覆盖并执行管线。"""

    # ArgumentParser 同时服务根目录 run_pipeline.py 薄入口和模块直接运行。
    parser = argparse.ArgumentParser(
        description="IPC 自适应医学图像数据集蒸馏：同构 IDM 模型池 + 可微隐空间扩散"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="全局 YAML 入口；其中 stage_configs 会自动加载各阶段配置",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=list(STAGES),
        help="只运行指定阶段；可重复。在线 IDM 会在 condense 阶段内部运行",
    )
    parser.add_argument("--from-stage", choices=FULL_PIPELINE_ORDER, default=None)
    parser.add_argument("--to-stage", choices=FULL_PIPELINE_ORDER, default=None)
    parser.add_argument("--ipc", action="append", type=int, choices=[1, 10, 50], help="只运行指定 IPC；可重复")
    parser.add_argument("--run-dir", default=None, help="覆盖 project.run_dir；该目录中的权重会自动续训")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="临时覆盖配置，例如 --set diffusion.epochs=400；值按 YAML 解析",
    )
    parser.add_argument("--dry-run", action="store_true", help="只显示阶段、IPC、设备和目录")
    # 到这里才真正解析 sys.argv，导入模块本身不会启动训练。
    args = parser.parse_args()
    # --set 的点号键转换为嵌套字典，并按 YAML 规则解析数字/布尔/列表。
    overrides = parse_cli_overrides(args.set)
    # --run-dir 是常用快捷参数，等价于覆盖 project.run_dir。
    if args.run_dir:
        overrides.setdefault("project", {})["run_dir"] = args.run_dir
    # 先加载全局文件，再加载各阶段文件，最后应用 CLI 覆盖并完整校验。
    config = load_config(args.config, overrides=overrides)
    # 阶段函数会自行寻找断点，入口无需传 resume 标志。
    try:
        result = run(
            config,
            explicit_stages=args.stage,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            selected_ipcs=args.ipc,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        # Windows 已让 Python 接管 Ctrl+C；用标准 130 退出码代替 forrtl 崩溃信息。
        print("\n运行已由用户取消；已有断点和日志不会被删除。", file=sys.stderr)
        raise SystemExit(130) from None
    # ensure_ascii=False 保留中文路径和状态说明。
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 只有作为脚本执行时才进入 CLI；测试可直接导入 run。
    main()
