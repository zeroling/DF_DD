"""D/C 一键编排器：每个阶段使用独立子进程并在完成后汇总。"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping

from Core.io_utils import atomic_write_json, read_json
from Core.config import DEFAULT_CONFIG_PATH
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    load_ablation_config,
    output_root,
)
from Core.experiment_runtime import (
    experiment_environment,
    remove_stale_temporary_files,
)


def _run_worker(
    config_path: Path,
    arguments: list[str],
    smoke: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "Pipeline.ablation_worker",
        "--config",
        str(config_path),
        *arguments,
    ]
    if smoke:
        command.append("--smoke")
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _metrics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key_parts = []
        for key in ("experiment", "condensation_seed", "architecture"):
            if key in record:
                key_parts.append(str(record[key]))
        grouped.setdefault("/".join(key_parts), []).append(record)
    summary: dict[str, Any] = {}
    for key, values in grouped.items():
        summary[key] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            samples = [
                float(value["test_metrics"][metric]) for value in values
            ]
            summary[key][metric] = {
                "values": samples,
                "mean": statistics.fmean(samples),
                "std": statistics.stdev(samples)
                if len(samples) > 1
                else 0.0,
            }
    return summary


def _collect_results(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in root.rglob("result.json"):
        payload = read_json(path)
        if isinstance(payload, Mapping) and "test_metrics" in payload:
            record = dict(payload)
            # D 组旧结果没有显式 experiment 时从目录推断。
            parts = path.relative_to(root).parts
            if "experiment" not in record and parts:
                record["experiment"] = parts[0]
            records.append(record)
    return records


def _manifest(
    config: Mapping[str, Any],
    group: str,
    profile: str,
    smoke: bool,
) -> None:
    root = output_root(config)
    if smoke:
        root = (
            Path(config["_runtime"]["project_root"])
            / "outputs"
            / "ablation_smoke"
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {
            "group": str(group),
            "profile": str(profile),
            "smoke": bool(smoke),
            "config": str(config["_runtime"]["config_path"]),
            "resume_policy": "最新可读断点自动恢复；不进行哈希校验",
            "environment": experiment_environment(),
        },
        root / f"{str(group).upper()}_run_manifest.json",
    )


def _write_resolved_config_snapshot(
    config: Mapping[str, Any],
    group: str,
    smoke: bool,
) -> Path:
    """固定本次运行的配置，避免长任务期间编辑 YAML 影响后续子进程。"""

    root = output_root(config)
    if smoke:
        root = (
            Path(config["_runtime"]["project_root"])
            / "outputs"
            / "ablation_smoke"
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{str(group).upper()}_resolved_config.json"
    atomic_write_json(dict(config), path)
    return path


def run_group_d(
    config: Mapping[str, Any],
    config_path: Path,
    only: set[str] | None,
    smoke: bool,
) -> Path:
    settings = ablation_settings(config)
    experiments = [
        value
        for value in map(str, settings["diagnostic"]["experiments"])
        if not only or value in only
    ]
    architectures = list(
        map(str, condensation_settings(config)["evaluation"]["architectures"])
    )
    repeats = 1 if smoke else int(settings["diagnostic"]["repeats"])
    for experiment in experiments:
        if experiment in {"D1", "D2"}:
            _run_worker(
                config_path,
                ["--task", "d-cache", "--experiment", experiment],
                smoke,
            )
        for architecture in architectures:
            for repeat in range(repeats):
                _run_worker(
                    config_path,
                    [
                        "--task",
                        "d-eval",
                        "--experiment",
                        experiment,
                        "--architecture",
                        architecture,
                        "--repeat",
                        str(repeat),
                    ],
                    smoke,
                )
    root = (
        Path(config["_runtime"]["project_root"])
        / "outputs"
        / "ablation_smoke"
        / "D"
        if smoke
        else output_root(config) / "D"
    )
    records = _collect_results(root)
    path = root / "summary.json"
    atomic_write_json(
        {"records": records, "aggregate": _metrics_summary(records)}, path
    )
    return path


def run_group_c(
    config: Mapping[str, Any],
    config_path: Path,
    profile: str,
    only: set[str] | None,
    smoke: bool,
) -> Path:
    settings = ablation_settings(config)
    experiments = [
        value
        for value in map(str, settings["methods"])
        if not only or value in only
    ]
    profile_settings = settings["profiles"][profile]
    condensation_seeds = (
        1 if smoke else int(profile_settings["condensation_seeds"])
    )
    repeats = 1 if smoke else int(profile_settings["evaluation_repeats"])
    architectures = list(
        map(str, condensation_settings(config)["evaluation"]["architectures"])
    )
    for experiment in experiments:
        for condensation_seed in range(condensation_seeds):
            _run_worker(
                config_path,
                [
                    "--task",
                    "c-condense",
                    "--experiment",
                    experiment,
                    "--condensation-seed",
                    str(condensation_seed),
                ],
                smoke,
            )
            for architecture in architectures:
                for repeat in range(repeats):
                    _run_worker(
                        config_path,
                        [
                            "--task",
                            "c-eval",
                            "--experiment",
                            experiment,
                            "--condensation-seed",
                            str(condensation_seed),
                            "--architecture",
                            architecture,
                            "--repeat",
                            str(repeat),
                        ],
                        smoke,
                    )
    root = (
        Path(config["_runtime"]["project_root"])
        / "outputs"
        / "ablation_smoke"
        / "C"
        if smoke
        else output_root(config) / "C"
    )
    records = _collect_results(root)
    path = root / "summary.json"
    atomic_write_json(
        {"records": records, "aggregate": _metrics_summary(records)}, path
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键运行 IDM D 组诊断或 C 组 IPC=1 消融"
    )
    parser.add_argument("--group", required=True, choices=("D", "C", "d", "c"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", choices=("pilot", "formal"))
    parser.add_argument(
        "--only",
        action="append",
        help="只运行指定实验，可重复使用，例如 --only C0 --only C2",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_ablation_config(config_path)
    settings = ablation_settings(config)
    profile = str(
        args.profile or settings["default_profile"]
    )
    if profile not in settings["profiles"]:
        raise ValueError(f"未知 profile：{profile}")
    only = set(args.only or []) or None
    root = output_root(config)
    if args.smoke:
        root = (
            Path(config["_runtime"]["project_root"])
            / "outputs"
            / "ablation_smoke"
        )
    removed = remove_stale_temporary_files(root)
    if removed:
        print(f"已清理 {removed} 个中断遗留的 .tmp 文件")
    _manifest(config, args.group, profile, args.smoke)
    resolved_config_path = _write_resolved_config_snapshot(
        config, args.group, args.smoke
    )
    if args.group.upper() == "D":
        summary = run_group_d(
            config, resolved_config_path, only, args.smoke
        )
    else:
        summary = run_group_c(
            config,
            resolved_config_path,
            profile,
            only,
            args.smoke,
        )
    print(f"\n完成。汇总：{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
