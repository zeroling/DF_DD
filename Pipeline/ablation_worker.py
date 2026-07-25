"""单子任务进程；退出后完整释放 CUDA 上下文。"""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime_compat import configure_runtime

configure_runtime()

from Core.config import DEFAULT_CONFIG_PATH
from Pipeline.Stages.condense import run_condensation
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    load_ablation_config,
    output_root,
)
from Pipeline.ablation_data import prepare_diagnostic_cache
from Pipeline.ablation_evaluate import (
    run_c_evaluation,
    run_d_evaluation,
)
from Core.experiment_runtime import (
    cleanup_memory,
    remove_stale_temporary_files,
)


def _smoke(config):
    config["_smoke"] = True
    config["_smoke_samples"] = 64
    root = (
        Path(config["_runtime"]["project_root"])
        / "outputs"
        / "ablation_smoke"
    )
    settings = ablation_settings(config)
    condensation = condensation_settings(config)
    settings["output_root"] = str(root.resolve())
    settings["diagnostic"]["full_data_epochs"] = 1
    settings["diagnostic"]["random_ipc_epochs"] = 1
    settings["diagnostic"]["checkpoint_interval_epochs"] = 1
    condensation["evaluation"]["epochs"] = {
        architecture: 1
        for architecture in condensation["evaluation"]["architectures"]
    }
    condensation["evaluation"]["checkpoint_interval_epochs"] = 1
    condensation["idm"]["iterations"] = 1
    # 冒烟仍保留官方有效 batch=128，用于真实压力测试；只缩短步数/迭代数。
    condensation["idm"]["batch_real"] = 128
    condensation["idm"]["batch_train"] = 128
    condensation["idm"]["model_train_steps"] = 1
    condensation["idm"]["net_num"] = 4
    condensation["idm"]["checkpoint_interval_iterations"] = 1
    condensation["idm"]["preview_interval_iterations"] = 0
    condensation["topology"]["calibration_iterations"] = [1]
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--task",
        required=True,
        choices=("d-cache", "d-eval", "c-condense", "c-eval"),
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--architecture")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--condensation-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = load_ablation_config(
        args.config if args.config else DEFAULT_CONFIG_PATH
    )
    if args.smoke:
        config = _smoke(config)
    remove_stale_temporary_files(output_root(config))
    try:
        if args.task == "d-cache":
            prepare_diagnostic_cache(config, args.experiment)
        elif args.task == "d-eval":
            if not args.architecture:
                raise ValueError("d-eval 必须指定 --architecture")
            run_d_evaluation(
                config,
                args.experiment,
                args.architecture,
                args.repeat,
            )
        elif args.task == "c-condense":
            run_condensation(
                config,
                args.experiment,
                args.condensation_seed,
            )
        else:
            if not args.architecture:
                raise ValueError("c-eval 必须指定 --architecture")
            run_c_evaluation(
                config,
                args.experiment,
                args.condensation_seed,
                args.architecture,
                args.repeat,
            )
    finally:
        cleanup_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
