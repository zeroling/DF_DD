"""项目快速与微型端到端测试。

默认测试配置合并、四种可调分类器、IDM/RBF 梯度、可微 DDIM 和无哈希断点恢复。
``--integration`` 额外在临时两类数据上用极小网络跑完整主流程和修改配置后的续训。
测试只验证代码契约，不代表真实性能或推荐超参数。
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path


def _sanitize_positive_thread_variable(name: str, fallback: int = 1) -> None:
    """在数值库导入前把非法的 OpenMP/MKL 线程数改成正整数。"""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = int(fallback)
    os.environ[name] = str(parsed_value if parsed_value > 0 else max(1, int(fallback)))


# 必须早于 NumPy/PyTorch；空串、0 或普通文本都会让 libgomp 在导入时报警。
_sanitize_positive_thread_variable("OMP_NUM_THREADS")
_sanitize_positive_thread_variable("MKL_NUM_THREADS")

import numpy as np
import torch
from PIL import Image

from Core.checkpoint import atomic_torch_save, restore_training_state
from Core.config import load_config
from Core.data import build_data_bundle, build_loader
from Net.Classification.factory import build_classifier_from_config
from Net.Condensation.losses import single_class_losses, weighted_total
from Net.Generative.diffusion import (
    build_ddim_scheduler,
    decode_latents,
    differentiable_ddim_sample,
)
from Net.Generative.models import build_autoencoder, build_diffusion_unet


def _make_dataset(root: Path, classes: list[str], size: int = 32) -> None:
    """创建颜色均值不同的微型 PNG 数据集，供测试训练与划分。"""

    # 使用独立 NumPy RNG，使测试图像不受项目全局 seed 影响。
    random_generator = np.random.default_rng(7)
    # 故意不创建 val，验证数据层能从 train 按类别自动划分。
    for split, count in (("train", 6), ("test", 3)):
        # 每个类别创建相同数量图像。
        for class_id, class_name in enumerate(classes):
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            # 类别通过不同基础亮度形成一个可学习但非常小的任务。
            for index in range(count):
                base = 50 + class_id * 130
                array = np.clip(
                    base + random_generator.normal(0, 20, (size, size, 3)),
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(array).save(directory / f"{index:03d}.png")


def _tiny_overrides(data_root: Path, run_root: Path) -> dict:
    """返回能在 CPU 快速运行但覆盖所有主模块的配置覆盖项。"""

    # 两类别足以验证类条件扩散和类别均衡队列。
    classes = ["class_a", "class_b"]
    # 只覆盖需要缩小的参数，未列出的值继续来自分文件 YAML。
    return {
        "project": {
            "run_dir": str(run_root),
            "device": "cpu",
            "num_workers": 0,
            "persistent_workers": False,
            "amp": False,
            "deterministic": False,
        },
        "data": {
            "root": str(data_root),
            "class_names": classes,
            "image": {"size": [32, 32], "channels": 3},
            "loader": {
                "train_batch_size": 4,
                "eval_batch_size": 4,
                "drop_last": False,
            },
            "augmentation": {"enabled": False},
        },
        "models": {
            "definitions": {
                "convnet": {
                    "widths": [8, 8, 8],
                    "kernel_size": 3,
                    "group_norm_groups": 8,
                    "pool_kernel_size": 2,
                    "feature_indices": {"shallow": 0, "middle": 1, "deep": 2},
                },
                "resnet18": {
                    "stage_widths": [8, 16, 32, 64],
                    "stage_blocks": [1, 1, 1, 1],
                    "stem_width": 8,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "use_max_pool": False,
                    "normalization": "group",
                    "group_norm_groups": 8,
                },
                "convnext_tiny": {
                    "custom": True,
                    "depths": [1, 1, 1, 1],
                    "dims": [8, 16, 32, 64],
                    "kernel_sizes": 3,
                    "patch_size": 2,
                    "drop_path_rate": 0.0,
                },
                "vit_tiny": {
                    "custom": True,
                    "patch_size": 8,
                    "embed_dim": 48,
                    "depth": 3,
                    "num_heads": 3,
                    "mlp_ratio": 2.0,
                    "drop_path_rate": 0.0,
                    "feature_indices": {"shallow": 0, "middle": 1, "deep": 2},
                },
            }
        },
        "autoencoder": {
            "epochs": 1,
            "batch_size": 4,
            "channels": [8, 16],
            "num_res_blocks": [1, 1],
            "attention_levels": [False, False],
            "latent_channels": 2,
            "norm_num_groups": 8,
            "gradient_checkpointing": False,
            # KL 从首轮就必须非零；两轮日程仍覆盖线性预热与断点续训。
            "kl_weight": 0.001,
            "kl_warmup_start_weight": 0.0005,
            "kl_warmup_epochs": 2,
            "preview_interval_epochs": 1,
            "checkpoint_interval_epochs": 1,
            "adversarial": {"enabled": False},
            "early_stopping": {"enabled": False},
        },
        "diffusion": {
            "epochs": 1,
            "batch_size": 4,
            "train_timesteps": 20,
            "channels": [16, 32],
            "num_res_blocks": [1, 1],
            "attention_levels": [False, False],
            "num_head_channels": [8, 8],
            "norm_num_groups": 8,
            "preview_interval_epochs": 1,
            "checkpoint_interval_epochs": 1,
            "preview_steps": 2,
            "validation_batches": 1,
            "early_stopping": {"enabled": False},
        },
        "condensation": {
            "real_per_class": 2,
            "checkpoint_interval_iterations": 1,
            "preview_interval_iterations": 0,
            "log_interval_iterations": 1,
            "synthetic_per_class_per_step": {"1": 1, "10": 2, "50": 2},
            "iterations": {"1": 1, "10": 1, "50": 1},
            "ddim_steps": {"1": 1, "10": 1, "50": 1},
            "idm_queue": {
                "architecture": "convnet",
                "initial_size": 2,
                "maximum_size": 3,
                "models_per_iteration": 1,
                "train_steps_per_model": 1,
                "generate_interval_iterations": 1,
                "batch_size": 4,
                "accuracy_ema_decay": 0.9,
                "minimum_reliability": 0.1,
                "optimization": {
                    "convnet": {"optimizer": "sgd", "learning_rate": 0.01}
                },
            },
            "topology": {
                "grids": {
                    "shallow": [2, 2],
                    "middle": [2, 2],
                    "deep": [1, 1],
                },
                "divergence": "js",
                "bandwidth": "real_median",
                "minimum_bandwidth": 0.0001,
            },
            "early_stopping": {"enabled": False},
        },
        "evaluation": {
            "architectures": ["convnet"],
            "repeats": 1,
            "batch_size": 2,
            "steps_per_epoch": 1,
            "epochs": {"1": 1, "10": 1, "50": 1},
            "sources": ["condensed"],
            "checkpoint_interval_epochs": 1,
            "validation_interval_epochs": 1,
            "optimization": {
                "convnet": {
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "min_learning_rate": 0.000001,
                }
            },
            "early_stopping": {"enabled": False, "minimum_epochs": 1},
        },
    }


def _assert_healthy_autoencoder_metrics(metrics: dict) -> None:
    """拒绝无 KL 约束、非有限值和明显失控的微型 VAE 潜空间。"""

    metric_names = (
        "generator",
        "reconstruction",
        "ssim",
        "kl",
        "weighted_kl",
        "kl_effective_weight",
        "validation_l1",
        "validation_raw_l1",
        "validation_zero_l1",
        "validation_raw_min",
        "validation_raw_max",
        "validation_raw_out_of_range_ratio",
        "validation_bounded_saturation_ratio",
        "latent_mean",
        "latent_std",
        "posterior_sigma_mean",
    )
    missing = [name for name in metric_names if name not in metrics]
    assert not missing, f"Autoencoder 断点缺少诊断指标：{missing}"
    non_finite = [name for name in metric_names if not math.isfinite(float(metrics[name]))]
    assert not non_finite, f"Autoencoder 出现非有限指标：{non_finite}"
    assert float(metrics["kl_effective_weight"]) > 0.0, "首轮 KL 权重不能为 0"
    assert float(metrics["weighted_kl"]) > 0.0, "KL 必须真实参与生成器目标"
    assert abs(float(metrics["latent_mean"])) < 3.0, "潜变量均值明显漂移"
    assert 0.05 < float(metrics["latent_std"]) < 3.0, "潜变量标准差坍缩或发散"
    assert 0.05 < float(metrics["posterior_sigma_mean"]) < 3.0, "后验 sigma 坍缩或发散"
    assert 0.0 <= float(metrics["validation_raw_out_of_range_ratio"]) <= 1.0
    assert 0.0 <= float(metrics["validation_bounded_saturation_ratio"]) <= 1.0


def quick_test() -> None:
    """执行不训练完整阶段的快速接口与梯度测试。"""

    # 临时目录退出后自动回收，不污染项目 outputs。
    with tempfile.TemporaryDirectory(prefix="miccai_quick_") as temporary:
        root = Path(temporary)
        data_root = root / "data"
        _make_dataset(data_root, ["class_a", "class_b"])
        # 多文件配置先合并，再应用微型覆盖。
        config = load_config(overrides=_tiny_overrides(data_root, root / "run"))
        bundle = build_data_bundle(config)
        # 验证数据层输出通道和空间尺寸。
        assert bundle.train[0]["image"].shape == (3, 32, 32)

        # 验证四种架构都能输出有限且非零的 IDM/RBF 输入梯度。
        real = torch.rand(3, 3, 32, 32)
        synthetic = torch.rand(1, 3, 32, 32, requires_grad=True)
        for architecture in ("convnet", "resnet18", "convnext_tiny", "vit_tiny"):
            model = build_classifier_from_config(
                config,
                architecture,
                bundle.num_classes,
            ).eval()
            real_output = model.forward_with_features(real)
            synthetic_output = model.forward_with_features(synthetic)
            losses = single_class_losses(
                real_output,
                synthetic_output,
                0,
                config["condensation"]["topology"],
            )
            gradient = torch.autograd.grad(
                weighted_total(losses, config["condensation"]["loss_profiles"]["1"]),
                synthetic,
                retain_graph=True,
            )[0]
            assert torch.isfinite(gradient).all()
            assert float(gradient.abs().sum()) > 0.0

        # 验证 DDIM+VAE 解码图可以把梯度传回初始隐变量。
        autoencoder = build_autoencoder(config)
        # Autoencoder 验证必须固定使用 decode(mean)，同一权重连续评估应完全一致。
        from Pipeline.Stages.train_autoencoder import _validation as validate_autoencoder

        validation_loader = build_loader(bundle.val, config, train=False, batch_size=4)
        first_validation = validate_autoencoder(autoencoder, validation_loader, torch.device("cpu"))
        second_validation = validate_autoencoder(autoencoder, validation_loader, torch.device("cpu"))
        assert first_validation == second_validation
        diffusion = build_diffusion_unet(config, bundle.num_classes)
        z_t = torch.randn(2, 2, 16, 16, requires_grad=True)
        labels = torch.tensor([0, 1])
        denoised = differentiable_ddim_sample(
            diffusion,
            build_ddim_scheduler(config),
            z_t,
            labels,
            bundle.num_classes,
            2,
            1.5,
        )
        decoded = decode_latents(autoencoder, denoised, 1.0)
        detached_decoded = decoded.detach()
        assert float(detached_decoded.min()) >= -1.0
        assert float(detached_decoded.max()) <= 1.0
        decoded.mean().backward()
        assert z_t.grad is not None and torch.isfinite(z_t.grad).all()

        # 验证只按权重文件进度恢复，不需要配置哈希。
        checkpoint_directory = root / "resume"
        model = torch.nn.Linear(3, 2)
        atomic_torch_save(
            {"epoch": 78, "model": model.state_dict()},
            checkpoint_directory / "epoch_0078.pt",
        )
        restored = torch.nn.Linear(3, 2)
        progress, path, _ = restore_training_state(checkpoint_directory, restored)
        assert progress == 78 and path is not None
    print("快速烟雾测试通过")


def integration_test() -> None:
    """在临时微型数据上跑主流程和修改目标后的续训。"""

    # 延迟导入避免快速测试仅导入阶段入口。
    from Pipeline.run_pipeline import run

    with tempfile.TemporaryDirectory(prefix="miccai_integration_") as temporary:
        root = Path(temporary)
        data_root = root / "data"
        run_root = root / "run"
        _make_dataset(data_root, ["class_a", "class_b"])
        # 第一轮所有阶段目标都设为 1。
        overrides = _tiny_overrides(data_root, run_root)
        config = load_config(overrides=overrides)
        outputs = run(config, selected_ipcs=[1])
        # 主流程已经不应生成静态专家库。
        assert not (run_root / "expert_bank").exists()
        assert (run_root / "autoencoder" / "checkpoint_last.pt").is_file()
        assert (run_root / "diffusion" / "checkpoint_last.pt").is_file()
        assert (run_root / "condensed" / "ipc_1" / "synthetic.pt").is_file()
        # 第 1 轮必须已有非零 KL 约束，并通过潜空间健康检查。
        first_autoencoder_payload = torch.load(
            run_root / "autoencoder" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        first_autoencoder_metrics = first_autoencoder_payload["metrics"]
        assert first_autoencoder_metrics["kl_effective_weight"] == 0.0005
        _assert_healthy_autoencoder_metrics(first_autoencoder_metrics)
        # 蒸馏断点必须包含同构 IDM 动态模型池当前状态。
        condensation_payload = torch.load(
            run_root / "condensed" / "ipc_1" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert condensation_payload["idm_queue"]["members"]
        assert outputs["stages"]["evaluate"]["records"]

        # 修改目标轮数和学习率后继续同一 run_dir，不比较任何哈希。
        resumed_overrides = _tiny_overrides(data_root, run_root)
        resumed_overrides["autoencoder"].update(
            {"epochs": 2, "learning_rate": 0.00008}
        )
        resumed_overrides["diffusion"].update(
            {"epochs": 2, "learning_rate": 0.00008}
        )
        resumed_overrides["condensation"].update(
            {
                "iterations": {"1": 2, "10": 1, "50": 1},
                "latent_learning_rate": 0.02,
            }
        )
        resumed_overrides["evaluation"].update(
            {"epochs": {"1": 2, "10": 1, "50": 1}}
        )
        resumed_overrides["evaluation"]["optimization"]["convnet"].update(
            {"learning_rate": 0.0007}
        )
        # 继续所有主阶段；在线 IDM 队列随 condensed 断点一起恢复。
        resumed_config = load_config(overrides=resumed_overrides)
        run(resumed_config, selected_ipcs=[1])
        # 验证各阶段真正从 1 继续到了 2。
        assert torch.load(
            run_root / "autoencoder" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )["epoch"] == 2
        # 两轮微型日程在第 2 轮必须精确到达配置的目标 KL 权重。
        resumed_autoencoder_payload = torch.load(
            run_root / "autoencoder" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert resumed_autoencoder_payload["metrics"]["kl_effective_weight"] == 0.001
        _assert_healthy_autoencoder_metrics(resumed_autoencoder_payload["metrics"])
        assert torch.load(
            run_root / "diffusion" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )["epoch"] == 2
        assert torch.load(
            run_root / "condensed" / "ipc_1" / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )["iteration"] == 2
    print("微型端到端集成测试通过")


def main() -> None:
    """解析是否运行端到端测试，并始终先执行快速测试。"""

    parser = argparse.ArgumentParser(description="在线 IDM 隐空间扩散项目测试")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="额外运行微型全流程与断点续训测试",
    )
    arguments = parser.parse_args()
    quick_test()
    if arguments.integration:
        integration_test()


if __name__ == "__main__":
    main()
