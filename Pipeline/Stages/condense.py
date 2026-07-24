"""原始 IDM 同构模型池指导冻结隐空间扩散生成器优化 ``z_T``。

本阶段是主方法的核心，参数更新关系严格固定为：

* AutoencoderKL：加载最新权重并冻结；
* DiffusionModelUNet：加载最新 EMA 权重并冻结；
* IDM 同构模型池：一次选择一种架构，持续注入随机初始化，只用真实 batch 更新；
* 合成隐变量 ``z_T``：通过可微 DDIM、VAE 解码、IDM/RBF 损失进行更新。

分类器在线更新与合成隐变量更新交替执行，但不会对分类器训练步骤求二阶梯度，也不会
用合成图像更新分类器。断点只保存当前活跃队列，不保存已经淘汰的训练轨迹。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import time
from typing import Any, Mapping

import torch
from torch import nn
from torchvision.utils import save_image

from Core.augmentations import (
    TensorBatchAugment,
    classifier_normalize,
    from_generator_range,
    to_generator_range,
)
from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_rng_state,
)
from Core.config import stage_dir
from Core.data import ClassImagePool, build_data_bundle
from Core.io_utils import atomic_write_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, resolve_device, stage_checkpoint_directory
from Core.seed import seed_everything
from Core.training import EarlyStopping, ExponentialMetric
from Net.Condensation.idm_queue import IDMModelQueue
from Net.Condensation.losses import detached_loss_values, single_class_losses, weighted_total
from Net.Generative.diffusion import (
    add_maximum_noise,
    build_ddim_scheduler,
    build_ddpm_scheduler,
    decode_latents,
    differentiable_ddim_sample,
    encode_latents,
)
from Net.Generative.models import (
    build_autoencoder,
    build_diffusion_unet,
    freeze_module,
    latent_spatial_size,
)


CONDENSATION_METHOD = "latent_diffusion_idm_homogeneous_topology_v2"


class LearnableLatentSet(nn.Module):
    """为每个类别保存恰好 IPC 个可学习扩散初始隐变量。"""

    def __init__(self, initial_latents: torch.Tensor) -> None:
        """把初始化张量注册为唯一需要优化的模型参数。"""

        # 初始化父类后才能注册 Parameter。
        super().__init__()
        # 标签由外部按类别顺序固定，只有该张量参与 Adam/AdamW 更新。
        self.latents = nn.Parameter(initial_latents)


def _load_generative_models(
    config: Mapping[str, Any],
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, float, Path, Path]:
    """加载并冻结最新 VAE 与扩散 EMA 权重。"""

    # 每个阶段只读取自己的固定输出目录。
    autoencoder_path = find_latest_checkpoint(stage_dir(config, "autoencoder", create=False))
    diffusion_path = find_latest_checkpoint(stage_dir(config, "diffusion", create=False))
    # 两个生成模块都是蒸馏的前置依赖。
    if autoencoder_path is None or diffusion_path is None:
        raise FileNotFoundError("condense 前必须完成 train_autoencoder 和 train_diffusion")
    # 完整训练断点还含优化器、普通权重和 EMA；先在 CPU 解包，避免重复副本污染显存。
    autoencoder_payload = load_checkpoint(autoencoder_path, "cpu")
    diffusion_payload = load_checkpoint(diffusion_path, "cpu")
    # latent_scale 优先读取扩散断点，以确保与扩散训练时的编码尺度一致。
    scale = diffusion_payload.get("latent_scale", autoencoder_payload.get("latent_scale"))
    if scale is None or float(scale) <= 0:
        raise ValueError("生成模型断点中没有有效 latent_scale")
    # 按当前配置重建 VAE 结构并加载最近权重。
    autoencoder = build_autoencoder(config)
    autoencoder.load_state_dict(model_state_from_checkpoint(autoencoder_payload), strict=True)
    # 按当前类别数重建条件扩散 U-Net。
    diffusion = build_diffusion_unet(config, int(num_classes))
    # 蒸馏优先使用训练期间维护的 EMA shadow，旧断点没有时回退到普通模型权重。
    ema_state = diffusion_payload.get("ema", {})
    ema_shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    diffusion.load_state_dict(
        ema_shadow
        if isinstance(ema_shadow, Mapping)
        else model_state_from_checkpoint(diffusion_payload),
        strict=True,
    )
    # 权重加载完成后只把最终模型各搬一次到 GPU；CPU payload 随函数返回立即释放。
    autoencoder = autoencoder.to(device)
    diffusion = diffusion.to(device)
    # 冻结模块参数，但后续仍允许梯度从图像穿过它们流向 z_T。
    return (
        freeze_module(autoencoder),
        freeze_module(diffusion),
        float(scale),
        autoencoder_path,
        diffusion_path,
    )


@torch.no_grad()
def _initial_latents(
    config: Mapping[str, Any],
    ipc: int,
    bundle,
    pool: ClassImagePool,
    autoencoder: nn.Module,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    """按 gaussian 或 real_noised 策略创建 ``类别数×IPC`` 个 ``z_T``。"""

    # VAE 下采样倍率由通道阶段数决定。
    latent_height, latent_width = latent_spatial_size(config)
    # 每个类别严格分配 IPC 个隐变量。
    shape = (
        bundle.num_classes * int(ipc),
        int(config["autoencoder"]["latent_channels"]),
        latent_height,
        latent_width,
    )
    # gaussian/random 都表示直接从扩散标准正态先验采样。
    initialization = str(config["condensation"].get("initialize", "real_noised")).lower()
    if initialization in {"gaussian", "random"}:
        return torch.randn(shape, device=device)
    # 只接受两个明确策略，防止拼写错误静默回退。
    if initialization != "real_noised":
        raise ValueError("condensation.initialize 只能是 real_noised 或 gaussian")
    # real_noised 先按类别抽取真实图像，再编码成干净隐变量。
    clean_parts: list[torch.Tensor] = []
    # 编码批大小沿用 Autoencoder 配置，避免一次编码 IPC=50 全部图像。
    encoding_batch = max(1, int(config["autoencoder"].get("batch_size", 16)))
    # 类别顺序与最终固定标签顺序保持一致。
    for class_id in range(bundle.num_classes):
        pixels = pool.sample(class_id, int(ipc)).to(device)
        # 分批编码真实图像均值，不采样 VAE 后验噪声。
        for start in range(0, pixels.shape[0], encoding_batch):
            clean_parts.append(
                encode_latents(
                    autoencoder,
                    to_generator_range(pixels[start : start + encoding_batch]),
                    scale,
                    sample=False,
                )
            )
    # 拼接成严格按类别排列的干净隐变量集合。
    clean_latents = torch.cat(clean_parts, dim=0)
    # 在 DDPM 最大时间步加噪，得到接近标准正态但保留弱初始化对应关系的 z_T。
    return add_maximum_noise(build_ddpm_scheduler(config), clean_latents)


def _build_latent_optimizer(
    settings: Mapping[str, Any],
    latent_set: LearnableLatentSet,
) -> torch.optim.Optimizer:
    """按配置创建只接收 ``z_T`` 参数的 Adam 或 AdamW。"""

    # 当前主方法只允许两种对隐变量稳定的自适应优化器。
    optimizer_name = str(settings.get("latent_optimizer", "adam")).lower()
    # 公共学习率从分阶段 YAML 读取。
    learning_rate = float(settings["latent_learning_rate"])
    # Adam 不使用解耦权重衰减。
    if optimizer_name == "adam":
        return torch.optim.Adam([latent_set.latents], lr=learning_rate)
    # AdamW 可通过 latent_weight_decay 对隐变量施加额外收缩。
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            [latent_set.latents],
            lr=learning_rate,
            weight_decay=float(settings.get("latent_weight_decay", 0.0)),
        )
    # 未实现优化器直接报错，避免悄悄改变实验。
    raise ValueError("condensation.latent_optimizer 只能是 adam 或 adamw")


def _restore_condensation(
    directory: Path,
    latent_set: LearnableLatentSet,
    optimizer: torch.optim.Optimizer,
    architecture: str,
) -> tuple[int, dict[str, Any] | None, Path | None]:
    """恢复同版本 IDM 隐变量、优化器和全局随机状态。"""

    # 优先读取 checkpoint_last，否则按进度寻找同目录其他完整权重。
    path = find_latest_checkpoint(directory)
    if path is None:
        return 0, None, None
    # 模型池断点保留在 CPU，逐成员恢复时再搬运必要状态。
    payload = load_checkpoint(path, "cpu")
    # 异构旧方法和当前同构 IDM 的模型池/优化轨迹没有可比性，必须从头开始。
    if payload.get("method") != CONDENSATION_METHOD:
        return 0, None, path
    saved_queue = payload.get("idm_queue", {})
    saved_architecture = (
        str(saved_queue.get("architecture", architecture)).lower()
        if isinstance(saved_queue, Mapping)
        else str(architecture).lower()
    )
    if saved_architecture != str(architecture).lower():
        raise ValueError(
            "不能在同一凝聚断点中切换 IDM 架构："
            f"断点={saved_architecture}，当前={architecture}；请指定新的 run_dir"
        )
    # 兼容当前 latents 命名和非常早期使用 model 命名的断点。
    latent_state = payload.get("latents", payload.get("model"))
    if not isinstance(latent_state, Mapping):
        raise KeyError(f"蒸馏断点缺少 latents：{path}")
    # 隐变量形状必须与当前数据类别数、IPC 和 VAE 尺寸兼容。
    latent_set.load_state_dict(latent_state, strict=True)
    # 恢复 Adam 动量；当前学习率稍后由 YAML 覆盖。
    if isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    # 恢复 Python、NumPy、Torch 和 CUDA 随机状态。
    restore_rng_state(payload.get("rng_state"))
    # 返回已完成迭代、完整载荷和实际路径。
    return int(payload.get("iteration", 0)), payload, path


def _save_condensation(
    directory: Path,
    iteration: int,
    latent_set: LearnableLatentSet,
    optimizer: torch.optim.Optimizer,
    queue: IDMModelQueue,
    pool: ClassImagePool,
    early_stopping: EarlyStopping,
    loss_smoother: ExponentialMetric,
    ipc: int,
    class_names: list[str],
    diagnostics: Mapping[str, float],
    snapshot_path: Path | None = None,
) -> Path:
    """原子保存继续蒸馏所需的完整当前状态。"""

    # 断点只含当前在线成员，不含已经淘汰的历史轨迹。
    payload = {
        "method": CONDENSATION_METHOD,
        "iteration": int(iteration),
        "latents": latent_set.state_dict(),
        "optimizer": optimizer.state_dict(),
        "idm_queue": queue.state_dict(),
        "real_pool": pool.state_dict(),
        "early_stopping": early_stopping.state_dict(),
        "loss_smoother": loss_smoother.state_dict(),
        "ipc": int(ipc),
        "class_names": list(class_names),
        "diagnostics": dict(diagnostics),
        "queue_summary": queue.summary(),
        "rng_state": capture_rng_state(),
    }
    # checkpoint_last 始终表示该 IPC 当前最新进度。
    latest_path = atomic_torch_save(payload, directory / "checkpoint_last.pt")
    # 长周期快照保存同一份完整状态，供回滚和逐阶段评估；普通 50 步断点不复制。
    if snapshot_path is not None:
        atomic_torch_save(payload, snapshot_path)
    return latest_path


@torch.no_grad()
def _decode_latent_batch(
    config: Mapping[str, Any],
    latent_batch: torch.Tensor,
    label_batch: torch.Tensor,
    autoencoder: nn.Module,
    diffusion: nn.Module,
    scale: float,
    ipc: int,
    num_classes: int,
) -> torch.Tensor:
    """使用当前 IPC 的 DDIM/CFG 参数把一批 ``z_T`` 解码为 ``[0,1]`` 图像。"""

    # 每次调用使用独立 DDIM 调度器，避免内部时间步状态互相覆盖。
    scheduler = build_ddim_scheduler(config)
    # 无梯度采样用于预览和最终导出，不保留昂贵计算图。
    denoised = differentiable_ddim_sample(
        diffusion,
        scheduler,
        latent_batch,
        label_batch,
        int(num_classes),
        int(config["condensation"]["ddim_steps"][str(ipc)]),
        float(config["condensation"]["guidance_scale"][str(ipc)]),
    )
    # VAE 解码输出位于 [-1,1]，转换并裁剪到数据集统一的 [0,1]。
    return from_generator_range(decode_latents(autoencoder, denoised, scale))


@torch.no_grad()
def _save_preview(
    config: Mapping[str, Any],
    directory: Path,
    iteration: int,
    latent_set: LearnableLatentSet,
    labels: torch.Tensor,
    autoencoder: nn.Module,
    diffusion: nn.Module,
    scale: float,
    ipc: int,
    num_classes: int,
) -> Path:
    """分批解码并保存当前完整 ``类别数×IPC`` 合成图网格。"""

    # 预览 batch 独立于训练 batch，防止 IPC=50 一次解码全部隐变量导致显存峰值。
    batch_size = max(
        1,
        int(config["condensation"].get("preview_batch_size", 16)),
    )
    image_parts: list[torch.Tensor] = []
    # 标签和隐变量顺序固定为 [0×IPC,1×IPC,...]，分批不会改变最终排列。
    for start in range(0, latent_set.latents.shape[0], batch_size):
        image_parts.append(
            _decode_latent_batch(
                config,
                latent_set.latents[start : start + batch_size],
                labels[start : start + batch_size],
                autoencoder,
                diffusion,
                scale,
                ipc,
                num_classes,
            ).cpu()
        )
    images = torch.cat(image_parts, dim=0)
    # 文件名包含迭代数；每行最多十张，避免 IPC=50 生成超宽 PNG。
    path = directory / f"preview_iteration_{int(iteration):06d}.png"
    save_image(images, path, nrow=min(10, int(ipc)))
    return path


@torch.no_grad()
def _export(
    config: Mapping[str, Any],
    directory: Path,
    latent_set: LearnableLatentSet,
    labels: torch.Tensor,
    autoencoder: nn.Module,
    diffusion: nn.Module,
    scale: float,
    ipc: int,
    class_names: list[str],
) -> Path:
    """把全部优化隐变量分批解码，保存 tensor 数据集、总预览和逐类 PNG。"""

    # 正式扩散训练 batch 通常远大于 16GB 显卡能承受的 DDIM 解码 batch；
    # 导出使用凝聚阶段自己的安全批大小，与训练权重和结果顺序无关。
    batch_size = max(
        1,
        int(
            config["condensation"].get(
                "export_batch_size",
                config["condensation"].get("preview_batch_size", 4),
            )
        ),
    )
    # 分批结果先放回 CPU，再统一拼接。
    image_parts: list[torch.Tensor] = []
    for start in range(0, latent_set.latents.shape[0], batch_size):
        image_parts.append(
            _decode_latent_batch(
                config,
                latent_set.latents[start : start + batch_size],
                labels[start : start + batch_size],
                autoencoder,
                diffusion,
                scale,
                ipc,
                len(class_names),
            ).cpu()
        )
    # 合并后的顺序仍是每类连续 IPC 张。
    images = torch.cat(image_parts, dim=0)
    # synthetic.pt 是评估阶段的唯一机器读取入口。
    dataset_path = directory / "synthetic.pt"
    atomic_torch_save(
        {
            "images": images,
            "labels": labels.cpu(),
            "ipc": int(ipc),
            "class_names": list(class_names),
            "pixel_range": [0.0, 1.0],
            "method": CONDENSATION_METHOD,
            "optimized_variable": "z_T",
        },
        dataset_path,
    )
    # 总览图按每类 IPC 排列，nrow 上限避免横向过宽。
    save_image(images, directory / "preview.png", nrow=min(10, int(ipc)))
    # 同时导出人类可直接查看的逐类 PNG。
    image_root = directory / "images"
    for index, image in enumerate(images):
        # 标签来自固定顺序，不参与任何字符串推断。
        class_id = int(labels[index])
        # Windows 文件名不允许的字符统一替换为下划线。
        safe_class_name = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]',
            "_",
            str(class_names[class_id]),
        ).strip(" .") or f"class_{class_id}"
        # 每个类别单独目录，方便人工抽查。
        class_directory = image_root / safe_class_name
        class_directory.mkdir(parents=True, exist_ok=True)
        # 类内编号从 0 到 IPC-1。
        class_local_index = index - class_id * int(ipc)
        save_image(image, class_directory / f"synthetic_{class_local_index:04d}.png")
    return dataset_path


def _run_ipc(
    config: Mapping[str, Any],
    ipc: int,
    bundle,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    """独立优化一个 IPC 档位并导出合成数据。"""

    # 读取该阶段分文件配置。
    settings = config["condensation"]
    # 不同 IPC 使用独立目录和断点，互不覆盖。
    directory = stage_checkpoint_directory(config, "condensed", f"ipc_{int(ipc)}")
    # 类别池按需读取真实图像，不把完整数据集常驻内存。
    pool = ClassImagePool(
        bundle.train,
        bundle.num_classes,
        int(config["project"]["seed"]) + int(ipc) * 17,
        cache_images=bool(settings.get("cache_real_images", False)),
    )
    # 加载冻结生成器。
    logger.info(
        "IPC=%d 正在从断点加载并冻结 Autoencoder 与 Diffusion（大文件首次加载可能需要一些时间）",
        ipc,
    )
    autoencoder, diffusion, scale, autoencoder_path, diffusion_path = _load_generative_models(
        config,
        bundle.num_classes,
        device,
    )
    logger.info(
        "IPC=%d 生成模型已就绪：autoencoder=%s diffusion=%s",
        ipc,
        autoencoder_path,
        diffusion_path,
    )
    # 创建初始 z_T；若有断点，稍后会整体覆盖该张量。
    logger.info("IPC=%d 正在初始化可学习隐变量 z_T", ipc)
    initial_latents = _initial_latents(
        config,
        int(ipc),
        bundle,
        pool,
        autoencoder,
        scale,
        device,
    )
    # 注册可学习隐变量集合。
    latent_set = LearnableLatentSet(initial_latents).to(device)
    # 优化器只持有 latent_set.latents 一个参数。
    optimizer = _build_latent_optimizer(settings, latent_set)
    # 先恢复 z_T 和公共状态。
    completed, resume_payload, resume_path = _restore_condensation(
        directory,
        latent_set,
        optimizer,
        str(settings["idm_queue"]["architecture"]),
    )
    # 恢复真实类别采样器位置，保证迭代级连续性。
    pool.load_state_dict((resume_payload or {}).get("real_pool"))
    # 当前 YAML 学习率覆盖断点旧值，允许用户修改后直接续训。
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = float(settings["latent_learning_rate"])
    # IDM 模型池从多个独立随机模型开始，并随迭代持续注入同架构新随机种子。
    logger.info(
        "IPC=%d 正在初始化 IDM 模型池：architecture=%s initial_size=%d",
        ipc,
        settings["idm_queue"]["architecture"],
        int(settings["idm_queue"].get("initial_size", 3)),
    )
    queue = IDMModelQueue(
        config=config,
        settings=settings["idm_queue"],
        num_classes=bundle.num_classes,
        device=device,
        seed=int(config["project"]["seed"]) + int(ipc) * 101,
    )
    # 同版本断点恢复完整大模型池和各成员优化器状态。
    idm_queue_state = (resume_payload or {}).get("idm_queue")
    queue.load_state_dict(idm_queue_state)
    # 构造/恢复池成员会消耗 Torch 初始化随机数；重新恢复一次全局 RNG，保证断点后
    # 的 z_T 子采样、真实数据采样和新模型初始化与不中断运行一致。
    if resume_payload is not None:
        restore_rng_state(resume_payload.get("rng_state"))
    logger.info("IPC=%d 初始化完成，IDM 模型池=%s", ipc, queue.summary())
    # 构造平滑损失与整个 IPC 的可选早停器。
    early_settings = settings.get("early_stopping", {})
    loss_smoother = ExponentialMetric(decay=float(early_settings.get("smoothing_decay", 0.95)))
    loss_smoother.load_state_dict((resume_payload or {}).get("loss_smoother"))
    early_stopping = EarlyStopping.from_config(
        early_settings,
        interval_key="check_interval_iterations",
        minimum_key="minimum_iterations",
    )
    early_stopping.load_state_dict(
        (resume_payload or {}).get("early_stopping"),
        reset=bool(early_settings.get("reset_on_resume", False)),
    )
    # 旧异构断点只被识别，不会把旧 z_T 或队列混入新方法。
    if resume_path and resume_payload is None:
        logger.warning(
            "IPC=%d 检测到不兼容的旧凝聚断点 %s；"
            "当前方法=%s，将从 iteration=0 重新开始并在首次保存时覆盖 checkpoint_last.pt",
            ipc,
            resume_path,
            CONDENSATION_METHOD,
        )
    elif resume_path:
        logger.info(
            "IPC=%d 从 %s 恢复：%d/%d iteration；IDM 模型池=%s",
            ipc,
            resume_path,
            completed,
            settings["iterations"][str(ipc)],
            "已恢复" if idm_queue_state else "从随机状态新建",
        )

    # 固定标签顺序为 [0×IPC, 1×IPC, ...]。
    labels = torch.arange(bundle.num_classes, device=device).repeat_interleave(int(ipc))
    # 当前 IPC 对应的损失权重配置。
    profile = settings["loss_profiles"][str(ipc)]
    # 真实匹配 batch 可按当前同构架构覆盖，控制大模型的特征激活峰值。
    architecture = str(settings["idm_queue"]["architecture"]).lower()
    real_per_class_setting = settings["real_per_class"]
    real_per_class = int(
        real_per_class_setting.get(architecture, 1)
        if isinstance(real_per_class_setting, Mapping)
        else real_per_class_setting
    )
    # 每次每类最多处理的合成隐变量数量。
    per_class = int(settings["synthetic_per_class_per_step"][str(ipc)])
    # 当前 IPC 最大迭代数。
    target_iterations = int(settings["iterations"][str(ipc)])
    logger.info(
        "IPC=%d 16GB 显存参数：real_per_class=%d synthetic_per_class=%d "
        "DDIM checkpoint=%s preview/export batch=%d/%d",
        ipc,
        real_per_class,
        per_class,
        bool(settings.get("diffusion_gradient_checkpointing", True)),
        int(settings.get("preview_batch_size", 4)),
        int(settings.get("export_batch_size", settings.get("preview_batch_size", 4))),
    )
    # 同一增强器分别用于在线真实训练和各专家的合成前向。
    augment = TensorBatchAugment.from_config(config["data"].get("augmentation", {}))
    # DDIM 调度器在训练循环内重复使用相同噪声定义。
    ddim_scheduler = build_ddim_scheduler(config)
    # 旧断点诊断只用于首次日志和无新增迭代的最终汇总。
    last_diagnostics: dict[str, float] = dict((resume_payload or {}).get("diagnostics", {}))
    # 实际完成进度初始为断点值。
    current_iteration = int(completed)

    # 从断点下一步开始，直到目标迭代或早停。
    for iteration in range(int(completed) + 1, target_iterations + 1):
        current_iteration = int(iteration)
        iteration_started = time.perf_counter()
        # 每轮单独记录峰值，便于区分模型常驻显存、缓存和瞬时反向激活。
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # 第一步：从同构大模型池随机抽取少量冻结模型，本轮所有类别共用。
        guidance_members = queue.sample_guidance_members()
        # 清除上一步遗留的 z_T 梯度。
        optimizer.zero_grad(set_to_none=True)
        # 累计所有类别和专家的可读损失诊断。
        loss_sums: dict[str, float] = defaultdict(float)
        # 收集本轮实际选择的隐变量，用于记录均值和标准差。
        selected_latent_values: list[torch.Tensor] = []
        # expert_count 用于把损失累计值还原为平均专家损失。
        expert_count = 0

        # 按类别依次建立和释放 DDIM 计算图，显著降低 224×224 峰值显存。
        for class_id in range(bundle.num_classes):
            # IPC 大于单步上限时随机选择该类部分隐变量。
            local_indices = torch.randperm(int(ipc), device=device)[: min(int(ipc), per_class)]
            # 转换为扁平 latent_set 中的全局索引。
            selected_indices = local_indices + class_id * int(ipc)
            # 取得当前类本轮要更新的 z_T。
            class_latents = latent_set.latents[selected_indices]
            # DDIM 使用固定类别条件。
            class_labels = torch.full(
                (class_latents.shape[0],),
                int(class_id),
                device=device,
                dtype=torch.long,
            )
            # 可微 DDIM 和 VAE 解码保留到 z_T 的计算图。
            with autocast_context(config, device):
                denoised_latents = differentiable_ddim_sample(
                    diffusion,
                    ddim_scheduler,
                    class_latents,
                    class_labels,
                    bundle.num_classes,
                    int(settings["ddim_steps"][str(ipc)]),
                    float(settings["guidance_scale"][str(ipc)]),
                    gradient_checkpointing=bool(
                        settings.get("diffusion_gradient_checkpointing", True)
                    ),
                )
                generated_pixels = from_generator_range(
                    decode_latents(autoencoder, denoised_latents, scale)
                )

            # 收集不同随机种子模型对生成图像的梯度，随后做普通算术平均。
            model_gradients: list[torch.Tensor] = []
            # 同一真实 batch 在本轮全部 IDM 模型间复用。
            real_pixels = pool.sample(
                class_id,
                real_per_class,
            ).to(device)
            # 每个随机模型独立提取真实/合成特征。
            for member in guidance_members:
                expert = member.model
                # 真实分支不需要任何梯度。
                with torch.no_grad(), autocast_context(config, device):
                    real_output = expert.forward_with_features(
                        classifier_normalize(augment(real_pixels), config)
                    )
                # 合成分支需要保留对 generated_pixels 的梯度，但专家参数已经冻结。
                with autocast_context(config, device):
                    synthetic_output = expert.forward_with_features(
                        classifier_normalize(augment(generated_pixels), config)
                    )
                # 原始 IDM 均值/分类目标加上保留的浅、中、深分层拓扑创新项。
                components = single_class_losses(
                    real_output,
                    synthetic_output,
                    class_id,
                    settings["topology"],
                )
                # 在线真实训练准确率只调节合成 CE，其他分布匹配项始终保留。
                expert_loss = weighted_total(
                    components,
                    profile,
                    expert_reliability=member.reliability(
                        float(settings["idm_queue"].get("minimum_reliability", 0.05))
                    ),
                )
                # 先求损失对生成图像的梯度，切断不同架构内部计算图。
                image_gradient = torch.autograd.grad(
                    expert_loss,
                    generated_pixels,
                    retain_graph=False,
                )[0]
                # 输入梯度无需再次求导，按随机种子保存待平均值。
                model_gradients.append(image_gradient.detach())
                # 保存各损失分量的普通数值用于日志。
                for name, value in detached_loss_values(components).items():
                    loss_sums[name] += value
                loss_sums["weighted_total"] += float(expert_loss.detach().item())
                expert_count += 1
                # 尽快释放专家中间特征。
                del real_output, synthetic_output

            # 同构 IDM 不做架构梯度再平衡，各随机模型等权平均。
            combined_gradient = torch.stack(model_gradients, dim=0).mean(dim=0)
            # 把合成图像梯度穿过冻结 VAE/DDIM 反传到本类 z_T。
            generated_pixels.backward(combined_gradient / float(bundle.num_classes))

            # z_T 的均值/标准差先验使用解析梯度，避免保留第二份 DDIM 图。
            prior_weight = float(profile.get("latent_prior", 0.0))
            if prior_weight > 0.0:
                # 先验统计不需要通过当前生成图求导。
                detached_latents = class_latents.detach()
                element_count = float(detached_latents.numel())
                latent_mean = detached_latents.mean()
                centered = detached_latents - latent_mean
                latent_std = centered.square().mean().add(1.0e-8).sqrt()
                # 目标是均值 0、标准差 1，并对类别数取平均。
                prior_gradient = prior_weight / float(bundle.num_classes) * (
                    2.0 * latent_mean / element_count
                    + 2.0
                    * (latent_std - 1.0)
                    * centered
                    / (element_count * latent_std)
                )
                # generated_pixels.backward 已经通常创建整张 latents.grad；这里兼容零损失情况。
                if latent_set.latents.grad is None:
                    latent_set.latents.grad = torch.zeros_like(latent_set.latents)
                # 只把先验梯度累加到本轮选中的隐变量位置。
                latent_set.latents.grad.index_add_(0, selected_indices, prior_gradient)
            # 保存本轮隐变量值用于诊断。
            selected_latent_values.append(class_latents.detach())
            # 主动释放本类生成图。
            del generated_pixels, denoised_latents

        # 可选裁剪 z_T 总梯度，防止在线随机模型产生极端步长。
        clip_norm = float(settings.get("gradient_clip_norm", 0.0))
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([latent_set.latents], clip_norm)
        # 第二步：只更新 z_T。
        optimizer.step()
        # 第三步：按 IDM 顺序在合成更新之后，用真实数据训练刚才抽到的模型；
        # 随后按固定周期注入全新随机模型，并在达到上限时 FIFO 淘汰。
        queue_diagnostics = queue.advance(
            iteration,
            guidance_members,
            pool,
            augment,
        )

        # 平均专家损失并合并队列、梯度与隐变量诊断。
        averaged_losses = {
            f"loss/{name}": value / max(1, expert_count)
            for name, value in loss_sums.items()
        }
        # 在线目标有噪声，因此额外记录总损失 EMA 供可选早停。
        smoothed_total = loss_smoother.update(averaged_losses.get("loss/weighted_total", 0.0))
        last_diagnostics = {
            **averaged_losses,
            **queue_diagnostics,
            "loss/smoothed_total": float(smoothed_total),
            "latent/mean": float(torch.cat(selected_latent_values).mean().item()),
            "latent/std": float(
                torch.cat(selected_latent_values).std(unbiased=False).item()
            ),
        }
        if device.type == "cuda":
            gibibyte = float(1024**3)
            last_diagnostics.update(
                {
                    "cuda/allocated_gib": torch.cuda.memory_allocated(device) / gibibyte,
                    "cuda/reserved_gib": torch.cuda.memory_reserved(device) / gibibyte,
                    "cuda/peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / gibibyte,
                }
            )
        last_diagnostics["time/iteration_seconds"] = float(
            time.perf_counter() - iteration_started
        )
        # 根据平滑总损失判断整个 IPC 是否达到可选早停条件。
        should_stop = early_stopping.update(float(smoothed_total), iteration)

        # 按迭代间隔保存轻量类别预览；0 表示关闭。
        preview_interval = int(settings.get("preview_interval_iterations", 0))
        if preview_interval > 0 and iteration % preview_interval == 0:
            preview_path = _save_preview(
                config,
                directory,
                iteration,
                latent_set,
                labels,
                autoencoder,
                diffusion,
                scale,
                int(ipc),
                bundle.num_classes,
            )
            logger.info(
                "IPC=%d iter=%d 已保存完整预览：%d 类×%d IPC -> %s",
                ipc,
                iteration,
                bundle.num_classes,
                int(ipc),
                preview_path,
            )

        # 每 1000 步等长周期保存一个不会被覆盖、可独立评估的完整快照。
        snapshot_interval = int(settings.get("snapshot_interval_iterations", 0))
        should_snapshot = (
            snapshot_interval > 0 and iteration % snapshot_interval == 0
        )
        snapshot_directory = (
            directory / "snapshots" / f"iteration_{int(iteration):06d}"
            if should_snapshot
            else None
        )

        # 定期、最终迭代、长周期快照或早停时保存完整断点。
        checkpoint_interval = int(settings.get("checkpoint_interval_iterations", 50))
        if (
            iteration % checkpoint_interval == 0
            or iteration == target_iterations
            or should_snapshot
            or should_stop
        ):
            _save_condensation(
                directory,
                iteration,
                latent_set,
                optimizer,
                queue,
                pool,
                early_stopping,
                loss_smoother,
                int(ipc),
                bundle.class_names,
                last_diagnostics,
                (
                    snapshot_directory / "checkpoint.pt"
                    if snapshot_directory is not None
                    else None
                ),
            )
        if snapshot_directory is not None:
            snapshot_dataset = _export(
                config,
                snapshot_directory,
                latent_set,
                labels,
                autoencoder,
                diffusion,
                scale,
                int(ipc),
                bundle.class_names,
            )
            logger.info(
                "IPC=%d iter=%d 已保存可评估快照：checkpoint=%s dataset=%s",
                ipc,
                iteration,
                snapshot_directory / "checkpoint.pt",
                snapshot_dataset,
            )

        # 按配置输出一次紧凑日志。
        log_interval = int(settings.get("log_interval_iterations", 10))
        if iteration == 1 or iteration % log_interval == 0 or should_stop:
            logger.info(
                "IPC=%d iter=%d/%d total=%.6f smooth=%.6f mean=%.6f "
                "topo(s/m/d)=%.6f/%.6f/%.6f z=%.4f±%.4f "
                "gpu=%.2f/%.2fG peak=%.2fG time=%.1fs queue=%s",
                ipc,
                iteration,
                target_iterations,
                last_diagnostics.get("loss/weighted_total", 0.0),
                last_diagnostics.get("loss/smoothed_total", 0.0),
                last_diagnostics.get("loss/mean", 0.0),
                last_diagnostics.get("loss/topology_shallow", 0.0),
                last_diagnostics.get("loss/topology_middle", 0.0),
                last_diagnostics.get("loss/topology_deep", 0.0),
                last_diagnostics["latent/mean"],
                last_diagnostics["latent/std"],
                last_diagnostics.get("cuda/allocated_gib", 0.0),
                last_diagnostics.get("cuda/reserved_gib", 0.0),
                last_diagnostics.get("cuda/peak_allocated_gib", 0.0),
                last_diagnostics.get("time/iteration_seconds", 0.0),
                queue.summary(),
            )
        # 满足早停后结束当前 IPC，但不影响其他 IPC 独立运行。
        if should_stop:
            logger.info(
                "IPC=%d 早停：连续 %d 次检查无改善，最佳平滑损失=%.6f@%d",
                ipc,
                early_stopping.bad_checks,
                float(early_stopping.best or 0.0),
                early_stopping.best_progress,
            )
            break

    # 若从恰好位于快照边界的旧断点恢复且没有新增迭代，补建该步快照。
    snapshot_interval = int(settings.get("snapshot_interval_iterations", 0))
    final_snapshot_directory = (
        directory / "snapshots" / f"iteration_{int(current_iteration):06d}"
        if snapshot_interval > 0
        and current_iteration > 0
        and current_iteration % snapshot_interval == 0
        else None
    )
    if (
        final_snapshot_directory is not None
        and not (final_snapshot_directory / "synthetic.pt").is_file()
    ):
        _save_condensation(
            directory,
            current_iteration,
            latent_set,
            optimizer,
            queue,
            pool,
            early_stopping,
            loss_smoother,
            int(ipc),
            bundle.class_names,
            last_diagnostics,
            final_snapshot_directory / "checkpoint.pt",
        )
        _export(
            config,
            final_snapshot_directory,
            latent_set,
            labels,
            autoencoder,
            diffusion,
            scale,
            int(ipc),
            bundle.class_names,
        )
        logger.info(
            "IPC=%d 从已有断点补建可评估快照：%s",
            ipc,
            final_snapshot_directory,
        )

    # 无论固定迭代还是早停，最终都用全部 z_T 导出完整合成集。
    dataset_path = _export(
        config,
        directory,
        latent_set,
        labels,
        autoencoder,
        diffusion,
        scale,
        int(ipc),
        bundle.class_names,
    )
    # 返回管线汇总需要的纯 Python 信息。
    return {
        "ipc": int(ipc),
        "iterations": int(current_iteration),
        "stopped_early": bool(current_iteration < target_iterations),
        "dataset": str(dataset_path.resolve()),
        "autoencoder_checkpoint": str(autoencoder_path),
        "diffusion_checkpoint": str(diffusion_path),
        "queue": queue.summary(),
        "diagnostics": last_diagnostics,
    }


def run(
    config: Mapping[str, Any],
    selected_ipcs: list[int] | None = None,
) -> dict[str, Any]:
    """依次运行选定 IPC，并写入主方法 summary.json。"""

    # 所有 IPC 共享 condensed 根目录，各自使用 ipc_N 子目录。
    directory = stage_dir(config, "condensed")
    # 阶段日志写入固定目录。
    logger = get_stage_logger("condense", directory)
    # 解析 auto/cpu/cuda 设备。
    device = resolve_device(config)
    logger.info("启动 condense：device=%s output=%s", device, directory.resolve())
    # 固定数据、初始化和全局随机状态。
    seed_everything(
        int(config["project"]["seed"]),
        bool(config["project"].get("deterministic", True)),
    )
    # 读取统一数据适配器和类别映射。
    # 凝聚统一在 GPU 批量增强；Dataset 保持确定性，才能安全缓存解码后的基础图像。
    bundle = build_data_bundle(config, train_augmentation={"enabled": False})
    logger.info(
        "数据集已就绪：classes=%d train=%d val=%d test=%d",
        bundle.num_classes,
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
    )
    # 命令行 --ipc 优先于阶段文件默认列表。
    requested = selected_ipcs or [
        int(value) for value in config["condensation"]["ipc_values"]
    ]
    # 每个 IPC 独立优化与恢复。
    results = [
        _run_ipc(config, int(ipc), bundle, device, logger)
        for ipc in requested
    ]
    # 主方法名称明确包含同构 IDM 动态模型池、分层拓扑与隐空间扩散。
    summary = {
        "method": CONDENSATION_METHOD,
        "class_names": bundle.class_names,
        "runs": {str(item["ipc"]): item for item in results},
    }
    # JSON 汇总不含模型权重，适合快速检查实验状态。
    atomic_write_json(summary, directory / "summary.json")
    return summary


if __name__ == "__main__":
    # 支持直接运行本文件进行调试，正式使用仍推荐根目录 run_pipeline.py。
    from Core.config import load_config

    run(load_config())
