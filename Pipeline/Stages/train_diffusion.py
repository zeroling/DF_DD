"""在冻结的 MONAI VAE 隐空间中从零训练共享类别条件扩散模型。

所有类别共用一个 U-Net，通过类别嵌入区分标签，并额外学习空类别用于 CFG。VAE 在本
阶段完全冻结。扩散断点保存当前 U-Net、EMA、优化器、随机状态和早停历史；蒸馏阶段
读取 EMA 权重并冻结，通过可微 DDIM 把 IDM/RBF 梯度传回待优化 ``z_T``。
"""

from __future__ import annotations

import math  # 计算余弦学习率倍率。
import time  # 记录纯训练阶段耗时与吞吐率。
from typing import Any, Mapping  # 合并配置与断点映射类型。

import torch  # 隐变量、模型训练、优化器和混合精度。
from torchvision.utils import save_image  # 定期保存类别条件采样预览。

from Core.augmentations import from_generator_range, to_generator_range
from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_progress,
    find_latest_checkpoint,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_rng_state,
)
from Core.config import stage_dir
from Core.data import build_data_bundle, build_loader, unpack_batch
from Core.io_utils import atomic_write_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, make_grad_scaler, resolve_device
from Core.seed import seed_everything
from Core.training import EarlyStopping, advance_scheduler_to
from Net.Generative.diffusion import (
    build_ddim_scheduler,
    build_ddpm_scheduler,
    decode_latents,
    differentiable_ddim_sample,
    diffusion_training_loss,
    encode_latents,
)
from Net.Generative.ema import ExponentialMovingAverage
from Net.Generative.models import (
    build_autoencoder,
    build_diffusion_unet,
    freeze_module,
    latent_spatial_size,
)


def _cosine_scheduler(optimizer, epochs: int, minimum_lr: float):
    """创建按当前 YAML 总轮数定义的余弦学习率调度器。"""

    # 以第一个参数组学习率为基准；本阶段 U-Net 使用单一参数组。
    base_lr = float(optimizer.param_groups[0]["lr"])
    # 最低倍率限制到不高于 1，避免“最小学习率”反而抬高基础学习率。
    minimum_factor = min(1.0, float(minimum_lr) / max(base_lr, 1.0e-12))

    def factor(epoch: int) -> float:
        """把完成轮数映射为有下界的余弦倍率。"""

        # 续训进度超过新总轮数时固定在曲线终点。
        progress = min(1.0, max(0.0, epoch / max(1, int(epochs))))
        return minimum_factor + (1.0 - minimum_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_autoencoder(config, device):
    """加载 VAE 阶段最新断点，验证 latent_scale 后严格冻结模型。"""

    # create=False 防止缺少 VAE 时创建空目录掩盖阶段未运行错误。
    directory = stage_dir(config, "autoencoder", create=False)
    # 用户只需给阶段文件夹，内部自动寻找 last/最大进度权重。
    path = find_latest_checkpoint(directory)
    if path is None:
        raise FileNotFoundError("训练扩散模型前必须完成 train_autoencoder 阶段")
    # 按扩散训练设备加载，减少二次复制。
    payload = load_checkpoint(path, device)
    # latent_scale 是扩散空间与原始 VAE 后验之间的必要缩放常数。
    scale = payload.get("latent_scale")
    if scale is None or float(scale) <= 0:
        raise ValueError(f"Autoencoder 断点缺少有效 latent_scale：{path}")
    # 当前 autoencoder.yaml 重建结构，严格载入确保完全兼容。
    model = build_autoencoder(config).to(device)
    model.load_state_dict(model_state_from_checkpoint(payload), strict=True)
    # eval+requires_grad(False)；本阶段永远不更新 VAE。
    return freeze_module(model), float(scale), path


def _restore(directory, model, optimizer, scaler, ema, device):
    """恢复扩散 U-Net、优化器、AMP、EMA、随机状态和阶段进度。"""

    path = find_latest_checkpoint(directory)
    # 首次训练返回统一空状态。
    if path is None:
        return 0, None, None
    # 不验证配置/数据哈希，只让模型 strict load 检查结构。
    payload = load_checkpoint(path, device)
    model.load_state_dict(model_state_from_checkpoint(payload), strict=True)
    # 兼容只有模型权重的旧断点，附加状态存在时才恢复。
    if isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    if isinstance(payload.get("scaler"), Mapping):
        scaler.load_state_dict(payload["scaler"])
    if isinstance(payload.get("ema"), Mapping):
        ema.load_state_dict(payload["ema"])
    # 随机状态恢复后，训练噪声、时间步和数据顺序从断点继续。
    restore_rng_state(payload.get("rng_state"))
    return checkpoint_progress(payload, path), payload, path


def _save(
    path,
    epoch,
    global_step,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    scale,
    metrics,
    class_names,
    early_stopping,
):
    """原子保存扩散模型、EMA、优化状态、验证指标和早停历史。"""

    # 使用同目录临时文件原子替换，避免写入中断损坏 last。
    atomic_torch_save(
        {
            # epoch 是阶段主进度，global_step 仅用于日志和精细统计。
            "epoch": int(epoch),
            "global_step": int(global_step),
            # 当前训练 U-Net 与优化状态。
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            # EMA 是后续隐空间蒸馏真正使用的生成权重。
            "ema": ema.state_dict(),
            # 与 VAE 断点一致保存，蒸馏阶段无需跨文件猜测。
            "latent_scale": float(scale),
            # 最近训练/验证指标及稳定类别顺序。
            "metrics": dict(metrics),
            "class_names": list(class_names),
            # best/bad_checks 让续训延续连续无改善计数。
            "early_stopping": early_stopping.state_dict(),
            "rng_state": capture_rng_state(),
        },
        path,
    )


@torch.no_grad()
def _preview(
    config,
    model,
    autoencoder,
    scale,
    num_classes,
    device,
    epoch,
    directory,
    variant: str,
    inference_steps: int,
):
    """从纯高斯隐变量生成每类一张确定性 DDIM 预览。"""

    # 预览步数和 CFG 强度来自 diffusion.yaml，不影响训练损失。
    settings = config["diffusion"]
    # 最多预览 8 类，避免类别很多时每次采样耗时过长。
    count = min(8, int(num_classes))
    # 依次选择类别 0...count-1。
    labels = torch.arange(count, device=device, dtype=torch.long)
    # 隐空间尺寸由输入尺寸和 VAE 下采样层数自动推导。
    latent_height, latent_width = latent_spatial_size(config)
    # 固定预览噪声；raw 与 EMA、不同 epoch 都从完全相同的 z_T 出发。
    preview_generator = torch.Generator(device=device).manual_seed(
        int(settings.get("preview_seed", config["project"].get("seed", 0)))
    )
    initial = torch.randn(
        count,
        int(config["autoencoder"]["latent_channels"]),
        latent_height,
        latent_width,
        device=device,
        generator=preview_generator,
    )
    # 预览使用独立 DDIM scheduler，避免修改训练 DDPM 或其他采样器状态。
    scheduler = build_ddim_scheduler(config)
    # no_grad 装饰器使同一可微函数在此不保留反向图。
    with autocast_context(config, device):
        latents = differentiable_ddim_sample(
            model,
            scheduler,
            initial,
            labels,
            int(num_classes),
            int(inference_steps),
            float(settings.get("preview_guidance_scale", 2.0)),
        )
        # 撤销 latent scale，经冻结 VAE 解码，再从 [-1,1] 转到 [0,1]。
        images = from_generator_range(decode_latents(autoencoder, latents, scale)).float()
    # 文件名区分即时训练权重和 EMA 权重，便于公平对照与回滚。
    save_image(images, directory / f"preview_{variant}_epoch_{epoch:04d}.png", nrow=count)


@torch.no_grad()
def _validation(
    config,
    model,
    autoencoder,
    latent_scale,
    loader,
    noise_scheduler,
    num_classes,
    device,
) -> float:
    """在真实验证图像隐变量上估计平均去噪损失，测试集不会参与。"""

    # 评估模式关闭随机失活；扩散噪声由下方独立固定 RNG 采样。
    model.eval()
    # 独立生成器让每个 epoch 使用相同验证噪声/时间步，同时不改变训练 RNG。
    validation_generator = torch.Generator(device=device).manual_seed(
        int(config["diffusion"].get("validation_seed", config["project"].get("seed", 0)))
    )
    # 可由配置限制验证 batch 数，以控制大型医学数据集的每轮开销。
    maximum_batches = max(0, int(config["diffusion"].get("validation_batches", 0)))
    # 按样本数累计损失。
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    sample_count = 0
    # 顺序遍历真实验证集。
    for batch_index, batch in enumerate(loader):
        # validation_batches=0 表示使用完整验证集。
        if maximum_batches > 0 and batch_index >= maximum_batches:
            break
        # 统一解包图像和类别标签。
        images, labels = unpack_batch(batch)
        images = to_generator_range(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True).long()
        # 验证使用后验均值，减少 VAE 采样噪声。
        with autocast_context(config, device):
            latents = encode_latents(autoencoder, images, latent_scale, sample=False)
            # class_dropout=0 保证验证的是正确类别条件下的噪声预测能力。
            loss = diffusion_training_loss(
                model,
                noise_scheduler,
                latents,
                labels,
                int(num_classes),
                0.0,
                float(config["diffusion"].get("min_snr_gamma", 5.0)),
                generator=validation_generator,
            )
        # 在 GPU 上累计，避免每个验证 batch 都用 .item() 强制同步。
        loss_sum.add_(loss.detach().float(), alpha=labels.numel())
        sample_count += labels.numel()
    # 空验证集意味着数据划分存在错误，不能用 0 假装最佳指标。
    if sample_count == 0:
        raise ValueError("扩散模型验证集为空")
    return float((loss_sum / sample_count).item())


def run(config: Mapping[str, Any], selected_ipcs: list[int] | None = None) -> dict[str, Any]:
    """训练或续训类别条件 LDM，并保存供蒸馏使用的 EMA 权重。"""

    # 扩散模型对所有 IPC 共用，统一阶段签名中的 selected_ipcs 在此不使用。
    del selected_ipcs
    # diffusion.yaml 合并后的唯一阶段参数节点。
    settings = config["diffusion"]
    # 固定目录使再次执行时自动恢复最新断点。
    directory = stage_dir(config, "diffusion")
    logger = get_stage_logger("diffusion", directory)
    # 解析全局设备并固定全部随机源。
    device = resolve_device(config)
    seed_everything(int(config["project"]["seed"]), bool(config["project"].get("deterministic", True)))
    # 数据层与疾病/文件格式解耦，输出统一 [0,1] RGB 张量。
    # 扩散模型可独立关闭增强；没有独立配置时仍兼容旧行为并使用全局设置。
    bundle = build_data_bundle(config, train_augmentation=settings.get("augmentation"))
    # 真实训练集使用类别平衡抽样，防止扩散模型忽略少数类别。
    train_loader = build_loader(bundle.train, config, True, int(settings["batch_size"]), balanced=True)
    # 真实验证集只用于扩散去噪损失早停，不参与参数更新。
    val_loader = build_loader(bundle.val, config, False, int(settings["batch_size"]))
    # 读取并冻结刚训练的 VAE，同时获得必要 latent scale。
    autoencoder, latent_scale, autoencoder_path = _load_autoencoder(config, device)
    # 一个共享 U-Net 从随机初始化开始，类别总数决定嵌入表大小 C+1。
    model = build_diffusion_unet(config, bundle.num_classes).to(device)
    # AdamW 学习率和权重衰减均可在阶段 YAML 调节。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1.0e-5)),
    )
    # 目标轮数是续训上限，而不是每次追加轮数。
    target_epochs = int(settings["epochs"])
    # 当前 YAML 重新定义完整余弦曲线。
    scheduler = _cosine_scheduler(optimizer, target_epochs, float(settings["min_learning_rate"]))
    # DDPM scheduler 只负责训练前向加噪和目标公式。
    noise_scheduler = build_ddpm_scheduler(config)
    # FP16 动态缩放；BF16/CPU 下使用禁用的统一接口。
    scaler = make_grad_scaler(config, device)
    # EMA 初始等于随机模型当前权重，随后每个优化 step 更新。
    ema = ExponentialMovingAverage(model, float(settings.get("ema_decay", 0.999)))
    # 自动读取 last/进度最大断点，不检查哈希。
    completed, resume_payload, resume_path = _restore(directory, model, optimizer, scaler, ema, device)
    # 当前配置优先：允许恢复后修改学习率与 EMA 衰减率。
    for group in optimizer.param_groups:
        group["lr"] = float(settings["learning_rate"])
    ema.decay = float(settings.get("ema_decay", 0.999))
    # 使用新目标轮数的调度器推进到已完成位置。
    if completed > 0:
        advance_scheduler_to(scheduler, completed)
        logger.info("从 %s 恢复扩散模型：%d/%d epoch", resume_path, completed, target_epochs)
    # 最近指标和全局更新步数从断点继续；旧权重缺失时从空/0 开始。
    last_metrics = dict((resume_payload or {}).get("metrics", {}))
    global_step = int((resume_payload or {}).get("global_step", 0))
    # 根据 validation_diffusion_loss 的停滞情况决定是否提前结束。
    early_settings = settings.get("early_stopping", {})
    early_stopping = EarlyStopping.from_config(
        early_settings,
        interval_key="check_interval_epochs",
        minimum_key="minimum_epochs",
    )
    early_stopping.load_state_dict(
        (resume_payload or {}).get("early_stopping"),
        reset=bool(early_settings.get("reset_on_resume", False)),
    )
    # 记录实际训练到的轮数，早停时可能小于 target_epochs。
    current_epoch = int(completed)
    # 从下一个未完成 epoch 训练；completed>=target 时循环为空并直接总结。
    for epoch in range(completed + 1, target_epochs + 1):
        current_epoch = int(epoch)
        # 验证函数会切 eval，每轮开始显式恢复训练模式。
        model.train()
        # 训练损失按样本数而非 batch 数累积。
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        sample_count = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        for batch in train_loader:
            # 解包并把真实图像/标签迁移到训练设备。
            images, labels = unpack_batch(batch)
            images = to_generator_range(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True).long()
            # VAE 已冻结，编码不需要保留计算图；训练时采样后验而非只取均值。
            with torch.no_grad(), autocast_context(config, device):
                latents = encode_latents(autoencoder, images, latent_scale, sample=True)
            # 只更新 U-Net 参数。
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                # 内部独立采样噪声/时间步，并按 class_dropout 训练 CFG 空类别。
                loss = diffusion_training_loss(
                    model,
                    noise_scheduler,
                    latents,
                    labels,
                    bundle.num_classes,
                    float(settings.get("class_dropout", 0.1)),
                    float(settings.get("min_snr_gamma", 5.0)),
                )
            # 混合精度缩放反传。
            scaler.scale(loss).backward()
            # 正值启用全模型梯度范数裁剪；裁剪前必须先 unscale。
            if float(settings.get("gradient_clip_norm", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip_norm", 1.0)))
            # 更新训练模型、动态缩放器和 EMA 影子权重。
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            # 全局 step 每个成功 batch 增加一次。
            global_step += 1
            # 按 batch 样本数加权累积，兼容最后一个小批次。
            # 在 GPU 上累计，整个 epoch 只在末尾同步一次。
            loss_sum.add_(loss.detach().float(), alpha=labels.numel())
            sample_count += labels.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = max(1.0e-9, time.perf_counter() - train_started)
        train_loss = float((loss_sum / max(1, sample_count)).item())
        samples_per_second = sample_count / train_seconds
        # 每个完整 epoch 按闭式余弦函数设置学习率，不调用会误报顺序警告的 scheduler.step。
        advance_scheduler_to(scheduler, epoch)
        # 每轮在真实验证隐变量上估计去噪损失。
        validation_loss = _validation(
            config,
            model,
            autoencoder,
            latent_scale,
            val_loader,
            noise_scheduler,
            bundle.num_classes,
            device,
        )
        last_metrics = {
            "min_snr_diffusion_loss": train_loss,
            "validation_diffusion_loss": float(validation_loss),
            "global_step": global_step,
            "train_seconds": float(train_seconds),
            "train_samples_per_second": float(samples_per_second),
        }
        # 验证损失越小越好。
        should_stop = early_stopping.update(validation_loss, epoch)
        # checkpoint_last 始终覆盖同一文件，断点目录不会堆积数百份轨迹权重。
        payload_path = directory / "checkpoint_last.pt"
        # 保存频率、预览频率和日志频率彼此独立可调。
        checkpoint_interval = int(settings.get("checkpoint_interval_epochs", 1))
        if epoch % checkpoint_interval == 0 or epoch == target_epochs or should_stop:
            _save(
                payload_path,
                epoch,
                global_step,
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                latent_scale,
                last_metrics,
                bundle.class_names,
                early_stopping,
            )
        preview_interval = int(settings.get("preview_interval_epochs", 10))
        if epoch % preview_interval == 0 or epoch == target_epochs or should_stop:
            # 常规预览使用较少步数；训练最终轮或早停轮使用完整采样步数。
            is_final_preview = epoch == target_epochs or should_stop
            requested_preview_steps = int(
                settings.get("final_preview_steps", settings.get("preview_steps", 250))
                if is_final_preview
                else settings.get("preview_steps", 250)
            )
            # 微型实验或调参时 train_timesteps 可能小于正式配置的预览步数；
            # MONAI 要求 DDIM 推理步数不能超过训练 scheduler 的总时间步。
            preview_steps = min(
                max(1, requested_preview_steps),
                int(settings.get("train_timesteps", 1000)),
            )
            # 当前训练权重与 EMA 权重使用同一批固定噪声，消除随机样本造成的误判。
            model.eval()
            _preview(
                config,
                model,
                autoencoder,
                latent_scale,
                bundle.num_classes,
                device,
                epoch,
                directory,
                "raw",
                preview_steps,
            )
            # 临时模型初始化会消费 RNG；预览完成后恢复，保证它不改变后续训练轨迹。
            preview_rng_state = capture_rng_state()
            ema_model = None
            try:
                ema_model = build_diffusion_unet(config, bundle.num_classes).to(device)
                ema.copy_to(ema_model)
                ema_model.eval()
                _preview(
                    config,
                    ema_model,
                    autoencoder,
                    latent_scale,
                    bundle.num_classes,
                    device,
                    epoch,
                    directory,
                    "ema",
                    preview_steps,
                )
            finally:
                del ema_model
                restore_rng_state(preview_rng_state)
        log_interval = int(settings.get("log_interval_epochs", 1))
        if epoch == 1 or epoch % log_interval == 0 or should_stop:
            logger.info(
                "LDM epoch=%d/%d train_loss=%.6f val_loss=%.6f step=%d "
                "throughput=%.1f img/s train_time=%.1fs latent_scale=%.6f",
                epoch,
                target_epochs,
                last_metrics["min_snr_diffusion_loss"],
                last_metrics["validation_diffusion_loss"],
                global_step,
                last_metrics["train_samples_per_second"],
                last_metrics["train_seconds"],
                latent_scale,
            )
        # 早停前已经保存 checkpoint_last 和预览图。
        if should_stop:
            logger.info(
                "Diffusion 早停：最佳 validation_diffusion_loss=%.6f@epoch %d",
                float(early_stopping.best or 0.0),
                early_stopping.best_progress,
            )
            break
    # 返回 JSON 可序列化摘要；大型模型只通过 checkpoint 路径引用。
    summary = {
        "complete": current_epoch >= target_epochs,
        "stopped_early": current_epoch < target_epochs,
        "epochs": current_epoch,
        "global_step": global_step,
        "latent_scale": latent_scale,
        "autoencoder_checkpoint": str(autoencoder_path),
        "checkpoint": str((directory / "checkpoint_last.pt").resolve()),
        "metrics": last_metrics,
    }
    # 固定 summary 文件供主流程和人工检查。
    atomic_write_json(summary, directory / "summary.json")
    return summary


if __name__ == "__main__":
    # 允许直接调试本阶段；推荐入口仍是根目录 run_pipeline.py。
    from Core.config import load_config

    run(load_config())
