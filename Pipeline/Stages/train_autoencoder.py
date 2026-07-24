"""从零训练 MONAI AutoencoderKL，并估计扩散隐变量缩放因子。

VAE 只学习当前真实训练集，不加载 MONAI 或其他项目权重。生成器目标由 L1、SSIM、KL
和可选 PatchGAN 组成；验证 L1 支持按配置早停。断点保存当前 epoch、两个优化器、
调度器、混合精度、随机状态和早停历史，恢复时只检查结构兼容，不比较配置哈希。
"""

from __future__ import annotations

import math  # 计算余弦学习率曲线。
from pathlib import Path  # 阶段断点路径类型。
from typing import Any, Mapping  # 合并配置与断点字典类型。

import torch  # 模型训练、优化器、调度器和张量运算。
import torch.nn.functional as F  # L1、hinge 判别器损失。
from torchvision.utils import save_image  # 保存真实/重建图像预览网格。

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
from Core.config import image_size, stage_dir
from Core.data import build_data_bundle, build_loader, unpack_batch
from Core.io_utils import atomic_write_json
from Core.logging_utils import get_stage_logger
from Core.run_context import autocast_context, make_grad_scaler, resolve_device
from Core.seed import seed_everything
from Core.training import EarlyStopping, advance_scheduler_to
from Net.Generative.diffusion import (
    bound_reconstruction,
    estimate_scale_factor,
    gaussian_kl,
)
from Net.Generative.models import build_autoencoder, build_patch_discriminator


def _cosine_scheduler(optimizer, target_epochs: int, minimum_lr: float):
    """构造从当前基础学习率平滑衰减到 ``minimum_lr`` 的 LambdaLR。"""

    # 第一个参数组定义基础学习率；本阶段所有组使用同一倍率。
    base_lr = float(optimizer.param_groups[0]["lr"])
    # minimum_factor 是最终学习率与基础学习率之比，并限制不超过 1。
    minimum_factor = min(1.0, float(minimum_lr) / max(base_lr, 1.0e-12))

    def factor(epoch: int) -> float:
        """把已完成 epoch 映射为 [minimum_factor,1] 的余弦倍率。"""

        # clamp 到 [0,1]，即使续训进度超过新目标也不会让曲线反向增长。
        progress = min(1.0, max(0.0, epoch / max(1, int(target_epochs))))
        return minimum_factor + (1.0 - minimum_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # LambdaLR 不持有配置哈希，恢复时可以按当前 target_epochs 重新推进。
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _kl_weight_for_epoch(settings: Mapping[str, Any], epoch: int) -> float:
    """返回当前 epoch 真正乘到 mean-KL 上的线性预热权重。

    本项目的 :func:`gaussian_kl` 对 batch、通道和空间维统一取均值，
    因此权重不会因图像尺寸或隐空间元素数变化而自动放大。
    预热在第 1 轮使用 ``kl_warmup_start_weight``，在指定的最后一个
    warmup epoch 精确到达 ``kl_weight``；之后始终保持目标值。
    """

    # kl_weight 是预热完成后的最终系数。
    target_weight = float(settings.get("kl_weight", 1.0e-3))
    # 起始权重默认为 0，先让编解码器建立基本重建能力。
    start_weight = float(settings.get("kl_warmup_start_weight", 0.0))
    # 0 或 1 表示不做多轮预热，从第 1 轮直接使用目标权重。
    warmup_epochs = int(settings.get("kl_warmup_epochs", 0))
    if warmup_epochs <= 1:
        return target_weight
    # epoch=1 时 progress=0；epoch=warmup_epochs 时 progress=1。
    progress = (int(epoch) - 1) / max(1, warmup_epochs - 1)
    progress = min(1.0, max(0.0, float(progress)))
    return start_weight + (target_weight - start_weight) * progress


def _set_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    """切换可选判别器参数梯度，生成器步骤中避免无用梯度。"""

    # PatchGAN 关闭时 module=None，直接跳过。
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(bool(enabled))


def _restore(
    directory: Path,
    autoencoder,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    scaler,
    device,
) -> tuple[int, dict | None, Path | None]:
    """从阶段目录恢复 VAE、可选 PatchGAN 及训练状态。"""

    # last 优先，否则选择进度最大且可读取的 pt/pth。
    path = find_latest_checkpoint(directory)
    # 没有断点表示从第 1 个 epoch 开始。
    if path is None:
        return 0, None, None
    # 映射到目标设备后读取完整 payload，不校验配置/数据哈希。
    payload = load_checkpoint(path, device)
    # VAE 结构必须与当前 YAML 完全兼容。
    autoencoder.load_state_dict(model_state_from_checkpoint(payload), strict=True)
    # 附加状态存在时恢复；兼容旧的纯模型权重。
    if isinstance(payload.get("optimizer"), Mapping):
        generator_optimizer.load_state_dict(payload["optimizer"])
    if isinstance(payload.get("scaler"), Mapping):
        scaler.load_state_dict(payload["scaler"])
    if discriminator is not None and isinstance(payload.get("discriminator"), Mapping):
        discriminator.load_state_dict(payload["discriminator"], strict=True)
    if discriminator_optimizer is not None and isinstance(payload.get("discriminator_optimizer"), Mapping):
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
    # 恢复随机状态，让数据增强和后验采样从断点位置继续。
    restore_rng_state(payload.get("rng_state"))
    # 进度读取 epoch 优先于 global_step。
    return checkpoint_progress(payload, path), payload, path


def _save(
    path: Path,
    epoch: int,
    autoencoder,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    generator_scheduler,
    discriminator_scheduler,
    scaler,
    metrics: Mapping[str, float],
    latent_scale: float | None,
    early_stopping: EarlyStopping,
) -> None:
    """保存 VAE、可选判别器、优化器、指标与早停历史。"""

    # 断点不包含配置哈希；网络结构兼容即可恢复。
    payload: dict[str, Any] = {
        # epoch 表示已经完整训练并验证完的轮数。
        "epoch": int(epoch),
        # VAE 模型与生成器优化状态。
        "model": autoencoder.state_dict(),
        "optimizer": generator_optimizer.state_dict(),
        # 调度器保存当前位置，但恢复后允许当前 YAML 重新推进/覆盖学习率。
        "scheduler": generator_scheduler.state_dict(),
        # FP16 GradScaler；BF16/CPU 时也保存统一的禁用状态。
        "scaler": scaler.state_dict(),
        # 最近一次训练/验证标量，便于无需重新评估就生成 summary。
        "metrics": dict(metrics),
        # 只有最终轮/早停才计算，扩散阶段必须读取到非空值。
        "latent_scale": latent_scale,
        # 保存 best/bad_checks，使续训延续“连续无改善”计数。
        "early_stopping": early_stopping.state_dict(),
        # Python/NumPy/Torch CPU/CUDA 全部随机状态。
        "rng_state": capture_rng_state(),
    }
    # 只有启用 PatchGAN 时才增加判别器三项状态。
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        payload["discriminator_scheduler"] = discriminator_scheduler.state_dict()
    # 临时文件写完后原子替换，避免中断损坏 checkpoint_last。
    atomic_torch_save(payload, path)


def _validation(autoencoder, loader, device) -> dict[str, float]:
    """确定性验证 bounded/raw 重建，并报告常数基线与输出饱和程度。"""

    # 固定网络推理行为。
    autoencoder.eval()
    # 使用总绝对误差/总像素数，最后不足批次不会改变权重。
    l1_sum = 0.0
    raw_l1_sum = 0.0
    zero_l1_sum = 0.0
    raw_out_of_range = 0
    bounded_saturated = 0
    raw_minimum = math.inf
    raw_maximum = -math.inf
    count = 0
    with torch.no_grad():
        for batch in loader:
            # 验证只需要图像，不使用类别标签。
            images, _ = unpack_batch(batch)
            # 数据层 [0,1] 转 VAE [-1,1] 并移动设备。
            images = to_generator_range(images.to(device))
            # 验证必须使用后验均值做确定性重建。AutoencoderKL.forward
            # 会执行 z=mean+epsilon*sigma，若在此直接调用，同一权重每次得到的
            # validation_l1 都会不同，从而污染早停判断。
            posterior_mean, _ = autoencoder.encode(images)
            raw_reconstruction = autoencoder.decode(posterior_mean)
            reconstruction = bound_reconstruction(raw_reconstruction)
            # reduction=sum 后跨 batch 累积所有通道与像素。
            l1_sum += float(F.l1_loss(reconstruction, images, reduction="sum").item())
            raw_l1_sum += float(
                F.l1_loss(raw_reconstruction, images, reduction="sum").item()
            )
            # 全零输出对应原像素域的中灰图，是判断 AE 是否学到有效映射的简单基线。
            zero_l1_sum += float(images.abs().sum().item())
            raw_out_of_range += int((raw_reconstruction.abs() > 1.0).sum().item())
            bounded_saturated += int((reconstruction.abs() > 0.98).sum().item())
            raw_minimum = min(raw_minimum, float(raw_reconstruction.min().item()))
            raw_maximum = max(raw_maximum, float(raw_reconstruction.max().item()))
            count += images.numel()
    if count == 0:
        raise ValueError("Autoencoder 验证集为空")
    return {
        "l1": l1_sum / count,
        "raw_l1": raw_l1_sum / count,
        "zero_l1": zero_l1_sum / count,
        "raw_min": raw_minimum,
        "raw_max": raw_maximum,
        "raw_out_of_range_ratio": raw_out_of_range / count,
        "bounded_saturation_ratio": bounded_saturated / count,
    }


@torch.no_grad()
def _validation_preview(autoencoder, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """固定重建验证集首批样本，保证不同 epoch 的预览可以逐图比较。"""

    autoencoder.eval()
    try:
        batch = next(iter(loader))
    except StopIteration as error:
        raise ValueError("Autoencoder 验证集为空，无法保存预览") from error
    images, _ = unpack_batch(batch)
    images = to_generator_range(images.to(device))
    posterior_mean, _ = autoencoder.encode(images)
    reconstruction = bound_reconstruction(autoencoder.decode(posterior_mean))
    return images[:8], reconstruction[:8]


def _scale_batches(loader):
    """把训练 DataLoader 转成只产生 ``[-1,1]`` 图像的惰性迭代器。"""

    for batch in loader:
        images, _ = unpack_batch(batch)
        # estimate_scale_factor 自己负责设备迁移，这里仅转换数值范围。
        yield to_generator_range(images)


def run(config: Mapping[str, Any], selected_ipcs: list[int] | None = None) -> dict[str, Any]:
    """训练或从目录续训 VAE，并返回扩散阶段需要的 latent_scale。"""

    # VAE 对所有 IPC 共用，阶段入口接受统一参数但不使用其值。
    del selected_ipcs
    # 阶段结构/训练参数来自 autoencoder.yaml 合并节点。
    settings = config["autoencoder"]
    # 固定阶段目录保证再次运行能自动找到 checkpoint_last.pt。
    directory = stage_dir(config, "autoencoder")
    logger = get_stage_logger("autoencoder", directory)
    # 按全局 auto/cpu/cuda 配置选择设备。
    device = resolve_device(config)
    # 模型初始化、数据增强与后验采样都从同一实验种子开始。
    seed_everything(int(config["project"]["seed"]), bool(config["project"].get("deterministic", True)))
    # 通用数据适配器构建互斥的 train/val/test；本阶段只用前两个。
    bundle = build_data_bundle(config)
    # 训练 loader 使用类别平衡抽样，避免 VAE 主要重建多数类。
    train_loader = build_loader(bundle.train, config, True, int(settings["batch_size"]), balanced=True)
    # 验证 loader 不增强、不打乱、不丢最后一批。
    val_loader = build_loader(bundle.val, config, False, int(settings["batch_size"]))
    # MONAI AutoencoderKL 从随机权重创建并移动到设备。
    autoencoder = build_autoencoder(config).to(device)
    # SSIMLoss 只在实际运行该阶段时延迟导入 MONAI。
    from monai.losses import SSIMLoss

    # 输入范围是 [-1,1]，因此 data_range=2；小尺寸测试自动选择不超过图像的奇数窗口。
    ssim_window = min(11, min(image_size(config)))
    # SSIM 高斯窗口要求奇数；偶数尺寸向下取最近奇数。
    ssim_window = ssim_window if ssim_window % 2 else ssim_window - 1
    if ssim_window < 3:
        raise ValueError("Autoencoder SSIM 要求图像最短边至少为 3")
    ssim_criterion = SSIMLoss(
        spatial_dims=2, data_range=2.0, win_size=ssim_window
    )
    # PatchGAN 可通过 autoencoder.adversarial.enabled 完全关闭。
    discriminator = build_patch_discriminator(config)
    if discriminator is not None:
        discriminator = discriminator.to(device)
    # 生成器 AdamW 的学习率和权重衰减都可配置。
    generator_optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1.0e-5)),
    )
    # 只有启用 PatchGAN 才创建第二个优化器。
    discriminator_optimizer = (
        torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(settings["discriminator_learning_rate"]),
            weight_decay=float(settings.get("discriminator_weight_decay", 1.0e-5)),
        )
        if discriminator is not None
        else None
    )
    # 当前 YAML 的 epochs 是续训目标：断点 78、目标 100 时只执行 79...100。
    target_epochs = int(settings["epochs"])
    # 两个优化器使用独立的余弦调度器，但共享最终 epoch。
    generator_scheduler = _cosine_scheduler(generator_optimizer, target_epochs, float(settings["min_learning_rate"]))
    discriminator_scheduler = (
        _cosine_scheduler(discriminator_optimizer, target_epochs, float(settings["min_learning_rate"]))
        if discriminator_optimizer is not None
        else None
    )
    # CUDA FP16 启用动态缩放，其余场景使用统一的禁用 scaler 接口。
    scaler = make_grad_scaler(config, device)
    # 自动读取目录最新权重；不比较任何哈希。
    completed, resume_payload, resume_path = _restore(
        directory,
        autoencoder,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        scaler,
        device,
    )
    # 当前配置优先于断点中的旧学习率，允许用户主动调 LR 后续训。
    for group in generator_optimizer.param_groups:
        group["lr"] = float(settings["learning_rate"])
    if discriminator_optimizer is not None:
        for group in discriminator_optimizer.param_groups:
            group["lr"] = float(settings["discriminator_learning_rate"])
    # 调度器按已完成轮数推进到当前位置，不依赖旧配置哈希。
    if completed > 0:
        advance_scheduler_to(generator_scheduler, completed)
        if discriminator_scheduler is not None:
            advance_scheduler_to(discriminator_scheduler, completed)
        logger.info("从 %s 恢复 AutoencoderKL：%d/%d epoch", resume_path, completed, target_epochs)

    # 对抗训练在 warmup 之后才启用，让重建分支先获得稳定输出。
    adversarial = settings.get("adversarial", {})
    warmup_epochs = int(adversarial.get("warmup_epochs", 10))
    # 续训时保留最近指标与可能已经估计的 latent_scale。
    last_metrics: dict[str, float] = dict((resume_payload or {}).get("metrics", {}))
    latent_scale = (resume_payload or {}).get("latent_scale")
    # 验证 L1 越小越好；当前阶段文件控制耐心、检查间隔和最低训练轮数。
    early_settings = settings.get("early_stopping", {})
    early_stopping = EarlyStopping.from_config(
        early_settings,
        interval_key="check_interval_epochs",
        minimum_key="minimum_epochs",
    )
    # 续训默认延续早停历史，也可以由 reset_on_resume 显式清空。
    early_stopping.load_state_dict(
        (resume_payload or {}).get("early_stopping"),
        reset=bool(early_settings.get("reset_on_resume", False)),
    )
    # 实际完成轮数可能小于目标轮数，因为早停会提前退出。
    current_epoch = int(completed)
    # Python range 从下一个未完成 epoch 开始；completed>=target 时循环自然为空。
    for epoch in range(completed + 1, target_epochs + 1):
        current_epoch = int(epoch)
        # 切回训练模式；验证函数会在每轮末切换 eval。
        autoencoder.train()
        if discriminator is not None:
            discriminator.train()
        # 本轮 KL 权重只由全局 epoch 决定，断点续训会自动落到正确位置。
        effective_kl_weight = _kl_weight_for_epoch(settings, epoch)
        # 其余三个生成器损失系数同样只读取一次，避免每 batch 重复解析配置。
        reconstruction_weight = float(settings.get("reconstruction_weight", 1.0))
        ssim_weight = float(settings.get("ssim_weight", 0.1))
        adversarial_weight = float(adversarial.get("weight", 0.01))
        # 每轮按 batch 累积原始损失和实际加权贡献，方便判断某项是否形同虚设。
        sums = {
            "generator": 0.0,
            "reconstruction": 0.0,
            "weighted_reconstruction": 0.0,
            "ssim": 0.0,
            "weighted_ssim": 0.0,
            "kl": 0.0,
            "weighted_kl": 0.0,
            "adversarial": 0.0,
            "weighted_adversarial": 0.0,
            "discriminator": 0.0,
        }
        # 下列一阶/二阶矩用于在不保留大张量的情况下计算全 epoch 潜变量统计。
        latent_value_sum = 0.0
        latent_square_sum = 0.0
        posterior_sigma_sum = 0.0
        posterior_sigma_square_sum = 0.0
        latent_value_count = 0
        batches = 0
        for batch in train_loader:
            # 类别标签对无条件 VAE 无用。
            images, _ = unpack_batch(batch)
            # 迁移后把统一数据范围 [0,1] 映射到 VAE [-1,1]。
            images = to_generator_range(images.to(device, non_blocking=True))
            # 判别器存在且已经越过 warmup 才激活对抗项。
            adversarial_active = discriminator is not None and epoch > warmup_epochs
            # 生成器步骤冻结判别器参数，但重建图仍可穿过判别器获得输入梯度。
            _set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            # 自动混合精度只包前向；敏感损失显式转 float32。
            with autocast_context(config, device):
                # MONAI Decoder 末层是裸卷积；先保留 raw 诊断，再用 tanh 映射到 [-1,1]。
                raw_reconstruction, mean, sigma = autoencoder(images)
                reconstruction = bound_reconstruction(raw_reconstruction)
                # L1 保留边缘且比 MSE 对异常像素更稳健。
                reconstruction_loss = F.l1_loss(reconstruction.float(), images.float())
                # ssim_weight=0 用于先验证基础重建，此时完全跳过 SSIM 计算与梯度。
                ssim_loss = (
                    ssim_criterion(reconstruction.float(), images.float())
                    if ssim_weight > 0.0
                    else reconstruction_loss * 0.0
                )
                # KL 把后验约束到标准高斯，使扩散潜空间连续可采样。
                kl_loss = gaussian_kl(mean.float(), sigma.float())
                # 生成器希望 PatchGAN 把重建判断为真，因此最小化负 logit 均值。
                adversarial_loss = -discriminator(reconstruction)[-1].mean() if adversarial_active else reconstruction_loss * 0.0
                # 先显式计算每一项的加权贡献，日志不再只报容易误读的原始大数。
                weighted_reconstruction = reconstruction_weight * reconstruction_loss
                weighted_ssim = ssim_weight * ssim_loss
                weighted_kl = effective_kl_weight * kl_loss
                weighted_adversarial = adversarial_weight * adversarial_loss
                # 总生成器损失就是四个已加权分量的和。
                generator_loss = (
                    weighted_reconstruction
                    + weighted_ssim
                    + weighted_kl
                    + weighted_adversarial
                )
            # 缩放反传并更新 VAE；scaler 在 BF16/CPU 下等价于普通 backward/step。
            scaler.scale(generator_loss).backward()
            scaler.step(generator_optimizer)

            # 默认构造与图/设备一致的零，便于统一日志。
            discriminator_loss = reconstruction_loss.detach() * 0.0
            if adversarial_active:
                # 判别器步骤重新解冻自身参数。
                _set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with autocast_context(config, device):
                    # 真实图和 detach 重建图均不向 VAE 回传梯度。
                    real_logits = discriminator(images.detach())[-1]
                    fake_logits = discriminator(reconstruction.detach())[-1]
                    # 标准 hinge discriminator loss。
                    discriminator_loss = 0.5 * (
                        F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
                    )
                # 更新判别器参数。
                scaler.scale(discriminator_loss).backward()
                scaler.step(discriminator_optimizer)
            # 一轮两个 step 后统一更新动态缩放因子。
            scaler.update()
            # 所有张量 detach 成 Python float，避免累积列表持有计算图。
            sums["generator"] += float(generator_loss.detach())
            sums["reconstruction"] += float(reconstruction_loss.detach())
            sums["weighted_reconstruction"] += float(weighted_reconstruction.detach())
            sums["ssim"] += float(ssim_loss.detach())
            sums["weighted_ssim"] += float(weighted_ssim.detach())
            sums["kl"] += float(kl_loss.detach())
            sums["weighted_kl"] += float(weighted_kl.detach())
            sums["adversarial"] += float(adversarial_loss.detach())
            sums["weighted_adversarial"] += float(weighted_adversarial.detach())
            sums["discriminator"] += float(discriminator_loss.detach())
            # 后验均值和 sigma 仅用 detach float32 做诊断，不参与损失反传。
            detached_mean = mean.detach().float()
            detached_sigma = sigma.detach().float()
            latent_value_sum += float(detached_mean.sum().item())
            latent_square_sum += float(detached_mean.square().sum().item())
            posterior_sigma_sum += float(detached_sigma.sum().item())
            posterior_sigma_square_sum += float(detached_sigma.square().sum().item())
            latent_value_count += int(detached_mean.numel())
            batches += 1
        # 每个 epoch 结束按闭式余弦函数设置学习率，不调用会误报顺序警告的 scheduler.step。
        advance_scheduler_to(generator_scheduler, epoch)
        if discriminator_scheduler is not None:
            advance_scheduler_to(discriminator_scheduler, epoch)
        # 验证集只计算确定性逐像素 L1。
        validation = _validation(autoencoder, val_loader, device)
        # 一阶/二阶矩还原整个 epoch 的均值和总体标准差，而不是简单平均 batch 标准差。
        latent_denominator = max(1, latent_value_count)
        latent_mean_value = latent_value_sum / latent_denominator
        latent_variance = max(
            0.0,
            latent_square_sum / latent_denominator - latent_mean_value**2,
        )
        posterior_sigma_mean = posterior_sigma_sum / latent_denominator
        posterior_sigma_variance = max(
            0.0,
            posterior_sigma_square_sum / latent_denominator - posterior_sigma_mean**2,
        )
        # 训练分量按实际 batch 数平均，并追加预热权重、潜变量统计和 validation L1。
        last_metrics = {
            **{key: value / max(1, batches) for key, value in sums.items()},
            "kl_effective_weight": float(effective_kl_weight),
            "latent_mean": float(latent_mean_value),
            "latent_std": float(math.sqrt(latent_variance)),
            "posterior_sigma_mean": float(posterior_sigma_mean),
            "posterior_sigma_std": float(math.sqrt(posterior_sigma_variance)),
            **{
                f"validation_{key}": float(value)
                for key, value in validation.items()
            },
        }
        # 只有达到检查间隔且连续无改善达到耐心值时才会返回 True。
        should_stop = early_stopping.update(last_metrics["validation_l1"], epoch)
        # 最终轮或早停时必须估计 latent_scale，后续扩散训练依赖该值。
        if epoch == target_epochs or should_stop:
            # 使用当前 VAE 后验均值抽取有限训练 batch，估计 1/std(z)。
            latent_scale = estimate_scale_factor(autoencoder, _scale_batches(train_loader), device)
        # checkpoint_interval_epochs 可减小磁盘写入频率；早停/最终轮始终强制保存。
        checkpoint_interval = int(settings.get("checkpoint_interval_epochs", 1))
        if epoch % checkpoint_interval == 0 or epoch == target_epochs or should_stop:
            _save(
                directory / "checkpoint_last.pt",
                epoch,
                autoencoder,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
                scaler,
                last_metrics,
                latent_scale,
                early_stopping,
            )
        # Keep an immutable, fully resumable snapshot for every completed epoch.
        # checkpoint_last.pt remains the automatic-resume target, while these
        # numbered files make manual rollback unambiguous.
        if bool(settings.get("save_epoch_checkpoints", False)):
            _save(
                directory / f"checkpoint_epoch_{epoch:04d}.pt",
                epoch,
                autoencoder,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
                scaler,
                last_metrics,
                latent_scale,
                early_stopping,
            )
        # 预览间隔独立于断点间隔，便于减少 PNG 产物数量。
        preview_interval = int(settings.get("preview_interval_epochs", 5))
        if epoch % preview_interval == 0 or epoch == target_epochs or should_stop:
            # 始终使用固定验证首批和 posterior mean，逐轮预览不再受 shuffle/采样噪声影响。
            preview_images, preview_reconstruction = _validation_preview(
                autoencoder, val_loader, device
            )
            preview = torch.cat((preview_images, preview_reconstruction), dim=0)
            save_image(
                from_generator_range(preview),
                directory / f"preview_epoch_{epoch:04d}.png",
                nrow=min(8, preview_images.shape[0]),
            )
        # 日志频率可调；首轮和早停轮始终输出。
        log_interval = int(settings.get("log_interval_epochs", 1))
        if epoch == 1 or epoch % log_interval == 0 or should_stop:
            logger.info(
                "AE epoch=%d/%d total=%.5f recon=%.5f "
                "ssim=%.5f(w=%.5f) kl=%.5f beta=%.7f(w=%.5f) "
                "adv=%.5f(w=%.5f) disc=%.5f "
                "val_l1=%.5f(zero=%.5f raw=%.5f) "
                "raw=[%.3f,%.3f] oor=%.4f tanh_sat=%.4f "
                "z_mu=%.5f z_std=%.5f post_sigma=%.5f scale=%s",
                epoch,
                target_epochs,
                last_metrics["generator"],
                last_metrics["reconstruction"],
                last_metrics["ssim"],
                last_metrics["weighted_ssim"],
                last_metrics["kl"],
                last_metrics["kl_effective_weight"],
                last_metrics["weighted_kl"],
                last_metrics["adversarial"],
                last_metrics["weighted_adversarial"],
                last_metrics["discriminator"],
                last_metrics["validation_l1"],
                last_metrics["validation_zero_l1"],
                last_metrics["validation_raw_l1"],
                last_metrics["validation_raw_min"],
                last_metrics["validation_raw_max"],
                last_metrics["validation_raw_out_of_range_ratio"],
                last_metrics["validation_bounded_saturation_ratio"],
                last_metrics["latent_mean"],
                last_metrics["latent_std"],
                last_metrics["posterior_sigma_mean"],
                "待估计" if latent_scale is None else f"{float(latent_scale):.6f}",
            )
        # 早停只结束 Autoencoder 阶段，已经保存的最佳进度和 scale 仍可供后续使用。
        if should_stop:
            logger.info(
                "Autoencoder 早停：最佳 validation_l1=%.6f@epoch %d",
                float(early_stopping.best or 0.0),
                early_stopping.best_progress,
            )
            break

    # 用户可能把目标轮数调小后直接恢复；旧断点没有 scale 时在这里补算一次。
    if latent_scale is None:
        # 此分支不会重新训练，只对当前已载入 VAE 估计尺度。
        latent_scale = estimate_scale_factor(autoencoder, _scale_batches(train_loader), device)
        # 将补算结果写回 last，后续扩散阶段无需重复估计。
        _save(
            directory / "checkpoint_last.pt",
            current_epoch,
            autoencoder,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            generator_scheduler,
            discriminator_scheduler,
            scaler,
            last_metrics,
            latent_scale,
            early_stopping,
        )
        # A completed run restored from an older checkpoint may reach this
        # branch only to estimate latent_scale; keep its numbered snapshot in
        # sync with checkpoint_last.pt as well.
        if bool(settings.get("save_epoch_checkpoints", False)):
            _save(
                directory / f"checkpoint_epoch_{current_epoch:04d}.pt",
                current_epoch,
                autoencoder,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
                scaler,
                last_metrics,
                latent_scale,
                early_stopping,
            )
    # summary 只包含 JSON 可序列化标量/路径，不嵌入模型权重。
    summary = {
        # 完整达到当前 YAML 目标时 complete=true；早停另有独立标志。
        "complete": current_epoch >= target_epochs,
        "stopped_early": current_epoch < target_epochs,
        "epochs": current_epoch,
        "latent_scale": float(latent_scale),
        "checkpoint": str((directory / "checkpoint_last.pt").resolve()),
        "metrics": last_metrics,
    }
    # 固定 summary 路径供扩散阶段和人工检查读取。
    atomic_write_json(summary, directory / "summary.json")
    return summary


if __name__ == "__main__":
    # 支持直接运行本文件调试；生产入口仍推荐根目录 run_pipeline.py。
    from Core.config import load_config

    run(load_config())
