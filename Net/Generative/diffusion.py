"""隐空间扩散的前向加噪、训练损失与可微 DDIM 反向采样。

扩散 U-Net 在 VAE 隐空间从零训练。普通预览可在 ``no_grad`` 中调用相同采样器；
蒸馏阶段则保留 DDIM 和 VAE 解码计算图，使 IDM/RBF 损失一直反传到初始噪声
``z_T``。生成网络参数会由调用方冻结，因此不会在蒸馏期间被更新。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping  # 配置映射与用于估计隐变量尺度的 batch 迭代器。

import torch  # 随机噪声、时间步、扩散运算和自动微分。


def _schedulers():
    """延迟导入 MONAI 的 DDIM/DDPM 调度器。"""

    # 配置检查不需要 MONAI，延迟导入能让仅运行前置阶段时更快失败/启动。
    try:
        from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
    except ImportError as error:
        raise ImportError("扩散调度器需要 MONAI") from error
    # 返回类定义，由训练/采样工厂分别实例化，避免共享可变 timesteps 状态。
    return DDIMScheduler, DDPMScheduler


def build_ddpm_scheduler(config: Mapping[str, Any]):
    """训练时使用的 DDPM 前向加噪调度器。"""

    # 训练只需要 DDPM 的随机时间步前向过程。
    _, DDPMScheduler = _schedulers()
    # 所有噪声日程参数来自 diffusion.yaml。
    settings = config["diffusion"]
    return DDPMScheduler(
        # 训练离散时间步总数，常用 1000。
        num_train_timesteps=int(settings.get("train_timesteps", 1000)),
        # beta 日程由 MONAI 名称选择，例如 scaled_linear_beta。
        schedule=str(settings.get("beta_schedule", "scaled_linear_beta")),
        # 网络可预测 epsilon 或 velocity，损失函数会同步选择目标。
        prediction_type=str(settings.get("prediction_type", "epsilon")),
        # 隐变量不应像像素一样裁剪到 [-1,1]。
        clip_sample=False,
    )


def build_ddim_scheduler(config: Mapping[str, Any]):
    """蒸馏和预览时使用的确定性 DDIM 反向调度器。"""

    # DDIM 与训练 DDPM 必须使用完全相同的训练噪声日程。
    DDIMScheduler, _ = _schedulers()
    settings = config["diffusion"]
    return DDIMScheduler(
        num_train_timesteps=int(settings.get("train_timesteps", 1000)),
        schedule=str(settings.get("beta_schedule", "scaled_linear_beta")),
        prediction_type=str(settings.get("prediction_type", "epsilon")),
        # 不裁剪潜变量，保留生成分布和可微梯度。
        clip_sample=False,
    )


def encode_latents(
    autoencoder: torch.nn.Module,
    images_minus_one_to_one: torch.Tensor,
    scale_factor: float,
    sample: bool = True,
) -> torch.Tensor:
    """编码并缩放隐变量；初始化时可用均值，扩散训练时使用重参数采样。"""

    # MONAI AutoencoderKL.encode 返回后验均值和标准差。
    mean, sigma = autoencoder.encode(images_minus_one_to_one)
    # 扩散训练采样后验以遵循 VAE；真实图初始化 z_T 时可用确定性均值。
    latent = autoencoder.sampling(mean, sigma) if sample else mean
    # scale_factor≈1/std(z)，让 U-Net 输入方差接近 1。
    return latent * float(scale_factor)


def bound_reconstruction(raw_reconstruction: torch.Tensor) -> torch.Tensor:
    """把 MONAI 裸卷积 Decoder 输出平滑限制到 VAE 的 ``[-1,1]`` 图像域。"""

    # AutoencoderKL 的 Decoder 最后一层没有输出激活；tanh 保留全域梯度，避免
    # 直接 clamp 在越界区域产生零梯度，同时统一 AE 训练、验证和扩散解码语义。
    return torch.tanh(raw_reconstruction)


def decode_latents(
    autoencoder: torch.nn.Module,
    latents: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """把缩放后的扩散隐变量解码回 ``[-1,1]`` 图像，保持输入梯度。"""

    # 先撤销扩散空间的方差缩放，再调用 VAE 第二阶段解码器。
    raw_reconstruction = autoencoder.decode_stage_2_outputs(
        latents / float(scale_factor)
    )
    # 与第一阶段 VAE 的训练/验证保持完全相同的输出范围契约。
    return bound_reconstruction(raw_reconstruction)


@torch.no_grad()
def estimate_scale_factor(
    autoencoder: torch.nn.Module,
    image_batches: Iterable[torch.Tensor],
    device: torch.device,
    maximum_batches: int = 16,
) -> float:
    """估计 ``1/std(z)``，使扩散模型看到近似单位方差的隐变量。"""

    # 仅收集每批均值潜变量的一维 CPU 副本，避免长期占用 GPU 显存。
    values: list[torch.Tensor] = []
    # 固定归一化/注意力推理行为；no_grad 装饰器已关闭计算图。
    autoencoder.eval()
    for index, images in enumerate(image_batches):
        # 只抽有限批次，估计足够稳定且不会额外扫描整个大型数据集。
        if index >= int(maximum_batches):
            break
        # 输入应已由调用方转换到 VAE 的 [-1,1] 范围。
        images = images.to(device)
        # 使用后验均值而非随机采样，估计不会受额外采样噪声影响。
        mean, _ = autoencoder.encode(images)
        values.append(mean.detach().float().cpu().flatten())
    # 空 DataLoader 或 maximum_batches=0 无法估计尺度。
    if not values:
        raise ValueError("无法从空数据加载器估计 latent scale")
    # 把所有抽样潜变量元素合并，计算总体标准差。
    standard_deviation = torch.cat(values).std(unbiased=False).item()
    # 近零标准差意味着 VAE 坍缩，继续训练扩散会导致无限缩放。
    if standard_deviation <= 1.0e-8:
        raise ValueError("VAE 隐变量标准差接近零，不能训练扩散模型")
    # 返回 Python float，便于存入 summary/断点并跨设备使用。
    return float(1.0 / standard_deviation)


def diffusion_training_loss(
    model: torch.nn.Module,
    scheduler,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    null_class_id: int,
    class_dropout: float,
    min_snr_gamma: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """计算类别条件扩散损失，并随机替换为空类别完成 CFG 训练。

    每个样本独立采样噪声和时间步。``class_dropout`` 决定多少标签替换为额外的
    ``null_class_id``；采样时同一 U-Net 因而同时能给出条件和无条件预测。
    ``min_snr_gamma>0`` 时使用 Min-SNR 权重抑制极端高信噪比时间步。可选
    ``generator`` 让验证阶段使用独立固定随机序列，不扰动训练随机状态。
    """

    # 目标高斯噪声与潜变量形状、dtype、device 完全一致；验证可传独立生成器。
    noise = torch.randn(
        latents.shape,
        dtype=latents.dtype,
        device=latents.device,
        generator=generator,
    )
    # 每个样本从完整训练日程均匀采样一个离散时间步。
    timesteps = torch.randint(
        0,
        int(scheduler.num_train_timesteps),
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.long,
        generator=generator,
    )
    # q(z_t | z_0) 按对应 alpha 累积系数混合干净潜变量与噪声。
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    # clone 避免原地替换破坏 DataLoader 提供的真实类别标签。
    conditioned_labels = class_labels.clone()
    # CFG 训练按样本 Bernoulli 丢弃类别条件。
    if float(class_dropout) > 0:
        drop = torch.rand(
            class_labels.shape,
            device=class_labels.device,
            generator=generator,
        ) < float(class_dropout)
        conditioned_labels[drop] = int(null_class_id)
    # U-Net 输出与 noisy_latents 同形状的 epsilon 或 velocity 预测。
    prediction = model(noisy_latents, timesteps=timesteps, class_labels=conditioned_labels)
    # 从调度器读取预测类型，确保训练目标与采样器公式一致。
    prediction_type = str(getattr(scheduler, "prediction_type", "epsilon"))
    # v_prediction 目标由调度器根据 z0、epsilon 和 t 计算。
    if prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    # epsilon 模式直接回归本轮加入的标准高斯噪声。
    elif prediction_type == "epsilon":
        target = noise
    else:
        raise ValueError(f"不支持的扩散 prediction_type：{prediction_type}")
    # 先对非 batch 维求均方误差，保留每个样本以便应用时间步权重。
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
    # gamma<=0 表示关闭 Min-SNR，保持普通均匀时间步 MSE。
    gamma = float(min_snr_gamma)
    if gamma > 0:
        # alpha_cumprod[t] 转到潜变量设备，并用 float32 计算 SNR。
        alpha_cumprod = scheduler.alphas_cumprod.to(latents.device)[timesteps].float()
        # SNR = alpha_bar / (1-alpha_bar)。
        snr = alpha_cumprod / (1.0 - alpha_cumprod).clamp_min(1.0e-8)
        # 把过大的 SNR 截断到 gamma，降低近干净时间步的主导作用。
        clipped = torch.minimum(snr, snr.new_tensor(gamma))
        # v_prediction 与 epsilon prediction 使用论文对应的不同归一化分母。
        weights = clipped / (snr + 1.0 if prediction_type == "v_prediction" else snr.clamp_min(1.0e-8))
        per_sample = per_sample * weights
    # 对 batch 样本等权平均得到最终标量损失。
    return per_sample.mean()


def differentiable_ddim_sample(
    model: torch.nn.Module,
    scheduler,
    initial_latents: torch.Tensor,
    class_labels: torch.Tensor,
    null_class_id: int,
    inference_steps: int,
    guidance_scale: float,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """不使用 ``no_grad`` 的 DDIM 反演，使分类损失能传回 ``z_T``。

    ``eta`` 固定为 0，保证同一隐变量和标签对应确定性图像；模型权重由调用方冻结，
    计算图只需要为待优化隐变量保留梯度。
    """

    # 根据可配置推理步数创建从大到小的 DDIM 时间步序列。
    scheduler.set_timesteps(int(inference_steps), device=initial_latents.device)
    # sample 始终保留到 initial_latents 的计算图；不要 clone/detach。
    sample = initial_latents
    # 按 DDIM 逆序时间步逐步去噪。
    for timestep in scheduler.timesteps:
        # MONAI step 接收 Python/int 时间步；同时兼容调度器返回 tensor 或 int。
        timestep_value = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        # U-Net 要求 batch 中每个样本都有一个时间步标签。
        timestep_batch = torch.full(
            (sample.shape[0],), timestep_value, device=sample.device, dtype=torch.long
        )

        # 用局部纯函数封装 U-Net 前向，供 torch.checkpoint 重算激活。
        def predict(current_sample, current_timesteps, current_labels):
            return model(
                current_sample,
                timesteps=current_timesteps,
                class_labels=current_labels,
            )

        # 激活检查点只在需要梯度时启用；预览 no_grad 下重算没有意义。
        if bool(gradient_checkpointing) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            # 条件分支使用真实类别标签。
            conditional = checkpoint(
                predict,
                sample,
                timestep_batch,
                class_labels,
                use_reentrant=False,
            )
        else:
            conditional = predict(sample, timestep_batch, class_labels)
        # guidance_scale=1 时条件预测已经是最终结果，不额外执行无条件 U-Net。
        if float(guidance_scale) != 1.0:
            # 额外类别编号 num_classes 代表空条件。
            null_labels = torch.full_like(class_labels, int(null_class_id))
            if bool(gradient_checkpointing) and torch.is_grad_enabled():
                # checkpoint 已在同一条件分支中导入；此条件与上面完全一致。
                unconditional = checkpoint(
                    predict,
                    sample,
                    timestep_batch,
                    null_labels,
                    use_reentrant=False,
                )
            else:
                unconditional = predict(sample, timestep_batch, null_labels)
            # 标准 CFG：eps_u + w(eps_c-eps_u)。
            prediction = unconditional + float(guidance_scale) * (conditional - unconditional)
        else:
            prediction = conditional
        # eta=0 去掉 DDIM 随机项，同一 z_T/标签必然生成同一图像且保持可微。
        sample, _ = scheduler.step(prediction, timestep_value, sample, eta=0.0)
    # 返回最终缩放潜变量 z_0，VAE 解码由调用方执行。
    return sample


def add_maximum_noise(scheduler, clean_latents: torch.Tensor) -> torch.Tensor:
    """把真实图像隐变量加噪到最后一个训练时刻，作为可学习 ``z_T`` 初值。"""

    # 每个真实隐变量使用独立标准高斯噪声。
    noise = torch.randn_like(clean_latents)
    # 所有样本都选择训练日程最后一时刻 T-1。
    timesteps = torch.full(
        (clean_latents.shape[0],),
        int(scheduler.num_train_timesteps) - 1,
        device=clean_latents.device,
        dtype=torch.long,
    )
    # 使用与 U-Net 训练相同的前向加噪公式，得到比纯高斯更贴近真实轨迹的初始化。
    return scheduler.add_noise(clean_latents, noise, timesteps)


def gaussian_kl(mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """AutoencoderKL 后验相对标准正态的全元素平均 KL。

    ``sigma`` 是标准差而非 log-variance。这里有意对 batch、通道和
    空间维全部取 mean，使同一 ``kl_weight`` 在不同图像分辨率下保持
    稳定语义。训练阶段通过线性 warmup 把这个平均值的权重逐步
    提高，因此不能把“潜变量求和”实现中的权重原样照搬到此处。
    """

    # 方差下界避免 sigma=0 时 log(0)；保持对 mean/sigma 的梯度。
    variance = sigma.square().clamp_min(1.0e-12)
    # KL(N(mean,var) || N(0,1)) 的逐元素闭式解，再对 batch/通道/空间取平均。
    return 0.5 * (mean.square() + variance - variance.log() - 1.0).mean()
