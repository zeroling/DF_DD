"""condense 与诊断实验共用的冻结 VAE / DDIM 参数化工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from Core.augmentations import from_generator_range, to_generator_range
from Core.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    model_state_from_checkpoint,
)
from Pipeline.ablation_config import (
    condensation_settings,
    generator_stage_directory,
)
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
)


@dataclass
class GenerativeBundle:
    autoencoder: nn.Module
    diffusion: nn.Module | None
    latent_scale: float
    autoencoder_checkpoint: str
    diffusion_checkpoint: str | None


def _checkpoint(stage_directory) -> tuple[dict[str, Any], str]:
    path = find_latest_checkpoint(stage_directory)
    if path is None:
        raise FileNotFoundError(f"没有找到生成模型断点：{stage_directory}")
    return load_checkpoint(path, "cpu"), str(path.resolve())


def load_generators(
    experiment_config: Mapping[str, Any],
    num_classes: int,
    device: torch.device,
    require_diffusion: bool,
) -> GenerativeBundle:
    """加载冻结生成器；只按张量形状兼容性恢复，不校验哈希。"""

    project_config = experiment_config
    ae_payload, ae_path = _checkpoint(
        generator_stage_directory(experiment_config, "autoencoder")
    )
    scale = ae_payload.get("latent_scale")
    autoencoder = build_autoencoder(project_config)
    autoencoder.load_state_dict(model_state_from_checkpoint(ae_payload), strict=True)
    autoencoder = freeze_module(autoencoder.to(device))
    del ae_payload

    diffusion: nn.Module | None = None
    diffusion_path: str | None = None
    if require_diffusion:
        diffusion_payload, diffusion_path = _checkpoint(
            generator_stage_directory(experiment_config, "diffusion")
        )
        scale = diffusion_payload.get("latent_scale", scale)
        diffusion = build_diffusion_unet(project_config, int(num_classes))
        ema_state = diffusion_payload.get("ema", {})
        ema_shadow = (
            ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
        )
        diffusion.load_state_dict(
            ema_shadow
            if isinstance(ema_shadow, Mapping)
            else model_state_from_checkpoint(diffusion_payload),
            strict=True,
        )
        diffusion = freeze_module(diffusion.to(device))
        del diffusion_payload
    if scale is None or float(scale) <= 0:
        raise ValueError("生成模型断点缺少有效 latent_scale")
    return GenerativeBundle(
        autoencoder=autoencoder,
        diffusion=diffusion,
        latent_scale=float(scale),
        autoencoder_checkpoint=ae_path,
        diffusion_checkpoint=diffusion_path,
    )


def encode_z0(
    bundle: GenerativeBundle,
    pixels_zero_to_one: torch.Tensor,
) -> torch.Tensor:
    return encode_latents(
        bundle.autoencoder,
        to_generator_range(pixels_zero_to_one),
        bundle.latent_scale,
        sample=False,
    )


def decode_z0(
    bundle: GenerativeBundle,
    latents: torch.Tensor,
) -> torch.Tensor:
    return from_generator_range(
        decode_latents(bundle.autoencoder, latents, bundle.latent_scale)
    )


def initialize_zT(
    experiment_config: Mapping[str, Any],
    clean_latents: torch.Tensor,
) -> torch.Tensor:
    return add_maximum_noise(
        build_ddpm_scheduler(experiment_config),
        clean_latents,
    )


def decode_zT(
    experiment_config: Mapping[str, Any],
    bundle: GenerativeBundle,
    initial_latents: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    if bundle.diffusion is None:
        raise RuntimeError("decode_zT 需要 Diffusion 模型")
    settings = condensation_settings(experiment_config)["latent"]
    denoised = differentiable_ddim_sample(
        bundle.diffusion,
        build_ddim_scheduler(experiment_config),
        initial_latents,
        labels,
        int(num_classes),
        int(settings["ddim_steps"]),
        float(settings["guidance_scale"]),
        gradient_checkpointing=bool(gradient_checkpointing),
    )
    return decode_z0(bundle, denoised)


def _guided_prediction(
    model: nn.Module,
    sample: torch.Tensor,
    timestep: int,
    labels: torch.Tensor,
    null_class_id: int,
    guidance_scale: float,
) -> torch.Tensor:
    timestep_batch = torch.full(
        (sample.shape[0],),
        int(timestep),
        dtype=torch.long,
        device=sample.device,
    )
    conditional = model(
        sample,
        timesteps=timestep_batch,
        class_labels=labels,
    )
    if float(guidance_scale) == 1.0:
        return conditional
    null_labels = torch.full_like(labels, int(null_class_id))
    unconditional = model(
        sample,
        timesteps=timestep_batch,
        class_labels=null_labels,
    )
    return unconditional + float(guidance_scale) * (
        conditional - unconditional
    )


@torch.no_grad()
def ddim_invert(
    experiment_config: Mapping[str, Any],
    bundle: GenerativeBundle,
    clean_latents: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    inference_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """确定性 DDIM inversion：z0→zT，供 D2 测量往返信息损失。"""

    if bundle.diffusion is None:
        raise RuntimeError("DDIM inversion 需要 Diffusion 模型")
    scheduler = build_ddim_scheduler(experiment_config)
    if str(getattr(scheduler, "prediction_type", "epsilon")) != "epsilon":
        raise ValueError("当前确定性 inversion 只支持 epsilon prediction")
    scheduler.set_timesteps(int(inference_steps), device=clean_latents.device)
    descending = [
        int(item.item()) if torch.is_tensor(item) else int(item)
        for item in scheduler.timesteps
    ]
    ascending = list(reversed(descending))
    alphas = scheduler.alphas_cumprod.to(
        device=clean_latents.device, dtype=torch.float32
    )
    sample = clean_latents
    for index in range(len(ascending) - 1):
        current_t = ascending[index]
        next_t = ascending[index + 1]
        epsilon = _guided_prediction(
            bundle.diffusion,
            sample,
            current_t,
            labels,
            int(num_classes),
            float(guidance_scale),
        ).float()
        alpha_current = alphas[current_t].clamp_min(1.0e-8)
        alpha_next = alphas[next_t].clamp_min(1.0e-8)
        predicted_clean = (
            sample.float()
            - (1.0 - alpha_current).sqrt() * epsilon
        ) / alpha_current.sqrt()
        sample = (
            alpha_next.sqrt() * predicted_clean
            + (1.0 - alpha_next).sqrt() * epsilon
        ).to(clean_latents.dtype)
    return sample


@torch.no_grad()
def reconstruct_vae(
    bundle: GenerativeBundle,
    images: torch.Tensor,
) -> torch.Tensor:
    return decode_z0(bundle, encode_z0(bundle, images))


@torch.no_grad()
def reconstruct_diffusion(
    experiment_config: Mapping[str, Any],
    bundle: GenerativeBundle,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    inference_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    clean = encode_z0(bundle, images)
    inverted = ddim_invert(
        experiment_config,
        bundle,
        clean,
        labels,
        num_classes,
        inference_steps,
        guidance_scale,
    )
    if bundle.diffusion is None:
        raise RuntimeError("D2 需要 Diffusion 模型")
    reconstructed = differentiable_ddim_sample(
        bundle.diffusion,
        build_ddim_scheduler(experiment_config),
        inverted,
        labels,
        int(num_classes),
        int(inference_steps),
        float(guidance_scale),
    )
    return decode_z0(bundle, reconstructed)
