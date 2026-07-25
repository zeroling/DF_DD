"""D 组派生缓存、C 组合成集与固定随机 IPC=1 数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from Core.checkpoint import atomic_torch_save
from Core.config import resolve_path
from Core.data import ClassImagePool, build_data_bundle
from Core.io_utils import atomic_write_json, read_json
from Core.run_context import autocast_context, resolve_device
from Pipeline.ablation_config import (
    ablation_settings,
    condensation_settings,
    output_root,
)
from Net.Condensation.generative import (
    load_generators,
    reconstruct_diffusion,
    reconstruct_vae,
)
from Core.experiment_runtime import (
    cleanup_memory,
    cuda_peak_megabytes,
    retry_cuda_oom,
)


def deterministic_bundle(config: Mapping[str, Any]):
    """构造确定性数据集；没有独立验证集时把自动切出的 10% 还给训练集。

    D/C 实验不使用验证集早停，因此 IDM 基线应使用原始 train 划分的全部
    样本。保留 ``bundle.val`` 仅是为了兼容现有 DataBundle 接口，测试集从不
    合并，也不会参与参数或超参数选择。
    """

    bundle = build_data_bundle(
        config,
        train_augmentation={"enabled": False},
    )
    data = config["data"]
    adapter = str(data.get("adapter", "folder")).lower()
    if adapter == "folder":
        root = resolve_path(config, data["root"])
        explicit_validation = (
            root / str(data.get("val_split", "val"))
        ).is_dir()
    else:
        configured = data.get("manifest", {}).get("val")
        explicit_validation = configured not in {None, "", "null"}
    if not explicit_validation:
        records = sorted(
            [*bundle.train.records, *bundle.val.records],
            key=lambda record: record.key,
        )
        bundle.train.records = records
        bundle.train.targets = [record.label for record in records]
    return bundle


class NpyImageDataset(Dataset):
    """按需 mmap 读取 `[N,C,H,W] uint8` 派生训练集。"""

    def __init__(self, directory: str | Path):
        root = Path(directory)
        manifest = read_json(root / "manifest.json")
        if not manifest or not bool(manifest.get("complete", False)):
            raise FileNotFoundError(f"派生数据缓存尚未完成：{root}")
        self.images = np.load(root / "images.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.targets = [int(value) for value in self.labels.tolist()]
        self.class_names = list(manifest["class_names"])

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        # copy 避免 PyTorch 对只读 memmap 发出不可写张量警告。
        image = torch.from_numpy(np.array(self.images[int(index)], copy=True)).float()
        return {
            "image": image.div_(255.0),
            "label": torch.tensor(int(self.labels[int(index)]), dtype=torch.long),
            "key": f"npy:{int(index)}",
        }


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.detach().cpu().float()
        self.labels = labels.detach().cpu().long()
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[int(index)],
            "label": self.labels[int(index)],
            "key": f"tensor:{int(index)}",
        }


def diagnostic_cache_directory(
    config: Mapping[str, Any],
    experiment: str,
) -> Path:
    return output_root(config) / "D" / str(experiment) / "cache"


def _open_cache_arrays(
    directory: Path,
    count: int,
    image_shape: tuple[int, int, int],
    resume: bool,
):
    mode = "r+" if resume else "w+"
    images = np.lib.format.open_memmap(
        directory / "images.npy",
        mode=mode,
        dtype=np.uint8,
        shape=(int(count), *map(int, image_shape)),
    )
    labels = np.lib.format.open_memmap(
        directory / "labels.npy",
        mode=mode,
        dtype=np.int64,
        shape=(int(count),),
    )
    return images, labels


def prepare_diagnostic_cache(
    config: Mapping[str, Any],
    experiment: str,
) -> dict[str, Any]:
    """生成 D1/D2 的一次性 uint8 mmap；中断后从 completed_count 继续。"""

    experiment = str(experiment)
    if experiment not in {"D1", "D2"}:
        raise ValueError("只有 D1/D2 需要生成派生缓存")
    directory = diagnostic_cache_directory(config, experiment)
    directory.mkdir(parents=True, exist_ok=True)
    bundle = deterministic_bundle(config)
    experiment_settings = ablation_settings(config)
    condensation = condensation_settings(config)
    count = len(bundle.train)
    if bool(config.get("_smoke", False)):
        count = min(count, int(config.get("_smoke_samples", 64)))
    sample = bundle.train[0]["image"]
    image_shape = tuple(map(int, sample.shape))
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path, default={}) or {}
    if bool(manifest.get("complete", False)):
        return manifest
    completed = int(manifest.get("completed_count", 0))
    resume = (
        completed > 0
        and (directory / "images.npy").is_file()
        and (directory / "labels.npy").is_file()
    )
    if not resume:
        completed = 0
    images_memmap, labels_memmap = _open_cache_arrays(
        directory, count, image_shape, resume
    )
    device = resolve_device(config)
    require_diffusion = experiment == "D2"
    generators = load_generators(
        config,
        bundle.num_classes,
        device,
        require_diffusion=require_diffusion,
    )
    requested = int(
        condensation["memory"]["reconstruction_batch"][
            "diffusion" if require_diffusion else "vae"
        ]
    )
    minimum = int(condensation["memory"].get("retry_minimum", 1))
    flush_interval = max(
        1,
        int(
            experiment_settings["diagnostic"][
                "cache_flush_interval_batches"
            ]
        ),
    )
    processed_batches = 0
    maximum_peak_mib = 0.0
    index = int(completed)
    while index < count:
        remaining = count - index

        def transform(batch_size: int):
            records = [bundle.train[offset] for offset in range(index, index + min(batch_size, remaining))]
            batch_images = torch.stack([item["image"] for item in records]).to(
                device, non_blocking=True
            )
            batch_labels = torch.stack([item["label"] for item in records]).to(
                device, non_blocking=True
            )
            with autocast_context(config, device):
                if experiment == "D1":
                    result = reconstruct_vae(generators, batch_images)
                else:
                    reconstruction = experiment_settings["diagnostic"][
                        "diffusion_reconstruction"
                    ]
                    result = reconstruct_diffusion(
                        config,
                        generators,
                        batch_images,
                        batch_labels,
                        bundle.num_classes,
                        int(reconstruction["inference_steps"]),
                        float(reconstruction["guidance_scale"]),
                    )
            return (
                result.detach().float().clamp(0, 1).cpu(),
                batch_labels.detach().cpu(),
            )

        (batch_result, batch_labels), actual_batch = retry_cuda_oom(
            transform,
            min(requested, remaining),
            minimum,
        )
        length = int(batch_result.shape[0])
        maximum_peak_mib = max(maximum_peak_mib, cuda_peak_megabytes())
        images_memmap[index : index + length] = (
            batch_result.mul(255).round().to(torch.uint8).numpy()
        )
        labels_memmap[index : index + length] = batch_labels.numpy()
        index += length
        requested = int(actual_batch)
        processed_batches += 1
        if processed_batches % flush_interval == 0 or index == count:
            images_memmap.flush()
            labels_memmap.flush()
            atomic_write_json(
                {
                    "experiment": experiment,
                    "complete": index == count,
                    "completed_count": int(index),
                    "count": int(count),
                    "shape": [int(count), *image_shape],
                    "dtype": "uint8",
                    "class_names": bundle.class_names,
                    "microbatch": int(requested),
                    "cuda_peak_mib": float(maximum_peak_mib),
                    "autoencoder_checkpoint": generators.autoencoder_checkpoint,
                    "diffusion_checkpoint": generators.diffusion_checkpoint,
                },
                manifest_path,
            )
    del images_memmap, labels_memmap
    generators = None
    cleanup_memory()
    return read_json(manifest_path)


def random_real_ipc1_dataset(
    config: Mapping[str, Any],
    repeat: int,
) -> TensorImageDataset:
    bundle = deterministic_bundle(config)
    pool = ClassImagePool(
        bundle.train,
        bundle.num_classes,
        int(config["project"]["seed"]) + int(repeat) * 1009,
        cache_images=False,
    )
    images, labels = pool.sample_all_classes(1)
    directory = output_root(config) / "D" / "D3" / f"repeat_{int(repeat)}"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "images": images,
            "labels": labels,
            "class_names": bundle.class_names,
            "ipc": 1,
            "method": "random_real_ipc1",
        },
        directory / "dataset.pt",
    )
    return TensorImageDataset(images, labels)


def load_c_synthetic(path: str | Path) -> TensorImageDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return TensorImageDataset(payload["images"], payload["labels"])
