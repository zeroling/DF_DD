"""与具体疾病、目录组织和医学文件格式解耦的数据层。

所有适配器最终只产生 ``SampleRecord``，所有解码器最终只产生 ``[0,1]`` 的
``float32 [C,H,W]`` 张量。分类网络、VAE 和扩散模型所需的不同归一化由训练阶段
显式完成，因此数据集更换不会把归一化细节传播到整条管线。
"""

from __future__ import annotations

import csv  # 读取带表头的 CSV 样本清单。
import json  # 读取 JSON/JSONL 清单。
import random  # 分层划分和 DataLoader worker 的 Python 随机源。
import sys  # Windows 使用单独的 DataLoader worker 安全默认值。
from dataclasses import dataclass  # 用轻量数据类保存样本索引与数据划分。
from pathlib import Path  # 统一处理绝对/相对路径和扩展名。
from typing import Any, Callable, Mapping, Sequence  # 描述可扩展解码器与配置接口。

import numpy as np  # 加载 NPY/NPZ、PIL 数组以及 worker 随机种子。
import torch  # 统一图像张量、采样器和数据加载器。
from PIL import Image  # 解码 PNG/JPEG/TIFF 等普通二维图像。
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler  # 通用数据接口。
from torchvision.transforms import InterpolationMode  # 明确 resize/affine 插值方式。
from torchvision.transforms import functional as TF  # 对张量执行尺寸与保守空间增强。

from Core.config import image_size, resolve_path  # 全局尺寸及相对项目根路径解析。


# 普通二维栅格格式由 Pillow 解码。
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# NumPy 数组格式用于预处理后的医学切片或特征图。
ARRAY_EXTENSIONS = {".npy", ".npz"}
# Torch 张量格式兼容保存为 tensor 或包含 image/data 键的字典。
TENSOR_EXTENSIONS = {".pt", ".pth"}
# DICOM 通过可选 pydicom 依赖按需解码。
DICOM_EXTENSIONS = {".dcm", ".dicom"}
# 文件夹扫描只收集已注册的扩展名；外部插件注册后也会动态加入该集合。
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | ARRAY_EXTENSIONS | TENSOR_EXTENSIONS | DICOM_EXTENSIONS


@dataclass(frozen=True)
class SampleRecord:
    """尚未解码的一条样本索引。"""

    path: Path  # 原始文件绝对路径，仅在 __getitem__ 时真正解码。
    label: int  # 连续类别编号，范围为 [0, num_classes-1]。
    class_name: str  # 人类可读类别名，便于输出和核对映射。
    key: str  # 数据集内稳定标识，用于追踪样本但不参与训练。


@dataclass
class DataBundle:
    """三个数据划分和全局一致的类别映射。"""

    train: "MedicalImageDataset"  # 训练集，可启用数据层增强。
    val: "MedicalImageDataset"  # 验证集，只执行确定性预处理。
    test: "MedicalImageDataset"  # 测试集，只在最终评价时使用。
    class_names: list[str]  # 按类别编号顺序排列的名称。
    class_to_idx: dict[str, int]  # 类别名称到连续编号的反向映射。

    @property
    def num_classes(self) -> int:
        """返回统一类别数，避免各阶段重复从记录中猜测。"""

        return len(self.class_names)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """读取 CSV、JSON 或 JSONL 清单。"""

    # 扩展名决定解析方式，清单字段名由 YAML 另行配置。
    suffix = path.suffix.lower()
    # utf-8-sig 同时兼容 Excel 导出的 BOM CSV 和标准 UTF-8 CSV。
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    # JSONL 每个非空行是一条独立 JSON 对象，适合大型清单流式生成。
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows
    # JSON 可直接是列表，也可用 samples/records 顶层键包裹。
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("samples", payload.get("records", []))
        # 其他顶层类型无法解释为样本序列，立即给出格式错误。
        if not isinstance(payload, list):
            raise TypeError(f"JSON 清单必须是列表或包含 samples 列表：{path}")
        return [dict(row) for row in payload]
    # 不根据内容猜格式，避免扩展名拼错时产生难理解的解析异常。
    raise ValueError(f"清单只支持 .csv/.json/.jsonl：{path}")


def _discover_folder_classes(root: Path, train_split: str) -> list[str]:
    """从训练划分的一级子目录按名称排序发现类别。"""

    # folder 适配器约定 root/train_split/class_name/files。
    split_root = root / train_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"训练目录不存在：{split_root}")
    # 排序保证不同操作系统和多次运行得到一致类别编号。
    names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    # 本项目是分类蒸馏，单类数据无法评价分类准确率。
    if len(names) < 2:
        raise ValueError(f"训练目录至少需要两个类别子目录：{split_root}")
    return names


def _folder_records(
    root: Path,
    split: str,
    class_names: Sequence[str],
    allow_missing: bool,
) -> list[SampleRecord]:
    """扫描一个 folder 划分并生成尚未解码的样本记录。"""

    # 每个划分在 root 下有独立目录。
    split_root = root / split
    # 验证集可缺失并由训练集分层切分，训练/测试集缺失则报错。
    if not split_root.is_dir():
        if allow_missing:
            return []
        raise FileNotFoundError(f"数据划分目录不存在：{split_root}")
    # 记录列表只保存索引信息，不在构建阶段读取大图像。
    records: list[SampleRecord] = []
    # enumerate(class_names) 是全项目类别编号的唯一来源。
    for label, class_name in enumerate(class_names):
        class_root = split_root / class_name
        # 某划分缺少类别会在 _verify_records 再次统一检查；这里给出具体目录信息。
        if not class_root.is_dir():
            if allow_missing:
                continue
            raise FileNotFoundError(f"类别目录不存在：{class_root}")
        # rglob 支持类别目录内按患者/医院继续分层组织文件。
        for path in sorted(class_root.rglob("*")):
            # 忽略未知辅助文件和目录，只接受已注册二维解码格式。
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append(
                    SampleRecord(
                        # resolve 固化绝对路径，运行目录变化不会影响后续解码。
                        path=path.resolve(),
                        label=label,
                        class_name=str(class_name),
                        # 相对 root 的 POSIX key 跨 Windows/Linux 保持统一分隔符。
                        key=path.relative_to(root).as_posix(),
                    )
                )
    return records


def _manifest_rows_for_split(config: Mapping[str, Any], split: str) -> tuple[list[dict], Path] | None:
    """读取某一划分的清单，并在共享清单中按 split 列筛选。"""

    # 每个划分既可配置独立文件，也可让 train/val/test 指向同一个带 split 列的文件。
    manifest_config = config["data"].get("manifest", {})
    configured = manifest_config.get(split)
    # YAML null、空字符串以及文字 null 都表示该划分没有清单。
    if configured in {None, "", "null"}:
        return None
    # 清单相对路径按全局 project.root 解析。
    manifest_path = resolve_path(config, configured)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{split} 清单不存在：{manifest_path}")
    # 统一读取为“每行一个普通字典”。
    rows = _read_manifest(manifest_path)
    # split_column 可针对不同公开数据集修改字段名。
    split_column = str(manifest_config.get("split_column", "split"))
    # 仅当至少一行真正提供 split 值时筛选；独立清单无需冗余 split 列。
    if any(split_column in row and str(row[split_column]).strip() for row in rows):
        rows = [row for row in rows if str(row.get(split_column, "")).lower() == split.lower()]
    return rows, manifest_path


def _manifest_class_names(config: Mapping[str, Any]) -> list[str]:
    """从显式配置或训练清单推断稳定类别名称顺序。"""

    # 显式 class_names 的顺序直接定义标签编号，最适合跨实验固定类别映射。
    configured = config["data"].get("class_names")
    if configured:
        return [str(value) for value in configured]
    # 未显式给出时，只允许从训练清单发现，不能由验证/测试集决定标签空间。
    train_manifest = _manifest_rows_for_split(config, "train")
    if train_manifest is None:
        raise ValueError("manifest 适配器至少需要 data.manifest.train")
    # label_column 可适配 diagnosis、target 等不同字段名。
    label_column = str(config["data"].get("manifest", {}).get("label_column", "label"))
    # 排序后建立确定性类别编号；如果需要自定义顺序应配置 class_names。
    return sorted({str(row[label_column]) for row in train_manifest[0]})


def _manifest_records(
    config: Mapping[str, Any],
    split: str,
    class_names: Sequence[str],
) -> list[SampleRecord]:
    """将清单行转换成统一 ``SampleRecord``，支持名称标签和数字标签。"""

    # 缺失划分返回空列表，稍后仅验证集允许自动拆分。
    manifest_data = _manifest_rows_for_split(config, split)
    if manifest_data is None:
        return []
    # 保留清单自身路径，以便解析相对清单目录的样本路径。
    rows, manifest_path = manifest_data
    settings = config["data"].get("manifest", {})
    # 两个核心列名均可由 YAML 覆盖。
    path_column = str(settings.get("path_column", "path"))
    label_column = str(settings.get("label_column", "label"))
    # 类别名称映射由训练集/显式配置统一确定。
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    # 相对样本路径优先相对于 data.root；不存在时再相对于清单目录。
    data_root = resolve_path(config, config["data"]["root"])
    records: list[SampleRecord] = []
    for row_index, row in enumerate(rows):
        # 每行至少必须能定位图像与标签。
        if path_column not in row or label_column not in row:
            raise KeyError(f"清单第 {row_index + 1} 行缺少 {path_column!r} 或 {label_column!r}")
        # expanduser 支持用户在外部清单中写入 ~，但本项目输出不会依赖它。
        raw_path = Path(str(row[path_column])).expanduser()
        # 两种相对路径约定按文件是否存在自动选择。
        if not raw_path.is_absolute():
            from_root = data_root / raw_path
            raw_path = from_root if from_root.exists() else manifest_path.parent / raw_path
        # 标签统一转字符串，以兼容 CSV 读取出的文本数字和 JSON 数字。
        raw_label = str(row[label_column])
        # 优先把标签解释为配置中的类别名称。
        if raw_label in class_to_idx:
            label = class_to_idx[raw_label]
        # 否则允许合法的连续数字编号。
        elif raw_label.isdigit() and 0 <= int(raw_label) < len(class_names):
            label = int(raw_label)
        else:
            raise ValueError(f"清单出现未知类别 {raw_label!r}：{manifest_path}")
        # 保存绝对路径并构造包含划分/行号的稳定 key，避免同名文件冲突。
        path = raw_path.resolve()
        records.append(SampleRecord(path, label, class_names[label], f"{split}:{row_index}:{path.name}"))
    return records


def _stratified_split(
    records: Sequence[SampleRecord], ratio: float, seed: int
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """逐类别确定性拆分验证集，保证小类别尽量同时出现在两侧。"""

    # 先按连续类别编号分桶，绝不在所有样本上直接随机切分。
    by_class: dict[int, list[SampleRecord]] = {}
    for record in records:
        by_class.setdefault(record.label, []).append(record)
    # 分别累积拆分后的训练和验证记录。
    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    # 排序类别保证处理顺序稳定；每类使用不同但可复现的随机种子。
    for label, values in sorted(by_class.items()):
        values = list(values)
        random.Random(int(seed) + int(label) * 1009).shuffle(values)
        count = len(values)
        # 类别有至少两张时，验证集至少取 1 张且至少给训练集留 1 张。
        val_count = min(count - 1, max(1, int(round(count * ratio)))) if count > 1 else 0
        val_records.extend(values[:val_count])
        train_records.extend(values[val_count:])
    return train_records, val_records


def _extract_tensor_payload(payload: Any, path: Path) -> torch.Tensor:
    """从裸张量、NumPy 数组或常见字典键中递归提取图像张量。"""

    # detach/cpu 防止数据文件中意外保存的计算图或 CUDA 设备泄漏到 Dataset worker。
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    # NumPy 数组可零拷贝转换，后续强度处理会按需转换 float32。
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload)
    # 兼容常见预处理脚本的字典封装键，并允许嵌套一层或多层。
    if isinstance(payload, Mapping):
        for key in ("image", "pixel", "pixels", "data", "array"):
            if key in payload:
                return _extract_tensor_payload(payload[key], path)
    # 不猜测其他任意字段，避免错误地把标签或元数据当成图像。
    raise TypeError(f"无法从文件中识别图像张量：{path}")


# 解码器接收文件路径和目标通道数，返回尚未标准化尺寸/强度的张量。
Decoder = Callable[[Path, int], torch.Tensor]
# 扩展名到解码函数的全局注册表，允许外部数据插件增量扩展。
_DECODER_REGISTRY: dict[str, Decoder] = {}


def register_image_decoder(extensions: str | Sequence[str], decoder: Decoder) -> None:
    """注册新的二维图像解码器。

    外部数据集只需在构建 ``DataBundle`` 前调用本函数，无需修改 Dataset、采样器或
    训练阶段。扩展名可写 ``.foo`` 或 ``foo``，解码器返回任意二维/HWC/CHW 张量。
    """

    # 单个扩展名先包装成列表，后续只维护一条注册逻辑。
    values = [extensions] if isinstance(extensions, str) else list(extensions)
    for extension in values:
        # 统一小写，使 Windows/Linux 上的后缀判断口径一致。
        normalized = str(extension).lower()
        # 调用者可写 dcm 或 .dcm，两种形式最终都规范为以点开头。
        normalized = normalized if normalized.startswith(".") else f".{normalized}"
        # 后注册的解码器覆盖旧实现，便于项目针对特殊 TIFF/DICOM 自定义读取。
        _DECODER_REGISTRY[normalized] = decoder
        # 文件夹扫描器也必须认识新扩展名，否则注册后仍发现不了文件。
        SUPPORTED_EXTENSIONS.add(normalized)


def _decode_raster(path: Path, channels: int) -> torch.Tensor:
    """使用 Pillow 解码普通栅格图，并按目标通道转成 L/RGB。"""

    # with 确保文件句柄在数组复制完成后立即关闭。
    with Image.open(path) as image:
        # 当前项目支持 1 或 3 通道；其他配置会在全局校验阶段拒绝。
        image = image.convert("L" if channels == 1 else "RGB")
        # copy=True 让返回数组脱离 Pillow 底层只读缓冲区。
        return torch.from_numpy(np.array(image, copy=True))


def _decode_npy(path: Path, channels: int) -> torch.Tensor:
    """读取单数组 NPY；通道转换由统一后处理完成。"""

    # 解码层不使用目标通道数，显式删除避免误导读者。
    del channels
    # allow_pickle=False 禁止清单文件执行任意 Python 对象反序列化。
    return torch.from_numpy(np.load(path, allow_pickle=False))


def _decode_npz(path: Path, channels: int) -> torch.Tensor:
    """读取 NPZ 中的第一个数组。"""

    del channels
    # 上下文管理器及时关闭压缩归档句柄。
    with np.load(path, allow_pickle=False) as payload:
        # 空归档无法构造图像。
        if not payload.files:
            raise ValueError(f"NPZ 中没有数组：{path}")
        # 简单约定使用第一个数组；复杂多字段格式可通过注册自定义解码器处理。
        return torch.from_numpy(payload[payload.files[0]])


def _decode_torch(path: Path, channels: int) -> torch.Tensor:
    """读取 Torch 张量或常见字典封装的图像。"""

    del channels
    # weights_only=False 是为了兼容合法的 NumPy/字典载荷；仅应加载用户自己的数据文件。
    return _extract_tensor_payload(
        torch.load(path, map_location="cpu", weights_only=False), path
    )


def _decode_dicom(path: Path, channels: int) -> torch.Tensor:
    """按 DICOM rescale slope/intercept 解码二维像素强度。"""

    del channels
    # pydicom 是可选依赖，只有实际遇到 DICOM 文件才要求安装。
    try:
        import pydicom
    except ImportError as error:
        raise ImportError("读取 DICOM 需要安装可选依赖 pydicom") from error
    # dcmread 读取数据集元信息与像素字节。
    dataset = pydicom.dcmread(str(path))
    # pixel_array 由 pydicom 根据 Transfer Syntax 解压成 NumPy 数组。
    tensor = torch.from_numpy(np.asarray(dataset.pixel_array))
    # CT 等模态需要 slope/intercept 才能还原物理强度；缺失时按恒等变换。
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return tensor.float().mul(slope).add(intercept)


# 模块导入时注册内置格式；外部项目仍可在 build_data_bundle 前覆盖或新增。
register_image_decoder(RASTER_EXTENSIONS, _decode_raster)
register_image_decoder(".npy", _decode_npy)
register_image_decoder(".npz", _decode_npz)
register_image_decoder(TENSOR_EXTENSIONS, _decode_torch)
register_image_decoder(DICOM_EXTENSIONS, _decode_dicom)


def _decode_raw(path: Path, channels: int) -> torch.Tensor:
    """通过解码器注册表读取原始数据，尚不改变尺寸和强度范围。"""

    # 扩展名统一小写后查注册表。
    decoder = _DECODER_REGISTRY.get(path.suffix.lower())
    # 清单可指向扫描阶段未验证的扩展名，因此这里仍需防御性检查。
    if decoder is None:
        raise ValueError(f"不支持的图像扩展名：{path}")
    # 目标通道数传给可能需要控制解码模式的自定义解码器。
    return decoder(path, int(channels))


def _to_chw(tensor: torch.Tensor) -> torch.Tensor:
    """把二维、HWC 或 CHW 输入统一为 CHW，并移除无意义单例维。"""

    # squeeze 兼容 [1,H,W,1] 等预处理工具留下的单例维度。
    tensor = tensor.squeeze()
    # 灰度二维数组增加通道轴，得到 [1,H,W]。
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    # 本项目处理二维切片，不在这里隐式选择三维体数据的切片方向。
    if tensor.ndim != 3:
        raise ValueError(f"图像必须是二维或三维数组，实际形状={tuple(tensor.shape)}")
    # 最后一维像通道且第一维不像通道时，将 HWC 转为 CHW。
    if tensor.shape[-1] in {1, 3, 4} and tensor.shape[0] not in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    # 已是 CHW 的张量原样返回；歧义小尺寸由用户自定义解码器处理。
    return tensor


def _robust_minmax(tensor: torch.Tensor, percentiles: Sequence[float]) -> torch.Tensor:
    """用可配置分位数裁剪离群值，再线性缩放到 [0,1]。"""

    # 分位数运算统一使用 float32，并在全通道/空间范围上估计强度窗。
    flat = tensor.float().flatten()
    # 空数组无法计算 quantile。
    if flat.numel() == 0:
        raise ValueError("图像数组为空")
    # YAML 使用直观百分比，例如 [0.5,99.5]；torch.quantile 使用 [0,1]。
    low_q, high_q = float(percentiles[0]) / 100.0, float(percentiles[1]) / 100.0
    # 计算低/高窗位。
    low = torch.quantile(flat, low_q)
    high = torch.quantile(flat, high_q)
    # 常量图像没有可缩放动态范围，返回全零避免 NaN。
    if float(high - low) < 1.0e-8:
        return torch.zeros_like(tensor, dtype=torch.float32)
    # 先裁剪离群值，再把 [low,high] 映射到 [0,1]。
    return tensor.float().clamp(low, high).sub(low).div(high - low)


def _scale_intensity(
    tensor: torch.Tensor, mode: str, percentiles: Sequence[float]
) -> torch.Tensor:
    """根据 preserve/minmax/auto 策略把任意医学强度映射到 [0,1]。"""

    # 原 dtype 用于 auto 判断整数的理论范围，之后运算统一 float32。
    original_dtype = tensor.dtype
    tensor = tensor.float()
    # 配置不区分大小写。
    mode = str(mode).lower()
    # preserve 假定输入已经归一化，只做安全裁剪。
    if mode == "preserve":
        return tensor.clamp(0.0, 1.0)
    # minmax 始终使用稳健分位窗，不根据原始 dtype 猜测。
    if mode == "minmax":
        return _robust_minmax(tensor, percentiles)
    # 其他未知策略应明确失败，避免静默改变医学窗宽。
    if mode != "auto":
        raise ValueError(f"未知 intensity_mode：{mode}")
    # 整数图像可用 dtype 判断是普通无符号栅格还是有符号医学强度。
    if not original_dtype.is_floating_point:
        info = torch.iinfo(original_dtype)
        if info.min < 0:
            # 有符号整型通常是 CT/MR 等医学强度，按整个 int16 范围缩放会严重压扁对比度。
            return _robust_minmax(tensor, percentiles)
        # 无符号 8/16 位图像按完整 dtype 范围线性归一化。
        return tensor.sub(float(info.min)).div(float(info.max - info.min)).clamp(0.0, 1.0)
    # 浮点图先观察实际最小/最大值，识别常见的 [0,1] 与 [0,255]。
    minimum, maximum = float(tensor.min()), float(tensor.max())
    # 容许极小浮点误差后直接裁剪已经归一化的数据。
    if minimum >= -1.0e-6 and maximum <= 1.0 + 1.0e-6:
        return tensor.clamp(0.0, 1.0)
    # 非负且不超过 255 的浮点图通常由 uint8 转换而来。
    if minimum >= 0.0 and maximum <= 255.0 + 1.0e-3:
        return tensor.div(255.0).clamp(0.0, 1.0)
    # CT/MR 或其他任意浮点范围回退到稳健分位窗。
    return _robust_minmax(tensor, percentiles)


def _convert_channels(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    """在 1/3 通道之间确定性转换，并丢弃可选 alpha/额外通道。"""

    # _to_chw 已保证第 0 维是当前通道数。
    current = int(tensor.shape[0])
    # 已满足目标时不复制。
    if current == channels:
        return tensor
    # 单通道医学图复制到 RGB，使 ImageNet 风格分类网络和 VAE 共用 3 通道接口。
    if channels == 3 and current == 1:
        return tensor.repeat(3, 1, 1)
    # RGB 转灰度使用标准亮度系数，只取前三个颜色通道。
    if channels == 1 and current >= 3:
        rgb = tensor[:3]
        return (0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]).unsqueeze(0)
    # RGBA 或多通道图转 RGB 时丢弃 alpha/额外通道。
    if channels == 3 and current >= 3:
        return tensor[:3]
    # 其他光谱/体数据通道语义不应由通用层猜测，应注册专用解码器。
    raise ValueError(f"无法把 {current} 通道图像转换为 {channels} 通道")


class TensorSpatialTransform:
    """对已解码张量执行统一 resize 与可选的保守医学图像增强。"""

    def __init__(self, size: tuple[int, int], augmentation: Mapping[str, Any] | None = None):
        # size 顺序固定为 (height,width)，由全局 data.image 配置提供。
        self.size = tuple(map(int, size))
        # 验证/测试集传 None，转换为空字典后默认关闭随机增强。
        self.augmentation = dict(augmentation or {})

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """将单张 CHW 图像转换为固定尺寸，并仅在训练集执行随机增强。"""

        # 先统一尺寸；BICUBIC+antialias 对缩小高分辨率医学图更平滑。
        image = TF.resize(image, list(self.size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        # 验证/测试必须确定性，仅裁剪范围并确保连续内存布局。
        if not bool(self.augmentation.get("enabled", False)):
            return image.clamp(0.0, 1.0).contiguous()
        # 水平翻转按单图采样；对方向敏感的数据集可在 YAML 设为 0。
        if torch.rand(()) < float(self.augmentation.get("horizontal_flip_probability", 0.5)):
            image = TF.hflip(image)
        # 读取空间增强范围，默认 0 旋转/平移与 1 倍缩放。
        degrees = float(self.augmentation.get("rotation_degrees", 0.0))
        translate_ratio = float(self.augmentation.get("translate_ratio", 0.0))
        scale_range = list(self.augmentation.get("scale_range", [1.0, 1.0]))
        # 角度在对称区间内均匀采样；关闭时不消耗随机数。
        angle = float(torch.empty(()).uniform_(-degrees, degrees)) if degrees else 0.0
        # 比例平移转换成当前目标尺寸下的最大整数像素数。
        maximum_dx = int(round(self.size[1] * translate_ratio))
        maximum_dy = int(round(self.size[0] * translate_ratio))
        # x/y 独立采样整数位移，范围包含两端。
        translate = [
            int(torch.randint(-maximum_dx, maximum_dx + 1, ()).item()) if maximum_dx else 0,
            int(torch.randint(-maximum_dy, maximum_dy + 1, ()).item()) if maximum_dy else 0,
        ]
        # 各向同性缩放在 YAML 上下界内均匀采样。
        scale = float(torch.empty(()).uniform_(float(scale_range[0]), float(scale_range[1])))
        # 一次 affine 合并旋转、平移和缩放，减少重复插值；不加入任务特定剪切。
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        # 颜色增强幅度同样来自数据配置，可对不同模态单独关闭。
        brightness = float(self.augmentation.get("brightness", 0.0))
        contrast = float(self.augmentation.get("contrast", 0.0))
        # torchvision brightness 的 factor=1 表示不变，因此围绕 1 采样。
        if brightness:
            factor = 1.0 + float(torch.empty(()).uniform_(-brightness, brightness))
            image = TF.adjust_brightness(image, factor)
        # 对比度也围绕 factor=1 对称采样。
        if contrast:
            factor = 1.0 + float(torch.empty(()).uniform_(-contrast, contrast))
            image = TF.adjust_contrast(image, factor)
        # 插值/颜色变换可能轻微越界，最终恢复 [0,1] 且连续存储。
        return image.clamp(0.0, 1.0).contiguous()


class MedicalImageDataset(Dataset):
    """基于记录和解码器注册表的通用二维医学图像数据集。"""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        size: tuple[int, int],
        channels: int,
        intensity_mode: str,
        percentile_range: Sequence[float],
        augmentation: Mapping[str, Any] | None = None,
    ):
        # 复制记录序列，避免调用方后续原地修改影响 Dataset。
        self.records = list(records)
        # targets 是 WeightedRandomSampler 与 ClassImagePool 使用的轻量标签索引。
        self.targets = [record.label for record in self.records]
        # 目标通道、强度策略和分位范围统一应用到每种文件格式。
        self.channels = int(channels)
        self.intensity_mode = str(intensity_mode)
        self.percentile_range = tuple(map(float, percentile_range))
        # 空间尺寸/增强单独封装，验证测试传 augmentation=None。
        self.transform = TensorSpatialTransform(size, augmentation)

    def __len__(self) -> int:
        """返回索引记录数，不提前解码图像。"""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按需解码一张图，并返回训练张量与可追踪元数据。"""

        # DataLoader 索引可能是 NumPy 整数，先规范成 Python int。
        record = self.records[int(index)]
        # 第一步只按扩展名解码原始数组，不改变其医学强度。
        raw = _decode_raw(record.path, self.channels)
        # 依次统一 CHW、[0,1] 强度和目标通道数。
        image = _convert_channels(
            _scale_intensity(_to_chw(raw), self.intensity_mode, self.percentile_range),
            self.channels,
        )
        # 返回字典便于添加 key/path；unpack_batch 会给训练代码统一取 image/label。
        return {
            # resize/增强最后执行，输出 float32 [C,H,W] 且范围 [0,1]。
            "image": self.transform(image),
            # 分类损失要求 long 标量标签，DataLoader 会堆叠为 [B]。
            "label": torch.tensor(record.label, dtype=torch.long),
            # key/path 仅用于调试和追踪，不会送入网络。
            "key": record.key,
            "path": str(record.path),
        }


def _verify_records(records: Sequence[SampleRecord], split: str, num_classes: int) -> None:
    """验证划分非空、类别齐全且所有记录文件存在。"""

    # 空训练/验证/测试集都无法完成本项目的标准流程。
    if not records:
        raise ValueError(f"{split} 数据集为空")
    # 每个划分都要求覆盖全部类别，Balanced Accuracy 才有稳定意义。
    present = {record.label for record in records}
    missing = sorted(set(range(num_classes)) - present)
    if missing:
        raise ValueError(f"{split} 缺少类别编号：{missing}")
    # 清单路径在真正训练前一次性检查，避免跑数小时后才遇到坏行。
    missing_files = [str(record.path) for record in records if not record.path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"{split} 有 {len(missing_files)} 个文件不存在，例如：{missing_files[0]}")


def build_data_bundle(
    config: Mapping[str, Any],
    train_augmentation: Mapping[str, Any] | None = None,
) -> DataBundle:
    """按适配器构造 train/val/test，并允许训练阶段覆盖全局增强配置。"""

    # data_config 包含格式适配、路径、图像预处理和 DataLoader 设置。
    data_config = config["data"]
    # folder 使用目录类别；其他值统一进入 manifest 适配器并由配置校验限制。
    adapter = str(data_config.get("adapter", "folder")).lower()
    # 数据根目录相对全局项目根解析，也允许显式绝对路径。
    root = resolve_path(config, data_config["root"])
    # class_names 可固定类别顺序；folder 未配置时从训练目录发现。
    configured_names = data_config.get("class_names")
    if adapter == "folder":
        # 目录结构：root/{train,val,test}/{class}/任意子目录/图像。
        class_names = [str(value) for value in configured_names] if configured_names else _discover_folder_classes(root, str(data_config.get("train_split", "train")))
        train_records = _folder_records(root, str(data_config.get("train_split", "train")), class_names, False)
        val_records = _folder_records(root, str(data_config.get("val_split", "val")), class_names, True)
        test_records = _folder_records(root, str(data_config.get("test_split", "test")), class_names, False)
    else:
        # 清单适配器通过可配置列名读取每个划分。
        class_names = _manifest_class_names(config)
        train_records = _manifest_records(config, "train", class_names)
        val_records = _manifest_records(config, "val", class_names)
        test_records = _manifest_records(config, "test", class_names)
    # 只有验证集允许缺失；此时从训练集按类别分层切出，不触碰测试集。
    if not val_records:
        validation = data_config.get("validation", {})
        train_records, val_records = _stratified_split(
            train_records,
            float(validation.get("ratio", 0.1)),
            int(validation.get("seed", config["project"].get("seed", 0))),
        )
    # 构建 Dataset 前统一验证三个划分，错误尽早暴露。
    for split, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        _verify_records(records, split, len(class_names))
    # 三个划分共享尺寸、通道和强度处理，保证输入定义一致。
    common = {
        "size": image_size(config),
        "channels": int(data_config["image"].get("channels", 3)),
        "intensity_mode": str(data_config["image"].get("intensity_mode", "auto")),
        "percentile_range": data_config["image"].get("percentile_range", [0.5, 99.5]),
    }
    # 默认读取全局增强；扩散等阶段可传入独立配置而不影响分类训练。
    augmentation = data_config.get("augmentation", {}) if train_augmentation is None else train_augmentation
    train = MedicalImageDataset(train_records, augmentation=augmentation, **common)
    val = MedicalImageDataset(val_records, augmentation=None, **common)
    test = MedicalImageDataset(test_records, augmentation=None, **common)
    # 同时返回正向名称列表和反向映射，所有阶段复用同一标签空间。
    return DataBundle(train, val, test, class_names, {name: index for index, name in enumerate(class_names)})


def _worker_init(worker_id: int) -> None:
    """让 Python/NumPy 的 worker 随机状态跟随 Torch DataLoader 种子。"""

    # Torch 会为每个 worker 设置不同 initial_seed；压到 NumPy 支持的 32 位范围。
    worker_seed = torch.initial_seed() % (2**32)
    # 再加 worker_id 明确区分多个进程中的 Python/NumPy 随机序列。
    random.seed(worker_seed + int(worker_id))
    np.random.seed(worker_seed + int(worker_id))


def build_loader(
    dataset: Dataset,
    config: Mapping[str, Any],
    train: bool,
    batch_size: int | None = None,
    balanced: bool = False,
    samples_per_epoch: int | None = None,
) -> DataLoader:
    """统一创建 DataLoader；可对不平衡训练集使用有放回加权采样。"""

    # project 控制进程/内存行为，data.loader 控制批次行为。
    project = config["project"]
    loader_config = config["data"].get("loader", {})
    # 调用方显式 batch_size 优先，否则按 train/eval 读取不同默认值。
    batch_size = int(batch_size or loader_config.get("train_batch_size" if train else "eval_batch_size", 64))
    # 负 worker 数没有意义，安全截断到单进程加载。
    # Windows 使用 spawn 创建 worker；每个进程都会重新导入 PyTorch/MONAI，
    # 高分辨率训练时容易瞬间占满内存。允许单独配置，默认退回单进程加载。
    worker_key = "windows_num_workers" if sys.platform == "win32" else "num_workers"
    workers = max(0, int(project.get(worker_key, 0)))
    # sampler=None 时由 shuffle 控制顺序；平衡采样会替代二者。
    sampler = None
    shuffle = bool(train)
    # balanced 仅应用到需要类别均衡的真实训练采样，不用于验证/测试。
    if balanced:
        # 普通 MedicalImageDataset 直接暴露 targets。
        targets = list(getattr(dataset, "targets", []))
        # 若调用方传 Subset，则按其 indices 从父数据集提取对应标签。
        if not targets and isinstance(dataset, Subset):
            parent_targets = getattr(dataset.dataset, "targets", [])
            targets = [parent_targets[index] for index in dataset.indices]
        # bincount 得到每类样本量；配置校验已保证标签连续非负。
        counts = torch.bincount(torch.tensor(targets, dtype=torch.long))
        # 每个样本权重是其类别频数倒数，使类别被抽中的总概率近似相等。
        weights = torch.tensor([1.0 / max(1, int(counts[label])) for label in targets], dtype=torch.double)
        # 有放回抽样允许少数类在一个 epoch 中重复出现。
        sampler = WeightedRandomSampler(
            weights,
            # samples_per_epoch 可控制在线队列每个“真实训练周期”的总步数。
            num_samples=int(samples_per_epoch or len(weights)),
            replacement=True,
        )
        # sampler 与 shuffle 互斥。
        shuffle = False
    # 所有阶段通过同一构造器获得一致 worker、pin memory 和 drop_last 行为。
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(project.get("pin_memory", True)) and torch.cuda.is_available(),
        "persistent_workers": bool(project.get("persistent_workers", True)) and workers > 0,
        "drop_last": bool(loader_config.get("drop_last", False)) and train,
        "worker_init_fn": _worker_init if workers > 0 else None,
    }
    # 多进程加载时预先准备更多 batch，隐藏图片解码、resize 与随机增强耗时。
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(
            1, int(project.get("prefetch_factor", 2))
        )
    return DataLoader(**loader_kwargs)


class ClassImagePool:
    """为分布匹配按类别随机读取真实图像，不把整个医学数据集塞进内存。"""

    def __init__(
        self,
        dataset: MedicalImageDataset,
        num_classes: int,
        seed: int,
        cache_images: bool = False,
    ):
        # 只保存 Dataset 引用，不缓存解码后的高分辨率图像。
        self.dataset = dataset
        # 为每个类别预建索引列表，后续采样不再扫描整个数据集。
        self.indices: dict[int, list[int]] = {class_id: [] for class_id in range(int(num_classes))}
        for index, label in enumerate(dataset.targets):
            self.indices[int(label)].append(index)
        # 分布匹配要求每个类别都能提供真实目标。
        if any(not values for values in self.indices.values()):
            raise ValueError("ClassImagePool 要求训练集每个类别至少有一个样本")
        # 使用独立 torch.Generator，蒸馏断点可精确保存其状态而不依赖全局 RNG。
        self.generator = torch.Generator().manual_seed(int(seed))
        # 凝聚会反复采样同一真实集；float16 CPU 缓存避免每次重新解码和 resize。
        self.cache_images = bool(cache_images)
        self._image_cache: dict[int, torch.Tensor] = {}

    def _image(self, index: int) -> torch.Tensor:
        """读取一张确定性基础图像，并按需缓存为 CPU float16。"""

        normalized_index = int(index)
        cached = self._image_cache.get(normalized_index)
        if cached is None:
            image = self.dataset[normalized_index]["image"]
            if not self.cache_images:
                return image
            cached = image.detach().to(device="cpu", dtype=torch.float16).contiguous()
            self._image_cache[normalized_index] = cached
        return cached

    def sample(self, class_id: int, count: int) -> torch.Tensor:
        """从指定类别有放回采样 ``count`` 张真实图像。"""

        # candidates 保存该类在 Dataset 中的原始索引。
        candidates = self.indices[int(class_id)]
        # randint 有放回采样，允许 IPC/真实 batch 大于少数类样本数。
        positions = torch.randint(len(candidates), (int(count),), generator=self.generator).tolist()
        # Dataset 按需解码并增强，最后堆叠为 [B,C,H,W] CPU 张量。
        # 缓存以 float16 节省主存，拼接后一次性恢复 float32，减少逐图转换开销。
        return torch.stack([self._image(candidates[position]) for position in positions]).float()

    def sample_all_classes(self, count_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        """为每个类别等量采样，并返回类别连续排列的图像与标签。"""

        # 分开累积后一次 cat，避免循环中反复重新分配大张量。
        images, labels = [], []
        # 排序类别保证输出顺序为 0...C-1。
        for class_id in sorted(self.indices):
            batch = self.sample(class_id, count_per_class)
            images.append(batch)
            # 标签数量使用实际 batch 大小，避免未来自定义采样器改变返回数量。
            labels.append(torch.full((batch.shape[0],), class_id, dtype=torch.long))
        return torch.cat(images), torch.cat(labels)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """保存独立类别采样器状态，用于迭代级精确续训。"""

        # Dataset 索引由当前配置重建，只需保存随机生成器状态。
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """若断点包含合法生成器状态则恢复，否则继续使用当前种子。"""

        # 兼容旧断点没有 class_pool 字段，并确保状态是张量。
        if state and torch.is_tensor(state.get("generator_state")):
            # torch.Generator 位于 CPU，载入前显式移动状态张量。
            self.generator.set_state(state["generator_state"].cpu())


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """兼容本项目字典 batch 和常见的 ``(image,label)`` batch。"""

    # 通用医学 Dataset 返回字典，保留 key/path 等元数据但训练阶段只取两项。
    if isinstance(batch, Mapping):
        return batch["image"], batch["label"]
    # 外部 Dataset 常用 tuple/list，按前两项解释图像和标签。
    return batch[0], batch[1]
