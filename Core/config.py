"""多文件 YAML 配置的读取、合并、命令行覆盖、校验与路径解析。

配置入口是 :mod:`configs/global.yaml`。该文件只保存跨阶段共享参数，并通过
``stage_configs`` 指向 Autoencoder、Diffusion、Condensation 与 Evaluation 配置。
加载顺序固定为“全局文件 → 各阶段文件 → 命令行/测试覆盖”，因此
``--set condensation.iterations.1=5000`` 始终拥有最高优先级。

断点恢复故意不计算配置、代码或数据哈希。用户修改任何可兼容参数后，只要继续指向
原来的 ``project.run_dir``，阶段代码就会读取最新权重并按照当前目标轮数继续训练。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml
from yaml.constructor import ConstructorError


# PROJECT_ROOT 是路径解析的唯一项目基准，避免从不同终端目录启动时得到不同路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 默认入口改为全局配置；各阶段文件由其中的 stage_configs 自动加载。
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "global.yaml"
# 四种分类器仍用于下游跨架构评估；condense 主方法只使用其中的 ConvNet。
SUPPORTED_CLASSIFIERS = {"convnet", "resnet18", "convnext_tiny", "vit_tiny"}
# 静态专家和额外对照阶段均不再注册；在线 IDM 只存在于 condense 阶段内部。
SUPPORTED_STAGES = {"train_autoencoder", "train_diffusion", "condense", "evaluate"}


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝单个 YAML 文件中的重复键，防止后写参数静默覆盖前写参数。"""


def _construct_unique_mapping(loader, node, deep: bool = False) -> dict:
    """把 YAML mapping 转成字典，并在发现重复键时报告准确文件位置。"""

    # 使用普通字典保留 YAML 原始顺序，便于日志与调试输出保持可读。
    mapping: dict = {}
    # node.value 中每项都是尚未构造的键节点和值节点。
    for key_node, value_node in node.value:
        # 先构造键，才能在插入前检查它是否已经出现。
        key = loader.construct_object(key_node, deep=deep)
        # 重复键通常意味着复制配置时忘记删除旧值，应直接阻止运行。
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"配置键重复：{key!r}",
                key_node.start_mark,
            )
        # 只有键唯一时才继续构造并保存对应值。
        mapping[key] = loader.construct_object(value_node, deep=deep)
    # 返回已经完成重复键检查的映射。
    return mapping


# 把自定义 mapping 构造器注册到 SafeLoader，其余 YAML 类型仍使用安全默认实现。
_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    """以 UTF-8 读取一个 YAML 文件，并要求其顶层必须是字典。"""

    # 提前检查路径可以给出比 yaml.open 更直观的错误信息。
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    # 所有配置和中文注释统一使用 UTF-8，避免 Windows 默认编码造成乱码。
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    # 阶段配置必须使用命名节点，禁止把列表或单个标量作为整个文件内容。
    if not isinstance(loaded, Mapping):
        raise TypeError(f"YAML 顶层必须是字典：{path}")
    # 深复制成普通 dict，避免把 YAML loader 内部对象传播到运行代码。
    return deepcopy(dict(loaded))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并字典；叶子值以 ``override`` 为准且不修改任何调用方对象。"""

    # 从基础配置的深复制开始，保证加载后的配置可安全修改。
    result = deepcopy(dict(base))
    # 逐项处理覆盖配置。
    for key, value in override.items():
        # 两侧都是映射时继续递归，使覆盖一个学习率不会删除同节点其他参数。
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            # 列表、标量或类型改变时整体替换该叶子。
            result[key] = deepcopy(value)
    # 返回新字典，基础配置和覆盖配置保持原样。
    return result


def set_by_dotted_key(config: MutableMapping[str, Any], key: str, value: Any) -> None:
    """设置 ``a.b.c`` 形式的嵌套键，供命令行 ``--set`` 使用。"""

    # 删除意外的空片段，例如开头或结尾的点。
    parts = [part for part in str(key).split(".") if part]
    # 空键没有明确目标，必须拒绝。
    if not parts:
        raise ValueError("配置覆盖键不能为空")
    # cursor 从配置根节点开始逐层向下定位。
    cursor: MutableMapping[str, Any] = config
    # 最后一段由循环结束后写入，前面各段必须是字典。
    for part in parts[:-1]:
        # 不存在的中间节点按字典创建，便于测试覆盖新增可选项。
        child = cursor.setdefault(part, {})
        # 若中间节点已经是标量，继续下钻会产生含义冲突。
        if not isinstance(child, MutableMapping):
            raise TypeError(f"无法在非字典配置 {part!r} 下继续设置 {key!r}")
        # 移动到下一层。
        cursor = child
    # 写入最终叶子值。
    cursor[parts[-1]] = value


def parse_cli_overrides(values: list[str] | None) -> dict[str, Any]:
    """解析若干 ``键=YAML值``，例如 ``project.device=cuda:1``。"""

    # 所有表达式最终合并到一个嵌套覆盖字典。
    result: dict[str, Any] = {}
    # argparse 未收到 --set 时 values 可能是 None。
    for expression in values or []:
        # 等号用于分隔点路径与 YAML 值，缺失时无法可靠解释。
        if "=" not in expression:
            raise ValueError(f"--set 必须使用 键=值 格式：{expression!r}")
        # 只切第一个等号，字符串值内部仍可包含等号。
        key, raw_value = expression.split("=", 1)
        # 使用 YAML 解析可自然支持整数、浮点、布尔、null 和列表。
        value = yaml.safe_load(raw_value)
        # 把点路径写入嵌套结果。
        set_by_dotted_key(result, key.strip(), value)
    # 返回可直接传给 load_config 的覆盖字典。
    return result


def image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    """把整数或 ``[高度, 宽度]`` 图像尺寸统一为二元整数元组。"""

    # 默认尺寸维持项目约定的 224×224。
    value = config["data"]["image"].get("size", [224, 224])
    # 单个整数代表正方形图像。
    if isinstance(value, int):
        return int(value), int(value)
    # 非二项序列无法唯一解释为二维图像尺寸。
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("data.image.size 必须是正整数或 [高度, 宽度]")
    # 明确转换为 int，避免 YAML 中的浮点尺寸悄悄流入网络。
    return int(value[0]), int(value[1])


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """解析用户路径；相对路径始终相对于项目根目录，而非当前终端目录。"""

    # expanduser 允许用户显式提供个人目录路径，但项目默认不会使用它。
    path = Path(value).expanduser()
    # 只有相对路径需要拼接固定项目根目录。
    if not path.is_absolute():
        path = Path(config["_runtime"]["project_root"]) / path
    # resolve 消除 ``..`` 并得到日志中清晰的绝对路径。
    return path.resolve()


def run_dir(config: Mapping[str, Any]) -> Path:
    """返回当前实验的固定输出根目录。"""

    return resolve_path(config, config["project"]["run_dir"])


def stage_dir(config: Mapping[str, Any], stage_name: str, create: bool = True) -> Path:
    """返回阶段输出目录，并按需创建；阶段间只通过这些目录交换产物。"""

    # 阶段名直接作为 run_dir 下一级目录，保持产物结构可预测。
    directory = run_dir(config) / str(stage_name)
    # 只读调用可传 create=False，避免查询操作意外创建空目录。
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    # 返回 Path 供调用方继续拼接具体文件名。
    return directory


def _require_positive(section: Mapping[str, Any], keys: tuple[str, ...], name: str) -> None:
    """验证一组必需数值都严格大于零。"""

    for key in keys:
        if float(section.get(key, 0)) <= 0:
            raise ValueError(f"{name}.{key} 必须大于 0")


def _validate_early_stopping(settings: Mapping[str, Any], prefix: str) -> None:
    """校验各阶段共用的早停基本字段，阶段特定间隔由调用处校验。"""

    # 关闭早停时仍校验 mode，避免以后启用才暴露拼写错误。
    if str(settings.get("mode", "min")).lower() not in {"min", "max"}:
        raise ValueError(f"{prefix}.mode 只能是 min 或 max")
    # patience 至少为 1，否则第一次检查就会立即停止。
    if int(settings.get("patience_checks", 1)) <= 0:
        raise ValueError(f"{prefix}.patience_checks 必须大于 0")
    # min_delta 是绝对改善阈值，负数会反转早停语义。
    if float(settings.get("min_delta", 0.0)) < 0:
        raise ValueError(f"{prefix}.min_delta 不能为负数")


def _validate(config: Mapping[str, Any]) -> None:
    """对跨文件合并后的完整配置执行集中、可读的启动前校验。"""

    # 这些节点分别来自全局文件和四个阶段文件，缺失通常表示 stage_configs 路径写错。
    required = {
        "stage_configs",
        "experiment_configs",
        "project",
        "data",
        "models",
        "autoencoder",
        "diffusion",
        "condensation",
        "evaluation",
        "pipeline",
        "ablation",
    }
    # 先报告所有缺失节点，避免用户逐次修复一个。
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"合并配置缺少顶层节点：{missing}")

    # 校验默认阶段顺序只包含入口实际支持的阶段。
    enabled_stages = [str(value) for value in config["pipeline"].get("enabled_stages", [])]
    unknown_stages = set(enabled_stages) - SUPPORTED_STAGES
    if unknown_stages:
        raise ValueError(f"pipeline.enabled_stages 含未知阶段：{sorted(unknown_stages)}")
    # train_experts 已经被删除，若旧配置仍引用它应明确报错。
    if "train_experts" in enabled_stages:
        raise ValueError("静态 train_experts 已移除，请从 pipeline.enabled_stages 删除")

    # 图像高宽必须为正整数。
    size = image_size(config)
    if min(size) <= 0:
        raise ValueError("data.image.size 中的高度和宽度必须为正数")
    # 当前数据读取和模型入口明确支持灰度或 RGB。
    channels = int(config["data"]["image"].get("channels", 3))
    if channels not in {1, 3}:
        raise ValueError("data.image.channels 当前只支持 1 或 3")
    # 分类归一化向量长度必须与输入通道一致。
    normalization = config["data"].get("classifier_normalization", {})
    if len(normalization.get("mean", [])) != channels or len(normalization.get("std", [])) != channels:
        raise ValueError("classifier_normalization 的 mean/std 长度必须等于图像通道数")
    # 标准差为零或负数会产生 Inf/翻转尺度。
    if any(float(value) <= 0 for value in normalization.get("std", [])):
        raise ValueError("classifier_normalization.std 必须全部大于 0")
    # 显式类别名必须至少两个且不能重复。
    class_names = config["data"].get("class_names")
    if class_names is not None:
        normalized_names = [str(value) for value in class_names]
        if len(normalized_names) < 2 or len(set(normalized_names)) != len(normalized_names):
            raise ValueError("data.class_names 至少包含两个互不重复的类别，或设为 null")
    # 自动验证集比例必须真正切出训练和验证两部分。
    ratio = float(config["data"].get("validation", {}).get("ratio", 0.1))
    if not 0.0 < ratio < 1.0:
        raise ValueError("data.validation.ratio 必须位于 (0,1)")
    # 数据适配器只接受已实现的目录和清单两种形式。
    adapter = str(config["data"].get("adapter", "folder")).lower()
    if adapter not in {"folder", "manifest"}:
        raise ValueError("data.adapter 只能是 folder 或 manifest")

    # 所有分类网络必须从随机初始化开始，防止无意引入 ImageNet 权重。
    if bool(config["models"].get("pretrained", False)):
        raise ValueError("本项目要求分类网络从零训练，models.pretrained 必须为 false")
    # 校验 ConvNet 至少三层并能提供浅、中、深特征。
    convnet = config["models"].get("definitions", {}).get("convnet", {})
    convnet_widths = [int(value) for value in convnet.get("widths", [])]
    if len(convnet_widths) < 3 or any(value <= 0 for value in convnet_widths):
        raise ValueError("models.definitions.convnet.widths 至少包含三个正整数")
    # 校验可调 ResNet 的四阶段宽度与块数。
    resnet = config["models"].get("definitions", {}).get("resnet18", {})
    if len(resnet.get("stage_widths", [])) != 4 or len(resnet.get("stage_blocks", [])) != 4:
        raise ValueError("resnet18.stage_widths 和 stage_blocks 必须各有四项")
    # 校验可调 ConvNeXt 的四个 stage 参数。
    convnext = config["models"].get("definitions", {}).get("convnext_tiny", {})
    if bool(convnext.get("custom", True)) and (
        len(convnext.get("depths", [])) != 4 or len(convnext.get("dims", [])) != 4
    ):
        raise ValueError("自定义 ConvNeXt 的 depths 和 dims 必须各有四项")
    # ViT patch 必须整除当前图像高宽，注意力头数必须整除 token 维度。
    vit = config["models"].get("definitions", {}).get("vit_tiny", {})
    patch_size = int(vit.get("patch_size", 16))
    if patch_size <= 0 or any(dimension % patch_size for dimension in size):
        raise ValueError("图像高宽必须能被 vit_tiny.patch_size 整除")
    if bool(vit.get("custom", True)) and int(vit.get("embed_dim", 0)) % int(vit.get("num_heads", 1)):
        raise ValueError("vit_tiny.embed_dim 必须能被 num_heads 整除")

    # Autoencoder 三个逐层列表必须长度一致，通道必须兼容 GroupNorm。
    autoencoder = config["autoencoder"]
    ae_channels = [int(value) for value in autoencoder.get("channels", [])]
    ae_blocks = list(autoencoder.get("num_res_blocks", []))
    ae_attention = list(autoencoder.get("attention_levels", []))
    if not ae_channels or len(ae_channels) != len(ae_blocks) or len(ae_channels) != len(ae_attention):
        raise ValueError("autoencoder.channels/num_res_blocks/attention_levels 长度必须一致")
    ae_groups = int(autoencoder.get("norm_num_groups", 32))
    if ae_groups <= 0 or any(value % ae_groups for value in ae_channels):
        raise ValueError("autoencoder.channels 每项必须能被 norm_num_groups 整除")
    # VAE 每增加一个通道阶段就多一次 2 倍下采样，图像尺寸必须整除总倍率。
    latent_factor = 2 ** (len(ae_channels) - 1)
    if any(dimension % latent_factor for dimension in size):
        raise ValueError(f"图像高宽必须能被 VAE 下采样倍率 {latent_factor} 整除")
    _require_positive(autoencoder, ("epochs", "batch_size", "latent_channels"), "autoencoder")
    # KL 使用分辨率无关的 mean reduction；目标权重可设 0 做无 KL 消融。
    kl_weight = float(autoencoder.get("kl_weight", 1.0e-3))
    kl_start_weight = float(autoencoder.get("kl_warmup_start_weight", 0.0))
    kl_warmup_epochs = int(autoencoder.get("kl_warmup_epochs", 0))
    if kl_weight < 0.0:
        raise ValueError("autoencoder.kl_weight 不能为负数")
    if kl_start_weight < 0.0 or kl_start_weight > kl_weight:
        raise ValueError(
            "autoencoder.kl_warmup_start_weight 必须位于 [0, kl_weight]"
        )
    if kl_warmup_epochs < 0:
        raise ValueError("autoencoder.kl_warmup_epochs 不能为负数")
    _validate_early_stopping(autoencoder.get("early_stopping", {}), "autoencoder.early_stopping")

    # Diffusion 各层列表必须长度一致，并满足 GroupNorm 与注意力头整除关系。
    diffusion = config["diffusion"]
    diffusion_channels = [int(value) for value in diffusion.get("channels", [])]
    diffusion_blocks = list(diffusion.get("num_res_blocks", []))
    diffusion_attention = list(diffusion.get("attention_levels", []))
    head_channels = [int(value) for value in diffusion.get("num_head_channels", [])]
    if (
        not diffusion_channels
        or len(diffusion_channels) != len(diffusion_blocks)
        or len(diffusion_channels) != len(diffusion_attention)
        or len(diffusion_channels) != len(head_channels)
    ):
        raise ValueError("diffusion 的 channels/num_res_blocks/attention_levels/num_head_channels 长度必须一致")
    diffusion_groups = int(diffusion.get("norm_num_groups", 32))
    if diffusion_groups <= 0 or any(value % diffusion_groups for value in diffusion_channels):
        raise ValueError("diffusion.channels 每项必须能被 norm_num_groups 整除")
    if any(head <= 0 or channel % head for channel, head in zip(diffusion_channels, head_channels)):
        raise ValueError("每个 diffusion.channels 必须能被对应 num_head_channels 整除")
    if str(diffusion.get("prediction_type", "epsilon")) not in {"epsilon", "v_prediction"}:
        raise ValueError("diffusion.prediction_type 只能是 epsilon 或 v_prediction")
    _require_positive(diffusion, ("epochs", "batch_size", "train_timesteps"), "diffusion")
    _validate_early_stopping(diffusion.get("early_stopping", {}), "diffusion.early_stopping")

    # condense 已替换为标准 IDM IPC=1；扩展方法只改变图像参数化和 topology。
    condensation = config["condensation"]
    requested_ipcs = [int(value) for value in condensation.get("ipc_values", [])]
    if requested_ipcs != [1] or int(condensation.get("idm", {}).get("ipc", 0)) != 1:
        raise ValueError("标准 IDM 消融入口固定为 IPC=1")
    idm = condensation.get("idm", {})
    _require_positive(
        idm,
        (
            "iterations",
            "image_learning_rate",
            "batch_real",
            "batch_train",
            "net_num",
            "fetch_net_num",
        ),
        "condensation.idm",
    )
    if int(idm.get("partition_expansion", 0)) != 2:
        raise ValueError("condensation.idm.partition_expansion 必须为 2")
    reserved_fraction = float(
        condensation.get("memory", {}).get("max_reserved_fraction", 0)
    )
    if not 0.25 <= reserved_fraction <= 0.95:
        raise ValueError(
            "condensation.memory.max_reserved_fraction 必须在 0.25–0.85"
        )
    # 通用 topology 接受三层正整数网格和 JS/KL/MSE 散度。
    topology = condensation.get("topology", {})
    if str(topology.get("divergence", "js")).lower() not in {"js", "kl", "mse"}:
        raise ValueError("condensation.topology.divergence 只能是 js、kl 或 mse")
    for level in ("shallow", "middle", "deep"):
        grid = topology.get("grids", {}).get(level)
        if not isinstance(grid, (list, tuple)) or len(grid) != 2 or min(map(int, grid)) <= 0:
            raise ValueError(f"condensation.topology.grids.{level} 必须是正整数 [高度,宽度]")
    methods = config["ablation"].get("methods", {})
    if set(map(str, methods)) != {"C0", "C1", "C2", "C3", "C4", "C5"}:
        raise ValueError("ablation.methods 必须完整包含 C0–C5")
    if str(condensation.get("default_method", "C0")) not in methods:
        raise ValueError("condensation.default_method 必须存在于 ablation.methods")

    # 评估必须至少有一个合法架构和一次重复，并且只读取主方法 condensed 产物。
    evaluation = config["evaluation"]
    unknown_sources = set(map(str, evaluation.get("sources", ["condensed"]))) - {"condensed"}
    if unknown_sources:
        raise ValueError("evaluation.sources 目前只能包含 condensed")
    if not evaluation.get("sources"):
        raise ValueError("evaluation.sources 不能为空")
    unknown_eval = set(map(str, evaluation.get("architectures", []))) - SUPPORTED_CLASSIFIERS
    if unknown_eval:
        raise ValueError(f"evaluation.architectures 含未知网络：{sorted(unknown_eval)}")
    if not evaluation.get("architectures"):
        raise ValueError("evaluation.architectures 不能为空")
    _require_positive(evaluation, ("repeats", "batch_size", "steps_per_epoch"), "evaluation")
    snapshot_iteration = evaluation.get("snapshot_iteration")
    if snapshot_iteration is not None and (
        isinstance(snapshot_iteration, bool)
        or not isinstance(snapshot_iteration, int)
        or snapshot_iteration <= 0
    ):
        raise ValueError("evaluation.snapshot_iteration 必须是 null 或正整数")
    _validate_early_stopping(evaluation.get("early_stopping", {}), "evaluation.early_stopping")

    # 混合精度类型和 DataLoader 进程数在最后统一验证。
    amp_dtype = str(config["project"].get("amp_dtype", "bf16")).lower()
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError("project.amp_dtype 只能是 fp16 或 bf16")
    if int(config["project"].get("num_workers", 0)) < 0:
        raise ValueError("project.num_workers 不能为负数")
    if int(config["project"].get("windows_num_workers", 0)) < 0:
        raise ValueError("project.windows_num_workers 不能为负数")


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """加载全局与阶段配置，应用覆盖项，校验后附加只读运行时路径信息。"""

    # 用户提供的 --config 指向全局入口文件，而非某个单独阶段文件。
    global_path = Path(config_path).expanduser().resolve()
    # 首先读取全局配置，后续需要从其中找到阶段文件映射。
    global_config = _read_yaml(global_path)
    # stage_configs 必须是“阶段名: 文件路径”的字典。
    stage_files = global_config.get("stage_configs", {})
    if not isinstance(stage_files, Mapping) or not stage_files:
        raise KeyError(f"全局配置缺少非空 stage_configs：{global_path}")
    # 阶段文件映射只允许当前四个主阶段，防止旧独立 IDM 配置被悄悄合并。
    unknown_stage_files = set(map(str, stage_files)) - SUPPORTED_STAGES
    if unknown_stage_files:
        raise ValueError(f"stage_configs 含已删除或未知阶段：{sorted(unknown_stage_files)}")
    # merged 从全局配置开始，依次递归并入每个阶段文件。
    merged: dict[str, Any] = deepcopy(global_config)
    # config_paths 记录本次真正读取过的全部文件，写入运行清单便于人工追踪。
    config_paths: dict[str, str] = {"global": str(global_path)}
    # 相对阶段路径以全局配置所在目录为基准，而不是项目根目录或终端目录。
    for stage_name, configured_path in stage_files.items():
        stage_path = Path(str(configured_path)).expanduser()
        if not stage_path.is_absolute():
            stage_path = global_path.parent / stage_path
        stage_path = stage_path.resolve()
        # 读取并递归合并该阶段配置。
        merged = _deep_merge(merged, _read_yaml(stage_path))
        # 保存解析后的绝对路径供日志和 README 排查。
        config_paths[str(stage_name)] = str(stage_path)
    # 实验矩阵与阶段算法配置分开维护，但仍由同一个入口原子地解析。
    experiment_files = global_config.get("experiment_configs", {})
    if not isinstance(experiment_files, Mapping) or set(map(str, experiment_files)) != {
        "ablation"
    }:
        raise ValueError("experiment_configs 必须且只能包含 ablation")
    for experiment_name, configured_path in experiment_files.items():
        experiment_path = Path(str(configured_path)).expanduser()
        if not experiment_path.is_absolute():
            experiment_path = global_path.parent / experiment_path
        experiment_path = experiment_path.resolve()
        merged = _deep_merge(merged, _read_yaml(experiment_path))
        config_paths[str(experiment_name)] = str(experiment_path)
    # 命令行和测试覆盖最后应用，因此始终优先于磁盘配置。
    config = _deep_merge(merged, overrides or {})
    # _runtime 只在内存中存在，不写回任何 YAML，也不参与断点兼容检查。
    config["_runtime"] = {
        "config_path": str(global_path),
        "config_paths": config_paths,
        "project_root": str(PROJECT_ROOT),
    }
    # 在创建模型和输出目录前集中发现配置错误。
    _validate(config)
    # 返回可以传给所有阶段的完整合并字典。
    return config
