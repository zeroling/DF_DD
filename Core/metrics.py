"""统一的分类评估指标。

本模块只负责“读取模型预测并计算指标”，不会训练模型，也不会修改模型参数。
医学数据集经常存在类别不平衡，因此除了普通准确率，还显式返回宏平均召回率
（Balanced Accuracy）、Macro-F1、每类召回率、每类精确率、每类特异度和混淆矩阵。
所有阶段都调用同一个函数，避免 IDM 基线、扩散蒸馏和最终测试使用不同口径。
"""

from __future__ import annotations

from typing import Any, Mapping  # Any 用于通用配置值；Mapping 允许普通字典和只读映射。

import numpy as np  # NumPy 用于混淆矩阵迹与宏平均统计。
import torch  # PyTorch 提供模型推理、设备迁移和交叉熵损失。
from sklearn.metrics import confusion_matrix, f1_score  # 使用成熟实现减少指标边界错误。

from Core.augmentations import classifier_normalize  # 按全局配置执行统一分类归一化。
from Core.data import unpack_batch  # 兼容 tuple、list 和字典形式的数据批次。


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader,
    device: str | torch.device,
    num_classes: int,
    config: Mapping[str, Any] | None = None,
    criterion: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """在真实验证集或真实测试集上计算完整分类指标。

    参数：
        model: 待评估分类器；函数会临时切换到 ``eval`` 模式，但不会更新参数。
        loader: 验证集或测试集 DataLoader；批次格式由 :func:`unpack_batch` 解耦。
        device: 推理设备，例如 ``"cuda"``、``"cpu"`` 或 ``torch.device``。
        num_classes: 数据集完整类别数；即使某个批次没有某类，也保留对应矩阵行列。
        config: 可选项目配置；传入后会应用与训练阶段一致的分类归一化。
        criterion: 可选评估损失；未提供时使用标准交叉熵。

    返回：
        包含标量指标、逐类指标和混淆矩阵的普通字典，可直接写入 JSON。
    """

    # 将字符串设备统一转换为 torch.device，后续迁移张量时只处理一种类型。
    device = torch.device(device)
    # 关闭 Dropout，并让 BatchNorm 使用已有统计量，保证评估结果可复现。
    model.eval()
    # 如果调用者没有指定损失函数，则使用与普通分类训练一致的交叉熵。
    criterion = criterion or torch.nn.CrossEntropyLoss()
    # total_loss 保存“批次平均损失 × 批次样本数”的总和，最后再除以总样本数。
    total_loss = 0.0
    # total_count 用来支持最后一个不足整批的批次，避免简单平均批次损失造成偏差。
    total_count = 0
    # predictions 和 labels 收集全数据集离散类别，以便一次性计算混淆矩阵和 F1。
    predictions: list[int] = []
    labels: list[int] = []
    # no_grad 装饰器已经关闭梯度，这里按 DataLoader 顺序遍历全部评估样本。
    for batch in loader:
        # 数据适配层将不同数据集插件的批次统一拆成图像与标签。
        images, targets = unpack_batch(batch)
        # non_blocking 在 DataLoader 开启 pinned memory 时允许异步 CPU→GPU 复制。
        images = images.to(device, non_blocking=True)
        # 交叉熵要求一维 long 标签；view(-1) 同时兼容 [B] 和 [B,1]。
        targets = targets.to(device, non_blocking=True).long().view(-1)
        # 合成图保存为 [0,1] RGB；分类器看到的输入必须与真实图采用相同归一化。
        if config is not None:
            images = classifier_normalize(images, config)
        # 前向传播返回形状 [B, num_classes] 的未归一化 logits。
        logits = model(images)
        # 损失只用于报告，不参与反向传播或早期停止之外的任何更新。
        loss = criterion(logits, targets)
        # 按样本数累积，确保不同大小批次具有正确权重。
        total_loss += float(loss.item()) * targets.numel()
        total_count += targets.numel()
        # argmax 得到每个样本的预测类别，并移动到 CPU 普通列表供 sklearn 使用。
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(targets.cpu().tolist())
    # 空数据集无法定义任何分类指标，直接报错比返回 NaN 更容易定位配置问题。
    if total_count == 0:
        raise ValueError("评估数据集为空")
    # 显式给出所有类别编号，保证混淆矩阵始终是 [C,C]，即使某类暂时没有样本。
    class_ids = list(range(int(num_classes)))
    # 矩阵约定：行是真实类别，列是预测类别。
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    # 三个列表按 class_ids 顺序保存逐类指标，便于定位少数类性能问题。
    recalls: list[float] = []
    precisions: list[float] = []
    specificities: list[float] = []
    # 将每个类别视为“一对其余”二分类问题，分别计算 TP/FP/FN/TN。
    for class_id in class_ids:
        # 对角元素是当前类别预测正确的样本数。
        true_positive = int(matrix[class_id, class_id])
        # 当前预测列减 TP，得到被误判成当前类别的样本数。
        false_positive = int(matrix[:, class_id].sum()) - true_positive
        # 当前真实行减 TP，得到当前类别被错分到其他类别的样本数。
        false_negative = int(matrix[class_id, :].sum()) - true_positive
        # 总样本减去 TP、FP、FN，剩余为当前类别的一对其余真阴性数。
        true_negative = int(matrix.sum()) - true_positive - false_positive - false_negative
        # max(1, denominator) 避免极小验证划分中某类缺失时出现除零。
        recalls.append(true_positive / max(1, true_positive + false_negative))
        precisions.append(true_positive / max(1, true_positive + false_positive))
        specificities.append(true_negative / max(1, true_negative + false_positive))
    # 返回 Python float/list，确保日志记录器可以直接序列化为 JSON，而不残留张量。
    return {
        # 全数据集逐样本平均交叉熵。
        "loss": total_loss / total_count,
        # 普通 Accuracy 等于混淆矩阵对角和除以总样本数。
        "accuracy": float(np.trace(matrix) / max(1, int(matrix.sum()))),
        # Balanced Accuracy 是所有类别召回率的算术平均，对类别规模不敏感。
        "balanced_accuracy": float(np.mean(recalls)),
        # Macro-F1 对每类 F1 等权平均；zero_division=0 明确定义无预测类别的结果。
        "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        # 以下逐类列表与 class_ids 一一对应。
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_specificity": specificities,
        # ndarray 转 list 后可以安全写入 metrics.json。
        "confusion_matrix": matrix.tolist(),
    }
