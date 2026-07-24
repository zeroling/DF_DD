"""可断点恢复的主流程阶段。在线 IDM 队列由 condense 阶段内部创建。"""

from Pipeline.Stages import condense, evaluate, train_autoencoder, train_diffusion

# 静态专家和独立像素 IDM 已移除；在线队列只在 condense 内部创建。
__all__ = ["train_autoencoder", "train_diffusion", "condense", "evaluate"]
