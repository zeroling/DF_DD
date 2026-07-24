# 同构 IDM 动态模型池 + 隐空间扩散的医学图像数据集浓缩

本项目面向二维医学图像分类数据集，实现一条完整主方法：冻结的 Autoencoder/扩散生成器
负责把可学习 `z_T` 映射成图像；一次实验选择 ConvNet-6、ResNet-18、
ConvNeXt-Tiny 或 ViT-Tiny 中的一种，同架构随机模型组成 IDM 动态池，只用真实数据
持续更新，再用其分布匹配、分类和浅/中/深 RBF 拓扑梯度优化 `z_T`。

项目不再训练或保存静态专家轨迹，不包含亚型发现、器官节点、解剖区域节点或特定疾病先验。

## 当前主流程

默认执行顺序只有四个阶段：

```text
train_autoencoder
        ↓
train_diffusion
        ↓
condense（同构 IDM 动态模型池在这里创建和训练）
        ↓
evaluate
```

### 1. `train_autoencoder`

使用 MONAI `AutoencoderKL` 从零训练图像编码器和解码器。MONAI 裸卷积 Decoder 的 raw 输出会统一经过 `tanh` 映射到 `[-1,1]`，AE 训练、确定性验证和扩散解码遵守同一契约。损失由 L1、可选 SSIM、KL 和可选 PatchGAN 构成；KL 对全隐变量取平均，并按 `autoencoder.yaml` 使用固定权重或从非零起点预热。日志同时报告原始/加权损失、zero-image L1 基线、raw 输出范围/越界率、tanh 饱和率和潜空间统计。训练结束后估计 `latent_scale = 1/std(z)`，供扩散训练和最终解码使用。

### 2. `train_diffusion`

冻结 Autoencoder，把真实训练图像编码到隐空间，从零训练共享的类别条件 `DiffusionModelUNet`。训练使用 DDPM 加噪、Min-SNR 加权、类别丢弃和 EMA；后续蒸馏优先加载 EMA 权重。

这一阶段不使用 IDM 队列，也不优化合成数据。

### 3. `condense`

对每个类别创建 IPC 个可学习 `z_T`。每次蒸馏迭代按以下顺序执行：

```text
从所选同构架构的模型池随机抽取两个模型并冻结
        ↓
z_T → 可微 DDIM → 冻结 VAE Decoder → 合成图像
        ↓
真实/合成图分别经过本轮随机模型
        ↓
IDM 分布损失 + 可靠性加权 CE + 三层 RBF 拓扑损失
        ↓
随机模型梯度等权平均并穿过 VAE/DDIM，只更新 z_T
        ↓
只用真实类别均衡 batch 训练本轮模型 K 步
        ↓
定期加入同架构全新随机模型，达到该架构容量后 FIFO 淘汰
```

在线分类器从不使用合成图像更新参数，因此不会和合成数据互相“迁就”。断点保存当前
活跃模型池及其优化器，不保存已经 FIFO 淘汰的历史权重。

### 4. `evaluate`

每个评估分类器重新随机初始化，只在 `condense` 生成的合成训练集上训练。真实验证集用于
选择最佳 epoch 和准确率停滞早停，真实测试集只在恢复最佳验证权重后评估一次。

报告指标包括：

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- 每类 Recall、Precision、Specificity；
- 混淆矩阵；
- 多次随机重复的均值和标准差。

## 为什么每轮抽 2 个不等于只有 2 个随机种子

模型池从 4 个随机模型开始，每 30 次凝聚加入一个全新随机初始化。默认 ConvNet-6
最多保留100个；较大的架构使用更小的安全容量。每轮只随机抽取2个模型参与梯度计算
和真实训练；这2个只是本轮样本，而不是随机种子总数。

因此磁盘只需保存当前状态：

```text
checkpoint_last.pt
├── z_T
├── z_T optimizer
├── 当前所选架构的全部池成员 + 各自 optimizer
├── 成员出生迭代、更新次数和准确率 EMA
├── 模型池随机采样器状态
└── 随机数与早停状态
```

达到 `maximum_size` 后先淘汰最老成员，再加入新随机成员。

## IPC 自适应损失

- IPC=1：使用类别特征均值、可靠性加权 CE、中/深层为主的拓扑约束和弱 `z_T` 标准正态先验；不计算单样本无法可靠估计的协方差、MMD 和多样性。
- IPC=10：继续使用原 IDM 均值匹配与分类正则，并叠加三层拓扑。
- IPC=50：降低分类正则权重并保留三层拓扑；每次只更新每类部分隐变量以控制显存。

三层拓扑节点只是规则空间网格，由所选网络统一提供的浅层、中层和深层空间特征构造
RBF 亲和图。它是保留在原 IDM 同构模型池之上的方法创新。

## 分文件配置

项目不再使用根目录单体 `config.yaml`。默认入口是 [global.yaml](/D:/PyWorkRoom/MICCAI/configs/global.yaml)，其中 `stage_configs` 自动加载各阶段文件：

```text
configs/
├── global.yaml          # 设备、数据、四网络逐层结构、默认阶段顺序
├── autoencoder.yaml     # VAE/PatchGAN、损失、优化器、保存与早停
├── diffusion.yaml       # U-Net、DDPM/Min-SNR/EMA、保存与早停
├── condensation.yaml    # IDM 同构模型池、z_T、DDIM、拓扑、IPC 损失
├── evaluation.yaml      # 新分类器训练、验证早停和真实测试
```

每个 YAML 参数旁都写有中文含义、有效范围或使用场景。加载顺序是：

```text
global.yaml
→ stage_configs 指向的各阶段 YAML
→ 命令行 --set 或测试覆盖
```

命令行覆盖拥有最高优先级，例如：

```bash
python run_pipeline.py --set condensation.idm_queue.models_per_iteration=4
python run_pipeline.py --set models.definitions.vit_tiny.embed_dim=256
python run_pipeline.py --set condensation.iterations.1=5000 --ipc 1
```

## 可调网络结构

[global.yaml](/D:/PyWorkRoom/MICCAI/configs/global.yaml) 中可以直接调节：

- ConvNet：每个卷积块通道 `widths`、卷积核、GroupNorm 组数、激活、池化和三层特征索引；
- ResNet：四阶段通道、每阶段 BasicBlock 数、stem 通道/卷积核/步长、是否最大池化、BatchNorm/GroupNorm；
- ConvNeXt：四阶段 `depths`、`dims`、卷积核、patch size、drop path 和 LayerScale；
- ViT：patch size、token 维度、block 数、注意力头数、MLP 比例、dropout/drop path 和三层 block 索引；
- Autoencoder：每层通道、残差块数、注意力层和隐变量通道；
- Diffusion U-Net：每层通道、残差块数、注意力层、注意力头通道和时间步。

改变网络结构后，旧权重张量若形状不兼容会正常报错；这不是哈希校验，而是权重无法装入不同结构。

## IDM 模型池可调参数

[condensation.yaml](/D:/PyWorkRoom/MICCAI/configs/condensation.yaml) 中包括：

- 同一次实验使用的同构架构：`convnet`、`resnet18`、`convnext_tiny` 或 `vit_tiny`；
- 初始模型池大小和最大容量；
- 每轮随机抽取多少个模型；
- 每个成员一次训练多少真实 batch；
- 每隔多少次凝聚加入一个全新随机种子；
- 在线 batch size、标签平滑和准确率 EMA；
- 每种可选架构的安全池容量、batch 和独立优化器策略。

新加入成员不预热，会真正从随机阶段进入模型池。

切换架构时必须使用新的 `run_dir`，例如：

```bash
python run_pipeline.py --stage condense --ipc 1 \
  --set condensation.idm_queue.architecture=convnext_tiny \
  --run-dir outputs/idm_topology_convnext
```

## 保存频率与早停

各阶段独立配置：

- `checkpoint_interval_epochs` 或 `checkpoint_interval_iterations`：保存 `checkpoint_last.pt` 的间隔；
- `preview_interval_epochs` 或 `preview_interval_iterations`：保存预览图的间隔；
- `log_interval_epochs` 或 `log_interval_iterations`：日志间隔；
- `patience_checks`：连续多少次检查无改善后停止；
- `min_delta`：至少改善多少才重置耐心计数；
- `minimum_epochs` / `minimum_iterations`：允许早停前的最低训练量；
- `reset_on_resume`：续训时是否清空早停历史。

`condense` 的定期预览会分批解码并保存当前完整的 `类别数×IPC` 合成图，而不是只取
每类第一张；`condensation.preview_batch_size` 只控制预览解码显存。

Autoencoder 监控真实验证 L1；Diffusion 监控真实验证去噪损失；Evaluation 监控真实验证 Balanced Accuracy。Condensation 的在线损失波动较大，因此完整阶段早停默认关闭，但成员级准确率停滞替换可独立开启。

## 安装

建议使用 Python 3.10–3.12。先根据服务器 CUDA 版本安装相匹配的 PyTorch，再安装其余依赖：

```bash
pip install torch torchvision --index-url <与你的 CUDA 对应的 PyTorch 源>
pip install -r requirements.txt
```

项目只使用安装后的 MONAI 网络和调度器定义，不下载 MONAI 或其他第三方预训练权重。

### Windows

PowerShell 中建议从项目根目录激活虚拟环境并先做配置检查：

```powershell
.\.venv\Scripts\Activate.ps1
python run_pipeline.py --dry-run
python run_pipeline.py --stage condense --ipc 1
```

入口文件名是 `run_pipeline.py`（不是 `run_pipline.py`）。Windows 下项目不会启用
当前平台不支持的 CUDA `expandable_segments`，并让 Python 正常处理 Ctrl+C；手动停止时
会以退出码 130 结束，不再由 Intel Fortran 运行库打印 `forrtl error (200)`。

Windows 的 DataLoader 默认读取 `project.windows_num_workers: 0`，避免 `spawn` 重复导入
PyTorch/MONAI 导致内存陡增。内存充足时可以在 `configs/global.yaml` 中逐步提高到 1、2
或 4；Linux 继续使用 `project.num_workers`。

#### 16GB 显卡运行 `condense`

默认 `condensation.yaml` 已按 224×224、RTX 4070 Ti SUPER 16GB 调整：开启可微 DDIM
激活检查点，ConvNet 的 `real_per_class=8`、在线训练 batch=16，IPC=10/50 每类每轮只
反传两个合成隐变量，预览和最终导出 batch 均为 4。这些调整不改变已经训练好的
Autoencoder/Diffusion 网络结构或权重；代价是 DDIM 反向会因重算激活而稍慢。

如果显卡同时被其他程序大量占用，可进一步临时使用保守档：

```powershell
python run_pipeline.py --stage condense --ipc 1 `
  --set condensation.real_per_class.convnet=4 `
  --set condensation.idm_queue.batch_size.convnet=8 `
  --set condensation.preview_batch_size=2 `
  --set condensation.export_batch_size=2
```

## 数据接口

数据层统一输出 `[0,1]` 的 `float32 [C,H,W]`。分类器 mean/std 归一化和生成模型 `[-1,1]` 映射分别在对应阶段完成。

### 文件夹格式

`data.adapter: folder` 时使用：

```text
data/COVID/
├── train/
│   ├── COVID/
│   ├── Lung_Opacity/
│   ├── Normal/
│   └── Viral Pneumonia/
├── val/                 # 可选；缺失时从 train 按类别确定性划分
│   └── ...同样类别目录
└── test/
    └── ...同样类别目录
```

### 清单格式

把 `data.adapter` 改为 `manifest` 后，可读取 CSV、JSON 或 JSONL：

```csv
path,label,split
images/a001.npy,DiseaseA,train
images/a002.dcm,DiseaseA,test
```

支持 PNG/JPEG/BMP/TIFF/WebP、`.npy/.npz`、`.pt/.pth` 和 DICOM。图像尺寸可配置为整数或 `[高度, 宽度]`，通道数支持 1 或 3，默认 `224×224×3`。

## 运行方式

检查合并后的阶段顺序、设备、IPC 和配置文件：

```bash
python run_pipeline.py --dry-run
```

运行完整四阶段：

```bash
python run_pipeline.py
```

只运行 IPC=1：

```bash
python run_pipeline.py --ipc 1
```

从指定阶段开始：

```bash
python run_pipeline.py --from-stage train_diffusion --ipc 1
```

单独运行某个阶段：

```bash
python run_pipeline.py --stage train_autoencoder
python run_pipeline.py --stage train_diffusion
python run_pipeline.py --stage condense --ipc 1
python run_pipeline.py --stage evaluate --ipc 1
```

使用另一份全局配置入口：

```bash
python run_pipeline.py --config configs/global.yaml
```

## 断点恢复规则

再次指向同一个 `project.run_dir` 或 `--run-dir` 时：

- 优先读取阶段目录中的 `checkpoint_last.pt`；没有时读取进度最大的完整 `.pt/.pth`；
- 断点 `epoch=78`、当前目标 `epochs=100` 时，从第 79 轮继续；
- 断点 `iteration=4000`、当前目标 `iterations=6000` 时，从第 4001 次继续；
- 当前 YAML 中的学习率和目标训练量覆盖断点旧值；
- 不生成或比较配置、代码、数据哈希；
- IDM 模型池恢复全部活跃成员、优化器、训练年龄、可靠性和随机采样状态；
- 减小最大容量时保留最新成员，增大容量后继续按固定间隔自然增长；
- 异构旧版凝聚断点不会恢复 `z_T` 或模型队列，而是明确提示并从第 0 次重新开始；
- 模型结构或类别数造成权重形状不兼容时会明确报错。

要完全从头开始，请指定新的 `run_dir`，不要删除已有实验目录。

## 输出结构

```text
outputs/online_idm_latent_diffusion/
├── autoencoder/
│   ├── checkpoint_last.pt
│   ├── summary.json
│   └── preview_epoch_*.png
├── diffusion/
│   ├── checkpoint_last.pt
│   ├── summary.json
│   └── preview_epoch_*.png
├── condensed/
│   └── ipc_<N>/
│       ├── checkpoint_last.pt
│       ├── synthetic.pt
│       ├── preview.png
│       ├── preview_iteration_*.png
│       └── images/<class>/synthetic_*.png
├── evaluation/
│   └── condensed/ipc_<N>/<architecture>/repeat_<N>/
│       ├── checkpoint_last.pt
│       ├── checkpoint_best.pt
│       └── result.json
├── run_manifest.json
└── pipeline_summary.json
```

## 测试

快速测试：

```bash
python smoke_test.py
```

微型端到端测试会运行完整四阶段，再修改轮数和学习率从 1 继续到 2；IDM 模型池随
`condensed` 断点一起恢复：

```bash
python smoke_test.py --integration
```

## 思想与上游实现来源

- IDM：<https://github.com/uitrbn/IDM>，*Improved Distribution Matching for Dataset Condensation*，CVPR 2023。
- MONAI Generative Models：<https://github.com/Project-MONAI/GenerativeModels>。项目使用安装版 MONAI 中的 `AutoencoderKL`、`DiffusionModelUNet`、调度器和可选 `PatchDiscriminator`，全部从零训练。
- 浅/中/深 RBF 拓扑构造改造自用户提供的 *Lite-MyoNet: An Edge-Based Network for Pathological Gait Analysis via Topology-Preserving Distillation*，已经删除时间动态、人体节点和任务特定结构。

正式运行不依赖克隆的第三方仓库。
