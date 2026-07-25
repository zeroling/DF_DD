# 医学图像 IPC=1 数据集浓缩实验

项目现在使用一套统一的标准 IDM 实现。`condense` 不再是另一套隐空间管线，
也不存在单独的 IDM 项目目录：

- `Pipeline/Stages/condense.py`：标准 IDM 及其 topology / VAE / diffusion 扩展；
- `Net/Condensation/idm_official.py`：官方 ConvNet-6、DSA、P&E 等基础组件；
- `Pipeline/run_ablation.py`：D/C 实验编排；
- `Pipeline/ablation_worker.py`：隔离每个子任务，结束后释放 CUDA 上下文。

## 配置结构

配置按职责拆分：

```text
configs/
├── global.yaml       # 设备、数据、网络结构和主阶段顺序
├── autoencoder.yaml  # VAE 训练
├── diffusion.yaml    # diffusion 训练
├── condense.yaml     # IDM、latent、topology、显存和评估参数
├── evaluation.yaml   # 原主流水线 evaluate 阶段
└── ablation.yaml     # 只定义 D/C 实验矩阵、seed 和重复次数
```

`global.yaml` 只引用这些文件，不再堆放消融算法参数。加载顺序是：

```text
global.yaml
→ stage_configs
→ experiment_configs.ablation
→ 命令行覆盖
```

本次运行开始时，D/C 启动器会把已经合并、校验过的配置写成
`D_resolved_config.json` 或 `C_resolved_config.json`。同一次长任务中的所有子进程都
读取这份快照，因此运行中编辑 YAML 不会造成 repeat 之间的配置漂移。重新执行命令时会
生成新快照。不做配置、代码或数据哈希校验。

## 实验定义

D 组用于定位生成模型的信息损失：

- D0：完整真实训练集；
- D1：完整训练集经过 VAE encode/decode；
- D2：完整训练集经过 DDIM inversion/reconstruction；
- D3：每类随机一张真实图。

C 组是统一 condense 实现上的六个消融：

```text
C0 标准 pixel IDM              C1 pixel IDM + topology
C2 VAE z0                      C3 VAE z0 + topology
C4 diffusion zT                C5 diffusion zT + topology
```

C0 就是消融表中的 IDM baseline，不是额外的一套项目或另一份代码。

## 一键运行

PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
.\run_D_experiments.cmd
.\run_C_experiments.cmd
```

正式 C 组默认使用 5 个 condensation seed，每个合成集评估 5 次。短流程和局部实验：

```powershell
.\run_C_experiments.cmd --profile pilot
.\run_C_experiments.cmd --only C0 --only C2
.\run_C_experiments.cmd --only C5 --smoke
.\run_D_experiments.cmd --only D3 --smoke
```

主流水线也可以直接调用统一后的默认 condense 方法：

```powershell
python run_pipeline.py --dry-run
python run_pipeline.py --stage condense --ipc 1
```

默认方法由 `configs/condense.yaml` 的 `condensation.default_method` 指定。当前标准
IDM 消融协议固定为 IPC=1。

## 断点与中断

- condensation、评估和 D1/D2 缓存都支持断点继续；
- 自动读取最新可读断点，不比较哈希；
- D1/D2 使用可续写的 uint8 memmap；
- 每个 condensation / architecture / repeat 使用独立子进程；
- 子进程退出后回收 Python 引用、CUDA cache 和 IPC 句柄；
- `checkpoint_last.pt` 使用原子替换，断电时最多损失当前保存间隔。

如果按 `Ctrl+C`，重新运行同一命令即可继续。入口文件是 `run_pipeline.py`，不是
`run_pipline.py`。

## 4070 Ti SUPER 16GB 显存设置

安全参数集中在 `configs/condense.yaml`：

```yaml
condensation:
  memory:
    max_reserved_fraction: 0.72
    real_feature_microbatch: 64
    real_train_microbatch: 32
```

预检会真实执行前向、DSA、反向和 optimizer step，并以
`torch.cuda.max_memory_reserved()` 判断，而不是只看偏低的 allocated memory。超过
物理显存的 72% 时自动把 batch 减半。有效 IDM 真实 batch 仍是 128，内部用 microbatch
累积保持算法定义。

RTX 4070 Ti SUPER 实测单轮峰值：

| 路径 | reserved 峰值 |
|---|---:|
| D0 / C0 官方 ConvNet-6 评估 | 5010 MiB |
| C0 pixel IDM condense | 6808 MiB |
| C2 VAE z0 condense | 7472 MiB |
| C4 diffusion zT condense | 7812 MiB |
| C5 diffusion zT + topology | 8348 MiB |

以上均未进入 Windows 共享 GPU 内存。不要把 `max_reserved_fraction` 调成 0.95；
Windows/WDDM 下这会允许 allocator 逼近 16GB 物理显存，容易再次溢出到共享内存。

## 权重要求

C0/C1 不需要 VAE 或 diffusion 权重。D1、C2、C3 需要：

```text
outputs/online_idm_latent_diffusion_tanh_v2/autoencoder/checkpoint_last.pt
```

D2、C4、C5 还需要：

```text
outputs/online_idm_latent_diffusion_tanh_v2/diffusion/checkpoint_last.pt
```

恢复权重时不校验哈希，但模型张量形状必须与当前 `autoencoder.yaml` /
`diffusion.yaml` 一致。

## 数据目录

默认 folder adapter：

```text
data/COVID/
├── train/<class name>/
├── val/<class name>/      # 可选
└── test/<class name>/
```

数据层输出 `[0,1]` 的 `float32 [C,H,W]`。分类器归一化与生成模型 `[-1,1]`
映射分别在对应模块内完成。Windows 默认 `project.windows_num_workers: 0`，避免
DataLoader spawn 重复导入大型依赖。

## 安装与快速检查

先安装适配本机 CUDA 的 PyTorch，再安装其余依赖：

```powershell
pip install -r requirements.txt
python -m unittest tests.test_idm_ablation
python run_pipeline.py --dry-run
```
