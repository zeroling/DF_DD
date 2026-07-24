"""项目根目录统一训练入口；配置默认从 configs/global.yaml 分阶段合并。"""

from __future__ import annotations

from runtime_compat import configure_runtime


# 必须在 Pipeline.run_pipeline 间接导入 PyTorch/NumPy 前执行。
configure_runtime()

# 所有参数解析和阶段调度集中在 Pipeline.run_pipeline，根文件只保留稳定入口。
from Pipeline.run_pipeline import main


if __name__ == "__main__":
    # 直接执行 ``python run_pipeline.py`` 时启动统一命令行。
    main()
