# 仓库结构说明（论文复现评估版）

本文档给出当前仓库的复现评估视角结构，不改变任何现有代码，仅用于快速定位入口。

## 一级目录

```text
src/
scripts/
tests/
outputs/
configs/
data/
README.md
requirements.txt
run_main_experiment.py
run_full_experiment_matrix.py
verify_bootstrap.py
```

## 关键目录职责

1. src/
- 核心实现（数据处理、迁移方法、实验调度、可视化等）
- 当前包含 data_processing、experiment、models、source_selection、transfer_methods、visualization 等子模块

2. scripts/
- 复现实验辅助脚本
- 典型入口：run_main_experiment.py、run_full_experiment_matrix.py、run_smoke_test.py、generate_results_report.py

3. tests/
- 单元/集成测试集合
- 覆盖核心迁移方法、实验调度、结果可视化模块

4. outputs/
- 运行产物输出目录
- experiment_results/: 原始实验结果
- matrix_runs/: 实验矩阵结果与快照
- results_reports/: 论文风格结果表与图表
- run_snapshots/、env_snapshots/: 配置和环境快照

5. configs/
- 运行配置与矩阵配置
- default_config.json、matrix_config.json

6. data/
- 数据准备与字段要求说明（README_data.md）

## 第三方复现建议入口

1. 安装依赖后先执行 verify_bootstrap.py 进行环境检查
2. 执行 run_main_experiment.py 产出基础结果
3. 执行 scripts/generate_results_report.py 产出图表与格式化结果
4. 如需批量评估，执行 run_full_experiment_matrix.py
