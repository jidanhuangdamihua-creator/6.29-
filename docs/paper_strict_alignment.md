# 严格论文对齐说明

本文档说明当前仓库如何在不破坏既有脚本入口的前提下，收敛到“可审计的严格论文协议”。

## 1. 严格模式开关

配置支持两个等价键：

- `paper_reproduction.strict_paper_mode`
- `paper_reproduction.paper_strict_mode`（兼容别名）

命令行脚本参数 `--strict-paper-mode` 优先级最高。

## 2. 指标口径对齐（metric_space）

当前代码真实评估口径：

- `current_metric_space = normalized_minmax_space`
- `RMSE`: 在 MinMax 归一化空间计算
- `Accuracy = 1 / (RMSE + 1e-8)`

严格模式下的硬校验：

- 若 `current_metric_space` 不是 `normalized_minmax_space`，直接判定为协议失败。

仍保留 TODO 的项：

- `paper_metric_space`
- `paper_accuracy_definition`

这两项如果未确认，仍会在报告中明确标记，不会伪装成 `ALIGNED`。

## 3. 数据切分窗口复刻

当前窗口协议：

- 目标域先取最近 `30(train+val) + 180(test)` 天窗口
- 再按 `0.067 / 0.067 / 0.866` 做时序切分

严格模式新增断言：

- `preprocessing.target_train_val_days/target_test_days` 必须与 `paper_reproduction.split_protocol.target_window` 完全一致。
- 运行时目标窗口的日期跨度必须等于期望天数，否则立即报错。

新增审计字段（写入结果与对齐报告）：

- `target_window_expected_days`
- `target_window_range_days`
- `target_window_unique_days`
- `target_strict_paper_mode`

## 4. 最多五个预训练 TL 模型约束

论文轨道定义：

- `paper_max_pretrained_models = 5`
- `paper_source_counts = [1, 3, 5]`
- `extended_source_counts = [6, 9]`

严格模式行为：

- 任意多源方法请求 `k` 不在论文轨道集合内，立即阻断。
- 矩阵运行器在 strict 模式会预检 `source_counts`，避免运行时混入扩展配置。

## 5. 统一自动校验脚本

脚本：`scripts/validate_paper_protocol_strict.py`

输出：

- `outputs/paper_alignment_reports/paper_protocol_strict_validation.csv`
- `outputs/paper_alignment_reports/paper_protocol_strict_validation.json`

覆盖检查维度：

1. 协议配置结构（metric/split/source）
2. 各数据集 split 运行时窗口一致性
3. source/pretrained 上限与轨道判定

## 6. TODO 原则

以下场景必须保持 TODO 标记，直到论文原文证据补全：

- 论文原始 metric space 的精确定义
- 论文原始 split 绝对边界/样本窗口证据

如果后续补证，请同步更新：

- `configs/default_config.json` 的 `paper_reproduction` 段
- 本文档
- `README.md` 中“论文对齐说明”章节
