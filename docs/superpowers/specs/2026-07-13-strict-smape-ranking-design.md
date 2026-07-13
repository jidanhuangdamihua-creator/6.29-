# 严格 sMAPE 跨数据集排名设计

## 目标

将原始销量空间的 sMAPE（百分比，越低越好）设为 D1–D6 正式跨数据集、跨方法排名和统计检验的唯一主指标。RMSE 保留为诊断指标，不参与正式排名。

## 范围与非目标

本次只收紧指标计算、结果事实字段、CSV 保留、正式结果筛选、聚合和统计检验。不改变模型训练、数据、候选池、KNN、Top-K、窗口、超参数、`parallel_mode_runner.sh`，也不运行全量实验。

历史 CSV 若不能证明 sMAPE 在原始销量空间计算，不进入正式排名；本次不尝试从 normalized sMAPE 反推或修复历史结果。

## Canonical sMAPE 合同

公式固定为：

`mean(2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred) + 1e-8)) * 100`

输入必须有限，输出单位为百分比。双零点的贡献为 0；只有一方为零的点趋近 200%。负值使用绝对值计算，但结果必须保留数据质量审计信息，不能把负销量解释为正常业务销量。

正式可比结果必须同时满足：

- `strict_paper_metrics=True`
- `paper_metric_space_requested=original_sales_space`
- `paper_metric_space_actual=original_sales_space`
- `inverse_transform_applied=True`
- `paper_metric_computed_valid=True`
- `paper_metric_status=valid`
- `smape_metric_space=original_sales_space`
- 主 `smape` 有限

不满足任一条件的行可以作为 debug 或错误记录保存，但不得参与正式排名、均值或统计检验。

## 计算与严格失败语义

`compute_metrics_with_protocol()` 是 sMAPE、RMSE、指标空间和 inverse 审计字段的唯一计算与事实来源。它返回 current、original/paper 和最终主指标的完整字段。

在严格模式下，调用方必须提供有限且同长度的 `y_true`、`y_pred`，以及可用于 `sales` 列反归一化的 `sales_scaler` 和 `feature_columns`。缺失或无效时抛出带缺失字段名称的 `ValueError`；不得返回先前的 normalized `rmse` 或 `smape`。反归一化或主指标计算不成功时同样不能返回有限主指标。

在 non-strict/debug 模式，normalized 指标仍可保留，但 `*_metric_space` 和 `paper_metric_status` 必须准确标示 fallback 或不可用状态。

## 多源方法 payload 合同

MSWA-TL、MSSB-TL、MSML-TL、MSML-TL-RFE 传给 `_extract_method_metrics()` 的选中 payload 都必须包含：

- `y_true`
- `y_pred`
- `sales_scaler`
- `feature_columns`

`_extract_method_metrics()` 接收 `metric_protocol`。若严格模式开启，它只接受上述 payload 并强制调用 canonical 计算函数，不透传 transfer 层预存的 `rmse` 或 `smape`。方法内部已计算的指标可作为诊断信息，但不是严格主指标来源。

## 结果与序列化合同

所有方法和 D4–D6 行保留：`smape`、`normalized_smape`、`smape_paper`、`original_scale_smape`、相应 RMSE 字段、空间字段、inverse 字段和计算状态字段。D4–D6 不得清空已计算的 paper/original-scale 指标。

`paper_metric_aligned` 只表达指标口径是否符合严格合同。是否存在外部论文参考值使用独立字段 `paper_reference_available` 和 `paper_reference_status`，不得覆盖前者语义。

## 正式排名、聚合与统计

排名按 information-sharing 场景分别产生。聚合顺序为：同一 `(dataset, target, method, horizon)` 先对 seed 平均；再对 target 做 macro mean；最后对 dataset 做 macro mean。这样每个 target 和每个 dataset 都只有一个等权贡献，避免 target、seed 或 horizon 较多的数据集主导结果。

aggregation、visualization 和 statistical tests 共同使用正式可比行筛选函数。排序、Friedman、Wilcoxon 和平均排名均以 `smape` 为值；RMSE 不参与这些正式结论。

每份正式摘要同时报告有效行数、排除行数和 zero-rate（若结果行提供），便于解释零/负销量对 sMAPE 的敏感性。

## 测试与基线

先为 strict 缺 scaler、缺 y_true、缺 sales 特征、成功 inverse、四个多源 payload、D4–D6 序列化保留字段、non-strict 空间标记、No-TL/SS-TL 回归、正式行筛选和 macro 聚合写失败测试。

基线 focused pytest 在提交 `ed4ce918` 上为 42 通过、1 失败。失败测试期望 D4–D6 透传 `strict_paper_metrics=False`，而当前生产合同已强制为 true。该测试会改为验证严格合同本身，而不是仅为消除失败而改断言。
