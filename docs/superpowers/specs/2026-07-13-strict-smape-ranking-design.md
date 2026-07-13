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

### 字段单一真源

`compute_metrics_with_protocol()` 在一次计算中构造唯一的 `computed_metric_space` 事实：它只能是实际成功计算出的 `original_sales_space`、实际 current space，或未计算。`paper_metric_space_requested` 只从输入 protocol 读取；`paper_metric_space_actual`、`rmse_metric_space`、`smape_metric_space`、`original_scale_*` 和严格主指标全部从 `computed_metric_space` 派生，调用方和 CSV 层不得独立赋值或覆盖。

严格且有效时，`paper_metric_space_requested`、`paper_metric_space_actual`、`rmse_metric_space`、`smape_metric_space` 必须全部为 `original_sales_space`。无效记录保留 requested 值以说明请求，但 actual 和各 metric-space 字段为空、主指标为 NaN，并以 `paper_metric_status=invalid` 和 `paper_metric_error` 说明原因；不得伪称已经在 original space 计算。

## 计算与严格失败语义

`compute_metrics_with_protocol()` 是 sMAPE、RMSE、指标空间和 inverse 审计字段的唯一计算与事实来源。它返回 current、original/paper 和最终主指标的完整字段。

在严格模式下，调用方必须提供有限且同长度的 `y_true`、`y_pred`，以及可用于 `sales` 列反归一化的 `sales_scaler` 和 `feature_columns`。缺失或无效时抛出带缺失字段名称的 `MetricProtocolError(ValueError)`；不得返回先前的 normalized `rmse` 或 `smape`。反归一化或主指标计算不成功时同样不能返回有限主指标。

这个异常只表示当前 method/entity 的严格指标无效。D4–D6 的 per-method/per-entity 边界必须捕获它，写出一条 `paper_metric_status=invalid`、`paper_metric_computed_valid=False`、`paper_metric_error` 非空且 `rmse/smape=NaN` 的错误行，再继续处理其他 method/entity。异常不得中断整批 D4–D6 运行；直接调用计算函数的单元测试仍必须观察到 raise。

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

`smape_paper` 的语义固定为“已在 requested paper space 实际算出的 sMAPE”；没有完成该计算时为 null，绝不能用 normalized 值填充。`original_scale_smape` 是前者在 requested/actual 均为 `original_sales_space` 时的兼容别名，二者必须相等。严格有效行中，`smape`、`smape_paper`、`original_scale_smape` 三者必须相等；non-strict 行可保留 `normalized_smape`，但不得用别名伪装为 paper/original 指标。

`paper_metric_aligned` 只表达指标口径是否符合严格合同。是否存在外部论文参考值使用独立字段 `paper_reference_available` 和 `paper_reference_status`，不得覆盖前者语义。

## 正式排名、聚合与统计

排名按 information-sharing 场景和 horizon 分别产生。正式排名单位为 `(dataset, horizon, scenario)`：同一 `(dataset, target, method, horizon, scenario)` 先对 seed 平均，再对 target 做 macro mean。跨数据集比较时，针对同一 `(horizon, scenario, method)` 再对 dataset 做 macro mean。这样每个 target 和每个 dataset 都只有一个等权贡献，避免 target、seed 或 horizon 较多的数据集主导结果。

不同 horizon 不得隐式合并成一个正式排名。若保留 `horizons_1_5` 等跨 horizon 汇总，它必须标为 descriptive/exploratory、明确是 horizon macro mean，且不得用作论文主表排名或正式 Friedman/Wilcoxon 结论。

aggregation、visualization 和 statistical tests 共同调用唯一的 `is_formally_comparable_smape_row(row)` 函数。排序、Friedman、Wilcoxon 和平均排名均以通过该函数的 `smape` 为值；RMSE 不参与这些正式结论。没有任何消费者可以直接根据 `smape` 有限值绕过该筛选。

每个计算结果行必须包含 `target_sales_zero_count`、`target_sales_zero_rate`、`target_sales_negative_count`、`target_sales_negative_rate`、`metric_sample_count` 和 `sales_distribution_available`。成功 original-space 计算从 original `y_true` 计算这些值；无法得到 original `y_true` 的 invalid 行仍保留字段，但 counts/rates 为空且 `sales_distribution_available=False`。正式摘要必须报告有效行数、排除行数、零值和负值的 count/rate，便于解释 sMAPE 对极端点的敏感性。

本轮明确不新增 `run_profile`、`paper_table_eligible` 或 `archive_eligible`：目标代码库不存在这些字段，新增它们会制造第二套准入机制。未来若引入 profile，它只能作为 `is_formally_comparable_smape_row()` 的额外输入；metric-space 严格合同仍由该函数唯一判定。

## 测试与基线

先为 strict 缺 scaler、缺 y_true、缺 sales 特征、成功 inverse、四空间字段单一真源、三种 strict sMAPE 别名一致性、四个多源 payload、D4–D6 序列化保留字段、per-method invalid 行不中断 entity 批处理、non-strict 空间标记、No-TL/SS-TL 回归、正式行筛选和 macro 聚合写失败测试。筛选测试必须构造 `strict_paper_metrics=False` 但 `smape` 有限的行，断言 aggregate、visualization、statistical tests 全部排除它。聚合测试必须覆盖同一 target 的多个 horizon，断言正式排名按 horizon 分表且不隐式混合。

基线 focused pytest 在提交 `ed4ce918` 上为 42 通过、1 失败。失败测试期望 D4–D6 透传 `strict_paper_metrics=False`，而当前生产合同已强制为 true。该测试会改为验证严格合同本身，而不是仅为消除失败而改断言。
