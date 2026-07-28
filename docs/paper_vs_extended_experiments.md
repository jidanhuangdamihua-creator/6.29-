# 论文复现与扩展实验边界说明

本文档用于回答一个审查问题：当前仓库的哪些结果可以归入“论文复现”，哪些结果只能归入“扩展实验”。

为避免误判，本文统一采用以下表达规则：

- 有代码和配置证据支持的内容，写“已实现”或“一致”。
- 仅能从相对窗口或工程推断得到的内容，写“部分一致”或“按论文相对窗口复刻”。
- 无证据项明确标注“未确认”。

当前项目统一数据集映射：Dataset1 = 需求预测挑战赛，Dataset2 = 意大利面需求，Dataset3 = Rossmann 门店。本文中所有 Dataset1/2/3 指代均以此映射为准。

## 1. 总表（审查入口）

| 主题 | 论文原始设定 | 仓库当前默认设定 | 严格论文模式设定 | 是否完全一致 | 仍未确认的部分 |
|---|---|---|---|---|---|
| 评估口径与 metric_space | 论文使用 RMSE/Accuracy，metric space 细节在仓库内证据不足 | 默认 normalized_minmax_space | 可强制 original_sales_space 审查口径 | 否，部分一致 | 论文是否强制原始量纲评估 |
| 数据切分窗口 | 目标域约 1 个月观测 + 约 6 个月预测 | 30 天 observed + 180 天 forecast 的相对窗口 | strict_paper_split=true 强制 30+180，不足报错 | 否，部分一致 | 绝对日期锚点与附录细节 |
| 最多五个预训练 TL 模型 | 上限为 5 | 默认可同时跑论文轨道与扩展轨道 | 严格模式下论文轨道上限 5，超限阻断 | 约束机制一致 | 论文对计数口径的更细定义 |
| source_count 与 pretrained_model_count | 论文强调预训练模型上限 | 两字段分开记录，可能出现不等值 | 严格模式要求可追踪并受上限约束 | 否，需逐结果核验 | 特定方法下字段映射规则 |
| 论文复现 vs 扩展实验归属 | 论文结果应限制在论文协议范围 | 默认允许扩展配置共存 | 严格模式阻断扩展配置混入论文结果 | 分流机制一致 | 旧结果是否全部重算清理 |

## 2. 评估口径与 metric_space

### 2.1 论文原始设定

- 论文指标体系可确认包含 RMSE、Accuracy。
- 仅凭仓库内信息无法确认论文是否明确要求在 original sales 空间计算。

### 2.2 仓库当前默认设定

- 默认输出记录在 normalized_minmax_space。
- Accuracy 由 RMSE 派生，公式为 1 / (RMSE + 1e-8)。

### 2.3 严格论文模式设定

- 启用 strict_paper_metrics 后，评估链路可在 original_sales_space 统一输出。
- 该机制是审查保护，不等价于“论文原始实现已证实”。

### 2.4 是否完全一致

- 结论：部分一致。

### 2.5 仍未确认的部分

- 论文 metric space 的逐字定义与边界条件。

## 3. 数据切分窗口

### 3.1 论文原始设定

- 目标域 observed 约 1 个月。
- 目标域 forecast 约 6 个月。

### 3.2 仓库当前默认设定

- 通过 paper_split_protocol 统一 observed=30、forecast=180。
- 使用时间顺序的相对窗口切分。

### 3.3 严格论文模式设定

- strict_paper_split=true 时，必须满足 30+180。
- 数据不足直接报错，不允许静默回退。
- strict_paper_mode 还会启用 dataset-specific split：
	- Dataset1: 15(train) + 15(val) + 180(test)
	- Dataset2: 14(train) + 15(val) + 179(test)
	- Dataset3: 15(train) + 15(val) + 180(test)

### 3.4 是否完全一致

- 结论：部分一致（按论文相对窗口复刻）。

### 3.5 仍未确认的部分

- 论文绝对时间边界是否固定到具体日期。

## 4. 最多五个预训练 TL 模型

### 4.1 论文原始设定

- 论文文字可确认上限为 5。

### 4.2 仓库当前默认设定

- 默认仍支持扩展 source_count，用于扩展实验。

### 4.3 严格论文模式设定

- 论文轨道约束 pretrained_model_count <= 5。
- 超出上限时终止执行论文轨道。

### 4.4 是否完全一致

- 结论：在代码约束层面一致。

### 4.5 仍未确认的部分

- 各方法内部“预训练模型”计数是否与论文术语完全同义。

## 5. source_count 与 pretrained_model_count 的关系

### 5.1 论文原始设定

- 论文强调多源与预训练上限，但工程字段命名需要映射。

### 5.2 仓库当前默认设定

- source_count 是选源规模。
- pretrained_model_count 是实际预训练并用于迁移的模型数量。

### 5.3 严格论文模式设定

- 在结果行记录 requested_source_count 与 actual_pretrained_model_count。
- 违反论文上限或轨道规则时阻断。
- 严格 source selection：SS-TL 固定 KNN top-1，多源方法固定 KNN top-3。

## 8. source identification 报告

- 新增：
	- `outputs/paper_alignment/source_identification_report.csv`
	- `outputs/paper_alignment/source_identification_report.json`
- 报告可用于核对 source_key、distance、weight。
- Dataset3 同区域约束在当前 CSV 下仍有 `PARTIAL/TODO`：缺少 region 元数据时会显式记录回退说明。

### 5.4 是否完全一致

- 结论：部分一致，需逐运行核验。

### 5.5 仍未确认的部分

- 样本不足、回退路径下两字段的可比性边界。

## 6. 哪些结果属于论文复现，哪些属于扩展实验

### 6.1 论文原始设定

- 论文复现结果应仅覆盖论文协议配置。

### 6.2 仓库当前默认设定

- 可同时生成论文轨道与扩展轨道结果。

### 6.3 严格论文模式设定

- 论文复现结果：full_paper_results.csv。
- 扩展实验结果：extended_results.csv。
- 严格模式禁止扩展配置写入论文复现结果。

### 6.4 是否完全一致

- 结论：分流机制一致。

### 6.5 仍未确认的部分

- 历史旧结果是否全部按当前协议重新生成。

## 7. 审查时建议输出

- 先看运行配置，再看结果字段，再看校验脚本报告。
- 对任何“部分一致”条目，必须保留未确认说明，不可改写为“已证实一致”。
- 关联文档：[docs/paper_protocol_alignment.md](docs/paper_protocol_alignment.md)。
