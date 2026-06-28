# 论文协议对齐说明

本文档面向“论文复现审查”，用于区分以下三类信息：

- 论文原始设定（可从论文文字直接确认）
- 仓库当前默认设定（代码当前真实行为）
- 严格论文模式设定（仓库为复现审查添加的强约束）

若缺少外部证据，本文只写“按论文相对窗口复刻”或“部分一致”，不会写成“已证实一致”。

当前项目统一数据集映射：Dataset1 = 需求预测挑战赛，Dataset2 = 意大利面需求，Dataset3 = Rossmann 门店。本文中所有 Dataset1/2/3 指代均以此映射为准。

## 结果层 vs 证据层

为避免后续阅读者把“结果层已完成”误解为“论文原始证据层已完全闭合”，本项目将最终状态拆分为两层说明。

- 结果层：关注代码、配置、导出结果与校验产物是否已经稳定落地。
- 证据层：关注论文正文、附录、外部说明是否足以把某个设定写成“已证实完全一致”。

当前最终状态如下：

| 项目 | 结果层状态 | 证据层状态 | 说明 |
|---|---|---|---|
| 结果 CSV 落地 | PASS | PASS | 结果文件已经真实生成并独立承载论文轨道与扩展轨道输出。 |
| paper/extended 分流 | PASS | PASS | 工程分流已完成，扩展轨道不再应被解释为论文主结果。 |
| metric 结果层 | PASS | PARTIAL | 结果层 metric 输出、对齐标记与错误清理已完成，但论文原始 metric space 的外部证据仍保守不足。 |
| split 相对窗口复刻 | PASS | PARTIAL | 相对窗口复刻已完成，但论文绝对日期边界证据仍不充分。 |
| 论文绝对边界证据 | 不适用 | PARTIAL | 当前保持保守表述，不把相对窗口复刻改写为绝对边界已证实。 |
| 论文原始 metric 外部证据 | 不适用 | PARTIAL | 当前保持保守表述，不把结果层修复改写为论文原始度量定义已完全证实。 |

收官结论：

- Dataset3 qty_key 报错已修复。
- error 行已从 19 降为 0。
- not_paper_original_metric_rows 已从 19 降为 0。
- overall_level 仍为 PARTIAL，原因不再是代码或结果错误，而是外部论文证据仍需保守缺口说明。
- 本文档保留 PARTIAL 与 TODO 语义，不会因为结果层已经完成而把证据层直接改写为已完成。

## 1. 总体对齐结论

| 协议维度 | 论文原始设定 | 仓库当前默认设定 | 严格论文模式设定 | 是否完全一致 | 仍未确认的部分 |
|---|---|---|---|---|---|
| 评估口径与 metric_space | 论文强调 RMSE 与 Accuracy，但 metric space 细节在仓库内证据不足 | 默认在 normalized_minmax_space 评估 | 可切换为 original_sales_space 的严格评估链路 | 否，当前为部分一致 | 论文是否要求必须在原始量纲评估 |
| 数据切分窗口 | 目标域约 1 个月可观测 + 约 6 个月预测 | 30 天 observed 与 180 天 forecast 的相对窗口 | strict_paper_split=true 时强制 30+180，不足即报错 | 否，当前为部分一致 | 论文绝对日期边界与跨数据集固定起止点 |
| 最多五个预训练 TL 模型 | 明确写到最多五个预训练模型 | 默认允许论文轨道与扩展轨道并存 | 论文轨道仅允许 <=5，超出即阻断 | 代码约束层面一致 | 是否存在论文未公开的额外筛选条件 |
| source_count 与 pretrained_model_count 关系 | 论文表达聚焦“预训练模型个数上限” | 两个字段均记录，可能在异常数据下不等值 | 严格模式要求论文轨道不超过 5 且关系可追踪 | 否，仍需逐运行核验 | 特定方法下两者一一对应的论文原始定义 |
| 论文结果与扩展实验边界 | 论文主结果不包含超上限扩展设置 | 默认可同时运行 paper 与 extended | 严格模式下扩展配置不进入论文轨道 | 代码分流一致 | 历史结果文件中旧记录是否已全部清理 |

## 2. 评估口径与 metric_space

### 2.1 论文原始设定

- 论文主指标是 RMSE 与 Accuracy。
- 仅从仓库内证据无法确认论文是否明确要求在原始 sales 量纲评估。

### 2.2 仓库当前默认设定

- 默认口径记录为 normalized_minmax_space。
- Accuracy 采用 1 / (RMSE + 1e-8)。

### 2.3 严格论文模式设定

- strict_paper_metrics=true 时，先反归一化，再在 original_sales_space 输出指标。
- 该行为用于审查时的可比性控制，不代表论文原始实现已被证实。

### 2.4 是否完全一致

- 当前结论：部分一致，不可写为“已证实一致”。

### 2.5 仍未确认的部分

- 论文对 metric space 的原文边界定义。
- 论文 Accuracy 与仓库公式的逐字等价性。

## 3. 数据切分窗口

### 3.1 论文原始设定

- 目标域可观测窗口约 1 个月。
- 目标域预测窗口约 6 个月。

### 3.2 仓库当前默认设定

- paper_split_protocol 中 observed=30 天，forecast=180 天。
- 切分采用按时间顺序的相对窗口策略。

### 3.3 严格论文模式设定

- strict_paper_split=true 时，目标域长度不足 30+180 立即报错。
- 严格模式禁止静默回退到非论文窗口。
- strict_paper_mode 下新增 dataset-specific target split：
	- Dataset1: 15(train) + 15(val) + 180(test)
	- Dataset2: 14(train) + 15(val) + 179(test)
	- Dataset3: 16(train) + 15(val) + 181(test)

### 3.4 是否完全一致

- 当前结论：部分一致（按论文相对窗口复刻）。

### 3.5 仍未确认的部分

- 论文是否给出统一的绝对日期锚点。
- 各数据集是否存在论文附录中的额外切分过滤。

## 4. 最多五个预训练 TL 模型

### 4.1 论文原始设定

- 论文文字强调最多五个预训练 TL 模型。

### 4.2 仓库当前默认设定

- 默认运行可包含论文轨道与扩展轨道。
- 扩展轨道可出现超过 5 的 source_count。

### 4.3 严格论文模式设定

- 论文轨道限制 pretrained_model_count <= 5。
- 超过上限的配置在严格模式下直接阻断。

### 4.4 是否完全一致

- 当前结论：约束机制层面一致。

### 4.5 仍未确认的部分

- 论文对“预训练模型”计数口径是否与仓库所有方法实现完全同义。

## 5. source_count 与 pretrained_model_count 的关系

### 5.1 论文原始设定

- 论文关注多源与预训练模型数量上限，但字段命名不一定与工程一致。

### 5.2 仓库当前默认设定

- source_count 表示参与选源或迁移的 source 数。
- pretrained_model_count 表示实际训练并参与迁移/融合的模型数。

### 5.3 严格论文模式设定

- 论文轨道会记录 requested_source_count 与 actual_pretrained_model_count，便于审查。
- 若出现违反论文上限或协议不一致的组合，严格模式拒绝执行。
- 严格模式 source selection 约束：
	- SS-TL 固定 KNN top-1 最近单源。
	- 多源方法固定 KNN top-3 最近源。

## 8. source identification 可审计产物

- 新增产物：
	- `outputs/paper_alignment/source_identification_report.csv`
	- `outputs/paper_alignment/source_identification_report.json`
- 报告字段覆盖 dataset/method/scenario/source_key/distance/weight，可用于与论文表 5/6 的 source-distance-weight 逻辑逐行核对。
- Dataset3 的同区域约束当前受原始 CSV 缺失 region 元数据影响，报告中会保留 `PARTIAL/TODO` 说明，不会伪装为 fully aligned。

### 5.4 是否完全一致

- 当前结论：部分一致，需要按运行结果逐行核验。

### 5.5 仍未确认的部分

- 某些方法在失败回退、样本不足时两字段是否应严格一一对应。

## 6. 哪些结果属于论文复现，哪些属于扩展实验

### 6.1 论文原始设定

- 论文复现结果应限定在论文协议允许的配置范围。

### 6.2 仓库当前默认设定

- 同一次批量运行可能同时产出 paper 轨道与 extended 轨道。

### 6.3 严格论文模式设定

- 论文轨道结果写入 full_paper_results.csv。
- 扩展轨道结果写入 extended_results.csv。
- 严格模式下扩展配置不会混入论文结果文件。

### 6.4 是否完全一致

- 当前结论：分流机制一致。

### 6.5 仍未确认的部分

- 旧历史产物是否已全部按新协议重算与清理。

## 7. 审查建议

- 审查时优先查看配置、运行日志、结果字段三者是否同向。
- 对“部分一致”条目，必须保留证据缺口说明，不可改写为“完全一致”。
- 建议与以下文档配合阅读：[docs/paper_vs_extended_experiments.md](docs/paper_vs_extended_experiments.md)。

## 9. Information-sharing Signature Default Policy

本节描述的是当前项目的工程默认策略，不是论文原文已证实要求。

### 9.1 变更背景

- 过去 with_information_sharing 的 static signature 默认包含多种 ID 类静态特征。
	- Dataset2: `brand_code`, `entity_id_code`, `item_id`, `promo`
	- Dataset3: `store_id`, `region_code`, `holiday_promo_profile`, `promo`, `state_holiday`, `school_holiday`, `open`
- 基于内部去 ID 消融验证，当前默认策略调整为“优先保留非身份型画像特征，默认移除强 ID 类静态编码”。

### 9.2 当前默认策略

- 默认移除的 ID 类静态特征：
	- `item_id`
	- `brand_code`
	- `entity_id_code`
	- `store_id`
	- `region_code`
- 默认保留的非身份型画像特征：
	- Dataset2: `promo`
	- Dataset3: `holiday_promo_profile`, `promo`, `state_holiday`, `school_holiday`, `open`

### 9.3 配置开关（保留消融能力）

- 配置项：`paper_reproduction.use_id_static_features_in_signature`
- 默认值：`false`
- 含义：
	- `false`：使用当前稳健默认策略（去 ID）。
	- `true`：恢复 ID 类静态特征，便于后续消融实验或回归对照。

### 9.4 内部验证结论（工程依据）

- 在 Dataset2 与 Dataset3 的 with_information_sharing 场景下，去掉 ID 类静态特征后：
	- source 选择基本稳定（未出现结构性异常变化）。
	- RMSE 未恶化，且在本轮验证中有改善。
- 因此，本项目将“默认去 ID、按需开关恢复”作为工程默认值，以降低 ID 偏好噪声/偏置风险。

### 9.5 边界声明

- 本策略结论来自项目内部验证，不应表述为“论文已证实要求”。
- 现阶段仍保留与论文外部证据相关的 PARTIAL/TODO 条目（例如 split 绝对边界证据、Dataset3 region 元数据缺口）。