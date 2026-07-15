# 论文复现评估版仓库（MSML-TL）

本仓库用于复现论文《A multi-source multi-layer-based transfer learning approach for forecasting customer demands of newly launched products》，并提供从原始 CSV → 标准化数据 → 迁移学习训练/评估 → 批量实验矩阵 → 结果表格/图像 → 统计显著性检验的一体化流水线。

仓库主入口以“可复现评审”为目标：配置集中、脚本可直接运行、输出目录固定、结果文件格式统一。

---

## 项目概览

### 主要能力

- 统一数据处理与切分：标准化为 date/entity_id/item_id/sales，并生成时间特征列。
- 统一训练与评估：输出 RMSE、Accuracy、prediction_shape，并记录失败原因。
- 论文尺度批量实验：三数据集 × 多方法 × 信息共享场景 × source_count 敏感性。
- 实验矩阵运行器：可生成矩阵快照与 [outputs/matrix_runs/master_results.csv](outputs/matrix_runs/master_results.csv)。
- 结果可视化与表格：生成格式化排名表与 RMSE/Accuracy 柱状图。
- 统计显著性分析：Friedman + Wilcoxon + 平均排名（Average Rank）。

### 已实现的方法

仓库内统一评估管线包含以下方法（与默认配置一致）：

1. No-TL：仅使用目标域数据训练，无迁移学习。
2. SS-TL：单源迁移学习基线。
3. MSWA-TL：多源加权聚合（weighted aggregation）。
4. MSSB-TL：多源分阶段迁移（stage-based）。
5. MSML-TL：多源多层迁移主方法。
6. MSML-TL-RFE：在 MSML-TL 上叠加 RFE 特征筛选。

说明：一键启动器的统一报告中会将 MSML-TL 显示为“MSADW-TL”（仅展示别名，不改变内部实现）。对应逻辑在 [scripts/launcher.py](scripts/launcher.py)。

### 指标口径（以代码为准）

- RMSE：标准回归 RMSE。
- Accuracy：派生指标，定义为 $1/(RMSE+\varepsilon)$（默认 $\varepsilon=1e-8$）。实现见 [src/evaluation/metrics.py](src/evaluation/metrics.py)。
- prediction_shape：预测张量/数组形状的可读表示。

## 评估口径对齐

- 论文 Accuracy 定义：当前仓库按论文复现协议使用 `1 / (RMSE + 1e-8)`。
- 当前仓库默认口径：默认在 `normalized_minmax_space`（归一化后的 sales 空间）计算 RMSE/Accuracy。
- 严格论文模式口径：当 `strict_paper_mode=true` 时会自动启用 `strict_paper_metrics=true`，统一在评估阶段先做反归一化，再在 `original_sales_space` 计算 RMSE/Accuracy。
- 两者是否完全一致：当前状态为 `PARTIAL`。
	- 已一致部分：严格模式下已实现统一反归一化评估链路，No-TL、SS-TL、MSWA-TL、MSSB-TL、MSML-TL、MSML-TL-RFE 全部走同一套评估函数。
	- 未确认部分：论文原文对 metric space 的逐字描述与边界细节仍需外部证据补齐，仓库不会声称“完全一致”。

结果 CSV 指标字段说明：

- `metric_space_current`：当前工程默认评估空间。
- `metric_space_paper`：严格论文口径目标评估空间。
- `paper_metric_aligned`：该行 `rmse/accuracy` 是否按论文口径输出。
- `inverse_transform_applied`：该行指标是否执行了反归一化。
- `metric_notes`：评估链路备注（含回退或占位信息）。

新增校验脚本：

- `python3 scripts/check_metric_alignment.py`
	- 输出 `outputs/paper_alignment_reports/metric_alignment_check_report.csv`
	- 输出 `outputs/paper_alignment_reports/metric_alignment_check_report.json`

---

## 当前状态

以下内容基于仓库当前文件与目录的“可见事实”，不假设你本机已成功跑通：

- 入口脚本齐全：单实验、论文全量、矩阵、对齐 smoke test、检查与统计分析脚本均在 [scripts](scripts) 下可见。
- 输出目录已存在且包含结果样例：例如 [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv)、[outputs/results_reports/full_paper_results_formatted.csv](outputs/results_reports/full_paper_results_formatted.csv) 等。
- 依赖清单在 [requirements.txt](requirements.txt)；Python 版本要求以启动器检查为准（启动器要求 Python >= 3.9）。
- 仓库同时存在“根目录实现”和“src 镜像实现”两套代码路径；主运行脚本默认走根目录模块，说明见 [docs/full_code_walkthrough.md](docs/full_code_walkthrough.md)。

无法确认但可能与复现相关的内容统一标记为“待补充”。

## 最终对齐状态

本节用于明确区分两层含义，避免把“工程/结果层已完成”误读为“论文原始证据层已完全闭合”。

- 工程/结果层：指结果 CSV、paper/extended 分流、结果层 metric 输出、相对窗口 split 复刻等已经在仓库产物与流程中落地的部分。
- 论文原始证据层：指需要论文正文、附录、原始实验说明或外部可核验证据进一步闭合的部分；这部分当前仍保持保守标记，不改写为已完成。

| 项目 | 当前状态 | 说明 |
|---|---|---|
| 结果 CSV 落地 | PASS | 论文轨道与扩展轨道结果已独立落盘，结果层产物已完成。 |
| paper/extended 分流 | PASS | 论文结果与扩展结果已分流，不再将扩展轨道表述为论文主结果。 |
| metric 结果层 | PASS | 结果层 metric 字段、对齐标记与错误清理已完成，结果输出层面不再保留未修复错误。 |
| split 相对窗口复刻 | PASS | 以 30 天 observed + 180 天 forecast 的相对窗口复刻已完成。 |
| 论文绝对边界证据 | PARTIAL | 论文绝对日期边界、附录细节或等价外部证据仍不足，因此继续保守标记。 |
| 论文原始 metric 外部证据 | PARTIAL | 结果层已完成，但论文对原始 metric space 的外部证据仍不充分，因此继续保守标记。 |

当前收官说明：

- Dataset3 qty_key 报错已修复。
- error 行已从 19 降为 0。
- not_paper_original_metric_rows 已从 19 降为 0。
- overall_level 仍为 PARTIAL，原因不再是代码或结果错误，而是论文原始证据层仍需保守保留缺口说明。
- 现有 PARTIAL 与 TODO 语义继续保留，用于标记外部论文证据尚未闭合的部分，而不是标记工程实现失败。

## 论文协议对齐状态

本仓库当前以“可审查的严格论文复现”为目标，不再默认把实验结果视为已与论文完全一致。所有协议状态都会进入配置、结果 CSV 与独立校验脚本。

### 总表（论文协议对齐状态）

| 协议主题 | 论文原始设定 | 仓库当前默认设定 | 严格论文模式设定 | 是否完全一致 | 仍未确认的部分 |
|---|---|---|---|---|---|
| 评估口径与 metric_space | 论文使用 RMSE/Accuracy，metric space 细节在仓库内证据不足 | 默认 normalized_minmax_space | strict_paper_metrics 可切换到 original_sales_space | 否，部分一致 | 论文是否要求固定在原始量纲评估 |
| 数据切分窗口 | 目标域约 1 个月 observed + 约 6 个月 forecast | 30 天 observed + 180 天 forecast（相对窗口） | strict_paper_split=true 时强制 30+180，不足报错 | 否，部分一致（按论文相对窗口复刻） | 论文绝对日期边界与附录细节 |
| 最多五个预训练 TL 模型 | 论文文字可确认上限为 5 | 默认允许论文轨道与扩展轨道并存 | 论文轨道强制 <=5，超限阻断 | 约束机制一致 | 论文“预训练模型”计数细则 |
| source_count 与 pretrained_model_count 关系 | 论文强调上限，但工程字段需映射 | 两字段分开记录，可能不总是相等 | 严格模式下记录 requested 与 actual 并做约束 | 否，部分一致 | 特定方法与异常路径下的严格对应关系 |
| 论文复现结果与扩展实验边界 | 论文结果应限定在论文协议范围 | 默认允许扩展配置共存运行 | 严格模式下扩展配置不能写入论文结果 | 分流机制一致 | 历史旧结果是否已全部按新协议重算 |

- Metric 对齐状态：`PARTIAL`
- Split 对齐状态：`PARTIAL`
- Source / pretrained-model 协议状态：`ALIGNED`

状态定义：

- `ALIGNED`：当前代码与配置已经有明确协议约束，且可自动校验。
- `PARTIAL`：当前代码行为已经被准确写清楚，但论文原始证据尚未完全确认，不能声称“完全一致”。
- `TODO`：尚未确认，必须保留占位，不允许伪装成已对齐。
- `EXTENDED`：该实验明确属于扩展实验，不应混入论文主结果表。

当前明确结论：

- 当前评估口径是 `normalized_minmax_space`，即在训练集拟合的 MinMax 归一化空间里计算 RMSE，并派生 `Accuracy = 1 / (RMSE + 1e-8)`；论文原始 metric space 仍未确认，因此标记为 `PARTIAL`。
- 当前目标域切分窗口已被显式写成“最近 30 天 train+val + 最近 180 天 test，再在该窗口内按 0.067 / 0.067 / 0.866 做时间切分”；该规则已可审查，但论文原始绝对窗口边界尚未确认，因此标记为 `PARTIAL`。
- 当前多源 TL 的论文轨道强制最多五个预训练 source model；`k > 5` 只进入扩展实验输出，并在 strict paper mode 下直接阻断。

## 论文原始 source / pretrained-model 设定

- 论文原始表述是什么：贡献部分明确写到“pre-training up to five different TL models”，即论文主设定关注的是“最多 5 个预训练迁移模型”的多源迁移评估。
- 当前仓库默认跑什么：默认多源敏感性包含 `source_count in {1,3,5,6,9}`，其中 `{1,3,5}` 为论文轨道，`{6,9}` 为仓库扩展轨道。
- 哪些属于论文内实验：
	- 多源 TL 方法（MSWA-TL / MSSB-TL / MSML-TL / MSML-TL-RFE）仅当 `source_count <= 5` 且属于 `paper_source_protocol.paper_source_counts` 时记为 `experiment_scope=paper`。
	- SS-TL 固定 1 个预训练模型，No-TL 为 0。
- 哪些属于仓库扩展实验：
	- `source_count in {6,9}` 或任何超过论文上限 5 的多源设置，统一记为 `experiment_scope=extended`，仅写入 `extended_results.csv`。
	- strict paper mode 下这类设置不会进入论文主结果，且会被阻断。

关键配置位于 [configs/default_config.json](configs/default_config.json)：

- `paper_reproduction.paper_source_protocol.max_pretrained_models_from_paper = 5`
- `paper_reproduction.paper_source_protocol.default_paper_multi_source_count = 3`
- `paper_reproduction.paper_source_protocol.allow_extended_source_counts = true/false`

字段语义（避免混淆）：

- `number_of_sources` / `source_count`：参与多源迁移的 source item 数量（例如 top-k 选源数）。
- `number_of_pretrained_models` / `pretrained_model_count`：实际训练并参与迁移或融合的源模型数。
- `number_of_methods`：一次运行启用的方法数量（方法维度，不等同于 source 数或模型数）。

独立校验脚本：

- [scripts/validate_metric_alignment.py](scripts/validate_metric_alignment.py)
- [scripts/validate_split_alignment.py](scripts/validate_split_alignment.py)
- [scripts/validate_source_pretrained_protocol.py](scripts/validate_source_pretrained_protocol.py)
- [scripts/validate_paper_protocol_strict.py](scripts/validate_paper_protocol_strict.py)
- 详细说明见 [docs/paper_protocol_alignment.md](docs/paper_protocol_alignment.md)
- 论文复现/扩展边界说明见 [docs/paper_vs_extended_experiments.md](docs/paper_vs_extended_experiments.md)
- 严格模式说明见 [docs/paper_strict_alignment.md](docs/paper_strict_alignment.md)

## 论文对齐说明

本仓库当前支持“严格论文协议开关”（`strict_paper_mode` 与兼容别名 `paper_strict_mode`）。

启用后会执行以下行为：

1. 统一协议预检：启动脚本会先检查 metric/split/source 配置结构，发现违反论文轨道硬约束会直接报错。
2. 切分窗口断言：目标域窗口必须与配置中的 `30(train+val)+180(test)` 一致；若运行时窗口日期跨度不一致会立即失败。
3. Source 模型上限：多源 TL 在论文轨道仅允许 `k in {1,3,5}` 且最多 5 个预训练模型；扩展设置不会混入论文结果。
4. TODO 保留策略：凡论文原文证据未确认项（如 paper metric space）保持 `TODO_*`，不会被伪装为已对齐。

统一校验命令：

		python3 scripts/validate_paper_protocol_strict.py

严格失败模式：

		python3 scripts/validate_paper_protocol_strict.py --strict-paper-mode

输出文件：

- [outputs/paper_alignment_reports/paper_protocol_strict_validation.csv](outputs/paper_alignment_reports/paper_protocol_strict_validation.csv)
- [outputs/paper_alignment_reports/paper_protocol_strict_validation.json](outputs/paper_alignment_reports/paper_protocol_strict_validation.json)

## 论文数据切分协议

论文正文的核心假设为：目标域可观测数据约 1 个月，用于训练+验证；预测后续约 6 个月。当前仓库对该协议采用“相对窗口复刻”策略，而非伪造绝对日期一致。

统一配置入口（默认）：

- `paper_reproduction.paper_split_protocol.target_observed_window_days = 30`
- `paper_reproduction.paper_split_protocol.target_forecast_window_days = 180`
- `paper_reproduction.paper_split_protocol.validation_strategy = time_holdout`
- `paper_reproduction.paper_split_protocol.rolling_or_fixed_split = rolling_recent_days`
- `paper_reproduction.paper_split_protocol.source_selection_window = full_history`
- `paper_reproduction.paper_split_protocol.source_pool_scope = all_source_items`

严格模式：

- `paper_reproduction.strict_paper_split = true`（兼容别名 `paper_strict_split`）
- 严格模式下若目标域可用长度不足 `30 + 180`，会直接报错，不允许静默回退。
- 严格模式会同时要求窗口范围与唯一日期数满足协议窗口。

每个数据集的实际切分摘要会自动落盘：

- `outputs/paper_alignment/split_protocol_dataset1.json`
- `outputs/paper_alignment/split_protocol_dataset2.json`
- `outputs/paper_alignment/split_protocol_dataset3.json`

自动校验脚本：

- `python3 scripts/check_paper_split_alignment.py`

该脚本会检查：

- target observed 是否约等于 1 个月
- target forecast 是否约等于 6 个月
- source/target 泄漏
- validation 是否存在且时间顺序合理

### 协议对齐表

| 维度 | 论文设定 | 当前仓库默认设定 | 严格论文模式设定 | 是否一致 |
|---|---|---|---|---|
| target observed window | 约 1 个月 | 30 天（相对窗口） | 强制 30 天，不足报错 | 部分一致（按论文相对窗口复刻） |
| target forecast window | 约 6 个月 | 180 天（相对窗口） | 强制 180 天，不足报错 | 部分一致（按论文相对窗口复刻） |
| target validation strategy | 训练窗口内含验证集 | 时间顺序划分（ratio/dates） | 强制时间顺序验证且不能为空 | 部分一致（按论文相对窗口复刻） |
| rolling or fixed split | 论文未给绝对日期细节 | rolling_recent_days | rolling_recent_days（硬约束） | 部分一致（按论文相对窗口复刻） |
| source selection window | 使用 source 历史支持 target 预测 | full_history | full_history | 一致（实现层） |
| source pool scope | 论文强调多源 TL | all_source_items（可叠加信息共享场景过滤） | all_source_items（可叠加场景过滤） | 一致（实现层） |

数据集说明（Dataset1/2/3）：当前仓库均为“按论文相对窗口复刻”。若论文原文未给出精确绝对日期，不宣称“绝对时间完全一致”。

### 严格论文版数据构造（dataset-specific）

在 `strict_paper_mode=true` 下，数据构造会切换到数据集专用协议：

- Dataset1：限定 store 1/2/3，source 候选为 item 1..9，target 固定为 store 1 的 item 10；target split 为 15 天 train + 15 天 val + 180 天 test。
- Dataset2：target 固定为 Brand B1 的 Item 10；target split 为 14 天 train + 15 天 val + 179 天 test。
- Dataset3：任务保持 store-level overall sales，target 固定为 Store 10；target split 为 16 天 train + 15 天 val + 181 天 test。

source selection 严格规则：

- SS-TL：KNN 最近单源（top-1）。
- 多源方法（MSWA-TL/MSSB-TL/MSML-TL/MSML-TL-RFE）：KNN 最近三源（top-3）。

Dataset3 区域约束说明：

- 论文要求“without_information_sharing 仅同区域 source”。
- 当前原始 CSV 不包含显式 region 元数据，因此该条在当前仓库标记为 `PARTIAL/TODO`，并在日志与 source identification 报告中保留说明，不伪装为完全一致。

source identification 审计产物：

- [outputs/paper_alignment/source_identification_report.csv](outputs/paper_alignment/source_identification_report.csv)
- [outputs/paper_alignment/source_identification_report.json](outputs/paper_alignment/source_identification_report.json)

---

## 方法说明（从调用与数据流角度）

主运行器以统一接口组织方法：先准备 source/target，再按方法调用并抽取指标。

- 统一实验运行器：根目录 [experiment_runner.py](experiment_runner.py)
	- `prepare_base_data_for_experiments(...)`：加载与标准化、切分 source/target。
	- `run_all_experiments(...)`：按 enabled_methods 依次运行各方法，统一返回结构。

方法实现既存在于根目录（如 [msml_tl.py](msml_tl.py)、[mswa_tl.py](mswa_tl.py) 等），也在 [src/transfer_methods](src/transfer_methods) 下有对应实现与辅助模块。

---

## 数据说明

### 支持的数据集

当前默认配置包含三套基准数据：

- Dataset1（需求预测挑战赛）：[Dataset 1/train.csv](Dataset%201/train.csv)
- Dataset2（意大利面需求）：[Dataset 2.csv](Dataset%202.csv)
- Dataset3（Rossmann 门店）：[Dataset 3/train ross.csv](Dataset%203/train%20ross.csv)

数据路径映射以 [configs/dataset_paths.json](configs/dataset_paths.json) 和 [configs/default_config.json](configs/default_config.json) 中的 dataset_paths 为准。

### 期望的统一字段

标准化后的最小字段要求：

- date
- entity_id
- item_id
- sales

说明与注意事项见 [data/README_data.md](data/README_data.md)。

---

## 配置说明

### 论文复现主配置（推荐）

论文复现/批量实验主入口默认读取 JSON 配置：

- [configs/default_config.json](configs/default_config.json)
	- dataset_paths：三数据集路径
	- features.default_feature_cols：默认特征列
	- methods：方法集合
	- single_experiment：单实验参数
	- matrix：矩阵参数（horizons、source_counts、weight_modes、keep_ratios 等）

如果你需要切换数据位置，优先修改：

- [configs/dataset_paths.json](configs/dataset_paths.json)
- 或者直接修改 [configs/default_config.json](configs/default_config.json) 的 dataset_paths

### 配置系统模块（Config/YAML）

仓库还包含一个更通用的配置系统（用于部分模块与脚本）：

- 配置文件：[config.yaml](config.yaml)、[supply_chain.yaml](supply_chain.yaml)
- 配置类：[config.py](config.py)
- 环境与日志：[environment.py](environment.py)
- 自检脚本：[init_check.py](init_check.py)、[verify_bootstrap.py](verify_bootstrap.py)

提示：主“论文复现脚本”（例如 [scripts/run_full_paper_experiments.py](scripts/run_full_paper_experiments.py)）不依赖 config.yaml；但根目录运行器中的部分函数在未显式传入 data_path/config 时，会回退使用 config.yaml 推断路径（见 [experiment_runner.py](experiment_runner.py)）。

---

## 运行方式

只有下文“D1–D6 正式协议运行”是 `d1_d6_sealed_v1` 的封存入口。其余一键入口、旧论文脚本、单实验、历史矩阵、smoke、统计和可视化路径均为**非封存兼容路径**：可用于诊断或历史复现，但其输出不能进入正式 acceptance，也不能被标记为 `SEALED_SUCCESS`。

### 1) 一键运行（macOS）

入口：[run_benchmark.command](run_benchmark.command)

该脚本会调用 [scripts/launcher.py](scripts/launcher.py)。启动器会：

- 在固定位置创建/复用共享虚拟环境：~/.msml_tl_env
- 按 [requirements.txt](requirements.txt) 安装依赖，并用指纹文件避免重复安装
- 运行论文全量实验脚本并打印统一报告

执行方式（两种其一）：

- Finder 双击运行 [run_benchmark.command](run_benchmark.command)
- 或在终端运行：

		./run_benchmark.command

### 2) 论文全量实验（推荐评审入口）

入口脚本：[scripts/run_full_paper_experiments.py](scripts/run_full_paper_experiments.py)

它会覆盖：

- DATASETS = Dataset1/2/3
- METHODS = No-TL / SS-TL / MSWA-TL / MSSB-TL / MSML-TL / MSML-TL-RFE
- 信息共享场景：with_information_sharing / without_information_sharing
- 论文轨道 source_count：k ∈ {1, 3, 5}（对 No-TL/SS-TL 固定为 k=1）
- 扩展轨道 source_count：k ∈ {6, 9}

运行：

		python3 scripts/run_full_paper_experiments.py --verbose-mode summary

严格论文模式：

		python3 scripts/run_full_paper_experiments.py --verbose-mode summary --strict-paper-mode

输出（示例路径）：

- [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv)
- [outputs/experiment_results/extended_results.csv](outputs/experiment_results/extended_results.csv)
- [outputs/results_reports/full_paper_results_formatted.csv](outputs/results_reports/full_paper_results_formatted.csv)
- [outputs/results_reports/full_paper_rmse_bar.png](outputs/results_reports/full_paper_rmse_bar.png)
- [outputs/results_reports/full_paper_accuracy_bar.png](outputs/results_reports/full_paper_accuracy_bar.png)

说明：

- [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv) 只保存论文轨道结果。
- [outputs/experiment_results/extended_results.csv](outputs/experiment_results/extended_results.csv) 只保存超出论文协议上限的扩展实验。
- 开启 `--strict-paper-mode` 后，扩展轨道配置会被拒绝执行，不会混入论文结果表。

### 3) 单实验（快速验证）

入口脚本（推荐用 scripts 版本，支持 verbose-mode 与 include_sales_in_knn 开关）：

- [scripts/run_main_experiment.py](scripts/run_main_experiment.py)

运行：

		python3 scripts/run_main_experiment.py --verbose-mode summary

默认会读取 [configs/default_config.json](configs/default_config.json) 的 single_experiment 配置，并把结果写入：

- [outputs/experiment_results/dataset1_results.csv](outputs/experiment_results/dataset1_results.csv)

仓库根目录也提供了一个简化入口 [run_main_experiment.py](run_main_experiment.py)（不带命令行参数）。

### 4) D1–D6 正式协议运行

正式运行由 [scripts/parallel_mode_runner.sh](scripts/parallel_mode_runner.sh) 统一监督；[scripts/parallel_runner.sh](scripts/parallel_runner.sh) 只是委托给同一 supervisor 的命令兼容别名，不提供任何数据或结果 fallback。supervisor 默认最多同时运行 6 个 `dataset × mode` worker，D5 固定最多 1 个。每个 mode 包含 5 个 seed bundle，每个 bundle 一次覆盖 h1–h5，因此全局计划固定为 12 个 mode、60 个 seed bundle。只有全部 mode 和 bundle 验收成功后，父进程才执行 global acceptance 与原子发布。

正式运行要求 Git worktree 干净。每次新运行会创建独立的 `outputs/runs/<run-id>_formal/`，保存不可变 `run_plan.json`、12 个隔离 mode 目录，以及带 acceptance、manifest 和 SHA-256 的正式结果。不要手工复制或拼接 CSV。

```bash
# 只读解析六个数据身份、provenance、日期窗口、完整 predictor/KNN schema、
# source repair 计数、12 个 cache 身份、schema digest、60-bundle 计划和线程依赖；
# 不创建 run root、attempt、cache 或正式结果
DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh

# 服务器 Terminal：新的正式全量运行；RUN_ROOT 必须尚不存在
MAX_JOBS=6 RUN_ROOT="outputs/runs/<new_run_id>" \
  bash scripts/parallel_mode_runner.sh

# 服务器 Terminal：恢复已有运行；仅复用身份完全匹配且已 accepted 的 bundle/mode
MAX_JOBS=6 RUN_ROOT="outputs/runs/<existing_run_id>" RESUME=1 \
  bash scripts/parallel_mode_runner.sh

# 封存前四-mode 服务器 probe；永不发布 global aggregate
python tools/protection/codex_timeout.py --timeout 180 \
  env PROBE=1 PUBLISH_GLOBAL=0 MAX_JOBS=4 \
  RUN_ROOT="outputs/runs/<probe-id>_formal" \
  bash scripts/parallel_mode_runner.sh
```

probe 若由保护器以退出码 124 终止，不要拆分、简化、重试或续跑；请在服务器 Terminal 手工执行同一条命令。

#### 恢复与封存语义

- `attempts/` 和 scheduler events 是 append-only attempts：恢复会新建 attempt，旧 attempt、失败证据和已接受记录不原地改写。只有哈希、代码、输入、协议与 schema 身份全部匹配的 accepted bundle 才可复用。
- resume 使用 heartbeat lease 与 fencing token。活跃 lease、过期 fencing token 或 compare-and-swap 冲突会以 `resume_lease_conflict` 失败；过期 in-flight 工作会被记为 orphaned，不能直接转成 accepted。
- rehydrate 只处理“正式 artifact 缺失且存在已注册可信副本”的情况，在新 attempt 中校验并原子切换 binding，过程不会调用 fit/predict。字节损坏或 digest 不匹配不会自动 rehydrate；下游一旦调度，binding 集合即冻结。
- `complete_unsealed` 表示 12 个 mode 已聚合但仍等待最终 trace gate。`sealed_success` 与 `sealed_failed` 都是终态；成功时 `SEALED_SUCCESS` 标记最后写入。终态之后禁止 resume、rehydrate 或覆盖发布。
- `outputs/experiment_results/`、旧 300-cell run plan、legacy CSV 聚合和各历史脚本输出都是非封存兼容路径，不得作为 resume 或正式验收来源。

### 5) 历史参数矩阵（非 D1–D6 正式入口）

入口脚本：

- [scripts/run_full_experiment_matrix.py](scripts/run_full_experiment_matrix.py)

运行：

		python3 scripts/run_full_experiment_matrix.py --verbose-mode summary

它会使用 [configs/default_config.json](configs/default_config.json) 的 matrix 参数，并在 [outputs/matrix_runs](outputs/matrix_runs) 下生成：

- 每个 experiment_id 目录下的结果文件（示例：[outputs/matrix_runs/dataset1_h1_k3_inverse_rfe50_smoke/results.csv](outputs/matrix_runs/dataset1_h1_k3_inverse_rfe50_smoke/results.csv)）
- 总表：[outputs/matrix_runs/master_results.csv](outputs/matrix_runs/master_results.csv)
- 矩阵快照：[outputs/matrix_runs/matrix_snapshot.json](outputs/matrix_runs/matrix_snapshot.json)

### 5) Paper alignment smoke test（小规模对齐验证）

入口脚本：[scripts/run_paper_alignment_smoke_test.py](scripts/run_paper_alignment_smoke_test.py)

严格模式运行：

		python3 scripts/run_paper_alignment_smoke_test.py --strict-paper-mode

输出示例：

- [outputs/paper_alignment_smoke_test/dataset1_alignment_smoke_results.csv](outputs/paper_alignment_smoke_test/dataset1_alignment_smoke_results.csv)

### 6) 结果可视化/报告

入口脚本：[scripts/run_result_visualization.py](scripts/run_result_visualization.py)

它会读取 [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv)，并在 [outputs/results_reports](outputs/results_reports) 下生成格式化表与图。

### 7) 统计显著性分析

入口脚本：[scripts/run_statistical_analysis.py](scripts/run_statistical_analysis.py)

默认输入优先选择 [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv)，并输出到：

- [outputs/statistical_reports/friedman_test_results.csv](outputs/statistical_reports/friedman_test_results.csv)
- [outputs/statistical_reports/wilcoxon_pairwise_results.csv](outputs/statistical_reports/wilcoxon_pairwise_results.csv)
- [outputs/statistical_reports/method_average_ranks.csv](outputs/statistical_reports/method_average_ranks.csv)
- [outputs/statistical_reports/method_average_rank_bar.png](outputs/statistical_reports/method_average_rank_bar.png)
- [outputs/statistical_reports/wilcoxon_significance_table.csv](outputs/statistical_reports/wilcoxon_significance_table.csv)

### 8) 运行前检查与自检

- 依赖/环境检查（脚本目录）：
	- [scripts/check_runtime_environment.py](scripts/check_runtime_environment.py)
	- [scripts/check_training_environment.py](scripts/check_training_environment.py)
- 数据管道检查：
	- [scripts/check_all_dataset_pipelines.py](scripts/check_all_dataset_pipelines.py)
	- 输出： [outputs/pipeline_checks/all_dataset_pipeline_check.csv](outputs/pipeline_checks/all_dataset_pipeline_check.csv)
- 最小训练检查：
	- [scripts/check_all_dataset_min_train.py](scripts/check_all_dataset_min_train.py)
	- 输出： [outputs/pipeline_checks/all_dataset_min_train_check.csv](outputs/pipeline_checks/all_dataset_min_train_check.csv)
- 配置系统自检（Config/YAML 路线）：
	- [init_check.py](init_check.py)
	- [verify_bootstrap.py](verify_bootstrap.py)

---

## 输出说明

### 核心输出目录

- [outputs/experiment_results](outputs/experiment_results)：单实验与论文全量实验的结果 CSV。
- [outputs/paper_alignment_reports](outputs/paper_alignment_reports)：metric / split / source-protocol 的独立校验结果。
- [outputs/results_reports](outputs/results_reports)：格式化表格与可视化图像（RMSE/Accuracy）。
- [outputs/pipeline_checks](outputs/pipeline_checks)：数据管道与最小训练检查结果。
- [outputs/paper_alignment_smoke_test](outputs/paper_alignment_smoke_test)：对齐 smoke test 输出。
- [outputs/statistical_reports](outputs/statistical_reports)：Friedman/Wilcoxon/平均排名等统计报告。
- [outputs/matrix_runs](outputs/matrix_runs)：矩阵跑批的分组结果与快照。

矩阵目录下的总表与快照文件示例：

- [outputs/matrix_runs/master_results.csv](outputs/matrix_runs/master_results.csv)
- [outputs/matrix_runs/matrix_snapshot.json](outputs/matrix_runs/matrix_snapshot.json)

### 结果文件的字段含义（以 [outputs/experiment_results/full_paper_results.csv](outputs/experiment_results/full_paper_results.csv) 为例）

CSV 字段通常包含：

- dataset：Dataset1/2/3
- method：方法名（内部名）
- information_sharing：信息共享场景
- source_count：k
- experiment_track：`paper` 或 `extended`
- strict_paper_mode：是否启用严格论文模式
- alignment_status：总对齐状态（`ALIGNED` / `PARTIAL` / `TODO` / `EXTENDED`）
- metric_alignment_status：评估口径对齐状态
- split_alignment_status：数据切分窗口对齐状态
- source_pretrained_alignment_status：source / pretrained-model 协议对齐状态
- metric_space：当前代码实际评估空间
- paper_metric_space：论文原始 metric space；若未确认则显式写成 `TODO_*`
- paper_split_reference：论文原始 split 依据；若未确认则显式写成 `TODO_*`
- requested_source_count：请求的 source 数量
- actual_pretrained_model_count：实际参与 TL 的 source model 数量
- rmse / accuracy / prediction_shape
- alignment_notes：当前对齐状态的说明文本
- error：失败原因（空字符串表示成功）

---

## 项目结构

建议先从“入口脚本 → 运行器 → 数据预处理 → 方法实现 → 评估与可视化”的路径阅读：

- 入口与脚本
	- [scripts](scripts)：推荐从这里运行（参数更全，报告更统一）。
	- [run_benchmark.command](run_benchmark.command)：macOS 一键入口。
- 配置
	- [configs/default_config.json](configs/default_config.json)：论文复现实验主配置。
	- [configs/dataset_paths.json](configs/dataset_paths.json)：数据路径映射。
- 核心流水线（根目录实现）
	- [experiment_runner.py](experiment_runner.py)：统一实验运行器。
	- [experiment_matrix_runner.py](experiment_matrix_runner.py)：矩阵运行器。
	- [data_preprocessing.py](data_preprocessing.py)：数据加载与标准化处理。
	- [result_visualizer.py](result_visualizer.py)：结果汇总与可视化。
- 核心模块（src 镜像实现）
	- [src](src)：与根目录模块对应的一套实现（用于结构化组织与复用）。
- 数据与产物
	- [data](data)：数据说明文档。
	- [outputs](outputs)：所有实验输出。
	- [results](results)：待补充（当前用途需结合内容确认）。

详细结构可参考 [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) 与阅读路线 [PROJECT_INDEX.md](PROJECT_INDEX.md)。

---

## 已知问题

以下问题来自当前仓库脚本/文档与常见运行日志，不阻止代码存在与输出文件生成，但可能影响复现体验：

1. Dataset3 可能出现 pandas DtypeWarning（StateHoliday 混合类型）；通常不阻塞流程。
2. TensorFlow 可能出现 retracing warning；通常不阻塞流程。
3. 与论文表格中的绝对 RMSE 数值可能存在偏差：评估口径、随机性、训练轮数、窗口定义等都会影响（对齐细节：待补充）。
4. 仓库存在两套配置/实现路径（根目录与 src 镜像；JSON 配置与 YAML Config 系统），初读可能混淆。
5. 数据文件名包含空格与括号（例如目录 [rossmann-store-sales (2)](rossmann-store-sales%20(2))）；在部分环境/脚本中可能需要注意路径转义或引用。

---

## 待补充

- 论文原始评估口径与本仓库 metric_space 的精确对齐说明。
- 各方法与论文公式/章节的逐条映射（如需要可补充到 docs）。
# multi
