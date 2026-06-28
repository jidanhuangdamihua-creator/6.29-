# paper_reproduction_detail_index.md

> 用途：后续代码审计时检查复现是否 `paper-aligned`。  
> 依据：论文 PDF 原文、图、表、公式与 Algorithm 1。上传的 markdown 文件内容显示为本次任务说明，而不是一份独立的 NotebookLM 草稿正文；因此第 17 节仅核查任务说明中列出的“NotebookLM 草稿高风险说法”。

## 状态标签说明

| 状态 | 使用含义 |
| ---- | -------- |
| 论文明确 | 论文正文、表、公式或算法有直接表述。 |
| 论文未明确 | 论文未给出足够可执行细节，复现代码需要显式工程决策。 |
| 论文存在冲突 | 正文、图、表、算法或叙述之间不一致。 |
| 图表可直接读出 | 主要依据来自图、表，而非正文完整文字。 |
| 工程建议，不可作为论文依据 | 为防泄漏、便于审计或稳定复现提出的工程处理，不能写成论文事实。 |
| NotebookLM 草稿待核实 | 只来自草稿/待核查清单，未被论文原文直接支撑。 |

---

## 1. 论文基本信息与复现边界

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| 论文题目为 *A multi-source multi-layer-based transfer learning approach for forecasting customer demands of newly launched products*。 | PDF p.1 标题。 | README、实验报告、结果目录命名应与论文题目一致。 | 论文明确 |
| 研究问题是：新上市产品/新开店在历史数据不足时，利用相似产品/商店的历史数据进行需求预测。 | Abstract；Section 1；Section 3 Problem definition，PDF pp.1,5。 | 检查代码是否围绕“limited target data + source transfer”设计，而不是普通全量监督学习。 | 论文明确 |
| 核心方法为 MSML-TL-RFE：Multi-Source Multi-Layer Transfer Learning + Recursive Feature Elimination。 | Abstract；Section 4；Fig. 2；Fig. 8；Algorithm 1，PDF pp.1,7,9,12。 | 检查是否包含多源预训练、权重/偏置融合、冻结层、目标域再训练、RFE。 | 论文明确 |
| 对比方法包括 No-TL、SS-TL、MSWA-TL、MSSB-TL、MSML-TL、MSML-TL-RFE。 | Section 4.1–4.3；Table 7/8 中列出 No-TL、SS-TL、MSWA-TL、MSSB-TL、MSML-TL-RFE；Section 4.3.3 单独描述 MSML-TL。 | 结果 CSV 至少能区分这些方法；若 MSML-TL 未单独输出，应标记为未完整复现。 | 论文明确 |
| Paper-aligned 复现边界一：三数据集预测实验。 | Section 5 Data collection；Section 6.1；Table 7、Table 8，PDF pp.10,15,16。 | 检查 Dataset 1/2/3 均有预测 RMSE 输出；方法、信息共享场景、target 构造一致。 | 论文明确 |
| Paper-aligned 复现边界二：no information sharing 与 with information sharing 两种场景。 | Section 3；Section 5.4.3；Section 5.4.4；Fig. 1；Table 5/6/7/8，PDF pp.5,14–16。 | 检查 `information_sharing=False/True` 是否只改变 source pool；输出 source_store_id/source_item_id/target_store_id/target_item_id。 | 论文明确 |
| Paper-aligned 复现边界三：Dataset 1 上的 supply chain cost evaluation。 | Section 3 cost model；Table 1；Section 5.5；Section 6.2；Table 11/12，PDF pp.6,15,17,18。 | 检查是否只把 Dataset 1 用于成本评估；是否包含 fixed LT 和 dynamic LT。 | 论文明确 |
| Sensitivity/behavioural analysis 是论文结果的一部分，但不是基础三源实验。 | Section 6.3；Table 13/14，PDF p.18。 | 单独标记 `experiment_scope=sensitivity`，不要混入基础 paper-aligned 默认结果。 | 论文明确 |
| DANN、稳定 RFE、多种 seed 搜索、额外防泄漏审计、超参数搜索属于工程扩展。 | 论文未出现 DANN/seed search/stability-RFE 等设定。 | 放入 `engineering_default` 或 `extension` 配置，不应宣称为论文设定。 | 工程建议，不可作为论文依据 |

---

## 2. 总体实验假设

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| 新产品被假设已上市约 1 个月。 | Section 3：all experiments based on assumption that a product has been recently launched for around one month；PDF p.5。 | 目标域 train+valid 应约为一个月；不要使用长目标历史训练。 | 论文明确 |
| 使用新产品约一个月销售模式识别相似 source，并用 target 的一个月数据 retrain，预测 6 个月数据。 | Section 3：retrained using one month of data ... to predict 6 months of data；PDF p.5。 | 检查 target train/valid/test 划分是否与 Table 2/3 一致。 | 论文明确 |
| Dataset 1 target 域为 15 天 train + 15 天 validation + 6 个月 test。 | Table 2，PDF p.12。 | 检查 Dataset 1 日期切片：2017-06-01 至 2017-06-15、2017-06-16 至 2017-06-30、2017-07-01 至 2017-12-31。 | 论文明确 |
| Source domain 是长期销售的相似产品/商店；Target domain 是假定新上市 Item X / Store 10。 | Section 3；Fig. 1；Section 5.4.2，PDF pp.5,13。 | 检查 DS/DT 是否按 item/store 分开，而不是随机行划分。 | 论文明确 |
| 层级供应链结构包含 1 个 supplier/distributor 和至少 3 个 retailers；三条 channel 为 retailer-supplier 两阶段结构。 | Section 3，PDF p.6。 | 成本模型中应有 3 个 channel；supplier-retailer 两级。 | 论文明确 |
| No information sharing：只从本 retailer/store 内长期销售的本地 items 中选 source。 | Section 3；Section 5.4.3；Fig. 11，PDF pp.5,14,15。 | `information_sharing=False` 时 source pool 不应跨 store/brand/region。 | 论文明确 |
| With information sharing：其他 retailers/stores 的 items/sales pattern 也进入 source pool。 | Section 3；Section 5.4.4；Table 6，PDF pp.5,14。 | `information_sharing=True` 时 source pool 应扩大，并能选到其他 store/brand/region。 | 论文明确 |
| 信息共享机制在论文中主要表现为 KNN source pool 扩大；没有明确说明会改变 CNN 架构或 cost model 参数。 | Section 3/5.4.4/6.2 未描述其他机制变化。 | 除 source pool 和由预测带来的成本结果外，不要擅自改库存参数。 | 论文未明确 |

---

## 3. 数据集信息

### 3.1 Dataset 1

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| Dataset 1 为 Kaggle Store Item Demand Challenge。 | Section 5.1，PDF p.10。 | 检查数据文件是否为 demand-forecasting-kernels-only/train.csv。 | 论文明确 |
| 时间范围为 2013–2017，共 50 items、10 stores。 | Section 5.1，PDF p.10。 | 检查 raw date range；不要只读取部分年份后再声明完整复现。 | 论文明确 |
| 原始属性：date of sales、Store ID、Item ID、number of items sold。 | Section 5.1，PDF p.10。 | 检查字段映射：date/store/item/sales。 | 论文明确 |
| 论文实际使用 first 3 stores，以及每个 store 对应的 first 10 sold items。 | Section 5.1，PDF p.10。 | 检查是否过滤为 Store 1–3 与 Item 1–10；成本评估三 channels 对应 Store 1/2/3。 | 论文明确 |
| 初始实验中 first 9 items 为 source products，10th item / Item X 为 target domain data。 | Section 5.1，PDF p.10。 | 检查 target_item_id=10；source items 不含 target item。 | 论文明确 |
| Channel 1 target 为 Store 1 Item 10；Channel 2 target 为 Store 2 Item 10；Channel 3 target 为 Store 3 Item 10。 | Section 5.5，PDF p.15。 | 成本模型三个 channel 的 target 应分别对应三家 store 的 Item 10。 | 论文明确 |
| No sharing 下 Channel 1 的三源为 Store 1 Item 7、Item 8、Item 2，距离为 101.58、103.50、115.28。 | Table 5，PDF p.14。 | 输出 selected_sources 和 distances，核对排序与表值。 | 图表可直接读出 |
| With sharing 下 Channel 1 的三源为 Store 3 Item 6、Store 2 Item 9、Store 3 Item 7，距离为 63.98、68.16、70.94。 | Table 6，PDF p.14。 | 检查 sharing 情况下能跨 store 选源。 | 图表可直接读出 |
| 同类别 sensitivity 中，Dataset 1 可用不同 stores 的 Item 10 作为同类 source。 | Section 6.3，Table 14 后正文，PDF pp.18–19。 | `same_category=True` 应单独作为 sensitivity scope，不应替代基础 Table 5/6 source pool。 | 论文明确 |

### 3.2 Dataset 2

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| Dataset 2 来自 Mancuso et al. (2021) 的 pasta demand 数据，Mendeley 链接。 | Section 5.1，PDF p.10。 | 检查数据来源与数据项命名，不要误用其他 pasta 数据。 | 论文明确 |
| 包含 118 条 daily time series，时间范围 01/01/2014–31/12/2018。 | Section 5.1，PDF p.10。 | 检查 date range 和 series 数量。 | 论文明确 |
| 数据除单变量时间序列外，还包含 promotional activities。 | Section 5.1；Section 5.2；Fig. 9，PDF pp.10,12,13。 | Dataset 2 特征应包含 promotion；不要把 promotion 的具体非线性影响写成论文事实。 | 论文明确 |
| Dataset 2 中 first two datasets 均假设每个 retailer/brand 的 Item 10 为新上市 target。 | Section 5.4.2，PDF p.13。 | 检查 target item 为 Brand 1 Item 10；成本扩展中 Channel 1/2/3 为 Brand 1/2/3 Item 10。 | 论文明确 |
| No sharing 下 Channel 1 的三源为 Brand 1 Item 4、Item 6、Item 8，距离 24.98、26.85、26.85。 | Table 5，PDF p.14。 | 核对 selected_sources、distances、weights。 | 图表可直接读出 |
| With sharing 下三源为 Brand 1 Item 4、Brand 2 Item 3、Brand 3 Item 2，距离 24.98、25.98、26.00。 | Table 6，PDF p.14。 | 检查 sharing 下允许跨 brand。 | 图表可直接读出 |
| “Promotion 对需求波动有显著非线性影响”不是论文明确结论。 | 论文只说 promotional activities are linked with sales；未给出非线性影响分析。 | 不要在 paper-aligned 索引中加入“显著非线性影响”。 | 论文未明确 |

### 3.3 Dataset 3

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| Dataset 3 为 Rossmann Store Sales data，Kaggle 平台。 | Section 5.1，PDF p.10。 | 检查 Rossmann 数据来源；确认 sales/store/date 字段。 | 论文明确 |
| 数据描述为 different Rossmann stores 的 daily sales。 | Section 5.1，PDF p.10。 | Dataset 3 应以 store-level sales 作为 entity，而非 item-level sales。 | 论文明确 |
| 正文称数据有 store type、holiday、Store Type 等 main features；Section 5.2 又称包含 promotional activities、holidays、open/closed、customer-related information。Fig. 9 显示 Year/Month/Week/Day/Customer/Open/Promotion/Holiday。 | Section 5.1；Section 5.2；Fig. 9，PDF pp.10,12,13。 | 字段应记录实际使用版本；store_type 是否使用需单独审计，因为 Fig. 9 未显示 store type。 | 论文存在冲突 |
| Dataset 3 取 first 10 stores 的 sales patterns 进行分析，Store 10 被假设为 target store。 | Section 5.4.2，PDF p.13。 | target_store_id=10；source stores 从其他 stores 选。 | 论文明确 |
| 论文假设 Store 1–10 位于 Region 1，Store 11–20 位于 Region 2，Store 21–30 位于 Region 3。 | Section 5.4.2，PDF p.14。 | 如果代码使用 region，应显式记录这是论文假设，不是 Rossmann 原始字段。 | 论文明确 |
| No sharing 下 Channel 1 的三源为 Store 6、Store 2、Store 1，距离 4901.97、5767.16、5895.77。 | Table 5，PDF p.14。 | 检查 no sharing 下只在 Region 1/first 10 stores 内选源。 | 图表可直接读出 |
| With sharing 下三源为 Store 23、Store 14、Store 6，距离 3723.62、3978.98、4901.97。 | Table 6，PDF p.14。 | 检查 sharing 下可跨 region 选 Store 23、14。 | 图表可直接读出 |
| Dataset 3 Category ‘a’ 只在 same category sensitivity 中出现，并非基础复现默认要求。 | Section 6.3/Table 14 后正文，PDF pp.18–19。 | `same_category_pass` 应作为 sensitivity/extension 字段，不应替代基础 Table 5/6。 | 论文明确 |

---

## 4. Source domain / Target domain 构造

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| Fig. 1 表示每个 store 有多个 source items，并有一个新产品 Item X 作为 target。 | Fig. 1；Section 3，PDF pp.5–6。 | 数据结构应保留 store/channel/item 层级。 | 图表可直接读出 |
| 每个 retailer/channel 可有一个 target product。 | Section 5.5：Dataset 1 Channel 1/2/3 分别为 Store 1/2/3 Item 10；PDF p.15。 | 成本评估时三个 channel 不应共用一个 target 序列。 | 论文明确 |
| No sharing 的 source pool 是 target 所在 retailer/store/brand 内的本地长期销售 items/stores。 | Section 3；Section 5.4.3；Fig. 11。 | 检查 source_pool 构造是否严格同 store/brand/region。 | 论文明确 |
| With sharing 的 source pool 扩展到其他 retailers/stores/brands/regions。 | Section 3；Section 5.4.4；Table 6。 | 检查 source_pool 中是否允许跨 store/brand/region。 | 论文明确 |
| KNN 在 no sharing 下是同 store/brand/region 内选源；with sharing 下是跨 store/brand/region 选源。 | Section 5.4.3/5.4.4；Table 5/6。 | 输出 `information_sharing`、`source_store_id`、`target_store_id`、`same_group`。 | 论文明确 |
| 论文没有说明三数据集的底层实体结构完全一致；Dataset 1/2 是 item/brand，Dataset 3 是 store-level sales。 | Section 5.4.2。 | 代码不应强行用同一 entity schema 覆盖 Dataset 3。 | 论文明确 |
| Dataset 3 的 source/target 划分依赖“first 10 stores/regions”的论文假设；论文未提供原始区域字段或映射依据。 | Section 5.4.2。 | 代码中 region 构造应显式标记为 paper assumption。 | 论文未明确 |

---

## 5. Train / Validation / Test 划分

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| Dataset 1 source train：1/1/2013–31/12/2016，3 years。 | Table 2，PDF p.12。 | 检查 source train 日期闭区间。 | 图表可直接读出 |
| Dataset 1 source validation：1/1/2017–30/6/2017，6 months。 | Table 2，PDF p.12。 | 检查 source validation 日期。 | 图表可直接读出 |
| Dataset 1 source test：1/7/2017–31/12/2017，6 months。 | Table 2，PDF p.12。 | 检查 source test 日期；注意 source test 是否实际用于训练流程需代码标记。 | 图表可直接读出 |
| Dataset 1 target train：1/6/2017–15/6/2017，15 days。 | Table 2，PDF p.12。 | 检查 target train 只包含 15 天。 | 图表可直接读出 |
| Dataset 1 target validation：16/6/2017–30/6/2017，15 days。 | Table 2，PDF p.12。 | 检查 target validation 只包含 15 天。 | 图表可直接读出 |
| Dataset 1 target test：1/7/2017–31/12/2017，6 months。 | Table 2，PDF p.12。 | 检查 target test 不进入训练/选源/RFE fitting。 | 图表可直接读出 |
| Table 3 Dataset 1 source time steps：train 1461/80.01%，valid 181/9.91%，test 184/10.08%。Target：train 15/6.98%，valid 15/6.98%，test 185/86.05%。 | Table 3，PDF p.12。 | 结果日志记录 row counts/time steps。 | 图表可直接读出 |
| Table 3 Dataset 2 source time steps：train 1443/80.26%，valid 176/9.79%，test 179/9.96%。Target：train 14/6.73%，valid 15/7.21%，test 179/86.06%。 | Table 3，PDF p.12。 | Dataset 2 只给 time steps，没有给具体日期；代码需记录实际 date split。 | 图表可直接读出 |
| Table 3 Dataset 3 source time steps：train 577/61.25%，valid 184/19.53%，test 181/19.21%。Target：train 16/7.55%，valid 15/7.08%，test 181/85.38%。 | Table 3，PDF p.12。 | 注意 Dataset 3 source 不是 80/10/10，应按表值或明确标记偏离。 | 图表可直接读出 |
| Dataset 2 和 Dataset 3 未提供具体日期切分，只提供 time steps 和比例。 | Table 3；Section 5.1/5.2 未给 split date。 | 复现代码需输出实际切分日期，但不能声称论文给出。 | 论文未明确 |
| “目标域前 30 天 observed window”可作为 Dataset 1 的 train+valid 概括；Dataset 2 为 14+15=29，Dataset 3 为 16+15=31，论文也泛称 30 days/one month。 | Table 2/3；Section 5.4.1。 | 配置名可用 `target_observed_window`，但日志必须保留各数据集真实 time steps。 | 论文明确 |
| KNN 使用 target domain 30 天 sales data，并抽取 source 在同期间对应 sales data；KNN 计算的是 30 天数据之间的距离。 | Section 5.4.1，PDF p.13。 | 检查 `source_selection_window` 是否使用 target observed window，不能使用 target test。 | 论文明确 |
| “严禁触碰测试集”是复现防泄漏要求，不是论文原文用语；但与 Section 5.4.1 的 30 天 KNN 窗口一致。 | 论文没有“严禁”字样。 | 审计中标记为防泄漏工程约束。 | 工程建议，不可作为论文依据 |

---

## 6. 特征工程与监督学习结构

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| year、month、week、day 由 date 通过 Python `to_datetime` 提取。 | Section 5.2；Fig. 9，PDF pp.12–13。 | 检查日期特征提取方式。 | 论文明确 |
| Dataset 1 至少 4 个特征：year、month、week、day；正文同时说 Dataset 1 没有 stores sales 之外信息。 | Section 5.2；Fig. 9。 | 不要给 Dataset 1 加 promotion/holiday 等论文外特征。 | 论文明确 |
| Dataset 2 在 date features 外加入 Promotion。 | Section 5.1/5.2；Fig. 9。 | 检查 promotion 是否进入特征矩阵。 | 论文明确 |
| Dataset 3 在 Fig. 9 中为 Year、Month、Week、Day、Customer、Open、Promotion、Holiday；正文另称 store-related information。 | Fig. 9；Section 5.2。 | 检查实际字段；若加入 store_type，应标记为论文正文支持但 Fig. 9 未显示。 | 论文存在冲突 |
| target variable 是某个 store item / store 的 sales。 | Section 5.2：sales of a particular store item being target variable；PDF p.12。 | 检查 y 是否为 sales。 | 论文明确 |
| independent variables X 包括 date-time features 与其他可用特征；正文还说 X train 包括 10 days of features and past sales information。 | Section 5.3；Fig. 10，PDF pp.12–13。 | 如果 sales 作为 X，必须是过去窗口内的 lagged sales，不得包含当前/未来 y。 | 论文明确 |
| Table 4 定义 days-ahead supervised learning：X 在 t−n−h 到 t−h，y 在 t−n+h 到 t+h；h 为 forecasting horizon。 | Table 4，PDF p.12。 | 检查 X/y shift 是否按 horizon 构造。 | 图表可直接读出 |
| 使用 Min–Max normalization，将特征缩放到 0–1，公式见 Eq. (5)。 | Section 5.3；Eq. (5)，PDF p.12。 | 检查 scaler fit 范围，建议避免 target test leakage。 | 论文明确 |
| CNN 输入使用 sliding window，转换为适合 CNN 的 3D arrays。 | Section 5.3，PDF p.12。 | 检查 X_train/X_val/X_test shape。 | 论文明确 |
| Window size 10 用于 1-day ahead 和 5-day ahead sales prediction。 | Section 5.3；Fig. 10，PDF pp.12–13。 | `window_size=10` 可作为 paper-aligned for reported 1–5 day experiments。 | 论文明确 |
| 论文没有给出所有 horizon 的完整 n/lookback 公式细节，也未说明是否对 15/30 horizon 实际实验。 | Section 5.3 同时提到 1、15、30 作为参数示例，但后文说限制为 5 days ahead。 | 不要把 15/30 day horizon 写成已报告实验矩阵。 | 论文存在冲突 |

---

## 7. KNN 选源机制

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| KNN 的作用是识别与 target item/store 相似的 source item/store。 | Section 3；Section 5.4.1。 | 检查 KNN 在 TL 前执行，而不是训练后筛选。 | 论文明确 |
| 距离度量为 Euclidean distance。 | Section 3；Section 5.4.1；Algorithm 1 lines 2–4。 | 输出 distance matrix；核对 Table 5/6。 | 论文明确 |
| KNN 使用 all available features of each dataset。 | Section 5.4.1；Table 5 前后正文。 | 检查 KNN 特征空间是否包含论文列；缺失字段需报告。 | 论文明确 |
| KNN 输入时间窗口为 target domain 的 30 days sales data 与 source 同期数据。 | Section 5.4.1。 | 检查 `source_selection_window=target_observed_window`。 | 论文明确 |
| 基础 multi-source 实验选择 top 3 closest items/stores。 | Section 3；Section 5.4.1；Table 5/6；Algorithm 1。 | `k=3` 应为基础默认。 | 论文明确 |
| SS-TL 使用最接近的单个 source；multi-source approaches 使用 3 个 source。 | Section 5.4.3；Fig. 11。 | 检查 SS-TL 的 selected source 是否为排序第一；MSWA/MSSB/MSML 是否使用前三源。 | 论文明确 |
| Sensitivity analysis 使用 6 sources 和 9 sources，with information-sharing concept enabled。 | Section 6.3；Table 13。 | 单独标记 `num_sources=6/9` 与 `experiment_scope=sensitivity`。 | 论文明确 |
| 论文贡献段落有 “pre-training up to five different TL models” 的描述，但结果表使用 3、6、9 sources，没有 5-source 结果表。 | Section 1.1；Table 13。 | 不要把 `max_sources=5` 写成结果矩阵默认；如实现 5 源，应标为工程扩展/待核实。 | 论文存在冲突 |
| KNN 是“跨源知识迁移的门控”这种表述不是论文术语。 | 论文未使用该术语。 | 可在注释中作为理解，不可作为 paper-aligned 字段。 | NotebookLM 草稿待核实 |

### 7.1 KNN 代码审计检查点

| 检查项 | 代码中应查什么 | 期望 paper-aligned 行为 | 状态 |
| ---- | ---- | ---- | ---- |
| source_selection_window | KNN fitting/距离计算是否使用 target train+valid/30 days | 使用 target observed window；不使用 target test | 论文明确 |
| selected_sources | source IDs、store/brand/item、rank | 与 Table 5/6 能对齐，至少 Channel 1 应可复现表中源 | 图表可直接读出 |
| distance matrix | 欧氏距离、特征缩放前后、字段顺序 | 使用 Euclidean distance；记录 feature columns | 论文明确 |
| information_sharing=False | source pool | 只在同 retailer/store/brand/region 内选源 | 论文明确 |
| information_sharing=True | source pool | 可跨 retailer/store/brand/region 选源 | 论文明确 |
| leakage check | target test、source future、scaler fit | 防止 KNN/RFE/scaler 使用 target test | 工程建议，不可作为论文依据 |

---

## 8. Source 权重计算

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| 正文称 Euclidean distances from KNN 用 inverse weighted distance 计算每个 similar item 的 weights。 | Section 3，PDF p.5。 | 检查权重是否为 inverse distance normalized。 | 论文明确 |
| MSWA-TL 中，每个 source model 的预测乘以对应权重后相加得到 target testing set final prediction。 | Section 4.3.1，PDF p.8。 | 检查 weighted prediction aggregation。 | 论文明确 |
| MSML-TL/MSML-TL-RFE 中，各 source CNN 的 weights/biases 用 weighted average 融合形成新 CNN 冻结层。 | Section 4.3.3/4.3.4，PDF p.9。 | 检查 layer-wise weights/biases fusion。 | 论文明确 |
| Table 5/6 标题和行名明确给出 inverse distance weights。 | Table 5/6，PDF p.14。 | 用表中距离计算 inverse-distance normalized weights，核对数值。 | 图表可直接读出 |
| Algorithm 1 lines 13–16 使用 `distance_i / sum_distance` 作为 `weights_avg`，这会给更远 source 更高权重。 | Algorithm 1，PDF p.12。 | 代码需明确采用 text-aligned 或 pseudo-code-aligned。 | 论文存在冲突 |
| Table 5 正文解释中 Item 7 权重写为 0.3793，但表格为 0.3493；按 inverse distance 计算也约为 0.349。 | Table 5 与其后正文，PDF p.14。 | 结果审计优先用表格数值并标记正文笔误。 | 论文存在冲突 |

### 8.1 权重公式冲突审计

| 项目 | 内容 | 代码审计检查点 | 状态 |
| ---- | ---- | ---- | ---- |
| 正文描述 | Section 3、Section 4.3.1、Table 5/6 均使用 inverse weighted distance / inverse distance weights；最近 source 权重应最大。 | `weight_i = (1/d_i) / sum_j(1/d_j)`。 | 论文明确 |
| Algorithm 1 伪代码 | lines 13–16：`sum_distance = d1+d2+d3`，`weights_avg=[d1/sum, d2/sum, d3/sum]`。 | `weight_i = d_i / sum_j(d_j)`。 | 图表可直接读出 |
| 冲突点 | 正文/表格与伪代码相反：正文使距离越小权重越大，伪代码使距离越大权重越大。 | 不能自行消解为单一论文事实。 | 论文存在冲突 |
| 对复现结果影响 | 影响 MSWA-TL 的 prediction aggregation，也影响 MSML/MSML-TL-RFE 的 weights/biases fusion。 | 结果 CSV 记录 `weight_formula=text_inverse_distance` 或 `pseudo_distance_sum`。 | 论文存在冲突 |
| 可选策略 1 | paper-text aligned：使用 inverse distance。 | 标记为 `paper_text_aligned`。 | 工程建议，不可作为论文依据 |
| 可选策略 2 | pseudo-code aligned：使用 distance/sum_distance。 | 标记为 `algorithm1_aligned`。 | 工程建议，不可作为论文依据 |
| 可选策略 3 | 双版本消融：两种都跑，报告差异。 | 标记为 `ambiguity_ablation`。 | 工程建议，不可作为论文依据 |

---

## 9. 方法索引与代码审计点

### 9.1 No-TL

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| No-TL 是只在 target domain 部署 simple CNN，不使用其他 source knowledge。 | Section 4.1，PDF p.7。 | 检查训练数据只来自 target train/valid。 | 论文明确 |
| CNN 使用新产品/target product 一个月数据训练和验证。 | Section 4.1；Table 2/3。 | 检查 target train+valid 数据量。 | 论文明确 |
| 测试集为 target domain test data，即未来约 6 个月。 | Table 2/3；Section 5.4.2。 | 检查 test 不进入训练。 | 论文明确 |
| No-TL 作为 benchmark，Table 7/8/13/11/12 均有结果。 | Tables 7,8,11,12,13。 | 结果输出 No-TL RMSE、cost、time/accuracy（Dataset 1 sensitivity）。 | 论文明确 |

### 9.2 SS-TL

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| SS-TL 首先用 KNN 识别一个 similar source product/store。 | Section 4.2；Section 5.4.3。 | selected source 为距离最小的 source。 | 论文明确 |
| 使用 source sales data 训练 CNN 获得 pre-trained model。 | Section 4.2。 | 检查 source train/val split 与 Table 2/3。 | 论文明确 |
| 冻结 pre-trained network 的 first 4 layers 的 weights/biases，并与 trainable lower-level layers 组成新 NN。 | Section 4.2；Fig. 4。 | 检查 frozen layers；建议按层名而不是 index 审计。 | 论文明确 |
| 新 NN 用 target data retrain/validate，再在 target test 上预测。 | Section 4.2。 | 检查 target train/valid/test 数据流。 | 论文明确 |
| 每个 day-ahead prediction 重复实验并计算 mean predictions。 | Section 4.2。 | 输出 horizon-level repetitions/mean aggregation。 | 论文明确 |
| 论文未说明重复次数、seed、epochs。 | Section 4.2 未给。 | 记录 engineering defaults。 | 论文未明确 |

### 9.3 MSWA-TL

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| MSWA-TL 使用三个不同 source 分别运行 SS-TL-like 模型。 | Section 4.3.1；Fig. 5。 | 检查每个 source 对应独立模型。 | 论文明确 |
| 每个 source model 在 target domain data 上 retrain。 | Section 4.3.1。 | 检查每个模型均 fine-tune target。 | 论文明确 |
| 各模型 target test predictions 按 source weight 加权求和。 | Section 4.3.1。 | 检查 prediction-level weighted aggregation。 | 论文明确 |
| 权重来自 KNN distance 的 inverse distance weights；但 Algorithm 1 对 MSML-RFE 伪代码存在冲突。 | Section 4.3.1；Table 5/6；Algorithm 1。 | 记录 weight formula。 | 论文存在冲突 |
| multiple experiments carried out and average weighted predictions used to remove randomness。 | Section 4.3.1。 | 记录 `n_repeats`；论文未给具体次数。 | 论文明确 |

### 9.4 MSSB-TL

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| MSSB-TL 对多个 source 分别训练 pre-trained networks，再生成各自 target CNN。 | Section 4.3.2；Fig. 6。 | 检查多个 source model 均存在。 | 论文明确 |
| switching 基于 target domain validation RMSE 的 minimum validation RMSE。 | Section 4.3.2，PDF p.9。 | 检查是否用 validation RMSE 选择模型；不能用 test RMSE。 | 论文明确 |
| 重复实验并取 average predictions；不同 forecasting horizons 重复。 | Section 4.3.2。 | 输出选择的 source/model、validation RMSE、horizon。 | 论文明确 |
| 论文未说明 tie-breaking、重复次数、训练细节。 | Section 4.3.2 未给。 | 记录 engineering defaults。 | 论文未明确 |

### 9.5 MSML-TL

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| 多个 source 分别训练同一 CNN，得到多个 pre-trained models。 | Section 4.3.3；Fig. 7。 | 检查每个 source 独立预训练。 | 论文明确 |
| 将每个 source CNN 对应层的 weights/biases 做 weighted average。 | Section 4.3.3。 | 检查 layer-wise fusion，shape 对齐。 | 论文明确 |
| 融合后的 weights/biases 形成新 CNN 的 frozen layers，其他 lower-level layers trainable。 | Section 4.3.3。 | 检查 freeze 状态与 target fine-tuning。 | 论文明确 |
| 初始 multi-source 关注 3 sources。 | Section 4.3.3。 | `num_sources=3` 为基础默认。 | 论文明确 |
| 每个 day-ahead prediction 多次实验并取平均。 | Section 4.3.3。 | 输出 repeat aggregation。 | 论文明确 |

### 9.6 MSML-TL-RFE

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| RFE 在喂入 MSML-TL 前执行。 | Section 4.3.4；Fig. 2；Fig. 8。 | 检查 RFE 是否在 source CNN 训练前执行。 | 论文明确 |
| Section 4.3.4 说重要特征 mainly selected from similar source data；Algorithm 1 line 5 对 Target 和 S1/S2/S3 都做 RFE。 | Section 4.3.4；Algorithm 1。 | 代码需明确 RFE fit/transform 对象；建议做泄漏审计。 | 论文存在冲突 |
| RFE 减少 target item and similar products 的 features。 | Algorithm 1 lines 5–6；Algorithm 1 解释。 | 检查 RFE 后 target/source 特征维度一致。 | 图表可直接读出 |
| 使用 RFE 提取约 40%–60% features。 | Section 4.3.4。 | 检查保留比例是否在 40%–60%，但具体 n_features 未明确。 | 论文明确 |
| 论文未说明 RFE estimator、ranking criterion、random_state、n_features_to_select、最终选中特征。 | Section 4.3.4/Algorithm 1 未给。 | 这些必须作为 engineering default 显式记录。 | 论文未明确 |
| RFE 可能影响 source 和 target 的输入维度；论文没有给出对不同数据集的最终 feature list。 | Section 4.3.4；Algorithm 1。 | 输出 `rfe_candidate_features`、`rfe_selected_features`、`final_input_features`。 | 论文未明确 |

---

## 10. CNN 网络结构与冻结层

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| CNN 为 one-dimensional CNN。 | Section 4.1；Fig. 3，PDF p.7。 | 检查 Conv1D 而非 Conv2D/LSTM/MLP。 | 论文明确 |
| 网络结构为 Input → Conv1D → MaxPooling → Conv1D → MaxPooling → Conv1D → Flatten → Dense → Output。 | Fig. 3；Section 4.1。 | 检查模型摘要层序列。 | 图表可直接读出 |
| 正文说明 three 1D convolutional layers with MaxPooling layers in between each；last two layers are Flatten and Dense。 | Section 4.1。 | 检查有 3 个 Conv1D、2 个 MaxPooling、Flatten、Dense。 | 论文明确 |
| Fig. 4 显示 frozen/transferred 的前四个计算层为 Conv1D、MaxPooling、Conv1D、MaxPooling。 | Fig. 4，PDF p.7。 | 不要把 Input 层计入冻结；但代码要记录具体 frozen layer names。 | 图表可直接读出 |
| 正文称 weights and biases from pre-trained network of first 4 layers are frozen。 | Section 4.2。 | 检查 `frozen_layers_count=4` 与 layer names。 | 论文明确 |
| filters、kernel size、activation、optimizer、loss、learning rate、epochs、batch size 未在论文中说明。 | Section 4/5 未提供。 | 全部归入 engineering defaults；不能写成论文事实。 | 论文未明确 |
| Adam、LR=0.001、Batch=32、Kernel=3、Filters=32/64 没有论文依据。 | 论文未出现这些具体超参数。 | 删除或标记为 engineering default。 | NotebookLM 草稿待核实 |
| 论文说明 MinMax scaling 是为了能处理可能的 sigmoid output layer，但没有明确 CNN output activation。 | Section 5.3。 | 不要把 sigmoid 输出层写成论文事实。 | 论文未明确 |

---

## 11. Horizon / days-ahead prediction

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| `days_ahead_prediction` 是 forecasting horizon。 | Table 4，PDF p.12。 | 代码应显式输出 horizon。 | 论文明确 |
| Table 4 用 X 的历史窗口预测 y 在 future horizon 的值。 | Table 4。 | 检查 X/y shift 方向。 | 图表可直接读出 |
| Section 5.3 说参数可设为 1、15、30 分别表示 short-term、mid-term、long-term。 | Section 5.3，PDF p.12。 | 不能直接把这些当作已报告结果表的 horizon 列表。 | 论文明确 |
| 同一段随后说 window size 10 用于 1-day ahead 和 5-day ahead；Fig. 10 说明 short-term/mid-term。 | Section 5.3；Fig. 10。 | 基础复现应至少实现 1–5 days。 | 论文明确 |
| Section 5.3 明确说由于 target data limited，本文 only 5 days ahead prediction is made。 | Section 5.3，PDF p.13。 | 不要声称论文实际跑到 15/30 天。 | 论文明确 |
| Section 6.1 说 To evaluate performance, 1 to 5 days ahead predictions were obtained from each individual model。 | Section 6.1，PDF p.15。 | 结果 RMSE 应明确是 1–5 horizons 的平均或聚合。 | 论文明确 |
| 结果表没有逐 horizon 展示 1、2、3、4、5 天 RMSE，只展示平均 RMSE/accuracy/cost。 | Table 7/8/11/12/13。 | 如代码输出 per-horizon，是补充审计输出，不是论文表格。 | 论文未明确 |

---

## 12. 实验重复与随机性

| 参数 | 论文是否明确 | 论文来源 | 代码复现建议如何标记 |
| -- | ------ | ---- | ---------- |
| multiple experiments | 明确提到，但未给次数 | Section 4.2、4.3.1、4.3.2、4.3.3；Section 6.1 | `n_repeats=engineering_default`，结果记录每次与均值 |
| average predictions | 明确 | Section 4.2/4.3.1/4.3.2/4.3.3；Section 6.1 | 输出 mean prediction / mean RMSE |
| random seed | 未明确 | 未给 | `random_seed=engineering_default` |
| repeated experiment count | 未明确 | 未给 | 记录 n_runs，不要写成论文设定 |
| train/validation/test shuffle | 未明确 | 未给 | 时间序列建议不 shuffle，但标为 engineering default |
| early stopping | 未明确 | 未给 | 若使用需标记 extension/default |
| normalization/scaling | 明确，Min–Max scaling Eq. (5) | Section 5.3 | `scaler=minmax`，记录 fit scope |
| batch size | 未明确 | 未给 | engineering default |
| epochs | 未明确 | 未给 | engineering default |
| learning rate | 未明确 | 未给 | engineering default |
| optimizer | 未明确 | 未给 | engineering default |
| loss function | 未明确 | 未给；只说 RMSE evaluation | training loss 为 engineering default；evaluation RMSE 为 paper-aligned |

---

## 13. Evaluation metrics

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| RMSE 用于 prediction performance evaluation。 | Algorithm 1 output/test_RMSE；Section 6.1；Table 7/8。 | 输出 test RMSE；明确 metric space。 | 论文明确 |
| Accuracy 定义为 reciprocal of RMSE。 | Abstract；Table 13 header：Accuracy (Reciprocal RMSE)。 | 若输出 accuracy，使用 `1 / RMSE` 并记录来源。 | 论文明确 |
| MSSB-TL 使用 target validation RMSE 做 switching。 | Section 4.3.2。 | 记录 validation RMSE 与选中模型。 | 论文明确 |
| 最终预测与 target domain test true values 比较计算 RMSE。 | Algorithm 1 lines 20–21；Section 6.1。 | 检查 test RMSE 不用于模型选择。 | 论文明确 |
| cost evaluation 使用预测结果进入 hierarchical SC model。 | Section 5.5；Section 6.2；Table 11/12。 | 输出 cost by channel and method。 | 论文明确 |
| runtime/computational time 只在 Table 13 报告 Dataset 1 的部分方法/源数量。 | Table 13，PDF p.18。 | 不要声称所有方法/数据集都有 runtime。 | 论文明确 |
| Table 8 中 With info Mean RMSE 表格为 0.1937，但正文写成 0.1973。 | Table 8 与其后正文，PDF p.16。 | 代码审计中标记 table/text conflict；优先报告原表与原文差异。 | 论文存在冲突 |
| Dataset 1 accuracy 数值 4.8340、5.1749、5.6668、5.5102、5.7538 来自 Table 13。 | Table 13，PDF p.18。 | 仅用于 Dataset 1 sensitivity/time comparison。 | 图表可直接读出 |

---

## 14. Supply chain cost model

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| cost evaluation 只考虑 Dataset 1。 | Section 3；Section 5.5；Section 6.2。 | 如果 Dataset 2/3 做 cost，应标为 future/extension。 | 论文明确 |
| SC model 为 hierarchical two-echelon supply chain，每个 channel 包含 retailer 和 supplier。 | Section 3，PDF p.6。 | 检查 channel 层级结构。 | 论文明确 |
| 无 backorder cost；retailer 无法满足需求时产生 lost sales cost；supplier 有库存并在 order 后交付。 | Section 3，PDF p.6。 | 成本逻辑中不应加 backorder cost。 | 论文明确 |
| Eq. (1) safety stock 同时考虑 demand variability 与 lead time variability。 | Eq. (1)，PDF p.6。 | 检查 safety stock 公式。 | 论文明确 |
| Eq. (2) fixed lead time 下 σL=0，safety stock 简化为 `Z σd sqrt(L)`。 | Eq. (2)，PDF p.6。 | fixed LT 时使用简化公式。 | 论文明确 |
| Eq. (3) reorder amount：RA = LTPD + ss。 | Eq. (3)，PDF p.6。 | 检查 reorder trigger。 | 论文明确 |
| 当 RA 大于 current inventory level + orders in delivery 时向 supplier 下单。 | Section 3，PDF p.6。 | 检查 order placement condition。 | 论文明确 |
| Eq. (4) total cost 为 purchasing cost、ordering cost、holding cost、lost sales cost 之和。 | Eq. (4)，PDF p.6。 | 输出 PC/OC/HC/LS/TC 分项或至少 TC。 | 论文明确 |
| Table 1 参数：Ordering $20/order，Purchasing $2/unit，Holding $7/unit/year，Lost sales $5/unit，Beginning inventory 250，Lot size Q=200。 | Table 1，PDF p.6。 | 参数应可配置并记录。 | 图表可直接读出 |
| fixed lead time = 5 days。 | Section 6.2.1；Table 11。 | 检查 fixed LT scenario。 | 论文明确 |
| dynamic lead time varies between 3 to 5 days；Table 12 标为 3+2 days；正文说 3 days LT + additional 1 to 2 days delay。 | Section 6.2.2；Table 12。 | 记录 dynamic LT 生成方式；论文未给随机分布。 | 论文明确 |
| dynamic lead time 的具体随机生成方式/概率分布未明确。 | Section 6.2.2 未给。 | 必须标记 engineering default，例如 uniform/random seed。 | 论文未明确 |
| holding cost increases/decreases linearly with inventory；lost sales fixed per unit and increases linearly as more goods are out of stock。 | Section 3 成本假设。 | 记录 daily/annual holding conversion；论文未给精确离散计算。 | 论文明确 |
| cost model 使用 TL forecasting algorithms 提供的 lead time predicted demand；预测来自 No-TL、SS-TL、MSML-TL-RFE。 | Section 3；Section 6.2。 | 检查 forecast-to-cost pipeline。 | 论文明确 |
| cost evaluation 是否用 true demand 扣减库存、如何计算每日 holding/lost sales 的精确顺序未完全展开。 | Section 3/6.2 未给伪代码。 | 成本复现需输出工程实现细节，不能宣称完全由论文规定。 | 论文未明确 |

---

## 15. Information sharing 实验

| 复现细节 | 论文依据/来源位置 | 代码审计检查点 | 状态 |
| ---- | --------- | ------- | -- |
| no information sharing 表示 retailer 不公开销售信息，source 只来自自身产品/本地区 stores。 | Section 3；Section 5.4.3。 | `information_sharing=False` 限制 source pool。 | 论文明确 |
| with information sharing 表示 competitors/stakeholders 共享数据，形成更大 source pool。 | Section 3；Section 5.4.4。 | `information_sharing=True` 扩展 source pool。 | 论文明确 |
| Fig. 1 用横向箭头表示 Store 1/2/3 之间 horizontal information sharing。 | Fig. 1，PDF p.6。 | 数据结构应保留跨 store source。 | 图表可直接读出 |
| Table 5 报告 without information sharing 的 closest three sources；Table 6 报告 with information sharing 的 closest three sources。 | Table 5/6，PDF p.14。 | 选源输出与表对照。 | 图表可直接读出 |
| Table 7/8 报告 no sharing 与 sharing 下各方法 RMSE；Fig. 12/13 可视化。 | Table 7/8；Fig. 12/13。 | 结果 CSV 需要 scenario 字段。 | 论文明确 |
| Table 11/12 cost 使用 Dataset 1 with information sharing concept。 | Table 11/12 标题，PDF pp.17–18。 | 成本评估应标记 `information_sharing=True`。 | 论文明确 |
| 论文没有说明 information sharing 改变 cost parameters；成本降低通过 forecast/prediction 进入模型体现。 | Section 6.2 未描述参数变化。 | 不要因 sharing 改 Table 1 参数。 | 论文未明确 |
| same-category/source type 限制只在 Section 6.3 Table 14 sensitivity 中出现。 | Section 6.3；Table 14。 | `same_category_pass` 不应成为基础 Table 7 默认。 | 论文明确 |

---

## 16. 论文结果表与实验矩阵

| Table/Figure | Dataset | Scenario | Method | Source 数量 | Horizon | Metric | 是否 cost | 代码应输出什么 |
| ------------ | ------- | -------- | ------ | --------- | ------- | ------ | ------- | ------- |
| Fig. 1 | All conceptual | no sharing / sharing | N/A | N/A | N/A | N/A | 否 | source/target/channel 构造图对应字段 |
| Table 1 | Dataset 1 | cost model | N/A | N/A | N/A | cost params | 是 | SC 参数配置 |
| Fig. 2 | All | proposed workflow | MSML-TL-RFE | 3 shown | N/A | workflow | 否 | RFE + multi-source MSML pipeline |
| Fig. 3 | All | No-TL CNN | No-TL | 0 | N/A | architecture | 否 | CNN model summary |
| Fig. 4 | All | SS-TL | SS-TL | 1 | per horizon | architecture | 否 | frozen layers + target fine-tune |
| Fig. 5 | All | MSWA-TL | MSWA-TL | 3 | per horizon | architecture | 否 | 3 SS models + weighted predictions |
| Fig. 6 | All | MSSB-TL | MSSB-TL | 3 | per horizon | validation RMSE switching | 否 | per-source validation RMSE + selected model |
| Fig. 7 | All | MSML-TL | MSML-TL | 3 | per horizon | architecture | 否 | layer-wise fused weights/biases |
| Fig. 8 | All | MSML-TL-RFE | MSML-TL-RFE | 3 | per horizon | workflow | 否 | RFE features + MSML pipeline |
| Table 2 | Dataset 1 | split | All | N/A | N/A | date ranges | 否 | source/target train/val/test dates |
| Table 3 | Dataset 1/2/3 | split | All | N/A | N/A | time steps/percentages | 否 | split counts and ratios |
| Table 4 | All | supervised structure | All | N/A | days_ahead_prediction | X/y formulation | 否 | horizon shift logic |
| Table 5 | Dataset 1/2/3 | without info sharing | TL methods | 3 | KNN window | distance + inverse weights | 否 | selected_sources/distances/weights |
| Table 6 | Dataset 1/2/3 | with info sharing | TL methods | 3 | KNN window | distance + inverse weights | 否 | selected_sources/distances/weights |
| Fig. 11 | Dataset 1 illustration | without info sharing | SS/MSSB/MSWA/MSML-RFE | 1 or 3 | test last 6 months | prediction evaluation | 否 | workflow by method |
| Table 7 | Dataset 1/2/3 | without/with info sharing | No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL-RFE | 0/1/3 | 1–5 days aggregated | Average RMSE | 否 | RMSE by dataset/method/scenario |
| Fig. 12 | Dataset 1/2/3 | without/with info sharing | same as Table 7 | 0/1/3 | aggregated | Average RMSE plot | 否 | optional plot |
| Table 8 | Combined mean | without/with info sharing | No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL-RFE | 0/1/3 | aggregated | Average RMSE | 否 | mean RMSE; note 0.1937/0.1973 conflict |
| Fig. 13 | Combined mean | without/with info sharing | same as Table 8 | 0/1/3 | aggregated | Average RMSE plot | 否 | optional plot |
| Table 9 | Combined | both scenarios | No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL-RFE | 0/1/3 | aggregated | Friedman mean rank | 否 | statistical rank output |
| Table 10 | Combined | both scenarios | selected pairs | N/A | aggregated | Wilcoxon p-value/decision | 否 | statistical test output |
| Table 11/Fig. 14 | Dataset 1 | with info sharing, fixed LT | No-TL, SS-TL, MSML-TL-RFE | 0/1/3 | up to 5 days | SC cost by channel | 是 | fixed LT cost table |
| Table 12 | Dataset 1 | with info sharing, dynamic LT | No-TL, SS-TL, MSML-TL-RFE | 0/1/3 | up to 5 days | SC cost by channel | 是 | dynamic LT cost table |
| Table 13 | Dataset 1 | sensitivity | No-TL, SS-TL w/o sharing, MSML-TL-RFE with sharing | 0/1/3/6/9 | aggregated | Time(s), Accuracy=1/RMSE | 否 | training time + accuracy; note 3915/3519 conflict |
| Table 14 | Dataset 1/2/3 | same-group sensitivity | MSML-TL-RFE | no sharing / 3 same-group / 9 same-group | aggregated | RMSE | 否 | same-category RMSE; Dataset 2 9-source missing |
| Fig. 15/16 | Dataset 1 channel 1 | fixed/dynamic LT | No-TL, SS-TL, MSML-TL-RFE/sensitivity | varies | aggregated | cost vs time vs accuracy | 是 | optional trade-off plot |

---

## 17. NotebookLM 草稿纠错表

| NotebookLM 草稿说法 | 是否有论文依据 | 正确处理 | 备注 |
| --------------- | ------- | ---- | -- |
| Dataset 3 重点聚焦 Category “a” | 部分有 | 可保留为论文事实，但仅限 Section 6.3 same-category sensitivity。 | 基础 Table 5/6 不是以 Category a 作为默认筛选。 |
| Dataset 1/2 target 是第 10 个项目 | 有 | 可保留为论文事实。 | Dataset 1 Store Item 10；Dataset 2 Brand Item 10。 |
| Dataset 3 target 是第 10 个商店 | 有 | 可保留为论文事实。 | 基础 Channel 1 Store 10；未来/扩展 channel 可为 Store 10/20/30。 |
| Window Size 固定为 10 | 有但有限定 | 可保留为论文事实，但限定为文中 1-day/5-day ahead 实验窗口。 | 不可扩展为所有可能 horizon 的论文事实。 |
| Sales 必须作为 X 特征输入 | 有但需谨慎 | 可保留为“past sales information in X train”；代码必须保证 lagged sales。 | 若当前/未来 sales 进入 X，是目标泄漏。 |
| KNN 只能使用目标域前 30 天 | 有 | 可保留为论文事实；“严禁触碰测试集”作为工程防泄漏表述。 | Section 5.4.1 明确 30 days KNN。 |
| K 扩展至 6 和 9 | 有 | 可保留为论文事实，但只属于 sensitivity analysis。 | Table 13。 |
| 反权距离公式应优先采用 | 有但有冲突 | 论文存在冲突，需双版本审计。 | 正文/表格支持 inverse distance；Algorithm 1 支持 distance/sum。 |
| 冻结前 4 层 | 有 | 可保留为论文事实。 | Fig. 4 显示 Conv1D+MaxPooling+Conv1D+MaxPooling。 |
| RFE 保留 40%-60% | 有 | 可保留为论文事实。 | 具体 n_features 未明确。 |
| horizon 包括 1、5、15、30 天 | 部分有且存在冲突 | 论文未明确，删除或移入未明确项；实际结果按 1–5 days。 | Section 5.3 提到 1/15/30 作为参数示例，但又说 only 5 days ahead。 |
| No-TL、SS-TL、MSML-TL-RFE runtime 数值 | 部分有 | 可保留 Table 13 中 Dataset 1 的部分 runtime；不要扩展到所有方法/数据集。 | Table 13 有 No-TL 49、SS-TL 458、MSML-RFE 3/6/9 sources 1304/2487/3915。 |
| fixed lead time = 5 days | 有 | 可保留为论文事实。 | Table 11/Section 6.2.1。 |
| dynamic lead time = 3+2 | 有 | 可保留为论文事实，但生成方式未明确。 | Table 12；正文：3 days + additional 1–2 days delay。 |
| Adam、LR=0.001、Batch=32 | 无 | 论文未明确，删除或移入未明确项。 | 工程建议，不可作为论文依据。 |
| Kernel=3、Filters=32/64 | 无 | 论文未明确，删除或移入未明确项。 | 工程建议，不可作为论文依据。 |
| RFE estimator = Random Forest Regressor | 无 | 论文未明确，删除或移入未明确项。 | 工程建议，不可作为论文依据。 |
| Dataset 2 Promotion 对需求波动有显著非线性影响 | 无 | 论文未明确，删除。 | 论文只说明包含 promotional activities。 |
| Dataset 3 必须包含 Customers、Open、Holiday、Store Type | 部分有且冲突 | 需要人工回查原文/数据列；Customer/Open/Promotion/Holiday 见 Fig. 9，store type 只见正文描述。 | Fig. 9 未显示 store type。 |

---

## 18. 论文未明确但复现必须决定的细节

| 细节 | 为什么会影响复现结果 | 代码中应该如何显式记录 | 应归入配置 |
| ---- | ---- | ---- | ---- |
| random seed | 影响 CNN 初始化、训练波动、RFE estimator 随机性。 | `random_seed`、每次 repeat seed。 | engineering-default |
| epochs | 影响收敛和过拟合。 | `source_epochs`、`target_epochs`。 | engineering-default |
| batch size | 影响优化路径和训练时间。 | `batch_size`。 | engineering-default |
| learning rate | 影响收敛速度和稳定性。 | `learning_rate`。 | engineering-default |
| optimizer | 影响训练结果。 | `optimizer`。 | engineering-default |
| loss function | 影响训练目标；论文只给 RMSE evaluation。 | `loss` 与 `evaluation_metric` 分开。 | engineering-default |
| normalization/scaling fit scope | 若 scaler fit 到 test 会泄漏。 | `scaler=minmax`、`fit_on=train_only/source_train_only` 等。 | paper-aligned + engineering防泄漏 |
| early stopping | 影响训练轮数。 | `early_stopping`、monitor、patience。 | engineering-default |
| KNN 输入窗口 | 决定 source 相似度；使用 test 会泄漏。 | `source_selection_window=target_observed_window`。 | paper-aligned |
| KNN 特征空间 | 决定距离和 selected sources。 | `knn_features`、字段顺序、是否 scaling。 | paper-aligned 但细节需工程记录 |
| source_selection_window | 直接决定 selected_sources。 | 每次运行输出 window date/time steps。 | paper-aligned |
| RFE estimator | 影响 selected_features。 | `rfe_estimator`、params、seed。 | engineering-default |
| RFE n_features_to_select | 影响输入维度和 RMSE。 | `n_features_to_select` 或 `feature_keep_ratio`。 | engineering-default，比例 40–60% 为 paper-aligned |
| RFE candidate features | 影响是否目标泄漏。 | `rfe_candidate_features`、是否包含 sales。 | engineering-default + leakage audit |
| 是否允许 sales 作为输入特征 | 过去 sales 可用，未来/current y 会泄漏。 | `include_lagged_sales=True/False`、lag construction。 | 论文部分明确；实现归工程审计 |
| CNN filters | 影响容量。 | `conv_filters`。 | engineering-default |
| CNN kernel size | 影响时间窗口感受野。 | `kernel_size`。 | engineering-default |
| activation | 影响非线性与输出范围。 | `activation`、`output_activation`。 | engineering-default |
| frozen layer index | index 可能因框架是否计 Input 层而变。 | frozen layer names：Conv1D_1, MaxPool_1, Conv1D_2, MaxPool_2。 | paper-aligned + engineering记录 |
| horizon 列表 | 决定评价结果聚合。 | `horizons=[1,2,3,4,5]` 或其他；结果按 horizon 输出。 | paper-aligned for 1–5；其他为 extension |
| 重复实验次数 | 影响 mean predictions 与统计稳定性。 | `n_repeats`。 | engineering-default |
| dynamic lead time 具体生成方式 | 影响 cost。 | `dynamic_lt_base=3`、`delay_distribution`、seed。 | engineering-default |
| 是否使用 validation RMSE 进行模型选择 | MSSB 明确；其他方法未说明。 | `model_selection_metric` by method。 | MSSB paper-aligned；其他 engineering-default |
| 是否对不同 dataset 使用完全一致配置 | 影响可比性。 | `dataset_config` 与 shared defaults。 | 论文未明确 |
| metric space normalized/original | 表中 RMSE 可能基于 normalized data；成本需要原始单位。 | `metric_space`、inverse scaling logic。 | 论文未明确 |

---

## 19. 高风险歧义点

| 歧义点 | 论文依据 | 为什么有歧义 | 可能影响的实验 | 建议代码审计如何标记 | 是否需要双版本实验 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 权重公式：inverse distance vs distance/sum | Section 3/4.3.1/Table 5/6 vs Algorithm 1 lines 13–16 | 正文表格与伪代码方向相反 | MSWA、MSML、MSML-RFE | `weight_formula` | 是 |
| Algorithm 1 写 “Train the new CNN network using target domain test data” | Algorithm 1 注释 line 19 前，但 line 19 实际用 Target_train_data, Target_val_data | 注释疑似笔误；若真用 test 训练会泄漏 | MSML-RFE | `algorithm_comment_conflict=True`，代码不得用 test 训练 | 不建议按 test 训练；可只标冲突 |
| KNN 是否可能使用 target test window | Section 5.4.1 明确 30 days，但 Algorithm 1 KNN 输入宽泛 | 若代码用完整 target window 会泄漏并改变 source | 所有 TL 方法 | `source_selection_window` | 否，paper-aligned 应用 observed window |
| RFE 是 source 上做、target 上做，还是 source+target 一起做 | Section 4.3.4 “mainly from similar source data”；Algorithm 1 对 Target/S1/S2/S3 做 RFE | fitting scope 不清楚，可能引发泄漏 | MSML-RFE | `rfe_fit_scope`、`rfe_transform_scope` | 是 |
| sales 是否作为输入特征导致目标泄漏 | Section 5.3 说 X train 含 past sales information；Table 4 不列具体列 | past lag sales 合理，未来 sales 泄漏 | 所有 CNN 方法 | `lagged_sales_only` | 需要做泄漏审计，不一定双版本 |
| frozen first 4 layers 精确含义 | Section 4.2；Fig. 4 | Keras/PyTorch 是否计 Input 层不同 | SS、MSWA、MSSB、MSML、MSML-RFE | frozen layer names | 可做冻结层敏感性，但不算论文事实 |
| horizon 是否单独训练 | Section 4.2/4.3.2/4.3.3 说 each day ahead repeated；Section 6.1 1–5 days | 表中只给平均，不给 per-horizon | 所有结果表 | `horizon`, `aggregate_method` | 建议输出 per-horizon 以审计 |
| Dataset 3 source/target 构造不够明确 | Section 5.4.2 假设 regions by store order；Rossmann 原始数据不一定有 region | 影响 KNN pool 与 Category a | Dataset 3 | `dataset3_region_rule` | 可做 paper-rule 与 data-native 两版 |
| information sharing 是否仅改变 source pool | Section 5.4.4 只描述 pool 扩大 | 未说明架构/成本参数变化 | Table 7/8/11/12 | `information_sharing_effect=source_pool_only` | 一般不需要 |
| cost model 是否完全可复现 | Eq. 1–4/Table 1 有框架，但缺日级伪代码 | holding/lost sales/lead time 分布会影响 cost | Table 11/12 | `cost_model_implementation_notes` | 需要实现审计，必要时做多版本 |
| Table 8 mean RMSE with info 0.1937 vs 正文 0.1973 | Table 8 与正文不一致 | 平均值引用冲突 | 综合结论 | `paper_table_text_conflict` | 否，报告两者 |
| Table 13 9-source time 表格 3915 vs 正文 3519 | Table 13 与正文不一致 | runtime 引用冲突 | sensitivity runtime | `paper_table_text_conflict` | 否，报告两者 |

---

## 20. 最终代码审计 Checklist

| 检查项 | 代码中应查什么 | 期望 paper-aligned 行为 | 风险等级 |
| --- | ------- | ------------------- | ---- |
| 数据集加载 | 文件路径、字段、日期范围 | Dataset 1/2/3 与论文来源一致 | 高 |
| Dataset 1 清洗 | Store/Item/Date/Sales | 使用 Store 1–3、Item 1–10；Item 10 target | 高 |
| Dataset 2 清洗 | brand/item/date/sales/promotion | Promotion 进入特征；Item 10 target | 高 |
| Dataset 3 清洗 | store/date/sales/customer/open/promo/holiday/store_type | Store 10 target；字段差异显式记录 | 高 |
| source/target 构造 | entity ID、target ID、source pool | target 不进入 source pool | 高 |
| no information sharing source pool | filter 条件 | 同 store/brand/region 内选源 | 高 |
| with information sharing source pool | filter 条件 | 跨 store/brand/region 选源 | 高 |
| same-category/store-type 限制 | `same_category_pass` | 只用于 Section 6.3 sensitivity | 中 |
| train/val/test split | date/time step counts | Table 2/3 对齐 | 高 |
| target observed window | target train+valid | KNN 使用约 30 天 observed window | 高 |
| KNN source selection | distance matrix、features、window | Euclidean distance；all available features；top 3 | 高 |
| selected_sources 输出 | source IDs/rank/distance | 可对照 Table 5/6 | 高 |
| distance 与 weight 计算 | `distance`, `weight` | 记录 inverse-distance 或 pseudo-code formula | 高 |
| inverse distance / pseudo-code distance 双版本 | 权重公式分支 | 不能混写；结果标记 formula | 高 |
| RFE 候选特征 | candidate columns | 不含未来 y；字段记录 | 高 |
| RFE 是否使用 target/test 泄漏 | fit scope | 不使用 target test fit RFE/scaler | 高 |
| final_selected_features | selected columns | 输出每 dataset/method/horizon | 中 |
| CNN 结构 | model summary | Conv1D-MaxPool-Conv1D-MaxPool-Conv1D-Flatten-Dense | 高 |
| frozen layers | layer names/freeze flags | 前 4 个计算层冻结 | 高 |
| No-TL | training data | target only，无 source | 高 |
| SS-TL | source pretrain + target fine-tune | 单最邻近 source | 高 |
| MSWA-TL | 3 source models + weighted predictions | prediction-level weighted average | 高 |
| MSSB-TL | validation RMSE switching | 用 target validation RMSE 选模型 | 高 |
| MSML-TL | weights/biases fusion | layer-wise weighted fusion | 高 |
| MSML-TL-RFE | RFE + MSML | RFE 在 MSML 前，保留约 40–60% | 高 |
| horizon | h values and aggregation | 1–5 days；每 horizon 单独/可追踪 | 中 |
| repeated experiments / seeds | repeat loop | 多次实验均值；seed 为工程默认 | 中 |
| metric space | normalized/original | RMSE 与 cost 使用空间分开记录 | 高 |
| RMSE / accuracy | formula | RMSE；accuracy=1/RMSE | 中 |
| supply chain cost | Eq. 1–4/Table 1 | Dataset 1 only；No-TL/SS-TL/MSML-RFE | 高 |
| static lead time | LT | fixed LT=5 days | 中 |
| dynamic lead time | LT | 3 days + 1–2 days delay；分布显式记录 | 高 |
| result CSV 字段 | schema | dataset, method, scenario, source_count, horizon, RMSE, cost, time, weight_formula | 高 |
| paper-aligned 默认配置 | config file | 与论文明确项分离：paper_aligned vs engineering_default | 高 |

---

## 最终文件建议

1. `paper_reproduction_detail_index.md`  
   严格论文事实索引，用于逐条对照代码是否 paper-aligned。

2. `paper_ambiguities_and_engineering_defaults.md`  
   单独存放论文未明确项、冲突项、工程默认项，避免把工程选择误写成论文事实。

3. `codex_code_audit_prompt.md`  
   发给 Codex 的审计指令，要求其只读检查代码与输出，不修改文件，先报告 paper-aligned / engineering-default / conflict 三类差异。
