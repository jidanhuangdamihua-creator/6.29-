# Gate 1 Frozen Transformation Contract

Contract ID：gate1_frozen_transformation_contract

Contract version：1.0.0

Contract status：CONTRACT FROZEN

Gate status：GATE 1I AUTHORIZED

Implementation execution：NOT STARTED

Freeze date：2026-07-16

## 1. Authority and precedence

本文件是 Gate 1 业务语义、时间可得性、字段准入、缺失处理、domain 和 raw authority 的唯一正式合同 authority。

正式输入只包括以下三份版本控制文件：

1. docs/protocol/gate1_frozen_transformation_contract.md：唯一语义 authority。
2. docs/protocol/gate1_implementation_scope.md：唯一实施范围 authority，不得改写语义。
3. docs/protocol/gate1_contract_traceability_matrix.md：唯一决策到实施和验收的追踪 authority，不得改写语义。

以下内容全部是审计附件或历史记录，不是并列 authority：/tmp/gate1_semantic_discovery_20260716_132656/ 下的全部文件、旧 candidate contract、旧 implementation scope、human decision closure、Discovery report、legacy runner 配置、private build、structural-only artifact。

发生冲突时，本文件优先。Implementation 只能读取正式路径；读取临时目录、旧文件或 runtime fallback 作为语义来源必须 fail closed。

## 2. Contract digest and raw authority

contract_digest 使用 SHA-256，作用于本文件的精确 UTF-8、LF 换行字节；digest sidecar、追踪矩阵和实施范围不纳入 contract_digest。合同内容任何变化都必须重新计算 digest，并同步更新 sidecar 与引用该 digest 的 manifest。

digest sidecar：docs/protocol/gate1_frozen_transformation_contract.sha256

raw authority 的 hash 使用 SHA-256 作用于文件原始字节，不做换行、编码或内容规范化。下表是本次冻结的文件、角色和 hash。

| dataset | frozen raw authority | role | SHA-256 |
|---|---|---|---|
| D1 | 数据集/原始数据/Dataset 1/train.csv | 历史 target authority | 038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed |
| D1 | 数据集/原始数据/Dataset 1/test.csv | 预测日期/键 authority | 16eb2eec677628c8ed62312a9d6749ebf46f59196dd2fece6fd1e5c147a918fb |
| D2 | 数据集/原始数据/Dataset 2/hierarchical_sales_data.csv | 历史 target 与原始 PROMO authority | 0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83 |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/train.csv | 历史 target 与官方日字段 authority | f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/test.csv | 预测日期/键 authority | e75f79972de046d88c2fd55da19df627f5ca654aaf418090d4d30c60ea7dbe26 |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/store.csv | 门店静态字段 authority | f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344 |
| D4 | 数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet | FreshRetailNet-LT 历史 authority | e744c537ed16a8254c50f8456de7c5614866d4d8b6787e3cc3b67be02ea2f1e3 |
| D4 | 数据集/原始数据/Dataset 4叮咚数据集/data/eval.parquet | FreshRetailNet-LT 评估日期/协变量 authority | 370ca7f995d2a059744e1659eb681e60db694bec2f80a84252426afdd2b6d626 |
| D5 | 数据集/原始数据/Dataset 5Favorita/train.csv | 历史 target 与 onpromotion authority | ccf4236a6b58b0db937b8c5006a0ad8fffef6acd06bed1c10cd5cd4d68d93248 |
| D5 | 数据集/原始数据/Dataset 5Favorita/test.csv | 预测日期/键/onpromotion authority | b9d3bef11ca9b058bdd4c1fa3a69b8a595f4ef39983051eb5ab416e255f13ae6 |
| D5 | 数据集/原始数据/Dataset 5Favorita/items.csv | 商品静态字段 authority | 1efd8295f52c8531ec5bf6c3de37228a56bad2f2c653e79c4869baaf637edcc6 |
| D5 | 数据集/原始数据/Dataset 5Favorita/stores.csv | 门店 city/state/type/cluster authority | af503b2bce11d7906d249f81cc0598f10e2addcc9f6c59aa2d95c9f2652c296b |
| D5 | 数据集/原始数据/Dataset 5Favorita/transactions.csv | 历史 transactions authority | e116384a6981af74932832436aa2f6a43121f77ca81accc44e3a5160158ca03c |
| D5 | 数据集/原始数据/Dataset 5Favorita/oil.csv | 历史油价 authority | 944b23b857580f9d804399346fd3ed69bffcb7facfd98c55fdb408b8d057cca7 |
| D5 | 数据集/原始数据/Dataset 5Favorita/holidays_events.csv | 官方假期 authority | 81a183d6c4d691b57f84a0fde6bbf734a5b3c36a74b97378bf33a592648e2999 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sales_train_validation.csv | 历史 target authority | f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sales_train_evaluation.csv | 预测完成后的 evaluation truth authority | 4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/calendar.csv | 唯一 future-known calendar authority | d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sell_prices.csv | future-known weekly price authority | 9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef |

hash mismatch、文件缺失、路径重定向、重复 authority 或未列出的 authority 必须 fail closed。

## 3. Common time and view rules

1. history reconstruction 只处理 origin 已完整观测的历史事实。
2. forecast blind generation 只读取 origin 时点可用的历史和本合同明确批准的 future-known covariates。
3. 预测完成前，真实 target、测试区间真实销量、同日实际交易量和任何 post-hoc 统计不得进入 worker、source selection、模型输入或 target view。
4. target truth 只能在预测完成后进入 evaluator 或不影响预测的事实审计。
5. year、month、day 是允许的确定性日期特征。未经本合同逐字段列出的字段不得进入任何正式 schema。
6. 缺失规则逐字段执行。通用 bfill、双向填充、未批准插值、未批准均值填充和通过未来行获得的值全部禁止。
7. fail closed 的含义是阻断该样本、字段、source pool 或运行并输出稳定失败证据；不得静默删除、补值、换 authority 或缩小任务范围。

## 4. Required formal path and component contracts

所有 D1–D6 正式运行必须经过：

frozen schema → availability gate → safe target view → KNN/CNN/RFE/transfer

history reconstruction 和 forecast blind generation 必须由两个独立 producer 实现。组件职责固定如下：

- authority producer：验证 raw path、文件 hash、snapshot identity 和字段来源。
- history reconstruction producer：只按历史规则重建已观测事实，输出 history view 与 repair proof。
- forecast blind producer：只生成 origin-blind forecast view，拒绝 target-day actual 和未来不可得字段。
- schema registry：维护每个 dataset/scenario/method 的显式字段名、顺序、dtype、role、transform 和允许 view。
- availability resolver：逐字段解析 available_at、revision policy、source authority 和 missing rule；解析失败即 fail closed。
- safe target view operator：分离 worker 输入、forecast covariate、label truth 和 evaluator truth。
- source pool operator：只从合同批准的 domain 和 schema 构造候选池；目标实体永不进入自身候选池。
- model operator：KNN、CNN、RFE、transfer 只能接收 safe target view 和 frozen predictor/KNN schema。
- proof writer：输出 authority identity、repair mask、availability decisions、view columns、exclusion reasons、source pool identity、schema identity 和 artifact hashes。
- formal preflight：在任何训练或预测前验证上述 identity、schema、as-of、view isolation、domain、row cardinality、missing policy 和 proof completeness。

D4–D6 legacy runner 不得作为平行正式入口。旧 runner 只能被测试用例证明为未被正式路径调用。

## 5. Dataset field contract

### 5.1 D1 common baseline

- 历史 sales 是 target truth；不得修改官方历史 truth。
- 预测窗口不把真实 target 作为输入。
- 允许的日期字段为 year、month、day 及 schema registry 明确列出的确定性日期字段。
- IDs 和静态 key 只能按显式 schema 使用；不得自动数值化或自动扩列。

### 5.2 D2 Promo

- 历史观测区间保留原始 PROMO 字段。
- 预测窗口完全排除 Promo。
- D2 没有预测期提前发布的促销计划。
- 不得读取完整数据中的预测期实际促销状态。
- 不得通过前向填充、反向填充或任何未来行填充生成预测期促销信息。

### 5.3 D3 Rossmann

Open：

- 历史缺失且同日销量大于 0 时填 1。
- 历史缺失且同日销量等于 0 时填 0。
- 该规则只用于完整观测历史事实重建。
- Open 不进入 KNN、CNN、RFE 或其他模型输入。
- Open 不影响 source pool、样本删除、预测日期过滤或评估样本范围。
- 预测完成前不得使用测试区间真实销量生成影响预测的 Open。

Promo：

- 历史保留原始 Promo。
- 预测窗口完全排除 Promo。
- Promo2、PromoInterval 等长期字段不得替代每日 Promo。
- 不读取预测期实际 Promo，不使用 ffill 或 bfill 生成预测期 Promo。

Customers：

- Customers 完全排除预测链路。
- Customers 不进入 KNN、CNN、RFE 或其他模型。
- Customers 不生成 lag、rolling 或外部客流预测。
- 原始 Customers 只可用于预测后审计，不得影响预测链路。

SchoolHoliday：

- 使用 D3 Rossmann train.csv/test.csv 的原始 SchoolHoliday 作为官方 benchmark authority。
- 缺失填 0。
- 统一字段名和 dtype。
- 不引入外部学校假期日历。
- 不由销量或 Promo 推导。

Domain：

- Store 1–10 映射 Region 1。
- Store 11–20 映射 Region 2。
- Store 21–30 映射 Region 3。
- 主目标是 Store 10。
- without-sharing 候选来源只允许 Store 1–9。
- with-sharing 候选来源只允许 Store 1–30 中排除 Store 10。
- StoreType 只用于同类别来源敏感性分析，不替代 Region 1–3。
- region = 1 和 TODO_REGION_UNAVAILABLE 禁止作为正式主实验 domain。

### 5.4 D4 FreshRetailNet-LT

D4 严格采用 FreshRetailNet-LT 官方 benchmark。

允许进入 forecast view 的 future-known covariates 只有：

activity_flag、discount、holiday_flag、precpt、avg_temperature、avg_humidity、avg_wind_level

上述字段的 contract class 和 manifest class 必须写为 benchmark-provided future covariate。该类别采用 benchmark 提供未来协变量的实验假设，不要求额外证明真实部署发布时间。

以下字段及其全部派生聚合禁止进入 KNN source selection、CNN、RFE、transfer model 和 forecast target view：

- hours_sale
- hours_stock_status
- stock_hour6_22_cnt
- 由上述字段生成的 sum、max、非零小时数和其他聚合

上述禁止字段只能用于历史分析、缺货识别、样本评估和审计。D4 通用 bfill 必须被拒绝。

### 5.5 D5 Favorita

onpromotion：

- 预测窗口允许使用数据集提供的 onpromotion。
- 缺失表示未促销，填 0。
- D5 不使用 D2/D3 的预测期排除 Promo 规则。
- 字段来源固定为 D5 train.csv/test.csv 的 onpromotion。

transactions：

- 历史缺失填 0。
- 预测窗口完全排除 transactions。
- 不使用同日实际 transactions。
- 不生成 transactions lag 或 rolling。
- 不使用外部交易量预测。
- transactions 不进入 KNN 或模型输入。

oil_price：

- 字段来源固定为 D5 oil.csv 的 dcoilwtico。
- 日期按升序处理。
- 只对历史缺失执行前向填充。
- 统一使用前一日或更早油价，并滞后一天。
- 禁止反向填充、插值和均值填充。
- 没有任何历史 prior 时 fail closed。
- 不得使用预测日期之后的首个有效油价反向补齐。

week：

- 删除 week。
- 不使用 ISO week、业务周或财务周。
- week 不进入 KNN 或 CNN。
- year、month、day 继续按显式日期 schema 使用。

is_holiday：

- Holiday、Additional、Bridge 映射为 1。
- 有效转移后的 Transfer 日期映射为 1。
- Work Day 映射为 0。
- National 适用于全部门店。
- Regional 按门店所属 state/region 适用。
- Local 按门店所属 city 适用。
- transferred=True 的原始日期不得重复记为假期。
- 实际转移到的新日期记为假期。
- 同日多个适用假期合并为单个值 1。
- merge 不得产生一对多销售行扩增。

### 5.6 D6 M5

D6 严格采用 M5 官方 benchmark。

sell_price：

- 预测窗口对应的未来周价格是 benchmark-provided future-known covariate。
- 字段来源固定为 sell_prices.csv 的 sell_price。
- 连接键严格为 store_id + item_id + wm_yr_wk。
- 缺键、重复键或价格缺失时 fail closed。
- 禁止跨门店、跨商品填充。
- 禁止从未来周 bfill。
- 该规则是 benchmark assumption，不表示真实部署价格发布时间已被证明。

event / SNAP calendar：

- calendar.csv 是 D6 唯一 future-known calendar authority。
- 允许字段只有 weekday、wday、wm_yr_wk、event_name_1、event_type_1、event_name_2、event_type_2、snap_CA、snap_TX、snap_WI。
- CA 门店只使用 snap_CA。
- TX 门店只使用 snap_TX。
- WI 门店只使用 snap_WI。
- 门店州未知或映射失败时 fail closed。
- 不得将三个州的 SNAP 值同时作为同一门店的有效输入。
- 其他日历版本不得静默覆盖冻结 calendar.csv。

## 6. Schema, resolver, proof and preflight outputs

每个 dataset/scenario/method 必须注册：

- ordered feature names
- ordered dtypes
- role：target、history-only、future-known、audit-only、key 或 static
- history reconstruction rule
- forecast generation rule
- availability class
- available_at
- revision policy
- missing policy
- allowed consumers
- exclusion reasons
- schema digest

availability resolver 必须输出逐字段 decision、authority、available_at、rule、view membership 和 failure code。safe target view 必须输出 worker columns、KNN columns、forecast covariate columns、label columns 和 evaluator-only columns 的分离证明。

proof writer 必须输出 raw authority hashes、contract_digest、schema digests、resolver decisions、history repair counts/digests、forecast exclusion counts、candidate pool/domain、view column lists、row cardinality checks、source-selection identity 和 artifact hashes。

formal preflight 必须在训练前阻断以下任一状态：contract digest 不匹配、raw hash 不匹配、schema drift、字段缺失或新增、available_at 超过 origin、target-day actual 存在、禁止字段存在、通用填充记录存在、domain 映射失败、候选池包含目标、merge 行数扩增、proof 缺失或任何 stable failure code。

## 7. Freeze transition

本次 Contract Consolidation 完成的判据为：三份正式文件已写入版本控制路径；本文件正文每一条准入均为确定性条款；traceability matrix 覆盖所有人工决定、组件和 acceptance test；raw authority hashes 已记录；contract_digest sidecar 已计算并通过核对；正式路径 authority 规则已写明。

满足上述判据后，状态为 CONTRACT FROZEN 和 GATE 1I AUTHORIZED。该状态只授权下一阶段按 scope 实施，不表示代码、数据重建、实验或正式 artifact 已经执行。
