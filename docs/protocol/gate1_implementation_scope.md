# Gate 1 Implementation Scope

Scope ID：gate1_implementation_scope

Scope status：GATE 1I AUTHORIZED

Execution status：NOT STARTED

Semantic authority：docs/protocol/gate1_frozen_transformation_contract.md

Traceability authority：docs/protocol/gate1_contract_traceability_matrix.md

## 1. Scope boundary

本 scope 覆盖唯一正式运行链路：

frozen schema → availability gate → safe target view → KNN/CNN/RFE/transfer

本 scope 不授权修改业务语义、不授权替换 raw authority、不授权从临时目录读取合同、不授权平行 legacy runner、不授权在 worker 中读取 target-day actual、不授权自动扩列或通用未来填充。

本 scope 的语义必须完全来自 gate1_frozen_transformation_contract.md。scope 与 traceability matrix 不得覆盖或修改合同条款。

## 2. Required components

| component | required responsibility | required inputs | required outputs | blocking conditions |
|---|---|---|---|---|
| authority producer | 读取正式 raw path，校验文件 hash，建立 snapshot identity | frozen contract raw table | authority manifest、file hashes、source identities | missing file、hash drift、path redirect、unlisted file |
| history reconstruction producer | 只重建完整观测历史事实 | raw authority、history rules、history view schema | history view、repair mask、repair proof | future row、target leakage、rule mismatch |
| forecast blind producer | 只生成 origin-blind 预测视图 | history view、forecast dates、approved future-known fields | blind forecast view、availability proof | target-day actual、post-origin field、unapproved fill |
| schema registry | 固定字段名、顺序、dtype、role、transform、consumer 和 schema digest | frozen contract | predictor schema、KNN schema、view schemas | extra field、missing field、dtype drift、digest drift |
| availability resolver | 逐字段解析 authority、available_at、revision policy、missing policy 和 view membership | contract clause、raw metadata、origin | field decisions、exclusion reasons、failure codes | unknown field、late availability、missing authority |
| safe target view operator | 分离 worker input、KNN input、forecast covariates、label 和 evaluator truth | history view、blind view、schemas | safe target view、view-isolation proof | truth in worker、view overlap、column mismatch |
| source pool operator | 执行 D3 synthetic region、with/without candidate pool 和 target exclusion | safe views、domain schema、KNN schema | candidate pool、ordered selection、pool digest | target included、domain failure、unapproved source |
| model operator | 只消费 frozen predictor/KNN schema 和 safe views | KNN pool、CNN view、RFE mask、transfer inputs | fitted bundles、predictions、method provenance | forbidden feature、target actual、schema mismatch |
| proof writer | 绑定 authority、contract、schema、resolver、view、pool 和 artifact identity | all producer/operator outputs | manifest、provenance、proof digests | missing proof、identity mismatch、unbound artifact |
| formal preflight | 在训练/预测前完成全部阻断检查 | contract digest、authority manifest、schemas、proofs、run config | pass report 或 stable failure report | 任一合同、as-of、schema、proof、domain 失败 |
| unified runner | 只编排上述组件，不实现第二套语义 | formal preflight pass、component APIs | one formal run path | call to legacy runner、runtime fallback |
| acceptance test suite | 固化追踪矩阵中的所有 acceptance tests | mini fixtures、mutations、fake spies | pass/fail evidence、failure code assertions | any untested clause or unexpected pass |

## 3. Producer implementation

### 3.1 Authority producer

实现一个单一 authority loader。它必须：

- 只读取合同列出的 raw path。
- 对每个文件重新计算 SHA-256 并与合同值比较。
- 保存文件大小、hash、相对路径、snapshot identity 和读取时间。
- 拒绝临时目录、旧 sealed/private build、未列出的副本和静默替换。
- 将 authority identity 传递给所有下游 producer、operator、proof 和 preflight。

### 3.2 History reconstruction producer

它必须：

- 只处理 origin 已完整观测的历史。
- 按合同逐字段执行历史缺失重建。
- 生成 repair mask、repair counts、affected rows 和 repair digest。
- 保留 target truth，不以预测期规则污染历史。
- 不输出可被 forecast producer 当作未来事实的共享 target-day fill。

必须实现的历史规则包括 D3 Open、D5 transactions、D5 oil_price 历史 forward fill、D5 is_holiday 历史映射，以及合同列出的其他明确规则。

### 3.3 Forecast blind producer

它必须：

- 接收 origin、forecast dates、history view 和 approved future-known source。
- 在生成 worker view 前切断 target truth、测试区间真实销量、同日实际 transactions、D4 禁止小时/库存字段和 D6 evaluation truth。
- 按字段 available_at 和 revision policy 做 as-of 解析。
- 仅输出合同允许的 future-known fields。
- 对无 prior、缺 key、重复 key、state 映射失败、字段缺失或 hash drift fail closed。

## 4. Operator implementation

### 4.1 Resolver

resolver 必须将每个候选字段解析为一个不可变 decision：

field、dataset、source authority、history rule、forecast rule、available_at、revision policy、missing rule、consumer set、view set、status、failure code。

resolver 不得根据 numeric dtype、字段前缀、JSON 缺失或当前 dataframe 自动添加字段。

### 4.2 Safe target view

operator 必须产生不重叠的逻辑视图：

- worker/model input：仅合同允许的 predictor fields。
- KNN input：仅合同允许的 source-selection fields。
- forecast covariates：仅合同允许的 future-known fields。
- label truth：仅 evaluator 在预测完成后读取。
- audit-only：只进入 proof 或 post-forecast audit，不进入 source/model path。

operator 必须记录每个 view 的列名、dtype、来源和 digest，并拒绝交叉污染。

### 4.3 Source pool and domain

D3 必须实现 synthetic Region 1–3；without-sharing 只能使用 Store 1–9；with-sharing 只能使用 Store 1–30 excluding Store 10。所有场景都必须排除 Store 10。source pool operator 必须输出 eligible source count、excluded target count、ordered selected sources、candidate pool digest 和 selection result digest。

### 4.4 KNN, CNN, RFE and transfer

各 method operator 只能从 safe target view 读取输入。KNN、CNN、RFE 和 transfer 不得读取原始完整 dataframe、evaluator truth、target-day actual、D3 Customers/Open/Promo、D4 禁止小时/库存字段、D5 transactions/week 或未经 schema registry 批准的列。

## 5. Proof and manifest implementation

每个正式 producer/operator 必须写入可消费 proof，至少包括：

- contract_digest 和 contract version。
- raw authority path、hash、snapshot identity。
- schema names、versions、digests 和 ordered columns。
- resolver 的逐字段 availability decision。
- history repair counts、affected-row digest 和 repair-mask digest。
- forecast blind exclusion counts 和 exclusion reasons。
- worker、KNN、forecast、label、audit view 的列清单与 digest。
- D3 domain/candidate pool identity。
- D5 oil prior edge status、D6 join status 和 D6 SNAP state mapping。
- artifact physical hash、logical identity 和 provenance code identity。

manifest 只有在所有 proof identity 一致、字段集合精确匹配、row cardinality 不变和 preflight 通过后才可发布。

## 6. Formal preflight

preflight 必须在任何训练或预测调用前完成：

1. contract_digest 校验。
2. 三份正式输入路径校验；临时目录和旧文件拒绝。
3. raw file hash 和 snapshot 校验。
4. schema names、order、dtype、role、transform 和 digest 校验。
5. availability/as-of 校验。
6. forecast view 与 history view 隔离校验。
7. target-day actual、evaluation truth 和禁止字段扫描。
8. D3 domain、target exclusion 和 candidate pool 校验。
9. D5 oil prior-only、lag-one、no-prior fail-closed 校验。
10. D5 holiday merge row cardinality 校验。
11. D6 price exact-key、duplicate-key、missing-price 校验。
12. D6 calendar authority 和 state-specific SNAP 校验。
13. proof completeness、manifest identity 和 artifact hash 校验。

任一检查失败都必须返回稳定 failure code 并阻断训练、预测、数据重建和 artifact 发布。

## 7. Test implementation

以下测试是本 scope 的强制 acceptance suite，测试名称和行为由 traceability matrix 固定：

- TC-001 authority precedence and temp-path rejection。
- TC-002 history/forecast producer isolation。
- TC-003 explicit schema rejects auto-expansion。
- TC-004 field-specific missing rules reject generic fill。
- TC-005 raw authority hash drift blocks preflight。
- TC-D1-01 target truth isolation。
- TC-D2-01 D2 Promo forecast exclusion。
- TC-D2-02 D2 future actual Promo exclusion。
- TC-D2-03 D2 no future fill。
- TC-D3-01 Open historical reconstruction only。
- TC-D3-02 Open exclusion from source/model/sample scope。
- TC-D3-03 Open target-day actual isolation。
- TC-D3-04 D3 Promo and long-term field separation。
- TC-D3-05 D3 Customers exclusion。
- TC-D3-06 D3 SchoolHoliday authority and dtype。
- TC-D3-07 D3 synthetic region mapping。
- TC-D3-08 D3 with/without candidate pools。
- TC-D3-09 D3 forbidden legacy domain tokens。
- TC-D4-01 approved benchmark future covariates only。
- TC-D4-02 hourly and stock risk exclusion。
- TC-D4-03 D4 generic fill rejection。
- TC-D5-01 onpromotion future view and zero fill。
- TC-D5-02 transactions forecast exclusion。
- TC-D5-03 oil prior-only forward-fill lag-one。
- TC-D5-04 oil no-prior fail closed。
- TC-D5-05 week removal。
- TC-D5-06 holiday mapping and no row expansion。
- TC-D5-07 transferred holiday deduplication and row cardinality。
- TC-D6-01 sell_price exact-key join。
- TC-D6-02 sell_price fail-closed conditions。
- TC-D6-03 calendar authority and allowed fields。
- TC-D6-04 state-specific SNAP mapping。
- TC-PROOF-01 proof completeness and identity binding。
- TC-PREFLIGHT-01 formal preflight blocks every forbidden state。
- TC-RUNNER-01 unified runner never calls legacy D4–D6 runners。

Acceptance tests 必须覆盖正常、缺失、重复、未来扰动、authority drift、schema drift、禁止字段注入、target-day actual 注入和 row-expansion 注入。只通过 structural validation 的旧 artifact 不得作为 acceptance evidence。

## 8. Execution order and non-goals

实施顺序固定为：

1. authority producer 与 contract/schema binding。
2. history reconstruction producer。
3. forecast blind producer 与 availability resolver。
4. safe target view、source pool 和 model operators。
5. proof writer、manifest 和 formal preflight。
6. acceptance test suite。
7. 由 preflight 通过的 unified runner dry-run。

本轮 Contract Consolidation 不执行上述任何代码或数据动作。当前只完成正式合同、scope、traceability matrix、digest sidecar 和静态合同核验。
