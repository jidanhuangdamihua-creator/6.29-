# Gate 1 Contract Traceability Matrix

Matrix ID：gate1_contract_traceability_matrix

Matrix status：FROZEN

Semantic authority：docs/protocol/gate1_frozen_transformation_contract.md

Implementation authority：docs/protocol/gate1_implementation_scope.md

Acceptance rule：每一行必须由对应 acceptance test 证明；缺少任一行的证明，formal preflight 必须阻断。

| ID | human decision | frozen contract clause | implementation component | acceptance test |
|---|---|---|---|---|
| G-01 | 统一正式路径为 frozen schema → availability gate → safe target view → KNN/CNN/transfer | Contract §4；所有正式运行必须经过统一路径 | schema registry、availability resolver、safe target view operator、unified runner | TC-001：正式入口只加载三份正式输入并拒绝临时目录 |
| G-02 | history reconstruction 与 forecast blind generation 分离 | Contract §3.1–§3.4；两个 producer 不共享 target-day fill | history reconstruction producer、forecast blind producer | TC-002：修改预测后 truth 不改变 blind view；worker 无 truth |
| G-03 | 正式字段只来自冻结 schema，禁止自动扩列和 runtime fallback | Contract §3.5、§6；显式 names/order/dtype/role/transform | schema registry、availability resolver | TC-003：注入 numeric、prefix、dtype 和 JSON fallback 字段全部阻断 |
| G-04 | 禁止通用 bfill、双向填充、未批准插值和均值填充 | Contract §3.6；缺失规则逐字段执行 | availability resolver、history/forecast producers | TC-004：未来行扰动和通用填充记录触发失败 |
| G-05 | raw authority、snapshot 和 hash 必须冻结 | Contract §2；hash mismatch、缺失和未列文件 fail closed | authority producer、proof writer、formal preflight | TC-005：修改任一 authority byte 后 preflight blocked |
| D1-01 | target truth 不被重建规则修改；预测窗口不读真实 target | Contract §5.1 | history reconstruction producer、forecast blind producer、safe target view operator | TC-D1-01：target truth 只在 evaluator view 出现，worker view 无 target |
| D2-01 | 历史保留原始 Promo；预测窗口排除 Promo | Contract §5.2 | D2 history/forecast producers、schema registry | TC-D2-01：历史 Promo 保留；forecast view、KNN view、model view 无 Promo |
| D2-02 | D2 无预测期提前计划；禁止读取预测期实际 Promo | Contract §5.2 | D2 forecast blind producer、availability resolver | TC-D2-02：在 raw 中注入未来实际 Promo 不改变 blind output 且该字段被排除 |
| D2-03 | D2 禁止通过 ffill/bfill 或未来行生成 Promo | Contract §5.2、§3.6 | field resolver、forecast blind producer | TC-D2-03：缺失 Promo 不得由未来行或前值生成 |
| D3-01 | Open 缺失且销量大于 0 填 1；销量等于 0 填 0；只用于历史事实重建 | Contract §5.3 Open | history reconstruction producer、repair proof writer | TC-D3-01：两类历史缺失得到精确值，预测窗口不执行该规则 |
| D3-02 | Open 不进入 KNN/CNN/RFE/model，不影响 source pool、样本和日期范围 | Contract §5.3 Open | schema registry、source pool operator、safe target view operator | TC-D3-02：Open 注入或变更不改变 source pool、sample range 和 model columns |
| D3-03 | 预测前不能用测试真实销量生成影响预测的 Open | Contract §5.3 Open | forecast blind producer、formal preflight | TC-D3-03：测试销量/target mutation 不改变 forecast；Open truth 访问被阻断 |
| D3-04 | D3 Promo 历史保留，预测窗口排除；Promo2/PromoInterval 不替代每日 Promo | Contract §5.3 Promo | D3 resolver、schema registry、forecast blind producer | TC-D3-04：长期字段不能填入每日 Promo；forecast view 无 Promo |
| D3-05 | Customers 完全排除，不进 KNN/CNN/RFE/model，不生成 lag/rolling/外部预测 | Contract §5.3 Customers | schema registry、history/forecast producers、model operator | TC-D3-05：Customers、lag、rolling、external forecast 注入全部被排除 |
| D3-06 | SchoolHoliday 使用 Rossmann raw，缺失 0，统一名称/dtype，不由销量/Promo推导 | Contract §5.3 SchoolHoliday | authority producer、history/forecast producers、schema registry | TC-D3-06：raw authority/hash、zero fill、dtype/name 和 no-sales-dependency 全部通过 |
| D3-07 | Region 由 Store 号构造：1–10、11–20、21–30；target Store 10 | Contract §5.3 Domain | source pool operator、domain resolver | TC-D3-07：每个 Store 映射唯一 Region，Store 10 固定 target |
| D3-08 | without-sharing 只用 Store 1–9；with-sharing 用 Store 1–30 排除 Store 10；StoreType 只作敏感性分析 | Contract §5.3 Domain | source pool operator、scenario resolver | TC-D3-08：候选池集合、target exclusion、StoreType 非主 domain 均被证明 |
| D3-09 | region = 1、TODO_REGION_UNAVAILABLE 禁止作为正式 domain | Contract §5.3 Domain | schema registry、formal preflight | TC-D3-09：旧 domain token 注入后 preflight blocked |
| D4-01 | 允许的 future-known covariates 只有七个批准字段，并标记 benchmark-provided future covariate | Contract §5.4 | D4 forecast blind producer、availability resolver、manifest writer | TC-D4-01：七字段进入 approved future view，其他字段不进入 |
| D4-02 | hours_sale、hours_stock_status、stock_hour6_22_cnt 及所有派生小时/库存聚合只能审计 | Contract §5.4 | D4 history producer、safe target view operator、schema registry | TC-D4-02：禁止字段注入 KNN/CNN/RFE/transfer/target view 全部阻断，audit view 可见 |
| D4-03 | 旧 D4 通用 bfill 必须删除 | Contract §5.4、§3.6 | availability resolver、formal preflight、unified runner | TC-D4-03：bfill provenance 或未来行填充记录触发失败 |
| D5-01 | onpromotion 预测期允许；缺失表示未促销并填 0；不套用 D2/D3 排除规则 | Contract §5.5 onpromotion | D5 forecast blind producer、field resolver | TC-D5-01：预测期 onpromotion 保留，缺失为 0，D2/D3 exclusion rule 不适用 |
| D5-02 | transactions 历史缺失填 0，预测窗口完全排除，不读同日实际、不做 lag/rolling/外部预测 | Contract §5.5 transactions | D5 history/forecast producers、schema registry | TC-D5-02：history zero fill 通过；forecast、KNN、model 无 transactions |
| D5-03 | oil_price 只用 prior；升序 forward fill；统一滞后一天；禁止 bfill/interpolation/mean | Contract §5.5 oil_price | D5 history/forecast producer、availability resolver | TC-D5-03：未来油价扰动不影响结果；lag-one 和 allowed ffill 精确匹配 |
| D5-04 | 没有任何历史 prior 时 fail closed，不能用预测后首个有效油价反向补齐 | Contract §5.5 oil_price | resolver、formal preflight、proof writer | TC-D5-04：无 prior 和 post-origin-only 两种 fixture 均 blocked |
| D5-05 | 删除 week；不使用 ISO、业务或财务周；保留 year/month/day | Contract §5.5 week | schema registry、D5 forecast producer | TC-D5-05：所有 view 无 week；year/month/day 按 schema 保留 |
| D5-06 | Holiday/Additional/Bridge/有效 Transfer=1；Work Day=0；按 National/Regional/Local 映射 | Contract §5.5 is_holiday | D5 holiday resolver、history/forecast producer | TC-D5-06：类型、地区层级和二元值逐项匹配 |
| D5-07 | transferred 原日期不重复；新 Transfer 日期为假期；同日合并为 1；merge 不扩行 | Contract §5.5 is_holiday | D5 holiday operator、row-cardinality proof | TC-D5-07：转移日、重复日和一对多 merge fixture 全部通过 |
| D6-01 | sell_price 是 benchmark future-known covariate，按 store_id + item_id + wm_yr_wk 精确连接 | Contract §5.6 sell_price | D6 price resolver、schema registry、forecast producer | TC-D6-01：exact-key join 通过；跨键填充和未来 bfill 被拒绝 |
| D6-02 | sell_price 缺键、重复键或价格缺失时 fail closed | Contract §5.6 sell_price | resolver、formal preflight、proof writer | TC-D6-02：三类异常各自返回稳定 failure code |
| D6-03 | calendar.csv 是唯一 future-known calendar authority，允许字段集合固定 | Contract §5.6 event/SNAP | authority producer、calendar resolver、schema registry | TC-D6-03：替代日历、额外字段和 authority drift 全部阻断 |
| D6-04 | CA/TX/WI 门店只使用对应 SNAP；未知州或映射失败 fail closed；不同时喂三州 | Contract §5.6 event/SNAP | D6 calendar resolver、safe target view operator | TC-D6-04：三州映射、未知州、三列并喂 fixture 全部按合同处理 |
| P-01 | provenance、manifest、schema、resolver 和 artifact 必须绑定同一 contract/raw identity | Contract §6 | proof writer、formal preflight | TC-PROOF-01：删除、篡改或替换任一 proof 后 preflight blocked |
| P-02 | preflight 必须在训练/预测前阻断全部 forbidden state | Contract §6 | formal preflight、unified runner | TC-PREFLIGHT-01：逐项注入 forbidden state，训练 spy 调用数保持 0 |
| R-01 | D4–D6 legacy runner 不再是平行正式入口 | Contract §4、scope §1 | unified runner、formal preflight | TC-RUNNER-01：正式调用图不出现 legacy D4–D6 runner |

## Acceptance test evidence contract

每个 acceptance test 必须记录：

- test ID
- contract_digest
- fixture/raw authority identity
- input mutation
- expected view/schema decision
- expected status 或 stable failure code
- producer/operator call evidence
- proof/manifest digest

测试通过不等于 Implementation 完成；只有三份正式输入、digest sidecar、实现代码、测试证据和 preflight 全部一致时，才可发布正式 run artifact。
