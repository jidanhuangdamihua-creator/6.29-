# Gate 1R Frozen Transformation Contract

Contract ID: `gate1_frozen_transformation_contract`

Contract version: `1R.1.0`

Contract status: `CONTRACT RE-FROZEN`

Gate 1X status: `PROHIBITED UNTIL ONE-TIME IMPLEMENTATION IS COMPLETE`

Freeze date: `2026-07-16`

## 1. Authority, precedence, and boundary

This document is the sole semantic authority for the Gate 1 transformation contract. The implementation scope and traceability matrix are companion authorities for execution responsibility and evidence; neither may change the meaning of this document.

The human decision authority is:

`/Users/ming/Desktop/复现实验/实验确定信息/Gate_1R_合同补全与重新冻结决策书_预填写.docx`

The decision-book SHA-256 is `sha256:4aaebe5f07d3dc0e61ada72dbe0625c82615ba74577e91b72b46b07a709c689d`.

All green-highlighted entries are approved final decisions. The old template legend that described green as mutually exclusive alternatives has no authority after human confirmation. Formal clauses below are affirmative, unique, and testable; Word color, checkboxes, and template wording have no contract effect.

This Gate 1R task is docs-only. It may update this contract, the implementation scope, the traceability matrix, the existing contract digest sidecar, the Gate 1R re-freeze record, and tests that prove contract identity and completeness. It must not modify producer, operator, transformation, model, data, authority, manifest, deployment, or experiment files.

The previous Gate 1X implementation commit `b8c1186ba9d8f96f98368a0fc5b438312a0e8813` is retained historically but is `SUPERSEDED AS FINAL GATE 1X IMPLEMENTATION BASIS` because it was implemented before the Gate 1R contract gap was fully closed. The failed controlled rerun is `20260716T112325Z-e2cfe3f889ec43fc983c4dac2dd1bb71`, dataset D3, stage producer, failure `HISTORY_FUTURE_ROW`.

## 2. Contract identity and raw authority

The contract digest is SHA-256 over this file's exact UTF-8 bytes with LF line endings. The sidecar and companion documents are excluded from `contract_digest`. Any byte change requires a new digest and sidecar update.

Sidecar: `docs/protocol/gate1_frozen_transformation_contract.sha256`

The formal identity records these hashes separately:

- contract: the contract digest in the sidecar;
- scope: SHA-256 over `docs/protocol/gate1_implementation_scope.md`;
- matrix: SHA-256 over `docs/protocol/gate1_contract_traceability_matrix.md`;
- decision book: the SHA-256 above.

The combined formal identity is the SHA-256 of these exact LF-terminated UTF-8 lines, in this order:

```text
decision_book_sha256=<hex>
contract_sha256=<hex>
scope_sha256=<hex>
matrix_sha256=<hex>
```

The re-freeze record stores the resulting component hashes and combined digest.

The frozen raw authority table is unchanged from the prior authority and remains byte-bound:

| dataset | frozen raw authority | role | SHA-256 |
|---|---|---|---|
| D1 | 数据集/原始数据/Dataset 1/train.csv | historical target authority | 038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed |
| D1 | 数据集/原始数据/Dataset 1/test.csv | forecast date/key authority | 16eb2eec677628c8ed62312a9d6749ebf46f59196dd2fece6fd1e5c147a918fb |
| D2 | 数据集/原始数据/Dataset 2/hierarchical_sales_data.csv | historical target and raw PROMO authority | 0dfd4a5bf801bf79ddb5fb1f6bc8d023487c36dd6a2a70b312cdfa9fa6568d83 |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/train.csv | historical target and official daily-field authority | f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/test.csv | forecast date/key authority | e75f79972de046d88c2fd55da19df627f5ca654aaf418090d4d30c60ea7dbe26 |
| D3 | 数据集/原始数据/Dataset 3 rossmann-store-sales/store.csv | store metadata authority | f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344 |
| D4 | 数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet | FreshRetailNet-LT historical authority | e744c537ed16a8254c50f8456de7c5614866d4d8b6787e3cc3b67be02ea2f1e3 |
| D4 | 数据集/原始数据/Dataset 4叮咚数据集/data/eval.parquet | FreshRetailNet-LT evaluation date/covariate authority | 370ca7f995d2a059744e1659eb681e60db694bec2f80a84252426afdd2b6d626 |
| D5 | 数据集/原始数据/Dataset 5Favorita/train.csv | historical target and onpromotion authority | ccf4236a6b58b0db937b8c5006a0ad8fffef6acd06bed1c10cd5cd4d68d93248 |
| D5 | 数据集/原始数据/Dataset 5Favorita/test.csv | forecast date/key/onpromotion authority | b9d3bef11ca9b058bdd4c1fa3a69b8a595f4ef39983051eb5ab416e255f13ae6 |
| D5 | 数据集/原始数据/Dataset 5Favorita/items.csv | item metadata authority | 1efd8295f52c8531ec5bf6c3de37228a56bad2f2c653e79c4869baaf637edcc6 |
| D5 | 数据集/原始数据/Dataset 5Favorita/stores.csv | store city/state/type/cluster authority | af503b2bce11d7906d249f81cc0598f10e2addcc9f6c59aa2d95c9f2652c296b |
| D5 | 数据集/原始数据/Dataset 5Favorita/transactions.csv | historical transactions authority | e116384a6981af74932832436aa2f6a43121f77ca81accc44e3a5160158ca03c |
| D5 | 数据集/原始数据/Dataset 5Favorita/oil.csv | historical oil authority | 944b23b857580f9d804399346fd3ed69bffcb7facfd98c55fdb408b8d057cca7 |
| D5 | 数据集/原始数据/Dataset 5Favorita/holidays_events.csv | official holiday authority | 81a183d6c4d691b57f84a0fde6bbf734a5b3c36a74b97378bf33a592648e2999 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sales_train_validation.csv | historical target authority | f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sales_train_evaluation.csv | post-forecast evaluation truth authority | 4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/calendar.csv | sole future-known calendar authority | d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5 |
| D6 | 数据集/原始数据/Dataset 6m5-forecasting-accuracy/sell_prices.csv | future-known weekly-price authority | 9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef |

Missing files, byte drift, path redirection, duplicate authority, or an unlisted authority fails closed.

## 3. Global frozen clauses

### G01 — Window authority

D1–D6 main experiments use one paper-style 180 Gregorian-natural-day backtest. Dataset benchmark test/eval data may serve only as an additional holdout, date/key authority, or approved future-known-covariate authority. It never replaces the main window. A benchmark holdout and the main backtest use different run identities.

### G02 — Forecast origin

For each dataset, `origin` is the final day of the target domain's continuous 30-day observed period. It is simultaneously the final validation day, source-history cutoff, source-pretrain cutoff, and KNN observation cutoff. Dataset-specific dates are fixed in Section 5.

### G03 — Origin role

Within the target role, `origin` belongs only to the final validation day; the blind window starts at `origin + 1 day`. Source history may end on `origin`, but source and target have separate entities, roles, views, and proof identities and are not one overlapping view.

### G04 — Source history

For every source entity, source history is the closed interval `[origin-179 days, origin]`, exactly 180 Gregorian natural days. Any missing, duplicate, or unauthorized row fails closed. Target observed history is separately represented as 15 train days plus 15 validation days.

### G05 — Target observed split

The target domain's 30 observed days are split into 15 target-train days followed by 15 validation days. Validation is excluded from target-train fitting and is used only for model selection, weight/switch decisions, and early stopping.

### G06 — Blind window and horizons

The blind interval is `[origin+1 day, origin+180 days]`, exactly 180 Gregorian natural days. Formal rolling horizons are 1, 2, 3, 4, and 5 days.

### G07 — Processing order

The only valid order is:

```text
full parent identity/key validation
→ entity/role/window slicing
→ calendarization only inside approved entities/windows
→ field-specific repair
→ worker-safe/evaluator-truth/audit views
→ schema validation
→ canonical digest
→ independent proof
→ formal preflight
→ publication validation
```

Full-parent calendarization or repair before slicing is prohibited.

### G08 — Gregorian and missing-day rules

Gregorian dates are authoritative and leap days are included naturally. Every exact entity key/date must be unique; duplicates fail closed. Calendarization is allowed only inside frozen entities and windows and only under the field-specific rules in this contract. A field without an approved missing authority fails closed. Generic backfill, interpolation, mean fill, and future-row fill are prohibited.

### G09 — Target scope

The exact canonical target keys are:

| dataset | canonical target keys |
|---|---|
| D1 | `(store_id=1,item_id=10)` |
| D2 | `(brand_id=1,item_id=10)` |
| D3 | `store_id=10` |
| D4 | `(166,258)`, `(166,432)`, `(166,433)`, `(166,313)`, `(166,311)` |
| D5 | `48/364606`, `48/1159415`, `48/1159414`, `48/1349808`, `48/320682` |
| D6 | `CA_1/FOODS_3_586`, `CA_1/FOODS_3_080`, `CA_1/FOODS_3_555`, `CA_1/FOODS_3_377`, `CA_1/FOODS_3_668` |

### G10 — Source eligibility

Source eligibility is resolved separately for without-sharing and with-sharing. Each candidate must be in the dataset's frozen domain, have a complete 180-day source history, satisfy the frozen KNN schema and field availability, and be excluded when its exact key is a target key. No runtime volume heuristic may add or remove a candidate.

### G11 — Blind cardinality

The expected main blind cardinalities are `D1=1×180=180`, `D2=1×180=180`, `D3=1×180=180`, `D4=5×180=900`, `D5=5×180=900`, and `D6=5×180=900`. The operator independently recomputes exact keys and dates. D2 and D5 missing-day rules below close their observed 175/180 and 885/900 gaps.

### G12 — KNN observation window

KNN observes `[origin-29 days, origin]`, exactly 30 natural days. Approved calendarization occurs before distance calculation. Each dataset's KNN fields and Top-3 rule are stated in its chapter.

### G13 — Artifact and view roles

The published bundle contains separate `source_history`, `target_observed`, `worker_safe_blind`, `evaluator_truth`, and `audit_view` objects. Worker-safe data contains observed target history, recursive forecast state, and approved future-known covariates only. Real blind/evaluation truth is restricted to evaluator truth and audit.

### G14 — Canonical digest

The published source, worker-safe view, evaluator truth, and audit view each have a separate canonical digest. Canonicalization fixes column names, column order, dtype, null representation, and stable sort by exact entity key then date. The operator recalculates every digest from the actual artifact or consumer frame independently.

### G15 — Independent proof

Proof is independently recomputed from the actual artifact and consumer frame. It covers raw/parent identity, approved inputs, window, exact key uniqueness, entity scope, cardinality, calendarization and repair masks, exclusions, candidate pool, schema/view/content digests, truth isolation, and no-leakage. A declared boolean without the underlying recomputation is insufficient.

### G16 — D1/D2 pass-through

D1 and D2 may retain a byte-identical physical copy, but the copy never proves readiness by itself. Exact key, null, schema, window, worker-safe/evaluator truth separation, and semantic proof are mandatory. Empty key/date-feature rows are rebuilt from the raw wide authority under the frozen exact key and are not silently deleted.

## 4. Common view, field, proof, and failure rules

History reconstruction and forecast blind generation are separate roles. Future target truth, evaluation sales, D5 same-day actual transactions, D3 Customers/Open, D4 hourly/stock fields, and any field not approved by a dataset chapter are excluded from worker, KNN, CNN, RFE, and transfer consumers. `evaluator_truth` and `audit_view` may contain prohibited fields only after prediction and only with their own digests.

Each formal artifact records its contract identity, raw authority identity, schema identity, availability decision, row cardinality, exclusion reasons, repair mask, view columns, canonical content digest, and independent proof digest. Any mismatch, unapproved merge expansion, missing key, duplicate key, unknown encoding, late field, or proof mismatch fails closed.

## 5. Dataset-specific frozen contracts

Every chapter is self-contained. Its unique executable conclusion is the only dataset-specific implementation input.

### D1 — Dataset 1

1. **Raw authority.** `Dataset 1/train.csv` is historical target authority; `Dataset 1/test.csv` is the extra forecast-date/key authority.
2. **Main backtest authority.** The main experiment is the fixed 180-day backtest, not raw test.
3. **Extra benchmark/holdout role.** Raw test `2018-01-01..2018-03-31` is an additional holdout/date-key authority with a distinct run identity.
4. **Forecast origin.** `2017-06-30`.
5. **Origin authority.** Paper Table 2 target split `2017-06-01..06-15` train and `2017-06-16..06-30` validation.
6. **Origin role.** Origin is validation's final day; source history may end the same day as a separate source role.
7. **Source history interval.** `2017-01-02..2017-06-30`, closed, 180 Gregorian days.
8. **Target train interval.** `2017-06-01..2017-06-15`, closed, 15 days.
9. **Validation interval.** `2017-06-16..2017-06-30`, closed, 15 days; excluded from target-train fitting.
10. **Blind interval.** `2017-07-01..2017-12-27`, closed, 180 Gregorian days.
11. **Target canonical keys.** `(store_id=1,item_id=10)`; the key is exact and immutable.
12. **Source eligibility.** Without-sharing uses Store 1's other items; with-sharing uses the approved Store 1–3 pool; target exact key is excluded and every candidate needs complete history.
13. **KNN window and fields.** `2017-06-01..2017-06-30`, 30 days; field `sales`; choose Top-3 by the frozen sales distance.
14. **Expected cardinality.** `1×180=180` blind rows.
15. **Calendar/missing rules.** Gregorian calendar; exact `(store_id,item_id,date)` uniqueness; approved entity/window calendarization only; no generic fill. `entity_id` is display/serialization metadata and cannot replace the composite source key.
16. **Field-specific availability.** Sales history is observed; date fields are deterministic; `week` is generated from ISO-8601 date rules with integer dtype.
17. **Worker-safe fields.** Exact key, history-derived sales/features, deterministic date fields, and recursive forecast state only.
18. **Evaluator truth.** Blind sales are stored only in `evaluator_truth` after forecasting.
19. **Audit-only fields.** Physical-copy identity, raw test comparison, repair masks, and proof metadata are audit material.
20. **Forbidden fields.** Blind actual sales, raw test labels, and any future field not explicitly approved are forbidden in worker/KNN/model views.
21. **Canonical digest objects.** Separate digests bind published source, worker-safe blind, evaluator truth, and audit view using the G14 rules.
22. **Fail-closed conditions.** Duplicate or missing composite keys, null key fields, wrong window, schema/dtype drift, or pass-through without semantic proof fails closed.
23. **Unique executable conclusion.** D1 uses origin `2017-06-30`, source history `2017-01-02..06-30`, train `06-01..06-15`, validation `06-16..06-30`, blind `07-01..12-27`, target `(1,10)`, 180 blind rows, Store 1/Store 1–3 sharing pools, sales-only KNN, ISO week, and four independently digested views.

### D2 — Dataset 2

1. **Raw authority.** `Dataset 2/hierarchical_sales_data.csv` is the single wide-table authority for date, sales, and original PROMO.
2. **Main backtest authority.** D2 uses the unified 180-day paper-style backtest; it has no independent raw test authority.
3. **Extra benchmark/holdout role.** No separate test/eval file exists; no other date source may replace the frozen backtest.
4. **Forecast origin.** `2018-06-30`.
5. **Origin authority.** Target split `2018-06-01..06-15` train and `2018-06-16..06-30` validation under the unified construction.
6. **Origin role.** Origin is validation's final day and source cutoff; the roles remain separate.
7. **Source history interval.** `2018-01-02..2018-06-30`, closed, 180 Gregorian days.
8. **Target train interval.** `2018-06-01..2018-06-15`, closed, 15 days.
9. **Validation interval.** `2018-06-16..2018-06-30`, closed, 15 days and excluded from target-train fitting.
10. **Blind interval.** `2018-07-01..2018-12-27`, closed, 180 Gregorian days.
11. **Target canonical keys.** `(brand_id=1,item_id=10)`.
12. **Source eligibility.** Without-sharing uses other items in Brand 1; with-sharing uses the approved Brand 1–3 pool; target exact key is excluded and candidates need complete 180-day history.
13. **KNN window and fields.** `2018-06-01..2018-06-30`, 30 days; fields `sales` and historical `PROMO`; choose Top-3 under the frozen schema.
14. **Expected cardinality.** `1×180=180` blind rows; the observed 175 rows are closed by the rule in item 15.
15. **Calendar/missing rules.** Calendarize only the frozen target and blind window; fill each missing blind sales day with `sales=0`; rebuild the approved covariates under the frozen authority; include leap days and reject duplicates.
16. **Field-specific availability.** Historical PROMO is retained; forecast PROMO is unavailable and must be excluded; empty key/date-feature rows are rebuilt from the raw wide table using the exact key, not silently deleted; `week` is deterministic ISO-8601 integer.
17. **Worker-safe fields.** Exact key, calendarized sales, approved historical features, deterministic date fields, and recursive forecast state; no forecast PROMO.
18. **Evaluator truth.** Blind sales truth is evaluator-only; it never enters worker, KNN, or model input before forecast completion.
19. **Audit-only fields.** Original future PROMO, raw wide-row lineage, missing-day masks, and reconstruction proof are audit-only.
20. **Forbidden fields.** Forecast PROMO, target-day actual sales, future-row fill values, and unapproved covariates are forbidden in worker/KNN/model views.
21. **Canonical digest objects.** Published source, worker-safe blind, evaluator truth, and audit view each receive an independent G14 digest.
22. **Fail-closed conditions.** Duplicate exact key/date, null key/date-feature row not rebuilt from raw, unauthorized fill, forecast PROMO exposure, wrong cardinality, schema drift, or missing proof fails closed.
23. **Unique executable conclusion.** D2 uses origin `2018-06-30`, source history `2018-01-02..06-30`, train `06-01..06-15`, validation `06-16..06-30`, blind `07-01..12-27`, target `(1,10)`, 180 rows, sales-zero plus approved-covariate reconstruction for the 175/180 gap, and complete forecast PROMO exclusion.

### D3 — Rossmann

1. **Raw authority.** `train.csv`, `test.csv`, and `store.csv` under the frozen D3 path are the target, date/key, and metadata authorities.
2. **Main backtest authority.** D3 uses the unified 180-day backtest ending at the frozen origin.
3. **Extra benchmark/holdout role.** Raw test `2015-08-01..2015-09-17` is an additional holdout/date-key authority with a distinct run identity.
4. **Forecast origin.** `2015-02-01`.
5. **Origin authority.** Unified target split `2015-01-03..01-17` train and `2015-01-18..02-01` validation.
6. **Origin role.** Origin belongs only to validation; source history may end on origin as a separate role.
7. **Source history interval.** `2014-08-06..2015-02-01`, closed, 180 Gregorian days.
8. **Target train interval.** `2015-01-03..2015-01-17`, closed, 15 days.
9. **Validation interval.** `2015-01-18..2015-02-01`, closed, 15 days and excluded from target-train fitting.
10. **Blind interval.** `2015-02-02..2015-07-31`, closed, 180 Gregorian days.
11. **Target canonical keys.** `store_id=10`; the exact composite source key remains `(store_id,item_id,date)` where item identity is present.
12. **Source eligibility.** Without-sharing is Store 1–9; with-sharing is Store 1–30 excluding Store 10; source and target are sliced before repair; target exact key is excluded.
13. **KNN window and fields.** `2015-01-03..2015-02-01`, 30 days; field `sales`; choose Top-3 after approved history reconstruction.
14. **Expected cardinality.** `1×180=180` blind rows.
15. **Calendar/missing rules.** Slice source and target by entity, role, and window before calendarization; use Gregorian dates; exact keys are unique; no generic fill.
16. **Field-specific availability.** Historical Open missing with same-day sales greater than zero maps to 1 and with sales equal to zero maps to 0; SchoolHoliday comes from official D3 fields and missing maps to 0; historical Promo is retained and future Promo is unavailable.
17. **Worker-safe fields.** Sales history, deterministic dates, approved SchoolHoliday future-known field, and recursive forecast state; no Open, Customers, or Promo.
18. **Evaluator truth.** Blind Sales is evaluator-only; all target rows at or before origin are assigned to history/train/validation/audit and never to the forecast producer.
19. **Audit-only fields.** Open repair evidence and Customers may appear only in audit; source-domain region grouping and raw test comparison are proof material.
20. **Forbidden fields.** Open, Customers, forecast Promo, blind Sales truth, same-day actuals, and any post-origin source row are forbidden in worker/KNN/CNN/RFE/transfer views.
21. **Canonical digest objects.** Separate G14 digests bind source history, worker-safe blind, evaluator truth, and audit view; proof includes the pre-slice boundary.
22. **Fail-closed conditions.** Parent rows after origin in history, target rows at or before origin in forecast producer, invalid Open rule, missing SchoolHoliday authority, domain failure, duplicate key/date, or truth leakage fails closed.
23. **Unique executable conclusion.** D3 uses origin `2015-02-01`, history `2014-08-06..02-01`, train `01-03..01-17`, validation `01-18..02-01`, blind `02-02..07-31`, Store 10 target, 180 rows, source Store 1–9/Store 1–30 excluding 10, sales-only KNN, historical Open repair only, SchoolHoliday=0 for missing, and full pre-slice isolation.

### D4 — FreshRetailNet-LT

1. **Raw authority.** `data/train.parquet` is historical authority and `data/eval.parquet` is evaluation-date and approved-covariate authority.
2. **Main backtest authority.** D4 main evaluation is the unified 180-day backtest.
3. **Extra benchmark/holdout role.** Raw eval is an independent `5×7=35` holdout and future-covariate authority with a distinct run identity.
4. **Forecast origin.** `2025-01-14`.
5. **Origin authority.** Target split `2024-12-16..12-30` train and `2024-12-31..2025-01-14` validation.
6. **Origin role.** Origin is validation's final day and source cutoff in separate roles.
7. **Source history interval.** `2024-07-19..2025-01-14`, closed, 180 Gregorian days.
8. **Target train interval.** `2024-12-16..2024-12-30`, closed, 15 days.
9. **Validation interval.** `2024-12-31..2025-01-14`, closed, 15 days and excluded from target-train fitting.
10. **Blind interval.** `2025-01-15..2025-07-13`, closed, 180 Gregorian days.
11. **Target canonical keys.** `(166,258)`, `(166,432)`, `(166,433)`, `(166,313)`, `(166,311)` with key `(store_id,product_id)`.
12. **Source eligibility.** Without-sharing is Store 166 excluding target product IDs; with-sharing may cross stores under the frozen domain; candidates require complete history and safe schema; never calendarize all 22,934 source entities.
13. **KNN window and fields.** `2024-12-16..2025-01-14`, 30 days; field `sales`; choose Top-3 after approved calendarization.
14. **Expected cardinality.** `5×180=900` main blind rows; extra eval holdout remains `5×7=35`.
15. **Calendar/missing rules.** Gregorian dates, exact key/date uniqueness, slice before calendarization, and only approved entity/window field repairs; generic fill is prohibited.
16. **Field-specific availability.** The only approved future-known fields are `activity_flag`, `discount`, `holiday_flag`, `precpt`, `avg_temperature`, `avg_humidity`, and `avg_wind_level`, all benchmark-provided.
17. **Worker-safe fields.** Sales history, deterministic dates, the seven approved future covariates, and recursive forecast state.
18. **Evaluator truth.** Blind sales truth and raw eval labels are evaluator-only and cannot affect worker, KNN, CNN, RFE, or transfer inputs.
19. **Audit-only fields.** `hours_sale`, `hours_stock_status`, `stock_hour6_22_cnt`, and every hourly/stock aggregate are audit-only.
20. **Forbidden fields.** All hourly/stock fields and their aggregates are forbidden in worker, KNN, CNN, RFE, transfer, and forecast target views; unapproved covariates are also forbidden.
21. **Canonical digest objects.** Main source, worker-safe blind, evaluator truth, audit view, and separate raw-eval holdout each carry identity-bound canonical digests.
22. **Fail-closed conditions.** Any all-entity calendarization, unapproved future covariate, hourly/stock leakage, duplicate key/date, missing authority, row expansion, schema drift, or truth leakage fails closed.
23. **Unique executable conclusion.** D4 uses the origin and intervals above, five fixed targets and 900 main blind rows; without-sharing is Store 166, with-sharing follows the frozen cross-store domain, only seven benchmark future covariates reach worker, all hourly/stock fields remain audit-only, and raw eval is a separate 35-row holdout.

### D5 — Favorita

1. **Raw authority.** `train.csv`, `test.csv`, `items.csv`, `stores.csv`, `transactions.csv`, `oil.csv`, and `holidays_events.csv` are the frozen authorities.
2. **Main backtest authority.** D5 uses the unified 180-day backtest.
3. **Extra benchmark/holdout role.** Raw test `2017-08-16..2017-08-31` is an additional 16-day holdout with a distinct run identity; it does not replace the main window.
4. **Forecast origin.** `2017-02-15`.
5. **Origin authority.** Target split `2017-01-17..01-31` train and `2017-02-01..02-15` validation.
6. **Origin role.** Origin is validation's final day and source cutoff in separate roles.
7. **Source history interval.** `2016-08-20..2017-02-15`, closed, 180 Gregorian days.
8. **Target train interval.** `2017-01-17..2017-01-31`, closed, 15 days.
9. **Validation interval.** `2017-02-01..2017-02-15`, closed, 15 days and excluded from target-train fitting.
10. **Blind interval.** `2017-02-16..2017-08-14`, closed, 180 Gregorian days.
11. **Target canonical keys.** `48/364606`, `48/1159415`, `48/1159414`, `48/1349808`, `48/320682`, represented by `(store_nbr,item_nbr)`.
12. **Source eligibility.** Candidates are in the same item family; without-sharing is Store 48, with-sharing may cross stores; target exact keys are excluded and candidates require complete history and safe schema.
13. **KNN window and fields.** `2017-01-17..2017-02-15`, 30 days; fields `sales`, `onpromotion`, and `oil_price`; choose Top-3.
14. **Expected cardinality.** `5×180=900` blind rows; the observed 885 rows are closed by item 15 and target `48/1159415` remains included.
15. **Calendar/missing rules.** Calendarize all five targets only inside the blind window; for the 15 missing blind dates of `48/1159415`, set `sales=0` and rebuild approved covariates; duplicates and unauthorized gaps fail closed.
16. **Field-specific availability.** `onpromotion`: True/true/1 maps to 1; False/false/0/null maps to 0; any other encoding fails with `ONPROMOTION_ENCODING`. `oil_price` authority is `oil.csv::dcoilwtico`, a unique globally ascending date series, history-only forward fill followed by lag-one; no historical prior fails closed. `is_holiday` comes from date plus store city/state applicability in `holidays_events.csv`; merges may not expand rows. `week` is deleted.
17. **Worker-safe fields.** Sales, normalized onpromotion, lag-one oil price, is_holiday, approved deterministic dates, and recursive forecast state; transactions never enter worker/KNN/model.
18. **Evaluator truth.** Blind sales and any forecast-period actual transactions are evaluator/audit only; worker contains no actual blind sales or transactions.
19. **Audit-only fields.** Forecast-period transactions, raw encodings, repair masks, holiday lineage, and source/evaluation comparisons are audit-only.
20. **Forbidden fields.** Transactions in forecast, same-day actual transactions, transaction lag/rolling/external forecasts, future oil backfill/interpolation/mean fill, week, and unauthorized holiday merge outputs are forbidden.
21. **Canonical digest objects.** Published source, worker-safe blind, evaluator truth, and audit view each receive a G14 digest; proof binds oil prior and holiday row-cardinality decisions.
22. **Fail-closed conditions.** Unknown onpromotion encoding, no oil prior, non-unique oil dates, cross-entity shift, duplicate or expanding holiday merge, missing price/field authority, dropped target `48/1159415`, wrong cardinality, or truth leakage fails closed.
23. **Unique executable conclusion.** D5 keeps all five listed targets, including `48/1159415`; closes 885/900 to 900 by 15 calendarized dates with sales zero and approved-covariate reconstruction; uses same-family sources, the specified three KNN fields, strict onpromotion encoding, history-only lag-one oil, applicable holidays, no transactions in forecast/model, and no week.

### D6 — M5

1. **Raw authority.** `sales_train_validation.csv`, `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` are the frozen M5 authorities.
2. **Main backtest authority.** D6 uses the unified 180-day backtest through `2016-05-22`.
3. **Extra benchmark/holdout role.** M5 evaluation/calendar dates through `2016-06-19` are an independent official holdout/date authority with a distinct run identity.
4. **Forecast origin.** `2015-11-24`.
5. **Origin authority.** Target split `2015-10-26..11-09` train and `2015-11-10..11-24` validation.
6. **Origin role.** Origin belongs only to validation; source cutoff is separate.
7. **Source history interval.** `2015-05-29..2015-11-24`, closed, 180 Gregorian days.
8. **Target train interval.** `2015-10-26..2015-11-09`, closed, 15 days.
9. **Validation interval.** `2015-11-10..2015-11-24`, closed, 15 days and excluded from target-train fitting.
10. **Blind interval.** `2015-11-25..2016-05-22`, closed, 180 Gregorian days.
11. **Target canonical keys.** `CA_1/FOODS_3_586`, `CA_1/FOODS_3_080`, `CA_1/FOODS_3_555`, `CA_1/FOODS_3_377`, `CA_1/FOODS_3_668`, key `(store_id,item_id)`.
12. **Source eligibility.** Candidates are same-department; without-sharing is CA_1 and with-sharing may cross stores; target exact keys are excluded and candidates require complete history and safe schema.
13. **KNN window and fields.** `2015-10-26..2015-11-24`, 30 days; field `sales`; choose Top-3.
14. **Expected cardinality.** `5×180=900` main blind rows.
15. **Calendar/missing rules.** Calendarize only frozen entity/windows; Gregorian leap-day rules apply; exact key/date is unique; missing calendar or price key fails closed.
16. **Field-specific availability.** `weekday` and `wday` both remain; `weekday` is the original `calendar.csv` string/object and `wday` is the original `calendar.csv` integer. Order is `weekday`, `wday`, `wm_yr_wk`. Missing parent `wday` is rebuilt only by exact `calendar.csv` join. `sell_price` joins exactly on `(store_id,item_id,wm_yr_wk)` from `sell_prices.csv`. `CA_1` uses only `snap_CA`, normalized to `snap`.
17. **Worker-safe fields.** Approved calendar fields, state-specific `snap`, exact future-known sell price, historical sales, and recursive forecast state; evaluation sales are excluded.
18. **Evaluator truth.** `sales_train_evaluation.csv` truth is restricted to evaluator truth and audit after forecast.
19. **Audit-only fields.** Evaluation truth lineage, excluded calendar columns, join proof, and raw benchmark comparisons are audit-only.
20. **Forbidden fields.** Real evaluation sales, all three SNAP state columns together, guessed weekday aliases, non-exact prices, cross-key fills, duplicate calendar rows, and future truth are forbidden in worker-safe views.
21. **Canonical digest objects.** Published source, worker-safe blind, evaluator truth, and audit view each receive a G14 digest; proof includes calendar and price join identities.
22. **Fail-closed conditions.** Missing or duplicate exact calendar/price keys, missing price, state mapping failure, weekday/wday dtype or order drift, evaluation truth in worker, or non-isolated holdout identity fails closed.
23. **Unique executable conclusion.** D6 uses origin `2015-11-24`, history `2015-05-29..11-24`, train `10-26..11-09`, validation `11-10..11-24`, blind `11-25..2016-05-22`, five fixed CA_1 targets and 900 rows; it retains both calendar weekday and wday in the fixed order, uses exact calendar and sell-price joins, maps `snap_CA` to `snap`, and isolates evaluation truth.

## 6. Freeze transition and next-stage constraint

This contract is re-frozen only when its bytes, sidecar, scope, matrix, re-freeze record, and freeze tests agree on identity and completeness. The re-freeze commit is the only valid Gate 1R commit. It authorizes a later one-time Gate 1X Implementation; it does not authorize implementation, readiness preflight, controlled rerun, publication, training, or deployment now.

The next Gate 1X Implementation must bind the final decision-book SHA-256, this contract digest, the scope SHA-256, the matrix SHA-256, and the Gate 1R freeze commit SHA. Gate 1X remains prohibited until that one-time implementation and its real-input readiness preflight are complete.
