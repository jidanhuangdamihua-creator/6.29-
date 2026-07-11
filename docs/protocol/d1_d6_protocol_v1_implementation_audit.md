# D1–D6 `d1_d6_protocol_v1` 实现审计

审计日期：2026-07-11
设计依据：`docs/superpowers/specs/2026-07-11-d1-d6-shared-experiment-protocol-design.md`

## 结论边界

本次审计确认的是严格协议代码、静态数据门禁、轻量单元测试和正式矩阵入口已经接通。Codex 没有重新生成 D1/D2 数据，也没有运行 D1–D6 模型训练。因此：

- 当前旧结果只能保留为 `legacy_unverified`，不能作为准确率提升的分母。
- 当前固化 D1-with 和 D2-with 候选池不完整，不能生成 `confirmed_baseline`。
- 用户补齐 D2 原始数据、重新生成 D1/D2、通过全部 preflight 并完成 5 horizon × 5 seed 正式矩阵后，完整性门禁通过的结果才可归档为 `confirmed_baseline`。

## 逐节证据

| 设计节 | 实现证据 | 测试证据 | 审计结果 |
|---|---|---|---|
| 4 双轨实验定义 | `src/protocols/experiment_protocol.py`；`src/protocols/runner_adapter.py` | `test_experiment_protocol_contract`；`test_runner_protocol_integration`；`test_protocol_preflight` | D1–D3 strict-paper 与 D4–D6 extended 的 target、with/without 候选规则固化；完整规范 key 排除目标；缺失或重复候选失败。 |
| 5 唯一截止日 | `experiment_protocol.py`；`candidate_pool.py`；`runner_adapter.py`；`entity_experiment.py` | `test_daily_knn_protocol`；`test_source_selector_shared_protocol`；`test_protocol_preflight` | KNN 使用明确的连续 30 日窗口；源切片在进入训练前按 `source_observation_cutoff` 截断；运行期不从文件最大日期推断。 |
| 6 KNN 选源 | `candidate_pool.py`；`src/utils/source_selector.py` | `test_daily_knn_protocol`；`test_candidate_pool_digest`；`test_source_selector_shared_protocol` | 仅使用 30 维逐日销量；任务内合法窗口 MinMax；float64 欧氏距离；锚定 tie `1e-12`；逆距离权重 `1e-8`；不自动缩小 K，不允许正式任务回退统计签名。 |
| 7 生产摘要 | `candidate_pool.py` | `test_candidate_pool_digest`；`test_daily_knn_protocol` | 候选池和选择结果使用唯一生产 digest 机制；候选键规范排序；字段顺序保留；浮点稳定序列化；with/without 金标独立。 |
| 8 KNN→CNN 溯源 | `src/protocols/provenance.py`；`src/data_processing/data_preprocessing.py`；`runner_adapter.py`；`source_selector.py`；各 TL 方法 | `test_knn_cnn_provenance`；`test_runner_protocol_integration`；`test_source_selector_shared_protocol` | 生产 selector 先校验入选源原始30日切片；实际源训练再于 `normalize_features → build_tabular_sequence` 路径用训练段 scaler 从原始分区重建并逐元素比对真正送入 CNN 的 X/y、日期、特征和标签。CNN 按 KNN 有序规范 key 查找源；严格运行禁止跳过失败源或重归一化剩余源。 |
| 9 CNN 时间切分 | `runner_adapter.py`；`entity_experiment.py`；`run_full_paper_experiments.py`；`run_d4_experiment.py` 至 `run_d6_experiment.py` | `test_runner_protocol_integration`；`test_d4_d6_source_authority` | 训练帧先按 cutoff 截断；沿用时间顺序切分；选源 key 与后续学习源一致。真实训练未在本次审计中执行。 |
| 10 滚动预测与公平评估 | `src/protocols/rolling_origin.py`；`src/data_processing/data_preprocessing.py`；`scripts/baselines/baseline_data_loader.py`；各 runner | `test_rolling_origin_protocol`；`test_baseline_protocol` | 1–5 horizon 共用不可变 manifest；CNN 构造每个 test tensor 时逐项核对 manifest 的 input dates 与 label date；基线消费同一10日模型输入窗口并使用特征可用性 allowlist。KNN 的30日销量序列仍仅用于选源。 |
| 11 指标、种子与汇总 | `src/protocols/reproducibility.py`；`scripts/run_strict_protocol_baseline.py`；`run_baselines_multiseed.py`；`result_schema.py` | `test_baseline_protocol`；`test_formal_protocol_matrix`；`test_strict_result_contract` | seeds 固定 42–46，horizons 固定 1–5；主要指标为空间为原始销量；结果最小粒度包含 dataset/track/scenario/target/method/horizon/seed；只有完整 25-cell 组可晋升。 |
| 12 输出、兼容与归档 | `src/constants.py`；`src/utils/result_schema.py`；`src/utils/result_validation.py`；`scripts/aggregate_d1_d6_results.py`；各 runner | `test_strict_result_contract`；`test_formal_protocol_matrix`；`test_unified_d1_d6_output_contract` | 强制协议字段已纳入结果；晋升门禁重算 candidate/selection digest，逐项比对 ordered Top-K 与 provenance keys，并校验原始指标空间、零失败源、K 未缩小、sample count 与25-cell完整性；任何未确认组使正式矩阵失败。 |
| 13 严格测试契约 | `tests/test_candidate_pool_digest.py`；`test_daily_knn_protocol.py`；`test_knn_cnn_provenance.py`；`test_rolling_origin_protocol.py` 等 | 2026-07-11 标准库 `unittest` 轻量套件 74/74 通过 | 覆盖未来扰动不变、观察期确定性翻转、锚定 tie、有序 Top-K、digest 金标、实际归一化 CNN X/y provenance、manifest 输入日期一致、RFE seed 传递、候选失败与公平评估。已有 pytest 风格兼容测试已更新并通过语法编译，但当前环境未安装 pytest，未执行其 pytest runner。 |
| 14 实施与验证边界 | 保护包装器；本实现计划与运行手册 | 所有 Python 验证均经 `codex_timeout.py --timeout 180`；`compileall` 通过 | 未执行正式再生成或训练；没有出现退出码 124。 |
| 15 验收标准 | 上述生产代码、测试、preflight、矩阵入口与运行手册 | 74 项协议测试 + 静态编译 | 代码改造达到交付用户正式重跑的条件；baseline 数据产物尚未达到确认条件。 |

## 当前数据门禁证据

只读 preflight 对现有固化数据给出预期失败：

- D1-with：缺少 Store2–3 × Item1–9。
- D2-with：缺少 Brand2–3 × Item1–9。
- 工作区不存在可证明覆盖 Brand1–3 × Item1–10 的 D2 原始文件；再生成脚本要求显式传入完整 `--d2-input`，不会从不完整 parquet 反向补造。

这些失败是协议门禁在工作，不是允许缩小候选池或 K 的理由。

## 本次验证命令

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest \
  tests.test_experiment_protocol_contract \
  tests.test_candidate_pool_digest \
  tests.test_daily_knn_protocol \
  tests.test_knn_cnn_provenance \
  tests.test_rolling_origin_protocol \
  tests.test_baseline_protocol \
  tests.test_source_selector_shared_protocol \
  tests.test_runner_protocol_integration \
  tests.test_strict_result_contract \
  tests.test_protocol_preprocessing_contract \
  tests.test_protocol_preflight \
  tests.test_formal_protocol_matrix \
  tests.test_unified_d1_d6_output_contract -v
```

结果：`Ran 74 tests ... OK`。

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m compileall -q src scripts tests
```

结果：退出码 0。

## 归档判定

可以归档当前分支和旧产物，名称应明确为“协议改造代码 + 改造前快照”，不得写成“已确认 baseline”。正式 baseline 的生成和归档步骤以 `docs/protocol/d1_d6_protocol_v1_runbook.md` 为准。
