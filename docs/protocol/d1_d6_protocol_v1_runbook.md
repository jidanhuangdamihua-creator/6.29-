# D1–D6 `d1_d6_protocol_v1` 运行与归档手册

本手册用于生成可被标记为 `confirmed_baseline` 的正式结果。代码改造和轻量测试不等于 baseline 已生成；正式训练、D1/D2 数据再生成均由用户在 Terminal 中执行。

## 1. 当前状态与前置条件

- D1–D3 属于 `strict_paper`；D4–D6 属于 `extended`。
- KNN 只使用从 `knn_observed_start` 开始的连续30个日历日销量序列。
- 30日 KNN 序列只用于选源；CNN 与 BL1–BL4 的评估输入窗口统一为10日，并逐日期消费同一 `sample_manifest`。两者角色不同，不得混成同一特征表示。
- 当前固化 D1-with 缺少 Store2–3 × Item1–9。
- 当前固化 D2-with 缺少 Brand2–3 × Item1–9。
- 因此，在 D1/D2 完成数据再生成前，with-sharing preflight 返回失败是正确行为。
- 工作区当前没有 D2 原始数据文件。不得用不完整的 `dataset2-source.parquet` 反向补造 Brand2/3；必须提供覆盖 Brand1–3 × Item1–10 的原始表。

## 2. 数据再生成

先备份现有 `数据集/固化数据`。D1 可从仓库原始 CSV 生成：

```bash
python scripts/regenerate_d1_d2_parquets.py \
  --dataset d1 \
  --d1-input "数据集/原始数据/Dataset 1/train.csv" \
  --output-dir "数据集/固化数据"
```

D2 需要显式提供完整原始表：

```bash
python scripts/regenerate_d1_d2_parquets.py \
  --dataset d2 \
  --d2-input "/absolute/path/to/complete_dataset2_raw.csv" \
  --output-dir "数据集/固化数据"
```

D2 输入至少包含可映射到 `date, brand_id, item_id, sales` 的字段；`promo` 缺失时生成器只允许将其作为已声明的零值协变量，不影响 KNN，因为 KNN 仅使用 `sales`。

生成器应得到：

- D1 source：Store1–3 × Item1–9，共27个 source key；target：Store1 × Item10。
- D2 source：Brand1–3 × Item1–9，共27个 source key；target：Brand1 × Item10。
- D3 source：Store1–30 排除 Store10，共29个 source key；target：Store10。

## 3. 只读 preflight

每个正式任务先执行只读 preflight。它不训练、不改写 parquet：

```bash
python scripts/validate_d1_d6_protocol_inputs.py --dataset d1 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d1 --scenario with --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d2 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d2 --scenario with --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d3 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d3 --scenario with --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d4 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d4 --scenario with --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d5 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d5 --scenario with --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d6 --scenario without --k 3
python scripts/validate_d1_d6_protocol_inputs.py --dataset d6 --scenario with --k 3
```

每个目标必须返回 `status=passed`，并包含 `candidate_pool_digest`、`selection_result_digest`、`knn_observed_start` 和 `knn_observed_end`。失败时不要缩小 K、补零候选日或启用旧统计签名。

preflight 内部仍使用完整规范化 `candidate_pool_digest_input` 计算 digest；默认控制台仅输出 `candidate_pool_digest_input_summary`，其中 candidate key 最多展示20项。候选排除审计使用 `candidate_exclusion_count`、`candidate_exclusion_reason_counts`、最多20项的 `candidate_exclusion_samples` 和 `candidate_exclusions_truncated`。ordered Top-K、distance、weight、tie group 和 CNN provenance 状态继续输出。完整内部排除信息和 digest 输入不因控制台截断而改变。

## 4. 正式迁移学习矩阵

以下入口为每个任务创建25个隔离 cell：horizon 1–5 × seed 42–46。cell 全部成功后才合并，并由代码判断哪些方法组可晋升为 `confirmed_baseline`。

先用 `--dry-run` 检查命令，不训练：

```bash
python scripts/run_strict_protocol_baseline.py \
  --dataset d1 --scenario without \
  --output-dir "outputs/protocol_v1/d1_without" \
  --dry-run
```

正式运行示例：

```bash
python scripts/run_strict_protocol_baseline.py \
  --dataset d1 --scenario without \
  --output-dir "outputs/protocol_v1/d1_without"
```

对 `d1` 至 `d6` 和 `without/with` 分别执行。输出位于：

```text
outputs/protocol_v1/<dataset>_<scenario>/results/dataset<id>_<scenario>_results.csv
```

## 5. BL1–BL4 target-only baseline

BL1–BL4 使用与 CNN 相同的 horizon、首个有效预测原点、label identity 和 `sample_manifest_digest`。每个数据集执行：

```bash
python scripts/baselines/run_baselines_multiseed.py --dataset d1 --seeds 42,43,44,45,46
```

依次替换为 `d2` 至 `d6`。确定性方法 BL1–BL3 也输出五个 seed 行，以保证结果粒度统一；BL3 只使用历史 lag 和提前可知的日历字段。

## 6. 结果完整性检查

每个严格结果行必须包含：

```text
protocol_track
protocol_version
knn_observed_start
knn_observed_end
knn_representation
target_test_excluded
source_future_excluded
candidate_pool_digest
selection_result_digest
horizon
seed
primary_metric_space
sample_manifest_digest
```

还应保留 `source_observation_cutoff`、`candidate_pool_digest_input`、ordered selected sources、distance、weight 和 tie group。

确认以下条件：

1. `protocol_version == d1_d6_protocol_v1`。
2. `primary_metric_space == original_sales`。
3. 每个 `(dataset, track, scenario, target, method)` 具有25个不重复的 `(horizon, seed)`。
4. seed 集合严格为 `{42,43,44,45,46}`，horizon 集合严格为 `{1,2,3,4,5}`。
5. 同一方法组 `sample_manifest_digest` 一致。
6. 同一选源任务在五个 seed 下 `candidate_pool_digest` 和 `selection_result_digest` 一致。
7. `target_test_excluded` 与 `source_future_excluded` 均为真。
8. RMSE、MAE、sMAPE、Accuracy 均为原始销量尺度的有限数值。
9. `result_status` 只有通过完整性门禁后才为 `confirmed_baseline`。
10. TL 方法的 `failed_source_count == 0`、`effective_k == requested_k` 且 `cnn_provenance_validated == true`；任何不满足条件的矩阵整体失败。

## 7. 汇总与归档

聚合器保留旧结果用于追踪，但指标汇总和最佳方法只读取 `confirmed_baseline`：

```bash
python scripts/aggregate_d1_d6_results.py \
  --run-dir "outputs/protocol_v1" \
  --output "outputs/final_summary/d1_d6_protocol_v1_all_results.csv" \
  --strict
```

归档时同时保存：

- 代码提交哈希与 `d1_d6_protocol_v1` 设计文档。
- 原始数据来源说明及固化 parquet 文件摘要。
- preflight 控制台输出。
- 每个 cell 的运行配置和结果。
- 合并结果、候选池 digest 输入、selection digest 与 sample manifest digest。
- Python、NumPy、TensorFlow、PyTorch 和硬件环境信息。

缺少新协议字段的旧 CSV 一律保持 `legacy_unverified`，不能与正式 baseline 混合，也不能用于后续准确率提升百分比的分母。
