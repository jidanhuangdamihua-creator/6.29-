# Unified Strict Metric Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make D1-D3 and D4-D6 share one fail-closed original-space sMAPE contract, validate the exact formal seed matrix, derive winners from seed means, and emit every Friedman stratum.

**Architecture:** Centralize manifest identity construction in the metric contract and send that identity through every method wrapper. Validate immutable seed rows before deriving target, dataset, and cross-dataset summaries. Return statistical results as dimension-preserving DataFrames.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, pytest.

## Global Constraints

- Formal sMAPE is canonical original-sales-space sMAPE; RMSE is diagnostic only.
- Formal seeds are exactly `FORMAL_SEEDS = (42, 43, 44, 45, 46)`.
- Missing, extra, or duplicate formal seed rows fail fast.
- D1-D3 and D4-D6 use the same identity fields and validator.
- Every Python or pytest command runs through `python tools/protection/codex_timeout.py`.
- Do not run experiments, training, validation jobs, data regeneration, or the full matrix.

---

### Task 1: Centralize Metric Identity and Close Formal Eligibility

**Files:**
- Modify: `src/evaluation/metric_contract.py`
- Modify: `src/utils/entity_experiment.py`
- Test: `tests/test_smape_metric_contract.py`
- Test: `tests/test_multi_source_metric_payload_contract.py`

**Interfaces:**
- Produces: `build_metric_identity_from_manifest(manifest: Any, *, horizon: int) -> dict[str, Any]`.
- Requires all members of `METRIC_IDENTITY_FIELDS` for formal eligibility.

- [ ] **Step 1: Write failing tests**

Add a parametrized test that removes each identity field and expects `missing:<field>`. Add manifest tests for a valid identity, no samples, and multiple target keys.

```python
@pytest.mark.parametrize("field", METRIC_IDENTITY_FIELDS)
def test_formal_smape_rejects_missing_metric_identity_field(field):
    row = _strict_row()
    row.pop(field)
    decision = is_formally_comparable_smape_row(row)
    assert decision["eligible"] is False
    assert f"missing:{field}" in decision["failure_reasons"]
```

- [ ] **Step 2: Verify RED**

```bash
python tools/protection/codex_timeout.py pytest -q tests/test_smape_metric_contract.py tests/test_multi_source_metric_payload_contract.py
```

Expected: the shared helper is missing and identity omissions remain eligible.

- [ ] **Step 3: Implement the shared helper**

```python
def build_metric_identity_from_manifest(manifest: Any, *, horizon: int) -> dict[str, Any]:
    records = tuple(manifest.for_horizon(int(horizon)))
    if not records:
        raise MetricProtocolError("metric_identity_mismatch", detail=f"manifest has no samples for horizon={horizon}")
    target_keys = {tuple(record.target_key) for record in records}
    if len(target_keys) != 1:
        raise MetricProtocolError("metric_identity_mismatch", detail=f"manifest has multiple target keys: {sorted(target_keys)}")
    label_dates = [str(record.label_date) for record in records]
    return {
        "metric_target_key": "/".join(str(value) for value in next(iter(target_keys))),
        "metric_horizon": int(horizon),
        "metric_sample_count": len(records),
        "metric_date_start": label_dates[0],
        "metric_date_end": label_dates[-1],
        "metric_index_digest": compute_metric_index_digest([record.sample_key for record in records]),
    }
```

Add all identity fields to formal required fields, validate positive horizon/count, and replace the entity-local implementation with the shared helper.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command, then:

```bash
git add src/evaluation/metric_contract.py src/utils/entity_experiment.py tests/test_smape_metric_contract.py tests/test_multi_source_metric_payload_contract.py
git commit -m "fix: centralize strict metric identity contract"
```

---

### Task 2: Route No-TL and SS-TL Through Strict Extraction

**Files:**
- Modify: `src/experiment/run_no_tl_experiment.py`
- Modify: `src/experiment/experiment_runner.py`
- Modify: `src/utils/entity_experiment.py`
- Test: `tests/test_multi_source_metric_payload_contract.py`
- Test: `tests/test_experiment_runner_parameter_forwarding.py`
- Test: `tests/test_metric_protocol_and_diagnostics.py`

**Interfaces:**
- Adds `expected_metric_identity` to No-TL and SS-TL signatures.
- No-TL bottom payload includes `y_true`, `y_pred`, scaler, feature columns, and identity.

- [ ] **Step 1: Write failing No-TL/SS-TL tests**

Assert both wrappers receive the expected identity, reject mismatches, return original-space audit fields, and produce a valid D4-D6 entity row rather than an error row.

```python
assert result["paper_metric_computed_valid"] is True
assert result["smape_metric_space"] == "original_sales_space"
assert {field: result[field] for field in IDENTITY} == IDENTITY
```

- [ ] **Step 2: Verify RED**

```bash
python tools/protection/codex_timeout.py pytest -q tests/test_multi_source_metric_payload_contract.py tests/test_experiment_runner_parameter_forwarding.py tests/test_metric_protocol_and_diagnostics.py
```

Expected: No-TL/SS-TL do not accept or preserve expected identity.

- [ ] **Step 3: Preserve the complete No-TL payload**

Return this shape from the bottom runner:

```python
return {
    "method": "No-TL",
    "y_true": np.asarray(y_test).reshape(-1),
    "y_pred": np.asarray(y_pred).reshape(-1),
    "sales_scaler": tgt_scaler,
    "feature_columns": list(tgt_feature_columns),
    "prediction_shape": tuple(y_pred.shape),
    **metric_result,
    **dict(expected_metric_identity or {}),
}
```

Make the public wrapper call `_extract_method_metrics(raw, method_name="No-TL", metric_protocol=metric_protocol, expected_metric_identity=expected_metric_identity)`.

- [ ] **Step 4: Add identity to SS-TL and entity dispatch**

Pass identity to SS-TL's strict extraction. In `entity_experiment`, pass `expected_metric_identity` to No-TL, SS-TL, and all multi-source methods; keep only `number_of_sources` conditional.

- [ ] **Step 5: Verify GREEN and commit**

Run the Step 2 command, then:

```bash
git add src/experiment/run_no_tl_experiment.py src/experiment/experiment_runner.py src/utils/entity_experiment.py tests/test_multi_source_metric_payload_contract.py tests/test_experiment_runner_parameter_forwarding.py tests/test_metric_protocol_and_diagnostics.py
git commit -m "fix: unify strict extraction for single-source baselines"
```

---

### Task 3: Complete D1-D3 Strict Contract Integration

**Files:**
- Modify: `scripts/run_full_paper_experiments.py`
- Test: `tests/test_multi_source_metric_payload_contract.py`
- Test: `tests/test_metric_protocol_and_diagnostics.py`

**Interfaces:**
- Consumes the shared manifest identity helper.
- Produces D1-D3 rows containing every metric audit and identity field.

- [ ] **Step 1: Write a failing orchestration test**

Patch all six method runners, capture kwargs, return strict payloads, and assert every runner receives the same manifest-derived identity. Assert the serialized row passes `is_formally_comparable_smape_row`.

- [ ] **Step 2: Verify RED**

```bash
python tools/protection/codex_timeout.py pytest -q tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py
```

Expected: D1-D3 omits identity forwarding and strict audit serialization.

- [ ] **Step 3: Build, forward, and serialize identity**

```python
expected_metric_identity = build_metric_identity_from_manifest(
    protocol_manifest,
    horizon=int(exp_cfg["horizon"]),
)
```

Pass it to every method. Copy `_metric_audit_values(raw)` and all identity fields into the D1-D3 result before schema alignment. Do not synthesize missing strict values.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command, then:

```bash
git add scripts/run_full_paper_experiments.py tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py
git commit -m "fix: connect d1 d3 to strict metric identity"
```

---

### Task 4: Validate Seeds and Derive Mean-Based Winners

**Files:**
- Modify: `src/evaluation/metric_contract.py`
- Modify: `scripts/aggregate_d1_d6_results.py`
- Modify: `src/analysis/statistical_tests.py`
- Test: `tests/test_aggregate_d1_d6_results.py`
- Test: `tests/test_formal_smape_statistics.py`

**Interfaces:**
- Changes `build_formal_smape_aggregates(frame, *, expected_seeds: Sequence[int] | None = None)`.
- Produces a seed-detail CSV and a mean-based best-method CSV.

- [ ] **Step 1: Write failing seed matrix tests**

Create valid rows for seeds 42-46, then separately remove, add, and duplicate a seed. Each mutation must raise `MetricProtocolError`. Add exact-value tests for two targets and two datasets covering seed mean, dataset macro, and cross-dataset macro.

- [ ] **Step 2: Write a failing lucky-seed winner test**

Method A has one winning seed but a worse five-seed mean than Method B. Assert every input row appears in seed detail, while `best_method_by_target` selects Method B and reports two candidate methods.

- [ ] **Step 3: Verify RED**

```bash
python tools/protection/codex_timeout.py pytest -q tests/test_aggregate_d1_d6_results.py tests/test_formal_smape_statistics.py
```

Expected: invalid seed matrices are accepted and Method A wins from one row.

- [ ] **Step 4: Implement strict validation**

Canonicalize sharing aliases, require dataset/target/method/horizon/scenario/seed, reject invalid numerics, and reject duplicate full keys:

```python
full_key = ["dataset", "target", "method", "horizon", "sharing_scenario", "seed"]
if work.duplicated(full_key, keep=False).any():
    raise MetricProtocolError("duplicate_formal_seed_row")
```

For every group excluding seed, compare actual seeds with `expected_seeds`; raise `formal_seed_set_mismatch` with missing and unexpected seeds in the detail.

- [ ] **Step 5: Implement detail and mean-winner outputs**

Rank every validated row within dataset/target/horizon/scenario/seed and write it. Select target winners from `aggregates["seed_mean"]`, not raw rows. Pass `FORMAL_SEEDS` from production aggregation and statistical callers. Never rank by RMSE.

- [ ] **Step 6: Verify GREEN and commit**

Run the Step 3 command, then:

```bash
git add src/evaluation/metric_contract.py scripts/aggregate_d1_d6_results.py src/analysis/statistical_tests.py tests/test_aggregate_d1_d6_results.py tests/test_formal_smape_statistics.py
git commit -m "fix: validate formal seeds before smape aggregation"
```

---

### Task 5: Preserve Every Friedman Stratum

**Files:**
- Modify: `src/analysis/statistical_tests.py`
- Modify: `scripts/run_statistical_analysis.py`
- Test: `tests/test_formal_smape_statistics.py`

**Interfaces:**
- Changes `run_friedman_test(results_dataframe: pd.DataFrame) -> pd.DataFrame`.
- Returns `horizon`, `sharing_scenario`, `n_datasets`, `n_methods`, `statistic`, `p_value`, and `status`.

- [ ] **Step 1: Write a failing multi-stratum test**

Build four horizon/scenario groups, one of them insufficient. Assert four output rows, unique dimension keys, and `insufficient_data` for the undersized group.

- [ ] **Step 2: Verify RED**

```bash
python tools/protection/codex_timeout.py pytest -q tests/test_formal_smape_statistics.py
```

Expected: the current function returns one dict and drops later strata.

- [ ] **Step 3: Return and write the full DataFrame**

Append one record per group. Valid groups get `status="ok"`; groups with fewer than two complete datasets or three methods get NaN statistics and `status="insufficient_data"`. Sort by horizon/scenario. In `run_statistical_analysis.py`, replace `pd.DataFrame([friedman])` with the returned DataFrame directly.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command, then:

```bash
git add src/analysis/statistical_tests.py scripts/run_statistical_analysis.py tests/test_formal_smape_statistics.py
git commit -m "fix: preserve stratified friedman results"
```

---

### Task 6: Focused Verification

**Files:**
- Verify only.

**Interfaces:**
- Confirms the unified contract without running experiments.

- [ ] **Step 1: Run the focused suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/protection/codex_timeout.py pytest -q tests/test_smape_metric_contract.py tests/test_multi_source_metric_payload_contract.py tests/test_experiment_runner_parameter_forwarding.py tests/test_metric_protocol_and_diagnostics.py tests/test_aggregate_d1_d6_results.py tests/test_formal_smape_statistics.py tests/test_d5_source_nan_repair.py
```

Expected: exit 0 with no failures. If the wrapper returns 124, stop immediately and provide this exact command for manual execution.

- [ ] **Step 2: Run static checks**

```bash
git diff --check
python tools/protection/codex_timeout.py python -m compileall -q src/evaluation/metric_contract.py src/experiment/experiment_runner.py src/experiment/run_no_tl_experiment.py src/utils/entity_experiment.py src/analysis/statistical_tests.py scripts/run_full_paper_experiments.py scripts/aggregate_d1_d6_results.py scripts/run_statistical_analysis.py
```

Expected: both commands exit 0. If the wrapper returns 124, stop without retrying.

- [ ] **Step 3: Review repository state**

```bash
git status --short
git diff --stat
git log -8 --oneline --decorate
```

Expected: only planned source, test, spec, and plan files changed; no outputs, data, parquet, JSON, or result CSV files.
