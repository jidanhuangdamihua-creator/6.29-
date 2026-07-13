# Strict Original-Sales sMAPE Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce fail-closed original-sales-space sMAPE computation for strict runs and make that contract the only basis for cross-dataset and cross-method ranking.

**Architecture:** A focused `metric_contract` module owns protocol constants, typed failures, row eligibility, and identity helpers. Metric computation derives all audit fields from one resolved state; experiment boundaries validate independent orchestration manifests and serialize invalid method rows without stopping a batch. Aggregation, visualization, and statistics consume one shared eligibility function and keep horizon and sharing scenario separate.

**Tech Stack:** Python 3, NumPy, pandas, SciPy, pytest, existing D1-D6 experiment runner and result schema.

## Global Constraints

- Formal ranking uses sMAPE in `original_sales_space`, lower is better; RMSE is diagnostic only.
- Formula identity is `smape_2abs_eps1e-8_pct_v1`: `100 * mean(2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred) + 1e-8))`, with range `[0, 200]` percent.
- Contract version is `smape_original_v1`; formal target policy is `clip_negative_to_zero_v1` in preprocessing, while the strict metric layer rejects negative targets and never clips predictions.
- Do not add `run_profile`, paper/debug configs, or paper/archive eligibility flags.
- Do not change training, source pools, KNN, Top-K, windows, hyperparameters, datasets, parquet files, or `parallel_mode_runner.sh`.
- Do not run a full matrix or full experiment; verification is limited to focused pytest suites.
- All Python commands use `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180` and execution stops on timeout exit code 124.

---

### Task 1: Canonical Metric Contract and Strict Computation

**Files:**
- Create: `src/evaluation/metric_contract.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `src/constants.py`
- Test: `tests/test_smape_metric_contract.py`
- Test: `tests/test_metrics_normalized_only.py`

**Interfaces:**
- Consumes: `compute_metrics_with_protocol(y_true, y_pred, scaler, feature_columns, metric_protocol)` and existing result dictionaries.
- Produces: `MetricProtocolError(ValueError)`, `SMAPE_CONTRACT_FIELDS`, `is_formally_comparable_smape_row(row) -> dict[str, object]`, `compute_metric_index_digest(values) -> str`, and complete metric audit fields from `compute_metrics_with_protocol`.

- [ ] **Step 1: Write strict computation and eligibility tests**

```python
def test_strict_missing_scaler_raises_typed_error():
    with pytest.raises(MetricProtocolError, match="missing_scaler"):
        compute_metrics_with_protocol(y_true, y_pred, None, ["sales"], strict_protocol)

def test_not_required_inverse_status_is_formally_comparable():
    row = valid_contract_row(inverse_transform_status="not_required", inverse_transform_applied=False)
    assert is_formally_comparable_smape_row(row)["eligible"] is True

def test_legacy_row_without_contract_fields_is_excluded_without_key_error():
    result = is_formally_comparable_smape_row({"smape": 12.0})
    assert result["eligible"] is False
    assert "missing:metric_contract_version" in result["failure_reasons"]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_smape_metric_contract.py tests/test_metrics_normalized_only.py -q`

Expected: FAIL because `src.evaluation.metric_contract` and the new strict fields do not exist.

- [ ] **Step 3: Implement the contract module and derive audit fields from resolved state**

```python
class MetricProtocolError(ValueError):
    def __init__(self, status: str, missing_fields=(), detail: str = ""):
        self.status = status
        self.missing_fields = tuple(missing_fields)
        message = f"metric protocol error: {status}"
        if self.missing_fields:
            message += f"; missing_fields={','.join(self.missing_fields)}"
        if detail:
            message += f"; {detail}"
        super().__init__(message)

def is_formally_comparable_smape_row(row):
    reasons = validate_required_contract_values(row)
    return {"eligible": not reasons, "failure_reasons": reasons}
```

Resolve `current_metric_space_actual`, paper requested/actual space, primary space, inverse status, validity, aliases, formula metadata, sample counts, and zero/negative audit values once. In strict mode, missing inputs, missing `sales`, unequal lengths, non-finite inputs, inverse failures, negative targets, or invalid sMAPE raise `MetricProtocolError`; no finite primary `rmse` or `smape` is returned on strict failure. In non-strict mode, retain current-space metrics with explicit `normalized` or `current_input_space` labels and never put a normalized value into a paper/original alias.

- [ ] **Step 4: Run focused metric tests and verify GREEN**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_smape_metric_contract.py tests/test_metrics_normalized_only.py tests/test_multi_source_smape_metrics.py -q`

Expected: PASS with no timeout.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/evaluation/metric_contract.py src/evaluation/metrics.py src/constants.py tests/test_smape_metric_contract.py tests/test_metrics_normalized_only.py
git commit -m "fix: enforce strict original-scale smape contract"
```

### Task 2: Strict Extraction, Multi-Source Payload Identity, and Serialization

**Files:**
- Modify: `src/experiment/experiment_runner.py`
- Modify: `src/transfer_methods/mswa_tl.py`
- Modify: `src/transfer_methods/mssb_tl.py`
- Modify: `src/transfer_methods/msml_tl.py`
- Modify: `src/transfer_methods/msml_tl_rfe.py`
- Modify: `src/utils/entity_experiment.py`
- Modify: `src/utils/result_schema.py`
- Test: `tests/test_metric_protocol_and_diagnostics.py`
- Test: `tests/test_strict_result_contract.py`
- Test: `tests/test_unified_d1_d6_output_contract.py`
- Test: `tests/test_multi_source_metric_payload_contract.py`

**Interfaces:**
- Consumes: Task 1 `MetricProtocolError`, contract fields, `compute_metric_index_digest`, and `compute_metrics_with_protocol`.
- Produces: `_extract_method_metrics(..., expected_metric_identity=None)` that cannot bypass strict recomputation; all four multi-source methods return values plus target/horizon/sample/date/index identity; entity boundaries serialize strict failures as invalid rows and continue.

- [ ] **Step 1: Add failing extraction, payload, and serialization tests**

```python
@pytest.mark.parametrize("method", ["MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"])
def test_multisource_strict_result_cannot_passthrough_internal_metrics(method):
    raw = payload_for(method, rmse=0.01, smape=0.01)
    result = _extract_method_metrics(raw, method, strict_protocol, expected_identity)
    assert result["smape"] == result["smape_paper"]
    assert result["smape_metric_space"] == "original_sales_space"

def test_d4_d6_row_preserves_computed_paper_metrics_and_alignment_semantics():
    row = _row_from_result(valid_strict_result, "MSWA-TL", config)
    assert row["smape_paper"] == valid_strict_result["smape_paper"]
    assert row["original_scale_smape"] == valid_strict_result["original_scale_smape"]
    assert row["paper_metric_aligned"] != "no_paper_reference"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_strict_result_contract.py tests/test_unified_d1_d6_output_contract.py -q`

Expected: FAIL because strict extraction can silently pass through and D4-D6 serialization clears fields.

- [ ] **Step 3: Make extraction fail closed and validate independent identity**

```python
required = ("y_true", "y_pred", "sales_scaler", "feature_columns")
missing = [name for name in required if selected.get(name) is None]
if strict and missing:
    raise MetricProtocolError("missing_metric_inputs", missing_fields=missing)
if strict:
    validate_metric_identity(selected, expected_metric_identity)
    return compute_metrics_with_protocol(...)
```

The expected identity must be constructed from `protocol_manifest` before model invocation: target key and horizon from the task; count, date range, and SHA-256 digest from ordered manifest sample keys. Never derive expected values from a prediction payload. Attach the same identity to each multi-source payload and verify exact equality before metric computation.

- [ ] **Step 4: Serialize per-method failures and preserve factual fields**

Catch `MetricProtocolError` at the method/entity boundary for No-TL, SS-TL, and multi-source methods. Write an invalid row with finite diagnostic fields only when explicitly labeled current-space, but set primary `rmse`/`smape` and paper/original aliases to `NaN`, set `paper_metric_status` to the typed status, and preserve a non-empty error. Keep `paper_metric_aligned` about metric-space alignment; use `paper_reference_available` and `paper_reference_status` for external reference availability. Extend the stable result schema with all contract, inverse, identity, and distribution-audit columns.

- [ ] **Step 5: Run focused runner/schema tests and verify GREEN**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_strict_result_contract.py tests/test_unified_d1_d6_output_contract.py tests/test_experiment_runner.py tests/test_mswa_tl.py tests/test_mssb_tl.py tests/test_msml_tl.py tests/test_msml_tl_rfe.py -q`

Expected: PASS, including No-TL and SS-TL regression assertions.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/experiment/experiment_runner.py src/transfer_methods/mswa_tl.py src/transfer_methods/mssb_tl.py src/transfer_methods/msml_tl.py src/transfer_methods/msml_tl_rfe.py src/utils/entity_experiment.py src/utils/result_schema.py tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_strict_result_contract.py tests/test_unified_d1_d6_output_contract.py
git commit -m "fix: close strict metric bypasses in experiment results"
```

### Task 3: Shared Formal Ranking and Statistical Consumers

**Files:**
- Modify: `scripts/aggregate_d1_d6_results.py`
- Modify: `src/analysis/statistical_tests.py`
- Modify: `src/visualization/result_visualizer.py`
- Test: `tests/test_aggregate_d1_d6_results.py`
- Test: `tests/test_result_visualizer.py`
- Create: `tests/test_formal_smape_statistics.py`

**Interfaces:**
- Consumes: Task 1 `is_formally_comparable_smape_row` and `smape` percent values.
- Produces: one shared formal-row filter, fixed-horizon/fixed-sharing-scenario rankings, target then dataset macro means, and complete-case dataset-level paired tests with Holm correction and effect sizes.

- [ ] **Step 1: Add failing consumer tests**

```python
def test_formal_aggregation_excludes_numeric_non_strict_row():
    rows = [valid_row(smape=20), invalid_row(strict_paper_metrics=False, smape=1)]
    result = aggregate_formal_smape(pd.DataFrame(rows))
    assert result["smape"].tolist() == [20]

def test_formal_ranking_never_merges_horizons_or_scenarios():
    result = aggregate_formal_smape(two_horizon_two_scenario_rows())
    assert result.groupby(["horizon", "sharing_scenario"]).ngroups == 4

def test_statistics_uses_complete_case_dataset_blocks():
    result = compare_methods_smape(rows_with_one_incomplete_dataset())
    assert result.loc[0, "n_datasets"] == 2
```

- [ ] **Step 2: Run consumer tests and verify RED**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_aggregate_d1_d6_results.py tests/test_result_visualizer.py tests/test_formal_smape_statistics.py -q`

Expected: FAIL because consumers currently average direct `smape` values and statistics are RMSE-oriented.

- [ ] **Step 3: Implement one eligibility path and explicit aggregation hierarchy**

```python
eligible = frame.apply(lambda row: is_formally_comparable_smape_row(row)["eligible"], axis=1)
formal = frame.loc[eligible].copy()
seed_mean = formal.groupby(["dataset", "target", "method", "horizon", "sharing_scenario"], as_index=False)["smape"].mean()
target_macro = seed_mean.groupby(["dataset", "method", "horizon", "sharing_scenario"], as_index=False)["smape"].mean()
dataset_macro = target_macro.groupby(["method", "horizon", "sharing_scenario"], as_index=False)["smape"].mean()
```

Expose excluded-row reason counts. Make visualizations call the same filter. At each fixed horizon/scenario, pivot dataset macro values, retain complete-case datasets, run paired Wilcoxon tests where sample size permits, report paired effect size and `n_datasets`, and apply Holm correction to the family of method comparisons. Label underpowered cases descriptive rather than significant.

- [ ] **Step 4: Run consumer tests and verify GREEN**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_aggregate_d1_d6_results.py tests/test_result_visualizer.py tests/test_formal_smape_statistics.py -q`

Expected: PASS with separate outputs for every horizon and sharing scenario.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/aggregate_d1_d6_results.py src/analysis/statistical_tests.py src/visualization/result_visualizer.py tests/test_aggregate_d1_d6_results.py tests/test_result_visualizer.py tests/test_formal_smape_statistics.py
git commit -m "feat: rank methods with eligible original-scale smape"
```

### Task 4: Focused Regression Verification and Handoff

**Files:**
- Modify only if a focused regression exposes a contract defect in files already listed above.
- Test: focused suites listed below.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence for the requested file list, four-method payload audit, test list, pytest result, and remaining-bypass statement.

- [ ] **Step 1: Run the complete focused suite**

Run: `/Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python tools/protection/codex_timeout.py --timeout 180 /Users/ming/Desktop/复现实验/完全保留版/.venv/bin/python -m pytest tests/test_smape_metric_contract.py tests/test_metrics_normalized_only.py tests/test_multi_source_smape_metrics.py tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_strict_result_contract.py tests/test_unified_d1_d6_output_contract.py tests/test_experiment_runner.py tests/test_mswa_tl.py tests/test_mssb_tl.py tests/test_msml_tl.py tests/test_msml_tl_rfe.py tests/test_aggregate_d1_d6_results.py tests/test_result_visualizer.py tests/test_formal_smape_statistics.py -q`

Expected: all selected tests PASS; exit code 0; no timeout.

- [ ] **Step 2: Inspect strict bypasses and changed files**

Run: `rg -n "strict_paper_metrics|compute_metrics_with_protocol|smape_paper|original_scale_smape|inverse_transform_applied" src scripts/aggregate_d1_d6_results.py`

Expected: every strict extraction path reaches protocol computation or typed invalid-row serialization; no strict path coalesces normalized/current-space sMAPE into primary or paper fields.

Run: `git diff --check && git status --short && git diff --stat ed4ce918...HEAD`

Expected: no whitespace errors; only scoped source, test, and documentation files changed.

- [ ] **Step 3: Commit any final focused regression adjustment**

```bash
git add src/evaluation/metric_contract.py src/evaluation/metrics.py src/constants.py src/experiment/experiment_runner.py src/transfer_methods/mswa_tl.py src/transfer_methods/mssb_tl.py src/transfer_methods/msml_tl.py src/transfer_methods/msml_tl_rfe.py src/utils/entity_experiment.py src/utils/result_schema.py scripts/aggregate_d1_d6_results.py src/analysis/statistical_tests.py src/visualization/result_visualizer.py tests/test_smape_metric_contract.py tests/test_metrics_normalized_only.py tests/test_multi_source_metric_payload_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_strict_result_contract.py tests/test_unified_d1_d6_output_contract.py tests/test_aggregate_d1_d6_results.py tests/test_result_visualizer.py tests/test_formal_smape_statistics.py
git commit -m "test: verify strict smape ranking contract"
```

- [ ] **Step 4: Report completion**

Report: modified files; per-method payload presence for MSWA-TL, MSSB-TL, MSML-TL, and MSML-TL-RFE; added/updated tests; exact focused pytest counts; pre-existing failures if any; and a direct yes/no answer to whether any `strict=True` path can still return normalized primary RMSE or sMAPE.
