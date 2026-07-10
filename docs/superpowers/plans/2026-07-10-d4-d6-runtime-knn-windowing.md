# D4–D6 Runtime KNN Windowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make D4–D6 formal runtime KNN selection leakage-free, date-aligned, runtime-authoritative, and traceable without changing features, signature statistics, distance, or scaling.

**Architecture:** The D4–D6 parquet loader materializes explicit 30-day observed and inclusive 300-day source-history metadata, while retaining the full target evaluation frame. `SourceSelector` detects that explicit protocol, aligns each source to the exact target observed dates, skips incomplete sources, and returns deterministic runtime metadata/digests. Transfer methods and result rows propagate that metadata, and regeneration uses the same domain policy while rebuilding metadata instead of copying it.

**Tech Stack:** Python 3, pandas, NumPy, hashlib/json, pytest.

## Global Constraints

- Apply only to D4–D6 runtime-authoritative selection; do not alter D1–D3 legacy behavior.
- `protocol_version = "runtime_knn_windowed_stats_v1"`.
- Target observed dates are `train_start` through `train_start + 29 days`, inclusive: 15 train + 15 validation days.
- Source history is the inclusive 300-day interval ending on `target_observed_end`; this is an engineering reproduction protocol, not a paper-specified constant.
- KNN target and source signatures use only the same 30 observed calendar dates.
- Preserve current `feature_cols`, mean/std/min/max/last signature, Euclidean distance, distance weighting, and no-scaling behavior.
- Do not run full D1–D6 experiments, representation ablations, or scaling ablations.
- Run every pytest/Python test command through `python tools/protection/codex_timeout.py ...`; if it exits 124, stop and hand the exact command to the user.
- Preserve the existing untracked `tests/test_solidified_knn_config_not_generated.py` unless the user separately asks to modify it.

---

### Task 1: Materialize and enforce D4–D6 runtime windows in the loader

**Files:**
- Modify: `src/constants.py`
- Modify: `src/utils/parquet_data_loader.py`
- Create: `tests/test_source_selector_window_leakage.py`

**Interfaces:**
- Produces constant `D4_D6_RUNTIME_KNN_PROTOCOL_VERSION: str`.
- Produces `derive_d4_d6_runtime_knn_windows(windows: dict[str, Any], source_history_days: int) -> dict[str, Any]`.
- `load_parquet_source_target(...)` returns a full target frame plus a source frame cropped to inclusive runtime history bounds, with identical protocol attrs attached to both.

- [ ] **Step 1: Write failing loader/window tests**

Add tests that monkeypatch `pandas.read_parquet` with small source/target frames, call `load_parquet_source_target()`, and assert:

```python
assert target_df["date"].min() == pd.Timestamp("2024-01-01")
assert target_df["date"].max() == pd.Timestamp("2024-07-28")
assert target_df.attrs["target_observed_start"] == pd.Timestamp("2024-01-01")
assert target_df.attrs["target_observed_end"] == pd.Timestamp("2024-01-30")
assert source_df.attrs["source_history_start"] == pd.Timestamp("2023-04-06")
assert source_df.attrs["source_history_end"] == pd.Timestamp("2024-01-30")
assert source_df["date"].between("2023-04-06", "2024-01-30").all()
assert target_df.attrs["selection_authority"] == "runtime"
assert target_df.attrs["protocol_version"] == "runtime_knn_windowed_stats_v1"
```

Also assert missing source/target `date`, missing `train_start`, invalid dates, and non-positive `source_history_days` raise `ValueError` only through this D4–D6 loader.

- [ ] **Step 2: Run the loader tests and confirm failure**

Run:

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_source_selector_window_leakage.py -q
```

Expected: FAIL because explicit runtime bounds and upper source cropping do not exist.

- [ ] **Step 3: Implement exact inclusive window derivation**

Add the constant to `src/constants.py` and implement the loader helper with this calculation:

```python
target_observed_start = pd.Timestamp(windows["train_start"]).normalize()
target_observed_end = target_observed_start + pd.Timedelta(days=29)
source_history_end = target_observed_end
source_history_start = source_history_end - pd.Timedelta(days=source_history_days - 1)
```

Return/attach these fields plus:

```python
{
    "selection_authority": "runtime",
    "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
    "target_test_excluded": True,
    "source_future_excluded": True,
    "source_alignment_mode": "exact_target_observed_dates",
    "representation": "mean_std_min_max_last",
    "scaling": "none",
    "scaler_fit_scope": "not_applicable",
}
```

Require the date columns, normalize them to pandas timestamps, reject `NaT`, and crop source rows with an inclusive `between(source_history_start, source_history_end)` mask. Keep target loading/reindexing through `test_end` unchanged. Replace the misleading comment claiming `SourceSelector` already excludes target test rows with a comment describing the explicit attrs.

- [ ] **Step 4: Re-run the loader tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the loader boundary change**

```bash
git add src/constants.py src/utils/parquet_data_loader.py tests/test_source_selector_window_leakage.py
git commit -m "fix: bound D4-D6 runtime KNN windows"
```

### Task 2: Align runtime KNN signatures to the exact observed dates

**Files:**
- Modify: `src/source_selection/source_selector.py`
- Modify: `tests/test_source_selector_window_leakage.py`

**Interfaces:**
- Adds private runtime-only helpers that consume the attrs from Task 1.
- `SourceSelector.select_top_k_sources(...) -> dict[str, object]` retains its public signature and legacy path.
- Runtime result `meta` contains required protocol fields, skip diagnostics, and digests.

- [ ] **Step 1: Add failing leakage/alignment tests**

Create a crafted D4–D6 fixture with 30 target observed dates, target test dates, complete source A/B groups, an incomplete source, and source-future rows. Attach the Task 1 attrs. Assert:

```python
baseline = selector.select_top_k_sources(...)
target_test_perturbed = selector.select_top_k_sources(...)
source_future_perturbed = selector.select_top_k_sources(...)
observed_perturbed = selector.select_top_k_sources(...)

assert source_keys(baseline) == source_keys(target_test_perturbed)
assert source_keys(baseline) == source_keys(source_future_perturbed)
assert source_keys(baseline) != source_keys(observed_perturbed)
assert baseline["meta"]["source_alignment_mode"] == "exact_target_observed_dates"
assert baseline["meta"]["source_skip_diagnostics"][0]["reason"] == "missing_target_observed_dates"
```

Add tests that D4–D6-marked frames missing observed attrs fail fast, while an unmarked legacy frame still uses the existing full-sequence path. Add target duplicate-date fail-fast and source duplicate-date skip coverage so every accepted source has exactly one row per observed date.

- [ ] **Step 2: Run the selector tests and confirm failure**

Run the Task 1 pytest command. Expected: FAIL because runtime alignment and diagnostics are missing.

- [ ] **Step 3: Implement a gated runtime-aligned signature path**

In `SourceSelector`, detect runtime mode only when `protocol_version == D4_D6_RUNTIME_KNN_PROTOCOL_VERSION` or `selection_authority == "runtime"`. Validate identical required bounds on source and target attrs.

Build `required_dates = pd.date_range(target_observed_start, target_observed_end, freq="D")`, slice the target to those dates, and require exact unique-date equality. For each source group:

```python
history = group[group["date"].between(source_history_start, source_history_end)]
aligned = history[history["date"].isin(required_dates)].sort_values("date")
```

Skip groups whose unique date set differs from `required_dates` or which have duplicate observed dates. Call the unchanged `_signature_from_df()` only with the 30-row target/aligned source frames. If no source remains, raise `ValueError` containing aggregate skip diagnostics.

Use deterministic strict JSON serialization plus SHA-256 helpers:

```python
candidate_pool_digest = sha256(sorted_eligible_source_keys)
selection_result_digest = sha256([
    {"source_rank": ..., "source_key": ..., "distance": ..., "weight": ...},
])
```

After ranking, populate `meta` with all required fields and `selected_sources_runtime = list(results)`. Do not change `_signature_from_df()`, Euclidean distance, weight calculations, or feature resolution.

- [ ] **Step 4: Re-run selector tests**

Run the Task 1 pytest command. Expected: PASS.

- [ ] **Step 5: Run the legacy selector unit check**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_source_selector.py -q
```

Expected: PASS or no collected pytest tests; the legacy callable behavior remains available.

- [ ] **Step 6: Commit aligned selection**

```bash
git add src/source_selection/source_selector.py tests/test_source_selector_window_leakage.py
git commit -m "fix: align runtime KNN signatures by observed date"
```

### Task 3: Propagate runtime authority metadata into D4–D6 results

**Files:**
- Modify: `src/transfer_methods/source_failure_tolerance.py`
- Modify: `src/transfer_methods/mswa_tl.py`
- Modify: `src/transfer_methods/mssb_tl.py`
- Modify: `src/transfer_methods/msml_tl.py`
- Modify: `src/transfer_methods/msml_tl_rfe.py`
- Modify: `src/utils/entity_experiment.py`
- Modify: `src/constants.py`
- Create: `tests/test_d4_d6_source_authority.py`

**Interfaces:**
- Produces `runtime_selection_meta(selection_result: Mapping[str, object]) -> dict[str, object]` in `source_failure_tolerance.py`.
- D4–D6 transfer result `meta` and CSV-ready rows expose the required runtime metadata.

- [ ] **Step 1: Write failing metadata propagation tests**

Test the shared extraction helper and `_row_from_result()` with a runtime selection whose Top-K differs from a fake JSON Top-K. Assert:

```python
assert row["selection_authority"] == "runtime"
assert row["protocol_version"] == "runtime_knn_windowed_stats_v1"
assert json.loads(row["selected_sources_runtime"]) == runtime_top_k
assert json.loads(row["selected_sources"]) == runtime_top_k
assert json_top_k != json.loads(row["selected_sources_runtime"])
assert row["candidate_pool_digest"] == "candidate-sha256"
assert row["selection_result_digest"] == "result-sha256"
```

Assert no row field labels JSON Top-K as actual/runtime selection.

- [ ] **Step 2: Run authority tests and confirm failure**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_d6_source_authority.py -q
```

Expected: FAIL because runtime fields are not propagated.

- [ ] **Step 3: Add a focused shared propagation helper**

Define the exact runtime field tuple once in `source_failure_tolerance.py` and copy only those keys from `selection_result["meta"]`. Spread that helper into the four multi-source transfer result `meta` dictionaries. Keep their existing `selected_sources` field as the actual runtime `SourceSelector` output.

Extend `_selection_meta()` and `_row_from_result()` so list/dict diagnostics are serialized with `_stable_json_dumps`, booleans remain booleans, dates become strings, and digest/protocol fields remain strings. Add the exact required columns to `RESULT_SCHEMA_COLUMNS` next to `selected_sources`.

- [ ] **Step 4: Re-run authority and result-schema tests**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_d6_source_authority.py tests/test_result_schema_golden_diff.py tests/test_unified_d1_d6_output_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit runtime metadata propagation**

```bash
git add src/constants.py src/transfer_methods/source_failure_tolerance.py src/transfer_methods/mswa_tl.py src/transfer_methods/mssb_tl.py src/transfer_methods/msml_tl.py src/transfer_methods/msml_tl_rfe.py src/utils/entity_experiment.py tests/test_d4_d6_source_authority.py
git commit -m "fix: record runtime KNN source authority"
```

### Task 4: Rebuild regenerated metadata and share runtime domain policy

**Files:**
- Modify: `scripts/regenerate_solidified_knn.py`
- Modify: `tests/test_regenerate_solidified_knn_check_only.py`
- Modify: `tests/test_d4_d6_domain_filter.py`

**Interfaces:**
- `_filter_source_for_scenario(...)` delegates to `apply_source_domain_policy()` for all D4–D6 datasets.
- Regeneration builds `selection_metadata[target_entity_id]` from the current `SourceSelector` result.
- No stale metadata survives from `old_payload`.

- [ ] **Step 1: Write failing regeneration/domain tests**

Add a unit test with an old payload containing unmistakable stale values:

```python
old_payload["selection_metadata"] = {
    "target-a": {"knn_representation": "paper_observed_sequence", "stale": True}
}
```

Call a new pure payload-builder helper and assert the output selection metadata equals newly supplied runtime metadata, contains `runtime_knn_windowed_stats_v1`, and contains neither `paper_observed_sequence` nor `stale`.

Parametrize D4/D5/D6 without-sharing filters using legacy `{column, value}`, list values, and multi-field filters. Assert `_filter_source_for_scenario()` yields the same frame and diagnostics semantics as `apply_source_domain_policy()`.

- [ ] **Step 2: Run regeneration/domain tests and confirm failure**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_regenerate_solidified_knn_check_only.py tests/test_d4_d6_domain_filter.py -q
```

Expected: FAIL because regeneration deep-copies stale metadata and only filters D4 specially.

- [ ] **Step 3: Replace deepcopy inheritance with explicit payload construction**

Import and call `apply_source_domain_policy()` in `_filter_source_for_scenario()` for every dataset/scenario. Preserve source attrs after filtering.

During target iteration, save both converted result rows and a strict-JSON-safe copy of `selected["meta"]`. Build the new payload from an explicit allowlist of stable config keys:

```python
stable_keys = (
    "dataset_id", "dataset", "info_sharing", "k", "window_size",
    "horizon", "target_train_window", "domain_filter", "group_cols",
)
new_payload = {key: copy.deepcopy(old_payload[key]) for key in stable_keys if key in old_payload}
new_payload.update({
    "selection_authority": "runtime",
    "protocol_version": "runtime_knn_windowed_stats_v1",
    "feature_cols": feature_cols,
    "feature_info": feature_info,
    "source_pool_size": int(len(source_df)),
    "results": new_results,
    "selection_metadata": new_selection_metadata,
})
```

Validate payload dataset/scenario identity, `k`, `group_cols`, and required without-sharing domain filter before selection. Do not read protocol bounds from old `selection_metadata`.

- [ ] **Step 4: Re-run regeneration/domain tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit regeneration safety**

```bash
git add scripts/regenerate_solidified_knn.py tests/test_regenerate_solidified_knn_check_only.py tests/test_d4_d6_domain_filter.py
git commit -m "fix: rebuild regenerated KNN metadata"
```

### Task 5: Remove redundant window attachment and run focused acceptance

**Files:**
- Modify: `scripts/run_d4_experiment.py`
- Modify: `scripts/run_d5_experiment.py`
- Modify: `scripts/run_d6_experiment.py`
- Modify: `scripts/regenerate_solidified_knn.py`
- Test: `tests/test_source_selector_window_leakage.py`
- Test: `tests/test_d4_d6_source_authority.py`
- Test: `tests/test_regenerate_solidified_knn_check_only.py`
- Test: `tests/test_d4_d6_domain_filter.py`

**Interfaces:**
- D4–D6 runners consume loader-attached protocol attrs without reattaching incomplete window dictionaries.

- [ ] **Step 1: Remove redundant `attach_window_attrs()` calls/imports**

The loader is the sole owner of runtime KNN window derivation. Remove the post-loader reattachment in each D4–D6 runner and in regeneration so explicit source-history metadata cannot be obscured by a second partial attachment. Do not change target-key or smoke/source-limit logic.

- [ ] **Step 2: Run the requested focused acceptance suite**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest \
  tests/test_source_selector_window_leakage.py \
  tests/test_d4_d6_source_authority.py \
  tests/test_regenerate_solidified_knn_check_only.py \
  tests/test_d4_d6_domain_filter.py \
  -q
```

Expected: PASS within 180 seconds. If exit code is 124, stop immediately and provide this exact command for manual execution.

- [ ] **Step 3: Run directly related loader/result contract tests**

```bash
python tools/protection/codex_timeout.py .venv/bin/python -m pytest \
  tests/test_read_dataset_windows_scenario_identity.py \
  tests/test_d5_knn_model_feature_consistency.py \
  tests/test_result_schema_golden_diff.py \
  tests/test_unified_d1_d6_output_contract.py \
  -q
```

Expected: PASS within 180 seconds. Do not broaden to full experiments.

- [ ] **Step 4: Run static syntax checks**

```bash
.venv/bin/python -m py_compile \
  src/source_selection/source_selector.py \
  src/utils/parquet_data_loader.py \
  src/utils/entity_experiment.py \
  src/transfer_methods/source_failure_tolerance.py \
  scripts/regenerate_solidified_knn.py \
  scripts/run_d4_experiment.py \
  scripts/run_d5_experiment.py \
  scripts/run_d6_experiment.py
```

Expected: exit code 0. This is a static syntax check allowed without the timeout wrapper.

- [ ] **Step 5: Inspect final diff and commit runner cleanup**

Confirm `git diff --check` is clean and that `tests/test_solidified_knn_config_not_generated.py` remains unmodified/untracked unless it was already user-owned. Then commit only task files:

```bash
git add scripts/run_d4_experiment.py scripts/run_d5_experiment.py scripts/run_d6_experiment.py scripts/regenerate_solidified_knn.py
git commit -m "fix: preserve D4-D6 runtime KNN window attrs"
```

The final report must list modified files, map changes to requirements, list tests, reproduce exact pytest commands/results, state remaining risks, and explicitly say that no full D1–D6 experiment was run.
