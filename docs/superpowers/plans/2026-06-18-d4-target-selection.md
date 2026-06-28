# D4 Target Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a D4 Dingdong target domain selection utility that selects one target `store_id`, 3-5 target `product_id` values, a same-`second_category_id` source pool, and MMD/structural-shift reports using no-leakage training history.

**Architecture:** Add one focused script with pure helper functions plus a CLI entry point. Unit tests exercise helper behavior on synthetic data; a smoke run verifies the script can read the real D4 paths and write the required artifacts.

**Tech Stack:** Python, pandas, numpy, scikit-learn `StandardScaler`/`rbf_kernel`, pytest/unittest-compatible tests.

## Global Constraints

- Raw D4 data path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet`
- Profile path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/dataset_profiles/Dataset4/`
- Target domain column: `store_id`
- SKU column: `product_id`
- Date column: `dt`
- Sales column: `sale_amount`
- Category column: `second_category_id`
- Window config: `15 train + 15 val + 180 test = 210`
- `source_history_days = 300`
- Minimum target SKU date coverage: `300 + 210 = 510`
- Final metrics must be recomputed from raw `train.parquet`, not copied from profile values.
- Selection metrics, MMD features, and structural-shift features must use `dt < val_start`.
- Source pool must exclude target store and align on `second_category_id`.
- MMD must use one `StandardScaler` fit jointly on target + source features.
- Candidate store selection must use the candidate MMD Q25-Q75 interval, not a fixed absolute threshold.

---

### Task 1: Unit Tests for Date Windows, Gaps, Metrics, and MMD

**Files:**
- Create: `tests/test_d4_target_selection.py`
- Create later: `scripts/select_d4_target_domain.py`

**Interfaces:**
- Consumes: none
- Produces expected imports from `scripts.select_d4_target_domain`: `WindowConfig`, `compute_target_windows`, `compute_source_window`, `date_coverage_days`, `max_calendar_gap_days`, `spike_ratio`, `compute_sku_metrics`, `extract_sku_summary`, `scaled_mmd`, `filter_eligible_skus`, `choose_target_skus`.

- [ ] **Step 1: Write the failing test**

```python
def test_target_and_source_windows_are_inclusive_and_ordered():
    cfg = selector.WindowConfig()
    windows = selector.compute_target_windows("2024-12-31", cfg)
    assert windows["test_start"] == pd.Timestamp("2024-07-05")
    assert windows["val_start"] == pd.Timestamp("2024-06-20")
    assert windows["target_train_start"] == pd.Timestamp("2024-06-05")
    source = selector.compute_source_window(windows["target_train_start"], 300)
    assert source["source_history_start"] == pd.Timestamp("2023-08-11")
    assert source["source_history_end"] == pd.Timestamp("2024-06-05")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_d4_target_selection.py -q`
Expected: import failure for missing `scripts.select_d4_target_domain`.

- [ ] **Step 3: Add more failing tests before implementation**

Add tests for inclusive `date_coverage_days`, long gap counting, SKU metric thresholds, category/CV target SKU ordering, and `scaled_mmd` using one shared scaler object.

### Task 2: Implement Pure Selection Helpers

**Files:**
- Modify: `scripts/select_d4_target_domain.py`
- Test: `tests/test_d4_target_selection.py`

**Interfaces:**
- Produces helper functions listed in Task 1 plus `WindowConfig` dataclass.
- Later tasks consume these helpers for candidate scan and CLI output.

- [ ] **Step 1: Implement minimal helper functions**

Implement window calculation, source window, coverage, gap, spike ratio, SKU metrics, SKU summaries, shared-scaler MMD, eligible filtering, and target SKU choice.

- [ ] **Step 2: Run helper tests**

Run: `pytest tests/test_d4_target_selection.py -q`
Expected: Task 1 tests pass.

### Task 3: Candidate Store Scan and Result Assembly

**Files:**
- Modify: `scripts/select_d4_target_domain.py`
- Modify: `tests/test_d4_target_selection.py`

**Interfaces:**
- Produces: `aggregate_store_profile(profile_dir)`, `scan_candidate_stores(df, store_profile, cfg)`, `select_target_store(scan_df)`, `build_selection_result(df, target_store_id, cfg)`.
- Consumes: pure helpers from Task 2.

- [ ] **Step 1: Write failing tests for end-to-end synthetic selection**

Use a synthetic dataframe with two stores, two categories, enough calendar coverage, and source rows before `target_train_start`. Assert the chosen store is inside the MMD interquartile interval, target SKUs are in the largest category sorted by CV, and source pool excludes the target store.

- [ ] **Step 2: Implement candidate scan**

Read profile only for store-quality narrowing, recompute final metrics on raw dataframe, build MMD features from no-leakage history, compute Q25/Q75, and choose the highest-quality candidate inside that interval.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_d4_target_selection.py -q`
Expected: all synthetic selection tests pass.

### Task 4: CLI Outputs

**Files:**
- Modify: `scripts/select_d4_target_domain.py`
- Modify: `tests/test_d4_target_selection.py`

**Interfaces:**
- Produces CLI: `python scripts/select_d4_target_domain.py --raw-path ... --profile-dir ... --output-dir ... --permutations 50`
- Writes: `store_candidate_profile.csv`, `warehouse_mmd_scan.csv`, `target_sku_metrics.csv`, `target_selection_result.json`, `target_selection_report.md`.

- [ ] **Step 1: Write failing output test**

Use temporary raw parquet/profile CSV fixtures. Run the Python API or CLI `main()` and assert all required output files exist and JSON contains `target_store_id`, `target_skus`, `target_categories`, `mmd`, `permutation_test`, and `structural_shift`.

- [ ] **Step 2: Implement output writers and report**

Write CSV/JSON/Markdown artifacts under `outputs/domain_adaptation/Dataset4/target_selection/` by default. Make permutation count configurable so tests can use a small number and real runs can use 500.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_d4_target_selection.py -q`
Expected: all tests pass.

### Task 5: Real D4 Smoke Run and Verification

**Files:**
- No code changes unless the smoke run exposes a defect.

**Interfaces:**
- Consumes CLI from Task 4.
- Produces real artifacts in `outputs/domain_adaptation/Dataset4/target_selection/`.

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/test_d4_target_selection.py -q`
Expected: all tests pass.

- [ ] **Step 2: Run real D4 target selection**

Run: `python scripts/select_d4_target_domain.py --permutations 100`
Expected: command exits 0 and prints the selected target store, SKU count, source pool size, and output directory.

- [ ] **Step 3: Inspect output artifacts**

Run: `python -m json.tool outputs/domain_adaptation/Dataset4/target_selection/target_selection_result.json`
Expected: JSON is valid and includes non-null `target_store_id`, 3-5 `target_skus`, source pool counts, MMD value/gamma, p-value, and structural-shift keys.
