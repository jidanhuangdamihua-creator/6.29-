# D6 Target Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a formal Dataset6 M5 selector that outputs one target store, 3-5 same-department target SKUs, concrete source entities, store-level and final MMD statistics, signed structural-shift diagnostics, and CSV/JSON/MD reports.

**Architecture:** Add one dedicated script with pure helpers and a CLI, matching the D4/D5 selector delivery pattern while using M5 wide-table semantics. Unit tests use synthetic M5-wide fixtures and are written before implementation; real-data smoke validates the final selector on `sales_train_evaluation.csv`.

**Tech Stack:** Python, pandas, numpy, scikit-learn `StandardScaler`/`rbf_kernel`/`pairwise_distances`, unittest-compatible tests.

## Global Constraints

- Raw D6 data path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 6m5-forecasting-accuracy`
- Output path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/domain_adaptation/Dataset6/target_selection`
- Required input files: `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`
- Required output files: `target_selection_result.json`, `store_candidate_profile.csv`, `target_sku_metrics.csv`, `target_selection_report.md`
- Sales table columns: `id,item_id,dept_id,cat_id,store_id,state_id,d_1...d_1941`
- Max date is resolved from `calendar.csv` where `d == "d_1941"` and must be `2016-05-22`
- Window config: `15 train + 15 val + 180 test = 210`, `source_history_days = 300`
- Correct windows: `train_start=2015-10-26`, `train_end=2015-11-09`, `val_start=2015-11-10`, `val_end=2015-11-24`, `test_start=2015-11-25`, `test_end=2016-05-22`, `source_start=2014-12-30`, `source_end=2015-10-25`
- Screening window is `d_1...d_1746`, derived from calendar mapping to `train_end`
- D6 zeros are explicit; do not fill missing days and do not clip sales
- Target SKUs must all share one `dept_id`; no cross-department mixing
- Preferred department is `FOODS_3`
- Fallback rounds: Round 1 `FOODS_3` thresholds `nonzero>=0.30`, `cv<=1.5`, `spike<=10`; Round 2 `FOODS_3` thresholds `nonzero>=0.20`, `cv<=2.0`, `spike<=15`; Round 3 `FOODS_1/2/3` with Round 1 thresholds, selecting the qualifying department with lowest median CV
- Store-level MMD samples up to 50 SKUs per store across all departments, deterministic by `random_seed=42`
- Store candidate selection uses Q25-Q75 inclusive and ranks by distance to median, state priority `CA > TX > WI`, then `store_id`
- MMD uses one shared `StandardScaler` fit jointly on target/source features
- Gamma uses median heuristic on standardized features; fallback `gamma=1.0`
- Permutation p-value uses corrected formula `(count + 1) / (n_perm + 1)` with `permutation_seed=43`
- Structural shift is signed `target - source`, and JSON includes `structural_shift_semantics = "signed_target_minus_source"`
- `source_entities` and `target_entities` are canonical concrete `{store_id,item_id}` objects

---

### Task 1: Unit Tests for Helpers and Output Contract

**Files:**
- Create: `tests/test_d6_target_selection.py`
- Create later: `scripts/select_d6_target_domain.py`

**Interfaces:**
- Consumes: none.
- Produces expected imports from `scripts.select_d6_target_domain`: `WindowConfig`, `resolve_max_date`, `compute_target_windows`, `compute_source_window`, `get_screening_d_cols`, `extract_sku_summary`, `compute_sku_screening_metrics`, `select_target_skus_with_fallback`, `build_source_entities`, `scaled_mmd`, `permutation_p_value`, `compute_structural_shift`, `run_target_selection`.

- [ ] **Step 1: Write failing helper tests**

Create tests covering corrected windows, feature edge cases, shared scaler, corrected permutation p-value, fallback rules, source entities, and output files. Use synthetic helper rows:

```python
def _make_wide_row(store_id, item_id, dept_id, cat_id, state_id, sales_array):
    row = {
        "id": f"{item_id}_{store_id}",
        "item_id": item_id,
        "dept_id": dept_id,
        "cat_id": cat_id,
        "store_id": store_id,
        "state_id": state_id,
    }
    for i, value in enumerate(sales_array, start=1):
        row[f"d_{i}"] = int(value)
    return row

def _make_calendar(n_days=1941, start="2011-01-29"):
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)], "date": dates})
```

- [ ] **Step 2: Run tests and verify red**

Run: `python -m unittest tests.test_d6_target_selection`

Expected: import failure for missing `scripts.select_d6_target_domain`.

### Task 2: Implement Core Helpers

**Files:**
- Create: `scripts/select_d6_target_domain.py`
- Test: `tests/test_d6_target_selection.py`

**Interfaces:**
- Produces calendar/window, feature, MMD, fallback, source entity, and structural shift helpers.

- [ ] **Step 1: Implement calendar and window helpers**

Implement `WindowConfig`, `resolve_max_date`, `compute_target_windows`, `compute_source_window`, and `get_screening_d_cols`.

- [ ] **Step 2: Implement feature extraction**

Implement `extract_sku_summary` with seven features: `mean`, `std`, `cv`, `zero_ratio`, `iqr`, `acf_lag7`, `trend_slope`. Use `ddof=0`, MMD `mean==0 -> cv=0`, edge-case `acf_lag7=0`, and sanitize NaN/inf to zero.

- [ ] **Step 3: Implement MMD and permutation helpers**

Implement `scaled_mmd` and `permutation_p_value`; keep scaler/gamma fixed during permutation.

- [ ] **Step 4: Implement screening/fallback/entity helpers**

Implement `compute_sku_screening_metrics`, `select_target_skus_with_fallback`, `build_source_entities`, and `compute_structural_shift`.

- [ ] **Step 5: Run helper tests**

Run: `python -m unittest tests.test_d6_target_selection`

Expected: helper tests pass; output tests may still fail until Task 4.

### Task 3: Implement Store Scan and Final Selection

**Files:**
- Modify: `scripts/select_d6_target_domain.py`
- Test: `tests/test_d6_target_selection.py`

**Interfaces:**
- Produces `scan_candidate_stores`, `rank_candidate_stores`, final target/source feature builders, and selected store fallback attempts.

- [ ] **Step 1: Implement wide-table loading and validation**

Validate required files and columns, read `sales_train_evaluation.csv`, detect `d_` columns, and keep wide format.

- [ ] **Step 2: Implement store-level MMD scan**

For each store, sample up to `store_sample_size` rows after sorting by `item_id`; compute features on screening columns; compute `store_mmd` vs all other stores.

- [ ] **Step 3: Implement store ranking and fallback loop**

Compute Q25/median/Q75, rank IQR stores, try SKU fallback per store, and record `fallback_attempts`.

- [ ] **Step 4: Run tests**

Run: `python -m unittest tests.test_d6_target_selection`

Expected: selection tests pass; output tests may still fail until writers are complete.

### Task 4: Implement Outputs and CLI

**Files:**
- Modify: `scripts/select_d6_target_domain.py`
- Test: `tests/test_d6_target_selection.py`

**Interfaces:**
- Produces `run_target_selection(dataset_root, output_dir, random_seed, permutation_seed, n_perm, store_sample_size, source_history_days)` and CLI `main()`.

- [ ] **Step 1: Implement final MMD and structural shift**

After target selection, compute final MMD on target department features, corrected permutation p-value, signed structural shift, and summary payload.

- [ ] **Step 2: Implement writers**

Write `store_candidate_profile.csv`, `target_sku_metrics.csv`, `target_selection_result.json`, and `target_selection_report.md`.

- [ ] **Step 3: Implement CLI**

Support `--dataset-root`, `--output-dir`, `--random-seed`, `--permutation-seed`, `--n-perm`, `--store-sample-size`, and `--source-history-days`.

- [ ] **Step 4: Run all D6 tests**

Run: `python -m unittest tests.test_d6_target_selection`

Expected: all tests pass.

### Task 5: Verification on Real D6 Data

**Files:**
- No intended code changes unless verification exposes defects.

**Interfaces:**
- Consumes CLI from Task 4.
- Produces real artifacts in `outputs/domain_adaptation/Dataset6/target_selection/`.

- [ ] **Step 1: Run focused tests**

Run: `python -m unittest tests.test_d6_target_selection`

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/select_d6_target_domain.py tests/test_d6_target_selection.py`

Expected: exits 0.

- [ ] **Step 3: Run real D6 selector smoke**

Run: `python scripts/select_d6_target_domain.py --n-perm 100`

Expected: exits 0 and prints target store, target department, target SKU count, source entity count, store MMD, final MMD, and output directory.

- [ ] **Step 4: Validate JSON artifact**

Run a small Python JSON checker that verifies required keys, 3-5 target SKUs from one department, `structural_shift_semantics == "signed_target_minus_source"`, and corrected time windows.
