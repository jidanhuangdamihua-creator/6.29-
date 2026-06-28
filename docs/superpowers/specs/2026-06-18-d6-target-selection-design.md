# D6 Target Selection Design

> Status: draft for review
> Scope: formal Dataset6 M5 target store/SKU selection utility, two-tier MMD reporting, structural-shift diagnostics, and output artifacts aligned with the D4/D5 selector shape.
> Out of scope: modifying raw M5 files, changing D4/D5 selector behavior, cold-start truncation preprocessing, or adding Dataset6 to the formal full experiment runner.

## Goal

Build a formal D6 M5 target selection script that chooses one target `store_id`, 3-5 target `item_id` values from one `dept_id`, concrete source entities, store-level and final MMD values, signed structural-shift diagnostics, and CSV/JSON/MD reports using no-leakage screening-window metrics.

## Current Context

Dataset6 raw M5 files are available under:

`/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 6m5-forecasting-accuracy/`

Required files:

- `sales_train_evaluation.csv`: `id,item_id,dept_id,cat_id,store_id,state_id,d_1...d_1941`
- `calendar.csv`: `d,date,wm_yr_wk,weekday,month,year,event_name_1,event_name_2,event_type_1,event_type_2,snap_CA,snap_TX,snap_WI`
- `sell_prices.csv`: `store_id,item_id,wm_yr_wk,sell_price`

The repository already has formal selector patterns for D4 and D5:

- `scripts/select_d4_target_domain.py`
- `scripts/select_d5_target_domain.py`

D6 should be implemented as a dedicated script, not folded into the profile scanner or the D1-D6 data-contract work.

## Approach Options

### Option 1: Add a Dedicated D6 Selector

Create `scripts/select_d6_target_domain.py` with pure helper functions plus a CLI entry point. Add `tests/test_d6_target_selection.py` with synthetic M5 wide-table fixtures.

Pros: clear alignment with D4/D5, low blast radius, easy to audit.

Cons: repeats some MMD/window helper logic until a later shared abstraction is worth extracting.

### Option 2: Refactor D4/D5/D6 Into a Shared Selector Framework

Extract common MMD, window, output, and structural-shift utilities into shared modules.

Pros: less duplication later.

Cons: too much refactor risk before D6 behavior is proven on real M5 data.

### Option 3: Use Dataset6 Profile Candidates Directly

Read the existing profile candidate target/source rows and write final artifacts from those.

Pros: fastest path to an output file.

Cons: profile candidates are explicitly marked unverified and must not become formal target/source facts.

Recommended approach: Option 1. Implement D6 as a dedicated selector now; revisit shared utilities only after D4-D6 selectors are stable.

## Output Contract

The CLI writes all final artifacts under:

`outputs/domain_adaptation/Dataset6/target_selection/`

Required files:

- `target_selection_result.json`
- `store_candidate_profile.csv`
- `target_sku_metrics.csv`
- `target_selection_report.md`

`target_selection_result.json` must include:

- `target_store`
- `target_department`
- `target_skus`
- `fallback_round`
- `target_entities`
- `source_entities`
- `store_mmd`
- `final_mmd`
- `final_mmd_gamma`
- `permutation_p_value`
- `store_selection_rank`
- `fallback_attempts`
- `q25_mmd`
- `median_mmd`
- `q75_mmd`
- `structural_shift`
- `structural_shift_semantics`
- `train_start`
- `train_end`
- `val_start`
- `val_end`
- `test_start`
- `test_end`
- `source_start`
- `source_end`
- `source_history_days`
- `selection_rules`
- `random_seed`
- `permutation_seed`
- `n_perm`
- `store_sample_size`
- `feature_columns`

`target_entities` and `source_entities` are canonical for downstream use. They must be concrete objects:

```json
[
  {"store_id": "CA_3", "item_id": "FOODS_3_586"}
]
```

`store_candidate_profile.csv` must include all 10 stores with:

- `store_id`
- `state_id`
- `store_mmd`
- `abs_mmd_distance`
- `q25_mmd`
- `median_mmd`
- `q75_mmd`
- `is_in_iqr`
- `rank_in_iqr`
- `n_foods3_eligible_r1`

`target_sku_metrics.csv` must include all screened SKUs from the selected store's fallback scope, with:

- `item_id`
- `dept_id`
- `cat_id`
- `nonzero_ratio`
- `cv`
- `spike_ratio`
- `mean`
- `std`
- `zero_ratio`
- `iqr`
- `acf_lag7`
- `trend_slope`
- `is_selected`

## Time Windows

Windows are computed from `calendar.csv`, never hardcoded from the example JSON.

`max_date` is the date mapped from `d_1941`, not the last row of `calendar.csv`. It must resolve to `2016-05-22`.

Using:

- target train days: 15
- target validation days: 15
- target test days: 180
- source history days: 300

The correct windows are:

- `train_start`: `2015-10-26` (`d_1732`)
- `train_end`: `2015-11-09` (`d_1746`)
- `val_start`: `2015-11-10` (`d_1747`)
- `val_end`: `2015-11-24` (`d_1761`)
- `test_start`: `2015-11-25` (`d_1762`)
- `test_end`: `2016-05-22` (`d_1941`)
- `source_start`: `2014-12-30` (`d_1432`)
- `source_end`: `2015-10-25` (`d_1731`)

The example dates in `/Users/ming/Downloads/D6_target_selection_codex_v3.md` are obsolete and must not be used.

The screening window for SKU screening, MMD features, and structural shift is `d_1` through the `d_xxx` mapped to `train_end`, inclusive. With the real M5 data this is `d_1...d_1746`.

The script outputs `source_start`, `source_end`, and `source_history_days` only. Downstream experiment code owns source train/validation/test splitting.

## D6 Data Semantics

D6 uses M5 wide-table semantics:

- sales are stored in columns `d_1...d_1941`
- zero sales are explicit `0` values
- no natural-day completion is needed
- no negative-sales clipping is needed
- feature extraction should operate directly on row slices
- `dept_id` is the authoritative department field; do not parse `item_id` to recover department

All 30,490 M5 entities have 1,941 days, so the 510-day minimum history requirement is asserted but not used as a filter.

## Two-Tier MMD

D6 reports two separate MMD values:

| Field | Purpose | Scope |
| --- | --- | --- |
| `store_mmd` | choose target store | each store vs remaining stores using all departments, up to 50 sampled SKUs per store |
| `final_mmd` | report paper-facing domain shift | selected target department only, target store SKUs vs source pool SKUs |

Both MMD values use the same feature extraction, shared-scaler, gamma, and RBF MMD protocol.

## Store Selection

For each of the 10 stores:

1. Sort available SKUs by `item_id`.
2. Sample up to `store_sample_size=50` SKUs without replacement using `np.random.RandomState(random_seed)`.
3. Sample across all departments, not only FOODS.
4. Extract 7-dimensional screening-window features for sampled SKUs.
5. Compute `store_mmd` for that store against concatenated sampled features from the other 9 stores.

MMD quantiles:

- compute Q25, median, and Q75 from all 10 `store_mmd` values
- use `np.percentile(..., interpolation="linear")` or the NumPy-version-compatible equivalent
- candidate stores satisfy `q25 <= store_mmd <= q75`, inclusive

Ranking within IQR:

1. `abs(store_mmd - median_mmd)` ascending
2. state priority: `CA > TX > WI`
3. `store_id` ascending lexicographically

Try ranked stores in order. If a store cannot produce at least 3 target SKUs after all fallback rounds, record the failed attempt and try the next ranked store.

## Target SKU Selection

Target SKUs must all come from one department. Cross-department mixing is forbidden.

Preferred starting department: `FOODS_3`.

Screening metrics use screening-window sales only:

- `nonzero_ratio = (sales > 0).sum() / len(screening_cols)`
- `cv = std(sales, ddof=0) / mean(sales)`, with `mean == 0` as `inf` for screening
- `spike_ratio = sales.max() / median(sales[sales > 0])`, with no nonzero sales as `inf`

Fallback rounds:

| Round | Department Scope | `nonzero_ratio` | `cv` | `spike_ratio` |
| --- | --- | --- | --- | --- |
| 1 | `FOODS_3` only | `>= 0.30` | `<= 1.5` | `<= 10` |
| 2 | `FOODS_3` only | `>= 0.20` | `<= 2.0` | `<= 15` |
| 3 | `FOODS_1`, `FOODS_2`, `FOODS_3` | `>= 0.30` | `<= 1.5` | `<= 10` |

Round 3 filters each FOODS department independently. Only departments with at least 3 eligible SKUs are candidates. If multiple departments qualify, choose the department with the lowest eligible-SKU median CV.

A successful round selects the 3-5 lowest-CV SKUs from one department and records:

- `target_department`
- `fallback_round`
- `target_skus`

If all rounds fail for a store, that store is ineligible and the selector tries the next ranked store.

## Source Pool

The source pool is all concrete `(store_id, item_id)` entities where:

- `store_id != target_store`
- `dept_id == target_department`

`source_entities` is the canonical downstream source pool. It must not be represented only as independent store and item lists.

## Feature Extraction

Every MMD and structural-shift feature vector has 7 dimensions:

- `mean`: `np.mean(sales)`
- `std`: `np.std(sales, ddof=0)`
- `cv`: `std / mean`, with `mean == 0` as `0.0` for MMD features
- `zero_ratio`: `(sales == 0).sum() / len(sales)`
- `iqr`: `np.percentile(sales, 75, interpolation="linear") - np.percentile(sales, 25, interpolation="linear")`
- `acf_lag7`: lag-7 autocorrelation
- `trend_slope`: `np.polyfit(np.arange(len(sales)), sales, 1)[0]`

Edge cases:

- constant series: `acf_lag7 = 0.0`
- all-zero series: `cv = 0.0`, `acf_lag7 = 0.0`
- `len(sales) < 14`: `acf_lag7 = 0.0`
- NaN autocorrelation: `acf_lag7 = 0.0`
- after extraction, replace any remaining NaN or infinite values with `0.0`

Use natural day index `0,1,2,...` for `trend_slope`; do not standardize the x-axis.

## MMD Protocol

Use one `StandardScaler` fit jointly on target and source features:

```python
combined = np.vstack([target_features, source_features])
scaler = StandardScaler().fit(combined)
X = scaler.transform(target_features)
Y = scaler.transform(source_features)
```

Use biased MMD squared with RBF kernel:

`mean(Kxx) + mean(Kyy) - 2 * mean(Kxy)`

Gamma uses the median heuristic on standardized features:

- compute pairwise Euclidean distances on `np.vstack([X, Y])`
- ignore diagonal and zero distances
- `gamma = 1 / (2 * median_dist ** 2)`
- if `median_dist == 0`, use `gamma = 1.0`

For permutation testing:

- use `permutation_seed=43` by default, distinct from store sampling seed
- shuffle the standardized combined matrix
- keep gamma and scaler fixed
- default `n_perm=500`
- use corrected one-sided p-value `(count + 1) / (n_perm + 1)`

Tests may use smaller `n_perm`.

## Structural Shift

Structural shift is computed on same-`target_department` features from target store and source pool.

Output semantics are signed:

`target mean - source mean`

The JSON must include:

```json
"structural_shift_semantics": "signed_target_minus_source"
```

Dimensions:

- `scale`: mean target `mean` minus mean source `mean`
- `volatility`: mean target `cv` minus mean source `cv`
- `sparsity`: mean target `zero_ratio` minus mean source `zero_ratio`
- `seasonality`: mean target `acf_lag7` minus mean source `acf_lag7`
- `trend`: mean target `trend_slope` minus mean source `trend_slope`

The Markdown report may additionally show absolute values for readability, but JSON `structural_shift` remains signed.

## CLI Design

Create:

`scripts/select_d6_target_domain.py`

Default arguments:

- `--dataset-root`: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 6m5-forecasting-accuracy`
- `--output-dir`: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/domain_adaptation/Dataset6/target_selection`
- `--random-seed`: `42`
- `--permutation-seed`: `43`
- `--n-perm`: `500`
- `--store-sample-size`: `50`
- `--source-history-days`: `300`

The CLI validates required files and columns before computation.

The CLI should print:

- target store
- target department
- target SKU count
- source entity count
- store MMD
- final MMD
- output directory

## File Responsibilities

### `scripts/select_d6_target_domain.py`

Responsibilities:

- parse CLI args
- validate M5 input files and columns
- load `calendar.csv` and resolve date/d-column windows
- load `sales_train_evaluation.csv`
- compute screening-window features directly from wide rows
- run store-level MMD scan
- rank candidate stores
- run target SKU fallback
- build target/source entity lists
- compute final MMD and permutation p-value
- compute signed structural shift
- write CSV, JSON, and Markdown artifacts

### `tests/test_d6_target_selection.py`

Responsibilities:

- verify calendar max-date resolution
- verify window and d-column mapping
- verify feature extraction edge cases
- verify shared-scaler MMD and corrected permutation p-value
- verify three-round fallback and no cross-department mixing
- verify source entity construction
- verify output contract on synthetic wide-table fixtures

## Testing Strategy

Use TDD for implementation. Add focused tests before production code:

1. Calendar/window helper tests.
2. Feature extraction edge-case tests.
3. MMD/scaler/permutation tests.
4. Department fallback tests.
5. Source entity construction tests.
6. End-to-end output contract test with small synthetic M5 fixtures.

The local environment may not have `pytest`; `python -m unittest tests.test_d6_target_selection` is an accepted fallback.

Real-data verification after unit tests:

```bash
python scripts/select_d6_target_domain.py --n-perm 100
python -m json.tool outputs/domain_adaptation/Dataset6/target_selection/target_selection_result.json
```

Use `--n-perm 100` for smoke verification and `--n-perm 500` for formal reporting.

## Acceptance Criteria

The implementation is accepted when:

1. `python -m unittest tests.test_d6_target_selection` passes.
2. `python -m py_compile scripts/select_d6_target_domain.py tests/test_d6_target_selection.py` passes.
3. The D6 selector CLI exits with code 0 on the real Dataset6 raw files.
4. The four required output files exist under `outputs/domain_adaptation/Dataset6/target_selection/`.
5. `target_selection_result.json` is valid JSON and includes both `store_mmd` and `final_mmd`.
6. `target_skus` contains 3-5 SKUs from exactly one `target_department`.
7. `source_entities` contains concrete `{store_id,item_id}` pairs from the same `target_department` and excludes the target store.
8. `structural_shift_semantics` equals `signed_target_minus_source`.
9. The JSON time windows match the formula-derived dates listed in this design.
10. `store_candidate_profile.csv` contains all 10 stores and the IQR ranking fields.
