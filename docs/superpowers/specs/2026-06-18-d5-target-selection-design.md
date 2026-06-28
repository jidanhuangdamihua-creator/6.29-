# D5 Target Selection Design

> Status: draft for review
> Scope: formal Dataset5 Favorita target store/SKU selection utility, MMD/domain-shift reporting, and output artifacts aligned with the D4 selector shape.
> Out of scope: changing the raw Favorita files, changing D4 selection behavior, adding D6 selection implementation in this phase, or running the full transfer-learning experiment matrix.

## Goal

Build a formal D5 Favorita target selection script that chooses one target `store_nbr`, 3-5 target `item_nbr` values, a source pool of concrete `(store_nbr, item_nbr)` entities, MMD statistics, and structural-shift diagnostics using no-leakage training-history metrics.

## Current Context

Dataset5 raw Favorita files are available under:

`/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 5Favorita/`

Required files:

- `train.csv`: `id,date,store_nbr,item_nbr,unit_sales,onpromotion`
- `items.csv`: `item_nbr,family,class,perishable`
- `stores.csv`: `store_nbr,city,state,type,cluster`
- `transactions.csv`: `date,store_nbr,transactions`
- `holidays_events.csv`, `oil.csv`, and `test.csv` are present but are not required for target selection.

The repository already has a D4 formal selector at `scripts/select_d4_target_domain.py`. D5 should follow that delivery pattern, not the older Dataset5 profile scanner. The existing `scripts/scan_dataset5_favorita_cold_start.py` remains a profile/candidate scanner and should not be overloaded with final target selection responsibilities.

## Approach Options

### Option 1: Add a Dedicated D5 Selector

Create `scripts/select_d5_target_domain.py` with pure helper functions plus a CLI entry point. Add `tests/test_d5_target_selection.py` for synthetic unit coverage and output contract coverage.

Pros: cleanest alignment with D4, low blast radius, easy to audit.

Cons: some helper logic overlaps with D4 until common abstractions are intentionally extracted.

### Option 2: Create a Shared D4/D5/D6 Selector Framework

Extract common MMD/window/output code and make D4-D6 configuration-driven.

Pros: less duplicated logic in the long run.

Cons: larger refactor now, more regression risk, and unnecessary abstraction before D5 behavior is proven.

### Option 3: Extend the Existing Dataset5 Scanner

Modify `scripts/scan_dataset5_favorita_cold_start.py` to also perform target selection.

Pros: fewer new files.

Cons: mixes profiling and formal experiment selection, making the output harder to interpret and test.

Recommended approach: Option 1. Implement D5 as a dedicated selector now, then consider shared selector helpers after D4-D6 are all stable.

## Output Contract

The CLI writes all final artifacts under:

`outputs/domain_adaptation/Dataset5/target_selection/`

Required files:

- `target_selection_result.json`
- `store_candidate_profile.csv`
- `target_sku_metrics.csv`
- `target_selection_report.md`

The JSON must include:

- `target_store`
- `target_skus`
- `target_family`
- `target_families`
- `family_selection_rule`
- `source_stores`
- `source_skus`
- `source_entities`
- `mmd_value`
- `mmd_gamma`
- `permutation_p_value`
- `structural_shift`
- `time_windows`
- `selection_rules`

`source_entities` is the canonical source pool and must be a list of concrete entity objects:

```json
[
  {"store_nbr": 1, "item_nbr": 12345},
  {"store_nbr": 2, "item_nbr": 67890}
]
```

`source_stores` and `source_skus` are summary lists only. They are not sufficient to reconstruct the experiment source pool because not every store sells every SKU.

## Time Windows

Use the formal experiment window:

- target train days: 15
- target validation days: 15
- target test days: 180
- target total days: 210
- max source history days: 300
- minimum SKU date coverage: 510 natural days

The global max date is `2017-08-15`. All target windows are counted backward from that date:

- `train_start`: `2017-01-17`
- `val_start`: `2017-02-01`
- `test_start`: `2017-02-16`
- `source_start`: `2016-04-22` when `source_history_days=300`

Selection metrics, MMD features, and structural-shift features use training-history data only:

- lower bound: each entity's own `min_date`
- upper bound: `2017-01-16`, the day before `val_start`

Do not force all selection metrics to start at `source_start`. `source_start` is for later experiment source-history extraction, not for judging long-run target SKU stability.

## Sales Cleaning for Selection Metrics

For target selection only, clean sales values as follows:

- negative `unit_sales` values are set to `0`
- rows are not dropped when `unit_sales` is negative
- missing natural dates are inserted for feature calculation only
- inserted missing dates use `unit_sales=0`
- inserted missing dates use `onpromotion=False`

This cleaning does not mutate raw files. It only gives CV, ACF, trend, MMD, and structural-shift features a uniform daily time axis.

Favorita records only positive sale observations in the main table. Therefore:

- `nonzero_ratio` from observed rows is not a meaningful filter
- use `coverage_ratio = observed_days / total_calendar_days`
- all feature time series should be natural-day complete with missing sales filled as zero

## Store Selection

### Candidate Store Filter

Start from `stores.csv` and prefer:

- `city == "Quito"`
- `type in {"A", "B"}`

The selection ranking uses Quito and type A as hard priorities. The implementation should still write non-selected candidate diagnostics where practical, but final choice should prefer Quito and type A according to the ranking rules.

### Transaction Continuity

Use `transactions.csv` at store level to detect store closure or data gaps. Main-table missing store-item dates are not treated as gaps because they may represent zero sales.

The hard continuity window is `2017-01-17` through `2017-08-15`, inclusive.

Hard exclusion threshold:

- exclude a store only if it has a continuous transaction gap greater than 30 days in the 210-day target window

Audit fields:

- `max_transaction_gap_days`
- `has_gap_gt_7_days`

`has_gap_gt_7_days` is an audit flag only. It must not exclude stores by default. This preserves a manual review hook without making the formal gate stricter than D4.

### Store-Level MMD

Store-level MMD is based on all product families, not only GROCERY I or BEVERAGES.

For each candidate store:

1. Use training-history data through `2017-01-16`.
2. Build per-SKU natural-day complete sales series, clipping negative sales to zero and filling missing dates with zero.
3. Randomly sample up to 50 SKUs from the store, without family filtering.
4. Compute the seven summary features for each sampled SKU:
   - `mean`
   - `std`
   - `cv`
   - `coverage_ratio`
   - `iqr`
   - `acf_lag7`
   - `trend_slope`
5. Compare the candidate store feature matrix against source-store feature matrices using one shared `StandardScaler` fit on target and source features together.
6. Compute RBF-kernel MMD with the median heuristic:
   `gamma = 1 / (2 * median_pairwise_distance^2)`.

Use a fixed default random seed for SKU sampling so repeated runs are deterministic.

MMD filtering:

- compute MMD for all candidate stores
- compute candidate-store Q25, median, and Q75
- keep stores in the Q25-Q75 interval
- rank those by closeness to the MMD median

Do not use a fixed absolute MMD threshold.

### Final Store Ranking

Final ranking priority:

1. `city == "Quito"`
2. `type == "A"`
3. passes transaction continuity hard gate
4. MMD is in Q25-Q75 and closest to all-candidate median
5. largest count of eligible target SKUs
6. lowest CV among eligible target SKUs

The chosen store must have at least three eligible target SKUs after family fallback rules are applied.

## Target SKU Selection

### Family Preference and Fallback

Preferred families:

1. `GROCERY I`
2. `BEVERAGES`

Excluded families:

- `BOOKS`
- `MAGAZINES`
- `LADIESWEAR`
- `BABY CARE`
- `HARDWARE`

Family fallback rules:

1. Evaluate `GROCERY I` first.
2. If fewer than 3 SKUs pass, evaluate `BEVERAGES`.
3. If `BEVERAGES` also has fewer than 3 passing SKUs, merge the passing candidates from `GROCERY I` and `BEVERAGES`.
4. If the merged pool has at least 3 SKUs, choose the 3-5 lowest-CV SKUs and mark the result as mixed family.
5. If the merged pool still has fewer than 3 SKUs, the store cannot be selected.

`target_family` should be the single family name for single-family selection. For mixed-family fallback, set `target_family` to `MIXED` and list the actual families in `target_families`.

### SKU Metrics

Compute SKU screening metrics from each entity's own `min_date` through `2017-01-16`.

For each target-store SKU:

- `date_coverage_days >= 510`
- `coverage_ratio >= 0.60`
- `onpromotion_ratio <= 0.30`
- `cv <= 1.0`
- `spike_ratio <= 8`

`date_coverage_days` is inclusive:

`(max_date - min_date).days + 1`

`coverage_ratio` is:

`observed_days / date_coverage_days`

`onpromotion_ratio` is calculated over the natural-day complete training-history series after missing dates are inserted with `onpromotion=False`.

`cv` is calculated over the natural-day complete sales series after negative sales are clipped to zero and missing dates are filled with zero.

`spike_ratio` is:

`sales.max() / median(sales[sales > 0])`

If a family has fewer than 3 passing SKUs, relax thresholds in this order for that family evaluation:

1. `cv <= 1.5`
2. `coverage_ratio >= 0.50`
3. `onpromotion_ratio <= 0.40`

The final chosen target SKUs are the lowest-CV 3-5 SKUs from the winning family pool or mixed fallback pool.

## Source Pool

The source pool uses concrete `(store_nbr, item_nbr)` entities, excluding the target store.

Source family alignment follows the final target SKU family set:

- if target SKUs are all `GROCERY I`, source pool uses only `GROCERY I`
- if target SKUs are all `BEVERAGES`, source pool uses only `BEVERAGES`
- if target SKUs are mixed `GROCERY I` and `BEVERAGES`, source pool uses the union of those actual families

Do not always take `GROCERY I + BEVERAGES` when a single family succeeded. This keeps target and source domain definitions symmetric.

Source entities should satisfy enough history for the later source-history extraction. At minimum, source entity `date_coverage_days` must cover `source_start` through `2017-01-16` when present in the raw data; if an entity has missing natural dates inside that range, the later modeling path can fill them with zero.

## MMD and Structural Shift Reporting

Use seven features for MMD:

- `mean`
- `std`
- `cv`
- `coverage_ratio`
- `iqr`
- `acf_lag7`
- `trend_slope`

Use one shared `StandardScaler` fit on source plus target features. Do not fit source and target scalers separately.

Use the median heuristic for RBF gamma. Report:

- `mmd_value`
- `mmd_gamma`
- `permutation_p_value`

Permutation test:

- default real run: at least 500 permutations
- tests may pass a smaller permutation count through CLI/API configuration

Report structural shift as a five-dimensional dictionary:

- `trend`: target slope minus source mean slope
- `seasonality`: target lag-7 ACF minus source mean lag-7 ACF
- `volatility`: target CV minus source mean CV
- `coverage`: target coverage ratio minus source mean coverage ratio
- `scale`: target mean sales minus source mean sales

## CLI Design

Create:

`scripts/select_d5_target_domain.py`

Default arguments:

- `--dataset-root`: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 5Favorita`
- `--output-dir`: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/domain_adaptation/Dataset5/target_selection`
- `--permutations`: `500`
- `--store-sku-sample`: `50`
- `--random-state`: `42`
- `--chunksize`: `5000000`

The CLI should print a concise completion summary containing:

- selected target store
- target SKU count
- target family or family list
- source entity count
- MMD value
- output directory

## File Responsibilities

### `scripts/select_d5_target_domain.py`

Responsibilities:

- parse CLI args
- load Favorita files
- stream or chunk `train.csv` where full-table operations would be expensive
- compute target windows
- clean sales values for selection metrics
- construct natural-day complete per-entity series for selected calculations
- compute store transaction gaps
- compute store-level MMD scan
- compute target SKU metrics and fallback
- build source entity pool
- compute final MMD and structural shift
- write CSV, JSON, and Markdown artifacts

### `tests/test_d5_target_selection.py`

Responsibilities:

- verify time windows
- verify negative sales clipping and natural-day zero fill
- verify transaction gap hard gate and audit fields
- verify store-level MMD uses all families and deterministic SKU sampling
- verify target family fallback
- verify source family alignment follows final target families
- verify JSON/CSV/MD output contract on synthetic fixtures

## Testing Strategy

Use TDD for implementation. Add focused tests before production code:

1. Window and cleaning helpers.
2. Transaction gap helper with both `>7` audit and `>30` hard-gate behavior.
3. SKU metric and threshold relaxation behavior.
4. Family fallback and source-family alignment.
5. MMD helper behavior: shared scaler, median gamma, deterministic sampling.
6. End-to-end output contract using small synthetic CSV fixtures.

Real-data verification after unit tests:

```bash
python scripts/select_d5_target_domain.py --permutations 100
python -m json.tool outputs/domain_adaptation/Dataset5/target_selection/target_selection_result.json
```

Use `--permutations 100` for smoke verification and `--permutations 500` for formal reporting.

## Acceptance Criteria

The implementation is accepted when:

1. `pytest tests/test_d5_target_selection.py -q` passes.
2. The D5 selector CLI exits with code 0 on the real Dataset5 raw files.
3. The four required output files exist under `outputs/domain_adaptation/Dataset5/target_selection/`.
4. `target_selection_result.json` is valid JSON and includes all fields in the output contract.
5. `source_entities` contains concrete `store_nbr` and `item_nbr` pairs and excludes the target store.
6. `store_candidate_profile.csv` includes `max_transaction_gap_days` and `has_gap_gt_7_days`.
7. The report states the selected family rule, including mixed-family fallback when used.
8. The report states that negative sales were clipped to zero and missing natural dates were filled with zero for feature calculation only.
