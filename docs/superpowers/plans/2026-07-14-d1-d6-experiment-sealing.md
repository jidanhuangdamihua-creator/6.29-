# D1–D6 Experiment Sealing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one leak-free, reproducible D1–D6 formal entry that uses fixed 30-day target observation, exact 180-day source pretraining, 180-day blind rollout, unified clipped original-space metrics, D5-first bounded scheduling, and acceptance-gated sealed outputs.

**Architecture:** Add a small immutable sealing protocol above the existing shared protocol, then make data loaders, KNN, model adapters, rollout, artifacts, and the supervisor consume it. Keep TensorFlow workers isolated, but reuse validated read-only mode caches. Replace 25 single-horizon cells per mode with five seed bundles; each bundle trains all horizons and uses its horizon-1 predictor as the only blind recursive feedback path.

**Tech Stack:** Python 3.9+, pandas, NumPy, PyArrow, TensorFlow/Keras through existing model code, pytest/unittest, Bash supervisor, SHA-256/JSON manifests, gzip CSV.

## Global Constraints

- D1–D6 target protocol is exactly 15 train days + 15 validation days + 180 blind-test natural days.
- D1 observed start is `2017-06-01`; D2 observed start is `2018-06-01`; D3 observed start remains `2015-01-03`.
- D4 target range is `2024-12-16..2025-07-13`; D5 is `2017-01-17..2017-08-14`; D6 is `2015-10-26..2016-05-22`.
- KNN uses exactly the target's 30 observed natural days and the same 30 source dates; target blind dates and source rows after target observed end are forbidden.
- KNN uses declared observed features in fixed column order and raw `float64` Euclidean distance; IDs and role labels are forbidden.
- D2 KNN must reproduce Item 4/6/8 with distances approximately 24.98/26.85/26.85 from raw sales + promotion.
- Every selected source pretraining pool is exactly 180 natural days ending at target observed end; the entire 180-day pool is eligible for fixed-epoch source fitting, and no 300-day formal source constant remains.
- Target blind truth is evaluator-owned; training, model selection, recursive input, scaler fit, and RFE fit cannot access it.
- At each blind origin, horizons 1–5 predict together; only inverse-transformed and nonnegative-clipped horizon 1 is fed back.
- Formal sample counts are h1=180, h2=179, h3=178, h4=177, h5=176.
- Formal sMAPE is original-sales-space `100 * mean(2*abs(pred-true)/(abs(true)+abs(pred)+1e-8))` on clipped predictions.
- Formal horizons remain `(1, 2, 3, 4, 5)` and seeds remain `(42, 43, 44, 45, 46)`.
- Formal scheduling starts `d5_without` first; `d5_with` starts immediately after successful `d5_without`; they never overlap.
- D5 gets 6 compute threads, ordinary workers get 2, and the concurrent total cannot exceed 16.
- Successful artifacts are immutable and preserved after later failure; incomplete runs are `partial_failed` and cannot create `SEALED_SUCCESS`.
- Existing D5 results cannot enter the new formal aggregate unless every input/protocol/cache digest matches.
- Codex does not run a full experiment. Every Python validation or data command in this plan is executed through `python tools/protection/codex_timeout.py --timeout 180 -- ...`. Exit 124 stops the work immediately without retrying, splitting, simplifying, resuming, or continuing.

---

## File and Interface Map

- `src/protocols/sealing_protocol.py`: immutable D1–D6 target/source windows, feature roles, formal status constants.
- `src/data_processing/sealed_daily.py`: shared entity/day aggregation, calendarization, missing-day provenance, model/evaluator view separation.
- `src/utils/sealed_parquet.py`: PyArrow column/date pushdown and validated table identity.
- `src/utils/mode_cache.py`: immutable run/mode cache manifests and atomic publication.
- `src/protocols/candidate_pool.py`: exact 30-day multifeature vectors and deterministic KNN digests.
- `src/protocols/provenance.py`: exact 180-day source slices and training provenance.
- `src/protocols/blind_rollout.py`: evaluator-sealed 180-day rollout and prediction trace.
- `src/experiment/fitted_predictor.py`: common adapters for single, weighted, switched, and fused Keras predictors.
- `src/evaluation/metrics.py`: one clipping and original-space metric boundary.
- `src/utils/prediction_artifacts.py`: atomic gzip prediction/source/failure trace publication.
- `scripts/run_strict_protocol_baseline.py`: five seed bundles per mode.
- `scripts/run_unified_d1_d6.py`: 60-cell plan, mode cache lifecycle, aggregation, sealed success.
- `scripts/parallel_mode_runner.sh`: D5 dependency lane and thread-aware scheduler.

---

### Task 1: Freeze Sealing Windows, Feature Roles, and Statuses

**Files:**
- Create: `src/protocols/sealing_protocol.py`
- Modify: `src/protocols/__init__.py`
- Modify: `src/constants.py`
- Test: `tests/test_sealing_protocol.py`

**Interfaces:**
- Produces: `TargetWindow`, `SourcePretrainWindow`, `FeatureRoles`, `get_target_window(dataset_id)`, `get_source_pretrain_window(dataset_id)`, `classify_feature_roles(...)`, `SOURCE_PRETRAIN_DAYS`, `SEALED_PROTOCOL_VERSION`.
- Consumed by: Tasks 2–12.

- [ ] **Step 1: Write failing protocol tests**

```python
from datetime import date

from src.protocols.sealing_protocol import (
    SOURCE_PRETRAIN_DAYS,
    get_source_pretrain_window,
    get_target_window,
)


def test_all_target_windows_are_exact_30_plus_180_natural_days():
    expected = {
        1: ("2017-06-01", "2017-12-27"),
        2: ("2018-06-01", "2018-12-27"),
        3: ("2015-01-03", "2015-07-31"),
        4: ("2024-12-16", "2025-07-13"),
        5: ("2017-01-17", "2017-08-14"),
        6: ("2015-10-26", "2016-05-22"),
    }
    for dataset_id, (start, end) in expected.items():
        window = get_target_window(dataset_id)
        assert window.start == date.fromisoformat(start)
        assert window.observed_days == 30
        assert window.test_days == 180
        assert window.end == date.fromisoformat(end)
        assert (window.end - window.start).days == 209


def test_source_pretrain_is_180_days_ending_at_observed_end():
    assert SOURCE_PRETRAIN_DAYS == 180
    window = get_source_pretrain_window(2)
    assert window.end.isoformat() == "2018-06-30"
    assert window.start.isoformat() == "2018-01-02"
    assert (window.end - window.start).days == 179
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealing_protocol.py -q
```

Expected: FAIL because `src.protocols.sealing_protocol` does not exist.

- [ ] **Step 3: Implement the immutable protocol**

Create the module with these exact public shapes:

```python
@dataclass(frozen=True)
class TargetWindow:
    dataset_id: int
    start: date
    train_days: int = 15
    validation_days: int = 15
    test_days: int = 180

    @property
    def observed_days(self) -> int:
        return self.train_days + self.validation_days

    @property
    def observed_end(self) -> date:
        return self.start + timedelta(days=self.observed_days - 1)

    @property
    def test_start(self) -> date:
        return self.observed_end + timedelta(days=1)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=self.observed_days + self.test_days - 1)


@dataclass(frozen=True)
class SourcePretrainWindow:
    dataset_id: int
    start: date
    end: date
    days: int = 180


@dataclass(frozen=True)
class FeatureRoles:
    knn_observed: tuple[str, ...]
    model_historical: tuple[str, ...]
    future_known: tuple[str, ...]
    evaluation_only: tuple[str, ...]
```

Set `SEALED_PROTOCOL_VERSION = "d1_d6_sealed_blind_v1"` and `SOURCE_PRETRAIN_DAYS = 180`. Replace `SOURCE_HISTORY_DAYS = 300` in `src/constants.py` with an import-compatible alias to 180, and change D5 `test_end` to `2017-08-14`.

Feature classification is fail-closed and uses these exact rules:

```python
IDENTIFIER_COLUMNS = frozenset({
    "entity_id", "item_id", "store_id", "store_nbr", "product_id",
    "brand_id", "brand_code", "region_id", "region_code",
    "category", "family", "department",
})
CALENDAR_COLUMNS = frozenset({"year", "month", "week", "day", "day_of_week"})
DECLARED_FUTURE_KNOWN = {
    1: CALENDAR_COLUMNS,
    2: CALENDAR_COLUMNS,
    3: CALENDAR_COLUMNS | {"open", "promo", "state_holiday", "school_holiday"},
    4: CALENDAR_COLUMNS,
    5: CALENDAR_COLUMNS,
    6: CALENDAR_COLUMNS,
}
```

For every dataset, observed finite numeric fields except identifiers may be `knn_observed` and `model_historical`. `sales` is historical and becomes `evaluation_only` after observed end. Only `DECLARED_FUTURE_KNOWN[dataset_id]` may enter blind model rows. D2 promo remains an observed KNN/historical feature but is not assumed known during blind dates. Any unclassified blind feature causes preflight failure.

- [ ] **Step 4: Verify GREEN and regress existing protocol tests**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealing_protocol.py tests/test_experiment_protocol_contract.py tests/test_d5_runtime_reconstruction_contract.py -q
```

Expected: all tests pass after updating existing expectations from 300 to 180 and D5 from 211 to 210 days.

- [ ] **Step 5: Commit**

```bash
git add src/protocols/sealing_protocol.py src/protocols/__init__.py src/constants.py tests/test_sealing_protocol.py tests/test_d5_runtime_reconstruction_contract.py
git commit -m "feat: freeze D1-D6 sealing windows"
```

### Task 2: Build Fixed D1/D2 Parquets from Raw Data

**Files:**
- Modify: `scripts/regenerate_d1_d2_parquets.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `tests/test_d1_d2_formal_input_paths.py`
- Create: `tests/test_d1_d2_sealed_builder.py`

**Interfaces:**
- Consumes: `get_target_window`, `SEALED_PROTOCOL_VERSION`.
- Produces: `calendarize_protocol_entities(...)`, `write_protocol_pair(...)`, `数据集/固化数据/dataset{1,2}-{source,target}.parquet`, `数据集/固化数据/dataset{1,2}-manifest.json`.

- [ ] **Step 1: Write failing path and D2 fingerprint tests**

```python
def test_formal_d1_d2_paths_are_project_solidified_paths():
    paths = _full_runner_solidified_paths()
    for dataset_name, dataset_id in (("Dataset1", 1), ("Dataset2", 2)):
        assert Path(paths[dataset_name]["source"]) == Path(
            f"数据集/固化数据/dataset{dataset_id}-source.parquet"
        )
        assert Path(paths[dataset_name]["target"]) == Path(
            f"数据集/固化数据/dataset{dataset_id}-target.parquet"
        )


def test_d2_builder_calendarizes_june_and_preserves_paper_knn_fingerprint(raw_d2):
    source, target, report = build_d2_protocol_frames(raw_d2, return_report=True)
    observed = pd.date_range("2018-06-01", "2018-06-30", freq="D")
    target_observed = target[target["date"].isin(observed)]
    assert tuple(target_observed["date"]) == tuple(observed)
    missing = target_observed[target_observed["is_synthetic_date"]]
    assert missing["date"].dt.strftime("%Y-%m-%d").tolist() == ["2018-06-02"]
    assert missing[["sales", "promo"]].to_numpy().tolist() == [[0.0, 0.0]]
    ranked = raw_d2_paper_distances(source, target)
    assert [item_id for item_id, _ in ranked[:3]] == [4, 6, 8]
    np.testing.assert_allclose(
        [distance for _, distance in ranked[:3]],
        [24.9799919936, 26.8514431642, 26.8514431642],
        atol=0.01,
        rtol=0.0,
    )
    assert report.protocol_version == "d1_d6_sealed_blind_v1"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_d1_d2_formal_input_paths.py tests/test_d1_d2_sealed_builder.py -q
```

Expected: path tests still point at `数据集/派生数据/d1d2_protocol_v1`; builder lacks calendar/report support.

- [ ] **Step 3: Implement deterministic solidified generation**

Change `DEFAULT_OUTPUT_DIR` to `ROOT / "数据集" / "固化数据"`. For every entity, reindex to the entity's raw daily range, add `is_synthetic_date`, fill absent sales/promo with zero, regenerate date features, and preserve raw-vs-synthetic counts. Write parquet and JSON through temporary siblings followed by `os.replace`. The manifest must contain:

```python
{
    "protocol_version": SEALED_PROTOCOL_VERSION,
    "dataset_id": dataset_id,
    "raw_input": {"path": str(input_path), "sha256": sha256_file(input_path)},
    "target_window": asdict(get_target_window(dataset_id)),
    "missing_dates": report.missing_dates,
    "source_sha256": sha256_file(source_path),
    "target_sha256": sha256_file(target_path),
    "columns": list(source.columns),
}
```

Change both formal path resolvers to use only `数据集/固化数据` for D1–D6. Do not fall back to the old derived directory.

`raw_d2_paper_distances` is a test-only helper defined in `tests/test_d1_d2_sealed_builder.py`; it filters the exact 30 June dates, flattens `(sales, promo)` in date order for target and Brand 1 items 1–9, and returns `(item_id, np.linalg.norm(source_vector - target_vector))` sorted by distance then item ID.

- [ ] **Step 4: Verify builder and path tests**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_d1_d2_formal_input_paths.py tests/test_d1_d2_sealed_builder.py tests/test_full_paper_runner_solidified_parquet.py -q
```

Expected: all pass on synthetic/temp data without writing formal artifacts.

- [ ] **Step 5: Generate the local fixed D1/D2 artifacts once**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python scripts/regenerate_d1_d2_parquets.py --dataset all --output-dir 数据集/固化数据
```

Expected: four parquet files and two manifest files are written. If exit code is 124, stop the entire implementation session and give this exact command to the user; do not retry or split it.

- [ ] **Step 6: Commit code and tests**

```bash
git add scripts/regenerate_d1_d2_parquets.py scripts/run_full_paper_experiments.py scripts/run_unified_d1_d6.py tests/test_d1_d2_formal_input_paths.py tests/test_d1_d2_sealed_builder.py
git commit -m "feat: fix D1-D2 solidified protocol inputs"
```

### Task 3: Unify Daily Cleaning and Seal Evaluator Truth

**Files:**
- Create: `src/data_processing/sealed_daily.py`
- Modify: `src/data_processing/data_preprocessing.py`
- Modify: `src/utils/parquet_data_loader.py`
- Create: `tests/test_sealed_daily_contract.py`
- Modify: `tests/test_protocol_preprocessing_contract.py`

**Interfaces:**
- Produces: `DailyCleaningReport`, `SealedTargetViews`, `frame_digest(...)`, `aggregate_entity_day(...)`, `calendarize_entity_day(...)`, `build_sealed_target_views(...)`.
- `SealedTargetViews.model_frame` excludes blind sales; `SealedTargetViews.truth_frame` contains only key/date/sales and is passed only to the evaluator.

- [ ] **Step 1: Write failing duplicate, missing-day, and truth-isolation tests**

```python
def test_shared_cleaning_aggregates_then_calendarizes_with_provenance():
    raw = pd.DataFrame({
        "entity_id": ["A", "A"],
        "date": ["2020-01-01", "2020-01-01"],
        "sales": [2.0, 3.0],
        "promo": [0, 1],
    })
    cleaned, report = calendarize_entity_day(
        raw,
        entity_cols=("entity_id",),
        start="2020-01-01",
        end="2020-01-03",
        aggregations={"sales": "sum", "promo": "max"},
        fill_values={"sales": 0.0, "promo": 0},
    )
    assert cleaned["sales"].tolist() == [5.0, 0.0, 0.0]
    assert cleaned["is_synthetic_date"].tolist() == [False, True, True]
    assert report.duplicate_input_rows == 2
    assert report.synthetic_rows == 2


def test_model_view_cannot_contain_blind_sales():
    views = build_sealed_target_views(frame_210_days, get_target_window(1), feature_roles)
    blind = views.model_frame[views.model_frame["date"] > "2017-06-30"]
    assert blind["sales"].isna().all()
    assert views.truth_frame.columns.tolist() == ["entity_id", "date", "sales", "is_synthetic_date"]
    changed_truth = views.truth_frame.copy()
    changed_truth.loc[:, "sales"] = changed_truth["sales"] + 10_000.0
    assert frame_digest(views.model_frame) == views.model_digest
    assert frame_digest(changed_truth) != frame_digest(views.truth_frame)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_daily_contract.py tests/test_protocol_preprocessing_contract.py -q
```

Expected: the new shared cleaning module is missing and current target frames retain blind sales.

- [ ] **Step 3: Implement one shared stage order**

Implement `DailyCleaningReport` as a frozen dataclass with input/output rows, entities, duplicate rows, invalid dates, synthetic rows/dates, per-column fill counts, negative actual-sales rows, and SHA-256 digest. `build_sealed_target_views` must copy only observed sales to `model_frame`, retain allowed future-known features, replace all evaluation-only fields with absent columns, and return the truth separately. Reject negative actual sales unless the dataset adapter explicitly sets `allow_negative_actual=True`.

- [ ] **Step 4: Route all D1–D6 loaders through the shared functions**

Replace dataset-specific date reindexing in `src/utils/parquet_data_loader.py` with `calendarize_entity_day`. Keep adapter aggregation maps explicit, but attach the same report fields and `15/15/180` split attrs for every dataset.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_daily_contract.py tests/test_protocol_preprocessing_contract.py tests/test_full_paper_runner_solidified_parquet.py tests/test_d5_calendar_reconstruction.py -q
```

Expected: all tests pass and no model frame exposes blind truth.

```bash
git add src/data_processing/sealed_daily.py src/data_processing/data_preprocessing.py src/utils/parquet_data_loader.py tests/test_sealed_daily_contract.py tests/test_protocol_preprocessing_contract.py
git commit -m "feat: unify D1-D6 daily cleaning and truth sealing"
```

### Task 4: Add Parquet Pushdown and Immutable Mode Caches

**Files:**
- Create: `src/utils/sealed_parquet.py`
- Create: `src/utils/mode_cache.py`
- Modify: `src/utils/parquet_data_loader.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `src/utils/entity_experiment.py`
- Create: `tests/test_sealed_parquet_pushdown.py`
- Create: `tests/test_mode_cache_contract.py`

**Interfaces:**
- Produces: `read_sealed_parquet(path, columns, start, end, date_col="date")`, `CacheIdentity`, `ModeCache`, `open_or_build_mode_cache(...)`.
- Cache artifacts: base cleaned parquet, D5 authority JSON, scenario candidate metadata, 30-day KNN pool, 180-day source pool, target model view, sealed truth reference.

- [ ] **Step 1: Write failing pushdown and cache-once tests**

```python
def test_reader_passes_columns_and_date_filter_to_pyarrow(monkeypatch, tmp_path):
    seen = {}
    class FakeDataset:
        schema = pa.schema([("date", pa.date32()), ("sales", pa.float64()), ("promo", pa.int8())])
        def to_table(self, *, columns, filter):
            seen["columns"] = tuple(columns)
            seen["filter"] = str(filter)
            return pa.table({"date": [], "sales": []})
    monkeypatch.setattr(ds, "dataset", lambda *a, **k: FakeDataset())
    read_sealed_parquet(tmp_path / "x.parquet", ("date", "sales"), "2020-01-01", "2020-01-30")
    assert seen["columns"] == ("date", "sales")
    assert "2020-01-01" in seen["filter"] and "2020-01-30" in seen["filter"]


def test_mode_cache_builds_once_and_invalidates_on_protocol_change(tmp_path):
    calls = []
    first = open_or_build_mode_cache(tmp_path, identity_v1, lambda root: calls.append(root) or payload)
    second = open_or_build_mode_cache(tmp_path, identity_v1, lambda root: calls.append(root) or payload)
    assert first.manifest_sha256 == second.manifest_sha256
    assert len(calls) == 1
    with pytest.raises(CacheIdentityMismatch):
        open_or_build_mode_cache(tmp_path, identity_v2, lambda root: payload, allow_rebuild=False)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_parquet_pushdown.py tests/test_mode_cache_contract.py -q
```

Expected: modules and interfaces are missing.

- [ ] **Step 3: Implement pushdown and atomic immutable caches**

Use `pyarrow.dataset.dataset(path, format="parquet")`, a conjunction of `date >= start` and `date <= end`, and `to_table(columns=list(columns), filter=predicate)`. Convert only the filtered table to pandas. Cache identity canonical JSON must include input SHA-256, protocol version, start/end, ordered columns, feature roles, fill rules, scenario, code revision, and D5 authority digest. Build into `<cache>.tmp.<pid>.<uuid>` and expose it with `os.replace` only after all artifact hashes match the manifest.

- [ ] **Step 4: Make mode workers consume one cache**

Construct the cache before seed bundles start. Pass only cache paths and digests to child commands. D5 authority and target reconstruction are built once per run-level base identity; scenario KNN/prepared pools are built once per mode identity.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_parquet_pushdown.py tests/test_mode_cache_contract.py tests/test_read_dataset_windows_scenario_identity.py tests/test_d5_runtime_reconstruction_contract.py -q
```

Expected: all pass; spy records real pushdown arguments and builders are invoked once.

```bash
git add src/utils/sealed_parquet.py src/utils/mode_cache.py src/utils/parquet_data_loader.py scripts/run_full_paper_experiments.py src/utils/entity_experiment.py tests/test_sealed_parquet_pushdown.py tests/test_mode_cache_contract.py
git commit -m "feat: add pushed-down reads and immutable mode caches"
```

### Task 5: Separate 30-Day Multifeature KNN from 180-Day Source Training

**Files:**
- Modify: `src/protocols/candidate_pool.py`
- Modify: `src/protocols/provenance.py`
- Modify: `src/source_selection/source_selector.py`
- Modify: `src/protocols/runner_adapter.py`
- Modify: `src/utils/parquet_data_loader.py`
- Modify: `tests/test_daily_knn_protocol.py`
- Modify: `tests/test_knn_cnn_provenance.py`
- Create: `tests/test_d2_paper_knn_fingerprint.py`
- Create: `tests/test_source_pretrain_180d.py`

**Interfaces:**
- Produces: `build_observed_feature_vector(frame, dates, feature_cols)`, `SelectionEntry.raw_vector` containing all ordered KNN features, `extract_selected_source_slices(..., training_start, training_end)` returning exact 180-day `SourceSliceRef` objects.

- [ ] **Step 1: Write failing multifeature and 180-day tests**

```python
def test_knn_vector_flattens_dates_then_declared_features_without_scaling():
    vector = build_observed_feature_vector(frame, dates, ("sales", "promo"))
    assert vector == (1.0, 0.0, 2.0, 1.0, 3.0, 0.0)


def test_source_slice_is_180_days_but_knn_subvector_is_last_30():
    selection = select_daily_sequence_sources(
        target_30, source_180, feature_cols=("sales", "promo"), k=1
    )
    slices = extract_selected_source_slices(
        selection,
        source_180,
        training_start="2018-01-02",
        training_end="2018-06-30",
        model_feature_cols=("sales", "promo", "year", "month", "week", "day"),
    )
    assert len(slices[0].dates) == 180
    assert slices[0].date_start == "2018-01-02"
    assert slices[0].date_end == "2018-06-30"
    assert selection.entries[0].observed_start == "2018-06-01"
    assert selection.entries[0].observed_end == "2018-06-30"
```

The D2 fingerprint test loads only June 2018 columns/rows from the raw fixture and asserts source items `(4, 6, 8)` plus distances with `abs=0.01`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_daily_knn_protocol.py tests/test_knn_cnn_provenance.py tests/test_d2_paper_knn_fingerprint.py tests/test_source_pretrain_180d.py -q
```

Expected: current sales-only vector/provenance and 300-day runtime assertions fail.

- [ ] **Step 3: Implement fixed-order raw multifeature vectors**

Flatten each date row in ascending date order and each feature in declared order, validate finite `float64`, and include `feature_cols` plus vector shape in both candidate and selection digests. Remove identifier-like columns from formal KNN inputs. Set formal representation to `daily_observed_features_flattened_30d_raw_v1`; delete the D4–D6 summary representation from the formal branch.

- [ ] **Step 4: Make source training consume the exact selected 180 days**

Resolve `training_start/end` exclusively through `get_source_pretrain_window`. Validate all 180 dates before building sequences. Feed the full source pool to fixed-epoch source fitting; do not apply the old `0.8/0.1/0.1` source test split in the sealed branch. Record natural days and structured source sample counts separately.

- [ ] **Step 5: Add positive and negative leakage controls**

Mutating source or target rows after observed end must leave candidate, selection, source-training, and prediction-input digests unchanged. Mutating a source observation within the 30-day KNN window must be able to change ranking. Mutating source rows in the first 150 pretraining days must leave KNN unchanged but change the source-training digest.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_daily_knn_protocol.py tests/test_knn_cnn_provenance.py tests/test_d2_paper_knn_fingerprint.py tests/test_source_pretrain_180d.py tests/test_source_selector_window_leakage.py tests/test_d4_d6_knn_window_perturbation.py -q
```

Expected: all pass, including exact D2 paper identity.

```bash
git add src/protocols/candidate_pool.py src/protocols/provenance.py src/source_selection/source_selector.py src/protocols/runner_adapter.py src/utils/parquet_data_loader.py tests/test_daily_knn_protocol.py tests/test_knn_cnn_provenance.py tests/test_d2_paper_knn_fingerprint.py tests/test_source_pretrain_180d.py
git commit -m "feat: separate 30-day KNN from 180-day pretraining"
```

### Task 6: Implement Blind Joint-Horizon Rollout and Clipped Metrics

**Files:**
- Create: `src/protocols/blind_rollout.py`
- Modify: `src/protocols/rolling_origin.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `src/evaluation/metric_contract.py`
- Create: `tests/test_blind_rollout_protocol.py`
- Modify: `tests/test_rolling_origin_protocol.py`
- Modify: `tests/test_smape_metric_contract.py`

**Interfaces:**
- Produces: `Predictor` protocol, `BlindPrediction`, `BlindRolloutTrace`, `clip_sales_prediction`, `run_blind_rollout(predictors, observed_frame, future_known_frame, truth_frame, ...)`.

- [ ] **Step 1: Write failing pure-NumPy rollout tests**

```python
class RecordingPredictor:
    def __init__(self, value):
        self.value = value
        self.inputs = []
    def predict_one(self, window):
        self.inputs.append(window.copy())
        return float(self.value)


def test_blind_rollout_uses_only_clipped_h1_feedback():
    predictors = {h: RecordingPredictor(-5.0 if h == 1 else 10.0 + h) for h in range(1, 6)}
    trace = run_blind_rollout(predictors, observed_30, future_known_180, sealed_truth_180)
    assert trace.sample_counts == {1: 180, 2: 179, 3: 178, 4: 177, 5: 176}
    assert trace.for_horizon(1)[0].y_pred_raw == -5.0
    assert trace.for_horizon(1)[0].y_pred_clipped == 0.0
    assert predictors[2].inputs[1][-1, sales_index] == 0.0


def test_mutating_blind_truth_changes_metrics_but_not_predictions():
    def make_predictors():
        return {h: RecordingPredictor(float(h)) for h in range(1, 6)}
    first = run_blind_rollout(make_predictors(), observed_30, future_known_180, truth_a)
    second = run_blind_rollout(make_predictors(), observed_30, future_known_180, truth_b)
    assert first.prediction_digest == second.prediction_digest
    assert first.input_digest == second.input_digest
    assert first.metric_digest != second.metric_digest
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_blind_rollout_protocol.py tests/test_rolling_origin_protocol.py tests/test_smape_metric_contract.py -q
```

Expected: current rolling manifest contains true blind input sales and no joint rollout exists.

- [ ] **Step 3: Implement the evaluator boundary**

Define:

```python
class Predictor(Protocol):
    horizon: int
    feature_columns: tuple[str, ...]
    window_size: int
    def predict_one(self, window: np.ndarray) -> float: ...


@dataclass(frozen=True)
class BlindPrediction:
    forecast_origin: str
    label_date: str
    horizon: int
    y_pred_raw: float
    y_pred_clipped: float
    was_clipped: bool
    y_true: float
    sample_key: str
```

The rollout owns an in-memory history copied from observed data. At every origin it constructs windows from observed/predicted sales plus allowlisted future-known fields, asks every valid horizon predictor, clips after inverse transform, appends only h1, and joins `truth_frame` only after the full prediction loop. `rolling_origin.SampleRecord` must no longer persist blind `input_sales` taken from the truth-containing frame.

- [ ] **Step 4: Make metrics consume clipped original-space pairs only**

Centralize `clip_sales_prediction(values) -> np.ndarray`. `compute_metrics_with_protocol` must reject formal calls without `prediction_policy="inverse_then_clip_nonnegative_v1"`. Set `smape`, `smape_paper`, and `original_scale_smape` from one value and retain normalized sMAPE only as a diagnostic.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_blind_rollout_protocol.py tests/test_rolling_origin_protocol.py tests/test_smape_metric_contract.py tests/test_metric_protocol_and_diagnostics.py tests/test_multi_source_smape_metrics.py -q
```

Expected: all pass; blind prediction/input digests are truth-invariant.

```bash
git add src/protocols/blind_rollout.py src/protocols/rolling_origin.py src/evaluation/metrics.py src/evaluation/metric_contract.py tests/test_blind_rollout_protocol.py tests/test_rolling_origin_protocol.py tests/test_smape_metric_contract.py
git commit -m "feat: add leak-free joint-horizon blind rollout"
```

### Task 7: Expose Fitted Predictors from Every Method

**Files:**
- Create: `src/experiment/fitted_predictor.py`
- Modify: `src/experiment/run_no_tl_experiment.py`
- Modify: `src/experiment/experiment_runner.py`
- Modify: `src/transfer_methods/mswa_tl.py`
- Modify: `src/transfer_methods/mssb_tl.py`
- Modify: `src/transfer_methods/msml_tl.py`
- Modify: `src/transfer_methods/msml_tl_rfe.py`
- Modify: `src/utils/entity_experiment.py`
- Create: `tests/test_fitted_predictor_adapters.py`
- Create: `tests/test_joint_horizon_method_bundle.py`

**Interfaces:**
- Produces: `KerasPredictor`, `WeightedPredictor`, `SwitchingPredictor`, `FittedMethodHorizon`, `fit_method_horizon(...)`, `fit_method_bundle(...)`.
- Consumes: `run_blind_rollout` from Task 6.

- [ ] **Step 1: Write failing adapter tests with fake Keras models**

```python
class FakeModel:
    def __init__(self, value): self.value = value
    def predict(self, tensor, verbose=0):
        return np.full((len(tensor), 1), self.value, dtype=float)


def test_weighted_predictor_fuses_models_without_truth():
    window = np.zeros((10, 3), dtype=float)
    first = KerasPredictor(1, FakeModel(2.0), ("sales", "year", "month"), 10, lambda value: value)
    second = KerasPredictor(1, FakeModel(10.0), ("sales", "year", "month"), 10, lambda value: value)
    predictor = WeightedPredictor(
        horizon=1,
        predictors=(first, second),
        weights=(0.75, 0.25),
    )
    assert predictor.predict_one(window) == pytest.approx(4.0)


def test_method_bundle_fits_each_horizon_once_then_rolls_out_once(monkeypatch, bundle_inputs):
    calls = []
    def fake_fit(**kwargs):
        horizon = kwargs["horizon"]
        calls.append(horizon)
        return FittedMethodHorizon(
            method="No-TL",
            horizon=horizon,
            predictor=KerasPredictor(horizon, FakeModel(float(horizon)), ("sales",), 10, lambda value: value),
            metadata={},
        )
    monkeypatch.setattr(fitted_predictor_module, "fit_method_horizon", fake_fit)
    result = fit_method_bundle(horizons=(1, 2, 3, 4, 5), **bundle_inputs)
    assert calls == [1, 2, 3, 4, 5]
    assert result.trace.sample_counts == {1: 180, 2: 179, 3: 178, 4: 177, 5: 176}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_fitted_predictor_adapters.py tests/test_joint_horizon_method_bundle.py -q
```

Expected: fitted predictor interfaces do not exist.

- [ ] **Step 3: Split fitting from evaluation**

Each formal method must stop constructing `X_test` from target truth. It returns a fitted predictor, target scaler/inverse function, feature columns, selected-source provenance, and training diagnostics. Map methods as follows:

The adapter constructors are fixed as:

```python
KerasPredictor(horizon, model, feature_columns, window_size, inverse_sales)
WeightedPredictor(horizon, predictors, weights)
SwitchingPredictor(horizon, selected_source_key, predictor, validation_rmse)
FittedMethodHorizon(method, horizon, predictor, metadata)
```

- No-TL and SS-TL: one `KerasPredictor`;
- MSWA-TL: `WeightedPredictor` over successful source-specific target models using normalized KNN weights;
- MSSB-TL: `SwitchingPredictor` holding the source-specific target model selected only by target validation RMSE;
- MSML-TL and MSML-TL-RFE: one fused `KerasPredictor`, with the RFE feature subset embedded in the adapter.

Use target train/validation only. Preserve current public run functions as non-formal compatibility wrappers, but make the formal branch call `fit_method_bundle` and Task 6 rollout.

- [ ] **Step 4: Reject accidental blind evaluation inside training**

Add a sentinel `SealedTruthFrame` whose DataFrame access raises before evaluator release. Pass it through every formal method test and assert all six methods can fit fake models without reading it.

- [ ] **Step 5: Verify GREEN and existing method tests**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_fitted_predictor_adapters.py tests/test_joint_horizon_method_bundle.py tests/test_notl_minimal_fix.py tests/test_single_source_tl.py tests/test_mswa_tl.py tests/test_mssb_tl.py tests/test_msml_tl.py tests/test_msml_tl_rfe.py -q
```

Expected: all complete within 180 seconds. If exit 124, stop and provide this exact command; do not run smaller subsets.

- [ ] **Step 6: Commit**

```bash
git add src/experiment/fitted_predictor.py src/experiment/run_no_tl_experiment.py src/experiment/experiment_runner.py src/transfer_methods/mswa_tl.py src/transfer_methods/mssb_tl.py src/transfer_methods/msml_tl.py src/transfer_methods/msml_tl_rfe.py src/utils/entity_experiment.py tests/test_fitted_predictor_adapters.py tests/test_joint_horizon_method_bundle.py
git commit -m "refactor: separate model fitting from blind evaluation"
```

### Task 8: Publish Detailed Prediction, Selection, and Failure Artifacts

**Files:**
- Create: `src/utils/prediction_artifacts.py`
- Modify: `src/utils/run_layout.py`
- Modify: `src/utils/run_artifacts.py`
- Modify: `src/constants.py`
- Modify: `src/utils/result_schema.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `src/utils/entity_experiment.py`
- Create: `tests/test_prediction_artifacts.py`
- Modify: `tests/test_result_schema_golden_diff.py`

**Interfaces:**
- Produces: `PredictionArtifactSet`, `publish_prediction_trace(...)`, `publish_source_selection(...)`, `publish_failure_record(...)` and new `RunLayout` paths.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_prediction_trace_is_atomic_gzip_csv_with_hash_manifest(tmp_path):
    layout = RunLayout(tmp_path)
    artifact = publish_prediction_trace(
        trace_frame,
        stable_path=layout.seed_predictions(5, "without", 42),
        identity=identity,
    )
    loaded = pd.read_csv(artifact.path, compression="gzip")
    assert loaded.columns.tolist() == REQUIRED_PREDICTION_COLUMNS
    assert artifact.sha256 == sha256_file(artifact.path)
    assert loaded.groupby("horizon").size().to_dict() == {1: 180, 2: 179, 3: 178, 4: 177, 5: 176}


def test_failed_publication_never_replaces_existing_trace(tmp_path):
    stable = existing_valid_trace(tmp_path)
    with pytest.raises(PredictionArtifactError):
        publish_prediction_trace(invalid_duplicate_dates, stable_path=stable, identity=identity)
    assert sha256_file(stable) == original_sha
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_prediction_artifacts.py tests/test_result_schema_golden_diff.py -q
```

Expected: artifact module and schema fields are missing.

- [ ] **Step 3: Implement artifact paths and required columns**

Add seed-bundle paths:

```text
d{dataset}_{mode}/cells/s{seed}/results/dataset{dataset}_{mode}_results.csv
d{dataset}_{mode}/cells/s{seed}/predictions/predictions.csv.gz
d{dataset}_{mode}/source_selection.csv
d{dataset}_{mode}/failures.csv
run_manifest.json
acceptance_report.json
acceptance_report.md
```

Set `RunLayout.aggregate_result` to `run_root / "results" / "experiment_results.csv"`. Any `d1_d6_results.csv` compatibility export is generated only after sealing and is never referenced by manifests or resume logic.

Prediction columns are exactly run/cell identity, dataset, scenario, target, method, seed, horizon, forecast origin, label date, raw prediction, clipped prediction, clipped flag, evaluator-joined truth, synthetic-date flag, sample key, input digest, and prediction-policy ID. Extend result schema with source 180-day dates/counts, KNN features/digest, model/future/evaluation feature roles, cleaning report digest, clipping count, prediction trace path/hash, and sealed protocol version.

`run_manifest.json` also records the Git commit/worktree identity, Python version, package versions, hostname/platform, input hashes, cache hashes, per-worker thread environment, start/end timestamps, and scheduler event-log hash.

- [ ] **Step 4: Publish traces before summary acceptance**

Cell result rows must reference an existing accepted trace manifest. A summary row cannot be accepted if its horizon sample count, metric date range, prediction digest, clipping count, or recomputed metric differs from the trace.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_prediction_artifacts.py tests/test_result_schema_golden_diff.py tests/test_unified_d1_d6_output_contract.py tests/test_strict_result_contract.py -q
```

Expected: all pass and corrupt traces fail closed.

```bash
git add src/utils/prediction_artifacts.py src/utils/run_layout.py src/utils/run_artifacts.py src/constants.py src/utils/result_schema.py scripts/run_full_paper_experiments.py src/utils/entity_experiment.py tests/test_prediction_artifacts.py tests/test_result_schema_golden_diff.py
git commit -m "feat: publish auditable blind prediction artifacts"
```

### Task 9: Replace 300 Horizon Cells with 60 Seed Bundles

**Files:**
- Modify: `scripts/run_strict_protocol_baseline.py`
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `scripts/run_d4_experiment.py`
- Modify: `scripts/run_d5_experiment.py`
- Modify: `scripts/run_d6_experiment.py`
- Modify: `src/utils/run_layout.py`
- Modify: `src/utils/result_acceptance.py`
- Modify: `src/utils/result_validation.py`
- Modify: `tests/test_formal_protocol_matrix.py`
- Modify: `tests/test_unified_parallel_lifecycle.py`
- Modify: `tests/test_run_layout_and_atomic_publication.py`

**Interfaces:**
- Replaces `MatrixTask(horizon, seed)` with `SeedBundleTask(seed, horizons=(1,2,3,4,5))`.
- One accepted seed bundle contains all six methods × all targets × five horizons for one dataset/mode/seed.

- [ ] **Step 1: Change tests to the approved 60-cell identity**

```python
def test_run_plan_locks_60_unique_seed_bundles():
    plan = build_run_plan(run_root, code_identity=identity, input_identity=inputs)
    assert len(plan["cells"]) == 60
    assert len({cell["result_path"] for cell in plan["cells"]}) == 60
    assert {tuple(cell["horizons"]) for cell in plan["cells"]} == {(1, 2, 3, 4, 5)}


def test_mode_worker_runs_exactly_five_seed_bundles(prepared_run, monkeypatch):
    seen = []
    monkeypatch.setattr(unified, "run_task", lambda task: successful(task, seen))
    unified.execute_mode_worker(prepared_run / "d2_with", "d2", "with", resume=False)
    assert [task.seed for task in seen] == [42, 43, 44, 45, 46]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_formal_protocol_matrix.py tests/test_unified_parallel_lifecycle.py tests/test_run_layout_and_atomic_publication.py -q
```

Expected: current plan produces 300 `h{horizon}_s{seed}` cells.

- [ ] **Step 3: Implement seed bundle commands and layout**

Each D1–D3 command uses `--all-horizons --seed <seed>`; each D4–D6 command does the same. Add that CLI flag to the dataset runners and reject `--horizon` together with `--all-horizons`. Use `cells/s{seed}` directories. Build cell acceptance with `horizons=FORMAL_HORIZONS` and `seeds=(seed,)`. The mode aggregate still contains the same formal result row keys; only process ownership changes.

- [ ] **Step 4: Update run-plan and resume identity**

Set `run_plan_version="formal_d1_d6_seed_bundle_v3"`, require exactly 60 unique cell paths, five per mode, and include mode cache digest plus expected prediction trace paths in every plan cell. Old 300-cell runs must fail resume identity validation instead of being partially reused.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_formal_protocol_matrix.py tests/test_unified_parallel_lifecycle.py tests/test_run_layout_and_atomic_publication.py tests/test_result_acceptance_scopes.py -q
```

Expected: all pass; result row coverage remains five horizons × five seeds.

```bash
git add scripts/run_strict_protocol_baseline.py scripts/run_unified_d1_d6.py scripts/run_full_paper_experiments.py scripts/run_d4_experiment.py scripts/run_d5_experiment.py scripts/run_d6_experiment.py src/utils/run_layout.py src/utils/result_acceptance.py src/utils/result_validation.py tests/test_formal_protocol_matrix.py tests/test_unified_parallel_lifecycle.py tests/test_run_layout_and_atomic_publication.py
git commit -m "refactor: group formal horizons by seed"
```

### Task 10: Implement D5-First Thread-Aware Failure-Preserving Scheduling

**Files:**
- Modify: `scripts/parallel_mode_runner.sh`
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `src/utils/run_artifacts.py`
- Modify: `tests/fixtures/fake_formal_worker.py`
- Modify: `tests/test_parallel_mode_supervisor.py`

**Interfaces:**
- Formal environment: `D5_THREADS=6`, `ORDINARY_THREADS=2`, `MAX_JOBS=6`, total budget 16.
- Scheduler dependency: `d5_without -> d5_with`; all other modes have no dependency.
- Produces: `RunStatus`, `write_run_status(...)`, and the `mark-status` lifecycle operation used by the shell supervisor.

- [ ] **Step 1: Write failing order, environment, and failure-preservation tests**

```python
def test_d5_without_starts_first_and_d5_with_starts_immediately_after_it(tmp_path):
    completed = _run_supervisor(tmp_path, MAX_JOBS="6", FAKE_SLEEP="0.2")
    events = _events(tmp_path)
    starts = [e for e in events if e["event"] == "start"]
    assert starts[0]["task"] == "d5_without"
    d5_without_finish = next(i for i, e in enumerate(events) if e["task"] == "d5_without" and e["event"] == "finish")
    next_start = next(e for e in events[d5_without_finish + 1:] if e["event"] == "start")
    assert next_start["task"] == "d5_with"


def test_thread_budget_is_six_for_d5_two_for_ordinary_and_never_over_16(tmp_path):
    completed = _run_supervisor(tmp_path, MAX_JOBS="6", FAKE_SLEEP="0.2")
    starts = [e for e in _events(tmp_path) if e["event"] == "start"]
    assert all(e["omp_threads"] == ("6" if e["task"].startswith("d5_") else "2") for e in starts)
    assert _peak_thread_sum(starts, _events(tmp_path)) <= 16


def test_d5_without_failure_blocks_with_but_preserves_finished_and_inflight(tmp_path):
    completed = _run_supervisor(tmp_path, MAX_JOBS="6", FAKE_FAIL_MODE="d5_without")
    events = _events(tmp_path)
    assert completed.returncode == 7
    assert not any(e["task"] == "d5_with" and e["event"] == "start" for e in events)
    assert any(e["event"] == "finish" for e in events if not e["task"].startswith("d5_"))
    assert _run_status(tmp_path) == "partial_failed"


def test_d5_with_failure_keeps_accepted_d5_without(tmp_path):
    completed = _run_supervisor(tmp_path, FAKE_FAIL_MODE="d5_with")
    assert completed.returncode == 7
    assert accepted_mode(tmp_path, "d5_without").is_file()
    assert _run_status(tmp_path) == "partial_failed"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_parallel_mode_supervisor.py -q
```

Expected: current task list launches D1 first, permits D5 modes by queue order only, exposes no thread env, and kills peers after any worker failure.

- [ ] **Step 3: Implement dependency-aware selection**

Order the queue as `d5_without`, then all D1–D4/D6 modes. Keep `d5_with` dependency-blocked. Once `d5_without` is `succeeded`, return `d5_with` before scanning ordinary queued work. If `d5_without` fails, mark `d5_with=blocked` and stop launching new tasks.

- [ ] **Step 4: Export thread limits per worker**

Launch through `env` with the task budget applied to `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `TF_NUM_INTRAOP_THREADS`; set `TF_NUM_INTEROP_THREADS=1`. Before launch, compute the active thread sum and refuse candidates that would exceed 16.

- [ ] **Step 5: Preserve in-flight results after worker failure**

Separate signal cleanup from task failure. Signals still terminate all process groups. A task failure sets `STOP_SCHEDULING=1`, records failure/blocked rows, waits for already running workers to reach atomic completion, skips aggregate, writes `partial_failed`, and exits with the first failure code. It must not delete any accepted mode/cell artifact.

The supervisor writes status through a new non-training command:

```bash
"${PYTHON}" "${UNIFIED_RUNNER}" --operation mark-status \
    --output-dir "${FORMAL_RUN_ROOT}" --run-status partial_failed
```

`run_unified_d1_d6.py` accepts `--operation mark-status --run-status running|partial_failed|complete_unsealed|sealed_failed|sealed_success` and delegates to `write_run_status`; invalid transitions exit 2.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_parallel_mode_supervisor.py -q
```

Expected: all scheduler tests pass in under 30 seconds.

```bash
git add scripts/parallel_mode_runner.sh scripts/run_unified_d1_d6.py src/utils/run_artifacts.py tests/fixtures/fake_formal_worker.py tests/test_parallel_mode_supervisor.py
git commit -m "feat: prioritize and isolate the D5 heavy lane"
```

### Task 11: Enforce Partial/Complete/Sealed State and Full Trace Acceptance

**Files:**
- Modify: `src/utils/result_acceptance.py`
- Modify: `src/utils/result_validation.py`
- Modify: `src/utils/run_artifacts.py`
- Modify: `scripts/run_unified_d1_d6.py`
- Create: `tests/test_sealed_run_acceptance.py`
- Modify: `tests/test_result_state_machine.py`
- Modify: `tests/test_unified_parallel_lifecycle.py`

**Interfaces:**
- Consumes: `RunStatus` and `write_run_status(...)` from Task 10.
- Produces: `accept_sealed_run(...)`, `publish_sealed_success(...)`.
- Valid terminal success requires 12 accepted modes, 60 accepted seed bundles, full formal row keys, accepted traces, source selections, no failures, and matching digests.

- [ ] **Step 1: Write failing state and tamper tests**

```python
@pytest.mark.parametrize(
    "mutation,reason",
    [
        (drop_one_result_row, "formal_key_coverage_mismatch"),
        (drop_one_prediction_date, "prediction_sample_coverage_mismatch"),
        (change_smape, "metric_trace_mismatch"),
        (change_source_digest, "selection_digest_mismatch"),
        (remove_mode_manifest, "mode_manifest_missing"),
    ],
)
def test_sealing_rejects_any_incomplete_or_tampered_artifact(valid_run, mutation, reason):
    mutation(valid_run)
    report = accept_sealed_run(valid_run)
    assert report.passed is False
    assert reason in report.reasons
    assert not (valid_run / "SEALED_SUCCESS").exists()


def test_only_complete_verified_run_gets_sealed_success(valid_run):
    marker = publish_sealed_success(valid_run, accept_sealed_run(valid_run))
    assert marker.read_text(encoding="utf-8").strip() == valid_run_identity(valid_run)
    assert json.loads((valid_run / "run_status.json").read_text())["status"] == "sealed_success"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_run_acceptance.py tests/test_result_state_machine.py tests/test_unified_parallel_lifecycle.py -q
```

Expected: no sealed-run acceptance or marker exists.

- [ ] **Step 3: Implement fail-closed global acceptance**

Recompute every result metric from its accepted prediction trace; require h1–h5 sample counts, date ranges, sample keys, clipping counts, and trace hashes. Require source selection rows and selected-source digests for transfer methods. Require cleaning, cache, input, code, and protocol identities from `run_plan.json` to match every cell/mode manifest.

The accepted global CSV path is exactly `results/experiment_results.csv`. `d1_d6_results.csv`, if retained for old readers, is a non-authoritative compatibility copy and cannot be used for resume or sealing.

- [ ] **Step 4: Implement run states**

Allow only these transitions:

```text
running -> partial_failed
running -> complete_unsealed
complete_unsealed -> sealed_success
complete_unsealed -> sealed_failed
```

`partial_failed` preserves successes and a `failures.csv`; it cannot transition directly to success without a validated resume completing all expected artifacts. Write `SEALED_SUCCESS` atomically only after the final report passes.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealed_run_acceptance.py tests/test_result_state_machine.py tests/test_unified_parallel_lifecycle.py tests/test_result_acceptance_scopes.py -q
```

Expected: all mutation cases are rejected and the valid fixture seals.

```bash
git add src/utils/result_acceptance.py src/utils/result_validation.py src/utils/run_artifacts.py scripts/run_unified_d1_d6.py tests/test_sealed_run_acceptance.py tests/test_result_state_machine.py tests/test_unified_parallel_lifecycle.py
git commit -m "feat: seal only complete trace-verified formal runs"
```

### Task 12: Wire the Single Formal Entry, Dry Run, and Final Lightweight Verification

**Files:**
- Modify: `scripts/run_unified_d1_d6.py`
- Modify: `scripts/parallel_mode_runner.sh`
- Modify: `scripts/validate_d1_d6_protocol_inputs.py`
- Modify: `README.md`
- Create: `tests/test_formal_entry_preflight.py`

**Interfaces:**
- User entry: `bash scripts/parallel_mode_runner.sh` from project root on the server.
- Preflight/dry-run: `DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh`.
- Produces: `build_formal_preflight(project_root, run_root)`, `ResumeDecision`, and `classify_resume_candidate(path, current_identity)`.

- [ ] **Step 1: Write failing formal-entry preflight tests**

```python
def test_preflight_reports_windows_features_cache_threads_and_60_cells(tmp_path):
    report = build_formal_preflight(project_root=fixture_project, run_root=tmp_path)
    assert report["passed"] is True
    assert report["planned_seed_bundles"] == 60
    assert report["planned_result_rows"] == report["expected_result_rows"]
    assert report["source_pretrain_days"] == 180
    assert report["blind_test_days"] == 180
    assert report["thread_budget"] == {"d5": 6, "ordinary": 2, "total": 16}
    assert report["first_mode"] == "d5_without"


def test_old_d5_output_without_matching_manifest_is_never_resumed(tmp_path):
    old = write_old_d5_csv(tmp_path)
    decision = classify_resume_candidate(old, current_identity)
    assert decision.reusable is False
    assert decision.reason == "missing_or_mismatched_sealed_identity"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_formal_entry_preflight.py -q
```

Expected: preflight does not yet report the sealed protocol, seed bundles, cache identity, and D5 order.

- [ ] **Step 3: Complete preflight and dry-run output**

Before creating a formal run, validate fixed D1–D6 parquet/manifests, exact target and source windows, feature roles, D2 fingerprint, cache identity, clean code identity, 60 unique seed bundles, 12 modes, thread limits, D5 dependency, expected result keys, and output ownership. Dry-run prints all resolved dates/columns and cache hit/miss decisions without creating output directories.

- [ ] **Step 4: Document the server entry and recovery behavior**

Document:

```bash
cd /path/to/保留的复现实验修改rfe
DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh
MAX_JOBS=6 RUN_ROOT=outputs/runs/<new_run_id> bash scripts/parallel_mode_runner.sh
MAX_JOBS=6 RUN_ROOT=outputs/runs/<existing_run_id> RESUME=1 bash scripts/parallel_mode_runner.sh
```

State that Codex never runs the second or third command. Resume reuses only accepted seed bundles with exact code/input/protocol/cache/trace identities. Old server D5 CSVs are comparison-only unless they already satisfy the new sealed manifest.

- [ ] **Step 5: Run the complete lightweight contract suite**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_sealing_protocol.py tests/test_d1_d2_formal_input_paths.py tests/test_d1_d2_sealed_builder.py tests/test_sealed_daily_contract.py tests/test_sealed_parquet_pushdown.py tests/test_mode_cache_contract.py tests/test_daily_knn_protocol.py tests/test_d2_paper_knn_fingerprint.py tests/test_source_pretrain_180d.py tests/test_blind_rollout_protocol.py tests/test_fitted_predictor_adapters.py tests/test_joint_horizon_method_bundle.py tests/test_smape_metric_contract.py tests/test_prediction_artifacts.py tests/test_formal_protocol_matrix.py tests/test_parallel_mode_supervisor.py tests/test_sealed_run_acceptance.py tests/test_formal_entry_preflight.py -q
```

Expected: all tests pass within 180 seconds and no D1–D6 training process or large output is created. If exit 124, stop immediately and return this exact command to the user; do not retry a subset.

- [ ] **Step 6: Run static checks**

Run:

```bash
python -m compileall -q src scripts tests
git diff --check
bash -n scripts/parallel_mode_runner.sh
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit the formal entry and documentation**

```bash
git add scripts/run_unified_d1_d6.py scripts/parallel_mode_runner.sh scripts/validate_d1_d6_protocol_inputs.py tests/test_formal_entry_preflight.py README.md
git commit -m "feat: finalize the D1-D6 sealed formal entry"
```

## Execution Notes

- Execute tasks strictly in order because later cache, result, and resume identities include earlier protocol fields.
- Do not retain compatibility fallbacks in the formal branch. Compatibility wrappers may remain only for non-formal callers and must be labeled non-sealed.
- After each task, review only that task's diff, run its exact tests, and commit before beginning the next task.
- Do not generate or run the full experiment during implementation. The final handoff includes dry-run evidence and the exact server command only.
