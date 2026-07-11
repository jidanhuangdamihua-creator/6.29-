# D1–D6 Shared Experiment Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved `d1_d6_protocol_v1` contract as the single source of truth for D1–D6 source selection, provenance, rolling-origin evaluation, result status, and preprocessing preflight without running formal experiments.

**Architecture:** Add a focused `src/protocols` package that owns immutable protocol definitions, canonical digests, deterministic daily-sequence KNN selection, provenance validation, and rolling-origin manifests. Existing D1–D6 runners and baseline loaders become adapters that consume these objects; they may not infer dates, rebuild source pools, or silently fall back. Result alignment marks old rows as `legacy_unverified` and new rows must carry the full protocol audit fields.

**Tech Stack:** Python 3.9+, pandas, NumPy, dataclasses, hashlib/json, standard-library `unittest`; existing TensorFlow/PyTorch code remains optional and no model training is used in protocol tests.

## Global Constraints

- Protocol version is exactly `d1_d6_protocol_v1`.
- Tracks are `strict_paper` for D1–D3 and `extended` for D4–D6.
- `knn_observed_end = target_observed_start + 29 calendar days`; `source_observation_cutoff = knn_observed_end`; target test dates are strictly later.
- KNN representation is a 30-element daily sales sequence in ascending date order, calculated in `float64`; IDs and static metadata are excluded.
- Tie tolerance is absolute `1e-12`; tie groups are anchored at each smallest ungrouped distance and ordered by normalized source key.
- Formal weights are inverse-distance with epsilon `1e-8`; K may never shrink silently.
- Formal horizons are exactly `(1, 2, 3, 4, 5)` and seeds exactly `(42, 43, 44, 45, 46)`.
- Primary metrics are original-scale RMSE, MAE, sMAPE, and `1 / (RMSE + 1e-8)`.
- Existing results without all required strict fields are `legacy_unverified` and cannot enter confirmed-baseline aggregation.
- No D1–D6 formal experiment or data regeneration is run by Codex.
- Every Python command is executed as `python tools/protection/codex_timeout.py --timeout 180 -- python ...`; exit 124 stops execution immediately.

---

### Task 1: Immutable Protocol Definitions and Exact Candidate-Pool Rules

**Files:**
- Create: `src/protocols/__init__.py`
- Create: `src/protocols/experiment_protocol.py`
- Test: `tests/test_experiment_protocol_contract.py`

**Interfaces:**
- Produces: `ExperimentProtocol`, `ObservationWindow`, `SourcePoolRule`, `get_experiment_protocol(dataset_id)`, `build_candidate_keys(protocol, scenario, target_key, available_keys)`.
- Consumed by: all later protocol, runner, preflight, and result tasks.

- [x] **Step 1: Write failing protocol tests**

```python
class ExperimentProtocolContractTest(unittest.TestCase):
    def test_tracks_windows_horizons_and_seeds_are_frozen(self):
        d1 = get_experiment_protocol("D1")
        d4 = get_experiment_protocol("D4")
        self.assertEqual(d1.track, "strict_paper")
        self.assertEqual(d4.track, "extended")
        self.assertEqual(d1.horizons, (1, 2, 3, 4, 5))
        self.assertEqual(d1.seeds, (42, 43, 44, 45, 46))
        window = d1.observation_window("2017-06-05")
        self.assertEqual(window.knn_observed_end.isoformat(), "2017-07-04")
        self.assertEqual(window.source_observation_cutoff, window.knn_observed_end)

    def test_d1_d3_paper_candidate_keys_are_exact(self):
        available = {(f"Store{s}", f"Item{i}") for s in range(1, 4) for i in range(1, 11)}
        d1 = get_experiment_protocol("D1")
        self.assertEqual(
            build_candidate_keys(d1, "without", ("Store1", "Item10"), available),
            tuple(("Store1", f"Item{i}") for i in range(1, 10)),
        )
        self.assertEqual(len(build_candidate_keys(d1, "with", ("Store1", "Item10"), available)), 27)

    def test_extended_pool_stays_in_group_and_excludes_target(self):
        d5 = get_experiment_protocol("D5")
        available = (
            SourceIdentity(("S1", "I1"), group_value="F1"),
            SourceIdentity(("S1", "I2"), group_value="F1"),
            SourceIdentity(("S2", "I2"), group_value="F1"),
            SourceIdentity(("S2", "I3"), group_value="F2"),
        )
        self.assertEqual(
            [entry.key for entry in build_candidate_keys(d5, "with", ("S1", "I1"), available)],
            [("S1", "I2"), ("S2", "I2")],
        )
```

- [x] **Step 2: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_experiment_protocol_contract -v`

Expected: import failure because `src.protocols.experiment_protocol` does not exist.

- [x] **Step 3: Implement immutable definitions and exact rules**

Implement frozen dataclasses, normalized dataset/scenario aliases, exact D1–D3 identities, D4–D6 group columns (`category`, `family`, `department`), target exclusion, duplicate detection, and fail-fast missing-paper-key diagnostics. `build_candidate_keys` returns keys in normalized lexical order.

- [x] **Step 4: Verify GREEN and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_experiment_protocol_contract -v`

Expected: all Task 1 tests pass without loading dataset files or ML libraries.

Commit: `feat: add strict D1-D6 protocol definitions`

### Task 2: Canonical Digests and Deterministic 30-Day KNN

**Files:**
- Create: `src/protocols/candidate_pool.py`
- Test: `tests/test_candidate_pool_digest.py`
- Test: `tests/test_daily_knn_protocol.py`

**Interfaces:**
- Consumes: `ExperimentProtocol`, `ObservationWindow`.
- Produces: `normalize_source_key`, `build_candidate_pool_digest(...)`, `build_selection_result_digest(...)`, `select_daily_sequence_sources(target_df, source_df, ...) -> SelectionResult`.

- [ ] **Step 1: Write failing digest goldens**

```python
WITHOUT_INPUT = dict(
    protocol_version="d1_d6_protocol_v1",
    dataset_id="D1",
    scenario="without",
    target_key=("Store1", "Item10"),
    group_cols=("store", "item"),
    candidate_keys=(("Store1", "Item1"), ("Store1", "Item2")),
    observed_start="2017-06-05",
    observed_end="2017-07-04",
    feature_cols=("sales",),
)

class CandidatePoolDigestTest(unittest.TestCase):
    def test_with_and_without_have_fixed_distinct_sha256(self):
        without = build_candidate_pool_digest(**WITHOUT_INPUT)
        with_sharing = build_candidate_pool_digest(
            **{**WITHOUT_INPUT, "scenario": "with", "candidate_keys": WITHOUT_INPUT["candidate_keys"] + (("Store2", "Item1"),)}
        )
        self.assertEqual(without, "7d7e0e0d6a08841426df0cea2273e420ae5d4b4dbc12c4c36e5cbf21e1328c72")
        self.assertEqual(with_sharing, "e3ea5ab06308c0b6a3826ab98ad1a9a33e026d66f3af4925698ffa8fd1941478")
        self.assertNotEqual(without, with_sharing)

    def test_each_contract_input_mutation_changes_digest(self):
        baseline = build_candidate_pool_digest(**WITHOUT_INPUT)
        for field, value in (("protocol_version", "v2"), ("dataset_id", "D2"), ("scenario", "with"), ("observed_end", "2017-07-05"), ("feature_cols", ("units",))):
            self.assertNotEqual(baseline, build_candidate_pool_digest(**{**WITHOUT_INPUT, field: value}))
```

The two literals above are SHA-256 values of the exact canonical JSON contract shown in this task. The production function must reproduce them; tests must not compute expectations with duplicate serialization logic.

- [ ] **Step 2: Write failing leakage, perturbation, tie, K, and missing-date tests**

Create deterministic 30-day synthetic frames and assert:

```python
self.assertEqual(before.ordered_source_keys, after_future_perturbation.ordered_source_keys)
np.testing.assert_array_equal(before.distances, after_future_perturbation.distances)
np.testing.assert_array_equal(before.weights, after_future_perturbation.weights)
self.assertEqual(before.selection_result_digest, after_future_perturbation.selection_result_digest)
self.assertGreater(original_margin, 0.5)
self.assertEqual(after_observed_extreme.ordered_source_keys[0], expected_former_far_source)
self.assertEqual(tied.ordered_source_keys[:2], (("S1", "I1"), ("S1", "I2")))
```

Also assert explicit failures for target missing day, duplicate target day, candidate missing day causing valid count below K, duplicate source key/date, target in source pool, future-only candidate data, non-finite distance, and `k > valid_source_count`.

- [ ] **Step 3: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_candidate_pool_digest tests.test_daily_knn_protocol -v`

Expected: missing production functions/classes.

- [ ] **Step 4: Implement canonical serialization and selection**

Canonical JSON uses UTF-8, `sort_keys=True`, `separators=(",", ":")`, ISO dates, sorted candidate keys, declared column order, and SHA-256. Selection filters to the exact dates, validates uniqueness/completeness, fits one legal-window scaler for the task, computes Euclidean `float64` distances, creates anchored tie groups, selects exactly K, computes inverse-distance weights, and persists digest inputs plus excluded-candidate diagnostics.

- [ ] **Step 5: Freeze goldens, verify GREEN, and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_candidate_pool_digest tests.test_daily_knn_protocol -v`

Expected: all digest/KNN tests pass and execute in seconds.

Commit: `feat: implement deterministic leak-free daily KNN`

### Task 3: Exact KNN-to-CNN Provenance

**Files:**
- Create: `src/protocols/provenance.py`
- Test: `tests/test_knn_cnn_provenance.py`

**Interfaces:**
- Consumes: `SelectionResult` and raw target/source DataFrames.
- Produces: `SourceSliceRef`, `TensorProvenance`, `extract_selected_source_slices`, `validate_cnn_tensor_provenance`.

- [ ] **Step 1: Write failing elementwise provenance tests**

Build two selected synthetic sources with known values and dates. Assert exact `(store, item, date_start, date_end)`, KNN vector equality to the raw 30-day slice, extractor key equality to ordered selection keys, CNN tensor/date/feature/label equality to the original rows, and explicit failure after key substitution, date reordering, or one-value mutation.

- [ ] **Step 2: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_knn_cnn_provenance -v`

Expected: provenance module import failure.

- [ ] **Step 3: Implement immutable provenance records and validators**

Use normalized keys and ISO date tuples. Extraction accepts only the ordered keys from `SelectionResult`; no fuzzy name lookup or candidate-pool reselection is exposed. Validators use `np.testing.assert_array_equal` semantics and raise a contract-specific `ProtocolViolation` with the mismatched key/date/element.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_knn_cnn_provenance -v`

Expected: all provenance tests pass without model training.

Commit: `feat: enforce KNN to CNN source provenance`

### Task 4: Rolling-Origin Manifest, Feature Availability, Metrics, and Seeds

**Files:**
- Create: `src/protocols/rolling_origin.py`
- Create: `src/protocols/reproducibility.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `scripts/baselines/baseline_data_loader.py`
- Modify: `scripts/baselines/run_baselines_multiseed.py`
- Test: `tests/test_rolling_origin_protocol.py`
- Test: `tests/test_baseline_protocol.py`

**Interfaces:**
- Produces: `SampleRecord`, `SampleManifest`, `build_sample_manifest`, `assert_same_sample_manifest`, `validate_feature_availability`, `set_protocol_seed`, `compute_original_scale_metrics`, `aggregate_protocol_results`.
- Consumed by: CNN adapters, all baseline methods, result validation.

- [ ] **Step 1: Write failing manifest/fairness tests**

Assert horizons `(1,2,3,4,5)`, every label date is after its input end, stable sample keys/digest, CNN and baseline ordered key equality, rejection of future `sales`, `transactions`, `stock`, and `customers`, acceptance only of explicitly allowed calendar/planned features, and failure when a method independently drops a sample.

- [ ] **Step 2: Write failing metric/seed/final-baseline tests**

Assert original-scale RMSE/MAE/sMAPE/accuracy against hand-calculated values, per-horizon and horizon-mean aggregation by five seeds, consistent Python/NumPy seed reset, and rejection of a final-baseline group missing any seed or horizon.

- [ ] **Step 3: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_rolling_origin_protocol tests.test_baseline_protocol -v`

Expected: missing manifest and protocol metric APIs.

- [ ] **Step 4: Implement shared manifest and baseline adapters**

The loader returns the protocol `SampleManifest` plus legal observed rows; BL1–BL4 receive the same manifest and emit one row per `target × method × horizon × seed`. Deterministic methods still emit all five seed rows with identical predictions so result cardinality is uniform. No method may build or shorten its own test slice.

- [ ] **Step 5: Verify GREEN and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_rolling_origin_protocol tests.test_baseline_protocol -v`

Expected: all tests pass without importing LightGBM/Torch unless a predictor is explicitly invoked.

Commit: `feat: unify rolling-origin baseline protocol`

### Task 5: SourceSelector and D1–D6 Runner Integration

**Files:**
- Modify: `src/source_selection/source_selector.py`
- Modify: `src/experiment/experiment_runner.py`
- Modify: `src/utils/parquet_data_loader.py`
- Modify: `src/utils/entity_experiment.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `scripts/run_d4_experiment.py`
- Modify: `scripts/run_d5_experiment.py`
- Modify: `scripts/run_d6_experiment.py`
- Modify: `scripts/run_unified_d1_d6.py`
- Test: `tests/test_source_selector_shared_protocol.py`
- Test: `tests/test_runner_protocol_integration.py`

**Interfaces:**
- Consumes: protocol, selection, provenance, and manifest objects from Tasks 1–4.
- Produces: unchanged public runner entrypoints with strict protocol metadata and fail-fast preflight.

- [ ] **Step 1: Write failing adapter tests**

Patch only the expensive training functions. Assert D1–D6 all call `select_daily_sequence_sources`, no runner calls the legacy statistics-signature path, formal runs reject `raw_distance`, K mismatch, or absent protocol metadata, D1/D2 old source pools fail with missing expected keys, selected source keys equal CNN extractor keys, and horizon/seed are forwarded to output rows.

- [ ] **Step 2: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_source_selector_shared_protocol tests.test_runner_protocol_integration -v`

Expected: runner still invokes legacy or D4–D6-only selection logic.

- [ ] **Step 3: Replace scattered selection with the shared adapter**

Keep existing method/runner signatures where possible. Convert DataFrame attrs to one `ExperimentProtocol` and `ObservationWindow`, delegate candidate construction and KNN to `src.protocols`, attach `SelectionResult`/provenance to method metadata, and remove runtime fallback, K shrinking, and D4–D6-specific digest implementations. Source training frames are clipped at `source_observation_cutoff` before any split or fitted transform.

- [ ] **Step 4: Verify GREEN and compatibility tests**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_source_selector_shared_protocol tests.test_runner_protocol_integration tests.test_experiment_runner tests.test_run_unified_d1_d6 -v`

Expected: protocol tests pass; compatible legacy entrypoint tests remain green.

Commit: `refactor: route D1-D6 runners through shared protocol`

### Task 6: Strict Result Contract and Legacy Isolation

**Files:**
- Modify: `src/constants.py`
- Modify: `src/utils/result_schema.py`
- Modify: `src/utils/result_validation.py`
- Modify: `scripts/aggregate_d1_d6_results.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Test: `tests/test_strict_result_contract.py`
- Test: `tests/test_aggregate_d1_d6_results.py`

**Interfaces:**
- Produces: `STRICT_PROTOCOL_FIELDS`, `classify_protocol_result`, `validate_confirmed_baseline_group`, strict result columns and legacy-safe aggregation.

- [ ] **Step 1: Write failing schema and aggregation tests**

Assert the mandatory fields `protocol_track`, `protocol_version`, `knn_observed_start`, `knn_observed_end`, `knn_representation`, `target_test_excluded`, `source_future_excluded`, `candidate_pool_digest`, `selection_result_digest`, `horizon`, `seed`, `primary_metric_space`, and `sample_manifest_digest`. Rows missing any field become `legacy_unverified`; strict and legacy rows cannot be aggregated together; only complete 5-seed × 5-horizon groups can be `confirmed_baseline`.

- [ ] **Step 2: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_strict_result_contract tests.test_aggregate_d1_d6_results -v`

Expected: current schema accepts incomplete rows and lacks strict field names.

- [ ] **Step 3: Implement classification and aggregation gates**

Add strict columns without deleting interpretable legacy columns. Alignment classifies rather than silently filling strict evidence fields. Aggregation separates `legacy_unverified`, `trial`, and `confirmed_baseline`, verifies original metric space and manifest consistency, and reports mean/std per horizon plus horizon 1–5 overall.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_strict_result_contract tests.test_aggregate_d1_d6_results -v`

Expected: strict contract and existing aggregation tests pass.

Commit: `feat: isolate legacy results from strict baselines`

### Task 7: Data-Regeneration Contracts and Static Preflight

**Files:**
- Modify: `scripts/regenerate_d1_d2_parquets.py`
- Modify: `scripts/preprocess_d1_d3_offline.py`
- Modify: `scripts/preprocess_d4_d6_offline.py`
- Create: `scripts/validate_d1_d6_protocol_inputs.py`
- Test: `tests/test_protocol_preprocessing_contract.py`
- Test: `tests/test_protocol_preflight.py`

**Interfaces:**
- Consumes: exact candidate-pool rules and observation windows.
- Produces: generation configuration capable of writing full required pools and a read-only preflight report/exit code.

- [ ] **Step 1: Write failing generation/static preflight tests**

Inspect configuration and synthetic frames rather than regenerating data. Assert D1 produces Store1–3 × Item1–10 coverage, D2 produces Brand1–3 × Item1–10 coverage, D3 covers Store1–30, D4–D6 preserve cross-store same-group candidates, dates are calendarized only under explicit zero-demand semantics, and existing incomplete D1/D2 fixtures fail with exact missing keys.

- [ ] **Step 2: Verify RED**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_protocol_preprocessing_contract tests.test_protocol_preflight -v`

Expected: D1/D2 current generation filters omit required stores/brands/items.

- [ ] **Step 3: Modify generation logic and add read-only preflight**

Generation scripts declare protocol version, key fields, explicit calendarization semantics, and full required pool filters. Preflight loads metadata/schema/key/date coverage only, calls production candidate/digest rules, reports missing/duplicate/future/insufficient-K issues, and never trains or rewrites data.

- [ ] **Step 4: Verify GREEN without regeneration and commit**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_protocol_preprocessing_contract tests.test_protocol_preflight -v`

Expected: synthetic/config tests pass; current incomplete artifacts are correctly classified as preflight failures rather than test failures.

Commit: `fix: regenerate complete D1-D6 protocol source pools`

### Task 8: Regression Verification and Requirement-by-Requirement Audit

**Files:**
- Modify: tests whose old expectations conflict with the approved protocol, including `tests/test_d4_d6_knn_window_perturbation.py`, `tests/test_knn_leakage_guards.py`, `tests/test_source_selector_window_leakage.py`, and `tests/test_unified_d1_d6_output_contract.py`.
- Create: `docs/protocol/d1_d6_protocol_v1_runbook.md`

**Interfaces:**
- Produces: documented user commands for regeneration, preflight, formal baseline, and archive acceptance; no commands are run by Codex beyond lightweight tests.

- [ ] **Step 1: Update conflicting tests to assert the production contract**

Remove assertions that permit statistics signatures, inferred cutoff fields, independent digest rules, candidate shrinking, normalized primary RMSE, or one-seed final results. Preserve unrelated behavioral coverage.

- [ ] **Step 2: Write the runbook**

Document prerequisites, expected preflight failure on old D1/D2 parquets, user-run regeneration commands, five-seed/horizon commands, required result fields, digest audit, and the exact conditions for labeling an archive `confirmed_baseline`.

- [ ] **Step 3: Run lightweight protocol suite**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m unittest tests.test_experiment_protocol_contract tests.test_candidate_pool_digest tests.test_daily_knn_protocol tests.test_knn_cnn_provenance tests.test_rolling_origin_protocol tests.test_baseline_protocol tests.test_source_selector_shared_protocol tests.test_runner_protocol_integration tests.test_strict_result_contract tests.test_protocol_preprocessing_contract tests.test_protocol_preflight -v`

Expected: all strict protocol tests pass in under 180 seconds. If exit code is 124, stop immediately and provide this exact command to the user.

- [ ] **Step 4: Run static checks**

Run: `python tools/protection/codex_timeout.py --timeout 180 -- python -m compileall -q src scripts tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors in files changed for this implementation; pre-existing unrelated worktree changes are reported separately and not modified.

- [ ] **Step 5: Audit every design requirement and commit**

Map design sections 4–15 to production files and passing tests. Record any environment-only or user-run evidence separately; do not claim a confirmed baseline because formal regeneration/training was intentionally not run.

Commit: `docs: add D1-D6 strict protocol runbook`
