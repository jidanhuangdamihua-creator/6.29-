# D2 Source Calendarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the frozen D2 180-day source calendarization rule before source completeness and KNN, and carry its identity through all existing selection digests.

**Architecture:** Add a D2-only source slice/calendarizer that owns the fixed interval and four-date allowlist. Integrate it at the shared protocol-frame boundary, then extend the existing candidate digest and selector metadata with the calendarization identities without changing non-D2 digest behavior.

**Tech Stack:** Python 3.9, pandas, NumPy, SHA-256/JSON, pytest, existing `ProtocolViolation` and shared protocol selector.

## Global Constraints

- Source interval is exactly `2018-01-02..2018-06-30`, 180 Gregorian calendar days.
- Only `2018-04-01`, `2018-04-25`, `2018-05-01`, and `2018-06-02` may be synthesized.
- Synthetic `sales` is numeric `0.0`; no forward fill, backward fill, mean fill, interpolation, or model inference may repair `sales`.
- Entity static keys are copied from the current source entity identity and calendar fields are regenerated from the actual date.
- Calendarization receives only the source frame and runs after source slicing and before source completeness/KNN.
- Target, validation, blind-period sales, source-window expansion, and other-date repair are out of scope.
- The rule version and calendarized result must enter source authority digest, consumer-frame fingerprint, candidate digest, and final sealed identity.
- Every Python validation command uses `python tools/protection/codex_timeout.py --timeout 180 -- ...`.

---

### Task 1: Add the D2 source calendarization module and contract tests

**Files:**
- Create: `src/protocols/d2_source_calendarization.py`
- Create: `tests/test_d2_source_calendarization.py`

**Interfaces:**
- Produces `D2_SOURCE_CALENDARIZATION_RULE_VERSION`, `D2_SOURCE_INTERVAL_START`, `D2_SOURCE_INTERVAL_END`, `D2_SOURCE_MISSING_DATES`, `D2_FROZEN_SOURCE_CANDIDATE_KEYS`, `slice_d2_source_frame(...)`, `calendarize_d2_source_frame(...)`, `build_d2_sealed_identity(...)`.
- `calendarize_d2_source_frame(source_slice, candidate_keys=...)` returns `(calendarized_source, report)` and never accepts a target frame.

- [x] **Step 1: Write failing tests for the frozen dates and source-only interface**

```python
def test_calendarizer_fills_only_the_four_frozen_dates_and_rebuilds_calendar_fields():
    source = _source_with_the_four_allowed_dates_missing()
    result, report = calendarize_d2_source_frame(
        slice_d2_source_frame(source),
        candidate_keys=(('1', '1'),),
    )

    assert len(result) == 180
    assert result['date'].nunique() == 180
    synthetic = result[result['date'].isin(pd.to_datetime(D2_SOURCE_MISSING_DATES))]
    assert synthetic['sales'].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert synthetic['brand_id'].tolist() == [1, 1, 1, 1]
    assert synthetic['item_id'].tolist() == [1, 1, 1, 1]
    assert synthetic['year'].tolist() == [2018] * 4
    assert synthetic['month'].tolist() == [4, 4, 5, 6]
    assert report.synthetic_row_count == 4
    assert report.rule_version == D2_SOURCE_CALENDARIZATION_RULE_VERSION


def test_calendarizer_rejects_any_missing_date_outside_the_allowlist():
    source = _source_with_the_four_allowed_dates_missing()
    source = source[source['date'] != pd.Timestamp('2018-03-01')]

    with pytest.raises(ProtocolViolation, match='unsupported missing source dates'):
        calendarize_d2_source_frame(
            slice_d2_source_frame(source),
            candidate_keys=(('1', '1'),),
        )


def test_calendarizer_does_not_read_target_or_fill_unauthorized_columns():
    source = _source_with_the_four_allowed_dates_missing().drop(
        columns=['date']
    )
    with pytest.raises(ProtocolViolation, match='date'):
        calendarize_d2_source_frame(source, candidate_keys=(('1', '1'),))
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q tests/test_d2_source_calendarization.py
```

Expected: collection fails because the new module and public constants do not exist.

- [x] **Step 3: Implement the minimal fail-closed calendarizer**

Implement fixed constants, source-only signatures, date normalization, candidate-key validation, duplicate/non-finite checks, exact-window validation, four-date-only insertion, static-key/calendar reconstruction, deterministic row ordering, and a report with source authority and consumer-frame digests. Inserted non-static/non-calendar columns remain missing; no `sales` value is copied from another date.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the same command from Step 2. Expected: all calendarizer tests pass.

### Task 2: Attach D2 calendarization before shared source eligibility and KNN

**Files:**
- Modify: `src/protocols/runner_adapter.py`
- Modify: `scripts/validate_d1_d6_protocol_inputs.py`
- Modify: `tests/test_protocol_preflight.py`
- Create: `tests/test_d2_source_calendarization_integration.py`

**Interfaces:**
- `configure_protocol_frames(...)` consumes frozen D2 source metadata and produces source/target frames carrying identical D2 calendarization identity metadata.
- `build_preflight_reports(...)` prepares its D2 pool from the calendarized 180-day source frame, including when the runtime source frame is a stub.

- [x] **Step 1: Write failing integration tests**

```python
def test_configure_protocol_frames_calendarizes_d2_source_before_pool_creation():
    source = _d2_source_with_four_missing_dates()
    target = _d2_target_after_observed_window()

    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id='D2',
        scenario='with',
        group_cols=('brand_id', 'item_id'),
        observed_start='2018-06-01',
    )

    assert configured_source.groupby(['brand_id', 'item_id']).date.nunique().eq(180).all()
    assert configured_source.attrs['d2_source_calendarization_rule_version']
    assert configured_source.attrs['d2_source_authority_digest'] == configured_target.attrs['d2_source_authority_digest']
    assert configured_source.attrs['d2_consumer_frame_fingerprint'] == configured_target.attrs['d2_consumer_frame_fingerprint']


def test_d2_preflight_prepares_pool_from_calendarized_source():
    source = _d2_source_with_four_missing_dates()
    target = _d2_target_after_observed_window()
    reports = build_preflight_reports(
        source,
        target,
        dataset_id='D2',
        scenario='with',
        group_cols=('brand_id', 'item_id'),
        observed_start='2018-06-01',
        k=1,
    )
    assert reports[0]['status'] == 'passed'
```

- [x] **Step 2: Run the focused integration tests and verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q tests/test_d2_source_calendarization_integration.py tests/test_protocol_preflight.py
```

Expected: the new assertions fail because D2 source currently retains the incomplete history and preflight builds its pool before calendarization.

- [x] **Step 3: Implement the source-slice/calendarizer ordering**

In the shared protocol-frame path, compute the frozen D2 candidate keys, slice only D2 source rows to the fixed interval, call the calendarizer, and attach report metadata before source is returned or a prepared pool is consumed. In the read-only preflight path, calendarize the D2 source before constructing `PreparedDailySequencePool`, then copy the report metadata to the empty source stub used by each target report. Preserve existing D1/D3–D6 behavior.

- [x] **Step 4: Run focused integration and regression tests**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q \
  tests/test_d2_source_calendarization.py \
  tests/test_d2_source_calendarization_integration.py \
  tests/test_protocol_preflight.py \
  tests/test_experiment_protocol_contract.py
```

Expected: all selected tests pass.

### Task 3: Propagate calendarization identity through KNN and sealed result metadata

**Files:**
- Modify: `src/protocols/candidate_pool.py`
- Modify: `src/source_selection/source_selector.py`
- Modify: `src/transfer_methods/source_failure_tolerance.py`
- Modify: `src/constants.py`
- Modify: `scripts/run_full_paper_experiments.py`
- Modify: `tests/test_candidate_pool_digest.py`
- Create: `tests/test_d2_calendarization_digest_chain.py`

**Interfaces:**
- `build_candidate_pool_digest(...)` remains backward-compatible for D1/D3–D6 and accepts optional D2 source identity fields.
- Shared selector metadata emits `d2_source_calendarization_rule_version`, `d2_source_authority_digest`, `d2_consumer_frame_fingerprint`, and `d2_sealed_identity`.

- [x] **Step 1: Write failing digest-chain tests**

```python
def test_d2_calendarization_identity_changes_candidate_and_sealed_digests():
    source, target = _configured_d2_frames()
    baseline = SourceSelector().select_top_k_sources(
        target, source, feature_cols=('sales',), k=1,
        group_cols=('brand_id', 'item_id'), weight_mode='inverse_distance',
    )
    mutated = source.copy()
    mutated.attrs = source.attrs.copy()
    mutated.attrs['d2_source_calendarization_rule_version'] = 'd2_source_calendarization_v2'
    changed = SourceSelector().select_top_k_sources(
        target, mutated, feature_cols=('sales',), k=1,
        group_cols=('brand_id', 'item_id'), weight_mode='inverse_distance',
    )

    assert baseline['meta']['candidate_pool_digest'] != changed['meta']['candidate_pool_digest']
    assert baseline['meta']['d2_sealed_identity'] != changed['meta']['d2_sealed_identity']


def test_legacy_candidate_digest_vectors_remain_unchanged_without_d2_identity():
    assert build_candidate_pool_digest(**WITHOUT_INPUT) == '7d7e0e0d6a08841426df0cea2273e420ae5d4b4dbc12c4c36e5cbf21e1328c72'
```

- [x] **Step 2: Run the digest tests and verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q tests/test_d2_calendarization_digest_chain.py tests/test_candidate_pool_digest.py
```

Expected: the new D2 metadata assertions fail because candidate digest and selector output do not yet include calendarization identity.

- [x] **Step 3: Implement optional digest inputs and final sealed identity propagation**

Extend the canonical candidate-pool payload only when all D2 identity fields are present. Compute the final sealed identity from rule version, source authority digest, consumer fingerprint, candidate digest, and selection-result digest. Add the fields to shared selection metadata, transfer metadata extraction, result schema, and serialized result rows. Keep existing non-D2 digest values byte-for-byte stable.

- [x] **Step 4: Run digest and result-contract tests**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q \
  tests/test_d2_calendarization_digest_chain.py \
  tests/test_candidate_pool_digest.py \
  tests/test_strict_result_contract.py \
  tests/test_d4_d6_source_authority.py
```

Expected: all selected tests pass and legacy digest fixtures remain unchanged.

### Task 4: Final verification

**Files:**
- Verify only; no additional production files.

- [x] **Step 1: Run the complete focused D2/protocol suite**

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q \
  tests/test_d2_source_calendarization.py \
  tests/test_d2_source_calendarization_integration.py \
  tests/test_d2_calendarization_digest_chain.py \
  tests/test_protocol_preflight.py \
  tests/test_candidate_pool_digest.py \
  tests/test_daily_knn_protocol.py \
  tests/test_source_selector_window_leakage.py \
  tests/test_strict_result_contract.py \
  tests/test_d4_d6_source_authority.py
```

Expected: all selected tests pass within 180 seconds.

- [x] **Step 2: Run static checks**

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m py_compile \
  src/protocols/d2_source_calendarization.py \
  src/protocols/candidate_pool.py \
  src/protocols/runner_adapter.py \
  src/source_selection/source_selector.py \
  scripts/validate_d1_d6_protocol_inputs.py \
  scripts/run_full_paper_experiments.py
git diff --check
```

Expected: compilation succeeds and `git diff --check` emits no output.

- [x] **Step 3: Review the diff and report exact verification evidence**

Confirm only the D2 source calendarization and identity-chain files changed; do not run any D1–D6 experiment or large data-generation command.
