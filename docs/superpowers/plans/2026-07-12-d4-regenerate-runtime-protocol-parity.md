# Dataset4 Regeneration/Runtime Protocol Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Dataset4 JSON regeneration select sources through the same runtime shared-protocol path as formal training and record auditable candidate diagnostics.

**Architecture:** The existing `configure_protocol_frames` and shared `SourceSelector` remain the authority for eligibility, 30-day completeness, scaling, ranking, and weights. Regeneration adds a Dataset4-only adapter that applies the runtime domain policy, validates JSON targets, configures each target, and then invokes that selector. Dataset5/Dataset6 regeneration retains its current path; shared selector diagnostics gain only non-behavioral audit fields.

**Tech Stack:** Python 3, pandas, numpy, pytest, existing protocol modules.

## Global Constraints

- Do not change D4 protocol definitions, `SourceSelector` distance/ranking/weight algorithms, parquet loading semantics, or training execution.
- Do not add D4-specific store/category predicates outside shared protocol rules.
- D4 historical JSON target filters are read and reported as-is; do not rewrite `second_category_id=20` to `first_category_id=15`.
- D5/D6 regeneration behavior remains unchanged; their same-group protocol is regression-tested.
- Never use `--write`, do not modify solidified JSON/parquet, do not commit or push.
- Run every Python pytest or dry-run command through `python tools/protection/codex_timeout.py`; stop immediately if it returns 124.

---

### Task 1: Expose shared-path candidate diagnostics

**Files:**
- Modify: `src/source_selection/source_selector.py:165-199`
- Modify: `tests/test_d4_d6_source_authority.py:8-43`

**Interfaces:**
- Consumes: `SelectionResult`, `target_df.attrs["protocol_candidate_keys"]`, and `result.excluded_candidates`.
- Produces: `selection["meta"]` fields `selection_path`, `eligible_candidate_keys`, `valid_30d_candidate_keys`, `eligible_candidate_count`, `valid_30d_candidate_count`, `selected_count`, and `observed_days` for every shared-protocol call.

- [ ] **Step 1: Write the failing metadata contract test**

```python
def test_shared_selection_reports_explicit_protocol_path_and_candidate_layers() -> None:
    result = _select_d4_shared_fixture_with_one_incomplete_candidate()
    assert result["meta"]["selection_path"] == "shared_protocol"
    assert result["meta"]["eligible_candidate_count"] == 4
    assert result["meta"]["valid_30d_candidate_count"] == 3
    assert result["meta"]["selected_count"] == 3
    assert result["meta"]["observed_days"] == 30
```

- [ ] **Step 2: Run the test to verify RED**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_regenerate_runtime_protocol_parity.py::test_shared_selection_reports_explicit_protocol_path_and_candidate_layers -q
```

Expected: FAIL because `selection_path` and candidate-layer fields are absent.

- [ ] **Step 3: Add audit-only metadata in the shared selector**

```python
excluded_keys = {tuple(item["source_key"]) for item in excluded}
eligible_keys = list(target_df.attrs["protocol_candidate_keys"])
valid_keys = [key for key in eligible_keys if tuple(key) not in excluded_keys]
meta.update({"selection_path": "shared_protocol", "eligible_candidate_keys": eligible_keys,
             "valid_30d_candidate_keys": valid_keys, "eligible_candidate_count": len(eligible_keys),
             "valid_30d_candidate_count": len(valid_keys), "selected_count": len(sources),
             "observed_days": 30})
```

- [ ] **Step 4: Run the focused shared-selector tests to verify GREEN**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_source_selector_shared_protocol.py tests/test_d4_d6_source_authority.py -q
```

Expected: PASS with the new metadata contract and no selection behavior change.

### Task 2: Route Dataset4 regeneration through runtime policy and shared frames

**Files:**
- Modify: `scripts/regenerate_solidified_knn.py:18-324`
- Modify: `tests/test_d4_d6_regenerate_schema_diagnostics_parity.py:17-153`

**Interfaces:**
- Consumes: `apply_runtime_source_domain_policy`, `validate_runtime_target_domain`, `configure_protocol_frames`, JSON `group_cols`, target `entity_id`, and runtime source/target frames.
- Produces: Dataset4 source policy diagnostics with target-only scope; configured target/source frames that force `selection_path="shared_protocol"`; per-target selection metadata retaining the shared selector diagnostics.

- [ ] **Step 1: Write failing Dataset4 policy/schema assertions**

```python
def test_d4_regeneration_policy_matches_runtime_target_only_scope() -> None:
    result = _filter_source_for_scenario(source, dataset_id=4, scenario="without", old_payload=payload)
    assert result.diagnostics["domain_filter_scope"] == "target_only"
    assert result.diagnostics["domain_filter_applied_to_source"] is False
    assert result.diagnostics["source_pool_policy"] == "without_information_sharing_same_store"
```

- [ ] **Step 2: Run the schema test to verify RED**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_d6_regenerate_schema_diagnostics_parity.py -q
```

Expected: FAIL for Dataset4 WITHOUT because the regeneration helper currently uses default source-pool filtering.

- [ ] **Step 3: Add the Dataset4-only runtime adapter**

```python
runtime_config = {"dataset_id": 4, "info_sharing": scenario, "entity_col": "entity_id"}
source_df = apply_runtime_source_domain_policy(source_df, old_payload, runtime_config)
validate_runtime_target_domain(target_df, list(old_payload.get("results", {})), old_payload, runtime_config)
configured_source, configured_target = configure_protocol_frames(
    source_df, target_entity_df, dataset_id=4, scenario=scenario,
    group_cols=group_cols, grouping_col=None, observed_start=observed_start)
selected = SourceSelector().select_top_k_sources(
    target_df=configured_target, source_df=configured_source,
    feature_cols=feature_cols, k=k, group_cols=group_cols)
assert selected["meta"]["selection_path"] == "shared_protocol"
```

Keep `_filter_source_for_scenario` behavior for Dataset5/Dataset6 unchanged. Attach `source_pool_entity_count` to source-policy diagnostics and preserve exact old target-filter content in dry-run output.

- [ ] **Step 4: Run regeneration schema and check-only tests to verify GREEN**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_d6_regenerate_schema_diagnostics_parity.py tests/test_regenerate_solidified_knn_check_only.py -q
```

Expected: PASS; Dataset4 diagnostics show target-only policy while Dataset5/Dataset6 assertions preserve their existing behavior.

### Task 3: Add end-to-end synthetic runtime/regeneration parity tests

**Files:**
- Create: `tests/test_d4_regenerate_runtime_protocol_parity.py`

**Interfaces:**
- Consumes: public regeneration selection adapter and the formal runtime sequence `configure_protocol_frames → SourceSelector`.
- Produces: direct comparisons of key sets, exclusions, Top-K ordering, distances, weights, shared path metadata, target-only source behavior, and insufficient-pool exception fields.

- [ ] **Step 1: Write the failing WITHOUT parity test**

```python
def test_d4_without_runtime_and_regeneration_select_identical_shared_candidates() -> None:
    runtime, regenerated = _select_both("without")
    assert _keys(runtime.meta["eligible_candidate_keys"]) == _keys(regenerated.meta["eligible_candidate_keys"])
    assert _keys(runtime.meta["valid_30d_candidate_keys"]) == _keys(regenerated.meta["valid_30d_candidate_keys"])
    assert _keys(runtime.sources) == _keys(regenerated.sources)
    np.testing.assert_allclose(_distances(runtime), _distances(regenerated))
    np.testing.assert_allclose(_weights(runtime), _weights(regenerated))
    assert regenerated.meta["selection_path"] == "shared_protocol"
```

Fixture members: same-store/same-category, same-store/different-category, cross-store/different-category, same product in another store, and an incomplete 29-day source. Assert only same-store, different-product, complete keys survive.

- [ ] **Step 2: Run the test to verify RED**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_regenerate_runtime_protocol_parity.py::test_d4_without_runtime_and_regeneration_select_identical_shared_candidates -q
```

Expected: FAIL before Task 2 implementation because regeneration uses the legacy path.

- [ ] **Step 3: Add WITH, target-only, fail-fast, and D5/D6 regression tests**

```python
def test_d4_with_allows_cross_store_cross_category_but_excludes_same_product() -> None:
    runtime, regenerated = _select_both("with")
    assert _keys(runtime.meta["eligible_candidate_keys"]) == {
        ("166", "259"), ("167", "260"), ("168", "261"), ("169", "262")
    }
    assert _keys(regenerated.meta["eligible_candidate_keys"]) == _keys(runtime.meta["eligible_candidate_keys"])
    assert ("168", "258") not in _keys(regenerated.meta["eligible_candidate_keys"])

def test_d4_domain_filter_is_target_only_and_invalid_target_fails_validation() -> None:
    source_before = _source_entities(_source_frame())
    selection = _regeneration_select("without", domain_filter={"second_category_id": 20})
    assert selection.source_policy["domain_filter_applied_to_source"] is False
    assert _source_entities(selection.source_frame) == source_before
    with pytest.raises(ProtocolViolation, match="target domain validation failed"):
        _regeneration_select("without", target_second_category_id=99, domain_filter={"second_category_id": 20})

def test_d4_k3_insufficient_pool_reports_the_same_core_failure_fields() -> None:
    runtime_error = _capture_insufficient_error(_runtime_select, "without", _two_valid_source_frame())
    regenerated_error = _capture_insufficient_error(_regeneration_select, "without", _two_valid_source_frame())
    assert type(regenerated_error) is type(runtime_error)
    assert (regenerated_error.required_k, regenerated_error.valid_count) == (3, 2)
    assert regenerated_error.exclusions == runtime_error.exclusions

def test_d5_and_d6_same_group_protocol_still_excludes_other_groups() -> None:
    for dataset_id, grouping_col in (("D5", "family"), ("D6", "dept_id")):
        source, target = _same_group_fixture(grouping_col)
        configured_source, configured_target = configure_protocol_frames(
            source, target, dataset_id=dataset_id, scenario="with",
            group_cols=("store_id", "product_id"), grouping_col=grouping_col,
            observed_start="2024-01-01")
        assert configured_target.attrs["protocol_candidate_keys"] == (("166", "259"), ("167", "260"))
```

For each insufficiency result, compare exception class, `required_k`, `valid_count`, candidate exclusions, configured target key, and scenario. For the target-only test, use the historical `second_category_id` filter as an input fixture without asserting it is the permanent production field.

- [ ] **Step 4: Run the new parity file to verify GREEN**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest tests/test_d4_regenerate_runtime_protocol_parity.py -q
```

Expected: PASS with all six required parity checks.

### Task 4: Verify focused contracts and execute protected Dataset4 dry-run

**Files:**
- No source changes expected.
- Generate only: `outputs/feature_consistency/d4_regenerate_runtime_parity_<timestamp>/`

**Interfaces:**
- Consumes: finalized CLI and existing Dataset4 parquet/JSON.
- Produces: non-destructive JSON copies, diff summary, and a Store 166 parity table for targets `166_258`, `166_432`, `166_433`, `166_313`, and `166_311` under WITH and WITHOUT.

- [ ] **Step 1: Confirm CLI defaults and write protection**

Run:
```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/regenerate_solidified_knn.py --help
```

Expected: help documents a non-overwriting default and the temporary output-root option.

- [ ] **Step 2: Run all requested focused tests**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python -m pytest \
  tests/test_d4_regenerate_runtime_protocol_parity.py \
  tests/test_d4_candidate_protocol.py \
  tests/test_d4_d6_domain_filter.py \
  tests/test_experiment_protocol_contract.py \
  tests/test_source_selector.py \
  tests/test_source_selector_shared_protocol.py \
  tests/test_source_selector_window_leakage.py \
  tests/test_d4_d6_regenerate_schema_diagnostics_parity.py \
  tests/test_d4_d6_source_authority.py -q
```

Expected: PASS; substitute only an actually absent requested filename after documenting the substitution.

- [ ] **Step 3: Run the protected, non-overwriting D4-only dry-run**

Run:
```bash
PYTHONPATH="$PWD" python tools/protection/codex_timeout.py .venv/bin/python scripts/regenerate_solidified_knn.py \
  --datasets 4 \
  --diff-out "outputs/feature_consistency/d4_regenerate_runtime_parity_$(date +%Y%m%d_%H%M%S)"
```

Expected: only generated JSON and summaries inside the timestamped output directory; snapshot verification confirms no file under `configs/solidified/knn/Dataset4/` changed.

- [ ] **Step 4: Inspect final diff evidence**

Run:
```bash
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors; diff limited to the planned code/tests/spec/plan files; existing untracked user files are unchanged.
