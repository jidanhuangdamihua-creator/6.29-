# Dataset4 Regeneration/Runtime Protocol Parity Design

## Goal

Make `scripts/regenerate_solidified_knn.py` reproduce Dataset4 source selection through the same shared protocol used by formal runtime training.  For every target and information-sharing scenario, candidate eligibility, 30-day validity, distances, ordering, Top-3 sources, inverse-distance weights, fail-fast behavior, and diagnostics must match.

## Scope

This change is limited to Dataset4 regeneration and its tests. It does not change formal training, source parquet generation, D1-D3, historical JSON files, or performance behavior. Dataset5 and Dataset6 retain their current same-group protocol; a regression test will lock that behavior.

## Existing Root Cause

Formal Dataset4 execution applies `apply_runtime_source_domain_policy`, validates JSON-selected targets with `validate_runtime_target_domain`, and configures each target through `configure_protocol_frames`. That attaches the protocol attrs that cause `SourceSelector` to delegate to `select_daily_sequence_sources`.

The regeneration script currently calls `apply_source_domain_policy` with its default `source_pool` scope and invokes `SourceSelector` on unconfigured frames. For Dataset4 WITHOUT, this can apply the stale category domain filter to sources. Because the frames lack shared protocol attrs, selection can use legacy statistical signatures instead of the formal 30-day daily-sales selection path.

## Design

### Shared runtime entrypoints

`regenerate_dataset_scenario` will use `apply_runtime_source_domain_policy` with a small regeneration config containing `dataset_id`, `info_sharing`, and `entity_col`. For Dataset4 this preserves all source rows and emits `domain_filter_scope="target_only"` and `domain_filter_applied_to_source=False`.

Before selecting any target, regeneration will call `validate_runtime_target_domain` for the JSON target keys. For each target it will derive the runtime observation start, call `configure_protocol_frames`, and pass the returned frames unchanged to `SourceSelector.select_top_k_sources`. No Dataset4-specific store/category predicates will be added to regeneration.

Historical Dataset4 JSON may still declare `second_category_id=20`. Regeneration will use that field only for target validation and diagnostics, report the exact filter used by the dry-run, and leave formal JSON unchanged. Migrating that field to `first_category_id=15` is a separate formal-JSON update and is not encoded as a permanent test assumption here.

The attrs returned by `configure_protocol_frames` are part of the interface: tests will assert `protocol_version`, `protocol_dataset_id="D4"`, `protocol_scenario`, `protocol_candidate_keys`, the observed start, and a 30-day window. Selector diagnostics will explicitly state `selection_path="shared_protocol"`; this prevents accidental legacy fallback even when legacy Top-K happens to match.

### Dataset4 protocol contract

The existing shared `ExperimentProtocol("D4")` remains authoritative:

| Scenario | Eligible source key rule |
| --- | --- |
| WITHOUT | same `store_id`; different `product_id` |
| WITH | any store; different `product_id` |

`require_same_group=False` means no category field participates in eligibility. The selector uses exactly 30 observed daily sales values, excludes incomplete candidates, applies stable tie-breaking, selects exactly K=3 or fails, computes Euclidean distance on the shared scaled representation, and computes inverse-distance weights.

### Diagnostics and JSON compatibility

Regenerated payloads retain their existing top-level schema and add protocol facts only inside the existing diagnostics/metadata structures. They record source-pool policy, domain-filter scope and source application status, same-group requirement, excluded candidate key fields, observed days, K, weight mode, selected path/version, and unambiguous counts: `source_pool_entity_count`, `eligible_candidate_count`, `valid_30d_candidate_count`, and `selected_count`. D4 policy labels will be `without_information_sharing_same_store` and `with_information_sharing_cross_store`; the stale same-domain label is not emitted for D4.

The command remains non-destructive by default. Generation writes only beneath an explicit temporary output root during acceptance; `--write` remains the only overwrite path and is not used.

### Tests and verification

Add `tests/test_d4_regenerate_runtime_protocol_parity.py` with synthetic fixtures covering both scenarios, cross-category/store membership, same-product exclusion, incomplete sequences, exact Top-3/distance/weight parity, equal K=3 insufficient-pool failures, target-only domain filtering, and a D5/D6 same-group regression. Insufficient-pool parity compares exception type plus target key, scenario, required K, eligible count, and valid count; only incidental prose may differ.

Existing regeneration schema tests will be updated only where Dataset4 diagnostics intentionally change. Focused test commands in the request are the verification set. A real Store 166 five-target, WITH/WITHOUT dry-run will generate only to a timestamped directory beneath `outputs/feature_consistency` and report parity diagnostics without training.

## Non-goals and safety

- No formal JSON overwrite, commit, push, or parquet rebuild.
- No full pytest, D1-D6 pipeline, or training command.
- Experiment-like Python commands use `python tools/protection/codex_timeout.py ...` and stop immediately if it returns 124.
