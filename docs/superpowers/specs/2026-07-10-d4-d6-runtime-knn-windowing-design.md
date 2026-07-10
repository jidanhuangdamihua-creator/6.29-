# D4–D6 Runtime-Authoritative KNN Windowing Design

## Scope

This change establishes a leakage-free D4–D6 KNN baseline without changing the existing feature set, statistical signature, Euclidean distance, or scaling behavior. It applies only to the D4–D6 runtime-authoritative path. D1–D3 behavior is outside this change.

The formal D4–D6 training path continues to select sources through the runtime `SourceSelector`. Solidified JSON Top-K rows may still define target lists and may restrict smoke/source-limit candidate pools, but they are not training evidence and must never be labeled as the sources actually selected for formal runtime training.

## Evidence and Protocol Boundary

The paper-supported core is:

- the target observed window contains target training plus target validation;
- the reference split is 15 training days plus 15 validation days, for 30 observed calendar days;
- KNN similarity uses the target's first 30 observed days and source rows from the same 30 calendar dates.

The 300-day source history length is not claimed as a paper-specified constant. It is an existing D4–D6 reproduction-engineering protocol used to standardize the source history and pretraining window.

The selected D4–D6 protocol is:

- `target_observed_start = target train_start`;
- `target_observed_end = target_observed_start + 29 days`;
- the observed date set contains exactly 30 calendar days and covers target train plus validation;
- `source_history_end = target_observed_end`;
- `source_history_start = source_history_end - 299 days`;
- both history bounds are inclusive, yielding exactly 300 calendar days;
- target test rows and source rows after `target_observed_end` are excluded from KNN;
- KNN signatures for both target and source are calculated only on the same 30 observed dates.

The solidified JSON `target_train_window` is a train-only legacy field and is not the runtime observed-window authority. Runtime observed bounds must be materialized explicitly in DataFrame attrs/window metadata by the D4–D6 loader.

## Data Flow

### Loader

`load_parquet_source_target()` keeps the full target evaluation frame from `train_start` through `test_end`, because downstream training and evaluation still need those rows.

For D4–D6 it must:

1. validate the target and source date columns;
2. derive the 30-day observed bounds from `train_start` and the existing 15+15 day split;
3. derive the inclusive 300-day source-history bounds from `target_observed_end`;
4. crop the loaded source frame to `[source_history_start, source_history_end]`;
5. attach explicit observed/history bounds and the runtime-authoritative protocol marker to both frames.

Missing dates or required bounds fail fast in this D4–D6 path. Target test rows remain in the returned target frame but are not eligible for signature construction.

### Runtime Source Selection

`SourceSelector` detects the D4–D6 runtime-authoritative protocol from explicit attrs. In that path it must:

1. require a valid `date` column and all required observed/history metadata;
2. slice the target to the exact 30-date observed set;
3. reject a target frame that does not contain every required observed date;
4. group sources using the existing group columns;
5. enforce the inclusive source-history bounds defensively, even if the loader already cropped the source frame;
6. reindex/filter each source group to the exact target observed date set;
7. skip a source missing any observed date and record a structured reason;
8. compute the unchanged mean/std/min/max/last signature on the aligned 30 rows;
9. compute the unchanged Euclidean distance and existing distance weights;
10. select runtime Top-K from the eligible aligned candidate pool.

No scaler or expanded 30-day sequence representation is introduced. Static/profile features already included by the existing information-sharing logic remain unchanged.

Legacy D1–D3 calls retain their current behavior. The new fail-fast checks are gated by the explicit D4–D6 runtime-authoritative protocol marker.

## Runtime Metadata

Each runtime selection result must expose metadata sufficient to identify the actual candidate pool and Top-K:

- `selection_authority = "runtime"`
- `protocol_version = "runtime_knn_windowed_stats_v1"`
- `target_observed_start`
- `target_observed_end`
- `source_history_start`
- `source_history_end`
- `target_test_excluded = true`
- `source_future_excluded = true`
- `source_alignment_mode` describing exact target-observed-date alignment
- `feature_cols`
- `representation = "mean_std_min_max_last"`
- `scaling = "none"`
- `scaler_fit_scope = "not_applicable"`
- `selected_sources_runtime`
- `candidate_pool_digest`
- `selection_result_digest`
- structured source-skip diagnostics

Digests use deterministic JSON serialization and SHA-256. The candidate-pool digest is calculated from the ordered eligible runtime source keys after domain filtering, source-history enforcement, and date-coverage validation. The selection-result digest is calculated from the deterministic runtime Top-K records, including source key, rank, distance, and weight.

Transfer methods must propagate the runtime selection metadata they receive from `SourceSelector`. D4–D6 result rows may store compact fields/digests while a diagnostics JSON stores the complete metadata, but `selected_sources` or `selected_sources_runtime` must always mean the actual runtime selection. JSON Top-K may only appear under an explicitly JSON-labeled field, if retained at all.

## Domain Filtering

Formal D4–D6 runtime filtering remains governed by `apply_source_domain_policy()`:

- without information sharing applies the normalized JSON domain filter;
- with information sharing keeps the full source pool while recording the JSON filter as non-applied metadata.

`regenerate_solidified_knn.py` must call the same shared policy rather than reproduce a D4-only special case. D4, D5, and D6 without-sharing therefore have identical filter semantics to runtime.

## Regenerated JSON Safety

When regeneration recomputes `results`, it must not copy an old `selection_metadata` block. The generated payload is rebuilt from explicitly allowed stable configuration fields plus new results, feature information, source-pool information, and metadata generated by the current runtime-compatible selection.

Required protocol inputs must be present or derivable from the D4–D6 fixed split and validated. Missing protocol metadata fails fast. Old `paper_observed_sequence`, old window diagnostics, and old date-alignment diagnostics are never inherited into a payload containing new results.

Generated JSON remains a configuration/diagnostic artifact. Its Top-K rows do not become formal training evidence under the runtime-authoritative protocol.

## Error Handling

- Missing D4–D6 observed/history metadata: fail fast.
- Missing or invalid target/source date column: fail fast.
- Target lacking any of the 30 required observed dates: fail fast.
- Individual source lacking any required observed date: skip it and record the missing-date reason.
- No eligible source after coverage checks: fail fast with aggregate diagnostics.
- Missing without-sharing domain-filter metadata: fail fast through the shared domain policy.
- Regeneration with insufficient protocol inputs: fail fast instead of inheriting old metadata.

## Tests and Acceptance

Focused tests will prove:

1. changing target test-period sales does not change runtime Top-K;
2. changing source rows after `target_observed_end` does not change runtime Top-K;
3. a crafted observed-period change can change runtime Top-K;
4. missing D4–D6 observed-window metadata fails fast;
5. runtime diagnostics/result metadata contains the actual runtime Top-K and authority marker;
6. JSON Top-K is not labeled as the actual training Top-K;
7. regenerated JSON cannot inherit stale `selection_metadata`;
8. D4–D6 without-sharing regeneration uses the same domain-filter semantics as runtime.

Only focused pytest files and directly related tests are run. No full D1–D6 experiment, representation ablation, or scaling ablation is part of acceptance.
