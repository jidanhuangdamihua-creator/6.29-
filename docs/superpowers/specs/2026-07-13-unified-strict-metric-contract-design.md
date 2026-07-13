# Unified Strict Metric Contract Design

## Goal

Make D1-D3 and D4-D6 use one fail-closed strict metric contract, preserve every seed-level result, enforce the formal `seed -> target -> dataset macro` aggregation hierarchy, and record every Friedman horizon/scenario stratum.

## Scope

This change covers:

- orchestration-owned metric identity construction and forwarding;
- strict original-sales-space metric extraction for No-TL, SS-TL, and all multi-source methods;
- D1-D6 result serialization and formal row eligibility;
- seed completeness and duplicate validation in formal aggregation;
- seed-level candidate output and mean-based target winners;
- stratified Friedman result output.

It does not change source selection, model training, dataset preparation, KNN configuration, formal seed values, or the definition of sMAPE.

## Shared Metric Identity

The metric contract owns one manifest-to-identity function. Given an orchestration sample manifest and a horizon, it returns exactly:

- `metric_target_key`
- `metric_horizon`
- `metric_sample_count`
- `metric_date_start`
- `metric_date_end`
- `metric_index_digest`

The function rejects empty manifests and manifests containing more than one target key. D1-D3 and D4-D6 both call this function; no runner derives identity from its own prediction payload.

Every method wrapper receives `expected_metric_identity`. In strict mode, missing identity, mismatched identity, or a prediction length that differs from `metric_sample_count` raises `MetricProtocolError`. No-TL, SS-TL, and the four multi-source methods follow this rule equally.

## Strict Metric Data Flow

The orchestration layer builds the identity from the sample manifest and passes it with the strict metric protocol. Method implementations return the prediction payload required by `_extract_method_metrics`: `y_true`, `y_pred`, scaler/feature metadata when inverse transformation is required, and the orchestration identity.

`_extract_method_metrics` is the single canonical boundary that computes or validates original-sales-space metrics and emits all strict audit fields. D1-D3 and D4-D6 result rows copy the complete audit and identity payload from that result.

No-TL continues to train only on target data. Its bottom runner must preserve the full metric audit payload and identity instead of manually selecting a legacy subset. Its wrapper passes the strict protocol and expected identity into `_extract_method_metrics`, preventing a valid strict result from being converted to an error row.

SS-TL receives and preserves the same expected identity. Multi-source behavior remains unchanged except that identity is mandatory in strict mode.

## Formal Eligibility

`is_formally_comparable_smape_row` requires every field in `METRIC_IDENTITY_FIELDS`, in addition to the existing strict contract fields. Identity fields must be non-empty; horizon and sample count must be positive integers; date bounds must be present; and the index digest must be present.

Formal aggregation never synthesizes a target such as `GLOBAL`. Missing dataset, target, method, horizon, scenario, seed, or identity is a contract violation rather than an exclusion or fallback.

## Seed Validation and Aggregation

`build_formal_smape_aggregates` accepts an explicit `expected_seeds` argument. Formal D1-D6 callers pass the protocol's `FORMAL_SEEDS` value.

Within each `(dataset, target, method, horizon, sharing_scenario)` group, the helper requires the actual seed set to equal `expected_seeds`. It rejects:

- a missing expected seed;
- an unexpected seed;
- more than one row for the same full key including seed.

After validation, aggregation is performed in this order:

1. retain every eligible seed row;
2. average seeds for each dataset/target/method/horizon/scenario;
3. average targets for each dataset/method/horizon/scenario;
4. average datasets for each method/horizon/scenario;
5. rank methods only within the same horizon/scenario.

Information-sharing values are canonicalized before uniqueness checks so aliases cannot split or merge groups unpredictably.

## Best-Method Outputs

The aggregation command writes a seed-level candidate artifact containing every eligible result and its within-seed rank. This preserves the accepted row-level view.

`best_method_by_target` is derived separately from method seed means. It contains one winner for each dataset/target/horizon/scenario after validating the complete seed set. Dataset win counts are based on these mean-based target winners, not lucky individual runs.

RMSE remains diagnostic and is never used to select or rank a formal winner.

## Friedman Output

`run_friedman_test` returns a DataFrame with one row per `(horizon, sharing_scenario)` stratum. Each row contains:

- `horizon`
- `sharing_scenario`
- `n_datasets`
- `n_methods`
- `statistic`
- `p_value`
- `status`

Eligible strata use complete dataset blocks. A stratum with fewer than two datasets or three methods is still emitted with NaN statistics and `insufficient_data`. The statistical analysis script writes the returned DataFrame directly, so no stratum is discarded.

## Error Semantics

Contract failures are fail-fast and typed. They are not silently converted into normalized-space metrics, incomplete aggregates, or successful formal rows. Existing typed experiment error-row handling remains available at the entity boundary, but those rows are not formally comparable.

## Testing Strategy

Implementation follows test-driven development. Focused tests cover:

- D1-D3 forwarding and serialization of the complete identity/audit payload;
- D4-D6 No-TL preservation of a valid strict result;
- No-TL and SS-TL identity forwarding and mismatch rejection;
- formal eligibility rejection when any identity field is missing;
- missing, extra, and duplicate seed failures;
- exact seed-to-target-to-dataset macro values;
- seed-level output plus mean-based target winner selection;
- one Friedman output row for every horizon/scenario, including insufficient strata.

All Python and pytest commands run through `python tools/protection/codex_timeout.py`. Full D1-D6 experiments and data regeneration are outside this change.
