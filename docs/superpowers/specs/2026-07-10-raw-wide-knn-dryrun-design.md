# Raw/Wide KNN Dry-run Design

## Goal

Add a read-only diagnostic for Dataset 4–6 that compares the current narrow
solidified source pool with a temporary, windowed, wide-clean source pool.

## Boundary

The new script may write only under its timestamped
`outputs/feature_consistency/raw_wide_knn_dryrun_<timestamp>/` directory. It
must never modify solidified parquet, KNN JSON, experiment-result CSV, or final
summary artifacts, and it never invokes transfer-learning training.

## Data and identity contract

`configs/solidified/knn/Dataset{N}/knn_*_info_sharing.json` is authoritative
for `group_cols`, `feature_cols`, domain filter, targets, and KNN settings.
The wide-clean adapter creates an entity key by joining the JSON `group_cols`;
it does not reuse legacy store-only or city-store `entity_id` conventions. It
normalizes aliases to the current solidified source schema, records mappings,
and fails a dataset if a required KNN feature cannot be built.

## Window contract

The diagnostic derives windows through the existing D4–D6 runtime functions:
the target signature uses exactly 30 observed days starting at the solidified
training start, and source signatures use the preceding inclusive 300-day
history ending on the same cutoff. Raw inputs are transformed and filtered to
those windows before KNN selection; no target-test or future source values are
used.

## Candidate pools and cap

The script loads a reusable `dataset{N}-wide-clean.parquet` from
`--clean-input-root` when available; otherwise it constructs that file under
the run's `intermediate/` directory. D4 combines raw train/eval as the widest
available input and reports the input row counts. D5 and D6 reuse the existing
cleaning-feature rules while adapting entity keys and schema.

Eligible entities are sorted by a SHA-256 digest of their JSON `group_cols`.
`--max-source-entities` keeps the first N stable keys after window eligibility,
and the same cap function is applied to narrow, wide-with, and wide-without
comparisons. Reports retain uncapped diagnostics.

## Output and failure model

For each selected dataset, write the requested CSV and JSON and append a
dataset section to the Markdown report. The script emits a failed status row,
with a concrete error, when construction, schema mapping, windows, or features
cannot meet the contract. It never substitutes the narrow source as the wide
candidate.

## Verification

Tests cover domain-vacuity/effectiveness, hash-cap determinism, schema
comparison, 30/300-day window metadata, overlap/non-configured-domain counts,
and output-path protections. Required legacy domain-policy tests remain part of
the verification command set.
