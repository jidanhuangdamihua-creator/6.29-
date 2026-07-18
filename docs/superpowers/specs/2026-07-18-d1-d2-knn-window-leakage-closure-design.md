# D1/D2 KNN Observed Window Leakage Closure

## Goal

Close the D1/D2 KNN observed-window leakage end to end. The only valid KNN
window for D1 is `2017-06-01..2017-06-30`; the only valid KNN window for D2
is `2018-06-01..2018-06-30`. Both are inclusive, contain exactly 30
Gregorian calendar days, and end at the dataset origin.

## Root cause

The current `ObservationWindow` derives `observed_end` from a caller-provided
start date. The formal D1/D2 runner and solidified configs still provide
`2017-06-05` and `2018-06-05`, producing the stale `06-05..07-04` window.
`configure_protocol_frames` attaches metadata to full model frames while the
shared selector later extracts the legal dates. This leaves the authority,
the actual selector frame, and the solidified KNN identity out of sync, and it
does not provide a frame-level proof that future rows were absent before KNN
feature construction.

## Architecture

### 1. One authoritative D1/D2 window source

`src/protocols/experiment_protocol.py` will own the frozen D1/D2 origins and
the inclusive-window formula:

```text
observed_end = origin
observed_start = origin - (observed_days - 1) calendar days
observed_days = 30
```

`ObservationWindow` will expose an origin-based constructor. D1/D2 callers
will derive their window from the protocol authority and reject a supplied
start/end that disagrees with it. D3–D6 retain their existing extended-track
window behavior and are not changed by this task. The old D1/D2 start-date
constants and duplicate derivations will be removed from formal runtime
routing.

### 2. Explicit KNN frames before selector work

`configure_protocol_frames` will preserve the full source/target model frames
needed for downstream training and evaluation, but will construct exact
observed KNN copies before returning. Those copies will be exposed through
protocol frame metadata and consumed by the shared selector path.

The KNN copies will:

- normalize and validate `date` values;
- include only the closed interval `[observed_start, observed_end]`;
- reject missing or duplicate required dates where the protocol requires a
  complete sequence;
- record their actual min/max dates and canonical frame digests;
- be passed to candidate-pool preparation, feature extraction, scaling,
  distance calculation, ranking, and Top-K selection.

The full model frames will remain available for target testing and source
training. No future row will be used to build a KNN feature, candidate pool,
selection result, or digest. D2 sealed-source verification remains before
consumer-frame construction and runtime calendarization remains forbidden.

### 3. Metadata and digest closure

Selection metadata will be generated from the actual KNN copies, not copied
from configuration constants. It will include:

```text
origin
observed_start
observed_end
observed_days
boundary = inclusive
source_frame_min_date
source_frame_max_date
target_frame_min_date
target_frame_max_date
source_frame_digest
target_frame_digest
candidate_pool_digest
selection_digest
```

The candidate-pool digest input and selection identity will bind the actual
frame digests, canonical window, candidate keys, ordered source results, and
all relevant D2 sealed identity fields. Rebuilding a digest from changed
future sentinel rows will therefore either produce the same legal identity
when those rows are excluded or fail closed when the legal frame changes.

### 4. Solidified regeneration and formal identity

The formal regeneration entry point will cover D1 and D2 as well as the
existing D4–D6 modes. It will load the sealed D1/D2 source/target parquet,
construct both information-sharing scenarios through the shared protocol,
and write generated JSON only from real selector output. The four D1/D2
solidified configs will then be promoted through the regeneration workflow;
their old window metadata, source order, distances, and digests will not be
retained.

The D1/D2 sealed dataset and deployment authority records will be updated by
the deterministic authority builder where their KNN identity is bound. Every
updated file hash and enclosing identity digest will be recomputed from the
resulting bytes. Hashes will never be edited by hand. D3–D6 authority content
and freeze rules remain outside scope.

## Error handling

The pipeline will fail closed when a D1/D2 frame has an invalid date, a date
outside the frozen window in the KNN copy, an incomplete required sequence,
metadata disagreement between source and target, a digest mismatch, or a
future row crossing the selector boundary. A stale caller-provided D1/D2
window will raise a protocol violation rather than silently being accepted.

## Testing strategy

Tests will be written before production changes and will cover:

1. D1 and D2 origin-based window formula, inclusive 30-day cardinality, and
   exact boundary values.
2. Source and target future sentinels at `origin + 1` and `origin + 4`.
3. Frame-level min/max bounds and exact required-date coverage from
   `configure_protocol_frames`.
4. Captured selector inputs proving no `date > origin` reaches KNN feature,
   scaling, distance, candidate, or ranking code.
5. Metadata and digest recomputation from actual canonical KNN frame bytes.
6. All four D1/D2 scenarios and the formal regeneration output, including
   source order, Top-K, distances, window metadata, and identity digests.
7. Regression coverage proving D3–D6 window behavior is unchanged.

Only lightweight protocol, unit, integration, and regeneration checks will be
run in this task. Formal model training and D1–D6 experiment execution will
not be run by Codex.

## Non-goals

- No D3–D6 freeze or model/baseline refactor.
- No selector-only end-of-pipeline date filter.
- No manual JSON/hash editing.
- No reuse or publication of old D1/D2 KNN selections or dependent results.
