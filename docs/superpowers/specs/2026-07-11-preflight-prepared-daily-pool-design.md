# D1–D6 Preflight Prepared Daily Pool Design

## Goal

Eliminate repeated full-source scans from `d1_d6_protocol_v1` preflight while preserving every selection and digest bit-for-bit and keeping formal `SourceSelector` and preflight on one production implementation.

## Architecture

`src.protocols.candidate_pool` owns an immutable `PreparedDailySequencePool`. `prepare_daily_sequence_pool(...)` performs one bounded-date projection, vectorized key normalization, duplicate detection, one groupby/pivot/reindex, and produces a key × 30-date float64 matrix plus validity and grouping indexes. It never uses `DataFrame.apply(axis=1)` and never scans the full dataframe inside a candidate loop.

`select_daily_sequence_sources(...)` accepts an optional prepared pool. Without one it prepares a pool once and follows the same selection path. With one it validates protocol group columns and observation dates, selects candidate rows through integer key indexes, computes one task-wide min/max, and calculates all Euclidean distances in a NumPy float64 batch. Ranking, anchored `1e-12` ties, inverse-distance `1e-8`, digests, failure behavior, and complete internal exclusions remain unchanged.

Preflight prepares one pool before iterating targets. A prepared-source configuration path builds exact protocol candidate keys from the pool's key/group metadata and attaches the same pool to a lightweight source view consumed by production `SourceSelector`; it does not copy or parse the full source per target.

## Diagnostics

Production selection retains complete `excluded_candidates`. Preflight JSON replaces the unbounded list with:

- `candidate_exclusion_count`
- `candidate_exclusion_reason_counts`
- `candidate_exclusion_samples` (default maximum 20)
- `candidate_exclusions_truncated`

Failure messages summarize counts and bounded samples rather than interpolating the complete exclusions. Diagnostic truncation does not enter either digest.

## Compatibility and Failure Semantics

Candidate and selection digest inputs are unchanged. Candidate ordering, raw/scaled vectors, distances, weights, ties, min/max, missing-date exclusions, duplicate-date failure, non-finite failure, and insufficient-K failure remain equivalent. No K shrink and no statistics fallback are permitted.

## D4 Without-Sharing Finding

The current D4 target keys are five products in store 166/category 20. The fixed source pool contains only product 242 and product 560 in that store/category, while the with-sharing category pool contains 1076 candidates. Thus `valid candidates=2 below K=3` is caused by the fixed source domain containing only two legal same-store/same-category sources, not by a category/store filter bug.

## Validation Boundary

Tests prove legacy-reference equivalence on small frames, one pool preparation across multiple targets, future invariance, observed sensitivity, all failure modes, bounded diagnostics, and the existing strict suite. The read-only D5 preflight is run only through the repository's 180-second wrapper. No training, regeneration, or fixed-data writes are allowed.
