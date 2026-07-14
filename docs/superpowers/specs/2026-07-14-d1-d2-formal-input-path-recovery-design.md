# D1/D2 Formal Input Path Recovery Design

## Scope

Restore the previously approved D1/D2 formal input routing without changing
candidate contracts, calendar requirements, validation strictness, parquet
contents, or experiment behavior outside input-path identity.

## Chosen approach

Use the dedicated protocol artifacts introduced by `585fc9b4` and already
used by D1/D2 preflight after `f1d6b18d`:

- `数据集/派生数据/d1d2_protocol_v1/dataset1-source.parquet`
- `数据集/派生数据/d1d2_protocol_v1/dataset1-target.parquet`
- `数据集/派生数据/d1d2_protocol_v1/dataset2-source.parquet`
- `数据集/派生数据/d1d2_protocol_v1/dataset2-target.parquet`

The D1-D3 formal runner and unified run-plan input discovery must resolve the
same authoritative files. D3-D6 paths remain unchanged.

## Rejected alternatives

- Do not fall back to `数据集/固化数据` when a D1/D2 protocol parquet is
  missing; that would silently reintroduce the regression.
- Do not calendarize or synthesize rows at runtime.
- Do not loosen candidate-pool or complete-calendar validation.
- Do not mutate or resume an existing run plan that locked the old files.

## Implementation

1. Restore D1/D2 entries in `SOLIDIFIED_DATASET_PATHS` to the protocol-derived
   directory.
2. Centralize formal parquet path resolution in the unified runner so D1/D2
   input identity uses that directory while D3-D6 continue using
   `数据集/固化数据`.
3. Add regression tests proving both the D1-D3 runner and formal run-plan
   discovery select the same D1/D2 paths and do not select the old files.

## Verification

- First observe the new regression tests fail against the current code.
- Apply only the path-resolution changes and observe the tests pass.
- Run related unified-runner, formal-parquet, and protocol-preflight tests
  through `tools/protection/codex_timeout.py`.
- Run syntax/static checks and confirm no parquet or protocol files changed.
