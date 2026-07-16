# Gate 1R Contract Re-freeze Record

Gate: `Gate 1R Contract Consolidation and Re-freeze`

Re-freeze date: `2026-07-16`

Record status: `CONTRACT RE-FROZEN`

## Human decision authority

- Source decision book: `/Users/ming/Desktop/复现实验/实验确定信息/Gate_1R_合同补全与重新冻结决策书_预填写.docx`
- Source decision book SHA-256: `sha256:4aaebe5f07d3dc0e61ada72dbe0625c82615ba74577e91b72b46b07a709c689d`
- Decision interpretation: `all green-highlighted entries are approved final decisions`
- Human business decision status: `COMPLETE`
- Genuine unresolved business decision: `NONE`

The old Word template legend is not used as a decision source. Green-highlighted decisions were normalized into affirmative contract clauses, including D2 175/180 closure, D5 885/900 closure with retention of `48/1159415`, and the D6 dual `weekday`/`wday` schema.

## Superseded identities

- Old contract digest: `sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e`
- Old contract status: `SUPERSEDED`
- Previous Gate 1X implementation commit: `b8c1186ba9d8f96f98368a0fc5b438312a0e8813`
- Previous implementation status: `SUPERSEDED AS FINAL GATE 1X IMPLEMENTATION BASIS`
- Supersede reason: implemented before the Gate 1R contract gap was fully closed
- Failed controlled rerun: `20260716T112325Z-e2cfe3f889ec43fc983c4dac2dd1bb71`, D3 producer, `HISTORY_FUTURE_ROW`

## New formal identity

- Contract file: `docs/protocol/gate1_frozen_transformation_contract.md`
- Contract SHA-256: `sha256:85713b9d13cae3c017c4856b6a0f42a49d6074aebbb729171d60b95baa42eb74`
- Scope file: `docs/protocol/gate1_implementation_scope.md`
- Scope SHA-256: `sha256:98107929ea310e7fc304d2631803092e68c01e51fa992da96a3c3118b628eeb4`
- Matrix file: `docs/protocol/gate1_contract_traceability_matrix.md`
- Matrix SHA-256: `sha256:80545e2739dacdedfd8e60857bd8828dbf2102db37fc310ae4fc994b194e1da3`
- Combined formal identity rule: SHA-256 of exact LF-terminated UTF-8 records in the order `decision_book_sha256`, `contract_sha256`, `scope_sha256`, `matrix_sha256`.
- Combined formal identity digest: `sha256:3d11fef7b4edeb9fc804cc61455095b59e2c995afda11ba7d2c2a8afed7000e6`
- Digest sidecar: `docs/protocol/gate1_frozen_transformation_contract.sha256`

## Freeze provenance

- Starting branch: `codex/zuihou`
- Starting HEAD: `b8c1186ba9d8f96f98368a0fc5b438312a0e8813`
- Starting working tree: `clean`
- Freeze commit: `resolved by the commit containing this record`
- Commit count required: `one independent Gate 1R commit`
- Suggested commit message: `docs: refreeze Gate 1R transformation contract`

## Files changed in this Gate 1R commit

1. `docs/protocol/gate1_frozen_transformation_contract.md` — consolidated G01–G16 and complete D1–D6 executable clauses.
2. `docs/protocol/gate1_implementation_scope.md` — froze the complete one-time D1–D6 implementation boundary.
3. `docs/protocol/gate1_contract_traceability_matrix.md` — closed decision-to-contract/component/view/proof/test/readiness/publication/failure mappings.
4. `docs/protocol/gate1_frozen_transformation_contract.sha256` — recorded new, old, component, and combined identities.
5. `docs/protocol/gate1r_contract_refreeze_record.md` — recorded supersede, provenance, scope, and handoff.
6. `tests/test_gate1_frozen_acceptance.py` — added contract completeness, traceability, digest, supersede, and scope-boundary checks.

## Prohibited changes and actions

The Gate 1R commit does not modify producer, operator, transformation, model, training, raw/parent/sealed authority, schema artifact, manifest artifact, deployment root, failed private build, historical report, formal output, or the source decision-book DOCX. It does not run materialization, producer, training, validation, deployment, publication, readiness preflight, or Gate 1X controlled rerun.

The files that must remain unchanged are:

```text
tools/operations/materialize_d1_d6_sealed_authority.py
scripts/adopt_and_seal_d3_d6.py
src/protocols/gate1_transformation.py
```

## Next legal stage

- Gate 1X rerun now allowed: `NO`
- Next legal stage: `one-time Gate 1X Implementation`
- Gate 1X implementation must bind: decision-book SHA-256, new contract digest, scope SHA-256, matrix SHA-256, and this Gate 1R freeze commit SHA.
- Controlled rerun remains prohibited until implementation, real-input readiness preflight, acceptance evidence, and publication identity checks are complete.
