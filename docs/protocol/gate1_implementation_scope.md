# Gate 1X One-Time Implementation Scope

Scope ID: `gate1_implementation_scope`

Scope version: `1R.1.0`

Scope status: `FROZEN FOR ONE-TIME GATE 1X IMPLEMENTATION`

Execution status: `NOT STARTED`

Semantic authority: `docs/protocol/gate1_frozen_transformation_contract.md`

Traceability authority: `docs/protocol/gate1_contract_traceability_matrix.md`

## 1. Boundary and authorization

This scope covers one parameterized D1–D6 implementation of the frozen contract. It does not authorize new business rules, a replacement raw authority, a parallel contract, an alternate runner, model redesign, metric changes, performance work, cloud work, baseline work, deployment, or an experiment rerun.

The only formal path is:

```text
contract/config resolution
→ parent/raw identity
→ entity/role/window slicing
→ approved calendarization
→ field-specific repair
→ safe-view generation
→ schema registry
→ canonical digest
→ independent proof
→ formal preflight
→ operator validation
→ publication gate
```

The implementation must cover all D1–D6 in one implementation and one commit. It must not create dataset-specific sub-Gates, failure-specific tickets, or patch checkpoints. An omission that belongs to this frozen contract is resolved in the same implementation.

The implementation must bind, without semantic substitution, the final decision-book SHA-256, contract digest, scope SHA-256, matrix SHA-256, and Gate 1R freeze commit SHA. Gate 1X is prohibited until this scope is implemented and real-input readiness preflight passes.

## 2. One-time work breakdown

| ID | required implementation responsibility | required artifact/evidence | readiness check | publication gate | stable failure family |
|---|---|---|---|---|---|
| I01 | Resolve the contract, decision-book identity, scope, matrix, and exact dataset parameters. | Immutable run identity with all component hashes. | Every identity matches the Gate 1R record. | Reject a run with any identity drift. | `CONTRACT_IDENTITY` |
| I02 | Validate parent/raw identity and the complete frozen raw authority set. | Authority manifest, file hashes, snapshot identity. | All listed files exist, match bytes, and are not redirected. | Publish only with an exact authority manifest. | `RAW_AUTHORITY` |
| I03 | Slice full parent data by entity, role, and window before repair. | Source history, target observed, blind, truth, and audit boundaries. | No source post-origin row or target at/before-origin row enters the wrong role. | Reject role overlap and future rows. | `ROLE_WINDOW` |
| I04 | Calendarize only approved entities and windows under Gregorian rules. | Exact key/date calendarization mask and row counts. | Every expected key/date is present exactly once. | Reject duplicate, missing, or out-of-window rows. | `CALENDARIZATION` |
| I05 | Apply field-specific history and forecast repairs. | Per-field availability decisions, repair masks, counts, and digests. | Every repair has a named authority and as-of rule. | Reject generic fill, unknown encoding, or missing authority. | `FIELD_REPAIR` |
| I06 | Close D2 175/180 and D5 885/900 cardinality gaps. | Missing-key inventory and approved reconstruction proof. | D2 is 180; D5 is 900 and includes all five targets. | Reject publication at 175 or 885. | `CARDINALITY` |
| I07 | Isolate D3 source/target roles and field roles. | D3 source pool, target-role proof, and field exclusion proof. | No target row at or before origin enters the forecast producer; Open/Customers/Promo rules hold. | Reject `HISTORY_FUTURE_ROW` and forecast truth leakage. | `D3_ROLE_ISOLATION` |
| I08 | Enforce D4 future covariates and audit exclusions. | Seven-field worker schema and hourly/stock audit schema. | Exactly seven approved future covariates reach worker; all hourly/stock fields are audit-only. | Reject any unapproved D4 future field. | `D4_FIELD_ROLE` |
| I09 | Implement D5 onpromotion, transactions, oil, holiday, and week rules. | Field decision report, lag-one oil proof, holiday merge proof, schema. | Encoding, prior, join scope, deletion, and row count all match contract. | Reject any forecast transactions, week, bad oil, or expanding holiday merge. | `D5_FIELD_RULE` |
| I10 | Implement D6 calendar, weekday/wday, SNAP, and sell-price exact joins. | Calendar schema, state SNAP mapping, price join proof. | `weekday,wday,wm_yr_wk` order and dtypes are exact; price keys are unique. | Reject alias guessing, wrong state SNAP, missing price, or non-exact join. | `D6_SCHEMA_JOIN` |
| I11 | Generate the five safe views with non-overlapping roles. | View manifests for source_history, target_observed, worker_safe_blind, evaluator_truth, audit_view. | Worker has no real evaluation sales or forbidden field. | Publish only if truth-isolation proof passes. | `VIEW_ISOLATION` |
| I12 | Register dataset/scenario/method schemas. | Ordered names, dtypes, roles, transforms, consumers, exclusions, schema digests. | Actual consumer frames equal registered schemas. | Reject extra, missing, reordered, or drifted fields. | `SCHEMA_REGISTRY` |
| I13 | Canonicalize and digest each formal object. | Published source, worker, truth, audit, schema, and content digests. | Independent recomputation equals declared identity. | Reject any digest mismatch or unstable sort. | `DIGEST_BINDING` |
| I14 | Bind independent proof to actual artifacts and consumer frames. | Proof covering authority, windows, keys, cardinality, repairs, exclusions, pools, views, and leakage. | Proof is recomputed from bytes/frames, not only declarations. | Reject incomplete or self-asserted proof. | `PROOF_BINDING` |
| I15 | Implement formal preflight before training or prediction. | Pass report or stable failure report. | Every contract, authority, as-of, schema, view, domain, and proof check runs before model calls. | Any failure blocks model, producer, and publication calls. | `PREFLIGHT` |
| I16 | Implement operator-independent validation. | Independent recomputation report and mutation evidence. | Operator does not reuse producer-declared booleans as proof. | Reject publication without independent evidence. | `OPERATOR_PROOF` |
| I17 | Implement manifest and publication gate identity checks. | Manifest with formal identity, artifact hashes, view digests, and proof digest. | Every artifact binds the same contract/raw/schema identity. | No deployment or publication on mismatch. | `PUBLICATION_IDENTITY` |
| I18 | Implement failure closure and rollback evidence. | Stable failure code, blocked status, and non-authoritative failure report. | A failed preflight cannot leave an authoritative artifact. | Reject partial or mixed-identity publication. | `FAILURE_CLOSURE` |
| I19 | Add real-input read-only readiness preflight. | D1–D6 readiness report without materialization, training, or publication. | Missing authority, schema, field, date, key, or proof blocks before producer. | Controlled rerun is unavailable until all six are ready. | `REAL_INPUT_NOT_READY` |
| I20 | Add parameterized positive and negative tests for D1–D6. | Acceptance evidence linked to matrix IDs and stable failure codes. | Tests cover normal, missing, duplicate, future perturbation, schema drift, leakage, and row expansion. | Freeze evidence must be complete before rerun authorization. | `ACCEPTANCE` |
| I21 | Deliver the production implementation as one Gate 1X commit. | One commit bound to Gate 1R identities. | No dataset patch commit and no semantic change outside the contract. | Review and identity verification precede any rerun. | `IMPLEMENTATION_COMMIT` |
| I22 | Enforce controlled-rerun prerequisites. | Preflight pass, test report, manifest identity, and run identity. | All D1–D6 are ready and no forbidden action occurred. | Only then can a separately authorized controlled rerun be considered. | `RERUN_PREREQUISITE` |

## 3. D1–D6 parameter coverage

The one-time implementation must parameterize every date, target key, source eligibility rule, KNN schema, cardinality rule, field role, safe view, digest object, and failure condition in the six dataset chapters of the frozen contract. It must explicitly cover D2's sales-zero/covariate reconstruction, D5's retained `48/1159415` target and field rules, and D6's dual weekday/wday schema and exact joins.

## 4. Explicit non-goals and prohibited changes

The implementation does not include changes to model architecture, model training policy, metrics, baseline comparison, cloud environment, deployment, raw data, parent authority, sealed authority, schema artifacts outside the implementation, manifest artifacts, failed private builds, historical reports, formal outputs, or the old Git commit.

The following files are outside this scope and must remain byte-identical during Gate 1R:

```text
tools/operations/materialize_d1_d6_sealed_authority.py
scripts/adopt_and_seal_d3_d6.py
src/protocols/gate1_transformation.py
```

No producer, operator, transformation, training, materialization, formal D1–D6 experiment, controlled rerun, deployment manifest, or publication may run in Gate 1R. The Gate 1R commit contains only contract-freeze documents, the digest sidecar, the re-freeze record, and directly relevant freeze tests.

## 5. Acceptance and handoff

The acceptance suite must prove G01–G16, all 23 contract items in each D1–D6 chapter, all matrix columns and mappings, unresolved-marker absence, sidecar byte identity, supersede relationship, and prohibited-file cleanliness. The handoff to Gate 1X is valid only after the single Gate 1R freeze commit is reviewed.
