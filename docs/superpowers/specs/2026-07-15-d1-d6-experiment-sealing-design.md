# D1–D6 Experiment Sealing Design

**Status:** Approved design, pending implementation

**Date:** 2026-07-15

**Implementation plan:** docs/superpowers/plans/2026-07-14-d1-d6-experiment-sealing.md

## 1. Purpose and normative scope

This document is the normative protocol for the D1–D6 formal experiment. It fixes data identity, predictor features, truth isolation, source selection and training, recursive rollout, artifact schemas, recovery, and final acceptance.

The implementation plan must implement this document exactly. If a task, test example, compatibility wrapper, or existing code path conflicts with this document, the conflict is an implementation defect rather than permission to choose the older behavior.

The design goals are:

1. prevent target blind truth from reaching fitting, selection, scaling, RFE, or recursive input;
2. keep all six methods comparable under one exact predictor schema per dataset;
3. recover from failure without rewriting accepted audit history; and
4. trace every accepted result to immutable data, schema, code, and artifact identities.

## 2. Fixed formal experiment contract

### 2.1 Target and source windows

For every dataset:

- target train: 15 natural days;
- target validation: 15 natural days;
- target blind evaluation: 180 natural days;
- source pretraining: one 180-day window ending at the target observed end;
- source KNN comparison: the final 30 days of that same 180-day source window.

The source window is 180 days total, not 30 plus 180:

~~~text
source 180-day window
├── first 150 days: source pretraining only
└── final 30 days: source pretraining and KNN
~~~

Formal horizons are (1, 2, 3, 4, 5). Formal seeds are (42, 43, 44, 45, 46).

| Dataset | Target start | Observed end | Blind end |
|---|---|---|---|
| D1 | 2017-06-01 | 2017-06-30 | 2017-12-27 |
| D2 | 2018-06-01 | 2018-06-30 | 2018-12-27 |
| D3 | 2015-01-03 | 2015-02-01 | 2015-07-31 |
| D4 | 2024-12-16 | 2025-01-14 | 2025-07-13 |
| D5 | 2017-01-17 | 2017-02-15 | 2017-08-14 |
| D6 | 2015-10-26 | 2015-11-24 | 2016-05-22 |

### 2.2 Formal sample counts

At blind origin O+i, horizon h predicts label date O+i+h. A prediction exists only when the label date remains inside the blind window.

~~~text
h1 = 180
h2 = 179
h3 = 178
h4 = 177
h5 = 176
~~~

Synthetic dates are not skipped, shifted, or removed. Runtime fill cannot change the sample calendar or these counts.

## 3. Immutable data sealing and provenance

### 3.1 Versioned sealed root

Formal resolvers read only a versioned immutable root such as:

~~~text
数据集/固化数据/d1_d6_sealed_v1/
~~~

Each dataset atomically publishes source, target, manifest, validation report, schema descriptors, and audit sidecars. The formal branch has no fallback to old derived directories or unsealed parquet files.

### 3.2 Hybrid provenance policy

- D1 and D2 use raw_rebuilt provenance.
- D3–D6 use adopted_solidified provenance for this protocol version.
- This version does not claim D3–D6 are traceable to raw data.
- An adopt-and-seal chain ends at the exact parent parquet observed during adoption.

Adopted manifests record at least:

~~~yaml
provenance_level: adopted_solidified
parent_artifact_sha256: "..."
parent_artifact_size_bytes: 0
parent_artifact_observed_at: "...Z"
parent_artifact_mtime_ns: null
parent_artifact_first_seen_at: null
parent_artifact_first_seen_source: null
parent_artifact_first_seen_reliability: unavailable
content_validation_level: structural_only
adopted_content_validated: false
content_validation_notes: >
  Only file identity, structural integrity, windows, schemas, KNN fingerprints,
  and feature protocol were validated. Historical numeric correctness was not
  reconstructed from raw data.
~~~

mtime is diagnostic metadata, not provenance authority. first_seen_at may be populated only from credible version control, object-store history, or immutable logs; otherwise it remains explicitly null.

### 3.3 Calendarization and fill engine

D1–D6 use one calendarize_and_fill engine and one rule format. Dataset-specific values may differ, but engine semantics and configuration identities are shared.

~~~yaml
fill_policy_engine_version: calendarize-fill/v1
fill_policy_shared_with_raw_rebuild: true
fill_policy_config_digest: "sha256:..."
~~~

Runtime fill policy is covariates_only. It cannot add, remove, reorder, or shift sample dates and cannot modify target truth.

Dataset canonicalization is a separate pre-sealing operation. The approved D2 raw rebuild encodes a dataset-authorized absent transaction day as canonical zero to preserve the paper KNN fingerprint. That operation is versioned and audited before the sealed target is published. No other target-sales fill is allowed without a new reviewed protocol version. Once sealed, worker and evaluator runtime code cannot fill or modify sales or y_true.

### 3.4 Source sales canonicalization

The source 180-day frame uses source_sales_canonicalization/v1 before KNN or scaling:

- source sales NaN becomes 0;
- finite negative source sales becomes 0;
- a calendarized missing source row produces NaN and therefore canonical sales=0;
- positive or negative infinity is a hard failure;
- KNN and source training use the same canonical value for every overlapping date.

This applies only to the source 180-day data used by pretraining and KNN. It does not authorize changing target observed sales or evaluator truth at runtime. Target observed sales and sealed evaluator truth must be finite and nonnegative after pre-sealing validation; an unapproved target repair is a hard failure.

Every repaired source row receives one reason:

~~~text
original_nan
original_negative
calendar_row_missing
~~~

The manifest records counts by reason, an affected-date digest, and a repair-mask SHA-256.

Non-sales columns use minimum intervention:

- unused audit or metadata fields retain nulls unchanged;
- only fields entering KNN, predictor schema, scaler, RFE, or recursive calculations are validated for finite input;
- there is no blanket numeric fillna(0) in the formal branch;
- a used field without an approved field-specific rule fails closed.

### 3.5 Dataset-level adopt failure

Adoption is atomic at dataset granularity. One required entity failure prevents publication of that dataset. Other sealed datasets remain sealed, but the six-dataset preflight is blocked.

There is no entity-level mixed provenance and no automatic fallback. An operator may explicitly choose a whole-dataset raw rebuild. If reliable reconstruction is unavailable, the dataset remains blocked.

## 4. Adopt validation and global preflight

### 4.1 Closed failure enumeration

AdoptValidationFailureReasonV1 is closed. An unknown validator exception maps to VALIDATOR_INTERNAL_ERROR and cannot become a warning.

~~~text
PARENT_ARTIFACT_MISSING
PARENT_ARTIFACT_UNREADABLE
PARENT_ARTIFACT_CORRUPT
PARENT_ARTIFACT_HASH_MISMATCH
PARENT_ARTIFACT_SIZE_MISMATCH
MANIFEST_SCHEMA_INVALID
MANIFEST_REQUIRED_FIELD_MISSING
PROVENANCE_METADATA_INVALID
SCHEMA_MISMATCH
COLUMN_ORDER_MISMATCH
DTYPE_MISMATCH
FEATURE_SCHEMA_DIGEST_MISMATCH
UNEXPECTED_COLUMN
REQUIRED_COLUMN_MISSING
ENTITY_SET_MISMATCH
ENTITY_DUPLICATED
ENTITY_REQUIRED_FIELD_NULL
ROW_COUNT_MISMATCH
PRIMARY_KEY_DUPLICATED
DATE_PARSE_FAILURE
DATE_DUPLICATED
DATE_ORDER_INVALID
DATE_DISCONTINUITY
TIME_WINDOW_OUT_OF_BOUNDS
TIME_WINDOW_LENGTH_MISMATCH
INSUFFICIENT_OBSERVATION_WINDOW
INSUFFICIENT_BLIND_WINDOW
FILL_POLICY_VERSION_MISMATCH
FILL_POLICY_CONFIG_MISMATCH
FILL_POLICY_EXECUTION_FAILURE
SYNTHETIC_DATE_COUNT_MISMATCH
KNN_WINDOW_LENGTH_MISMATCH
KNN_WINDOW_ALIGNMENT_MISMATCH
KNN_FEATURE_SCHEMA_MISMATCH
KNN_FINGERPRINT_MISMATCH
KNN_FINGERPRINT_NON_UNIQUE
KNN_FINGERPRINT_COLLISION
FUTURE_KNOWN_AUDIT_FAILED
FORBIDDEN_FEATURE_DETECTED
DEPENDENCY_CUTOFF_VIOLATION
SALES_DERIVED_FUTURE_LEAKAGE
OUTPUT_ARTIFACT_HASH_MISMATCH
OUTPUT_ATOMIC_PUBLISH_FAILURE
VALIDATOR_INTERNAL_ERROR
~~~

KNN_FINGERPRINT_COLLISION means different canonical bytes produced the same digest. Equal digests alone do not prove a collision.

~~~text
failure_reasons non-empty => status=failed
status=failed             => no sealed dataset publication
~~~

Free-text evidence cannot override an enum decision.

### 4.2 Immutable validation policy

Every proof pins:

~~~yaml
validation_policy_version: adopt-policy/v1
validation_policy_digest: "sha256:..."
validator_code_digest: "sha256:..."
~~~

Changing thresholds, enums, or tolerances requires a new policy version, new digest, new seal attempt, actor record, reason, and policy diff. Old reports are append-only. Global preflight accepts only the required policy digest.

### 4.3 Layered state display

Preflight distinguishes dataset sealing from experiment readiness:

- an already sealed dataset remains sealed;
- a failed dataset is adopt_validation_failed and has no formal artifact;
- six-dataset preflight becomes blocked and reports dataset, failure codes, report path, and report SHA-256;
- no supervisor or formal attempt is created while blocked.

## 5. Predictor feature protocol

### 5.1 Exact schema alignment

Each dataset has one immutable PredictorFeatureSchema shared by:

- all six methods;
- with- and without-information-sharing scenarios;
- source and target;
- train, validation, and blind rollout;
- h1–h5; and
- seeds 42–46.

The contract freezes ordered names, ordered dtypes, roles, transforms, dimension, and schema digest. Dataset schemas may differ; methods within a dataset may not.

The formal branch forbids select_dtypes, feature intersections, fallback columns, silent drops, and runtime column discovery.

### 5.2 Feature roles

Every field has exactly one role:

~~~text
target_signal
future_known
static_known
observed_dynamic
recursive_derived
evaluation_only
identifier_group_only
~~~

Future-known lineage records source type, authority, availability cutoff, dependencies, generation rule, and code digest. Its dependency graph cannot contain sales, truth, predictions, or sales-derived statistics.

Schedule fields qualify only when an independent pre-cutoff authority can reconstruct them. Otherwise they become observed-only or audit-only. Missing authority fails closed.

### 5.3 Observed-only dynamics

KNN use has a separate sealed disposition, KnnObservedDispositionV1 = knn_observed | audit_only. This disposition does not add a field to the formal predictor-role vocabulary.

The following fields are not formal predictors unless a later schema version proves future availability:

- D2: promo;
- D3: Customers, Open, Promo;
- D4: stock, activity, discount, and weather dynamics;
- D5: onpromotion, transactions, oil_price;
- D6: sell_price.

They are never formal predictors. Each must be classified before sealing as either knn_observed or audit_only; there is no runtime choice or unclassified fallback. A knn_observed field may enter only the observed 30-day distance vector and must be present and finite for every compared date. An audit_only field cannot enter distance, fitting, scaling, RFE, or rollout. D2 freezes KNN order as sales then promo to preserve the approved paper fingerprint. D3–D6 freeze their corresponding classifications and ordered KNN fields in the dataset schema; D4 leakage-risk fields remain audit_only.

Identifiers, group codes, and category codes do not enter the formal predictor. The currently approved static predictor exception is D5 perishable.

### 5.4 RFE alignment

MSML-TL-RFE receives the same full ordered tensor plus an equal-length boolean mask. Unselected transformed fields are zeroed for training and inference. Columns are never deleted or reordered. Sales is always retained.

The predictor schema digest remains common. RFE has a separate mask digest; non-RFE methods use the deterministic full-mask digest rather than null.

## 6. Four data views and truth isolation

Each target has four typed views:

1. knn_observed_frame: exact 30 observed days and exact KNN features;
2. observed_model_frame: exact 30 observed days under the predictor schema, including observed sales;
3. blind_covariate_frame: exact 180 blind days with future-known/static fields and safe identities only;
4. evaluator_truth_frame: exact 180 blind days with keys, date, y_true, is_synthetic_date, and truth_key.

Worker and evaluator caches are separate. Worker commands receive only worker cache. Evaluator paths use a high-entropy capability ID whose mapping exists only in the evaluator control plane. Worker manifests cannot contain truth paths, evaluator paths, capability IDs, path templates, or reconstructing parameters.

The prevention goal is accidental truth leakage through formal APIs, paths, manifests, and schemas. It is not a defense against a malicious same-UID process. Stronger isolation requires a separate UID or container and a different reported isolation level.

### 6.1 Truth tripwire

Formal fitting APIs do not accept truth. Tests use an external TruthAccessTripwire that logs every access before raising SealedTruthAccessError.

A test passes only when:

~~~text
fit completed successfully
attempted_access_count == 0
evaluator_loader_call_count == 0
~~~

Catching the exception still fails because the counter is nonzero.

### 6.2 Trace type boundary

WorkerPredictionTraceSchemaV1 does not contain y_true. Exact validation rejects extra, missing, reordered, or mistyped fields. Only evaluator code creates EvaluatedPredictionTraceSchemaV1, which adds y_true, is_synthetic_date, and join status.

## 7. KNN and source training

Calendarize and canonicalize the full 180-day source before eligibility. Only eligible 180-day sources enter KNN ranking. This prevents a source ranking on the final 30 days and failing later in the first 150 days.

KNN uses:

- target exact 30 observed dates;
- the same final 30 dates from each eligible source;
- declared features in fixed order;
- unscaled canonical float64 values;
- deterministic distance-then-key tie breaking.

IDs and role labels are forbidden. A non-sales null outside KNN/predictor projection remains untouched. A required field without an approved rule fails eligibility.

D2 retains the approved paper fingerprint: Items 4, 6, and 8 with approximate distances 24.98, 26.85, and 26.85.

## 8. Blind joint-horizon rollout

### 8.1 Independent streams

Each mutable history belongs to one exact stream:

~~~text
rollout_stream_key = SHA256(
  protocol_digest,
  dataset_id,
  target_entity_key,
  scenario,
  method,
  seed
)
~~~

Different datasets, targets, scenarios, methods, or seeds never share mutable history. Read-only observed bases and blind covariates may be shared.

For ensemble methods, only the final combined prediction may be committed. Internal source-model predictions never enter history.

### 8.2 Origin barrier and feedback

At every origin:

1. create one immutable history snapshot containing dates no later than the origin;
2. use observed sales for observed dates and previously committed clipped h1 for elapsed blind dates;
3. predict every valid horizon from the same snapshot;
4. inverse-transform and verify finite values;
5. clip to nonnegative original sales space;
6. commit current clipped h1 only after every valid horizon succeeds.

h2–h5 never enter history. No horizon sees the h1 produced at its own origin. Any horizon failure prevents the h1 commit and fails the seed bundle.

Sales-derived lags and rolling fields are recursive_derived. They use only observed sales plus committed clipped h1 and cannot depend on a date after the forecast origin.

### 8.3 Deterministic keys

~~~text
truth_key = SHA256(
  "truth-key/v1",
  evaluation_contract_digest,
  dataset_id,
  target_entity_key,
  label_date
)

sample_key = SHA256(
  "prediction-sample-key/v1",
  truth_key,
  forecast_origin,
  horizon
)

prediction_row_key = SHA256(
  "prediction-row-key/v1",
  sample_key,
  scenario,
  method,
  seed
)
~~~

Keys use fixed-order canonical UTF-8, ISO dates, typed entity values, integer horizons, and full 64-character SHA-256 hex. run_id is excluded. Evaluator verifies label_date = forecast_origin + horizon.

### 8.4 History hash chain

Each trace records pre-origin and post-h1-commit history digests. Adjacent origins form one continuous chain for the same stream. h2–h5 cannot appear in the commit chain.

## 9. Append-only recovery and state machine

### 9.1 Attempts and transitions

One immutable run root contains append-only attempts:

~~~text
attempts/<attempt_id>/attempt_manifest.json
attempts/<attempt_id>/scheduler_events/
attempts/<attempt_id>/worker_logs/
attempts/<attempt_id>/attempt_result.json
~~~

Allowed transitions:

~~~text
running -> partial_failed
running -> complete_unsealed
partial_failed -> running
complete_unsealed -> sealed_success
complete_unsealed -> sealed_failed
~~~

sealed_success and sealed_failed are terminal. sealed_failed requires a new run_id.

### 9.2 Lease and fencing

Resume uses lease TTL and heartbeat expiry, not PID existence. Host/PID are diagnostic. partial_failed -> running is an atomic CAS that increments a monotonic fencing token. Every event, cell publication, aggregate, and artifact binding includes the token. Stale supervisors cannot publish. Concurrent resume losers exit resume_lease_conflict.

### 9.3 Cell states

~~~text
queued
in_flight
accepted
failed
orphaned
~~~

Only accepted cells are reusable. Failed and orphaned cells are rerunnable. Expired attempts convert unresolved in-flight cells to orphaned. Acceptance occurs only after atomic directory publication of all required artifacts, schemas, hashes, and reports.

### 9.4 Actor and crash-consistent logging

Every authenticated transition logs subject/type, auth context ID without secrets, hostname, OS user, PID, command digest, attempt ID, fencing token, timestamp, reason, and before/after states.

Write order:

1. write and fsync accepted cell content;
2. temp-write, fsync, rename, and directory-fsync attempt_result.json;
3. publish one immutable hash-chained event file;
4. atomically refresh rebuildable state.json;
5. release lease last.

Events are authoritative. state.json is rebuildable. SEALED_SUCCESS is written last.

## 10. Artifact schemas and deterministic serialization

### 10.1 Registered exact schemas

Register at least:

~~~text
WorkerPredictionTraceSchemaV1
EvaluatedPredictionTraceSchemaV1
SourceSelectionTraceSchemaV1
FormalResultRowSchemaV1
WorkerManifestSchemaV1
CellResultManifestSchemaV1
RunManifestSchemaV1
PreflightReportSchemaV1
~~~

Descriptors freeze field name/order, Arrow dtype, nullability, enums, primary key, physical sort key, semantic columns, semantic sort key, serialization policy, and additionalProperties=false.

Readers use the full tuple (schema_name, schema_version, schema_digest). Unknown tuples fail. Same version with a different digest is SCHEMA_DEFINITION_DRIFT. Known V1/V2 readers may coexist, but one run cannot mix versions for one artifact type. There is no silent upcast. Migration creates an append-only derived artifact and cannot resume an old run under a new schema.

### 10.2 Trace fields

WorkerPredictionTraceSchemaV1 freezes this exact field order:

~~~text
run_id
cell_id
attempt_id
dataset_id
scenario
target_entity_key
method
seed
rollout_stream_key
forecast_origin
label_date
horizon
truth_key
sample_key
prediction_row_key
y_pred_raw
y_pred_clipped
was_clipped
history_snapshot_digest
history_after_h1_commit_digest
input_digest
prediction_policy_id
predictor_feature_schema_digest
feature_mask_digest
~~~

All worker fields are non-null. target_entity_key has one schema-declared canonical representation. Worker traces do not contain y_true, is_synthetic_date, evaluator paths, or evaluator capability IDs.

WorkerPredictionTraceSchemaV1 uses these exact Arrow types:

- UTF-8 string: run_id, cell_id, attempt_id, dataset_id, scenario, target_entity_key, method, prediction_policy_id;
- date32: forecast_origin, label_date;
- int32: seed;
- int8: horizon;
- float64: y_pred_raw, y_pred_clipped;
- bool: was_clipped; and
- lowercase 64-hex UTF-8: rollout_stream_key, truth_key, sample_key, prediction_row_key, history_snapshot_digest, history_after_h1_commit_digest, input_digest, predictor_feature_schema_digest, feature_mask_digest.

dataset_id is the closed enum D1–D6. scenario and method are closed enums declared by the protocol descriptor. Invalid enum strings, noncanonical entity keys, nonfinite predictions, and malformed digests fail schema validation.

EvaluatedPredictionTraceSchemaV1 contains the exact worker fields in the same order, followed by:

~~~text
y_true
is_synthetic_date
evaluator_join_status
~~~

All evaluated additions are non-null. y_true is float64, is_synthetic_date is bool, and evaluator_join_status is a closed UTF-8 enum. y_true is finite and nonnegative after sealed-data validation. evaluator_join_status must equal matched in an accepted trace.

Before any formal run, FormalResultRowSchemaV1 must freeze explicit field names, order, Arrow types, nullability, and enums in its repository descriptor. It must include result identity (dataset, scenario, target, method, seed, horizon), sample count and date bounds, RMSE/sMAPE/accuracy, clipping count, accepted trace paths and hashes, semantic prediction digest, source-selection identity, predictor schema and mask digests, protocol/input/code identities, status, and a closed failure code. A failed result row has null metric fields and cannot be accepted.

### 10.3 Frozen semantic sorting

Prediction semantic_sort_key is mandatory and frozen:

~~~text
dataset_id
scenario_enum_order
target_entity_key_canonical
method_enum_order
seed
forecast_origin
horizon
label_date
~~~

Semantic hashing validates schema, projects frozen semantic columns, sorts stably, rejects duplicate/null keys, canonically encodes, and hashes. Callers cannot override the key. A change requires a new schema version.

### 10.4 Canonical files

CSV uses exact column order, stable sorting, UTF-8 without BOM, LF newlines, ISO dates, lowercase booleans, the literal two-character null token backslash followed by N, finite float64 with frozen 17-significant-digit formatting, normalized negative zero, and no locale formatting.

Gzip uses fixed compression policy, mtime=0, and empty filename. JSON uses sorted keys, compact separators, UTF-8, ensure_ascii=false, allow_nan=false, one trailing newline, and UTC Z timestamps. Paths are POSIX paths relative to run root.

Publication is temp-write, fsync, atomic rename, and directory fsync.

### 10.5 Digest layers

~~~yaml
schema_digest: "sha256:..."
canonical_content_sha256: "sha256:..."
artifact_sha256: "sha256:..."
semantic_prediction_digest: "sha256:..."
~~~

Schema digest identifies interpretation. Canonical digest identifies normalized logical content. Artifact digest identifies physical bytes. Semantic prediction digest excludes run/attempt ownership.

Schema descriptors are immutable versioned files in the repository and copied to run_root/schemas/<schema_digest>.json. Unit tests pin literal V1 digests.

## 11. Artifact reuse and rehydration

### 11.1 Logical versus physical identity

~~~text
schema match + canonical match + artifact match => direct reuse
schema match + canonical match + artifact differs => no model recompute; explicit rehydrate
schema match + canonical differs => recompute cell
schema differs => reject
~~~

Missing and mismatched bytes differ:

- ARTIFACT_MISSING may rehydrate automatically from a registered trusted content-addressed replica, with a new attempt and event;
- missing without a trusted replica requires cell recomputation;
- ARTIFACT_BYTES_MISMATCH is corruption/tamper suspicion and requires authenticated rehydrate from a different trusted candidate;
- a mismatched file cannot sign itself.

Rehydrate applies only before sealing. For sealed runs, restore exact backup bytes or create a new run ID.

### 11.2 Atomic authority rebinding

Every artifact has stable logical_artifact_id and one active physical reference per attempt. Resume completes validation/rehydration before downstream scheduling, atomically publishes artifact_binding_set.json by CAS with the current fencing token, freezes its digest, and then starts downstream work.

All downstream resolvers use the frozen binding set. They cannot read paths embedded in old manifests. Old artifacts remain audit-only. If rehydration becomes necessary after work begins, the attempt becomes partial_failed and a new attempt performs the rebind.

Rehydrate must keep fit and predict call counts at zero.

## 12. Metrics and result authority

Predictions are inverse-transformed and clipped to nonnegative original sales space before feedback and metrics.

Formal sMAPE:

~~~text
100 * mean(2 * abs(pred - true) / (abs(true) + abs(pred) + 1e-8))
~~~

EvaluatedPredictionTraceSchemaV1 is metric authority. Result CSV is a derived summary. A successful row must match sample count, dates, keys, clipping count, trace hashes, and recomputed metrics. Failed rows contain no partial metric values.

## 13. Final preflight and sealed acceptance

Preflight validates:

- six sealed dataset identities and validation-policy digests;
- D1/D2 raw-rebuild and D3–D6 adopted provenance;
- exact target/source windows and source-sales canonicalization reports;
- exact predictor and KNN schemas;
- future-known lineage;
- D2 KNN fingerprint;
- separated worker/evaluator caches;
- registered artifact schemas and frozen digests;
- 12 modes, 60 seed bundles, result keys, D5 scheduling, and thread budget;
- code/input/protocol/cache identities;
- output ownership and append-only recovery layout.

Global sealing requires 12 accepted modes, 60 accepted bundles, all formal result keys, accepted worker/evaluated traces, source-selection proofs, no unresolved failures, and matching identities.

## 14. Required tests

The plan must test:

- predictor schema equality across methods/scenarios/horizons/seeds;
- RFE mask-only behavior and forbidden observed-dynamic/ID fields;
- future-known cutoff and sales-derived leakage;
- source sales repair and unused non-sales null preservation;
- four-view separation and truth tripwire non-swallowing;
- worker trace rejection of y_true;
- independent histories and h1-only hash chains;
- exact sample counts and deterministic keys;
- lease TTL, fencing, CAS conflict, orphan classification, and crash recovery;
- closed adopt enums and preflight blocker display;
- schema digest freezing, exact readers, and same-run V1/V2 rejection;
- semantic digest invariance to row order;
- missing artifact versus byte mismatch;
- rehydrate without fit/predict and downstream binding switch;
- tamper rejection and SEALED_SUCCESS written last.

## 15. Non-goals

- This version does not rebuild D3–D6 from raw data.
- It does not defend against a malicious same-UID process.
- It does not silently migrate schemas.
- It does not allow entity-level mixed provenance.
- It does not allow operator waivers to override validation failure enums.
- It does not run full D1–D6 experiments during implementation or verification.
