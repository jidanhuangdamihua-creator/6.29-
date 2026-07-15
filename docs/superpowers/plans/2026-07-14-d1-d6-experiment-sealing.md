# D1–D6 Experiment Sealing Implementation Plan

> **Normative design:** docs/superpowers/specs/2026-07-15-d1-d6-experiment-sealing-design.md
>
> Implement this plan task by task. If any existing code, fixture, compatibility wrapper, or example conflicts with the normative design, treat the conflict as a defect. Do not choose the older behavior.

**Goal:** Build one leak-free, reproducible D1–D6 formal entry with fixed 30-day target observation, one 180-day source window whose final 30 days are used by KNN, 180-day blind joint-horizon rollout, exact predictor schemas, separated worker/evaluator truth, append-only recovery, and acceptance-gated sealed outputs.

**Architecture:** Publish immutable versioned D1–D6 data first; build exact worker/evaluator views and caches; validate complete 180-day sources before KNN; fit all six methods through one predictor contract; roll out h1–h5 from independent per-method/seed histories; publish typed traces; and let an append-only supervisor seal only complete trace-verified results.

**Tech stack:** Python 3.9+, pandas, NumPy, PyArrow, TensorFlow/Keras through existing model code, pytest/unittest, Bash supervisor, SHA-256, canonical JSON, and canonical gzip CSV.

## Execution policy

- Do not run a full D1–D6 experiment during implementation.
- Every Python test, validation, builder, dataset, or model command in this plan must run through:

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- <command...>
~~~

- If the wrapper exits 124, stop the entire implementation session. Do not retry, split, simplify, resume, or continue. Return the exact timed-out command for manual execution.
- Static searches, git diff checks, shell syntax checks, and small import/compile checks may run directly.
- Preserve unrelated user changes. Review each task diff before committing it.

## Frozen formal contract

- Target: 15 train days + 15 validation days + 180 blind natural days.
- Source: exactly 180 natural days ending at target observed end.
- KNN: the final 30 days inside that same source window.
- Horizons: 1–5. Seeds: 42–46.
- Sample counts: h1=180, h2=179, h3=178, h4=177, h5=176.
- Six methods share one exact PredictorFeatureSchema per dataset.
- Only clipped h1 feeds back, independently for each dataset/target/scenario/method/seed.
- D5 scheduling begins with d5_without; d5_with starts immediately after its success and never overlaps it.
- D5 uses 6 compute threads, ordinary workers 2, with total active budget at most 16.
- Sealed success is written last and only after full trace verification.

## Planned files and interfaces

- src/protocols/sealing_protocol.py: windows, protocol identity, feature roles.
- src/protocols/feature_schema.py: exact predictor/KNN schemas, masks, lineage audit.
- src/protocols/artifact_schemas.py: exact artifact descriptors, reader registry, digest freezing.
- src/data_processing/sealed_daily.py: shared calendarization, canonicalization, four views.
- src/utils/sealed_parquet.py: validated projection and date pushdown.
- src/utils/mode_cache.py: separate worker/evaluator cache contracts.
- src/protocols/candidate_pool.py: exact 30-day KNN over eligible 180-day sources.
- src/protocols/provenance.py: source slices and tensor provenance.
- src/protocols/blind_rollout.py: truth-free joint-horizon rollout.
- src/experiment/fitted_predictor.py: common fitted predictor adapters.
- src/utils/prediction_artifacts.py: canonical typed artifact publication.
- src/utils/run_recovery.py: attempts, leases, fencing, events, cell states.
- src/utils/artifact_rehydration.py: trusted replica validation and atomic rebinding.
- scripts/run_unified_d1_d6.py: 60-cell plan, preflight, aggregation, sealing.
- scripts/parallel_mode_runner.sh: D5 dependency lane and bounded scheduler.

---

### Task 1: Freeze protocol windows, feature schemas, roles, and masks

**Files**

- Create: src/protocols/sealing_protocol.py
- Create: src/protocols/feature_schema.py
- Modify: src/protocols/__init__.py
- Modify: src/constants.py
- Create: tests/test_sealing_protocol.py
- Create: tests/test_predictor_feature_schema.py
- Create: tests/test_future_known_lineage.py

**Required interfaces**

- TargetWindow and SourcePretrainWindow.
- PredictorFeatureSchema with ordered names, dtypes, roles, transforms, dimension, and digest.
- FeatureRole enum: target_signal, future_known, static_known, observed_dynamic, recursive_derived, evaluation_only, identifier_group_only.
- KnnObservedDispositionV1 enum: knn_observed, audit_only.
- PredictorFeatureMask with the same length as the full schema.
- get_target_window, get_source_pretrain_window, get_predictor_schema, get_knn_schema, audit_future_known_lineage.

**Steps**

- [ ] Write failing tests for exact 30+180 target windows and exact 180-day source windows.
- [ ] Freeze one predictor schema per dataset and assert exact equality across all six methods, both scenarios, source/target, train/validation/blind, horizons, and seeds.
- [ ] Freeze KNN schemas separately from predictor schemas.
- [ ] Exclude D2 promo; D3 Customers/Open/Promo; D4 stock/activity/discount/weather; D5 onpromotion/transactions/oil_price; and D6 sell_price from formal predictors.
- [ ] Classify every such observed-only dynamic before sealing as knn_observed or audit_only; forbid runtime inference and unclassified fallback.
- [ ] Freeze D2 KNN order as sales then promo. Freeze the ordered D3–D6 KNN classifications in their dataset schemas; keep D4 leakage-risk fields audit_only.
- [ ] Exclude identifiers/group/category codes from predictors; permit only the approved D5 perishable static exception.
- [ ] Implement RFE as a full-schema boolean mask. Zero unselected transformed fields; never delete or reorder columns; always retain sales.
- [ ] Audit every future-known dependency and fail if it depends on sales, truth, prediction, or a date after the cutoff.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_sealing_protocol.py   tests/test_predictor_feature_schema.py   tests/test_future_known_lineage.py   tests/test_experiment_protocol_contract.py -q
~~~

**Acceptance**

- All methods receive identical ordered predictor shapes within a dataset.
- Promo/IDs cannot enter formal predictor tensors.
- RFE changes only the mask digest.
- Future-known fields have explicit authority and cutoff lineage.

---

### Task 2: Build the versioned hybrid sealed data root

**Files**

- Modify: scripts/regenerate_d1_d2_parquets.py
- Create: scripts/adopt_and_seal_d3_d6.py
- Create: src/data_processing/sealed_daily.py
- Create: src/protocols/adopt_validation.py
- Modify: scripts/run_full_paper_experiments.py
- Modify: scripts/run_unified_d1_d6.py
- Create: tests/test_hybrid_sealed_builder.py
- Create: tests/test_adopt_validation_contract.py
- Modify: tests/test_d1_d2_formal_input_paths.py
- Modify: tests/test_d1_d2_sealed_builder.py

**Required output**

~~~text
数据集/固化数据/d1_d6_sealed_v1/
  dataset1/
  dataset2/
  dataset3/
  dataset4/
  dataset5/
  dataset6/
~~~

Each dataset directory contains source/target artifacts, manifest, validation report, schemas, and audit sidecars. Formal path resolvers use only this root.

**Steps**

- [ ] Rebuild D1/D2 from raw inputs with raw_rebuilt provenance.
- [ ] Preserve the approved D2 June calendarization and Item 4/6/8 KNN fingerprint.
- [ ] Adopt current D3–D6 parquets with adopted_solidified provenance.
- [ ] Record parent SHA-256, size, observation time, mtime metadata, nullable first-seen evidence, and structural-only content validation disclosure.
- [ ] Route all D1–D6 calendarization through the same versioned engine and dataset configuration format.
- [ ] Implement the closed AdoptValidationFailureReasonV1 enum from the normative design. Unknown exceptions become VALIDATOR_INTERNAL_ERROR.
- [ ] Pin validation policy/version/code digests.
- [ ] Publish at dataset granularity only. One required entity failure prevents that dataset directory from becoming sealed.
- [ ] Forbid entity-level mixed provenance and automatic fallback. Whole-dataset raw rebuild is an explicit operator action.
- [ ] Make dataset directory publication temp-write, fsync, atomic rename, and directory fsync.

**Source sales rule**

For each source 180-day frame, before KNN/scaling:

- NaN -> 0;
- finite negative -> 0;
- calendar-missing row -> canonical sales 0;
- positive/negative infinity -> failure;
- unused non-sales nulls remain unchanged;
- used non-sales fields require an approved rule or fail.

Record repair reason, counts, dates digest, and repair-mask digest.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_hybrid_sealed_builder.py   tests/test_adopt_validation_contract.py   tests/test_d1_d2_formal_input_paths.py   tests/test_d1_d2_sealed_builder.py   tests/test_full_paper_runner_solidified_parquet.py -q
~~~

**Local artifact generation**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python   scripts/regenerate_d1_d2_parquets.py   --dataset all   --output-dir 数据集/固化数据/d1_d6_sealed_v1
~~~

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python   scripts/adopt_and_seal_d3_d6.py   --output-dir 数据集/固化数据/d1_d6_sealed_v1
~~~

If either command exits 124, stop and return that exact command.

---

### Task 3: Create four typed target views and separated caches

**Files**

- Modify: src/data_processing/sealed_daily.py
- Create: src/utils/sealed_parquet.py
- Create: src/utils/mode_cache.py
- Create: src/utils/truth_isolation.py
- Modify: src/utils/parquet_data_loader.py
- Modify: src/utils/entity_experiment.py
- Create: tests/test_target_view_contract.py
- Create: tests/test_truth_isolation.py
- Create: tests/test_mode_cache_contract.py
- Create: tests/test_worker_manifest_truth_paths.py

**Required views**

- knn_observed_frame: exact 30 observed days and KNN schema.
- observed_model_frame: exact 30 observed days and predictor schema, including observed sales.
- blind_covariate_frame: exact 180 days with future-known/static fields and safe keys only.
- evaluator_truth_frame: exact 180 days with key/date/y_true/is_synthetic_date/truth_key.

**Steps**

- [ ] Define exact Arrow/runtime schemas for all four views.
- [ ] Reject blind sales, observed-only dynamics, truth references, and unknown fields in blind_covariate_frame.
- [ ] Build physically separate worker and evaluator caches.
- [ ] Give workers only WorkerRunLayout and worker cache.
- [ ] Address evaluator bundles with a high-entropy capability ID mapped only in evaluator control plane.
- [ ] Give worker manifests a fixed schema with no truth/evaluator/capability/path-template fields or reconstructing parameters.
- [ ] Add TruthAccessTripwire that records each access before raising.
- [ ] Assert fitting succeeds with attempted_access_count=0 and evaluator_loader_call_count=0; caught exceptions still fail.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_target_view_contract.py   tests/test_truth_isolation.py   tests/test_mode_cache_contract.py   tests/test_worker_manifest_truth_paths.py   tests/test_sealed_parquet_pushdown.py -q
~~~

---

### Task 4: Validate complete sources before 30-day KNN

**Files**

- Modify: src/protocols/candidate_pool.py
- Modify: src/protocols/provenance.py
- Modify: src/source_selection/source_selector.py
- Modify: src/protocols/runner_adapter.py
- Modify: src/utils/parquet_data_loader.py
- Create: tests/test_source_pretrain_180d.py
- Modify: tests/test_daily_knn_protocol.py
- Modify: tests/test_knn_cnn_provenance.py
- Modify: tests/test_d2_paper_knn_fingerprint.py
- Create: tests/test_source_sales_canonicalization.py

**Steps**

- [ ] Calendarize and canonicalize all 180 source dates before eligibility.
- [ ] Validate only used predictor/KNN fields; preserve unused non-sales nulls.
- [ ] Exclude sources with infinity or unresolved required feature values.
- [ ] Rank only eligible sources using the final 30 days.
- [ ] Flatten dates ascending and KNN fields in frozen schema order as raw canonical float64.
- [ ] Include KNN schema, vector shape, date range, source repair digest, and vector digest in selection identity.
- [ ] Use the full 180 days for fixed-epoch source fitting. Do not retain the old 300-day constant or source test split.
- [ ] Prove every source training tensor and label against the canonical source slice.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_source_pretrain_180d.py   tests/test_source_sales_canonicalization.py   tests/test_daily_knn_protocol.py   tests/test_knn_cnn_provenance.py   tests/test_d2_paper_knn_fingerprint.py   tests/test_source_selector_window_leakage.py -q
~~~

**Acceptance**

- Mutating the first 150 days changes training digest but not KNN.
- Mutating the final 30 observed days may change ranking.
- No selected source can later fail because an unvalidated first-150-day value was ignored.

---

### Task 5: Implement truth-free joint-horizon rollout

**Files**

- Create: src/protocols/blind_rollout.py
- Modify: src/protocols/rolling_origin.py
- Modify: src/evaluation/metrics.py
- Modify: src/evaluation/metric_contract.py
- Create: tests/test_blind_rollout_protocol.py
- Modify: tests/test_rolling_origin_protocol.py
- Modify: tests/test_smape_metric_contract.py

**Required interface**

run_blind_rollout receives fitted predictors, observed_model_frame, blind_covariate_frame, schema identities, and rollout identity. It does not accept evaluator truth.

**Steps**

- [ ] Define rollout_stream_key by protocol/dataset/target/scenario/method/seed.
- [ ] Create a separate mutable history for every stream.
- [ ] At each origin, freeze one snapshot for all valid horizons.
- [ ] Inverse-transform and finite-check all predictions, then clip nonnegative.
- [ ] Commit clipped h1 only after all valid horizons succeed.
- [ ] Never commit h2–h5 or expose same-origin h1 to another horizon.
- [ ] Recompute recursive_derived fields only from observed sales and prior committed h1.
- [ ] Generate truth_key, sample_key, and prediction_row_key using the exact normative formulas.
- [ ] Record a continuous history hash chain.
- [ ] Emit WorkerPredictionTraceSchemaV1 with no y_true.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_blind_rollout_protocol.py   tests/test_rolling_origin_protocol.py   tests/test_smape_metric_contract.py   tests/test_metric_protocol_and_diagnostics.py   tests/test_multi_source_smape_metrics.py -q
~~~

**Acceptance**

- Exact h1–h5 sample counts.
- All horizons at one origin share history_snapshot_digest.
- Histories never cross seed/scenario/method.
- Truth mutation cannot change predictions or input digests.

---

### Task 6: Expose fitted predictors without evaluator access

**Files**

- Create: src/experiment/fitted_predictor.py
- Modify: src/experiment/run_no_tl_experiment.py
- Modify: src/experiment/experiment_runner.py
- Modify: src/transfer_methods/mswa_tl.py
- Modify: src/transfer_methods/mssb_tl.py
- Modify: src/transfer_methods/msml_tl.py
- Modify: src/transfer_methods/msml_tl_rfe.py
- Modify: src/utils/entity_experiment.py
- Create: tests/test_fitted_predictor_adapters.py
- Create: tests/test_joint_horizon_method_bundle.py

**Steps**

- [ ] Define KerasPredictor, WeightedPredictor, SwitchingPredictor, and FittedMethodHorizon.
- [ ] Fit h1–h5 once per method/seed bundle using target train/validation only.
- [ ] Make No-TL and SS-TL return one fitted predictor per horizon.
- [ ] Make MSWA-TL combine successful source-specific target models using frozen weights.
- [ ] Make MSSB-TL select only by target validation RMSE.
- [ ] Make MSML-TL and RFE expose fused predictors under the same full schema.
- [ ] Remove evaluator truth parameters from every formal fitting signature.
- [ ] Run all six methods against TruthAccessTripwire and evaluator-loader spies.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_fitted_predictor_adapters.py   tests/test_joint_horizon_method_bundle.py   tests/test_truth_isolation.py   tests/test_notl_minimal_fix.py   tests/test_single_source_tl.py   tests/test_mswa_tl.py   tests/test_mssb_tl.py   tests/test_msml_tl.py   tests/test_msml_tl_rfe.py -q
~~~

---

### Task 7: Implement typed artifact schemas and deterministic publication

**Files**

- Create: src/protocols/artifact_schemas.py
- Create: src/protocols/schemas/worker_prediction_trace_v1.json
- Create: src/protocols/schemas/evaluated_prediction_trace_v1.json
- Create: src/protocols/schemas/source_selection_trace_v1.json
- Create: src/protocols/schemas/formal_result_row_v1.json
- Create: src/protocols/schemas/worker_manifest_v1.json
- Create: src/protocols/schemas/cell_result_manifest_v1.json
- Create: src/protocols/schemas/run_manifest_v1.json
- Create: src/protocols/schemas/preflight_report_v1.json
- Create: src/utils/prediction_artifacts.py
- Modify: src/utils/result_schema.py
- Create: tests/test_artifact_schema_contract.py
- Create: tests/test_prediction_artifacts.py
- Modify: tests/test_result_schema_golden_diff.py

**Steps**

- [ ] Freeze exact fields, order, dtype, nullability, enums, primary key, physical sort key, semantic columns, semantic sort key, and serialization policy.
- [ ] Register readers by exact name/version/digest tuple.
- [ ] Reject unknown schemas, same-version digest drift, extra fields, reordered fields, and V1/V2 mixing in one run.
- [ ] Freeze prediction semantic_sort_key exactly as specified in the design.
- [ ] Canonically serialize JSON and gzip CSV.
- [ ] Compute schema, canonical content, physical artifact, and semantic prediction digests separately.
- [ ] Copy exact schema descriptors into each run under schemas/<digest>.json.
- [ ] Freeze literal V1 digests in tests; field changes require new version files.
- [ ] Let evaluator validate worker trace, join truth one-to-one, and publish EvaluatedPredictionTraceSchemaV1.
- [ ] Derive result rows only from evaluated traces; failed rows contain no partial metrics.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_artifact_schema_contract.py   tests/test_prediction_artifacts.py   tests/test_result_schema_golden_diff.py   tests/test_unified_d1_d6_output_contract.py   tests/test_strict_result_contract.py -q
~~~

**Required tests**

- Random physical row permutations produce the same semantic digest.
- Worker trace rejects y_true at schema validation.
- Same run rejects mixed schema versions.
- Recreated metrics match evaluated trace exactly.

---

### Task 8: Implement append-only attempts, leases, fencing, and cell recovery

**Files**

- Create: src/utils/run_recovery.py
- Modify: src/utils/run_layout.py
- Modify: src/utils/run_artifacts.py
- Modify: scripts/run_unified_d1_d6.py
- Create: tests/test_run_recovery_state_machine.py
- Modify: tests/test_result_state_machine.py
- Modify: tests/test_run_layout_and_atomic_publication.py

**Steps**

- [ ] Create append-only attempts/<attempt_id> manifests, events, logs, and attempt results.
- [ ] Implement allowed run transitions only: running->partial_failed, running->complete_unsealed, partial_failed->running, complete_unsealed->sealed_success/sealed_failed.
- [ ] Make sealed states terminal.
- [ ] Use heartbeat TTL for lease expiry; PID/hostname are diagnostic only.
- [ ] Make resume transition an atomic CAS that increments fencing token.
- [ ] Include fencing token in state events, cell publication, aggregation, and bindings.
- [ ] Implement queued/in_flight/accepted/failed/orphaned cell states.
- [ ] Convert unresolved in-flight cells from expired attempts to orphaned.
- [ ] Reuse only accepted cells with exact run-plan, code/input/protocol/cache/schema/content identities.
- [ ] Publish cells by atomic directory rename after all artifacts validate.
- [ ] Record authenticated actor identity and crash-consistent hash-chained events.
- [ ] Write SEALED_SUCCESS last.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_run_recovery_state_machine.py   tests/test_result_state_machine.py   tests/test_run_layout_and_atomic_publication.py   tests/test_unified_parallel_lifecycle.py -q
~~~

---

### Task 9: Add artifact rehydration and atomic authority binding

**Files**

- Create: src/utils/artifact_rehydration.py
- Modify: src/utils/run_recovery.py
- Modify: src/utils/run_layout.py
- Modify: src/utils/result_acceptance.py
- Create: tests/test_artifact_rehydration.py

**Steps**

- [ ] Give every artifact a stable logical_artifact_id.
- [ ] Distinguish ARTIFACT_MISSING from ARTIFACT_BYTES_MISMATCH.
- [ ] Auto-rehydrate missing artifacts only from registered trusted content-addressed replicas.
- [ ] Require authenticated explicit rehydrate from a different trusted candidate after byte mismatch.
- [ ] Never let a mismatched file sign itself.
- [ ] Preserve schema and canonical digest; publish a new physical SHA under a new attempt.
- [ ] Keep fit_call_count and predict_call_count at zero.
- [ ] Atomically publish and freeze artifact_binding_set.json before downstream scheduling.
- [ ] Make downstream resolvers use only the frozen binding set.
- [ ] Keep old manifests and bytes audit-only.
- [ ] Disallow rehydrate after sealed terminal states.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_artifact_rehydration.py   tests/test_run_recovery_state_machine.py   tests/test_run_layout_and_atomic_publication.py -q
~~~

**Required tests**

- Missing trusted replica rehydrates without model calls.
- Byte mismatch does not auto-rehydrate.
- Rehydrate preserves schema/canonical/semantic digests and changes only physical SHA.
- All downstream references switch to the new attempt binding.
- Old manifest remains byte-for-byte unchanged.

---

### Task 10: Replace 300 horizon cells with 60 seed bundles

**Files**

- Modify: scripts/run_strict_protocol_baseline.py
- Modify: scripts/run_unified_d1_d6.py
- Modify: scripts/run_full_paper_experiments.py
- Modify: scripts/run_d4_experiment.py
- Modify: scripts/run_d5_experiment.py
- Modify: scripts/run_d6_experiment.py
- Modify: src/utils/run_layout.py
- Modify: src/utils/result_acceptance.py
- Modify: src/utils/result_validation.py
- Modify: tests/test_formal_protocol_matrix.py
- Modify: tests/test_unified_parallel_lifecycle.py
- Modify: tests/test_result_acceptance_scopes.py

**Steps**

- [ ] Replace MatrixTask(horizon, seed) with SeedBundleTask(seed, horizons=(1,2,3,4,5)).
- [ ] Use five cells per dataset/mode and 60 cells globally.
- [ ] Use cells/s<seed> directories.
- [ ] Make every dataset runner accept --all-horizons --seed and reject simultaneous --horizon.
- [ ] Pin mode cache, schema registry, feature schema, source-repair, and expected trace identities in each plan cell.
- [ ] Reject old 300-cell run plans during resume.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_formal_protocol_matrix.py   tests/test_unified_parallel_lifecycle.py   tests/test_run_layout_and_atomic_publication.py   tests/test_result_acceptance_scopes.py -q
~~~

---

### Task 11: Implement D5-first bounded failure-preserving scheduling

**Files**

- Modify: scripts/parallel_mode_runner.sh
- Modify: scripts/run_unified_d1_d6.py
- Modify: src/utils/run_artifacts.py
- Modify: tests/fixtures/fake_formal_worker.py
- Modify: tests/test_parallel_mode_supervisor.py

**Steps**

- [ ] Queue d5_without first.
- [ ] Keep d5_with dependency-blocked until d5_without succeeds.
- [ ] Start d5_with immediately after that success and before scanning ordinary queued work.
- [ ] Never overlap D5 modes.
- [ ] Apply 6 threads to D5, 2 to ordinary workers, and refuse launches above total 16.
- [ ] On task failure, stop new scheduling, preserve accepted and in-flight atomic completions, mark dependent tasks blocked, and write partial_failed.
- [ ] Signals terminate process groups; ordinary task failure does not delete accepted artifacts.
- [ ] Route every status write through the fenced run-recovery API.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_parallel_mode_supervisor.py   tests/test_unified_parallel_lifecycle.py -q
~~~

---

### Task 12: Implement layered preflight and full sealed acceptance

**Files**

- Modify: scripts/run_unified_d1_d6.py
- Modify: scripts/validate_d1_d6_protocol_inputs.py
- Modify: src/utils/result_acceptance.py
- Modify: src/utils/result_validation.py
- Create: tests/test_formal_entry_preflight.py
- Create: tests/test_sealed_run_acceptance.py

**Steps**

- [ ] Validate six dataset seal proofs and one required validation-policy digest before creating an attempt.
- [ ] Report each dataset seal state separately from global preflight state.
- [ ] Block with dataset/failure codes/report path/report SHA when any dataset fails.
- [ ] Do not start supervisor or create a formal attempt while blocked.
- [ ] Validate exact target/source windows, source repair reports, predictor/KNN schemas, future-known lineage, D2 fingerprint, cache isolation, artifact schema registry, 12 modes, 60 cells, result keys, D5 dependency, and thread budget.
- [ ] Recompute every metric from accepted evaluated traces.
- [ ] Require exact horizon counts/dates/keys, clipping counts, source-selection proofs, and all code/input/protocol/cache/schema/content identities.
- [ ] Treat results/experiment_results.csv as a derived authoritative aggregate only after trace validation.
- [ ] Keep any d1_d6_results.csv export non-authoritative.
- [ ] Publish sealed_failed or SEALED_SUCCESS only from complete_unsealed; sealed states are terminal.

**Verification**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_formal_entry_preflight.py   tests/test_sealed_run_acceptance.py   tests/test_result_state_machine.py   tests/test_result_acceptance_scopes.py   tests/test_artifact_schema_contract.py   tests/test_artifact_rehydration.py -q
~~~

---

### Task 13: Wire the formal entry, documentation, and lightweight verification

**Files**

- Modify: scripts/run_unified_d1_d6.py
- Modify: scripts/parallel_mode_runner.sh
- Modify: README.md
- Modify: docs/superpowers/plans/2026-07-14-d1-d6-experiment-sealing.md as implementation discoveries require
- Modify: docs/superpowers/specs/2026-07-15-d1-d6-experiment-sealing-design.md only through reviewed design-version changes

**Formal entry**

~~~bash
cd /path/to/保留的复现实验修改rfe
DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh
MAX_JOBS=6 RUN_ROOT=outputs/runs/<new_run_id> bash scripts/parallel_mode_runner.sh
MAX_JOBS=6 RUN_ROOT=outputs/runs/<existing_run_id> RESUME=1 bash scripts/parallel_mode_runner.sh
~~~

Codex does not run the second or third command.

**Steps**

- [ ] Make dry-run print resolved dataset identities, provenance, dates, exact feature/KNN schemas, source repair counts, cache identities, schema digests, 60 cells, thread budget, and D5 dependency without creating run output.
- [ ] Document append-only attempts, resume lease conflict, rehydrate behavior, terminal sealed states, and manual server commands.
- [ ] Remove or label every non-formal compatibility fallback as non-sealed.
- [ ] Run the complete lightweight contract suite once.

**Complete lightweight suite**

~~~bash
python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest   tests/test_sealing_protocol.py   tests/test_predictor_feature_schema.py   tests/test_future_known_lineage.py   tests/test_hybrid_sealed_builder.py   tests/test_adopt_validation_contract.py   tests/test_target_view_contract.py   tests/test_truth_isolation.py   tests/test_mode_cache_contract.py   tests/test_source_pretrain_180d.py   tests/test_source_sales_canonicalization.py   tests/test_daily_knn_protocol.py   tests/test_d2_paper_knn_fingerprint.py   tests/test_blind_rollout_protocol.py   tests/test_fitted_predictor_adapters.py   tests/test_joint_horizon_method_bundle.py   tests/test_smape_metric_contract.py   tests/test_artifact_schema_contract.py   tests/test_prediction_artifacts.py   tests/test_run_recovery_state_machine.py   tests/test_artifact_rehydration.py   tests/test_formal_protocol_matrix.py   tests/test_parallel_mode_supervisor.py   tests/test_sealed_run_acceptance.py   tests/test_formal_entry_preflight.py -q
~~~

If this exits 124, stop and return the exact command. Do not retry a subset.

**Static checks**

~~~bash
python -m compileall -q src scripts tests
git diff --check
bash -n scripts/parallel_mode_runner.sh
~~~

## Task completion discipline

For each task:

1. implement only that task;
2. review only its diff;
3. run its exact verification;
4. confirm no formal training or large run output was created;
5. commit the task with an explicit message before moving to the next task.

Do not retain formal fallbacks, silently skip failed selected sources, fill arbitrary numeric columns, expose evaluator truth to workers, mix artifact schema versions, mutate accepted attempts, or continue after timeout.
