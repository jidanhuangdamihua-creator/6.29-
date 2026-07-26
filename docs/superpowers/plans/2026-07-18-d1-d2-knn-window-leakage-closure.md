# D1/D2 KNN Observed Window Leakage Closure Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with review checkpoints. Steps use checkbox syntax for tracking.

**Goal:** Make D1/D2 KNN selection use the authoritative inclusive 30-day windows ending at the origin, then regenerate and bind all four solidified KNN configs to real frame and selection digests.

**Architecture:** Keep full source/target model frames intact for training and evaluation, while configure_protocol_frames constructs exact observed KNN copies before selector work. The shared selector consumes those copies and emits metadata whose frame, candidate, selection, config, and formal authority identities are recomputable from real bytes.

**Tech Stack:** Python 3, pandas, NumPy, pytest, PyArrow Parquet, deterministic JSON/SHA-256 authority builders.

## Global Constraints

- D1 KNN window is inclusive 2017-06-01..2017-06-30, origin 2017-06-30, exactly 30 days.
- D2 KNN window is inclusive 2018-06-01..2018-06-30, origin 2018-06-30, exactly 30 days.
- Every KNN date must satisfy observed_start <= date <= observed_end and date <= origin.
- D3–D6 freeze rules and model/baseline code are outside scope.
- No selector-only final filter, manual JSON/hash edit, old selection reuse, or formal model training.
- Every Python command uses python tools/protection/codex_timeout.py --timeout 180 -- ...; exit code 124 stops the work.
- Use apply_patch for source, test, documentation, and configuration edits.

---

### Task 1: Freeze the single D1/D2 origin-based window authority

Files:
- Modify: src/protocols/experiment_protocol.py
- Modify: src/constants.py
- Modify: scripts/run_full_paper_experiments.py
- Modify: scripts/validate_d1_d6_protocol_inputs.py
- Test: tests/test_experiment_protocol_contract.py

Interfaces:
- Produce D1_D2_KNN_ORIGINS, STRICT_KNN_OBSERVED_DAYS, ObservationWindow.from_origin(origin, observed_days=30), and ExperimentProtocol.observation_window(observed_start=None).
- D1/D2 observation_window derives from the protocol origin and rejects a stale supplied start; D3–D6 continue deriving from their existing caller-provided start.

- [ ] Write tests first. Assert D1 maps to origin 2017-06-30, start 2017-06-01, end 2017-06-30, and 30 calendar dates; assert the analogous D2 values; assert a D1 supplied start 2017-06-05 raises ProtocolViolation; assert D4 still maps a supplied 2020-01-01 start to 2020-01-30.
- [ ] Run the RED test: python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_experiment_protocol_contract.py -q. It must fail because the old implementation requires a start and returns the offset window.
- [ ] Implement the authority. Add D1_D2_KNN_ORIGINS and STRICT_KNN_OBSERVED_DAYS in experiment_protocol.py. Add ObservationWindow.from_origin, origin property, and observed_days property. Make D1/D2 observation_window use from_origin and reject disagreement; preserve from_start for D3–D6. Remove unused stale D1_TARGET_TRAIN_WINDOW and D2_TARGET_TRAIN_WINDOW constants. Remove D1/D2 entries from run_full_paper_experiments.py STRICT_KNN_OBSERVED_START and derive D1/D2 starts from the protocol in validate_d1_d6_protocol_inputs.py.
- [ ] Run the focused tests and rg -n '2017-06-05|2018-06-05|STRICT_KNN_OBSERVED_START' src scripts tests configs/solidified/knn --glob '*.py' --glob '*.json'. Tests must pass and no D1/D2 runtime/config authority occurrence may remain.
- [ ] Commit: git add the five changed files; git commit -m "fix: derive D1 D2 KNN windows from origin".

### Task 2: Build explicit observed KNN frames before selector work

Files:
- Create: src/protocols/knn_frames.py
- Modify: src/protocols/runner_adapter.py
- Test: tests/test_d1_d2_knn_window_closure.py
- Test: tests/test_runner_protocol_integration.py

Interfaces:
- Produce build_observed_knn_frame(frame, window, role, group_cols) and get_configured_knn_frame(frame, role).
- configure_protocol_frames returns full model frames with attrs["protocol_knn_observed_frame"] holding a validated observed copy for each role.

- [ ] Add a new frame-level test with D1 and D2 source/target rows covering the legal interval plus origin+1 and origin+4 sentinels. Assert the configured KNN source and target frames have only the inclusive legal dates, exact target max date equal to origin, and the returned full target frame still retains future rows. Add invalid-date failure coverage.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_d1_d2_knn_window_closure.py -q. It must fail because the accessor does not exist.
- [ ] Implement knn_frames.py. Normalize and validate date, select only the inclusive interval, reject invalid/empty frames, preserve keys/data, and compute a deterministic sorted canonical frame digest from group columns, date, columns, dtypes, and values.
- [ ] Integrate in configure_protocol_frames. Resolve the protocol window before candidate discovery; build target observed frame immediately; after D2 verification and candidate-key filtering build source observed frame; use the pre-origin source copy for D1/D2 candidate discovery; store both observed copies and metadata with origin, observed_days, boundary, min/max, and digests. Keep full model frames intact.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_d1_d2_knn_window_closure.py tests/test_runner_protocol_integration.py -q. All selected tests must pass, including D4/D5 behavior.
- [ ] Commit: git add knn_frames.py, runner_adapter.py, and the two test files; git commit -m "fix: construct bounded KNN frames before selection".

### Task 3: Bind selector, candidate, and selection digests to real frames

Files:
- Modify: src/protocols/candidate_pool.py
- Modify: src/source_selection/source_selector.py
- Test: tests/test_daily_knn_protocol.py
- Test: tests/test_source_selector_window_leakage.py
- Test: tests/test_source_selector_shared_protocol.py
- Test: tests/test_candidate_pool_digest.py

Interfaces:
- build_candidate_pool_digest accepts source_frame_digest and target_frame_digest and includes them when supplied.
- select_daily_sequence_sources consumes only configured observed frames and places both frame digests in candidate_pool_digest_input.
- Shared selector metadata exposes selection_digest as the canonical selection_result_digest alias.

- [ ] Add tests that capture the actual frames passed from the shared selector into select_daily_sequence_sources and assert max(date) <= origin and exact min/max bounds for D1 and D2. Change only future sentinel values and assert source order, distances, weights, frame digests, candidate digest, and selection digest are unchanged. Change an observed value as a positive control and assert identity changes.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_daily_knn_protocol.py tests/test_source_selector_window_leakage.py tests/test_source_selector_shared_protocol.py tests/test_candidate_pool_digest.py -q. New assertions must fail because full frames are still passed and candidate digests lack frame identity.
- [ ] In SourceSelector._select_with_shared_protocol retrieve get_configured_knn_frame for source and target, verify bounds and shared metadata, and pass only those copies to select_daily_sequence_sources. Keep full frames for CNN provenance/model training.
- [ ] In candidate_pool.py compute canonical digests from actual inputs, include them in digest input, reject dates outside the protocol interval, and add origin, observed_days, boundary, frame bounds, frame digests, candidate digest, and selection digest to shared metadata. Keep D3–D6 runtime digest behavior compatible.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_daily_knn_protocol.py tests/test_source_selector_window_leakage.py tests/test_source_selector_shared_protocol.py tests/test_candidate_pool_digest.py tests/test_d2_calendarization_digest_chain.py -q. All must pass.
- [ ] Commit: git add candidate_pool.py, source_selector.py, and the five test files; git commit -m "fix: bind KNN selection identity to observed frames".

### Task 4: Extend formal regeneration to D1/D2

Files:
- Modify: scripts/regenerate_solidified_knn.py
- Modify: src/utils/parquet_data_loader.py
- Create: tests/test_regenerate_d1_d2_knn.py
- Modify: tests/test_regenerate_solidified_knn_check_only.py

Interfaces:
- regenerate_dataset_scenario supports dataset IDs 1–6; D1/D2 use shared protocol authority and D4–D6 retain runtime authority.
- D1/D2 payloads use protocol_version == PROTOCOL_VERSION, selection_authority == shared_protocol, and real metadata for both scenarios.

- [ ] Write a temporary Parquet fixture test with one target, required candidate keys, legal dates, and future sentinels. Assert generated D1 and D2 payloads contain origin, observed_start, observed_end, observed_days, boundary, actual source/target frame min/max, and non-empty frame/candidate/selection digests; assert old 06-05..07-04 identity is absent and check-only does not mutate solidified files.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_regenerate_d1_d2_knn.py tests/test_regenerate_solidified_knn_check_only.py -q. It must fail because the CLI defaults to D4–D6 and always writes runtime authority.
- [ ] Add a strict D1/D2 loader that reads resolver-selected sealed Parquet source/target files, applies the scenario domain policy, isolates the configured target key, and calls configure_protocol_frames with the protocol-owned window. Add a shared selection helper using D1 store_id/item_id or D2 brand_id/item_id and feature_cols=("sales",), and never copy old results or digests.
- [ ] Make _build_regenerated_payload select shared protocol authority/version for D1/D2 and runtime authority/version for D4–D6. For D1/D2 write the canonical observed-window object, real results, selection_metadata, source-pool size, and feature info from the selector. Add datasets 1 and 2 to explicit CLI coverage; preserve D4–D6 behavior.
- [ ] Run the D1/D2 check-only command: python tools/protection/codex_timeout.py --timeout 180 -- python scripts/regenerate_solidified_knn.py --datasets 1 2 --diff-out /tmp/d1-d2-knn-window-check. It must not mutate configs/solidified/knn.
- [ ] Commit regeneration source/tests with git commit -m "fix: regenerate D1 D2 KNN authority from sealed frames".

### Task 5: Regenerate the four solidified configs

Files:
- Modify: configs/solidified/knn/Dataset1/knn_with_info_sharing.json
- Modify: configs/solidified/knn/Dataset1/knn_without_info_sharing.json
- Modify: configs/solidified/knn/Dataset2/knn_with_info_sharing.json
- Modify: configs/solidified/knn/Dataset2/knn_without_info_sharing.json

- [ ] Confirm the worktree is clean for unrelated files, then run the authorized write command: python tools/protection/codex_timeout.py --timeout 180 -- python scripts/regenerate_solidified_knn.py --datasets 1 2 --write --diff-out /tmp/d1-d2-knn-regeneration. Exit 124 stops the task without retry.
- [ ] Parse all four JSON files and verify identical per-dataset window identity, observed_end == origin, 30 days, no post-origin KNN date, real source/target frame digests, candidate/selection digests, and scenario-specific source order/distances/weights.
- [ ] Commit only the four generated configs with git commit -m "data: reseal D1 D2 KNN selections at origin".

### Task 6: Bind D1/D2 KNN config identity into formal authority

Files:
- Modify: src/protocols/formal_deployment_manifest.py
- Modify: tests/test_formal_deployment_manifest.py
- Generated: 数据集/固化数据/d1_d6_sealed_v1/dataset1/formal-proof.json
- Generated: 数据集/固化数据/d1_d6_sealed_v1/dataset2/formal-proof.json
- Generated: 数据集/固化数据/d1_d6_sealed_v1/deployment-manifest.json
- Generated: 数据集/固化数据/d1_d6_sealed_v1/deployment-manifest.sha256
- Generated: 数据集/固化数据/d1_d6_sealed_v1/code-inventory.json

Interfaces:
- build_formal_proof adds deterministic D1/D2 KNN authority data containing both scenario config paths, SHA-256 hashes, window identity, frame/candidate/selection digests, and a canonical authority digest.
- build_root_manifest exposes d1_d2_selection_authority; validate_deployment_manifest recomputes it.

- [ ] Add tests that load the root manifest, assert D1/D2 with/without files, recompute each config SHA-256 and authority digest, and fail closed after mutating a temporary copy. Keep D4 tests.
- [ ] Run python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_formal_deployment_manifest.py -q. It must fail because the current manifest binds only D4 authority.
- [ ] Add _d1_d2_authority(repository_root, dataset_id). It must validate dataset/scenario/shared protocol/window metadata and return a canonical object with config SHA-256 and its own digest. Include it in D1/D2 dataset-specific proof, root manifest, and validation.
- [ ] After code/config commits, get the current branch and HEAD, then pass those exact values to the official builder:

  git branch --show-current
  authority_head=$(git rev-parse HEAD)
  python tools/protection/codex_timeout.py --timeout 180 -- python tools/operations/build_d1_d6_root_manifest.py --repository-root . --sealed-root 数据集/固化数据/d1_d6_sealed_v1 --expected-branch codex/改 --expected-head "$authority_head"

  The builder must preserve all Parquet bytes. Exit 124 stops without retry.
- [ ] Run the authority tests, then commit generated authority with git commit -m "fix: bind D1 D2 KNN configs into formal authority".

### Task 7: Scoped acceptance and final report

Files:
- Test/Modify if needed: tests/test_d1_d2_knn_window_closure.py
- Scratch only: /tmp/d1-d2-knn-window-final

- [ ] Run the scoped regression suite through the wrapper: python tools/protection/codex_timeout.py --timeout 180 -- python -m pytest tests/test_experiment_protocol_contract.py tests/test_d1_d2_knn_window_closure.py tests/test_runner_protocol_integration.py tests/test_daily_knn_protocol.py tests/test_source_selector_window_leakage.py tests/test_source_selector_shared_protocol.py tests/test_candidate_pool_digest.py tests/test_d2_calendarization_digest_chain.py tests/test_regenerate_d1_d2_knn.py tests/test_formal_deployment_manifest.py -q.
- [ ] Run static closure checks for stale D1/D2 dates, diff --check, and git status. Any remaining old date must be listed as a non-KNN historical reference; no KNN metadata/config/manifest may contain date > origin.
- [ ] Read the four configs and formal manifest and report separately for D1 and D2: origin, observed start/end, source/target frame min/max, source/target frame digest, candidate digest, selection digest, solidified config SHA-256, and formal manifest SHA-256. If any identity fails to recompute, report REJECTED_BLOCKING_DEFECTS.
- [ ] Only after all checks pass, commit any test-only adjustment and verify the final worktree is clean.

## Plan self-review

- Task 1 covers the frozen formula and all D1/D2 runtime caller authority.
- Task 2 covers frame-level filtering before candidate/selector work and future sentinels.
- Task 3 covers actual selector inputs and digest closure.
- Tasks 4–5 cover both scenarios and formal regeneration of all four configs.
- Task 6 covers proof/root/deployment identity recomputation.
- Task 7 covers scoped regression, stale-window audit, clean-tree, and final evidence.
- No D3–D6 freeze change or formal model execution is included.
