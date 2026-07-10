# Raw/Wide KNN Dry-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only D4–D6 raw/wide KNN comparison diagnostic with deterministic, windowed source selection.

**Architecture:** A single script owns safe paths, raw-to-wide adapters, source diagnostics, runtime-window reconstruction, deterministic entity capping, KNN comparisons, and report writing. It imports the existing source selector, KNN payload loader, domain policy, and raw cleaning helpers but changes no main experiment path.

**Tech Stack:** Python, pandas, NumPy, pyarrow/parquet, pytest.

## Global Constraints

- Never modify `数据集/固化数据/`, `configs/solidified/knn/`, experiment result CSVs, or final summary files.
- Outputs may only be under `outputs/feature_consistency/raw_wide_knn_dryrun_<timestamp>/`.
- Derive source entity identity from JSON `group_cols`.
- Reconstruct the existing 30-day target and 300-day source runtime windows before KNN.
- Missing required JSON features fail explicitly; narrow data is never substituted for wide data.
- Use a stable SHA-256 entity-key cap only after runtime eligibility is determined.

### Task 1: Specify and test diagnostic primitives

**Files:**
- Create: `tests/test_raw_wide_knn_dryrun_compare.py`
- Create: `scripts/dryrun_raw_wide_knn_compare.py`

**Interfaces:**
- Produces `normalize_entity_key(frame, group_cols)`, `compare_feature_schema`, `stable_cap_entities`, `pool_diagnostics`, `topk_overlap_ratio`, and `validate_output_dir`.

- [ ] Write failing unit tests for one-domain vacuity, multi-domain effectiveness, exact/partial/zero overlap, configured-domain selection count, missing/extra features, deterministic caps, and protected output directories.
- [ ] Run `python3 tools/protection/codex_timeout.py --timeout 180 .venv/bin/python -m pytest tests/test_raw_wide_knn_dryrun_compare.py -q`; confirm failures are due to missing script symbols.
- [ ] Implement only the tested pure functions with JSON-safe results.
- [ ] Re-run the test command; confirm all primitive tests pass.

### Task 2: Add runtime windows and wide-clean adapters

**Files:**
- Modify: `scripts/dryrun_raw_wide_knn_compare.py`
- Modify: `tests/test_raw_wide_knn_dryrun_compare.py`

**Interfaces:**
- Consumes `derive_d4_d6_runtime_knn_windows`, JSON payloads, and raw roots.
- Produces `reconstruct_runtime_windows`, `build_or_load_wide_clean`, and a normalized wide DataFrame with `entity_id` plus JSON features.

- [ ] Add failing synthetic tests that assert 30 target days, 300 source days, compound keys from `group_cols`, alias mapping, and explicit feature-missing failure.
- [ ] Run the targeted tests and confirm RED.
- [ ] Implement window reconstruction and test-fixture/raw adapters for D4, D5, and D6; write intermediates only in the validated run directory.
- [ ] Re-run targeted tests and confirm GREEN.

### Task 3: Add KNN comparison, artifacts, and CLI

**Files:**
- Modify: `scripts/dryrun_raw_wide_knn_compare.py`
- Modify: `tests/test_raw_wide_knn_dryrun_compare.py`

**Interfaces:**
- Consumes normalized narrow/wide pools, target data, JSON payloads, and CLI arguments.
- Produces `dataset{N}_raw_wide_knn_compare.csv`, `dataset{N}_raw_wide_knn_compare_summary.json`, and `raw_wide_knn_compare_report.md`.

- [ ] Add a failing tiny-fixture CLI test asserting CSV/JSON/Markdown paths, status fields, window/schema report sections, and no writes outside the temporary output root.
- [ ] Run the test and confirm RED.
- [ ] Implement `--dataset`, `--mode`, `--top-k`, cap, roots, reuse, missing-wide, and debug arguments; use `SourceSelector` with runtime attrs for narrow, wide-with, and wide-without comparisons.
- [ ] Re-run the full new test module and confirm GREEN.

### Task 4: Verify integration and protected artifacts

**Files:**
- Test: `tests/test_raw_wide_knn_dryrun_compare.py`
- Test: `tests/test_d4_d6_domain_filter.py`

- [ ] Run both required pytest commands through the 180-second wrapper.
- [ ] Inspect `git diff --name-status` and assert no protected path changed.
- [ ] Run a synthetic CLI dry-run under a temporary output directory and inspect all three artifacts.
- [ ] Commit the implementation and test changes with an intentional message.
