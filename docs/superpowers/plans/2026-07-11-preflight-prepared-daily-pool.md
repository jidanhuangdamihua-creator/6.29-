# D1–D6 Preflight Prepared Daily Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi-target D1–D6 preflight reuse one vectorized 30-day source pool while preserving strict selection outputs exactly.

**Architecture:** Add an immutable prepared-pool production object in `candidate_pool.py`, route both direct selection and `SourceSelector` through it, and let preflight attach one shared pool to target-specific protocol metadata without reprocessing the complete source. Bound only the presentation of exclusions.

**Tech Stack:** Python 3.9+, pandas, NumPy, standard-library unittest.

## Global Constraints

- No formal training or experiment run.
- No data generation and no writes to fixed parquet/JSON/CSV artifacts.
- Every Python command uses `tools/protection/codex_timeout.py --timeout 180`.
- Exit code 124 stops all further experiment/validation attempts.
- Tie tolerance remains absolute `1e-12`; inverse-distance epsilon remains `1e-8`.
- No K shrink and no statistics-signature fallback.

---

### Task 1: Prepared Pool and Equivalent Selection

**Files:**
- Modify: `src/protocols/candidate_pool.py`
- Create: `tests/test_prepared_daily_sequence_pool.py`

**Interfaces:**
- Produces: `PreparedDailySequencePool`, `prepare_daily_sequence_pool(...)`, and optional `prepared_pool` on `select_daily_sequence_sources(...)`.

- [x] Write failing tests comparing prepared selection against a frozen reference result for digests, keys, vectors, distance, weight, tie, min/max, and exclusions.
- [x] Verify RED with the protected unittest command.
- [x] Implement vectorized key construction, one pivot/reindex, integer key lookup, and batched float64 distance.
- [x] Verify GREEN, future invariance, observed sensitivity, missing/duplicate/non-finite/insufficient-K behavior.

### Task 2: Single Preparation Across Multi-Target Preflight

**Files:**
- Modify: `src/protocols/runner_adapter.py`
- Modify: `src/source_selection/source_selector.py`
- Modify: `scripts/validate_d1_d6_protocol_inputs.py`
- Modify: `tests/test_protocol_preflight.py`

**Interfaces:**
- Consumes: `PreparedDailySequencePool`.
- Produces: prepared-source target configuration and one preparation per CLI dataset/scenario.

- [x] Write a failing call-count test with two targets asserting one source-pool preparation.
- [x] Verify RED.
- [x] Refactor candidate construction to consume prepared key/group indexes and attach the pool to a lightweight source frame.
- [x] Route `SourceSelector` to `select_daily_sequence_sources(..., prepared_pool=...)`.
- [x] Verify GREEN and unchanged production selection outputs.

### Task 3: Bounded Preflight Diagnostics

**Files:**
- Modify: `scripts/validate_d1_d6_protocol_inputs.py`
- Modify: `src/protocols/candidate_pool.py`
- Modify: `tests/test_protocol_preflight.py`

**Interfaces:**
- Produces: count/reason/sample/truncated output with default sample limit 20.

- [x] Write a failing test with more than20 exclusions and exact reason counts.
- [x] Verify RED.
- [x] Add presentation-only summarization and bounded insufficient-K messages.
- [x] Verify GREEN and confirm digests do not depend on truncation.

### Task 4: Regression and Server Acceptance

**Files:**
- Modify: implementation audit/runbook only if command/output contract changed.

- [x] Run the new focused unittest suite through the 180-second wrapper.
- [x] Run the existing 74-test strict protocol suite through the wrapper.
- [x] Run `compileall` and scoped `git diff --check`.
- [x] Run the exact D5-without protected preflight command; if exit124, stop immediately.
- [x] Record D4-without evidence and report no formal experiment was run.
