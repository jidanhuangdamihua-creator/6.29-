# Mode-Level Bounded Parallel Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore bounded `dataset × mode` parallel execution while preserving the existing cell, mode, and global acceptance and atomic-publication contract.

**Architecture:** `scripts/run_unified_d1_d6.py` remains the only Python authority for formal plan creation, mode execution, artifact validation, and global publication. `scripts/parallel_mode_runner.sh` becomes a process-only supervisor: it prepares one immutable 300-cell run, schedules 12 isolated mode workers with global and D5 caps, and requests one global publication only after every worker succeeds.

**Tech Stack:** Python 3, Bash, `pytest`/`unittest`, existing `RunLayout`, result acceptance, manifests, SHA-256, POSIX process groups, Linux `setsid --wait`.

## Global Constraints

- Implementation base is `7d8221f06baa22da959eaccf71207b304d8e3c0e` on `codex/preseal-blocker-fixes`.
- Do not change models, datasets, protocols, metrics, target keys, result schemas, or `src/utils/result_acceptance.py` semantics.
- Do not restore CSV-existence validation, CSV copying, shell concatenation, or manifest-free collection.
- A full plan is exactly 300 unique cells; one mode worker is exactly 25 cells; the full run is exactly 12 modes.
- Default `MAX_JOBS=6`, accepted range `1..12`; D5 concurrency is always capped at one.
- Any experiment-like Python command must use `python tools/protection/codex_timeout.py <command...>` and work stops immediately on exit 124.
- Do not launch the full experiment during implementation verification.

---

### Task 1: Immutable Formal Run Plan Lifecycle

**Files:**
- Modify: `scripts/run_unified_d1_d6.py:6-345`
- Test: `tests/test_unified_parallel_lifecycle.py`

**Interfaces:**
- Produces: `build_run_plan(run_root: Path, code_identity: CodeIdentity, input_identity: dict) -> dict[str, object]`.
- Produces: `prepare_formal_run(run_root: Path, *, resume: bool) -> dict[str, object]`.
- Produces: `load_validated_run_plan(run_root: Path) -> tuple[dict[str, object], CodeIdentity]`.
- `run_identity` is SHA-256 of canonical JSON for the plan payload before the `run_identity` field is inserted.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_build_run_plan_locks_300_unique_cells_and_identity(tmp_path, monkeypatch):
    identity = CodeIdentity("abc123", False, "f" * 64)
    monkeypatch.setattr(unified, "PROJECT_ROOT", tmp_path)
    plan = unified.build_run_plan(tmp_path / "run", identity, {"input": {"sha256": "a"}})
    assert len(plan["cells"]) == 300
    assert len({cell["result_path"] for cell in plan["cells"]}) == 300
    assert len(plan["run_identity"]) == 64
    assert plan["code_identity"] == identity.to_dict()

def test_prepare_resume_rejects_changed_code_identity(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    clean = CodeIdentity("abc123", False, "a" * 64)
    changed = CodeIdentity("def456", False, "b" * 64)
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: clean)
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})
    unified.prepare_formal_run(run_root, resume=False)
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: changed)
    with pytest.raises(RuntimeError, match="plan|identity|resume"):
        unified.prepare_formal_run(run_root, resume=True)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py -k 'run_plan or prepare_resume'`

Expected: collection or assertion failure because the lifecycle functions do not exist.

- [ ] **Step 3: Implement canonical plan creation and validation**

```python
def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def build_run_plan(run_root: Path, code_identity: CodeIdentity, input_identity: dict[str, dict[str, object]]) -> dict[str, object]:
    tasks = build_tasks(None, smoke=False, run_dir=run_root)
    paths = [str(task.expected_result_path) for task in tasks]
    if len(tasks) != 300 or len(set(paths)) != 300:
        raise RuntimeError("formal run plan must contain exactly 300 unique cells")
    payload = {
        "run_plan_version": "formal_d1_d6_run_plan_v2",
        "code_identity": code_identity.to_dict(),
        "schema_registry_version": RESULT_SCHEMA_REGISTRY_VERSION,
        "schema_registry_digest": result_schema_registry_digest(),
        "input_identity": input_identity,
        "methods": list(FORMAL_METHODS),
        "horizons": list(FORMAL_HORIZONS),
        "seeds": list(FORMAL_SEEDS),
        "cells": [_task_plan_entry(task) for task in tasks],
    }
    return {**payload, "run_identity": _canonical_digest(payload)}
```

`prepare_formal_run` must reserve a new root or require an existing resume root, reject dirty code, compute identities, and call the existing atomic `write_or_validate_run_plan`. `load_validated_run_plan` must rebuild the current plan and require exact equality before returning it.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py -k 'run_plan or prepare_resume'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_unified_d1_d6.py tests/test_unified_parallel_lifecycle.py
git commit -m "feat: add immutable formal run lifecycle"
```

### Task 2: Fail-Closed Artifact Revalidation

**Files:**
- Modify: `src/utils/run_artifacts.py:100-420`
- Test: `tests/test_run_layout_and_atomic_publication.py`

**Interfaces:**
- Produces: `verify_formal_cell_artifact(stable_path, acceptance_path, expected, code_identity) -> None`.
- Produces: `verify_formal_mode_artifact(stable_path, acceptance_path, cell_paths, expected, code_identity) -> None`.
- Existing publication and acceptance semantics remain unchanged.

- [ ] **Step 1: Write failing artifact-tamper tests**

```python
def test_verify_cell_requires_passing_acceptance_sidecar(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    path = layout.cell_result(1, "without", 1, 42)
    expected = build_formal_cell_contract(
        dataset_id=1, mode="without", targets=("1/10",), horizon=1, seed=42
    )
    identity = CodeIdentity("abc", False, "1" * 64)
    publish_formal_cell_frame(
        _valid_cell(), stable_path=path, expected=expected, code_identity=identity
    )
    layout.cell_acceptance_report(1, "without", 1, 42).unlink()
    with pytest.raises(ResultAcceptanceError, match="acceptance report"):
        verify_formal_cell_artifact(
            path,
            acceptance_path=layout.cell_acceptance_report(1, "without", 1, 42),
            expected=expected,
            code_identity=identity,
        )

@pytest.fixture
def accepted_mode_artifacts(
    tmp_path: Path,
) -> tuple[RunLayout, list[Path], ExpectedResultContract, CodeIdentity]:
    layout = RunLayout(tmp_path / "run")
    identity = CodeIdentity("abc", False, "2" * 64)
    expected = ExpectedResultContract(
        scope=AcceptanceScope.MODE_MATRIX,
        formal=True,
        dataset_ids=(1,),
        modes=("without",),
        protocol_tracks=("strict_paper",),
        targets_by_dataset_mode={(1, "without"): ("1/10",)},
        methods=FORMAL_METHODS,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        confirmation_eligible=True,
    )
    paths = []
    for horizon in expected.horizons:
        for seed in expected.seeds:
            path = layout.cell_result(1, "without", horizon, seed)
            cell_expected = ExpectedResultContract(
                **{
                    **expected.__dict__,
                    "scope": AcceptanceScope.CELL,
                    "horizons": (horizon,),
                    "seeds": (seed,),
                }
            )
            publish_formal_cell_frame(
                _valid_cell().assign(horizon=horizon, seed=seed),
                stable_path=path,
                expected=cell_expected,
                code_identity=identity,
            )
            paths.append(path)
    publish_mode_matrix(
        paths,
        stable_path=layout.mode_result(1, "without"),
        expected=expected,
        code_identity=identity,
    )
    return layout, paths, expected, identity

def test_verify_mode_rejects_manifest_hash_mismatch(
    accepted_mode_artifacts: tuple[RunLayout, list[Path], ExpectedResultContract, CodeIdentity],
) -> None:
    layout, cell_paths, mode_expected, identity = accepted_mode_artifacts
    mode_path = layout.mode_result(1, "without")
    mode_path.write_text(mode_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ResultAcceptanceError, match="hash"):
        verify_formal_mode_artifact(
            mode_path,
            acceptance_path=layout.mode_acceptance_report(1, "without"),
            cell_paths=cell_paths,
            expected=mode_expected,
            code_identity=identity,
        )
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_run_layout_and_atomic_publication.py -k 'verify_cell or verify_mode'`

Expected: import failure because the verification functions do not exist.

- [ ] **Step 3: Implement sidecar, manifest, hash, and acceptance revalidation**

```python
def _require_passing_acceptance_report(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ResultAcceptanceError(f"acceptance report unreadable: {path}") from exc
    if payload.get("passed") is not True:
        raise ResultAcceptanceError(f"acceptance report did not pass: {path}")
    return payload

def verify_formal_cell_artifact(
    stable_path: Path,
    *,
    acceptance_path: Path,
    expected: ExpectedResultContract,
    code_identity: CodeIdentity,
) -> None:
    _require_passing_acceptance_report(acceptance_path)
    _require_matching_artifact_manifest(stable_path, artifact_type="formal_cell", code_identity=code_identity)
    outcome = accept_cell_csv(stable_path, expected=expected)
    if not outcome.report.passed:
        raise ResultAcceptanceError("cell acceptance revalidation failed: " + ",".join(outcome.report.reasons))
```

The mode verifier must verify all cells first, verify the mode acceptance sidecar and manifest, and call `accept_mode_matrix(cell_paths, expected=expected, candidate_mode_csv=stable_path)`.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_run_layout_and_atomic_publication.py -k 'verify_cell or verify_mode or mode_and_selection'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/run_artifacts.py tests/test_run_layout_and_atomic_publication.py
git commit -m "feat: revalidate accepted formal artifacts"
```

### Task 3: Supervised Mode Worker Operation

**Files:**
- Modify: `scripts/run_unified_d1_d6.py:165-409`
- Test: `tests/test_unified_parallel_lifecycle.py`

**Interfaces:**
- CLI: `--operation mode-worker --only dN --info-sharing without|with --output-dir <run-root>/dN_<mode> [--resume]`.
- Produces: `execute_mode_worker(mode_dir: Path, dataset: str, mode: str, *, resume: bool) -> Path`.
- Worker reads `<mode-dir>/../run_plan.json`, executes exactly 25 matching planned tasks, and returns the accepted mode CSV.

- [ ] **Step 1: Write failing worker-selection and failure tests**

```python
@pytest.fixture
def prepared_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_root = tmp_path / "run"
    identity = CodeIdentity("abc123", False, "a" * 64)
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: identity)
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})
    unified.prepare_formal_run(run_root, resume=False)
    return run_root

def test_mode_worker_selects_exactly_25_plan_cells(
    prepared_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    def complete(task: unified.Task) -> unified.Task:
        seen.append(task)
        return replace(
            task,
            result_paths=[task.expected_result_path],
            returncode=0,
            elapsed_seconds=0.01,
        )
    monkeypatch.setattr(unified, "run_task", complete)
    monkeypatch.setattr(unified, "verify_formal_cell_artifact", lambda *a, **k: None)
    monkeypatch.setattr(unified, "publish_mode_matrix", lambda *a, **k: None)
    monkeypatch.setattr(unified, "verify_formal_mode_artifact", lambda *a, **k: None)
    unified.execute_mode_worker(prepared_run / "d2_with", "d2", "with", resume=False)
    assert len(seen) == 25
    assert {(task.dataset_token, task.scenario) for task in seen} == {("d2", "with")}

def test_mode_worker_does_not_publish_after_cell_failure(
    prepared_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(unified, "run_task", lambda task: replace(task, returncode=1))
    publish = Mock()
    monkeypatch.setattr(unified, "publish_mode_matrix", publish)
    with pytest.raises(RuntimeError, match="failed"):
        unified.execute_mode_worker(prepared_run / "d2_with", "d2", "with", resume=False)
    publish.assert_not_called()
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py -k mode_worker`

Expected: failure because the worker operation is missing.

- [ ] **Step 3: Implement exact plan selection, strict resume, and mode publication**

```python
def execute_mode_worker(mode_dir: Path, dataset: str, mode: str, *, resume: bool) -> Path:
    dataset_id = int(dataset[1:])
    run_root = Path(mode_dir).parent
    plan, code_identity = load_validated_run_plan(run_root)
    tasks = build_tasks([dataset], smoke=False, run_dir=run_root, info_sharing=mode)
    _require_tasks_match_plan(tasks, plan, expected_count=25)
    expected = build_mode_expected_contract(dataset=dataset, scenario=mode)
    for task in tasks:
        cell_expected = replace(expected, scope=AcceptanceScope.CELL, horizons=(task.horizon,), seeds=(task.seed,))
        if resume and _cell_is_reusable(task, cell_expected, code_identity):
            continue
        completed = run_task(task)
        if completed.returncode != 0 or not completed.result_paths:
            raise RuntimeError(f"formal cell failed: {task.label}")
        verify_formal_cell_artifact(task.expected_result_path, task.expected_result_path.with_suffix(".acceptance.json"), cell_expected, code_identity)
    cell_paths = [task.expected_result_path for task in tasks]
    output = RunLayout(run_root).mode_result(dataset_id, mode)
    publish_mode_matrix(cell_paths, stable_path=output, expected=expected, code_identity=code_identity)
    verify_formal_mode_artifact(output, output.with_suffix(".acceptance.json"), cell_paths, expected, code_identity)
    return output
```

If all cells and the existing mode artifact revalidate on resume, return without republishing. If cells pass but the mode artifact fails, republish through `publish_mode_matrix`.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py -k mode_worker`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_unified_d1_d6.py tests/test_unified_parallel_lifecycle.py
git commit -m "feat: add accepted formal mode worker"
```

### Task 4: Parent-Only Global Publication

**Files:**
- Modify: `scripts/run_unified_d1_d6.py:241-409`
- Test: `tests/test_unified_parallel_lifecycle.py`

**Interfaces:**
- CLI: `--operation aggregate --output-dir <run-root>`.
- Produces: `aggregate_prepared_run(run_root: Path) -> Path`.
- Consumes exactly 12 mode artifacts from the immutable full plan.

- [ ] **Step 1: Write failing aggregate gate tests**

```python
def test_aggregate_requires_all_twelve_verified_modes(
    prepared_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    paths_seen = []
    monkeypatch.setattr(unified, "verify_formal_mode_artifact", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(unified, "publish_global_aggregate", lambda paths, **k: paths_seen.extend(paths))
    unified.aggregate_prepared_run(prepared_run)
    assert len(calls) == 12
    assert len(paths_seen) == 12

def test_aggregate_never_publishes_when_one_mode_fails_validation(
    prepared_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_on_d3_with(path: Path, *args, **kwargs) -> None:
        if "d3_with" in str(path):
            raise ResultAcceptanceError("mode artifact failed validation")
    monkeypatch.setattr(unified, "verify_formal_mode_artifact", fail_on_d3_with)
    publish = Mock()
    monkeypatch.setattr(unified, "publish_global_aggregate", publish)
    with pytest.raises(ResultAcceptanceError):
        unified.aggregate_prepared_run(prepared_run)
    publish.assert_not_called()
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py -k aggregate`

Expected: failure because `aggregate_prepared_run` is missing.

- [ ] **Step 3: Implement revalidation and one global publication call**

Build all 12 mode contracts and paths from `RunLayout`, verify every mode against its exact 25 plan cells, require the plan to be the full 300-cell profile, recheck formal input identity, then call:

```python
publish_global_aggregate(
    mode_paths,
    stable_path=layout.aggregate_result,
    expected=_global_contract(mode_contracts),
    code_identity=code_identity,
)
```

The `main()` dispatcher must preserve current standalone behavior while adding `prepare`, `mode-worker`, and `aggregate`; invalid combinations exit 2 before filesystem mutation.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_unified_parallel_lifecycle.py tests/test_unified_d1_d6_output_contract.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_unified_d1_d6.py tests/test_unified_parallel_lifecycle.py tests/test_unified_d1_d6_output_contract.py
git commit -m "feat: gate global publication on accepted modes"
```

### Task 5: Static Dry-Run and Validated Supervisor Interface

**Files:**
- Modify: `scripts/parallel_mode_runner.sh`
- Modify: `scripts/parallel_runner.sh`
- Test: `tests/test_parallel_mode_supervisor.py`

**Interfaces:**
- Environment: `MAX_JOBS=1..12`, `RUN_ROOT=<path>`, `RESUME=0|1`, `DRY_RUN=0|1`, `PROBE=0|1`, `PUBLISH_GLOBAL=0|1`.
- `PROBE=1` requires `MAX_JOBS=4` and `PUBLISH_GLOBAL=0`, and selects exactly `d1_without d1_with d2_without d2_with`.
- Full mode requires all 12 tasks and `PUBLISH_GLOBAL=1`.
- `parallel_runner.sh` delegates to `parallel_mode_runner.sh` so there is one supported scheduler.

- [ ] **Step 1: Write failing static interface tests**

```python
def test_dry_run_prints_twelve_unique_mode_workers_without_launch(tmp_path):
    completed = run_supervisor(tmp_path, DRY_RUN="1", RUN_ROOT=str(tmp_path / "formal"))
    assert completed.returncode == 0
    lines = [line for line in completed.stdout.splitlines() if line.startswith("[MODE]")]
    assert len(lines) == 12
    assert len(set(lines)) == 12
    assert "cells=300 unique=300" in completed.stdout
    assert not (tmp_path / "formal").exists()

@pytest.mark.parametrize("value", ["0", "13", "x", "1.5", ""])
def test_invalid_max_jobs_fails_before_run_root_creation(tmp_path, value):
    completed = run_supervisor(tmp_path, MAX_JOBS=value, RUN_ROOT=str(tmp_path / "formal"))
    assert completed.returncode == 2
    assert not (tmp_path / "formal").exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_parallel_mode_supervisor.py -k 'dry_run or invalid_max_jobs'`

Expected: failures because the wrapper still delegates one serial plan.

- [ ] **Step 3: Implement validation and no-launch dry-run**

The shell must validate all environment variables before resolving `setsid`, creating directories, or invoking Python. Construct the fixed mode list, print one worker command per mode with the mode directory as `--output-dir`, and print `[FORMAL PLAN] cells=300 unique=300`. Dry-run exits immediately.

`parallel_runner.sh` becomes:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/parallel_mode_runner.sh" "$@"
```

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_parallel_mode_supervisor.py -k 'dry_run or invalid_max_jobs or probe'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/parallel_mode_runner.sh scripts/parallel_runner.sh tests/test_parallel_mode_supervisor.py
git commit -m "feat: define bounded formal supervisor interface"
```

### Task 6: Bounded Scheduler and Process-Group Failure Handling

**Files:**
- Modify: `scripts/parallel_mode_runner.sh`
- Test: `tests/test_parallel_mode_supervisor.py`
- Create: `tests/fixtures/fake_formal_worker.py`

**Interfaces:**
- Scheduler state per task: queued, starting, running, succeeded, failed, interrupted.
- `pids.tsv` columns: `task`, `launcher_pid`, `pid`, `pgid`, `status`, `event_time`, `elapsed_seconds`, `exit_code`, `log_file`, `output_dir`.
- On first failure: stop launches, TERM all active worker PGIDs, wait a bounded grace period, KILL survivors, reap launchers, return nonzero, and skip aggregate.

- [ ] **Step 1: Write failing fake-worker concurrency tests**

```python
def test_scheduler_enforces_global_and_d5_caps_and_overlaps(tmp_path):
    completed = run_with_fake_worker(tmp_path, MAX_JOBS="4", FAKE_SLEEP="0.15")
    events = read_events(tmp_path)
    assert completed.returncode == 0
    assert peak_running(events) <= 4
    assert peak_running(events, prefix="d5_") == 1
    assert has_non_d5_overlap(events)

def test_first_worker_failure_terminates_peers_and_skips_global(tmp_path):
    completed = run_with_fake_worker(tmp_path, MAX_JOBS="4", FAKE_FAIL_MODE="d2_with")
    assert completed.returncode != 0
    assert "aggregate" not in fake_invocations(tmp_path)
    assert all_processes_reaped(events=read_events(tmp_path))
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_parallel_mode_supervisor.py -k 'caps or overlap or failure or signal'`

Expected: failures because bounded scheduling and group cleanup are absent.

- [ ] **Step 3: Implement process supervision**

Implement queue scanning that launches while `running < MAX_JOBS`, skipping a queued D5 task while another D5 worker is active but continuing to scan later non-D5 tasks. Launch each worker through `setsid --wait`, resolve a safe PID/PGID, append atomic state events, and poll/reap launchers. Use one idempotent cleanup function for launch failure, worker failure, INT, and TERM. Never parse result CSVs in shell.

After all selected workers succeed, invoke `--operation aggregate` only when the selected set is all 12 modes and `PUBLISH_GLOBAL=1`.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_parallel_mode_supervisor.py`

Expected: PASS with measured overlap, global peak within `MAX_JOBS`, D5 peak exactly one, failure fan-out cleanup, and no aggregate after failure.

- [ ] **Step 5: Commit**

```bash
git add scripts/parallel_mode_runner.sh tests/test_parallel_mode_supervisor.py tests/fixtures/fake_formal_worker.py
git commit -m "feat: supervise bounded formal mode workers"
```

### Task 7: Documentation and Complete Verification

**Files:**
- Modify: `README.md:400`
- Modify: `docs/superpowers/specs/2026-07-14-mode-level-bounded-parallel-supervisor-design.md` only if implementation names differ from the approved interface.

**Interfaces:**
- Documents normal, resume, dry-run, and four-mode probe commands.

- [ ] **Step 1: Update operator commands**

```bash
# Full new run
MAX_JOBS=6 bash scripts/parallel_mode_runner.sh

# Exact-identity resume
RUN_ROOT=outputs/runs/<run-id> RESUME=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh

# No-launch plan inspection
DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh

# Server probe; never globally published
PROBE=1 PUBLISH_GLOBAL=0 MAX_JOBS=4 RUN_ROOT=outputs/runs/<probe-id> bash scripts/parallel_mode_runner.sh
```

- [ ] **Step 2: Run focused non-experiment checks**

Run:

```bash
pytest -q tests/test_unified_parallel_lifecycle.py tests/test_parallel_mode_supervisor.py tests/test_unified_d1_d6_output_contract.py tests/test_run_layout_and_atomic_publication.py tests/test_result_acceptance_scopes.py
python -m compileall -q scripts/run_unified_d1_d6.py src/utils/run_artifacts.py
bash -n scripts/parallel_mode_runner.sh scripts/parallel_runner.sh
git diff --check
DRY_RUN=1 MAX_JOBS=6 bash scripts/parallel_mode_runner.sh
```

Expected: all tests pass; compile, Bash syntax, diff check, and dry-run exit 0; dry-run prints 12 workers and `cells=300 unique=300` without creating the run root.

- [ ] **Step 3: Run the complete unit test suite through the required wrapper**

Run: `python tools/protection/codex_timeout.py python -m pytest -q`

Expected: PASS within 180 seconds. If exit code is 124, stop immediately and give this exact command to the user without retrying or splitting the suite.

- [ ] **Step 4: Commit documentation and any verification-only fixes**

```bash
git add README.md docs/superpowers/specs/2026-07-14-mode-level-bounded-parallel-supervisor-design.md
git commit -m "docs: document bounded formal execution"
```

- [ ] **Step 5: Do not run the full experiment; hand off the protected probe command**

Manual server command:

```bash
python tools/protection/codex_timeout.py env PROBE=1 PUBLISH_GLOBAL=0 MAX_JOBS=4 RUN_ROOT=outputs/runs/<probe-id> bash scripts/parallel_mode_runner.sh
```

If Codex is explicitly asked to run it and the wrapper exits 124, stop all work immediately and return the same command for manual Terminal execution.
