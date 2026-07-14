# D1/D2 Formal Input Path Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore D1/D2 formal execution and run-plan identity to the previously generated protocol-derived parquet directory.

**Architecture:** Keep the existing D1-D3 runner and unified lifecycle intact. Change only their authoritative D1/D2 path resolution, with regression tests that inspect real module constants and the path list passed to input-identity discovery.

**Tech Stack:** Python 3.9, pathlib, pytest, existing formal lifecycle utilities.

## Global Constraints

- Do not modify parquet files or generate replacement data.
- Do not change candidate contracts, calendar requirements, or validation strictness.
- D1/D2 must fail if their protocol-derived parquet files are absent; there is no fallback.
- D3-D6 continue using `数据集/固化数据`.
- Every Python test command runs through `python tools/protection/codex_timeout.py --timeout 180 --`.

---

### Task 1: Lock the desired formal input paths with failing tests

**Files:**
- Create: `tests/test_d1_d2_formal_input_paths.py`
- Test: `tests/test_d1_d2_formal_input_paths.py`

**Interfaces:**
- Consumes: `scripts.run_full_paper_experiments.SOLIDIFIED_DATASET_PATHS` and `scripts.run_unified_d1_d6.discover_formal_input_identity(Path)`.
- Produces: Regression coverage for runner paths, run-plan identity paths, and unchanged D3-D6 roots.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from scripts import run_full_paper_experiments as full_runner
from scripts import run_unified_d1_d6 as unified


def test_d1_d2_formal_runner_uses_protocol_derived_parquets() -> None:
    expected_root = Path("数据集/派生数据/d1d2_protocol_v1")
    for dataset_name, dataset_id in (("Dataset1", 1), ("Dataset2", 2)):
        paths = full_runner.SOLIDIFIED_DATASET_PATHS[dataset_name]
        assert Path(paths["source"]) == expected_root / f"dataset{dataset_id}-source.parquet"
        assert Path(paths["target"]) == expected_root / f"dataset{dataset_id}-target.parquet"


def test_run_plan_identity_locks_protocol_derived_d1_d2_parquets(
    tmp_path: Path, monkeypatch,
) -> None:
    captured = []

    def capture(root: Path, paths):
        captured.extend(Path(path) for path in paths)
        return {}

    monkeypatch.setattr(unified, "discover_input_identity", capture)
    unified.discover_formal_input_identity(tmp_path)
    expected_root = tmp_path / "数据集/派生数据/d1d2_protocol_v1"
    for dataset_id in (1, 2):
        assert expected_root / f"dataset{dataset_id}-source.parquet" in captured
        assert expected_root / f"dataset{dataset_id}-target.parquet" in captured
        assert tmp_path / "数据集/固化数据" / f"dataset{dataset_id}-source.parquet" not in captured
        assert tmp_path / "数据集/固化数据" / f"dataset{dataset_id}-target.parquet" not in captured


def test_run_plan_identity_keeps_d3_d6_on_solidified_parquets(
    tmp_path: Path, monkeypatch,
) -> None:
    captured = []
    monkeypatch.setattr(
        unified,
        "discover_input_identity",
        lambda root, paths: captured.extend(Path(path) for path in paths) or {},
    )
    unified.discover_formal_input_identity(tmp_path)
    for dataset_id in range(3, 7):
        expected_root = tmp_path / "数据集/固化数据"
        assert expected_root / f"dataset{dataset_id}-source.parquet" in captured
        assert expected_root / f"dataset{dataset_id}-target.parquet" in captured
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q tests/test_d1_d2_formal_input_paths.py
```

Expected: the first two tests fail because current code selects `数据集/固化数据`; the D3-D6 preservation test passes.

- [ ] **Step 3: Commit the failing regression tests**

```bash
git add tests/test_d1_d2_formal_input_paths.py
git commit -m "test: lock D1-D2 formal input paths"
```

### Task 2: Restore the minimal path resolution

**Files:**
- Modify: `scripts/run_full_paper_experiments.py:108-116`
- Modify: `scripts/run_unified_d1_d6.py:60-87`
- Test: `tests/test_d1_d2_formal_input_paths.py`

**Interfaces:**
- Consumes: project root `Path` and integer dataset IDs 1-6.
- Produces: `_formal_parquet_dir(project_root: Path, dataset_id: int) -> Path`, used by formal input identity discovery.

- [ ] **Step 1: Restore the D1/D2 runner constants**

```python
SOLIDIFIED_DATASET_PATHS = {
    "Dataset1": {
        "source": "数据集/派生数据/d1d2_protocol_v1/dataset1-source.parquet",
        "target": "数据集/派生数据/d1d2_protocol_v1/dataset1-target.parquet",
    },
    "Dataset2": {
        "source": "数据集/派生数据/d1d2_protocol_v1/dataset2-source.parquet",
        "target": "数据集/派生数据/d1d2_protocol_v1/dataset2-target.parquet",
    },
}
```

- [ ] **Step 2: Route run-plan identity through a dataset-aware parquet root**

```python
def _formal_parquet_dir(project_root: Path, dataset_id: int) -> Path:
    root = Path(project_root)
    if int(dataset_id) in (1, 2):
        return root / "数据集" / "派生数据" / "d1d2_protocol_v1"
    return root / "数据集" / "固化数据"
```

Use this helper for each dataset's source and target entries in `discover_formal_input_identity`.

- [ ] **Step 3: Run the new tests and verify GREEN**

Run:

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q tests/test_d1_d2_formal_input_paths.py
```

Expected: `3 passed`.

- [ ] **Step 4: Commit the production fix**

```bash
git add scripts/run_full_paper_experiments.py scripts/run_unified_d1_d6.py
git commit -m "fix: restore D1-D2 formal input paths"
```

### Task 3: Verify related lifecycle behavior

**Files:**
- Verify: `tests/test_unified_parallel_lifecycle.py`
- Verify: `tests/test_protocol_preflight.py`
- Verify: `tests/test_d1_d2_formal_input_paths.py`

**Interfaces:**
- Consumes: the restored path resolver and unchanged formal lifecycle.
- Produces: evidence that lifecycle selection, preflight routing, and path identity agree.

- [ ] **Step 1: Run focused related tests**

```bash
python tools/protection/codex_timeout.py --timeout 180 -- \
  .venv/bin/python -m pytest -q \
  tests/test_d1_d2_formal_input_paths.py \
  tests/test_unified_parallel_lifecycle.py \
  tests/test_protocol_preflight.py
```

Expected: all selected tests pass within 180 seconds.

- [ ] **Step 2: Run syntax and diff checks**

```bash
.venv/bin/python -m py_compile \
  scripts/run_full_paper_experiments.py \
  scripts/run_unified_d1_d6.py \
  tests/test_d1_d2_formal_input_paths.py
git diff --check
git status --short
```

Expected: syntax and diff checks pass; only planned code/test changes or commits are present; no parquet file is modified.

- [ ] **Step 3: Review deployment limitation**

Confirm that no local or rescue-server protocol-derived D1/D2 parquet exists. Report deployment synchronization as a remaining external prerequisite rather than generating data or weakening tests.
