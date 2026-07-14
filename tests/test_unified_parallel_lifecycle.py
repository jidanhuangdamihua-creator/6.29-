from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_unified_d1_d6 as unified
from src.utils.run_artifacts import CodeIdentity


def _identity(commit: str = "abc123", digest: str = "a" * 64) -> CodeIdentity:
    return CodeIdentity(commit, False, digest)


def test_build_run_plan_locks_300_unique_cells_and_identity(tmp_path: Path) -> None:
    identity = _identity()

    plan = unified.build_run_plan(
        tmp_path / "run",
        code_identity=identity,
        input_identity={"input": {"sha256": "b" * 64, "bytes": 1}},
    )

    cells = plan["cells"]
    assert len(cells) == 300
    assert len({cell["result_path"] for cell in cells}) == 300
    assert len(plan["run_identity"]) == 64
    assert plan["code_identity"] == identity.to_dict()
    assert {
        (cell["dataset_id"], cell["mode"])
        for cell in cells
    } == {
        (dataset_id, mode)
        for dataset_id in range(1, 7)
        for mode in ("without", "with")
    }


def test_prepare_resume_rejects_changed_code_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    current = _identity()
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: current)
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})

    prepared = unified.prepare_formal_run(run_root, resume=False)

    assert run_root.is_dir()
    assert json.loads((run_root / "run_plan.json").read_text(encoding="utf-8")) == prepared

    current = _identity("different", "c" * 64)
    with pytest.raises(RuntimeError, match="plan|identity|resume"):
        unified.prepare_formal_run(run_root, resume=True)


def test_prepare_rejects_dirty_code_before_creating_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    monkeypatch.setattr(
        unified,
        "discover_code_identity",
        lambda _: CodeIdentity("abc123", True, "d" * 64),
    )

    with pytest.raises(RuntimeError, match="clean git worktree"):
        unified.prepare_formal_run(run_root, resume=False)

    assert not run_root.exists()


def test_load_validated_run_plan_returns_locked_code_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    identity = _identity()
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: identity)
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})
    prepared = unified.prepare_formal_run(run_root, resume=False)

    loaded, loaded_identity = unified.load_validated_run_plan(run_root)

    assert loaded == prepared
    assert loaded_identity == identity
