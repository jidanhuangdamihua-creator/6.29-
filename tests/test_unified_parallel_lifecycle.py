from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

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


@pytest.fixture
def prepared_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    run_root = tmp_path / "run"
    identity = _identity()
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: identity)
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})
    unified.prepare_formal_run(run_root, resume=False)
    return run_root


def test_mode_worker_selects_exactly_25_plan_cells(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[unified.Task] = []

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

    output = unified.execute_mode_worker(
        prepared_run / "d2_with",
        "d2",
        "with",
        resume=False,
    )

    assert len(seen) == 25
    assert {(task.dataset_token, task.scenario) for task in seen} == {("d2", "with")}
    assert output == prepared_run / "d2_with" / "results" / "dataset2_with_results.csv"


def test_mode_worker_does_not_publish_after_cell_failure(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        unified,
        "run_task",
        lambda task: replace(
            task,
            result_paths=[],
            returncode=9,
            elapsed_seconds=0.01,
        ),
    )
    publish = Mock()
    monkeypatch.setattr(unified, "publish_mode_matrix", publish)

    with pytest.raises(RuntimeError, match="formal cell failed"):
        unified.execute_mode_worker(
            prepared_run / "d2_with",
            "d2",
            "with",
            resume=False,
        )

    publish.assert_not_called()


def test_mode_worker_reuses_fully_verified_mode_without_republishing(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_task = Mock()
    publish = Mock()
    cell_verify = Mock()
    mode_verify = Mock()
    monkeypatch.setattr(unified, "run_task", run_task)
    monkeypatch.setattr(unified, "publish_mode_matrix", publish)
    monkeypatch.setattr(unified, "verify_formal_cell_artifact", cell_verify)
    monkeypatch.setattr(unified, "verify_formal_mode_artifact", mode_verify)

    unified.execute_mode_worker(
        prepared_run / "d5_without",
        "d5",
        "without",
        resume=True,
    )

    assert cell_verify.call_count == 25
    mode_verify.assert_called_once()
    run_task.assert_not_called()
    publish.assert_not_called()


def test_mode_worker_rejects_nested_or_wrong_mode_output_directory(
    prepared_run: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical mode directory"):
        unified.execute_mode_worker(
            prepared_run / "d2_with" / "d2_with",
            "d2",
            "with",
            resume=False,
        )
