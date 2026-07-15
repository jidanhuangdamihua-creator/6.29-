from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_unified_d1_d6 as unified
from src.utils.result_acceptance import ResultAcceptanceError
from src.utils.run_artifacts import CodeIdentity


def _identity(commit: str = "abc123", digest: str = "a" * 64) -> CodeIdentity:
    return CodeIdentity(commit, False, digest)


def test_build_run_plan_locks_60_unique_seed_bundles_and_identity(tmp_path: Path) -> None:
    identity = _identity()

    plan = unified.build_run_plan(
        tmp_path / "run",
        code_identity=identity,
        input_identity={"input": {"sha256": "b" * 64, "bytes": 1}},
    )

    cells = plan["cells"]
    assert len(cells) == 60
    assert len({cell["result_path"] for cell in cells}) == 60
    assert {tuple(cell["horizons"]) for cell in cells} == {(1, 2, 3, 4, 5)}
    assert all("horizon" not in cell for cell in cells)
    for cell in cells:
        assert {
            "mode_cache_identity",
            "artifact_schema_registry_identity",
            "predictor_feature_schema_identity",
            "source_repair_identity",
            "expected_trace_identities",
        }.issubset(cell)
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


def test_mode_worker_selects_exactly_five_plan_bundles(
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

    assert len(seen) == 5
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

    assert cell_verify.call_count == 5
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


def test_aggregate_requires_all_twelve_verified_modes(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified: list[Path] = []
    published: list[Path] = []
    monkeypatch.setattr(
        unified,
        "verify_formal_mode_artifact",
        lambda path, **kwargs: verified.append(path),
    )
    monkeypatch.setattr(
        unified,
        "publish_global_aggregate",
        lambda paths, **kwargs: published.extend(paths),
    )

    output = unified.aggregate_prepared_run(prepared_run)

    assert len(verified) == 12
    assert len(set(verified)) == 12
    assert published == verified
    assert output == prepared_run / "results" / "d1_d6_results.csv"


def test_aggregate_never_publishes_when_one_mode_fails_validation(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_d3_with(path: Path, **kwargs) -> None:
        if "d3_with" in str(path):
            raise ResultAcceptanceError("mode artifact failed validation")

    publish = Mock()
    monkeypatch.setattr(unified, "verify_formal_mode_artifact", fail_on_d3_with)
    monkeypatch.setattr(unified, "publish_global_aggregate", publish)

    with pytest.raises(ResultAcceptanceError, match="failed validation"):
        unified.aggregate_prepared_run(prepared_run)

    publish.assert_not_called()


def test_aggregate_rejects_non_full_plan_before_publication(
    prepared_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = prepared_run / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cells"] = plan["cells"][:-1]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    publish = Mock()
    monkeypatch.setattr(unified, "publish_global_aggregate", publish)

    with pytest.raises(RuntimeError, match="plan|identity"):
        unified.aggregate_prepared_run(prepared_run)

    publish.assert_not_called()


def test_parse_args_exposes_internal_lifecycle_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_unified_d1_d6.py",
            "--operation",
            "mode-worker",
            "--only",
            "d4",
            "--info-sharing",
            "with",
            "--output-dir",
            "/tmp/formal/d4_with",
            "--resume",
        ],
    )

    args = unified._parse_args()

    assert args.operation == "mode-worker"
    assert args.only == ["d4"]
    assert args.info_sharing == "with"
    assert args.output_dir == Path("/tmp/formal/d4_with")
    assert args.resume is True


def test_main_dispatches_prepare_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    prepare = Mock(
        return_value={"run_identity": "a" * 64, "cells": [None] * 60}
    )
    monkeypatch.setattr(
        unified,
        "_parse_args",
        lambda: argparse.Namespace(
            operation="prepare",
            only=None,
            info_sharing=None,
            output_dir=run_root,
            dry_run=False,
            resume=False,
            smoke=False,
        ),
    )
    monkeypatch.setattr(unified, "prepare_formal_run", prepare)

    unified.main()

    prepare.assert_called_once_with(run_root, resume=False)


def test_prepare_resume_rejects_legacy_300_cell_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "legacy"
    run_root.mkdir()
    (run_root / "run_plan.json").write_text(
        json.dumps(
            {
                "run_plan_version": "formal_d1_d6_run_plan_v2",
                "cells": [{"horizon": 1, "seed": 42}] * 300,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(unified, "discover_code_identity", lambda _: _identity())
    monkeypatch.setattr(unified, "discover_formal_input_identity", lambda _: {})

    with pytest.raises(RuntimeError, match="legacy 300-cell"):
        unified.prepare_formal_run(run_root, resume=True)


def test_main_dispatches_mode_worker_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_dir = tmp_path / "run" / "d4_with"
    worker = Mock(return_value=mode_dir / "results" / "dataset4_with_results.csv")
    monkeypatch.setattr(
        unified,
        "_parse_args",
        lambda: argparse.Namespace(
            operation="mode-worker",
            only=["d4"],
            info_sharing="with",
            output_dir=mode_dir,
            dry_run=False,
            resume=True,
            smoke=False,
        ),
    )
    monkeypatch.setattr(unified, "execute_mode_worker", worker)

    unified.main()

    worker.assert_called_once_with(mode_dir, "d4", "with", resume=True)


def test_main_dispatches_aggregate_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    aggregate = Mock(return_value=run_root / "results" / "d1_d6_results.csv")
    monkeypatch.setattr(
        unified,
        "_parse_args",
        lambda: argparse.Namespace(
            operation="aggregate",
            only=None,
            info_sharing=None,
            output_dir=run_root,
            dry_run=False,
            resume=False,
            smoke=False,
        ),
    )
    monkeypatch.setattr(unified, "aggregate_prepared_run", aggregate)

    unified.main()

    aggregate.assert_called_once_with(run_root)
