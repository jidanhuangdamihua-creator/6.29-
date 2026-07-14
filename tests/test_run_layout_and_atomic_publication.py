from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from src.constants import SCHEMA_FAMILY_D1_D3
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.utils.result_acceptance import (
    AcceptanceScope,
    AggregateProfile,
    ExpectedResultContract,
    build_formal_cell_contract,
)
from src.utils.run_artifacts import (
    CodeIdentity,
    discover_code_identity,
    publish_formal_cell_frame,
    publish_global_aggregate,
    publish_mode_matrix,
    resumable_formal_cell,
    write_or_validate_run_plan,
)
from src.utils.run_layout import RunLayout
from test_strict_result_contract import _strict_row


def _valid_cell() -> pd.DataFrame:
    rows = []
    for method in FORMAL_METHODS:
        row = _strict_row(horizon=1, seed=42)
        row.update(
            {
                "dataset_id": "D1",
                "target_entity_key": "Store1/Item10",
                "scenario": "without",
                "information_sharing": "without",
                "method": method,
                "schema_family": SCHEMA_FAMILY_D1_D3,
                "result_status": "trial",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_run_layout_has_one_canonical_owner_for_every_artifact(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "formal-run")

    assert layout.mode_dir(5, "without") == tmp_path / "formal-run" / "d5_without"
    assert layout.cell_dir(5, "without", 3, 44) == (
        tmp_path / "formal-run" / "d5_without" / "cells" / "h3_s44"
    )
    assert layout.cell_result(5, "without", 3, 44) == (
        tmp_path
        / "formal-run"
        / "d5_without"
        / "cells"
        / "h3_s44"
        / "results"
        / "dataset5_without_results.csv"
    )
    assert layout.mode_result(5, "without") == (
        tmp_path / "formal-run" / "d5_without" / "results" / "dataset5_without_results.csv"
    )
    assert layout.aggregate_result == tmp_path / "formal-run" / "results" / "d1_d6_results.csv"

    with pytest.raises(ValueError, match="dataset_id"):
        layout.mode_dir(0, "without")
    with pytest.raises(ValueError, match="mode"):
        layout.mode_dir(1, "invalid")


def test_invalid_candidate_never_replaces_stable_cell(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    stable = layout.cell_result(1, "without", 1, 42)
    invalid = _valid_cell().iloc[:-1]
    expected = build_formal_cell_contract(
        dataset_id=1,
        mode="without",
        targets=("Store1/Item10",),
        horizon=1,
        seed=42,
    )

    with pytest.raises(RuntimeError, match="acceptance failed"):
        publish_formal_cell_frame(
            invalid,
            stable_path=stable,
            expected=expected,
            code_identity=CodeIdentity("abc", False, "0" * 64),
        )

    assert not stable.exists()
    assert not list(stable.parent.glob(f"{stable.name}.tmp.*"))
    report = layout.cell_acceptance_report(1, "without", 1, 42)
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_accepted_cell_is_atomic_hashed_and_resume_requires_same_code(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    stable = layout.cell_result(1, "without", 1, 42)
    expected = build_formal_cell_contract(
        dataset_id=1,
        mode="without",
        targets=("Store1/Item10",),
        horizon=1,
        seed=42,
    )
    identity = CodeIdentity("abc", True, "1" * 64)

    manifest = publish_formal_cell_frame(
        _valid_cell(),
        stable_path=stable,
        expected=expected,
        code_identity=identity,
    )

    assert stable.is_file()
    assert manifest["sha256"]
    manifest_path = layout.cell_manifest(1, "without", 1, 42)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"] == manifest["sha256"]
    assert resumable_formal_cell(
        stable_path=stable,
        manifest_path=manifest_path,
        expected=expected,
        code_identity=identity,
    )
    assert not resumable_formal_cell(
        stable_path=stable,
        manifest_path=manifest_path,
        expected=expected,
        code_identity=CodeIdentity("different", True, "1" * 64),
    )

    stable.write_text(stable.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not resumable_formal_cell(
        stable_path=stable,
        manifest_path=manifest_path,
        expected=expected,
        code_identity=identity,
    )


def test_dirty_code_identity_hashes_contents_not_only_git_status(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    tracked.write_text("first dirty value\n", encoding="utf-8")
    first = discover_code_identity(repo)
    tracked.write_text("second dirty value\n", encoding="utf-8")
    second = discover_code_identity(repo)

    assert first.git_commit == second.git_commit
    assert first.dirty and second.dirty
    assert first.worktree_digest != second.worktree_digest


def test_mode_and_selection_aggregate_are_also_acceptance_gated(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "run")
    identity = CodeIdentity("abc", False, "2" * 64)
    mode_expected = ExpectedResultContract(
        scope=AcceptanceScope.MODE_MATRIX,
        formal=True,
        dataset_ids=(1,),
        modes=("without",),
        protocol_tracks=("strict_paper",),
        targets_by_dataset_mode={(1, "without"): ("Store1/Item10",)},
        methods=FORMAL_METHODS,
        horizons=(1, 2, 3, 4, 5),
        seeds=(42, 43, 44, 45, 46),
        confirmation_eligible=True,
    )
    cell_paths = []
    for horizon in mode_expected.horizons:
        for seed in mode_expected.seeds:
            path = layout.cell_result(1, "without", horizon, seed)
            frame = _valid_cell().assign(horizon=horizon, seed=seed)
            publish_formal_cell_frame(
                frame,
                stable_path=path,
                expected=ExpectedResultContract(
                    **{
                        **mode_expected.__dict__,
                        "scope": AcceptanceScope.CELL,
                        "horizons": (horizon,),
                        "seeds": (seed,),
                    }
                ),
                code_identity=identity,
            )
            cell_paths.append(path)

    invalid_mode = layout.mode_result(1, "without")
    with pytest.raises(RuntimeError, match="mode_matrix acceptance failed"):
        publish_mode_matrix(
            cell_paths[:-1],
            stable_path=invalid_mode,
            expected=mode_expected,
            code_identity=identity,
        )
    assert not invalid_mode.exists()

    publish_mode_matrix(
        cell_paths,
        stable_path=invalid_mode,
        expected=mode_expected,
        code_identity=identity,
    )
    assert invalid_mode.is_file()
    assert layout.mode_manifest(1, "without").is_file()

    aggregate_expected = ExpectedResultContract(
        **{
            **mode_expected.__dict__,
            "scope": AcceptanceScope.GLOBAL_AGGREGATE,
            "aggregate_profile": AggregateProfile.RUN_SELECTION_AGGREGATE,
        }
    )
    publish_global_aggregate(
        [invalid_mode],
        stable_path=layout.aggregate_result,
        expected=aggregate_expected,
        code_identity=identity,
    )
    assert layout.aggregate_result.is_file()
    assert layout.aggregate_manifest.is_file()


def test_resume_requires_the_exact_immutable_run_plan(tmp_path: Path) -> None:
    path = tmp_path / "run" / "run_plan.json"
    payload = {"version": "v1", "cells": [{"dataset_id": 1, "seed": 42}]}

    write_or_validate_run_plan(path, payload, resume=False)
    write_or_validate_run_plan(path, payload, resume=True)

    with pytest.raises(RuntimeError, match="does not match"):
        write_or_validate_run_plan(
            path,
            {"version": "v1", "cells": [{"dataset_id": 1, "seed": 43}]},
            resume=True,
        )
    with pytest.raises(FileNotFoundError, match="requires an existing run plan"):
        write_or_validate_run_plan(tmp_path / "missing.json", payload, resume=True)
