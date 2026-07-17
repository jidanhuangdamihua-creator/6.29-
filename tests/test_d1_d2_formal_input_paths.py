from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import run_unified_d1_d6 as unified


ROOT = Path(__file__).resolve().parents[1]


def _full_runner_solidified_paths() -> dict[str, dict[str, str]]:
    runner_path = ROOT / "scripts" / "run_full_paper_experiments.py"
    module = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "SOLIDIFIED_DATASET_PATHS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("SOLIDIFIED_DATASET_PATHS assignment not found")


def _captured_formal_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    captured: list[Path] = []

    def capture(_root: Path, paths: list[Path] | tuple[Path, ...]) -> dict:
        captured.extend(Path(path) for path in paths)
        return {}

    monkeypatch.setattr(unified, "discover_input_identity", capture)
    unified.discover_formal_input_identity(tmp_path)
    return captured


def test_d1_d2_formal_runner_uses_protocol_derived_parquets() -> None:
    expected_root = Path("数据集/固化数据")
    solidified_paths = _full_runner_solidified_paths()

    for dataset_name, dataset_id in (("Dataset1", 1), ("Dataset2", 2)):
        paths = solidified_paths[dataset_name]
        assert Path(paths["source"]) == (
            expected_root / f"dataset{dataset_id}-source.parquet"
        )
        assert Path(paths["target"]) == (
            expected_root / f"dataset{dataset_id}-target.parquet"
        )


def test_run_plan_identity_locks_protocol_derived_d1_d2_parquets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _captured_formal_input_paths(tmp_path, monkeypatch)
    expected_root = tmp_path / "数据集" / "派生数据" / "d1d2_protocol_v1"
    stale_root = tmp_path / "数据集" / "固化数据"

    for dataset_id in (1, 2):
        assert expected_root / f"dataset{dataset_id}-source.parquet" in captured
        assert expected_root / f"dataset{dataset_id}-target.parquet" in captured
        assert stale_root / f"dataset{dataset_id}-source.parquet" not in captured
        assert stale_root / f"dataset{dataset_id}-target.parquet" not in captured


def test_run_plan_identity_keeps_d3_d6_on_solidified_parquets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _captured_formal_input_paths(tmp_path, monkeypatch)
    expected_root = tmp_path / "数据集" / "固化数据"

    for dataset_id in range(3, 7):
        assert expected_root / f"dataset{dataset_id}-source.parquet" in captured
        assert expected_root / f"dataset{dataset_id}-target.parquet" in captured
