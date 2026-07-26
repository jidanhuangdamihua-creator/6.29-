from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_unified_d1_d6 as unified
from src.protocols import formal_input_paths as resolver_module
from src.protocols.formal_input_paths import (
    FORMAL_SEALED_ROOT_RELATIVE,
    FormalInputPathError,
    formal_input_paths,
    resolve_all_formal_dataset_paths,
    resolve_formal_dataset_paths,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_resolver_tree(root: Path) -> None:
    for dataset_id in range(1, 7):
        directory = root / FORMAL_SEALED_ROOT_RELATIVE / f"dataset{dataset_id}"
        directory.mkdir(parents=True, exist_ok=True)
        for filename in (
            "source.parquet",
            "target.parquet",
            "manifest.json",
            "source_schema.json",
            "target_schema.json",
        ):
            (directory / filename).write_bytes(b"resolver-fixture")


def _assert_error(code: str, callback) -> None:
    with pytest.raises(FormalInputPathError) as captured:
        callback()
    assert captured.value.code == code


@pytest.mark.parametrize("dataset_id", range(1, 7))
def test_exact_d1_d6_path_matrix(tmp_path: Path, dataset_id: int) -> None:
    _write_resolver_tree(tmp_path)
    resolved = resolve_formal_dataset_paths(dataset_id, repository_root=tmp_path)
    sealed_root = (tmp_path / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True)
    dataset_root = sealed_root / f"dataset{dataset_id}"

    assert resolved.dataset_id == dataset_id
    assert resolved.sealed_root == sealed_root
    assert resolved.dataset_root == dataset_root
    assert str(resolved.source_path) == str(dataset_root / "source.parquet")
    assert str(resolved.target_path) == str(dataset_root / "target.parquet")
    assert resolved.dataset_manifest_path == dataset_root / "manifest.json"
    assert resolved.source_schema_path == dataset_root / "source_schema.json"
    assert resolved.target_schema_path == dataset_root / "target_schema.json"
    assert formal_input_paths(tmp_path, dataset_id) == {
        "source": resolved.source_path,
        "target": resolved.target_path,
    }


def test_resolver_rejects_invalid_dataset_ids(tmp_path: Path) -> None:
    _write_resolver_tree(tmp_path)
    for dataset_id in (0, 7, True, "1.0", "D1"):
        _assert_error(
            "FORMAL_DATASET_ID_INVALID",
            lambda dataset_id=dataset_id: resolve_formal_dataset_paths(
                dataset_id,
                repository_root=tmp_path,
            ),
        )


@pytest.mark.parametrize(
    ("filename", "code"),
    (
        ("source.parquet", "FORMAL_SOURCE_MISSING"),
        ("target.parquet", "FORMAL_TARGET_MISSING"),
        ("manifest.json", "FORMAL_MANIFEST_MISSING"),
        ("source_schema.json", "FORMAL_SCHEMA_MISSING"),
        ("target_schema.json", "FORMAL_SCHEMA_MISSING"),
    ),
)
def test_required_artifact_missing_fails_closed(
    tmp_path: Path,
    filename: str,
    code: str,
) -> None:
    _write_resolver_tree(tmp_path)
    (tmp_path / FORMAL_SEALED_ROOT_RELATIVE / "dataset1" / filename).unlink()
    _assert_error(
        code,
        lambda: resolve_formal_dataset_paths(1, repository_root=tmp_path),
    )


@pytest.mark.parametrize("dataset_id", range(1, 7))
def test_missing_sealed_file_never_falls_back_to_legacy_files(
    tmp_path: Path,
    dataset_id: int,
) -> None:
    _write_resolver_tree(tmp_path)
    legacy_root = tmp_path / "数据集/固化数据"
    (legacy_root / f"dataset{dataset_id}-source.parquet").write_bytes(b"legacy")
    (legacy_root / f"dataset{dataset_id}-target.parquet").write_bytes(b"legacy")
    derived_root = tmp_path / "数据集/派生数据/d1d2_protocol_v1"
    derived_root.mkdir(parents=True, exist_ok=True)
    (derived_root / f"dataset{dataset_id}-source.parquet").write_bytes(b"derived")
    (derived_root / f"dataset{dataset_id}-target.parquet").write_bytes(b"derived")

    sealed_source = (
        tmp_path
        / FORMAL_SEALED_ROOT_RELATIVE
        / f"dataset{dataset_id}"
        / "source.parquet"
    )
    sealed_source.unlink()
    _assert_error(
        "FORMAL_SOURCE_MISSING",
        lambda: resolve_formal_dataset_paths(
            dataset_id,
            repository_root=tmp_path,
        ),
    )


def test_d1_resolves_without_old_derived_tree(tmp_path: Path) -> None:
    _write_resolver_tree(tmp_path)
    resolved = resolve_formal_dataset_paths(1, repository_root=tmp_path)
    assert resolved.source_path.name == "source.parquet"
    assert resolved.target_path.name == "target.parquet"
    assert "d1d2_protocol_v1" not in str(resolved.source_path)


def test_symlink_escape_fails_closed(tmp_path: Path) -> None:
    _write_resolver_tree(tmp_path)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside")
    source = tmp_path / FORMAL_SEALED_ROOT_RELATIVE / "dataset1/source.parquet"
    source.unlink()
    source.symlink_to(outside)
    _assert_error(
        "FORMAL_PATH_OUTSIDE_SEALED_ROOT",
        lambda: resolve_formal_dataset_paths(1, repository_root=tmp_path),
    )


def test_parent_traversal_in_relative_root_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_resolver_tree(tmp_path)
    monkeypatch.setattr(
        resolver_module,
        "FORMAL_SEALED_ROOT_RELATIVE",
        Path("数据集/固化数据/../escape"),
    )
    _assert_error(
        "FORMAL_PATH_OUTSIDE_SEALED_ROOT",
        lambda: resolve_formal_dataset_paths(1, repository_root=tmp_path),
    )


def test_unified_child_commands_receive_exact_resolver_paths() -> None:
    resolved = {item.dataset_id: item for item in resolve_all_formal_dataset_paths(repository_root=ROOT)}
    tasks = unified.build_tasks(
        None,
        smoke=False,
        run_dir=Path("/tmp/formal-resolver-test"),
        repository_root=ROOT,
    )
    assert len(tasks) == 300
    for task in tasks:
        source_index = task.cmd.index("--formal-source-path")
        target_index = task.cmd.index("--formal-target-path")
        assert task.cmd[source_index + 1] == str(resolved[task.dataset_id].source_path)
        assert task.cmd[target_index + 1] == str(resolved[task.dataset_id].target_path)
