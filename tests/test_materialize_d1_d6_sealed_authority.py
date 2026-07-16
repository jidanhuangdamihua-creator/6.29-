from __future__ import annotations

from copy import deepcopy
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.operations import materialize_d1_d6_sealed_authority as operator


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "operations" / "materialize_d1_d6_sealed_authority.py"


def _proof() -> dict:
    return {
        "version": "source-sales-canonicalization/v1",
        "status": "canonicalized",
        "rows_examined": 2,
        "affected_rows": [],
        "repair_reason_counts": {
            "original_nan": 0,
            "original_negative": 0,
            "calendar_row_missing": 0,
        },
        "repair_mask_sha256": "a" * 64,
        "affected_date_digest": "sha256:" + "b" * 64,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _make_dataset(root: Path, dataset_id: int) -> None:
    dataset = root / f"dataset{dataset_id}"
    dataset.mkdir()
    proof = _proof()
    manifest = {
        "dataset_id": f"D{dataset_id}",
        "source_sales_repair": proof,
        "source_sales_repair_mask_sha256": proof["repair_mask_sha256"],
        "source_sales_repair_reason_counts": proof["repair_reason_counts"],
        "artifacts": {
            "source": {"path": "source.parquet"},
            "target": {"path": "target.parquet"},
        },
    }
    for name in operator._artifact_names(dataset_id):
        path = dataset / name
        if name == "source.parquet":
            path.write_bytes(f"D{dataset_id}-source-bytes".encode())
        elif name == "target.parquet":
            path.write_bytes(f"D{dataset_id}-target-bytes".encode())
        elif name == "manifest.json":
            _write_json(path, manifest)
        elif name == "source_sales_canonicalization.json":
            _write_json(path, proof)
        elif name == "adopt_validation_report.json":
            _write_json(path, {"source_sales_repair": proof})
        else:
            _write_json(path, {"dataset_id": f"D{dataset_id}", "artifact": name})


def _fixture(tmp_path: Path) -> tuple[operator.MaterializationConfig, Path, Path]:
    old = tmp_path / "old"
    parent = tmp_path / "parents"
    deploy = tmp_path / "deployments"
    reports = tmp_path / "reports"
    outputs = tmp_path / "outputs-runs"
    for path in (old, parent, deploy, reports, outputs):
        path.mkdir()
    for dataset_id in range(1, 7):
        _make_dataset(old, dataset_id)
    config = operator.MaterializationConfig(
        old_sealed_root=old,
        parent_root=parent,
        private_build_root=deploy / ".private-build",
        final_deployment_parent=deploy,
        report_output=reports / "execution-report.json",
        manifest_candidate_output=reports / "manifest-candidate.json",
    )
    return config, old, outputs


def _copying_producer(old: Path, calls: list[int], *, fail_at: int | None = None):
    def run(dataset_id: int, parent_root: Path, output_root: Path):
        calls.append(dataset_id)
        if fail_at == dataset_id:
            return subprocess.CompletedProcess([], 7, "", "fixture producer failed")
        shutil.copytree(old / f"dataset{dataset_id}", output_root / f"dataset{dataset_id}")
        return subprocess.CompletedProcess([], 0, f"sealed D{dataset_id}\n", "")

    return run


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _final_root(config: operator.MaterializationConfig) -> Path:
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    return Path(report["final_root"])


def test_cli_help_lists_only_fixed_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout
    for option in (
        "--old-sealed-root",
        "--parent-root",
        "--private-build-root",
        "--final-deployment-parent",
        "--report-output",
        "--manifest-candidate-output",
        "--dry-run",
    ):
        assert option in help_text
    for forbidden in (
        "--timeout",
        "--entity",
        "--batch",
        "--date",
        "--legacy",
        "--smoke",
        "--fallback",
    ):
        assert forbidden not in help_text


def test_success_preserves_old_root_copies_d1_d2_and_calls_each_dataset_once(
    tmp_path: Path,
) -> None:
    config, old, outputs = _fixture(tmp_path)
    before = _tree_bytes(old)
    (outputs / "sentinel.txt").write_bytes(b"unchanged-run-output")
    outputs_before = _tree_bytes(outputs)
    calls: list[int] = []

    def producer(dataset_id: int, parent_root: Path, output_root: Path):
        assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
        return _copying_producer(old, calls)(dataset_id, parent_root, output_root)

    status = operator.materialize(
        config,
        producer_runner=producer,
        outputs_run_root=outputs,
    )

    assert status == 0
    assert calls == [3, 4, 5, 6]
    assert _tree_bytes(old) == before
    assert _tree_bytes(outputs) == outputs_before
    final = _final_root(config)
    assert final.is_dir()
    assert not config.private_build_root.exists()
    for dataset_id in (1, 2):
        for name in operator._artifact_names(dataset_id):
            assert (final / f"dataset{dataset_id}" / name).read_bytes() == (
                old / f"dataset{dataset_id}" / name
            ).read_bytes()
    manifest = json.loads(config.manifest_candidate_output.read_text(encoding="utf-8"))
    assert sum(len(items) for items in manifest["datasets"].values()) == 70
    assert set(manifest) == {
        "manifest_version",
        "sealed_root_version",
        "deployment_root",
        "content_set_digest",
        "datasets",
    }
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    assert [item["dataset"] for item in report["source_target_comparisons"]] == [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    ]
    assert config.manifest_candidate_output.read_bytes().endswith(b"\n")
    assert not config.manifest_candidate_output.read_bytes().endswith(b"\n\n")


def test_failure_keeps_old_root_unchanged_and_publishes_no_final_root(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    before = _tree_bytes(old)
    calls: list[int] = []

    status = operator.materialize(
        config,
        producer_runner=_copying_producer(old, calls, fail_at=4),
        outputs_run_root=outputs,
    )

    assert status != 0
    assert calls == [3, 4]
    assert _tree_bytes(old) == before
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
    assert not config.manifest_candidate_output.exists()
    assert (config.private_build_root / "NON_AUTHORITATIVE.json").is_file()
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure_dataset"] == "D4"
    assert report["error_code"] == "PRODUCER_FAILED"


def test_source_or_target_identity_drift_blocks_publication(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def drift(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id == 3:
            (output_root / "dataset3" / "source.parquet").write_bytes(b"drift")
        return completed

    assert operator.materialize(config, producer_runner=drift, outputs_run_root=outputs) != 0
    assert calls == [3]
    assert not config.manifest_candidate_output.exists()
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
    assert json.loads(config.report_output.read_text())["error_code"] == "SOURCE_TARGET_IDENTITY_DRIFT"


@pytest.mark.parametrize("damage", ["missing", "identity"])
def test_incomplete_or_mismatched_repair_proof_blocks_publication(
    tmp_path: Path, damage: str
) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def damaged(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id == 3:
            sidecar_path = output_root / "dataset3" / "source_sales_canonicalization.json"
            sidecar = json.loads(sidecar_path.read_text())
            if damage == "missing":
                sidecar["status"] = None
                manifest_path = output_root / "dataset3" / "manifest.json"
                report_path = output_root / "dataset3" / "adopt_validation_report.json"
                manifest = json.loads(manifest_path.read_text())
                report = json.loads(report_path.read_text())
                manifest["source_sales_repair"] = sidecar
                report["source_sales_repair"] = sidecar
                _write_json(manifest_path, manifest)
                _write_json(report_path, report)
            else:
                sidecar["repair_mask_sha256"] = "c" * 64
            _write_json(sidecar_path, sidecar)
        return completed

    assert operator.materialize(config, producer_runner=damaged, outputs_run_root=outputs) != 0
    assert calls == [3]
    assert not config.manifest_candidate_output.exists()
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))


def test_cli_rejects_existing_private_or_final_root(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    config.private_build_root.mkdir()
    assert operator.materialize(config, producer_runner=_copying_producer(old, []), outputs_run_root=outputs) != 0

    second = tmp_path / "second"
    second.mkdir()
    config, old, outputs = _fixture(second)
    entries = operator.inventory_artifacts(old)
    digest = operator.content_set_digest(entries)
    (config.final_deployment_parent / f"d1_d6_sealed_v1_deploy_{digest[:16]}").mkdir()
    calls: list[int] = []
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, calls), outputs_run_root=outputs
    ) != 0
    assert calls == [3, 4, 5, 6]
    assert not config.manifest_candidate_output.exists()
    assert (config.private_build_root / "NON_AUTHORITATIVE.json").is_file()


def test_cli_rejects_cross_device_private_build_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, old, outputs = _fixture(tmp_path)
    devices = iter((1, 2))
    monkeypatch.setattr(operator, "_device_id", lambda _path: next(devices))
    calls: list[int] = []
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, calls), outputs_run_root=outputs
    ) != 0
    assert calls == []
    assert not config.private_build_root.exists()


def test_manifest_candidate_rejects_extra_missing_absolute_parent_or_symlink_paths(
    tmp_path: Path,
) -> None:
    config, old, _outputs = _fixture(tmp_path)
    entries = operator.inventory_artifacts(old)
    digest = operator.content_set_digest(entries)
    candidate = operator.build_manifest_candidate(entries, digest, f"d1_d6_sealed_v1_deploy_{digest[:16]}")
    operator.validate_manifest_candidate(candidate, old, entries)

    mutations = []
    extra = deepcopy(candidate)
    extra["datasets"]["D1"].append(
        {"logical_role": "extra", "path": "dataset1/extra", "size_bytes": 0, "sha256": "0" * 64}
    )
    mutations.append(extra)
    missing = deepcopy(candidate)
    missing["datasets"]["D1"].pop()
    mutations.append(missing)
    absolute = deepcopy(candidate)
    absolute["datasets"]["D1"][0]["path"] = "/tmp/escape"
    mutations.append(absolute)
    parent = deepcopy(candidate)
    parent["datasets"]["D1"][0]["path"] = "dataset1/../escape"
    mutations.append(parent)
    for mutation in mutations:
        with pytest.raises(operator.MaterializationError):
            operator.validate_manifest_candidate(mutation, old, entries)

    symlink_root = tmp_path / "symlink-root"
    shutil.copytree(old, symlink_root)
    victim = symlink_root / "dataset1" / "source.parquet"
    victim.unlink()
    victim.symlink_to(old / "dataset1" / "source.parquet")
    with pytest.raises(operator.MaterializationError):
        operator.validate_manifest_candidate(candidate, symlink_root, entries)


def test_content_set_digest_and_manifest_are_deterministic(tmp_path: Path) -> None:
    _config, old, _outputs = _fixture(tmp_path)
    first_entries = operator.inventory_artifacts(old)
    second_entries = operator.inventory_artifacts(old)
    first_digest = operator.content_set_digest(first_entries)
    second_digest = operator.content_set_digest(second_entries)
    assert first_digest == second_digest
    assert operator.build_manifest_candidate(first_entries, first_digest, "deploy") == (
        operator.build_manifest_candidate(second_entries, second_digest, "deploy")
    )


def test_dry_run_performs_no_io_or_producer_call(tmp_path: Path) -> None:
    base = tmp_path / "does-not-exist"
    paths = {
        "old": base / "old",
        "parents": base / "parents",
        "build": base / "build",
        "deployments": tmp_path / "deployments",
        "report": base / "report.json",
        "manifest": base / "manifest.json",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--dry-run",
            "--old-sealed-root",
            str(paths["old"]),
            "--parent-root",
            str(paths["parents"]),
            "--private-build-root",
            str(paths["build"]),
            "--final-deployment-parent",
            str(paths["deployments"]),
            "--report-output",
            str(paths["report"]),
            "--manifest-candidate-output",
            str(paths["manifest"]),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["producer_calls_performed"] == 0
    assert plan["writes_performed"] is False
    assert plan["data_traversal_performed"] is False
    assert not base.exists()
    assert not paths["deployments"].exists()


def test_dataset_is_never_split_below_dataset_level(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, calls), outputs_run_root=outputs
    ) == 0
    assert calls == [3, 4, 5, 6]
    report = json.loads(config.report_output.read_text())
    assert [record["dataset"] for record in report["datasets"]] == ["D3", "D4", "D5", "D6"]
    assert all(record["returncode"] == 0 for record in report["datasets"])
    assert all(
        not ({"entity", "batch", "date_range", "file_fragment"} & set(record))
        for record in report["datasets"]
    )


def test_operator_has_no_training_or_formal_publication_calls() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert not (
        {
            "fit",
            "fit_transform",
            "predict",
            "run_unified_d1_d6",
            "publish_run",
            "create_attempt",
        }
        & called_names
    )


def test_real_producer_command_is_one_whole_dataset_without_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    completed = operator._run_producer(5, Path("/tmp/parents"), Path("/tmp/build"))
    assert completed.returncode == 0
    assert captured["argv"] == [
        sys.executable,
        "scripts/adopt_and_seal_d3_d6.py",
        "--dataset",
        "d5",
        "--parent-root",
        "/tmp/parents",
        "--output-dir",
        "/tmp/build",
    ]
    assert captured["shell"] is False
    assert captured["timeout"] is None
    assert captured["check"] is False
