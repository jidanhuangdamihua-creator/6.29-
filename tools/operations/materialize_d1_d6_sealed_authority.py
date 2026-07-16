#!/usr/bin/env python3
"""Materialize one immutable D1-D6 sealed authority as an operator action."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.prediction_artifacts import (  # noqa: E402
    _atomic_bytes,
    _fsync_directory,
    canonical_json_bytes,
)
from src.utils.run_artifacts import sha256_file  # noqa: E402


OUTPUTS_RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
SEALED_DATA_PARENT = PROJECT_ROOT / "数据集" / "固化数据"
PRODUCER = PROJECT_ROOT / "scripts" / "adopt_and_seal_d3_d6.py"

_COMMON_ARTIFACTS = (
    "source.parquet",
    "target.parquet",
    "manifest.json",
    "validation_report.json",
    "source_schema.json",
    "target_schema.json",
    "predictor_schema.json",
    "knn_schema.json",
    "calendarization_audit.json",
    "source_sales_canonicalization.json",
    "provenance.json",
)
_ADOPT_ARTIFACT = "adopt_validation_report.json"
_REPAIR_REASONS = (
    "original_nan",
    "original_negative",
    "calendar_row_missing",
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFIXED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_LOGICAL_ROLES = {
    "source.parquet": "source",
    "target.parquet": "target",
    "manifest.json": "dataset_manifest",
    "validation_report.json": "validation_report",
    "source_schema.json": "source_schema",
    "target_schema.json": "target_schema",
    "predictor_schema.json": "predictor_schema",
    "knn_schema.json": "knn_schema",
    "calendarization_audit.json": "calendarization_audit",
    "source_sales_canonicalization.json": "source_sales_canonicalization",
    "provenance.json": "provenance",
    "adopt_validation_report.json": "adopt_validation_report",
}


@dataclass(frozen=True)
class MaterializationConfig:
    old_sealed_root: Path
    parent_root: Path
    private_build_root: Path
    final_deployment_parent: Path
    report_output: Path
    manifest_candidate_output: Path


class MaterializationError(RuntimeError):
    """A stable fail-closed materialization error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        dataset: str | None = None,
    ) -> None:
        self.code = code
        self.stage = stage
        self.dataset = dataset
        super().__init__(message)


ProducerRunner = Callable[[int, Path, Path], subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_without_newline(value: Any) -> bytes:
    encoded = canonical_json_bytes(value)
    if not encoded.endswith(b"\n"):
        raise MaterializationError(
            "CANONICAL_JSON_CONTRACT",
            "canonical JSON helper did not produce its required trailing newline",
            stage="canonicalization",
        )
    return encoded[:-1]


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(parent)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _expected_relative_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for dataset_id in range(1, 7):
        names = _COMMON_ARTIFACTS + ((_ADOPT_ARTIFACT,) if dataset_id >= 3 else ())
        paths.extend(f"dataset{dataset_id}/{name}" for name in names)
    return tuple(sorted(paths))


EXPECTED_RELATIVE_PATHS = _expected_relative_paths()


def _artifact_names(dataset_id: int) -> tuple[str, ...]:
    return _COMMON_ARTIFACTS + ((_ADOPT_ARTIFACT,) if dataset_id >= 3 else ())


def _assert_leaf_not_symlink(path: Path, *, code: str, stage: str) -> None:
    candidate = Path(path)
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise MaterializationError(code, f"symlink path rejected: {candidate}", stage=stage)


def _assert_existing_directory(path: Path, *, label: str) -> None:
    _assert_leaf_not_symlink(path, code="SYMLINK_ROOT", stage="path_validation")
    if not Path(path).is_dir():
        raise MaterializationError(
            "INPUT_ROOT_MISSING",
            f"{label} must be an existing directory: {path}",
            stage="path_validation",
        )


def _assert_new_path(path: Path, *, label: str) -> None:
    _assert_leaf_not_symlink(path, code="SYMLINK_OUTPUT", stage="path_validation")
    if os.path.lexists(path):
        raise MaterializationError(
            "OUTPUT_ALREADY_EXISTS",
            f"{label} must not already exist: {path}",
            stage="path_validation",
        )


def _assert_write_path_allowed(path: Path) -> None:
    candidate = _lexical_absolute(path)
    if _is_within(candidate, OUTPUTS_RUN_ROOT):
        raise MaterializationError(
            "OUTPUTS_RUNS_PROTECTED",
            f"writes below outputs/runs are forbidden: {candidate}",
            stage="path_validation",
        )
    if _is_within(candidate, PROJECT_ROOT) and not _is_within(candidate, SEALED_DATA_PARENT):
        raise MaterializationError(
            "GIT_WORKTREE_WRITE_FORBIDDEN",
            f"operator output inside the tracked worktree is forbidden: {candidate}",
            stage="path_validation",
        )


def _validate_nonoverlap(config: MaterializationConfig) -> None:
    old = _lexical_absolute(config.old_sealed_root)
    private = _lexical_absolute(config.private_build_root)
    final_parent = _lexical_absolute(config.final_deployment_parent)
    report = _lexical_absolute(config.report_output)
    manifest = _lexical_absolute(config.manifest_candidate_output)
    if private == final_parent or _is_within(final_parent, private):
        raise MaterializationError(
            "OVERLAPPING_PATHS",
            "private build root cannot contain the final deployment parent",
            stage="path_validation",
        )
    if final_parent == old or _is_within(final_parent, old):
        raise MaterializationError(
            "OVERLAPPING_PATHS",
            "final deployment parent cannot be the old sealed root or lie below it",
            stage="path_validation",
        )
    for output in (private, report, manifest):
        if output == old or _is_within(output, old) or _is_within(old, output):
            raise MaterializationError(
                "OVERLAPPING_PATHS",
                f"output overlaps the immutable old sealed root: {output}",
                stage="path_validation",
            )
    if report == manifest:
        raise MaterializationError(
            "OVERLAPPING_PATHS",
            "report and manifest candidate outputs must differ",
            stage="path_validation",
        )
    for external in (report, manifest):
        if _is_within(external, private) or _is_within(external, final_parent):
            raise MaterializationError(
                "OVERLAPPING_PATHS",
                f"external output must be outside build/deployment roots: {external}",
                stage="path_validation",
            )


def _device_id(path: Path) -> int:
    return int(Path(path).stat().st_dev)


def _validate_paths(config: MaterializationConfig) -> None:
    _validate_nonoverlap(config)
    _assert_existing_directory(config.old_sealed_root, label="old sealed root")
    _assert_existing_directory(config.parent_root, label="parent root")
    _assert_existing_directory(config.final_deployment_parent, label="final deployment parent")
    _assert_existing_directory(config.private_build_root.parent, label="private build parent")
    _assert_existing_directory(config.report_output.parent, label="report parent")
    _assert_existing_directory(config.manifest_candidate_output.parent, label="manifest parent")
    _assert_new_path(config.private_build_root, label="private build root")
    _assert_new_path(config.report_output, label="execution report")
    _assert_new_path(config.manifest_candidate_output, label="manifest candidate")
    for output in (
        config.private_build_root,
        config.report_output,
        config.manifest_candidate_output,
    ):
        _assert_write_path_allowed(output)
    if _device_id(config.private_build_root.parent) != _device_id(config.final_deployment_parent):
        raise MaterializationError(
            "CROSS_DEVICE_PUBLICATION",
            "private build root and final deployment parent must share a filesystem",
            stage="path_validation",
        )


def _regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for candidate in sorted(Path(root).rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise MaterializationError(
                "SYMLINK_ARTIFACT",
                f"symlink artifact rejected: {candidate}",
                stage="inventory",
            )
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise MaterializationError(
                "NON_REGULAR_ARTIFACT",
                f"non-regular artifact rejected: {candidate}",
                stage="inventory",
            )
    return tuple(files)


def inventory_tree(root: Path) -> dict[str, Any]:
    candidate = Path(root)
    if not candidate.exists():
        return {"exists": False, "entries": []}
    entries = []
    for path in _regular_files(candidate):
        entries.append(
            {
                "relative_path": path.relative_to(candidate).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"exists": True, "entries": entries}


def inventory_artifacts(root: Path, *, allow_non_authoritative_marker: bool = False) -> list[dict[str, Any]]:
    tree = inventory_tree(root)
    if not tree["exists"]:
        raise MaterializationError(
            "SEALED_ROOT_MISSING", f"sealed root missing: {root}", stage="inventory"
        )
    entries = tree["entries"]
    if allow_non_authoritative_marker:
        entries = [
            entry for entry in entries if entry["relative_path"] != "NON_AUTHORITATIVE.json"
        ]
    actual = tuple(entry["relative_path"] for entry in entries)
    if actual != EXPECTED_RELATIVE_PATHS:
        missing = sorted(set(EXPECTED_RELATIVE_PATHS) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_RELATIVE_PATHS))
        raise MaterializationError(
            "ARTIFACT_SET_MISMATCH",
            f"sealed artifact set mismatch; missing={missing}, extra={extra}",
            stage="inventory",
        )
    return entries


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    if sha256_file(left) != sha256_file(right):
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            left_chunk = first.read(1024 * 1024)
            right_chunk = second.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _copy_file_fsync(source: Path, target: Path) -> None:
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _copy_d1_d2(old_root: Path, build_root: Path) -> None:
    for dataset_id in (1, 2):
        source_dir = old_root / f"dataset{dataset_id}"
        destination_dir = build_root / f"dataset{dataset_id}"
        destination_dir.mkdir(exist_ok=False)
        for name in _artifact_names(dataset_id):
            source = source_dir / name
            destination = destination_dir / name
            if source.is_symlink() or not source.is_file():
                raise MaterializationError(
                    "D1_D2_ARTIFACT_INVALID",
                    f"D{dataset_id} source artifact is missing or unsafe: {source}",
                    stage="copy_d1_d2",
                    dataset=f"D{dataset_id}",
                )
            _copy_file_fsync(source, destination)
            if not _files_equal(source, destination):
                raise MaterializationError(
                    "D1_D2_COPY_MISMATCH",
                    f"D{dataset_id} byte copy mismatch: {name}",
                    stage="copy_d1_d2",
                    dataset=f"D{dataset_id}",
                )
        _fsync_directory(destination_dir)


def _load_json(path: Path, *, dataset: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(
            "INVALID_JSON_ARTIFACT",
            f"invalid JSON artifact {path}: {exc}",
            stage="dataset_validation",
            dataset=dataset,
        ) from exc
    if not isinstance(value, dict):
        raise MaterializationError(
            "INVALID_JSON_ARTIFACT",
            f"JSON artifact must contain an object: {path}",
            stage="dataset_validation",
            dataset=dataset,
        )
    return value


def validate_repair_proof(dataset_dir: Path, dataset_id: int) -> dict[str, Any]:
    dataset = f"D{dataset_id}"
    sidecar = _load_json(dataset_dir / "source_sales_canonicalization.json", dataset=dataset)
    manifest = _load_json(dataset_dir / "manifest.json", dataset=dataset)
    report = _load_json(dataset_dir / "adopt_validation_report.json", dataset=dataset)
    manifest_proof = manifest.get("source_sales_repair")
    report_proof = report.get("source_sales_repair")
    if not (sidecar == manifest_proof == report_proof):
        raise MaterializationError(
            "REPAIR_PROOF_IDENTITY_MISMATCH",
            f"{dataset} repair proof differs across sidecar, manifest, and adoption report",
            stage="dataset_validation",
            dataset=dataset,
        )
    proof = sidecar
    if proof.get("status") in {None, "unavailable", "not_reconstructed_during_adoption"}:
        raise MaterializationError(
            "REPAIR_PROOF_INCOMPLETE",
            f"{dataset} repair proof lacks a success status",
            stage="dataset_validation",
            dataset=dataset,
        )
    counts = proof.get("repair_reason_counts")
    affected = proof.get("affected_rows")
    examined = proof.get("rows_examined")
    if not isinstance(counts, dict) or set(counts) != set(_REPAIR_REASONS):
        raise MaterializationError(
            "REPAIR_PROOF_REASON_SET",
            f"{dataset} repair reason set is not closed",
            stage="dataset_validation",
            dataset=dataset,
        )
    if (
        not isinstance(affected, list)
        or not isinstance(examined, int)
        or isinstance(examined, bool)
        or examined < 0
        or len(affected) > examined
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
        or sum(counts.values()) != len(affected)
    ):
        raise MaterializationError(
            "REPAIR_PROOF_COUNTS",
            f"{dataset} repair proof counts do not close",
            stage="dataset_validation",
            dataset=dataset,
        )
    actual = Counter(row.get("reason") for row in affected if isinstance(row, dict))
    if len(actual) > len(_REPAIR_REASONS) or any(actual[reason] != counts[reason] for reason in _REPAIR_REASONS):
        raise MaterializationError(
            "REPAIR_PROOF_ROWS",
            f"{dataset} repair rows do not match reason counts",
            stage="dataset_validation",
            dataset=dataset,
        )
    mask = proof.get("repair_mask_sha256")
    date_digest = proof.get("affected_date_digest")
    if not isinstance(mask, str) or _HEX_SHA256.fullmatch(mask) is None:
        raise MaterializationError(
            "REPAIR_MASK_DIGEST",
            f"{dataset} repair mask digest is invalid",
            stage="dataset_validation",
            dataset=dataset,
        )
    if not isinstance(date_digest, str) or _PREFIXED_SHA256.fullmatch(date_digest) is None:
        raise MaterializationError(
            "AFFECTED_DATE_DIGEST",
            f"{dataset} affected-date digest is invalid",
            stage="dataset_validation",
            dataset=dataset,
        )
    if manifest.get("source_sales_repair_mask_sha256") != mask:
        raise MaterializationError(
            "REPAIR_MASK_IDENTITY",
            f"{dataset} manifest mask identity mismatch",
            stage="dataset_validation",
            dataset=dataset,
        )
    if manifest.get("source_sales_repair_reason_counts") != counts:
        raise MaterializationError(
            "REPAIR_COUNT_IDENTITY",
            f"{dataset} manifest reason-count identity mismatch",
            stage="dataset_validation",
            dataset=dataset,
        )
    return {
        "proof_sha256": hashlib.sha256(_canonical_without_newline(proof)).hexdigest(),
        "repair_mask_sha256": mask,
        "affected_date_digest": date_digest,
        "repair_reason_counts": counts,
        "rows_examined": examined,
        "affected_row_count": len(affected),
    }


def _validate_dataset_artifacts(root: Path, dataset_id: int) -> None:
    dataset_dir = root / f"dataset{dataset_id}"
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise MaterializationError(
            "DATASET_DIRECTORY_INVALID",
            f"dataset directory missing or unsafe: {dataset_dir}",
            stage="dataset_validation",
            dataset=f"D{dataset_id}",
        )
    actual = []
    for path in _regular_files(dataset_dir):
        actual.append(path.relative_to(dataset_dir).as_posix())
    expected = sorted(_artifact_names(dataset_id))
    if actual != expected:
        raise MaterializationError(
            "DATASET_ARTIFACT_SET_MISMATCH",
            f"D{dataset_id} artifact set mismatch; expected={expected}, actual={actual}",
            stage="dataset_validation",
            dataset=f"D{dataset_id}",
        )


def _validate_source_target_identity(old_root: Path, build_root: Path, dataset_id: int) -> dict[str, Any]:
    comparison: dict[str, Any] = {"dataset": f"D{dataset_id}"}
    for role in ("source", "target"):
        name = f"{role}.parquet"
        old = old_root / f"dataset{dataset_id}" / name
        new = build_root / f"dataset{dataset_id}" / name
        if old.is_symlink() or new.is_symlink() or not old.is_file() or not new.is_file():
            raise MaterializationError(
                "SOURCE_TARGET_MISSING",
                f"D{dataset_id} {role} artifact missing or unsafe",
                stage="dataset_validation",
                dataset=f"D{dataset_id}",
            )
        identical = _files_equal(old, new)
        comparison[role] = {
            "old_size_bytes": int(old.stat().st_size),
            "new_size_bytes": int(new.stat().st_size),
            "old_sha256": sha256_file(old),
            "new_sha256": sha256_file(new),
            "bytes_identical": identical,
        }
        if not identical:
            raise MaterializationError(
                "SOURCE_TARGET_IDENTITY_DRIFT",
                f"D{dataset_id} {role} bytes differ from the old authority",
                stage="dataset_validation",
                dataset=f"D{dataset_id}",
            )
    return comparison


def content_set_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_without_newline(list(entries))).hexdigest()


def build_manifest_candidate(
    entries: Sequence[Mapping[str, Any]], digest: str, deployment_root: str
) -> dict[str, Any]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset_id in range(1, 7):
        prefix = f"dataset{dataset_id}/"
        artifacts = []
        for entry in entries:
            relative = str(entry["relative_path"])
            if not relative.startswith(prefix):
                continue
            filename = relative[len(prefix) :]
            artifacts.append(
                {
                    "logical_role": _LOGICAL_ROLES[filename],
                    "path": relative,
                    "size_bytes": int(entry["size_bytes"]),
                    "sha256": str(entry["sha256"]),
                }
            )
        datasets[f"D{dataset_id}"] = artifacts
    return {
        "manifest_version": "d1_d6_sealed_deployment_manifest_v1",
        "sealed_root_version": "d1_d6_sealed_v1",
        "deployment_root": deployment_root,
        "content_set_digest": digest,
        "datasets": datasets,
    }


def validate_manifest_candidate(
    candidate: Mapping[str, Any], root: Path, expected_entries: Sequence[Mapping[str, Any]]
) -> None:
    expected_top = {
        "manifest_version",
        "sealed_root_version",
        "deployment_root",
        "content_set_digest",
        "datasets",
    }
    if set(candidate) != expected_top:
        raise MaterializationError(
            "MANIFEST_TOP_LEVEL_FIELDS",
            "manifest candidate top-level fields differ from the frozen contract",
            stage="manifest_validation",
        )
    deployment_root = candidate.get("deployment_root")
    if (
        not isinstance(deployment_root, str)
        or not deployment_root
        or PurePosixPath(deployment_root).name != deployment_root
        or "/" in deployment_root
        or "\\" in deployment_root
    ):
        raise MaterializationError(
            "MANIFEST_DEPLOYMENT_ROOT",
            "deployment_root must be one basename",
            stage="manifest_validation",
        )
    digest = candidate.get("content_set_digest")
    if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
        raise MaterializationError(
            "MANIFEST_CONTENT_DIGEST",
            "manifest content-set digest is invalid",
            stage="manifest_validation",
        )
    if digest != content_set_digest(expected_entries):
        raise MaterializationError(
            "MANIFEST_CONTENT_DIGEST_MISMATCH",
            "manifest content-set digest does not bind the inventory",
            stage="manifest_validation",
        )
    datasets = candidate.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {f"D{i}" for i in range(1, 7)}:
        raise MaterializationError(
            "MANIFEST_DATASET_SET",
            "manifest dataset set must be exactly D1-D6",
            stage="manifest_validation",
        )
    expected_by_path = {str(entry["relative_path"]): entry for entry in expected_entries}
    seen: set[str] = set()
    artifact_fields = {"logical_role", "path", "size_bytes", "sha256"}
    for dataset_id in range(1, 7):
        artifacts = datasets[f"D{dataset_id}"]
        if not isinstance(artifacts, list):
            raise MaterializationError(
                "MANIFEST_ARTIFACT_LIST",
                f"D{dataset_id} artifacts must be a list",
                stage="manifest_validation",
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != artifact_fields:
                raise MaterializationError(
                    "MANIFEST_ARTIFACT_FIELDS",
                    "manifest artifact fields differ from the frozen contract",
                    stage="manifest_validation",
                )
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
                raise MaterializationError(
                    "MANIFEST_PATH_UNSAFE", "invalid manifest path", stage="manifest_validation"
                )
            pure = PurePosixPath(raw_path)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != raw_path:
                raise MaterializationError(
                    "MANIFEST_PATH_UNSAFE",
                    f"non-canonical manifest path rejected: {raw_path}",
                    stage="manifest_validation",
                )
            if raw_path in seen:
                raise MaterializationError(
                    "MANIFEST_DUPLICATE_PATH",
                    f"duplicate manifest path: {raw_path}",
                    stage="manifest_validation",
                )
            seen.add(raw_path)
            expected = expected_by_path.get(raw_path)
            if expected is None:
                raise MaterializationError(
                    "MANIFEST_EXTRA_PATH",
                    f"manifest path is outside the expected artifact set: {raw_path}",
                    stage="manifest_validation",
                )
            target = root.joinpath(*pure.parts)
            current = root
            for component in pure.parts:
                current = current / component
                if current.is_symlink():
                    raise MaterializationError(
                        "MANIFEST_PATH_UNSAFE",
                        f"manifest path crosses a symlink: {raw_path}",
                        stage="manifest_validation",
                    )
            if not target.is_file():
                raise MaterializationError(
                    "MANIFEST_PATH_UNSAFE",
                    f"manifest target is missing or a symlink: {raw_path}",
                    stage="manifest_validation",
                )
            try:
                target.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise MaterializationError(
                    "MANIFEST_PATH_ESCAPE",
                    f"manifest path escapes its sealed root: {raw_path}",
                    stage="manifest_validation",
                ) from exc
            if artifact["logical_role"] != _LOGICAL_ROLES[PurePosixPath(raw_path).name]:
                raise MaterializationError(
                    "MANIFEST_LOGICAL_ROLE",
                    f"logical role mismatch for {raw_path}",
                    stage="manifest_validation",
                )
            if (
                artifact["size_bytes"] != expected["size_bytes"]
                or artifact["sha256"] != expected["sha256"]
            ):
                raise MaterializationError(
                    "MANIFEST_IDENTITY_MISMATCH",
                    f"manifest identity mismatch for {raw_path}",
                    stage="manifest_validation",
                )
    if seen != set(expected_by_path):
        missing = sorted(set(expected_by_path) - seen)
        raise MaterializationError(
            "MANIFEST_MISSING_PATH",
            f"manifest candidate is missing artifacts: {missing}",
            stage="manifest_validation",
        )


def _run_producer(dataset_id: int, parent_root: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PRODUCER.relative_to(PROJECT_ROOT)),
            "--dataset",
            f"d{dataset_id}",
            "--parent-root",
            str(parent_root),
            "--output-dir",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=None,
    )


def _log_path(report_output: Path, dataset_id: int, stream: str) -> Path:
    return report_output.parent / f".{report_output.name}.d{dataset_id}.{stream}.log"


def _write_log(path: Path, text: str) -> None:
    _assert_new_path(path, label="dataset log")
    _assert_write_path_allowed(path)
    _atomic_bytes(path, text.encode("utf-8"))


def _marker_payload(status: str, *, error_code: str | None = None) -> dict[str, Any]:
    return {
        "authority_status": "non_authoritative",
        "materialization_status": status,
        "error_code": error_code,
    }


def _ensure_non_authoritative_marker(build_root: Path, *, status: str, error_code: str | None = None) -> None:
    if not build_root.is_dir():
        return
    marker = build_root / "NON_AUTHORITATIVE.json"
    marker.unlink(missing_ok=True)
    _atomic_bytes(marker, canonical_json_bytes(_marker_payload(status, error_code=error_code)))
    _fsync_directory(build_root)


def _fsync_complete_root(root: Path) -> None:
    for path in _regular_files(root):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for directory in sorted(
        (path for path in Path(root).rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _publish_external(path: Path, payload: bytes) -> None:
    _assert_new_path(path, label="external output")
    _atomic_bytes(path, payload)
    _fsync_directory(path.parent)


def _stage_external(path: Path, payload: bytes) -> Path:
    temporary = path.parent / f".{path.name}.materialization-candidate.{os.getpid()}"
    _assert_new_path(temporary, label="external output candidate")
    _assert_write_path_allowed(temporary)
    _atomic_bytes(temporary, payload)
    _fsync_directory(temporary.parent)
    return temporary


def _replace_staged_payload(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_staged(temporary: Path, destination: Path) -> None:
    _assert_new_path(destination, label="external output")
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _failed_report(
    *,
    started_at: str,
    error: BaseException,
    dataset_records: Sequence[Mapping[str, Any]],
    config: MaterializationConfig,
) -> dict[str, Any]:
    if isinstance(error, MaterializationError):
        code = error.code
        stage = error.stage
        dataset = error.dataset
    else:
        code = "UNEXPECTED_ERROR"
        stage = "unexpected"
        dataset = None
    return {
        "report_version": "d1_d6_sealed_materialization_report_v1",
        "status": "failed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "failure_stage": stage,
        "failure_dataset": dataset,
        "exception_type": type(error).__name__,
        "error_code": code,
        "error": str(error),
        "datasets": list(dataset_records),
        "private_build_root": str(_lexical_absolute(config.private_build_root)),
    }


def materialize(
    config: MaterializationConfig,
    *,
    producer_runner: ProducerRunner = _run_producer,
    outputs_run_root: Path | None = None,
) -> int:
    started_at = _utc_now()
    dataset_records: list[dict[str, Any]] = []
    build_created = False
    published_final: Path | None = None
    manifest_published = False
    staged_manifest: Path | None = None
    staged_report: Path | None = None
    try:
        _validate_paths(config)
        protected_outputs = Path(outputs_run_root) if outputs_run_root is not None else OUTPUTS_RUN_ROOT
        old_inventory = inventory_artifacts(config.old_sealed_root)
        old_tree = inventory_tree(config.old_sealed_root)
        outputs_before = inventory_tree(protected_outputs)

        config.private_build_root.mkdir(exist_ok=False)
        build_created = True
        _ensure_non_authoritative_marker(config.private_build_root, status="building")
        _copy_d1_d2(config.old_sealed_root, config.private_build_root)

        proof_identities: dict[str, Any] = {}
        comparisons = [
            _validate_source_target_identity(
                config.old_sealed_root, config.private_build_root, dataset_id
            )
            for dataset_id in (1, 2)
        ]
        for dataset_id in range(3, 7):
            dataset = f"D{dataset_id}"
            record: dict[str, Any] = {
                "dataset": dataset,
                "started_at": _utc_now(),
                "finished_at": None,
                "returncode": None,
                "status": "running",
                "stdout_log_path": str(_log_path(config.report_output, dataset_id, "stdout")),
                "stderr_log_path": str(_log_path(config.report_output, dataset_id, "stderr")),
                "error": None,
                "validation_result": None,
            }
            dataset_records.append(record)
            completed = producer_runner(dataset_id, config.parent_root, config.private_build_root)
            _write_log(Path(record["stdout_log_path"]), completed.stdout or "")
            _write_log(Path(record["stderr_log_path"]), completed.stderr or "")
            record["returncode"] = int(completed.returncode)
            record["finished_at"] = _utc_now()
            if completed.returncode != 0:
                record["status"] = "failed"
                record["error"] = f"producer returned {completed.returncode}"
                raise MaterializationError(
                    "PRODUCER_FAILED",
                    record["error"],
                    stage="producer",
                    dataset=dataset,
                )
            _validate_dataset_artifacts(config.private_build_root, dataset_id)
            comparison = _validate_source_target_identity(
                config.old_sealed_root, config.private_build_root, dataset_id
            )
            proof_identity = validate_repair_proof(
                config.private_build_root / f"dataset{dataset_id}", dataset_id
            )
            comparisons.append(comparison)
            proof_identities[dataset] = proof_identity
            record["status"] = "success"
            record["validation_result"] = {
                "artifact_set": "valid",
                "source_target_identity": "identical",
                "repair_proof": "valid",
                "proof_identity": proof_identity,
            }

        marker = config.private_build_root / "NON_AUTHORITATIVE.json"
        new_inventory = inventory_artifacts(
            config.private_build_root, allow_non_authoritative_marker=True
        )
        digest = content_set_digest(new_inventory)
        final_name = f"d1_d6_sealed_v1_deploy_{digest[:16]}"
        final_root = config.final_deployment_parent / final_name
        _assert_new_path(final_root, label="content-addressed final root")
        candidate = build_manifest_candidate(new_inventory, digest, final_name)
        validate_manifest_candidate(candidate, config.private_build_root, new_inventory)

        staged_manifest = _stage_external(
            config.manifest_candidate_output, canonical_json_bytes(candidate)
        )
        staged_report = _stage_external(
            config.report_output,
            canonical_json_bytes(
                {
                    "report_version": "d1_d6_sealed_materialization_report_v1",
                    "status": "publication_pending",
                    "started_at": started_at,
                    "content_set_digest": digest,
                    "datasets": dataset_records,
                }
            ),
        )

        if inventory_tree(config.old_sealed_root) != old_tree:
            raise MaterializationError(
                "OLD_ROOT_CHANGED",
                "old sealed root inventory changed during materialization",
                stage="prepublication_recheck",
            )
        if inventory_tree(protected_outputs) != outputs_before:
            raise MaterializationError(
                "OUTPUTS_RUNS_CHANGED",
                "outputs/runs inventory changed during materialization",
                stage="prepublication_recheck",
            )

        marker.unlink()
        _fsync_complete_root(config.private_build_root)
        _fsync_directory(config.private_build_root.parent)
        os.replace(config.private_build_root, final_root)
        published_final = final_root
        _fsync_directory(config.final_deployment_parent)

        final_inventory = inventory_artifacts(final_root)
        if final_inventory != new_inventory or content_set_digest(final_inventory) != digest:
            raise MaterializationError(
                "POSTPUBLICATION_IDENTITY_MISMATCH",
                "published root failed its content-set identity recheck",
                stage="postpublication_recheck",
            )

        finished_at = _utc_now()
        report = {
            "report_version": "d1_d6_sealed_materialization_report_v1",
            "status": "success",
            "started_at": started_at,
            "finished_at": finished_at,
            "operator_code_sha256": sha256_file(Path(__file__)),
            "gate1a_producer_commit": "c4a905cd",
            "old_sealed_root": str(_lexical_absolute(config.old_sealed_root)),
            "old_root_inventory": old_inventory,
            "final_root": str(_lexical_absolute(final_root)),
            "final_root_name": final_name,
            "content_set_digest": digest,
            "new_root_inventory": final_inventory,
            "source_target_comparisons": comparisons,
            "proof_identities": proof_identities,
            "datasets": dataset_records,
            "outputs_runs_inventory_before": outputs_before,
            "outputs_runs_inventory_after": inventory_tree(protected_outputs),
        }
        _replace_staged_payload(staged_report, canonical_json_bytes(report))
        _publish_staged(staged_manifest, config.manifest_candidate_output)
        staged_manifest = None
        manifest_published = True
        _publish_staged(staged_report, config.report_output)
        staged_report = None
        return 0
    except BaseException as error:
        for temporary in (staged_manifest, staged_report):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
                _fsync_directory(temporary.parent)
        if manifest_published:
            config.manifest_candidate_output.unlink(missing_ok=True)
            _fsync_directory(config.manifest_candidate_output.parent)
        if published_final is not None and published_final.exists():
            if config.private_build_root.exists():
                raise RuntimeError("cannot roll back published root because private path now exists") from error
            os.replace(published_final, config.private_build_root)
            _fsync_directory(config.final_deployment_parent)
            build_created = True
        if build_created:
            code = error.code if isinstance(error, MaterializationError) else "UNEXPECTED_ERROR"
            _ensure_non_authoritative_marker(
                config.private_build_root, status="failed", error_code=code
            )
        failed = _failed_report(
            started_at=started_at,
            error=error,
            dataset_records=dataset_records,
            config=config,
        )
        if not config.report_output.exists() and config.report_output.parent.is_dir():
            try:
                _publish_external(config.report_output, canonical_json_bytes(failed))
            except Exception:
                pass
        return 1


def planned_operation(config: MaterializationConfig) -> dict[str, Any]:
    _validate_nonoverlap(config)
    return {
        "operation": "materialize_d1_d6_sealed_authority",
        "mode": "dry_run",
        "producer_order": ["D3", "D4", "D5", "D6"],
        "producer_timeout": None,
        "old_sealed_root": str(_lexical_absolute(config.old_sealed_root)),
        "parent_root": str(_lexical_absolute(config.parent_root)),
        "private_build_root": str(_lexical_absolute(config.private_build_root)),
        "final_deployment_parent": str(_lexical_absolute(config.final_deployment_parent)),
        "expected_final_root": "d1_d6_sealed_v1_deploy_<content_set_digest[:16]>",
        "report_output": str(_lexical_absolute(config.report_output)),
        "manifest_candidate_output": str(_lexical_absolute(config.manifest_candidate_output)),
        "writes_performed": False,
        "data_traversal_performed": False,
        "producer_calls_performed": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sealed-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--private-build-root", type=Path, required=True)
    parser.add_argument("--final-deployment-parent", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-candidate-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = MaterializationConfig(
        old_sealed_root=args.old_sealed_root,
        parent_root=args.parent_root,
        private_build_root=args.private_build_root,
        final_deployment_parent=args.final_deployment_parent,
        report_output=args.report_output,
        manifest_candidate_output=args.manifest_candidate_output,
    )
    if args.dry_run:
        try:
            sys.stdout.buffer.write(canonical_json_bytes(planned_operation(config)))
            return 0
        except MaterializationError as error:
            sys.stderr.write(f"{error.code}: {error}\n")
            return 2
    return materialize(config)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_RELATIVE_PATHS",
    "MaterializationConfig",
    "MaterializationError",
    "build_manifest_candidate",
    "content_set_digest",
    "inventory_artifacts",
    "main",
    "materialize",
    "planned_operation",
    "validate_manifest_candidate",
    "validate_repair_proof",
]
