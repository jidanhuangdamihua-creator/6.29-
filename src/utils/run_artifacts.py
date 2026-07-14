"""Atomic, acceptance-gated publication of formal result artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from src.utils.result_acceptance import (
    AcceptanceOutcome,
    ExpectedResultContract,
    ResultAcceptanceError,
    accept_cell_csv,
    accept_global_aggregate,
    accept_mode_matrix,
    build_formal_cell_contract,
)


@dataclass(frozen=True)
class CodeIdentity:
    git_commit: str
    dirty: bool
    worktree_digest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_code_identity(project_root: Path) -> CodeIdentity:
    root = Path(project_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    identity_payload = bytearray(status.encode("utf-8"))
    identity_payload.extend(b"\0TRACKED\0")
    identity_payload.extend(tracked_diff)
    for raw_path in sorted(value for value in untracked_output.split(b"\0") if value):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        identity_payload.extend(b"\0UNTRACKED\0")
        identity_payload.extend(raw_path)
        identity_payload.extend(b"\0")
        identity_payload.extend(hashlib.sha256(candidate.read_bytes()).digest())
    return CodeIdentity(
        git_commit=commit,
        dirty=bool(status.strip()),
        worktree_digest=hashlib.sha256(identity_payload).hexdigest(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_input_identity(
    project_root: Path,
    paths: list[Path] | tuple[Path, ...],
) -> dict[str, dict[str, object]]:
    root = Path(project_root).resolve()
    identity: dict[str, dict[str, object]] = {}
    for raw_path in sorted({Path(path) for path in paths}, key=lambda value: str(value)):
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"formal input file missing: {path}")
        try:
            label = str(path.resolve().relative_to(root))
        except ValueError:
            label = str(path.resolve())
        identity[label] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return identity


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_sidecar(path: Path, suffix: str) -> Path:
    return Path(path).with_suffix(suffix)


def write_or_validate_run_plan(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    destination = Path(path)
    if destination.exists():
        if not resume:
            raise FileExistsError(f"run plan already exists: {destination}")
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"run plan is unreadable: {destination}") from exc
        if existing != dict(payload):
            raise RuntimeError("resume run plan does not match current formal plan")
        return
    if resume:
        raise FileNotFoundError(f"resume requires an existing run plan: {destination}")
    _atomic_json(destination, payload)


def _require_matching_artifact_manifest(
    artifact_path: Path,
    *,
    artifact_type: str,
    code_identity: CodeIdentity,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    manifest_path = artifact.with_suffix(".manifest.json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise ResultAcceptanceError(f"accepted artifact or manifest missing: {artifact}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ResultAcceptanceError(f"artifact manifest unreadable: {manifest_path}") from exc
    if manifest.get("artifact_type") != artifact_type:
        raise ResultAcceptanceError(f"artifact manifest type mismatch: {manifest_path}")
    if manifest.get("code_identity") != code_identity.to_dict():
        raise ResultAcceptanceError(f"artifact code identity mismatch: {artifact}")
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
        raise ResultAcceptanceError(f"artifact manifest is not accepted: {artifact}")
    if manifest.get("sha256") != sha256_file(artifact):
        raise ResultAcceptanceError(f"artifact hash mismatch: {artifact}")
    return manifest


def publish_formal_cell_frame(
    frame: pd.DataFrame,
    *,
    stable_path: Path,
    expected: ExpectedResultContract,
    code_identity: CodeIdentity,
) -> dict[str, object]:
    """Write a candidate, accept it, hash it, and only then expose it."""
    destination = Path(stable_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid4().hex}")
    report_path = _artifact_sidecar(destination, ".acceptance.json")
    manifest_path = _artifact_sidecar(destination, ".manifest.json")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())

        outcome = accept_cell_csv(temporary, expected=expected)
        _atomic_json(report_path, outcome.report.to_dict())
        if not outcome.report.passed:
            raise ResultAcceptanceError(
                "cell acceptance failed: " + ",".join(outcome.report.reasons)
            )

        candidate_sha = sha256_file(temporary)
        candidate_size = temporary.stat().st_size
        os.replace(temporary, destination)
        if sha256_file(destination) != candidate_sha:
            raise RuntimeError("published result hash differs from accepted candidate")

        manifest: dict[str, object] = {
            "artifact_type": "formal_cell",
            "path": str(destination),
            "sha256": candidate_sha,
            "bytes": candidate_size,
            "rows": int(len(outcome.accepted_rows)),
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "acceptance": outcome.report.to_dict(),
            "code_identity": code_identity.to_dict(),
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def publish_formal_cell_output_frame(
    frame: pd.DataFrame,
    *,
    stable_path: Path,
    dataset_id: int,
    mode: str,
    targets: tuple[str, ...] | list[str],
    horizon: int,
    seed: int,
    project_root: Path,
) -> dict[str, object]:
    identity = discover_code_identity(project_root)
    if identity.dirty:
        raise ResultAcceptanceError(
            "formal cell publication requires a clean git worktree"
        )
    expected = build_formal_cell_contract(
        dataset_id=dataset_id,
        mode=mode,
        targets=targets,
        horizon=horizon,
        seed=seed,
    )
    return publish_formal_cell_frame(
        frame,
        stable_path=stable_path,
        expected=expected,
        code_identity=identity,
    )


def _publish_accepted_rows(
    outcome: AcceptanceOutcome,
    *,
    stable_path: Path,
    code_identity: CodeIdentity,
    artifact_type: str,
    candidate_validator,
) -> dict[str, object]:
    destination = Path(stable_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid4().hex}")
    report_path = _artifact_sidecar(destination, ".acceptance.json")
    manifest_path = _artifact_sidecar(destination, ".manifest.json")
    if not outcome.report.passed:
        _atomic_json(report_path, outcome.report.to_dict())
        raise ResultAcceptanceError(
            f"{outcome.report.scope.value} acceptance failed: "
            + ",".join(outcome.report.reasons)
        )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            outcome.accepted_rows.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        final_outcome = candidate_validator(temporary)
        _atomic_json(report_path, final_outcome.report.to_dict())
        if not final_outcome.report.passed:
            raise ResultAcceptanceError(
                f"{final_outcome.report.scope.value} acceptance failed: "
                + ",".join(final_outcome.report.reasons)
            )
        candidate_sha = sha256_file(temporary)
        candidate_size = temporary.stat().st_size
        os.replace(temporary, destination)
        if sha256_file(destination) != candidate_sha:
            raise RuntimeError("published result hash differs from accepted candidate")
        manifest: dict[str, object] = {
            "artifact_type": artifact_type,
            "path": str(destination),
            "sha256": candidate_sha,
            "bytes": candidate_size,
            "rows": int(len(final_outcome.accepted_rows)),
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "acceptance": final_outcome.report.to_dict(),
            "code_identity": code_identity.to_dict(),
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def publish_mode_matrix(
    cell_paths: tuple[Path, ...] | list[Path],
    *,
    stable_path: Path,
    expected: ExpectedResultContract,
    code_identity: CodeIdentity,
) -> dict[str, object]:
    paths = tuple(Path(path) for path in cell_paths)
    for path in paths:
        _require_matching_artifact_manifest(
            path,
            artifact_type="formal_cell",
            code_identity=code_identity,
        )
    outcome = accept_mode_matrix(paths, expected=expected)
    return _publish_accepted_rows(
        outcome,
        stable_path=stable_path,
        code_identity=code_identity,
        artifact_type="formal_mode_matrix",
        candidate_validator=lambda temporary: accept_mode_matrix(
            paths,
            expected=expected,
            candidate_mode_csv=temporary,
        ),
    )


def publish_global_aggregate(
    mode_paths: tuple[Path, ...] | list[Path],
    *,
    stable_path: Path,
    expected: ExpectedResultContract,
    code_identity: CodeIdentity,
) -> dict[str, object]:
    paths = tuple(Path(path) for path in mode_paths)
    for path in paths:
        _require_matching_artifact_manifest(
            path,
            artifact_type="formal_mode_matrix",
            code_identity=code_identity,
        )
    outcome = accept_global_aggregate(paths, expected=expected)
    return _publish_accepted_rows(
        outcome,
        stable_path=stable_path,
        code_identity=code_identity,
        artifact_type="formal_global_aggregate",
        candidate_validator=lambda temporary: accept_global_aggregate(
            paths,
            expected=expected,
            candidate_aggregate_csv=temporary,
        ),
    )


def resumable_formal_cell(
    *,
    stable_path: Path,
    manifest_path: Path,
    expected: ExpectedResultContract,
    code_identity: CodeIdentity,
) -> bool:
    result = Path(stable_path)
    manifest_file = Path(manifest_path)
    if not result.is_file() or not manifest_file.is_file():
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if manifest.get("code_identity") != code_identity.to_dict():
        return False
    if manifest.get("sha256") != sha256_file(result):
        return False
    outcome = accept_cell_csv(result, expected=expected)
    return bool(outcome.report.passed)
