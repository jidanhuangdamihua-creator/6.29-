"""Operator-side Gate 1X publication gate.

The operator validates readiness and proof independently.  It never trains or
predicts.  The normal implementation phase uses ``--dry-run`` only; actual
publication remains a later authorized stage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from src.protocols.gate1_transformation import (
    CONTRACT_DIGEST,
    COMBINED_FORMAL_IDENTITY_DIGEST,
    FormalPreflight,
    Gate1Failure,
    ProofWriter,
    dataset_contract,
    load_formal_identity,
    validate_proof_digest,
)


class MaterializationError(RuntimeError):
    pass


@dataclass
class MaterializationConfig:
    project_root: Path | None = None
    parent_root: Path | None = None
    old_sealed_root: Path | None = None
    old_root: Path | None = None
    deployment_root: Path | None = None
    private_build_root: Path | None = None
    report_output: Path | None = None
    readiness_report: Path | None = None
    dry_run: bool = False
    run_id: str = "gate1x-implementation"

    @property
    def root(self) -> Path:
        return Path(self.project_root or Path(__file__).resolve().parents[2]).resolve()

    @property
    def parent(self) -> Path:
        return Path(self.parent_root or self.root).resolve()

    @property
    def old(self) -> Path:
        return Path(self.old_sealed_root or self.old_root or self.root).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_set_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps([dict(entry) for entry in entries], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inventory_tree(root: Path) -> dict[str, object]:
    root = Path(root)
    files = []
    if root.exists():
        for path in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
            files.append({"path": path.relative_to(root).as_posix(), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)})
    return {"root": str(root), "files": files, "content_set_digest": content_set_digest(files)}


def validate_gate1_publication_proof(proof: Mapping[str, Any]) -> dict[str, object]:
    try:
        validate_proof_digest(proof)
    except Gate1Failure as exc:
        return {"status": "failed", "failure_code": exc.code, "error": str(exc)}
    result = FormalPreflight().check({"proof": proof})
    if result.get("status") != "passed":
        return result
    if proof.get("contract_digest") != CONTRACT_DIGEST or proof.get("formal_identity", {}).get("combined_formal_identity_digest") != COMBINED_FORMAL_IDENTITY_DIGEST:
        return {"status": "failed", "failure_code": "FORMAL_IDENTITY"}
    return {"status": "passed", "failure_code": None, "proof_digest": proof["proof_digest"]}


def _load_readiness(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MaterializationError("READINESS_NOT_PASSED: readiness report is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise MaterializationError("READINESS_NOT_PASSED: readiness report did not pass")
    required = {"read_only": True, "writes_performed": False, "producer_calls_performed": 0, "private_build_created": False, "deployment_created": False}
    for field, expected in required.items():
        if report.get(field) != expected:
            raise MaterializationError(f"READINESS_IDENTITY: {field} is not {expected!r}")
    if len(report.get("datasets", [])) != 6 or any(item.get("status") != "passed" for item in report["datasets"]):
        raise MaterializationError("READINESS_NOT_PASSED: all D1-D6 datasets must pass")
    identity = report.get("formal_identity", {})
    if identity.get("combined_formal_identity_digest") != COMBINED_FORMAL_IDENTITY_DIGEST:
        raise MaterializationError("READINESS_IDENTITY: formal identity mismatch")
    report["report_sha256"] = _sha256(path)
    return report


def _artifact_paths(root: Path, dataset: int) -> tuple[Path, Path]:
    base = root / "数据集" / "固化数据"
    return base / f"dataset{dataset}-source.parquet", base / f"dataset{dataset}-target.parquet"


def validate_pass_through_dataset(old_root: Path, build_root: Path, dataset_id: int) -> dict[str, object]:
    old_source, old_target = _artifact_paths(Path(old_root), dataset_id)
    new_source, new_target = _artifact_paths(Path(build_root), dataset_id)
    if not old_source.is_file() or not old_target.is_file():
        raise MaterializationError(f"AUTHORITY_MISSING: D{dataset_id} old sealed parquet missing")
    if not new_source.is_file() or not new_target.is_file():
        raise MaterializationError(f"ARTIFACT_MISSING: D{dataset_id} build parquet missing")
    if _sha256(old_source) != _sha256(new_source) or _sha256(old_target) != _sha256(new_target):
        raise MaterializationError(f"ARTIFACT_HASH: D{dataset_id} pass-through bytes changed")
    return {"dataset": f"D{dataset_id}", "status": "passed", "source_sha256": _sha256(new_source), "target_sha256": _sha256(new_target)}


def planned_operation(config: MaterializationConfig) -> dict[str, object]:
    identity = load_formal_identity(config.root)
    return {"status": "planned", "dry_run": bool(config.dry_run), "formal_identity": identity, "project_root": str(config.root), "parent_root": str(config.parent), "old_root": str(config.old), "readiness_report": None if config.readiness_report is None else str(config.readiness_report), "writes_performed": False, "producer_calls_performed": 0, "private_build_created": False, "deployment_created": False}


def materialize(config: MaterializationConfig) -> dict[str, object]:
    config = config if isinstance(config, MaterializationConfig) else MaterializationConfig(**dict(config))
    identity = load_formal_identity(config.root)
    if config.dry_run:
        return {"status": "dry_run", "formal_identity": identity, "writes_performed": False, "producer_calls_performed": 0, "private_build_created": False, "deployment_created": False}
    if config.readiness_report is None:
        raise MaterializationError("READINESS_NOT_PASSED: --readiness-report is required")
    readiness = _load_readiness(Path(config.readiness_report))
    destination = Path(config.deployment_root or (config.root / "deployments"))
    private = Path(config.private_build_root or (destination / f".private-build-{config.run_id}"))
    final = destination / "d1-d6-sealed-v1"
    if private.exists() or final.exists():
        raise MaterializationError("PUBLICATION_ROOT_EXISTS: refuse reuse or overwrite")
    private.mkdir(parents=True, exist_ok=False)
    try:
        for dataset in range(1, 7):
            dataset_dir = private / f"dataset{dataset}"
            dataset_dir.mkdir()
            for path in _artifact_paths(config.parent, dataset):
                if not path.is_file():
                    raise MaterializationError(f"AUTHORITY_MISSING: {path}")
                shutil.copy2(path, dataset_dir / path.name)
        manifest = {"formal_identity": identity, "readiness": {"status": readiness["status"], "report_sha256": readiness["report_sha256"]}, "datasets": [f"D{i}" for i in range(1, 7)], "ownership": {"status": "NON_AUTHORITATIVE", "run_id": config.run_id}}
        (private / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        private.rename(final)
        result = {"status": "published", "formal_identity": identity, "readiness": readiness, "writes_performed": True, "producer_calls_performed": 0, "private_build_created": True, "deployment_created": True, "final_root": str(final)}
    except Exception:
        shutil.rmtree(private, ignore_errors=True)
        raise
    if config.report_output:
        Path(config.report_output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 1X sealed authority operator")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--parent-root", type=Path, default=None)
    parser.add_argument("--old-sealed-root", type=Path, default=None)
    parser.add_argument("--deployment-root", type=Path, default=None)
    parser.add_argument("--private-build-root", type=Path, default=None)
    parser.add_argument("--readiness-report", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = MaterializationConfig(project_root=args.root, parent_root=args.parent_root, old_sealed_root=args.old_sealed_root, deployment_root=args.deployment_root, private_build_root=args.private_build_root, readiness_report=args.readiness_report, report_output=args.report_output, dry_run=args.dry_run)
    try:
        result = materialize(config)
    except (Gate1Failure, MaterializationError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
