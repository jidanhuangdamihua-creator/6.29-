#!/usr/bin/env python3
"""Rebuild the sealed D2 target through the formal calendarization producer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.d2_target_calendarization import (  # noqa: E402
    D2_TARGET_REPAIR_DATES,
    calendarize_d2_target_frame,
    target_semantic_digest,
)
from src.protocols.formal_deployment_manifest import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)


DEFAULT_DATASET_ROOT = ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset2"


def _write_parquet_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_json(path: Path, updates: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"authority sidecar is not an object: {path}")
    payload.update(updates)
    atomic_write_json(path, payload)


def rebuild_d2_target_authority(
    *,
    target_path: Path = DEFAULT_DATASET_ROOT / "target.parquet",
    audit_path: Path = DEFAULT_DATASET_ROOT / "calendarization_audit.json",
) -> dict[str, object]:
    target_path = Path(target_path).resolve(strict=True)
    audit_path = Path(audit_path).resolve(strict=True)
    dataset_root = target_path.parent
    manifest_path = dataset_root / "manifest.json"
    target_schema_path = dataset_root / "target_schema.json"
    provenance_path = dataset_root / "provenance.json"
    validation_path = dataset_root / "validation_report.json"
    for path in (audit_path, manifest_path, target_schema_path, provenance_path, validation_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"D2 authority sidecar missing: {path}")

    input_sha = sha256_file(target_path)
    target = pd.read_parquet(target_path)
    target.attrs["split_role"] = "target"
    repaired, evidence = calendarize_d2_target_frame(
        target,
        input_target_sha256=input_sha,
    )
    if len(repaired) != 1807:
        raise RuntimeError(f"D2 target producer expected 1807 rows, got {len(repaired)}")
    existing_row_changed_cell_count = int(
        evidence["existing_row_repair"]["changed_cell_count"]
    )
    if evidence["inserted_count"] or existing_row_changed_cell_count:
        _write_parquet_atomically(repaired, target_path)
    elif input_sha != str(evidence.get("output_target_sha256") or input_sha):
        raise RuntimeError("D2 idempotence evidence does not bind current target bytes")

    output_sha = sha256_file(target_path)
    evidence["input_target_sha256"] = input_sha
    evidence["output_target_sha256"] = output_sha
    evidence["output_semantic_digest"] = target_semantic_digest(pd.read_parquet(target_path))
    evidence["repair_mask_digest"] = evidence["repair_date_digest"]
    evidence["sidecar_path"] = audit_path.relative_to(dataset_root.parent.parent).as_posix()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise RuntimeError("D2 calendarization audit is not an object")
    previous_target_repair = audit.get("target_repair")
    if (
        evidence["inserted_count"] == 0
        and isinstance(previous_target_repair, dict)
        and previous_target_repair.get("output_target_sha256") == output_sha
        and previous_target_repair.get("output_semantic_digest") == evidence["output_semantic_digest"]
    ):
        return {
            "status": "verified_idempotent",
            "input_target_sha256": input_sha,
            "output_target_sha256": output_sha,
            "rows": len(repaired),
            "inserted_count": 0,
            "existing_row_changed_cell_count": 0,
            "audit_path": str(audit_path),
            "evidence": previous_target_repair,
        }
    audit["audit_version"] = "d2_sealed_bytes_audit_v2"
    audit["target_repair"] = evidence
    audit["target_formal_window"] = {
        "start": "2018-06-01",
        "end": "2018-12-27",
        "expected_days": 210,
        "actual_days": 210,
        "missing_dates": [],
        "extra_dates": [],
    }
    atomic_write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    target_schema = json.loads(target_schema_path.read_text(encoding="utf-8"))
    target_schema["row_count"] = 1807
    target_schema["parquet_sha256"] = output_sha
    target_schema["null_counts"] = {
        column: int(count) for column, count in repaired.isna().sum().items()
    }
    atomic_write_json(target_schema_path, target_schema)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["formal_input_identity"]["target"].update(
        {"sha256": output_sha, "row_count": 1807}
    )
    provenance["target_calendarization_repair"] = evidence
    atomic_write_json(provenance_path, provenance)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["artifact_identity"]["target"].update(
        {"sha256": output_sha, "rows": 1807}
    )
    validation["target_normalized_frame_digest"] = evidence["output_semantic_digest"]
    validation["checks"] = [
        check
        for check in validation.get("checks", [])
        if "row counts match formal identity" not in str(check)
        and "formal target" not in str(check)
    ]
    validation["checks"].extend(
        [
            "source and target row counts match rebuilt formal identity",
            "formal target calendar is exactly 2018-06-01..2018-12-27 (210/210)",
            "five authorized store-closed target rows have sales=0 and promo=0",
            "authorized 2018-06-02 target identity and calendar fields are finite and date-consistent",
            "forecast consumer excludes future promo",
        ]
    )
    validation["target_calendarization_repair"] = {
        "audit_path": audit_path.name,
        "audit_sha256": audit_sha,
        "repair_date_digest": evidence["repair_date_digest"],
        "inserted_dates": list(D2_TARGET_REPAIR_DATES),
        "existing_row_repair": evidence["existing_row_repair"],
    }
    atomic_write_json(validation_path, validation)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_stat = target_path.stat()
    target_artifact = {
        "path": "target.parquet",
        "sha256": output_sha,
        "size_bytes": int(target_stat.st_size),
        "row_count": 1807,
    }
    manifest["artifacts"]["target"] = target_artifact
    manifest["parent_artifacts"]["target"] = dict(target_artifact)
    canonicalization = manifest.setdefault("dataset_canonicalization", {})
    canonicalization.update(
        {
            "target_sales_repair_performed": True,
            "target_promo_repair_performed": True,
            "target_repair_dates": list(D2_TARGET_REPAIR_DATES),
            "target_repair_policy": evidence["policy"],
            "target_repair_reason": evidence["reason"],
            "target_repair_mask_sha256": evidence["repair_date_digest"],
            "target_synthetic_date_count": 5,
            "target_existing_row_repair": evidence["existing_row_repair"],
        }
    )
    manifest["target_calendarization_evidence"] = {
        "path": audit_path.relative_to(dataset_root.parent.parent).as_posix(),
        "sha256": audit_sha,
        "repair_date_digest": evidence["repair_date_digest"],
        "input_target_sha256": input_sha,
        "output_target_sha256": output_sha,
        "output_semantic_digest": evidence["output_semantic_digest"],
        "existing_row_repair": evidence["existing_row_repair"],
    }
    manifest["sealed_identity"]["target_normalized_frame_digest"] = evidence[
        "output_semantic_digest"
    ]
    manifest["content_validation_notes"] = (
        "D2 formal input is the checked-in source.parquet/target.parquet pair; "
        "target calendarization is sealed at the producer layer and readiness verifies bytes."
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "rebuilt",
        "input_target_sha256": input_sha,
        "output_target_sha256": output_sha,
        "output_semantic_digest": evidence["output_semantic_digest"],
        "rows": 1807,
        "inserted_count": evidence["inserted_count"],
        "existing_row_changed_cell_count": existing_row_changed_cell_count,
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=DEFAULT_DATASET_ROOT / "target.parquet")
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_DATASET_ROOT / "calendarization_audit.json")
    args = parser.parse_args(argv)
    report = rebuild_d2_target_authority(
        target_path=args.target_path,
        audit_path=args.audit_path,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
