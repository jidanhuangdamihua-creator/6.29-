#!/usr/bin/env python3
"""Repair and rebuild the sealed D2 source calendar-field authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.d2_source_calendarization import (  # noqa: E402
    D2_FROZEN_SOURCE_CANDIDATE_KEYS,
    D2_SOURCE_CALENDARIZATION_RULE_VERSION,
    repair_d2_source_calendar_fields,
    slice_d2_source_frame,
    verify_d2_source_frame,
)
from src.protocols.formal_deployment_manifest import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    sha256_bytes,
)


DEFAULT_DATASET_ROOT = ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset2"
CALENDAR_FIELDS = ("year", "month", "week", "day")


def _write_parquet_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"D2 authority sidecar is not an object: {path}")
    return payload


def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(frame[column].isna().sum()) for column in frame.columns}


def _schema_digest(frame: pd.DataFrame) -> str:
    payload = {
        "column_order": list(frame.columns),
        "columns": [
            {
                "name": str(column),
                "pandas_dtype": str(frame[column].dtype),
                "null_count": int(frame[column].isna().sum()),
            }
            for column in frame.columns
        ],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _source_calendar_identity(
    source: pd.DataFrame,
) -> tuple[dict[str, object], object]:
    source = source.copy()
    source.attrs["split_role"] = "source"
    verified, report = verify_d2_source_frame(
        slice_d2_source_frame(source),
        candidate_keys=D2_FROZEN_SOURCE_CANDIDATE_KEYS,
    )
    del verified
    return {
        "source_authority_digest": report.source_authority_digest,
        "consumer_frame_fingerprint": report.consumer_frame_fingerprint,
        "candidate_count": len(D2_FROZEN_SOURCE_CANDIDATE_KEYS),
        "candidate_day_count": 180,
        "candidate_rows": 180 * len(D2_FROZEN_SOURCE_CANDIDATE_KEYS),
    }, report


def rebuild_d2_source_authority(
    *,
    source_path: Path = DEFAULT_DATASET_ROOT / "source.parquet",
    audit_path: Path = DEFAULT_DATASET_ROOT / "calendarization_audit.json",
) -> dict[str, object]:
    source_path = Path(source_path).resolve(strict=True)
    dataset_root = source_path.parent
    audit_path = Path(audit_path).resolve(strict=True)
    sidecars = {
        "audit": audit_path,
        "manifest": dataset_root / "manifest.json",
        "source_schema": dataset_root / "source_schema.json",
        "provenance": dataset_root / "provenance.json",
        "validation": dataset_root / "validation_report.json",
        "source_sales": dataset_root / "source_sales_canonicalization.json",
    }
    for name, path in sidecars.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"D2 {name} sidecar missing: {path}")

    input_sha = sha256_file(source_path)
    source = pd.read_parquet(source_path)
    source.attrs["split_role"] = "source"
    repaired, repair_evidence = repair_d2_source_calendar_fields(source)
    if len(repaired) != len(source):
        raise RuntimeError("D2 source calendar-field repair changed row count")
    for column in ("date", "brand_id", "item_id", "entity_id", "sales", "promo"):
        if not source[column].equals(repaired[column]):
            raise RuntimeError(f"D2 source calendar-field repair changed protected column: {column}")
    if repair_evidence["changed_cell_count"]:
        _write_parquet_atomically(repaired, source_path)

    output_source = pd.read_parquet(source_path)
    output_source.attrs["split_role"] = "source"
    output_sha = sha256_file(source_path)
    identity, calendar_report = _source_calendar_identity(output_source)
    null_counts = _null_counts(output_source)
    schema_digest = _schema_digest(output_source)
    if any(null_counts.get(field, 0) for field in CALENDAR_FIELDS):
        raise RuntimeError(f"D2 source calendar fields remain nullable: {null_counts}")

    evidence: dict[str, object] = {
        **repair_evidence,
        "artifact": "source",
        "dataset": "Dataset2",
        "input_source_sha256": input_sha,
        "output_source_sha256": output_sha,
        "output_source_size_bytes": int(source_path.stat().st_size),
        "source_authority_digest": identity["source_authority_digest"],
        "consumer_frame_fingerprint": identity["consumer_frame_fingerprint"],
        "null_counts_after": null_counts,
        "producer_identity": {
            "module": "src.protocols.d2_source_calendarization",
            "function": "repair_d2_source_calendar_fields",
            "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        },
        "policy": "recompute_date_fields_on_approved_source_dates",
        "reason": "repair_nonfinite_date_derived_fields",
        "calendar_fields": list(CALENDAR_FIELDS),
    }

    audit = _read_object(audit_path)
    previous = audit.get("source_calendar_field_repair")
    current_schema = _read_object(sidecars["source_schema"])
    current_manifest = _read_object(sidecars["manifest"])
    sidecars_coherent = (
        current_schema.get("parquet_sha256") == output_sha
        and current_schema.get("schema_digest") == schema_digest
        and current_manifest.get("artifacts", {}).get("source", {}).get("sha256") == output_sha
        and current_manifest.get("schema_fingerprints", {}).get("source") == schema_digest
    )
    if (
        int(repair_evidence["changed_cell_count"]) == 0
        and isinstance(previous, dict)
        and previous.get("output_source_sha256") == output_sha
        and previous.get("source_authority_digest") == evidence["source_authority_digest"]
        and sidecars_coherent
    ):
        return {
            "status": "verified_idempotent",
            "input_source_sha256": input_sha,
            "output_source_sha256": output_sha,
            "rows": len(output_source),
            "changed_cell_count": 0,
            "source_authority_digest": evidence["source_authority_digest"],
            "consumer_frame_fingerprint": evidence["consumer_frame_fingerprint"],
        }

    audit["audit_version"] = "d2_sealed_bytes_audit_v3"
    audit["source"] = {
        **dict(audit.get("source", {})),
        "path": "source.parquet",
        "row_count": int(len(output_source)),
        "sha256": output_sha,
        "size_bytes": int(source_path.stat().st_size),
        **identity,
        "synthetic_date_count": 0,
        "verified_zero_sales_rows": {
            date_text: int(
                (
                    output_source.loc[
                        output_source["date"].eq(pd.Timestamp(date_text)), "sales"
                    ]
                    == 0
                ).sum()
            )
            for date_text in repair_evidence["approved_dates"]
        },
    }
    audit["source_calendar_field_repair"] = evidence
    atomic_write_json(audit_path, audit)

    source_schema = _read_object(sidecars["source_schema"])
    source_schema["parquet_sha256"] = output_sha
    source_schema["row_count"] = int(len(output_source))
    source_schema["schema_digest"] = schema_digest
    source_schema["null_counts"] = null_counts
    source_schema["calendar_field_repair"] = evidence
    atomic_write_json(sidecars["source_schema"], source_schema)

    provenance = _read_object(sidecars["provenance"])
    provenance.setdefault("formal_input_identity", {}).setdefault("source", {}).update(
        {"sha256": output_sha, "row_count": int(len(output_source))}
    )
    provenance["source_authority_digest"] = identity["source_authority_digest"]
    provenance["consumer_frame_fingerprint"] = identity["consumer_frame_fingerprint"]
    provenance["source_calendar_field_repair"] = evidence
    atomic_write_json(sidecars["provenance"], provenance)

    validation = _read_object(sidecars["validation"])
    validation.setdefault("artifact_identity", {}).setdefault("source", {}).update(
        {"sha256": output_sha, "rows": int(len(output_source))}
    )
    validation["source_authority_digest"] = identity["source_authority_digest"]
    validation["consumer_frame_fingerprint"] = identity["consumer_frame_fingerprint"]
    checks = [str(check) for check in validation.get("checks", [])]
    checks.extend(
        [
            "approved D2 source dates have date-consistent finite year/month/week/day fields",
            "D2 source calendar fields are finite for all formal source rows",
            "D2 source sales and promo are unchanged by calendar-field repair",
        ]
    )
    validation["checks"] = list(dict.fromkeys(checks))
    validation["source_calendar_field_repair"] = evidence
    atomic_write_json(sidecars["validation"], validation)

    source_sales = _read_object(sidecars["source_sales"])
    source_sales["source_sha256"] = output_sha
    source_sales["calendar_field_repair"] = {
        "status": "date_fields_only",
        "repair_dates": list(repair_evidence["approved_dates"]),
        "changed_cell_count": int(repair_evidence["changed_cell_count"]),
        "sales_unchanged": True,
        "promo_unchanged": True,
    }
    atomic_write_json(sidecars["source_sales"], source_sales)

    manifest = _read_object(sidecars["manifest"])
    source_artifact = {
        "path": "source.parquet",
        "sha256": output_sha,
        "size_bytes": int(source_path.stat().st_size),
        "row_count": int(len(output_source)),
    }
    manifest.setdefault("artifacts", {})["source"] = source_artifact
    manifest.setdefault("parent_artifacts", {})["source"] = dict(source_artifact)
    manifest.setdefault("schema_fingerprints", {})["source"] = schema_digest
    canonicalization = manifest.setdefault("dataset_canonicalization", {})
    canonicalization.update(
        {
            "source_calendar_field_repair_performed": True,
            "source_calendar_field_repair_dates": list(repair_evidence["approved_dates"]),
            "source_calendar_field_repair_policy": evidence["policy"],
            "source_calendar_field_repair_reason": evidence["reason"],
            "source_calendar_field_changed_cell_count": int(repair_evidence["changed_cell_count"]),
        }
    )
    manifest.setdefault("sealed_identity", {}).update(
        {
            "source_authority_digest": identity["source_authority_digest"],
            "consumer_frame_fingerprint": identity["consumer_frame_fingerprint"],
        }
    )
    manifest["source_calendar_field_repair"] = evidence
    manifest["content_validation_notes"] = (
        "D2 source calendar fields are repaired only at the sealed producer for "
        "approved dates; runtime verification is fail-closed and performs no repair."
    )
    atomic_write_json(sidecars["manifest"], manifest)

    audit_sha = sha256_file(audit_path)
    return {
        "status": "rebuilt",
        "input_source_sha256": input_sha,
        "output_source_sha256": output_sha,
        "rows": len(output_source),
        "changed_cell_count": int(repair_evidence["changed_cell_count"]),
        "repair_dates": repair_evidence["repair_dates"],
        "source_authority_digest": identity["source_authority_digest"],
        "consumer_frame_fingerprint": identity["consumer_frame_fingerprint"],
        "audit_sha256": audit_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_DATASET_ROOT / "source.parquet")
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_DATASET_ROOT / "calendarization_audit.json")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            rebuild_d2_source_authority(
                source_path=args.source_path,
                audit_path=args.audit_path,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
