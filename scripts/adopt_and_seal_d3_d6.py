#!/usr/bin/env python3
"""Adopt the existing D3-D6 solidified parquets into the sealed v1 root."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.sealed_daily import (
    FILL_POLICY_ENGINE_VERSION,
    RUNTIME_FILL_POLICY,
    SHARED_FILL_POLICY_CONFIG_DIGEST,
    SOURCE_SALES_CANONICALIZATION_VERSION,
    calendarize_and_fill,
    canonicalize_source_sales,
    publish_sealed_dataset,
    sha256_file,
)
from src.protocols.adopt_validation import (
    VALIDATION_POLICY_DIGEST,
    VALIDATION_POLICY_VERSION,
    validate_adopted_pair,
    validator_code_digest,
)
from src.protocols.feature_schema import get_knn_schema, get_predictor_schema
from src.protocols.sealing_protocol import SEALING_PROTOCOL_VERSION, get_source_pretrain_window, get_target_window


DEFAULT_PARENT_ROOT = ROOT / "数据集" / "固化数据"
DEFAULT_OUTPUT_ROOT = DEFAULT_PARENT_ROOT / "d1_d6_sealed_v1"

_SOURCE_GROUP_COLUMNS = {
    3: ("store_id",),
    4: ("store_id", "product_id"),
    5: ("store_nbr", "item_nbr"),
    6: ("store_id", "item_id"),
}
_SOURCE_REPAIR_REASONS = (
    "original_nan",
    "original_negative",
    "calendar_row_missing",
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_PREFIXED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class AdoptionValidationError(RuntimeError):
    """Raised when one whole dataset cannot be adopted and sealed."""

    def __init__(self, dataset_id: int, report: Dict[str, Any]) -> None:
        self.dataset_id = int(dataset_id)
        self.report = report
        reasons = ",".join(report.get("failure_reasons", [])) or "unknown"
        super().__init__(f"D{dataset_id} adoption failed: {reasons}")


def _date(value: object) -> str:
    return str(value)


def _parent_artifact_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "size_bytes": int(stat.st_size),
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mtime_ns": int(stat.st_mtime_ns),
        "first_seen_at": None,
        "first_seen_source": None,
        "first_seen_reliability": "unavailable",
    }


def _window_descriptor(dataset_id: int) -> Dict[str, Any]:
    target = get_target_window(dataset_id)
    source = get_source_pretrain_window(dataset_id)
    return {
        "target": {
            "train_start": _date(target.train_start),
            "train_end": _date(target.train_end),
            "validation_start": _date(target.validation_start),
            "validation_end": _date(target.validation_end),
            "blind_start": _date(target.blind_start),
            "blind_end": _date(target.blind_end),
        },
        "source": {
            "pretrain_start": _date(source.pretrain_start),
            "pretrain_end": _date(source.pretrain_end),
            "knn_start": _date(source.knn_start),
            "knn_end": _date(source.knn_end),
        },
    }


def _parent_paths(dataset_id: int, parent_root: Path) -> tuple[Path, Path]:
    root = Path(parent_root)
    return (
        root / f"dataset{dataset_id}-source.parquet",
        root / f"dataset{dataset_id}-target.parquet",
    )


def _calendarize_and_canonicalize_source(
    dataset_id: int,
    source_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    parent = pd.read_parquet(source_path)
    window = get_source_pretrain_window(dataset_id)
    required_dates = pd.date_range(window.pretrain_start, window.pretrain_end, freq="D")
    calendarized = calendarize_and_fill(
        parent,
        group_cols=_SOURCE_GROUP_COLUMNS[dataset_id],
        additional_dates=required_dates,
    )
    calendar_row_missing = calendarized.attrs["calendar_row_missing_mask"]
    canonicalized, audit = canonicalize_source_sales(
        calendarized,
        calendar_row_missing=calendar_row_missing,
    )
    proof = dict(audit)
    proof["status"] = "canonicalized"
    return parent, canonicalized, proof


def _assert_complete_source_repair_proof(proof: Dict[str, Any]) -> None:
    if proof.get("version") != SOURCE_SALES_CANONICALIZATION_VERSION:
        raise ValueError("source-sales repair proof version mismatch")
    if proof.get("status") in {None, "not_reconstructed_during_adoption", "unavailable"}:
        raise ValueError("source-sales repair proof has no success status")
    counts = proof.get("repair_reason_counts")
    if not isinstance(counts, dict) or tuple(counts) != _SOURCE_REPAIR_REASONS:
        raise ValueError("source-sales repair reasons are not the closed reason set")
    affected_rows = proof.get("affected_rows")
    rows_examined = proof.get("rows_examined")
    if not isinstance(affected_rows, list) or not isinstance(rows_examined, int):
        raise ValueError("source-sales repair rows are invalid")
    if rows_examined < 0 or len(affected_rows) > rows_examined:
        raise ValueError("source-sales repair row counts do not close")
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("source-sales repair reason counts are invalid")
    if sum(counts.values()) != len(affected_rows):
        raise ValueError("source-sales repair reason counts do not match affected rows")
    actual_counts = Counter(str(row.get("reason")) for row in affected_rows)
    if any(actual_counts[reason] != counts[reason] for reason in _SOURCE_REPAIR_REASONS):
        raise ValueError("source-sales repair row reasons do not match reason counts")
    repair_digest = proof.get("repair_mask_sha256")
    date_digest = proof.get("affected_date_digest")
    if not isinstance(repair_digest, str) or _SHA256_HEX.fullmatch(repair_digest) is None:
        raise ValueError("source-sales repair mask digest is invalid")
    if not isinstance(date_digest, str) or _PREFIXED_SHA256.fullmatch(date_digest) is None:
        raise ValueError("source-sales affected-date digest is invalid")


def _assert_published_proof_identity(
    proof: Dict[str, Any],
    manifest: Dict[str, Any],
    report: Dict[str, Any],
    sidecars: Dict[str, Dict[str, Any]],
) -> None:
    _assert_complete_source_repair_proof(proof)
    if not (
        proof
        == manifest.get("source_sales_repair")
        == report.get("source_sales_repair")
        == sidecars.get("source_sales_canonicalization.json")
    ):
        raise ValueError("source-sales repair proof identity mismatch before publication")
    if manifest.get("source_sales_repair_mask_sha256") != proof["repair_mask_sha256"]:
        raise ValueError("source-sales repair mask identity mismatch before publication")
    if manifest.get("source_sales_repair_reason_counts") != proof["repair_reason_counts"]:
        raise ValueError("source-sales repair count identity mismatch before publication")


def adopt_and_seal_dataset(
    dataset_id: int,
    *,
    source_path: Optional[Path] = None,
    target_path: Optional[Path] = None,
    parent_root: Path = DEFAULT_PARENT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Validate and atomically adopt both parent parquets as one dataset."""

    number = int(dataset_id)
    if number not in range(3, 7):
        raise ValueError("adoption is approved only for D3-D6")
    default_source, default_target = _parent_paths(number, parent_root)
    source = Path(source_path or default_source)
    target = Path(target_path or default_target)
    # File identity is collected before any output directory is created.
    source_parent = _parent_artifact_record(source)
    target_parent = _parent_artifact_record(target)
    validation = validate_adopted_pair(source, target)
    report = validation.to_dict()
    if not validation.passed:
        raise AdoptionValidationError(number, report)

    parent_source_frame, canonical_source_frame, source_sales_proof = (
        _calendarize_and_canonicalize_source(number, source)
    )
    source_bytes_unchanged = canonical_source_frame.equals(parent_source_frame)
    report["source_sales_repair"] = source_sales_proof

    predictor = get_predictor_schema(f"D{number}")
    knn = get_knn_schema(f"D{number}")
    parent_artifacts = {"source": source_parent, "target": target_parent}
    manifest = {
        "manifest_version": "sealed_dataset_manifest_v1",
        "dataset_id": f"D{number}",
        "sealed_root_version": SEALING_PROTOCOL_VERSION,
        "provenance_level": "adopted_solidified",
        "content_validation_level": "structural_only",
        "adopted_content_validated": False,
        "content_validation_notes": (
            "Only file identity, structural integrity, windows, schemas, KNN fingerprints, "
            "and feature protocol were validated. Historical numeric correctness was not "
            "reconstructed from raw data."
        ),
        "parent_artifacts": parent_artifacts,
        "parent_artifact_sha256": source_parent["sha256"],
        "parent_artifact_size_bytes": source_parent["size_bytes"],
        "parent_artifact_observed_at": source_parent["observed_at"],
        "parent_artifact_mtime_ns": source_parent["mtime_ns"],
        "parent_artifact_first_seen_at": None,
        "parent_artifact_first_seen_source": None,
        "parent_artifact_first_seen_reliability": "unavailable",
        "fill_policy_engine_version": FILL_POLICY_ENGINE_VERSION,
        "fill_policy_shared_with_raw_rebuild": True,
        "fill_policy_config_digest": SHARED_FILL_POLICY_CONFIG_DIGEST,
        "runtime_fill_policy": RUNTIME_FILL_POLICY,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "validation_policy_digest": VALIDATION_POLICY_DIGEST,
        "validator_code_digest": validator_code_digest(),
        "source_sales_canonicalization_version": source_sales_proof["version"],
        "source_sales_repair_mask_sha256": source_sales_proof["repair_mask_sha256"],
        "source_sales_repair_reason_counts": source_sales_proof["repair_reason_counts"],
        "source_sales_repair": source_sales_proof,
        "predictor_feature_schema_digest": predictor.digest,
        "knn_feature_schema_digest": knn.digest,
        "windows": _window_descriptor(number),
    }
    sidecars = {
        "provenance.json": parent_artifacts,
        "adopt_validation_report.json": report,
        "calendarization_audit.json": {
            "engine_version": canonical_source_frame.attrs["fill_policy_engine_version"],
            "runtime_policy": canonical_source_frame.attrs["runtime_fill_policy"],
            "content_validation_level": "structural_only",
            "source": {
                "synthetic_date_count": canonical_source_frame.attrs["synthetic_date_count"],
                "config_digest": canonical_source_frame.attrs["fill_policy_config_digest"],
                "rule_descriptor": canonical_source_frame.attrs[
                    "calendarization_rule_descriptor"
                ],
            },
            "source_sales_repair": source_sales_proof["status"],
        },
        "source_sales_canonicalization.json": source_sales_proof,
    }
    _assert_published_proof_identity(source_sales_proof, manifest, report, sidecars)
    return publish_sealed_dataset(
        output_dir,
        number,
        source_path=source if source_bytes_unchanged else None,
        source_frame=None if source_bytes_unchanged else canonical_source_frame,
        target_path=target,
        manifest=manifest,
        validation_report=report,
        sidecars=sidecars,
        predictor_schema=predictor.descriptor(),
        knn_schema=knn.descriptor(),
    )


def adopt_and_seal_all(
    *,
    parent_root: Path = DEFAULT_PARENT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    dataset_ids: Iterable[int] = range(3, 7),
) -> list[Path]:
    return [
        adopt_and_seal_dataset(
            int(dataset_id),
            parent_root=parent_root,
            output_dir=output_dir,
        )
        for dataset_id in dataset_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("d3", "d4", "d5", "d6", "all"), default="all")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    ids = range(3, 7) if args.dataset == "all" else (int(args.dataset[1:]),)
    for dataset_id in ids:
        path = adopt_and_seal_dataset(
            dataset_id,
            parent_root=args.parent_root,
            output_dir=args.output_dir,
        )
        print(f"sealed D{dataset_id}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "AdoptionValidationError",
    "adopt_and_seal_all",
    "adopt_and_seal_dataset",
    "main",
]
