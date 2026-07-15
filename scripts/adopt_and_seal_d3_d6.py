#!/usr/bin/env python3
"""Adopt the existing D3-D6 solidified parquets into the sealed v1 root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.sealed_daily import (
    FILL_POLICY_ENGINE_VERSION,
    RUNTIME_FILL_POLICY,
    SHARED_FILL_POLICY_CONFIG_DIGEST,
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
        "predictor_feature_schema_digest": predictor.digest,
        "knn_feature_schema_digest": knn.digest,
        "windows": _window_descriptor(number),
    }
    sidecars = {
        "provenance.json": parent_artifacts,
        "adopt_validation_report.json": report,
        "calendarization_audit.json": {
            "engine_version": FILL_POLICY_ENGINE_VERSION,
            "runtime_policy": RUNTIME_FILL_POLICY,
            "content_validation_level": "structural_only",
            "source_sales_repair": "not_reconstructed_during_adoption",
        },
        "source_sales_canonicalization.json": {
            "version": "source_sales_canonicalization/v1",
            "status": "not_reconstructed_during_adoption",
            "content_validation_level": "structural_only",
        },
    }
    return publish_sealed_dataset(
        output_dir,
        number,
        source_path=source,
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
