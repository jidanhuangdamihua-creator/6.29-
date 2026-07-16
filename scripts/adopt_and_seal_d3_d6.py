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
from src.protocols.gate1_transformation import (
    CONTRACT_DIGEST,
    CONTRACT_VERSION,
    AuthorityProducer,
    AvailabilityResolver,
    ForecastBlindProducer,
    FormalInputLoader,
    FormalPreflight,
    HistoryReconstructionProducer,
    ProofWriter,
    SafeTargetViewOperator,
    SchemaRegistry,
    canonical_digest,
    normalized_frame_digest,
)
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
_FORMAL_PATHS = {
    "contract": ROOT / "docs/protocol/gate1_frozen_transformation_contract.md",
    "implementation_scope": ROOT / "docs/protocol/gate1_implementation_scope.md",
    "traceability_matrix": ROOT / "docs/protocol/gate1_contract_traceability_matrix.md",
}
_CODE_PATHS = {
    "operator": ROOT / "tools/operations/materialize_d1_d6_sealed_authority.py",
    "producer": Path(__file__).resolve(),
    "gate1_transformation": ROOT / "src/protocols/gate1_transformation.py",
    "feature_schema": ROOT / "src/protocols/feature_schema.py",
    "sealed_daily": ROOT / "src/data_processing/sealed_daily.py",
    "adopt_validation": ROOT / "src/protocols/adopt_validation.py",
    "sealing_protocol": ROOT / "src/protocols/sealing_protocol.py",
}


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


def _formal_identity() -> Dict[str, Any]:
    FormalInputLoader(ROOT).load()
    files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in _FORMAL_PATHS.items()
    }
    return {
        "contract_digest": CONTRACT_DIGEST,
        "contract_version": CONTRACT_VERSION,
        "files": files,
        "formal_input_set_digest": canonical_digest(files),
    }


def _raw_authority_identity(dataset_id: int) -> Dict[str, Any]:
    frozen = AuthorityProducer.from_frozen_contract(ROOT)
    prefix = f"D{int(dataset_id)}:"
    selected_files = {
        name: path for name, path in frozen.files.items() if name.startswith(prefix)
    }
    selected_hashes = {
        name: digest
        for name, digest in frozen.expected_hashes.items()
        if name.startswith(prefix)
    }
    manifest = AuthorityProducer(
        root=ROOT,
        files=selected_files,
        expected_hashes=selected_hashes,
    ).load()
    files = [
        {
            "name": name,
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for name, record in sorted(manifest.files.items())
        if name.startswith(prefix)
    ]
    if not files:
        raise ValueError(f"D{dataset_id} has no frozen raw authority")
    return {
        "dataset": f"D{int(dataset_id)}",
        "files": files,
        "approved_input_set_digest": canonical_digest(files),
        "snapshot_identity": canonical_digest(
            [(item["path"], item["sha256"]) for item in files]
        ),
        "verified_from_bytes": True,
    }


def _code_identity() -> Dict[str, Any]:
    files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in _CODE_PATHS.items()
    }
    return {"files": files, "code_set_digest": canonical_digest(files)}


def _key_status(frame: pd.DataFrame, columns: tuple[str, ...]) -> Dict[str, Any]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"key columns missing: {missing}")
    keys = frame.loc[:, list(columns)]
    if keys.isna().any().any():
        raise ValueError(f"key columns contain nulls: {columns}")
    return {
        "columns": list(columns),
        "unique": not bool(keys.duplicated().any()),
        "sorted": keys.reset_index(drop=True).equals(
            keys.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
        ),
        "digest": canonical_digest(keys.astype(str).values.tolist()),
    }


def _dataset_exclusions(dataset_id: int) -> Dict[str, str]:
    values = {
        3: ("Open", "Customers", "Promo", "Promo2", "PromoInterval"),
        4: ("hours_sale", "hours_stock_status", "stock_hour6_22_cnt"),
        5: ("transactions", "week"),
        6: ("evaluation_truth", "snap_CA", "snap_TX", "snap_WI"),
    }
    return {
        name: "excluded from forecast/model consumers by frozen contract"
        for name in values[dataset_id]
    }


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


def _build_gate1_publication_proof(
    dataset_id: int,
    *,
    parent_source: pd.DataFrame,
    source: pd.DataFrame,
    target: pd.DataFrame,
    parent_artifacts: Dict[str, Any],
    repair: Dict[str, Any],
    parent_root: Path,
    output_root: Path,
    history_result: Any,
    blind_result: Any,
    safe_views: Any,
) -> Dict[str, Any]:
    dataset = f"D{int(dataset_id)}"
    raw = _raw_authority_identity(dataset_id)
    formal = _formal_identity()
    code = _code_identity()
    predictor = get_predictor_schema(dataset)
    knn = get_knn_schema(dataset)
    registry = SchemaRegistry()
    source_key = tuple(_SOURCE_GROUP_COLUMNS[dataset_id]) + ("date",)
    target_key = tuple(_SOURCE_GROUP_COLUMNS[dataset_id]) + ("date",)
    source_status = _key_status(source, source_key)
    target_status = _key_status(target, target_key)
    if not all((source_status["unique"], source_status["sorted"])):
        raise ValueError(f"{dataset} source key proof failed")
    if not all((target_status["unique"], target_status["sorted"])):
        raise ValueError(f"{dataset} target key proof failed")
    windows = _window_descriptor(dataset_id)
    blind_start = pd.Timestamp(windows["target"]["blind_start"])
    blind_end = pd.Timestamp(windows["target"]["blind_end"])
    configuration = {
        "dataset": dataset,
        "parent_root": str(Path(parent_root).resolve()),
        "source_input": parent_artifacts["source"]["path"],
        "target_input": parent_artifacts["target"]["path"],
        "source_group_columns": list(_SOURCE_GROUP_COLUMNS[dataset_id]),
        "producer_entrypoint": _CODE_PATHS["producer"].relative_to(ROOT).as_posix(),
        "command_argv": [
            sys.executable,
            _CODE_PATHS["producer"].relative_to(ROOT).as_posix(),
            "--dataset",
            dataset.lower(),
            "--parent-root",
            str(Path(parent_root).resolve()),
            "--output-dir",
            str(Path(output_root).resolve()),
        ],
    }
    approved_inputs = {
        "artifacts": parent_artifacts,
        "digest": canonical_digest(parent_artifacts),
    }
    schemas = {
        "predictor": predictor.descriptor(),
        "predictor_digest": predictor.digest,
        "knn": knn.descriptor(),
        "knn_digest": knn.digest,
        "worker_digest": registry.digest(dataset, "worker"),
        "knn_view_digest": registry.digest(dataset, "knn"),
    }
    exclusions = _dataset_exclusions(dataset_id)
    views = {
        "worker": {"columns": list(safe_views.worker.columns), "digest": safe_views.digests["worker"]},
        "knn": {"columns": list(safe_views.knn.columns), "digest": safe_views.digests["knn"]},
        "forecast": {"columns": list(safe_views.forecast.columns), "digest": safe_views.digests["forecast"]},
        "label": {"columns": list(safe_views.label_truth.columns), "digest": safe_views.digests["label"]},
        "audit": {"columns": list(safe_views.audit.columns), "digest": safe_views.digests["audit"]},
    }
    availability = AvailabilityResolver()
    decisions = []
    origin = windows["source"]["pretrain_end"]
    for name in dict.fromkeys((*predictor.ordered_names, *knn.ordered_names)):
        if name in {"sales", "date"}:
            continue
        decision = availability.resolve(dataset, name, available_at=origin, origin=origin)
        decisions.append(
            {
                "field": decision.field,
                "authority": decision.authority,
                "available_at": decision.available_at,
                "history_rule": decision.history_rule,
                "forecast_rule": decision.forecast_rule,
                "missing_rule": decision.missing_rule,
                "status": decision.status,
            }
        )
    resolver = {
        "status": "passed",
        "approved_fields": list(predictor.ordered_names),
        "knn_fields": list(knn.ordered_names),
        "exclusions": exclusions,
        "decisions": decisions,
    }
    base_proof = ProofWriter().build(
        contract_digest=CONTRACT_DIGEST,
        authority=raw,
        schemas=schemas,
        resolver=resolver,
        views=views,
        artifacts=approved_inputs,
    )
    preflight = FormalPreflight().check(
        {"contract_digest": CONTRACT_DIGEST, "proof": base_proof}
    )
    proof: Dict[str, Any] = {
        "version": "gate1_publication_proof_v1",
        "status": "publication_ready",
        "dataset": dataset,
        "formal_identity": formal,
        "raw_authority": raw,
        "code_identity": code,
        "producer_configuration": configuration,
        "producer_configuration_digest": canonical_digest(configuration),
        "approved_inputs": approved_inputs,
        "parent_lineage": {
            "status": "bound_to_frozen_raw_snapshot",
            "raw_snapshot_identity": raw["snapshot_identity"],
            "parent_digest": approved_inputs["digest"],
        },
        "key_window": {
            "dataset": dataset,
            "source_key": source_status,
            "target_key": target_status,
            "windows": windows,
            "horizon_days": int((blind_end - blind_start).days + 1),
            "source_history_days": 180,
            "knn_observation_days": 30,
        },
        "transformation": {
            "status": "contract_validated",
            "calendarization": "gregorian_daily",
            "source_rows_before": int(len(parent_source)),
            "source_rows_after": int(len(source)),
            "target_rows_before": int(len(target)),
            "target_rows_after": int(len(target)),
            "repair_mask_sha256": repair["repair_mask_sha256"],
            "affected_date_digest": repair["affected_date_digest"],
            "repair_reason_counts": repair["repair_reason_counts"],
            "history_field_repair_counts": dict(history_result.repair_counts),
            "history_repair_mask_digest": history_result.repair_mask_digest,
            "blind_view_digest": blind_result.digest,
            "blind_exclusions": dict(blind_result.exclusions),
            "dataset_rule": f"gate1_frozen_{dataset.lower()}_v1",
            "generic_fill": False,
            "backward_fill": False,
        },
        "availability_no_leakage": {
            "status": "passed",
            "resolver": resolver,
            "target_truth_isolated": True,
            "target_day_actual_isolated": True,
            "audit_only_isolated": True,
            "forbidden_fields_isolated": True,
        },
        "schemas": schemas,
        "views": views,
        "content": {
            "source_normalized_row_digest": normalized_frame_digest(source),
            "target_normalized_row_digest": normalized_frame_digest(target),
            "normalization": "gate1_logical_values_columns_dtypes_v1",
        },
        "formal_preflight_proof": base_proof,
        "formal_preflight": preflight,
    }
    proof["proof_digest"] = canonical_digest(proof)
    return proof


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
    source_key = list(_SOURCE_GROUP_COLUMNS[number]) + ["date"]
    canonical_source_frame = canonical_source_frame.sort_values(
        source_key, kind="mergesort"
    ).reset_index(drop=True)
    target_frame = pd.read_parquet(target)
    target_key = list(_SOURCE_GROUP_COLUMNS[number]) + ["date"]
    target_frame = target_frame.sort_values(target_key, kind="mergesort").reset_index(drop=True)
    source_window = get_source_pretrain_window(number)
    history_result = HistoryReconstructionProducer().build(
        f"D{number}",
        canonical_source_frame,
        origin=source_window.pretrain_end,
    )
    canonical_source_frame = history_result.frame.sort_values(
        source_key, kind="mergesort"
    ).reset_index(drop=True)
    blind_result = ForecastBlindProducer().build(
        f"D{number}",
        history_result,
        target_frame,
        origin=source_window.pretrain_end,
    )
    safe_views = SafeTargetViewOperator().build(
        f"D{number}", history_result.frame, target_frame
    )
    source_bytes_unchanged = canonical_source_frame.equals(parent_source_frame)
    report["source_sales_repair"] = source_sales_proof

    predictor = get_predictor_schema(f"D{number}")
    knn = get_knn_schema(f"D{number}")
    parent_artifacts = {"source": source_parent, "target": target_parent}
    publication_proof = _build_gate1_publication_proof(
        number,
        parent_source=parent_source_frame,
        source=canonical_source_frame,
        target=target_frame,
        parent_artifacts=parent_artifacts,
        repair=source_sales_proof,
        parent_root=parent_root,
        output_root=output_dir,
        history_result=history_result,
        blind_result=blind_result,
        safe_views=safe_views,
    )
    report["gate1_publication_proof"] = publication_proof
    manifest = {
        "manifest_version": "sealed_dataset_manifest_v1",
        "dataset_id": f"D{number}",
        "sealed_root_version": SEALING_PROTOCOL_VERSION,
        "provenance_level": "gate1_contract_transformed",
        "content_validation_level": "gate1_contract_validated",
        "adopted_content_validated": True,
        "content_validation_notes": (
            "Validated against the frozen Gate 1 raw authority, formal identities, "
            "key/window contract, safe schemas, repair evidence, availability isolation, "
            "canonical content identity, and formal preflight."
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
        "gate1_publication_proof": publication_proof,
    }
    sidecars = {
        "provenance.json": parent_artifacts,
        "adopt_validation_report.json": report,
        "calendarization_audit.json": {
            "engine_version": canonical_source_frame.attrs["fill_policy_engine_version"],
            "runtime_policy": canonical_source_frame.attrs["runtime_fill_policy"],
            "content_validation_level": "gate1_contract_validated",
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
        target_frame=target_frame,
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
