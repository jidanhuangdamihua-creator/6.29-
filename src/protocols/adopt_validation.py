"""Closed validation contract for adopting solidified D3-D6 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


class AdoptValidationFailureReasonV1(str, Enum):
    PARENT_ARTIFACT_MISSING = "PARENT_ARTIFACT_MISSING"
    PARENT_ARTIFACT_UNREADABLE = "PARENT_ARTIFACT_UNREADABLE"
    PARENT_ARTIFACT_CORRUPT = "PARENT_ARTIFACT_CORRUPT"
    PARENT_ARTIFACT_HASH_MISMATCH = "PARENT_ARTIFACT_HASH_MISMATCH"
    PARENT_ARTIFACT_SIZE_MISMATCH = "PARENT_ARTIFACT_SIZE_MISMATCH"
    MANIFEST_SCHEMA_INVALID = "MANIFEST_SCHEMA_INVALID"
    MANIFEST_REQUIRED_FIELD_MISSING = "MANIFEST_REQUIRED_FIELD_MISSING"
    PROVENANCE_METADATA_INVALID = "PROVENANCE_METADATA_INVALID"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    COLUMN_ORDER_MISMATCH = "COLUMN_ORDER_MISMATCH"
    DTYPE_MISMATCH = "DTYPE_MISMATCH"
    FEATURE_SCHEMA_DIGEST_MISMATCH = "FEATURE_SCHEMA_DIGEST_MISMATCH"
    UNEXPECTED_COLUMN = "UNEXPECTED_COLUMN"
    REQUIRED_COLUMN_MISSING = "REQUIRED_COLUMN_MISSING"
    ENTITY_SET_MISMATCH = "ENTITY_SET_MISMATCH"
    ENTITY_DUPLICATED = "ENTITY_DUPLICATED"
    ENTITY_REQUIRED_FIELD_NULL = "ENTITY_REQUIRED_FIELD_NULL"
    ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
    PRIMARY_KEY_DUPLICATED = "PRIMARY_KEY_DUPLICATED"
    DATE_PARSE_FAILURE = "DATE_PARSE_FAILURE"
    DATE_DUPLICATED = "DATE_DUPLICATED"
    DATE_ORDER_INVALID = "DATE_ORDER_INVALID"
    DATE_DISCONTINUITY = "DATE_DISCONTINUITY"
    TIME_WINDOW_OUT_OF_BOUNDS = "TIME_WINDOW_OUT_OF_BOUNDS"
    TIME_WINDOW_LENGTH_MISMATCH = "TIME_WINDOW_LENGTH_MISMATCH"
    INSUFFICIENT_OBSERVATION_WINDOW = "INSUFFICIENT_OBSERVATION_WINDOW"
    INSUFFICIENT_BLIND_WINDOW = "INSUFFICIENT_BLIND_WINDOW"
    FILL_POLICY_VERSION_MISMATCH = "FILL_POLICY_VERSION_MISMATCH"
    FILL_POLICY_CONFIG_MISMATCH = "FILL_POLICY_CONFIG_MISMATCH"
    FILL_POLICY_EXECUTION_FAILURE = "FILL_POLICY_EXECUTION_FAILURE"
    SYNTHETIC_DATE_COUNT_MISMATCH = "SYNTHETIC_DATE_COUNT_MISMATCH"
    KNN_WINDOW_LENGTH_MISMATCH = "KNN_WINDOW_LENGTH_MISMATCH"
    KNN_WINDOW_ALIGNMENT_MISMATCH = "KNN_WINDOW_ALIGNMENT_MISMATCH"
    KNN_FEATURE_SCHEMA_MISMATCH = "KNN_FEATURE_SCHEMA_MISMATCH"
    KNN_FINGERPRINT_MISMATCH = "KNN_FINGERPRINT_MISMATCH"
    KNN_FINGERPRINT_NON_UNIQUE = "KNN_FINGERPRINT_NON_UNIQUE"
    KNN_FINGERPRINT_COLLISION = "KNN_FINGERPRINT_COLLISION"
    FUTURE_KNOWN_AUDIT_FAILED = "FUTURE_KNOWN_AUDIT_FAILED"
    FORBIDDEN_FEATURE_DETECTED = "FORBIDDEN_FEATURE_DETECTED"
    DEPENDENCY_CUTOFF_VIOLATION = "DEPENDENCY_CUTOFF_VIOLATION"
    SALES_DERIVED_FUTURE_LEAKAGE = "SALES_DERIVED_FUTURE_LEAKAGE"
    OUTPUT_ARTIFACT_HASH_MISMATCH = "OUTPUT_ARTIFACT_HASH_MISMATCH"
    OUTPUT_ATOMIC_PUBLISH_FAILURE = "OUTPUT_ATOMIC_PUBLISH_FAILURE"
    VALIDATOR_INTERNAL_ERROR = "VALIDATOR_INTERNAL_ERROR"


VALIDATION_POLICY_VERSION = "adopt-policy/v1"
_POLICY_PAYLOAD = {
    "version": VALIDATION_POLICY_VERSION,
    "failure_reasons": [reason.value for reason in AdoptValidationFailureReasonV1],
}
VALIDATION_POLICY_DIGEST = "sha256:" + hashlib.sha256(
    json.dumps(_POLICY_PAYLOAD, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

AdoptValidationFailureReason = AdoptValidationFailureReasonV1
REQUIRED_ADOPTED_MANIFEST_FIELDS = (
    "manifest_version",
    "dataset_id",
    "sealed_root_version",
    "provenance_level",
    "parent_artifacts",
    "content_validation_level",
    "adopted_content_validated",
    "content_validation_notes",
    "fill_policy_engine_version",
    "fill_policy_config_digest",
    "validation_policy_version",
    "validation_policy_digest",
    "validator_code_digest",
    "predictor_feature_schema_digest",
    "knn_feature_schema_digest",
)


def validator_code_digest() -> str:
    source = Path(__file__).read_bytes()
    return "sha256:" + hashlib.sha256(source).hexdigest()


@dataclass(frozen=True)
class AdoptValidationResult:
    status: str
    failure_reasons: Tuple[AdoptValidationFailureReasonV1, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    validation_policy_version: str = VALIDATION_POLICY_VERSION
    validation_policy_digest: str = VALIDATION_POLICY_DIGEST
    validator_code_digest: str = field(default_factory=validator_code_digest)

    def __post_init__(self) -> None:
        normalized = tuple(AdoptValidationFailureReasonV1(reason) for reason in self.failure_reasons)
        if normalized and self.status != "failed":
            raise ValueError("non-empty failure_reasons require status=failed")
        if self.status not in {"validated", "failed"}:
            raise ValueError(f"unsupported adopt validation status: {self.status!r}")
        object.__setattr__(self, "failure_reasons", normalized)

    @property
    def passed(self) -> bool:
        return self.status == "validated" and not self.failure_reasons

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["failure_reasons"] = [reason.value for reason in self.failure_reasons]
        payload["passed"] = self.passed
        return payload


def map_validator_exception(_exc: BaseException) -> AdoptValidationFailureReasonV1:
    """Map every unclassified validator exception to the closed internal code."""

    return AdoptValidationFailureReasonV1.VALIDATOR_INTERNAL_ERROR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed(reasons: Iterable[AdoptValidationFailureReasonV1], **evidence: Any) -> AdoptValidationResult:
    ordered = tuple(dict.fromkeys(AdoptValidationFailureReasonV1(reason) for reason in reasons))
    return AdoptValidationResult(status="failed", failure_reasons=ordered, evidence=evidence)


def _scan_small_table(path: Path, columns: Sequence[str]) -> Dict[str, Any]:
    frame = pd.read_parquet(path, columns=list(columns))
    evidence: Dict[str, Any] = {"rows_scanned": int(len(frame))}
    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        evidence["date_min"] = None if parsed.dropna().empty else parsed.min().strftime("%Y-%m-%d")
        evidence["date_max"] = None if parsed.dropna().empty else parsed.max().strftime("%Y-%m-%d")
    return evidence


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset_id: Optional[object] = None,
    expected_provenance: Optional[str] = "adopted_solidified",
    expected_fill_policy_config_digest: Optional[str] = None,
) -> AdoptValidationResult:
    """Validate the immutable manifest portion of an adopted dataset."""

    if not isinstance(manifest, Mapping):
        return _failed([AdoptValidationFailureReasonV1.MANIFEST_SCHEMA_INVALID])
    missing = [field for field in REQUIRED_ADOPTED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        return _failed(
            [AdoptValidationFailureReasonV1.MANIFEST_REQUIRED_FIELD_MISSING],
            missing_fields=missing,
        )
    reasons: list[AdoptValidationFailureReasonV1] = []
    evidence: Dict[str, Any] = {}
    if dataset_id is not None and str(manifest["dataset_id"]).upper() != str(dataset_id).upper():
        reasons.append(AdoptValidationFailureReasonV1.MANIFEST_SCHEMA_INVALID)
    provenance = str(manifest["provenance_level"])
    if expected_provenance is not None and provenance != str(expected_provenance):
        reasons.append(AdoptValidationFailureReasonV1.PROVENANCE_METADATA_INVALID)
    if manifest["sealed_root_version"] != "d1_d6_sealed_v1":
        reasons.append(AdoptValidationFailureReasonV1.MANIFEST_SCHEMA_INVALID)
    if manifest["validation_policy_version"] != VALIDATION_POLICY_VERSION:
        reasons.append(AdoptValidationFailureReasonV1.MANIFEST_SCHEMA_INVALID)
    if manifest["validation_policy_digest"] != VALIDATION_POLICY_DIGEST:
        reasons.append(AdoptValidationFailureReasonV1.MANIFEST_SCHEMA_INVALID)
    if manifest["fill_policy_engine_version"] != "calendarize-fill/v1":
        reasons.append(AdoptValidationFailureReasonV1.FILL_POLICY_VERSION_MISMATCH)
    if (
        expected_fill_policy_config_digest is not None
        and manifest["fill_policy_config_digest"] != expected_fill_policy_config_digest
    ):
        reasons.append(AdoptValidationFailureReasonV1.FILL_POLICY_CONFIG_MISMATCH)
    if provenance == "adopted_solidified":
        parents = manifest["parent_artifacts"]
        if not isinstance(parents, Mapping) or set(("source", "target")).difference(parents):
            reasons.append(AdoptValidationFailureReasonV1.PROVENANCE_METADATA_INVALID)
        else:
            for role in ("source", "target"):
                record = parents[role]
                required = {
                    "path",
                    "sha256",
                    "size_bytes",
                    "observed_at",
                    "mtime_ns",
                    "first_seen_at",
                    "first_seen_source",
                    "first_seen_reliability",
                }
                if not isinstance(record, Mapping) or required.difference(record):
                    reasons.append(AdoptValidationFailureReasonV1.PROVENANCE_METADATA_INVALID)
                    continue
                if record["first_seen_at"] is None and record["first_seen_reliability"] != "unavailable":
                    reasons.append(AdoptValidationFailureReasonV1.PROVENANCE_METADATA_INVALID)
    if manifest["content_validation_level"] == "structural_only" and manifest["adopted_content_validated"] is not False:
        reasons.append(AdoptValidationFailureReasonV1.PROVENANCE_METADATA_INVALID)
    evidence["dataset_id"] = manifest["dataset_id"]
    evidence["provenance_level"] = provenance
    if reasons:
        return _failed(reasons, **evidence)
    return AdoptValidationResult(status="validated", evidence=evidence)


validate_adopt_manifest = validate_manifest
validate_adopted_manifest = validate_manifest


def validate_adopted_artifact(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_size_bytes: Optional[int] = None,
    expected_columns: Optional[Sequence[str]] = None,
    required_columns: Sequence[str] = (),
    expected_row_count: Optional[int] = None,
    validation_policy_version: str = VALIDATION_POLICY_VERSION,
    max_rows_to_scan: int = 200_000,
) -> AdoptValidationResult:
    """Validate one parent parquet and never downgrade a failure to a warning."""

    candidate = Path(path)
    if validation_policy_version != VALIDATION_POLICY_VERSION:
        return _failed(
            [AdoptValidationFailureReason.MANIFEST_SCHEMA_INVALID],
            validation_policy_version=validation_policy_version,
        )
    if not candidate.exists():
        return _failed([AdoptValidationFailureReason.PARENT_ARTIFACT_MISSING], path=str(candidate))
    if not candidate.is_file():
        return _failed([AdoptValidationFailureReason.PARENT_ARTIFACT_UNREADABLE], path=str(candidate))

    reasons: list[AdoptValidationFailureReasonV1] = []
    evidence: Dict[str, Any] = {"path": str(candidate)}
    try:
        size = candidate.stat().st_size
        evidence["size_bytes"] = int(size)
        if expected_sha256 is not None:
            actual_hash = _sha256_file(candidate)
            evidence["sha256"] = actual_hash
            if actual_hash.lower() != str(expected_sha256).lower():
                reasons.append(AdoptValidationFailureReasonV1.PARENT_ARTIFACT_HASH_MISMATCH)
        if expected_size_bytes is not None and int(size) != int(expected_size_bytes):
            reasons.append(AdoptValidationFailureReasonV1.PARENT_ARTIFACT_SIZE_MISMATCH)
        parquet = pq.ParquetFile(candidate)
        schema = parquet.schema_arrow
        names = list(schema.names)
        evidence["columns"] = names
        evidence["row_count"] = int(parquet.metadata.num_rows)
        if expected_row_count is not None and int(parquet.metadata.num_rows) != int(expected_row_count):
            reasons.append(AdoptValidationFailureReasonV1.ROW_COUNT_MISMATCH)
        if expected_columns is not None:
            expected = list(expected_columns)
            if names != expected:
                if set(names) == set(expected):
                    reasons.append(AdoptValidationFailureReasonV1.COLUMN_ORDER_MISMATCH)
                else:
                    reasons.append(AdoptValidationFailureReasonV1.SCHEMA_MISMATCH)
        missing = [column for column in required_columns if column not in names]
        if missing:
            evidence["missing_columns"] = missing
            reasons.append(AdoptValidationFailureReasonV1.REQUIRED_COLUMN_MISSING)
        if int(parquet.metadata.num_rows) <= int(max_rows_to_scan):
            scan_columns = [column for column in ("date", "entity_id", "store_id", "item_id", "brand_id") if column in names]
            if scan_columns:
                table_evidence = _scan_small_table(candidate, scan_columns)
                evidence.update(table_evidence)
                if "date" in scan_columns:
                    dates = pd.to_datetime(pd.read_parquet(candidate, columns=["date"])["date"], errors="coerce")
                    if dates.isna().any():
                        reasons.append(AdoptValidationFailureReasonV1.DATE_PARSE_FAILURE)
                    elif dates.duplicated().any() and len(scan_columns) == 1:
                        reasons.append(AdoptValidationFailureReasonV1.DATE_DUPLICATED)
    except (OSError, PermissionError):
        reasons.append(AdoptValidationFailureReasonV1.PARENT_ARTIFACT_UNREADABLE)
    except (ValueError, TypeError, KeyError, EOFError):
        reasons.append(AdoptValidationFailureReasonV1.PARENT_ARTIFACT_CORRUPT)
    except Exception as exc:  # noqa: BLE001 - closed enum requires fail-closed mapping
        reasons.append(map_validator_exception(exc))

    if reasons:
        return _failed(reasons, **evidence)
    return AdoptValidationResult(status="validated", evidence=evidence)


def validate_adopted_pair(
    source_path: Path,
    target_path: Path,
    *,
    expected_source_columns: Optional[Sequence[str]] = None,
    expected_target_columns: Optional[Sequence[str]] = None,
) -> AdoptValidationResult:
    """Validate source and target as one dataset-level adoption unit."""

    source = validate_adopted_artifact(
        source_path,
        expected_columns=expected_source_columns,
        required_columns=("date", "sales"),
    )
    target = validate_adopted_artifact(
        target_path,
        expected_columns=expected_target_columns,
        required_columns=("date", "sales"),
    )
    reasons = tuple(dict.fromkeys((*source.failure_reasons, *target.failure_reasons)))
    evidence = {"source": source.to_dict(), "target": target.to_dict()}
    if reasons:
        return AdoptValidationResult(status="failed", failure_reasons=reasons, evidence=evidence)
    return AdoptValidationResult(status="validated", evidence=evidence)


__all__ = [
    "AdoptValidationFailureReasonV1",
    "AdoptValidationFailureReason",
    "AdoptValidationResult",
    "REQUIRED_ADOPTED_MANIFEST_FIELDS",
    "VALIDATION_POLICY_DIGEST",
    "VALIDATION_POLICY_VERSION",
    "map_validator_exception",
    "validate_adopt_manifest",
    "validate_adopted_artifact",
    "validate_adopted_manifest",
    "validate_adopted_pair",
    "validate_manifest",
    "validator_code_digest",
]
