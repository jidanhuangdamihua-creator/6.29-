"""Fail-closed loading for the optional precomputed D5 source history."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.utils.d5_calendar_reconstruction import D5AuthorityBundle


D5_PRECOMPUTED_SOURCE_FILENAME = "prepared_180day_source.parquet"
D5_PRECOMPUTED_MANIFEST_FILENAME = "prepared_180day_source_manifest.json"
D5_FORCE_RECOMPUTE_ENV = "D5_FORCE_RECOMPUTE"
D5_PRECOMPUTED_AUXILIARY_FILES = (
    "oil",
    "transactions",
    "items",
    "stores",
    "holidays",
)


class D5PrecomputedSourceHistoryError(ValueError):
    """Base error for malformed or unsafe precomputed D5 artifacts."""


class D5PrecomputedSourceHistoryHashMismatch(D5PrecomputedSourceHistoryError):
    """Raised when a manifest-bound input or artifact has changed."""

    def __init__(self, *, file_label: str, expected: str, actual: str) -> None:
        self.file_label = str(file_label)
        self.expected = str(expected)
        self.actual = str(actual)
        super().__init__(
            "D5_PRECOMPUTED_HASH_MISMATCH "
            f"file={self.file_label} expected={self.expected} actual={self.actual}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return "MISSING"
    return digest.hexdigest()


def _iso_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise D5PrecomputedSourceHistoryError(f"invalid source-history date: {value!r}")
    return timestamp.normalize().date().isoformat()


def _frame_schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in frame.dtypes.items()
    ]


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D5PrecomputedSourceHistoryError(
            f"D5_PRECOMPUTED_MANIFEST_INVALID field={field}"
        )
    return value


def _require_hash_record(
    manifest_section: Mapping[str, Any],
    *,
    manifest_field: str,
    file_label: str,
    path: Path,
) -> None:
    record = manifest_section.get(manifest_field)
    if not isinstance(record, Mapping) or not record.get("sha256"):
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_MANIFEST_INVALID "
            f"missing_sha256={file_label}"
        )
    expected = str(record["sha256"])
    actual = _sha256_file(path)
    if actual != expected:
        raise D5PrecomputedSourceHistoryHashMismatch(
            file_label=file_label,
            expected=expected,
            actual=actual,
        )


def _precomputed_paths(source_path: Path) -> tuple[Path, Path]:
    directory = Path(source_path).parent
    return (
        directory / D5_PRECOMPUTED_SOURCE_FILENAME,
        directory / D5_PRECOMPUTED_MANIFEST_FILENAME,
    )


def load_precomputed_d5_source_history(
    *,
    source_path: Path,
    authorities: D5AuthorityBundle,
    source_history_start: object,
    source_history_end: object,
    source_history_days: int,
    key_fields: Sequence[str],
) -> tuple[pd.DataFrame, Mapping[str, Any]] | None:
    """Load a validated static D5 frame, or return None for compatibility.

    Returning ``None`` is limited to the two intentional bypass cases:
    ``D5_FORCE_RECOMPUTE=1`` or either optional artifact being absent.  Once
    both artifacts exist, every manifest-bound input and the artifact itself
    must validate; failures never fall back to runtime reconstruction.
    """
    if os.environ.get(D5_FORCE_RECOMPUTE_ENV) == "1":
        return None

    output_path, manifest_path = _precomputed_paths(Path(source_path))
    if not output_path.is_file() or not manifest_path.is_file():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise D5PrecomputedSourceHistoryError(
            f"D5_PRECOMPUTED_MANIFEST_INVALID path={manifest_path}"
        ) from exc
    manifest = _require_mapping(manifest, field="root")

    history = _require_mapping(manifest.get("source_history"), field="source_history")
    expected_days = int(history.get("source_history_days", -1))
    if expected_days != int(source_history_days):
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_MANIFEST_INVALID "
            f"source_history_days expected={source_history_days} actual={expected_days}"
        )
    for field, expected in (
        ("source_history_start", _iso_date(source_history_start)),
        ("source_history_end", _iso_date(source_history_end)),
    ):
        actual = str(history.get(field, ""))
        if actual != expected:
            raise D5PrecomputedSourceHistoryError(
                "D5_PRECOMPUTED_MANIFEST_INVALID "
                f"{field} expected={expected} actual={actual}"
            )
    actual_key_fields = tuple(str(field) for field in history.get("key_fields", ()))
    expected_key_fields = tuple(str(field) for field in key_fields)
    if actual_key_fields != expected_key_fields:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_MANIFEST_INVALID "
            f"key_fields expected={expected_key_fields!r} actual={actual_key_fields!r}"
        )
    frame_digest = str(history.get("source_history_frame_digest", ""))
    if len(frame_digest) != 64:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_MANIFEST_INVALID "
            "missing_or_invalid=source_history_frame_digest"
        )

    source_manifest = _require_mapping(manifest.get("source"), field="source")
    _require_hash_record(
        {"source": source_manifest},
        manifest_field="source",
        file_label="source.parquet",
        path=Path(source_path),
    )
    auxiliary_manifest = _require_mapping(
        manifest.get("auxiliary_csvs"),
        field="auxiliary_csvs",
    )
    for name in D5_PRECOMPUTED_AUXILIARY_FILES:
        evidence = authorities.files.get(name)
        if evidence is None or not evidence.used:
            raise D5PrecomputedSourceHistoryError(
                "D5_PRECOMPUTED_MANIFEST_INVALID "
                f"missing_current_authority={name}.csv"
            )
        _require_hash_record(
            auxiliary_manifest,
            manifest_field=name,
            file_label=f"{name}.csv",
            path=Path(evidence.path),
        )

    artifact = _require_mapping(manifest.get("artifact"), field="artifact")
    expected_artifact_hash = str(artifact.get("sha256", ""))
    if not expected_artifact_hash:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_MANIFEST_INVALID missing_sha256=prepared_180day_source.parquet"
        )
    actual_artifact_hash = _sha256_file(output_path)
    if actual_artifact_hash != expected_artifact_hash:
        raise D5PrecomputedSourceHistoryHashMismatch(
            file_label=D5_PRECOMPUTED_SOURCE_FILENAME,
            expected=expected_artifact_hash,
            actual=actual_artifact_hash,
        )

    frame = pd.read_parquet(output_path)
    expected_rows = int(artifact.get("row_count", -1))
    if len(frame) != expected_rows:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_ARTIFACT_INVALID "
            f"row_count expected={expected_rows} actual={len(frame)}"
        )
    expected_schema = artifact.get("schema")
    actual_schema = _frame_schema(frame)
    if actual_schema != expected_schema:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_ARTIFACT_INVALID "
            f"schema expected={expected_schema!r} actual={actual_schema!r}"
        )
    missing_columns = [field for field in (*expected_key_fields, "date") if field not in frame.columns]
    if missing_columns:
        raise D5PrecomputedSourceHistoryError(
            "D5_PRECOMPUTED_ARTIFACT_INVALID "
            f"missing_columns={missing_columns!r}"
        )
    return frame, manifest


__all__ = [
    "D5_FORCE_RECOMPUTE_ENV",
    "D5_PRECOMPUTED_AUXILIARY_FILES",
    "D5_PRECOMPUTED_MANIFEST_FILENAME",
    "D5_PRECOMPUTED_SOURCE_FILENAME",
    "D5PrecomputedSourceHistoryError",
    "D5PrecomputedSourceHistoryHashMismatch",
    "load_precomputed_d5_source_history",
]
