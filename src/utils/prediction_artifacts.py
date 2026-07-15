"""Deterministic validation, hashing, and publication of formal artifacts."""

from __future__ import annotations

import csv
import datetime as _datetime
import gzip
import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from src.protocols.artifact_schemas import (
    METHOD_ENUM,
    PREDICTION_SEMANTIC_SORT_KEY,
    SCENARIO_ENUM,
    ArtifactSchemaDescriptor,
    SchemaValidationError,
    get_artifact_schema,
    repository_descriptor_bytes,
)


class ArtifactPublicationError(RuntimeError):
    """An artifact failed validation, hashing, or atomic publication."""


@dataclass(frozen=True)
class ArtifactIdentity:
    schema_name: str
    schema_version: str
    schema_digest: str
    canonical_content_sha256: str
    artifact_sha256: str
    semantic_prediction_digest: str
    artifact_path: str
    artifact_bytes: int
    row_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def canonical_content_digest(self) -> str:
        return self.canonical_content_sha256

    @property
    def physical_artifact_sha256(self) -> str:
        return self.artifact_sha256


def _sha256_prefixed(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _date(value: Any) -> _datetime.date:
    if isinstance(value, _datetime.datetime):
        return value.date()
    if isinstance(value, _datetime.date):
        return value
    if isinstance(value, str):
        return _datetime.date.fromisoformat(value)
    if hasattr(value, "date"):
        candidate = value.date()
        if isinstance(candidate, _datetime.date):
            return candidate
    raise ArtifactPublicationError("expected a date32 value")


def _date_text(value: Any) -> str:
    return _date(value).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    def default(item: Any) -> str:
        if isinstance(item, _datetime.datetime):
            if item.tzinfo is None:
                raise ValueError("naive datetime cannot be serialized")
            return item.astimezone(_datetime.timezone.utc).isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z")
        if isinstance(item, _datetime.date):
            return item.isoformat()
        if hasattr(item, "item"):
            return default(item.item())
        raise TypeError(type(item).__name__)

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=default,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError("value is not canonical JSON") from exc


def _normalized_rows(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> Tuple[Dict[str, Any], ...]:
    try:
        return schema.validate_records(records)
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise ArtifactPublicationError("artifact records failed schema validation") from exc


def _sort_value(row: Mapping[str, Any], key: str) -> Any:
    if key == "scenario_enum_order":
        try:
            return SCENARIO_ENUM.index(str(row["scenario"]))
        except ValueError as exc:
            raise ArtifactPublicationError("scenario is not in the frozen enum") from exc
    if key == "method_enum_order":
        try:
            return METHOD_ENUM.index(str(row["method"]))
        except ValueError as exc:
            raise ArtifactPublicationError("method is not in the frozen enum") from exc
    if key == "target_entity_key_canonical":
        return str(row["target_entity_key"])
    value = row[key]
    if isinstance(value, (_datetime.date, _datetime.datetime)):
        return _date_text(value)
    if isinstance(value, (list, tuple, dict)):
        return _canonical_json_bytes(value)
    return value


def _sorted_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Tuple[Mapping[str, Any], ...]:
    return tuple(sorted(rows, key=lambda row: tuple(_sort_value(row, key) for key in keys)))


def _physical_rows(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> Tuple[Dict[str, Any], ...]:
    rows = _normalized_rows(records, schema)
    return tuple(_sorted_rows(rows, schema.physical_sort_key))  # type: ignore[return-value]


def _semantic_rows(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> Tuple[Dict[str, Any], ...]:
    rows = _normalized_rows(records, schema)
    return tuple(_sorted_rows(rows, schema.semantic_sort_key))  # type: ignore[return-value]


def _float_text(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactPublicationError("CSV cannot serialize non-finite float")
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _csv_cell(value: Any, dtype: str) -> str:
    if value is None:
        return "\\N"
    if dtype == "date32":
        return _date_text(value)
    if dtype == "float64":
        return _float_text(value)
    if dtype == "bool":
        return "true" if bool(value) else "false"
    if dtype == "json":
        return _canonical_json_bytes(value).decode("utf-8").rstrip("\n")
    return str(value)


def canonical_csv_bytes(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> bytes:
    """Return canonical uncompressed CSV bytes with frozen field ordering."""

    rows = _physical_rows(records, schema)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(schema.field_names)
    field_map = schema.field_map
    for row in rows:
        writer.writerow([_csv_cell(row[name], field_map[name].arrow_dtype) for name in schema.field_names])
    return output.getvalue().encode("utf-8")


def canonical_json_artifact_bytes(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> bytes:
    rows = _physical_rows(records, schema)
    return _canonical_json_bytes(list(rows))


def canonical_json_bytes(value: Any) -> bytes:
    """Public canonical JSON helper used by manifest publishers."""
    return _canonical_json_bytes(value)


def canonical_content_sha256(
    records: Iterable[Mapping[str, Any]],
    schema: Optional[ArtifactSchemaDescriptor] = None,
    *,
    format: str = "csv.gz",
) -> str:
    descriptor = schema or get_artifact_schema("WorkerPredictionTraceSchemaV1")
    content = (
        canonical_json_artifact_bytes(records, descriptor)
        if format == "json"
        else canonical_csv_bytes(records, descriptor)
    )
    return _sha256_prefixed(content)


def semantic_prediction_sha256(
    records: Iterable[Mapping[str, Any]],
    schema: Optional[ArtifactSchemaDescriptor] = None,
) -> str:
    descriptor = schema or get_artifact_schema("WorkerPredictionTraceSchemaV1")
    if descriptor.schema_name not in {
        "WorkerPredictionTraceSchemaV1",
        "EvaluatedPredictionTraceSchemaV1",
    }:
        raise ArtifactPublicationError("semantic prediction digest requires a prediction trace")
    rows = _semantic_rows(records, descriptor)
    seen = set()
    for row in rows:
        key = tuple(_sort_value(row, name) for name in PREDICTION_SEMANTIC_SORT_KEY)
        if key in seen:
            raise ArtifactPublicationError("duplicate prediction semantic sort key")
        seen.add(key)
    projected = [
        {name: row[name] for name in descriptor.semantic_columns}
        for row in rows
    ]
    return _sha256_prefixed(_canonical_json_bytes(projected))


def _gzip_deterministic(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0
    ) as handle:
        handle.write(payload)
    return buffer.getvalue()


def canonical_gzip_csv_bytes(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor
) -> bytes:
    return _gzip_deterministic(canonical_csv_bytes(records, schema))


def _artifact_bytes(
    records: Iterable[Mapping[str, Any]], schema: ArtifactSchemaDescriptor, format: str
) -> Tuple[bytes, bytes]:
    if format == "json":
        canonical = canonical_json_artifact_bytes(records, schema)
        return canonical, canonical
    if format not in {"csv.gz", "csv"}:
        raise ArtifactPublicationError("unsupported artifact format: %s" % format)
    canonical = canonical_csv_bytes(records, schema)
    return canonical, _gzip_deterministic(canonical) if format == "csv.gz" else canonical


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%s.%s" % (path.name, os.getpid(), uuid4().hex))
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def copy_schema_descriptor(run_root: Path, schema: ArtifactSchemaDescriptor) -> Path:
    """Copy the exact repository descriptor into ``run_root/schemas``."""

    destination = Path(run_root) / "schemas" / (schema.schema_digest + ".json")
    descriptor_bytes = repository_descriptor_bytes(schema)
    if destination.exists():
        if destination.read_bytes() != descriptor_bytes:
            raise ArtifactPublicationError("run schema descriptor bytes drifted")
        return destination
    _atomic_bytes(destination, descriptor_bytes)
    return destination


def publish_prediction_artifact(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    schema: Optional[ArtifactSchemaDescriptor] = None,
    schema_name: str = "WorkerPredictionTraceSchemaV1",
    run_root: Optional[Path] = None,
    format: Optional[str] = None,
) -> ArtifactIdentity:
    """Validate, hash, and atomically publish a typed artifact."""

    descriptor = schema or get_artifact_schema(schema_name)
    destination = Path(path)
    chosen_format = format or ("csv.gz" if str(destination).endswith(".csv.gz") else "json" if destination.suffix == ".json" else "csv.gz")
    try:
        rows = _normalized_rows(records, descriptor)
        canonical, physical = _artifact_bytes(rows, descriptor, chosen_format)
        semantic = (
            semantic_prediction_sha256(rows, descriptor)
            if descriptor.schema_name in {"WorkerPredictionTraceSchemaV1", "EvaluatedPredictionTraceSchemaV1"}
            else "sha256:" + "0" * 64
        )
        if run_root is not None:
            copy_schema_descriptor(Path(run_root), descriptor)
        _atomic_bytes(destination, physical)
        if destination.read_bytes() != physical:
            raise ArtifactPublicationError("published artifact bytes differ from candidate")
        return ArtifactIdentity(
            schema_name=descriptor.schema_name,
            schema_version=descriptor.schema_version,
            schema_digest=descriptor.schema_digest,
            canonical_content_sha256=_sha256_prefixed(canonical),
            artifact_sha256=_sha256_prefixed(physical),
            semantic_prediction_digest=semantic,
            artifact_path=str(destination),
            artifact_bytes=len(physical),
            row_count=len(rows),
        )
    except (SchemaValidationError, ArtifactPublicationError):
        raise
    except Exception as exc:
        raise ArtifactPublicationError("artifact publication failed") from exc


def _parse_csv_cell(value: str, dtype: str) -> Any:
    if value == "\\N":
        return None
    if dtype == "date32":
        return _datetime.date.fromisoformat(value)
    if dtype == "int8" or dtype == "int32":
        return int(value)
    if dtype == "float64":
        return float(value)
    if dtype == "bool":
        if value == "true":
            return True
        if value == "false":
            return False
        raise ArtifactPublicationError("invalid canonical boolean")
    if dtype == "json":
        return json.loads(value)
    return value


def _read_logical_bytes(path: Path) -> Tuple[bytes, str]:
    physical = path.read_bytes()
    if str(path).endswith(".csv.gz"):
        try:
            return gzip.decompress(physical), "csv.gz"
        except OSError as exc:
            raise ArtifactPublicationError("artifact gzip bytes are corrupt") from exc
    if path.suffix == ".json":
        return physical, "json"
    return physical, "csv"


def _read_records(path: Path, schema: ArtifactSchemaDescriptor) -> Tuple[Dict[str, Any], ...]:
    logical, format = _read_logical_bytes(path)
    try:
        if format == "json":
            value = json.loads(logical.decode("utf-8"))
            if not isinstance(value, list):
                raise ArtifactPublicationError("JSON artifact must contain a record list")
            return _normalized_rows(value, schema)
        text = logical.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != schema.field_names:
            raise ArtifactPublicationError("CSV header does not match the exact schema order")
        field_map = schema.field_map
        rows = []
        for row in reader:
            rows.append(
                {
                    name: _parse_csv_cell(row[name], field_map[name].arrow_dtype)
                    for name in schema.field_names
                }
            )
        return _normalized_rows(rows, schema)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        raise ArtifactPublicationError("artifact logical bytes are corrupt") from exc


def _expected_value(expected: Any, key: str) -> Optional[str]:
    if expected is None:
        return None
    if isinstance(expected, ArtifactIdentity):
        return getattr(expected, key)
    if isinstance(expected, Mapping):
        value = expected.get(key)
        return None if value is None else str(value)
    return None


def read_prediction_artifact(
    path: Path,
    *,
    expected: Optional[Any] = None,
    schema: Optional[ArtifactSchemaDescriptor] = None,
    schema_name: str = "WorkerPredictionTraceSchemaV1",
) -> Tuple[Dict[str, Any], ...]:
    """Read and revalidate a published artifact without invoking model code."""

    destination = Path(path)
    if not destination.is_file():
        raise ArtifactPublicationError("artifact is missing: %s" % destination)
    descriptor = schema or get_artifact_schema(schema_name)
    physical = destination.read_bytes()
    expected_artifact = _expected_value(expected, "artifact_sha256")
    if expected_artifact is not None and expected_artifact != _sha256_prefixed(physical):
        raise ArtifactPublicationError("artifact SHA-256 does not match its identity")
    rows = _read_records(destination, descriptor)
    chosen_format = "json" if destination.suffix == ".json" else "csv.gz" if str(destination).endswith(".csv.gz") else "csv"
    canonical = canonical_content_sha256(rows, descriptor, format=chosen_format)
    semantic = (
        semantic_prediction_sha256(rows, descriptor)
        if descriptor.schema_name in {"WorkerPredictionTraceSchemaV1", "EvaluatedPredictionTraceSchemaV1"}
        else "sha256:" + "0" * 64
    )
    expected_canonical = _expected_value(expected, "canonical_content_sha256")
    expected_semantic = _expected_value(expected, "semantic_prediction_digest")
    if expected_canonical is not None and expected_canonical != canonical:
        raise ArtifactPublicationError("canonical content SHA-256 does not match its identity")
    if expected_semantic is not None and expected_semantic != semantic:
        raise ArtifactPublicationError("semantic prediction digest does not match its identity")
    return rows


def write_prediction_artifact(*args: Any, **kwargs: Any) -> ArtifactIdentity:
    """Compatibility name for the typed atomic publisher."""
    return publish_prediction_artifact(*args, **kwargs)


def load_prediction_artifact(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], ...]:
    return read_prediction_artifact(*args, **kwargs)


def semantic_prediction_digest(
    records: Iterable[Mapping[str, Any]],
    schema: Optional[ArtifactSchemaDescriptor] = None,
) -> str:
    return semantic_prediction_sha256(records, schema)


def validate_worker_trace(records: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")
    rows = _normalized_rows(records, schema)
    for row in rows:
        if "y_true" in row:
            raise SchemaValidationError("worker trace may not contain y_true")
    return rows


def validate_evaluated_trace(records: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    schema = get_artifact_schema("EvaluatedPredictionTraceSchemaV1")
    return _normalized_rows(records, schema)


def join_worker_trace_with_truth(
    worker_records: Iterable[Mapping[str, Any]],
    truth_records: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Join evaluator-only truth exactly once to each validated worker row."""

    workers = validate_worker_trace(worker_records)
    truth_by_key: Dict[str, Mapping[str, Any]] = {}
    for truth in truth_records:
        key = truth.get("truth_key")
        if not isinstance(key, str) or len(key) != 64 or key.lower() != key:
            raise SchemaValidationError("truth record has an invalid truth_key")
        if key in truth_by_key:
            raise SchemaValidationError("truth join is not one-to-one: duplicate truth_key")
        truth_by_key[key] = truth
    evaluated = []
    for worker in workers:
        truth = truth_by_key.get(worker["truth_key"])
        if truth is None:
            raise SchemaValidationError("truth join is not one-to-one: missing truth_key")
        if _date_text(truth.get("label_date")) != _date_text(worker["label_date"]):
            raise SchemaValidationError("truth label_date does not match worker trace")
        if str(truth.get("target_entity_key")) != worker["target_entity_key"]:
            raise SchemaValidationError("truth target_entity_key does not match worker trace")
        y_true = truth.get("y_true")
        if isinstance(y_true, bool) or not isinstance(y_true, (int, float)) or not math.isfinite(float(y_true)) or float(y_true) < 0:
            raise SchemaValidationError("evaluator truth must be finite and nonnegative")
        row = dict(worker)
        row["y_true"] = float(y_true)
        row["is_synthetic_date"] = bool(truth.get("is_synthetic_date"))
        row["evaluator_join_status"] = "matched"
        evaluated.append(row)
    return validate_evaluated_trace(evaluated)


def derive_formal_result_row(
    evaluated_records: Iterable[Mapping[str, Any]],
    *,
    trace_identity: Optional[Mapping[str, Any]] = None,
    status: str = "accepted",
    failure_code: str = "NONE",
) -> Dict[str, Any]:
    """Derive one result row from evaluator trace rows only."""

    rows = validate_evaluated_trace(evaluated_records)
    if not rows:
        raise SchemaValidationError("cannot derive a result row from an empty trace")
    if status == "accepted" and any(row["evaluator_join_status"] != "matched" for row in rows):
        raise SchemaValidationError("accepted result requires matched evaluator joins")
    first = rows[0]
    identity = dict(trace_identity or {})
    y_true = [float(row["y_true"]) for row in rows]
    y_pred = [float(row["y_pred_clipped"]) for row in rows]
    errors = [prediction - truth for prediction, truth in zip(y_pred, y_true)]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    smape = 100.0 * sum(
        2.0 * abs(error) / (abs(truth) + abs(prediction) + 1e-8)
        for error, truth, prediction in zip(errors, y_true, y_pred)
    ) / len(errors)
    defaults = {
        "source_selection_identity": "0" * 64,
        "predictor_feature_schema_digest": "0" * 64,
        "feature_mask_digest": "0" * 64,
        "protocol_identity": "0" * 64,
        "input_identity": "0" * 64,
        "code_identity": "0" * 64,
    }
    result: Dict[str, Any] = {
        "dataset_id": first["dataset_id"],
        "scenario": first["scenario"],
        "target_entity_key": first["target_entity_key"],
        "method": first["method"],
        "seed": first["seed"],
        "horizon": first["horizon"],
        "sample_count": len(rows),
        "forecast_origin_start": min(row["forecast_origin"] for row in rows),
        "forecast_origin_end": max(row["forecast_origin"] for row in rows),
        "label_date_start": min(row["label_date"] for row in rows),
        "label_date_end": max(row["label_date"] for row in rows),
        "rmse": rmse if status == "accepted" else None,
        "smape": smape if status == "accepted" else None,
        "accuracy": (1.0 / (rmse + 1e-8)) if status == "accepted" else None,
        "clipping_count": sum(1 for row in rows if row["was_clipped"]),
        "accepted_trace_path": identity.get("artifact_path") if status == "accepted" else None,
        "accepted_trace_sha256": identity.get("artifact_sha256") if status == "accepted" else None,
        "accepted_trace_canonical_content_sha256": identity.get("canonical_content_sha256") if status == "accepted" else None,
        "accepted_trace_artifact_sha256": identity.get("artifact_sha256") if status == "accepted" else None,
        "semantic_prediction_digest": identity.get("semantic_prediction_digest") if status == "accepted" else None,
        "source_selection_identity": identity.get("source_selection_identity", defaults["source_selection_identity"]),
        "predictor_feature_schema_digest": identity.get("predictor_feature_schema_digest", defaults["predictor_feature_schema_digest"]),
        "feature_mask_digest": identity.get("feature_mask_digest", defaults["feature_mask_digest"]),
        "protocol_identity": identity.get("protocol_identity", defaults["protocol_identity"]),
        "input_identity": identity.get("input_identity", defaults["input_identity"]),
        "code_identity": identity.get("code_identity", defaults["code_identity"]),
        "status": status,
        "failure_code": failure_code,
    }
    return get_artifact_schema("FormalResultRowSchemaV1").validate_record(result)


__all__ = [
    "ArtifactIdentity",
    "ArtifactPublicationError",
    "canonical_content_sha256",
    "canonical_csv_bytes",
    "canonical_gzip_csv_bytes",
    "canonical_json_bytes",
    "canonical_json_artifact_bytes",
    "copy_schema_descriptor",
    "derive_formal_result_row",
    "join_worker_trace_with_truth",
    "load_prediction_artifact",
    "publish_prediction_artifact",
    "read_prediction_artifact",
    "semantic_prediction_digest",
    "semantic_prediction_sha256",
    "validate_evaluated_trace",
    "validate_worker_trace",
    "write_prediction_artifact",
]
