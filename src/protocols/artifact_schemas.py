"""Typed, versioned artifact schemas for the sealed D1--D6 run.

This module deliberately keeps the artifact boundary independent from pandas.
Records are validated as ordered mappings, then serialized through one canonical
policy.  A schema is identified by its full ``(name, version, digest)`` tuple;
the version alone is never sufficient to read an artifact.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ARTIFACT_SCHEMA_REGISTRY_VERSION = "artifact_schema_registry_v1"
SCHEMA_VERSION_V1 = "v1"
SCHEMA_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

PREDICTION_SEMANTIC_SORT_KEY = (
    "dataset_id",
    "scenario_enum_order",
    "target_entity_key_canonical",
    "method_enum_order",
    "seed",
    "forecast_origin",
    "horizon",
    "label_date",
)

DATASET_ENUM = ("D1", "D2", "D3", "D4", "D5", "D6")
SCENARIO_ENUM = ("without", "with")
METHOD_ENUM = (
    "No-TL",
    "SS-TL",
    "MSWA-TL",
    "MSSB-TL",
    "MSML-TL",
    "MSML-TL-RFE",
)

_SCHEMA_DIR = Path(__file__).with_name("schemas")

WORKER_PREDICTION_TRACE_SCHEMA_NAME = "WorkerPredictionTraceSchemaV1"
EVALUATED_PREDICTION_TRACE_SCHEMA_NAME = "EvaluatedPredictionTraceSchemaV1"
SOURCE_SELECTION_TRACE_SCHEMA_NAME = "SourceSelectionTraceSchemaV1"
FORMAL_RESULT_ROW_SCHEMA_NAME = "FormalResultRowSchemaV1"
WORKER_MANIFEST_SCHEMA_NAME = "WorkerManifestSchemaV1"
CELL_RESULT_MANIFEST_SCHEMA_NAME = "CellResultManifestSchemaV1"
RUN_MANIFEST_SCHEMA_NAME = "RunManifestSchemaV1"
PREFLIGHT_REPORT_SCHEMA_NAME = "PreflightReportSchemaV1"


class ArtifactSchemaError(ValueError):
    """Base class for fail-closed schema and artifact contract errors."""


class UnknownSchemaError(ArtifactSchemaError):
    """The name/version/digest tuple is not registered."""


class SchemaDefinitionDriftError(ArtifactSchemaError):
    """A known schema name/version was presented with a different digest."""


class SchemaVersionMixError(ArtifactSchemaError):
    """One run attempted to use multiple versions of one artifact type."""


class SchemaValidationError(ArtifactSchemaError):
    """A record or descriptor violates its registered schema."""


def _sha256_prefixed(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON using the repository-wide canonical JSON policy."""

    def default(item: Any) -> str:
        if isinstance(item, _datetime.datetime):
            if item.tzinfo is None:
                raise ValueError("naive datetime cannot be canonically serialized")
            utc = item.astimezone(_datetime.timezone.utc)
            return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
        if isinstance(item, _datetime.date):
            return item.isoformat()
        if hasattr(item, "item"):
            return default(item.item())
        raise TypeError("unsupported canonical JSON value: %s" % type(item).__name__)

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=default,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("value is not canonical JSON serializable") from exc
    return (text + "\n").encode("utf-8")


@dataclass(frozen=True)
class ArtifactField:
    """One ordered field in a typed artifact descriptor."""

    name: str
    arrow_dtype: str
    nullable: bool = False
    enum: Tuple[str, ...] = ()
    value_format: Optional[str] = None
    finite: bool = False
    nonnegative: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactField":
        try:
            return cls(
                name=str(payload["name"]),
                arrow_dtype=str(payload["arrow_dtype"]),
                nullable=bool(payload.get("nullable", False)),
                enum=tuple(str(value) for value in payload.get("enum", ())),
                value_format=(
                    None if payload.get("format") is None else str(payload.get("format"))
                ),
                finite=bool(payload.get("finite", False)),
                nonnegative=bool(payload.get("nonnegative", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaValidationError("invalid field descriptor") from exc

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "arrow_dtype": self.arrow_dtype,
            "nullable": self.nullable,
        }
        if self.enum:
            result["enum"] = list(self.enum)
        if self.value_format is not None:
            result["format"] = self.value_format
        if self.finite:
            result["finite"] = True
        if self.nonnegative:
            result["nonnegative"] = True
        return result

    @property
    def dtype(self) -> str:
        """Compatibility alias for callers that call Arrow dtype simply dtype."""
        return self.arrow_dtype

    @property
    def arrow_type(self) -> Any:
        """Return the corresponding pyarrow type without importing pyarrow at module load."""
        import pyarrow as pa

        return {
            "string": pa.string(),
            "date32": pa.date32(),
            "int8": pa.int8(),
            "int32": pa.int32(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
            # JSON fields are stored as canonical UTF-8 JSON text in CSV.
            "json": pa.string(),
        }[self.arrow_dtype]


def _date_value(value: Any, field_name: str) -> _datetime.date:
    if isinstance(value, _datetime.datetime):
        return value.date()
    if isinstance(value, _datetime.date):
        return value
    if hasattr(value, "date") and not isinstance(value, str):
        candidate = value.date()
        if isinstance(candidate, _datetime.date):
            return candidate
    if isinstance(value, str):
        try:
            parsed = _datetime.date.fromisoformat(value)
        except ValueError as exc:
            raise SchemaValidationError("%s must be an ISO date" % field_name) from exc
        return parsed
    raise SchemaValidationError("%s must be date32" % field_name)


def _scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


@dataclass(frozen=True)
class ArtifactSchemaDescriptor:
    """Immutable interpretation and ordering contract for one artifact type."""

    schema_name: str
    schema_version: str
    fields: Tuple[ArtifactField, ...]
    primary_key: Tuple[str, ...]
    physical_sort_key: Tuple[str, ...]
    semantic_columns: Tuple[str, ...]
    semantic_sort_key: Tuple[str, ...]
    serialization_policy: Mapping[str, Any]
    additional_properties: bool
    schema_digest: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSchemaDescriptor":
        required = (
            "schema_name",
            "schema_version",
            "fields",
            "primary_key",
            "physical_sort_key",
            "semantic_columns",
            "semantic_sort_key",
            "serialization_policy",
            "additionalProperties",
            "schema_digest",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise SchemaValidationError("descriptor missing fields: %s" % ", ".join(missing))
        fields = tuple(ArtifactField.from_dict(item) for item in payload["fields"])
        descriptor = cls(
            schema_name=str(payload["schema_name"]),
            schema_version=str(payload["schema_version"]),
            fields=fields,
            primary_key=tuple(str(value) for value in payload["primary_key"]),
            physical_sort_key=tuple(str(value) for value in payload["physical_sort_key"]),
            semantic_columns=tuple(str(value) for value in payload["semantic_columns"]),
            semantic_sort_key=tuple(str(value) for value in payload["semantic_sort_key"]),
            serialization_policy=dict(payload["serialization_policy"]),
            additional_properties=bool(payload["additionalProperties"]),
            schema_digest=str(payload["schema_digest"]),
        )
        descriptor._validate_descriptor_payload()
        expected = descriptor.computed_schema_digest
        if descriptor.schema_digest != expected:
            raise SchemaDefinitionDriftError(
                "%s declares %s but computes %s"
                % (descriptor.schema_name, descriptor.schema_digest, expected)
            )
        return descriptor

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def name(self) -> str:
        return self.schema_name

    @property
    def version(self) -> str:
        return self.schema_version

    @property
    def digest(self) -> str:
        return self.schema_digest

    @property
    def field_map(self) -> Dict[str, ArtifactField]:
        return {field.name: field for field in self.fields}

    @property
    def computed_schema_digest(self) -> str:
        return _sha256_prefixed(_canonical_json_bytes(self.to_dict(include_digest=False)))

    def to_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "fields": [field.to_dict() for field in self.fields],
            "primary_key": list(self.primary_key),
            "physical_sort_key": list(self.physical_sort_key),
            "semantic_columns": list(self.semantic_columns),
            "semantic_sort_key": list(self.semantic_sort_key),
            "serialization_policy": dict(self.serialization_policy),
            "additionalProperties": self.additional_properties,
        }
        if include_digest:
            result["schema_digest"] = self.schema_digest
        return result

    def descriptor_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict(include_digest=True))

    def to_pyarrow_schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [pa.field(field.name, field.arrow_type, nullable=field.nullable) for field in self.fields]
        )

    def _validate_descriptor_payload(self) -> None:
        if not self.schema_name or not self.schema_version:
            raise SchemaValidationError("schema name and version must be non-empty")
        if self.additional_properties is not False:
            raise SchemaValidationError("artifact schemas must set additionalProperties=false")
        if not SCHEMA_DIGEST_RE.fullmatch(self.schema_digest):
            raise SchemaValidationError("schema_digest must use sha256:<64 lowercase hex>")
        if not self.fields or len(self.field_names) != len(set(self.field_names)):
            raise SchemaValidationError("schema fields must be non-empty and unique")
        all_fields = set(self.field_names)
        for key_name, values in (
            ("primary_key", self.primary_key),
            ("physical_sort_key", self.physical_sort_key),
            ("semantic_columns", self.semantic_columns),
        ):
            if not values or not set(values).issubset(all_fields):
                raise SchemaValidationError("%s contains an unknown or empty field" % key_name)
        if not self.semantic_sort_key:
            raise SchemaValidationError("semantic_sort_key must be frozen and non-empty")
        if self.schema_name in {
            "WorkerPredictionTraceSchemaV1",
            "EvaluatedPredictionTraceSchemaV1",
        } and self.semantic_sort_key != PREDICTION_SEMANTIC_SORT_KEY:
            raise SchemaValidationError("prediction semantic_sort_key drifted")

    def validate_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(record, Mapping):
            raise SchemaValidationError("artifact row must be a mapping")
        actual = tuple(str(key) for key in record.keys())
        expected = self.field_names
        if actual != expected:
            extras = [key for key in actual if key not in expected]
            missing = [key for key in expected if key not in actual]
            if extras:
                raise SchemaValidationError("unknown field(s): %s" % ", ".join(extras))
            if missing:
                raise SchemaValidationError("missing field(s): %s" % ", ".join(missing))
            raise SchemaValidationError("artifact row must use the exact field order")
        validated: Dict[str, Any] = {}
        for field in self.fields:
            value = _scalar_value(record[field.name])
            if value is None:
                if field.nullable:
                    validated[field.name] = None
                    continue
                raise SchemaValidationError("%s is not nullable" % field.name)
            self._validate_value(field, value)
            validated[field.name] = value
        if self.schema_name in {
            "WorkerPredictionTraceSchemaV1",
            "EvaluatedPredictionTraceSchemaV1",
        }:
            if validated["seed"] not in (42, 43, 44, 45, 46):
                raise SchemaValidationError("seed is outside the formal seed set")
            if validated["horizon"] not in (1, 2, 3, 4, 5):
                raise SchemaValidationError("horizon is outside the formal horizon set")
        if self.schema_name == "FormalResultRowSchemaV1":
            if validated["status"] == "failed":
                metric_fields = ("rmse", "smape", "accuracy", "clipping_count")
                if any(validated[name] is not None for name in metric_fields):
                    raise SchemaValidationError("failed result rows cannot contain partial metrics")
            elif validated["failure_code"] != "NONE":
                raise SchemaValidationError("accepted result rows must use failure_code=NONE")
        if self.schema_name == "WorkerManifestSchemaV1":
            path = validated["artifact_path"]
            if not _is_safe_relative_path(path):
                raise SchemaValidationError("worker artifact_path must be a safe relative path")
        if self.schema_name == "CellResultManifestSchemaV1":
            for name in ("worker_trace_path", "evaluated_trace_path", "result_path"):
                if not _is_safe_relative_path(validated[name]):
                    raise SchemaValidationError("manifest path must be a safe relative path")
        return validated

    def validate_records(self, records: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
        materialized = tuple(self.validate_record(record) for record in records)
        primary_keys = set()
        for row in materialized:
            key = tuple(row[name] for name in self.primary_key)
            if key in primary_keys:
                raise SchemaValidationError("duplicate primary key: %r" % (key,))
            primary_keys.add(key)
        return materialized

    @staticmethod
    def _validate_value(field: ArtifactField, value: Any) -> None:
        dtype = field.arrow_dtype
        if field.enum and value not in field.enum:
            raise SchemaValidationError(
                "%s is outside its closed enum: %r" % (field.name, value)
            )
        if dtype == "string":
            if not isinstance(value, str):
                raise SchemaValidationError("%s must be string" % field.name)
        elif dtype == "date32":
            _date_value(value, field.name)
        elif dtype in ("int8", "int32"):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise SchemaValidationError("%s must be %s" % (field.name, dtype))
            number = int(value)
            bounds = (-128, 127) if dtype == "int8" else (-2147483648, 2147483647)
            if not bounds[0] <= number <= bounds[1]:
                raise SchemaValidationError("%s is outside %s range" % (field.name, dtype))
        elif dtype == "float64":
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise SchemaValidationError("%s must be float64" % field.name)
            number = float(value)
            if field.finite and not math.isfinite(number):
                raise SchemaValidationError("%s must be finite" % field.name)
            if field.nonnegative and number < 0:
                raise SchemaValidationError("%s must be nonnegative" % field.name)
        elif dtype == "bool":
            if not isinstance(value, bool):
                raise SchemaValidationError("%s must be bool" % field.name)
        elif dtype == "json":
            _canonical_json_bytes(value)
        else:
            raise SchemaValidationError("unsupported Arrow dtype: %s" % dtype)

        if field.value_format == "sha256" and (
            not isinstance(value, str) or not HEX_DIGEST_RE.fullmatch(value)
        ):
            raise SchemaValidationError("%s must be lowercase 64-hex" % field.name)
        if field.value_format == "canonical_entity_key":
            if not isinstance(value, str) or not value or value != value.strip() or "/" in value:
                raise SchemaValidationError("%s is not canonical" % field.name)
        if field.value_format == "posix_relative" and not _is_safe_relative_path(value):
            raise SchemaValidationError("%s must be a safe relative POSIX path" % field.name)


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    lowered = value.lower()
    forbidden = ("truth", "evaluator", "capability", "template", "reconstruct", "${", "$(")
    return not any(token in lowered for token in forbidden)


class ArtifactSchemaRegistry:
    """Exact schema registry and reader lookup boundary."""

    def __init__(self, descriptors: Iterable[ArtifactSchemaDescriptor]) -> None:
        self._by_name: Dict[str, Dict[str, ArtifactSchemaDescriptor]] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def schema_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def register(self, descriptor: ArtifactSchemaDescriptor) -> None:
        versions = self._by_name.setdefault(descriptor.schema_name, {})
        previous = versions.get(descriptor.schema_version)
        if previous is not None:
            if previous.schema_digest != descriptor.schema_digest:
                raise SchemaDefinitionDriftError(
                    "same-version schema drift: %s/%s" % (descriptor.schema_name, descriptor.schema_version)
                )
            return
        versions[descriptor.schema_version] = descriptor

    def get(self, schema_name: str, schema_version: Optional[str] = None) -> ArtifactSchemaDescriptor:
        try:
            versions = self._by_name[str(schema_name)]
        except KeyError as exc:
            raise UnknownSchemaError("unknown schema: %s" % schema_name) from exc
        if schema_version is not None:
            try:
                return versions[str(schema_version)]
            except KeyError as exc:
                raise UnknownSchemaError(
                    "unknown schema version: %s/%s" % (schema_name, schema_version)
                ) from exc
        if SCHEMA_VERSION_V1 in versions:
            return versions[SCHEMA_VERSION_V1]
        if len(versions) == 1:
            return next(iter(versions.values()))
        raise UnknownSchemaError("schema version is required for: %s" % schema_name)

    def resolve(
        self, schema_name: str, schema_version: str, schema_digest: str
    ) -> ArtifactSchemaDescriptor:
        descriptor = self.get(schema_name, schema_version)
        if descriptor.schema_digest != str(schema_digest):
            raise SchemaDefinitionDriftError(
                "schema digest drift: %s/%s" % (schema_name, schema_version)
            )
        return descriptor


def _load_registry() -> ArtifactSchemaRegistry:
    descriptors = []
    for path in sorted(_SCHEMA_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise SchemaValidationError("cannot read schema descriptor: %s" % path) from exc
        descriptors.append(ArtifactSchemaDescriptor.from_dict(payload))
    if len(descriptors) != 8:
        raise SchemaValidationError("expected 8 registered artifact schema descriptors")
    return ArtifactSchemaRegistry(descriptors)


_REGISTRY: Optional[ArtifactSchemaRegistry] = None


def get_artifact_schema_registry() -> ArtifactSchemaRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_registry()
    return _REGISTRY


def get_artifact_schema(schema_name: str) -> ArtifactSchemaDescriptor:
    return get_artifact_schema_registry().get(schema_name)


def get_worker_prediction_trace_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(WORKER_PREDICTION_TRACE_SCHEMA_NAME)


def get_evaluated_prediction_trace_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(EVALUATED_PREDICTION_TRACE_SCHEMA_NAME)


def get_source_selection_trace_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(SOURCE_SELECTION_TRACE_SCHEMA_NAME)


def get_formal_result_row_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(FORMAL_RESULT_ROW_SCHEMA_NAME)


def get_worker_manifest_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(WORKER_MANIFEST_SCHEMA_NAME)


def get_cell_result_manifest_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(CELL_RESULT_MANIFEST_SCHEMA_NAME)


def get_run_manifest_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(RUN_MANIFEST_SCHEMA_NAME)


def get_preflight_report_schema() -> ArtifactSchemaDescriptor:
    return get_artifact_schema(PREFLIGHT_REPORT_SCHEMA_NAME)


def validate_schema_reference(
    schema_name: str, schema_version: str, schema_digest: str
) -> ArtifactSchemaDescriptor:
    return get_artifact_schema_registry().resolve(schema_name, schema_version, schema_digest)


def artifact_schema_registry_digest() -> str:
    payload = [
        {
            "schema_name": name,
            "schema_version": get_artifact_schema(name).schema_version,
            "schema_digest": get_artifact_schema(name).schema_digest,
        }
        for name in get_artifact_schema_registry().schema_names
    ]
    return _sha256_prefixed(_canonical_json_bytes(payload))


def validate_run_schema_versions(references: Iterable[Mapping[str, Any]]) -> None:
    """Validate exact tuples and reject V1/V2 mixing per artifact type."""

    versions: Dict[str, str] = {}
    for reference in references:
        descriptor = validate_schema_reference(
            str(reference["schema_name"]),
            str(reference["schema_version"]),
            str(reference["schema_digest"]),
        )
        previous = versions.get(descriptor.schema_name)
        if previous is not None and previous != descriptor.schema_version:
            raise SchemaVersionMixError(
                "one run cannot mix %s and %s for %s"
                % (previous, descriptor.schema_version, descriptor.schema_name)
            )
        versions[descriptor.schema_name] = descriptor.schema_version


def schema_descriptor_path(schema_digest: str) -> Path:
    if not SCHEMA_DIGEST_RE.fullmatch(str(schema_digest)):
        raise SchemaValidationError("invalid schema digest path")
    return _SCHEMA_DIR / (str(schema_digest) + ".json")


def repository_descriptor_bytes(schema: ArtifactSchemaDescriptor) -> bytes:
    """Return the exact checked-in descriptor bytes for a registered schema."""
    for path in sorted(_SCHEMA_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("schema_name") == schema.schema_name:
            return path.read_bytes()
    raise UnknownSchemaError("repository descriptor missing: %s" % schema.schema_name)


__all__ = [
    "ARTIFACT_SCHEMA_REGISTRY_VERSION",
    "CELL_RESULT_MANIFEST_SCHEMA_NAME",
    "EVALUATED_PREDICTION_TRACE_SCHEMA_NAME",
    "FORMAL_RESULT_ROW_SCHEMA_NAME",
    "PREFLIGHT_REPORT_SCHEMA_NAME",
    "RUN_MANIFEST_SCHEMA_NAME",
    "SOURCE_SELECTION_TRACE_SCHEMA_NAME",
    "WORKER_MANIFEST_SCHEMA_NAME",
    "WORKER_PREDICTION_TRACE_SCHEMA_NAME",
    "ArtifactField",
    "ArtifactSchemaDescriptor",
    "ArtifactSchemaError",
    "ArtifactSchemaRegistry",
    "artifact_schema_registry_digest",
    "DATASET_ENUM",
    "METHOD_ENUM",
    "PREDICTION_SEMANTIC_SORT_KEY",
    "SCENARIO_ENUM",
    "SCHEMA_VERSION_V1",
    "SchemaDefinitionDriftError",
    "SchemaValidationError",
    "SchemaVersionMixError",
    "UnknownSchemaError",
    "get_artifact_schema",
    "get_artifact_schema_registry",
    "get_cell_result_manifest_schema",
    "get_evaluated_prediction_trace_schema",
    "get_formal_result_row_schema",
    "get_preflight_report_schema",
    "get_run_manifest_schema",
    "get_source_selection_trace_schema",
    "get_worker_manifest_schema",
    "get_worker_prediction_trace_schema",
    "repository_descriptor_bytes",
    "schema_descriptor_path",
    "validate_run_schema_versions",
    "validate_schema_reference",
]
