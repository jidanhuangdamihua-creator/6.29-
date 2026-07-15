"""Frozen predictor, KNN, RFE-mask, and future-known lineage contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .experiment_protocol import FORMAL_METHODS, ProtocolViolation, normalize_scenario
from .sealing_protocol import (
    FORMAL_HORIZONS,
    FORMAL_SEEDS,
    SEALING_PROTOCOL_VERSION,
    get_target_window,
    normalize_dataset_id,
)


FEATURE_SCHEMA_VERSION = "predictor_feature_schema_v1"
KNN_SCHEMA_VERSION = "knn_observed_schema_v1"
MASK_SCHEMA_VERSION = "predictor_feature_mask_v1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError) as exc:
        raise FutureKnownLineageViolation(f"invalid lineage date: {value!r}") from exc


class FeatureRole(str, Enum):
    TARGET_SIGNAL = "target_signal"
    FUTURE_KNOWN = "future_known"
    STATIC_KNOWN = "static_known"
    OBSERVED_DYNAMIC = "observed_dynamic"
    RECURSIVE_DERIVED = "recursive_derived"
    EVALUATION_ONLY = "evaluation_only"
    IDENTIFIER_GROUP_ONLY = "identifier_group_only"


class KnnObservedDispositionV1(str, Enum):
    KNN_OBSERVED = "knn_observed"
    AUDIT_ONLY = "audit_only"


@dataclass(frozen=True)
class PredictorFeature:
    name: str
    dtype: str
    role: FeatureRole
    transform: str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        dtype = str(self.dtype).strip()
        transform = str(self.transform).strip()
        if not name or not dtype or not transform:
            raise ProtocolViolation("predictor feature name, dtype, and transform must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "role", FeatureRole(self.role))
        object.__setattr__(self, "transform", transform)

    def descriptor(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role.value,
            "transform": self.transform,
        }


@dataclass(frozen=True)
class PredictorFeatureSchema:
    dataset_id: str
    fields: Tuple[PredictorFeature, ...]
    version: str = FEATURE_SCHEMA_VERSION
    protocol_version: str = SEALING_PROTOCOL_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        dataset_id = normalize_dataset_id(self.dataset_id)
        fields = tuple(self.fields)
        names = tuple(item.name for item in fields)
        if not fields or len(names) != len(set(names)):
            raise ProtocolViolation(f"{dataset_id} predictor schema requires unique fields")
        if names[0] != "sales" or fields[0].role is not FeatureRole.TARGET_SIGNAL:
            raise ProtocolViolation(f"{dataset_id} predictor schema must start with target_signal sales")
        forbidden_roles = {
            FeatureRole.OBSERVED_DYNAMIC,
            FeatureRole.EVALUATION_ONLY,
            FeatureRole.IDENTIFIER_GROUP_ONLY,
        }
        if any(item.role in forbidden_roles for item in fields):
            raise ProtocolViolation(f"{dataset_id} predictor schema contains a forbidden feature role")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "digest", _canonical_digest(self.descriptor(include_digest=False)))

    @property
    def ordered_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.fields)

    @property
    def ordered_dtypes(self) -> Tuple[str, ...]:
        return tuple(item.dtype for item in self.fields)

    @property
    def ordered_roles(self) -> Tuple[FeatureRole, ...]:
        return tuple(item.role for item in self.fields)

    @property
    def ordered_transforms(self) -> Tuple[str, ...]:
        return tuple(item.transform for item in self.fields)

    @property
    def dimension(self) -> int:
        return len(self.fields)

    def field(self, name: str) -> PredictorFeature:
        try:
            return self.fields[self.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def index(self, name: str) -> int:
        return self.ordered_names.index(str(name))

    def descriptor(self, *, include_digest: bool = True) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "version": self.version,
            "protocol_version": self.protocol_version,
            "dataset_id": self.dataset_id,
            "dimension": len(self.fields),
            "fields": [item.descriptor() for item in self.fields],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class KnnObservedFieldV1:
    name: str
    dtype: str
    disposition: KnnObservedDispositionV1

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        dtype = str(self.dtype).strip()
        if not name or not dtype:
            raise ProtocolViolation("KNN field name and dtype must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "disposition", KnnObservedDispositionV1(self.disposition))

    def descriptor(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True)
class KnnFeatureSchema:
    dataset_id: str
    fields: Tuple[KnnObservedFieldV1, ...]
    version: str = KNN_SCHEMA_VERSION
    protocol_version: str = SEALING_PROTOCOL_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        dataset_id = normalize_dataset_id(self.dataset_id)
        fields = tuple(self.fields)
        names = tuple(item.name for item in fields)
        if not fields or len(names) != len(set(names)):
            raise ProtocolViolation(f"{dataset_id} KNN schema requires unique classified fields")
        if fields[0].name != "sales" or (
            fields[0].disposition is not KnnObservedDispositionV1.KNN_OBSERVED
        ):
            raise ProtocolViolation(f"{dataset_id} KNN schema must start with knn_observed sales")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "digest", _canonical_digest(self.descriptor(include_digest=False)))

    @property
    def ordered_names(self) -> Tuple[str, ...]:
        return tuple(
            item.name
            for item in self.fields
            if item.disposition is KnnObservedDispositionV1.KNN_OBSERVED
        )

    @property
    def ordered_dtypes(self) -> Tuple[str, ...]:
        return tuple(
            item.dtype
            for item in self.fields
            if item.disposition is KnnObservedDispositionV1.KNN_OBSERVED
        )

    @property
    def classified_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.fields)

    @property
    def dimension(self) -> int:
        return len(self.ordered_names)

    def field(self, name: str) -> KnnObservedFieldV1:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(name)

    def descriptor(self, *, include_digest: bool = True) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "version": self.version,
            "protocol_version": self.protocol_version,
            "dataset_id": self.dataset_id,
            "dimension": self.dimension,
            "fields": [item.descriptor() for item in self.fields],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class PredictorFeatureMask:
    schema_digest: str
    values: Tuple[bool, ...]
    ordered_names: Tuple[str, ...]
    version: str = MASK_SCHEMA_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        schema_digest = str(self.schema_digest).strip().lower()
        values = tuple(bool(value) for value in self.values)
        names = tuple(str(name).strip() for name in self.ordered_names)
        if not re.fullmatch(r"[0-9a-f]{64}", schema_digest):
            raise ValueError("schema_digest must be a SHA-256 digest")
        if not names or len(values) != len(names):
            raise ValueError("predictor mask must have the same length as the full schema")
        if len(names) != len(set(names)):
            raise ValueError("predictor mask names must be unique and ordered")
        if "sales" not in names or not values[names.index("sales")]:
            raise ValueError("predictor mask must always retain sales")
        object.__setattr__(self, "schema_digest", schema_digest)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "ordered_names", names)
        object.__setattr__(
            self,
            "digest",
            _canonical_digest(
                {
                    "version": self.version,
                    "schema_digest": schema_digest,
                    "ordered_names": list(names),
                    "values": list(values),
                }
            ),
        )

    @classmethod
    def full(cls, schema: PredictorFeatureSchema) -> "PredictorFeatureMask":
        return cls(schema.digest, (True,) * schema.dimension, schema.ordered_names)

    @classmethod
    def from_selected_names(
        cls,
        schema: PredictorFeatureSchema,
        selected_names: Iterable[str],
    ) -> "PredictorFeatureMask":
        selected = {str(name).strip() for name in selected_names}
        unknown = selected.difference(schema.ordered_names)
        if unknown:
            raise ValueError(f"RFE selected unknown predictor fields: {sorted(unknown)}")
        selected.add("sales")
        return cls(
            schema.digest,
            tuple(name in selected for name in schema.ordered_names),
            schema.ordered_names,
        )

    @property
    def selected_names(self) -> Tuple[str, ...]:
        return tuple(name for name, selected in zip(self.ordered_names, self.values) if selected)

    def apply(self, transformed_tensor: object):
        """Zero unselected transformed fields without deleting or reordering columns."""

        import numpy as np

        values = np.asarray(transformed_tensor)
        if values.ndim == 0 or values.shape[-1] != len(self.values):
            raise ValueError("transformed tensor final dimension does not match predictor mask")
        mask = np.asarray(self.values, dtype=bool)
        return np.where(mask, values, np.zeros((), dtype=values.dtype))


class FutureKnownLineageViolation(ProtocolViolation):
    """Raised when future-known authority or dependencies fail closed."""


@dataclass(frozen=True)
class FutureKnownLineage:
    feature_name: str
    source_type: str
    authority: str
    available_at: date
    dependencies: Tuple[str, ...]
    generation_rule: str
    code_digest: str

    def __post_init__(self) -> None:
        feature_name = str(self.feature_name).strip()
        source_type = str(self.source_type).strip()
        authority = str(self.authority).strip()
        generation_rule = str(self.generation_rule).strip()
        dependencies = tuple(str(value).strip() for value in self.dependencies)
        code_digest = str(self.code_digest).strip().lower()
        if not feature_name:
            raise FutureKnownLineageViolation("future-known feature_name is required")
        if not source_type:
            raise FutureKnownLineageViolation(f"{feature_name} source_type is required")
        if not authority:
            raise FutureKnownLineageViolation(f"{feature_name} authority is required")
        if not generation_rule:
            raise FutureKnownLineageViolation(f"{feature_name} generation_rule is required")
        if not dependencies or any(not value for value in dependencies):
            raise FutureKnownLineageViolation(f"{feature_name} dependencies must be explicit")
        if not re.fullmatch(r"[0-9a-f]{64}", code_digest):
            raise FutureKnownLineageViolation(f"{feature_name} code_digest must be a SHA-256 digest")
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "available_at", _as_date(self.available_at))
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "generation_rule", generation_rule)
        object.__setattr__(self, "code_digest", code_digest)

    @property
    def availability_cutoff(self) -> date:
        return self.available_at

    def descriptor(self) -> Dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "source_type": self.source_type,
            "authority": self.authority,
            "available_at": self.available_at.isoformat(),
            "dependencies": list(self.dependencies),
            "generation_rule": self.generation_rule,
            "code_digest": self.code_digest,
        }


@dataclass(frozen=True)
class FutureKnownLineageAuditReport:
    valid: bool
    schema_digest: str
    cutoff: date
    audited_fields: Tuple[str, ...]
    lineage_digest: str


_FORBIDDEN_DEPENDENCY_PATTERN = re.compile(
    r"(^|[^a-z0-9])(sales?|y_true|truth|predictions?|y_pred)([^a-z0-9]|$)",
    re.IGNORECASE,
)


def audit_future_known_lineage(
    schema: PredictorFeatureSchema,
    lineage: Sequence[FutureKnownLineage],
    *,
    cutoff: object,
) -> FutureKnownLineageAuditReport:
    """Audit exact coverage, independent authority, cutoff, and dependency safety."""

    cutoff_date = _as_date(cutoff)
    items = tuple(lineage)
    names = tuple(item.feature_name for item in items)
    expected = tuple(
        item.name for item in schema.fields if item.role is FeatureRole.FUTURE_KNOWN
    )
    if names != expected:
        raise FutureKnownLineageViolation(
            f"future-known lineage must exactly follow schema order: expected={expected!r} got={names!r}"
        )
    if len(names) != len(set(names)):
        raise FutureKnownLineageViolation("future-known lineage contains duplicate fields")

    for item in items:
        if not item.authority:
            raise FutureKnownLineageViolation(f"{item.feature_name} authority is required")
        if item.available_at > cutoff_date:
            raise FutureKnownLineageViolation(
                f"{item.feature_name} is available after cutoff {cutoff_date.isoformat()}"
            )
        for dependency in item.dependencies:
            normalized = dependency.strip().lower().replace("-", "_")
            if _FORBIDDEN_DEPENDENCY_PATTERN.search(normalized) or (
                "sales" in normalized and normalized != ""
            ):
                raise FutureKnownLineageViolation(
                    f"{item.feature_name} has forbidden dependency {dependency!r}"
                )
            if normalized in {"prediction", "predicted", "forecast", "label"}:
                raise FutureKnownLineageViolation(
                    f"{item.feature_name} has forbidden dependency {dependency!r}"
                )

    payload = [item.descriptor() for item in items]
    return FutureKnownLineageAuditReport(
        valid=True,
        schema_digest=schema.digest,
        cutoff=cutoff_date,
        audited_fields=names,
        lineage_digest=_canonical_digest(payload),
    )


def _feature(
    name: str,
    role: FeatureRole,
    *,
    dtype: str = "float64",
    transform: str = "minmax_fit_observed_only",
) -> PredictorFeature:
    return PredictorFeature(name=name, dtype=dtype, role=role, transform=transform)


def _calendar(name: str) -> PredictorFeature:
    return _feature(name, FeatureRole.FUTURE_KNOWN, dtype="int64", transform="identity")


def _base_fields() -> Tuple[PredictorFeature, ...]:
    return (
        _feature("sales", FeatureRole.TARGET_SIGNAL),
        _calendar("year"),
        _calendar("month"),
        _calendar("week"),
        _calendar("day"),
    )


_PREDICTOR_SCHEMAS: Mapping[str, PredictorFeatureSchema] = MappingProxyType(
    {
        "D1": PredictorFeatureSchema("D1", _base_fields()),
        "D2": PredictorFeatureSchema("D2", _base_fields()),
        "D3": PredictorFeatureSchema("D3", _base_fields() + (_calendar("SchoolHoliday"),)),
        "D4": PredictorFeatureSchema("D4", _base_fields() + (_calendar("holiday_flag"),)),
        "D5": PredictorFeatureSchema(
            "D5",
            _base_fields()
            + (
                _feature("perishable", FeatureRole.STATIC_KNOWN, dtype="int64", transform="identity"),
                _calendar("is_holiday"),
            ),
        ),
        "D6": PredictorFeatureSchema(
            "D6",
            _base_fields()
            + (
                _calendar("weekday"),
                _calendar("is_event_1"),
                _calendar("is_event_2"),
                _calendar("snap"),
            ),
        ),
    }
)


def _knn_field(
    name: str,
    disposition: KnnObservedDispositionV1 = KnnObservedDispositionV1.KNN_OBSERVED,
    *,
    dtype: str = "float64",
) -> KnnObservedFieldV1:
    return KnnObservedFieldV1(name, dtype, disposition)


_AUDIT = KnnObservedDispositionV1.AUDIT_ONLY
_KNN_SCHEMAS: Mapping[str, KnnFeatureSchema] = MappingProxyType(
    {
        "D1": KnnFeatureSchema("D1", (_knn_field("sales"),)),
        "D2": KnnFeatureSchema("D2", (_knn_field("sales"), _knn_field("promo"))),
        "D3": KnnFeatureSchema(
            "D3",
            (
                _knn_field("sales"),
                _knn_field("Customers"),
                _knn_field("Open"),
                _knn_field("Promo"),
            ),
        ),
        "D4": KnnFeatureSchema(
            "D4",
            (
                _knn_field("sales"),
                _knn_field("stock_hour6_22_cnt", _AUDIT),
                _knn_field("activity_flag", _AUDIT),
                _knn_field("discount", _AUDIT),
                _knn_field("precpt", _AUDIT),
                _knn_field("avg_temperature", _AUDIT),
                _knn_field("avg_humidity", _AUDIT),
                _knn_field("avg_wind_level", _AUDIT),
                _knn_field("hours_sale_sum_leakage_risk", _AUDIT),
                _knn_field("hours_sale_max_leakage_risk", _AUDIT),
                _knn_field("hours_sale_nonzero_hours_leakage_risk", _AUDIT),
                _knn_field("hours_stock_sum_leakage_risk", _AUDIT),
                _knn_field("hours_stock_max_leakage_risk", _AUDIT),
                _knn_field("hours_stock_nonzero_hours_leakage_risk", _AUDIT),
            ),
        ),
        "D5": KnnFeatureSchema(
            "D5",
            (
                _knn_field("sales"),
                _knn_field("onpromotion"),
                _knn_field("transactions"),
                _knn_field("oil_price"),
            ),
        ),
        "D6": KnnFeatureSchema("D6", (_knn_field("sales"), _knn_field("sell_price"))),
    }
)


def _validate_context(
    *,
    method: Optional[str],
    scenario: Optional[str],
    domain: Optional[str],
    partition: Optional[str],
    horizon: Optional[int],
    seed: Optional[int],
) -> None:
    if method is not None and str(method) not in FORMAL_METHODS:
        raise ProtocolViolation(f"unsupported formal method: {method!r}")
    if scenario is not None:
        normalize_scenario(scenario)
    if domain is not None and str(domain).strip().lower() not in {"source", "target"}:
        raise ProtocolViolation(f"unsupported feature domain: {domain!r}")
    if partition is not None and str(partition).strip().lower() not in {
        "train",
        "validation",
        "blind",
    }:
        raise ProtocolViolation(f"unsupported feature partition: {partition!r}")
    if horizon is not None and int(horizon) not in FORMAL_HORIZONS:
        raise ProtocolViolation(f"unsupported formal horizon: {horizon!r}")
    if seed is not None and int(seed) not in FORMAL_SEEDS:
        raise ProtocolViolation(f"unsupported formal seed: {seed!r}")


def get_predictor_schema(
    dataset_id: object,
    *,
    method: Optional[str] = None,
    scenario: Optional[str] = None,
    domain: Optional[str] = None,
    partition: Optional[str] = None,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
) -> PredictorFeatureSchema:
    """Return the one immutable schema after validating optional formal context."""

    _validate_context(
        method=method,
        scenario=scenario,
        domain=domain,
        partition=partition,
        horizon=horizon,
        seed=seed,
    )
    return _PREDICTOR_SCHEMAS[normalize_dataset_id(dataset_id)]


def get_knn_schema(dataset_id: object) -> KnnFeatureSchema:
    return _KNN_SCHEMAS[normalize_dataset_id(dataset_id)]


_CALENDAR_RULES = {
    "year": "date.year",
    "month": "date.month",
    "week": "date.isocalendar().week",
    "day": "date.day",
    "weekday": "sealed_calendar.weekday",
    "SchoolHoliday": "sealed_calendar.school_holiday",
    "holiday_flag": "sealed_calendar.holiday_flag",
    "is_holiday": "sealed_calendar.is_holiday",
    "is_event_1": "sealed_calendar.primary_event_present",
    "is_event_2": "sealed_calendar.secondary_event_present",
    "snap": "sealed_calendar.snap_schedule",
}


def _lineage_authority(dataset_id: str, feature_name: str) -> Tuple[str, str]:
    if feature_name in {"year", "month", "week", "day"}:
        return "deterministic_calendar", "ISO/Gregorian calendar standard"
    authorities = {
        "D3": "Rossmann sealed school-holiday calendar",
        "D4": "D4 sealed public-holiday calendar",
        "D5": "Favorita sealed holidays_events calendar",
        "D6": "M5 sealed calendar and SNAP schedule",
    }
    return "sealed_external_calendar", authorities[dataset_id]


def _build_lineage(dataset_id: str) -> Tuple[FutureKnownLineage, ...]:
    schema = _PREDICTOR_SCHEMAS[dataset_id]
    available_at = get_target_window(dataset_id).target_start - timedelta(days=1)
    items = []
    for feature in schema.fields:
        if feature.role is not FeatureRole.FUTURE_KNOWN:
            continue
        source_type, authority = _lineage_authority(dataset_id, feature.name)
        rule = _CALENDAR_RULES[feature.name]
        items.append(
            FutureKnownLineage(
                feature_name=feature.name,
                source_type=source_type,
                authority=authority,
                available_at=available_at,
                dependencies=("date",),
                generation_rule=rule,
                code_digest=hashlib.sha256(
                    f"{FEATURE_SCHEMA_VERSION}:{dataset_id}:{feature.name}:{rule}".encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(items)


_FUTURE_KNOWN_LINEAGE: Mapping[str, Tuple[FutureKnownLineage, ...]] = MappingProxyType(
    {dataset_id: _build_lineage(dataset_id) for dataset_id in _PREDICTOR_SCHEMAS}
)


def get_future_known_lineage(dataset_id: object) -> Tuple[FutureKnownLineage, ...]:
    return _FUTURE_KNOWN_LINEAGE[normalize_dataset_id(dataset_id)]


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "KNN_SCHEMA_VERSION",
    "MASK_SCHEMA_VERSION",
    "FeatureRole",
    "FutureKnownLineage",
    "FutureKnownLineageAuditReport",
    "FutureKnownLineageViolation",
    "KnnFeatureSchema",
    "KnnObservedDispositionV1",
    "KnnObservedFieldV1",
    "PredictorFeature",
    "PredictorFeatureMask",
    "PredictorFeatureSchema",
    "audit_future_known_lineage",
    "get_future_known_lineage",
    "get_knn_schema",
    "get_predictor_schema",
]
