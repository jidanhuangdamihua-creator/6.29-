"""Shared, pre-sealing daily calendar and sales canonicalization primitives.

The formal branch uses these helpers before any source selection or model code.
They intentionally keep runtime fill out of the target truth path: calendar
rows and source repairs are explicit, deterministic, and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.protocols.experiment_protocol import ProtocolViolation


FILL_POLICY_ENGINE_VERSION = "calendarize-fill/v1"
SOURCE_SALES_CANONICALIZATION_VERSION = "source_sales_canonicalization/v1"
RUNTIME_FILL_POLICY = "covariates_only"
D2_CANONICALIZATION_RULE_ID = "d2_june_absent_transaction_day_v1"
D2_APPROVED_ABSENT_TRANSACTION_DATES = ("2018-06-02",)
TARGET_VIEW_SCHEMA_VERSION = "target_views/v1"


class TargetViewContractError(ProtocolViolation):
    """A typed target view cannot be constructed without violating the protocol."""


class TargetViewName(str, Enum):
    KNN_OBSERVED = "knn_observed_frame"
    OBSERVED_MODEL = "observed_model_frame"
    BLIND_COVARIATE = "blind_covariate_frame"
    EVALUATOR_TRUTH = "evaluator_truth_frame"


@dataclass(frozen=True)
class TargetViewSchema:
    """Exact runtime/Arrow contract for one target view."""

    name: str
    fields: Tuple[Tuple[str, str, bool], ...]
    version: str = TARGET_VIEW_SCHEMA_VERSION
    digest: str = ""

    def __post_init__(self) -> None:
        names = tuple(str(field[0]) for field in self.fields)
        if not names or len(names) != len(set(names)):
            raise TargetViewContractError(f"{self.name} schema requires unique fields")
        if self.digest:
            return
        computed = _digest(
            {
                "name": str(self.name),
                "version": str(self.version),
                "fields": [
                    {"name": name, "arrow_dtype": dtype, "nullable": bool(nullable)}
                    for name, dtype, nullable in self.fields
                ],
            }
        )
        object.__setattr__(self, "digest", computed)

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(str(field[0]) for field in self.fields)

    def descriptor(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "version": str(self.version),
            "fields": [
                {"name": name, "arrow_dtype": dtype, "nullable": bool(nullable)}
                for name, dtype, nullable in self.fields
            ],
            "digest": self.digest,
        }

    def to_pyarrow_schema(self) -> Any:
        import pyarrow as pa

        types = {
            "string": pa.string(),
            "date32": pa.date32(),
            "int64": pa.int64(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
        }
        return pa.schema(
            [
                pa.field(name, types[dtype], nullable=bool(nullable))
                for name, dtype, nullable in self.fields
            ]
        )

    def validate_runtime_frame(self, frame: pd.DataFrame) -> None:
        """Reject extra, missing, reordered, or incompatible runtime fields."""

        if tuple(frame.columns) != self.column_names:
            raise TargetViewContractError(
                f"{self.name} requires exact column order {self.column_names!r}"
            )
        for name, dtype, nullable in self.fields:
            series = frame[name]
            if not nullable and series.isna().any():
                raise TargetViewContractError(f"{self.name}.{name} is not nullable")
            if dtype == "string":
                if not series.map(lambda value: isinstance(value, str)).all():
                    raise TargetViewContractError(f"{self.name}.{name} must be string")
            elif dtype == "date32":
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.isna().any():
                    raise TargetViewContractError(f"{self.name}.{name} must be date32")
            elif dtype == "int64":
                if not pd.api.types.is_integer_dtype(series):
                    raise TargetViewContractError(f"{self.name}.{name} must be int64")
            elif dtype == "float64":
                if not pd.api.types.is_numeric_dtype(series):
                    raise TargetViewContractError(f"{self.name}.{name} must be float64")
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
                if not np.isfinite(values).all():
                    raise TargetViewContractError(f"{self.name}.{name} must be finite")
            elif dtype == "bool" and not pd.api.types.is_bool_dtype(series):
                raise TargetViewContractError(f"{self.name}.{name} must be bool")


@dataclass(frozen=True)
class TypedTargetView:
    """One materialized view with a frozen column order and schema identity."""

    name: str
    frame: pd.DataFrame
    schema: TargetViewSchema
    dataset_id: str

    @property
    def schema_digest(self) -> str:
        return self.schema.digest

    @property
    def columns(self) -> Tuple[str, ...]:
        return self.schema.column_names

    def __post_init__(self) -> None:
        if tuple(self.frame.columns) != self.schema.column_names:
            raise TargetViewContractError(
                f"{self.name} columns do not match the exact view schema"
            )


@dataclass(frozen=True)
class TargetViews:
    """The four target views, exposed both as attributes and a small mapping."""

    knn_observed_frame: pd.DataFrame
    observed_model_frame: pd.DataFrame
    blind_covariate_frame: pd.DataFrame
    evaluator_truth_frame: pd.DataFrame
    schemas: Mapping[str, TargetViewSchema]
    dataset_id: str

    _NAMES = (
        "knn_observed_frame",
        "observed_model_frame",
        "blind_covariate_frame",
        "evaluator_truth_frame",
    )

    def __iter__(self) -> Iterator[str]:
        return iter(self._NAMES)

    def __len__(self) -> int:
        return len(self._NAMES)

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def keys(self) -> Tuple[str, ...]:
        return self._NAMES

    def items(self):
        return tuple((name, self[name]) for name in self._NAMES)

    def typed(self, name: str) -> TypedTargetView:
        if name not in self._NAMES:
            raise KeyError(name)
        return TypedTargetView(name, self[name], self.schemas[name], self.dataset_id)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


SHARED_FILL_POLICY_CONFIG_DIGEST = "sha256:" + _digest(
    {
        "engine_version": FILL_POLICY_ENGINE_VERSION,
        "runtime_policy": RUNTIME_FILL_POLICY,
        "sales_runtime_mutation": False,
        "non_sales_default": "preserve_null",
    }
)


def _normalize_date(value: object, *, label: str = "date") -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ProtocolViolation(f"{label} contains an invalid date: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _normalize_dates(values: Iterable[object], *, label: str = "date") -> pd.DatetimeIndex:
    parsed = pd.to_datetime(list(values), errors="coerce")
    if pd.isna(parsed).any():
        raise ProtocolViolation(f"{label} contains an invalid date")
    return pd.DatetimeIndex(parsed).normalize().sort_values().unique()


def _validate_group_columns(frame: pd.DataFrame, group_cols: Sequence[str]) -> Tuple[str, ...]:
    groups = tuple(str(column) for column in group_cols)
    if not groups:
        raise ProtocolViolation("calendarization requires at least one group column")
    missing = [column for column in groups if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"calendarization input is missing group columns: {missing}")
    return groups


def calendarize_and_fill(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    date_col: str = "date",
    start: Optional[object] = None,
    end: Optional[object] = None,
    additional_dates: Sequence[object] = (),
    fill_rules: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Calendarize groups and apply only explicitly declared row fill rules.

    Existing rows are never reordered within a group except for the required
    ascending daily order.  Synthetic rows are tracked in ``DataFrame.attrs``
    instead of being silently smuggled into the artifact schema.  A rule may
    be ``"zero"``, a scalar value, or a callable accepting the column name.
    No unspecified column receives a blanket fill value.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("calendarize_and_fill requires a pandas DataFrame")
    groups = _validate_group_columns(frame, group_cols)
    if date_col not in frame.columns:
        raise ProtocolViolation(f"calendarization input is missing date column: {date_col}")
    prepared = frame.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce").dt.normalize()
    if prepared[date_col].isna().any():
        raise ProtocolViolation("calendarization input contains invalid dates")
    if prepared.duplicated([*groups, date_col]).any():
        raise ProtocolViolation("calendarization input contains duplicate entity/date rows")

    lower = _normalize_date(start, label="calendar start") if start is not None else None
    upper = _normalize_date(end, label="calendar end") if end is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ProtocolViolation("calendar start is after calendar end")
    requested_dates = _normalize_dates(additional_dates, label="additional calendar dates")
    if lower is not None and (requested_dates < lower).any():
        raise ProtocolViolation("additional calendar date precedes requested calendar start")
    if upper is not None and (requested_dates > upper).any():
        raise ProtocolViolation("additional calendar date exceeds requested calendar end")

    rules: Dict[str, Any] = dict(fill_rules or {})
    unknown_rules = sorted(set(rules).difference(prepared.columns))
    if unknown_rules:
        raise ProtocolViolation(f"calendarization fill rule references unknown columns: {unknown_rules}")
    for column, rule in rules.items():
        if callable(rule):
            continue
        if rule == "zero":
            continue
        if isinstance(rule, (str, bytes)):
            raise ProtocolViolation(f"unsupported calendar fill rule for {column}: {rule!r}")

    rows = []
    synthetic_mask: list[bool] = []
    for raw_key, group in prepared.groupby(list(groups), sort=False, dropna=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        group = group.sort_values(date_col).copy()
        group_start = lower if lower is not None else group[date_col].min()
        group_end = upper if upper is not None else group[date_col].max()
        if len(requested_dates) and lower is None and upper is None:
            # An explicit pre-sealing canonicalization list is intentionally
            # sparse.  It must not silently become blanket target-sales fill.
            calendar = pd.DatetimeIndex(pd.DatetimeIndex(group[date_col]).union(requested_dates)).sort_values()
        else:
            calendar = pd.date_range(group_start, group_end, freq="D")
            if len(requested_dates):
                calendar = pd.DatetimeIndex(calendar.union(requested_dates)).sort_values()
        indexed = group.set_index(date_col).reindex(calendar)
        indexed.index.name = date_col
        missing_row = ~indexed.index.isin(group[date_col])
        for column, value in zip(groups, key):
            indexed[column] = value
        for column, rule in rules.items():
            if callable(rule):
                fill_value = rule(column)
            elif rule == "zero":
                fill_value = 0
            else:
                fill_value = rule
            if fill_value is not None:
                indexed.loc[missing_row, column] = fill_value
        rows.append(indexed.reset_index().loc[:, prepared.columns])
        synthetic_mask.extend(bool(value) for value in missing_row)

    result = pd.concat(rows, ignore_index=True) if rows else prepared.iloc[0:0].copy()
    result = result.sort_values([*groups, date_col]).reset_index(drop=True)

    # Sorting changes the row order, so rebuild the synthetic mask by key.
    original_keys = set(
        tuple(row[column] for column in (*groups, date_col))
        for _, row in prepared.iterrows()
    )
    sorted_mask = [
        tuple(row[column] for column in (*groups, date_col)) not in original_keys
        for _, row in result.iterrows()
    ]
    rule_descriptor = {
        "engine_version": FILL_POLICY_ENGINE_VERSION,
        "date_column": date_col,
        "group_columns": list(groups),
        "start": None if lower is None else lower.strftime("%Y-%m-%d"),
        "end": None if upper is None else upper.strftime("%Y-%m-%d"),
        "additional_dates": [value.strftime("%Y-%m-%d") for value in requested_dates],
        "fill_rules": {column: str(rule) for column, rule in sorted(rules.items())},
    }
    result.attrs.update(
        {
            "fill_policy_engine_version": FILL_POLICY_ENGINE_VERSION,
            "fill_policy_shared_with_raw_rebuild": True,
            "fill_policy_config_digest": "sha256:" + _digest(rule_descriptor),
            "runtime_fill_policy": RUNTIME_FILL_POLICY,
            "calendarization_rule_descriptor": rule_descriptor,
            "calendar_row_missing_mask": sorted_mask,
            "synthetic_date_count": int(sum(sorted_mask)),
        }
    )
    return result


def canonicalize_source_sales(
    frame: pd.DataFrame,
    *,
    sales_col: str = "sales",
    date_col: str = "date",
    calendar_row_missing: Optional[Sequence[bool]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Canonicalize source sales to finite non-negative float64 values.

    The returned audit identifies every changed row by one of the three closed
    reasons required by the protocol.  Infinity is a hard failure.
    """

    if sales_col not in frame.columns:
        raise ProtocolViolation(f"source sales canonicalization requires {sales_col!r}")
    result = frame.copy()
    values = pd.to_numeric(result[sales_col], errors="coerce").astype("float64")
    if np.isinf(values.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ValueError("source sales contains infinity")
    if calendar_row_missing is None:
        raw_mask = result.attrs.get("calendar_row_missing_mask")
        if raw_mask is None:
            synthetic = np.zeros(len(result), dtype=bool)
        else:
            synthetic = np.asarray(raw_mask, dtype=bool)
    else:
        synthetic = np.asarray(list(calendar_row_missing), dtype=bool)
    if len(synthetic) != len(result):
        raise ValueError("calendar_row_missing mask length does not match source frame")

    nan_mask = values.isna().to_numpy()
    negative_mask = (values < 0).to_numpy() & ~nan_mask
    repaired = nan_mask | negative_mask
    reasons = np.full(len(result), "", dtype=object)
    reasons[nan_mask & synthetic] = "calendar_row_missing"
    reasons[nan_mask & ~synthetic] = "original_nan"
    reasons[negative_mask] = "original_negative"
    values.loc[repaired] = 0.0
    result[sales_col] = values.astype("float64")

    records = []
    group_columns = [column for column in result.columns if column not in {sales_col, date_col}]
    for index in np.flatnonzero(repaired):
        row = result.iloc[int(index)]
        records.append(
            {
                "row_index": int(index),
                "date": pd.Timestamp(row[date_col]).strftime("%Y-%m-%d") if date_col in result else None,
                "reason": str(reasons[index]),
                "group": [str(row[column]) for column in group_columns[:3]],
            }
        )
    records.sort(key=lambda item: (str(item["date"]), item["row_index"]))
    reason_counts = {
        reason: int(sum(1 for value in reasons if value == reason))
        for reason in ("original_nan", "original_negative", "calendar_row_missing")
    }
    dates = sorted({record["date"] for record in records if record["date"] is not None})
    audit = {
        "version": SOURCE_SALES_CANONICALIZATION_VERSION,
        "repair_reason_counts": reason_counts,
        "affected_date_digest": "sha256:" + _digest(dates),
        "repair_mask_sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
        "affected_rows": records,
        "rows_examined": int(len(result)),
    }
    if not np.isfinite(result[sales_col].to_numpy(dtype="float64")).all():
        raise ProtocolViolation("source sales canonicalization left non-finite values")
    if (result[sales_col] < 0).any():
        raise ProtocolViolation("source sales canonicalization left negative values")
    result.attrs["source_sales_canonicalization"] = audit
    return result, audit


def validate_target_truth(frame: pd.DataFrame, *, sales_col: str = "sales") -> None:
    """Validate sealed target sales without applying a repair."""

    if sales_col not in frame.columns:
        raise ProtocolViolation(f"target truth requires {sales_col!r}")
    values = pd.to_numeric(frame[sales_col], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ProtocolViolation("target sales must be finite after pre-sealing validation")
    if (values < 0).any():
        raise ProtocolViolation("target sales must be non-negative after pre-sealing validation")


_VIEW_BASE_KEY_COLUMNS = ("target_entity_key", "date")


def _arrow_dtype_for_feature(dtype: str) -> str:
    normalized = str(dtype).lower()
    if normalized in {"int", "int32", "int64", "integer"}:
        return "int64"
    if normalized in {"bool", "boolean"}:
        return "bool"
    return "float64"


def _target_key_columns(frame: pd.DataFrame, dataset_id: str) -> Tuple[str, ...]:
    if "target_entity_key" in frame.columns:
        return ("target_entity_key",)
    from src.protocols.experiment_protocol import get_experiment_protocol

    protocol = get_experiment_protocol(dataset_id)
    declared = tuple(protocol.source_pool_rule.key_fields)
    if all(column in frame.columns for column in declared):
        return declared
    if "entity_id" in frame.columns:
        return ("entity_id",)
    raise TargetViewContractError(
        f"target frame has no canonical key columns for {dataset_id}: expected {declared!r}"
    )


def _canonical_key_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise TargetViewContractError("target entity key cannot be null")
    text = str(value).strip()
    if not text or "/" in text and text.startswith("/"):
        raise TargetViewContractError(f"invalid target entity key component: {value!r}")
    return text


def _attach_canonical_target_key(frame: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    result = frame.copy()
    key_columns = _target_key_columns(result, dataset_id)
    if key_columns == ("target_entity_key",):
        result["target_entity_key"] = result["target_entity_key"].map(_canonical_key_value)
    else:
        result["target_entity_key"] = result.loc[:, list(key_columns)].apply(
            lambda row: "/".join(_canonical_key_value(value) for value in row), axis=1
        )
    keys = tuple(result["target_entity_key"].drop_duplicates())
    if len(keys) != 1:
        raise TargetViewContractError(
            f"target view requires exactly one target entity key, got {keys!r}"
        )
    result.attrs["target_key_columns"] = tuple(key_columns)
    return result


def _prepare_target_view_input(frame: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("target view construction requires a pandas DataFrame")
    if "date" not in frame.columns:
        raise TargetViewContractError("target view input requires date")
    result = _attach_canonical_target_key(frame, dataset_id)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise TargetViewContractError("target view input contains invalid dates")
    if result.duplicated(["target_entity_key", "date"]).any():
        raise TargetViewContractError("target view input contains duplicate entity/date rows")
    return result.sort_values(["target_entity_key", "date"]).reset_index(drop=True)


def _date_slice(
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
    label: str,
) -> pd.DataFrame:
    expected = pd.date_range(start, end, freq="D")
    in_window = frame.loc[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    dates = pd.DatetimeIndex(in_window["date"].unique()).sort_values()
    if not dates.equals(expected):
        missing = expected.difference(dates)
        raise TargetViewContractError(
            f"{label} must contain exact natural-day calendar: "
            f"expected={len(expected)} missing={list(missing[:3])}"
        )
    counts = in_window.groupby("target_entity_key", sort=False)["date"].nunique()
    if not (counts == len(expected)).all():
        raise TargetViewContractError(f"{label} has an incomplete entity calendar")
    selected = frame.loc[frame["date"].isin(expected)].copy()
    return selected.sort_values(["target_entity_key", "date"]).reset_index(drop=True)


def _validate_numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
    nonnegative: Sequence[str] = (),
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise TargetViewContractError(f"{label} is missing required field {column!r}")
        converted = pd.to_numeric(result[column], errors="coerce")
        values = converted.to_numpy(dtype="float64", na_value=np.nan)
        if not np.isfinite(values).all():
            raise TargetViewContractError(f"{label} field {column!r} must be finite and non-null")
        if column in nonnegative and (values < 0).any():
            raise TargetViewContractError(f"{label} field {column!r} must be non-negative")
        result[column] = converted
    return result


def _view_schema(
    name: TargetViewName,
    *,
    predictor: Any,
    knn: Any,
) -> TargetViewSchema:
    fields: list[Tuple[str, str, bool]] = [
        ("target_entity_key", "string", False),
        ("date", "date32", False),
    ]
    if name is TargetViewName.KNN_OBSERVED:
        fields.extend((field, "float64", False) for field in knn.ordered_names)
    elif name is TargetViewName.OBSERVED_MODEL:
        fields.extend(
            (field.name, _arrow_dtype_for_feature(field.dtype), False)
            for field in predictor.fields
        )
    elif name is TargetViewName.BLIND_COVARIATE:
        fields.extend(
            (field.name, _arrow_dtype_for_feature(field.dtype), False)
            for field in predictor.fields
            if field.role.value in {"future_known", "static_known"}
        )
    else:
        fields.extend(
            [
                ("y_true", "float64", False),
                ("is_synthetic_date", "bool", False),
                ("truth_key", "string", False),
            ]
        )
    return TargetViewSchema(name.value, tuple(fields))


def _strict_blind_projection(
    frame: pd.DataFrame,
    *,
    predictor: Any,
    key_columns: Sequence[str],
    allow_source_sales: bool = False,
) -> None:
    allowed = {
        "target_entity_key",
        "date",
        *key_columns,
        "entity_id",
        *(
            field.name
            for field in predictor.fields
            if field.role.value in {"future_known", "static_known"}
        ),
    }
    if allow_source_sales:
        # The sealed target artifact necessarily contains observed sales so
        # evaluator_truth_frame can be built.  Sales is never copied into the
        # blind output below.
        allowed.add("sales")
    forbidden = {
        "y_true",
        "truth_key",
        "is_synthetic_date",
    }
    forbidden.update(
        name
        for name in frame.columns
        if name not in allowed and name.lower() in {"promo", "customers", "open", "stock", "activity", "discount", "sell_price", "onpromotion", "transactions", "oil_price"}
    )
    leaked = sorted(set(frame.columns).intersection(forbidden))
    unknown = sorted(set(frame.columns).difference(allowed))
    if leaked:
        raise TargetViewContractError(
            f"blind_covariate_frame contains forbidden field(s): {leaked}"
        )
    if unknown:
        raise TargetViewContractError(
            f"blind_covariate_frame contains unknown field(s): {unknown}"
        )


def _synthetic_flags(frame: pd.DataFrame) -> pd.Series:
    if "is_synthetic_date" in frame.columns:
        values = frame["is_synthetic_date"]
        if not pd.api.types.is_bool_dtype(values):
            lowered = values.astype("string").str.lower()
            if not lowered.isin({"true", "false"}).all():
                raise TargetViewContractError("is_synthetic_date must be boolean")
            values = lowered.eq("true")
        return values.astype(bool)
    attrs_mask = frame.attrs.get("calendar_row_missing_mask")
    if attrs_mask is not None and len(attrs_mask) == len(frame):
        return pd.Series(np.asarray(attrs_mask, dtype=bool), index=frame.index)
    # A sealed target has already been validated.  This default is explicit:
    # runtime view construction never infers or repairs target sales.
    return pd.Series(False, index=frame.index, dtype=bool)


def _build_view_frame(
    frame: pd.DataFrame,
    schema: TargetViewSchema,
    *,
    name: TargetViewName,
    predictor: Any,
    knn: Any,
    window: Any,
) -> pd.DataFrame:
    if name is TargetViewName.KNN_OBSERVED:
        selected = _date_slice(frame, start=window.observed_start, end=window.observed_end, label=name.value)
        selected = _validate_numeric_columns(
            selected,
            knn.ordered_names,
            label=name.value,
            nonnegative=("sales",),
        )
        result = selected.loc[:, ["target_entity_key", "date", *knn.ordered_names]].copy()
    elif name is TargetViewName.OBSERVED_MODEL:
        selected = _date_slice(frame, start=window.observed_start, end=window.observed_end, label=name.value)
        names = list(predictor.ordered_names)
        selected = _validate_numeric_columns(
            selected,
            names,
            label=name.value,
            nonnegative=("sales",),
        )
        result = selected.loc[:, ["target_entity_key", "date", *names]].copy()
    elif name is TargetViewName.BLIND_COVARIATE:
        key_columns = tuple(
            frame.attrs.get("target_key_columns")
            or _target_key_columns(frame, window.dataset_id)
        )
        selected = _date_slice(frame, start=window.blind_start, end=window.blind_end, label=name.value)
        names = [
            field.name
            for field in predictor.fields
            if field.role.value in {"future_known", "static_known"}
        ]
        selected = _validate_numeric_columns(selected, names, label=name.value)
        result = selected.loc[:, ["target_entity_key", "date", *names]].copy()
        _strict_blind_projection(
            result,
            predictor=predictor,
            key_columns=key_columns,
        )
    else:
        selected = _date_slice(frame, start=window.blind_start, end=window.blind_end, label=name.value)
        selected = _validate_numeric_columns(
            selected,
            ("sales",),
            label=name.value,
            nonnegative=("sales",),
        )
        flags = _synthetic_flags(selected)
        result = selected.loc[:, ["target_entity_key", "date"]].copy()
        result["y_true"] = selected["sales"].astype("float64")
        result["is_synthetic_date"] = flags.to_numpy(dtype=bool)
        result["truth_key"] = result.apply(
            lambda row: hashlib.sha256(
                f"{window.dataset_id}|{row['target_entity_key']}|{pd.Timestamp(row['date']).date().isoformat()}".encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
    if tuple(result.columns) != schema.column_names:
        raise TargetViewContractError(f"{name.value} produced an unexpected column order")
    for field_name, arrow_dtype, _nullable in schema.fields:
        if arrow_dtype == "string":
            result[field_name] = result[field_name].astype("string").astype(object)
        elif arrow_dtype == "date32":
            result[field_name] = pd.to_datetime(result[field_name]).dt.normalize()
        elif arrow_dtype == "int64":
            result[field_name] = pd.to_numeric(result[field_name], errors="raise").astype("int64")
        elif arrow_dtype == "float64":
            result[field_name] = pd.to_numeric(result[field_name], errors="raise").astype("float64")
        elif arrow_dtype == "bool":
            result[field_name] = result[field_name].astype(bool)
    schema.validate_runtime_frame(result)
    result.attrs.update(
        {
            "target_view_name": name.value,
            "target_view_schema_version": schema.version,
            "target_view_schema_digest": schema.digest,
            "dataset_id": window.dataset_id,
            "runtime_fill_policy": RUNTIME_FILL_POLICY,
        }
    )
    return result.reset_index(drop=True)


def validate_target_view_frame(
    frame: pd.DataFrame,
    dataset_id: object,
    view_name: object,
    *,
    window: Any | None = None,
) -> None:
    """Validate a materialized view independently of its source artifact."""

    from src.protocols.feature_schema import get_knn_schema, get_predictor_schema
    from src.protocols.sealing_protocol import get_target_window, normalize_dataset_id

    normalized = normalize_dataset_id(dataset_id)
    try:
        name = TargetViewName(str(view_name))
    except ValueError as exc:
        raise TargetViewContractError(f"unknown target view: {view_name!r}") from exc
    target_window = window or get_target_window(normalized)
    schema = _view_schema(
        name,
        predictor=get_predictor_schema(normalized),
        knn=get_knn_schema(normalized),
    )
    schema.validate_runtime_frame(frame)
    prepared = _prepare_target_view_input(frame, normalized)
    if name in {TargetViewName.KNN_OBSERVED, TargetViewName.OBSERVED_MODEL}:
        _date_slice(
            prepared,
            start=target_window.observed_start,
            end=target_window.observed_end,
            label=name.value,
        )
    else:
        _date_slice(
            prepared,
            start=target_window.blind_start,
            end=target_window.blind_end,
            label=name.value,
        )
    if name is TargetViewName.BLIND_COVARIATE:
        _strict_blind_projection(
            frame,
            predictor=get_predictor_schema(normalized),
            key_columns=(),
        )
    if name is TargetViewName.EVALUATOR_TRUTH:
        values = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values < 0).any():
            raise TargetViewContractError("evaluator truth must be finite and non-negative")


def build_target_views(
    target_frame: pd.DataFrame,
    dataset_id: object,
    *,
    window: Any | None = None,
) -> TargetViews:
    """Build the four exact target views without mutating target truth."""

    from src.protocols.feature_schema import get_knn_schema, get_predictor_schema
    from src.protocols.sealing_protocol import get_target_window, normalize_dataset_id

    normalized = normalize_dataset_id(dataset_id)
    target_window = window or get_target_window(normalized)
    if str(getattr(target_window, "dataset_id", normalized)) != normalized:
        raise TargetViewContractError("target view window dataset does not match dataset_id")
    prepared = _prepare_target_view_input(target_frame, normalized)
    predictor = get_predictor_schema(normalized)
    knn = get_knn_schema(normalized)
    schemas = {
        name.value: _view_schema(name, predictor=predictor, knn=knn)
        for name in TargetViewName
    }
    built = {
        name.value: _build_view_frame(
            prepared,
            schemas[name.value],
            name=name,
            predictor=predictor,
            knn=knn,
            window=target_window,
        )
        for name in TargetViewName
    }
    return TargetViews(
        knn_observed_frame=built[TargetViewName.KNN_OBSERVED.value],
        observed_model_frame=built[TargetViewName.OBSERVED_MODEL.value],
        blind_covariate_frame=built[TargetViewName.BLIND_COVARIATE.value],
        evaluator_truth_frame=built[TargetViewName.EVALUATOR_TRUTH.value],
        schemas=MappingProxyType(schemas),
        dataset_id=normalized,
    )


def build_knn_observed_frame(target_frame: pd.DataFrame, dataset_id: object, *, window: Any | None = None) -> pd.DataFrame:
    return build_target_views(target_frame, dataset_id, window=window).knn_observed_frame


def build_observed_model_frame(target_frame: pd.DataFrame, dataset_id: object, *, window: Any | None = None) -> pd.DataFrame:
    return build_target_views(target_frame, dataset_id, window=window).observed_model_frame


def build_blind_covariate_frame(target_frame: pd.DataFrame, dataset_id: object, *, window: Any | None = None) -> pd.DataFrame:
    return build_target_views(target_frame, dataset_id, window=window).blind_covariate_frame


def build_evaluator_truth_frame(target_frame: pd.DataFrame, dataset_id: object, *, window: Any | None = None) -> pd.DataFrame:
    return build_target_views(target_frame, dataset_id, window=window).evaluator_truth_frame


def d2_approved_calendarize(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    fill_sales: bool = True,
) -> pd.DataFrame:
    """Apply only the reviewed D2 absent-transaction-day canonicalization."""

    return calendarize_and_fill(
        frame,
        group_cols=group_cols,
        additional_dates=D2_APPROVED_ABSENT_TRANSACTION_DATES,
        fill_rules={
            "sales": "zero",
            "promo": "zero",
        }
        if fill_sales
        else {"promo": "zero"},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parquet_schema_descriptor(path: Path) -> Dict[str, Any]:
    schema = pq.read_schema(path)
    return {
        "columns": [
            {"name": name, "arrow_dtype": str(schema.field(name).type)}
            for name in schema.names
        ],
        "column_order": list(schema.names),
    }


def publish_sealed_dataset(
    output_root: Path,
    dataset_id: object,
    *,
    manifest: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    source_frame: Optional[pd.DataFrame] = None,
    target_frame: Optional[pd.DataFrame] = None,
    source_path: Optional[Path] = None,
    target_path: Optional[Path] = None,
    sidecars: Optional[Mapping[str, Mapping[str, Any]]] = None,
    predictor_schema: Optional[Mapping[str, Any]] = None,
    knn_schema: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Publish one dataset directory with a temp-write/rename/fsync boundary."""

    normalized = str(dataset_id).strip().lower()
    if normalized.startswith("d"):
        normalized = normalized[1:]
    if normalized.startswith("dataset"):
        normalized = normalized[len("dataset") :]
    dataset_name = f"dataset{int(normalized)}"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / dataset_name
    if final_dir.exists():
        raise FileExistsError(f"sealed dataset already exists and is immutable: {final_dir}")
    temporary = root / f".{dataset_name}.tmp.{os.getpid()}.{uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        if (source_frame is None) == (source_path is None):
            raise ValueError("provide exactly one of source_frame or source_path")
        if (target_frame is None) == (target_path is None):
            raise ValueError("provide exactly one of target_frame or target_path")
        if source_frame is not None:
            source_frame.to_parquet(temporary / "source.parquet", index=False)
        else:
            shutil.copy2(Path(source_path), temporary / "source.parquet")
        if target_frame is not None:
            target_frame.to_parquet(temporary / "target.parquet", index=False)
        else:
            shutil.copy2(Path(target_path), temporary / "target.parquet")

        source_artifact = temporary / "source.parquet"
        target_artifact = temporary / "target.parquet"
        source_schema = _parquet_schema_descriptor(source_artifact)
        target_schema = _parquet_schema_descriptor(target_artifact)
        _write_json_fsync(temporary / "source_schema.json", source_schema)
        _write_json_fsync(temporary / "target_schema.json", target_schema)
        _write_json_fsync(temporary / "predictor_schema.json", dict(predictor_schema or {}))
        _write_json_fsync(temporary / "knn_schema.json", dict(knn_schema or {}))
        for name, payload in (sidecars or {}).items():
            _write_json_fsync(temporary / name, dict(payload))

        final_manifest = dict(manifest)
        final_manifest.setdefault("dataset_id", f"D{int(normalized)}")
        final_manifest.setdefault("sealed_root_version", "d1_d6_sealed_v1")
        final_manifest["artifacts"] = {
            "source": {
                "path": "source.parquet",
                "sha256": sha256_file(source_artifact),
                "size_bytes": source_artifact.stat().st_size,
            },
            "target": {
                "path": "target.parquet",
                "sha256": sha256_file(target_artifact),
                "size_bytes": target_artifact.stat().st_size,
            },
        }
        _write_json_fsync(temporary / "validation_report.json", dict(validation_report))
        _write_json_fsync(temporary / "manifest.json", final_manifest)
        for file_path in temporary.iterdir():
            if file_path.is_file():
                with file_path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, final_dir)
        _fsync_directory(root)
        return final_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "D2_APPROVED_ABSENT_TRANSACTION_DATES",
    "D2_CANONICALIZATION_RULE_ID",
    "FILL_POLICY_ENGINE_VERSION",
    "SHARED_FILL_POLICY_CONFIG_DIGEST",
    "RUNTIME_FILL_POLICY",
    "SOURCE_SALES_CANONICALIZATION_VERSION",
    "TARGET_VIEW_SCHEMA_VERSION",
    "TargetViewContractError",
    "TargetViewName",
    "TargetViewSchema",
    "TargetViews",
    "TypedTargetView",
    "build_blind_covariate_frame",
    "build_evaluator_truth_frame",
    "build_knn_observed_frame",
    "build_observed_model_frame",
    "build_target_views",
    "calendarize_and_fill",
    "canonicalize_source_sales",
    "d2_approved_calendarize",
    "publish_sealed_dataset",
    "sha256_file",
    "validate_target_view_frame",
    "validate_target_truth",
]
