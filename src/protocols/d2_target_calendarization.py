"""Deterministic producer for the five frozen D2 target repair dates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .experiment_protocol import ProtocolViolation, normalize_source_key
from .gate1_transformation import _add_date_fields


D2_TARGET_CALENDARIZATION_RULE_VERSION = "d2_target_calendarization_v1"
D2_TARGET_REPAIR_POLICY = "closed_day_zero_demand"
D2_TARGET_REPAIR_REASON = "store_closed"
D2_TARGET_KEY = ("1", "10")
D2_TARGET_FORMAL_START = "2018-06-01"
D2_TARGET_FORMAL_END = "2018-12-27"
D2_TARGET_REPAIR_DATES = (
    "2018-08-15",
    "2018-11-01",
    "2018-12-08",
    "2018-12-25",
    "2018-12-26",
)
_DATE_FEATURES = ("year", "month", "week", "day")
D2_TARGET_EXISTING_ROW_REPAIR_DATE = "2018-06-02"
D2_TARGET_EXISTING_ROW_REPAIR_FIELDS = ("entity_id", *_DATE_FEATURES)
_STATIC_IDENTITY_NAMES = frozenset({"entity_id", "brand_id", "item_id", "brand", "item"})


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize().strftime("%Y-%m-%d")
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ProtocolViolation("D2 target digest received a non-finite value")
        return format(value, ".17g")
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _canonical_payload(frame: pd.DataFrame) -> dict[str, object]:
    columns = list(frame.columns)
    ordered = frame.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    ordered = ordered.sort_values(["brand_id", "item_id", "date"], kind="mergesort")
    return {
        "columns": columns,
        "rows": [
            [_canonical_value(value) for value in row]
            for row in ordered.loc[:, columns].itertuples(index=False, name=None)
        ],
    }


def target_semantic_digest(frame: pd.DataFrame) -> str:
    payload = _canonical_payload(frame)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_identity(frame: pd.DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _authoritative_value(frame: pd.DataFrame, column: str) -> object:
    values = frame[column].dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def _coerce_to_original_dtype(value: object, dtype: object) -> object:
    if pd.isna(value):
        return value
    if pd.api.types.is_integer_dtype(dtype):
        return int(value)
    if pd.api.types.is_float_dtype(dtype):
        return float(value)
    return value


def _repair_existing_target_row(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Repair only the one frozen D2 target row with incomplete date identity."""

    result = source.copy()
    repair_date = pd.Timestamp(D2_TARGET_EXISTING_ROW_REPAIR_DATE)
    repair_mask = result["date"].eq(repair_date)
    if int(repair_mask.sum()) != 1:
        raise ProtocolViolation(
            "D2 target existing-row repair requires exactly one row at "
            f"{D2_TARGET_EXISTING_ROW_REPAIR_DATE}"
        )

    expected_entity_id = _authoritative_value(result.loc[~repair_mask], "entity_id")
    if pd.isna(expected_entity_id):
        raise ProtocolViolation("D2 target existing-row repair has no entity_id authority")
    expected_values: dict[str, pd.Series] = {
        "entity_id": pd.Series(expected_entity_id, index=result.index),
        "year": result["date"].dt.year,
        "month": result["date"].dt.month,
        "week": result["date"].dt.isocalendar().week,
        "day": result["date"].dt.day,
    }

    changed_fields: list[str] = []
    for field, expected in expected_values.items():
        if field in _DATE_FEATURES:
            numeric = pd.to_numeric(result[field], errors="coerce")
            matches = pd.Series(
                np.isfinite(numeric.to_numpy(dtype=np.float64)), index=result.index
            ) & numeric.eq(pd.to_numeric(expected, errors="coerce"))
        else:
            matches = pd.Series(
                (
                    not pd.isna(value)
                    and str(value) == str(expected.loc[index])
                    for index, value in result[field].items()
                ),
                index=result.index,
            )
        outside_repair = ~matches & ~repair_mask
        if outside_repair.any():
            offending_dates = (
                result.loc[outside_repair, "date"]
                .dt.strftime("%Y-%m-%d")
                .drop_duplicates()
                .tolist()
            )
            raise ProtocolViolation(
                f"D2 target {field} is missing or inconsistent outside the authorized "
                f"existing-row repair date: {offending_dates!r}"
            )
        if not bool(matches.loc[repair_mask].all()):
            repair_value = _coerce_to_original_dtype(
                expected.loc[repair_mask].iloc[0], result[field].dtype
            )
            result.loc[repair_mask, field] = repair_value
            changed_fields.append(field)

    return result, {
        "date": D2_TARGET_EXISTING_ROW_REPAIR_DATE,
        "fields": list(D2_TARGET_EXISTING_ROW_REPAIR_FIELDS),
        "changed_fields": changed_fields,
        "changed_cell_count": len(changed_fields),
        "policy": "recompute_identity_and_date_fields_on_authorized_existing_row",
        "reason": "repair_nonfinite_existing_target_row",
    }


def calendarize_d2_target_frame(
    target_frame: pd.DataFrame,
    *,
    input_target_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Insert only the authorized D2 closed-day rows and preserve every old row."""

    required = (
        "date",
        "brand_id",
        "item_id",
        "sales",
        "promo",
        "entity_id",
        *_DATE_FEATURES,
    )
    missing = [column for column in required if column not in target_frame.columns]
    if missing:
        raise ProtocolViolation(f"D2 target repair schema is missing columns: {missing!r}")
    source = target_frame.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    if source["date"].isna().any():
        raise ProtocolViolation("D2 target repair received invalid dates")
    key_series = source.loc[:, ["brand_id", "item_id"]].apply(
        lambda row: normalize_source_key(tuple(row.tolist())), axis=1
    )
    if key_series.nunique() != 1 or key_series.iloc[0] != D2_TARGET_KEY:
        raise ProtocolViolation(
            f"D2 target repair requires exactly target key {D2_TARGET_KEY!r}, got {sorted(key_series.unique())!r}"
        )
    if source.duplicated(["brand_id", "item_id", "date"]).any():
        raise ProtocolViolation("D2 target repair received duplicate entity/date keys")
    before_digest = target_semantic_digest(source)
    protected_before_digest = target_semantic_digest(
        source.loc[source["date"].ne(pd.Timestamp(D2_TARGET_EXISTING_ROW_REPAIR_DATE))]
    )
    source, existing_row_repair = _repair_existing_target_row(source)
    existing_dates = set(pd.DatetimeIndex(source["date"]))
    authorized = tuple(pd.Timestamp(value) for value in D2_TARGET_REPAIR_DATES)
    existing_authorized = source.loc[source["date"].isin(authorized)]
    if (
        not existing_authorized["sales"].eq(0).all()
        or not existing_authorized["promo"].eq(0).all()
    ):
        raise ProtocolViolation(
            "D2 target authorized repair dates must have sales=0 and promo=0"
        )
    inserted_dates = tuple(timestamp for timestamp in authorized if timestamp not in existing_dates)
    rows: list[dict[str, object]] = []
    for timestamp in inserted_dates:
        row: dict[str, object] = {column: pd.NA for column in source.columns}
        for column in source.columns:
            name = str(column).strip().lower()
            if name in _STATIC_IDENTITY_NAMES or name.endswith("_id") or name.endswith("_code"):
                row[column] = _authoritative_value(source, column)
        row["brand_id"] = _coerce_to_original_dtype(1, source["brand_id"].dtype)
        row["item_id"] = _coerce_to_original_dtype(10, source["item_id"].dtype)
        row["date"] = timestamp
        row["sales"] = _coerce_to_original_dtype(0, source["sales"].dtype)
        row["promo"] = _coerce_to_original_dtype(0, source["promo"].dtype)
        generated = _add_date_fields(pd.DataFrame([row]))
        for column in _DATE_FEATURES:
            row[column] = _coerce_to_original_dtype(generated.iloc[0][column], source[column].dtype)
        rows.append(row)
    if rows:
        inserted = pd.DataFrame(rows, columns=list(source.columns))
        for column in source.columns:
            try:
                inserted[column] = inserted[column].astype(source[column].dtype)
            except (TypeError, ValueError) as exc:
                raise ProtocolViolation(
                    f"D2 target repair cannot preserve dtype for column {column!r}"
                ) from exc
        result = pd.concat([source, inserted], ignore_index=True)
    else:
        result = source.copy()
    result = result.sort_values(["brand_id", "item_id", "date"], kind="mergesort").reset_index(drop=True)
    result.attrs = target_frame.attrs.copy()
    protected_result = result.loc[
        result["date"].isin(source["date"])
        & result["date"].ne(pd.Timestamp(D2_TARGET_EXISTING_ROW_REPAIR_DATE))
    ]
    if target_semantic_digest(protected_result) != protected_before_digest:
        raise ProtocolViolation("D2 target repair changed a protected existing row")
    evidence: dict[str, Any] = {
        "dataset": "Dataset2",
        "artifact": "target",
        "target_key": list(D2_TARGET_KEY),
        "formal_window_start": D2_TARGET_FORMAL_START,
        "formal_window_end": D2_TARGET_FORMAL_END,
        "policy": D2_TARGET_REPAIR_POLICY,
        "reason": D2_TARGET_REPAIR_REASON,
        "inserted_dates": [timestamp.strftime("%Y-%m-%d") for timestamp in inserted_dates],
        "repair_dates": list(D2_TARGET_REPAIR_DATES),
        "sales_fill": 0,
        "promo_fill": 0,
        "inserted_count": len(inserted_dates),
        "original_row_present": len(inserted_dates) == 0,
        "existing_row_repair": existing_row_repair,
        "repair_date_digest": hashlib.sha256(
            json.dumps(list(D2_TARGET_REPAIR_DATES), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "input_target_sha256": input_target_sha256,
        "output_target_sha256": None,
        "producer_identity": {
            "module": "src.protocols.d2_target_calendarization",
            "function": "calendarize_d2_target_frame",
            "rule_version": D2_TARGET_CALENDARIZATION_RULE_VERSION,
        },
        "schema_identity": {
            "columns": list(source.columns),
            "dtypes": {column: str(source[column].dtype) for column in source.columns},
            "digest": _schema_identity(source),
        },
        "input_semantic_digest": before_digest,
        "output_semantic_digest": target_semantic_digest(result),
        "created_rebuilt_deterministically": True,
    }
    return result, evidence


__all__ = [
    "D2_TARGET_CALENDARIZATION_RULE_VERSION",
    "D2_TARGET_EXISTING_ROW_REPAIR_DATE",
    "D2_TARGET_EXISTING_ROW_REPAIR_FIELDS",
    "D2_TARGET_REPAIR_DATES",
    "calendarize_d2_target_frame",
    "target_semantic_digest",
]
