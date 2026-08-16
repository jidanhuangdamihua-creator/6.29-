"""Fail-closed calendarization for the frozen D2 source interval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .experiment_protocol import ProtocolViolation, normalize_source_key


D2_SOURCE_CALENDARIZATION_RULE_VERSION = "d2_source_calendarization_v1"
D2_SOURCE_INTERVAL_START = date(2018, 1, 2)
D2_SOURCE_INTERVAL_END = date(2018, 6, 30)
D2_SOURCE_GROUP_COLS = ("brand_id", "item_id")
D2_SOURCE_MISSING_DATES = (
    "2018-04-01",
    "2018-04-25",
    "2018-05-01",
    "2018-06-02",
)
D2_FROZEN_SOURCE_CANDIDATE_KEYS = tuple(
    (str(brand), str(item))
    for brand in range(1, 4)
    for item in range(1, 10)
)

_EXPECTED_DATES = pd.date_range(
    D2_SOURCE_INTERVAL_START,
    D2_SOURCE_INTERVAL_END,
    freq="D",
)
_ALLOWED_MISSING_DATES = frozenset(pd.to_datetime(D2_SOURCE_MISSING_DATES))
_CALENDAR_FIELDS = frozenset({"year", "month", "week", "day", "day_of_week"})
_STATIC_FIELDS = frozenset({"entity_id", "brand", "item", "brand_id", "item_id"})


@dataclass(frozen=True)
class D2SourceCalendarizationReport:
    rule_version: str
    source_interval_start: str
    source_interval_end: str
    source_entity_keys: tuple[tuple[str, ...], ...]
    synthetic_entity_date_keys: tuple[tuple[tuple[str, ...], str], ...]
    synthetic_row_count: int
    source_authority_digest: str
    consumer_frame_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "source_interval_start": self.source_interval_start,
            "source_interval_end": self.source_interval_end,
            "source_entity_keys": [list(key) for key in self.source_entity_keys],
            "synthetic_entity_date_keys": [
                {"source_key": list(key), "date": date_text}
                for key, date_text in self.synthetic_entity_date_keys
            ],
            "synthetic_row_count": int(self.synthetic_row_count),
            "source_authority_digest": self.source_authority_digest,
            "consumer_frame_fingerprint": self.consumer_frame_fingerprint,
        }


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
        return value.normalize().strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ProtocolViolation("D2 digest received a non-finite value")
        return format(value, ".17g")
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _canonical_frame(frame: pd.DataFrame) -> dict[str, object]:
    columns = list(frame.columns)
    order = [column for column in (*D2_SOURCE_GROUP_COLS, "date") if column in columns]
    ordered = frame.copy()
    if "date" in ordered.columns:
        ordered["date"] = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    ordered = ordered.sort_values(order, kind="mergesort").reset_index(drop=True)
    return {
        "columns": columns,
        "rows": [
            [_canonical_value(value) for value in row]
            for row in ordered.loc[:, columns].itertuples(index=False, name=None)
        ],
    }


def _sha256_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_source_role(source_df: pd.DataFrame) -> None:
    if source_df.attrs.get("split_role") != "source":
        raise ProtocolViolation("D2 calendarizer requires source role")


def _normalize_frame_keys(source_df: pd.DataFrame) -> pd.Series:
    return source_df.loc[:, list(D2_SOURCE_GROUP_COLS)].apply(
        lambda row: normalize_source_key(tuple(row.tolist())),
        axis=1,
    )


def canonical_d2_entity_id(brand_id: object) -> str:
    """Return the canonical D2 domain identity produced from ``brand_id``."""

    try:
        missing = pd.isna(brand_id)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ProtocolViolation("D2 brand_id may not be missing")
    return normalize_source_key((brand_id,))[0]


def _expected_d2_entity_ids(source_df: pd.DataFrame) -> pd.Series:
    if "brand_id" not in source_df.columns:
        raise ProtocolViolation("D2 entity identity requires brand_id")
    return source_df["brand_id"].map(canonical_d2_entity_id)


def verify_d2_source_entity_identity(source_frame: pd.DataFrame) -> None:
    """Require every D2 ``entity_id`` to equal canonical ``brand_id`` identity."""

    if "entity_id" not in source_frame.columns:
        raise ProtocolViolation("D2 sealed source missing columns: ['entity_id']")
    if source_frame["entity_id"].isna().any():
        raise ProtocolViolation("D2 sealed source entity_id contains missing values")
    expected = _expected_d2_entity_ids(source_frame)
    actual = source_frame["entity_id"].map(lambda value: normalize_source_key((value,))[0])
    mismatch = actual.ne(expected)
    if mismatch.any():
        offending = source_frame.loc[
            mismatch, ["brand_id", "item_id", "entity_id"]
        ].drop_duplicates()
        raise ProtocolViolation(
            "D2 sealed source entity_id does not match canonical brand identity: "
            f"{offending.to_dict(orient='records')!r}"
        )


def repair_d2_source_entity_identity(
    source_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Repair only missing D2 entity identities on approved sealed dates."""

    _assert_source_role(source_frame)
    required = (*D2_SOURCE_GROUP_COLS, "date", "entity_id")
    missing_columns = [column for column in required if column not in source_frame.columns]
    if missing_columns:
        raise ProtocolViolation(
            f"D2 source entity-identity repair missing columns: {missing_columns}"
        )

    result = source_frame.copy()
    result.attrs = source_frame.attrs.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ProtocolViolation("D2 source entity-identity repair received invalid dates")

    expected = _expected_d2_entity_ids(result)
    missing = result["entity_id"].isna()
    actual = result["entity_id"].map(
        lambda value: None if pd.isna(value) else normalize_source_key((value,))[0]
    )
    inconsistent = ~missing & actual.ne(expected)
    if inconsistent.any():
        offending = result.loc[
            inconsistent, [*D2_SOURCE_GROUP_COLS, "date", "entity_id"]
        ]
        raise ProtocolViolation(
            "D2 source entity_id does not match canonical brand identity: "
            f"{offending.to_dict(orient='records')!r}"
        )

    outside_approved = missing & ~result["date"].isin(_ALLOWED_MISSING_DATES)
    if outside_approved.any():
        offending = result.loc[
            outside_approved, [*D2_SOURCE_GROUP_COLS, "date"]
        ]
        raise ProtocolViolation(
            "D2 source entity_id is missing outside approved repair dates: "
            f"{offending.to_dict(orient='records')!r}"
        )

    result.loc[missing, "entity_id"] = expected.loc[missing]
    verify_d2_source_entity_identity(result)
    repair_dates = sorted(
        result.loc[missing, "date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )
    evidence = {
        "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        "repair_dates": repair_dates,
        "approved_dates": list(D2_SOURCE_MISSING_DATES),
        "repaired_row_count": int(missing.sum()),
        "changed_cell_count": int(missing.sum()),
        "identity_column": "entity_id",
        "canonical_constructor": "canonical_d2_entity_id_from_brand_id",
        "row_count": int(len(result)),
    }
    return result, evidence


def _calendar_value(column: str, timestamp: pd.Timestamp) -> object:
    if column == "year":
        return int(timestamp.year)
    if column == "month":
        return int(timestamp.month)
    if column == "week":
        return int(timestamp.isocalendar().week)
    if column == "day":
        return int(timestamp.day)
    if column == "day_of_week":
        return int(timestamp.dayofweek)
    raise ProtocolViolation(f"unsupported D2 calendar field: {column!r}")


def _is_static_field(column: str) -> bool:
    name = str(column).strip().lower()
    return name in _STATIC_FIELDS or name.endswith("_id") or name.endswith("_code")


def slice_d2_source_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    """Slice a source-role frame to the one frozen 180-day D2 interval."""

    _assert_source_role(source_df)
    if "date" not in source_df.columns:
        raise ProtocolViolation("D2 source slicing requires date column")
    parsed = pd.to_datetime(source_df["date"], errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ProtocolViolation("D2 source slicing received invalid date")
    sliced = source_df.copy()
    sliced["date"] = parsed
    sliced = sliced.loc[
        sliced["date"].between(
            pd.Timestamp(D2_SOURCE_INTERVAL_START),
            pd.Timestamp(D2_SOURCE_INTERVAL_END),
            inclusive="both",
        )
    ].copy()
    if sliced.empty:
        raise ProtocolViolation("D2 source slice is empty")
    sliced.attrs["d2_source_slicing_complete"] = True
    sliced.attrs["d2_source_interval_start"] = D2_SOURCE_INTERVAL_START.isoformat()
    sliced.attrs["d2_source_interval_end"] = D2_SOURCE_INTERVAL_END.isoformat()
    return sliced


def calendarize_d2_source_frame(
    source_slice: pd.DataFrame,
    *,
    candidate_keys: Iterable[Sequence[object]],
) -> tuple[pd.DataFrame, D2SourceCalendarizationReport]:
    """Calendarize only the frozen D2 source candidates, fail-closed."""

    _assert_source_role(source_slice)
    if source_slice.attrs.get("d2_source_slicing_complete") is not True:
        raise ProtocolViolation("D2 calendarizer requires completed source slicing")
    missing_columns = [
        column
        for column in (*D2_SOURCE_GROUP_COLS, "date", "sales")
        if column not in source_slice.columns
    ]
    if missing_columns:
        raise ProtocolViolation(f"D2 source calendarizer missing columns: {missing_columns}")

    candidates = tuple(normalize_source_key(key) for key in candidate_keys)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ProtocolViolation("D2 source candidate keys must be non-empty and unique")

    source = source_slice.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    if source["date"].isna().any():
        raise ProtocolViolation("D2 source calendarizer received invalid date")
    if not source["date"].between(
        pd.Timestamp(D2_SOURCE_INTERVAL_START),
        pd.Timestamp(D2_SOURCE_INTERVAL_END),
        inclusive="both",
    ).all():
        raise ProtocolViolation("D2 source calendarizer received dates outside frozen interval")

    source_keys = _normalize_frame_keys(source)
    if source.duplicated([*D2_SOURCE_GROUP_COLS, "date"]).any():
        raise ProtocolViolation("D2 source contains duplicate entity/date keys")
    numeric_sales = pd.to_numeric(source["sales"], errors="coerce")
    if numeric_sales.isna().any() or not np.isfinite(numeric_sales.to_numpy(dtype=np.float64)).all():
        raise ProtocolViolation("D2 source sales contain non-finite values")
    source["sales"] = numeric_sales.astype(float)
    verify_d2_source_entity_identity(source)

    actual_keys = set(source_keys.tolist())
    missing_entities = sorted(set(candidates).difference(actual_keys))
    if missing_entities:
        raise ProtocolViolation(f"D2 source candidate entity missing: {missing_entities!r}")

    rows = []
    synthetic_keys: list[tuple[tuple[str, ...], str]] = []
    for candidate in candidates:
        entity = source.loc[source_keys.map(lambda key: key == candidate)].copy()
        actual_dates = pd.DatetimeIndex(entity["date"])
        missing_dates = tuple(_EXPECTED_DATES.difference(actual_dates))
        unsupported = tuple(
            timestamp for timestamp in missing_dates if timestamp not in _ALLOWED_MISSING_DATES
        )
        if unsupported:
            rendered = [timestamp.strftime("%Y-%m-%d") for timestamp in unsupported]
            raise ProtocolViolation(f"unsupported missing source dates: {rendered!r}")

        rows.extend(entity.to_dict(orient="records"))
        representative = entity.iloc[0]
        for timestamp in missing_dates:
            row = {column: pd.NA for column in source.columns}
            for column in source.columns:
                if column in D2_SOURCE_GROUP_COLS or _is_static_field(column):
                    row[column] = representative[column]
            if "entity_id" in source.columns:
                row["entity_id"] = canonical_d2_entity_id(row["brand_id"])
            row["date"] = timestamp
            row["sales"] = 0.0
            for column in source.columns:
                if column in _CALENDAR_FIELDS:
                    row[column] = _calendar_value(column, timestamp)
            rows.append(row)
            synthetic_keys.append((candidate, timestamp.strftime("%Y-%m-%d")))

    result = pd.DataFrame(rows, columns=list(source.columns))
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["sales"] = pd.to_numeric(result["sales"], errors="raise").astype(float)
    for column in _CALENDAR_FIELDS.intersection(result.columns):
        result[column] = result["date"].map(
            lambda timestamp, field=column: _calendar_value(field, timestamp)
        )
    result = result.sort_values([*D2_SOURCE_GROUP_COLS, "date"], kind="mergesort").reset_index(
        drop=True
    )
    result.attrs = source_slice.attrs.copy()

    result_keys = _normalize_frame_keys(result)
    for candidate in candidates:
        candidate_dates = pd.DatetimeIndex(
            result.loc[result_keys.map(lambda key: key == candidate), "date"]
        )
        if not candidate_dates.equals(_EXPECTED_DATES):
            raise ProtocolViolation(
                f"D2 source candidate does not cover exact 180 days: {candidate!r}"
            )

    canonical_frame = _canonical_frame(result)
    source_payload = {
        "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        "source_interval": [
            D2_SOURCE_INTERVAL_START.isoformat(),
            D2_SOURCE_INTERVAL_END.isoformat(),
        ],
        "candidate_keys": [list(key) for key in sorted(candidates)],
        "synthetic_entity_date_keys": [
            {"source_key": list(key), "date": date_text}
            for key, date_text in synthetic_keys
        ],
        "frame": canonical_frame,
    }
    source_authority_digest = _sha256_payload(source_payload)
    consumer_frame_fingerprint = _sha256_payload(
        {
            "consumer": "d2_source_completeness_knn_v1",
            "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
            "source_authority_digest": source_authority_digest,
            "frame": canonical_frame,
        }
    )
    report = D2SourceCalendarizationReport(
        rule_version=D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        source_interval_start=D2_SOURCE_INTERVAL_START.isoformat(),
        source_interval_end=D2_SOURCE_INTERVAL_END.isoformat(),
        source_entity_keys=tuple(sorted(candidates)),
        synthetic_entity_date_keys=tuple(synthetic_keys),
        synthetic_row_count=len(synthetic_keys),
        source_authority_digest=source_authority_digest,
        consumer_frame_fingerprint=consumer_frame_fingerprint,
    )
    result.attrs.update(
        {
            "d2_source_calendarization_rule_version": report.rule_version,
            "d2_source_authority_digest": report.source_authority_digest,
            "d2_consumer_frame_fingerprint": report.consumer_frame_fingerprint,
            "d2_synthetic_source_row_count": report.synthetic_row_count,
            "d2_synthetic_source_entity_date_keys": report.to_dict()[
                "synthetic_entity_date_keys"
            ],
            "d2_source_calendarization_report": report.to_dict(),
        }
    )
    return result, report


def repair_d2_source_calendar_fields(
    source_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Repair date-derived fields only on the four approved source dates.

    The sealed source stores ``date`` and its calendar projections as separate
    columns.  This producer repairs an already materialized frame without
    changing demand, promotion, identity, or row/date keys.  Any non-finite
    or date-inconsistent calendar field outside the approved repair dates is a
    hard protocol violation.
    """

    _assert_source_role(source_frame)
    required = (
        *D2_SOURCE_GROUP_COLS,
        "date",
        "sales",
        "promo",
        "year",
        "month",
        "week",
        "day",
    )
    missing_columns = [column for column in required if column not in source_frame.columns]
    if missing_columns:
        raise ProtocolViolation(
            f"D2 source calendar-field repair missing columns: {missing_columns}"
        )

    result = source_frame.copy()
    result.attrs = source_frame.attrs.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ProtocolViolation("D2 source calendar-field repair received invalid dates")

    approved_mask = result["date"].isin(_ALLOWED_MISSING_DATES)
    mismatch_masks: dict[str, pd.Series] = {}
    for field in ("year", "month", "week", "day"):
        numeric = pd.to_numeric(result[field], errors="coerce")
        expected = result["date"].map(
            lambda timestamp, field=field: _calendar_value(field, timestamp)
        )
        mismatch_masks[field] = numeric.isna() | ~np.isclose(
            numeric.to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            equal_nan=False,
        )

    mismatch_frame = pd.DataFrame(mismatch_masks, index=result.index)
    outside_approved = mismatch_frame.any(axis=1) & ~approved_mask
    if outside_approved.any():
        offending = result.loc[outside_approved, [*D2_SOURCE_GROUP_COLS, "date"]]
        raise ProtocolViolation(
            "D2 source calendar fields are non-finite or date-inconsistent "
            f"outside approved repair dates: {offending.to_dict(orient='records')!r}"
        )

    changed = mismatch_frame.any(axis=1) & approved_mask
    for field in ("year", "month", "week", "day"):
        expected = result["date"].map(
            lambda timestamp, field=field: _calendar_value(field, timestamp)
        )
        result.loc[approved_mask & changed, field] = expected.loc[approved_mask & changed]

    remaining_nonfinite = {
        field: int(
            (~np.isfinite(pd.to_numeric(result[field], errors="coerce").to_numpy(dtype=np.float64)))
            .sum()
        )
        for field in ("year", "month", "week", "day")
    }
    if any(remaining_nonfinite.values()):
        raise ProtocolViolation(
            f"D2 source calendar-field repair left non-finite values: {remaining_nonfinite}"
        )

    repaired_dates = sorted(
        result.loc[approved_mask & changed, "date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )
    evidence = {
        "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        "repair_dates": repaired_dates,
        "approved_dates": list(D2_SOURCE_MISSING_DATES),
        "repaired_row_count": int(changed.sum()),
        "changed_cell_count": int(mismatch_frame.loc[approved_mask].to_numpy(dtype=bool).sum()),
        "calendar_fields": ["year", "month", "week", "day"],
        "sales_unchanged": True,
        "promo_unchanged": True,
        "row_count": int(len(result)),
    }
    return result, evidence


def verify_d2_source_frame(
    source_slice: pd.DataFrame,
    *,
    candidate_keys: Iterable[Sequence[object]],
) -> tuple[pd.DataFrame, D2SourceCalendarizationReport]:
    """Verify an already-calendarized D2 source without adding or changing rows."""

    _assert_source_role(source_slice)
    if source_slice.attrs.get("d2_source_slicing_complete") is not True:
        raise ProtocolViolation("D2 source verification requires completed source slicing")
    required = (*D2_SOURCE_GROUP_COLS, "date", "entity_id", "sales")
    missing_columns = [column for column in required if column not in source_slice.columns]
    if missing_columns:
        raise ProtocolViolation(f"D2 sealed source missing columns: {missing_columns}")

    candidates = tuple(normalize_source_key(key) for key in candidate_keys)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ProtocolViolation("D2 source candidate keys must be non-empty and unique")

    source = source_slice.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    if source["date"].isna().any():
        raise ProtocolViolation("D2 sealed source contains invalid date")
    if source.duplicated([*D2_SOURCE_GROUP_COLS, "date"]).any():
        raise ProtocolViolation("D2 sealed source contains duplicate entity/date keys")
    numeric_sales = pd.to_numeric(source["sales"], errors="coerce")
    if numeric_sales.isna().any() or not np.isfinite(numeric_sales.to_numpy(dtype=np.float64)).all():
        raise ProtocolViolation("D2 sealed source sales contain non-finite values")
    source["sales"] = numeric_sales.astype(float)
    verify_d2_source_entity_identity(source)

    _, calendar_field_evidence = repair_d2_source_calendar_fields(source)
    if int(calendar_field_evidence["changed_cell_count"]) != 0:
        raise ProtocolViolation(
            "D2 sealed source calendar repair requires producer repair: "
            f"{calendar_field_evidence!r}"
        )

    source_keys = _normalize_frame_keys(source)
    actual_keys = set(source_keys.tolist())
    missing_entities = sorted(set(candidates).difference(actual_keys))
    if missing_entities:
        raise ProtocolViolation(f"D2 sealed source candidate entity missing: {missing_entities!r}")

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        entity = source.loc[source_keys.map(lambda key: key == candidate)].copy()
        actual_dates = pd.DatetimeIndex(entity["date"])
        missing_dates = tuple(_EXPECTED_DATES.difference(actual_dates))
        if missing_dates:
            rendered = [timestamp.strftime("%Y-%m-%d") for timestamp in missing_dates]
            raise ProtocolViolation(
                f"D2 sealed source does not cover exact 180 days: {rendered!r}"
            )
        frozen_dates = entity.loc[entity["date"].isin(_ALLOWED_MISSING_DATES), "sales"]
        if not frozen_dates.eq(0).all():
            raise ProtocolViolation("D2 sealed source frozen-date sales must be zero")
        rows.extend(entity.to_dict(orient="records"))

    result = pd.DataFrame(rows, columns=list(source.columns))
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["sales"] = pd.to_numeric(result["sales"], errors="raise").astype(float)
    result = result.sort_values([*D2_SOURCE_GROUP_COLS, "date"], kind="mergesort").reset_index(drop=True)
    result.attrs = source_slice.attrs.copy()

    canonical_frame = _canonical_frame(result)
    source_payload = {
        "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        "source_interval": [
            D2_SOURCE_INTERVAL_START.isoformat(),
            D2_SOURCE_INTERVAL_END.isoformat(),
        ],
        "candidate_keys": [list(key) for key in sorted(candidates)],
        "synthetic_entity_date_keys": [],
        "frame": canonical_frame,
    }
    source_authority_digest = _sha256_payload(source_payload)
    consumer_frame_fingerprint = _sha256_payload(
        {
            "consumer": "d2_source_completeness_knn_v1",
            "rule_version": D2_SOURCE_CALENDARIZATION_RULE_VERSION,
            "source_authority_digest": source_authority_digest,
            "frame": canonical_frame,
        }
    )
    report = D2SourceCalendarizationReport(
        rule_version=D2_SOURCE_CALENDARIZATION_RULE_VERSION,
        source_interval_start=D2_SOURCE_INTERVAL_START.isoformat(),
        source_interval_end=D2_SOURCE_INTERVAL_END.isoformat(),
        source_entity_keys=tuple(sorted(candidates)),
        synthetic_entity_date_keys=(),
        synthetic_row_count=0,
        source_authority_digest=source_authority_digest,
        consumer_frame_fingerprint=consumer_frame_fingerprint,
    )
    result.attrs.update(
        {
            "d2_source_calendarization_rule_version": report.rule_version,
            "d2_source_authority_digest": report.source_authority_digest,
            "d2_consumer_frame_fingerprint": report.consumer_frame_fingerprint,
            "d2_synthetic_source_row_count": 0,
            "d2_synthetic_source_entity_date_keys": [],
            "d2_source_calendarization_report": report.to_dict(),
        }
    )
    return result, report


def build_d2_sealed_identity(
    *,
    rule_version: str,
    source_authority_digest: str,
    consumer_frame_fingerprint: str,
    candidate_pool_digest: str,
    selection_result_digest: str,
) -> str:
    """Build the final D2 sealed identity from every upstream identity."""

    return _sha256_payload(
        {
            "rule_version": str(rule_version),
            "source_authority_digest": str(source_authority_digest),
            "consumer_frame_fingerprint": str(consumer_frame_fingerprint),
            "candidate_pool_digest": str(candidate_pool_digest),
            "selection_result_digest": str(selection_result_digest),
        }
    )
