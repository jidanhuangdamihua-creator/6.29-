"""Build one additional target's 210-day series from the sealed target authority.

D5 targets are read from ``dataset5/source.parquet`` after dropping dates after
2017-08-14. ``prepared_180day_source.parquet`` is the source calendar only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from additional_target_robustness.paths import SEALED_ROOT
from additional_target_robustness.protocol import (
    TRAIN_DAYS,
    VAL_DAYS,
    TEST_DAYS,
    Calendar,
    ProtocolError,
    calendar_for,
)


def target_authority_path(dataset: str) -> Path:
    calendar = calendar_for(dataset)
    path = SEALED_ROOT / calendar.target_filename
    if dataset == "D5" and path.name != "source.parquet":
        raise ProtocolError("D5 target authority must be source.parquet")
    if "prepared_180day_source" in path.name:
        raise ProtocolError("prepared_180day_source.parquet cannot supply a target series")
    return path


def source_cache_path(dataset: str) -> Path:
    return SEALED_ROOT / calendar_for(dataset).source_filename


def split_target_frame(frame: pd.DataFrame, calendar: Calendar) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = _as_daily(frame)
    _require_span(ordered, calendar.train_start, calendar.test_end, TRAIN_DAYS + VAL_DAYS + TEST_DAYS)
    train = _slice(ordered, calendar.train_start, calendar.train_end, TRAIN_DAYS)
    validation = _slice(ordered, calendar.val_start, calendar.val_end, VAL_DAYS)
    test = _slice(ordered, calendar.test_start, calendar.test_end, TEST_DAYS)
    if train["date"].max() >= validation["date"].min() or validation["date"].max() >= test["date"].min():
        raise ProtocolError(f"{calendar.dataset} target partitions overlap or are out of order")
    return train, validation, test


def load_target_series(dataset: str, identity: str) -> pd.DataFrame:
    calendar = calendar_for(dataset)
    path = target_authority_path(dataset)
    columns = _columns(calendar)
    frame = _read_filtered(
        path,
        key_fields=calendar.key_fields,
        identity=identity,
        date_min=calendar.train_start,
        date_max=calendar.test_end if dataset != "D5" else "2017-08-14",
        columns=columns,
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if dataset == "D5":
        cutoff = pd.Timestamp("2017-08-14")
        banned = pd.Timestamp("2017-08-15")
        frame = frame.loc[frame["date"] <= cutoff].copy()
        if (frame["date"] == banned).any() or (frame["date"] > cutoff).any():
            raise ProtocolError("D5 target series still contains 2017-08-15 or a later date")
    _require_identity(frame, calendar.key_fields, identity)
    for column in calendar.model_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolError(f"{dataset} {identity} has a non-finite {column}")
    train, validation, test = split_target_frame(frame, calendar)
    del train, validation, test
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


def sales_by_date(frame: pd.DataFrame) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for day, value in zip(frame["date"], frame["sales"]):
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        if key in lookup:
            raise ProtocolError(f"duplicate target date {key}")
        number = float(value)
        if not np.isfinite(number):
            raise ProtocolError(f"non-finite sales on {key}")
        lookup[key] = number
    return lookup


def _columns(calendar: Calendar) -> list[str]:
    columns = ["date", *calendar.key_fields, *calendar.model_features, *calendar.similarity_variables]
    if calendar.group_field is not None:
        columns.append(calendar.group_field)
    return list(dict.fromkeys(columns))


def _read_filtered(
    path: Path,
    *,
    key_fields: tuple[str, ...],
    identity: str,
    date_min: str,
    date_max: str,
    columns: list[str],
) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = tuple(part.strip() for part in identity.split("/"))
    if len(parts) != len(key_fields) or any(part == "" for part in parts):
        raise ProtocolError(f"target identity {identity!r} does not match {key_fields!r}")
    schema = pq.read_schema(path)
    filters: list[tuple] = []
    for field, part in zip(key_fields, parts):
        arrow_type = schema.field(field).type
        if pa.types.is_integer(arrow_type):
            value = pa.scalar(int(part), type=arrow_type)
        else:
            value = part
        filters.append((field, "==", value))
    filters.append(("date", ">=", pa.scalar(np.datetime64(date_min, "ns"))))
    filters.append(("date", "<=", pa.scalar(np.datetime64(date_max, "ns"))))
    present = [column for column in columns if column in schema.names]
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise ProtocolError(f"{path.name} is missing columns {missing}")
    table = pq.read_table(path, columns=present, filters=filters)
    return table.to_pandas()


def _as_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ProtocolError("target frame has no date column")
    ordered = frame.copy()
    ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    ordered = ordered.sort_values("date", kind="mergesort").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ProtocolError("target frame has duplicate dates")
    return ordered


def _require_span(frame: pd.DataFrame, start: str, end: str, expected: int) -> None:
    dates = pd.Series(pd.date_range(start, end, freq="D"))
    got = frame["date"].reset_index(drop=True)
    if len(dates) != expected or len(got) != expected or not got.equals(dates):
        raise ProtocolError(
            f"target span {start}..{end} has {len(got)} rows, expected {expected} ordered days"
        )


def _slice(frame: pd.DataFrame, start: str, end: str, expected: int) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].reset_index(drop=True)
    _require_span(sliced, start, end, expected)
    return sliced


def _require_identity(frame: pd.DataFrame, key_fields: tuple[str, ...], identity: str) -> None:
    expected = tuple(part.strip() for part in identity.split("/"))
    if frame.empty:
        raise ProtocolError(f"target {identity} has no sealed rows")
    for field, part in zip(key_fields, expected):
        found = {str(value).strip() for value in frame[field].tolist()}
        if found != {part}:
            raise ProtocolError(f"target {identity} {field} values are {sorted(found)!r}")
