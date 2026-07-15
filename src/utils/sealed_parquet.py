"""Fail-closed projection and date-pushdown reads for sealed Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq


class SealedParquetProjectionError(ValueError):
    """Raised when a sealed Parquet projection cannot be proven safe."""


def _normalise_columns(columns: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(column) for column in columns)
    if not values:
        raise SealedParquetProjectionError("sealed projection requires explicit columns")
    if len(values) != len(set(values)):
        raise SealedParquetProjectionError("sealed projection contains duplicate columns")
    return values


def _normalise_bound(value: object, *, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise SealedParquetProjectionError(f"invalid {label}: {value!r}")
    return pd.Timestamp(parsed).normalize()


def read_sealed_projection(
    path: str | Path,
    *,
    columns: Sequence[str],
    date_col: str = "date",
    date_start: object | None = None,
    date_end: object | None = None,
    expected_arrow_dtypes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read exactly one declared projection from one sealed Parquet artifact.

    The Arrow reader receives both the explicit projection and date filters.  A
    second in-memory date check is intentional: it proves the returned frame is
    within the requested interval even when a Parquet implementation ignores a
    filter for an unusual physical date representation.
    """

    artifact = Path(path)
    if not artifact.is_file():
        raise SealedParquetProjectionError(f"sealed Parquet artifact is missing: {artifact}")
    requested = _normalise_columns(columns)
    date_name = str(date_col)
    if date_name not in requested:
        raise SealedParquetProjectionError(
            f"sealed projection must include its date column: {date_name!r}"
        )

    try:
        schema = pq.read_schema(artifact)
    except Exception as exc:  # pragma: no cover - exact backend exception is not stable
        raise SealedParquetProjectionError(f"sealed Parquet schema is unreadable: {artifact}") from exc
    available = tuple(schema.names)
    missing = [column for column in requested if column not in available]
    if missing:
        raise SealedParquetProjectionError(
            f"sealed projection has unknown or missing columns: {missing}"
        )
    if expected_arrow_dtypes:
        mismatches = {
            column: (str(schema.field(column).type), str(expected_arrow_dtypes[column]))
            for column in requested
            if column in expected_arrow_dtypes
            and str(schema.field(column).type) != str(expected_arrow_dtypes[column])
        }
        if mismatches:
            raise SealedParquetProjectionError(f"sealed projection dtype mismatch: {mismatches}")

    lower = _normalise_bound(date_start, label="date_start") if date_start is not None else None
    upper = _normalise_bound(date_end, label="date_end") if date_end is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise SealedParquetProjectionError("date_start is after date_end")

    filters = []
    if lower is not None:
        filters.append((date_name, ">=", lower.to_pydatetime()))
    if upper is not None:
        filters.append((date_name, "<=", upper.to_pydatetime()))
    try:
        table = pq.read_table(
            artifact,
            columns=list(requested),
            filters=filters or None,
        )
        frame = table.to_pandas()
    except Exception as exc:  # pragma: no cover - exact backend exception is not stable
        raise SealedParquetProjectionError(f"sealed Parquet projection is unreadable: {artifact}") from exc

    if tuple(frame.columns) != requested:
        frame = frame.loc[:, list(requested)].copy()
    frame[date_name] = pd.to_datetime(frame[date_name], errors="coerce").dt.normalize()
    if frame[date_name].isna().any():
        raise SealedParquetProjectionError("sealed projection contains invalid dates")
    if lower is not None:
        frame = frame.loc[frame[date_name] >= lower].copy()
    if upper is not None:
        frame = frame.loc[frame[date_name] <= upper].copy()
    return frame.reset_index(drop=True)


def read_sealed_parquet(*args, **kwargs) -> pd.DataFrame:
    """Compatibility alias with the same fail-closed projection contract."""

    return read_sealed_projection(*args, **kwargs)


__all__ = [
    "SealedParquetProjectionError",
    "read_sealed_parquet",
    "read_sealed_projection",
]
