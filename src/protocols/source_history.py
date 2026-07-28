"""Exact Gregorian source-history eligibility for D4--D6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import pandas as pd

from src.constants import SOURCE_HISTORY_CALENDAR, SOURCE_HISTORY_COMPLETENESS_POLICY
from src.utils.dataframe_attrs import copy_frame_with_lightweight_attrs


SOURCE_HISTORY_DIGEST_CHUNK_ROWS = 100_000
SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS = 10_000
SOURCE_HISTORY_DIGEST_LARGE_FRAME_ROWS = 1_000_000
SOURCE_HISTORY_DIGEST_ALGORITHM = "source_history_frame_digest_v2"


def _normalize_key(values: Sequence[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if value is None or value is pd.NA:
            raise ValueError("source history entity key contains null")
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            raise ValueError("source history entity key contains null")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = str(value).strip()
        if not text:
            raise ValueError("source history entity key contains an empty component")
        normalized.append(text)
    if not normalized:
        raise ValueError("source history entity key is empty")
    return tuple(normalized)


def expected_source_history_dates(
    origin: object,
    *,
    source_history_days: int,
) -> pd.DatetimeIndex:
    """Return the inclusive, normalized Gregorian source-history calendar."""
    days = int(source_history_days)
    if isinstance(source_history_days, bool) or days <= 0:
        raise ValueError("source_history_days must be a positive integer")
    end = pd.Timestamp(origin)
    if pd.isna(end):
        raise ValueError(f"invalid source-history origin: {origin!r}")
    end = end.normalize()
    return pd.DatetimeIndex(pd.date_range(end=end, periods=days, freq="D")).normalize()


@dataclass(frozen=True)
class SourceHistoryEligibility:
    """The exact eligible candidate frame and auditable rejection facts."""

    candidate_frame: pd.DataFrame
    eligible_keys: tuple[tuple[str, ...], ...]
    incomplete_keys: Mapping[tuple[str, ...], int]
    duplicate_keys: tuple[tuple[str, ...], ...]
    expected_dates: pd.DatetimeIndex
    outside_window_row_count: int

    @property
    def expected_count(self) -> int:
        return int(len(self.expected_dates))


def source_history_frame_digest(
    frame: pd.DataFrame,
    *,
    key_fields: Sequence[str],
) -> str:
    """Digest the normalized, sorted candidate-frame bytes and column order."""
    fields = tuple(str(field) for field in key_fields)
    missing = [field for field in (*fields, "date") if field not in frame.columns]
    if missing:
        raise ValueError(f"source history frame is missing columns: {missing!r}")
    canonical_order = frame.attrs.get("source_history_canonical_order") is True
    if canonical_order:
        ordered = copy_frame_with_lightweight_attrs(frame, deep=False)
        dates = pd.to_datetime(ordered["date"], errors="coerce")
        if dates.isna().any() or not dates.equals(dates.dt.normalize()):
            raise ValueError("source history frame contains invalid or non-normalized dates")
    else:
        ordered = copy_frame_with_lightweight_attrs(frame)
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce").dt.normalize()
        if ordered["date"].isna().any():
            raise ValueError("source history frame contains invalid dates")
        ordered = ordered.sort_values([*fields, "date"], kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(SOURCE_HISTORY_DIGEST_ALGORITHM.encode("utf-8"))
    digest.update("\x1f".join(map(str, ordered.columns)).encode("utf-8"))
    chunk_rows = int(SOURCE_HISTORY_DIGEST_CHUNK_ROWS)
    if chunk_rows <= 0:
        raise ValueError("SOURCE_HISTORY_DIGEST_CHUNK_ROWS must be positive")
    small_frame_rows = int(SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS)
    if small_frame_rows < 0:
        raise ValueError("SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS must be non-negative")
    if len(ordered) <= small_frame_rows:
        digest.update(b"small_frame_json")
        digest.update(
            ordered.to_json(
                orient="split",
                date_format="iso",
                date_unit="ns",
                double_precision=15,
                force_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()
    # Keep one column's temporary hash buffer alive at a time.  On wide,
    # object-heavy D5 frames, row-major slicing retains several temporary
    # object blocks across iterations and can make the process exceed its
    # memory limit even though each individual hash is small.
    for column in ordered.columns:
        digest.update(str(column).encode("utf-8"))
        if len(ordered) >= int(SOURCE_HISTORY_DIGEST_LARGE_FRAME_ROWS):
            digest.update(
                pd.util.hash_pandas_object(ordered[column], index=False).to_numpy().tobytes()
            )
            continue
        for start in range(0, len(ordered), chunk_rows):
            column_chunk = ordered[column].iloc[start : start + chunk_rows]
            digest.update(
                pd.util.hash_pandas_object(column_chunk, index=False).to_numpy().tobytes()
            )
    return digest.hexdigest()


def build_exact_source_history_candidate_frame(
    frame: pd.DataFrame,
    *,
    key_fields: Sequence[str],
    origin: object,
    source_history_days: int,
) -> SourceHistoryEligibility:
    """Filter a source frame and keep only exact Gregorian-date candidates.

    Rows outside the inclusive expected window are discarded before eligibility
    is evaluated.  A candidate must have the exact expected date set and no
    duplicate entity/date in that window; row count or non-null value count is
    never used as a substitute for the date-set check.
    """
    fields = tuple(str(field) for field in key_fields)
    if not fields:
        raise ValueError("source history requires at least one entity key field")
    missing = [field for field in (*fields, "date") if field not in frame.columns]
    if missing:
        raise ValueError(f"source history frame is missing columns: {missing!r}")

    expected = expected_source_history_dates(
        origin,
        source_history_days=source_history_days,
    )
    start = expected[0]
    end = expected[-1]

    if frame.attrs.get("source_history_prevalidated_exact") is True:
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        in_window = dates.between(start, end, inclusive="both")
        duplicate = frame.duplicated([*fields, "date"]).any()
        if not dates.isna().any() and bool(in_window.all()) and not duplicate:
            group_sizes = frame.groupby(list(fields), sort=True, dropna=False).size()
            if not group_sizes.empty and bool(group_sizes.eq(len(expected)).all()):
                eligible = tuple(
                    sorted(
                        _normalize_key(
                            raw_key if isinstance(raw_key, tuple) else (raw_key,)
                        )
                        for raw_key in group_sizes.index
                    )
                )
                candidate = copy_frame_with_lightweight_attrs(frame, deep=False)
                candidate.attrs.update(
                    {
                        "source_history_days": int(source_history_days),
                        "source_history_start": start,
                        "source_history_end": end,
                        "source_history_expected_date_count": int(len(expected)),
                        "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
                        "source_history_calendar": SOURCE_HISTORY_CALENDAR,
                        "source_history_inclusive_end": True,
                        "source_history_eligible_key_count": len(eligible),
                        "source_history_incomplete_key_count": 0,
                        "source_history_duplicate_key_count": 0,
                        "source_history_outside_window_row_count": 0,
                        "source_history_validation_path": "prevalidated_calendarized",
                    }
                )
                return SourceHistoryEligibility(
                    candidate_frame=candidate,
                    eligible_keys=eligible,
                    incomplete_keys={},
                    duplicate_keys=(),
                    expected_dates=expected,
                    outside_window_row_count=0,
                )

    prepared = copy_frame_with_lightweight_attrs(frame)
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any():
        raise ValueError("source history frame contains invalid dates")
    for field in fields:
        values = prepared[field]
        if values.isna().any() or values.astype("string").str.strip().eq("").any():
            raise ValueError(f"source history entity key contains null or empty values: {field!r}")
    in_window = prepared["date"].between(start, end, inclusive="both")
    window = prepared.loc[in_window].copy()
    outside_count = int((~in_window).sum())
    # The boundary filter above guarantees that every retained date belongs to
    # the finite expected Gregorian calendar.  Therefore, for a group inside
    # that bounded frame, ``row_count == unique_date_count == expected_count``
    # is equivalent to exact expected-date-set equality, provided duplicate
    # entity-date rows are rejected explicitly.  Keep the duplicate check
    # separate so a repeated date can never be mistaken for a complete set.
    grouped = window.groupby(list(fields), sort=True, dropna=False, observed=True)
    row_counts = grouped.size()
    unique_date_counts = grouped["date"].nunique(dropna=True)
    duplicate_rows = window.duplicated([*fields, "date"], keep=False)
    duplicate_counts = (
        window.loc[duplicate_rows]
        .groupby(list(fields), sort=True, dropna=False, observed=True)
        .size()
    )
    eligible_mask = row_counts.eq(len(expected)) & unique_date_counts.eq(len(expected))
    if not duplicate_counts.empty:
        eligible_mask &= ~row_counts.index.isin(duplicate_counts.index)

    def _normalized_group_key(raw_key: object) -> tuple[str, ...]:
        return _normalize_key(raw_key if isinstance(raw_key, tuple) else (raw_key,))

    eligible_raw_keys = row_counts.index[eligible_mask]
    eligible = [_normalized_group_key(raw_key) for raw_key in eligible_raw_keys]
    incomplete_keys = row_counts.index[~eligible_mask]
    incomplete_counts = unique_date_counts.loc[~eligible_mask].to_numpy()
    incomplete = {
        _normalized_group_key(raw_key): int(count)
        for raw_key, count in zip(incomplete_keys.tolist(), incomplete_counts)
    }
    duplicate_keys = [
        _normalized_group_key(raw_key)
        for raw_key in duplicate_counts.index
    ]

    eligible_set = set(eligible)
    eligible_raw_set = set(eligible_raw_keys.tolist())
    if len(fields) == 1:
        candidate_mask = window[fields[0]].isin(eligible_raw_set)
    else:
        candidate_index = pd.MultiIndex.from_frame(window.loc[:, list(fields)])
        candidate_mask = candidate_index.isin(eligible_raw_set)
    candidate = window.loc[candidate_mask].copy()
    candidate = candidate.sort_values([*fields, "date"], kind="mergesort").reset_index(drop=True)
    candidate.attrs.update(
        {
            "source_history_days": int(source_history_days),
            "source_history_start": start,
            "source_history_end": end,
            "source_history_expected_date_count": int(len(expected)),
            "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
            "source_history_calendar": SOURCE_HISTORY_CALENDAR,
            "source_history_inclusive_end": True,
            "source_history_eligible_key_count": len(eligible),
            "source_history_incomplete_key_count": len(incomplete),
            "source_history_duplicate_key_count": len(duplicate_keys),
            "source_history_outside_window_row_count": outside_count,
        }
    )
    return SourceHistoryEligibility(
        candidate_frame=candidate,
        eligible_keys=tuple(sorted(eligible)),
        incomplete_keys=dict(sorted(incomplete.items())),
        duplicate_keys=tuple(sorted(duplicate_keys)),
        expected_dates=expected,
        outside_window_row_count=outside_count,
    )


__all__ = [
    "SOURCE_HISTORY_COMPLETENESS_POLICY",
    "SourceHistoryEligibility",
    "build_exact_source_history_candidate_frame",
    "expected_source_history_dates",
    "source_history_frame_digest",
]
