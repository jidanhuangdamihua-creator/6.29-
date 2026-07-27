"""Authority-derived formal target windows and fail-closed scoping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .experiment_protocol import ProtocolViolation, normalize_source_key
from .gate1_transformation import dataset_contract


@dataclass(frozen=True)
class FormalTargetWindow:
    dataset_id: str
    target_keys: tuple[tuple[str, ...], ...]
    start: pd.Timestamp
    end: pd.Timestamp
    observed_end: pd.Timestamp
    forecast_horizon_days: int
    expected_days: int

    @property
    def expected_rows(self) -> int:
        return self.expected_days * len(self.target_keys)


def resolve_formal_target_window(dataset_id: object) -> FormalTargetWindow:
    """Resolve the formal window from the frozen contract, never from row position."""

    spec = dataset_contract(dataset_id)
    start = pd.Timestamp(spec.target_train_start).normalize()
    end = pd.Timestamp(spec.blind_end).normalize()
    expected_days = int((end - start).days + 1)
    return FormalTargetWindow(
        dataset_id=str(spec.dataset),
        target_keys=tuple(tuple(str(part) for part in key) for key in spec.target_keys),
        start=start,
        end=end,
        observed_end=pd.Timestamp(spec.origin).normalize(),
        forecast_horizon_days=int((end - pd.Timestamp(spec.blind_start).normalize()).days + 1),
        expected_days=expected_days,
    )


def _normalized_key_frame(frame: pd.DataFrame, key_fields: tuple[str, ...]) -> pd.Series:
    return frame.loc[:, list(key_fields)].apply(
        lambda row: normalize_source_key(tuple(row.tolist())), axis=1
    )


def evaluate_formal_target_calendar(
    frame: pd.DataFrame,
    *,
    dataset_id: object,
) -> dict[str, Any]:
    """Compare actual sealed rows to the formal date set without adding rows."""

    window = resolve_formal_target_window(dataset_id)
    spec = dataset_contract(dataset_id)
    required = (*spec.key_fields, "date")
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ProtocolViolation(
            f"formal target calendar is missing columns: {missing_columns!r}"
        )
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ProtocolViolation("formal target calendar contains invalid dates")
    keys = _normalized_key_frame(frame, spec.key_fields)
    expected_dates = pd.date_range(window.start, window.end, freq="D")
    formal_mask = dates.between(window.start, window.end, inclusive="both")
    formal = frame.loc[formal_mask].copy()
    formal_dates = dates.loc[formal_mask]
    formal_keys = keys.loc[formal_mask]
    duplicate_count = int(
        pd.DataFrame(
            {**{field: formal[field] for field in spec.key_fields}, "date": formal_dates}
        ).duplicated([*spec.key_fields, "date"]).sum()
    )
    expected_key_set = set(window.target_keys)
    actual_key_set = set(formal_keys.tolist())
    unexpected_keys = sorted(actual_key_set.difference(expected_key_set))
    missing_exact_keys: list[dict[str, object]] = []
    for target_key in window.target_keys:
        key_dates = pd.DatetimeIndex(
            formal_dates.loc[formal_keys == target_key]
        ).drop_duplicates()
        for timestamp in expected_dates.difference(key_dates):
            missing_exact_keys.append(
                {"key": list(target_key), "date": timestamp.strftime("%Y-%m-%d")}
            )
    actual_dates = set(pd.DatetimeIndex(formal_dates).drop_duplicates())
    extra_dates = [
        timestamp.strftime("%Y-%m-%d")
        for timestamp in sorted(actual_dates.difference(set(expected_dates)))
    ]
    actual = int(len(formal))
    expected = int(window.expected_rows)
    ready = not missing_exact_keys and not extra_dates and not unexpected_keys and duplicate_count == 0 and actual == expected
    return {
        "dataset_id": window.dataset_id,
        "formal_window_start": window.start.strftime("%Y-%m-%d"),
        "formal_window_end": window.end.strftime("%Y-%m-%d"),
        "actual": actual,
        "expected": expected,
        "unique_dates": int(formal_dates.nunique()),
        "missing_exact_keys": missing_exact_keys,
        "extra_dates": extra_dates,
        "unexpected_keys": [list(key) for key in unexpected_keys],
        "duplicate_exact_keys": duplicate_count,
        "ready": bool(ready),
    }


def scope_target_to_formal_window(
    frame: pd.DataFrame,
    *,
    dataset_id: object,
) -> pd.DataFrame:
    """Return the exact formal target window after validating its actual bytes."""

    window = resolve_formal_target_window(dataset_id)
    spec = dataset_contract(dataset_id)
    report = evaluate_formal_target_calendar(frame, dataset_id=dataset_id)
    if not report["ready"]:
        raise ProtocolViolation(
            "formal target scope is incomplete: "
            f"missing formal target dates={report['missing_exact_keys']!r} "
            f"extra_dates={report['extra_dates']!r} "
            f"duplicates={report['duplicate_exact_keys']} "
            f"actual={report['actual']} expected={report['expected']}"
        )
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    keys = _normalized_key_frame(frame, spec.key_fields)
    mask = dates.between(window.start, window.end, inclusive="both") & keys.isin(
        set(window.target_keys)
    )
    scoped = frame.loc[mask].copy()
    scoped["date"] = dates.loc[mask]
    scoped = scoped.sort_values([*spec.key_fields, "date"], kind="mergesort").reset_index(drop=True)
    scoped.attrs = frame.attrs.copy()
    scoped.attrs.update(
        {
            "formal_target_scope": {
                "dataset_id": window.dataset_id,
                "start": window.start.strftime("%Y-%m-%d"),
                "end": window.end.strftime("%Y-%m-%d"),
                "expected_days": window.expected_days,
                "expected_rows": window.expected_rows,
                "actual_rows": len(scoped),
            },
            "target_window_expected_days": int(window.expected_days),
            "target_window_range_days": int((window.end - window.start).days + 1),
            "target_window_unique_days": int(scoped["date"].nunique()),
        }
    )
    return scoped


__all__ = [
    "FormalTargetWindow",
    "evaluate_formal_target_calendar",
    "resolve_formal_target_window",
    "scope_target_to_formal_window",
]
