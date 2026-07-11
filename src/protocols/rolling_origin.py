"""Shared rolling-origin sample manifests and strict baseline aggregation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .experiment_protocol import (
    FORMAL_HORIZONS,
    FORMAL_SEEDS,
    ProtocolViolation,
    normalize_scenario,
    normalize_source_key,
)


@dataclass(frozen=True)
class SampleRecord:
    dataset_id: str
    track: str
    scenario: str
    target_key: Tuple[str, ...]
    forecast_origin: str
    input_start: str
    input_end: str
    input_dates: Tuple[str, ...]
    input_sales: Tuple[float, ...]
    horizon: int
    label_date: str
    label: float
    sample_key: str


@dataclass(frozen=True)
class SampleManifest:
    records: Tuple[SampleRecord, ...]
    digest: str
    horizons: Tuple[int, ...] = FORMAL_HORIZONS

    def for_horizon(self, horizon: int) -> Tuple[SampleRecord, ...]:
        return tuple(record for record in self.records if record.horizon == int(horizon))

    @property
    def sample_keys(self) -> Tuple[str, ...]:
        return tuple(record.sample_key for record in self.records)


def _float_text(value: float) -> str:
    converted = np.float64(value)
    if not np.isfinite(converted):
        raise ProtocolViolation("sample manifest sales must be finite")
    return format(float(converted), ".17g")


def _manifest_digest(records: Sequence[SampleRecord]) -> str:
    payload = [
        {
            "dataset_id": record.dataset_id,
            "track": record.track,
            "scenario": record.scenario,
            "target_key": list(record.target_key),
            "forecast_origin": record.forecast_origin,
            "input_start": record.input_start,
            "input_end": record.input_end,
            "input_dates": list(record.input_dates),
            "input_sales": [_float_text(value) for value in record.input_sales],
            "horizon": record.horizon,
            "label_date": record.label_date,
            "label": _float_text(record.label),
            "sample_key": record.sample_key,
        }
        for record in records
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_sample_manifest(
    frame: pd.DataFrame,
    *,
    dataset_id: object,
    track: str,
    scenario: object,
    target_key: Sequence[object],
    observed_end: object,
    input_window: int,
    horizons: Sequence[int] = FORMAL_HORIZONS,
    date_col: str = "date",
    sales_col: str = "sales",
) -> SampleManifest:
    """Build the sole ordered sample identity used by every evaluated method."""

    if date_col not in frame.columns or sales_col not in frame.columns:
        raise ProtocolViolation(f"sample frame requires {date_col!r} and {sales_col!r}")
    if not isinstance(input_window, int) or isinstance(input_window, bool) or input_window <= 0:
        raise ProtocolViolation("input_window must be a positive integer")
    normalized_horizons = tuple(int(value) for value in horizons)
    if normalized_horizons != FORMAL_HORIZONS:
        raise ProtocolViolation(f"formal horizons must be exactly {FORMAL_HORIZONS}")
    if track not in {"strict_paper", "extended"}:
        raise ProtocolViolation(f"unsupported protocol track: {track!r}")

    prepared = frame.loc[:, [date_col, sales_col]].copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce").dt.normalize()
    prepared[sales_col] = pd.to_numeric(prepared[sales_col], errors="coerce")
    if prepared[date_col].isna().any() or not np.isfinite(
        prepared[sales_col].to_numpy(dtype=np.float64)
    ).all():
        raise ProtocolViolation("sample frame contains invalid dates or non-finite sales")
    if prepared[date_col].duplicated().any():
        raise ProtocolViolation("sample frame contains duplicate dates")
    prepared = prepared.sort_values(date_col).reset_index(drop=True)
    expected_calendar = pd.date_range(
        prepared[date_col].iloc[0], prepared[date_col].iloc[-1], freq="D"
    )
    if not pd.DatetimeIndex(prepared[date_col]).equals(expected_calendar):
        raise ProtocolViolation("sample frame must cover a complete daily calendar")

    cutoff = pd.Timestamp(observed_end).normalize()
    matching = prepared.index[prepared[date_col] == cutoff]
    if len(matching) != 1:
        raise ProtocolViolation("observed_end must identify exactly one sample-frame date")
    first_origin_index = int(matching[0])
    if first_origin_index + 1 < input_window:
        raise ProtocolViolation("insufficient observed history for input_window")
    if first_origin_index >= len(prepared) - 1:
        raise ProtocolViolation("sample frame has no target test dates after observed_end")

    normalized_dataset = str(dataset_id).strip().upper()
    normalized_scenario = normalize_scenario(scenario)
    normalized_target = normalize_source_key(target_key)
    records = []
    for horizon in normalized_horizons:
        last_origin_index = len(prepared) - horizon - 1
        for origin_index in range(first_origin_index, last_origin_index + 1):
            input_start_index = origin_index - input_window + 1
            input_rows = prepared.iloc[input_start_index : origin_index + 1]
            label_index = origin_index + horizon
            label_row = prepared.iloc[label_index]
            input_dates = tuple(input_rows[date_col].dt.strftime("%Y-%m-%d"))
            input_sales = tuple(float(value) for value in input_rows[sales_col])
            forecast_origin = prepared.loc[origin_index, date_col].strftime("%Y-%m-%d")
            label_date = label_row[date_col].strftime("%Y-%m-%d")
            if pd.Timestamp(label_date) <= pd.Timestamp(forecast_origin):
                raise ProtocolViolation("sample label date must be after forecast origin")
            sample_key = "|".join(
                (
                    normalized_dataset,
                    track,
                    normalized_scenario,
                    "/".join(normalized_target),
                    forecast_origin,
                    f"h{horizon}",
                    label_date,
                )
            )
            records.append(
                SampleRecord(
                    dataset_id=normalized_dataset,
                    track=track,
                    scenario=normalized_scenario,
                    target_key=normalized_target,
                    forecast_origin=forecast_origin,
                    input_start=input_dates[0],
                    input_end=input_dates[-1],
                    input_dates=input_dates,
                    input_sales=input_sales,
                    horizon=horizon,
                    label_date=label_date,
                    label=float(label_row[sales_col]),
                    sample_key=sample_key,
                )
            )

    manifest_records = tuple(records)
    return SampleManifest(manifest_records, _manifest_digest(manifest_records))


def assert_same_sample_manifest(
    manifest: SampleManifest,
    actual_sample_keys: Iterable[str],
    *,
    method: str,
) -> None:
    actual = tuple(str(value) for value in actual_sample_keys)
    if actual != manifest.sample_keys:
        raise ProtocolViolation(
            f"{method} sample manifest mismatch: expected={len(manifest.sample_keys)} actual={len(actual)}"
        )


def validate_feature_availability(
    feature_cols: Sequence[str],
    *,
    allowlist: Mapping[str, str],
) -> None:
    """Reject every forecast-time feature that is not explicitly allowlisted."""

    for column in feature_cols:
        availability = str(allowlist.get(str(column), "")).strip().lower()
        if availability not in {"known_at_origin", "known_in_advance"}:
            raise ProtocolViolation(
                f"feature {column!r} is not available at forecast origin"
            )


_IDENTITY_COLUMNS = ("dataset_id", "target_entity_key", "scenario", "method")
_METRIC_COLUMNS = ("rmse", "mae", "smape", "accuracy")


def aggregate_protocol_results(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete five-seed/five-horizon original-scale results."""

    required = set(_IDENTITY_COLUMNS) | {
        "horizon",
        "seed",
        "primary_metric_space",
        "sample_manifest_digest",
        *_METRIC_COLUMNS,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ProtocolViolation(f"protocol result rows are missing columns: {missing}")
    if rows.empty:
        raise ProtocolViolation("protocol result rows may not be empty")

    output = []
    for identity, group in rows.groupby(list(_IDENTITY_COLUMNS), dropna=False, sort=False):
        if set(group["primary_metric_space"].astype(str)) != {"original_sales"}:
            raise ProtocolViolation("formal aggregation requires original_sales metric space")
        expected_pairs = {(horizon, seed) for horizon in FORMAL_HORIZONS for seed in FORMAL_SEEDS}
        actual_pairs = set(
            zip(group["horizon"].astype(int), group["seed"].astype(int))
        )
        if actual_pairs != expected_pairs or len(group) != len(expected_pairs):
            raise ProtocolViolation(
                "formal seed coverage requires exactly seeds 42-46 for horizons 1-5"
            )
        if group["sample_manifest_digest"].astype(str).nunique() != 1:
            raise ProtocolViolation("formal aggregation mixes sample manifests")
        base = dict(zip(_IDENTITY_COLUMNS, identity if isinstance(identity, tuple) else (identity,)))
        for horizon in FORMAL_HORIZONS:
            horizon_rows = group[group["horizon"].astype(int) == horizon]
            aggregate = {
                **base,
                "aggregate_scope": "horizon",
                "horizon": horizon,
                "seed_count": len(FORMAL_SEEDS),
                "sample_manifest_digest": horizon_rows["sample_manifest_digest"].iloc[0],
            }
            for metric in _METRIC_COLUMNS:
                values = horizon_rows[metric].astype(float)
                aggregate[f"{metric}_mean"] = float(values.mean())
                aggregate[f"{metric}_std"] = float(values.std(ddof=1))
            output.append(aggregate)

        per_seed = group.groupby(group["seed"].astype(int))[list(_METRIC_COLUMNS)].mean()
        overall = {
            **base,
            "aggregate_scope": "horizons_1_5",
            "horizon": 0,
            "seed_count": len(FORMAL_SEEDS),
            "sample_manifest_digest": group["sample_manifest_digest"].iloc[0],
        }
        for metric in _METRIC_COLUMNS:
            overall[f"{metric}_mean"] = float(per_seed[metric].mean())
            overall[f"{metric}_std"] = float(per_seed[metric].std(ddof=1))
        output.append(overall)
    return pd.DataFrame(output)
