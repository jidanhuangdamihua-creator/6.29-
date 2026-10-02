"""Build target, validation, and rolling test windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sita_ablation.data import ContractError

WINDOW_SIZE = 10
TARGET_PARTITION_DAYS = 15
TARGET_TEST_DAYS = 180


@dataclass(frozen=True)
class WindowBatch:
    x: np.ndarray
    x_sales: np.ndarray
    y_scaled: np.ndarray
    y_original: np.ndarray
    input_start: np.ndarray
    input_end: np.ndarray
    label_date: np.ndarray
    partition: str
    unscored_test_dates: tuple[str, ...] = ()


def expected_target_partition_count(horizon: int) -> int:
    """Return the frozen target train or validation sample count, ``6 - horizon``.

    A 15-day target partition and a 10-day window produce
    ``15 - 10 - horizon + 1`` samples. This is the expected count. Callers
    still have to read the count from the built ``WindowBatch``.
    """
    if type(horizon) is not int or horizon not in (1, 3, 5):
        raise ContractError(f"horizon must be 1, 3, or 5, got {horizon!r}")
    return TARGET_PARTITION_DAYS - WINDOW_SIZE - horizon + 1


def validate_target_sample_counts(
    train: WindowBatch,
    validation: WindowBatch,
    test: WindowBatch,
    horizon: int,
) -> tuple[int, int, int]:
    """Check the actual target batches. Does not invent a count from horizon alone."""
    expected = expected_target_partition_count(horizon)
    n_train = _window_sample_count(train, "train")
    n_validation = _window_sample_count(validation, "validation")
    n_test = _window_sample_count(test, "test")
    if n_train != expected or n_validation != expected:
        raise ContractError(
            f"target train and validation WindowBatch counts are {n_train} and {n_validation}, expected {expected}"
        )
    if n_test != TARGET_TEST_DAYS:
        raise ContractError(f"target test WindowBatch has {n_test} samples, expected {TARGET_TEST_DAYS}")
    return n_train, n_validation, n_test


def build_target_partition_windows(
    rows: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
    key: tuple[str, ...],
    feature_columns: Sequence[str],
    horizon: int,
    window_size: int = WINDOW_SIZE,
    partition: str = "train",
) -> WindowBatch:
    """Build one target train or validation batch and require its actual sample count."""
    batch = build_partition_windows(
        rows,
        key_fields=key_fields,
        key=key,
        feature_columns=feature_columns,
        horizon=horizon,
        window_size=window_size,
        partition=partition,
    )
    expected = expected_target_partition_count(horizon)
    actual = _window_sample_count(batch, partition)
    if actual != expected:
        raise ContractError(
            f"target {partition} WindowBatch has {actual} samples, expected {expected} for horizon {horizon}"
        )
    return batch


def build_partition_windows(
    rows: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
    key: tuple[str, ...],
    feature_columns: Sequence[str],
    horizon: int,
    window_size: int = WINDOW_SIZE,
    partition: str = "train",
) -> WindowBatch:
    """Slide inside one split. Inputs and labels both stay in `rows`."""
    series = _one_series(rows, key_fields, key)
    _require_transformed(series)
    _require_horizon(horizon, window_size)
    dates = series["date"].to_list()
    if partition not in {"train", "validation"}:
        raise ContractError(f"partition windows only cover train or validation, got {partition!r}")
    if len(dates) < window_size + horizon:
        raise ContractError(
            f"partition has {len(dates)} days, which cannot hold window {window_size} and horizon {horizon}"
        )
    features = _feature_matrix(series, feature_columns)
    sales_scaled = series["sales"].to_numpy(dtype=np.float32)
    sales_original = series["sales_original"].to_numpy(dtype=np.float64)
    inputs = []
    scaled_labels = []
    original_labels = []
    input_starts = []
    input_ends = []
    label_dates = []
    last_end = len(series) - horizon
    for end_idx in range(window_size - 1, last_end):
        start_idx = end_idx - window_size + 1
        label_idx = end_idx + horizon
        inputs.append(features[start_idx : end_idx + 1])
        scaled_labels.append(sales_scaled[label_idx])
        original_labels.append(sales_original[label_idx])
        input_starts.append(dates[start_idx])
        input_ends.append(dates[end_idx])
        label_dates.append(dates[label_idx])
    return _pack(
        inputs,
        scaled_labels,
        original_labels,
        input_starts,
        input_ends,
        label_dates,
        feature_columns,
        partition=partition,
    )


def build_test_windows(
    observed_rows: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
    feature_columns: Sequence[str],
    horizon: int,
    window_size: int = WINDOW_SIZE,
) -> WindowBatch:
    """Score every frozen test day. The first inputs come from pre-test history."""
    _require_transformed(observed_rows)
    _require_horizon(horizon, window_size)
    series = observed_rows.sort_values("date", kind="mergesort").reset_index(drop=True)
    if series["date"].duplicated().any():
        raise ContractError("test observation frame has duplicate dates")
    by_date = {pd.Timestamp(day): index for index, day in enumerate(series["date"])}
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    label_days = list(pd.date_range(start, end, freq="D"))
    if len(label_days) != 180:
        raise ContractError(f"test span {test_start}..{test_end} has {len(label_days)} days, expected 180")
    features = _feature_matrix(series, feature_columns)
    sales_scaled = series["sales"].to_numpy(dtype=np.float32)
    sales_original = series["sales_original"].to_numpy(dtype=np.float64)
    inputs = []
    scaled_labels = []
    original_labels = []
    input_starts = []
    input_ends = []
    for label_day in label_days:
        input_end = label_day - pd.Timedelta(days=horizon)
        input_start = input_end - pd.Timedelta(days=window_size - 1)
        window_days = list(pd.date_range(input_start, input_end, freq="D"))
        missing = [str(day.date()) for day in window_days if day not in by_date]
        if label_day not in by_date or missing:
            raise ContractError(
                f"test label {label_day.date()} is missing inputs {missing} or the label row"
            )
        positions = [by_date[day] for day in window_days]
        if any(day >= label_day for day in window_days):
            raise ContractError(f"test input for {label_day.date()} is not strictly before the label")
        inputs.append(features[positions])
        label_pos = by_date[label_day]
        scaled_labels.append(sales_scaled[label_pos])
        original_labels.append(sales_original[label_pos])
        input_starts.append(input_start)
        input_ends.append(input_end)
    batch = _pack(
        inputs,
        scaled_labels,
        original_labels,
        input_starts,
        input_ends,
        label_days,
        feature_columns,
        partition="test",
    )
    if batch.unscored_test_dates:
        raise ContractError(f"test windows dropped dates {batch.unscored_test_dates}")
    if _window_sample_count(batch, "test") != TARGET_TEST_DAYS:
        raise ContractError("target test WindowBatch does not contain 180 samples")
    return batch


def _pack(
    inputs: list[np.ndarray],
    scaled_labels: list[float],
    original_labels: list[float],
    input_starts: list[pd.Timestamp],
    input_ends: list[pd.Timestamp],
    label_dates: list[pd.Timestamp],
    feature_columns: Sequence[str],
    *,
    partition: str,
) -> WindowBatch:
    if not inputs:
        raise ContractError(f"{partition} window builder produced no samples")
    if feature_columns[0] != "sales":
        raise ContractError(f"sales must be channel 0, found {feature_columns[0]!r}")
    x = np.stack(inputs).astype(np.float32, copy=False)
    y_scaled = np.asarray(scaled_labels, dtype=np.float32)
    y_original = np.asarray(original_labels, dtype=np.float64)
    return WindowBatch(
        x=x,
        x_sales=x[:, :, 0:1].astype(np.float32, copy=False),
        y_scaled=y_scaled,
        y_original=y_original,
        input_start=_dates(input_starts),
        input_end=_dates(input_ends),
        label_date=_dates(label_dates),
        partition=partition,
        unscored_test_dates=(),
    )


def _window_sample_count(batch: WindowBatch, partition: str) -> int:
    if batch.partition != partition:
        raise ContractError(f"expected a {partition} WindowBatch, got {batch.partition!r}")
    counts = (
        int(np.asarray(batch.x).shape[0]),
        int(np.asarray(batch.x_sales).shape[0]),
        int(np.asarray(batch.y_scaled).shape[0]),
        int(np.asarray(batch.y_original).shape[0]),
        int(np.asarray(batch.input_start).shape[0]),
        int(np.asarray(batch.input_end).shape[0]),
        int(len(batch.label_date)),
    )
    if len(set(counts)) != 1:
        raise ContractError(f"{partition} WindowBatch axes do not share one sample count")
    return counts[0]


def _one_series(rows: pd.DataFrame, key_fields: tuple[str, ...], key: tuple[str, ...]) -> pd.DataFrame:
    if len(key_fields) != len(key):
        raise ContractError(f"key {key!r} does not match {key_fields!r}")
    mask = pd.Series(True, index=rows.index)
    for field, value in zip(key_fields, key):
        mask &= rows[field].map(lambda item: str(item).strip()) == str(value)
    series = rows.loc[mask].sort_values("date", kind="mergesort").reset_index(drop=True)
    if series.empty:
        raise ContractError(f"window rows do not contain key {key!r}")
    if series["date"].duplicated().any():
        raise ContractError(f"window rows for {key!r} contain duplicate dates")
    return series


def _feature_matrix(series: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in series.columns]
    if missing:
        raise ContractError(f"window frame is missing features {missing}")
    values = series.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError("window features contain a non-finite value")
    return values.astype(np.float32, copy=False)


def _require_transformed(frame: pd.DataFrame) -> None:
    if "sales_original" not in frame.columns or "sales" not in frame.columns:
        raise ContractError("window frame must already be transformed and keep sales_original")


def _require_horizon(horizon: int, window_size: int) -> None:
    if horizon not in (1, 3, 5):
        raise ContractError(f"horizon must be 1, 3, or 5, got {horizon}")
    if window_size != WINDOW_SIZE:
        raise ContractError(f"window size is frozen at {WINDOW_SIZE}, got {window_size}")


def _dates(values: Sequence[pd.Timestamp]) -> np.ndarray:
    return np.asarray([pd.Timestamp(value).to_datetime64().astype("datetime64[D]") for value in values])
