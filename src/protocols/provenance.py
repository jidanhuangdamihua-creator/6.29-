"""Exact source-slice and tensor provenance for KNN-selected CNN inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .candidate_pool import SelectionResult
from .experiment_protocol import ProtocolViolation, SourceKey, normalize_source_key


@dataclass(frozen=True)
class SourceSliceRef:
    source_key: SourceKey
    date_start: str
    date_end: str
    dates: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    values: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class TensorProvenance:
    source_key: SourceKey
    source_date_start: str
    source_date_end: str
    feature_cols: Tuple[str, ...]
    label_col: str
    horizon: int
    input_dates: Tuple[Tuple[str, ...], ...]
    label_dates: Tuple[str, ...]
    input_tensor: np.ndarray
    labels: np.ndarray


def _prepare_source(
    source_df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    missing = [column for column in (*group_cols, "date") if column not in source_df.columns]
    if missing:
        raise ProtocolViolation(f"source provenance dataframe missing columns: {missing}")
    prepared = source_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any():
        raise ProtocolViolation("source provenance dataframe contains invalid dates")
    prepared["__protocol_source_key__"] = prepared.loc[:, list(group_cols)].apply(
        lambda row: normalize_source_key(tuple(row.tolist())), axis=1
    )
    return prepared


def extract_selected_source_slices(
    selection: SelectionResult,
    source_df: pd.DataFrame,
    *,
    training_start: object,
    model_feature_cols: Sequence[str],
    training_end: object | None = None,
) -> Tuple[SourceSliceRef, ...]:
    """Extract only selected sources and prove their KNN vectors against raw rows."""

    if tuple(selection.group_cols) == ():
        raise ProtocolViolation("selection group columns may not be empty")
    missing_features = [column for column in model_feature_cols if column not in source_df.columns]
    if missing_features:
        raise ProtocolViolation(f"source dataframe missing model features: {missing_features}")
    if "sales" not in source_df.columns:
        raise ProtocolViolation("source dataframe requires sales for KNN provenance")

    start = pd.Timestamp(training_start).normalize()
    cutoff = pd.Timestamp(selection.source_observation_cutoff).normalize()
    end = cutoff if training_end is None else pd.Timestamp(training_end).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ProtocolViolation("invalid CNN source training date range")
    if end > cutoff:
        raise ProtocolViolation(
            "CNN source training end exceeds source_observation_cutoff"
        )

    prepared = _prepare_source(source_df, selection.group_cols)
    expected_dates = pd.date_range(start, end, freq="D")
    extracted = []
    for entry in selection.entries:
        candidate = prepared[
            prepared["__protocol_source_key__"] == entry.source_key
        ]
        if candidate.empty:
            raise ProtocolViolation(
                f"selected source key is absent from CNN extractor: {entry.source_key!r}"
            )
        sliced = candidate[candidate["date"].between(start, end, inclusive="both")].sort_values(
            "date"
        )
        if sliced["date"].duplicated().any():
            raise ProtocolViolation(
                f"selected source {entry.source_key!r} has duplicate CNN dates"
            )
        actual_dates = pd.DatetimeIndex(sliced["date"])
        if not actual_dates.equals(expected_dates):
            missing = expected_dates.difference(actual_dates).strftime("%Y-%m-%d").tolist()
            raise ProtocolViolation(
                f"selected source {entry.source_key!r} missing CNN dates: {missing}"
            )

        observed_start = pd.Timestamp(entry.observed_start).normalize()
        observed_end = pd.Timestamp(entry.observed_end).normalize()
        knn_rows = sliced[
            sliced["date"].between(observed_start, observed_end, inclusive="both")
        ]
        raw_sales = knn_rows["sales"].to_numpy(dtype=np.float64)
        if not np.array_equal(raw_sales, np.asarray(entry.raw_vector, dtype=np.float64)):
            raise ProtocolViolation(
                f"selected source {entry.source_key!r} KNN vector differs from raw slice"
            )

        values = sliced.loc[:, list(model_feature_cols)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolViolation(
                f"selected source {entry.source_key!r} CNN features contain non-finite values"
            )
        extracted.append(
            SourceSliceRef(
                source_key=entry.source_key,
                date_start=start.strftime("%Y-%m-%d"),
                date_end=end.strftime("%Y-%m-%d"),
                dates=tuple(actual_dates.strftime("%Y-%m-%d")),
                feature_cols=tuple(model_feature_cols),
                values=tuple(tuple(float(value) for value in row) for row in values),
            )
        )

    if tuple(item.source_key for item in extracted) != selection.ordered_source_keys:
        raise ProtocolViolation("CNN extractor source key order differs from KNN selection")
    return tuple(extracted)


def build_cnn_tensor_provenance(
    source_slice: SourceSliceRef,
    *,
    window_size: int,
    horizon: int,
    label_col: str,
) -> TensorProvenance:
    """Build supervised CNN arrays together with exact row/date provenance."""

    if window_size <= 0 or horizon <= 0:
        raise ProtocolViolation("window_size and horizon must be positive")
    if label_col not in source_slice.feature_cols:
        raise ProtocolViolation(f"label column {label_col!r} is not in source features")
    values = np.asarray(source_slice.values, dtype=np.float64)
    dates = tuple(source_slice.dates)
    sample_count = len(dates) - window_size - horizon + 1
    if sample_count <= 0:
        raise ProtocolViolation("source slice is too short for CNN window and horizon")
    label_index = source_slice.feature_cols.index(label_col)

    inputs = []
    labels = []
    input_dates = []
    label_dates = []
    for start in range(sample_count):
        input_end = start + window_size
        target_index = input_end + horizon - 1
        inputs.append(values[start:input_end])
        labels.append(values[target_index, label_index])
        input_dates.append(dates[start:input_end])
        label_dates.append(dates[target_index])

    return TensorProvenance(
        source_key=source_slice.source_key,
        source_date_start=source_slice.date_start,
        source_date_end=source_slice.date_end,
        feature_cols=source_slice.feature_cols,
        label_col=label_col,
        horizon=int(horizon),
        input_dates=tuple(tuple(group) for group in input_dates),
        label_dates=tuple(label_dates),
        input_tensor=np.asarray(inputs, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
    )


def validate_cnn_tensor_provenance(
    provenance: TensorProvenance,
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> None:
    """Validate every CNN tensor element and label against the original source rows."""

    prepared = _prepare_source(source_df, group_cols)
    normalized_key = normalize_source_key(provenance.source_key)
    source = prepared[prepared["__protocol_source_key__"] == normalized_key].copy()
    if source.empty:
        raise ProtocolViolation(f"CNN provenance source key is absent: {normalized_key!r}")
    if source["date"].duplicated().any():
        raise ProtocolViolation(f"CNN provenance source key has duplicate dates: {normalized_key!r}")
    missing_features = [column for column in provenance.feature_cols if column not in source.columns]
    if missing_features or provenance.label_col not in source.columns:
        raise ProtocolViolation(
            f"CNN provenance source is missing features: {missing_features!r}"
        )
    if len(provenance.input_dates) != len(provenance.label_dates):
        raise ProtocolViolation("CNN provenance input and label date counts differ")
    if provenance.input_tensor.shape[0] != len(provenance.input_dates):
        raise ProtocolViolation("CNN provenance tensor sample count differs from dates")
    if provenance.labels.shape != (len(provenance.label_dates),):
        raise ProtocolViolation("CNN provenance label shape differs from label dates")

    by_date = source.set_index("date", verify_integrity=True)
    for sample_index, (date_group, label_date) in enumerate(
        zip(provenance.input_dates, provenance.label_dates)
    ):
        parsed_dates = tuple(pd.Timestamp(value).normalize() for value in date_group)
        if tuple(sorted(parsed_dates)) != parsed_dates or len(set(parsed_dates)) != len(parsed_dates):
            raise ProtocolViolation(
                f"CNN provenance date order is invalid at sample {sample_index}"
            )
        parsed_label_date = pd.Timestamp(label_date).normalize()
        if not parsed_dates or parsed_label_date <= parsed_dates[-1]:
            raise ProtocolViolation(
                f"CNN provenance label date is not after input at sample {sample_index}"
            )
        try:
            expected_input = by_date.loc[
                list(parsed_dates), list(provenance.feature_cols)
            ].to_numpy(dtype=np.float64)
            expected_label = float(by_date.loc[parsed_label_date, provenance.label_col])
        except KeyError as exc:
            raise ProtocolViolation(
                f"CNN provenance references an absent raw date at sample {sample_index}"
            ) from exc
        actual_input = np.asarray(provenance.input_tensor[sample_index], dtype=np.float64)
        if not np.array_equal(actual_input, expected_input):
            raise ProtocolViolation(
                f"CNN provenance input tensor differs from raw rows at sample {sample_index}"
            )
        if float(provenance.labels[sample_index]) != expected_label:
            raise ProtocolViolation(
                f"CNN provenance label differs from raw row at sample {sample_index}"
            )
