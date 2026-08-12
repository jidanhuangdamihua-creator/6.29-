"""Exact source-slice and tensor provenance for KNN-selected CNN inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .candidate_pool import CanonicalSourceIndex, SelectionResult
from .experiment_protocol import ProtocolViolation, SourceKey, normalize_source_key
from src.utils.dataframe_attrs import (
    context_with,
    get_protocol_frame_context,
    select_rows_with_lightweight_attrs,
    set_protocol_frame_context,
)


def bind_actual_cnn_source_frame(
    frame: pd.DataFrame,
    *,
    source_key: Sequence[object],
    group_cols: Sequence[str],
    feature_cols: Sequence[str],
) -> None:
    """Bind an exact selected key to the dataframe that will enter CNN training."""
    normalized_key = normalize_source_key(source_key)
    # The frame entering CNN training is already a selected small frame.  Scan
    # it locally so a stale parent index can never hide an extra/wrong key.
    prepared = _prepare_source(frame, group_cols)
    actual_keys = tuple(sorted(set(prepared["__protocol_source_key__"])))
    if actual_keys != (normalized_key,):
        raise ProtocolViolation(
            f"actual CNN source frame key mismatch: expected={normalized_key!r} actual={actual_keys!r}"
        )
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"actual CNN source frame missing features: {missing!r}")
    context = get_protocol_frame_context(frame)
    audit = (
        {key: dict(value) for key, value in (context.actual_cnn_audit or {}).items()}
        if context is not None
        else {}
    )
    audit[normalized_key] = {
        "bound": True,
        "actual_tensor_validated": False,
        "feature_cols": tuple(str(column) for column in feature_cols),
    }
    frame.attrs["protocol_actual_source_key"] = normalized_key
    set_protocol_frame_context(
        frame,
        context_with(
            context,
            actual_source_key=normalized_key,
            actual_cnn_audit=audit,
        ),
    )


def validate_actual_cnn_arrays_against_raw(
    frame: pd.DataFrame,
    *,
    input_tensor: np.ndarray,
    labels: np.ndarray,
    feature_cols: Sequence[str],
    window_size: int,
    horizon: int,
) -> None:
    """Rebuild the exact normalized CNN arrays from raw rows and compare elementwise."""
    context = get_protocol_frame_context(frame)
    source_key = (
        context.actual_source_key if context is not None else frame.attrs.get("protocol_actual_source_key")
    )
    if source_key is None:
        return
    raw = context.raw_partition if context is not None else None
    scaler = context.fitted_scaler if context is not None else None
    scaler_features = tuple(context.scaler_feature_cols if context is not None else ())
    features = tuple(str(column) for column in feature_cols)
    if not isinstance(raw, pd.DataFrame) or scaler is None or scaler_features != features:
        raise ProtocolViolation("actual CNN provenance is missing raw partition or fitted scaler")
    ordered_raw = raw.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)
    expected_scaled = ordered_raw.copy()
    transformed = scaler.transform(ordered_raw.loc[:, list(features)])
    for index, column in enumerate(features):
        expected_scaled[column] = transformed[:, index]

    expected_x = []
    expected_y = []
    input_dates = []
    label_dates = []
    for _, group in expected_scaled.groupby(["entity_id", "item_id"], sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        values = group.loc[:, list(features)].to_numpy(dtype=np.float32)
        sales = group["sales"].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["date"], errors="raise").dt.strftime("%Y-%m-%d")
        for end_index in range(window_size - 1, len(group) - horizon):
            start_index = end_index - window_size + 1
            label_index = end_index + horizon
            expected_x.append(values[start_index : end_index + 1])
            expected_y.append(float(sales[label_index]))
            input_dates.append(tuple(dates.iloc[start_index : end_index + 1]))
            label_dates.append(dates.iloc[label_index])
    rebuilt_x = np.asarray(expected_x, dtype=np.float32)
    rebuilt_y = np.asarray(expected_y, dtype=np.float32)
    actual_x = np.asarray(input_tensor, dtype=np.float32)
    actual_y = np.asarray(labels, dtype=np.float32)
    if not np.array_equal(actual_x, rebuilt_x) or not np.array_equal(actual_y, rebuilt_y):
        raise ProtocolViolation("actual CNN input tensor or labels differ from raw source mapping")

    audit = (
        {key: dict(value) for key, value in (context.actual_cnn_audit or {}).items()}
        if context is not None
        else {}
    )
    normalized_key = normalize_source_key(source_key)
    if not isinstance(audit, dict) or normalized_key not in audit:
        raise ProtocolViolation("actual CNN provenance audit binding is missing")
    audit[normalized_key].update(
        {
            "actual_tensor_validated": True,
            "window_size": int(window_size),
            "horizon": int(horizon),
            "sample_count": int(len(actual_y)),
            "input_dates": tuple(input_dates),
            "label_dates": tuple(label_dates),
        }
    )
    set_protocol_frame_context(
        frame,
        context_with(context, actual_cnn_audit=audit),
    )


def assert_actual_cnn_training_validated(
    frame: pd.DataFrame,
    *,
    source_key: Sequence[object],
) -> None:
    """Fail unless the arrays actually sent to CNN training passed provenance."""
    normalized_key = normalize_source_key(source_key)
    context = get_protocol_frame_context(frame)
    audit = context.actual_cnn_audit if context is not None else None
    entry = audit.get(normalized_key, {}) if isinstance(audit, Mapping) else {}
    if not entry.get("actual_tensor_validated") or int(entry.get("sample_count", 0)) <= 0:
        raise ProtocolViolation(
            f"actual CNN training provenance was not validated for {normalized_key!r}"
        )


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
    if not source_df.columns.is_unique:
        duplicates = list(
            dict.fromkeys(
                source_df.columns[source_df.columns.duplicated(keep=False)].tolist()
            )
        )
        raise ProtocolViolation(
            f"source provenance dataframe contains duplicate columns: {duplicates!r}"
        )
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


def _selected_source_rows(
    source_df: pd.DataFrame,
    source_key: Sequence[object],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Use the shared index for lookup, then validate only selected rows."""

    normalized_key = normalize_source_key(source_key)
    missing = [column for column in (*group_cols, "date") if column not in source_df.columns]
    if missing:
        raise ProtocolViolation(f"source provenance dataframe missing columns: {missing}")
    if not source_df.columns.is_unique:
        return _prepare_source(source_df, group_cols).loc[
            lambda frame: frame["__protocol_source_key__"] == normalized_key
        ].copy()
    context = get_protocol_frame_context(source_df)
    source_index = context.source_index if context is not None else None
    mask = None
    if (
        isinstance(source_index, CanonicalSourceIndex)
        and source_index.group_cols == tuple(str(column) for column in group_cols)
    ):
        mask = source_index.mask_for_normalized_key(source_df, normalized_key)
    if mask is None:
        prepared = _prepare_source(source_df, group_cols)
        return prepared[prepared["__protocol_source_key__"] == normalized_key].copy()
    selected = select_rows_with_lightweight_attrs(source_df, mask)
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    if selected["date"].isna().any():
        raise ProtocolViolation("source provenance dataframe contains invalid dates")
    selected_keys = [
        normalize_source_key(tuple(row))
        for row in selected.loc[:, list(group_cols)].itertuples(index=False, name=None)
    ]
    if any(key != normalized_key for key in selected_keys):
        raise ProtocolViolation(
            f"canonical source index returned incorrect rows for {normalized_key!r}"
        )
    selected["__protocol_source_key__"] = selected_keys
    return selected


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
    knn_feature_cols = tuple(selection.feature_cols)
    missing_knn_features = [
        column for column in knn_feature_cols if column not in source_df.columns
    ]
    if missing_knn_features:
        raise ProtocolViolation(
            f"source dataframe missing KNN features: {missing_knn_features}"
        )

    start = pd.Timestamp(training_start).normalize()
    cutoff = pd.Timestamp(selection.source_observation_cutoff).normalize()
    end = cutoff if training_end is None else pd.Timestamp(training_end).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ProtocolViolation("invalid CNN source training date range")
    if end > cutoff:
        raise ProtocolViolation(
            "CNN source training end exceeds source_observation_cutoff"
        )

    expected_dates = pd.date_range(start, end, freq="D")
    extracted = []
    for entry in selection.entries:
        candidate = _selected_source_rows(
            source_df,
            entry.source_key,
            selection.group_cols,
        )
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
        raw_vector = np.concatenate(
            [
                pd.to_numeric(knn_rows[column], errors="coerce").to_numpy(dtype=np.float64)
                for column in knn_feature_cols
            ]
        )
        if not np.isfinite(raw_vector).all():
            raise ProtocolViolation(
                f"selected source {entry.source_key!r} KNN features contain non-finite values"
            )
        if not np.array_equal(raw_vector, np.asarray(entry.raw_vector, dtype=np.float64)):
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

    normalized_key = normalize_source_key(provenance.source_key)
    source = _selected_source_rows(source_df, normalized_key, group_cols)
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
