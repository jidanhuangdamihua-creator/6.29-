"""Load target-only baseline windows from the formal sealed resolver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.protocols.experiment_protocol import (
    get_experiment_protocol,
    serialize_canonical_target_key,
)
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.formal_target_scope import scope_target_to_formal_window
from src.protocols.gate1_transformation import dataset_contract
from src.protocols.rolling_origin import build_sample_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_MODEL_WINDOW = 10
FORMAL_TRAIN_DAYS = 15
FORMAL_VALIDATION_DAYS = 15
FORMAL_FORECAST_DAYS = 180


def _filter_entity(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
    entity_values: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in key_fields if column not in frame.columns]
    if missing:
        raise AssertionError(f"formal target is missing entity columns: {missing}")
    if len(key_fields) != len(entity_values):
        raise AssertionError(
            f"formal target key arity mismatch: fields={key_fields} values={entity_values}"
        )

    mask = pd.Series(True, index=frame.index)
    for column, expected in zip(key_fields, entity_values):
        mask &= frame[column].astype(str).eq(str(expected))
    entity = frame.loc[mask].copy()
    if entity.empty:
        raise AssertionError(f"formal target entity {entity_values!r} was not found")
    return entity


def _validate_sales(values: pd.Series, *, where: str) -> np.ndarray:
    sales = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if sales.size == 0 or not np.isfinite(sales).all():
        raise AssertionError(f"{where} sales must be non-empty and finite")
    return sales


def _build_entity_slice(
    window: pd.DataFrame,
    dataset_id: str,
    entity_key: str,
    entity_values: tuple[str, ...] | None = None,
) -> dict:
    """Close caller-resolved target rows into explicit train/validation/forecast roles."""

    prepared = window.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any():
        raise AssertionError(f"{dataset_id}/{entity_key} contains invalid target dates")
    if prepared["date"].duplicated().any():
        raise AssertionError(f"{dataset_id}/{entity_key} contains duplicate target dates")
    prepared = prepared.sort_values("date").reset_index(drop=True)
    if len(prepared) < FORMAL_TRAIN_DAYS + FORMAL_VALIDATION_DAYS + 1:
        raise AssertionError(
            f"{dataset_id}/{entity_key} needs 30 observed days plus forecast data, "
            f"got {len(prepared)}"
        )

    observed_df = prepared.iloc[: FORMAL_TRAIN_DAYS + FORMAL_VALIDATION_DAYS].copy()
    train_df = observed_df.iloc[:FORMAL_TRAIN_DAYS].copy()
    validation_df = observed_df.iloc[FORMAL_TRAIN_DAYS:].copy()
    test_df = prepared.iloc[FORMAL_TRAIN_DAYS + FORMAL_VALIDATION_DAYS :].copy()
    if len(train_df) != FORMAL_TRAIN_DAYS or len(validation_df) != FORMAL_VALIDATION_DAYS:
        raise AssertionError(
            f"{dataset_id}/{entity_key} formal split must be "
            f"{FORMAL_TRAIN_DAYS}+{FORMAL_VALIDATION_DAYS}"
        )

    train_sales = _validate_sales(
        train_df["sales"],
        where=f"{dataset_id}/{entity_key} train",
    )
    validation_sales = _validate_sales(
        validation_df["sales"],
        where=f"{dataset_id}/{entity_key} validation",
    )
    observed_sales = np.concatenate((train_sales, validation_sales))
    test_sales = _validate_sales(
        test_df["sales"],
        where=f"{dataset_id}/{entity_key} forecast",
    )

    feature_df = observed_df.select_dtypes(include=[np.number]).reset_index(drop=True)
    if "sales" not in feature_df.columns:
        raise AssertionError(f"{dataset_id}/{entity_key} numeric feature frame requires sales")
    test_feature_df = (
        test_df.select_dtypes(include=[np.number])
        .drop(columns=["sales"], errors="ignore")
        .reset_index(drop=True)
    )

    protocol = get_experiment_protocol(dataset_id)
    target_key = (
        tuple(entity_values)
        if entity_values is not None
        else tuple(str(entity_key).replace("/", "_").split("_"))
    )
    manifest = build_sample_manifest(
        prepared,
        dataset_id=protocol.dataset_id,
        track=protocol.track,
        scenario="without",
        target_key=target_key,
        observed_end=observed_df["date"].max(),
        first_forecast_origin=(
            observed_df["date"].max() + pd.Timedelta(days=PROTOCOL_MODEL_WINDOW)
        ),
        input_window=PROTOCOL_MODEL_WINDOW,
    )

    return {
        "entity_key": str(entity_key),
        "entity_values": tuple(target_key),
        "observed_sales": observed_sales,
        "test_sales": test_sales,
        "test_len": int(test_sales.size),
        "feature_df": feature_df,
        "test_feature_df": test_feature_df,
        "train_sales": train_sales,
        "val_sales": validation_sales,
        "train_dates": tuple(train_df["date"].dt.strftime("%Y-%m-%d")),
        "validation_dates": tuple(validation_df["date"].dt.strftime("%Y-%m-%d")),
        "lookback": PROTOCOL_MODEL_WINDOW,
        "dataset_id": dataset_id,
        "target_window": prepared.copy(),
        "sample_manifest": manifest,
        "sample_manifest_digest": manifest.digest,
        "protocol_version": protocol.protocol_version,
        "protocol_track": protocol.track,
        "primary_metric_space": protocol.primary_metric_space,
        "knn_observed_start": observed_df["date"].min().strftime("%Y-%m-%d"),
        "knn_observed_end": observed_df["date"].max().strftime("%Y-%m-%d"),
    }


def load_baseline_data(dataset_id: str) -> list[dict]:
    """Return formal target-only slices resolved from sealed D1-D6 authority."""

    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError(f"dataset_id must be d1 through d6, got {dataset_id!r}")
    dataset_number = int(normalized_id[1:])
    spec = dataset_contract(dataset_number)
    paths = resolve_formal_dataset_paths(
        dataset_number,
        repository_root=PROJECT_ROOT,
    )
    target = pd.read_parquet(paths.target_path)
    if "date" not in target.columns or "sales" not in target.columns:
        raise AssertionError(
            f"{normalized_id} formal target requires date and sales columns: {paths.target_path}"
        )
    target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    if target["date"].isna().any():
        raise AssertionError(f"{normalized_id} formal target contains invalid dates")
    scoped = scope_target_to_formal_window(target, dataset_id=dataset_number)

    slices = []
    expected_calendar = pd.date_range(spec.target_train_start, spec.blind_end, freq="D")
    for target_key in spec.target_keys:
        entity_values = tuple(str(value) for value in target_key)
        entity = _filter_entity(
            scoped,
            key_fields=tuple(spec.key_fields),
            entity_values=entity_values,
        ).sort_values("date").reset_index(drop=True)
        actual_calendar = pd.DatetimeIndex(entity["date"])
        if not actual_calendar.equals(expected_calendar):
            raise AssertionError(
                f"{normalized_id}/{entity_values} formal calendar mismatch: "
                f"expected={len(expected_calendar)} actual={len(actual_calendar)}"
            )
        if len(entity) != FORMAL_TRAIN_DAYS + FORMAL_VALIDATION_DAYS + FORMAL_FORECAST_DAYS:
            raise AssertionError(
                f"{normalized_id}/{entity_values} formal target must contain 210 rows, "
                f"got {len(entity)}"
            )
        slices.append(
            _build_entity_slice(
                entity,
                normalized_id,
                serialize_canonical_target_key(dataset_number, entity_values),
                entity_values=entity_values,
            )
        )
    return slices
