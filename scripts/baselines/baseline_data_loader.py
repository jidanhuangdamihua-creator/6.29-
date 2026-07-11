"""Load target-only baseline windows from the solidified parquet files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.rolling_origin import build_sample_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_MODEL_WINDOW = 10

_DATASET_WINDOWS = {
    "d1": {"train_start": "2017-06-05", "train_end": "2017-06-19",
           "val_start": "2017-06-20", "val_end": "2017-07-04",
           "test_start": "2017-07-05", "test_end": "2017-12-31"},
    "d2": {"train_start": "2018-06-05", "train_end": "2018-06-19",
           "val_start": "2018-06-20", "val_end": "2018-07-04",
           "test_start": "2018-07-05", "test_end": "2018-12-31"},
    "d3": {"train_start": "2015-01-03", "train_end": "2015-01-17",
           "val_start": "2015-01-18", "val_end": "2015-02-01",
           "test_start": "2015-02-02", "test_end": "2015-07-31"},
    "d4": {"train_start": "2024-12-16", "train_end": "2024-12-30",
           "val_start": "2024-12-31", "val_end": "2025-01-14",
           "test_start": "2025-01-15", "test_end": "2025-07-13"},
    "d5": {"train_start": "2017-01-17", "train_end": "2017-01-31",
           "val_start": "2017-02-01", "val_end": "2017-02-15",
           "test_start": "2017-02-16", "test_end": "2017-08-15"},
    "d6": {"train_start": "2015-10-26", "train_end": "2015-11-09",
           "val_start": "2015-11-10", "val_end": "2015-11-24",
           "test_start": "2015-11-25", "test_end": "2016-05-22"},
}


TARGET_ENTITIES = {
    "d1": [("1", "10")],
    "d2": [("1", "10")],
    "d3": [("10",)],
    "d4": [("166", "258"), ("166", "432"), ("166", "433"), ("166", "313"), ("166", "311")],
    "d5": [("48", "364606"), ("48", "1159415"), ("48", "1159414"), ("48", "1349808"), ("48", "320682")],
    "d6": [
        ("CA_1", "FOODS_3_586"),
        ("CA_1", "FOODS_3_080"),
        ("CA_1", "FOODS_3_555"),
        ("CA_1", "FOODS_3_377"),
        ("CA_1", "FOODS_3_668"),
    ],
}

ENTITY_COLUMNS = {
    "d1": ("store_id", "item_id"),
    "d2": ("brand_id", "item_id"),
    "d3": ("store_id",),
    "d4": ("store_id", "item_id"),
    "d5": ("store_nbr", "item_id"),
    "d6": ("store_id", "item_id"),
}


def _filter_entity(
    frame: pd.DataFrame,
    dataset_id: str,
    entity_values: tuple[str, ...],
) -> pd.DataFrame:
    columns = ENTITY_COLUMNS[dataset_id]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AssertionError(f"{dataset_id} target parquet is missing entity columns: {missing}")

    mask = pd.Series(True, index=frame.index)
    for column, expected in zip(columns, entity_values):
        mask &= frame[column].astype(str).eq(str(expected))
    entity = frame.loc[mask].copy()
    if entity.empty:
        key = "_".join(entity_values)
        raise AssertionError(f"{dataset_id} target entity {key!r} was not found")
    return entity


def _prepare_configured_window(
    entity: pd.DataFrame,
    dataset_id: str,
    windows: dict,
) -> pd.DataFrame:
    parsed = {key: pd.Timestamp(value) for key, value in windows.items()}
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["val_start"]
        <= parsed["val_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise AssertionError(
            f"{dataset_id} has invalid configured window order: {windows}"
        )

    window = entity[
        entity["date"].between(parsed["train_start"], parsed["test_end"])
    ].copy()
    if window["date"].duplicated().any():
        raise AssertionError(f"{dataset_id} target window contains duplicate dates")
    window = window.sort_values("date").reset_index(drop=True)

    if int(dataset_id[1:]) <= 3:
        if len(window) != 210 or window["date"].nunique() != 210:
            raise AssertionError(
                f"{dataset_id} configured window must contain 210 actual dates, "
                f"got rows={len(window)} unique_dates={window['date'].nunique()}"
            )
        return window

    calendar = pd.date_range(
        parsed["train_start"],
        parsed["test_end"],
        freq="D",
    )
    original_columns = list(window.columns)
    original_dtypes = window.dtypes.to_dict()
    window = window.sort_values("date").set_index("date").reindex(calendar)
    window["date"] = calendar
    window["sales"] = pd.to_numeric(window["sales"], errors="coerce").fillna(0.0)

    for column in original_columns:
        if column in {"date", "sales"}:
            continue
        dtype = original_dtypes[column]
        if pd.api.types.is_bool_dtype(dtype):
            window[column] = window[column].fillna(False).astype(dtype)
        elif pd.api.types.is_numeric_dtype(dtype):
            window[column] = window[column].fillna(0)
        else:
            window[column] = window[column].ffill().bfill()

    return window.reset_index(drop=True).loc[:, original_columns]


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
    if window["date"].duplicated().any():
        raise AssertionError(f"{dataset_id}/{entity_key} contains duplicate target dates")
    if len(window) < 31:
        raise AssertionError(
            f"{dataset_id}/{entity_key} needs 30 observed days plus test data, got {len(window)}"
        )

    observed_df = window.iloc[:30].copy()
    test_df = window.iloc[30:].copy()
    if len(observed_df) != 30:
        raise AssertionError(f"{dataset_id}/{entity_key} observed window must contain 30 rows")

    observed_sales = _validate_sales(
        observed_df["sales"],
        where=f"{dataset_id}/{entity_key} observed",
    )
    test_sales = _validate_sales(
        test_df["sales"],
        where=f"{dataset_id}/{entity_key} test",
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
    scenario = "without"
    target_key = tuple(entity_values) if entity_values is not None else tuple(entity_key.split("_"))
    manifest = build_sample_manifest(
        window,
        dataset_id=protocol.dataset_id,
        track=protocol.track,
        scenario=scenario,
        target_key=target_key,
        observed_end=observed_df["date"].max(),
        first_forecast_origin=observed_df["date"].max()
        + pd.Timedelta(days=PROTOCOL_MODEL_WINDOW),
        input_window=PROTOCOL_MODEL_WINDOW,
    )

    return {
        "entity_key": entity_key,
        "observed_sales": observed_sales,
        "test_sales": test_sales,
        "test_len": int(test_sales.size),
        "feature_df": feature_df,
        "test_feature_df": test_feature_df,
        "train_sales": observed_sales[:25].copy(),
        "val_sales": observed_sales[25:].copy(),
        "dataset_id": dataset_id,
        "target_window": window.copy(),
        "sample_manifest": manifest,
        "sample_manifest_digest": manifest.digest,
        "protocol_version": protocol.protocol_version,
        "protocol_track": protocol.track,
        "primary_metric_space": protocol.primary_metric_space,
        "knn_observed_start": observed_df["date"].min().strftime("%Y-%m-%d"),
        "knn_observed_end": observed_df["date"].max().strftime("%Y-%m-%d"),
    }


def load_baseline_data(dataset_id: str) -> list[dict]:
    """Return one target-only baseline slice per configured entity."""
    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in TARGET_ENTITIES:
        raise ValueError(f"dataset_id must be one of {sorted(TARGET_ENTITIES)}, got {dataset_id!r}")

    dataset_number = int(normalized_id[1:])
    parquet_path = (
        PROJECT_ROOT
        / "数据集"
        / "固化数据"
        / f"dataset{dataset_number}-target.parquet"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing target parquet: {parquet_path}")

    target = pd.read_parquet(parquet_path)
    if "date" not in target.columns:
        raise AssertionError(f"{normalized_id} target parquet requires a date column")
    if "sales" not in target.columns:
        raise AssertionError(f"{normalized_id} target parquet requires a sales column")
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    if target["date"].isna().any():
        raise AssertionError(f"{normalized_id} target parquet contains invalid dates")

    slices = []
    windows = _DATASET_WINDOWS[normalized_id]
    for entity_values in TARGET_ENTITIES[normalized_id]:
        entity_key = "_".join(entity_values)
        entity = _filter_entity(target, normalized_id, entity_values)
        window = _prepare_configured_window(entity, normalized_id, windows)
        slices.append(
            _build_entity_slice(
                window,
                normalized_id,
                entity_key,
                entity_values=tuple(entity_values),
            )
        )
    return slices
