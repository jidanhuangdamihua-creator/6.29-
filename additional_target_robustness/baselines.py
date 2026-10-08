"""Evaluation-index adapters for Seasonal Naive, PatchTST, and TimesNet.

Model definitions stay in the existing baseline modules. These functions only
bind those models to the frozen 180-date index.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from additional_target_robustness.paths import ensure_baseline_path
from additional_target_robustness.protocol import SEEDS, EvaluationIndex, ProtocolError


def predict_seasonal(
    dataset: str,
    identity: str,
    horizon: int,
    index: EvaluationIndex,
    sales: dict[str, float],
) -> np.ndarray:
    ensure_baseline_path()
    from bl5_seasonal_naive import predict_bl5

    predictions = np.empty(len(index.rows), dtype=np.float64)
    for position, row in enumerate(index.rows):
        record = SimpleNamespace(
            dataset_id=dataset,
            target_key=tuple(identity.split("/")),
            horizon=int(horizon),
            forecast_origin=row.forecast_origin,
            label_date=row.label_date,
            input_dates=row.input_dates,
            input_sales=tuple(_sales(sales, day) for day in row.input_dates),
        )
        predictions[position] = float(predict_bl5(record))
    if predictions.shape != (len(index.rows),):
        raise ProtocolError("seasonal naive did not score every evaluation date")
    return predictions


def materialize_seasonal_seeds(predictions: np.ndarray) -> tuple[dict[str, object], ...]:
    """Copy one seasonal forecast onto seeds 42–46. The values stay identical."""
    base = np.asarray(predictions, dtype=np.float64)
    records = []
    for seed in SEEDS:
        copied = np.array(base, copy=True)
        records.append(
            {
                "seed": int(seed),
                "method": "SeasonalNaive",
                "deterministic_reuse": True,
                "y_pred": copied,
            }
        )
    first = records[0]["y_pred"]
    if any(not np.array_equal(first, record["y_pred"]) for record in records):
        raise ProtocolError("deterministic seasonal reuse produced different predictions")
    return tuple(records)


def fit_and_predict_patchtst(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl9_patchtst import fit_bl9

    fitted = fit_bl9(
        np.asarray(train_sales, dtype=np.float64),
        np.asarray(validation_sales, dtype=np.float64),
        horizon=int(horizon),
        seed=int(seed),
    )
    return _predict_indexed(fitted.predict, index, sales)


def fit_and_predict_timesnet(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl10_timesnet import fit_bl10

    fitted = fit_bl10(
        np.asarray(train_sales, dtype=np.float64),
        np.asarray(validation_sales, dtype=np.float64),
        horizon=int(horizon),
        seed=int(seed),
    )
    return _predict_indexed(fitted.predict, index, sales)


def fit_and_predict_lstm(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl4_lstm import fit_bl4

    fitted = fit_bl4(
        np.asarray(train_sales, dtype=np.float64),
        np.asarray(validation_sales, dtype=np.float64),
        horizon=int(horizon),
        seed=int(seed),
    )
    return _predict_indexed(fitted.predict, index, sales)


def fit_and_predict_dlinear(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl6_dlinear import fit_bl6

    fitted = fit_bl6(
        np.asarray(train_sales, dtype=np.float64),
        np.asarray(validation_sales, dtype=np.float64),
        horizon=int(horizon),
        seed=int(seed),
    )
    return _predict_indexed(fitted.predict, index, sales)


def fit_and_predict_tcn(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl7_tcn import fit_bl7

    fitted = fit_bl7(
        np.asarray(train_sales, dtype=np.float64),
        np.asarray(validation_sales, dtype=np.float64),
        horizon=int(horizon),
        seed=int(seed),
    )
    return _predict_indexed(fitted.predict, index, sales)


def fit_and_predict_lightgbm(
    train_sales: np.ndarray,
    train_dates: np.ndarray,
    index: EvaluationIndex,
    sales: dict[str, float],
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    ensure_baseline_path()
    from bl3_lightgbm import predict_bl3

    history = np.asarray(train_sales, dtype=np.float64)
    history_dates = pd.DatetimeIndex(pd.to_datetime(train_dates, errors="raise")).normalize()
    resolved_horizon = int(horizon)
    feature_rows = []
    for origin_index in range(6, len(history) - resolved_horizon):
        label_index = origin_index + resolved_horizon
        label_date = history_dates[label_index]
        row = {f"lag_{lag}": float(history[origin_index - lag + 1]) for lag in range(1, 8)}
        row.update(
            {
                "day_of_week": int(label_date.dayofweek),
                "day_of_month": int(label_date.day),
                "month": int(label_date.month),
                "year": int(label_date.year),
                "sales": float(history[label_index]),
            }
        )
        feature_rows.append(row)
    if not feature_rows:
        raise ProtocolError(f"LightGBM cannot form direct-h samples for horizon={resolved_horizon}")
    test_rows = []
    for row in index.rows:
        visible = np.asarray([_sales(sales, day) for day in row.input_dates], dtype=np.float64)
        if visible.size != 10:
            raise ProtocolError("LightGBM evaluation input is not a 10-day window")
        label_date = pd.Timestamp(row.label_date).normalize()
        test_row = {f"lag_{lag}": float(visible[-lag]) for lag in range(1, 8)}
        test_row.update(
            {
                "day_of_week": int(label_date.dayofweek),
                "day_of_month": int(label_date.day),
                "month": int(label_date.month),
                "year": int(label_date.year),
            }
        )
        test_rows.append(test_row)
    predictions = np.asarray(
        predict_bl3(pd.DataFrame(feature_rows), pd.DataFrame(test_rows), random_state=int(seed)),
        dtype=np.float64,
    )
    if predictions.shape != (len(index.rows),):
        raise ProtocolError("LightGBM did not score every evaluation date")
    return predictions


def partition_sales(frame: pd.DataFrame, start: str, end: str) -> np.ndarray:
    ordered = _partition_frame(frame, start, end)
    values = pd.to_numeric(ordered["sales"], errors="raise").to_numpy(dtype=np.float64)
    if values.size != 15 or not np.isfinite(values).all():
        raise ProtocolError(f"baseline fit slice {start}..{end} is not 15 finite sales")
    return values


def partition_dates(frame: pd.DataFrame, start: str, end: str) -> np.ndarray:
    ordered = _partition_frame(frame, start, end)
    dates = pd.to_datetime(ordered["date"]).dt.normalize().to_numpy()
    if dates.size != 15:
        raise ProtocolError(f"baseline fit slice {start}..{end} is not 15 dates")
    return dates


def _partition_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced = frame.loc[(pd.to_datetime(frame["date"]) >= start_ts) & (pd.to_datetime(frame["date"]) <= end_ts)]
    return sliced.sort_values("date", kind="mergesort")


def _predict_indexed(predict, index: EvaluationIndex, sales: dict[str, float]) -> np.ndarray:
    predictions = np.empty(len(index.rows), dtype=np.float64)
    for position, row in enumerate(index.rows):
        visible = np.asarray([_sales(sales, day) for day in row.input_dates], dtype=np.float64)
        if visible.size != 10 or np.any(pd.to_datetime(list(row.input_dates)) > row.forecast_origin):
            raise ProtocolError("evaluation input leaves the forecast origin")
        predictions[position] = float(predict(visible))
    return predictions


def _sales(sales: dict[str, float], day: pd.Timestamp) -> float:
    key = pd.Timestamp(day).strftime("%Y-%m-%d")
    if key not in sales:
        raise ProtocolError(f"sales history has no {key}")
    value = float(sales[key])
    if not np.isfinite(value):
        raise ProtocolError(f"sales history is non-finite on {key}")
    return value
