"""BL5: strict seven-day seasonal-naive baseline."""

from __future__ import annotations

import math

import pandas as pd


def _identity_text(record) -> str:
    target_key = getattr(record, "target_key", ())
    if isinstance(target_key, (tuple, list)):
        return "/".join(str(value) for value in target_key)
    return str(target_key)


def predict_bl5(record) -> float:
    """Return y[target_date - 7 days] using history known at forecast origin."""
    dataset = str(getattr(record, "dataset_id", ""))
    target_identity = _identity_text(record)
    horizon = int(record.horizon)
    forecast_origin = pd.Timestamp(record.forecast_origin).normalize()
    target_date = pd.Timestamp(record.label_date).normalize()
    expected_target_date = forecast_origin + pd.Timedelta(days=horizon)
    if target_date != expected_target_date:
        raise ValueError(
            "SEASONAL_NAIVE_TARGET_DATE_MISMATCH "
            f"dataset={dataset} target_identity={target_identity} "
            f"forecast_origin={forecast_origin.date()} target_date={target_date.date()} "
            f"expected_target_date={expected_target_date.date()} horizon={horizon}"
        )

    required_lag_date = target_date - pd.Timedelta(days=7)
    if required_lag_date > forecast_origin:
        raise ValueError(
            "SEASONAL_NAIVE_FUTURE_TARGET_ACCESS_FORBIDDEN "
            f"dataset={dataset} target_identity={target_identity} "
            f"forecast_origin={forecast_origin.date()} target_date={target_date.date()} "
            f"required_lag_date={required_lag_date.date()} horizon={horizon}"
        )

    input_dates = tuple(pd.Timestamp(value).normalize() for value in record.input_dates)
    input_sales = tuple(float(value) for value in record.input_sales)
    if len(input_dates) != len(input_sales):
        raise ValueError("SEASONAL_NAIVE_HISTORY_LENGTH_MISMATCH")
    if len(set(input_dates)) != len(input_dates):
        raise ValueError("SEASONAL_NAIVE_HISTORY_DUPLICATE_DATE")
    if any(date > forecast_origin for date in input_dates):
        raise ValueError("SEASONAL_NAIVE_HISTORY_AFTER_FORECAST_ORIGIN")

    history = dict(zip(input_dates, input_sales))
    if required_lag_date not in history:
        raise ValueError(
            "SEASONAL_NAIVE_LAG7_HISTORY_MISSING "
            f"dataset={dataset} target_identity={target_identity} "
            f"forecast_origin={forecast_origin.date()} target_date={target_date.date()} "
            f"required_lag_date={required_lag_date.date()} horizon={horizon}"
        )
    prediction = history[required_lag_date]
    if not math.isfinite(prediction):
        raise ValueError(
            "SEASONAL_NAIVE_LAG7_HISTORY_NONFINITE "
            f"dataset={dataset} target_identity={target_identity} "
            f"forecast_origin={forecast_origin.date()} target_date={target_date.date()} "
            f"required_lag_date={required_lag_date.date()} horizon={horizon}"
        )
    return prediction
