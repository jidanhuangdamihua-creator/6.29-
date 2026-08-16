"""BL5: strict seven-day seasonal-naive baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


SEASONAL_LAG_DAYS = 7


def _target_identity_text(target_identity: Sequence[object] | object) -> str:
    if isinstance(target_identity, (str, bytes)):
        return str(target_identity)
    try:
        values = tuple(target_identity)  # type: ignore[arg-type]
    except TypeError:
        return str(target_identity)
    return "/".join(str(value) for value in values)


def _context(
    *,
    dataset_id: object,
    target_identity: Sequence[object] | object,
    forecast_origin: pd.Timestamp,
    target_date: pd.Timestamp,
    required_lag_date: pd.Timestamp,
    horizon: int,
) -> str:
    return (
        f"dataset={str(dataset_id).upper()} "
        f"target_identity={_target_identity_text(target_identity)} "
        f"forecast_origin={forecast_origin.strftime('%Y-%m-%d')} "
        f"target_date={target_date.strftime('%Y-%m-%d')} "
        f"required_lag_date={required_lag_date.strftime('%Y-%m-%d')} "
        f"horizon={horizon}"
    )


def predict_bl5(
    input_dates,
    input_sales,
    *,
    forecast_origin,
    horizon,
    dataset_id,
    target_identity,
) -> float:
    """Predict ``y[target_date - 7 days]`` using only history legal at origin.

    The function deliberately performs an explicit calendar-date lookup rather than
    assuming positional daily continuity.  It never repairs a missing lag and never
    consumes a sales observation after ``forecast_origin``.
    """

    dates = pd.DatetimeIndex(pd.to_datetime(list(input_dates), errors="coerce")).normalize()
    sales = np.asarray(input_sales, dtype=float).reshape(-1)
    origin = pd.Timestamp(forecast_origin).normalize()
    try:
        horizon_value = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SEASONAL_NAIVE_INVALID_HORIZON horizon={horizon!r}") from exc

    if horizon_value <= 0:
        raise ValueError(f"SEASONAL_NAIVE_INVALID_HORIZON horizon={horizon_value}")
    if len(dates) != sales.size or sales.size == 0:
        raise ValueError(
            "SEASONAL_NAIVE_INVALID_HISTORY "
            f"date_count={len(dates)} sales_count={sales.size}"
        )
    if dates.isna().any() or not np.isfinite(sales).all():
        raise ValueError("SEASONAL_NAIVE_INVALID_HISTORY dates_and_sales_must_be_finite")
    if dates.duplicated().any():
        raise ValueError("SEASONAL_NAIVE_INVALID_HISTORY duplicate_input_dates")

    target_date = origin + pd.Timedelta(days=horizon_value)
    required_lag_date = target_date - pd.Timedelta(days=SEASONAL_LAG_DAYS)
    context = _context(
        dataset_id=dataset_id,
        target_identity=target_identity,
        forecast_origin=origin,
        target_date=target_date,
        required_lag_date=required_lag_date,
        horizon=horizon_value,
    )

    if (dates > origin).any():
        raise ValueError(f"SEASONAL_NAIVE_INPUT_AFTER_FORECAST_ORIGIN {context}")
    if required_lag_date > origin:
        raise ValueError(f"SEASONAL_NAIVE_LAG7_NOT_OBSERVED_AT_ORIGIN {context}")

    matches = np.flatnonzero(dates == required_lag_date)
    if matches.size != 1:
        raise ValueError(f"SEASONAL_NAIVE_LAG7_HISTORY_MISSING {context}")

    prediction = float(sales[int(matches[0])])
    if not np.isfinite(prediction):  # defensive; covered by the vector guard above
        raise ValueError(f"SEASONAL_NAIVE_NONFINITE_PREDICTION {context}")
    return prediction
