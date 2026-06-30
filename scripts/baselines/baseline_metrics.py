"""Metrics shared by the target-only baseline methods."""

from __future__ import annotations

import numpy as np


def _as_finite_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_metrics(y_true, y_pred) -> dict:
    """Compute sMAPE, RMSE, and MAE in the original sales space."""
    true = _as_finite_vector(y_true, name="y_true")
    pred = _as_finite_vector(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )

    error = pred - true
    smape = float(
        np.mean(
            2.0 * np.abs(error)
            / (np.abs(true) + np.abs(pred) + 1e-8)
        )
        * 100.0
    )
    metrics = {
        "smape": smape,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in metrics.values()):
        raise ValueError(f"computed metrics must be finite and non-negative: {metrics}")
    return metrics
