"""Frozen original-sales RMSE and sMAPE.

This is the same rule as ``sita_ablation.metrics.evaluate_original_sales``:
clip negatives to 0, then

RMSE = sqrt(mean((pred - true)^2))
sMAPE = 100 * mean(2 * |pred - true| / (|true| + |pred| + 1e-8))
"""

from __future__ import annotations

import numpy as np


def original_sales_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return RMSE and sMAPE on one paired original-sales sample."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape or true.size == 0:
        raise ValueError("original-sales scores require equal non-empty 1-d vectors")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("original-sales scores require finite values")
    true = np.maximum(true, 0.0)
    pred = np.maximum(pred, 0.0)
    diff = pred - true
    rmse = float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))
    denom = np.abs(true) + np.abs(pred) + np.float64(1e-8)
    smape = float(np.float64(100.0) * np.mean(np.float64(2.0) * np.abs(diff) / denom, dtype=np.float64))
    return rmse, smape
