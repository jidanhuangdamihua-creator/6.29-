"""BL1: historical mean baseline."""

from __future__ import annotations

import numpy as np


def predict_bl1(observed_sales, test_len):
    """Repeat the mean of the 30-day observed sales window."""
    observed = np.asarray(observed_sales, dtype=float).reshape(-1)
    horizon = int(test_len)
    if observed.size == 0 or not np.isfinite(observed).all():
        raise ValueError("observed_sales must be a non-empty finite vector")
    if horizon <= 0:
        raise ValueError("test_len must be positive")
    return np.full(horizon, float(observed.mean()), dtype=float)
