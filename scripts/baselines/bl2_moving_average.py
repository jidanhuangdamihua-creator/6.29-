"""BL2: seven-day moving-average baseline."""

from __future__ import annotations

import numpy as np


def predict_bl2(observed_sales, test_len):
    """Repeat the mean of the final seven observed sales values."""
    observed = np.asarray(observed_sales, dtype=float).reshape(-1)
    horizon = int(test_len)
    if observed.size < 7 or not np.isfinite(observed).all():
        raise ValueError("observed_sales must contain at least seven finite values")
    if horizon <= 0:
        raise ValueError("test_len must be positive")
    return np.full(horizon, float(observed[-7:].mean()), dtype=float)
