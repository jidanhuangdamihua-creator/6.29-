from __future__ import annotations

import numpy as np

from src.transfer_methods.mssb_tl import evaluate_predictions
from src.transfer_methods.mswa_tl import evaluate_fused_predictions


def test_mswa_fused_evaluation_returns_smape() -> None:
    result = evaluate_fused_predictions(
        y_true=np.array([10.0, 20.0, 30.0]),
        y_pred=np.array([[12.0], [18.0], [33.0]]),
    )

    assert np.isfinite(result["smape"])
    assert result["prediction_shape"] == (3, 1)


def test_mssb_evaluation_returns_smape() -> None:
    result = evaluate_predictions(
        y_true=np.array([10.0, 20.0, 30.0]),
        y_pred=np.array([[12.0], [18.0], [33.0]]),
    )

    assert np.isfinite(result["smape"])
    assert result["prediction_shape"] == (3, 1)
