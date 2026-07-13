import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metrics import compute_metrics_with_protocol


def test_metrics_ignore_inverse_transform_and_use_normalized_values():
    scaler = MinMaxScaler()
    scaler.fit(np.array([[10.0, 2024.0], [110.0, 2025.0]]))

    result = compute_metrics_with_protocol(
        y_true=np.array([0.2, 0.8]),
        y_pred=np.array([0.1, 0.6]),
        metric_protocol={
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "normalized_minmax_space",
            "current_accuracy_definition": "1/(RMSE+1e-8)",
            "paper_accuracy_definition": "1/(RMSE+1e-8)",
            # This fixture intentionally checks normalized-only, non-strict behavior.
            "strict_paper_metrics": False,
        },
        sales_scaler=scaler,
        feature_columns=["sales", "year"],
    )

    expected_rmse = float(np.sqrt(np.mean((np.array([0.1, 0.2])) ** 2)))
    assert np.isclose(result["rmse"], expected_rmse)
    assert np.isclose(result["rmse_current"], expected_rmse)
    assert np.isclose(result["rmse_paper"], expected_rmse)
    assert result["metric_space"] == "normalized_minmax_space"
    assert result["metric_space_used"] == "normalized_minmax_space"
    assert result["rmse_metric_space"] == "normalized_minmax_space"
    assert result["smape_metric_space"] == "normalized_minmax_space"
    assert result["paper_metric_aligned"] is False
    assert result["inverse_transform_applied"] is False
    assert result["inverse_transform_available"] is True


def test_non_strict_missing_inverse_keeps_normalized_diagnostic_but_not_paper_aliases():
    result = compute_metrics_with_protocol(
        y_true=np.array([0.2, 0.8]),
        y_pred=np.array([0.1, 0.6]),
        metric_protocol={
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": False,
        },
        sales_scaler=None,
        feature_columns=None,
    )

    assert np.isfinite(result["smape"])
    assert result["smape_metric_space"] == "normalized_minmax_space"
    assert result["paper_metric_space_actual"] == "unavailable"
    assert result["inverse_transform_status"] == "unavailable"
    assert result["paper_metric_computed_valid"] is False
    assert result["paper_metric_status"] == "missing_scaler"
    assert result["smape_paper"] is None
    assert result["original_scale_smape"] is None
