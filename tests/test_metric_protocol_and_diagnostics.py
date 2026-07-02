from __future__ import annotations

import numpy as np

from src.experiment.experiment_runner import _extract_method_metrics
from src.utils.entity_experiment import _row_from_result


class DummyMinMaxScaler:
    data_min_ = np.array([10.0])
    data_max_ = np.array([20.0])


def test_extract_method_metrics_recomputes_multi_source_strict_paper_metrics() -> None:
    raw = {
        "fused_result": {
            "rmse": 0.25,
            "accuracy": 4.0,
            "smape": 10.0,
            "y_true": np.array([0.0, 0.0]),
            "y_pred": np.array([[0.25], [0.25]]),
            "prediction_shape": (2, 1),
            "sales_scaler": DummyMinMaxScaler(),
            "feature_columns": ["sales"],
        },
        "meta": {"selected_sources": [{"source_key": ("s1", "i1")}]},
    }

    result = _extract_method_metrics(
        raw,
        method_name="MSWA-TL",
        metric_protocol={
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": True,
        },
    )

    assert result["metric_space_used"] == "original_sales_space"
    assert result["paper_metric_aligned"] is True
    assert result["inverse_transform_applied"] is True
    assert result["normalized_rmse"] == 0.25
    assert result["rmse"] == 2.5


def test_validate_finite_array_reports_nan_prediction_counts() -> None:
    from src.utils.finite_diagnostics import NonFiniteArrayError, validate_finite_array

    try:
        validate_finite_array(np.array([[np.nan], [np.inf]]), name="y_pred")
    except NonFiniteArrayError as exc:
        assert exc.diagnostics["y_pred_nan_count"] == 1
        assert exc.diagnostics["y_pred_inf_count"] == 1
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("validate_finite_array should reject NaN/Inf predictions")


def test_entity_row_marks_present_shape_missing_metrics_as_diagnostic_failure() -> None:
    row = _row_from_result(
        {
            "rmse": np.nan,
            "accuracy": np.nan,
            "smape": np.nan,
            "prediction_shape": (170, 1),
            "y_pred_nan_count": 170,
        },
        method="SS-TL",
        entity_key="48_1159415",
        config={"dataset_id": 5, "dataset_name": "Dataset5", "info_sharing": "without"},
        elapsed=1.0,
    )

    assert row["error"]
    assert "silent_metric_failure" in row["error"]
    assert row["y_pred_nan_count"] == 170
