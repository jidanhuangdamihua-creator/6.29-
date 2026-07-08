from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_with_protocol
from src.experiment.experiment_runner import _extract_method_metrics
from src.utils import entity_experiment
from src.utils.entity_experiment import _row_from_result
from src.utils.result_validation import annotate_silent_metric_failure


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
    assert result["rmse_metric_space"] == "original_sales_space"
    assert result["smape_metric_space"] == "original_sales_space"
    assert result["paper_metric_aligned"] is True
    assert result["inverse_transform_applied"] is True
    assert result["normalized_rmse"] == 0.25
    assert result["rmse"] == 2.5


def test_non_strict_inverse_available_reports_mixed_metric_spaces() -> None:
    result = compute_metrics_with_protocol(
        y_true=np.array([0.0, 0.5]),
        y_pred=np.array([0.25, 0.25]),
        metric_protocol={
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": False,
        },
        sales_scaler=DummyMinMaxScaler(),
        feature_columns=["sales"],
    )

    assert result["metric_space_used"] == "mixed_metric_space"
    assert result["rmse_metric_space"] == "normalized_minmax_space"
    assert result["smape_metric_space"] == "original_sales_space"
    assert result["paper_metric_aligned"] is False
    assert result["inverse_transform_applied"] is True
    assert (
        result["metric_protocol_note"]
        == "non-strict protocol uses normalized RMSE and original-scale sMAPE when inverse transform is available"
    )


def test_entity_row_reports_mixed_note_without_overwriting_existing_note() -> None:
    config = {
        "dataset_id": 5,
        "dataset_name": "Dataset5",
        "info_sharing": "without",
        "metric_protocol": {
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": False,
        },
    }
    mixed_raw = {
        "rmse": 0.25,
        "accuracy": 4.0,
        "smape": 10.0,
        "metric_space_used": "mixed_metric_space",
        "rmse_metric_space": "normalized_minmax_space",
        "smape_metric_space": "original_sales_space",
        "paper_metric_aligned": False,
        "inverse_transform_applied": True,
    }
    mixed_row = _row_from_result(
        mixed_raw,
        method="SS-TL",
        entity_key="target",
        config=config,
        elapsed=1.0,
    )

    assert mixed_row["rmse_metric_space"] == "normalized_minmax_space"
    assert mixed_row["smape_metric_space"] == "original_sales_space"
    assert (
        mixed_row["metric_protocol_note"]
        == "non-strict protocol uses normalized RMSE and original-scale sMAPE when inverse transform is available"
    )

    noted_row = _row_from_result(
        {
            **mixed_raw,
            "metric_protocol_note": "keep this diagnostic",
        },
        method="SS-TL",
        entity_key="target",
        config=config,
        elapsed=1.0,
    )

    assert noted_row["metric_protocol_note"] == "keep this diagnostic"


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
    assert row["result_status"] == "failed"
    assert row["failure_type"] == "silent_metric_failure"
    assert row["y_pred_nan_count"] == 170


def test_silent_metric_failure_annotation_preserves_training_exception() -> None:
    row = annotate_silent_metric_failure(
        {
            "rmse": np.nan,
            "smape": np.nan,
            "prediction_shape": (8, 1),
            "error": "RuntimeError: optimizer exploded",
        }
    )

    assert row["error"] == "RuntimeError: optimizer exploded"
    assert row["result_status"] == "failed"
    assert row["failure_type"] == "training_exception"


def test_silent_metric_failure_annotation_marks_missing_metric_without_exception() -> None:
    row = annotate_silent_metric_failure(
        {
            "rmse": np.nan,
            "smape": 12.0,
            "prediction_shape": (8, 1),
            "error": "",
        }
    )

    assert row["result_status"] == "failed"
    assert row["failure_type"] == "silent_metric_failure"
    assert row["error"].startswith("silent_metric_failure: rmse")


def test_source_failure_diagnostics_propagate_to_extracted_metrics_and_entity_row() -> None:
    failed_sources = [
        {
            "failed_source_key": ("bad", "item"),
            "exception_type": "NonFiniteArrayError",
            "exception_message": "model weights contain non-finite values: nan_count=1 inf_count=0",
        }
    ]
    raw = {
        "fused_result": {
            "rmse": 0.25,
            "accuracy": 4.0,
            "smape": 10.0,
            "prediction_shape": (2, 1),
        },
        "meta": {
            "selected_sources": [{"source_key": ("bad", "item")}, {"source_key": ("good", "item")}],
            "valid_source_count": 1,
            "skipped_source_count": 1,
            "failed_source_count": 1,
            "failed_source_keys": [("bad", "item")],
            "skipped_nonfinite_source_count": 1,
            "failed_sources": failed_sources,
            "selected_source_count": 2,
            "source_failure_messages": [
                "('bad', 'item'): NonFiniteArrayError: model weights contain non-finite values: nan_count=1 inf_count=0"
            ],
        },
    }

    extracted = _extract_method_metrics(raw, method_name="MSWA-TL")
    row = _row_from_result(
        extracted,
        method="MSWA-TL",
        entity_key="target",
        config={"dataset_id": 5, "dataset_name": "Dataset5", "info_sharing": "without", "source_count": 2},
        elapsed=1.0,
    )

    assert extracted["failed_sources"] == failed_sources
    assert row["valid_source_count"] == 1
    assert row["skipped_source_count"] == 1
    assert row["failed_source_count"] == 1
    assert row["selected_source_count"] == 2
    assert row["skipped_nonfinite_source_count"] == 1
    assert row["failed_source_keys"] == [("bad", "item")]
    assert json.loads(row["failed_sources"]) == [
        {
            "exception_message": "model weights contain non-finite values: nan_count=1 inf_count=0",
            "exception_type": "NonFiniteArrayError",
            "failed_source_key": ["bad", "item"],
        }
    ]
    assert json.loads(row["selected_sources"]) == [
        {"source_key": ["bad", "item"]},
        {"source_key": ["good", "item"]},
    ]
    assert json.loads(row["source_failure_messages"]) == [
        "('bad', 'item'): NonFiniteArrayError: model weights contain non-finite values: nan_count=1 inf_count=0"
    ]


def test_entity_experiment_forwards_metric_protocol_and_marks_unavailable_inverse_transform(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    source_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["source"] * len(dates),
            "item_id": [1] * len(dates),
            "sales": np.arange(1.0, 9.0),
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["target"] * len(dates),
            "item_id": [2] * len(dates),
            "sales": np.arange(2.0, 10.0),
        }
    )
    metric_protocol = {
        "current_metric_space": "normalized_minmax_space",
        "paper_metric_space": "original_sales_space",
        "strict_paper_metrics": False,
    }
    received_protocols: dict[str, dict] = {}

    def fake_runner(method: str):
        def runner(**kwargs):
            received_protocols[method] = kwargs["metric_protocol"]
            return {
                "rmse": 1.0,
                "accuracy": 0.5,
                "smape": 10.0,
                "error": "",
            }

        return runner

    monkeypatch.setattr(entity_experiment, "_method_runner", fake_runner)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="target",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=["sales"],
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 1,
            "horizon": 1,
            "window_size": 1,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
            "metric_protocol": metric_protocol,
        },
        enabled_methods=["No-TL", "MSWA-TL"],
    )

    assert received_protocols == {"No-TL": metric_protocol, "MSWA-TL": metric_protocol}
    for row in rows:
        assert row["metric_protocol"] == json.dumps(metric_protocol, ensure_ascii=False, sort_keys=True)
        assert row["metric_space_used"] == "normalized_minmax_space"
        assert row["paper_metric_aligned"] == "no_paper_reference"
        assert row["inverse_transform_applied"] is False
        assert row["metric_protocol_note"] == "inverse transform not available for solidified parquet path"


def test_d4_d6_entity_row_does_not_fabricate_paper_or_scale_metrics() -> None:
    row = _row_from_result(
        {
            "rmse": 1.25,
            "accuracy": 0.5,
            "smape": 7.5,
            "prediction_shape": (12, 1),
            "error": "",
        },
        method="MSWA-TL",
        entity_key="target",
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 3,
        },
        elapsed=1.0,
    )

    assert row["paper_metric_aligned"] == "no_paper_reference"
    assert row["rmse_paper"] == ""
    assert row["smape_paper"] == ""
    assert row["normalized_rmse"] == ""
    assert row["original_scale_rmse"] == ""


def test_should_skip_source_exception_only_skips_numeric_source_failures() -> None:
    from src.transfer_methods.source_failure_tolerance import should_skip_source_exception
    from src.utils.finite_diagnostics import NonFiniteArrayError

    skip_cases = [
        NonFiniteArrayError(
            "model weights contain non-finite values: nan_count=1 inf_count=0",
            diagnostics={"model_weight_nan_count": 1, "model_weight_inf_count": 0},
        ),
        FloatingPointError("overflow encountered in source model"),
        ValueError("prediction contains NaN values for source_key=('a', 'b')"),
        RuntimeError("SS-TL failed for source_key=('a', 'b'): invalid value encountered in divide"),
        RuntimeError("model weights contain non-finite values: nan_count=32545 inf_count=0"),
    ]
    for exc in skip_cases:
        assert should_skip_source_exception(exc), f"expected source skip for {exc!r}"

    fail_fast_cases = [
        ValueError("sales must remain in model_feature_cols for sequence target construction"),
        ValueError("Selected source_key not found in source_df: ('missing', 'item')"),
        ValueError("Inconsistent target y_test across source runs; cannot fuse predictions"),
        RuntimeError("schema mismatch: feature columns changed"),
        RuntimeError("target dataframe is missing required dates"),
        RuntimeError("config path is invalid"),
        RuntimeError("source optimizer failed"),
    ]
    for exc in fail_fast_cases:
        assert not should_skip_source_exception(exc), f"expected fail-fast for {exc!r}"
