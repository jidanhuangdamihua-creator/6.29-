from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.transfer_methods import mswa_tl
from src.transfer_methods.mssb_tl import evaluate_predictions
from src.transfer_methods.source_failure_tolerance import AllSourcesFailedError
from src.transfer_methods.mswa_tl import evaluate_fused_predictions
from src.utils.finite_diagnostics import NonFiniteArrayError
from src.utils import entity_experiment


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


class _FakeSelector:
    def select_top_k_sources(self, **kwargs):
        return {
            "sources": [
                {"source_key": ("bad", "item"), "distance": 0.1, "weight": 0.25},
                {"source_key": ("good", "item"), "distance": 0.2, "weight": 0.75},
            ]
        }


def _multi_source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.DataFrame(
        {
            "entity_id": ["bad", "bad", "good", "good"],
            "item_id": ["item", "item", "item", "item"],
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    )
    target_df = pd.DataFrame(
        {
            "entity_id": ["target", "target", "target"],
            "item_id": ["item", "item", "item"],
            "sales": [10.0, 20.0, 30.0],
        }
    )
    return source_df, target_df


def test_mswa_skips_nonfinite_source_and_records_diagnostics(monkeypatch) -> None:
    source_df, target_df = _multi_source_frames()
    y_test = np.array([10.0, 20.0])
    y_pred = np.array([[11.0], [19.0]])

    def fake_single_source(source_sequence_df, **kwargs):
        source_key = tuple(source_sequence_df[["entity_id", "item_id"]].iloc[0])
        if source_key == ("bad", "item"):
            raise NonFiniteArrayError(
                "model weights contain non-finite values: nan_count=1 inf_count=0",
                diagnostics={"model_weight_nan_count": 1, "model_weight_inf_count": 0},
            )
        return {
            "rmse": 1.0,
            "accuracy": 1.0,
            "y_pred": y_pred,
            "y_test": y_test,
            "prediction_shape": y_pred.shape,
            "sales_scaler": None,
            "feature_columns": ["sales"],
        }

    monkeypatch.setattr(mswa_tl, "SourceSelector", lambda: _FakeSelector())
    monkeypatch.setattr(mswa_tl, "temporal_split_by_ratio_or_dates", lambda df: (df, df, df))
    monkeypatch.setattr(mswa_tl, "run_single_source_tl_for_mswa", fake_single_source)

    result = mswa_tl.run_mswa_tl(source_df=source_df, target_df=target_df, feature_cols=["sales"], k=2)

    assert result["meta"]["valid_source_count"] == 1
    assert result["meta"]["failed_source_count"] == 1
    assert result["meta"]["skipped_source_count"] == 1
    assert result["meta"]["skipped_nonfinite_source_count"] == 1
    assert result["meta"]["failed_source_keys"] == [("bad", "item")]
    assert result["meta"]["failed_sources"][0]["exception_type"] == "NonFiniteArrayError"
    assert len(result["individual_results"]) == 1
    assert result["fused_result"]["prediction_shape"] == y_pred.shape
    assert np.isfinite(result["fused_result"]["rmse"])
    assert np.isfinite(result["fused_result"]["smape"])


def test_mswa_plain_runtime_error_remains_fail_fast(monkeypatch) -> None:
    source_df, target_df = _multi_source_frames()

    def fake_single_source(source_sequence_df, **kwargs):
        source_key = tuple(source_sequence_df[["entity_id", "item_id"]].iloc[0])
        if source_key == ("bad", "item"):
            raise RuntimeError("source optimizer failed")
        return {
            "rmse": 2.0,
            "accuracy": 0.5,
            "y_pred": np.array([[1.0], [2.0]]),
            "y_test": np.array([1.0, 2.0]),
            "prediction_shape": (2, 1),
            "sales_scaler": None,
            "feature_columns": ["sales"],
        }

    monkeypatch.setattr(mswa_tl, "SourceSelector", lambda: _FakeSelector())
    monkeypatch.setattr(mswa_tl, "temporal_split_by_ratio_or_dates", lambda df: (df, df, df))
    monkeypatch.setattr(mswa_tl, "run_single_source_tl_for_mswa", fake_single_source)

    with pytest.raises(RuntimeError, match="source optimizer failed"):
        mswa_tl.run_mswa_tl(source_df=source_df, target_df=target_df, feature_cols=["sales"], k=2)


def test_mswa_skipped_numeric_runtime_error_is_not_counted_as_nonfinite(monkeypatch) -> None:
    source_df, target_df = _multi_source_frames()

    def fake_single_source(source_sequence_df, **kwargs):
        source_key = tuple(source_sequence_df[["entity_id", "item_id"]].iloc[0])
        if source_key == ("bad", "item"):
            raise RuntimeError("prediction contains overflow for source_key=('bad', 'item')")
        return {
            "rmse": 2.0,
            "accuracy": 0.5,
            "y_pred": np.array([[1.0], [2.0]]),
            "y_test": np.array([1.0, 2.0]),
            "prediction_shape": (2, 1),
            "sales_scaler": None,
            "feature_columns": ["sales"],
        }

    monkeypatch.setattr(mswa_tl, "SourceSelector", lambda: _FakeSelector())
    monkeypatch.setattr(mswa_tl, "temporal_split_by_ratio_or_dates", lambda df: (df, df, df))
    monkeypatch.setattr(mswa_tl, "run_single_source_tl_for_mswa", fake_single_source)

    result = mswa_tl.run_mswa_tl(source_df=source_df, target_df=target_df, feature_cols=["sales"], k=2)

    assert result["meta"]["valid_source_count"] == 1
    assert result["meta"]["skipped_source_count"] == 1
    assert result["meta"]["failed_source_count"] == 1
    assert result["meta"]["skipped_nonfinite_source_count"] == 0
    assert result["meta"]["selected_source_count"] == 2
    assert result["meta"]["source_failure_messages"] == [
        "('bad', 'item'): RuntimeError: prediction contains overflow for source_key=('bad', 'item')"
    ]


def test_mswa_all_sources_failed_raises_typed_error(monkeypatch) -> None:
    source_df, target_df = _multi_source_frames()

    def fake_single_source(source_sequence_df, **kwargs):
        raise NonFiniteArrayError(
            "model weights contain non-finite values: nan_count=1 inf_count=0",
            diagnostics={"model_weight_nan_count": 1, "model_weight_inf_count": 0},
        )

    monkeypatch.setattr(mswa_tl, "SourceSelector", lambda: _FakeSelector())
    monkeypatch.setattr(mswa_tl, "temporal_split_by_ratio_or_dates", lambda df: (df, df, df))
    monkeypatch.setattr(mswa_tl, "run_single_source_tl_for_mswa", fake_single_source)

    with pytest.raises(AllSourcesFailedError) as captured:
        mswa_tl.run_mswa_tl(source_df=source_df, target_df=target_df, feature_cols=["sales"], k=2)

    assert captured.value.method_name == "MSWA-TL"
    assert len(captured.value.failed_sources) == 2
    assert len(captured.value.selected_sources) == 2


def test_entity_loop_writes_error_row_for_all_sources_failed(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    source_df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "entity_id": ["bad"] * 3 + ["also_bad"] * 3,
            "item_id": ["item"] * 6,
            "sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["target"] * 3,
            "item_id": ["item"] * 3,
            "sales": [10.0, 11.0, 12.0],
        }
    )
    failed_sources = [
        {
            "failed_source_key": ("bad", "item"),
            "exception_type": "NonFiniteArrayError",
            "exception_message": "model weights contain non-finite values: nan_count=1 inf_count=0",
        },
        {
            "failed_source_key": ("also_bad", "item"),
            "exception_type": "RuntimeError",
            "exception_message": "prediction contains NaN values",
        },
    ]
    selected_sources = [
        {"source_key": ("bad", "item"), "distance": 0.1, "weight": 0.5},
        {"source_key": ("also_bad", "item"), "distance": 0.2, "weight": 0.5},
    ]

    def fake_runner(**kwargs):
        raise AllSourcesFailedError("MSWA-TL", failed_sources, selected_sources=selected_sources)

    monkeypatch.setattr(entity_experiment, "run_mswa_experiment", fake_runner)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="target_item",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=["sales"],
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 2,
            "horizon": 1,
            "window_size": 1,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
            "target_entity_id": "target",
            "target_store_id": "target-store",
            "target_item_id": "item",
        },
        enabled_methods=["MSWA-TL"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "MSWA-TL"
    assert row["dataset"] == "Dataset5"
    assert row["dataset_id"] == 5
    assert row["information_sharing"] == "without"
    assert row["target_entity_key"] == "target_item"
    assert row["target_entity_id"] == "target"
    assert row["target_store_id"] == "target-store"
    assert row["target_item_id"] == "item"
    assert row["valid_source_count"] == 0
    assert row["selected_source_count"] == 2
    assert row["skipped_source_count"] == 2
    assert np.isnan(row["rmse"])
    assert np.isnan(row["smape"])
    assert "all selected sources failed" in row["error"].lower()
    assert json.loads(row["failed_sources"])[0]["failed_source_key"] == ["bad", "item"]
    assert len(json.loads(row["source_failure_messages"])) == 2
    assert len(json.loads(row["selected_sources"])) == 2


def test_transfer_methods_do_not_use_old_direct_ss_tl_source_raise() -> None:
    needle = 'raise RuntimeError(f"SS-TL failed for source_key={source_key}: {exc}") from exc'
    hits = [
        str(path)
        for path in Path("src/transfer_methods").rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    ]
    assert hits == []
