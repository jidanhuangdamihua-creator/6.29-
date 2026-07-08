from __future__ import annotations

import numpy as np
import pandas as pd

from src.transfer_methods import mswa_tl
from src.transfer_methods.mssb_tl import evaluate_predictions
from src.transfer_methods.mswa_tl import evaluate_fused_predictions
from src.utils.finite_diagnostics import NonFiniteArrayError


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


def test_mswa_skipped_runtime_error_is_not_counted_as_nonfinite(monkeypatch) -> None:
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

    result = mswa_tl.run_mswa_tl(source_df=source_df, target_df=target_df, feature_cols=["sales"], k=2)

    assert result["meta"]["valid_source_count"] == 1
    assert result["meta"]["skipped_source_count"] == 1
    assert result["meta"]["failed_source_count"] == 1
    assert result["meta"]["skipped_nonfinite_source_count"] == 0
