from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest

from src.experiment import experiment_runner
from src.experiment.experiment_runner import (
    run_msml_experiment,
    run_msml_rfe_experiment,
    run_mssb_experiment,
    run_mswa_experiment,
)


def _result() -> Dict[str, Any]:
    return {
        "fused_result": {
            "rmse": 1.0,
            "accuracy": 0.5,
            "prediction_shape": (1, 1),
        }
    }


@pytest.mark.parametrize(
    ("wrapper", "module_path", "function_name", "is_rfe"),
    [
        (
            run_mswa_experiment,
            "src.transfer_methods.mswa_tl",
            "run_mswa_tl",
            False,
        ),
        (
            run_mssb_experiment,
            "src.transfer_methods.mssb_tl",
            "run_mssb_tl",
            False,
        ),
        (
            run_msml_experiment,
            "src.transfer_methods.msml_tl",
            "run_msml_tl",
            False,
        ),
        (
            run_msml_rfe_experiment,
            "src.transfer_methods.msml_tl_rfe",
            "run_msml_tl_rfe",
            True,
        ),
    ],
)
def test_multi_source_wrappers_translate_number_of_sources_to_k(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., Dict[str, Any]],
    module_path: str,
    function_name: str,
    is_rfe: bool,
) -> None:
    received: Dict[str, Any] = {}

    if is_rfe:

        def strict_runner(
            source_df: pd.DataFrame,
            target_df: pd.DataFrame,
            feature_cols: list[str],
            k: int,
            group_cols: tuple[str, ...],
            horizon: int,
            window_size: int,
            weight_mode: str,
            estimator_name: str,
            keep_ratio: float,
            learning_rate: float,
            source_epochs: int,
            target_epochs: int,
            batch_size: int,
            random_state: int = 42,
            metric_identity: dict[str, Any] | None = None,
        ) -> Dict[str, Any]:
            received.update(k=k, group_cols=group_cols, metric_identity=metric_identity)
            return _result()

    else:

        def strict_runner(
            source_df: pd.DataFrame,
            target_df: pd.DataFrame,
            feature_cols: list[str],
            k: int,
            group_cols: tuple[str, ...],
            horizon: int,
            window_size: int,
            weight_mode: str,
            learning_rate: float,
            source_epochs: int,
            target_epochs: int,
            batch_size: int,
            metric_identity: dict[str, Any] | None = None,
        ) -> Dict[str, Any]:
            received.update(k=k, group_cols=group_cols, metric_identity=metric_identity)
            return _result()

    module = __import__(module_path, fromlist=[function_name])
    monkeypatch.setattr(module, function_name, strict_runner)

    kwargs: Dict[str, Any] = {
        "source_df": pd.DataFrame(),
        "target_df": pd.DataFrame(),
        "feature_cols": ["sales"],
        "k": 3,
        "number_of_sources": 7,
        "include_sales_in_knn": False,
        "metric_protocol": {"strict_paper_metrics": False},
        "group_cols": ("entity_id", "item_id"),
    }
    if is_rfe:
        kwargs.update(estimator_name="random_forest", keep_ratio=0.5)

    result = wrapper(**kwargs)

    assert received["k"] == 7
    assert received["group_cols"] == ("entity_id", "item_id")
    assert received["metric_identity"] is None
    assert result["rmse"] == 1.0
    assert np.isnan(result["smape"])


def test_ss_tl_wrapper_does_not_forward_metric_protocol_to_bottom_evaluator() -> None:
    tree = ast.parse(Path("src/experiment/experiment_runner.py").read_text(encoding="utf-8"))
    wrapper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_ss_tl_experiment"
    )
    calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_regression_model"
    ]

    assert len(calls) == 1
    assert {kw.arg for kw in calls[0].keywords} == {"model", "X_test", "y_test"}


def test_single_source_wrappers_accept_expected_metric_identity() -> None:
    assert "expected_metric_identity" in inspect.signature(
        experiment_runner.run_no_tl_experiment
    ).parameters
    assert "expected_metric_identity" in inspect.signature(
        experiment_runner.run_ss_tl_experiment
    ).parameters
