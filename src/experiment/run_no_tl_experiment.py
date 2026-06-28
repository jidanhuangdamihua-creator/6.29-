"""No-TL experiment runner.

No-TL trains only on target-domain data and does not use any source-domain
samples or transferred weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.models.no_tl_model import build_no_tl_cnn_model
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.runtime_control import keras_verbose


def run_no_tl_experiment(
    target_df: pd.DataFrame,
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: dict | None = None,
):
    """Run No-TL baseline on target data only.

    Returns:
        dict with keys: method, rmse, accuracy, prediction_shape.
    """
    target_min_df = target_df.copy()

    tgt_train, tgt_val, tgt_test = temporal_split_by_ratio_or_dates(target_min_df)
    tgt_train, tgt_val, tgt_test, tgt_scaler, tgt_feature_columns = normalize_features(tgt_train, tgt_val, tgt_test)

    x_train, y_train = build_tabular_sequence(tgt_train, horizon=horizon, window_size=window_size)
    x_val, y_val = build_tabular_sequence(tgt_val, horizon=horizon, window_size=window_size)
    x_test, y_test = build_tabular_sequence(tgt_test, horizon=horizon, window_size=window_size)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("No-TL target windows are empty; adjust window_size/horizon.")

    x_train = to_cnn_tensor(x_train)
    x_val = to_cnn_tensor(x_val)
    x_test = to_cnn_tensor(x_test)

    input_shape = x_train.shape[1:]
    model = build_no_tl_cnn_model(input_shape=input_shape, learning_rate=learning_rate)
    fit_kwargs = {"epochs": target_epochs, "batch_size": batch_size, "verbose": keras_verbose()}
    if len(y_val) > 0:
        fit_kwargs["validation_data"] = (x_val, y_val)

    model.fit(x_train, y_train, **fit_kwargs)

    y_pred = model.predict(x_test, verbose=0)
    metric_result = compute_metrics_with_protocol(
        y_true=y_test,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=tgt_scaler,
        feature_columns=tgt_feature_columns,
    )

    return {
        "method": "No-TL",
        "rmse": float(metric_result["rmse"]),
        "accuracy": float(metric_result["accuracy"]),
        "mae": float(metric_result.get("mae", float("nan"))),
        "mape": float(metric_result.get("mape", float("nan"))),
        "smape": float(metric_result["smape"]),
        "rmse_current": float(metric_result.get("rmse_current", float("nan"))),
        "accuracy_current": float(metric_result.get("accuracy_current", float("nan"))),
        "mae_current": float(metric_result.get("mae_current", float("nan"))),
        "mape_current": float(metric_result.get("mape_current", float("nan"))),
        "smape_current": float(metric_result.get("smape_current", float("nan"))),
        "rmse_paper": float(metric_result.get("rmse_paper", float("nan"))),
        "accuracy_paper": float(metric_result.get("accuracy_paper", float("nan"))),
        "mae_paper": float(metric_result.get("mae_paper", float("nan"))),
        "mape_paper": float(metric_result.get("mape_paper", float("nan"))),
        "smape_paper": float(metric_result.get("smape_paper", float("nan"))),
        "normalized_rmse": metric_result.get("normalized_rmse"),
        "normalized_accuracy": metric_result.get("normalized_accuracy"),
        "normalized_mae": metric_result.get("normalized_mae"),
        "normalized_mape": metric_result.get("normalized_mape"),
        "normalized_smape": metric_result.get("normalized_smape"),
        "original_scale_rmse": metric_result.get("original_scale_rmse"),
        "original_scale_accuracy": metric_result.get("original_scale_accuracy"),
        "original_scale_mae": metric_result.get("original_scale_mae"),
        "original_scale_mape": metric_result.get("original_scale_mape"),
        "original_scale_smape": metric_result.get("original_scale_smape"),
        "metric_space": str(metric_result.get("metric_space", metric_result["metric_space_current"])),
        "metric_space_used": str(metric_result.get("metric_space_used", metric_result.get("metric_space", ""))),
        "metric_space_current": str(metric_result["metric_space_current"]),
        "metric_space_paper": str(metric_result["metric_space_paper"]),
        "paper_metric_aligned": bool(metric_result["paper_metric_aligned"]),
        "inverse_transform_applied": bool(metric_result["inverse_transform_applied"]),
        "inverse_transform_available": bool(metric_result.get("inverse_transform_available", False)),
        "metric_notes": str(metric_result["metric_notes"]),
        "prediction_shape": tuple(y_pred.shape),
    }
