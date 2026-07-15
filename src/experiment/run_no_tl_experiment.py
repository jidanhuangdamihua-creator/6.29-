"""No-TL experiment runner.

No-TL trains only on target-domain data and does not use any source-domain
samples or transferred weights.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.models.no_tl_model import build_no_tl_cnn_model
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.runtime_control import keras_verbose


def fit_no_tl_predictor(
    *,
    target_train_df: pd.DataFrame,
    target_validation_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    window_size: int = 10,
    learning_rate: float = 0.001,
    target_epochs: int = 3,
    batch_size: int = 16,
    feature_mask=None,
):
    """Fit the formal No-TL predictor without a test or evaluator argument."""

    from src.experiment.fitted_predictor import KerasPredictor

    columns = [str(column) for column in feature_cols]
    train_scaled, val_scaled, _, input_scaler, normalized_columns = normalize_features(
        target_train_df,
        target_validation_df,
        target_validation_df,
        feature_columns=columns,
    )
    x_train, y_train = build_tabular_sequence(
        train_scaled,
        horizon=int(horizon),
        window_size=int(window_size),
        feature_columns=normalized_columns,
    )
    x_val, y_val = build_tabular_sequence(
        val_scaled,
        horizon=int(horizon),
        window_size=int(window_size),
        feature_columns=normalized_columns,
    )
    if len(y_train) == 0:
        raise ValueError("No-TL target training windows are empty")
    x_train = to_cnn_tensor(x_train)
    x_val = to_cnn_tensor(x_val)
    model = build_no_tl_cnn_model(input_shape=x_train.shape[1:], learning_rate=learning_rate)
    fit_kwargs = {
        "epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "verbose": keras_verbose(),
    }
    if len(y_val):
        fit_kwargs["validation_data"] = (x_val, y_val)
    model.fit(x_train, y_train, **fit_kwargs)
    return KerasPredictor(
        model=model,
        feature_mask=feature_mask,
        input_scaler=input_scaler,
    )


def run_no_tl_experiment(
    target_df: pd.DataFrame,
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: dict | None = None,
    feature_cols: Sequence[str] | None = None,
    expected_metric_identity: dict[str, Any] | None = None,
):
    """Run No-TL baseline on target data only.

    Returns:
        dict with keys: method, rmse, accuracy, prediction_shape.
    """
    target_min_df = target_df.copy()
    explicit_feature_cols = [str(col) for col in feature_cols] if feature_cols is not None else None

    tgt_train, tgt_val, tgt_test = temporal_split_by_ratio_or_dates(target_min_df)
    tgt_train, tgt_val, tgt_test, tgt_scaler, tgt_feature_columns = normalize_features(
        tgt_train, tgt_val, tgt_test, feature_columns=explicit_feature_cols
    )

    x_train, y_train = build_tabular_sequence(
        tgt_train, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    x_val, y_val = build_tabular_sequence(
        tgt_val, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    x_test, y_test = build_tabular_sequence(
        tgt_test, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )

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
        "y_true": np.asarray(y_test).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
        "sales_scaler": tgt_scaler,
        "feature_columns": list(tgt_feature_columns),
        "prediction_shape": tuple(y_pred.shape),
        **metric_result,
        **dict(expected_metric_identity or {}),
    }
