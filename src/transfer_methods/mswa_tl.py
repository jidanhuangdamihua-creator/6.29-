"""
Module 6: MSWA-TL (Multi-Source Weighted Average Transfer Learning)

This module implements output-level multi-source transfer fusion:
1. Select top-k similar sources from source pool
2. Run one SS-TL pipeline per selected source
3. Collect predictions from all source-specific target models
4. Fuse predictions via source weights
5. Evaluate fused predictions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.transfer_methods.single_source_tl import (
    build_target_model_from_source,
    evaluate_regression_model,
    fine_tune_target_model,
    train_source_model,
)
from src.source_selection.source_selector import SourceSelector
from src.evaluation.metrics import smape
from src.utils.finite_diagnostics import validate_finite_array
from src.utils.source_fillna import fill_source_numeric_na
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    SOURCE_LEVEL_EXCEPTIONS,
    make_failed_source,
    normalize_successful_source_weights,
    enforce_formal_source_success,
    runtime_selection_meta,
    should_skip_source_exception,
    source_failure_meta,
)
from src.protocols.runner_adapter import source_key_mask
from src.protocols.provenance import (
    assert_actual_cnn_training_validated,
    bind_actual_cnn_source_frame,
)


LOGGER_NAME = "experiment"


def _get_logger() -> logging.Logger:
    """Get project-level logger and initialize fallback logging if needed."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


def _validate_feature_cols(df: pd.DataFrame, feature_cols: Sequence[str], where: str) -> List[str]:
    """Validate feature columns existence and return a concrete list."""
    cols = list(feature_cols)
    if not cols:
        raise ValueError("feature_cols must not be empty")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {where}: {missing}")
    return cols


def _prepare_single_source_split(source_sequence_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one source sequence into train/val/test using source-style ratio config."""
    source_one = source_sequence_df.copy()
    source_one.attrs["split_role"] = "source"
    source_one.attrs["split_mode"] = "ratio"
    source_one.attrs["split_config"] = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    return temporal_split_by_ratio_or_dates(source_one)


def run_single_source_tl_for_mswa(
    source_sequence_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
) -> Dict[str, object]:
    """
    Run one SS-TL pipeline for MSWA using one source sequence and shared target split.

    Steps:
    - Build source/target tabular windows with module 2
    - Train source model with module 4
    - Transfer to target model and fine-tune on target train/val
    - Evaluate on target test and return y_pred for fusion

    Returns:
        {
          "source_model_trained": bool,
          "target_model_built": bool,
          "fine_tuned": bool,
          "rmse": float,
          "accuracy": float,
          "y_pred": np.ndarray,
          "y_test": np.ndarray,
          "prediction_shape": tuple,
        }
    """
    logger = _get_logger()
    logger.info("[run_single_source_tl_for_mswa] Start.")

    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")
    _validate_feature_cols(target_train_df, feature_cols, where="target_train_df")
    _validate_feature_cols(target_val_df, feature_cols, where="target_val_df")
    _validate_feature_cols(target_test_df, feature_cols, where="target_test_df")

    source_sequence_df = fill_source_numeric_na(source_sequence_df, feature_columns=feature_cols)
    src_train_df, src_val_df, src_test_df = _prepare_single_source_split(source_sequence_df)

    src_train_df, src_val_df, src_test_df, _, _ = normalize_features(
        src_train_df, src_val_df, src_test_df, feature_columns=feature_cols
    )
    tgt_train_df, tgt_val_df, tgt_test_df, tgt_scaler, tgt_feature_columns = normalize_features(
        target_train_df, target_val_df, target_test_df, feature_columns=feature_cols
    )

    x_source, y_source = build_tabular_sequence(
        src_train_df, horizon=horizon, window_size=window_size, feature_columns=feature_cols
    )
    if src_train_df.attrs.get("protocol_actual_source_key") is not None:
        assert_actual_cnn_training_validated(
            src_train_df,
            source_key=src_train_df.attrs["protocol_actual_source_key"],
        )
    x_tgt_train, y_tgt_train = build_tabular_sequence(
        tgt_train_df, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    x_tgt_val, y_tgt_val = build_tabular_sequence(
        tgt_val_df, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    x_tgt_test, y_tgt_test = build_tabular_sequence(
        tgt_test_df, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )

    if len(y_source) == 0:
        raise ValueError("Source sequence produced zero training windows; adjust window_size/horizon.")
    if len(y_tgt_train) == 0:
        raise ValueError("Target train split produced zero windows; adjust window_size/horizon.")
    if len(y_tgt_test) == 0:
        raise ValueError("Target test split produced zero windows; adjust window_size/horizon.")

    x_source = to_cnn_tensor(x_source)
    x_tgt_train = to_cnn_tensor(x_tgt_train)
    x_tgt_val = to_cnn_tensor(x_tgt_val)
    x_tgt_test = to_cnn_tensor(x_tgt_test)

    if x_source.shape[1:] != x_tgt_train.shape[1:]:
        raise ValueError(
            "Shape mismatch between source and target train tensors: "
            f"source={x_source.shape[1:]} target_train={x_tgt_train.shape[1:]}"
        )
    if x_tgt_val.shape[1:] != x_tgt_train.shape[1:] or x_tgt_test.shape[1:] != x_tgt_train.shape[1:]:
        raise ValueError(
            "Target split tensor shape mismatch: "
            f"train={x_tgt_train.shape[1:]} val={x_tgt_val.shape[1:]} test={x_tgt_test.shape[1:]}"
        )

    input_shape = x_source.shape[1:]

    source_model = train_source_model(
        X_source=x_source,
        y_source=y_source,
        input_shape=input_shape,
        learning_rate=learning_rate,
        epochs=source_epochs,
        batch_size=batch_size,
    )

    target_model, _ = build_target_model_from_source(
        source_model=source_model,
        input_shape=input_shape,
        learning_rate=learning_rate,
        freeze_first_n_layers=4,
    )

    target_model = fine_tune_target_model(
        target_model=target_model,
        X_target_train=x_tgt_train,
        y_target_train=y_tgt_train,
        X_target_val=x_tgt_val,
        y_target_val=y_tgt_val,
        epochs=target_epochs,
        batch_size=batch_size,
    )

    eval_result = evaluate_regression_model(target_model, x_tgt_test, y_tgt_test)
    y_pred = target_model.predict(x_tgt_test, verbose=0)
    diagnostics = {
        key: value
        for key, value in eval_result.items()
        if key.endswith("_nan_count")
        or key.endswith("_inf_count")
        or key in {"X_test_shape", "y_true_shape", "y_pred_shape"}
    }
    diagnostics.update(validate_finite_array(y_pred, name="y_pred", context=diagnostics))

    logger.info(
        "[run_single_source_tl_for_mswa] Finished. rmse=%.4f accuracy=%.4f pred_shape=%s",
        float(eval_result["rmse"]),
        float(eval_result["accuracy"]),
        tuple(y_pred.shape),
    )

    return {
        "source_model_trained": True,
        "target_model_built": True,
        "fine_tuned": True,
        "rmse": float(eval_result["rmse"]),
        "accuracy": float(eval_result["accuracy"]),
        "y_pred": np.asarray(y_pred),
        "y_test": np.asarray(y_tgt_test),
        "prediction_shape": tuple(y_pred.shape),
        "sales_scaler": tgt_scaler,
        "feature_columns": tgt_feature_columns,
        **diagnostics,
    }


def weighted_prediction_fusion(predictions_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """
    Fuse multiple predictions by weighted average.

    Args:
        predictions_list: Sequence of prediction arrays with identical shape.
        weights: Sequence of source weights.

    Returns:
        Weighted prediction with the same shape as one prediction input.
    """
    if not predictions_list:
        raise ValueError("predictions_list must not be empty")

    preds = [np.asarray(p, dtype=np.float64) for p in predictions_list]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    if len(preds) != w.shape[0]:
        raise ValueError(
            "Number of predictions and weights must match: "
            f"predictions={len(preds)} weights={w.shape[0]}"
        )

    first_shape = preds[0].shape
    for i, p in enumerate(preds):
        if p.shape != first_shape:
            raise ValueError(
                "All prediction shapes must match for fusion: "
                f"index={i} shape={p.shape} expected={first_shape}"
            )

    stacked = np.stack(preds, axis=0)
    # Broadcast weights from (k,) to (k, 1, ..., 1) to match prediction dims.
    weight_shape = (w.shape[0],) + (1,) * (stacked.ndim - 1)
    fused = np.sum(stacked * w.reshape(weight_shape), axis=0)
    return fused


def evaluate_fused_predictions(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, object]:
    """
    Evaluate fused predictions with RMSE and Accuracy=1/(RMSE+eps).

    Args:
        y_true: Ground-truth values.
        y_pred: Fused predictions.
        eps: Numerical stability term.

    Returns:
        {
          "rmse": float,
          "accuracy": float,
          "prediction_shape": tuple,
        }
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    y_pred_flat = y_pred_arr.reshape(-1)
    diagnostics = {}
    diagnostics.update(validate_finite_array(y_true_arr, name="y_true"))
    diagnostics.update(validate_finite_array(y_pred_arr, name="y_pred", context=diagnostics))

    if y_true_arr.shape[0] != y_pred_flat.shape[0]:
        raise ValueError(
            "y_true and y_pred size mismatch: "
            f"y_true={y_true_arr.shape[0]} y_pred={y_pred_flat.shape[0]}"
        )

    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_arr) ** 2)))
    accuracy = float(1.0 / (rmse + eps))
    smape_value = float(smape(y_true_arr, y_pred_flat, epsilon=eps))

    return {
        "rmse": rmse,
        "accuracy": accuracy,
        "smape": smape_value,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "prediction_shape": tuple(y_pred_arr.shape),
        **diagnostics,
    }


def run_mswa_tl(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, object]:
    """
    Run MSWA-TL: top-k source selection + per-source SS-TL + weighted output fusion.

    Args:
        source_df: Source pool dataframe.
        target_df: Target dataframe.
        feature_cols: Feature columns used by source selector and metadata trace.
        k: Number of selected sources.
        horizon: Forecast horizon for sequence windows.
        window_size: Lookback window size.
        weight_mode: Weight mode for source selector.
        learning_rate: Learning rate for source/target model training.
        source_epochs: Source pretrain epochs.
        target_epochs: Target fine-tune epochs.
        batch_size: Mini-batch size.

    Returns:
        {
          "meta": {
            "method": "MSWA-TL",
            "k": int,
            "weight_mode": str,
            "feature_cols": list[str],
            "selected_sources": list[dict],
          },
          "individual_results": [
            {
              "source_key": tuple,
              "distance": float,
              "weight": float,
              "rmse": float,
              "accuracy": float,
              "prediction_shape": tuple,
            },
            ...
          ],
          "fused_result": {
            "rmse": float,
            "accuracy": float,
            "prediction_shape": tuple,
          },
        }
    """
    logger = _get_logger()
    logger.info("[run_mswa_tl] Start. k=%d weight_mode=%s", k, weight_mode)

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")
    resolved_group_cols = tuple(group_cols)

    selector = SourceSelector()
    selection_result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=resolved_group_cols,
        weight_mode=weight_mode,
    )

    selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
    if not selected_sources:
        raise ValueError("No source selected from source pool.")

    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)

    predictions: List[np.ndarray] = []
    weights: List[float] = []
    individual_results: List[Dict[str, object]] = []
    y_test_reference: np.ndarray | None = None
    target_scaler_reference = None
    target_feature_columns_reference = None
    failed_sources: List[Dict[str, object]] = []

    for selected in selected_sources:
        source_key = tuple(selected["source_key"]) if isinstance(selected["source_key"], (list, tuple)) else (selected["source_key"],)
        if len(source_key) != len(resolved_group_cols):
            raise ValueError(f"Invalid source_key format: {source_key}")

        source_mask = source_key_mask(source_df, resolved_group_cols, source_key)
        source_sequence_df = source_df[source_mask].copy()

        if source_sequence_df.empty:
            raise ValueError(f"Selected source_key not found in source_df: {source_key}")
        bind_actual_cnn_source_frame(
            source_sequence_df,
            source_key=source_key,
            group_cols=resolved_group_cols,
            feature_cols=feature_cols,
        )

        try:
            one_result = run_single_source_tl_for_mswa(
                source_sequence_df=source_sequence_df,
                target_train_df=target_train_df,
                target_val_df=target_val_df,
                target_test_df=target_test_df,
                feature_cols=feature_cols,
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                source_epochs=source_epochs,
                target_epochs=target_epochs,
                batch_size=batch_size,
            )
        except SOURCE_LEVEL_EXCEPTIONS as exc:
            enforce_formal_source_success(source_df, source_key, exc)
            if not should_skip_source_exception(exc):
                raise
            failed_source = make_failed_source(source_key, exc)
            failed_sources.append(failed_source)
            logger.warning(
                "[run_mswa_tl] Skipping failed source_key=%s exception_type=%s message=%s",
                source_key,
                failed_source["exception_type"],
                failed_source["exception_message"],
            )
            continue

        y_pred = np.asarray(one_result["y_pred"])
        y_test = np.asarray(one_result["y_test"])

        if y_test_reference is None:
            y_test_reference = y_test
            target_scaler_reference = one_result.get("sales_scaler")
            target_feature_columns_reference = one_result.get("feature_columns")
        else:
            if y_test_reference.shape != y_test.shape or not np.allclose(y_test_reference, y_test, atol=1e-8):
                raise ValueError(
                    "Inconsistent target y_test across source runs; cannot fuse predictions. "
                    f"reference_shape={y_test_reference.shape} current_shape={y_test.shape}"
                )

        predictions.append(y_pred)
        weights.append(float(selected["weight"]))
        individual_results.append(
            {
                "source_key": source_key,
                "distance": float(selected["distance"]),
                "weight": float(selected["weight"]),
                "rmse": float(one_result["rmse"]),
                "accuracy": float(one_result["accuracy"]),
                "prediction_shape": tuple(one_result["prediction_shape"]),
            }
        )

    if not individual_results:
        raise AllSourcesFailedError(
            "MSWA-TL",
            failed_sources,
            selected_sources=selected_sources,
            selection_meta=runtime_selection_meta(selection_result),
        )

    normalized_weights = normalize_successful_source_weights(weights)
    for item, normalized_weight in zip(individual_results, normalized_weights):
        item["weight"] = float(normalized_weight)

    fused_pred = weighted_prediction_fusion(predictions_list=predictions, weights=normalized_weights)
    if y_test_reference is None:
        raise ValueError("No valid y_test found from source runs.")

    fused_result = evaluate_fused_predictions(y_true=y_test_reference, y_pred=fused_pred)
    fused_result["sales_scaler"] = target_scaler_reference
    fused_result["feature_columns"] = target_feature_columns_reference

    result = {
        "meta": {
            "method": "MSWA-TL",
            "k": int(k),
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "knn_feature_mode": (selection_result.get("meta", {}) or {}).get("knn_feature_mode", ""),
            "selected_sources": selected_sources,
            **runtime_selection_meta(selection_result),
            **source_failure_meta(
                requested_k=k,
                selected_sources=selected_sources,
                valid_source_count=len(individual_results),
                failed_sources=failed_sources,
            ),
        },
        "individual_results": individual_results,
        "fused_result": fused_result,
    }

    logger.info(
        "[run_mswa_tl] Finished. selected=%d fused_rmse=%.4f fused_accuracy=%.4f",
        len(individual_results),
        float(fused_result["rmse"]),
        float(fused_result["accuracy"]),
    )
    return result
