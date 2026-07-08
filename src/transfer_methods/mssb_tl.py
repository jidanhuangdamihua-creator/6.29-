"""
Module 7: MSSB-TL (Multi-Source Switching-Based Transfer Learning)

This module implements model switching across multiple source-specific SS-TL models:
1. Select top-k similar sources from source pool
2. Train one SS-TL model per source
3. Evaluate each model on target validation/test splits
4. Select the best source model by minimum validation RMSE
5. Use the selected model's test performance as final MSSB-TL result
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

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
    fine_tune_target_model,
    train_source_model,
)
from src.source_selection.source_selector import SourceSelector
from src.evaluation.metrics import smape
from src.utils.finite_diagnostics import validate_finite_array
from src.transfer_methods.source_failure_tolerance import (
    SOURCE_LEVEL_EXCEPTIONS,
    all_sources_failed_message,
    make_failed_source,
    should_skip_source_exception,
    source_failure_meta,
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, object]:
    """
    Evaluate predictions with RMSE and Accuracy=1/(RMSE+eps).

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
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


def run_single_source_tl_for_mssb(
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
    Run one SS-TL pipeline for MSSB using one source sequence and shared target split.

    Steps:
    - Build source/target tabular windows with module 2
    - Train source model with module 4
    - Transfer to target model and fine-tune on target train/val
    - Predict and evaluate on target validation and target test

    Returns:
        {
          "source_model_trained": bool,
          "target_model_built": bool,
          "fine_tuned": bool,
          "val_rmse": float,
          "val_accuracy": float,
          "test_rmse": float,
          "test_accuracy": float,
          "y_val_pred": np.ndarray,
          "y_test_pred": np.ndarray,
          "y_val_true": np.ndarray,
          "y_test_true": np.ndarray,
          "prediction_shape": tuple,
        }
    """
    logger = _get_logger()
    logger.info("[run_single_source_tl_for_mssb] Start.")

    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")
    _validate_feature_cols(target_train_df, feature_cols, where="target_train_df")
    _validate_feature_cols(target_val_df, feature_cols, where="target_val_df")
    _validate_feature_cols(target_test_df, feature_cols, where="target_test_df")

    src_train_df, src_val_df, src_test_df = _prepare_single_source_split(source_sequence_df)

    src_train_df, src_val_df, src_test_df, _, _ = normalize_features(src_train_df, src_val_df, src_test_df)
    tgt_train_df, tgt_val_df, tgt_test_df, tgt_scaler, tgt_feature_columns = normalize_features(
        target_train_df, target_val_df, target_test_df
    )

    x_source, y_source = build_tabular_sequence(src_train_df, horizon=horizon, window_size=window_size)
    x_tgt_train, y_tgt_train = build_tabular_sequence(tgt_train_df, horizon=horizon, window_size=window_size)
    x_tgt_val, y_tgt_val = build_tabular_sequence(tgt_val_df, horizon=horizon, window_size=window_size)
    x_tgt_test, y_tgt_test = build_tabular_sequence(tgt_test_df, horizon=horizon, window_size=window_size)

    if len(y_source) == 0:
        raise ValueError("Source sequence produced zero training windows; adjust window_size/horizon.")
    if len(y_tgt_train) == 0:
        raise ValueError("Target train split produced zero windows; adjust window_size/horizon.")
    if len(y_tgt_val) == 0:
        raise ValueError("Target val split produced zero windows; adjust window_size/horizon.")
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

    y_val_pred = target_model.predict(x_tgt_val, verbose=0)
    y_test_pred = target_model.predict(x_tgt_test, verbose=0)
    diagnostics = {}
    diagnostics.update(validate_finite_array(x_tgt_test, name="X_test"))
    diagnostics.update(validate_finite_array(y_tgt_test, name="y_true", context=diagnostics))
    diagnostics.update(validate_finite_array(y_test_pred, name="y_pred", context=diagnostics))

    val_eval = evaluate_predictions(y_true=y_tgt_val, y_pred=y_val_pred)
    test_eval = evaluate_predictions(y_true=y_tgt_test, y_pred=y_test_pred)

    if y_val_pred.shape != y_test_pred.shape and y_test_pred.ndim != y_val_pred.ndim:
        raise ValueError(
            "Prediction ndim mismatch between val and test outputs: "
            f"val_shape={y_val_pred.shape} test_shape={y_test_pred.shape}"
        )

    logger.info(
        "[run_single_source_tl_for_mssb] Finished. val_rmse=%.4f test_rmse=%.4f pred_shape=%s",
        float(val_eval["rmse"]),
        float(test_eval["rmse"]),
        tuple(y_test_pred.shape),
    )

    return {
        "source_model_trained": True,
        "target_model_built": True,
        "fine_tuned": True,
        "val_rmse": float(val_eval["rmse"]),
        "val_accuracy": float(val_eval["accuracy"]),
        "val_smape": float(val_eval["smape"]),
        "test_rmse": float(test_eval["rmse"]),
        "test_accuracy": float(test_eval["accuracy"]),
        "test_smape": float(test_eval["smape"]),
        "y_val_pred": np.asarray(y_val_pred),
        "y_test_pred": np.asarray(y_test_pred),
        "y_val_true": np.asarray(y_tgt_val),
        "y_test_true": np.asarray(y_tgt_test),
        "prediction_shape": tuple(y_test_pred.shape),
        "sales_scaler": tgt_scaler,
        "feature_columns": tgt_feature_columns,
        **diagnostics,
    }


def select_best_source_model(individual_results: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], int]:
    """
    Select best source model by minimum validation RMSE.

    Args:
        individual_results: Per-source result dict list containing val_rmse.

    Returns:
        (best_result, best_index)
    """
    if not individual_results:
        raise ValueError("individual_results must not be empty")

    best_index = int(np.argmin([float(r["val_rmse"]) for r in individual_results]))
    return dict(individual_results[best_index]), best_index


def run_mssb_tl(
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
) -> Dict[str, object]:
    """
    Run MSSB-TL: top-k source selection + per-source SS-TL + val-based model switching.

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
            "method": "MSSB-TL",
            "k": int,
            "weight_mode": str,
            "feature_cols": list[str],
            "selected_sources": list[dict],
          },
          "individual_results": [...],
          "best_source_result": {...},
          "final_result": {
            "rmse": float,
            "accuracy": float,
            "prediction_shape": tuple,
          },
        }
    """
    logger = _get_logger()
    logger.info("[run_mssb_tl] Start. k=%d weight_mode=%s", k, weight_mode)

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")

    selector = SourceSelector()
    selection_result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=k,
        weight_mode=weight_mode,
    )

    selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
    if not selected_sources:
        raise ValueError("No source selected from source pool.")

    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)

    individual_results: List[Dict[str, object]] = []
    test_shape_reference: Optional[Tuple[int, ...]] = None
    failed_sources: List[Dict[str, object]] = []

    for selected in selected_sources:
        source_key = tuple(selected["source_key"]) if isinstance(selected["source_key"], (list, tuple)) else (selected["source_key"],)
        if len(source_key) < 2:
            raise ValueError(f"Invalid source_key format: {source_key}")

        entity_id, item_id = source_key[0], source_key[1]
        source_sequence_df = source_df[
            (source_df["entity_id"] == entity_id) & (source_df["item_id"] == item_id)
        ].copy()

        if source_sequence_df.empty:
            raise ValueError(f"Selected source_key not found in source_df: {source_key}")

        try:
            one_result = run_single_source_tl_for_mssb(
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
            if not should_skip_source_exception(exc):
                raise
            failed_source = make_failed_source(source_key, exc)
            failed_sources.append(failed_source)
            logger.warning(
                "[run_mssb_tl] Skipping failed source_key=%s exception_type=%s message=%s",
                source_key,
                failed_source["exception_type"],
                failed_source["exception_message"],
            )
            continue

        pred_shape = tuple(one_result["prediction_shape"])
        if test_shape_reference is None:
            test_shape_reference = pred_shape
        elif test_shape_reference != pred_shape:
            raise ValueError(
                "Inconsistent prediction shape across source runs: "
                f"reference={test_shape_reference} current={pred_shape}"
            )

        individual_results.append(
            {
                "source_key": source_key,
                "distance": float(selected["distance"]),
                "weight": float(selected["weight"]),
                "val_rmse": float(one_result["val_rmse"]),
                "val_accuracy": float(one_result["val_accuracy"]),
                "val_smape": float(one_result["val_smape"]),
                "test_rmse": float(one_result["test_rmse"]),
                "test_accuracy": float(one_result["test_accuracy"]),
                "test_smape": float(one_result["test_smape"]),
                "y_test_true": np.asarray(one_result["y_test_true"]),
                "y_test_pred": np.asarray(one_result["y_test_pred"]),
                "sales_scaler": one_result.get("sales_scaler"),
                "feature_columns": one_result.get("feature_columns"),
                "prediction_shape": pred_shape,
            }
        )

    if not individual_results:
        raise RuntimeError(all_sources_failed_message("MSSB-TL", failed_sources))

    best_source_result, best_index = select_best_source_model(individual_results)

    logger.info(
        "[run_mssb_tl] Best source selected. index=%d source_key=%s val_rmse=%.4f",
        best_index,
        best_source_result["source_key"],
        float(best_source_result["val_rmse"]),
    )

    final_result = {
        "rmse": float(best_source_result["test_rmse"]),
        "accuracy": float(best_source_result["test_accuracy"]),
        "smape": float(best_source_result["test_smape"]),
        "y_true": np.asarray(best_source_result["y_test_true"]),
        "y_pred": np.asarray(best_source_result["y_test_pred"]),
        "sales_scaler": best_source_result.get("sales_scaler"),
        "feature_columns": best_source_result.get("feature_columns"),
        "prediction_shape": tuple(best_source_result["prediction_shape"]),
    }
    public_individual_results = []
    for item in individual_results:
        public_item = dict(item)
        public_item.pop("sales_scaler", None)
        public_item.pop("feature_columns", None)
        public_individual_results.append(public_item)
    public_best_source_result = dict(best_source_result)
    public_best_source_result.pop("sales_scaler", None)
    public_best_source_result.pop("feature_columns", None)

    result = {
        "meta": {
            "method": "MSSB-TL",
            "k": int(k),
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "selected_sources": selected_sources,
            **source_failure_meta(
                requested_k=k,
                selected_sources=selected_sources,
                valid_source_count=len(individual_results),
                failed_sources=failed_sources,
            ),
        },
        "individual_results": public_individual_results,
        "best_source_result": public_best_source_result,
        "final_result": final_result,
    }

    logger.info(
        "[run_mssb_tl] Finished. final_rmse=%.4f final_accuracy=%.4f",
        float(final_result["rmse"]),
        float(final_result["accuracy"]),
    )
    return result
