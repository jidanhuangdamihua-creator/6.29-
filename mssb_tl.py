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
    from environment import setup_logging
except ImportError:
    setup_logging = None

from data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.experiment_hyperparams import FIXED_EPOCHS, FIXED_LEARNING_RATE, fixed_hyperparams_summary
from src.utils.source_fillna import fill_source_numeric_na
from single_source_tl import (
    build_target_model_from_source,
    fine_tune_target_model,
    train_source_model,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
)
from source_selector import SourceSelector


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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    metric_protocol: dict | None = None,
    sales_scaler: object | None = None,
    feature_columns: object | None = None,
) -> Dict[str, object]:
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
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    metric_result = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred_arr,
        metric_protocol=metric_protocol,
        sales_scaler=sales_scaler,
        feature_columns=feature_columns,
        eps=eps,
    )

    return {
        "rmse": float(metric_result["rmse"]),
        "accuracy": float(metric_result["accuracy"]),
        "mae": float(metric_result.get("mae", float("nan"))),
        "mape": float(metric_result.get("mape", float("nan"))),
        "smape": float(metric_result.get("smape", float("nan"))),
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
        "normalized_rmse": float(metric_result.get("normalized_rmse", metric_result.get("rmse", float("nan")))),
        "normalized_accuracy": float(metric_result.get("normalized_accuracy", metric_result.get("accuracy", float("nan")))),
        "normalized_mae": float(metric_result.get("normalized_mae", metric_result.get("mae", float("nan")))),
        "normalized_mape": metric_result.get("normalized_mape"),
        "normalized_smape": metric_result.get("normalized_smape"),
        "original_scale_rmse": metric_result.get("original_scale_rmse"),
        "original_scale_accuracy": metric_result.get("original_scale_accuracy"),
        "original_scale_mae": metric_result.get("original_scale_mae"),
        "original_scale_mape": metric_result.get("original_scale_mape"),
        "original_scale_smape": metric_result.get("original_scale_smape"),
        "metric_space": str(metric_result.get("metric_space", metric_result.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(metric_result.get("metric_space_used", metric_result.get("metric_space", "normalized_minmax_space"))),
        "prediction_shape": tuple(y_pred_arr.shape),
        "metric_space_current": str(metric_result["metric_space_current"]),
        "metric_space_paper": str(metric_result["metric_space_paper"]),
        "paper_metric_aligned": bool(metric_result["paper_metric_aligned"]),
        "inverse_transform_applied": bool(metric_result["inverse_transform_applied"]),
        "inverse_transform_available": bool(metric_result.get("inverse_transform_available", False)),
        "metric_notes": str(metric_result["metric_notes"]),
    }


def run_single_source_tl_for_mssb(
    source_sequence_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: dict | None = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
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
    logger.info(
        "[run_single_source_tl_for_mssb] Start. hyperparams=%s. clipnorm=None means gradient clipping is disabled.",
        fixed_hyperparams_summary(),
    )

    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")
    _validate_feature_cols(target_train_df, feature_cols, where="target_train_df")
    _validate_feature_cols(target_val_df, feature_cols, where="target_val_df")
    _validate_feature_cols(target_test_df, feature_cols, where="target_test_df")

    src_train_df, src_val_df, src_test_df = _prepare_single_source_split(source_sequence_df)

    src_train_df = fill_source_numeric_na(src_train_df)
    src_val_df = fill_source_numeric_na(src_val_df)
    src_test_df = fill_source_numeric_na(src_test_df)
    src_train_df, src_val_df, src_test_df, _, _ = normalize_features(src_train_df, src_val_df, src_test_df)
    tgt_train_df, tgt_val_df, tgt_test_df, tgt_scaler, tgt_feature_columns = normalize_features(target_train_df, target_val_df, target_test_df)

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
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
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
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    y_val_pred = target_model.predict(x_tgt_val, verbose=0)
    y_test_pred = target_model.predict(x_tgt_test, verbose=0)

    val_eval = evaluate_predictions(
        y_true=y_tgt_val,
        y_pred=y_val_pred,
        metric_protocol=metric_protocol,
        sales_scaler=tgt_scaler,
        feature_columns=tgt_feature_columns,
    )
    test_eval = evaluate_predictions(
        y_true=y_tgt_test,
        y_pred=y_test_pred,
        metric_protocol=metric_protocol,
        sales_scaler=tgt_scaler,
        feature_columns=tgt_feature_columns,
    )

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
        "test_rmse": float(test_eval["rmse"]),
        "test_accuracy": float(test_eval["accuracy"]),
        "test_mae": float(test_eval.get("mae", float("nan"))),
        "test_mape": float(test_eval.get("mape", float("nan"))),
        "test_smape": float(test_eval.get("smape", float("nan"))),
        "test_rmse_current": float(test_eval.get("rmse_current", float("nan"))),
        "test_accuracy_current": float(test_eval.get("accuracy_current", float("nan"))),
        "test_mae_current": float(test_eval.get("mae_current", float("nan"))),
        "test_mape_current": float(test_eval.get("mape_current", float("nan"))),
        "test_smape_current": float(test_eval.get("smape_current", float("nan"))),
        "test_rmse_paper": float(test_eval.get("rmse_paper", float("nan"))),
        "test_accuracy_paper": float(test_eval.get("accuracy_paper", float("nan"))),
        "test_mae_paper": float(test_eval.get("mae_paper", float("nan"))),
        "test_mape_paper": float(test_eval.get("mape_paper", float("nan"))),
        "test_smape_paper": float(test_eval.get("smape_paper", float("nan"))),
        "y_val_pred": np.asarray(y_val_pred),
        "y_test_pred": np.asarray(y_test_pred),
        "y_val_true": np.asarray(y_tgt_val),
        "y_test_true": np.asarray(y_tgt_test),
        "prediction_shape": tuple(y_test_pred.shape),
        "metric_space": str(test_eval.get("metric_space", "normalized_minmax_space")),
        "metric_space_used": str(test_eval.get("metric_space_used", test_eval.get("metric_space", "normalized_minmax_space"))),
        "metric_space_current": str(test_eval.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(test_eval.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(test_eval.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(test_eval.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(test_eval.get("inverse_transform_available", False)),
        "metric_notes": str(test_eval.get("metric_notes", "")),
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
    target_df_for_selection: pd.DataFrame | None = None,
    k: int = 3,
    number_of_sources: int | None = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    include_sales_in_knn: bool = True,
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: dict | None = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: int | None = None,
    distance_jump_threshold: float = 0.5,
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
    effective_number_of_sources = int(k if number_of_sources is None else number_of_sources)
    logger.info(
        "[run_mssb_tl] Start. number_of_sources=%d weight_mode=%s",
        effective_number_of_sources,
        weight_mode,
    )
    logger.info(
        "[run_mssb_tl] Training hyperparameters: %s. clipnorm=None means gradient clipping is disabled.",
        fixed_hyperparams_summary(),
    )

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")

    selection_target_df = target_df if target_df_for_selection is None else target_df_for_selection
    _validate_feature_cols(selection_target_df, feature_cols, where="target_df_for_selection")

    selector = SourceSelector()
    selection_result = selector.select_top_k_sources(
        target_df=selection_target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=effective_number_of_sources,
        weight_mode=weight_mode,
        include_sales_in_knn=include_sales_in_knn,
        adaptive_source_selection=adaptive_source_selection,
        min_sources=min_sources,
        max_sources=max_sources,
        distance_jump_threshold=distance_jump_threshold,
    )
    selection_meta = selection_result.get("meta", {}) if isinstance(selection_result, dict) else {}

    selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
    if not selected_sources:
        raise ValueError("No source selected from source pool.")

    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)

    individual_results: List[Dict[str, object]] = []
    test_shape_reference: Optional[Tuple[int, ...]] = None

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
                metric_protocol=metric_protocol,
                early_stopping_enabled=early_stopping_enabled,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
            )
        except Exception as exc:
            raise RuntimeError(f"SS-TL failed for source_key={source_key}: {exc}") from exc

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
                "test_rmse": float(one_result["test_rmse"]),
                "test_accuracy": float(one_result["test_accuracy"]),
                "test_mae": float(one_result.get("test_mae", float("nan"))),
                "test_mape": float(one_result.get("test_mape", float("nan"))),
                "test_smape": float(one_result.get("test_smape", float("nan"))),
                "test_rmse_current": float(one_result.get("test_rmse_current", float("nan"))),
                "test_accuracy_current": float(one_result.get("test_accuracy_current", float("nan"))),
                "test_mae_current": float(one_result.get("test_mae_current", float("nan"))),
                "test_mape_current": float(one_result.get("test_mape_current", float("nan"))),
                "test_smape_current": float(one_result.get("test_smape_current", float("nan"))),
                "test_rmse_paper": float(one_result.get("test_rmse_paper", float("nan"))),
                "test_accuracy_paper": float(one_result.get("test_accuracy_paper", float("nan"))),
                "test_mae_paper": float(one_result.get("test_mae_paper", float("nan"))),
                "test_mape_paper": float(one_result.get("test_mape_paper", float("nan"))),
                "test_smape_paper": float(one_result.get("test_smape_paper", float("nan"))),
                "test_normalized_smape": one_result.get("normalized_smape"),
                "test_original_scale_smape": one_result.get("original_scale_smape"),
                "prediction_shape": pred_shape,
                "metric_space": str(one_result.get("metric_space", "normalized_minmax_space")),
                "metric_space_used": str(one_result.get("metric_space_used", one_result.get("metric_space", "normalized_minmax_space"))),
                "metric_space_current": str(one_result.get("metric_space_current", "normalized_minmax_space")),
                "metric_space_paper": str(one_result.get("metric_space_paper", "original_sales_space")),
                "paper_metric_aligned": bool(one_result.get("paper_metric_aligned", False)),
                "inverse_transform_applied": bool(one_result.get("inverse_transform_applied", False)),
                "inverse_transform_available": bool(one_result.get("inverse_transform_available", False)),
                "metric_notes": str(one_result.get("metric_notes", "")),
            }
        )

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
        "mae": float(best_source_result.get("test_mae", float("nan"))),
        "mape": float(best_source_result.get("test_mape", float("nan"))),
        "smape": float(best_source_result.get("test_smape", float("nan"))),
        "rmse_current": float(best_source_result.get("test_rmse_current", float("nan"))),
        "accuracy_current": float(best_source_result.get("test_accuracy_current", float("nan"))),
        "mae_current": float(best_source_result.get("test_mae_current", float("nan"))),
        "mape_current": float(best_source_result.get("test_mape_current", float("nan"))),
        "smape_current": float(best_source_result.get("test_smape_current", float("nan"))),
        "rmse_paper": float(best_source_result.get("test_rmse_paper", float("nan"))),
        "accuracy_paper": float(best_source_result.get("test_accuracy_paper", float("nan"))),
        "mae_paper": float(best_source_result.get("test_mae_paper", float("nan"))),
        "mape_paper": float(best_source_result.get("test_mape_paper", float("nan"))),
        "smape_paper": float(best_source_result.get("test_smape_paper", float("nan"))),
        "normalized_rmse": float(best_source_result.get("test_normalized_rmse", best_source_result.get("test_rmse", float("nan")))),
        "normalized_accuracy": float(best_source_result.get("test_normalized_accuracy", best_source_result.get("test_accuracy", float("nan")))),
        "normalized_mae": float(best_source_result.get("test_normalized_mae", best_source_result.get("test_mae", float("nan")))),
        "normalized_smape": best_source_result.get("test_normalized_smape"),
        "original_scale_rmse": best_source_result.get("test_original_scale_rmse", best_source_result.get("test_rmse_paper")),
        "original_scale_accuracy": best_source_result.get("test_original_scale_accuracy", best_source_result.get("test_accuracy_paper")),
        "original_scale_mae": best_source_result.get("test_original_scale_mae", best_source_result.get("test_mae_paper")),
        "original_scale_smape": best_source_result.get("test_original_scale_smape", best_source_result.get("test_smape_paper")),
        "prediction_shape": tuple(best_source_result["prediction_shape"]),
        "metric_space": str(best_source_result.get("metric_space", "normalized_minmax_space")),
        "metric_space_used": str(best_source_result.get("metric_space_used", best_source_result.get("metric_space", "normalized_minmax_space"))),
        "metric_space_current": str(best_source_result.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(best_source_result.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(best_source_result.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(best_source_result.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(best_source_result.get("inverse_transform_available", False)),
        "metric_notes": str(best_source_result.get("metric_notes", "")),
    }

    result = {
        "meta": {
            "method": "MSSB-TL",
            "k": int(effective_number_of_sources),
            "number_of_sources": int(effective_number_of_sources),
            "number_of_pretrained_models": int(len(individual_results)),
            "number_of_methods": 1,
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "selected_sources": selected_sources,
            "requested_k": int(selection_meta.get("requested_k", effective_number_of_sources)),
            "effective_k": int(selection_meta.get("effective_k", len(selected_sources))),
            "valid_source_count": int(selection_meta.get("valid_source_count", len(selected_sources))),
            "skipped_source_count": int(selection_meta.get("skipped_source_count", 0)),
            "date_alignment_mode": str(selection_meta.get("date_alignment_mode", "")),
            "date_alignment_diagnostics": selection_meta.get("date_alignment_diagnostics", {}),
        },
        "individual_results": individual_results,
        "best_source_result": best_source_result,
        "final_result": final_result,
    }

    logger.info(
        "[run_mssb_tl] Finished. final_rmse=%.4f final_accuracy=%.4f",
        float(final_result["rmse"]),
        float(final_result["accuracy"]),
    )
    return result
