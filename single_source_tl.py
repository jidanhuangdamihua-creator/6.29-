"""
Module 4: Single-Source Transfer Learning (SS-TL)

本模块实现单源迁移学习流程：
1. 在 source 数据上训练 CNN 基础模型
2. 将 source 模型权重迁移到 target 模型
3. 冻结前 N 层后在 target 数据上微调
4. 评估回归模型性能
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from cnn_model import (
    build_base_cnn,
    EarlyStoppingConfig,
    create_training_callbacks,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
)
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.experiment_hyperparams import FIXED_EPOCHS, FIXED_LEARNING_RATE, fixed_hyperparams_summary
from src.utils.runtime_control import keras_verbose

LOGGER_NAME = "experiment"

# 导出早停默认值供其他模块使用
__all__ = [
    "train_source_model",
    "build_target_model_from_source",
    "fine_tune_target_model",
    "evaluate_regression_model",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_EARLY_STOPPING_MIN_DELTA",
]


def _get_logger() -> logging.Logger:
    """获取项目统一日志器。"""
    return logging.getLogger(LOGGER_NAME)


def train_source_model(
    X_source: np.ndarray,
    y_source: np.ndarray,
    input_shape: Tuple[int, ...],
    learning_rate: float = FIXED_LEARNING_RATE,
    epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
):
    """
    在 source 数据上训练 CNN 基础模型。

    Args:
        X_source: Source 输入数据，形状 (samples, window_size, num_features)。
        y_source: Source 标签，形状 (samples,)。
        input_shape: 单个样本的形状，例如 (10, 7)。
        learning_rate: Adam 优化器学习率。
        epochs: 训练轮数。
        batch_size: 批大小。
        X_val: 可选的验证输入数据。
        y_val: 可选的验证标签。
        early_stopping_enabled: 是否启用早停。
        early_stopping_patience: 早停耐心值。
        early_stopping_min_delta: 早停最小改进阈值。

    Returns:
        tf.keras.Model: 训练好的 source 模型。
        或者 dict: 包含模型和早停信息（如果需要详细返回）。
    """
    logger = _get_logger()
    logger.info(
        "[train_source_model] Start. input_shape=%s batch_size=%d early_stopping=%s hyperparams=%s. clipnorm=None means gradient clipping is disabled.",
        input_shape, batch_size, early_stopping_enabled, fixed_hyperparams_summary(),
    )

    model = build_base_cnn(input_shape, learning_rate=learning_rate)

    # 配置早停回调
    callbacks = []
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        if early_stopping_enabled:
            early_stopping_config = EarlyStoppingConfig(
                enabled=True,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                monitor="val_loss",
                restore_best_weights=True,
            )
            callbacks = create_training_callbacks(early_stopping_config=early_stopping_config)

    history = model.fit(
        X_source, y_source,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None,
        verbose=keras_verbose(),
    )

    # 检查是否早停
    early_stopped = False
    stopped_epoch = epochs
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, "stopped_epoch") and cb.stopped_epoch > 0:
                early_stopped = True
                stopped_epoch = cb.stopped_epoch + 1
                break

    logger.info(
        "[train_source_model] Finished training on source data. early_stopped=%s stopped_epoch=%d",
        early_stopped, stopped_epoch,
    )
    return model


def build_target_model_from_source(
    source_model,
    input_shape: Tuple[int, ...],
    learning_rate: float = FIXED_LEARNING_RATE,
    freeze_first_n_layers: int = 4,
):
    """
    基于 source 模型构建 target 模型：复制权重并冻结前 N 层。

    Args:
        source_model: 训练好的 source Keras 模型。
        input_shape: 单个样本的形状，需与 source 一致。
        learning_rate: Adam 优化器学习率。
        freeze_first_n_layers: 冻结的前 N 层数量（从第一层开始计数，
            不含 InputLayer）。

    Returns:
        tf.keras.Model: 编译好的 target 模型，前 N 层已冻结。
    """
    import tensorflow as tf

    logger = _get_logger()
    logger.info("[build_target_model_from_source] Start. freeze_first_n_layers=%d",
                freeze_first_n_layers)

    # 新建一个结构完全相同的 target 模型
    target_model = build_base_cnn(input_shape, learning_rate=learning_rate)

    # 逐层复制权重
    for src_layer, tgt_layer in zip(source_model.layers, target_model.layers):
        weights = src_layer.get_weights()
        if weights:
            tgt_layer.set_weights(weights)

    # 冻结前 freeze_first_n_layers 层（跳过 InputLayer）
    non_input_layers = [l for l in target_model.layers
                        if not l.__class__.__name__ == "InputLayer"]
    frozen_names = []
    for i, layer in enumerate(non_input_layers):
        if i < freeze_first_n_layers:
            layer.trainable = False
            frozen_names.append(layer.name)
        else:
            layer.trainable = True

    logger.info("[build_target_model_from_source] Frozen layers: %s", frozen_names)

    # 重新编译（冻结后必须重新编译才能生效）
    target_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info("[build_target_model_from_source] Target model compiled.")
    return target_model, frozen_names


def fine_tune_target_model(
    target_model,
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_val: Optional[np.ndarray] = None,
    y_target_val: Optional[np.ndarray] = None,
    epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
):
    """
    在 target 训练集上微调已迁移的 target 模型。

    Args:
        target_model: 已迁移权重并冻结部分层的 target 模型。
        X_target_train: Target 训练输入。
        y_target_train: Target 训练标签。
        X_target_val: Target 验证输入（可选）。
        y_target_val: Target 验证标签（可选）。
        epochs: 微调轮数。
        batch_size: 批大小。
        early_stopping_enabled: 是否启用早停。
        early_stopping_patience: 早停耐心值。
        early_stopping_min_delta: 早停最小改进阈值。

    Returns:
        tf.keras.Model: 微调后的 target 模型。
    """
    logger = _get_logger()
    logger.info(
        "[fine_tune_target_model] Start. batch_size=%d early_stopping=%s hyperparams=%s. clipnorm=None means gradient clipping is disabled.",
        batch_size, early_stopping_enabled, fixed_hyperparams_summary(),
    )

    # 配置早停回调
    callbacks = []
    validation_data = None
    if X_target_val is not None and y_target_val is not None:
        validation_data = (X_target_val, y_target_val)
        if early_stopping_enabled:
            early_stopping_config = EarlyStoppingConfig(
                enabled=True,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                monitor="val_loss",
                restore_best_weights=True,
            )
            callbacks = create_training_callbacks(early_stopping_config=early_stopping_config)

    history = target_model.fit(
        X_target_train, y_target_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None,
        verbose=keras_verbose(),
    )

    # 检查是否早停
    early_stopped = False
    stopped_epoch = epochs
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, "stopped_epoch") and cb.stopped_epoch > 0:
                early_stopped = True
                stopped_epoch = cb.stopped_epoch + 1
                break

    logger.info(
        "[fine_tune_target_model] Fine-tuning completed. early_stopped=%s stopped_epoch=%d",
        early_stopped, stopped_epoch,
    )
    return target_model


def evaluate_regression_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_protocol: dict | None = None,
    sales_scaler: object | None = None,
    feature_columns: object | None = None,
) -> Dict[str, object]:
    """
    评估回归模型：计算 RMSE 和 Accuracy = 1/RMSE。

    Args:
        model: 训练或微调后的 Keras 模型。
        X_test: 测试输入。
        y_test: 测试标签。

    Returns:
        dict: 包含 rmse、accuracy、y_pred_shape。
    """
    logger = _get_logger()
    logger.info("[evaluate_regression_model] Start. test_samples=%d", len(X_test))

    y_pred = model.predict(X_test, verbose=0)
    metric_result = compute_metrics_with_protocol(
        y_true=y_test,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=sales_scaler,
        feature_columns=feature_columns,
    )

    logger.info(
        "[evaluate_regression_model] RMSE=%.4f Accuracy=%.4f current_space=%s paper_space=%s",
        float(metric_result["rmse"]),
        float(metric_result["accuracy"]),
        str(metric_result["metric_space_current"]),
        str(metric_result["metric_space_paper"]),
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
        "metric_space_current": str(metric_result["metric_space_current"]),
        "metric_space_paper": str(metric_result["metric_space_paper"]),
        "paper_metric_aligned": bool(metric_result["paper_metric_aligned"]),
        "inverse_transform_applied": bool(metric_result["inverse_transform_applied"]),
        "inverse_transform_available": bool(metric_result.get("inverse_transform_available", False)),
        "metric_notes": str(metric_result["metric_notes"]),
        "y_pred_shape": y_pred.shape,
    }
