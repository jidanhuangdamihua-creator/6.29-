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

from cnn_model import build_base_cnn
from src.utils.runtime_control import keras_verbose

LOGGER_NAME = "experiment"


def _get_logger() -> logging.Logger:
    """获取项目统一日志器。"""
    return logging.getLogger(LOGGER_NAME)


def train_source_model(
    X_source: np.ndarray,
    y_source: np.ndarray,
    input_shape: Tuple[int, ...],
    learning_rate: float = 0.001,
    epochs: int = 3,
    batch_size: int = 16,
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

    Returns:
        tf.keras.Model: 训练好的 source 模型。
    """
    logger = _get_logger()
    logger.info("[train_source_model] Start. input_shape=%s epochs=%d batch_size=%d",
                input_shape, epochs, batch_size)

    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    model.fit(X_source, y_source, epochs=epochs, batch_size=batch_size, verbose=keras_verbose())

    logger.info("[train_source_model] Finished training on source data.")
    return model


def build_target_model_from_source(
    source_model,
    input_shape: Tuple[int, ...],
    learning_rate: float = 0.001,
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
    epochs: int = 3,
    batch_size: int = 16,
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

    Returns:
        tf.keras.Model: 微调后的 target 模型。
    """
    logger = _get_logger()
    logger.info("[fine_tune_target_model] Start. epochs=%d batch_size=%d", epochs, batch_size)

    fit_kwargs = dict(epochs=epochs, batch_size=batch_size, verbose=keras_verbose())
    if X_target_val is not None and y_target_val is not None:
        fit_kwargs["validation_data"] = (X_target_val, y_target_val)

    target_model.fit(X_target_train, y_target_train, **fit_kwargs)

    logger.info("[fine_tune_target_model] Fine-tuning completed.")
    return target_model


def evaluate_regression_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
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
    y_pred_flat = y_pred.flatten()

    rmse = float(np.sqrt(np.mean((y_pred_flat - y_test) ** 2)))

    eps = 1e-8
    accuracy = 1.0 / (rmse + eps)

    logger.info("[evaluate_regression_model] RMSE=%.4f Accuracy=%.4f", rmse, accuracy)
    return {
        "rmse": rmse,
        "accuracy": accuracy,
        "y_pred_shape": y_pred.shape,
    }
