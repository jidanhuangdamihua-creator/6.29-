"""
Module 3: CNN Base Model Module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils.experiment_hyperparams import (
    FIXED_CLIPNORM,
    FIXED_DROPOUT,
    FIXED_LEARNING_RATE,
)


# 早停默认配置
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
DEFAULT_EARLY_STOPPING_RESTORE_BEST = True


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """早停配置数据类。"""
    enabled: bool = True
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = DEFAULT_EARLY_STOPPING_RESTORE_BEST
    verbose: int = 0


def create_early_stopping_callback(config: EarlyStoppingConfig | None = None) -> EarlyStopping:
    """
    创建 EarlyStopping callback。

    Args:
        config: 早停配置，如果为 None 使用默认配置

    Returns:
        EarlyStopping callback 实例
    """
    if config is None:
        config = EarlyStoppingConfig()

    return EarlyStopping(
        monitor=config.monitor,
        patience=config.patience,
        min_delta=config.min_delta,
        mode=config.mode,
        restore_best_weights=config.restore_best_weights,
        verbose=config.verbose,
    )


def create_training_callbacks(
    early_stopping_config: EarlyStoppingConfig | None = None,
    use_reduce_lr: bool = False,
) -> List[Callback]:
    """
    创建训练回调列表。

    Args:
        early_stopping_config: 早停配置
        use_reduce_lr: 是否使用学习率衰减

    Returns:
        回调列表
    """
    callbacks: List[Callback] = []

    if early_stopping_config is not None and early_stopping_config.enabled:
        callbacks.append(create_early_stopping_callback(early_stopping_config))

    if use_reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            )
        )

    return callbacks

CNN_ABLATION_VARIANTS = [
    "original",
    "change1_batch_size_1",
    "change2_no_batch_norm",
    "change3_low_lr_clipnorm",
    "change123_all",
]


@dataclass(frozen=True)
class CnnAblationTrainingConfig:
    cnn_ablation_variant: str
    change1_batch_size_1_enabled: bool
    change2_no_batch_norm_enabled: bool
    change3_low_lr_clipnorm_enabled: bool
    batch_norm_enabled: bool
    cnn_normalization: str
    original_batch_size: int
    effective_batch_size: int
    learning_rate: float
    clipnorm: float | None
    optimizer_name: str
    model_name: str = "current_3layer_cnn"


def _adam_optimizer(learning_rate=FIXED_LEARNING_RATE, clipnorm=FIXED_CLIPNORM):
    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm
    return tf.keras.optimizers.Adam(**optimizer_kwargs)


def build_base_cnn(input_shape, learning_rate=FIXED_LEARNING_RATE, dropout=FIXED_DROPOUT, clipnorm=FIXED_CLIPNORM):
    """
    Build and compile a base 1D-CNN regression model.

    Args:
        input_shape (tuple): Shape of a single input sample, e.g. (10, 7).
        learning_rate (float): Learning rate for the Adam optimizer.
        dropout (float): Dropout rate before the output layer.
        clipnorm: Optional Adam gradient clipping norm. None disables clipping.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)

    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)

    x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(x)

    x = Flatten(name="flatten")(x)
    x = Dropout(rate=float(dropout), name="dropout")(x)

    outputs = Dense(1, name="dense_out")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=_adam_optimizer(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _validate_cnn_ablation_variant(cnn_ablation_variant: str) -> str:
    variant = str(cnn_ablation_variant or "original")
    if variant not in CNN_ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown cnn_ablation_variant={variant!r}. "
            f"Expected one of: {', '.join(CNN_ABLATION_VARIANTS)}"
        )
    return variant


def _build_current_3layer_cnn_no_batch_norm(input_shape, learning_rate=FIXED_LEARNING_RATE, clipnorm=FIXED_CLIPNORM, dropout=FIXED_DROPOUT):
    """Audit-only copy of the current 3-layer CNN with no BatchNormalization."""
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)

    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)

    x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(x)

    x = Flatten(name="flatten")(x)
    x = Dropout(rate=float(dropout), name="dropout")(x)
    outputs = Dense(1, name="dense_out")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=_adam_optimizer(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="mse",
        metrics=["mae"],
    )
    return model


def resolve_cnn_ablation_training_config(
    cnn_ablation_variant: str = "original",
    original_batch_size: int = 16,
    original_learning_rate: float = FIXED_LEARNING_RATE,
) -> CnnAblationTrainingConfig:
    """Resolve audit-only CNN ablation switches under the fixed core optimizer config."""
    variant = _validate_cnn_ablation_variant(cnn_ablation_variant)
    change1_enabled = variant in {"change1_batch_size_1", "change123_all"}
    change2_enabled = variant in {"change2_no_batch_norm", "change123_all"}
    change3_enabled = variant in {"change3_low_lr_clipnorm", "change123_all"}

    effective_batch_size = 1 if change1_enabled else int(original_batch_size)
    learning_rate = float(original_learning_rate)
    clipnorm = FIXED_CLIPNORM

    # The preserved current 3-layer CNN in this repository has no BN layers.
    original_batch_norm_enabled = False
    batch_norm_enabled = False if change2_enabled else original_batch_norm_enabled
    cnn_normalization = "batch_norm" if batch_norm_enabled else "none"

    return CnnAblationTrainingConfig(
        cnn_ablation_variant=variant,
        change1_batch_size_1_enabled=change1_enabled,
        change2_no_batch_norm_enabled=change2_enabled,
        change3_low_lr_clipnorm_enabled=change3_enabled,
        batch_norm_enabled=batch_norm_enabled,
        cnn_normalization=cnn_normalization,
        original_batch_size=int(original_batch_size),
        effective_batch_size=int(effective_batch_size),
        learning_rate=float(learning_rate),
        clipnorm=clipnorm,
        optimizer_name="Adam",
    )


def build_cnn_ablation_variant(
    input_shape,
    learning_rate=FIXED_LEARNING_RATE,
    cnn_ablation_variant: str = "original",
):
    """Build an audit-only CNN variant under fixed lr/dropout and no clipnorm."""
    meta = resolve_cnn_ablation_training_config(
        cnn_ablation_variant=cnn_ablation_variant,
        original_learning_rate=learning_rate,
    )
    if meta.cnn_ablation_variant in {"original", "change1_batch_size_1"}:
        return build_base_cnn(input_shape=input_shape, learning_rate=learning_rate)
    if meta.cnn_ablation_variant == "change2_no_batch_norm":
        return _build_current_3layer_cnn_no_batch_norm(
            input_shape=input_shape,
            learning_rate=learning_rate,
            clipnorm=FIXED_CLIPNORM,
            dropout=FIXED_DROPOUT,
        )
    if meta.cnn_ablation_variant in {"change3_low_lr_clipnorm", "change123_all"}:
        builder = build_base_cnn if meta.cnn_ablation_variant == "change3_low_lr_clipnorm" else _build_current_3layer_cnn_no_batch_norm
        if builder is build_base_cnn:
            return builder(input_shape=input_shape, learning_rate=meta.learning_rate, clipnorm=meta.clipnorm)
        return builder(input_shape=input_shape, learning_rate=meta.learning_rate, clipnorm=meta.clipnorm, dropout=FIXED_DROPOUT)
    raise ValueError(f"Unhandled cnn_ablation_variant={meta.cnn_ablation_variant!r}")


def set_trainable_layers(model, trainable_layer_names):
    """
    Set the trainability of model layers by name.

    Args:
        model (tf.keras.Model): The Keras model to modify.
        trainable_layer_names (list[str]): Names of layers that should be trainable.
            All other layers will be frozen.
    """
    for layer in model.layers:
        layer.trainable = layer.name in trainable_layer_names


def get_model_summary_dict(model):
    """
    Return a summary dictionary for the given model.

    Args:
        model (tf.keras.Model): A compiled Keras model.

    Returns:
        dict: Contains input_shape, output_shape, total_params,
              trainable_params, and layer_names.
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    return {
        "input_shape": tuple(model.input_shape),
        "output_shape": tuple(model.output_shape),
        "total_params": total_params,
        "trainable_params": int(trainable_params),
        "layer_names": [layer.name for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)],
    }
