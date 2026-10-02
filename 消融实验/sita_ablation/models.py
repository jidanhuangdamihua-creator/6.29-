"""CNN and TCN builders per INTERFACES §6 and tcn_frozen.json.

Builds networks only. No data loading, training loops, early stopping, or
BL7 fit hyper-parameters.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model

# Frozen source-model compile LR (INTERFACES §2 / §9). Not a BL7 target LR.
_SOURCE_CNN_LEARNING_RATE = 1e-4
_SOURCE_DROPOUT = 0.1
_TARGET_FEATURE_DROPOUT = 0.1

# INTERFACES §6.1 / protocol_frozen.json cnn.fusion
CNN_FUSIBLE_LAYERS: Tuple[str, ...] = ("conv1", "conv2", "conv3")

# tcn_frozen.json fused_tensors / INTERFACES §6.2
TCN_FUSIBLE_LAYERS: Tuple[str, ...] = (
    "blocks.0.conv1",
    "blocks.0.conv2",
    "blocks.0.residual",
    "blocks.1.conv1",
    "blocks.1.conv2",
)

TCN_CHANNELS = 16
TCN_LEVELS = 2
TCN_KERNEL_SIZE = 3
TCN_DILATIONS: Tuple[int, ...] = (1, 2)
TCN_INPUT_CHANNELS = 1

# INTERFACES §6.2 Keras shapes for fusible TCN layers.
TCN_FUSIBLE_SHAPES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "blocks.0.conv1": ((3, 1, 16), (16,)),
    "blocks.0.conv2": ((3, 16, 16), (16,)),
    "blocks.0.residual": ((1, 1, 16), (16,)),
    "blocks.1.conv1": ((3, 16, 16), (16,)),
    "blocks.1.conv2": ((3, 16, 16), (16,)),
}


def _cnn_backbone(inputs: tf.Tensor) -> tf.Tensor:
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)
    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)
    x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(x)
    return Flatten(name="flatten")(x)


def build_source_cnn(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Source CNN: conv stack, Dropout 0.1, dense_out. Adam+MSE, no clipnorm."""
    if len(input_shape) != 2:
        raise ValueError(f"CNN input_shape must be (time, C), got {input_shape!r}")
    inputs = Input(shape=input_shape, dtype=tf.float32, name="x")
    flat = _cnn_backbone(inputs)
    dropped = Dropout(rate=_SOURCE_DROPOUT, name="dropout")(flat)
    outputs = Dense(1, name="dense_out")(dropped)
    model = Model(inputs=inputs, outputs=outputs, name="source_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=_SOURCE_CNN_LEARNING_RATE,
            clipnorm=None,
        ),
        loss="mse",
    )
    return model


def build_target_cnn(input_shape: tuple[int, int]) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Target CNN: full head_target network and flatten extractor.

    Returns (full_network, flatten_extractor). Prediction is head_target(z_T).
    """
    if len(input_shape) != 2:
        raise ValueError(f"CNN input_shape must be (time, C), got {input_shape!r}")
    inputs = Input(shape=input_shape, dtype=tf.float32, name="x")
    flat = _cnn_backbone(inputs)
    dropped = Dropout(rate=_TARGET_FEATURE_DROPOUT, name="feature_dropout")(flat)
    outputs = Dense(1, name="head_target")(dropped)
    full = Model(inputs=inputs, outputs=outputs, name="target_cnn")
    extractor = Model(inputs=inputs, outputs=flat, name="target_cnn_flatten")
    return full, extractor


def _causal_conv1d(
    values: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dilation: int,
    name: str,
    activation=None,
) -> tf.Tensor:
    # tcn_frozen.json: left pad (kernel_size - 1) * dilation, then valid conv.
    left = (int(kernel_size) - 1) * int(dilation)
    padded = ZeroPadding1D(padding=(left, 0), name=f"{name}_pad")(values)
    return Conv1D(
        filters=int(filters),
        kernel_size=int(kernel_size),
        dilation_rate=int(dilation),
        padding="valid",
        activation=activation,
        name=name,
    )(padded)


def _tcn_block(
    values: tf.Tensor,
    *,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    level: int,
) -> tf.Tensor:
    in_channels = int(values.shape[-1])
    residual = values
    if in_channels != int(out_channels):
        residual = Conv1D(
            filters=int(out_channels),
            kernel_size=1,
            padding="valid",
            name=f"blocks.{level}.residual",
        )(values)
    output = _causal_conv1d(
        values,
        filters=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        name=f"blocks.{level}.conv1",
        activation="relu",
    )
    output = _causal_conv1d(
        output,
        filters=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        name=f"blocks.{level}.conv2",
        activation=None,
    )
    return Activation("relu", name=f"blocks.{level}.relu_out")(output + residual)


def build_tcn(input_shape: tuple[int, int] = (10, 1)) -> tf.keras.Model:
    """Causal TCN: 2 blocks, 16 channels, dilations 1/2, last-timestep Dense(1).

    Input is x_sales with shape (N, 10, 1). Not compiled here: Stage A / adaptation
    attach the loss and optimizer. Does not carry BL7 early stopping or fit LR.
    """
    if len(input_shape) != 2:
        raise ValueError(f"TCN input_shape must be (time, 1), got {input_shape!r}")
    if int(input_shape[-1]) != TCN_INPUT_CHANNELS:
        raise ValueError(
            f"TCN input channels must be {TCN_INPUT_CHANNELS}, got {input_shape[-1]}"
        )
    if len(TCN_DILATIONS) != TCN_LEVELS:
        raise ValueError("frozen TCN levels and dilations disagree")
    inputs = Input(shape=input_shape, dtype=tf.float32, name="x_sales")
    x = inputs
    for level, dilation in enumerate(TCN_DILATIONS):
        x = _tcn_block(
            x,
            out_channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dilation=int(dilation),
            level=level,
        )
    # Last timestep Linear(16, 1); keep Functional-graph compatible.
    last = tf.keras.layers.Lambda(lambda tensor: tensor[:, -1, :], name="last_timestep")(x)
    outputs = Dense(1, name="head")(last)
    return Model(inputs=inputs, outputs=outputs, name="tcn")


def fusible_layer_weights(model: tf.keras.Model, layer_names: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    """Return {name: [kernel, bias]} for the named Conv1D layers."""
    out: dict[str, list[np.ndarray]] = {}
    for name in layer_names:
        layer = model.get_layer(name)
        weights = layer.get_weights()
        if len(weights) != 2:
            raise ValueError(f"layer {name!r} must expose kernel and bias, got {len(weights)} tensors")
        out[name] = [np.asarray(weights[0]), np.asarray(weights[1])]
    return out


def cnn_fusible_weights(model: tf.keras.Model) -> dict[str, list[np.ndarray]]:
    return fusible_layer_weights(model, CNN_FUSIBLE_LAYERS)


def tcn_fusible_weights(model: tf.keras.Model) -> dict[str, list[np.ndarray]]:
    return fusible_layer_weights(model, TCN_FUSIBLE_LAYERS)


def expected_cnn_fusible_shapes(channels: int) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    c = int(channels)
    return {
        "conv1": ((3, c, 32), (32,)),
        "conv2": ((3, 32, 64), (64,)),
        "conv3": ((3, 64, 128), (128,)),
    }
