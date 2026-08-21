"""Reversible Instance Normalization (Kim et al., ICLR 2022).

Port of ts-kim/RevIN (~60 lines). Applied per input window after the existing
MinMax scaler: subtract instance mean, divide by instance std, optionally apply
a per-channel affine map, then reverse the transform on the target-channel
forecast.

Statistics are detached (stop_gradient) so gradients do not flow through
mean/std. Zero-variance windows are protected by `eps`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

DEFAULT_REVIN_EPS = 1e-5
DEFAULT_REVIN_AFFINE = True
DEFAULT_REVIN_TARGET_CHANNEL = 0


class RevIN(tf.keras.layers.Layer):
    """RevIN wrap: normalize (batch, time, features), denormalize (batch, 1)."""

    def __init__(
        self,
        num_features: int,
        eps: float = DEFAULT_REVIN_EPS,
        affine: bool = DEFAULT_REVIN_AFFINE,
        target_channel: int = DEFAULT_REVIN_TARGET_CHANNEL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_features = int(num_features)
        if n_features < 1:
            raise ValueError(f"RevIN num_features must be >= 1, got {num_features!r}")
        if float(eps) <= 0.0:
            raise ValueError(f"RevIN eps must be positive, got {eps!r}")
        channel = int(target_channel)
        if channel < 0 or channel >= n_features:
            raise ValueError(
                f"RevIN target_channel={target_channel!r} is outside [0, {n_features})"
            )
        self.num_features = n_features
        self.eps = float(eps)
        self.affine = bool(affine)
        self.target_channel = channel
        self.affine_weight: Optional[tf.Variable] = None
        self.affine_bias: Optional[tf.Variable] = None

    def build(self, input_shape):
        if self.affine and self.affine_weight is None:
            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=(self.num_features,),
                initializer="ones",
                trainable=True,
            )
            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(self.num_features,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _require_rank3(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        rank = x.shape.rank
        if rank != 3:
            raise ValueError(
                f"RevIN.normalize expects rank-3 (batch, time, features), got rank={rank}"
            )
        return x

    def normalize(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Instance-normalize along the time axis. Returns (x_norm, mean, stdev)."""
        x = self._require_rank3(x)
        if not self.built:
            self.build(x.shape)
        mean = tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
        variance = tf.stop_gradient(tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True))
        stdev = tf.sqrt(variance + self.eps)
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, mean, stdev

    def denormalize_sequence(
        self,
        x_norm: tf.Tensor,
        mean: tf.Tensor,
        stdev: tf.Tensor,
    ) -> tf.Tensor:
        """Invert normalize() for a full (batch, time, features) tensor."""
        x_norm = tf.convert_to_tensor(x_norm, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        stdev = tf.convert_to_tensor(stdev, dtype=tf.float32)
        x = x_norm
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * stdev + mean

    def denormalize(
        self,
        y_norm: tf.Tensor,
        mean: tf.Tensor,
        stdev: tf.Tensor,
        include_mean: bool = True,
    ) -> tf.Tensor:
        """Invert RevIN for a target-channel forecast of shape (batch, 1).

        Default ``include_mean=True`` restores location+scale (V3 sales path):
        affine_inv(y_norm) * stdev + mean.

        ``include_mean=False`` is the residual-aware scale-only inverse:
        affine_inv(y_norm) * stdev. Location is not added back because a
        residual target is already a difference.
        """
        y = tf.reshape(tf.convert_to_tensor(y_norm, dtype=tf.float32), (-1, 1))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        stdev = tf.convert_to_tensor(stdev, dtype=tf.float32)
        mean_t = tf.reshape(mean[:, 0, self.target_channel], (-1, 1))
        stdev_t = tf.reshape(stdev[:, 0, self.target_channel], (-1, 1))
        if self.affine:
            weight = self.affine_weight[self.target_channel]
            bias = self.affine_bias[self.target_channel]
            y = (y - bias) / (weight + self.eps * self.eps)
        scaled = y * stdev_t
        if include_mean:
            return scaled + mean_t
        return scaled

    def normalize_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_norm, mean, stdev = self.normalize(tf.constant(x, dtype=tf.float32))
        return x_norm.numpy(), mean.numpy(), stdev.numpy()

    def denormalize_numpy(
        self,
        y_norm: np.ndarray,
        mean: np.ndarray,
        stdev: np.ndarray,
        include_mean: bool = True,
    ) -> np.ndarray:
        y = self.denormalize(
            tf.constant(np.asarray(y_norm, dtype=np.float32)),
            tf.constant(mean, dtype=tf.float32),
            tf.constant(stdev, dtype=tf.float32),
            include_mean=bool(include_mean),
        )
        return y.numpy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "eps": self.eps,
                "affine": self.affine,
                "target_channel": self.target_channel,
            }
        )
        return config
