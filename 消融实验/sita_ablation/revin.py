"""RevIN per frozen INTERFACES §6.3 and protocol_frozen.json revin."""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf

# Frozen: protocol_frozen.json revin / DECISIONS STOP-4.
_REVIN_EPS = 1e-5
_REVIN_AFFINE = True
_REVIN_TARGET_CHANNEL = 0


class RevIN(tf.keras.layers.Layer):
    """Instance normalize (N, 10, F); denormalize sales forecast (N, 1)."""

    def __init__(self, num_features: int, **kwargs):
        super().__init__(**kwargs)
        n_features = int(num_features)
        if n_features < 1:
            raise ValueError(f"RevIN num_features must be >= 1, got {num_features!r}")
        self.num_features = n_features
        self.eps = float(_REVIN_EPS)
        self.affine = bool(_REVIN_AFFINE)
        self.target_channel = int(_REVIN_TARGET_CHANNEL)
        self.affine_weight: Optional[tf.Variable] = None
        self.affine_bias: Optional[tf.Variable] = None

    def build(self, input_shape=None):
        if self.affine and self.affine_weight is None:
            # ones/zeros do not consume the TensorFlow RNG stream.
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

    def normalize(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Normalize along the time axis. Returns (x_norm, mean, stdev)."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if x.shape.rank != 3:
            raise ValueError(
                f"RevIN.normalize expects (N, 10, F), got rank={x.shape.rank}"
            )
        if not self.built:
            self.build(x.shape)
        mean = tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
        variance = tf.stop_gradient(
            tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
        )
        stdev = tf.sqrt(variance + self.eps)
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, mean, stdev

    def denormalize(
        self,
        y: tf.Tensor,
        mean: tf.Tensor,
        stdev: tf.Tensor,
    ) -> tf.Tensor:
        """Invert RevIN for channel-0 sales predictions of shape (N, 1)."""
        y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), (-1, 1))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        stdev = tf.convert_to_tensor(stdev, dtype=tf.float32)
        mean_t = tf.reshape(mean[:, 0, self.target_channel], (-1, 1))
        stdev_t = tf.reshape(stdev[:, 0, self.target_channel], (-1, 1))
        if self.affine:
            weight = self.affine_weight[self.target_channel]
            bias = self.affine_bias[self.target_channel]
            # denom = gamma + eps^2 = gamma + 1e-10
            y = (y - bias) / (weight + self.eps * self.eps)
        return y * stdev_t + mean_t

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features})
        return config


def build_revin(num_features: int) -> RevIN:
    """Build RevIN with frozen eps/affine/target_channel and materialize weights."""
    layer = RevIN(num_features=int(num_features))
    layer.build((None, 10, int(num_features)))
    return layer
