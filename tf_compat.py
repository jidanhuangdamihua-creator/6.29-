"""
TensorFlow compatibility layer for Apple Silicon.

This module must be imported before any tensorflow/keras imports in project
entry scripts. When legacy Adam is usable, it redirects common Adam access
paths to that implementation to avoid the known slow v2.11+ optimizer path on
M1/M2 Macs. Keras 3 exposes a legacy namespace that raises at construction
time, so this module leaves the default Adam in place when legacy Adam is not
actually supported.
"""

import warnings
import tensorflow as tf


def _patch_adam_to_legacy() -> None:
    try:
        legacy_adam = tf.keras.optimizers.legacy.Adam
        legacy_adam(learning_rate=0.001)
    except (AttributeError, ImportError):
        warnings.warn(
            "tf.keras.optimizers.legacy.Adam is not supported in this "
            "TensorFlow/Keras environment; using tf.keras.optimizers.Adam.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    tf.keras.optimizers.Adam = legacy_adam

    try:
        from tensorflow.keras import optimizers as tensorflow_keras_optimizers
    except ImportError:
        tensorflow_keras_optimizers = None
    if tensorflow_keras_optimizers is not None:
        tensorflow_keras_optimizers.Adam = legacy_adam

    try:
        from keras import optimizers as keras_optimizers
    except ImportError:
        keras_optimizers = None
    if keras_optimizers is not None:
        keras_optimizers.Adam = legacy_adam


_patch_adam_to_legacy()


class _SlowAdamWarningFilter:
    def filter(self, record) -> bool:
        return "runs slowly on M1/M2" not in record.getMessage()


try:
    from absl import logging as absl_logging
except ImportError:
    absl_logging = None

if absl_logging is not None:
    absl_logging.get_absl_logger().addFilter(_SlowAdamWarningFilter())
