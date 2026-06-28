"""No-TL baseline model definition.

The architecture is intentionally aligned with the SS-TL backbone CNN to keep
model capacity comparable in fairness-sensitive comparisons.
"""

from __future__ import annotations

from src.models.cnn_model import build_base_cnn


def build_no_tl_cnn_model(input_shape, learning_rate: float = 0.001):
    """Build No-TL CNN model with the same backbone as SS-TL.

    Args:
        input_shape: Input sample shape, e.g. (window_size, num_features).
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    return build_base_cnn(input_shape=input_shape, learning_rate=learning_rate)
