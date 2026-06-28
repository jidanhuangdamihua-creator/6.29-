"""Compatibility wrapper for the canonical CNN implementation.

All CNN model definitions are maintained in src/models/cnn_model.py.
This file re-exports them for backward compatibility with existing code
that imports from the root-level cnn_model module.

DO NOT add new code here. Edit src/models/cnn_model.py instead.
"""

from src.models.cnn_model import (
    CNN_ABLATION_VARIANTS,
    CnnAblationTrainingConfig,
    EarlyStoppingConfig,
    build_base_cnn,
    build_cnn_ablation_variant,
    create_early_stopping_callback,
    create_training_callbacks,
    get_model_summary_dict,
    resolve_cnn_ablation_training_config,
    set_trainable_layers,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_RESTORE_BEST,
)

__all__ = [
    "CNN_ABLATION_VARIANTS",
    "CnnAblationTrainingConfig",
    "EarlyStoppingConfig",
    "build_base_cnn",
    "build_cnn_ablation_variant",
    "create_early_stopping_callback",
    "create_training_callbacks",
    "get_model_summary_dict",
    "resolve_cnn_ablation_training_config",
    "set_trainable_layers",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_EARLY_STOPPING_MIN_DELTA",
    "DEFAULT_EARLY_STOPPING_RESTORE_BEST",
]
