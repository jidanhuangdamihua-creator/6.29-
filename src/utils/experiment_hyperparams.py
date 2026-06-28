"""Unified core hyperparameters for reproduction experiments."""

from __future__ import annotations

FIXED_LEARNING_RATE = 1e-4
FIXED_EPOCHS = 50
FIXED_CLIPNORM = None
FIXED_DROPOUT = 0.1

FORMAL_HYPERPARAMS = {
    "lr": 1e-4,
    "epochs": 50,
    "clipnorm": None,
    "dropout": 0.1,
}

DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
DEFAULT_EARLY_STOPPING_RESTORE_BEST = True

LR_LIST = [FIXED_LEARNING_RATE]
EPOCH_LIST = [FIXED_EPOCHS]
CLIPNORM_LIST = [FIXED_CLIPNORM]
DROPOUT_LIST = [FIXED_DROPOUT]


def fixed_hyperparams_dict() -> dict[str, object]:
    return {
        "learning_rate": FIXED_LEARNING_RATE,
        "epochs": FIXED_EPOCHS,
        "clipnorm": FIXED_CLIPNORM,
        "dropout": FIXED_DROPOUT,
    }


def fixed_hyperparams_summary() -> str:
    return "lr=1e-4, epochs=50, clipnorm=None, dropout=0.1"


def fixed_hyperparams_slug() -> str:
    return "lr-1e-4_epochs-50_clipnorm-None_dropout-0.1"
