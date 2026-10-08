"""BL4 LSTM + SITA-contract RevIN robustness wrapper.

The formal LSTM in ``bl4_lstm.py`` is imported, not rewritten. The only
addition is affine RevIN after MinMax and before the LSTM, with sales-channel
denormalization on the scalar forecast.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

try:
    from .bl4_lstm import (
        DEFAULT_LOOKBACK,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        FittedBL4,
        _direct_h_arrays,
        _finite_vector,
        _load_torch,
        _scale_partitions,
        fit_bl4,
        predict_bl4,
    )
except ImportError:  # pragma: no cover - direct script import compatibility
    from bl4_lstm import (
        DEFAULT_LOOKBACK,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        FittedBL4,
        _direct_h_arrays,
        _finite_vector,
        _load_torch,
        _scale_partitions,
        fit_bl4,
        predict_bl4,
    )

from src.transfer_methods.revin_torch import TorchRevIN, REVIN_CONTRACT

PARENT_METHOD = "BL4_LSTM"
PAPER_METHOD = "LSTM_RevIN"
METHOD_ID = "BL4_LSTM_RevIN"
ROBUSTNESS_VARIANT = "RevIN"


def lstm_revin_config(*, revin_enabled: bool = True) -> dict[str, Any]:
    return {
        "method": METHOD_ID,
        "paper_method": PAPER_METHOD,
        "parent_method": PARENT_METHOD,
        "robustness_variant": ROBUSTNESS_VARIANT,
        "revin_enabled": bool(revin_enabled),
        "revin": dict(REVIN_CONTRACT) if revin_enabled else {"enabled": False},
        "formal_baseline_config_identity": {
            "parent_module": "scripts/baselines/bl4_lstm.py",
            "lookback": DEFAULT_LOOKBACK,
            "hidden": 32,
            "layers": 2,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "MSE",
        },
    }


def fit_bl4_revin(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
    hidden: int = 32,
    layers: int = 2,
    seed: int = 42,
    revin_enabled: bool = True,
) -> FittedBL4:
    """Fit formal LSTM, optionally wrapping the backbone with SITA RevIN."""

    if not revin_enabled:
        return fit_bl4(
            train_sales,
            validation_sales,
            horizon=horizon,
            lookback=lookback,
            hidden=hidden,
            layers=layers,
            seed=seed,
        )

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    hidden_size = int(hidden)
    layer_count = int(layers)
    if resolved_horizon <= 0 or resolved_lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    if hidden_size <= 0 or layer_count <= 0:
        raise ValueError("hidden and layers must be positive")

    train = _finite_vector(train_sales, name="train_sales")
    validation = _finite_vector(validation_sales, name="validation_sales")
    train_scaled, validation_scaled, scaler = _scale_partitions(train, validation)
    train_x, train_y = _direct_h_arrays(
        train_scaled,
        horizon=resolved_horizon,
        lookback=resolved_lookback,
        name="train_sales",
    )
    validation_x, validation_y = _direct_h_arrays(
        validation_scaled,
        horizon=resolved_horizon,
        lookback=resolved_lookback,
        name="validation_sales",
    )
    if float(np.ptp(train)) <= 1e-12:
        return FittedBL4(
            resolved_horizon,
            resolved_lookback,
            scaler,
            None,
            None,
            float(train[0]),
        )

    torch, nn = _load_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    class LSTMRevINNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.revin = TorchRevIN(num_features=1)
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=layer_count,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, inputs):
            x_norm, mean, stdev = self.revin.normalize(inputs)
            sequence, _ = self.lstm(x_norm)
            y_norm = self.output(sequence[:, -1, :])
            return self.revin.denormalize(y_norm, mean, stdev)

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    model = LSTMRevINNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(train_x_tensor), train_y_tensor)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(model(validation_x_tensor), validation_y_tensor).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("BL4 RevIN training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL4(resolved_horizon, resolved_lookback, scaler, torch, model)


def predict_bl4_revin(fitted_model: FittedBL4, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL4):
        raise TypeError("fitted_model must be returned by fit_bl4_revin or fit_bl4")
    return predict_bl4(fitted_model, input_sales)
