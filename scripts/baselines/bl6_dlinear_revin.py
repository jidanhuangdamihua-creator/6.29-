"""BL6 DLinear + SITA-contract RevIN robustness wrapper.

The formal DLinear implementation in ``bl6_dlinear.py`` is imported, not
rewritten. The only conceptual addition is RevIN around the unchanged
backbone / scaler / optimizer / early-stopping recipe.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

try:
    from .bl4_lstm import _direct_h_arrays, _finite_vector, _scale_partitions
    from .bl6_dlinear import (
        DLINEAR_KERNEL_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        FittedBL6,
        _load_torch,
        _moving_average,
        fit_bl6,
        predict_bl6,
    )
except ImportError:  # pragma: no cover - direct script import compatibility
    from bl4_lstm import _direct_h_arrays, _finite_vector, _scale_partitions
    from bl6_dlinear import (
        DLINEAR_KERNEL_SIZE,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        FittedBL6,
        _load_torch,
        _moving_average,
        fit_bl6,
        predict_bl6,
    )

from src.transfer_methods.revin_torch import TorchRevIN, REVIN_CONTRACT

PARENT_METHOD = "BL6_DLinear"
PAPER_METHOD = "DLinear_RevIN"
METHOD_ID = "BL6_DLinear_RevIN"
ROBUSTNESS_VARIANT = "RevIN"


def dlinear_revin_config(*, revin_enabled: bool = True) -> dict[str, Any]:
    return {
        "method": METHOD_ID,
        "paper_method": PAPER_METHOD,
        "parent_method": PARENT_METHOD,
        "robustness_variant": ROBUSTNESS_VARIANT,
        "revin_enabled": bool(revin_enabled),
        "revin": dict(REVIN_CONTRACT) if revin_enabled else {"enabled": False},
        "formal_baseline_config_identity": {
            "parent_module": "scripts/baselines/bl6_dlinear.py",
            "kernel_size": DLINEAR_KERNEL_SIZE,
            "lookback": 10,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "MSE",
        },
    }


def fit_bl6_revin(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = 10,
    kernel_size: int = DLINEAR_KERNEL_SIZE,
    seed: int = 42,
    revin_enabled: bool = True,
) -> FittedBL6:
    """Fit formal DLinear, optionally wrapping the backbone with RevIN."""

    if not revin_enabled:
        return fit_bl6(
            train_sales,
            validation_sales,
            horizon=horizon,
            lookback=lookback,
            kernel_size=kernel_size,
            seed=seed,
        )

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    if resolved_horizon <= 0 or resolved_lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    if int(kernel_size) != DLINEAR_KERNEL_SIZE:
        raise ValueError(f"BL6 kernel_size is fixed at {DLINEAR_KERNEL_SIZE}")

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
        return FittedBL6(
            resolved_horizon,
            resolved_lookback,
            scaler,
            None,
            None,
            float(train[0]),
        )

    torch, nn, functional = _load_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    class DLinearRevINNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.revin = TorchRevIN(num_features=1)
            self.seasonal = nn.Linear(resolved_lookback, 1)
            self.trend = nn.Linear(resolved_lookback, 1)

        def forward(self, inputs):
            x_norm, mean, stdev = self.revin.normalize(inputs)
            trend = _moving_average(
                x_norm,
                kernel_size=DLINEAR_KERNEL_SIZE,
                functional=functional,
            )
            seasonal = x_norm - trend
            seasonal_output = self.seasonal(seasonal.transpose(1, 2)).squeeze(1)
            trend_output = self.trend(trend.transpose(1, 2)).squeeze(1)
            y_norm = (seasonal_output + trend_output).reshape(-1, 1)
            return self.revin.denormalize(y_norm, mean, stdev)

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    model = DLinearRevINNet()
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
        raise RuntimeError("BL6 RevIN training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL6(resolved_horizon, resolved_lookback, scaler, torch, model)


def predict_bl6_revin(fitted_model: FittedBL6, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL6):
        raise TypeError("fitted_model must be returned by fit_bl6_revin or fit_bl6")
    return predict_bl6(fitted_model, input_sales)
