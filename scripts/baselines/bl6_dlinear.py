"""BL6: fixed, target-only DLinear direct-horizon baseline."""

from __future__ import annotations

import copy
import importlib
import random
from dataclasses import dataclass

import numpy as np

try:
    from .bl4_lstm import _direct_h_arrays, _finite_vector, _scale_partitions
except ImportError:  # pragma: no cover - direct script import compatibility
    from bl4_lstm import _direct_h_arrays, _finite_vector, _scale_partitions


DLINEAR_KERNEL_SIZE = 3
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        raise ImportError(
            "BL6 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn, functional


def _moving_average(values, *, kernel_size: int, functional):
    padding = (kernel_size - 1) // 2
    channels_first = values.transpose(1, 2)
    padded = functional.pad(channels_first, (padding, padding), mode="replicate")
    return functional.avg_pool1d(
        padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        count_include_pad=False,
    ).transpose(1, 2)


@dataclass(frozen=True)
class FittedBL6:
    horizon: int
    lookback: int
    scaler: object
    torch: object | None
    model: object | None
    constant_prediction: float | None = None

    def predict(self, input_sales) -> float:
        visible = _finite_vector(input_sales, name="input_sales")
        if visible.size != self.lookback:
            raise ValueError(
                f"input_sales must contain exactly {self.lookback} values, got {visible.size}"
            )
        if self.constant_prediction is not None:
            return float(self.constant_prediction)
        if self.torch is None or self.model is None:
            raise RuntimeError("BL6 fitted model is incomplete")
        normalized = self.scaler.transform(visible.reshape(-1, 1))
        model_input = self.torch.as_tensor(
            normalized.reshape(1, self.lookback, 1), dtype=self.torch.float32
        )
        self.model.eval()
        with self.torch.no_grad():
            normalized_prediction = float(self.model(model_input).item())
        normalized_prediction = float(np.clip(normalized_prediction, 0.0, 1.0))
        prediction = float(self.scaler.inverse_transform([[normalized_prediction]])[0, 0])
        if not np.isfinite(prediction):
            raise ValueError("BL6 produced a non-finite prediction")
        return prediction


def fit_bl6(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = 10,
    kernel_size: int = DLINEAR_KERNEL_SIZE,
    seed: int = 42,
) -> FittedBL6:
    """Fit one fixed-architecture target-only DLinear direct-H model."""

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

    class DLinearNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.seasonal = nn.Linear(resolved_lookback, 1)
            self.trend = nn.Linear(resolved_lookback, 1)

        def forward(self, inputs):
            trend = _moving_average(
                inputs,
                kernel_size=DLINEAR_KERNEL_SIZE,
                functional=functional,
            )
            seasonal = inputs - trend
            seasonal_output = self.seasonal(seasonal.transpose(1, 2)).squeeze(1)
            trend_output = self.trend(trend.transpose(1, 2)).squeeze(1)
            return (seasonal_output + trend_output).reshape(-1, 1)

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    model = DLinearNet()
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
        raise RuntimeError("BL6 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL6(resolved_horizon, resolved_lookback, scaler, torch, model)


def predict_bl6(fitted_model: FittedBL6, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL6):
        raise TypeError("fitted_model must be returned by fit_bl6")
    return fitted_model.predict(input_sales)
