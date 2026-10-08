"""BL7: fixed, causal target-only TCN direct-horizon baseline."""

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


TCN_CHANNELS = 16
TCN_LEVELS = 2
TCN_KERNEL_SIZE = 3
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        raise ImportError(
            "BL7 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn


class CausalConv1d:
    """Factory namespace for a left-padded causal convolution."""


@dataclass(frozen=True)
class _CausalSpec:
    convolution: object
    dilation: int
    kernel_size: int


class TCNResidualBlock:
    """Residual block whose convolutions only read current and past positions."""

    causal = True

    def __new__(cls, in_channels, out_channels, kernel_size, dilation):
        torch, nn = _load_torch()

        class _Block(nn.Module):
            causal = True

            def __init__(self):
                super().__init__()
                self.kernel_size = int(kernel_size)
                self.dilation = int(dilation)
                self.conv1 = nn.Conv1d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    padding=0,
                )
                self.conv2 = nn.Conv1d(
                    int(out_channels),
                    int(out_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    padding=0,
                )
                self.activation = nn.ReLU()
                self.residual = (
                    nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
                    if int(in_channels) != int(out_channels)
                    else nn.Identity()
                )

            def _causal(self, conv, values):
                left = (self.kernel_size - 1) * self.dilation
                padded = torch.nn.functional.pad(values, (left, 0))
                return conv(padded)

            def forward(self, values):
                residual = self.residual(values)
                output = self.activation(self._causal(self.conv1, values))
                output = self._causal(self.conv2, output)
                return self.activation(output + residual)

        return _Block()


@dataclass(frozen=True)
class FittedBL7:
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
            raise RuntimeError("BL7 fitted model is incomplete")
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
            raise ValueError("BL7 produced a non-finite prediction")
        return prediction


def fit_bl7(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = 10,
    channels: int = TCN_CHANNELS,
    levels: int = TCN_LEVELS,
    kernel_size: int = TCN_KERNEL_SIZE,
    seed: int = 42,
) -> FittedBL7:
    """Fit one fixed-architecture causal TCN direct-H model."""

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    if resolved_horizon <= 0 or resolved_lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    if (int(channels), int(levels), int(kernel_size)) != (
        TCN_CHANNELS,
        TCN_LEVELS,
        TCN_KERNEL_SIZE,
    ):
        raise ValueError(
            "BL7 architecture is fixed at channels=16, levels=2, kernel_size=3"
        )

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
        return FittedBL7(
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

    class TCNNet(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = 1
            for level in range(TCN_LEVELS):
                blocks.append(
                    TCNResidualBlock(
                        in_channels,
                        TCN_CHANNELS,
                        TCN_KERNEL_SIZE,
                        2**level,
                    )
                )
                in_channels = TCN_CHANNELS
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Linear(TCN_CHANNELS, 1)

        def forward(self, inputs):
            encoded = self.blocks(inputs.transpose(1, 2))
            return self.head(encoded[:, :, -1])

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    model = TCNNet()
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
        raise RuntimeError("BL7 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL7(resolved_horizon, resolved_lookback, scaler, torch, model)


def predict_bl7(fitted_model: FittedBL7, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL7):
        raise TypeError("fitted_model must be returned by fit_bl7")
    return fitted_model.predict(input_sales)
