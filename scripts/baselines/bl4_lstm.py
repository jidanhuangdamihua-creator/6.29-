"""BL4: target-only direct-horizon LSTM baseline."""

from __future__ import annotations

import copy
import importlib
import random
from dataclasses import dataclass

import numpy as np

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "BL4 requires scikit-learn. Please install it manually with: pip install scikit-learn"
    ) from exc


DEFAULT_LOOKBACK = 10
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4


def _finite_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scale_partitions(train_sales, validation_sales):
    """Fit the scaler on caller-owned train rows and transform validation only."""

    train = _finite_vector(train_sales, name="train_sales")
    validation = _finite_vector(validation_sales, name="validation_sales")
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
    validation_scaled = scaler.transform(validation.reshape(-1, 1)).reshape(-1)
    return train_scaled, validation_scaled, scaler


def _direct_h_arrays(
    scaled: np.ndarray,
    *,
    horizon: int,
    lookback: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X[t-lookback+1:t] -> y[t+horizon] without recursive labels."""

    values = _finite_vector(scaled, name=name)
    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    if resolved_horizon <= 0:
        raise ValueError("horizon must be positive")
    if resolved_lookback <= 0:
        raise ValueError("lookback must be positive")
    required = resolved_lookback + resolved_horizon
    if values.size < required:
        raise ValueError(
            f"{name} requires at least {required} values for "
            f"lookback={resolved_lookback} horizon={resolved_horizon}, got {values.size}"
        )

    sample_count = values.size - required + 1
    x = np.stack(
        [values[index : index + resolved_lookback] for index in range(sample_count)],
        axis=0,
    )[..., np.newaxis]
    y = np.asarray(
        [values[index + resolved_lookback + resolved_horizon - 1] for index in range(sample_count)],
        dtype=np.float32,
    )[:, np.newaxis]
    return x.astype(np.float32), y.astype(np.float32)


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:
        raise ImportError(
            "BL4 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class FittedBL4:
    """A fitted direct-H model; scientific split ownership remains with the caller."""

    horizon: int
    lookback: int
    scaler: MinMaxScaler
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
        if self.torch is None or self.model is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("BL4 fitted model is incomplete")

        normalized = self.scaler.transform(visible.reshape(-1, 1))
        model_input = self.torch.as_tensor(
            normalized.reshape(1, self.lookback, 1),
            dtype=self.torch.float32,
        )
        self.model.eval()
        with self.torch.no_grad():
            normalized_prediction = float(self.model(model_input).item())
        normalized_prediction = float(np.clip(normalized_prediction, 0.0, 1.0))
        prediction = float(
            self.scaler.inverse_transform([[normalized_prediction]])[0, 0]
        )
        if not np.isfinite(prediction):
            raise ValueError("BL4 produced a non-finite prediction")
        return prediction


def fit_bl4(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
    hidden: int = 32,
    layers: int = 2,
    seed: int = 42,
) -> FittedBL4:
    """Fit one caller-specified direct-H LSTM without owning the 15/15 split."""

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    hidden_size = int(hidden)
    layer_count = int(layers)
    if resolved_horizon <= 0:
        raise ValueError("horizon must be positive")
    if resolved_lookback <= 0:
        raise ValueError("lookback must be positive")
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

    class SalesLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=layer_count,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, inputs):
            sequence, _ = self.lstm(inputs)
            return self.output(sequence[:, -1, :])

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)

    model = SalesLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_prediction = model(train_x_tensor)
        train_loss = criterion(train_prediction, train_y_tensor)
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
        raise RuntimeError("BL4 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL4(
        resolved_horizon,
        resolved_lookback,
        scaler,
        torch,
        model,
    )


def predict_bl4(fitted_model: FittedBL4, input_sales) -> float:
    """Predict one direct-H label from the legal values visible at one origin."""

    if not isinstance(fitted_model, FittedBL4):
        raise TypeError("fitted_model must be returned by fit_bl4")
    return fitted_model.predict(input_sales)
