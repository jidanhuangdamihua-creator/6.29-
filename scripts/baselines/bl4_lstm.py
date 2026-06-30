"""BL4: target-only recursive autoregressive LSTM baseline."""

from __future__ import annotations

import copy
import importlib
import random

import numpy as np

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "BL4 requires scikit-learn. Please install it manually with: pip install scikit-learn"
    ) from exc


WINDOW_SIZE = 10
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4


def _finite_vector(values, *, name: str, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scale_observed_sales(train_sales, val_sales):
    """Fit MinMaxScaler on all 30 observed values and return one-dimensional data."""
    train = _finite_vector(train_sales, name="train_sales", expected_size=25)
    val = _finite_vector(val_sales, name="val_sales", expected_size=5)
    observed = np.concatenate([train, val])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaled = scaler.fit_transform(observed.reshape(-1, 1)).reshape(-1)
    return observed, scaled, scaler


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:
        raise ImportError(
            "BL4 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn


def _training_arrays(scaled: np.ndarray) -> tuple[np.ndarray, ...]:
    train = scaled[:25]
    train_x = np.stack(
        [train[index : index + WINDOW_SIZE] for index in range(15)],
        axis=0,
    )[..., np.newaxis]
    train_y = np.asarray(
        [train[index + WINDOW_SIZE] for index in range(15)],
        dtype=np.float32,
    )[:, np.newaxis]

    val_x = np.stack(
        [scaled[index - WINDOW_SIZE : index] for index in range(25, 30)],
        axis=0,
    )[..., np.newaxis]
    val_y = scaled[25:30, np.newaxis]
    return (
        train_x.astype(np.float32),
        train_y.astype(np.float32),
        val_x.astype(np.float32),
        val_y.astype(np.float32),
    )


def predict_bl4(train_sales, val_sales, test_len, hidden=32, layers=2, seed=42):
    """Train an LSTM and recursively forecast in the 30-day MinMax space."""
    horizon = int(test_len)
    hidden_size = int(hidden)
    layer_count = int(layers)
    if horizon <= 0:
        raise ValueError("test_len must be positive")
    if hidden_size <= 0 or layer_count <= 0:
        raise ValueError("hidden and layers must be positive")

    observed, scaled, scaler = _scale_observed_sales(train_sales, val_sales)
    if float(np.ptp(observed)) <= 1e-12:
        return np.full(horizon, float(observed[0]), dtype=float)

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

    train_x, train_y, val_x, val_y = _training_arrays(scaled)
    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    val_x_tensor = torch.as_tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.as_tensor(val_y, dtype=torch.float32)

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
            validation_loss = float(criterion(model(val_x_tensor), val_y_tensor).item())
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

    raw_window = observed[-WINDOW_SIZE:].astype(float).tolist()
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            normalized_window = scaler.transform(
                np.asarray(raw_window[-WINDOW_SIZE:], dtype=float).reshape(-1, 1)
            )
            model_input = torch.as_tensor(
                normalized_window.reshape(1, WINDOW_SIZE, 1),
                dtype=torch.float32,
            )
            normalized_prediction = float(model(model_input).item())
            normalized_prediction = float(np.clip(normalized_prediction, 0.0, 1.0))
            raw_prediction = float(
                scaler.inverse_transform([[normalized_prediction]])[0, 0]
            )
            predictions.append(raw_prediction)
            raw_window.append(raw_prediction)

    result = np.asarray(predictions, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("BL4 produced non-finite predictions")
    return result
