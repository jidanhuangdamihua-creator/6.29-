"""BL10 TimesNet + SITA-contract affine RevIN.

Reuses the formal TimesNet blocks from ``bl10_timesnet.py``. The parameter-free
window mean/std inside ``TimesNetForecast.forward`` is replaced by SITA RevIN
(affine γ, β; sales-channel denormalize). The two normalizers are not stacked.
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
        _direct_h_arrays,
        _finite_vector,
        _scale_partitions,
    )
    from .bl10_timesnet import (
        FEATURE_COLUMNS,
        FittedBL10,
        METHOD,
        TARGET_COLUMN,
        TIMESNET_CONFIG,
        _build_timesnet_modules,
        _load_torch,
        assert_timesnet_config,
        predict_bl10,
        target_channel_index,
    )
except ImportError:  # pragma: no cover
    from bl4_lstm import (
        DEFAULT_LOOKBACK,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        _direct_h_arrays,
        _finite_vector,
        _scale_partitions,
    )
    from bl10_timesnet import (
        FEATURE_COLUMNS,
        FittedBL10,
        METHOD,
        TARGET_COLUMN,
        TIMESNET_CONFIG,
        _build_timesnet_modules,
        _load_torch,
        assert_timesnet_config,
        predict_bl10,
        target_channel_index,
    )

from src.transfer_methods.revin_torch import TorchRevIN, REVIN_CONTRACT

PARENT_METHOD = "BL10_TimesNet"
PAPER_METHOD = "TimesNet_RevIN"
METHOD_ID = "BL10_TimesNet_RevIN"


def timesnet_revin_config() -> dict[str, Any]:
    return {
        "method": METHOD_ID,
        "paper_method": PAPER_METHOD,
        "parent_method": PARENT_METHOD,
        "parent_method_name": METHOD,
        "revin": dict(REVIN_CONTRACT),
        "timesnet": dict(TIMESNET_CONFIG),
        "replaced_internal_standardization": True,
        "formal_baseline_config_identity": {
            "parent_module": "scripts/baselines/bl10_timesnet.py",
            "lookback": DEFAULT_LOOKBACK,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "MSE",
        },
    }


def _attach_sita_revin(model, channel: int):
    """Replace TimesNet's mean/std wrap with SITA affine RevIN; keep the rest."""

    revin = TorchRevIN(num_features=1, target_channel=0)
    model.add_module("revin", revin)

    def forward(self, values):
        x_norm, mean, stdev = self.revin.normalize(values)
        encoded = self.enc_embedding(x_norm)
        encoded = self.predict_linear(encoded.permute(0, 2, 1)).permute(0, 2, 1)
        observed_k = self.effective_top_k
        for block in self.blocks:
            encoded, observed_k = block(encoded)
            encoded = self.layer_norm(encoded)
        self.last_effective_top_k = int(observed_k)
        decoded = self.projection(encoded)
        y_norm = decoded[:, -self.pred_len :, int(channel)]
        y = self.revin.denormalize(y_norm, mean, stdev)
        batch = int(values.shape[0])
        return y.reshape(batch, self.pred_len, 1)

    model.forward = forward.__get__(model, type(model))
    return model


def fit_bl10_revin(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
    seed: int = 42,
    feature_columns=FEATURE_COLUMNS,
) -> FittedBL10:
    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    if resolved_horizon <= 0 or resolved_lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    if resolved_lookback != int(TIMESNET_CONFIG["seq_len"]):
        raise ValueError("BL10 seq_len is frozen at 10")
    columns = tuple(str(column) for column in feature_columns)
    channel = target_channel_index(columns)
    config = assert_timesnet_config()

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
        return FittedBL10(
            resolved_horizon,
            resolved_lookback,
            scaler,
            None,
            None,
            columns,
            channel,
            int(config["effective_top_k"]),
            float(train[0]),
            tuple(),
            None,
            0,
            None,
        )

    torch, nn, _ = _load_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model, _ = _build_timesnet_modules(n_vars=1)
    model = _attach_sita_revin(model, channel)
    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []
    epochs_completed = 0
    last_k = int(config["effective_top_k"])
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_output = model(train_x_tensor)[:, -1, 0].unsqueeze(-1)
        if train_output.shape != train_y_tensor.shape:
            raise RuntimeError(
                f"BL10 RevIN train output shape {tuple(train_output.shape)} != {tuple(train_y_tensor.shape)}"
            )
        train_loss = criterion(train_output, train_y_tensor)
        if not torch.isfinite(train_loss):
            raise RuntimeError("BL10 RevIN training loss is not finite")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_output = model(validation_x_tensor)[:, -1, 0].unsqueeze(-1)
            validation_loss = float(criterion(validation_output, validation_y_tensor).item())
        last_k = int(getattr(model, "last_effective_top_k", last_k))
        improved = validation_loss < best_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": validation_loss,
                "is_best": bool(improved),
            }
        )
        epochs_completed = epoch
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("BL10 RevIN training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL10(
        resolved_horizon,
        resolved_lookback,
        scaler,
        torch,
        model,
        columns,
        channel,
        int(last_k),
        None,
        tuple(history),
        best_epoch,
        epochs_completed,
        float(best_loss),
    )


def predict_bl10_revin(fitted_model: FittedBL10, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL10):
        raise TypeError("fitted_model must be returned by fit_bl10_revin")
    return predict_bl10(fitted_model, input_sales)
