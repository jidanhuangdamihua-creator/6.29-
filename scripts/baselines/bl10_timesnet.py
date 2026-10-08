"""BL10: target-only TimesNet direct-horizon baseline.

Keeps the official TimesNet blocks (FFT period discovery, 1D-to-2D
reshape, Inception-style 2D convolution, adaptive aggregation, residual
TimesBlock, LayerNorm) with pred_len frozen at 1 for this paper's
direct-H task.
"""

from __future__ import annotations

import copy
import importlib
import math
import random
from dataclasses import dataclass, field
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
except ImportError:  # pragma: no cover - direct script import compatibility
    from bl4_lstm import (
        DEFAULT_LOOKBACK,
        LEARNING_RATE,
        MAX_EPOCHS,
        PATIENCE,
        _direct_h_arrays,
        _finite_vector,
        _scale_partitions,
    )


METHOD = "TimesNet"
METHOD_ID = "BL10_TimesNet"
FEATURE_COLUMNS = ("sales",)
TARGET_COLUMN = "sales"

TIMESNET_CONFIG = {
    "seq_len": 10,
    "pred_len": 1,
    "e_layers": 2,
    "d_model": 32,
    "d_ff": 64,
    "top_k": 3,
    "num_kernels": 4,
    "dropout": 0.10,
}


def rfft_nonzero_frequency_bins(length: int) -> int:
    resolved = int(length)
    if resolved <= 0:
        raise ValueError("FFT length must be positive")
    return max(int(resolved // 2), 0)


def effective_top_k(seq_len: int, top_k: int, extra_len: int = 0) -> int:
    available = rfft_nonzero_frequency_bins(int(seq_len) + int(extra_len))
    return int(min(int(top_k), available))


def assert_timesnet_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    checked = dict(TIMESNET_CONFIG if config is None else config)
    if int(checked["pred_len"]) != 1:
        raise ValueError("BL10 pred_len must be 1 for direct-H forecasting")
    seq_len = int(checked["seq_len"])
    pred_len = int(checked["pred_len"])
    top_k = int(checked["top_k"])
    available_seq = rfft_nonzero_frequency_bins(seq_len)
    available_block = rfft_nonzero_frequency_bins(seq_len + pred_len)
    resolved = int(min(top_k, available_seq, available_block))
    if resolved <= 0:
        raise ValueError("TimesNet has no usable nonzero FFT bins")
    checked["available_nonzero_frequency_bins_seq"] = available_seq
    checked["available_nonzero_frequency_bins_block"] = available_block
    checked["effective_top_k"] = resolved
    return checked


def target_channel_index(feature_columns=FEATURE_COLUMNS) -> int:
    columns = [str(column) for column in feature_columns]
    if TARGET_COLUMN not in columns:
        raise ValueError(f"feature metadata must include {TARGET_COLUMN!r}: {columns}")
    return int(columns.index(TARGET_COLUMN))


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as exc:  # pragma: no cover - runtime dependency boundary
        raise ImportError(
            "BL10 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn, functional


def _build_timesnet_modules(n_vars: int, *, fft_length: int | None = None):
    torch, nn, functional = _load_torch()
    config = assert_timesnet_config()
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    d_model = int(config["d_model"])
    d_ff = int(config["d_ff"])
    top_k = int(config["effective_top_k"])
    num_kernels = int(config["num_kernels"])
    dropout = float(config["dropout"])
    block_len = seq_len + pred_len
    probe_length = int(fft_length) if fft_length is not None else block_len

    class PositionalEmbedding(nn.Module):
        def __init__(self, model_dim, max_len=5000):
            super().__init__()
            position = torch.arange(max_len).float().unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
            )
            pe = torch.zeros(max_len, model_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, values):
            return self.pe[:, : values.size(1)]

    class TokenEmbedding(nn.Module):
        def __init__(self, c_in, model_dim):
            super().__init__()
            self.tokenConv = nn.Conv1d(
                in_channels=c_in,
                out_channels=model_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            )
            nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

        def forward(self, values):
            return self.tokenConv(values.permute(0, 2, 1)).transpose(1, 2)

    class DataEmbedding(nn.Module):
        def __init__(self, c_in, model_dim, drop):
            super().__init__()
            self.value_embedding = TokenEmbedding(c_in, model_dim)
            self.position_embedding = PositionalEmbedding(model_dim)
            self.dropout = nn.Dropout(drop)

        def forward(self, values):
            return self.dropout(self.value_embedding(values) + self.position_embedding(values))

    class InceptionBlockV1(nn.Module):
        def __init__(self, in_channels, out_channels, kernels):
            super().__init__()
            self.kernels = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=2 * index + 1, padding=index)
                    for index in range(int(kernels))
                ]
            )
            for kernel in self.kernels:
                nn.init.kaiming_normal_(kernel.weight, mode="fan_in", nonlinearity="leaky_relu")
                if kernel.bias is not None:
                    nn.init.zeros_(kernel.bias)

        def forward(self, values):
            outputs = torch.stack([kernel(values) for kernel in self.kernels], dim=-1)
            return outputs.mean(-1)

    class TimesBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = top_k
            self.seq_len = block_len
            self.conv = nn.Sequential(
                InceptionBlockV1(d_model, d_ff, num_kernels),
                nn.GELU(),
                InceptionBlockV1(d_ff, d_model, num_kernels),
            )

        def _periods(self, values):
            xf = torch.fft.rfft(values, dim=1)
            frequency_list = torch.abs(xf).mean(0).mean(-1)
            frequency_list = frequency_list.clone()
            frequency_list[0] = 0.0
            available = int(frequency_list.numel() - 1)
            k = int(min(self.k, max(available, 1)))
            _, top_list = torch.topk(frequency_list, k)
            periods = []
            for index in top_list.detach().tolist():
                period = int(values.shape[1] // max(int(index), 1))
                periods.append(max(period, 1))
            return periods, torch.abs(xf).mean(-1)[:, top_list], k

        def forward(self, values):
            batch, time, channels = values.shape
            period_list, period_weight, k = self._periods(values)
            residuals = []
            for period in period_list:
                if time % period != 0:
                    length = ((time // period) + 1) * period
                    padding = values.new_zeros(batch, length - time, channels)
                    padded = torch.cat([values, padding], dim=1)
                else:
                    length = time
                    padded = values
                reshaped = (
                    padded.reshape(batch, length // period, period, channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                conv = self.conv(reshaped)
                conv = conv.permute(0, 2, 3, 1).reshape(batch, -1, channels)
                residuals.append(conv[:, :time, :])
            stacked = torch.stack(residuals, dim=-1)
            weights = functional.softmax(period_weight, dim=1)
            weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, time, channels, 1)
            aggregated = torch.sum(stacked * weights, dim=-1)
            return aggregated + values, k

    class TimesNetForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.n_vars = int(n_vars)
            self.requested_top_k = int(config["top_k"])
            self.effective_top_k = top_k
            self.enc_embedding = DataEmbedding(self.n_vars, d_model, dropout)
            self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
            self.blocks = nn.ModuleList([TimesBlock() for _ in range(int(config["e_layers"]))])
            self.layer_norm = nn.LayerNorm(d_model)
            self.projection = nn.Linear(d_model, self.n_vars)
            self.last_effective_top_k = top_k

        def forward(self, values):
            # values: [batch, seq_len, channels]
            means = values.mean(1, keepdim=True).detach()
            centered = values - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = centered / stdev
            encoded = self.enc_embedding(normalized)
            encoded = self.predict_linear(encoded.permute(0, 2, 1)).permute(0, 2, 1)
            observed_k = self.effective_top_k
            for block in self.blocks:
                encoded, observed_k = block(encoded)
                encoded = self.layer_norm(encoded)
            self.last_effective_top_k = int(observed_k)
            decoded = self.projection(encoded)
            decoded = decoded * stdev.repeat(1, self.seq_len + self.pred_len, 1)
            decoded = decoded + means.repeat(1, self.seq_len + self.pred_len, 1)
            return decoded[:, -self.pred_len :, :]

    model = TimesNetForecast()
    # Touch the probe length so tests can assert FFT bin arithmetic without a silent mismatch.
    if rfft_nonzero_frequency_bins(probe_length) < top_k:
        raise ValueError(
            f"TimesNet FFT bins at length={probe_length} cannot support top_k={top_k}"
        )
    return model, config


@dataclass
class FittedBL10:
    horizon: int
    lookback: int
    scaler: object
    torch: object | None
    model: object | None
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    target_channel_index: int = 0
    effective_top_k: int = 3
    constant_prediction: float | None = None
    training_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    best_epoch: int | None = None
    epochs_completed: int = 0
    best_val_loss: float | None = None

    def predict(self, input_sales) -> float:
        visible = _finite_vector(input_sales, name="input_sales")
        if visible.size != self.lookback:
            raise ValueError(
                f"input_sales must contain exactly {self.lookback} values, got {visible.size}"
            )
        if self.constant_prediction is not None:
            return float(self.constant_prediction)
        if self.torch is None or self.model is None:
            raise RuntimeError("BL10 fitted model is incomplete")
        normalized = self.scaler.transform(visible.reshape(-1, 1))
        model_input = self.torch.as_tensor(
            normalized.reshape(1, self.lookback, 1),
            dtype=self.torch.float32,
        )
        self.model.eval()
        with self.torch.no_grad():
            output = self.model(model_input)
        channel = int(self.target_channel_index)
        normalized_prediction = float(output[:, -1, channel].reshape(-1)[0].item())
        normalized_prediction = float(np.clip(normalized_prediction, 0.0, 1.0))
        prediction = float(self.scaler.inverse_transform([[normalized_prediction]])[0, 0])
        if not np.isfinite(prediction):
            raise ValueError("BL10 produced a non-finite prediction")
        return prediction


def fit_bl10(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
    seed: int = 42,
    feature_columns=FEATURE_COLUMNS,
) -> FittedBL10:
    """Fit one fixed-architecture target-only TimesNet direct-H model."""

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
        train_output = model(train_x_tensor)[:, -1, channel].unsqueeze(-1)
        if train_output.shape != train_y_tensor.shape:
            raise RuntimeError(
                f"BL10 train output shape {tuple(train_output.shape)} != {tuple(train_y_tensor.shape)}"
            )
        train_loss = criterion(train_output, train_y_tensor)
        if not torch.isfinite(train_loss):
            raise RuntimeError("BL10 training loss is not finite")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_output = model(validation_x_tensor)[:, -1, channel].unsqueeze(-1)
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
        raise RuntimeError("BL10 training did not produce a valid checkpoint")
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


def predict_bl10(fitted_model: FittedBL10, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL10):
        raise TypeError("fitted_model must be returned by fit_bl10")
    return fitted_model.predict(input_sales)


def timesnet_forward_probe(batch_size: int = 2, seq_len: int = 10, n_vars: int = 1):
    torch, _, _ = _load_torch()
    model, config = _build_timesnet_modules(n_vars=n_vars)
    model.eval()
    dummy = torch.zeros((int(batch_size), int(seq_len), int(n_vars)), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy)
    return output, config, int(getattr(model, "last_effective_top_k", config["effective_top_k"]))
