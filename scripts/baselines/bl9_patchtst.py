"""BL9: target-only PatchTST direct-horizon baseline.

Architecture follows the official supervised PatchTST backbone
(patching, channel independence, Transformer encoder, flatten head,
positional encoding, residual attention, model-native RevIN) with a
short-window patch config required by lookback=10.

Training, scaler, and direct-H arrays are owned by the existing BL4
learning-baseline contract. This module does not import SITA RevIN.
"""

from __future__ import annotations

import copy
import importlib
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


METHOD = "PatchTST"
METHOD_ID = "BL9_PatchTST"
FEATURE_COLUMNS = ("sales",)
TARGET_COLUMN = "sales"

PATCHTST_CONFIG = {
    "context_window": 10,
    "target_window": 1,
    "patch_len": 4,
    "stride": 2,
    "padding_patch": "end",
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "d_ff": 128,
    "dropout": 0.10,
    "attn_dropout": 0.00,
    "fc_dropout": 0.05,
    "head_dropout": 0.00,
    "head_type": "flatten",
    "individual": False,
    "revin": True,
    "affine": True,
    "subtract_last": False,
    "res_attention": True,
    "pre_norm": False,
}


def compute_patch_num(
    context_window: int = PATCHTST_CONFIG["context_window"],
    patch_len: int = PATCHTST_CONFIG["patch_len"],
    stride: int = PATCHTST_CONFIG["stride"],
    padding_patch: str = PATCHTST_CONFIG["padding_patch"],
) -> int:
    window = int(context_window)
    length = int(patch_len)
    step = int(stride)
    if window <= 0 or length <= 0 or step <= 0:
        raise ValueError("context_window, patch_len, and stride must be positive")
    if length > window:
        raise ValueError(
            f"patch_len={length} exceeds context_window={window}"
        )
    patch_num = int((window - length) / step + 1)
    if str(padding_patch) == "end":
        patch_num += 1
    if patch_num <= 0:
        raise ValueError("PatchTST produced a non-positive patch sequence")
    return patch_num


def assert_patchtst_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    checked = dict(PATCHTST_CONFIG if config is None else config)
    if int(checked["d_model"]) % int(checked["n_heads"]) != 0:
        raise ValueError(
            f"d_model={checked['d_model']} is not divisible by n_heads={checked['n_heads']}"
        )
    if int(checked["target_window"]) != 1:
        raise ValueError("BL9 target_window / pred_len must be 1")
    if str(checked["head_type"]) != "flatten":
        raise ValueError("BL9 head_type must remain flatten")
    if bool(checked["individual"]):
        raise ValueError("BL9 individual must remain False")
    checked["patch_num"] = compute_patch_num(
        int(checked["context_window"]),
        int(checked["patch_len"]),
        int(checked["stride"]),
        str(checked["padding_patch"]),
    )
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
            "BL9 requires PyTorch. Please install it manually with: pip install torch"
        ) from exc
    return torch, nn, functional


def _build_patchtst_modules(n_vars: int):
    torch, nn, functional = _load_torch()
    config = assert_patchtst_config()
    patch_num = int(config["patch_num"])
    d_model = int(config["d_model"])
    n_heads = int(config["n_heads"])
    d_ff = int(config["d_ff"])
    head_dim = d_model // n_heads

    class PatchTSTRevIN(nn.Module):
        """Official PatchTST RevIN, independent of the SITA RevIN adapter."""

        def __init__(self):
            super().__init__()
            self.num_features = int(n_vars)
            self.eps = 1e-5
            self.affine = bool(config["affine"])
            self.subtract_last = bool(config["subtract_last"])
            if self.affine:
                self.affine_weight = nn.Parameter(torch.ones(self.num_features))
                self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

        def _get_statistics(self, values):
            dim2reduce = tuple(range(1, values.ndim - 1))
            if self.subtract_last:
                self.last = values[:, -1, :].unsqueeze(1).detach()
            else:
                self.mean = torch.mean(values, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(values, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()

        def _normalize(self, values):
            if self.subtract_last:
                values = values - self.last
            else:
                values = values - self.mean
            values = values / self.stdev
            if self.affine:
                values = values * self.affine_weight + self.affine_bias
            return values

        def _denormalize(self, values):
            if self.affine:
                values = (values - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            values = values * self.stdev
            if self.subtract_last:
                values = values + self.last
            else:
                values = values + self.mean
            return values

        def forward(self, values, mode: str):
            if mode == "norm":
                self._get_statistics(values)
                return self._normalize(values)
            if mode == "denorm":
                return self._denormalize(values)
            raise ValueError(f"unsupported PatchTST RevIN mode: {mode!r}")

    class _ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.res_attention = bool(config["res_attention"])
            self.attn_dropout = nn.Dropout(float(config["attn_dropout"]))
            self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

        def forward(self, query, key, value, prev=None):
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            if prev is not None:
                attn_scores = attn_scores + prev
            attn_weights = functional.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, value)
            if self.res_attention:
                return output, attn_weights, attn_scores
            return output, attn_weights

    class _MultiheadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = head_dim
            self.res_attention = bool(config["res_attention"])
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.attention = _ScaledDotProductAttention()

        def _reshape(self, values):
            batch, seq_len, _ = values.shape
            return values.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        def forward(self, query, key, value, prev=None):
            query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
            query, key, value = self._reshape(query), self._reshape(key), self._reshape(value)
            if self.res_attention:
                output, attn_weights, attn_scores = self.attention(query, key, value, prev)
            else:
                output, attn_weights = self.attention(query, key, value)
                attn_scores = None
            output = output.transpose(1, 2).contiguous().view(query.shape[0], -1, d_model)
            output = self.out_proj(output)
            if self.res_attention:
                return output, attn_weights, attn_scores
            return output, attn_weights

    class TSTEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.res_attention = bool(config["res_attention"])
            self.pre_norm = bool(config["pre_norm"])
            self.self_attn = _MultiheadAttention()
            self.dropout_attn = nn.Dropout(float(config["dropout"]))
            self.norm_attn = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(float(config["dropout"])),
                nn.Linear(d_ff, d_model),
            )
            self.dropout_ff = nn.Dropout(float(config["dropout"]))
            self.norm_ff = nn.LayerNorm(d_model)

        def forward(self, src, prev=None):
            residual = src
            if self.pre_norm:
                src = self.norm_attn(src)
            if self.res_attention:
                src2, _, scores = self.self_attn(src, src, src, prev)
            else:
                src2, _ = self.self_attn(src, src, src)
                scores = None
            src = residual + self.dropout_attn(src2)
            if not self.pre_norm:
                src = self.norm_attn(src)
            residual = src
            if self.pre_norm:
                src = self.norm_ff(src)
            src = residual + self.dropout_ff(self.ff(src))
            if not self.pre_norm:
                src = self.norm_ff(src)
            if self.res_attention:
                return src, scores
            return src

    class TSTEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.res_attention = bool(config["res_attention"])
            self.layers = nn.ModuleList(
                [TSTEncoderLayer() for _ in range(int(config["n_layers"]))]
            )

        def forward(self, src):
            output = src
            scores = None
            for layer in self.layers:
                if self.res_attention:
                    output, scores = layer(output, prev=scores)
                else:
                    output = layer(output)
            return output

    class FlattenHead(nn.Module):
        def __init__(self):
            super().__init__()
            if str(config["head_type"]) != "flatten":
                raise ValueError("BL9 requires flatten head")
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_model * patch_num, int(config["target_window"]))
            self.dropout = nn.Dropout(float(config["head_dropout"]))

        def forward(self, values):
            return self.dropout(self.linear(self.flatten(values)))

    class PatchTSTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_vars = int(n_vars)
            self.patch_len = int(config["patch_len"])
            self.stride = int(config["stride"])
            self.patch_num = patch_num
            self.padding_patch = str(config["padding_patch"])
            self.revin_enabled = bool(config["revin"])
            if self.padding_patch == "end":
                self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.revin_layer = PatchTSTRevIN() if self.revin_enabled else nn.Identity()
            self.W_P = nn.Linear(self.patch_len, d_model)
            self.W_pos = nn.Parameter(torch.empty(patch_num, d_model))
            nn.init.uniform_(self.W_pos, -0.02, 0.02)
            self.dropout = nn.Dropout(float(config["dropout"]))
            self.encoder = TSTEncoder()
            self.head = FlattenHead()
            self.fc_dropout = nn.Dropout(float(config["fc_dropout"]))

        def forward(self, values):
            # values: [batch, seq_len, channels] -> official [batch, channels, seq_len]
            channels_first = values.permute(0, 2, 1)
            if self.revin_enabled:
                normalized = self.revin_layer(channels_first.permute(0, 2, 1), "norm")
                channels_first = normalized.permute(0, 2, 1)
            if self.padding_patch == "end":
                channels_first = self.padding_patch_layer(channels_first)
            patches = channels_first.unfold(
                dimension=-1, size=self.patch_len, step=self.stride
            )
            # [batch, channels, patch_num, patch_len]
            batch, n_vars, actual_patch_num, _ = patches.shape
            if actual_patch_num != self.patch_num:
                raise RuntimeError(
                    f"PatchTST patch_num mismatch: expected={self.patch_num} actual={actual_patch_num}"
                )
            encoded = self.W_P(patches)  # [B, C, patch_num, d_model]
            encoded = encoded.reshape(batch * n_vars, actual_patch_num, d_model)
            encoded = self.dropout(encoded + self.W_pos)
            encoded = self.encoder(encoded)
            encoded = encoded.reshape(batch, n_vars, actual_patch_num, d_model)
            encoded = encoded.permute(0, 1, 3, 2)
            forecast = self.fc_dropout(self.head(encoded))  # [B, C, target_window]
            if self.revin_enabled:
                forecast = self.revin_layer(forecast.permute(0, 2, 1), "denorm").permute(0, 2, 1)
            return forecast

    return PatchTSTBackbone(), config


@dataclass
class FittedBL9:
    horizon: int
    lookback: int
    scaler: object
    torch: object | None
    model: object | None
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    target_channel_index: int = 0
    patch_num: int = 5
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
            raise RuntimeError("BL9 fitted model is incomplete")
        normalized = self.scaler.transform(visible.reshape(-1, 1))
        model_input = self.torch.as_tensor(
            normalized.reshape(1, self.lookback, 1),
            dtype=self.torch.float32,
        )
        self.model.eval()
        with self.torch.no_grad():
            output = self.model(model_input)
        channel = int(self.target_channel_index)
        normalized_prediction = float(output[:, channel, -1].reshape(-1)[0].item())
        normalized_prediction = float(np.clip(normalized_prediction, 0.0, 1.0))
        prediction = float(self.scaler.inverse_transform([[normalized_prediction]])[0, 0])
        if not np.isfinite(prediction):
            raise ValueError("BL9 produced a non-finite prediction")
        return prediction


def fit_bl9(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = DEFAULT_LOOKBACK,
    seed: int = 42,
    feature_columns=FEATURE_COLUMNS,
) -> FittedBL9:
    """Fit one fixed-architecture target-only PatchTST direct-H model."""

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    if resolved_horizon <= 0 or resolved_lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    if resolved_lookback != int(PATCHTST_CONFIG["context_window"]):
        raise ValueError("BL9 context_window is frozen at 10")
    columns = tuple(str(column) for column in feature_columns)
    channel = target_channel_index(columns)
    config = assert_patchtst_config()

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
        return FittedBL9(
            resolved_horizon,
            resolved_lookback,
            scaler,
            None,
            None,
            columns,
            channel,
            int(config["patch_num"]),
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

    model, _ = _build_patchtst_modules(n_vars=1)
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
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_output = model(train_x_tensor)[:, channel, -1].unsqueeze(-1)
        if train_output.shape != train_y_tensor.shape:
            raise RuntimeError(
                f"BL9 train output shape {tuple(train_output.shape)} != {tuple(train_y_tensor.shape)}"
            )
        train_loss = criterion(train_output, train_y_tensor)
        if not torch.isfinite(train_loss):
            raise RuntimeError("BL9 training loss is not finite")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_output = model(validation_x_tensor)[:, channel, -1].unsqueeze(-1)
            validation_loss = float(criterion(validation_output, validation_y_tensor).item())
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
        raise RuntimeError("BL9 training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedBL9(
        resolved_horizon,
        resolved_lookback,
        scaler,
        torch,
        model,
        columns,
        channel,
        int(config["patch_num"]),
        None,
        tuple(history),
        best_epoch,
        epochs_completed,
        float(best_loss),
    )


def predict_bl9(fitted_model: FittedBL9, input_sales) -> float:
    if not isinstance(fitted_model, FittedBL9):
        raise TypeError("fitted_model must be returned by fit_bl9")
    return fitted_model.predict(input_sales)


def patchtst_forward_probe(batch_size: int = 2, seq_len: int = 10, n_vars: int = 1):
    """Return a detached forward pass used by smoke/unit tests."""

    torch, _, _ = _load_torch()
    model, config = _build_patchtst_modules(n_vars=n_vars)
    model.eval()
    dummy = torch.zeros((int(batch_size), int(seq_len), int(n_vars)), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy)
    return output, config
