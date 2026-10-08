"""PyTorch RevIN adapter that matches the official SITA TensorFlow contract.

The trainable SITA layer lives in ``src/transfer_methods/revin.py`` and cannot
own affine parameters inside a PyTorch autograd tape. This module implements
the same equations, epsilon, affine inverse, axes, and denormalize procedure.
Numerical parity against the TensorFlow layer is enforced by tests.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

DEFAULT_REVIN_EPS = 1e-5
DEFAULT_REVIN_AFFINE = True
DEFAULT_REVIN_TARGET_CHANNEL = 0
REVIN_CONTRACT = {
    "enabled": True,
    "affine": DEFAULT_REVIN_AFFINE,
    "eps": DEFAULT_REVIN_EPS,
    "target_channel": DEFAULT_REVIN_TARGET_CHANNEL,
    "include_mean": True,
    "normalization_axis": 1,
    "input_rank": 3,
    "statistics_source": "current_input_window_only",
}


def _load_torch():
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    return torch, nn


def official_revin_contract() -> dict[str, Any]:
    """Return the live SITA RevIN constants; used to prove this adapter did not drift."""

    official = importlib.import_module("src.transfer_methods.revin")
    return {
        "eps": float(official.DEFAULT_REVIN_EPS),
        "affine": bool(official.DEFAULT_REVIN_AFFINE),
        "target_channel": int(official.DEFAULT_REVIN_TARGET_CHANNEL),
    }


def assert_matches_official_revin_contract() -> dict[str, Any]:
    official = official_revin_contract()
    if float(DEFAULT_REVIN_EPS) != float(official["eps"]):
        raise AssertionError(f"RevIN eps drifted: adapter={DEFAULT_REVIN_EPS} official={official['eps']}")
    if bool(DEFAULT_REVIN_AFFINE) is not bool(official["affine"]):
        raise AssertionError("RevIN affine contract drifted")
    if int(DEFAULT_REVIN_TARGET_CHANNEL) != int(official["target_channel"]):
        raise AssertionError("RevIN target_channel contract drifted")
    return official


class TorchRevIN:
    """Factory for an instance-norm module over rank-3 `(batch, time, features)`."""

    def __new__(
        cls,
        num_features: int = 1,
        eps: float = DEFAULT_REVIN_EPS,
        affine: bool = DEFAULT_REVIN_AFFINE,
        target_channel: int = DEFAULT_REVIN_TARGET_CHANNEL,
    ):
        torch, nn = _load_torch()
        n_features = int(num_features)
        if n_features < 1:
            raise ValueError(f"RevIN num_features must be >= 1, got {num_features!r}")
        if float(eps) <= 0.0:
            raise ValueError(f"RevIN eps must be positive, got {eps!r}")
        channel = int(target_channel)
        if channel < 0 or channel >= n_features:
            raise ValueError(
                f"RevIN target_channel={target_channel!r} is outside [0, {n_features})"
            )

        class _RevIN(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_features = n_features
                self.eps = float(eps)
                self.affine = bool(affine)
                self.target_channel = channel
                if self.affine:
                    self.affine_weight = nn.Parameter(torch.ones(n_features))
                    self.affine_bias = nn.Parameter(torch.zeros(n_features))
                else:
                    self.register_parameter("affine_weight", None)
                    self.register_parameter("affine_bias", None)

            def _require_rank3(self, values):
                tensor = torch.as_tensor(values, dtype=torch.float32)
                if tensor.dim() != 3:
                    raise ValueError(
                        "RevIN.normalize expects rank-3 (batch, time, features), "
                        f"got rank={tensor.dim()}"
                    )
                return tensor

            def normalize(self, values):
                """Return `(x_norm, mean, stdev)` using only the provided input window."""

                x = self._require_rank3(values)
                mean = x.mean(dim=1, keepdim=True).detach()
                variance = ((x - mean) ** 2).mean(dim=1, keepdim=True).detach()
                stdev = torch.sqrt(variance + self.eps)
                x_norm = (x - mean) / stdev
                if self.affine:
                    x_norm = x_norm * self.affine_weight + self.affine_bias
                return x_norm, mean, stdev

            def denormalize(self, y_norm, mean, stdev, include_mean: bool = True):
                y = torch.as_tensor(y_norm, dtype=torch.float32).reshape(-1, 1)
                mean_t = torch.as_tensor(mean, dtype=torch.float32)[:, 0, self.target_channel].reshape(-1, 1)
                stdev_t = torch.as_tensor(stdev, dtype=torch.float32)[:, 0, self.target_channel].reshape(-1, 1)
                if self.affine:
                    weight = self.affine_weight[self.target_channel]
                    bias = self.affine_bias[self.target_channel]
                    y = (y - bias) / (weight + self.eps * self.eps)
                scaled = y * stdev_t
                if include_mean:
                    return scaled + mean_t
                return scaled

            def denormalize_sequence(self, x_norm, mean, stdev):
                x = torch.as_tensor(x_norm, dtype=torch.float32)
                mean_t = torch.as_tensor(mean, dtype=torch.float32)
                stdev_t = torch.as_tensor(stdev, dtype=torch.float32)
                if self.affine:
                    x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
                return x * stdev_t + mean_t

            def normalize_numpy(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                x_norm, mean, stdev = self.normalize(values)
                return (
                    np.asarray(x_norm.detach().cpu().numpy()),
                    np.asarray(mean.detach().cpu().numpy()),
                    np.asarray(stdev.detach().cpu().numpy()),
                )

            def denormalize_numpy(
                self,
                y_norm: np.ndarray,
                mean: np.ndarray,
                stdev: np.ndarray,
                include_mean: bool = True,
            ) -> np.ndarray:
                y = self.denormalize(y_norm, mean, stdev, include_mean=bool(include_mean))
                return np.asarray(y.detach().cpu().numpy())

            def config(self) -> dict[str, Any]:
                return {
                    "enabled": True,
                    "affine": bool(self.affine),
                    "eps": float(self.eps),
                    "target_channel": int(self.target_channel),
                    "num_features": int(self.num_features),
                    "include_mean": True,
                    "normalization_axis": 1,
                    "statistics_source": "current_input_window_only",
                }

        return _RevIN()
