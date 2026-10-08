"""SITA-TCN backbone-generalization primitives.

Reuses the formal BL7 TCN+RevIN architecture. The only intended experimental
variable is backbone initialization (random vs source-informed). This module
does not retune BL7 hyperparameters and does not write paper freeze artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.baselines.bl4_lstm import _direct_h_arrays, _finite_vector, _scale_partitions
from scripts.baselines.bl7_tcn import (
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_LEVELS,
    FittedBL7,
    TCNResidualBlock,
    _load_torch,
    predict_bl7,
)
from src.source_selection.source_selector import SourceSelector
from src.transfer_methods.revin_torch import TorchRevIN, REVIN_CONTRACT

METHOD_NAME = "SITA-TCN"
BL7_METHOD_NAME = "TCN_RevIN"
BL7_METHOD_ID = "BL7_TCN_RevIN"
WEIGHT_EPS = 1e-8
WEIGHT_SUM_TOLERANCE = 1e-6
MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR = 1e-6
FORWARD_FINITE_ATOL = 1e-12

EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS = (
    "blocks.0.conv1.bias",
    "blocks.0.conv1.weight",
    "blocks.0.conv2.bias",
    "blocks.0.conv2.weight",
    "blocks.0.residual.bias",
    "blocks.0.residual.weight",
    "blocks.1.conv1.bias",
    "blocks.1.conv1.weight",
    "blocks.1.conv2.bias",
    "blocks.1.conv2.weight",
)
EXPECTED_TARGET_ONLY_KEYS = (
    "head.bias",
    "head.weight",
    "revin.affine_bias",
    "revin.affine_weight",
)
HEAD_KEY_PREFIXES = ("head.",)
REVIN_KEY_PREFIXES = ("revin.",)


class SITATCNContractError(RuntimeError):
    """Fail-closed SITA-TCN protocol error."""


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def hash_state_dict(state: Mapping[str, Any], keys: Sequence[str] | None = None) -> str:
    digest = hashlib.sha256()
    selected = list(keys) if keys is not None else sorted(state.keys())
    for name in selected:
        if name not in state:
            raise SITATCNContractError(f"missing state key for hash: {name}")
        array = np.ascontiguousarray(_as_numpy(state[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def classify_parameter_keys(state: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    transferable: list[str] = []
    target_only: list[str] = []
    unknown: list[str] = []
    for name in state:
        if name.startswith(HEAD_KEY_PREFIXES) or name.startswith(REVIN_KEY_PREFIXES):
            target_only.append(name)
        elif name.startswith("blocks."):
            transferable.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise SITATCNContractError(f"unclassified TCN parameter keys: {unknown}")
    return {
        "transferable": tuple(sorted(transferable)),
        "target_only": tuple(sorted(target_only)),
    }


def assert_transfer_scope(state: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    classified = classify_parameter_keys(state)
    if classified["transferable"] != EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS:
        raise SITATCNContractError(
            "transferable TCN keys drifted: "
            f"{classified['transferable']} != {EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS}"
        )
    if classified["target_only"] != EXPECTED_TARGET_ONLY_KEYS:
        raise SITATCNContractError(
            f"target-only keys drifted: {classified['target_only']} != {EXPECTED_TARGET_ONLY_KEYS}"
        )
    return classified


def transferable_backbone_keys(state: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    if state is None:
        return EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS
    return assert_transfer_scope(state)["transferable"]


def seed_torch(seed: int) -> None:
    torch, _ = _load_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def build_formal_bl7_tcn_revin():
    """Construct the exact BL7 TCN+RevIN module used by fit_bl7_revin."""

    torch, nn = _load_torch()

    class TCNRevINNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.revin = TorchRevIN(num_features=1)
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
            x_norm, mean, stdev = self.revin.normalize(inputs)
            encoded = self.blocks(x_norm.transpose(1, 2))
            y_norm = self.head(encoded[:, :, -1])
            return self.revin.denormalize(y_norm, mean, stdev)

        def backbone_activation(self, inputs):
            x_norm, _, _ = self.revin.normalize(inputs)
            encoded = self.blocks(x_norm.transpose(1, 2))
            return encoded[:, :, -1]

    model = TCNRevINNet()
    assert_transfer_scope(model.state_dict())
    if uses_weight_norm(model.state_dict()):
        raise SITATCNContractError("formal BL7 TCN must not use weight_norm")
    return model


def uses_weight_norm(state: Mapping[str, Any]) -> bool:
    names = list(state.keys())
    return any(name.endswith("weight_g") or name.endswith("weight_v") or "parametrizations" in name for name in names)


def materialize_effective_tcn_state(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert a TCN state_dict into effective convolution weights.

    Formal BL7 has no weight_norm, so this is an identity copy. If weight_g /
    weight_v appear, they are reconstructed as w = g * v / ||v|| and the
    parametrization keys are dropped. Direct averaging of g and v is forbidden.
    """

    materialized: dict[str, np.ndarray] = {}
    consumed = set()
    for name, tensor in state.items():
        if name in consumed:
            continue
        if name.endswith("weight_g"):
            prefix = name[: -len("weight_g")]
            g_key, v_key = prefix + "weight_g", prefix + "weight_v"
            if v_key not in state:
                raise SITATCNContractError(f"weight_g without weight_v: {name}")
            g = _as_numpy(state[g_key])
            v = _as_numpy(state[v_key])
            axes = tuple(range(1, v.ndim))
            v_norm = np.sqrt(np.sum(v * v, axis=axes, keepdims=True))
            v_norm = np.maximum(v_norm, 1e-12)
            g_shaped = g.reshape(g.shape + (1,) * (v.ndim - g.ndim))
            materialized[prefix + "weight"] = g_shaped * (v / v_norm)
            consumed.update({g_key, v_key})
            continue
        if name.endswith("weight_v"):
            continue
        if "parametrizations" in name:
            raise SITATCNContractError(
                f"unsupported weight parametrization key {name}; fuse materialized weights only"
            )
        materialized[name] = _as_numpy(tensor)
    return materialized


def overlay_backbone(model, fused_backbone: Mapping[str, Any], *, keys: Sequence[str] | None = None) -> None:
    torch, _ = _load_torch()
    current = model.state_dict()
    classified = assert_transfer_scope(current)
    selected = tuple(keys) if keys is not None else classified["transferable"]
    forbidden = set(classified["target_only"])
    overlap = set(selected).intersection(forbidden)
    if overlap:
        raise SITATCNContractError(f"attempted to transfer target-only keys: {sorted(overlap)}")
    updated = dict(current)
    for name in selected:
        if name not in fused_backbone:
            raise SITATCNContractError(f"fused backbone missing {name}")
        reference = current[name]
        array = np.asarray(fused_backbone[name], dtype=np.float32)
        if tuple(array.shape) != tuple(reference.shape):
            raise SITATCNContractError(
                f"fused shape mismatch for {name}: {array.shape} != {tuple(reference.shape)}"
            )
        if not np.isfinite(array).all():
            raise SITATCNContractError(f"fused tensor {name} is non-finite")
        updated[name] = torch.as_tensor(array, dtype=reference.dtype)
    model.load_state_dict(updated)


def fuse_materialized_backbones(
    states: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    if len(states) != len(weights):
        raise SITATCNContractError("source state count must equal weight count")
    if len(states) == 0:
        raise SITATCNContractError("fusion requires at least one source")
    materialized = [materialize_effective_tcn_state(state) for state in states]
    selected = tuple(keys) if keys is not None else transferable_backbone_keys(states[0])
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0):
        raise SITATCNContractError(f"fusion weights must be finite and non-negative: {weight_array}")
    fused: dict[str, np.ndarray] = {}
    for name in selected:
        acc = None
        for source_state, weight in zip(materialized, weight_array):
            if name not in source_state:
                raise SITATCNContractError(f"source missing transferable key {name}")
            tensor = np.asarray(source_state[name], dtype=np.float64)
            acc = weight * tensor if acc is None else acc + weight * tensor
        if acc is None or not np.isfinite(acc).all():
            raise SITATCNContractError(f"fused tensor {name} is non-finite")
        fused[name] = acc.astype(np.float32)
    return fused


def official_inverse_distance_weights(distances: Sequence[float], *, eps: float = WEIGHT_EPS) -> np.ndarray:
    selector = SourceSelector()
    weights = np.asarray(
        selector.compute_source_weights(np.asarray(distances, dtype=np.float64), mode="inverse_distance", eps=float(eps)),
        dtype=np.float64,
    )
    if weights.size == 0:
        raise SITATCNContractError("inverse-distance weighting produced no weights")
    if abs(float(np.sum(weights)) - 1.0) >= WEIGHT_SUM_TOLERANCE:
        raise SITATCNContractError(f"source weights must sum to 1, got {float(np.sum(weights))}")
    return weights


def parameter_alignment_rows(
    states: Sequence[Mapping[str, Any]],
    *,
    source_ids: Sequence[str],
    keys: Sequence[str] | None = None,
    dataset: str = "",
    horizon: int = 0,
) -> list[dict[str, Any]]:
    if len(states) != len(source_ids):
        raise SITATCNContractError("alignment requires one source id per state")
    materialized = [materialize_effective_tcn_state(state) for state in states]
    selected = tuple(keys) if keys is not None else transferable_backbone_keys(states[0])
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(materialized):
        for j, right in enumerate(materialized):
            if j <= i:
                continue
            for name in selected:
                a = np.asarray(left[name], dtype=np.float64).reshape(-1)
                b = np.asarray(right[name], dtype=np.float64).reshape(-1)
                norm_i = float(np.linalg.norm(a))
                norm_j = float(np.linalg.norm(b))
                denom = norm_i * norm_j
                cosine = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
                relative = abs(norm_i - norm_j) / max(norm_i, 1e-12)
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "source_i": source_ids[i],
                        "source_j": source_ids[j],
                        "parameter_key": name,
                        "norm_i": norm_i,
                        "norm_j": norm_j,
                        "cosine_similarity": cosine,
                        "relative_norm_diff": float(relative),
                    }
                )
    return rows


def _activation_diagnostics(model, inputs) -> dict[str, float]:
    torch, _ = _load_torch()
    model.eval()
    with torch.no_grad():
        encoded = model.backbone_activation(inputs)
        predictions = model(inputs)
        return {
            "activation_norm_epoch0": float(torch.linalg.vector_norm(encoded).item()),
            "prediction_mean_epoch0": float(predictions.mean().item()),
            "prediction_std_epoch0": float(predictions.std(unbiased=False).item()) if predictions.numel() > 1 else 0.0,
            "output_shape": list(predictions.shape),
        }


def _finite_loss(value: float, *, name: str) -> float:
    loss = float(value)
    if not np.isfinite(loss):
        raise SITATCNContractError(f"{name} is not finite: {loss}")
    return loss


def canonical_source_init_state(*, seed: int) -> dict[str, Any]:
    seed_torch(int(seed))
    model = build_formal_bl7_tcn_revin()
    return copy.deepcopy(model.state_dict())


def split_source_sales(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match SITA source split ratios 0.8/0.1/0.1 without importing the CNN trainer."""

    values = _finite_vector(sales, name="source_sales")
    n_train = int(values.size * 0.8)
    n_val = int(values.size * 0.1)
    if n_train <= 0 or n_val <= 0 or values.size - n_train - n_val <= 0:
        raise SITATCNContractError(f"source series too short for 80/10/10 split: n={values.size}")
    train = values[:n_train]
    validation = values[n_train : n_train + n_val]
    test = values[n_train + n_val :]
    return train, validation, test


@dataclass
class TCNFitDiagnostics:
    train_loss_epoch0: float
    val_loss_epoch0: float
    prediction_mean_epoch0: float
    prediction_std_epoch0: float
    activation_norm_epoch0: float
    early_trace: list[dict[str, float]] = field(default_factory=list)
    best_val_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_run: int = 0
    target_random_init_hash: str = ""
    target_head_init_hash: str = ""
    target_revin_init_hash: str = ""
    fused_backbone_hash: str = ""
    paired_head_initialization: str = "NA"


def fit_tcn_revin_with_optional_backbone(
    train_sales,
    validation_sales,
    *,
    horizon: int,
    lookback: int = 10,
    seed: int = 42,
    fused_backbone: Mapping[str, Any] | None = None,
    init_state: Mapping[str, Any] | None = None,
    collect_early_epochs: int = 5,
) -> tuple[FittedBL7, TCNFitDiagnostics]:
    """BL7 TCN+RevIN training loop with optional source-informed backbone overlay."""

    resolved_horizon = int(horizon)
    resolved_lookback = int(lookback)
    train = _finite_vector(train_sales, name="train_sales")
    validation = _finite_vector(validation_sales, name="validation_sales")
    train_scaled, validation_scaled, scaler = _scale_partitions(train, validation)
    train_x, train_y = _direct_h_arrays(
        train_scaled, horizon=resolved_horizon, lookback=resolved_lookback, name="train_sales"
    )
    validation_x, validation_y = _direct_h_arrays(
        validation_scaled, horizon=resolved_horizon, lookback=resolved_lookback, name="validation_sales"
    )
    if float(np.ptp(train)) <= 1e-12:
        diagnostics = TCNFitDiagnostics(
            train_loss_epoch0=0.0,
            val_loss_epoch0=0.0,
            prediction_mean_epoch0=float(train[0]),
            prediction_std_epoch0=0.0,
            activation_norm_epoch0=0.0,
            paired_head_initialization="CONSTANT_SERIES",
        )
        return (
            FittedBL7(resolved_horizon, resolved_lookback, scaler, None, None, float(train[0])),
            diagnostics,
        )

    torch, nn = _load_torch()
    seed_torch(int(seed))
    model = build_formal_bl7_tcn_revin()
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(dict(init_state)))
    random_state = copy.deepcopy(model.state_dict())
    classified = assert_transfer_scope(random_state)
    head_hash = hash_state_dict(random_state, [k for k in classified["target_only"] if k.startswith("head.")])
    revin_hash = hash_state_dict(random_state, [k for k in classified["target_only"] if k.startswith("revin.")])
    random_hash = hash_state_dict(random_state)
    fused_hash = ""
    if fused_backbone is not None:
        overlay_backbone(model, fused_backbone, keys=classified["transferable"])
        fused_hash = hash_state_dict(model.state_dict(), classified["transferable"])
        after = model.state_dict()
        if hash_state_dict(after, [k for k in classified["target_only"] if k.startswith("head.")]) != head_hash:
            raise SITATCNContractError("backbone overlay mutated the prediction head")
        if hash_state_dict(after, [k for k in classified["target_only"] if k.startswith("revin.")]) != revin_hash:
            raise SITATCNContractError("backbone overlay mutated RevIN affine parameters")

    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32)
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        train_loss0 = _finite_loss(float(criterion(model(train_x_tensor), train_y_tensor).item()), name="train_loss_epoch0")
        val_loss0 = _finite_loss(float(criterion(model(validation_x_tensor), validation_y_tensor).item()), name="val_loss_epoch0")
    epoch0 = _activation_diagnostics(model, train_x_tensor)
    if not np.isfinite(epoch0["activation_norm_epoch0"]):
        raise SITATCNContractError("activation_norm_epoch0 is not finite")
    if list(epoch0["output_shape"]) != [int(train_x_tensor.shape[0]), 1]:
        raise SITATCNContractError(f"unexpected epoch-0 output shape {epoch0['output_shape']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    trace: list[dict[str, float]] = [
        {
            "epoch": 0,
            "train_loss": train_loss0,
            "val_loss": val_loss0,
            "learning_rate": float(LEARNING_RATE),
        }
    ]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(train_x_tensor), train_y_tensor)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x_tensor), validation_y_tensor).item())
            train_eval = float(criterion(model(train_x_tensor), train_y_tensor).item())
        if epoch <= int(collect_early_epochs):
            trace.append(
                {
                    "epoch": int(epoch),
                    "train_loss": _finite_loss(train_eval, name=f"train_loss_epoch{epoch}"),
                    "val_loss": _finite_loss(validation_loss, name=f"val_loss_epoch{epoch}"),
                    "learning_rate": float(LEARNING_RATE),
                }
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise SITATCNContractError("TCN+RevIN training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    diagnostics = TCNFitDiagnostics(
        train_loss_epoch0=train_loss0,
        val_loss_epoch0=val_loss0,
        prediction_mean_epoch0=float(epoch0["prediction_mean_epoch0"]),
        prediction_std_epoch0=float(epoch0["prediction_std_epoch0"]),
        activation_norm_epoch0=float(epoch0["activation_norm_epoch0"]),
        early_trace=trace,
        best_val_epoch=int(best_epoch),
        best_val_loss=_finite_loss(best_loss, name="best_val_loss"),
        epochs_run=int(epoch),
        target_random_init_hash=random_hash,
        target_head_init_hash=head_hash,
        target_revin_init_hash=revin_hash,
        fused_backbone_hash=fused_hash,
        paired_head_initialization="PASS",
    )
    return FittedBL7(resolved_horizon, resolved_lookback, scaler, torch, model), diagnostics


def train_source_tcn(
    source_sales,
    *,
    horizon: int,
    lookback: int = 10,
    init_state: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    train, validation, _test = split_source_sales(source_sales)
    fitted, diagnostics = fit_tcn_revin_with_optional_backbone(
        train,
        validation,
        horizon=int(horizon),
        lookback=int(lookback),
        seed=int(seed),
        init_state=init_state,
        fused_backbone=None,
    )
    if fitted.model is None:
        raise SITATCNContractError("source TCN collapsed to a constant series")
    state = copy.deepcopy(fitted.model.state_dict())
    if not all(np.isfinite(_as_numpy(tensor)).all() for tensor in state.values()):
        raise SITATCNContractError("source TCN checkpoint contains non-finite values")
    return {
        "state_dict": state,
        "materialized": materialize_effective_tcn_state(state),
        "diagnostics": diagnostics,
        "n_train": int(np.asarray(train).size),
        "n_val": int(np.asarray(validation).size),
    }


def materialized_forward_parity_error(state: Mapping[str, Any], inputs: np.ndarray) -> float:
    """Compare a live model forward with a clone loaded from materialized weights."""

    torch, _ = _load_torch()
    left = build_formal_bl7_tcn_revin()
    left.load_state_dict(copy.deepcopy(dict(state)))
    right = build_formal_bl7_tcn_revin()
    materialized = materialize_effective_tcn_state(state)
    current = right.state_dict()
    reloaded = {}
    for name, tensor in current.items():
        if name not in materialized:
            raise SITATCNContractError(f"materialized state missing {name}")
        reloaded[name] = torch.as_tensor(np.asarray(materialized[name], dtype=np.float32), dtype=tensor.dtype)
    right.load_state_dict(reloaded)
    x = torch.as_tensor(np.asarray(inputs, dtype=np.float32), dtype=torch.float32)
    left.eval()
    right.eval()
    with torch.no_grad():
        error = float(torch.max(torch.abs(left(x) - right(x))).item())
    if error > MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR:
        raise SITATCNContractError(
            f"MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR={error} exceeds {MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR}"
        )
    return error


def predict_target_only(fitted_model: FittedBL7, input_sales) -> float:
    if hasattr(fitted_model, "source_models") or hasattr(fitted_model, "source_series"):
        raise SITATCNContractError("fitted target model retained source artifacts")
    return predict_bl7(fitted_model, input_sales)


def transfer_scope_audit_payload(model=None) -> dict[str, Any]:
    if model is None:
        seed_torch(0)
        model = build_formal_bl7_tcn_revin()
    state = model.state_dict()
    classified = assert_transfer_scope(state)
    named_params = list(model.named_parameters())
    buffers = list(model.named_buffers())
    transferred = classified["transferable"]
    target_only = classified["target_only"]
    transferred_count = int(sum(int(np.prod(state[name].shape)) for name in transferred))
    total_count = int(sum(int(np.prod(tensor.numel() if hasattr(tensor, "numel") else np.prod(_as_numpy(tensor).shape))) for _, tensor in named_params))
    return {
        "TOTAL_PARAMETER_TENSORS": int(len(named_params)),
        "TRANSFERRED_PARAMETER_TENSORS": int(len(transferred)),
        "TARGET_ONLY_PARAMETER_TENSORS": int(len(target_only)),
        "NONTRAINABLE_BUFFERS": int(len(buffers)),
        "TRANSFERRED_PARAMETER_COUNT": transferred_count,
        "TOTAL_PARAMETER_COUNT": total_count,
        "TRANSFERABLE_TCN_BACKBONE_KEYS": list(transferred),
        "TARGET_ONLY_KEYS": list(target_only),
        "BUFFER_KEYS": [name for name, _ in buffers],
        "WEIGHT_NORM_STATUS": "ABSENT" if not uses_weight_norm(state) else "PRESENT",
        "REVIN_CONTRACT": dict(REVIN_CONTRACT),
    }


def dump_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
