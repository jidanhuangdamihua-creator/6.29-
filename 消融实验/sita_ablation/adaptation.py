"""Target adaptation for one RunSpec.

Consumes target windows, the target scaler, a caller-supplied init template,
and, for sourced arms, a Stage A bundle. Fusion uses ``fuse_init``.
This module does not rank sources, fit scalers, write run files, or fit a
TCN source. Signed error is scored with the test labels. Alpha entropy uses
the frozen natural-log formula when fusion weights exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from sita_ablation.arms import BLOCKED_GAP_TCN_SOURCE_FIT, RunSpec, frozen_tensors, validate_fusible_weights
from sita_ablation.data import ContractError, load_dataset_contract
from sita_ablation.models import CNN_FUSIBLE_LAYERS, TCN_FUSIBLE_LAYERS, build_target_cnn, build_tcn
from sita_ablation.revin import RevIN, build_revin
from sita_ablation.scaling import ModelScaler, inverse_sales
from sita_ablation.source_selection import inverse_distance_weights, uniform_weights
from sita_ablation.stage_a import NOT_APPLICABLE, StageABundle, fuse_init, tcn_source_fit_is_complete
from sita_ablation.windows import WINDOW_SIZE, WindowBatch, validate_target_sample_counts

UNRESOLVED = "UNRESOLVED"
DEFINED = "DEFINED"
UNDEFINED_ZERO_INIT_NORM = "UNDEFINED_ZERO_INIT_NORM"
COMPLETED = "COMPLETED"
ABSENT = "ABSENT"
PRESENT = "PRESENT"
_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "protocol_frozen.json"
_FUSE_MODES = {"inverse_distance", "uniform", "single"}


@dataclass(frozen=True)
class WeightSnapshot:
    """Named float32 tensors. The caller owns the shared random template."""

    tensors: dict[str, tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class TargetWindows:
    train: WindowBatch
    validation: WindowBatch
    test: WindowBatch


@dataclass(frozen=True)
class ParameterShift:
    absolute: float
    relative: float | None
    status: str


@dataclass(frozen=True)
class TargetProtocol:
    """Frozen target-training settings. The learning rate is the RunSpec value."""

    epochs: int
    batch_size: int
    loss_name: str
    huber_delta: float
    optimizer_name: str
    schedule: str
    schedule_passes_alpha: bool
    learning_rate: float
    ema_momentum: float
    final_weight_state: str
    validation_weight_variant: str
    early_stopping: bool
    steps_per_epoch: int
    decay_steps: int


@dataclass(frozen=True)
class TargetTrainRequest:
    """Update arrays are the target train slice only."""

    x: np.ndarray
    y_scaled: np.ndarray
    label_date: np.ndarray
    protocol: TargetProtocol
    seed: int
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    target_scaler: ModelScaler


@dataclass(frozen=True)
class RecordOnlyValidation:
    role: str
    x: np.ndarray
    y_scaled: np.ndarray
    y_original: np.ndarray
    label_date: np.ndarray
    participates_in_update: bool
    participates_in_model_selection: bool


@dataclass(frozen=True)
class TargetTrainOutput:
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    val_rmse: tuple[float, ...]
    val_smape: tuple[float, ...]
    epoch0_val_rmse: float
    gradient_steps: int
    raw_weights: dict[str, list[np.ndarray]]
    ema_weights: dict[str, list[np.ndarray]]
    used_validation_for_update: bool
    used_validation_for_selection: bool


@dataclass(frozen=True)
class AdaptationResult:
    status: str
    spec: RunSpec
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    val_rmse: tuple[float, ...]
    val_smape: tuple[float, ...]
    epoch0_val_rmse: float | None
    validation_rmse: float | None
    validation_smape: float | None
    validation_weight_variant: str | None
    gradient_steps: int | None
    raw_shift: dict[str, ParameterShift]
    ema_shift: dict[str, ParameterShift]
    test_predictions: np.ndarray
    label_date: np.ndarray
    unscored_test_dates: tuple[str, ...]
    test_weight_variant: str | None
    test_weight_sha256_before: str | None
    test_weight_sha256_after: str | None
    signed_error: float | None
    alpha_entropy: float | None
    effective_source_count: float | None
    source_info: str
    source_validation_metric: str | None
    source_validation_value_space: str | None
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    revin_enabled: bool
    learning_rate: float | None
    checkpoint_selection: bool
    n_train: int | None = None
    n_validation: int | None = None


TargetTrainer = Callable[
    [TargetTrainRequest, tf.keras.Model, Optional[RevIN], RecordOnlyValidation],
    TargetTrainOutput,
]


def adapt_target(
    spec: RunSpec,
    windows: TargetWindows,
    target_scaler: ModelScaler,
    init_template: WeightSnapshot,
    stage_a: StageABundle | None,
    trainer: TargetTrainer | None = None,
) -> AdaptationResult:
    """Adapt one target arm and return in-memory predictions. Does not write artifacts."""
    if spec.backbone == "tcn" and spec.selection_rule != "none" and not tcn_source_fit_is_complete():
        return _blocked(spec)
    n_train, n_validation = _require_windows(spec, windows)
    _require_target_scaler(target_scaler, windows)
    channels = _input_channels(spec, windows)
    initialized = _initial_tensors(spec, init_template, stage_a, channels)
    protocol = _protocol_for(spec, sample_count=int(windows.train.y_scaled.shape[0]))
    model = _build_model(spec, channels)
    _assign_snapshot(model, None, initialized)
    revin = _build_optional_revin(spec, channels, initialized)
    trainable_names, frozen_names, _variables = _trainable_state(spec, model, revin)
    initial = _snapshot_dict(model, revin)
    _assert_train_labels(spec, windows.train)
    tf.keras.utils.set_random_seed(int(spec.seed))
    request = TargetTrainRequest(
        x=_model_input(spec, windows.train),
        y_scaled=np.array(windows.train.y_scaled, dtype=np.float32, copy=True),
        label_date=np.array(windows.train.label_date, copy=True),
        protocol=protocol,
        seed=int(spec.seed),
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        target_scaler=target_scaler,
    )
    validation = RecordOnlyValidation(
        role="record_only",
        x=_model_input(spec, windows.validation),
        y_scaled=np.array(windows.validation.y_scaled, dtype=np.float32, copy=True),
        y_original=np.array(windows.validation.y_original, dtype=np.float64, copy=True),
        label_date=np.array(windows.validation.label_date, copy=True),
        participates_in_update=False,
        participates_in_model_selection=False,
    )
    output = (run_target_training if trainer is None else trainer)(request, model, revin, validation)
    _require_trainer_output(protocol, output)
    _assert_frozen_held(spec, initial, output.raw_weights, output.ema_weights)
    raw_shift = _shift_map(initial, output.raw_weights, backbone=spec.backbone, revin=spec.revin)
    ema_shift = _shift_map(initial, output.ema_weights, backbone=spec.backbone, revin=spec.revin)
    _assign_snapshot(model, revin, output.ema_weights)
    before = _snapshot_dict(model, revin)
    before_digest = _digest(before)
    predictions = _predict_original(model, revin, _model_input(spec, windows.test), target_scaler)
    after = _snapshot_dict(model, revin)
    after_digest = _digest(after)
    if before_digest != after_digest:
        raise ContractError("test inference changed target parameters")
    if predictions.shape != (180,) or predictions.dtype != np.float64:
        raise ContractError(f"test predictions must be float64 (180,), got {predictions.shape} {predictions.dtype}")
    if windows.test.unscored_test_dates:
        raise ContractError("test inference left unscored dates")
    return AdaptationResult(
        status=COMPLETED,
        spec=spec,
        train_loss=tuple(float(value) for value in output.train_loss),
        val_loss=tuple(float(value) for value in output.val_loss),
        val_rmse=tuple(float(value) for value in output.val_rmse),
        val_smape=tuple(float(value) for value in output.val_smape),
        epoch0_val_rmse=float(output.epoch0_val_rmse),
        validation_rmse=float(output.val_rmse[-1]),
        validation_smape=float(output.val_smape[-1]),
        validation_weight_variant=protocol.validation_weight_variant,
        gradient_steps=int(output.gradient_steps),
        raw_shift=raw_shift,
        ema_shift=ema_shift,
        test_predictions=predictions,
        label_date=np.array(windows.test.label_date, copy=True),
        unscored_test_dates=(),
        test_weight_variant=protocol.final_weight_state,
        test_weight_sha256_before=before_digest,
        test_weight_sha256_after=after_digest,
        signed_error=None,
        alpha_entropy=_alpha_entropy(spec, stage_a),
        effective_source_count=_effective_source_count(spec, stage_a),
        source_info=ABSENT if spec.selection_rule == "none" else PRESENT,
        source_validation_metric=None if spec.selection_rule == "none" else NOT_APPLICABLE,
        source_validation_value_space=None if spec.selection_rule == "none" else NOT_APPLICABLE,
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        revin_enabled=revin is not None,
        learning_rate=float(spec.target_learning_rate),
        checkpoint_selection=False,
        n_train=n_train,
        n_validation=n_validation,
    )


def run_target_training(
    request: TargetTrainRequest,
    model: tf.keras.Model,
    revin: RevIN | None,
    validation: RecordOnlyValidation,
) -> TargetTrainOutput:
    """Adam, Huber, cosine, and EMA. Validation is recorded on RAW weights and does not select."""
    protocol = request.protocol
    if protocol.early_stopping:
        raise ContractError("frozen target protocol disables early stopping")
    if protocol.schedule_passes_alpha:
        raise ContractError("cosine decay does not receive an alpha argument")
    if validation.role != "record_only" or validation.participates_in_update or validation.participates_in_model_selection:
        raise ContractError("validation cannot update parameters or select the model")
    if protocol.validation_weight_variant != "RAW":
        raise ContractError("in-loop validation must use RAW weights")
    if protocol.final_weight_state != "EMA":
        raise ContractError("test weights must be the EMA state")
    tf.keras.utils.set_random_seed(int(request.seed))
    names, frozen_names, variables = _trainable_state_from_model(model, revin, frozen=set(request.frozen_names))
    if tuple(names) != tuple(request.trainable_names) or tuple(frozen_names) != tuple(request.frozen_names):
        raise ContractError("optimizer variable list does not match the arm's frozen tensors")
    steps = int(protocol.decay_steps)
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        float(protocol.learning_rate),
        decay_steps=steps,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    huber = tf.keras.losses.Huber(delta=float(protocol.huber_delta))
    ema_values = [tf.identity(tf.cast(var, tf.float32)) for var in variables]
    momentum = float(protocol.ema_momentum)
    x_train = tf.constant(request.x, dtype=tf.float32)
    y_train = tf.constant(np.asarray(request.y_scaled, dtype=np.float32).reshape(-1, 1))
    count = int(request.x.shape[0])
    generator = np.random.default_rng(int(request.seed))

    def observe() -> tuple[float, float, float]:
        pred = _forward(model, revin, validation.x, training=False)
        loss = huber(tf.constant(validation.y_scaled.reshape(-1, 1), dtype=tf.float32), pred)
        original = inverse_sales(request.target_scaler, pred.numpy().reshape(-1))
        rmse, smape = _recorded_original_errors(validation.y_original, original)
        return float(loss.numpy()), rmse, smape

    _epoch0_loss, epoch0_rmse, _epoch0_smape = observe()
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_rmses: list[float] = []
    val_smapes: list[float] = []
    for _epoch in range(int(protocol.epochs)):
        order = generator.permutation(count)
        batch_losses: list[float] = []
        for step in range(int(protocol.steps_per_epoch)):
            start = step * int(protocol.batch_size)
            batch_idx = order[start : start + int(protocol.batch_size)]
            if int(batch_idx.size) == 0:
                batch_idx = order
            xb = tf.gather(x_train, batch_idx)
            yb = tf.gather(y_train, batch_idx)
            with tf.GradientTape() as tape:
                pred = _forward(model, revin, xb, training=True)
                loss = huber(yb, pred)
            grads = tape.gradient(loss, variables)
            pairs = []
            for grad, var in zip(grads, variables):
                if grad is None:
                    raise ContractError(f"missing gradient for trainable variable {var.name}")
                if not bool(tf.reduce_all(tf.math.is_finite(grad)).numpy()):
                    raise ContractError(f"non-finite gradient for {var.name}")
                pairs.append((grad, var))
            optimizer.apply_gradients(pairs)
            ema_values = [
                momentum * ema + (1.0 - momentum) * tf.cast(var, tf.float32)
                for ema, var in zip(ema_values, variables)
            ]
            batch_losses.append(float(loss.numpy()))
        train_losses.append(float(np.mean(batch_losses)))
        val_loss, val_rmse, val_smape = observe()
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_smapes.append(val_smape)
    raw_weights = _snapshot_dict(model, revin)
    for var, ema in zip(variables, ema_values):
        var.assign(ema)
    ema_weights = _snapshot_dict(model, revin)
    return TargetTrainOutput(
        train_loss=tuple(train_losses),
        val_loss=tuple(val_losses),
        val_rmse=tuple(val_rmses),
        val_smape=tuple(val_smapes),
        epoch0_val_rmse=float(epoch0_rmse),
        gradient_steps=steps,
        raw_weights=raw_weights,
        ema_weights=ema_weights,
        used_validation_for_update=False,
        used_validation_for_selection=False,
    )


def parameter_shift(
    initial: Sequence[np.ndarray],
    final: Sequence[np.ndarray],
    *,
    unchanged_is_undefined: bool = False,
) -> ParameterShift:
    """L2 of the concatenated tensors. A zero initial norm has no relative shift.

    RevIN's beta starts at 0. An unchanged RevIN group therefore reports
    ``UNDEFINED_ZERO_INIT_NORM`` and a null relative shift, including when gamma
    is 1 and would otherwise make the concatenated norm positive.
    """
    init = _concat(initial)
    done = _concat(final)
    if init.shape != done.shape:
        raise ContractError("parameter shift tensors changed shape")
    absolute = float(np.linalg.norm(done - init))
    init_norm = float(np.linalg.norm(init))
    if init_norm == 0.0 or (unchanged_is_undefined and absolute == 0.0):
        return ParameterShift(absolute=absolute, relative=None, status=UNDEFINED_ZERO_INIT_NORM)
    return ParameterShift(absolute=absolute, relative=absolute / init_norm, status=DEFINED)


def _blocked(spec: RunSpec) -> AdaptationResult:
    return AdaptationResult(
        status=BLOCKED_GAP_TCN_SOURCE_FIT,
        spec=spec,
        train_loss=(),
        val_loss=(),
        val_rmse=(),
        val_smape=(),
        epoch0_val_rmse=None,
        validation_rmse=None,
        validation_smape=None,
        validation_weight_variant=None,
        gradient_steps=None,
        raw_shift={},
        ema_shift={},
        test_predictions=np.zeros((0,), dtype=np.float64),
        label_date=np.zeros((0,), dtype="datetime64[D]"),
        unscored_test_dates=(),
        test_weight_variant=None,
        test_weight_sha256_before=None,
        test_weight_sha256_after=None,
        signed_error=None,
        alpha_entropy=None,
        effective_source_count=None,
        source_info=BLOCKED_GAP_TCN_SOURCE_FIT,
        source_validation_metric=None,
        source_validation_value_space=None,
        trainable_names=(),
        frozen_names=(),
        revin_enabled=False,
        learning_rate=None,
        checkpoint_selection=False,
    )


def _protocol_for(spec: RunSpec, *, sample_count: int) -> TargetProtocol:
    file_protocol = _read_protocol_file()
    if sample_count < 1:
        raise ContractError("target training received no training samples")
    steps = max(1, math.ceil(sample_count / file_protocol["batch_size"]))
    epochs = file_protocol["epochs"]
    final_state = "EMA" if file_protocol["ema_for_test"] else "RAW"
    return TargetProtocol(
        epochs=epochs,
        batch_size=file_protocol["batch_size"],
        loss_name=file_protocol["loss_name"],
        huber_delta=file_protocol["huber_delta"],
        optimizer_name=file_protocol["optimizer_name"],
        schedule=file_protocol["schedule"],
        schedule_passes_alpha=False,
        learning_rate=float(spec.target_learning_rate),
        ema_momentum=file_protocol["ema_momentum"],
        final_weight_state=final_state,
        validation_weight_variant=file_protocol["validation_weights"],
        early_stopping=file_protocol["early_stopping"],
        steps_per_epoch=steps,
        decay_steps=epochs * steps,
    )


def _read_protocol_file() -> dict[str, object]:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if "huber" not in payload or "target_optimizer" not in payload or "ema" not in payload:
        raise ContractError("frozen protocol is missing the target optimizer, Huber, or EMA section")
    optimizer = payload["target_optimizer"]
    huber = payload["huber"]
    ema = payload["ema"]
    parsed = {
        "epochs": int(optimizer["epochs"]["value"]),
        "batch_size": int(optimizer["batch_size"]["value"]),
        "optimizer_name": str(optimizer["name"]["value"]),
        "schedule": str(optimizer["schedule"]["value"]),
        "early_stopping": bool(optimizer["early_stopping"]["value"]),
        "loss_name": "huber",
        "huber_delta": float(huber["delta"]["value"]),
        "ema_momentum": float(ema["momentum"]["value"]),
        "ema_for_test": bool(ema["enabled_for_test"]["value"]),
        "validation_weights": str(ema["in_loop_validation_weights"]["value"]),
    }
    if parsed["optimizer_name"] != "Adam" or parsed["schedule"] != "cosine":
        raise ContractError("frozen target optimizer must be Adam with a cosine schedule")
    if parsed["early_stopping"]:
        raise ContractError("frozen target protocol disables early stopping")
    if parsed["validation_weights"] != "RAW" or not parsed["ema_for_test"]:
        raise ContractError("frozen validation uses RAW weights and the test state is EMA")
    if parsed["epochs"] < 1 or parsed["batch_size"] < 1:
        raise ContractError("frozen epoch count and batch size must be positive")
    if not (0.0 < float(parsed["ema_momentum"]) < 1.0):
        raise ContractError("frozen EMA momentum must be inside (0, 1)")
    return parsed


def _initial_tensors(
    spec: RunSpec,
    template: WeightSnapshot,
    bundle: StageABundle | None,
    channels: int,
) -> dict[str, list[np.ndarray]]:
    initialized = _copy_weights(template.tensors)
    if spec.selection_rule == "none":
        if bundle is not None:
            raise ContractError("a source-free arm must not receive a Stage A bundle")
        _require_template(spec, initialized)
        return initialized
    if bundle is None:
        raise ContractError("a sourced arm requires an existing Stage A bundle")
    _require_bundle(spec, bundle)
    if spec.weight_mode not in _FUSE_MODES:
        raise ContractError(f"sourced arm weight mode is not frozen, got {spec.weight_mode!r}")
    fused = fuse_init(bundle, spec.weight_mode)  # type: ignore[arg-type]
    validate_fusible_weights(spec.backbone, fused, channels=channels)
    for name, tensors in fused.items():
        if name not in initialized:
            raise ContractError(f"template is missing fusible parameter {name}")
        initialized[name] = [np.asarray(array, dtype=np.float32) for array in tensors]
    _require_template(spec, initialized)
    return initialized


def _require_bundle(spec: RunSpec, bundle: StageABundle) -> None:
    key = bundle.key
    if key.dataset != spec.dataset or key.horizon != spec.horizon or key.seed != spec.seed:
        raise ContractError("Stage A bundle identity does not match this run")
    if key.backbone != spec.backbone:
        raise ContractError("CNN and TCN Stage A weights are not interchangeable")
    if spec.backbone != "cnn" and not (spec.backbone == "tcn" and tcn_source_fit_is_complete()):
        raise ContractError(BLOCKED_GAP_TCN_SOURCE_FIT)
    span_days = int((pd.Timestamp(key.comparison_end) - pd.Timestamp(key.comparison_start)).days) + 1
    if spec.weight_mode == "single":
        if key.trained_k != 3 or key.selection_rule != "top" or spec.comparison_days != 30 or span_days != 30:
            raise ContractError("P2 reads the Top-3 cache and does not train a separate K=1 source")
        return
    if key.selection_rule != spec.selection_rule or key.trained_k != spec.requested_k or span_days != spec.comparison_days:
        raise ContractError("Stage A bundle does not match this arm's source rule, K, or comparison window")


def _require_template(spec: RunSpec, weights: Mapping[str, Sequence[np.ndarray]]) -> None:
    required = (*CNN_FUSIBLE_LAYERS, "head_target") if spec.backbone == "cnn" else (*TCN_FUSIBLE_LAYERS, "head")
    missing = [name for name in required if name not in weights]
    if missing:
        raise ContractError(f"init template is missing {missing}")


def _build_model(spec: RunSpec, channels: int) -> tf.keras.Model:
    if spec.backbone == "cnn":
        model, _extractor = build_target_cnn((10, channels))
        return model
    if spec.backbone == "tcn":
        if spec.selection_rule != "none" and not tcn_source_fit_is_complete():
            raise ContractError(BLOCKED_GAP_TCN_SOURCE_FIT)
        return build_tcn((10, 1))
    raise ContractError(f"unsupported backbone {spec.backbone!r}")


def _build_optional_revin(spec: RunSpec, channels: int, weights: Mapping[str, Sequence[np.ndarray]]) -> RevIN | None:
    if not spec.revin:
        return None
    width = 1 if spec.backbone == "tcn" else channels
    revin = build_revin(width)
    if "affine_weight" in weights or "affine_bias" in weights:
        if "affine_weight" not in weights or "affine_bias" not in weights:
            raise ContractError("RevIN template must include both affine_weight and affine_bias")
        revin.affine_weight.assign(np.asarray(weights["affine_weight"][0], dtype=np.float32))
        revin.affine_bias.assign(np.asarray(weights["affine_bias"][0], dtype=np.float32))
    return revin


def _trainable_state(
    spec: RunSpec,
    model: tf.keras.Model,
    revin: RevIN | None,
) -> tuple[tuple[str, ...], tuple[str, ...], list[tf.Variable]]:
    """Freeze P1 conv variables by layer-object identity, then list optimizer variables."""
    frozen = set(frozen_tensors(spec))
    for layer in model.layers:
        if not layer.weights:
            continue
        labels = [_layer_weight_label(layer, var) for var in layer.weights]
        if any(label in frozen for label in labels) and not all(label in frozen for label in labels):
            raise ContractError(f"{layer.name} mixes frozen and trainable tensors")
        layer.trainable = not any(label in frozen for label in labels)
    if revin is not None:
        revin.trainable = True
    return _trainable_state_from_model(model, revin, frozen=frozen)


def _trainable_state_from_model(
    model: tf.keras.Model,
    revin: RevIN | None,
    *,
    frozen: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], list[tf.Variable]]:
    trainable: list[tf.Variable] = []
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    seen: set[int] = set()

    def take(var: tf.Variable, label: str, *, frozen_slot: bool) -> None:
        if id(var) in seen:
            return
        seen.add(id(var))
        if frozen_slot:
            if label not in frozen:
                raise ContractError(f"{label} left the optimizer without being a frozen tensor")
            frozen_names.append(label)
            return
        if label in frozen:
            raise ContractError(f"frozen tensor {label} entered the optimizer list")
        trainable.append(var)
        trainable_names.append(label)

    for layer in model.layers:
        for var in layer.trainable_weights:
            take(var, _layer_weight_label(layer, var), frozen_slot=False)
        for var in layer.non_trainable_weights:
            take(var, _layer_weight_label(layer, var), frozen_slot=True)
    if revin is not None:
        for var in revin.trainable_weights:
            take(var, _revin_weight_label(revin, var), frozen_slot=False)
        for var in revin.non_trainable_weights:
            take(var, _revin_weight_label(revin, var), frozen_slot=True)
    if set(frozen_names) != frozen:
        raise ContractError(f"frozen tensors {tuple(frozen_names)} != {tuple(sorted(frozen))}")
    if not trainable:
        raise ContractError("target adaptation has no trainable variables")
    _require_optimizer_membership(model, revin, trainable, frozen)
    return tuple(trainable_names), tuple(frozen_names), trainable


def _layer_weight_label(layer: tf.keras.layers.Layer, var: tf.Variable) -> str:
    """Name a weight from the layer object. Does not read Variable.name."""
    if getattr(layer, "kernel", None) is var:
        return f"{layer.name}.kernel"
    if getattr(layer, "bias", None) is var:
        return f"{layer.name}.bias"
    raise ContractError(f"{layer.name} has a weight that is neither its kernel nor its bias")


def _revin_weight_label(revin: RevIN, var: tf.Variable) -> str:
    """Name a RevIN affine variable from the layer attributes, not Variable.name."""
    if revin.affine_weight is var:
        return "affine_weight"
    if revin.affine_bias is var:
        return "affine_bias"
    raise ContractError("RevIN has a weight that is neither affine_weight nor affine_bias")


def _require_optimizer_membership(
    model: tf.keras.Model,
    revin: RevIN | None,
    trainable: Sequence[tf.Variable],
    frozen: set[str],
) -> None:
    """P1 conv kernel/bias stay out of the optimizer; head and RevIN stay in."""
    optimizer_ids = {id(var) for var in trainable}
    frozen_ids: set[int] = set()
    for layer in model.layers:
        for var in layer.weights:
            if _layer_weight_label(layer, var) in frozen:
                frozen_ids.add(id(var))
                if id(var) in optimizer_ids:
                    raise ContractError(f"frozen variable on {layer.name} entered the optimizer list")
    if len(frozen_ids) != len(frozen):
        raise ContractError("frozen conv variables do not match the P1 tensor set")
    if any(layer.name == "head_target" for layer in model.layers):
        head = model.get_layer("head_target")
        for var in (head.kernel, head.bias):
            if id(var) not in optimizer_ids:
                raise ContractError("head_target left the optimizer list")
    if revin is not None:
        for var in (revin.affine_weight, revin.affine_bias):
            if var is None or id(var) not in optimizer_ids:
                raise ContractError("RevIN left the optimizer list")


def _assert_frozen_held(
    spec: RunSpec,
    initial: Mapping[str, Sequence[np.ndarray]],
    raw: Mapping[str, Sequence[np.ndarray]],
    ema: Mapping[str, Sequence[np.ndarray]],
) -> None:
    for name in frozen_tensors(spec):
        layer, slot = name.rsplit(".", 1)
        index = 0 if slot == "kernel" else 1
        for final in (raw, ema):
            if layer not in final or not _same(initial[layer][index], final[layer][index]):
                raise ContractError(f"frozen tensor {name} changed during target adaptation")


def _shift_map(
    initial: Mapping[str, Sequence[np.ndarray]],
    final: Mapping[str, Sequence[np.ndarray]],
    *,
    backbone: str,
    revin: bool,
) -> dict[str, ParameterShift]:
    groups: dict[str, ParameterShift] = {}
    if backbone == "cnn":
        for name in CNN_FUSIBLE_LAYERS:
            groups[name] = parameter_shift(initial[name], final[name])
        groups["head"] = parameter_shift(initial["head_target"], final["head_target"])
    else:
        for name in TCN_FUSIBLE_LAYERS:
            groups[name] = parameter_shift(initial[name], final[name])
        groups["head"] = parameter_shift(initial["head"], final["head"])
    if revin:
        groups["revin"] = parameter_shift(
            [initial["affine_weight"][0], initial["affine_bias"][0]],
            [final["affine_weight"][0], final["affine_bias"][0]],
            unchanged_is_undefined=True,
        )
    return groups


def _predict_original(
    model: tf.keras.Model,
    revin: RevIN | None,
    x: np.ndarray,
    scaler: ModelScaler,
) -> np.ndarray:
    pred = _forward(model, revin, x, training=False)
    return inverse_sales(scaler, pred.numpy().reshape(-1))


def _forward(model: tf.keras.Model, revin: RevIN | None, x: np.ndarray | tf.Tensor, *, training: bool) -> tf.Tensor:
    values = tf.convert_to_tensor(x, dtype=tf.float32)
    if revin is None:
        return model(values, training=training)
    normalized, mean, stdev = revin.normalize(values)
    pred = model(normalized, training=training)
    return revin.denormalize(pred, mean, stdev)


def _recorded_original_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    true_raw = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_raw = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true_raw.shape != pred_raw.shape or not np.isfinite(true_raw).all() or not np.isfinite(pred_raw).all():
        raise ContractError("recorded validation metrics received a non-finite or misaligned sales vector")
    true = np.maximum(true_raw, 0.0)
    pred = np.maximum(pred_raw, 0.0)
    rmse = float(np.sqrt(np.mean(np.square(pred - true))))
    denom = np.abs(true) + np.abs(pred) + 1e-8
    smape = float(100.0 * np.mean(2.0 * np.abs(pred - true) / denom))
    return rmse, smape


def _require_trainer_output(protocol: TargetProtocol, output: TargetTrainOutput) -> None:
    if output.used_validation_for_update or output.used_validation_for_selection:
        raise ContractError("validation entered a parameter update or model selection")
    width = int(protocol.epochs)
    curves = (output.train_loss, output.val_loss, output.val_rmse, output.val_smape)
    if any(len(curve) != width for curve in curves):
        raise ContractError("target training must finish every frozen epoch; validation cannot stop it")
    if int(output.gradient_steps) != int(protocol.decay_steps):
        raise ContractError("gradient steps do not match epochs times steps per epoch")


def _fusion_alpha(spec: RunSpec, bundle: StageABundle | None) -> np.ndarray | None:
    if spec.selection_rule == "none" or bundle is None:
        return None
    if spec.weight_mode == "single":
        return np.array([1.0], dtype=np.float64)
    if spec.weight_mode == "uniform":
        return uniform_weights(len(bundle.sources))
    return inverse_distance_weights(bundle.distances)


def _alpha_entropy(spec: RunSpec, bundle: StageABundle | None) -> float | None:
    alpha = _fusion_alpha(spec, bundle)
    if alpha is None:
        return None
    from sita_ablation.metrics import alpha_entropy

    return alpha_entropy(alpha)


def _effective_source_count(spec: RunSpec, bundle: StageABundle | None) -> float | None:
    alpha = _fusion_alpha(spec, bundle)
    if alpha is None:
        return None
    total = float(np.sum(np.square(alpha), dtype=np.float64))
    if not np.isfinite(total) or total == 0.0:
        raise ContractError("effective source count is undefined")
    return float(np.float64(1.0) / np.float64(total))


def _require_windows(spec: RunSpec, windows: TargetWindows) -> tuple[int, int]:
    n_train, n_validation, _n_test = validate_target_sample_counts(
        windows.train,
        windows.validation,
        windows.test,
        spec.horizon,
    )
    contract = load_dataset_contract(spec.dataset)
    if windows.train.partition != "train" or windows.validation.partition != "validation" or windows.test.partition != "test":
        raise ContractError("target windows must be train, validation, and test")
    _partition_inside(windows.train, contract.target_train, spec.horizon)
    _partition_inside(windows.validation, contract.target_val, spec.horizon)
    expected = tuple(pd.date_range(contract.target_test.start, contract.target_test.end, freq="D"))
    got = _timestamps(windows.test.label_date)
    if len(expected) != 180 or got != expected:
        raise ContractError("test labels must equal the frozen 180-day interval")
    if windows.test.unscored_test_dates:
        raise ContractError(f"test windows dropped dates {windows.test.unscored_test_dates}")
    ends = _timestamps(windows.test.input_end)
    if any(end >= label for end, label in zip(ends, got)):
        raise ContractError("a test input ends on or after its label date")
    train_days = set(_timestamps(windows.train.label_date))
    val_days = set(_timestamps(windows.validation.label_date))
    test_days = set(got)
    if train_days & val_days or train_days & test_days or val_days & test_days:
        raise ContractError("train, validation, and test labels overlap")
    return n_train, n_validation


def _assert_train_labels(spec: RunSpec, train: WindowBatch) -> None:
    contract = load_dataset_contract(spec.dataset)
    _labels_inside(train, contract.target_train)


def _require_target_scaler(scaler: ModelScaler, windows: TargetWindows) -> None:
    if scaler.role != "target_model":
        raise ContractError("target adaptation requires the target model scaler")
    if not scaler.columns or scaler.columns[0] != "sales":
        raise ContractError("sales must be the first target-scaler column")
    test_days = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in _timestamps(windows.test.label_date)}
    if test_days & set(scaler.fit_dates):
        raise ContractError("target scaler fit dates include a test date")


def _partition_inside(batch: WindowBatch, span, horizon: int) -> None:
    """Keep every input day and every label inside one 15-day slice."""
    _labels_inside(batch, span)
    starts = _timestamps(batch.input_start)
    ends = _timestamps(batch.input_end)
    labels = _timestamps(batch.label_date)
    if len(starts) != len(labels) or len(ends) != len(labels):
        raise ContractError(f"{batch.partition} inputs and labels are not aligned")
    for start, end, label in zip(starts, ends, labels):
        if start < span.start or end > span.end or start > end:
            raise ContractError(f"{batch.partition} inputs leave the frozen interval")
        if int((end - start).days) != WINDOW_SIZE - 1:
            raise ContractError(f"{batch.partition} input window is not {WINDOW_SIZE} days")
        if int((label - end).days) != horizon or label < span.start or label > span.end:
            raise ContractError(f"{batch.partition} label is not {horizon} days after an in-slice input")


def _labels_inside(batch: WindowBatch, span) -> None:
    labels = _timestamps(batch.label_date)
    if not labels:
        raise ContractError(f"{batch.partition} has no labels")
    if labels[0] < span.start or labels[-1] > span.end or any(day < span.start or day > span.end for day in labels):
        raise ContractError(f"{batch.partition} labels leave the frozen interval")


def _timestamps(values: np.ndarray) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(day) for day in pd.to_datetime(np.asarray(values)))


def _input_channels(spec: RunSpec, windows: TargetWindows) -> int:
    if spec.backbone == "tcn":
        if windows.train.x_sales.shape[-1] != 1:
            raise ContractError("TCN target input must be the sales channel")
        return 1
    return int(windows.train.x.shape[-1])


def _model_input(spec: RunSpec, batch: WindowBatch) -> np.ndarray:
    source = batch.x_sales if spec.backbone == "tcn" else batch.x
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 10:
        raise ContractError(f"model input must have shape (N, 10, C), got {values.shape}")
    return np.array(values, dtype=np.float32, copy=True)


def _assign_snapshot(model: tf.keras.Model, revin: RevIN | None, weights: Mapping[str, Sequence[np.ndarray]]) -> None:
    for layer in model.layers:
        if layer.name not in weights or not layer.weights:
            continue
        try:
            layer.set_weights([np.asarray(array, dtype=np.float32) for array in weights[layer.name]])
        except ValueError as exc:
            raise ContractError(f"cannot load {layer.name}: {exc}") from exc
    if revin is None:
        return
    if "affine_weight" not in weights or "affine_bias" not in weights:
        raise ContractError("EMA snapshot is missing RevIN affine parameters")
    revin.affine_weight.assign(np.asarray(weights["affine_weight"][0], dtype=np.float32))
    revin.affine_bias.assign(np.asarray(weights["affine_bias"][0], dtype=np.float32))


def _snapshot_dict(model: tf.keras.Model, revin: RevIN | None) -> dict[str, list[np.ndarray]]:
    named: dict[str, list[np.ndarray]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            named[layer.name] = [np.asarray(array, dtype=np.float32) for array in arrays]
    if revin is not None:
        named["affine_weight"] = [np.asarray(revin.affine_weight.numpy(), dtype=np.float32)]
        named["affine_bias"] = [np.asarray(revin.affine_bias.numpy(), dtype=np.float32)]
    return named


def _copy_weights(weights: Mapping[str, Sequence[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {name: [np.array(array, dtype=np.float32, copy=True) for array in tensors] for name, tensors in weights.items()}


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32))


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ContractError("parameter shift received no tensors")
    return np.concatenate([np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays])


def _digest(snapshot: Mapping[str, Sequence[np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        digest.update(name.encode("utf-8"))
        for array in snapshot[name]:
            values = np.ascontiguousarray(array, dtype=np.float32)
            digest.update(str(values.shape).encode("utf-8"))
            digest.update(values.tobytes())
    return digest.hexdigest()
