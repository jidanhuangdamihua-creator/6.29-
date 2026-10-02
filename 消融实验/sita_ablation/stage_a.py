"""Stage A source-cache: selection, per-source CNN fit, and resume.

TCN source fit runs only when ``tcn_source_fit`` is complete. This module does
not write fused tensors or official run artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from sita_ablation.data import (
    ComparisonDays,
    ConditionFrames,
    ContractError,
    DatasetId,
    clip_source_history,
    split_source_frame,
)
from sita_ablation.models import CNN_FUSIBLE_LAYERS, TCN_FUSIBLE_LAYERS, build_source_cnn, build_tcn
from sita_ablation.scaling import ModelScaler, fit_source_model_scaler, transform_model_frame
from sita_ablation.source_selection import (
    STATUS_COMPLETED,
    STATUS_SKIPPED,
    inverse_distance_weights,
    rank_candidates,
    uniform_weights,
)
from sita_ablation.windows import build_partition_windows

Backbone = Literal["cnn", "tcn"]
SelectionRule = Literal["top", "bottom", "random3"]
CellSeed = Literal[42, 43, 44, 45, 46]
Horizon = Literal[1, 3, 5]
WeightMode = Literal["inverse_distance", "uniform", "single"]
ResumeDecision = Literal["RUN", "SKIP", "HOLD_HASH_MISMATCH", "RETRY_ONCE", "HOLD_FAILED"]

SCALING_RULE = "per_similarity_variable_minmax"
DRAW_RULE_RANDOM3 = "sorted_permutation_v1"
NOT_APPLICABLE = "NOT_APPLICABLE"
SOURCE_EPOCHS = 50
SOURCE_BATCH_SIZE = 16
_CELL_SEEDS = {42, 43, 44, 45, 46}
_HORIZONS = {1, 3, 5}
_TRAINED_K = {3, 5, 10}

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PACKAGE_ROOT / "configs"
_FROZEN_CONFIGS = ("features_frozen.json", "protocol_frozen.json", "tcn_frozen.json")


@dataclass(frozen=True)
class StageAKey:
    dataset: DatasetId
    horizon: Horizon
    seed: CellSeed
    backbone: Backbone
    selection_rule: SelectionRule
    trained_k: int
    comparison_start: str
    comparison_end: str
    source_history_start: str
    source_history_end: str
    model_features: tuple[str, ...]
    similarity_variables: tuple[str, ...]
    scaling_rule: str
    draw_rule: str | None
    frozen_config_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in {"D1", "D4", "D5", "D6"}:
            raise ContractError(f"unsupported dataset {self.dataset!r}")
        if self.horizon not in _HORIZONS:
            raise ContractError(f"horizon must be 1, 3, or 5, got {self.horizon!r}")
        if self.seed not in _CELL_SEEDS:
            raise ContractError(f"seed must be a cell seed, got {self.seed!r}")
        if self.backbone not in {"cnn", "tcn"}:
            raise ContractError(f"backbone must be cnn or tcn, got {self.backbone!r}")
        if self.selection_rule not in {"top", "bottom", "random3"}:
            raise ContractError(f"selection rule must be top, bottom, or random3, got {self.selection_rule!r}")
        if self.trained_k not in _TRAINED_K:
            raise ContractError(f"trained_k must be 3, 5, or 10, got {self.trained_k!r}")
        if self.scaling_rule != SCALING_RULE:
            raise ContractError(f"scaling_rule must be {SCALING_RULE!r}")
        expected_draw = DRAW_RULE_RANDOM3 if self.selection_rule == "random3" else None
        if self.draw_rule != expected_draw:
            raise ContractError(f"draw_rule must be {expected_draw!r} for {self.selection_rule}")
        if "sales" not in self.model_features:
            raise ContractError("model features must include sales")
        if self.comparison_start > self.comparison_end or self.source_history_start > self.source_history_end:
            raise ContractError("Stage A date span ends before it starts")


@dataclass(frozen=True)
class SourceTrainRequest:
    """Train-split windows only. Validation arrays are not on this object."""

    rank: int
    source_key: tuple[str, ...]
    x: np.ndarray
    y_scaled: np.ndarray
    label_date: np.ndarray
    backbone: Backbone
    seed: int


@dataclass
class SourceRecord:
    rank: int
    source_key: tuple[str, ...]
    weights: dict[str, list[np.ndarray]]
    scaler: ModelScaler
    sample_count: int


@dataclass
class StageABundle:
    key: StageAKey
    status: str
    distances: np.ndarray
    selected: tuple[tuple[str, ...], ...]
    sources: tuple[SourceRecord, ...]
    source_validation_metric: str
    source_validation_value_space: str
    cache_dir: Path


SourceTrainer = Callable[[SourceTrainRequest], dict[str, list[np.ndarray]]]


@dataclass(frozen=True)
class TcnSourceFit:
    """Frozen TCN source-fit contract. Missing or mistyped fields keep ``complete`` false."""

    loss: str | None
    optimizer: str | None
    learning_rate: float | None
    batch_size: int | None
    epochs: int | None
    clipnorm: float | None
    shuffle: bool | None
    validation_used_for_fit: bool | None
    validation_for_early_stopping: bool | None
    validation_for_checkpoint_selection: bool | None
    validation_for_source_ranking: bool | None
    ema: bool | None
    final_weights: str | None
    source_revin: str | None
    seed: str | None
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            not self.missing
            and self.loss == "mse"
            and self.optimizer == "adam"
            and self.learning_rate == 1e-4
            and type(self.batch_size) is int
            and self.batch_size == 16
            and type(self.epochs) is int
            and self.epochs == 50
            and self.clipnorm is None
            and self.shuffle is True
            and self.validation_used_for_fit is False
            and self.validation_for_early_stopping is False
            and self.validation_for_checkpoint_selection is False
            and self.validation_for_source_ranking is False
            and self.ema is False
            and self.final_weights == "last_epoch_raw"
            and self.source_revin == "UNUSED"
            and self.seed == "cell_seed_reset_per_stage_a_key"
        )


def load_tcn_source_fit() -> TcnSourceFit:
    """Read ``tcn_source_fit`` from ``tcn_frozen.json``."""
    payload = json.loads((_CONFIG_DIR / "tcn_frozen.json").read_text(encoding="utf-8"))
    section = payload.get("tcn_source_fit")
    if not isinstance(section, dict):
        raise ContractError("tcn_frozen.json is missing tcn_source_fit")
    return assess_tcn_source_fit(section)


def assess_tcn_source_fit(section: dict) -> TcnSourceFit:
    """Return completeness from one contract object. Does not train or fill defaults.

    A missing key stays missing. JSON null stays null. An old nested contract is
    not rewritten into the frozen values.
    """
    if not isinstance(section, dict):
        raise ContractError("tcn_source_fit must be an object")
    missing: list[str] = []
    for name in ("patience", "early_stopping", "best_state", "huber", "cosine"):
        if name in section:
            missing.append(name)
    loss = _exact_text(section, "loss", "mse", missing)
    optimizer = _exact_text(section, "optimizer", "adam", missing)
    learning_rate = _exact_rate(section, "learning_rate", 1e-4, missing)
    batch_size = _exact_count(section, "batch_size", 16, missing)
    epochs = _exact_count(section, "epochs", 50, missing)
    clipnorm = _exact_null(section, "clipnorm", missing)
    shuffle = _exact_bool(section, "shuffle", True, missing)
    validation_used_for_fit = _exact_bool(section, "validation_used_for_fit", False, missing)
    validation_for_early_stopping = _exact_bool(section, "validation_for_early_stopping", False, missing)
    validation_for_checkpoint_selection = _exact_bool(section, "validation_for_checkpoint_selection", False, missing)
    validation_for_source_ranking = _exact_bool(section, "validation_for_source_ranking", False, missing)
    ema = _exact_bool(section, "ema", False, missing)
    final_weights = _exact_text(section, "final_weights", "last_epoch_raw", missing)
    source_revin = _exact_text(section, "source_revin", "UNUSED", missing)
    seed = _exact_text(section, "seed", "cell_seed_reset_per_stage_a_key", missing)
    return TcnSourceFit(
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        clipnorm=clipnorm,
        shuffle=shuffle,
        validation_used_for_fit=validation_used_for_fit,
        validation_for_early_stopping=validation_for_early_stopping,
        validation_for_checkpoint_selection=validation_for_checkpoint_selection,
        validation_for_source_ranking=validation_for_source_ranking,
        ema=ema,
        final_weights=final_weights,
        source_revin=source_revin,
        seed=seed,
        missing=tuple(missing),
    )


def tcn_source_fit_is_complete() -> bool:
    return load_tcn_source_fit().complete


def _exact_text(section: dict, name: str, expected: str, missing: list[str]) -> str | None:
    if name not in section or type(section[name]) is not str or section[name] != expected:
        missing.append(name)
        return None
    return expected


def _exact_rate(section: dict, name: str, expected: float, missing: list[str]) -> float | None:
    value = section.get(name, None)
    if name not in section or isinstance(value, bool) or not isinstance(value, (int, float)):
        missing.append(name)
        return None
    number = float(value)
    if number != float(expected):
        missing.append(name)
        return None
    return number


def _exact_count(section: dict, name: str, expected: int, missing: list[str]) -> int | None:
    if name not in section or type(section[name]) is not int or section[name] != expected:
        missing.append(name)
        return None
    return expected


def _exact_bool(section: dict, name: str, expected: bool, missing: list[str]) -> bool | None:
    if name not in section or type(section[name]) is not bool or section[name] is not expected:
        missing.append(name)
        return None
    return expected


def _exact_null(section: dict, name: str, missing: list[str]) -> None:
    """Accept only an explicit JSON null. A missing key is not filled in."""
    if name not in section or section[name] is not None:
        missing.append(name)
    return None


def frozen_config_sha256() -> str:
    """SHA256 of the three frozen JSON files concatenated in filename order."""
    digest = hashlib.sha256()
    for name in _FROZEN_CONFIGS:
        digest.update((_CONFIG_DIR / name).read_bytes())
    return digest.hexdigest()


def make_stage_a_key(
    frames: ConditionFrames,
    *,
    horizon: Horizon,
    seed: CellSeed,
    backbone: Backbone,
    selection_rule: SelectionRule,
    trained_k: int,
    comparison_days: ComparisonDays,
) -> StageAKey:
    if comparison_days not in (15, 30):
        raise ContractError(f"comparison window must be 15 or 30, got {comparison_days!r}")
    span = frames.comparison_15 if comparison_days == 15 else frames.comparison_30
    features = ("sales",) if backbone == "tcn" else tuple(frames.model_features)
    start, end = span.iso()
    history_start, history_end = frames.source_history.iso()
    return StageAKey(
        dataset=frames.dataset,
        horizon=horizon,
        seed=seed,
        backbone=backbone,
        selection_rule=selection_rule,
        trained_k=int(trained_k),
        comparison_start=start,
        comparison_end=end,
        source_history_start=history_start,
        source_history_end=history_end,
        model_features=features,
        similarity_variables=tuple(frames.similarity_variables),
        scaling_rule=SCALING_RULE,
        draw_rule=DRAW_RULE_RANDOM3 if selection_rule == "random3" else None,
        frozen_config_sha256=frozen_config_sha256(),
    )


def stage_a_cache_dir(key: StageAKey, cache_root: Path) -> Path:
    return Path(cache_root) / stage_a_key_sha256(key)


def stage_a_key_sha256(key: StageAKey) -> str:
    payload = json.dumps(canonical_key(key), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_key(key: StageAKey) -> dict[str, object]:
    return {
        "dataset": key.dataset,
        "horizon": key.horizon,
        "seed": key.seed,
        "backbone": key.backbone,
        "selection_rule": key.selection_rule,
        "trained_k": key.trained_k,
        "comparison_start": key.comparison_start,
        "comparison_end": key.comparison_end,
        "source_history_start": key.source_history_start,
        "source_history_end": key.source_history_end,
        "model_features": list(key.model_features),
        "similarity_variables": list(key.similarity_variables),
        "scaling_rule": key.scaling_rule,
        "draw_rule": key.draw_rule,
        "frozen_config_sha256": key.frozen_config_sha256,
    }


def fit_source_cnn(x: np.ndarray, y_scaled: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Fit one source CNN with the frozen SOURCE-CNN-FIT settings. No validation data."""
    if x.ndim != 3 or y_scaled.ndim != 1 or x.shape[0] != y_scaled.shape[0]:
        raise ContractError(f"source CNN fit expected x (N, 10, C) and y (N,), got {x.shape} and {y_scaled.shape}")
    model = build_source_cnn((int(x.shape[1]), int(x.shape[2])))
    model.fit(
        x.astype(np.float32, copy=False),
        y_scaled.astype(np.float32, copy=False),
        epochs=SOURCE_EPOCHS,
        batch_size=SOURCE_BATCH_SIZE,
        verbose=0,
    )
    named: dict[str, list[np.ndarray]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            named[layer.name] = [np.asarray(array) for array in arrays]
    return named


def fit_source_tcn(x: np.ndarray, y_scaled: np.ndarray, contract: TcnSourceFit) -> dict[str, list[np.ndarray]]:
    """Fit one source TCN from a complete contract.

    ``x`` and ``y_scaled`` are the source-train Min–Max sales channel. The fit
    call does not receive validation data, callbacks, early stopping, a
    checkpoint, or an EMA copy. The returned tensors are the raw weights left
    in the model when the last epoch returns.
    """
    if not contract.complete:
        detail = ", ".join(contract.missing) or "contract"
        raise ContractError(f"GAP-TCN-SOURCE-FIT: tcn_source_fit is incomplete: {detail}")
    if x.ndim != 3 or y_scaled.ndim != 1 or x.shape[0] != y_scaled.shape[0] or int(x.shape[-1]) != 1:
        raise ContractError(f"source TCN fit expected x (N, 10, 1) and y (N,), got {x.shape} and {y_scaled.shape}")
    model = build_tcn((int(x.shape[1]), 1))
    if any("revin" in layer.name for layer in model.layers):
        raise ContractError("source-side RevIN is unused")
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(contract.learning_rate),
            clipnorm=contract.clipnorm,
        ),
        loss=str(contract.loss),
    )
    model.fit(
        x.astype(np.float32, copy=False),
        y_scaled.astype(np.float32, copy=False),
        epochs=int(contract.epochs),
        batch_size=int(contract.batch_size),
        shuffle=bool(contract.shuffle),
        verbose=0,
    )
    named: dict[str, list[np.ndarray]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            named[layer.name] = [np.asarray(array) for array in arrays]
    return named


def load_or_train_stage_a(
    key: StageAKey,
    frames: ConditionFrames,
    *,
    cache_root: Path,
    trainer: SourceTrainer | None = None,
) -> StageABundle:
    """Load a matching cache or train sources. Incomplete TCN contracts are refused."""
    if frames.dataset != key.dataset:
        raise ContractError(f"frames dataset {frames.dataset!r} does not match key {key.dataset!r}")
    directory = stage_a_cache_dir(key, cache_root)
    decision = inspect_stage_a(directory, key)
    if decision == "SKIP":
        return _read_bundle(directory, key)
    if decision == "HOLD_HASH_MISMATCH":
        raise ContractError("HOLD_HASH_MISMATCH")
    if decision == "HOLD_FAILED":
        raise ContractError("HOLD_FAILED")
    if decision == "RETRY_ONCE":
        _preserve_failed_attempt(directory)
    contract = load_tcn_source_fit()
    if key.backbone == "tcn" and not contract.complete:
        detail = ", ".join(contract.missing) or "contract"
        raise ContractError(f"GAP-TCN-SOURCE-FIT: tcn_source_fit is incomplete: {detail}")
    try:
        bundle = _train_bundle(key, frames, directory, trainer)
    except ContractError:
        raise
    except Exception as exc:
        _write_failure(directory, key, exc)
        raise
    return bundle


def inspect_stage_a(directory: Path, key: StageAKey) -> ResumeDecision:
    directory = Path(directory)
    done_path = directory / "DONE.json"
    if done_path.exists():
        payload = _read_json(done_path)
        if payload is None or not _done_matches(directory, key, payload):
            return "HOLD_HASH_MISMATCH"
        return "SKIP"
    failed_path = directory / "FAILED.json"
    if failed_path.exists():
        payload = _read_json(failed_path)
        count = None if payload is None else payload.get("attempt_count")
        if count == 1:
            return "RETRY_ONCE"
        return "HOLD_FAILED"
    return "RUN"


def fuse_init(bundle: StageABundle, weight_mode: WeightMode) -> dict[str, list[np.ndarray]]:
    """Position-wise weighted average of fusible tensors. Does not write a cache."""
    if bundle.status != STATUS_COMPLETED:
        raise ContractError("fusion requires a completed Stage A bundle")
    if weight_mode not in {"inverse_distance", "uniform", "single"}:
        raise ContractError(f"unknown weight mode {weight_mode!r}")
    count = len(bundle.sources)
    if weight_mode == "inverse_distance":
        weights = inverse_distance_weights(bundle.distances)
    elif weight_mode == "uniform":
        weights = uniform_weights(count)
    else:
        weights = np.zeros(count, dtype=np.float64)
        weights[0] = 1.0
    names = CNN_FUSIBLE_LAYERS if bundle.key.backbone == "cnn" else TCN_FUSIBLE_LAYERS
    fused: dict[str, list[np.ndarray]] = {}
    for name in names:
        pieces = [_layer_arrays(source.weights, name) for source in bundle.sources]
        width = len(pieces[0])
        fused[name] = [
            sum(float(weights[index]) * pieces[index][slot] for index in range(count))
            for slot in range(width)
        ]
    return fused


def _train_bundle(
    key: StageAKey,
    frames: ConditionFrames,
    directory: Path,
    trainer: SourceTrainer | None,
) -> StageABundle:
    comparison_days = _comparison_days(key)
    _require_key_matches_frames(key, frames, comparison_days)
    selection = rank_candidates(
        key.dataset,
        frames,
        comparison_days=comparison_days,
        rule=key.selection_rule,
        k=key.trained_k,
        seed=key.seed,
    )
    if selection.status == STATUS_SKIPPED:
        _write_skipped(directory, key, selection)
        return _read_bundle(directory, key)
    _reset_tensorflow_seed(int(key.seed))
    if trainer is None:
        trainer = _tcn_trainer if key.backbone == "tcn" else _cnn_trainer
    sources: list[SourceRecord] = []
    for selected in selection.selected:
        request, scaler = _train_request(key, frames, selected.source_key, selected.rank)
        weights = trainer(request)
        _require_weights(key.backbone, weights)
        sources.append(
            SourceRecord(
                rank=selected.rank,
                source_key=selected.source_key,
                weights=weights,
                scaler=scaler,
                sample_count=int(request.y_scaled.shape[0]),
            )
        )
    _write_completed(directory, key, selection, tuple(sources))
    return _read_bundle(directory, key)


def _train_request(
    key: StageAKey,
    frames: ConditionFrames,
    source_key: tuple[str, ...],
    rank: int,
) -> tuple[SourceTrainRequest, ModelScaler]:
    clipped = clip_source_history(frames, source_key)
    split = split_source_frame(clipped, key.model_features)
    scaler = fit_source_model_scaler(split.train, key.model_features)
    transformed = transform_model_frame(scaler, split.train)
    windows = build_partition_windows(
        transformed,
        key_fields=frames.key_fields,
        key=source_key,
        feature_columns=key.model_features,
        horizon=key.horizon,
        partition="train",
    )
    if key.backbone == "tcn":
        model_input = windows.x_sales
    else:
        model_input = windows.x
    request = SourceTrainRequest(
        rank=rank,
        source_key=source_key,
        x=model_input,
        y_scaled=windows.y_scaled,
        label_date=windows.label_date,
        backbone=key.backbone,
        seed=int(key.seed),
    )
    return request, scaler


def _cnn_trainer(request: SourceTrainRequest) -> dict[str, list[np.ndarray]]:
    if request.backbone != "cnn":
        raise ContractError("the frozen source-CNN trainer cannot fit a TCN")
    return fit_source_cnn(request.x, request.y_scaled)


def _tcn_trainer(request: SourceTrainRequest) -> dict[str, list[np.ndarray]]:
    if request.backbone != "tcn":
        raise ContractError("the TCN source trainer cannot fit a CNN")
    return fit_source_tcn(request.x, request.y_scaled, load_tcn_source_fit())


def _reset_tensorflow_seed(seed: int) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))


def _write_completed(directory: Path, key: StageAKey, selection, sources: tuple[SourceRecord, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    selection_payload = _selection_payload(selection)
    selection_payload["sample_counts"] = [source.sample_count for source in sources]
    payloads = {
        "key.json": canonical_key(key),
        "selection.json": selection_payload,
    }
    for source in sources:
        payloads[f"source_scaler_{source.rank}.json"] = _scaler_payload(source.scaler)
    written = [ _atomic_json(directory / name, payload) for name, payload in payloads.items() ]
    for source in sources:
        written.append(_atomic_weights(directory / f"source_{source.rank}.npz", source.weights))
    done = {
        "stage_a_key_sha256": stage_a_key_sha256(key),
        "status": STATUS_COMPLETED,
        "artifact_sha256": {path.name: _sha256(path) for path in written},
        "source_validation_metric": NOT_APPLICABLE,
        "source_validation_value_space": NOT_APPLICABLE,
    }
    _atomic_json(directory / "DONE.json", done)


def _write_skipped(directory: Path, key: StageAKey, selection) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    key_path = _atomic_json(directory / "key.json", canonical_key(key))
    selection_path = _atomic_json(directory / "selection.json", _selection_payload(selection))
    done = {
        "stage_a_key_sha256": stage_a_key_sha256(key),
        "status": STATUS_SKIPPED,
        "artifact_sha256": {
            key_path.name: _sha256(key_path),
            selection_path.name: _sha256(selection_path),
        },
        "source_validation_metric": NOT_APPLICABLE,
        "source_validation_value_space": NOT_APPLICABLE,
    }
    _atomic_json(directory / "DONE.json", done)


def _write_failure(directory: Path, key: StageAKey, exc: Exception) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 2 if (directory / "FAILED.attempt1.json").exists() else 1
    _atomic_json(
        directory / "FAILED.json",
        {
            "stage_a_key_sha256": stage_a_key_sha256(key),
            "attempt_count": attempt,
            "error": f"{type(exc).__name__}: {exc}",
        },
    )


def _preserve_failed_attempt(directory: Path) -> None:
    failed = directory / "FAILED.json"
    preserved = directory / "FAILED.attempt1.json"
    if failed.exists() and not preserved.exists():
        os.replace(failed, preserved)


def _read_bundle(directory: Path, key: StageAKey) -> StageABundle:
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    status = str(selection["status"])
    distances = np.asarray(selection["distances"], dtype=np.float64)
    selected = tuple(tuple(item) for item in selection["selected_keys"])
    sources: list[SourceRecord] = []
    if status == STATUS_COMPLETED:
        for rank, source_key in enumerate(selected):
            sources.append(
                SourceRecord(
                    rank=rank,
                    source_key=source_key,
                    weights=_read_weights(directory / f"source_{rank}.npz"),
                    scaler=_read_scaler(directory / f"source_scaler_{rank}.json"),
                    sample_count=int(selection["sample_counts"][rank]),
                )
            )
    return StageABundle(
        key=key,
        status=status,
        distances=distances,
        selected=selected,
        sources=tuple(sources),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=directory,
    )


def _selection_payload(selection) -> dict[str, object]:
    return {
        "status": selection.status,
        "selected_keys": [list(item.source_key) for item in selection.selected],
        "ranks": [item.rank for item in selection.selected],
        "distances": [float(item.distance) for item in selection.selected],
        "n_legal": int(selection.n_legal),
        "comparison_start": selection.comparison_start,
        "comparison_end": selection.comparison_end,
        "source_validation_metric": NOT_APPLICABLE,
        "source_validation_value_space": NOT_APPLICABLE,
    }


def _scaler_payload(scaler: ModelScaler) -> dict[str, object]:
    return {
        "role": scaler.role,
        "columns": list(scaler.columns),
        "data_min": scaler.data_min.tolist(),
        "data_max": scaler.data_max.tolist(),
        "scale": scaler.scale.tolist(),
        "min": scaler.min.tolist(),
        "fit_dates": list(scaler.fit_dates),
    }


def _read_scaler(path: Path) -> ModelScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelScaler(
        role=str(payload["role"]),
        columns=tuple(payload["columns"]),
        data_min=np.asarray(payload["data_min"], dtype=np.float64),
        data_max=np.asarray(payload["data_max"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
        min=np.asarray(payload["min"], dtype=np.float64),
        fit_dates=tuple(payload["fit_dates"]),
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _atomic_weights(path: Path, weights: dict[str, list[np.ndarray]]) -> Path:
    temporary = path.with_suffix(".tmp.npz")
    arrays = {}
    for name, tensors in weights.items():
        for index, tensor in enumerate(tensors):
            arrays[f"{name}__{index}"] = np.asarray(tensor)
    np.savez(temporary, **arrays)
    os.replace(temporary, path)
    return path


def _read_weights(path: Path) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path) as payload:
        for name in payload.files:
            layer, index_text = name.rsplit("__", 1)
            grouped.setdefault(layer, {})[int(index_text)] = np.asarray(payload[name])
    return {
        layer: [slots[index] for index in range(len(slots))]
        for layer, slots in grouped.items()
    }


def _done_matches(directory: Path, key: StageAKey, payload: dict[str, object]) -> bool:
    if payload.get("stage_a_key_sha256") != stage_a_key_sha256(key):
        return False
    if payload.get("status") not in {STATUS_COMPLETED, STATUS_SKIPPED}:
        return False
    artifacts = payload.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for name, digest in artifacts.items():
        path = directory / str(name)
        if not path.is_file() or _sha256(path) != digest:
            return False
    if payload["status"] == STATUS_SKIPPED:
        return set(artifacts) == {"key.json", "selection.json"}
    if payload["status"] == STATUS_COMPLETED:
        expected = {"key.json", "selection.json"}
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        for rank in range(len(selection["selected_keys"])):
            expected.add(f"source_{rank}.npz")
            expected.add(f"source_scaler_{rank}.json")
        if set(artifacts) != expected:
            return False
    return True


def _require_weights(backbone: Backbone, weights: dict[str, list[np.ndarray]]) -> None:
    names = CNN_FUSIBLE_LAYERS if backbone == "cnn" else TCN_FUSIBLE_LAYERS
    missing = [name for name in names if name not in weights or len(weights[name]) < 2]
    if missing:
        raise ContractError(f"source weights are missing fusible tensors {missing}")


def _layer_arrays(weights: dict[str, list[np.ndarray]], name: str) -> list[np.ndarray]:
    if name not in weights:
        raise ContractError(f"source weights have no fusible layer {name!r}")
    return [np.asarray(array, dtype=np.float64) for array in weights[name]]


def _comparison_days(key: StageAKey) -> ComparisonDays:
    days = _span_days(key.comparison_start, key.comparison_end)
    if days == 15:
        return 15
    if days == 30:
        return 30
    raise ContractError("Stage A comparison span must be 15 or 30 days")


def _require_key_matches_frames(key: StageAKey, frames: ConditionFrames, comparison_days: ComparisonDays) -> None:
    span = frames.comparison_15 if comparison_days == 15 else frames.comparison_30
    if span.iso() != (key.comparison_start, key.comparison_end):
        raise ContractError("Stage A key comparison dates do not match the frames")
    if frames.source_history.iso() != (key.source_history_start, key.source_history_end):
        raise ContractError("Stage A key source history does not match the frames")
    if tuple(frames.similarity_variables) != key.similarity_variables:
        raise ContractError("Stage A key similarity variables do not match the frames")
    expected = ("sales",) if key.backbone == "tcn" else tuple(frames.model_features)
    if key.model_features != expected:
        raise ContractError("Stage A key model features do not match this backbone")


def _span_days(start: str, end: str) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 1


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
