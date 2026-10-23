"""Frozen arm identities, Stage A keys, and fusible initialization.

Does not train, adapt, or write run artifacts. Sourced TCN arms stay blocked
while ``tcn_source_fit`` is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from sita_ablation.data import ConditionFrames, ContractError
from sita_ablation.models import (
    CNN_FUSIBLE_LAYERS,
    TCN_FUSIBLE_LAYERS,
    TCN_FUSIBLE_SHAPES,
    expected_cnn_fusible_shapes,
)
from sita_ablation.stage_a import (
    NOT_APPLICABLE,
    StageABundle,
    StageAKey,
    fuse_init,
    make_stage_a_key,
    tcn_source_fit_is_complete,
)

ArmName = Literal["F1", "F2", "F3", "F4", "P1", "P2", "P3", "SITA_TCN", "TCN_REVIN"]
ProtocolTag = Literal["main", "alt_lr", "random3", "bottom3", "k5", "k10", "ell15"]
DatasetId = Literal["D1", "D4", "D5", "D6"]
Horizon = Literal[1, 3, 5]
CellSeed = Literal[42, 43, 44, 45, 46]
Backbone = Literal["cnn", "tcn"]
SelectionRule = Literal["none", "top", "bottom", "random3"]
WeightMode = Literal["none", "inverse_distance", "uniform", "single"]
PrepStatus = Literal["READY", "BLOCKED_GAP_TCN_SOURCE_FIT"]

DATASETS: tuple[DatasetId, ...] = ("D1", "D4", "D5", "D6")
HORIZONS: tuple[Horizon, ...] = (1, 3, 5)
SEEDS: tuple[CellSeed, ...] = (42, 43, 44, 45, 46)
BLOCKED_GAP_TCN_SOURCE_FIT = "BLOCKED_GAP_TCN_SOURCE_FIT"

_MAIN_LR = {"D1": 3e-5, "D4": 3e-5, "D5": 3e-4, "D6": 3e-5}
_ALT_LR = {"D1": 3e-4, "D4": 3e-4, "D5": 3e-5, "D6": 3e-4}
_FUSIBLE = {"cnn": CNN_FUSIBLE_LAYERS, "tcn": TCN_FUSIBLE_LAYERS}


@dataclass(frozen=True)
class _ArmDef:
    arm: ArmName
    protocol_tag: ProtocolTag
    revin: bool
    freeze_conv: bool
    selection_rule: SelectionRule
    requested_k: int | None
    weight_mode: WeightMode
    comparison_days: Literal[15, 30] | None
    backbone: Backbone


def _sourced(
    arm: ArmName,
    tag: ProtocolTag,
    *,
    revin: bool,
    freeze_conv: bool = False,
    rule: SelectionRule,
    k: int,
    weight_mode: WeightMode,
    comparison_days: Literal[15, 30] = 30,
    backbone: Backbone = "cnn",
) -> _ArmDef:
    return _ArmDef(arm, tag, revin, freeze_conv, rule, k, weight_mode, comparison_days, backbone)


def _plain(arm: ArmName, tag: ProtocolTag, *, revin: bool, backbone: Backbone = "cnn") -> _ArmDef:
    return _ArmDef(arm, tag, revin, False, "none", None, "none", None, backbone)


_PHASE_ARMS: dict[int, tuple[_ArmDef, ...]] = {
    1: (
        _plain("F1", "main", revin=False),
        _plain("F2", "main", revin=True),
        _sourced("F3", "main", revin=False, rule="top", k=3, weight_mode="inverse_distance"),
        _sourced("F4", "main", revin=True, rule="top", k=3, weight_mode="inverse_distance"),
        _sourced("P1", "main", revin=True, freeze_conv=True, rule="top", k=3, weight_mode="inverse_distance"),
        _sourced("P2", "main", revin=True, rule="top", k=1, weight_mode="single"),
        _sourced("P3", "main", revin=True, rule="top", k=3, weight_mode="uniform"),
    ),
    2: (
        _plain("F1", "alt_lr", revin=False),
        _plain("F2", "alt_lr", revin=True),
        _sourced("F3", "alt_lr", revin=False, rule="top", k=3, weight_mode="inverse_distance"),
        _sourced("F4", "alt_lr", revin=True, rule="top", k=3, weight_mode="inverse_distance"),
        _sourced("F4", "random3", revin=True, rule="random3", k=3, weight_mode="inverse_distance"),
        _sourced("F4", "bottom3", revin=True, rule="bottom", k=3, weight_mode="inverse_distance"),
        _sourced("F4", "k5", revin=True, rule="top", k=5, weight_mode="inverse_distance"),
        _sourced("F4", "k10", revin=True, rule="top", k=10, weight_mode="inverse_distance"),
        _sourced("F4", "ell15", revin=True, rule="top", k=3, weight_mode="inverse_distance", comparison_days=15),
    ),
    3: (
        _sourced("SITA_TCN", "main", revin=True, rule="top", k=3, weight_mode="inverse_distance", backbone="tcn"),
        _plain("TCN_REVIN", "main", revin=True, backbone="tcn"),
    ),
    4: (
        _sourced("F4", "ell15", revin=True, rule="top", k=3, weight_mode="inverse_distance", comparison_days=15),
    ),
    5: (
        _sourced("F4", "k5", revin=True, rule="top", k=5, weight_mode="inverse_distance"),
    ),
}

_NO_REVIN_P_ARMS: tuple[_ArmDef, ...] = (
    _sourced("P1", "main", revin=False, freeze_conv=True, rule="top", k=3, weight_mode="inverse_distance"),
    _sourced("P2", "main", revin=False, rule="top", k=1, weight_mode="single"),
    _sourced("P3", "main", revin=False, rule="top", k=3, weight_mode="uniform"),
)


@dataclass(frozen=True)
class RunSpec:
    arm: ArmName
    protocol_tag: ProtocolTag
    dataset: DatasetId
    horizon: Horizon
    seed: CellSeed
    backbone: Backbone
    revin: bool
    freeze_conv: bool
    selection_rule: SelectionRule
    requested_k: int | None
    weight_mode: WeightMode
    comparison_days: Literal[15, 30] | None
    target_learning_rate: float

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS or self.horizon not in HORIZONS or self.seed not in SEEDS:
            raise ContractError("RunSpec dataset, horizon, or seed is outside the frozen grid")
        if self.backbone not in {"cnn", "tcn"}:
            raise ContractError(f"backbone must be cnn or tcn, got {self.backbone!r}")
        if self.freeze_conv and self.arm != "P1":
            raise ContractError("only P1 freezes conv1-3")
        if self.selection_rule == "none":
            if self.requested_k is not None or self.weight_mode != "none" or self.comparison_days is not None:
                raise ContractError("a source-free arm has no K, weight mode, or comparison window")
            return
        if self.requested_k not in {1, 3, 5, 10}:
            raise ContractError(f"requested_k must be 1, 3, 5, or 10, got {self.requested_k!r}")
        if self.comparison_days not in {15, 30}:
            raise ContractError("comparison_days must be 15 or 30; windows longer than 30 are rejected")
        if self.weight_mode == "none":
            raise ContractError("a sourced arm needs a frozen weight mode")
        if self.weight_mode == "single" and self.requested_k != 1:
            raise ContractError("single initialization is the Top-1 arm")
        if self.weight_mode == "uniform" and self.requested_k != 3:
            raise ContractError("uniform initialization is defined at K=3")


@dataclass(frozen=True)
class ArmPreparation:
    spec: RunSpec
    status: PrepStatus
    stage_a_key: StageAKey | None
    fused: dict[str, list[np.ndarray]] | None
    initialized: dict[str, list[np.ndarray]] | None
    frozen_tensors: tuple[str, ...]
    source_validation_metric: str
    source_validation_value_space: str


def expand_runs(phase: Literal[1, 2, 3, 4, 5], *, include_no_revin_variants: bool = False) -> tuple[RunSpec, ...]:
    """Expand one frozen phase. Tier 3 arms X1, X2, and X3 are never returned.

    Phase 4 is only ``F4/ell15``. Phase 5 is only ``F4/k5``.
    Phase 2 keeps both tags in its own list.
    """
    if phase not in _PHASE_ARMS:
        raise ContractError(f"phase must be 1, 2, 3, 4, or 5, got {phase!r}")
    definitions = list(_PHASE_ARMS[phase])
    if include_no_revin_variants and phase == 1:
        definitions.extend(_NO_REVIN_P_ARMS)
    specs = []
    for definition in definitions:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    specs.append(_spec_from_definition(definition, dataset, horizon, seed))
    return tuple(specs)


def target_learning_rate(dataset: DatasetId, protocol_tag: ProtocolTag) -> float:
    table = _ALT_LR if protocol_tag == "alt_lr" else _MAIN_LR
    if dataset not in table:
        raise ContractError(f"no frozen target learning rate for {dataset}")
    return float(table[dataset])


def run_directory(spec: RunSpec, root: Path | None = None) -> Path:
    """Return the frozen run path. Does not create the directory.

    Optional no-RevIN P1/P2/P3 share the frozen prefix with the main arm, so
    those three paths gain a final ``revin0`` segment. That is the existing
    ``RunSpec.revin`` flag, not a new protocol tag.
    """
    base = Path("消融实验/runs") if root is None else Path(root)
    path = base / spec.arm / spec.protocol_tag / f"{spec.dataset}_h{spec.horizon}" / f"seed{spec.seed}"
    if spec.arm in {"P1", "P2", "P3"} and spec.revin is False:
        path = path / "revin0"
    return path


def stage_a_key_for(spec: RunSpec, frames: ConditionFrames) -> StageAKey | None:
    """Stage A identity for a sourced arm. P2 reuses the Top-3 key and does not use K=1."""
    if spec.selection_rule == "none":
        return None
    if spec.weight_mode == "single":
        trained_k = 3
        rule = "top"
        comparison_days: Literal[15, 30] | None = 30
    else:
        trained_k = int(spec.requested_k)
        rule = spec.selection_rule
        comparison_days = spec.comparison_days
    if comparison_days not in {15, 30}:
        raise ContractError("comparison windows longer than 30 days are rejected")
    return make_stage_a_key(
        frames,
        horizon=spec.horizon,
        seed=spec.seed,
        backbone=spec.backbone,
        selection_rule=rule,  # top, bottom, or random3; source-free specs already returned
        trained_k=trained_k,
        comparison_days=comparison_days,
    )


def frozen_tensors(spec: RunSpec) -> tuple[str, ...]:
    """P1 freezes the six conv1-3 kernel and bias tensors. Every other tensor stays trainable."""
    if not spec.freeze_conv:
        return ()
    if spec.backbone != "cnn":
        raise ContractError("conv freezing is only defined for the CNN P1 arm")
    return tuple(f"{layer}.{slot}" for layer in CNN_FUSIBLE_LAYERS for slot in ("kernel", "bias"))


def tensor_trainable(spec: RunSpec, name: str) -> bool:
    return name not in set(frozen_tensors(spec))


def validate_fusible_weights(
    backbone: Backbone,
    weights: Mapping[str, Sequence[np.ndarray]],
    *,
    channels: int,
) -> None:
    """Require the frozen fusible names, kernel-then-bias order, and shapes."""
    expected_names = _FUSIBLE[backbone]
    present = tuple(weights)
    if present != tuple(expected_names):
        raise ContractError(f"fusible parameter names or order {present} != {expected_names}")
    shapes = expected_cnn_fusible_shapes(channels) if backbone == "cnn" else TCN_FUSIBLE_SHAPES
    for name in expected_names:
        tensors = list(weights[name])
        if len(tensors) != 2:
            raise ContractError(f"{name} must provide kernel then bias, got {len(tensors)} tensors")
        kernel_shape, bias_shape = shapes[name]
        if tuple(np.asarray(tensors[0]).shape) != kernel_shape or tuple(np.asarray(tensors[1]).shape) != bias_shape:
            raise ContractError(
                f"{name} shape {(tuple(np.asarray(tensors[0]).shape), tuple(np.asarray(tensors[1]).shape))} "
                f"!= {(kernel_shape, bias_shape)}"
            )


def prepare_arm(
    spec: RunSpec,
    frames: ConditionFrames,
    template: Mapping[str, Sequence[np.ndarray]],
    bundle: StageABundle | None = None,
) -> ArmPreparation:
    """Overlay one arm's fusible init. Does not fit a source model or run adaptation."""
    key = stage_a_key_for(spec, frames)
    blocked = spec.backbone == "tcn" and spec.selection_rule != "none" and not tcn_source_fit_is_complete()
    if blocked:
        return ArmPreparation(
            spec=spec,
            status=BLOCKED_GAP_TCN_SOURCE_FIT,
            stage_a_key=key,
            fused=None,
            initialized=None,
            frozen_tensors=frozen_tensors(spec),
            source_validation_metric=NOT_APPLICABLE,
            source_validation_value_space=NOT_APPLICABLE,
        )
    if key is None:
        return ArmPreparation(
            spec=spec,
            status="READY",
            stage_a_key=None,
            fused=None,
            initialized=_copy_weights(template),
            frozen_tensors=frozen_tensors(spec),
            source_validation_metric=NOT_APPLICABLE,
            source_validation_value_space=NOT_APPLICABLE,
        )
    if bundle is None:
        raise ContractError("a sourced arm requires an existing Stage A bundle; this module does not train")
    if bundle.key != key:
        raise ContractError("Stage A bundle key does not match this arm; refusing to reuse another cache")
    if spec.weight_mode not in {"inverse_distance", "uniform", "single"}:
        raise ContractError(f"sourced arm weight mode is not frozen, got {spec.weight_mode!r}")
    fused = fuse_init(bundle, spec.weight_mode)
    channels = len(key.model_features)
    validate_fusible_weights(spec.backbone, fused, channels=channels)
    initialized = _overlay(template, fused, channels=channels, backbone=spec.backbone)
    return ArmPreparation(
        spec=spec,
        status="READY",
        stage_a_key=key,
        fused=fused,
        initialized=initialized,
        frozen_tensors=frozen_tensors(spec),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
    )


def _spec_from_definition(definition: _ArmDef, dataset: DatasetId, horizon: Horizon, seed: CellSeed) -> RunSpec:
    return RunSpec(
        arm=definition.arm,
        protocol_tag=definition.protocol_tag,
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        backbone=definition.backbone,
        revin=definition.revin,
        freeze_conv=definition.freeze_conv,
        selection_rule=definition.selection_rule,
        requested_k=definition.requested_k,
        weight_mode=definition.weight_mode,
        comparison_days=definition.comparison_days,
        target_learning_rate=target_learning_rate(dataset, definition.protocol_tag),
    )


def _overlay(
    template: Mapping[str, Sequence[np.ndarray]],
    fused: Mapping[str, Sequence[np.ndarray]],
    *,
    channels: int,
    backbone: Backbone,
) -> dict[str, list[np.ndarray]]:
    validate_fusible_weights(backbone, fused, channels=channels)
    out = _copy_weights(template)
    for name, tensors in fused.items():
        if name not in out:
            raise ContractError(f"template is missing fusible parameter {name}")
        if [tuple(np.asarray(array).shape) for array in out[name]] != [tuple(np.asarray(array).shape) for array in tensors]:
            raise ContractError(f"template and fused shapes disagree for {name}")
        out[name] = [np.array(array, copy=True) for array in tensors]
    return out


def _copy_weights(weights: Mapping[str, Sequence[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {name: [np.array(array, copy=True) for array in tensors] for name, tensors in weights.items()}
