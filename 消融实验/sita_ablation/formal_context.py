"""Assemble in-memory formal-run inputs after preflight has already passed.

This module is the only place the CLI loads sealed tables. It reuses the
existing frame, window, scaler, ranking, Stage A, and adaptation functions.
D5 history is rebuilt later by ``rank_candidates``, not here. Nothing in this
module reads or writes a cell result file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from sita_ablation.adaptation import TargetWindows, WeightSnapshot, adapt_target
from sita_ablation.arms import RunSpec, stage_a_key_for
from sita_ablation.data import (
    ConditionFrames,
    ContractError,
    load_condition_frames,
    observed_through_test,
    partition_rows,
)
from sita_ablation.io import inspect_done, write_cell_failed, write_run_atomically
from sita_ablation.metrics import evaluate_test
from sita_ablation.models import build_target_cnn, build_tcn
from sita_ablation.revin import build_revin
from sita_ablation.scaling import fit_target_model_scaler, transform_model_frame
from sita_ablation.source_selection import inverse_distance_weights, rank_candidates, uniform_weights
from sita_ablation.stage_a import StageABundle, load_or_train_stage_a
from sita_ablation.windows import build_target_partition_windows, build_test_windows

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_STAGE_A_CACHE = _PACKAGE_ROOT / "cache" / "stage_a"
_LOGS_ROOT = _PACKAGE_ROOT / "logs"
_SELECTION_RULES = ("top", "bottom", "random3")


@dataclass(frozen=True)
class PreparedCondition:
    windows: TargetWindows
    target_scaler: object
    y_true: np.ndarray


def load_plan_frames(specs: Sequence[RunSpec]) -> dict[str, ConditionFrames]:
    """Load each planned dataset once. An empty plan is refused before any read."""
    if not specs:
        raise ContractError("formal context requires the planned cells before any sealed table is read")
    datasets: list[str] = []
    for spec in specs:
        if spec.dataset not in datasets:
            datasets.append(spec.dataset)
    return {dataset: load_condition_frames(dataset) for dataset in datasets}


def prepare_conditions(
    specs: Sequence[RunSpec],
    frames: dict[str, ConditionFrames],
) -> dict[tuple[str, int], PreparedCondition]:
    """Build target windows only for dataset/horizon pairs present in the plan."""
    prepared: dict[tuple[str, int], PreparedCondition] = {}
    for spec in specs:
        key = (spec.dataset, int(spec.horizon))
        if key not in prepared:
            if spec.dataset not in frames:
                raise ContractError(f"sealed frames for {spec.dataset} were not loaded")
            prepared[key] = _prepare_one(frames[spec.dataset], int(spec.horizon))
    return prepared


def assemble_formal_context(specs: Sequence[RunSpec]):
    """Return the cell context for an already accepted plan. Does not start a cell."""
    try:
        frames = load_plan_frames(specs)
        conditions = prepare_conditions(specs, frames)
        templates = _templates_for(specs, frames)
        return _context(frames, conditions, templates)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"formal context could not be assembled: {exc}") from exc


def _prepare_one(frames: ConditionFrames, horizon: int) -> PreparedCondition:
    features = tuple(frames.model_features)
    scaler = fit_target_model_scaler(partition_rows(frames, "train"), features)
    train = build_target_partition_windows(
        transform_model_frame(scaler, partition_rows(frames, "train")),
        key_fields=frames.key_fields,
        key=frames.target_key,
        feature_columns=features,
        horizon=horizon,
        partition="train",
    )
    validation = build_target_partition_windows(
        transform_model_frame(scaler, partition_rows(frames, "validation")),
        key_fields=frames.key_fields,
        key=frames.target_key,
        feature_columns=features,
        horizon=horizon,
        partition="validation",
    )
    test_start, test_end = frames.target_test.iso()
    test = build_test_windows(
        transform_model_frame(scaler, observed_through_test(frames)),
        test_start=test_start,
        test_end=test_end,
        feature_columns=features,
        horizon=horizon,
    )
    return PreparedCondition(
        windows=TargetWindows(train=train, validation=validation, test=test),
        target_scaler=scaler,
        y_true=np.array(test.y_original, dtype=np.float64, copy=True),
    )


def _templates_for(
    specs: Sequence[RunSpec],
    frames: dict[str, ConditionFrames],
) -> dict[tuple[str, int, bool, int], WeightSnapshot]:
    templates: dict[tuple[str, int, bool, int], WeightSnapshot] = {}
    for spec in specs:
        channels = 1 if spec.backbone == "tcn" else len(tuple(frames[spec.dataset].model_features))
        key = (spec.backbone, channels, bool(spec.revin), int(spec.seed))
        if key not in templates:
            templates[key] = _weight_snapshot(spec.backbone, channels, bool(spec.revin), int(spec.seed))
    return templates


def _weight_snapshot(backbone: str, channels: int, revin: bool, seed: int) -> WeightSnapshot:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    if backbone == "cnn":
        model, _extractor = build_target_cnn((10, channels))
        width = channels
    elif backbone == "tcn":
        model = build_tcn((10, 1))
        width = 1
    else:
        raise ContractError(f"unsupported backbone {backbone!r}")
    tensors: dict[str, tuple[np.ndarray, ...]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            tensors[layer.name] = tuple(np.asarray(array, dtype=np.float32) for array in arrays)
    if revin:
        affine = build_revin(width)
        affine.build((None, 10, width))
        tensors["affine_weight"] = (np.asarray(affine.affine_weight.numpy(), dtype=np.float32),)
        tensors["affine_bias"] = (np.asarray(affine.affine_bias.numpy(), dtype=np.float32),)
    return WeightSnapshot(tensors=tensors)


def _context(frames, conditions, templates):
    from sita_ablation.runner import CellContext

    alphas: dict[tuple[str, str, str, int, int], np.ndarray | None] = {}

    def select(spec: RunSpec):
        rule = spec.selection_rule
        if rule not in _SELECTION_RULES or spec.comparison_days not in (15, 30) or spec.requested_k is None:
            raise ContractError(f"{spec.arm} has no legal candidate rule to apply")
        return rank_candidates(
            spec.dataset,
            frames[spec.dataset],
            comparison_days=spec.comparison_days,
            rule=rule,
            k=int(spec.requested_k),
            seed=spec.seed,
        )

    def adapt(spec: RunSpec):
        condition = conditions[(spec.dataset, int(spec.horizon))]
        frame = frames[spec.dataset]
        channels = 1 if spec.backbone == "tcn" else len(tuple(frame.model_features))
        template = templates[(spec.backbone, channels, bool(spec.revin), int(spec.seed))]
        key = stage_a_key_for(spec, frame)
        bundle = None if key is None else load_or_train_stage_a(key, frame, cache_root=_STAGE_A_CACHE)
        result = adapt_target(spec, condition.windows, condition.target_scaler, template, bundle)
        alphas[_spec_key(spec)] = _alpha_for(spec, bundle)
        return result

    def evaluate(result):
        spec = result.spec
        condition = conditions[(spec.dataset, int(spec.horizon))]
        return evaluate_test(result, condition.y_true, alpha=alphas[_spec_key(spec)])

    windows = None
    scaler = None
    y_true = None
    template = None
    if len(conditions) == 1:
        condition = next(iter(conditions.values()))
        windows = condition.windows
        scaler = condition.target_scaler
        y_true = condition.y_true
    if len(templates) == 1:
        template = next(iter(templates.values()))
    context = CellContext(
        clock=lambda: datetime.now(timezone.utc),
        inspect=inspect_done,
        write_run=write_run_atomically,
        write_failed=write_cell_failed,
        logs_root=_LOGS_ROOT,
        adapt=adapt,
        evaluate=evaluate,
        select=select,
        windows=windows,
        target_scaler=scaler,
        init_template=template,
        y_true=y_true,
        frames=frames,
        conditions=conditions,
    )
    return context


def _alpha_for(spec: RunSpec, bundle: StageABundle | None) -> np.ndarray | None:
    if spec.selection_rule == "none" or bundle is None:
        return None
    if spec.weight_mode == "single":
        return np.array([1.0], dtype=np.float64)
    if spec.weight_mode == "uniform":
        return uniform_weights(len(bundle.sources))
    return inverse_distance_weights(bundle.distances)


def _spec_key(spec: RunSpec) -> tuple[str, str, str, int, int]:
    return (spec.arm, spec.protocol_tag, spec.dataset, int(spec.horizon), int(spec.seed))
