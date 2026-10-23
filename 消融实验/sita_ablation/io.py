"""Serialize one cell's frozen artifacts and resume by hash.

Schema v1. Writes predictions, ``cell_record.json``, and ``DONE.json`` for a
completed cell, or only ``selection.json`` and ``DONE.json`` when fewer than
K legal candidates remain. Does not train, adapt, rank sources, read sealed
tables, or retry a failed cell. Signed error and alpha entropy use the closed formulas.

``cell_record.json`` uses ``CELL_RECORD_KEYS``. ``n_train`` and
``n_validation`` are taken from the adaptation result, which reads the target
``WindowBatch``. ``top3_overlap_count`` is accepted only when the caller
already computed it. This module does not derive either value by calling
source selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sita_ablation.adaptation import (
    ABSENT,
    COMPLETED,
    DEFINED,
    PRESENT,
    UNDEFINED_ZERO_INIT_NORM,
    UNRESOLVED,
    AdaptationResult,
    ParameterShift,
)
from sita_ablation.arms import RunSpec, run_directory
from sita_ablation.data import ContractError, load_dataset_contract
from sita_ablation.metrics import ORIGINAL_SALES, TEST_DAYS, MetricPair, TestMetrics, signed_error as signed_error_value
from sita_ablation.source_selection import (
    STATUS_COMPLETED,
    STATUS_SKIPPED,
    SelectionResult,
)
from sita_ablation.windows import expected_target_partition_count
from sita_ablation.stage_a import (
    NOT_APPLICABLE,
    ResumeDecision,
    StageAKey,
    frozen_config_sha256,
    stage_a_key_sha256,
)

PREDICTIONS_FILE = "predictions.parquet"
CELL_RECORD_FILE = "cell_record.json"
SELECTION_FILE = "selection.json"
DONE_FILE = "DONE.json"
FAILED_FILE = "FAILED.json"
_EMA = "EMA"
_RAW = "RAW"
_PREDICTION_COLUMNS = ("label_date", "y_true_original", "y_pred_original", "weight_variant")
_COMPLETED_NAMES = {PREDICTIONS_FILE, CELL_RECORD_FILE, DONE_FILE, FAILED_FILE}
_SKIP_NAMES = {SELECTION_FILE, DONE_FILE}
_SPEC_FIELDS = (
    "arm",
    "protocol_tag",
    "dataset",
    "horizon",
    "seed",
    "backbone",
    "revin",
    "freeze_conv",
    "selection_rule",
    "requested_k",
    "weight_mode",
    "comparison_days",
    "target_learning_rate",
)
_PAIR_FIELDS = (
    "rmse",
    "smape",
    "zero_sales_count",
    "smape_zero_denominator_count",
    "clipped_true_negative_count",
    "clipped_pred_negative_count",
    "value_space",
)
_CNN_SHIFT_GROUPS = ("conv1", "conv2", "conv3", "head")
SCHEMA_VERSION = 1
PREDICTION_PATH = PREDICTIONS_FILE
RUN_SPEC_KEYS = _SPEC_FIELDS
PAIR_KEYS = _PAIR_FIELDS
SHIFT_GROUP_KEYS = ("absolute", "relative", "status")
SOURCE_INFO_KEYS = (
    "alpha_entropy",
    "distances",
    "effective_source_count",
    "inverse_distance_weights",
    "ranks",
    "selected_keys",
    "source_validation_metric",
    "source_validation_value_space",
    "uniform_weights",
)
CELL_RECORD_KEYS = (
    "alpha_entropy",
    "artifact_sha256",
    "config_digest",
    "elapsed_seconds",
    "ema_shift",
    "epoch0_val_rmse",
    "finished_at",
    "frozen_config_sha256",
    "git_head",
    "gradient_steps",
    "n_test",
    "n_train",
    "n_validation",
    "pair",
    "prediction_path",
    "raw_shift",
    "run_spec",
    "signed_error",
    "source_info",
    "stage_a_key_sha256",
    "started_at",
    "status",
    "tensorflow_version",
    "test_weight_variant",
    "train_loss",
    "unscored_test_dates",
    "val_loss",
    "val_rmse",
    "val_smape",
    "validation_rmse",
    "validation_smape",
    "validation_weight_variant",
    "workspace_dirty",
)
PHASE_AUDIT_CATEGORIES = ("missing", "hash_mismatch", "failed", "insufficient_candidates")
CELL_FAILED_KEYS = ("attempt_count",)
_OVERLAP_TAGS = {"random3", "bottom3"}
_OVERLAP_RULES = {"random3", "bottom"}
_REVIN0_LOG_ARMS = {"P1", "P2", "P3"}
_CELL_ID_RE = re.compile(
    r"^(?:F1|F2|F3|F4|P1|P2|P3|SITA_TCN|TCN_REVIN):"
    r"(?:main|alt_lr|random3|bottom3|k5|k10|ell15):"
    r"(?:D1|D4|D5|D6):h(?:1|3|5):seed(?:42|43|44|45|46)$"
)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_PROTOCOL_PATH = _PACKAGE_ROOT / "configs" / "protocol_frozen.json"


@dataclass(frozen=True)
class RunPayload:
    """One cell to serialize. The caller supplies the result objects."""

    spec: RunSpec
    started_at: datetime
    finished_at: datetime
    adaptation: AdaptationResult | None = None
    metrics: TestMetrics | None = None
    stage_a_key: StageAKey | None = None
    selection: SelectionResult | None = None
    top3_overlap_count: int | None = None


@dataclass(frozen=True)
class DoneRecord:
    run_spec: RunSpec
    status: str
    artifact_sha256: dict[str, str]
    stage_a_key_sha256: str | None
    started_at: str
    finished_at: str
    elapsed_seconds: float


def log_path(spec: RunSpec, root: Path | None = None) -> Path:
    """Log path. No-RevIN P1/P2/P3 names end in ``__revin0``. The file is not created here."""
    base = Path("消融实验/logs") if root is None else Path(root)
    name = f"{spec.arm}__{spec.protocol_tag}__{spec.dataset}_h{spec.horizon}_seed{spec.seed}"
    if spec.arm in _REVIN0_LOG_ARMS and spec.revin is False:
        name = f"{name}__revin0"
    return base / f"{name}.log"


def config_digest(frozen_hash: str, spec: RunSpec) -> str:
    """SHA256 of the frozen-config hash plus the 13-field run spec.

    ``git_head`` is not an input. The same pair of inputs always hashes to the
    same digest.
    """
    if type(frozen_hash) is not str or len(frozen_hash) != 64:
        raise ContractError("frozen_config_sha256 is not a sha256 digest")
    if any(character not in "0123456789abcdef" for character in frozen_hash):
        raise ContractError("frozen_config_sha256 is not a sha256 digest")
    identity = {
        "frozen_config_sha256": frozen_hash,
        "run_spec": _spec_payload(spec),
    }
    if set(identity) != {"frozen_config_sha256", "run_spec"} or set(identity["run_spec"]) != set(RUN_SPEC_KEYS):
        raise ContractError("config_digest identity is not the frozen pair")
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    """UTF-8 canonical JSON: recursive key sort, compact separators, no NaN."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    return text.encode("utf-8")


def require_clean_workspace_for_formal_run(workspace_dirty: bool) -> None:
    """Refuse a formal run when the worktree is dirty.

    This is the workspace flag itself. It is not a code-diff digest.
    """
    if type(workspace_dirty) is not bool:
        raise ContractError("workspace_dirty must stay a boolean")
    if workspace_dirty:
        raise ContractError("formal run refused because workspace_dirty is true")


def phase_audit_path(phase: int) -> Path:
    """Production audit path ``消融实验/audit/phase_<phase>_audit.json``."""
    _require_phase(phase)
    return Path("消融实验") / "audit" / f"phase_{phase}_audit.json"


def phase_horizon_audit_path(phase: int, horizon: int) -> Path:
    """Audit for one horizon shard. It does not replace ``phase_<phase>_audit.json``."""
    _require_phase(phase)
    if type(horizon) is not int or horizon not in (1, 3, 5):
        raise ContractError("only-horizon must be 1, 3, or 5")
    return Path("消融实验") / "audit" / f"phase_{phase}_h{horizon}_audit.json"


def cell_id(spec: RunSpec) -> str:
    """Frozen audit id ``<arm>:<protocol_tag>:<dataset>:h<horizon>:seed<seed>``."""
    text = f"{spec.arm}:{spec.protocol_tag}:{spec.dataset}:h{spec.horizon}:seed{spec.seed}"
    _require_cell_id(text)
    return text


def write_cell_failed(directory: Path, attempt_count: int) -> None:
    """Write a target-cell ``FAILED.json`` whose only field is ``attempt_count``.

    Exception text stays in the cell log. This is not the Stage A cache failure
    file, which keeps its own ``stage_a_key_sha256`` and ``error`` fields.
    """
    count = _cell_failed_attempt({"attempt_count": attempt_count})
    _atomic_json(Path(directory) / FAILED_FILE, {"attempt_count": count})


def write_run_atomically(directory: Path, payload: RunPayload) -> DoneRecord:
    """Write a cell, or return the existing record when its frozen hashes match.

    ``DONE.json`` is renamed into place only after the required files are on
    disk and check out. A hash or identity mismatch leaves every existing
    file untouched.
    """
    directory = Path(directory)
    if not _directory_matches(directory, payload.spec):
        raise ContractError("run directory does not match run_directory")
    if directory.exists() and ((directory / DONE_FILE).exists() or (directory / FAILED_FILE).exists()):
        decision = inspect_done(directory)
        if decision == "SKIP":
            record = _read_done(directory)
            if _same_spec(record.run_spec, payload.spec):
                return record
            raise ContractError("HOLD_HASH_MISMATCH")
        if decision == "HOLD_HASH_MISMATCH":
            raise ContractError("HOLD_HASH_MISMATCH")
        if decision == "HOLD_FAILED":
            raise ContractError("HOLD_FAILED")
        if decision != "RETRY_ONCE":
            raise ContractError(f"cannot write a run from resume decision {decision}")
    if payload.selection is not None and payload.selection.status == STATUS_SKIPPED:
        return _write_skipped(directory, payload)
    return _write_completed(directory, payload)


def inspect_done(directory: Path) -> ResumeDecision:
    """Read resume state. This function does not create, replace, or delete files."""
    directory = Path(directory)
    done_path = directory / DONE_FILE
    if done_path.exists():
        payload = _read_json(done_path)
        if payload is None or not _done_matches(directory, payload):
            return "HOLD_HASH_MISMATCH"
        return "SKIP"
    failed_path = directory / FAILED_FILE
    if failed_path.exists():
        payload = _read_json(failed_path)
        if payload is None:
            return "HOLD_FAILED"
        count = _cell_failed_attempt(payload)
        if count == 1:
            return "RETRY_ONCE"
        return "HOLD_FAILED"
    return "RUN"


def read_cell_record(directory: Path) -> dict[str, object]:
    """Return ``cell_record.json`` with its status strings unchanged."""
    payload = _read_json(Path(directory) / CELL_RECORD_FILE)
    if payload is None:
        raise ContractError("cell_record.json is missing or unreadable")
    _require_cell_identity(payload)
    return payload


def read_predictions(directory: Path) -> pa.Table:
    """Return the prediction table without clipping or re-rounding values."""
    path = Path(directory) / PREDICTIONS_FILE
    if not path.is_file():
        raise ContractError("predictions.parquet is missing")
    table = pq.read_table(path)
    _require_prediction_schema(table)
    return table


def write_phase_audit(
    path: Path,
    *,
    phase: int,
    missing: Sequence[str] = (),
    hash_mismatch: Sequence[str] = (),
    failed: Sequence[str] = (),
    insufficient_candidates: Sequence[str] = (),
) -> None:
    """Write the four audit categories. Rows that were not supplied stay absent."""
    _require_phase(phase)
    destination = Path(path)
    allowed = {phase_audit_path(phase).name}
    allowed.update(phase_horizon_audit_path(phase, horizon).name for horizon in (1, 3, 5))
    if destination.name not in allowed:
        raise ContractError("phase audit file name must be phase_<phase>_audit.json")
    seen: set[str] = set()
    payload = {
        "missing": _audit_ids(missing, seen),
        "hash_mismatch": _audit_ids(hash_mismatch, seen),
        "failed": _audit_ids(failed, seen),
        "insufficient_candidates": _audit_ids(insufficient_candidates, seen),
    }
    if set(payload) != set(PHASE_AUDIT_CATEGORIES):
        raise ContractError("phase audit categories are not the frozen set")
    _atomic_json(destination, payload)


def _write_completed(directory: Path, payload: RunPayload) -> DoneRecord:
    _require_completed_payload(payload)
    blocking = _blocking_names(directory, _COMPLETED_NAMES)
    if blocking:
        raise ContractError(f"completed cell directory has unexpected files {sorted(blocking)}")
    started_at, finished_at, elapsed = _times(payload)
    version = _version_fields()
    frame = _prediction_table(payload)
    record = _cell_body(payload, version, started_at, finished_at, elapsed, predictions_sha256=None)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_table(directory / PREDICTIONS_FILE, frame)
    predictions_sha256 = _sha256(directory / PREDICTIONS_FILE)
    record["artifact_sha256"] = {PREDICTIONS_FILE: predictions_sha256}
    _require_cell_identity(record)
    _atomic_json(directory / CELL_RECORD_FILE, record)
    _require_written_cell(directory, payload.spec, record)
    _require_prediction_bytes(directory, payload)
    done = _done_body(
        payload,
        status=COMPLETED,
        artifacts={
            PREDICTIONS_FILE: predictions_sha256,
            CELL_RECORD_FILE: _sha256(directory / CELL_RECORD_FILE),
        },
        started_at=started_at,
        finished_at=finished_at,
        elapsed=elapsed,
    )
    _commit_done(directory, done)
    return _read_done(directory)


def _write_skipped(directory: Path, payload: RunPayload) -> DoneRecord:
    if payload.adaptation is not None or payload.metrics is not None:
        raise ContractError("an insufficient-candidate cell has no predictions or metrics")
    if payload.top3_overlap_count is not None:
        raise ContractError("an insufficient-candidate cell does not record top3_overlap_count")
    selection = payload.selection
    if selection is None or selection.status != STATUS_SKIPPED or selection.selected:
        raise ContractError("a skipped cell requires an empty skipped selection")
    if payload.spec.selection_rule == "none":
        raise ContractError("a source-free arm cannot be skipped for candidates")
    _require_stage_key(payload)
    if selection.dataset != payload.spec.dataset or selection.requested_k != payload.spec.requested_k:
        raise ContractError("skipped selection does not match the run spec")
    if selection.rule != payload.spec.selection_rule or selection.seed != payload.spec.seed:
        raise ContractError("skipped selection does not match the run spec")
    blocking = _blocking_names(directory, _SKIP_NAMES)
    if blocking:
        raise ContractError(f"skipped cell directory has unexpected files {sorted(blocking)}")
    started_at, finished_at, elapsed = _times(payload)
    body = {
        "status": STATUS_SKIPPED,
        "selected_keys": [],
        "ranks": [],
        "distances": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / SELECTION_FILE, body)
    written = _read_json(directory / SELECTION_FILE)
    if written != body:
        raise ContractError("selection.json did not round-trip")
    done = _done_body(
        payload,
        status=STATUS_SKIPPED,
        artifacts={SELECTION_FILE: _sha256(directory / SELECTION_FILE)},
        started_at=started_at,
        finished_at=finished_at,
        elapsed=elapsed,
    )
    _commit_done(directory, done)
    names = _listed_names(directory)
    if names != {SELECTION_FILE, DONE_FILE}:
        raise ContractError(f"skipped cell wrote unexpected files {sorted(names)}")
    return _read_done(directory)


def _require_completed_payload(payload: RunPayload) -> None:
    adaptation = payload.adaptation
    metrics = payload.metrics
    if adaptation is None or metrics is None:
        raise ContractError("a completed cell requires adaptation and test metrics")
    if payload.selection is not None and payload.selection.status == STATUS_SKIPPED:
        raise ContractError("a skipped selection cannot be written as a completed cell")
    if adaptation.status != COMPLETED or adaptation.spec != payload.spec:
        raise ContractError("adaptation result does not match the completed run spec")
    if metrics.dataset != payload.spec.dataset or metrics.n_test != TEST_DAYS:
        raise ContractError("test metrics do not cover the frozen 180 target days")
    if metrics.test_weight_variant != _EMA or adaptation.test_weight_variant != _EMA:
        raise ContractError("test predictions must be marked EMA")
    if adaptation.validation_weight_variant != _RAW:
        raise ContractError("recorded validation must stay on RAW weights")
    if adaptation.checkpoint_selection:
        raise ContractError("target adaptation does not select a checkpoint")
    _require_same_score(adaptation.signed_error, metrics.signed_error, "signed error", allow_null=False)
    _require_same_score(
        adaptation.alpha_entropy,
        metrics.alpha_entropy,
        "alpha entropy",
        allow_null=payload.spec.selection_rule == "none",
    )
    if adaptation.source_info != metrics.source_info:
        raise ContractError("source info disagrees between adaptation and metrics")
    if adaptation.unscored_test_dates or metrics.unscored_test_dates:
        raise ContractError("unscored test dates must stay an empty list")
    _require_source_payload(payload)
    _require_curves(adaptation)
    _require_shifts(payload.spec, adaptation.raw_shift, adaptation.ema_shift)
    _require_sales_alignment(payload)
    if adaptation.validation_rmse != adaptation.val_rmse[-1]:
        raise ContractError("validation RMSE is the last RAW epoch")
    if adaptation.validation_smape != adaptation.val_smape[-1]:
        raise ContractError("validation sMAPE is the last RAW epoch")
    if metrics.effective_source_count != adaptation.effective_source_count:
        raise ContractError("effective source count disagrees between adaptation and metrics")
    _require_recorded_sample_counts(payload.spec, adaptation)


def _require_source_payload(payload: RunPayload) -> None:
    spec = payload.spec
    adaptation = payload.adaptation
    metrics = payload.metrics
    assert adaptation is not None and metrics is not None
    if spec.selection_rule == "none":
        if adaptation.source_info != ABSENT or metrics.source_info != ABSENT:
            raise ContractError("a source-free arm's source info is ABSENT")
        if payload.stage_a_key is not None or payload.selection is not None:
            raise ContractError("a source-free arm has no Stage A key and no selection")
        if adaptation.source_validation_metric is not None or adaptation.source_validation_value_space is not None:
            raise ContractError("a source-free arm does not record a source-validation metric")
        if adaptation.effective_source_count is not None:
            raise ContractError("a source-free arm has no effective source count")
        return
    if adaptation.source_info != PRESENT:
        raise ContractError("a sourced arm's source info is PRESENT")
    if adaptation.source_validation_metric != NOT_APPLICABLE or adaptation.source_validation_value_space != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    if metrics.source_validation_metric != NOT_APPLICABLE or metrics.source_validation_value_space != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    selection = payload.selection
    if selection is None or selection.status != STATUS_COMPLETED:
        raise ContractError("a sourced cell requires its completed selection")
    if selection.dataset != spec.dataset or selection.rule != spec.selection_rule:
        raise ContractError("selection does not match the run spec")
    if selection.requested_k != spec.requested_k or selection.seed != spec.seed:
        raise ContractError("selection does not match the run spec")
    if selection.comparison_days != spec.comparison_days:
        raise ContractError("selection comparison window does not match the run spec")
    _require_stage_key(payload)
    key = payload.stage_a_key
    assert key is not None
    if selection.comparison_start != key.comparison_start or selection.comparison_end != key.comparison_end:
        raise ContractError("selection dates do not match the Stage A key")


def _require_stage_key(payload: RunPayload) -> None:
    spec = payload.spec
    key = payload.stage_a_key
    if spec.selection_rule == "none" or key is None:
        raise ContractError("a sourced arm requires its Stage A key")
    if key.frozen_config_sha256 != frozen_config_sha256():
        raise ContractError("Stage A key frozen-config hash does not match the current files")
    if (key.dataset, key.horizon, key.seed, key.backbone) != (spec.dataset, spec.horizon, spec.seed, spec.backbone):
        raise ContractError("Stage A key does not match the run spec")
    span = _span_days(key.comparison_start, key.comparison_end)
    if spec.weight_mode == "single":
        if key.selection_rule != "top" or key.trained_k != 3 or span != 30:
            raise ContractError("P2 reuses the Top-3 Stage A key")
        return
    if key.selection_rule != spec.selection_rule or key.trained_k != spec.requested_k or span != spec.comparison_days:
        raise ContractError("Stage A key does not match the arm's selection")


def _require_curves(adaptation: AdaptationResult) -> None:
    epochs = _target_epochs()
    curves = {
        "train_loss": adaptation.train_loss,
        "val_loss": adaptation.val_loss,
        "val_rmse": adaptation.val_rmse,
        "val_smape": adaptation.val_smape,
    }
    for name, values in curves.items():
        if len(values) != epochs:
            raise ContractError(f"{name} must contain the frozen {epochs} epochs")
        for value in values:
            _finite_float(value, name)
    _finite_float(adaptation.epoch0_val_rmse, "epoch0_val_rmse")
    if type(adaptation.gradient_steps) is not int or isinstance(adaptation.gradient_steps, bool):
        raise ContractError("gradient_steps must stay an integer")
    if adaptation.gradient_steps < 1:
        raise ContractError("gradient_steps must stay an integer")


def _require_shifts(
    spec: RunSpec,
    raw_shift: Mapping[str, ParameterShift],
    ema_shift: Mapping[str, ParameterShift],
) -> None:
    if set(raw_shift) != set(ema_shift):
        raise ContractError("RAW and EMA parameter shifts name different groups")
    if "head" not in raw_shift:
        raise ContractError("parameter shifts are missing the prediction head")
    if spec.revin != ("revin" in raw_shift):
        raise ContractError("RevIN shift presence does not match the arm")
    if spec.backbone == "cnn" and any(name not in raw_shift for name in _CNN_SHIFT_GROUPS):
        raise ContractError("CNN parameter shifts are missing conv1, conv2, conv3, or the head")
    for group in raw_shift.values():
        _require_shift(group)
    for group in ema_shift.values():
        _require_shift(group)


def _require_shift(group: ParameterShift) -> None:
    _finite_float(group.absolute, "absolute shift")
    if group.status == DEFINED:
        _finite_float(group.relative, "relative shift")
        return
    if group.status == UNDEFINED_ZERO_INIT_NORM and group.relative is None:
        return
    raise ContractError("a zero initial norm keeps a null relative shift and UNDEFINED_ZERO_INIT_NORM")


def _require_sales_alignment(payload: RunPayload) -> None:
    adaptation = payload.adaptation
    metrics = payload.metrics
    assert adaptation is not None and metrics is not None
    predicted = np.asarray(adaptation.test_predictions, dtype=np.float64)
    observed = np.asarray(metrics.y_pred, dtype=np.float64)
    truth = np.asarray(metrics.y_true, dtype=np.float64)
    if predicted.shape != (TEST_DAYS,) or observed.shape != (TEST_DAYS,) or truth.shape != (TEST_DAYS,):
        raise ContractError("test sales vectors must have length 180")
    if not np.array_equal(predicted, observed):
        raise ContractError("adaptation predictions and test metrics are not the same values")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ContractError("test sales contain a non-finite value")
    dates = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in metrics.label_date)
    contract = load_dataset_contract(payload.spec.dataset)
    expected = tuple(
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")
    )
    if dates != expected:
        raise ContractError("test dates are not the frozen 180-day interval")


def _cell_body(
    payload: RunPayload,
    version: Mapping[str, object],
    started_at: str,
    finished_at: str,
    elapsed: float,
    *,
    predictions_sha256: str | None,
) -> dict[str, object]:
    adaptation = payload.adaptation
    metrics = payload.metrics
    assert adaptation is not None and metrics is not None
    artifacts = {} if predictions_sha256 is None else {PREDICTIONS_FILE: predictions_sha256}
    frozen_hash = version["frozen_config_sha256"]
    if type(frozen_hash) is not str:
        raise ContractError("frozen config hash does not match the current files")
    n_train, n_validation = _require_recorded_sample_counts(payload.spec, adaptation)
    record = {
        "run_spec": _spec_payload(payload.spec),
        "status": COMPLETED,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "git_head": version["git_head"],
        "workspace_dirty": version["workspace_dirty"],
        "frozen_config_sha256": frozen_hash,
        "config_digest": config_digest(frozen_hash, payload.spec),
        "tensorflow_version": version["tensorflow_version"],
        "pair": _pair_payload(metrics.pair),
        "validation_rmse": _finite_float(adaptation.validation_rmse, "validation_rmse"),
        "validation_smape": _finite_float(adaptation.validation_smape, "validation_smape"),
        "validation_weight_variant": _RAW,
        "train_loss": _float_list(adaptation.train_loss, "train_loss"),
        "val_loss": _float_list(adaptation.val_loss, "val_loss"),
        "val_rmse": _float_list(adaptation.val_rmse, "val_rmse"),
        "val_smape": _float_list(adaptation.val_smape, "val_smape"),
        "epoch0_val_rmse": _finite_float(adaptation.epoch0_val_rmse, "epoch0_val_rmse"),
        "gradient_steps": adaptation.gradient_steps,
        "raw_shift": _shift_payload(adaptation.raw_shift),
        "ema_shift": _shift_payload(adaptation.ema_shift),
        "signed_error": _finite_float(metrics.signed_error, "signed_error"),
        "alpha_entropy": None if metrics.alpha_entropy is None else _finite_float(metrics.alpha_entropy, "alpha_entropy"),
        "source_info": _source_payload(payload),
        "unscored_test_dates": [],
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": TEST_DAYS,
        "test_weight_variant": _EMA,
        "prediction_path": PREDICTION_PATH,
        "artifact_sha256": artifacts,
        "stage_a_key_sha256": None if payload.stage_a_key is None else stage_a_key_sha256(payload.stage_a_key),
    }
    if set(record) != set(CELL_RECORD_KEYS):
        raise ContractError("cell_record.json keys are not the frozen schema v1 set")
    return record


def _source_payload(payload: RunPayload) -> str | dict[str, object]:
    if payload.spec.selection_rule == "none":
        return ABSENT
    selection = payload.selection
    adaptation = payload.adaptation
    assert selection is not None and adaptation is not None
    count = adaptation.effective_source_count
    if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(float(count)):
        raise ContractError("effective source count must stay the computed finite value")
    body: dict[str, object] = {
        "source_validation_metric": NOT_APPLICABLE,
        "source_validation_value_space": NOT_APPLICABLE,
        "alpha_entropy": _finite_float(adaptation.alpha_entropy, "alpha_entropy"),
        "effective_source_count": _finite_float(count, "effective_source_count"),
        "selected_keys": [list(item.source_key) for item in selection.selected],
        "ranks": [int(item.rank) for item in selection.selected],
        "distances": _float_list([item.distance for item in selection.selected], "distances"),
        "inverse_distance_weights": _float_list(selection.inverse_distance_weights, "inverse_distance_weights"),
        "uniform_weights": _float_list(selection.uniform_weights, "uniform_weights"),
    }
    overlap = payload.top3_overlap_count
    if _overlap_cell(payload.spec) and payload.spec.comparison_days == 30 and payload.spec.requested_k == 3:
        if overlap is not None:
            body["top3_overlap_count"] = _require_overlap_value(overlap)
    elif overlap is not None:
        raise ContractError("top3_overlap_count is only recorded for completed random3 and bottom3 cells")
    if not set(SOURCE_INFO_KEYS) <= set(body):
        raise ContractError("source info is missing a frozen field")
    return body


def _done_body(
    payload: RunPayload,
    *,
    status: str,
    artifacts: Mapping[str, str],
    started_at: str,
    finished_at: str,
    elapsed: float,
) -> dict[str, object]:
    if status not in {COMPLETED, STATUS_SKIPPED}:
        raise ContractError(f"run status must be completed or skipped, got {status!r}")
    key_hash = None if payload.stage_a_key is None else stage_a_key_sha256(payload.stage_a_key)
    if payload.spec.selection_rule == "none":
        key_hash = None
    return {
        "run_spec": _spec_payload(payload.spec),
        "status": status,
        "artifact_sha256": dict(artifacts),
        "stage_a_key_sha256": key_hash,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
    }


def _prediction_table(payload: RunPayload) -> pa.Table:
    metrics = payload.metrics
    assert metrics is not None
    dates = [pd.Timestamp(day).date() for day in metrics.label_date]
    true = np.array(metrics.y_true, dtype=np.float64, copy=True)
    pred = np.array(metrics.y_pred, dtype=np.float64, copy=True)
    return pa.table(
        {
            "label_date": pa.array(dates, type=pa.date32()),
            "y_true_original": pa.array(true, type=pa.float64()),
            "y_pred_original": pa.array(pred, type=pa.float64()),
            "weight_variant": pa.array([_EMA] * TEST_DAYS, type=pa.string()),
        }
    )


def _require_written_cell(directory: Path, spec: RunSpec, record: dict[str, object]) -> None:
    disk = _read_json(directory / CELL_RECORD_FILE)
    if disk != record:
        raise ContractError("cell_record.json did not round-trip")
    table = read_predictions(directory)
    _require_prediction_values(table, record)
    if disk is None or disk.get("stage_a_key_sha256") != record.get("stage_a_key_sha256"):
        raise ContractError("stage_a_key_sha256 was not preserved")
    if spec.selection_rule == "none" and disk.get("source_info") != ABSENT:
        raise ContractError("a source-free arm's source info is ABSENT")
    artifacts = disk.get("artifact_sha256")
    if not isinstance(artifacts, dict) or artifacts.get(PREDICTIONS_FILE) != _sha256(directory / PREDICTIONS_FILE):
        raise ContractError("prediction hash does not match predictions.parquet")


def _done_matches(directory: Path, payload: dict[str, object]) -> bool:
    try:
        spec = _spec_from_payload(payload.get("run_spec"))
    except (ContractError, TypeError, ValueError):
        return False
    if not _directory_matches(directory, spec):
        return False
    status = payload.get("status")
    if status not in {COMPLETED, STATUS_SKIPPED}:
        return False
    artifacts = payload.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for name, digest in artifacts.items():
        path = directory / str(name)
        if not isinstance(digest, str) or not path.is_file() or _sha256(path) != digest:
            return False
    key_hash = payload.get("stage_a_key_sha256")
    if spec.selection_rule == "none":
        if key_hash is not None:
            return False
    elif not isinstance(key_hash, str) or len(key_hash) != 64:
        return False
    try:
        _require_timestamp_pair(payload.get("started_at"), payload.get("finished_at"), payload.get("elapsed_seconds"))
    except ContractError:
        return False
    if status == STATUS_SKIPPED:
        return _skipped_matches(directory, spec, artifacts)
    if set(artifacts) != {PREDICTIONS_FILE, CELL_RECORD_FILE}:
        return False
    return _completed_matches(directory, spec, payload, artifacts)


def _skipped_matches(directory: Path, spec: RunSpec, artifacts: Mapping[object, object]) -> bool:
    if spec.selection_rule == "none" or set(artifacts) != {SELECTION_FILE}:
        return False
    if _listed_names(directory) != {SELECTION_FILE, DONE_FILE}:
        return False
    selection = _read_json(directory / SELECTION_FILE)
    if selection is None:
        return False
    return (
        selection.get("status") == STATUS_SKIPPED
        and selection.get("selected_keys") == []
        and selection.get("ranks") == []
        and selection.get("distances") == []
        and set(selection) == {"status", "selected_keys", "ranks", "distances"}
    )


def _completed_matches(
    directory: Path,
    spec: RunSpec,
    done: Mapping[str, object],
    artifacts: Mapping[object, object],
) -> bool:
    record = _read_json(directory / CELL_RECORD_FILE)
    if record is None:
        return False
    try:
        _require_cell_identity(record)
        record_spec = _spec_from_payload(record.get("run_spec"))
    except (ContractError, TypeError, ValueError):
        return False
    if not _same_spec(record_spec, spec):
        return False
    if record.get("status") != COMPLETED or record.get("started_at") != done.get("started_at"):
        return False
    if record.get("finished_at") != done.get("finished_at") or record.get("elapsed_seconds") != done.get("elapsed_seconds"):
        return False
    if record.get("stage_a_key_sha256") != done.get("stage_a_key_sha256"):
        return False
    recorded = record.get("artifact_sha256")
    if not isinstance(recorded, dict) or recorded.get(PREDICTIONS_FILE) != artifacts.get(PREDICTIONS_FILE):
        return False
    try:
        table = read_predictions(directory)
        _require_prediction_values(table, record)
    except (ContractError, OSError, pa.ArrowException, ValueError):
        return False
    return True


def _require_cell_identity(record: Mapping[str, object]) -> None:
    if set(record) != set(CELL_RECORD_KEYS):
        raise ContractError("cell_record.json keys are not the frozen schema v1 set")
    spec = _spec_from_payload(record.get("run_spec"))
    if record.get("status") != COMPLETED:
        raise ContractError("cell record status is not completed")
    _finite_float(record.get("signed_error"), "signed_error")
    if spec.selection_rule == "none":
        if record.get("alpha_entropy") is not None:
            raise ContractError("a source-free arm has no alpha entropy")
    else:
        _finite_float(record.get("alpha_entropy"), "alpha_entropy")
    if record.get("unscored_test_dates") != [] or record.get("n_test") != TEST_DAYS:
        raise ContractError("test coverage is not the frozen 180 days")
    if record.get("test_weight_variant") != _EMA or record.get("validation_weight_variant") != _RAW:
        raise ContractError("weight variants do not match the frozen RAW and EMA rules")
    pair = record.get("pair")
    if not isinstance(pair, dict) or set(pair) != set(PAIR_KEYS) or pair.get("value_space") != ORIGINAL_SALES:
        raise ContractError("test metrics are in original sales")
    _finite_float(pair.get("rmse"), "rmse")
    _finite_float(pair.get("smape"), "smape")
    for name in _PAIR_FIELDS[2:-1]:
        if type(pair.get(name)) is not int or isinstance(pair.get(name), bool) or int(pair[name]) < 0:
            raise ContractError(f"{name} must stay a non-negative count")
    _require_disk_shifts(spec, record.get("raw_shift"))
    _require_disk_shifts(spec, record.get("ema_shift"))
    _require_disk_source(spec, record.get("source_info"))
    for name in ("train_loss", "val_loss", "val_rmse", "val_smape"):
        _float_list(record.get(name), name)
    _finite_float(record.get("validation_rmse"), "validation_rmse")
    _finite_float(record.get("validation_smape"), "validation_smape")
    _finite_float(record.get("epoch0_val_rmse"), "epoch0_val_rmse")
    if type(record.get("git_head")) is not str or type(record.get("tensorflow_version")) is not str:
        raise ContractError("version fields are missing")
    if type(record.get("workspace_dirty")) is not bool:
        raise ContractError("workspace_dirty must stay a boolean")
    if record.get("frozen_config_sha256") != frozen_config_sha256():
        raise ContractError("frozen config hash does not match the current files")
    if record.get("config_digest") != config_digest(str(record.get("frozen_config_sha256")), spec):
        raise ContractError("config_digest does not match the frozen config and run spec")
    if record.get("prediction_path") != PREDICTION_PATH or Path(str(record.get("prediction_path"))).is_absolute():
        raise ContractError("prediction_path must be predictions.parquet relative to the run directory")
    _require_recorded_counts(spec.horizon, record.get("n_train"), record.get("n_validation"))
    _require_timestamp_pair(record.get("started_at"), record.get("finished_at"), record.get("elapsed_seconds"))
    artifacts = record.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {PREDICTIONS_FILE}:
        raise ContractError("cell record artifact hash contains only predictions.parquet")
    if "raw_shift" not in record or "ema_shift" not in record:
        raise ContractError("RAW and EMA parameter shifts are both required")


def _require_disk_source(spec: RunSpec, source_info: object) -> None:
    if spec.selection_rule == "none":
        if source_info != ABSENT or type(source_info) is not str:
            raise ContractError("a source-free arm's source info is ABSENT")
        return
    if not isinstance(source_info, dict):
        raise ContractError("a sourced arm records a source-info block")
    keys = set(source_info)
    if not set(SOURCE_INFO_KEYS) <= keys:
        raise ContractError("source info is missing a frozen field")
    extra = keys - set(SOURCE_INFO_KEYS)
    if _overlap_cell(spec):
        if extra not in (set(), {"top3_overlap_count"}):
            raise ContractError("source info keys are not the frozen set")
        if "top3_overlap_count" in source_info:
            if spec.comparison_days != 30 or spec.requested_k != 3:
                raise ContractError("top3 overlap is only defined for K=3 and a 30-day window")
            _require_overlap_value(source_info.get("top3_overlap_count"))
    elif extra:
        raise ContractError("source info keys are not the frozen set")
    if source_info.get("source_validation_metric") != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    if source_info.get("source_validation_value_space") != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    if type(source_info.get("source_validation_metric")) is not str or type(source_info.get("source_validation_value_space")) is not str:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    _finite_float(source_info.get("alpha_entropy"), "alpha_entropy")
    _finite_float(source_info.get("effective_source_count"), "effective_source_count")
    if source_info.get("source_info") in {ABSENT, NOT_APPLICABLE, UNRESOLVED, ""}:
        raise ContractError("source info must be ABSENT or the sourced block")


def _require_disk_shifts(spec: RunSpec, shifts: object) -> None:
    if not isinstance(shifts, dict) or "head" not in shifts:
        raise ContractError("parameter shifts are missing the prediction head")
    if spec.revin != ("revin" in shifts):
        raise ContractError("RevIN shift presence does not match the arm")
    if spec.backbone == "cnn" and any(name not in shifts for name in _CNN_SHIFT_GROUPS):
        raise ContractError("CNN parameter shifts are missing conv1, conv2, conv3, or the head")
    for group in shifts.values():
        if not isinstance(group, dict) or set(group) != set(SHIFT_GROUP_KEYS):
            raise ContractError("parameter shift groups keep absolute, relative, and status")
        _finite_float(group.get("absolute"), "absolute shift")
        status = group.get("status")
        if status == DEFINED:
            _finite_float(group.get("relative"), "relative shift")
            continue
        if status == UNDEFINED_ZERO_INIT_NORM and group.get("relative") is None:
            continue
        raise ContractError("a zero initial norm keeps a null relative shift and UNDEFINED_ZERO_INIT_NORM")


def _require_prediction_schema(table: pa.Table) -> None:
    if table.column_names != list(_PREDICTION_COLUMNS):
        raise ContractError("predictions.parquet columns are not the frozen set")
    if table.num_rows != TEST_DAYS:
        raise ContractError("predictions.parquet does not cover 180 test days")
    types = {
        "label_date": pa.date32(),
        "y_true_original": pa.float64(),
        "y_pred_original": pa.float64(),
        "weight_variant": pa.string(),
    }
    for name, expected in types.items():
        if table.schema.field(name).type != expected:
            raise ContractError(f"predictions.parquet column {name} has the wrong type")


def _require_prediction_values(table: pa.Table, record: Mapping[str, object]) -> None:
    _require_prediction_schema(table)
    variants = table.column("weight_variant").to_pylist()
    if variants != [_EMA] * TEST_DAYS:
        raise ContractError("test rows are fixed to EMA")
    dates = [item.isoformat() for item in table.column("label_date").to_pylist()]
    spec = _spec_from_payload(record.get("run_spec"))
    contract = load_dataset_contract(spec.dataset)
    expected = [
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")
    ]
    if dates != expected:
        raise ContractError("prediction dates are not the frozen test interval")
    true = np.asarray(table.column("y_true_original").to_numpy(zero_copy_only=False), dtype=np.float64)
    pred = np.asarray(table.column("y_pred_original").to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ContractError("prediction sales contain a non-finite value")
    computed = signed_error_value(true, pred)
    recorded = _finite_float(record.get("signed_error"), "signed_error")
    if np.float64(recorded) != np.float64(computed):
        raise ContractError("signed error is not mean(pred - true)")


def _version_fields() -> dict[str, object]:
    head, dirty = _git_identity()
    try:
        from importlib.metadata import version

        tensorflow_version = version("tensorflow")
    except Exception as exc:
        raise ContractError("TensorFlow version could not be read") from exc
    if type(tensorflow_version) is not str or not tensorflow_version:
        raise ContractError("TensorFlow version could not be read")
    return {
        "git_head": head,
        "workspace_dirty": dirty,
        "frozen_config_sha256": frozen_config_sha256(),
        "tensorflow_version": tensorflow_version,
    }


def _git_identity() -> tuple[str, bool]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("git HEAD and workspace state are required and could not be read") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ContractError("git HEAD is not a revision")
    return head, bool(porcelain.strip())


def _times(payload: RunPayload) -> tuple[str, str, float]:
    started = _utc_timestamp(payload.started_at, "started_at")
    finished = _utc_timestamp(payload.finished_at, "finished_at")
    if finished < started:
        raise ContractError("finished_at is earlier than started_at")
    seconds = (finished - started).total_seconds()
    if not math.isfinite(seconds):
        raise ContractError("elapsed_seconds is non-finite")
    return (
        started.isoformat(timespec="microseconds"),
        finished.isoformat(timespec="microseconds"),
        seconds,
    )


def _target_epochs() -> int:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    return int(payload["target_optimizer"]["epochs"]["value"])


def _spec_payload(spec: RunSpec) -> dict[str, object]:
    return {name: getattr(spec, name) for name in _SPEC_FIELDS}


def _spec_from_payload(payload: object) -> RunSpec:
    if not isinstance(payload, dict) or set(payload) != set(_SPEC_FIELDS):
        raise ContractError("run_spec fields do not match the frozen identity")
    if type(payload["revin"]) is not bool or type(payload["freeze_conv"]) is not bool:
        raise ContractError("revin and freeze_conv must stay booleans")
    if type(payload["horizon"]) is not int or type(payload["seed"]) is not int:
        raise ContractError("horizon and seed must stay integers")
    requested = payload["requested_k"]
    comparison = payload["comparison_days"]
    if requested is not None and type(requested) is not int:
        raise ContractError("requested_k must stay an integer or null")
    if comparison is not None and type(comparison) is not int:
        raise ContractError("comparison_days must stay an integer or null")
    learning_rate = payload["target_learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ContractError("target learning rate must stay a finite number")
    return RunSpec(
        arm=payload["arm"],
        protocol_tag=payload["protocol_tag"],
        dataset=payload["dataset"],
        horizon=payload["horizon"],
        seed=payload["seed"],
        backbone=payload["backbone"],
        revin=payload["revin"],
        freeze_conv=payload["freeze_conv"],
        selection_rule=payload["selection_rule"],
        requested_k=requested,
        weight_mode=payload["weight_mode"],
        comparison_days=comparison,
        target_learning_rate=float(learning_rate),
    )


def _same_spec(left: RunSpec, right: RunSpec) -> bool:
    return all(getattr(left, name) == getattr(right, name) for name in _SPEC_FIELDS)


def _pair_payload(pair: MetricPair) -> dict[str, object]:
    return {
        "rmse": _finite_float(pair.rmse, "rmse"),
        "smape": _finite_float(pair.smape, "smape"),
        "zero_sales_count": pair.zero_sales_count,
        "smape_zero_denominator_count": pair.smape_zero_denominator_count,
        "clipped_true_negative_count": pair.clipped_true_negative_count,
        "clipped_pred_negative_count": pair.clipped_pred_negative_count,
        "value_space": pair.value_space,
    }


def _shift_payload(shifts: Mapping[str, ParameterShift]) -> dict[str, dict[str, object]]:
    payload = {}
    for name in shifts:
        group = shifts[name]
        _require_shift(group)
        relative = None if group.relative is None else _finite_float(group.relative, f"{name} relative shift")
        payload[name] = {
            "absolute": _finite_float(group.absolute, f"{name} absolute shift"),
            "relative": relative,
            "status": group.status,
        }
    return payload


def _directory_matches(directory: Path, spec: RunSpec) -> bool:
    expected = run_directory(spec, Path("anchor")).relative_to("anchor").parts
    return tuple(Path(directory).parts[-len(expected) :]) == expected


def _span_days(start: str, end: str) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 1


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, (str, bytes)) or value is None:
        raise ContractError(f"{name} must stay a finite number")
    if isinstance(value, (int, float, np.floating, np.integer)):
        number = float(value)
        if math.isfinite(number):
            return number
    raise ContractError(f"{name} must stay a finite number")


def _float_list(values: object, name: str) -> list[float]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ContractError(f"{name} must stay a list of finite numbers")
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must stay a list of finite numbers")
    return [_finite_float(value, name) for value in values]


def _require_same_score(left: object, right: object, name: str, *, allow_null: bool) -> None:
    if allow_null and left is None and right is None:
        return
    left_number = _finite_float(left, name)
    right_number = _finite_float(right, name)
    if np.float64(left_number) != np.float64(right_number):
        raise ContractError(f"{name} disagrees between adaptation and metrics")


def _require_same_token(left: object, right: object, expected: str, name: str) -> None:
    if type(left) is not str or type(right) is not str or left != expected or right != expected:
        raise ContractError(f"{name} is unresolved")


def _audit_ids(values: Sequence[str], seen: set[str]) -> list[str]:
    rows = []
    for value in values:
        _require_cell_id(value)
        if value in seen:
            raise ContractError(f"phase audit repeats {value}")
        seen.add(value)
        rows.append(value)
    return sorted(rows)


def _require_phase(phase: object) -> None:
    if type(phase) is not int or phase not in (1, 2, 3, 4, 5):
        raise ContractError(f"phase must be 1, 2, 3, 4, or 5, got {phase!r}")


def _require_cell_id(value: object) -> None:
    if type(value) is not str or _CELL_ID_RE.fullmatch(value) is None:
        raise ContractError(
            "phase audit cell id must be <arm>:<protocol_tag>:<dataset>:h<horizon>:seed<seed>, "
            f"got {value!r}"
        )


def _cell_failed_attempt(payload: object) -> int:
    if not isinstance(payload, dict) or set(payload) != set(CELL_FAILED_KEYS):
        raise ContractError("cell FAILED.json must contain only attempt_count")
    count = payload["attempt_count"]
    if type(count) is not int or count not in (1, 2):
        raise ContractError("cell FAILED.json attempt_count must be 1 or 2")
    return count


def _overlap_cell(spec: RunSpec) -> bool:
    return spec.protocol_tag in _OVERLAP_TAGS and spec.selection_rule in _OVERLAP_RULES


def _require_overlap_value(value: object) -> int:
    if type(value) is not int or value < 0 or value > 3:
        raise ContractError("top3_overlap_count must be an integer from 0 to 3")
    return value


def _require_recorded_sample_counts(spec: RunSpec, adaptation: AdaptationResult) -> tuple[int, int]:
    return _require_recorded_counts(spec.horizon, adaptation.n_train, adaptation.n_validation)


def _require_recorded_counts(horizon: int, n_train: object, n_validation: object) -> tuple[int, int]:
    expected = expected_target_partition_count(horizon)
    if type(n_train) is not int or type(n_validation) is not int:
        raise ContractError("n_train and n_validation must come from the target WindowBatch")
    if n_train != expected or n_validation != expected:
        raise ContractError(
            f"n_train and n_validation are {n_train} and {n_validation}, expected {expected} from the target WindowBatch"
        )
    return n_train, n_validation


def _utc_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ContractError(f"{name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError(f"{name} rejects a naive datetime")
    utc = value.astimezone(timezone.utc)
    if utc.utcoffset() != timedelta(0):
        raise ContractError(f"{name} must be recorded in UTC")
    return utc


def _parse_timestamp(value: object, name: str) -> datetime:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        raise ContractError(f"{name} must be a UTC timestamp with microseconds and +00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ContractError(f"{name} must be a UTC timestamp with microseconds and +00:00")
    if parsed.isoformat(timespec="microseconds") != value:
        raise ContractError(f"{name} must keep six microsecond digits and +00:00")
    return parsed


def _require_timestamp_pair(started: object, finished: object, elapsed: object) -> None:
    start = _parse_timestamp(started, "started_at")
    end = _parse_timestamp(finished, "finished_at")
    if end < start:
        raise ContractError("finished_at is earlier than started_at")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
        raise ContractError("elapsed_seconds is non-finite")
    if float(elapsed) != (end - start).total_seconds():
        raise ContractError("elapsed_seconds must equal the UTC timestamp difference")


def _blocking_names(directory: Path, allowed: set[str]) -> set[str]:
    return {name for name in _listed_names(directory) if name not in allowed}


def _listed_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    names = set()
    for path in directory.iterdir():
        if path.name.endswith(".tmp"):
            continue
        names.add(path.name)
    return names


def _read_done(directory: Path) -> DoneRecord:
    payload = _read_json(directory / DONE_FILE)
    if payload is None or not _done_matches(directory, payload):
        raise ContractError("HOLD_HASH_MISMATCH")
    elapsed = payload["elapsed_seconds"]
    key_hash = payload["stage_a_key_sha256"]
    return DoneRecord(
        run_spec=_spec_from_payload(payload["run_spec"]),
        status=str(payload["status"]),
        artifact_sha256={str(name): str(digest) for name, digest in payload["artifact_sha256"].items()},
        stage_a_key_sha256=None if key_hash is None else str(key_hash),
        started_at=str(payload["started_at"]),
        finished_at=str(payload["finished_at"]),
        elapsed_seconds=float(elapsed),
    )


def _require_prediction_bytes(directory: Path, payload: RunPayload) -> None:
    metrics = payload.metrics
    assert metrics is not None
    table = read_predictions(directory)
    true = np.asarray(table.column("y_true_original").to_numpy(zero_copy_only=False), dtype=np.float64)
    pred = np.asarray(table.column("y_pred_original").to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.array_equal(true, np.asarray(metrics.y_true, dtype=np.float64)):
        raise ContractError("predictions.parquet changed the true sales values")
    if not np.array_equal(pred, np.asarray(metrics.y_pred, dtype=np.float64)):
        raise ContractError("predictions.parquet changed the predicted sales values")


def _commit_done(directory: Path, payload: dict[str, object]) -> None:
    """Rename DONE into place only after the other files are already valid."""
    path = directory / DONE_FILE
    _atomic_json(path, payload)
    if inspect_done(directory) == "SKIP":
        return
    if path.exists():
        path.unlink()
    raise ContractError("cell failed its own resume check and DONE was removed")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_text(path, json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, temporary)
        _replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
