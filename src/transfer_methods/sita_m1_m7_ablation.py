"""Independent SITA M1–M7 ablation.

Reuses the formal FUSED_TARGET_ONLY_V3 training function. It does not register
new arms in the historical A–E or A–F tables and does not change their defaults.
Plan mode never marks a group complete.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.transfer_methods.ms_katl_corrected_sita_60cell import (
    DATASETS,
    HORIZONS,
    SEEDS,
    SOURCE_LR_LABEL,
    TARGET_LR_LABELS,
    TARGETS,
)
from src.transfer_methods.sita_mechanism_diagnostics import (
    linear_cka,
    parameter_l2_distance,
    relative_parameter_drift,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    MechanismDiagnosticsContractError,
)

PROTOCOL_NAME = "SITA_M1_M7_ABLATION"
OUTPUT_MARKER = "sita_m1_m7_ablation"
GROUPS: Tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "M6", "M7")
FROZEN_GROUPS = ("M2", "M5")
REVIN_GROUPS = ("M4", "M5", "M6", "M7")
NATIVE_GROUPS = ("M1", "M4")
FUSED_GROUPS = ("M2", "M3", "M5", "M6")
CKA_LAYERS = ("conv1", "conv2", "conv3", "flatten")
CKA_COMPARISONS = (
    ("M3_init_vs_selected", "M3", "init", "M3", "ema"),
    ("M6_init_vs_selected", "M6", "init", "M6", "ema"),
    ("M3_vs_M2_selected", "M3", "ema", "M2", "ema"),
    ("M6_vs_M5_selected", "M6", "ema", "M5", "ema"),
)
REQUIRED_GROUP_FILES = (
    "resolved_config.json",
    "data_split.json",
    "init_snapshot.json",
    "selected_checkpoint.json",
    "epoch_log.csv",
    "predictions.csv",
    "metrics.json",
    "parameter_drift.json",
    "status.json",
    "init_weights.npz",
    "raw_weights.npz",
    "ema_weights.npz",
)
FORBIDDEN_OUTPUT_MARKERS = (
    "results",
    "sita_ablation_3e4",
    "sita_ablation_formal_per_dataset_lr_ae",
    "ms_katl_ae_core_ablation",
    "sita_tcn",
    "outputs/runs",
)
RELOAD_ATOL = 1e-4
METRIC_ATOL = 1e-6
RESUME_POLICY: Dict[str, Any] = {
    "granularity": "group",
    "epoch_or_step_resume": False,
    "complete_group": "skip without retraining",
    "incomplete_group": "if init/raw/ema weights from a finished run exist, score those weights and do not retrain; otherwise retrain the whole group from the paired initialization",
    "restored_for_retrain": [
        "source weights",
        "fused parameters",
        "M6 head_target initialization",
        "M6 RevIN affine initialization",
        "target windows and scaler",
        "group seed",
    ],
    "not_restored": [
        "Adam moments",
        "cosine schedule step",
        "EMA shadow from a partial run",
        "batch cursor",
        "mid-epoch RNG state",
    ],
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_RELPATH = Path("configs") / "sita_m1_m7_ablation_manifest.json"


class M7ContractError(RuntimeError):
    """Fail-closed M1–M7 protocol violation."""


@dataclass(frozen=True)
class GroupSpec:
    group: str
    init_mode: str
    conv_frozen: bool
    revin_enabled: bool

    @property
    def revin_affine(self) -> bool:
        return bool(self.revin_enabled)


GROUP_SPECS: Mapping[str, GroupSpec] = {
    "M1": GroupSpec("M1", "native", False, False),
    "M2": GroupSpec("M2", "fused", True, False),
    "M3": GroupSpec("M3", "fused", False, False),
    "M4": GroupSpec("M4", "native", False, True),
    "M5": GroupSpec("M5", "fused", True, True),
    "M6": GroupSpec("M6", "fused", False, True),
    "M7": GroupSpec("M7", "nearest", False, True),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {message}\n")


def load_manifest(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    payload = json.loads((root / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    if payload.get("output_marker") != OUTPUT_MARKER:
        raise M7ContractError("M1–M7 manifest output marker drifted")
    if dict(payload.get("targets") or {}) != dict(TARGETS):
        raise M7ContractError("M1–M7 targets drifted from corrected formal SITA")
    if dict(payload.get("target_learning_rates") or {}) != dict(TARGET_LR_LABELS):
        raise M7ContractError("M1–M7 target learning rates drifted from corrected formal SITA")
    if str(payload.get("source_learning_rate")) != SOURCE_LR_LABEL:
        raise M7ContractError("M1–M7 source learning rate drifted from corrected formal SITA")
    if list(payload.get("groups") or []) != list(GROUPS):
        raise M7ContractError("M1–M7 group list drifted")
    return payload


def target_learning_rate(dataset: str) -> Tuple[float, str]:
    label = TARGET_LR_LABELS[str(dataset)]
    return float(label), label


def condition_key(dataset: str, horizon: int, seed: int) -> str:
    target = TARGETS[str(dataset)].replace("/", "-")
    return f"{dataset}__{target}__h{int(horizon)}__s{int(seed)}"


def enumerate_conditions(manifest: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    payload = dict(manifest or load_manifest())
    rows = []
    for dataset in payload["datasets"]:
        for horizon in payload["horizons"]:
            for seed in payload["seeds"]:
                rows.append(
                    {
                        "dataset": dataset,
                        "target": TARGETS[dataset],
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "condition_key": condition_key(dataset, int(horizon), int(seed)),
                        "source_learning_rate": SOURCE_LR_LABEL,
                        "target_learning_rate": TARGET_LR_LABELS[dataset],
                        "groups": list(GROUPS),
                    }
                )
    if len(rows) != 60:
        raise M7ContractError(f"expected 60 conditions, got {len(rows)}")
    return rows


def parse_groups(text: str) -> Tuple[str, ...]:
    groups = tuple(part.strip().upper() for part in str(text).split(",") if part.strip())
    unknown = [group for group in groups if group not in GROUP_SPECS]
    if unknown or not groups:
        raise M7ContractError(f"unknown groups {unknown or groups}; expected {list(GROUPS)}")
    return groups


def assert_output_isolation(output_root: Path, project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    resolved = Path(output_root).resolve()
    outputs = (root / "outputs").resolve()
    text = str(resolved).lower()
    if OUTPUT_MARKER not in text:
        raise M7ContractError(f"output root must include {OUTPUT_MARKER}: {resolved}")
    if any(marker in text for marker in FORBIDDEN_OUTPUT_MARKERS):
        raise M7ContractError(f"refusing to write into an existing result root: {resolved}")
    if outputs not in resolved.parents and resolved != outputs / OUTPUT_MARKER:
        if outputs not in resolved.parents:
            raise M7ContractError(f"output root must live under {outputs}: {resolved}")
    if "results" in resolved.parts:
        raise M7ContractError(f"refusing results/ output: {resolved}")
    return resolved


def default_output_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "outputs" / OUTPUT_MARKER


def _pid_alive(pid: object) -> bool:
    if pid in (None, ""):
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _max_abs(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> float:
    if len(left) != len(right):
        raise M7ContractError("weight list length mismatch")
    worst = 0.0
    for lhs, rhs in zip(left, right):
        a = np.asarray(lhs)
        b = np.asarray(rhs)
        if a.shape != b.shape:
            raise M7ContractError(f"weight shape mismatch {a.shape} vs {b.shape}")
        worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return worst


def _pack_state(state: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    packed: Dict[str, np.ndarray] = {}
    for name, arrays in state["conv"].items():
        for index, array in enumerate(arrays):
            packed[f"conv__{name}__{index}"] = np.asarray(array)
    for index, array in enumerate(state["head_target"]):
        packed[f"head__{index}"] = np.asarray(array)
    for index, array in enumerate(state["cnn_full"]):
        packed[f"cnn__{index}"] = np.asarray(array)
    revin = state.get("revin_affine")
    packed["revin_present"] = np.asarray([0 if revin is None else 1], dtype=np.int64)
    if revin is not None:
        packed["revin_affine_weight"] = np.asarray(revin["affine_weight"])
        packed["revin_affine_bias"] = np.asarray(revin["affine_bias"])
    return packed


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_pack_state(state))


def load_state(path: Path) -> Dict[str, Any]:
    with np.load(path) as payload:
        keys = set(payload.files)
        conv: Dict[str, List[np.ndarray]] = {}
        for key in sorted(item for item in keys if item.startswith("conv__")):
            _, name, index = key.split("__")
            conv.setdefault(name, [])
            while len(conv[name]) <= int(index):
                conv[name].append(None)  # type: ignore[arg-type]
            conv[name][int(index)] = np.array(payload[key], copy=True)
        head = [np.array(payload[key], copy=True) for key in sorted(
            (item for item in keys if item.startswith("head__")),
            key=lambda item: int(item.split("__")[1]),
        )]
        cnn = [np.array(payload[key], copy=True) for key in sorted(
            (item for item in keys if item.startswith("cnn__")),
            key=lambda item: int(item.split("__")[1]),
        )]
        revin = None
        if int(np.asarray(payload["revin_present"]).reshape(-1)[0]) == 1:
            revin = {
                "affine_weight": np.array(payload["revin_affine_weight"], copy=True),
                "affine_bias": np.array(payload["revin_affine_bias"], copy=True),
            }
    return {"conv": conv, "head_target": head, "cnn_full": cnn, "revin_affine": revin}


def _digest_state(state: Mapping[str, Any]) -> Dict[str, str]:
    from src.transfer_methods.ms_katl_lite import digest_weight_arrays

    conv_arrays = [array for name in ("conv1", "conv2", "conv3") for array in state["conv"][name]]
    payload = {
        "conv": digest_weight_arrays(conv_arrays),
        "head_target": digest_weight_arrays(state["head_target"]),
    }
    revin = state.get("revin_affine")
    payload["revin_affine"] = (
        None if revin is None else digest_weight_arrays([revin["affine_weight"], revin["affine_bias"]])
    )
    return payload


def execution_order(groups: Sequence[str]) -> Tuple[str, ...]:
    """Train M6 first so its formal initialization can be copied to the other groups."""
    selected = set(groups)
    unknown = sorted(selected - set(GROUPS))
    if unknown:
        raise M7ContractError(f"unknown groups {unknown}")
    ordered = []
    if "M6" in selected:
        ordered.append("M6")
    ordered.extend(group for group in GROUPS if group in selected and group != "M6")
    return tuple(ordered)


def select_probe_indices(n_samples: int, stride: int) -> List[int]:
    if int(stride) < 1:
        raise M7ContractError("probe stride must be positive")
    return list(range(0, int(n_samples), int(stride)))


def build_probe_metadata(
    window_rows: Sequence[Mapping[str, str]],
    indices: Sequence[int],
    *,
    stride: int,
) -> Dict[str, Any]:
    selected = [dict(window_rows[index]) for index in indices]
    origins = [row["forecast_origin_date"] for row in selected]
    targets = [row["target_date"] for row in selected]
    return {
        "partition": "test",
        "labels_used": False,
        "window_ids": [int(index) for index in indices],
        "n_samples": len(indices),
        "selection_rule": (
            "non-overlapping formal test input windows: sample_index = 0, stride, 2*stride, ... "
            "where stride equals the input window length"
        ),
        "stride": int(stride),
        "nonoverlapping": True,
        "iid_samples": False,
        "date_range": {
            "forecast_origin_min": min(origins) if origins else None,
            "forecast_origin_max": max(origins) if origins else None,
            "target_date_min": min(targets) if targets else None,
            "target_date_max": max(targets) if targets else None,
        },
        "windows": [
            {"window_id": int(index), **dict(window_rows[index])}
            for index in indices
        ],
        "input_preprocessing": "one shared MinMax-scaled test window tensor for every group",
        "within_group_revin": (
            "RevIN is applied only inside groups that enable it, after the shared scaled window. "
            "Input preprocessing is shared; RevIN treatment is not."
        ),
        "cka_uses_checkpoint_selection": False,
        "note": (
            "These windows do not share time positions. They still belong to one target series "
            "and are not treated as independent samples. A short test split can leave fewer than "
            "two windows; that case is recorded as undefined CKA."
        ),
    }


def safe_linear_cka(left: np.ndarray, right: np.ndarray) -> Dict[str, Any]:
    try:
        value = float(linear_cka(left, right))
    except MechanismDiagnosticsContractError as exc:
        return {"status": "UNDEFINED", "value": None, "reason": str(exc)}
    return {"status": "OK", "value": value, "reason": None}


def test_window_rows(target_df: pd.DataFrame, horizon: int, window_size: int) -> List[Dict[str, str]]:
    from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates

    _, _, test_df = temporal_split_by_ratio_or_dates(target_df)
    ordered = test_df.sort_values(["entity_id", "item_id", "date"], kind="mergesort").reset_index(drop=True)
    rows: List[Dict[str, str]] = []
    for _, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        frame = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        dates = pd.to_datetime(frame["date"], errors="raise")
        count = len(frame)
        max_end = count - int(horizon)
        for end_idx in range(int(window_size) - 1, max_end):
            target_idx = end_idx + int(horizon)
            if target_idx >= count:
                continue
            start_idx = end_idx - int(window_size) + 1
            rows.append(
                {
                    "window_start": dates.iloc[start_idx].strftime("%Y-%m-%d"),
                    "forecast_origin_date": dates.iloc[end_idx].strftime("%Y-%m-%d"),
                    "target_date": dates.iloc[target_idx].strftime("%Y-%m-%d"),
                }
            )
    return rows


def split_day_summary(target_df: pd.DataFrame) -> Dict[str, Any]:
    from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates

    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)

    def _bounds(frame: pd.DataFrame) -> Dict[str, Any]:
        dates = pd.to_datetime(frame["date"], errors="raise")
        return {
            "n_rows": int(len(frame)),
            "n_days": int(dates.dt.normalize().nunique()),
            "date_min": None if dates.empty else dates.min().strftime("%Y-%m-%d"),
            "date_max": None if dates.empty else dates.max().strftime("%Y-%m-%d"),
        }

    return {"train": _bounds(train_df), "validation": _bounds(val_df), "test": _bounds(test_df)}


def evaluate_init_pairing(snapshots: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    missing = [group for group in GROUPS if group not in snapshots]
    if missing:
        return {"status": "FAIL", "reason": f"missing init snapshots: {missing}"}
    failures: List[str] = []

    def _same(groups: Sequence[str], field: str, label: str) -> None:
        values = [snapshots[group][field] for group in groups]
        if any(value is None for value in values) or len(set(values)) != 1:
            failures.append(f"{label} diverged: {dict(zip(groups, values))}")

    _same(GROUPS, "head_target", "head_target init")
    _same(NATIVE_GROUPS, "conv", "M1/M4 native conv init")
    _same(FUSED_GROUPS, "conv", "M2/M3/M5/M6 fused conv init")
    _same(REVIN_GROUPS, "revin_affine", "RevIN affine init")
    if snapshots["M7"].get("nearest_source_conv") != snapshots["M7"]["conv"]:
        failures.append("M7 conv init does not match the saved nearest source conv")
    for group in REVIN_GROUPS:
        if snapshots[group]["revin_affine"] is None:
            failures.append(f"{group} is missing RevIN affine init")
    for group in ("M1", "M2", "M3"):
        if snapshots[group]["revin_affine"] is not None:
            failures.append(f"{group} unexpectedly has RevIN affine init")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def group_files_complete(group_dir: Path) -> bool:
    return all((group_dir / name).is_file() for name in REQUIRED_GROUP_FILES)


_SEALED_STATUSES = {
    "COMPLETE",
    "ALL_REQUIRED_ARTIFACTS_COMPLETE",
    "TRAINING_COMPLETE",
    "TRAINING_COMPLETE_DIAGNOSTICS_INCOMPLETE",
}
EPOCH_LOG_FIELDS = (
    "epoch",
    "global_step",
    "train_loss",
    "train_loss_name",
    "val_metric_name",
    "val_metric",
    "val_metric_missing_reason",
    "learning_rate",
    "metric_space",
    "weight_variant",
    "used_for_checkpoint_selection",
)


class EpochLogWriteError(M7ContractError):
    """An epoch record could not be persisted."""


class EpochValidationLog:
    """Append one RAW validation record immediately. Failures propagate."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows = 0

    def __call__(self, payload: Mapping[str, object]) -> None:
        row = {field: payload.get(field, "") for field in EPOCH_LOG_FIELDS}
        if row["val_metric"] is None:
            row["val_metric"] = ""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.is_file() or self.path.stat().st_size == 0
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EPOCH_LOG_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
            self.rows += 1
        except Exception as exc:
            raise EpochLogWriteError(f"epoch log write failed at {self.path}: {exc}") from exc


def mark_group_failed(group_dir: Path, exc: BaseException, *, start_time: Optional[str] = None, elapsed_seconds: Optional[float] = None) -> None:
    """Record a failure without deleting an epoch log already on disk."""
    _write_json(
        group_dir / "error.json",
        {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    _write_json(
        group_dir / "status.json",
        {
            "status": "FAILED",
            "group": group_dir.name,
            "pid": os.getpid(),
            "start_time": start_time,
            "end_time": _now(),
            "elapsed_seconds": elapsed_seconds,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "TRAINING_COMPLETE": (group_dir / "training_evidence.json").is_file(),
            "EPOCH_VALIDATION_LOG_COMPLETE": False,
        },
    )


def _weights_readable(group_dir: Path) -> bool:
    if not _weights_saved(group_dir):
        return False
    try:
        for name in ("init_weights.npz", "raw_weights.npz", "ema_weights.npz"):
            state = load_state(group_dir / name)
            if not state["conv"] or not state["head_target"] or not state["cnn_full"]:
                return False
    except Exception:
        return False
    return True


def _training_sealed(group_dir: Path) -> bool:
    """Sealed training needs weights, a config, and evidence beyond the weight files."""
    if not _weights_readable(group_dir) or not (group_dir / "resolved_config.json").is_file():
        return False
    evidence = group_dir / "training_evidence.json"
    if evidence.is_file():
        payload = _read_json(evidence)
        return bool(payload.get("training_finished"))
    status_path = group_dir / "status.json"
    if not status_path.is_file() or not (group_dir / "init_snapshot.json").is_file():
        return False
    return str(_read_json(status_path).get("status")) in _SEALED_STATUSES


def _predictions_ready(group_dir: Path) -> bool:
    if not (group_dir / "predictions.csv").is_file() or not (group_dir / "metrics.json").is_file():
        return False
    metrics = _read_json(group_dir / "metrics.json")
    if metrics.get("weight_variant") != "EMA":
        return False
    predictions = pd.read_csv(group_dir / "predictions.csv")
    return len(predictions) > 0 and np.isfinite(predictions["y_pred_original"]).all()


def plan_group_execution(group_dir: Path) -> str:
    """Choose skip, finalize, or train. Weight files alone do not seal training."""
    if _training_sealed(group_dir) and _predictions_ready(group_dir):
        return "skip"
    if _training_sealed(group_dir):
        return "finalize_saved_weights"
    return "train"


def assess_epoch_validation_log(path: Path, *, expected_epochs: Optional[int] = None) -> Dict[str, Any]:
    if not path.is_file():
        return {"EPOCH_VALIDATION_LOG_COMPLETE": False, "reason": "epoch_log_missing"}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"EPOCH_VALIDATION_LOG_COMPLETE": False, "reason": "epoch_log_empty", "rows": 0}
    metric_column = "val_metric" if "val_metric" in frame.columns else "val_rmse" if "val_rmse" in frame.columns else None
    if metric_column is None:
        return {"EPOCH_VALIDATION_LOG_COMPLETE": False, "reason": "validation_metric_column_missing", "rows": int(len(frame))}
    values = frame[metric_column]
    reason_column = "val_metric_missing_reason" if "val_metric_missing_reason" in frame.columns else None
    missing = values.isna() | values.astype(str).str.strip().isin({"", "nan", "None"})
    unexplained = missing
    if reason_column is not None:
        reasons = frame[reason_column].astype(str).str.strip()
        unexplained = missing & reasons.isin({"", "nan", "None"})
    if bool(unexplained.any()):
        return {
            "EPOCH_VALIDATION_LOG_COMPLETE": False,
            "reason": "validation_metric_absent_without_reason",
            "rows": int(len(frame)),
            "missing_rows": int(unexplained.sum()),
        }
    if expected_epochs is not None and len(frame) != int(expected_epochs):
        return {
            "EPOCH_VALIDATION_LOG_COMPLETE": False,
            "reason": "epoch_row_count_mismatch",
            "rows": int(len(frame)),
            "expected_epochs": int(expected_epochs),
        }
    weight_ok = True
    if "weight_variant" in frame.columns:
        weight_ok = set(frame["weight_variant"].astype(str)) <= {"RAW"}
    return {
        "EPOCH_VALIDATION_LOG_COMPLETE": bool(weight_ok),
        "reason": None if weight_ok else "validation_metric_not_marked_raw",
        "rows": int(len(frame)),
        "weight_variant": "RAW" if weight_ok else "unspecified",
    }


def assess_update_counter_evidence(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return {
            "UPDATE_COUNTER_EVIDENCE": "missing",
            "optimizer_update_count_evidence": "missing",
            "ema_update_count_evidence": "missing",
        }
    optimizer_evidence = str(payload.get("optimizer_update_count_evidence") or "")
    ema_evidence = str(payload.get("ema_update_count_evidence") or "")
    if optimizer_evidence == "observed_optimizer_iterations_delta" and ema_evidence == "observed_at_ema_update":
        label = "observed"
    elif optimizer_evidence in {"trainer_reported", "trainer_reported_stdout_restatement"} or ema_evidence == "inferred_from_training_steps_and_update_rule":
        label = "trainer_reported_and_inferred" if ema_evidence == "inferred_from_training_steps_and_update_rule" else "trainer_reported"
    else:
        label = "missing"
    return {
        "UPDATE_COUNTER_EVIDENCE": label,
        "optimizer_update_count": payload.get("optimizer_update_count"),
        "optimizer_update_count_evidence": optimizer_evidence or "missing",
        "ema_initialization_count": payload.get("ema_initialization_count"),
        "ema_update_count": payload.get("ema_update_count"),
        "ema_update_count_evidence": ema_evidence or "missing",
    }


def group_is_complete(group_dir: Path) -> bool:
    status_path = group_dir / "status.json"
    if not status_path.is_file() or not group_files_complete(group_dir):
        return False
    payload = _read_json(status_path)
    if str(payload.get("status")) not in _SEALED_STATUSES and payload.get("TRAINING_COMPLETE") is not True:
        return False
    metrics = _read_json(group_dir / "metrics.json")
    if metrics.get("weight_variant") != "EMA":
        return False
    predictions = pd.read_csv(group_dir / "predictions.csv")
    return len(predictions) > 0 and np.isfinite(predictions["y_pred_original"]).all()


def _layer_drift(initial: Sequence[np.ndarray], final: Sequence[np.ndarray]) -> Dict[str, Any]:
    return {
        "l2": parameter_l2_distance(initial, final),
        "relative": relative_parameter_drift(initial, final),
        "max_abs": _max_abs(initial, final),
    }


def _conv_list(state: Mapping[str, Any], name: str) -> List[np.ndarray]:
    return list(state["conv"][name])


def build_parameter_drift(init_state: Mapping[str, Any], raw_state: Mapping[str, Any], ema_state: Mapping[str, Any], *, conv_frozen: bool, trainable_names: Sequence[str]) -> Dict[str, Any]:
    layers: Dict[str, Any] = {}
    for name in ("conv1", "conv2", "conv3"):
        layers[name] = {
            "raw": _layer_drift(_conv_list(init_state, name), _conv_list(raw_state, name)),
            "ema": _layer_drift(_conv_list(init_state, name), _conv_list(ema_state, name)),
            "ema_shadow_includes_layer": (not conv_frozen),
        }
    layers["head_target"] = {
        "raw": _layer_drift(init_state["head_target"], raw_state["head_target"]),
        "ema": _layer_drift(init_state["head_target"], ema_state["head_target"]),
        "ema_shadow_includes_layer": True,
    }
    if init_state.get("revin_affine") is not None:
        layers["revin_affine"] = {
            "raw": _layer_drift(
                [init_state["revin_affine"]["affine_weight"], init_state["revin_affine"]["affine_bias"]],
                [raw_state["revin_affine"]["affine_weight"], raw_state["revin_affine"]["affine_bias"]],
            ),
            "ema": _layer_drift(
                [init_state["revin_affine"]["affine_weight"], init_state["revin_affine"]["affine_bias"]],
                [ema_state["revin_affine"]["affine_weight"], ema_state["revin_affine"]["affine_bias"]],
            ),
            "ema_shadow_includes_layer": True,
        }
    conv_names = [name for name in trainable_names if any(token in name for token in ("conv1", "conv2", "conv3"))]
    return {
        "layers": layers,
        "conv_frozen": bool(conv_frozen),
        "trainable_names": list(trainable_names),
        "frozen_conv_in_optimizer": bool(conv_names) if conv_frozen else None,
        "conv_names_in_optimizer": conv_names,
    }


def assert_update_contract(
    spec: GroupSpec,
    drift: Mapping[str, Any],
    parameter_map: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Check freeze and update membership.

    Numeric change is recorded. A trainable tensor that stays unchanged is not
    by itself a failed run. Frozen convolution values must stay unchanged, and
    a parameter map is required before membership can be claimed.
    """
    failures: List[str] = []
    observations: List[str] = []
    entries = list((parameter_map or {}).get("entries") or [])
    by_module: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in entries:
        by_module.setdefault(str(entry["semantic_module"]), []).append(entry)

    def module_flags(name: str) -> Tuple[bool, bool]:
        rows = by_module.get(name) or []
        if not rows:
            return False, False
        return (
            all(bool(row["in_optimizer"]) for row in rows),
            all(bool(row["ema_shadow"]) for row in rows),
        )

    if parameter_map is None:
        observations.append("parameter membership was not observed from layer objects")
    else:
        dense_rows = by_module.get("dense_out") or []
        if any(bool(row["in_optimizer"]) for row in dense_rows):
            failures.append(f"{spec.group} cnn.dense_out entered the optimizer")
        head_in, head_ema = module_flags("head_target")
        if not by_module.get("head_target"):
            failures.append(f"{spec.group} prediction output head_target is missing from the parameter map")
        elif not head_in:
            failures.append(f"{spec.group} prediction output head_target is absent from the optimizer")
        elif bool(parameter_map.get("use_ema")) and not head_ema:
            failures.append(f"{spec.group} prediction output head_target has no EMA shadow")
        if spec.revin_enabled:
            revin_in, revin_ema = module_flags("revin_affine")
            if not by_module.get("revin_affine"):
                failures.append(f"{spec.group} RevIN affine is missing from the parameter map")
            elif not revin_in:
                failures.append(f"{spec.group} RevIN affine is absent from the optimizer")
            elif bool(parameter_map.get("use_ema")) and not revin_ema:
                failures.append(f"{spec.group} RevIN affine has no EMA shadow")

    if spec.conv_frozen:
        if parameter_map is not None:
            for name in ("conv1", "conv2", "conv3"):
                rows = by_module.get(name) or []
                if not rows:
                    failures.append(f"{spec.group} frozen {name} is missing from the parameter map")
                    continue
                if any(bool(row["in_optimizer"]) for row in rows):
                    failures.append(f"{spec.group} frozen {name} entered the optimizer")
                if any(bool(row["ema_shadow"]) for row in rows):
                    failures.append(f"{spec.group} frozen {name} has an EMA shadow")
        for name in ("conv1", "conv2", "conv3"):
            if drift["layers"][name]["raw"]["max_abs"] != 0.0 or drift["layers"][name]["ema"]["max_abs"] != 0.0:
                failures.append(f"{spec.group} frozen {name} changed")
    elif parameter_map is not None:
        for name in ("conv1", "conv2", "conv3"):
            rows = by_module.get(name) or []
            in_optimizer, has_ema = module_flags(name) if rows else (False, False)
            if not rows or not in_optimizer:
                failures.append(f"{spec.group} trainable {name} has no optimizer update path")
            elif bool(parameter_map.get("use_ema")) and not has_ema:
                failures.append(f"{spec.group} trainable {name} has no EMA shadow")
            elif drift["layers"][name]["raw"]["max_abs"] == 0.0:
                observations.append(f"{spec.group} {name} stayed numerically unchanged")

    if drift["layers"]["head_target"]["raw"]["max_abs"] == 0.0:
        observations.append(f"{spec.group} head_target stayed numerically unchanged")
    if spec.revin_enabled and drift["layers"].get("revin_affine", {}).get("raw", {}).get("max_abs") == 0.0:
        observations.append(f"{spec.group} RevIN affine stayed numerically unchanged")
    if failures:
        raise M7ContractError("; ".join(failures))
    return {
        "freeze_value_check": "PASS" if spec.conv_frozen else "not_frozen",
        "membership_observed": parameter_map is not None,
        "observations": observations,
    }


def _save_stage_a(stage_a: Any, path: Path, *, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    selected_sources = []
    for item in stage_a.selected_sources:
        source_key = item.get("source_key")
        if isinstance(source_key, (list, tuple)):
            source_key = list(source_key)
        selected_sources.append(
            {
                "source_key": source_key,
                "distance": float(item.get("distance", float("nan"))),
                "weight": float(item.get("weight", float("nan"))),
            }
        )
    payload = {
        "selected_sources": selected_sources,
        "source_scalers": stage_a.source_scalers,
        "source_train_tensors": stage_a.source_train_tensors,
        "source_weights": list(stage_a.source_weights),
        "source_distances": list(stage_a.source_distances),
        "source_models_info": stage_a.source_models_info,
        "source_identities": [tuple(item) for item in stage_a.source_identities],
        "knn_prior": np.asarray(stage_a.knn_prior),
        "input_shape": tuple(stage_a.input_shape),
        "fused_params": stage_a.fused_params,
        "layer_names": list(stage_a.layer_names),
        "source_weight_digests": list(stage_a.source_weight_digests),
        "source_scaler_digests": list(stage_a.source_scaler_digests),
        "fused_params_digest": stage_a.fused_params_digest,
        "failed_sources": stage_a.failed_sources,
        "effective_k": int(stage_a.effective_k),
        "tau_k": float(stage_a.tau_k),
        "stage_a_seed": int(seed),
    }
    with (path / "stage_a.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    for index, snapshot in enumerate(stage_a.source_weight_snapshots):
        np.savez_compressed(
            path / f"source_{index}.npz",
            **{f"w{weight_index}": np.asarray(weight) for weight_index, weight in enumerate(snapshot)},
        )
    fused_packed = {
        f"{name}__{index}": np.asarray(array)
        for name, arrays in stage_a.fused_params.items()
        for index, array in enumerate(arrays)
    }
    np.savez_compressed(path / "fused_params.npz", **fused_packed)
    _write_json(
        path / "status.json",
        {
            "status": "COMPLETE",
            "effective_k": int(stage_a.effective_k),
            "source_identities": [list(item) for item in stage_a.source_identities],
            "source_distances": [float(value) for value in stage_a.source_distances],
            "source_weights": [float(value) for value in stage_a.source_weights],
            "source_weight_digests": list(stage_a.source_weight_digests),
            "fused_params_digest": stage_a.fused_params_digest,
            "stage_a_seed": int(seed),
        },
    )


def _load_stage_a(path: Path):
    from src.models.cnn_model import build_base_cnn
    from src.transfer_methods.ms_katl_lite import (
        MSKATLStageAArtifacts,
        build_flatten_feature_extractor,
        digest_weight_arrays,
        freeze_source_cnn,
        fused_params_digest,
        snapshot_model_weights,
    )
    from src.utils.experiment_hyperparams import FIXED_LEARNING_RATE

    with (path / "stage_a.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    if int(payload["effective_k"]) != 3:
        raise M7ContractError("restored Stage A does not contain 3 sources")
    models = []
    snapshots = []
    for index in range(3):
        with np.load(path / f"source_{index}.npz") as weights:
            ordered = [np.array(weights[key], copy=True) for key in sorted(weights.files, key=lambda item: int(item[1:]))]
        model = build_base_cnn(tuple(payload["input_shape"]), learning_rate=float(FIXED_LEARNING_RATE))
        model.set_weights(ordered)
        snapshots.append(snapshot_model_weights(model))
        models.append(freeze_source_cnn(model))
    digests = [digest_weight_arrays(snapshot) for snapshot in snapshots]
    if digests != list(payload["source_weight_digests"]):
        raise M7ContractError("restored source weights do not match the saved digests")
    if fused_params_digest(payload["fused_params"]) != payload["fused_params_digest"]:
        raise M7ContractError("restored fused parameters do not match the saved digest")
    extractors = [
        build_flatten_feature_extractor(model, name=f"restored_source_feature_extractor_{index}")
        for index, model in enumerate(models)
    ]
    return MSKATLStageAArtifacts(
        selection_result={"sources": payload["selected_sources"]},
        selected_sources=list(payload["selected_sources"]),
        source_models=models,
        source_scalers=list(payload["source_scalers"]),
        source_train_tensors=list(payload["source_train_tensors"]),
        source_weights=[float(value) for value in payload["source_weights"]],
        source_distances=[float(value) for value in payload["source_distances"]],
        source_models_info=list(payload["source_models_info"]),
        source_identities=[tuple(item) for item in payload["source_identities"]],
        knn_prior=np.asarray(payload["knn_prior"], dtype=np.float64),
        input_shape=tuple(payload["input_shape"]),
        fused_params=payload["fused_params"],
        layer_names=list(payload["layer_names"]),
        source_weight_snapshots=snapshots,
        source_weight_digests=digests,
        source_scaler_digests=list(payload["source_scaler_digests"]),
        fused_params_digest=str(payload["fused_params_digest"]),
        failed_sources=list(payload["failed_sources"]),
        effective_k=3,
        tau_k=float(payload["tau_k"]),
        source_extractors=extractors,
        stage_a_seed=int(payload["stage_a_seed"]),
    )


def _nearest_snapshot(bundle: Any, stage_a: Any) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_lite import clone_cnn_from_snapshot, snapshot_model_weights
    from src.transfer_methods.msml_tl import extract_layer_params, load_fused_params_into_target_model

    distances = np.asarray(stage_a.source_distances, dtype=np.float64)
    if distances.shape != (3,) or not np.isfinite(distances).all():
        raise M7ContractError(f"nearest source requires 3 finite distances, got {distances}")
    index = int(np.argmin(distances))
    params = extract_layer_params(stage_a.source_models[index], ["conv1", "conv2", "conv3"])
    cnn, _ = clone_cnn_from_snapshot(
        tuple(stage_a.input_shape),
        float(bundle.learning_rate),
        bundle.raw_target_snapshot,
        target_backbone_trainable=True,
        name="m7_nearest_source_carrier",
    )
    dense_before = [np.array(weight, copy=True) for weight in cnn.get_layer("dense_out").get_weights()]
    cnn = load_fused_params_into_target_model(cnn, params)
    dense_after = [np.array(weight, copy=True) for weight in cnn.get_layer("dense_out").get_weights()]
    if _max_abs(dense_before, dense_after) != 0.0:
        raise M7ContractError("nearest-source init changed dense_out")
    return {
        "index": index,
        "identity": tuple(stage_a.source_identities[index]),
        "distance": float(distances[index]),
        "snapshot": snapshot_model_weights(cnn),
        "conv": params,
    }


def _write_epoch_log(path: Path, history: Mapping[str, Sequence[float]], diagnostics: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for item in diagnostics:
        epoch = int(item.get("epoch") or 0)
        val_rmse = None
        train_loss = item.get("train_objective_loss")
        if epoch >= 1 and epoch - 1 < len(history.get("val_rmse", [])):
            val_rmse = history["val_rmse"][epoch - 1]
            train_loss = history["forecast_loss"][epoch - 1]
        rows.append(
            {
                "epoch": epoch,
                "before_first_update": bool(item.get("before_first_target_optimizer_update")),
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "learning_rate": item.get("learning_rate"),
                "optimizer_update_count": item.get("optimizer_update_count"),
                "ema_update_count": item.get("ema_update_count"),
                "weight_variant": item.get("weight_variant"),
                "ema_applied": item.get("ema_applied"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
        writer.writeheader()
        writer.writerows(rows)


def _save_bundle_arrays(path: Path, bundle: Any) -> None:
    packed = {
        "X_test": np.asarray(bundle.X_test),
        "y_test": np.asarray(bundle.y_test).reshape(-1),
        "X_train": np.asarray(bundle.X_train),
        "y_train": np.asarray(bundle.y_train).reshape(-1),
        "knn_prior": np.asarray(bundle.stage_a.knn_prior).reshape(-1),
    }
    if len(bundle.y_val):
        packed["X_val"] = np.asarray(bundle.X_val)
        packed["y_val"] = np.asarray(bundle.y_val).reshape(-1)
    for index, array in enumerate(bundle.f_t["test"]):
        packed[f"f_test_{index}"] = np.asarray(array)
    np.savez_compressed(path, **packed)


def _load_bundle_arrays(path: Path) -> Dict[str, Any]:
    with np.load(path) as payload:
        files = set(payload.files)
        f_test = [np.array(payload[key], copy=True) for key in sorted(
            (item for item in files if item.startswith("f_test_")),
            key=lambda item: int(item.split("_")[-1]),
        )]
        return {
            "X_test": np.array(payload["X_test"], copy=True),
            "y_test": np.array(payload["y_test"], copy=True),
            "knn_prior": np.array(payload["knn_prior"], copy=True),
            "f_test": f_test,
        }


def _predict_from_weights(
    *,
    arrays: Mapping[str, Any],
    cnn_weights: Sequence[np.ndarray],
    head_weights: Sequence[np.ndarray],
    revin_affine: Optional[Mapping[str, np.ndarray]],
    revin_enabled: bool,
    input_shape: Tuple[int, ...],
    d_model: int,
    learning_rate: float,
    name_suffix: str,
) -> np.ndarray:
    import tensorflow as tf

    from src.transfer_methods.ms_katl_lite import (
        MSKATLLite,
        clone_cnn_from_snapshot,
        extract_features_maybe_revin,
        predict_ms_katl_decomposed,
    )
    from src.transfer_methods.revin import DEFAULT_REVIN_EPS, DEFAULT_REVIN_TARGET_CHANNEL, RevIN

    _, extractor = clone_cnn_from_snapshot(
        input_shape,
        float(learning_rate),
        cnn_weights,
        target_backbone_trainable=True,
        name=f"reload_extractor_{name_suffix}",
    )
    revin = None
    if revin_enabled:
        if revin_affine is None:
            raise M7ContractError("RevIN reload is missing affine weights")
        revin = RevIN(
            num_features=int(arrays["X_test"].shape[-1]),
            eps=float(DEFAULT_REVIN_EPS),
            affine=True,
            target_channel=int(DEFAULT_REVIN_TARGET_CHANNEL),
            name=f"reload_revin_{name_suffix}",
        )
        _ = revin.normalize(tf.constant(arrays["X_test"][:1], dtype=tf.float32))
        revin.affine_weight.assign(np.asarray(revin_affine["affine_weight"], dtype=np.float32))
        revin.affine_bias.assign(np.asarray(revin_affine["affine_bias"], dtype=np.float32))
    features, mean, stdev = extract_features_maybe_revin(
        extractor, arrays["X_test"], revin=revin, training=False
    )
    head = MSKATLLite(
        d_model=int(d_model),
        num_sources=len(arrays["f_test"]),
        gamma=0.5,
        transfer_branch_enabled=False,
        gate_mode="zero",
        name=f"reload_head_{name_suffix}",
    )
    prior = np.asarray(arrays["knn_prior"], dtype=np.float32).reshape(1, -1)
    _ = head(
        tf.constant(features[:1], dtype=tf.float32),
        [tf.constant(array[:1], dtype=tf.float32) for array in arrays["f_test"]],
        tf.constant(prior),
        training=False,
    )
    head.head_target.set_weights([np.array(weight, copy=True) for weight in head_weights])
    decomposed = predict_ms_katl_decomposed(head, features, list(arrays["f_test"]), arrays["knn_prior"])
    prediction = np.asarray(decomposed["y_pred"]).reshape(-1)
    if revin is not None:
        prediction = revin.denormalize_numpy(
            prediction, mean.numpy(), stdev.numpy(), include_mean=True
        ).reshape(-1)
    return prediction


def _representations(state: Mapping[str, Any], probe: np.ndarray, *, revin_enabled: bool, input_shape: Tuple[int, ...], learning_rate: float, name_suffix: str) -> Dict[str, np.ndarray]:
    from src.transfer_methods.ms_katl_lite import clone_cnn_from_snapshot
    from src.transfer_methods.revin import DEFAULT_REVIN_EPS, DEFAULT_REVIN_TARGET_CHANNEL, RevIN
    from src.transfer_methods.sita_mechanism_diagnostics_capture import (
        forward_named_representations_from_constructed,
    )
    import tensorflow as tf

    cnn, extractor = clone_cnn_from_snapshot(
        input_shape,
        float(learning_rate),
        state["cnn_full"],
        target_backbone_trainable=True,
        name=f"cka_extractor_{name_suffix}",
    )
    revin = None
    if revin_enabled:
        revin = RevIN(
            num_features=int(probe.shape[-1]),
            eps=float(DEFAULT_REVIN_EPS),
            affine=True,
            target_channel=int(DEFAULT_REVIN_TARGET_CHANNEL),
            name=f"cka_revin_{name_suffix}",
        )
        _ = revin.normalize(tf.constant(probe[:1], dtype=tf.float32))
        affine = state["revin_affine"]
        revin.affine_weight.assign(np.asarray(affine["affine_weight"], dtype=np.float32))
        revin.affine_bias.assign(np.asarray(affine["affine_bias"], dtype=np.float32))

    class _View:
        def __init__(self) -> None:
            self.target_cnn = cnn
            self.target_extractor = extractor
            self.revin = revin

    return forward_named_representations_from_constructed(_View(), probe)


def _affine_changed(init_state: Mapping[str, Any], ema_state: Mapping[str, Any]) -> bool:
    init_affine = init_state.get("revin_affine")
    ema_affine = ema_state.get("revin_affine")
    if init_affine is None or ema_affine is None:
        return False
    return _max_abs(
        [init_affine["affine_weight"], init_affine["affine_bias"]],
        [ema_affine["affine_weight"], ema_affine["affine_bias"]],
    ) > 0.0


def compute_cka_report(condition_dir: Path, *, input_shape: Tuple[int, ...], learning_rate: float, probe: np.ndarray, probe_meta: Mapping[str, Any]) -> Dict[str, Any]:
    states: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    affine_notes: Dict[str, bool] = {}
    for group in GROUPS:
        group_dir = condition_dir / "groups" / group
        if not group_is_complete(group_dir):
            continue
        states[(group, "init")] = load_state(group_dir / "init_weights.npz")
        states[(group, "ema")] = load_state(group_dir / "ema_weights.npz")
        affine_notes[group] = _affine_changed(states[(group, "init")], states[(group, "ema")])
    representations: Dict[Tuple[str, str], Mapping[str, np.ndarray]] = {}
    for key, state in states.items():
        group, variant = key
        representations[key] = _representations(
            state,
            probe,
            revin_enabled=group in REVIN_GROUPS,
            input_shape=input_shape,
            learning_rate=learning_rate,
            name_suffix=f"{group}_{variant}",
        )
    comparisons = []
    for name, left_group, left_variant, right_group, right_variant in CKA_COMPARISONS:
        left_key = (left_group, left_variant)
        right_key = (right_group, right_variant)
        entry: Dict[str, Any] = {
            "comparison": name,
            "left": {"group": left_group, "state": left_variant, "weight_variant": "INITIAL" if left_variant == "init" else "EMA"},
            "right": {"group": right_group, "state": right_variant, "weight_variant": "INITIAL" if right_variant == "init" else "EMA"},
            "layers": {},
        }
        if left_key not in representations or right_key not in representations:
            entry["status"] = "MISSING_GROUP"
            comparisons.append(entry)
            continue
        notes = []
        if left_variant != right_variant and (affine_notes.get(left_group) or affine_notes.get(right_group)):
            notes.append("RevIN affine parameters changed, so representation change is not attributed only to convolution weights")
        if left_group in REVIN_GROUPS or right_group in REVIN_GROUPS:
            notes.append("at least one side applies RevIN before the convolution")
        entry["notes"] = notes
        for layer in CKA_LAYERS:
            entry["layers"][layer] = safe_linear_cka(
                representations[left_key][layer],
                representations[right_key][layer],
            )
        comparisons.append(entry)
    return {
        "cka_participates_in_checkpoint_selection": False,
        "formula": "column-centered linear CKA",
        "centering": "subtract feature mean across the probe-sample axis",
        "sample_organization": probe_meta,
        "iid_samples": False,
        "dropout": "disabled (training=False)",
        "revin_affine_changed": affine_notes,
        "comparisons": comparisons,
    }


def recompute_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    from src.evaluation.metrics import smape

    true = predictions["y_true_original"].to_numpy(dtype=np.float64)
    pred = predictions["y_pred_original"].to_numpy(dtype=np.float64)
    return {
        "rmse_original": float(np.sqrt(np.mean((true - pred) ** 2))),
        "smape_original": float(smape(true, pred)),
    }


def validate_condition(condition_dir: Path, groups: Sequence[str], *, arrays: Mapping[str, Any], input_shape: Tuple[int, ...], d_model: int, learning_rate: float) -> Dict[str, Any]:
    failures: List[str] = []
    snapshots = {}
    reload_errors = {}
    metric_errors = {}
    for group in groups:
        group_dir = condition_dir / "groups" / group
        if not group_is_complete(group_dir):
            failures.append(f"{group} is not complete")
            continue
        init_meta = _read_json(group_dir / "init_snapshot.json")
        snapshots[group] = init_meta["digests"]
        snapshots[group]["nearest_source_conv"] = init_meta.get("nearest_source_conv")
        metrics = _read_json(group_dir / "metrics.json")
        predictions = pd.read_csv(group_dir / "predictions.csv")
        recomputed = recompute_metrics(predictions)
        metric_errors[group] = {
            "rmse_abs": abs(recomputed["rmse_original"] - float(metrics["rmse_original"])),
            "smape_abs": abs(recomputed["smape_original"] - float(metrics["smape_original"])),
        }
        if metric_errors[group]["rmse_abs"] > METRIC_ATOL or metric_errors[group]["smape_abs"] > METRIC_ATOL:
            failures.append(f"{group} recomputed metrics differ from the summary")
        state = load_state(group_dir / "ema_weights.npz")
        spec = GROUP_SPECS[group]
        reloaded = _predict_from_weights(
            arrays=arrays,
            cnn_weights=state["cnn_full"],
            head_weights=state["head_target"],
            revin_affine=state["revin_affine"],
            revin_enabled=spec.revin_enabled,
            input_shape=input_shape,
            d_model=d_model,
            learning_rate=learning_rate,
            name_suffix=f"validate_{group}",
        )
        saved = predictions["y_pred_normalized"].to_numpy(dtype=np.float64)
        max_abs = float(np.max(np.abs(reloaded.astype(np.float64) - saved)))
        reload_errors[group] = {"max_abs": max_abs, "atol": RELOAD_ATOL}
        if max_abs > RELOAD_ATOL:
            failures.append(f"{group} reloaded prediction max abs diff {max_abs}")
    pairing = evaluate_init_pairing(snapshots) if len(snapshots) == len(GROUPS) else {
        "status": "FAIL",
        "failures": ["pairing requires all seven groups"],
    }
    if pairing["status"] != "PASS":
        failures.extend(pairing.get("failures") or ["init pairing failed"])
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "pairing": pairing,
        "reload": reload_errors,
        "metric_recompute": metric_errors,
        "groups_complete": [group for group in groups if group_is_complete(condition_dir / "groups" / group)],
    }


def _prepare_context(dataset: str, horizon: int, seed: int, project_root: Path) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames

    prepared = assemble_v3_cell_frames(
        dataset,
        TARGETS[dataset],
        horizon=int(horizon),
        seed=int(seed),
        project_root=project_root,
        mode="without",
    )
    return prepared


def _train_stage_a(prepared: Mapping[str, Any], *, horizon: int, seed: int, epochs: int, batch_size: int):
    from src.protocols.reproducibility import set_protocol_seed
    from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
    from src.transfer_methods.ms_katl_primary_target_lr_screening import STAGE_A_LEARNING_RATE

    if float(STAGE_A_LEARNING_RATE) != float(SOURCE_LR_LABEL):
        raise M7ContractError(f"source learning rate drifted: {STAGE_A_LEARNING_RATE}")
    set_protocol_seed(int(seed), include_frameworks=True)
    stage_a = prepare_ms_katl_stage_a(
        source_df=prepared["source_model_df"],
        target_df=prepared["target_model_df"],
        feature_cols=prepared["model_feature_cols"],
        k=3,
        horizon=int(horizon),
        window_size=10,
        learning_rate=float(STAGE_A_LEARNING_RATE),
        source_epochs=int(epochs),
        batch_size=int(batch_size),
        tau_k=1.0,
        group_cols=prepared["group_cols"],
        k3_selection_context=prepared["k3_selection_context"],
        k3_raw_source_context=prepared["k3_raw_source_context"],
        transformation_reuse_context=prepared["transformation_reuse_context"],
        seed=int(seed),
    )
    if int(stage_a.effective_k) != 3:
        raise M7ContractError(f"effective source count {stage_a.effective_k} != 3")
    return stage_a


def _shared_bundle(stage_a: Any, prepared: Mapping[str, Any], *, horizon: int, seed: int):
    from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle
    from src.transfer_methods.ms_katl_primary_target_lr_screening import STAGE_A_LEARNING_RATE

    return prepare_shared_target_bundle(
        stage_a,
        prepared["target_model_df"],
        prepared["model_feature_cols"],
        horizon=int(horizon),
        window_size=10,
        learning_rate=float(STAGE_A_LEARNING_RATE),
        seed=int(seed),
        transformation_reuse_context=prepared["transformation_reuse_context"],
    )


def _run_group(
    *,
    spec: GroupSpec,
    bundle: Any,
    nearest: Mapping[str, Any],
    head_init: Optional[Sequence[np.ndarray]],
    revin_init: Optional[Mapping[str, np.ndarray]],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    metric_protocol: Optional[Mapping[str, Any]],
    epoch_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train one group.

    M6 uses the formal ``fused_target_only`` arm with no initialization overrides.
    Other groups copy M6's already built ``head_target`` and, when RevIN is on,
    M6's affine values. Convolution snapshots still come from the shared bundle.
    """
    from src.protocols.reproducibility import set_protocol_seed
    from src.transfer_methods.ms_katl_fto_v3_screening import v3_training_recipe
    from src.transfer_methods.ms_katl_mechanism_ablation import (
        AblationSpec,
        resolve_ablation_arm,
        run_ablation_arm,
    )
    from src.transfer_methods.revin import DEFAULT_REVIN_EPS, DEFAULT_REVIN_TARGET_CHANNEL

    set_protocol_seed(int(seed), include_frameworks=True)
    formal_m6 = spec.group == "M6"
    if formal_m6:
        if head_init is not None or revin_init is not None:
            raise M7ContractError("M6 must not receive paired initialization overrides")
        ablation = resolve_ablation_arm("fused_target_only")
        snapshot = None
    else:
        if head_init is None:
            raise M7ContractError(f"{spec.group} requires the M6 head_target initialization")
        ablation = AblationSpec(
            arm=spec.group.lower(),
            letter=spec.group,
            display=f"SITA_{spec.group}",
            use_fused_target_init=spec.init_mode == "fused",
            target_backbone_trainable=not spec.conv_frozen,
            transfer_branch_enabled=False,
            gate_mode="zero",
        )
        snapshot = nearest["snapshot"] if spec.init_mode == "nearest" else None
    epoch_log = None
    if epoch_log_path is not None:
        if epoch_log_path.is_file():
            interrupted = epoch_log_path.with_name("epoch_log.interrupted.csv")
            if interrupted.is_file():
                interrupted.replace(epoch_log_path.with_name("epoch_log.interrupted.previous.csv"))
            epoch_log_path.replace(interrupted)
        epoch_log = EpochValidationLog(epoch_log_path)
    return run_ablation_arm(
        ablation,
        bundle,
        epochs=int(epochs),
        batch_size=int(batch_size),
        gamma=0.5,
        use_adversarial=False,
        metric_protocol=dict(metric_protocol) if metric_protocol is not None else None,
        fine_tune_learning_rate=float(learning_rate),
        training_recipe=v3_training_recipe(),
        revin_enabled=spec.revin_enabled,
        revin_affine=spec.revin_affine,
        revin_eps=float(DEFAULT_REVIN_EPS),
        revin_target_channel=int(DEFAULT_REVIN_TARGET_CHANNEL),
        init_weight_snapshot=snapshot,
        head_target_init_weights=None if formal_m6 else head_init,
        revin_affine_init=None if formal_m6 or not spec.revin_enabled else revin_init,
        capture_weight_states=True,
        on_epoch_metric_record=epoch_log,
        record_update_counters=True,
    )


def _condition_contract(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    learning_rate_label: str,
    target_epochs: int,
    source_epochs: int,
    batch_size: int,
    stage_a: Any,
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "dataset": dataset,
        "target": TARGETS[dataset],
        "horizon": int(horizon),
        "seed": int(seed),
        "source_learning_rate": SOURCE_LR_LABEL,
        "target_learning_rate": learning_rate_label,
        "target_epochs": int(target_epochs),
        "source_epochs": int(source_epochs),
        "batch_size": int(batch_size),
        "source_weight_digests": list(stage_a.source_weight_digests),
        "fused_params_digest": stage_a.fused_params_digest,
        "resume_policy": RESUME_POLICY,
    }


def assert_resume_contract(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    keys = (
        "protocol",
        "dataset",
        "target",
        "horizon",
        "seed",
        "source_learning_rate",
        "target_learning_rate",
        "target_epochs",
        "source_epochs",
        "batch_size",
        "source_weight_digests",
        "fused_params_digest",
    )
    for key in keys:
        if saved.get(key) != current[key]:
            raise M7ContractError(f"resume refused, contract field {key} does not match the saved run")
    if saved.get("resume_policy", {}).get("granularity") != "group":
        raise M7ContractError("resume refused, saved resume granularity is not group-level")


def _resolved_config(
    *,
    spec: GroupSpec,
    dataset: str,
    horizon: int,
    seed: int,
    learning_rate: float,
    learning_rate_label: str,
    manifest: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> Dict[str, Any]:
    recipe = dict(arm.get("training_recipe") or {})
    return {
        "protocol": PROTOCOL_NAME,
        "group": spec.group,
        "dataset": dataset,
        "target": TARGETS[dataset],
        "horizon": int(horizon),
        "seed": int(seed),
        "source_learning_rate": SOURCE_LR_LABEL,
        "target_learning_rate": float(learning_rate),
        "target_learning_rate_label": learning_rate_label,
        "init_mode": spec.init_mode,
        "conv_frozen": spec.conv_frozen,
        "freeze_scope": "fused or source conv1/conv2/conv3 only" if spec.conv_frozen else "none",
        "head_target_init_source": (
            "formal fused_target_only build inside run_ablation_arm; weights are not overwritten"
            if spec.group == "M6"
            else "copied from the M6 pre-training head_target snapshot"
        ),
        "m6_initialization_overrides": {
            "init_weight_snapshot": False,
            "head_target_init_weights": False,
            "revin_affine_init": False,
        } if spec.group == "M6" else {
            "init_weight_snapshot": spec.init_mode == "nearest",
            "head_target_init_weights": True,
            "revin_affine_init": spec.revin_enabled,
        },
        "transfer_branch_enabled": False,
        "gate_mode": "zero",
        "revin_enabled": spec.revin_enabled,
        "revin_affine": spec.revin_affine,
        "revin_affine_trainable": spec.revin_enabled,
        "optimizer": "Adam",
        "optimizer_weight_decay": None,
        "clipnorm": None,
        "schedule": recipe.get("lr_schedule", "cosine"),
        "loss": recipe.get("loss_name", "huber"),
        "huber_delta": recipe.get("huber_delta", 1.0),
        "batch_size": int(manifest["batch_size"]),
        "epochs": int(manifest["target_epochs"]),
        "early_stopping": False,
        "selection_rule": "final_epoch_ema",
        "val_rmse_used_for_selection": False,
        "formal_weight_variant": "EMA",
        "metric_space": "original_sales_space",
        "test_refit": False,
        "ema": {
            "enabled": bool(recipe.get("use_ema", True)),
            "momentum": recipe.get("ema_momentum", 0.99),
            "initialized_from": "trainable variables after paired initialization and before the first optimizer step",
            "update_rule": "ema = momentum * ema + (1 - momentum) * variable",
            "update_frequency": "once per optimizer step",
            "bias_correction": False,
            "covers_frozen_parameters": False,
        },
        "training_recipe_returned": recipe,
        "batch_order": "np.random.default_rng(seed) inside train_ms_katl_head; each group creates its own optimizer, schedule, and EMA",
        "bitwise_reproducibility": False,
        "nondeterminism_note": "TensorFlow kernels are not claimed to be bitwise reproducible",
    }


def _weights_saved(group_dir: Path) -> bool:
    return all((group_dir / name).is_file() for name in ("init_weights.npz", "raw_weights.npz", "ema_weights.npz"))


def _formal_step_count(n_train: int, epochs: int, batch_size: int) -> int:
    steps_per_epoch = max(1, int(np.ceil(int(n_train) / int(batch_size))))
    return int(steps_per_epoch) * int(epochs)


def _write_completed_step_log(path: Path, *, epochs: int, total_steps: int) -> None:
    """Record the completed-run step count when the in-memory epoch log was not flushed."""
    steps_per_epoch = int(total_steps) // int(epochs)
    rows = []
    for epoch in range(1, int(epochs) + 1):
        count = epoch * steps_per_epoch
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update_count": count,
                "ema_update_count": count,
                "val_rmse": "",
                "record_source": "completed_run_step_count",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _historical_condition_locked(condition_dir: Path, groups: Sequence[str]) -> bool:
    status_path = condition_dir / "condition_status.json"
    validation_path = condition_dir / "validation_report.json"
    if not status_path.is_file() or not validation_path.is_file():
        return False
    if str(_read_json(status_path).get("status")) != "COMPLETE":
        return False
    return all(plan_group_execution(condition_dir / "groups" / group) == "skip" for group in groups)


def finalize_saved_group(
    group_dir: Path,
    *,
    spec: GroupSpec,
    arrays: Mapping[str, Any],
    window_rows: Sequence[Mapping[str, Any]],
    target_scaler: Any,
    feature_columns: Sequence[str],
    input_shape: Tuple[int, ...],
    d_model: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    n_train: int,
) -> Dict[str, Any]:
    """Score sealed weights. This function does not train and does not replace an existing epoch log."""
    if not _training_sealed(group_dir):
        raise M7ContractError(f"{spec.group} is not sealed training evidence; weight files alone cannot be finalized")
    if _predictions_ready(group_dir):
        return _read_json(group_dir / "metrics.json")
    from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

    init_state = load_state(group_dir / "init_weights.npz")
    raw_state = load_state(group_dir / "raw_weights.npz")
    ema_state = load_state(group_dir / "ema_weights.npz")
    evidence = _read_json(group_dir / "training_evidence.json") if (group_dir / "training_evidence.json").is_file() else {}
    parameter_map = _read_json(group_dir / "parameter_map.json") if (group_dir / "parameter_map.json").is_file() else None
    drift = build_parameter_drift(
        init_state,
        raw_state,
        ema_state,
        conv_frozen=spec.conv_frozen,
        trainable_names=[],
    )
    contract_result = assert_update_contract(spec, drift, parameter_map)
    if not (group_dir / "parameter_drift.json").is_file():
        _write_json(group_dir / "parameter_drift.json", drift)
    y_pred = _predict_from_weights(
        arrays=arrays,
        cnn_weights=ema_state["cnn_full"],
        head_weights=ema_state["head_target"],
        revin_affine=ema_state["revin_affine"],
        revin_enabled=spec.revin_enabled,
        input_shape=input_shape,
        d_model=d_model,
        learning_rate=learning_rate,
        name_suffix=f"finalize_{spec.group}",
    )
    y_true = np.asarray(arrays["y_test"]).reshape(-1)
    metrics = compute_screening_metrics(
        y_true,
        y_pred,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
    )
    epoch_assessment = assess_epoch_validation_log(group_dir / "epoch_log.csv", expected_epochs=int(epochs))
    counter_assessment = assess_update_counter_evidence(evidence)
    if not (group_dir / "selected_checkpoint.json").is_file():
        _write_json(
            group_dir / "selected_checkpoint.json",
            {
                "last_epoch": int(epochs),
                "selected_epoch": int(epochs),
                "selected_weight_file": "ema_weights.npz",
                "selected_weight_variant": "EMA",
                "selection_rule": "final_epoch_ema",
                "early_stopping": False,
                "val_rmse_used_for_selection": False,
                "optimizer_update_count": evidence.get("optimizer_update_count"),
                "ema_initialization_count": evidence.get("ema_initialization_count"),
                "ema_update_count": evidence.get("ema_update_count"),
                "optimizer_update_count_evidence": evidence.get("optimizer_update_count_evidence", "missing"),
                "ema_update_count_evidence": evidence.get("ema_update_count_evidence", "missing"),
                "per_epoch_val_rmse_retained": bool(epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]),
            },
        )
    frame = pd.DataFrame(list(window_rows))
    frame.insert(0, "sample_index", np.arange(len(frame)))
    frame["y_true_normalized"] = y_true
    frame["y_pred_normalized"] = np.asarray(y_pred).reshape(-1)
    frame["y_true_original"] = np.asarray(metrics["y_true_original"]).reshape(-1)
    frame["y_pred_original"] = np.asarray(metrics["y_pred_original"]).reshape(-1)
    frame.to_csv(group_dir / "predictions.csv", index=False)
    _write_json(
        group_dir / "metrics.json",
        {
            "rmse_original": metrics["rmse_original"],
            "smape_original": metrics["smape_original"],
            "rmse_normalized": metrics["rmse_normalized"],
            "smape_normalized": metrics["smape_normalized"],
            "metric_space": metrics["metric_space"],
            "weight_variant": "EMA",
            "checkpoint": "ema_weights.npz",
            "selected_epoch": int(epochs),
            "last_epoch": int(epochs),
            "selection_rule": "final_epoch_ema",
            "prediction_source": "saved final-epoch EMA weights; no retraining",
        },
    )
    diagnostic_complete = bool(
        epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]
        and counter_assessment["UPDATE_COUNTER_EVIDENCE"] == "observed"
        and parameter_map is not None
    )
    _write_json(
        group_dir / "status.json",
        {
            "status": "ALL_REQUIRED_ARTIFACTS_COMPLETE" if diagnostic_complete else "TRAINING_COMPLETE_DIAGNOSTICS_INCOMPLETE",
            "group": spec.group,
            "pid": os.getpid(),
            "end_time": _now(),
            "rmse_original": metrics["rmse_original"],
            "smape_original": metrics["smape_original"],
            "selected_epoch": int(epochs),
            "TRAINING_COMPLETE": True,
            "PREDICTION_VERIFICATION": "PASS",
            "FREEZE_VERIFICATION": contract_result["freeze_value_check"] if spec.conv_frozen else "update_path_checked",
            "EPOCH_VALIDATION_LOG_COMPLETE": bool(epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]),
            "EPOCH_VALIDATION_LOG_REASON": epoch_assessment.get("reason"),
            "UPDATE_COUNTER_EVIDENCE": counter_assessment["UPDATE_COUNTER_EVIDENCE"],
            "DIAGNOSTIC_ARTIFACTS_COMPLETE": diagnostic_complete,
            "finalized_from_saved_weights": True,
            "epoch_history_backfilled": False,
        },
    )
    if not _predictions_ready(group_dir):
        raise M7ContractError(f"{spec.group} saved weights could not be finalized")
    return metrics


def execute_condition(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    output_root: Path,
    groups: Sequence[str] = GROUPS,
    resume: bool = False,
    project_root: Optional[Path] = None,
    epochs: Optional[int] = None,
    source_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    import tensorflow as tf

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    manifest = load_manifest(root)
    if dataset not in DATASETS or int(horizon) not in HORIZONS or int(seed) not in SEEDS:
        raise M7ContractError(f"condition outside the formal matrix: {dataset} h={horizon} seed={seed}")
    selected_groups = tuple(groups)
    resolved_root = assert_output_isolation(output_root, root)
    condition_dir = resolved_root / "conditions" / condition_key(dataset, int(horizon), int(seed))
    if condition_dir.exists() and not resume:
        raise M7ContractError(f"condition already exists; use resume: {condition_dir}")
    if resume and _historical_condition_locked(condition_dir, selected_groups):
        return {
            "status": "UNCHANGED",
            "condition_dir": str(condition_dir),
            "writes": False,
            "reason": "historical condition artifacts are locked; resume will not retrain or rewrite them",
        }
    condition_dir.mkdir(parents=True, exist_ok=True)
    log_path = condition_dir / "execution_log.txt"
    target_epochs = int(manifest["target_epochs"] if epochs is None else epochs)
    stage_epochs = int(manifest["source_epochs"] if source_epochs is None else source_epochs)
    batch_size = int(manifest["batch_size"])
    learning_rate, learning_rate_label = target_learning_rate(dataset)
    _write_json(
        condition_dir / "condition_status.json",
        {"status": "RUNNING", "pid": os.getpid(), "start_time": _now(), "groups": list(selected_groups)},
    )
    _log(log_path, f"START dataset={dataset} horizon={horizon} seed={seed} groups={','.join(selected_groups)}")
    started = time.perf_counter()
    try:
        prepared = _prepare_context(dataset, int(horizon), int(seed), root)
        source_dir = condition_dir / "sources"
        if resume and (source_dir / "status.json").is_file() and _read_json(source_dir / "status.json").get("status") == "COMPLETE":
            _log(log_path, "RESTORE_STAGE_A")
            stage_a = _load_stage_a(source_dir)
        else:
            _log(log_path, "TRAIN_STAGE_A")
            stage_a = _train_stage_a(
                prepared, horizon=int(horizon), seed=int(seed), epochs=stage_epochs, batch_size=batch_size
            )
            _save_stage_a(stage_a, source_dir, seed=int(seed))
        bundle = _shared_bundle(stage_a, prepared, horizon=int(horizon), seed=int(seed))
        nearest = _nearest_snapshot(bundle, stage_a)
        from src.transfer_methods.ms_katl_lite import digest_weight_arrays

        nearest_conv_digest = digest_weight_arrays(
            [array for name in ("conv1", "conv2", "conv3") for array in nearest["conv"][name]]
        )
        _write_json(
            source_dir / "nearest_source.json",
            {
                "index": nearest["index"],
                "identity": list(nearest["identity"]),
                "distance": nearest["distance"],
                "distances": [float(value) for value in stage_a.source_distances],
                "selection_rule": "argmin euclidean distance within the same top-3; not test error",
                "conv_digest": nearest_conv_digest,
            },
        )
        contract = _condition_contract(
            dataset=dataset,
            horizon=int(horizon),
            seed=int(seed),
            learning_rate_label=learning_rate_label,
            target_epochs=target_epochs,
            source_epochs=stage_epochs,
            batch_size=batch_size,
            stage_a=stage_a,
        )
        contract_path = condition_dir / "condition_contract.json"
        if resume and contract_path.is_file():
            assert_resume_contract(_read_json(contract_path), contract)
        elif resume and any(group_is_complete(condition_dir / "groups" / group) for group in GROUPS):
            raise M7ContractError("resume refused: completed groups exist without a matching condition contract")
        elif not contract_path.is_file():
            _write_json(contract_path, contract)
        window_rows = test_window_rows(prepared["target_model_df"], int(horizon), int(manifest["input_window"]))
        if len(window_rows) != len(np.asarray(bundle.y_test).reshape(-1)):
            raise M7ContractError(
                f"test window metadata {len(window_rows)} != y_test {len(np.asarray(bundle.y_test).reshape(-1))}"
            )
        data_summary = {
            "feature_columns": list(bundle.target_feature_columns),
            "configured_split": {
                "train_days": int(manifest["target_train_days"]),
                "val_days": int(manifest["target_val_days"]),
                "test_days": int(manifest["target_test_days"]),
                "input_window": int(manifest["input_window"]),
            },
            "actual_split": split_day_summary(prepared["target_model_df"]),
            "n_train_windows": int(len(np.asarray(bundle.y_train).reshape(-1))),
            "n_val_windows": int(len(np.asarray(bundle.y_val).reshape(-1))),
            "n_test_windows": int(len(window_rows)),
            "sample_index_start": 0,
            "sample_index_end": int(len(window_rows) - 1),
            "scaler_data_min": np.asarray(bundle.target_scaler.data_min_, dtype=np.float64).tolist(),
            "scaler_data_max": np.asarray(bundle.target_scaler.data_max_, dtype=np.float64).tolist(),
            "split_digest": dict(bundle.target_split_digest),
        }
        _save_bundle_arrays(condition_dir / "bundle_arrays.npz", bundle)
        probe_indices = select_probe_indices(len(window_rows), int(manifest["input_window"]))
        probe_meta = build_probe_metadata(window_rows, probe_indices, stride=int(manifest["input_window"]))
        _write_json(condition_dir / "probe_windows.json", probe_meta)
        np.save(condition_dir / "probe_inputs.npy", np.asarray(bundle.X_test)[probe_indices])
        from src.utils.d4_d6_runtime import load_default_metric_protocol

        metric_protocol = load_default_metric_protocol(root)
        paired_head: Optional[List[np.ndarray]] = None
        paired_revin: Optional[Dict[str, np.ndarray]] = None
        m6_dir = condition_dir / "groups" / "M6"
        if resume and group_is_complete(m6_dir):
            m6_init = load_state(m6_dir / "init_weights.npz")
            paired_head = m6_init["head_target"]
            paired_revin = m6_init["revin_affine"]
            saved_contract = _read_json(contract_path)
            if saved_contract.get("m6_head_target_digest") not in (None, _digest_state(m6_init)["head_target"]):
                raise M7ContractError("resume refused: M6 head_target digest does not match the condition contract")
        for group in execution_order(selected_groups):
            spec = GROUP_SPECS[group]
            group_dir = condition_dir / "groups" / group
            group_dir.mkdir(parents=True, exist_ok=True)
            if plan_group_execution(group_dir) == "skip":
                _log(log_path, f"SKIP_COMPLETE {group}")
                continue
            if plan_group_execution(group_dir) == "finalize_saved_weights":
                _log(log_path, f"FINALIZE_SAVED_WEIGHTS {group}")
                try:
                    metrics = finalize_saved_group(
                        group_dir,
                        spec=spec,
                        arrays=_load_bundle_arrays(condition_dir / "bundle_arrays.npz"),
                        window_rows=window_rows,
                        target_scaler=bundle.target_scaler,
                        feature_columns=bundle.target_feature_columns,
                        input_shape=tuple(stage_a.input_shape),
                        d_model=int(bundle.d_model),
                        learning_rate=learning_rate,
                        epochs=target_epochs,
                        batch_size=batch_size,
                        n_train=int(len(np.asarray(bundle.y_train).reshape(-1))),
                    )
                    if group == "M6":
                        m6_init = load_state(group_dir / "init_weights.npz")
                        paired_head = m6_init["head_target"]
                        paired_revin = m6_init["revin_affine"]
                    _log(log_path, f"COMPLETE {group} rmse={metrics['rmse_original']} smape={metrics['smape_original']}")
                except Exception as exc:
                    _write_json(
                        group_dir / "error.json",
                        {
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    _write_json(
                        group_dir / "status.json",
                        {
                            "status": "FAILED",
                            "group": group,
                            "pid": os.getpid(),
                            "end_time": _now(),
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                    )
                    _log(log_path, f"FAILED {group} {type(exc).__name__}: {exc}")
                tf.keras.backend.clear_session()
                continue
            status_path = group_dir / "status.json"
            if status_path.is_file():
                previous = _read_json(status_path)
                if str(previous.get("status")) == "RUNNING" and _pid_alive(previous.get("pid")):
                    raise M7ContractError(f"{group} is already running as pid {previous.get('pid')}")
                if str(previous.get("status")) == "RUNNING":
                    previous["status"] = "INTERRUPTED"
                    previous["end_time"] = _now()
                    _write_json(status_path, previous)
            group_started = time.perf_counter()
            _write_json(status_path, {"status": "RUNNING", "group": group, "pid": os.getpid(), "start_time": _now()})
            _log(log_path, f"TRAIN {group}")
            try:
                if group == "M6":
                    group_head = None
                    group_revin = None
                else:
                    if paired_head is None or (spec.revin_enabled and paired_revin is None):
                        raise M7ContractError(
                            f"{group} cannot start because the formal M6 initialization has not been captured"
                        )
                    group_head = paired_head
                    group_revin = paired_revin
                arm = _run_group(
                    spec=spec,
                    bundle=bundle,
                    nearest=nearest,
                    head_init=group_head,
                    revin_init=group_revin,
                    learning_rate=learning_rate,
                    epochs=target_epochs,
                    batch_size=batch_size,
                    seed=int(seed),
                    metric_protocol=metric_protocol,
                    epoch_log_path=group_dir / "epoch_log.csv",
                )
                states = arm["weight_states"]
                if group == "M6":
                    paired_head = [np.array(weight, copy=True) for weight in states["init"]["head_target"]]
                    paired_revin = states["init"]["revin_affine"]
                    contract["m6_head_target_digest"] = _digest_state(states["init"])["head_target"]
                    contract["m6_revin_affine_digest"] = _digest_state(states["init"])["revin_affine"]
                    contract["m6_formal_correspondence"] = {
                        "runtime_arm": "fused_target_only",
                        "initialization_overrides": False,
                        "head_target_source": "weights built by run_ablation_arm after set_protocol_seed and before the first optimizer step",
                        "conv_source": "bundle.fused_target_snapshot selected by use_fused_target_init",
                        "revin_affine_source": "RevIN ones/zeros initializer inside run_ablation_arm",
                        "other_groups_copy_head_and_revin": True,
                        "config_match_is_not_bitwise_reproduction": True,
                        "checkpoint_selection": "final epoch EMA; validation RMSE is recorded and does not select the checkpoint",
                    }
                    _write_json(contract_path, contract)
                    _write_json(condition_dir / "m6_formal_correspondence.json", contract["m6_formal_correspondence"])
                save_state(group_dir / "init_weights.npz", states["init"])
                save_state(group_dir / "raw_weights.npz", states["raw"])
                save_state(group_dir / "ema_weights.npz", states["ema"])
                counters = dict(states.get("update_counters") or {})
                _write_json(
                    group_dir / "training_evidence.json",
                    {
                        "training_finished": True,
                        "epochs_requested": int(target_epochs),
                        "epochs_completed": int(target_epochs),
                        **counters,
                    },
                )
                if states.get("parameter_map") is not None:
                    _write_json(group_dir / "parameter_map.json", states["parameter_map"])
                digests = _digest_state(states["init"])
                if group == "M7":
                    digests["nearest_source_conv"] = nearest_conv_digest
                _write_json(
                    group_dir / "init_snapshot.json",
                    {
                        "group": group,
                        "init_mode": spec.init_mode,
                        "digests": {
                            "conv": digests["conv"],
                            "head_target": digests["head_target"],
                            "revin_affine": digests["revin_affine"],
                        },
                        "nearest_source_conv": nearest_conv_digest if group == "M7" else None,
                        "nearest_source_identity": list(nearest["identity"]) if group == "M7" else None,
                    },
                )
                _write_json(group_dir / "data_split.json", data_summary)
                resolved = _resolved_config(
                    spec=spec,
                    dataset=dataset,
                    horizon=int(horizon),
                    seed=int(seed),
                    learning_rate=learning_rate,
                    learning_rate_label=learning_rate_label,
                    manifest={**manifest, "target_epochs": target_epochs, "batch_size": batch_size},
                    arm=arm,
                )
                _write_json(group_dir / "resolved_config.json", resolved)
                drift = build_parameter_drift(
                    states["init"],
                    states["raw"],
                    states["ema"],
                    conv_frozen=spec.conv_frozen,
                    trainable_names=list(states["trainable_names"]),
                )
                _write_json(group_dir / "parameter_drift.json", drift)
                contract_result = assert_update_contract(spec, drift, states.get("parameter_map"))
                epoch_assessment = assess_epoch_validation_log(
                    group_dir / "epoch_log.csv",
                    expected_epochs=int(target_epochs),
                )
                counter_assessment = assess_update_counter_evidence(counters)
                from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

                metrics = compute_screening_metrics(
                    arm["y_true"],
                    arm["y_pred"],
                    target_scaler=bundle.target_scaler,
                    feature_columns=bundle.target_feature_columns,
                )
                selected = {
                    "last_epoch": int(target_epochs),
                    "selected_epoch": int(target_epochs),
                    "last_epoch_weight_file": "raw_weights.npz",
                    "last_epoch_weight_variant": "RAW",
                    "selected_weight_file": "ema_weights.npz",
                    "selected_weight_variant": "EMA",
                    "selection_rule": "final_epoch_ema",
                    "early_stopping": False,
                    "val_rmse_used_for_selection": False,
                    "optimizer_update_count": counters.get("optimizer_update_count"),
                    "ema_initialization_count": counters.get("ema_initialization_count"),
                    "ema_update_count": counters.get("ema_update_count"),
                    "optimizer_update_count_evidence": counters.get("optimizer_update_count_evidence"),
                    "ema_update_count_evidence": counters.get("ema_update_count_evidence"),
                    "per_epoch_val_rmse_retained": bool(epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]),
                    "val_metric_weight_variant": "RAW",
                }
                _write_json(group_dir / "selected_checkpoint.json", selected)
                frame = pd.DataFrame(window_rows)
                frame.insert(0, "sample_index", np.arange(len(frame)))
                frame["y_true_normalized"] = np.asarray(arm["y_true"]).reshape(-1)
                frame["y_pred_normalized"] = np.asarray(arm["y_pred"]).reshape(-1)
                frame["y_true_original"] = np.asarray(metrics["y_true_original"]).reshape(-1)
                frame["y_pred_original"] = np.asarray(metrics["y_pred_original"]).reshape(-1)
                frame.to_csv(group_dir / "predictions.csv", index=False)
                _write_json(
                    group_dir / "metrics.json",
                    {
                        "rmse_original": metrics["rmse_original"],
                        "smape_original": metrics["smape_original"],
                        "rmse_normalized": metrics["rmse_normalized"],
                        "smape_normalized": metrics["smape_normalized"],
                        "metric_space": metrics["metric_space"],
                        "weight_variant": "EMA",
                        "checkpoint": "ema_weights.npz",
                        "selected_epoch": int(target_epochs),
                        "last_epoch": int(target_epochs),
                        "selection_rule": "final_epoch_ema",
                    },
                )
                assert_update_contract_result = contract_result
                diagnostic_complete = bool(
                    epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]
                    and counter_assessment["UPDATE_COUNTER_EVIDENCE"] == "observed"
                    and states.get("parameter_map") is not None
                )
                group_status = (
                    "ALL_REQUIRED_ARTIFACTS_COMPLETE"
                    if diagnostic_complete
                    else "TRAINING_COMPLETE_DIAGNOSTICS_INCOMPLETE"
                )
                elapsed = time.perf_counter() - group_started
                _write_json(
                    status_path,
                    {
                        "status": group_status,
                        "group": group,
                        "pid": os.getpid(),
                        "start_time": _read_json(status_path).get("start_time"),
                        "end_time": _now(),
                        "elapsed_seconds": elapsed,
                        "rmse_original": metrics["rmse_original"],
                        "smape_original": metrics["smape_original"],
                        "selected_epoch": int(target_epochs),
                        "TRAINING_COMPLETE": True,
                        "PREDICTION_VERIFICATION": "PASS",
                        "FREEZE_VERIFICATION": contract_result["freeze_value_check"] if spec.conv_frozen else "update_path_checked",
                        "EPOCH_VALIDATION_LOG_COMPLETE": bool(epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]),
                        "EPOCH_VALIDATION_LOG_REASON": epoch_assessment.get("reason"),
                        "UPDATE_COUNTER_EVIDENCE": counter_assessment["UPDATE_COUNTER_EVIDENCE"],
                        "DIAGNOSTIC_ARTIFACTS_COMPLETE": diagnostic_complete,
                        "optimizer_update_count": counters.get("optimizer_update_count"),
                        "ema_update_count": counters.get("ema_update_count"),
                        "unchanged_parameter_observations": assert_update_contract_result["observations"],
                    },
                )
                if not group_is_complete(group_dir):
                    raise M7ContractError(f"{group} finished without the required artifacts")
                _log(log_path, f"COMPLETE {group} rmse={metrics['rmse_original']} smape={metrics['smape_original']}")
            except Exception as exc:
                start_time = None
                if status_path.is_file():
                    start_time = str(_read_json(status_path).get("start_time") or "")
                mark_group_failed(
                    group_dir,
                    exc,
                    start_time=start_time,
                    elapsed_seconds=time.perf_counter() - group_started,
                )
                _log(log_path, f"FAILED {group} {type(exc).__name__}: {exc}")
            tf.keras.backend.clear_session()
        arrays = _load_bundle_arrays(condition_dir / "bundle_arrays.npz")
        probe = np.load(condition_dir / "probe_inputs.npy")
        from src.transfer_methods.ms_katl_primary_target_lr_screening import STAGE_A_LEARNING_RATE

        cka = compute_cka_report(
            condition_dir,
            input_shape=tuple(bundle.stage_a.input_shape),
            learning_rate=float(STAGE_A_LEARNING_RATE),
            probe=probe,
            probe_meta=probe_meta,
        )
        _write_json(condition_dir / "cka.json", cka)
        validation = validate_condition(
            condition_dir,
            selected_groups,
            arrays=arrays,
            input_shape=tuple(bundle.stage_a.input_shape),
            d_model=int(bundle.d_model),
            learning_rate=float(STAGE_A_LEARNING_RATE),
        )
        validation["elapsed_seconds"] = time.perf_counter() - started
        diagnostic_gaps = []
        for group in selected_groups:
            group_dir = condition_dir / "groups" / group
            epoch_assessment = assess_epoch_validation_log(group_dir / "epoch_log.csv", expected_epochs=target_epochs)
            evidence_path = group_dir / "training_evidence.json"
            evidence = _read_json(evidence_path) if evidence_path.is_file() else {}
            counter_assessment = assess_update_counter_evidence(evidence)
            if not epoch_assessment["EPOCH_VALIDATION_LOG_COMPLETE"]:
                diagnostic_gaps.append(f"{group} epoch validation log: {epoch_assessment.get('reason')}")
            if counter_assessment["UPDATE_COUNTER_EVIDENCE"] != "observed":
                diagnostic_gaps.append(f"{group} update counters: {counter_assessment['UPDATE_COUNTER_EVIDENCE']}")
            if not (group_dir / "parameter_map.json").is_file():
                diagnostic_gaps.append(f"{group} parameter map missing")
        validation["diagnostic_gaps"] = diagnostic_gaps
        validation["all_required_artifacts_complete"] = validation["status"] == "PASS" and not diagnostic_gaps
        if validation["status"] != "PASS":
            condition_status = "FAILED"
        elif diagnostic_gaps:
            condition_status = "TRAINING_COMPLETE_DIAGNOSTICS_INCOMPLETE"
        else:
            condition_status = "ALL_REQUIRED_ARTIFACTS_COMPLETE"
        validation["condition_status"] = condition_status
        _write_json(condition_dir / "validation_report.json", validation)
        condition_status = validation["condition_status"]
        _write_json(
            condition_dir / "condition_status.json",
            {
                "status": condition_status,
                "pid": os.getpid(),
                "end_time": _now(),
                "elapsed_seconds": validation["elapsed_seconds"],
                "groups": list(selected_groups),
                "validation_status": validation["status"],
            },
        )
        _log(log_path, f"CONDITION_{condition_status}")
        return {"status": condition_status, "condition_dir": str(condition_dir), "validation": validation}
    except Exception as exc:
        _write_json(
            condition_dir / "condition_error.json",
            {"exception_type": type(exc).__name__, "exception_message": str(exc), "traceback": traceback.format_exc()},
        )
        _write_json(
            condition_dir / "condition_status.json",
            {"status": "FAILED", "pid": os.getpid(), "end_time": _now(), "exception_type": type(exc).__name__, "exception_message": str(exc)},
        )
        _log(log_path, f"CONDITION_FAILED {type(exc).__name__}: {exc}")
        raise


def write_supplemental_audit(condition_dir: Path) -> Path:
    """Write an audit beside a finished condition without changing its historical files."""
    condition_dir = Path(condition_dir)
    dest = condition_dir / "supplemental_audit"
    dest.mkdir(parents=True, exist_ok=True)
    historical_status = _read_json(condition_dir / "condition_status.json")
    validation = _read_json(condition_dir / "validation_report.json")
    probe = _read_json(condition_dir / "probe_windows.json")
    cka = _read_json(condition_dir / "cka.json")
    execution_log = (condition_dir / "execution_log.txt").read_text(encoding="utf-8")
    groups = []
    for group in GROUPS:
        group_dir = condition_dir / "groups" / group
        status = _read_json(group_dir / "status.json")
        checkpoint = _read_json(group_dir / "selected_checkpoint.json")
        drift = _read_json(group_dir / "parameter_drift.json")
        init_meta = _read_json(group_dir / "init_snapshot.json")
        metrics = _read_json(group_dir / "metrics.json")
        epoch_assessment = assess_epoch_validation_log(group_dir / "epoch_log.csv")
        groups.append(
            {
                "group": group,
                "TRAINING_COMPLETE": True,
                "training_evidence": (
                    "Saved init/RAW/EMA weights, resolved config, and the retained trainer restatement "
                    "in selected_checkpoint.json. execution_log.txt records TRAIN then the later "
                    "FINALIZE_SAVED_WEIGHTS pass. The original train_ms_katl_head Finished line was "
                    "stdout and is not stored in execution_log.txt."
                ),
                "PREDICTION_VERIFICATION": "PASS",
                "prediction_reload_max_abs": validation["reload"][group]["max_abs"],
                "metric_recompute_rmse_abs": validation["metric_recompute"][group]["rmse_abs"],
                "metric_recompute_smape_abs": validation["metric_recompute"][group]["smape_abs"],
                "rmse_original": metrics["rmse_original"],
                "smape_original": metrics["smape_original"],
                "n_predictions": int(pd.read_csv(group_dir / "predictions.csv").shape[0]),
                "INITIALIZATION_PAIRING_VERIFICATION": validation["pairing"]["status"],
                "init_digests": init_meta["digests"],
                "FREEZE_VERIFICATION": (
                    "value_freeze_observed_membership_not_observed"
                    if group in FROZEN_GROUPS
                    else "update_path_not_observed_from_layer_objects"
                ),
                "frozen_conv_raw_max_abs": (
                    max(drift["layers"][name]["raw"]["max_abs"] for name in ("conv1", "conv2", "conv3"))
                    if group in FROZEN_GROUPS
                    else None
                ),
                "frozen_conv_ema_max_abs": (
                    max(drift["layers"][name]["ema"]["max_abs"] for name in ("conv1", "conv2", "conv3"))
                    if group in FROZEN_GROUPS
                    else None
                ),
                "EPOCH_VALIDATION_LOG_COMPLETE": False,
                "epoch_log_assessment": epoch_assessment,
                "UPDATE_COUNTER_EVIDENCE": "trainer_reported_and_inferred",
                "optimizer_update_count": checkpoint.get("optimizer_update_count"),
                "optimizer_update_count_evidence": "trainer_reported_stdout_restatement",
                "ema_update_count": checkpoint.get("ema_update_count"),
                "ema_initialization_count": None,
                "ema_update_count_evidence": "inferred_from_training_steps_and_update_rule",
                "DIAGNOSTIC_ARTIFACTS_COMPLETE": False,
                "historical_group_status": status.get("status"),
                "parameter_map_saved_at_training": False,
            }
        )
    cka_rows = []
    for comparison in cka["comparisons"]:
        revin_note = any("RevIN" in str(note) for note in comparison.get("notes") or [])
        for layer, result in comparison["layers"].items():
            cka_rows.append(
                {
                    "comparison": comparison["comparison"],
                    "left_group": comparison["left"]["group"],
                    "left_state": "INIT" if comparison["left"]["state"] == "init" else "FINAL_EMA",
                    "right_group": comparison["right"]["group"],
                    "right_state": "INIT" if comparison["right"]["state"] == "init" else "FINAL_EMA",
                    "layer": layer,
                    "cka": result["value"],
                    "status": result["status"],
                    "degeneracy_reason": result.get("reason"),
                    "n_samples": probe["n_samples"],
                    "window_ids": probe["window_ids"],
                    "date_range": probe["date_range"],
                    "partition": probe["partition"],
                    "nonoverlapping": probe["nonoverlapping"],
                    "within_group_revin": revin_note or comparison["left"]["group"] in REVIN_GROUPS or comparison["right"]["group"] in REVIN_GROUPS,
                    "numerically_degenerate": result["status"] != "OK",
                    "probe_statement": "CKA is computable on this 17-window probe. That is not a claim of statistical sample sufficiency.",
                }
            )
    report = {
        "historical_condition_status": historical_status.get("status"),
        "historical_condition_status_scope": (
            "The retained CONDITION_STATUS=COMPLETE means the seven groups finished training and the "
            "later prediction reload plus metric recompute passed. It does not mean the per-epoch "
            "validation log or directly observed optimizer and EMA counters were complete."
        ),
        "historical_validation_report_preserved": True,
        "historical_execution_log_preserved": "FINALIZE_SAVED_WEIGHTS" in execution_log and "CONDITION_COMPLETE" in execution_log,
        "groups": groups,
        "irrecoverable": [
            "Per-epoch validation RMSE was computed on RAW weights during training and was not flushed before the post-training audit raised. It cannot be reconstructed from the final checkpoint.",
            "Optimizer iterations were not read at the start and end of that run. The retained count 50 is the trainer-reported steps value, restated during finalize, not an observed iterations delta.",
            "EMA initialization count was not recorded. The retained EMA update count was inferred from steps=50 and the one-update-per-step rule.",
            "No layer-object parameter map was saved, so optimizer membership and EMA-shadow membership were not directly observed.",
        ],
        "cka_notes": [
            "The 17 probe windows make CKA computable. They are not claimed to be statistically sufficient.",
            "A high CKA does not mean the parameters were unchanged.",
            "A high CKA does not by itself prove that source information was retained or that target adaptation was effective.",
            "M6 representation change can include both convolution updates and RevIN affine updates.",
            "CKA was not used for training, checkpoint selection, or tuning.",
            "This table is rendered from the existing cka.json. No new forward pass was run.",
        ],
        "full_run_readiness": (
            "Future runs can persist RAW validation rows as each epoch finishes, record optimizer iteration deltas and EMA counts at the update site, and map parameters by layer object. "
            "A run is not marked fully complete when those diagnostic records are missing, and a diagnostic gap does not retrain a sealed group. "
            "The 420-condition matrix has not been started."
        ),
    }
    _write_json(dest / "report.json", report)
    with (dest / "cka_layer_detail.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cka_rows[0]))
        writer.writeheader()
        for row in cka_rows:
            flat = dict(row)
            flat["window_ids"] = " ".join(str(item) for item in row["window_ids"])
            flat["date_range"] = json.dumps(row["date_range"], ensure_ascii=False)
            writer.writerow(flat)
    return dest


def render_plan(
    *,
    dataset: Optional[str] = None,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
    groups: Sequence[str] = GROUPS,
    output_root: Optional[Path] = None,
) -> str:
    manifest = load_manifest()
    if dataset is None and horizon is None and seed is None:
        conditions = enumerate_conditions(manifest)
    else:
        if dataset is None or horizon is None or seed is None:
            raise M7ContractError("a single condition requires dataset, horizon, and seed together")
        conditions = [
            {
                "dataset": dataset,
                "target": TARGETS[dataset],
                "horizon": int(horizon),
                "seed": int(seed),
                "condition_key": condition_key(dataset, int(horizon), int(seed)),
                "source_learning_rate": SOURCE_LR_LABEL,
                "target_learning_rate": TARGET_LR_LABELS[dataset],
                "groups": list(groups),
            }
        ]
    lines = [
        "PROTOCOL=SITA_M1_M7_ABLATION",
        "EXECUTION_STATUS=PLANNED",
        "TRAINING_CALLED=NO",
        f"SOURCE_LR={SOURCE_LR_LABEL}",
        f"TARGET_LR_D1={TARGET_LR_LABELS['D1']}",
        f"TARGET_LR_D4={TARGET_LR_LABELS['D4']}",
        f"TARGET_LR_D5={TARGET_LR_LABELS['D5']}",
        f"TARGET_LR_D6={TARGET_LR_LABELS['D6']}",
        f"CONDITIONS={len(conditions)}",
        f"GROUPS={','.join(groups)}",
        f"PLANNED_GROUP_RUNS={len(conditions) * len(groups)}",
        f"OUTPUT_ROOT={output_root if output_root is not None else default_output_root()}",
        "SELECTION_RULE=final_epoch_ema",
        "EARLY_STOPPING=OFF",
    ]
    for row in conditions:
        lines.append(
            "PLAN {dataset} {target} h={horizon} seed={seed} lr={target_learning_rate} groups={groups}".format(
                **{**row, "groups": ",".join(groups)}
            )
        )
    return "\n".join(lines) + "\n"


def write_plan_file(
    output_root: Path,
    text: str,
    *,
    dataset: Optional[str],
    horizon: Optional[int],
    seed: Optional[int],
    groups: Sequence[str],
    project_root: Optional[Path] = None,
) -> None:
    resolved = assert_output_isolation(output_root, project_root)
    resolved.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PLANNED",
        "training_called": False,
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "groups": list(groups),
        "text": text,
    }
    _write_json(resolved / "plan.json", payload)
    if str(payload["status"]) == "COMPLETE":
        raise M7ContractError("plan file must not be marked COMPLETE")
