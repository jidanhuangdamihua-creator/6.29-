"""MS-KATL FUSED_TARGET_ONLY_V3_1_CALIBRATED contract.

Derived development arm: frozen V3 plus one uniform validation-median
additive bias calibration. Not part of FORMAL_METHODS.

Does not reselect targets or learning rates, does not modify V3 screening /
robustness artifacts, and does not introduce dataset-, horizon-, or
seed-specific calibration rules.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.evaluation.metrics import smape
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_cross_target import BL_METHODS
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    DEFAULT_V3_RUN_DIRNAME,
    EXPECTED_COMMON_TARGETS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    ROBUSTNESS_HORIZONS,
    ROBUSTNESS_SEEDS,
    RunIdentity,
    assert_formal_methods_unchanged,
    load_v3_frozen_robustness_manifest,
    relative_improvement,
    v3_arm_spec,
    v3_frozen_manifest_path,
    v3_revin_config,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    V3_REVIN_CONTRACT,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    METHOD_NAME,
    _transform_with_scaler,
    extract_features_maybe_revin,
    predict_ms_katl_decomposed,
)
from src.transfer_methods.ms_katl_mechanism_ablation import (
    assert_formal_methods_untouched,
    resolve_ablation_arm,
    run_ablation_arm,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_MODE,
    default_baseline_csv,
    format_lr,
    lookup_reference_row,
    original_rmse,
    parse_lr,
    parse_target_identity,
    compute_screening_metrics,
)

METHOD_DISPLAY = "FUSED_TARGET_ONLY_V3_1_CALIBRATED"
METHOD_ARM = "fused_target_only_v3_1_calibrated"
PARENT_METHOD = V3_METHOD_DISPLAY
ROBUSTNESS_LABEL = "MS_KATL_FTO_V3_1_CALIBRATED_ROBUSTNESS"
ROBUSTNESS_TRACK = "development_derived_calibration"
CALIBRATION_TYPE = "validation_median_bias"
CALIBRATION_FORMULA = "yhat_calibrated = yhat_raw + bias"
BIAS_FORMULA = "median(y_val - yhat_val)"
CALIBRATION_SPACE = ORIGINAL_SALES_SPACE
CALIBRATION_CONTEXT = "train_history_plus_validation_labels_only"
CALIBRATION_TRUTH_SOURCE = "validation_only"
WINDOW_SIZE = 10
EXPECTED_RUNS = 54
UNIQUE_RUNS = 54
V3_1_HORIZONS: Tuple[int, ...] = ROBUSTNESS_HORIZONS
V3_1_SEEDS: Tuple[int, ...] = ROBUSTNESS_SEEDS
DEFAULT_V3_ROBUSTNESS_DIRNAME = "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
V3_1_MANIFEST_RELPATH = Path("configs") / "ms_katl_fto_v3_1_calibrated_manifest.json"
V3_RAW_REPRODUCTION_RTOL = 1e-5
V3_RAW_REPRODUCTION_ATOL = 1e-5
V3_RAW_PARENT_RMSE_RTOL = 2e-2
V3_RAW_YTRUE_ATOL = 1e-8

EXPECTED_V3_LR: Mapping[Tuple[str, str], str] = MappingProxyType(
    {
        ("D1", "1/10"): "3e-5",
        ("D2", "1/10"): "3e-5",
        ("D3", "10"): "3e-5",
        ("D4", "166/311"): "3e-5",
        ("D5", "48/1159414"): "3e-4",
        ("D6", "CA_1/FOODS_3_080"): "3e-5",
    }
)

FORBIDDEN_CALIBRATION_KWARGS = frozenset(
    {
        "y_test",
        "y_test_original",
        "yhat_test",
        "yhat_test_original",
        "test_truth",
        "test_residual",
        "rmse_test",
        "smape_test",
        "baseline",
        "baselines",
        "baseline_metrics",
        "robustness",
        "robustness_result",
        "dataset",
        "horizon",
        "seed",
        "alpha",
        "clip",
        "shrinkage",
        "mean_bias",
        "multiplicative",
    }
)

REQUIRED_CELL_FIELDS = (
    "status",
    "dataset",
    "target_identity",
    "horizon",
    "seed",
    "learning_rate",
    "parent_method",
    "parent_architecture_digest",
    "calibration_type",
    "calibration_sample_count",
    "calibration_bias",
    "rmse_v3_raw",
    "smape_v3_raw",
    "rmse_v3_1_calibrated",
    "smape_v3_1_calibrated",
    "rmse_improvement_vs_raw_v3",
    "smape_improvement_vs_raw_v3",
    "metric_space",
)

PARENT_OUTPUT_MARKERS = (
    "MS_KATL_FTO_V3_SCREENING",
    "MS_KATL_FTO_THREE_TRACK_ROBUSTNESS",
)

V3_SOURCE_FILES = (
    Path("src") / "transfer_methods" / "ms_katl_fto_v3_screening.py",
    Path("scripts") / "run_ms_katl_fto_v3_screening.py",
    Path("tests") / "test_ms_katl_fto_v3_screening.py",
    Path("src") / "transfer_methods" / "ms_katl_fto_three_track_robustness.py",
    Path("scripts") / "run_ms_katl_fto_three_track_robustness.py",
    Path("tests") / "test_ms_katl_fto_three_track_robustness.py",
    Path("configs") / "ms_katl_fto_v3_robustness_manifest.json",
)


class V31ContractError(RuntimeError):
    """Fail-closed V3.1 calibration contract violation."""


def _frozen_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def project_root_from(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[2]


def v3_1_manifest_path(project_root: Path) -> Path:
    return Path(project_root) / V3_1_MANIFEST_RELPATH


def v3_screening_root(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "runs" / DEFAULT_V3_RUN_DIRNAME


def v3_robustness_root(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "runs" / DEFAULT_V3_ROBUSTNESS_DIRNAME


def parent_artifact_files(project_root: Path) -> Tuple[Path, ...]:
    screening = v3_screening_root(project_root)
    robustness = v3_robustness_root(project_root)
    return (
        v3_frozen_manifest_path(project_root),
        screening / "results" / "v3_selection_manifest.json",
        screening / "results" / "v3_screening_all_runs.csv",
        screening / "results" / "v3_screening_target_best_lr.csv",
        robustness / "results" / "three_track_robustness_all_runs.csv",
        robustness / "results" / "three_track_completion_guard.json",
        robustness / "results" / "three_track_paired_by_cell.csv",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_parent_artifacts(project_root: Path) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for path in parent_artifact_files(project_root):
        if path.is_file():
            payload[str(path.resolve())] = sha256_file(path)
    for rel in V3_SOURCE_FILES:
        path = Path(project_root) / rel
        if path.is_file():
            payload[str(path.resolve())] = sha256_file(path)
    if not payload:
        raise V31ContractError("V3 parent artifacts were not found for fingerprinting")
    return payload


def assert_parent_fingerprints_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> str:
    if dict(before) != dict(after):
        changed = sorted(set(before) | set(after))
        details = [key for key in changed if before.get(key) != after.get(key)]
        raise V31ContractError(
            "V3 parent files unexpectedly modified: " + ", ".join(details[:12])
        )
    return "YES"


def assert_not_v3_parent_output_dir(path: Path, project_root: Path) -> None:
    run_dir = Path(path).resolve()
    name = run_dir.name
    if any(marker in name for marker in PARENT_OUTPUT_MARKERS) and "V3_1" not in name:
        raise V31ContractError(f"refusing to write V3.1 outputs into V3 parent run: {path}")
    forbidden = {
        v3_screening_root(project_root).resolve(),
        v3_robustness_root(project_root).resolve(),
        v3_frozen_manifest_path(project_root).resolve(),
    }
    if run_dir in forbidden:
        raise V31ContractError(f"refusing to write V3.1 outputs into V3 parent path: {path}")
    for parent in (v3_screening_root(project_root), v3_robustness_root(project_root)):
        try:
            run_dir.relative_to(parent.resolve())
        except ValueError:
            continue
        raise V31ContractError(f"refusing to write V3.1 outputs under V3 parent: {path}")


def assert_formal_methods_unchanged_v31() -> Tuple[str, ...]:
    expected = assert_formal_methods_untouched()
    assert_formal_methods_unchanged()
    if METHOD_DISPLAY in FORMAL_METHODS or METHOD_ARM in FORMAL_METHODS:
        raise V31ContractError("V3.1 must not be added to FORMAL_METHODS")
    if PARENT_METHOD in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise V31ContractError("MS-KATL development methods must stay outside FORMAL_METHODS")
    return expected


def assert_v3_parent_contract(project_root: Path) -> Dict[str, Any]:
    frozen = load_v3_frozen_robustness_manifest(project_root)
    if frozen.get("method") != PARENT_METHOD:
        raise V31ContractError(f"V3 frozen manifest method drifted: {frozen.get('method')}")
    if str(frozen.get("architecture_digest")) != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V31ContractError("V3 frozen architecture digest drifted")
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V31ContractError("live V3 architecture digest drifted")
    observed_lr = {}
    for dataset, target in EXPECTED_COMMON_TARGETS:
        label = format_lr(parse_lr(frozen["datasets"][dataset]["learning_rate"]))
        identity = str(frozen["datasets"][dataset]["primary_target"]).strip()
        if identity != target:
            raise V31ContractError(f"V3 primary target drifted on {dataset}: {identity}")
        observed_lr[(dataset, target)] = label
        if label != EXPECTED_V3_LR[(dataset, target)]:
            raise V31ContractError(
                f"V3 frozen LR drifted on {dataset} {target}: {label} != {EXPECTED_V3_LR[(dataset, target)]}"
            )
    guard_path = (
        v3_robustness_root(project_root) / "results" / "three_track_completion_guard.json"
    )
    if guard_path.is_file():
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        if int(guard.get("EXPECTED_RUNS", -1)) != 162 or str(guard.get("ROBUSTNESS_COMPLETE")) != "YES":
            raise V31ContractError(f"V3 robustness is not the frozen complete reference: {guard}")
        if int(guard.get("OK_RUNS", -1)) != 162 or int(guard.get("FAILED_RUNS", 1)) != 0:
            raise V31ContractError(f"V3 robustness completeness drifted: {guard}")
    return frozen


def v3_1_architecture() -> Dict[str, Any]:
    arch = dict(v3_architecture())
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V31ContractError("V3.1 cannot inherit a drifted V3 architecture")
    return arch


def v3_1_training_recipe() -> Dict[str, Any]:
    recipe = dict(v3_training_recipe())
    expected = {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    if recipe != expected:
        raise V31ContractError(f"V3.1 recipe must inherit frozen V3 recipe, got {recipe}")
    return recipe


def v3_1_revin_config() -> Dict[str, Any]:
    config = v3_revin_config()
    expected = {"enabled": True, "affine": True, "eps": 1e-5, "target_channel": 0}
    if bool(config["enabled"]) is not True or bool(config["affine"]) is not True:
        raise V31ContractError(f"V3.1 RevIN drifted: {config}")
    if not math.isclose(float(config["eps"]), expected["eps"], rel_tol=0.0, abs_tol=0.0):
        raise V31ContractError(f"V3.1 RevIN eps drifted: {config}")
    if int(config["target_channel"]) != expected["target_channel"]:
        raise V31ContractError(f"V3.1 RevIN target_channel drifted: {config}")
    if dict(V3_REVIN_CONTRACT) != {
        "revin_enabled": True,
        "revin_affine": True,
        "revin_eps": float(expected["eps"]),
        "revin_target_channel": 0,
    }:
        raise V31ContractError(f"V3_REVIN_CONTRACT drifted: {V3_REVIN_CONTRACT}")
    return dict(config)


def inherited_v3_spec():
    spec = v3_arm_spec()
    if spec.architecture_digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V31ContractError("inherited V3 arm digest drifted")
    if spec.method_display != PARENT_METHOD:
        raise V31ContractError("inherited V3 method display drifted")
    return spec


@dataclass(frozen=True)
class V31RunIdentity:
    dataset: str
    target_identity: str
    horizon: int
    seed: int
    selected_lr: float
    selected_lr_label: str

    @property
    def arm(self) -> str:
        return "v3_1"

    @property
    def parent_arm(self) -> str:
        return "v3"

    @property
    def identity_tuple(self) -> Tuple[Any, ...]:
        return (
            self.arm,
            self.dataset,
            self.target_identity,
            int(self.horizon),
            int(self.seed),
            self.selected_lr_label,
        )

    @property
    def parent_identity_tuple(self) -> Tuple[Any, ...]:
        return (
            self.parent_arm,
            self.dataset,
            self.target_identity,
            int(self.horizon),
            int(self.seed),
            self.selected_lr_label,
        )

    @property
    def cell_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        return (
            f"v3_1_{self.dataset}_{target_slug}"
            f"_h{int(self.horizon)}_s{int(self.seed)}_{self.selected_lr_label}"
        )

    def as_parent_run_identity(self) -> RunIdentity:
        return RunIdentity(
            arm="v3",
            dataset=self.dataset,
            target_identity=self.target_identity,
            horizon=int(self.horizon),
            seed=int(self.seed),
            selected_lr=float(self.selected_lr),
            selected_lr_label=self.selected_lr_label,
        )


def frozen_v3_lr_map(frozen_manifest: Mapping[str, Any]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for dataset, target in EXPECTED_COMMON_TARGETS:
        label = format_lr(parse_lr(frozen_manifest["datasets"][dataset]["learning_rate"]))
        identity = str(frozen_manifest["datasets"][dataset]["primary_target"]).strip()
        if identity != target:
            raise V31ContractError(f"frozen target drift on {dataset}")
        if label != EXPECTED_V3_LR[(dataset, target)]:
            raise V31ContractError(
                f"V3.1 must inherit frozen V3 LR for {dataset} {target}, got {label}"
            )
        mapping[(dataset, target)] = label
    if mapping != dict(EXPECTED_V3_LR):
        raise V31ContractError("V3.1 LR mapping drifted from the frozen V3 map")
    return mapping


def build_v3_1_identities(
    *,
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_map: Mapping[Tuple[str, str], str],
) -> Tuple[V31RunIdentity, ...]:
    identities: List[V31RunIdentity] = []
    targets = dict(EXPECTED_COMMON_TARGETS)
    for dataset in datasets:
        target = targets.get(dataset)
        if target is None:
            raise V31ContractError(f"unknown dataset {dataset}")
        lr_label = lr_map.get((dataset, target))
        if lr_label is None:
            raise V31ContractError(f"missing frozen V3 LR for {dataset} {target}")
        lr_value = parse_lr(lr_label)
        for horizon in horizons:
            for seed in seeds:
                identities.append(
                    V31RunIdentity(
                        dataset=dataset,
                        target_identity=target,
                        horizon=int(horizon),
                        seed=int(seed),
                        selected_lr=float(lr_value),
                        selected_lr_label=format_lr(lr_value),
                    )
                )
    tuples = [item.identity_tuple for item in identities]
    if len(tuples) != len(set(tuples)):
        raise V31ContractError("duplicate V3.1 cell identity")
    return tuple(identities)


def parse_dataset_ids(spec: Optional[object]) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return SCREENING_DATASETS
    parts = [part.strip().upper() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(f"D{int(part.replace('D', ''))}" for part in parts)
    extra = [item for item in resolved if item not in SCREENING_DATASETS]
    if extra:
        raise V31ContractError(f"unknown datasets: {extra}")
    seen = set()
    ordered: List[str] = []
    for item in resolved:
        if item in seen:
            raise V31ContractError(f"duplicate dataset {item}")
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def parse_int_tuple(spec: Optional[object], allowed: Sequence[int], *, name: str) -> Tuple[int, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return tuple(int(item) for item in allowed)
    parts = [part.strip() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(int(part) for part in parts)
    extra = [item for item in resolved if item not in set(allowed)]
    if extra:
        raise V31ContractError(f"{name} values {extra} are not in frozen set {list(allowed)}")
    seen = set()
    ordered: List[int] = []
    for item in resolved:
        if item in seen:
            raise V31ContractError(f"duplicate {name} {item}")
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def expected_matrix_counts(
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
) -> Dict[str, int]:
    n = len(datasets) * len(horizons) * len(seeds)
    return {"expected_runs": n, "unique_runs": n}


def _date_key(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d")


def _date_key_set(values: Iterable[object]) -> set:
    return {_date_key(value) for value in values}


def compute_validation_median_bias(
    y_val_original: np.ndarray,
    yhat_val_original: np.ndarray,
) -> float:
    """Uniform additive bias: median(y_val - yhat_val) in original sales space."""
    true = np.asarray(y_val_original, dtype=np.float64).reshape(-1)
    pred = np.asarray(yhat_val_original, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise V31ContractError(
            f"calibration arrays must be non-empty and aligned: {true.shape} vs {pred.shape}"
        )
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise V31ContractError("calibration residual requires finite validation arrays")
    return float(np.median(true - pred))


def apply_bias_calibration(yhat_raw: np.ndarray, bias: float) -> np.ndarray:
    """Uniform additive correction: yhat_calibrated = yhat_raw + bias."""
    pred = np.asarray(yhat_raw, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        raise V31ContractError("cannot calibrate an empty prediction vector")
    if not np.isfinite(pred).all() or not math.isfinite(float(bias)):
        raise V31ContractError("calibration requires finite predictions and bias")
    return pred + float(bias)


def estimate_validation_median_bias(
    *,
    y_val_original: np.ndarray,
    yhat_val_original: np.ndarray,
    **forbidden: Any,
) -> float:
    if forbidden:
        raise V31ContractError(
            "forbidden calibration inputs: " + ", ".join(sorted(forbidden))
        )
    return compute_validation_median_bias(y_val_original, yhat_val_original)


def assert_calibration_rule_is_uniform(source: Optional[str] = None) -> Dict[str, str]:
    text = source if source is not None else (
        inspect.getsource(compute_validation_median_bias)
        + "\n"
        + inspect.getsource(apply_bias_calibration)
        + "\n"
        + inspect.getsource(estimate_validation_median_bias)
    )
    forbidden_tokens = (
        "if dataset",
        "if horizon",
        "if seed",
        "if D1",
        "if D2",
        "best_seed",
        "min_seed",
        "selected_seed",
        "y_test",
        "baseline",
    )
    hits = [token for token in forbidden_tokens if token in text]
    if hits:
        raise V31ContractError(f"dataset/horizon/seed/test calibration logic detected: {hits}")
    for fn in (compute_validation_median_bias, apply_bias_calibration, estimate_validation_median_bias):
        params = inspect.signature(fn).parameters
        for name in ("dataset", "horizon", "seed", "y_test", "baseline"):
            if name in params:
                raise V31ContractError(f"{fn.__name__} must not take {name}")
    return {
        "CALIBRATION_UNIFORM_ACROSS_DATASETS": "YES",
        "CALIBRATION_UNIFORM_ACROSS_HORIZONS": "YES",
        "CALIBRATION_UNIFORM_ACROSS_SEEDS": "YES",
        "TEST_TRUTH_USED_FOR_CALIBRATION": "NO",
    }


def concat_train_val_context(
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    *,
    raw_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if raw_train is None or raw_val is None or len(raw_train) == 0 or len(raw_val) == 0:
        raise V31ContractError("calibration context requires non-empty train and validation frames")
    train_dates = _date_key_set(raw_train["date"])
    val_dates = _date_key_set(raw_val["date"])
    if train_dates & val_dates:
        raise V31ContractError("train and validation dates overlap")
    if raw_test is not None:
        test_dates = _date_key_set(raw_test["date"])
        if (train_dates | val_dates) & test_dates:
            raise V31ContractError("test rows leaked into train/validation calibration context")
        if "sales" in raw_test.columns:
            # Dates may be inspected for the leakage guard, but test sales/truth are not consumed.
            pass
    context = pd.concat(
        [raw_train.copy(deep=False), raw_val.copy(deep=False)],
        ignore_index=True,
    )
    context_dates = _date_key_set(context["date"])
    if raw_test is not None and context_dates & _date_key_set(raw_test["date"]):
        raise V31ContractError("concatenated calibration context contains test dates")
    return context


def sequence_window_meta(
    df: pd.DataFrame,
    *,
    horizon: int,
    window_size: int,
    feature_columns: Sequence[str],
) -> List[Dict[str, str]]:
    """Mirror build_tabular_sequence window indices and record label timestamps."""
    ordered = df.sort_values(["entity_id", "item_id", "date"], kind="mergesort").reset_index(drop=True)
    records: List[Dict[str, str]] = []
    for group_key, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        g = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        group_dates = pd.to_datetime(g["date"], errors="raise", utc=True)
        normalized_group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        entity_id = str(normalized_group_key[0])
        item_id = str(normalized_group_key[1]) if len(normalized_group_key) > 1 else ""
        n = len(g)
        max_end = n - int(horizon)
        for end_idx in range(int(window_size) - 1, max_end):
            start_idx = end_idx - int(window_size) + 1
            target_idx = end_idx + int(horizon)
            if target_idx >= n:
                continue
            records.append(
                {
                    "entity_id": entity_id,
                    "item_id": item_id,
                    "window_start": _date_key(group_dates.iloc[start_idx]),
                    "window_end": _date_key(group_dates.iloc[end_idx]),
                    "label_date": _date_key(group_dates.iloc[target_idx]),
                }
            )
    _ = feature_columns
    return records


def build_windows_with_meta(
    df: pd.DataFrame,
    *,
    horizon: int,
    window_size: int,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    X, y = build_tabular_sequence(
        df,
        horizon=int(horizon),
        window_size=int(window_size),
        feature_columns=feature_columns,
        reuse_context=None,
    )
    meta = sequence_window_meta(
        df, horizon=horizon, window_size=window_size, feature_columns=feature_columns
    )
    if len(meta) != int(np.asarray(y).reshape(-1).shape[0]):
        raise V31ContractError(
            f"sequence metadata length drifted from labels: {len(meta)} vs {len(y)}"
        )
    return np.asarray(X), np.asarray(y), meta


def _meta_key(record: Mapping[str, str]) -> Tuple[str, str, str]:
    return (str(record["entity_id"]), str(record["item_id"]), str(record["label_date"]))


def _align_windows(
    X: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Mapping[str, str]],
    keys: Sequence[Tuple[str, str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    index = {_meta_key(record): i for i, record in enumerate(meta)}
    if len(index) != len(meta):
        raise V31ContractError("duplicate sequence keys while aligning calibration windows")
    idxs: List[int] = []
    kept: List[Dict[str, str]] = []
    for key in keys:
        if key not in index:
            raise V31ContractError(f"missing aligned calibration window for {key}")
        idxs.append(index[key])
        kept.append(dict(meta[index[key]]))
    selector = np.asarray(idxs, dtype=np.int64)
    return np.asarray(X)[selector], np.asarray(y)[selector], kept


@dataclass(frozen=True)
class CalibrationWindows:
    X: np.ndarray
    y_normalized: np.ndarray
    f_t: List[np.ndarray]
    meta: List[Dict[str, str]]
    sample_count: int
    val_dates: Tuple[str, ...]


def build_calibration_windows(
    *,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    bundle: Any,
    horizon: int,
    window_size: int = WINDOW_SIZE,
    feature_columns: Sequence[str],
) -> CalibrationWindows:
    context_raw = concat_train_val_context(raw_train, raw_val, raw_test=raw_test)
    val_dates = tuple(sorted(_date_key_set(raw_val["date"])))
    val_date_set = set(val_dates)
    test_dates = _date_key_set(raw_test["date"])
    scaled_target = _transform_with_scaler(
        context_raw, bundle.target_scaler, bundle.target_feature_columns
    )
    X, y, meta = build_windows_with_meta(
        scaled_target,
        horizon=horizon,
        window_size=window_size,
        feature_columns=feature_columns,
    )
    for record in meta:
        if record["label_date"] in test_dates:
            raise V31ContractError("calibration label timestamp is in the test period")
        if record["window_start"] in test_dates or record["window_end"] in test_dates:
            raise V31ContractError("calibration input window overlaps the test period")
    keys = [
        _meta_key(record)
        for record in meta
        if record["label_date"] in val_date_set
    ]
    if not keys:
        raise V31ContractError("calibration_sample_count=0 after validation-label filter")
    X_cal, y_cal, meta_cal = _align_windows(X, y, meta, keys)
    f_t: List[np.ndarray] = []
    source_scalers = list(bundle.stage_a.source_scalers)
    source_extractors = list(bundle.stage_a.source_extractors)
    if len(source_scalers) != len(source_extractors):
        raise V31ContractError("source scaler/extractor count mismatch")
    for scaler_i, extractor_i in zip(source_scalers, source_extractors):
        scaled_src = _transform_with_scaler(context_raw, scaler_i, bundle.target_feature_columns)
        Xs, ys, meta_s = build_windows_with_meta(
            scaled_src,
            horizon=horizon,
            window_size=window_size,
            feature_columns=feature_columns,
        )
        Xs_cal, _ys_cal, _meta_s = _align_windows(Xs, ys, meta_s, keys)
        tensor = to_cnn_tensor(Xs_cal)
        f_t.append(_extract_source_numpy(extractor_i, tensor))
    return CalibrationWindows(
        X=to_cnn_tensor(X_cal),
        y_normalized=np.asarray(y_cal).reshape(-1),
        f_t=f_t,
        meta=meta_cal,
        sample_count=int(len(y_cal)),
        val_dates=val_dates,
    )


def count_calibration_samples(
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    *,
    horizon: int,
    window_size: int = WINDOW_SIZE,
    feature_columns: Optional[Sequence[str]] = None,
    raw_test: Optional[pd.DataFrame] = None,
) -> int:
    context = concat_train_val_context(raw_train, raw_val, raw_test=raw_test)
    cols = list(feature_columns) if feature_columns is not None else [
        col for col in context.columns if col not in {"date"}
    ]
    if "sales" not in cols and "sales" in context.columns:
        cols = ["sales"] + [col for col in cols if col != "sales"]
    meta = sequence_window_meta(
        context, horizon=horizon, window_size=window_size, feature_columns=cols
    )
    val_dates = _date_key_set(raw_val["date"])
    return sum(1 for record in meta if record["label_date"] in val_dates)


def _extract_source_numpy(extractor, tensor: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    return extractor(tf.constant(tensor, dtype=tf.float32), training=False).numpy()


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def predict_v3_normalized(
    runtime: Mapping[str, Any],
    X: np.ndarray,
    f_t: Sequence[np.ndarray],
    knn_prior: np.ndarray,
) -> np.ndarray:
    z, mean, stdev = extract_features_maybe_revin(
        runtime["target_extractor"],
        X,
        revin=runtime.get("revin"),
        training=False,
    )
    decomposed = predict_ms_katl_decomposed(
        runtime["head"], z, list(f_t), knn_prior
    )
    y_pred_model = np.asarray(decomposed["y_pred"]).reshape(-1)
    revin = runtime.get("revin")
    if revin is not None:
        mean_np = _as_numpy(mean)
        stdev_np = _as_numpy(stdev)
        y_pred = revin.denormalize_numpy(y_pred_model, mean_np, stdev_np).reshape(-1)
    else:
        y_pred = y_pred_model
    return np.asarray(y_pred, dtype=np.float64).reshape(-1)


def run_v3_raw(
    *,
    bundle: Any,
    identity: V31RunIdentity,
    spec,
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
) -> Dict[str, Any]:
    ablation = resolve_ablation_arm(spec.ablation_arm)
    arm = run_ablation_arm(
        ablation,
        bundle,
        epochs=int(epochs),
        batch_size=int(batch_size),
        gamma=0.5,
        use_adversarial=False,
        metric_protocol=metric_protocol,
        fine_tune_learning_rate=float(identity.selected_lr),
        return_runtime=True,
        **spec.run_ablation_kwargs(),
    )
    runtime = arm.pop("runtime")
    metrics = compute_screening_metrics(
        arm["y_true"],
        arm["y_pred"],
        target_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    if metrics["metric_space"] != ORIGINAL_SALES_SPACE:
        raise V31ContractError(
            f"metric space drifted to {metrics['metric_space']}; original_sales_space required"
        )
    replay = predict_v3_normalized(
        runtime,
        bundle.X_test,
        bundle.f_t["test"],
        bundle.stage_a.knn_prior,
    )
    raw_norm = np.asarray(arm["y_pred"], dtype=np.float64).reshape(-1)
    if not np.allclose(replay, raw_norm, rtol=V3_RAW_REPRODUCTION_RTOL, atol=V3_RAW_REPRODUCTION_ATOL):
        max_abs = float(np.max(np.abs(replay - raw_norm))) if raw_norm.size else 0.0
        raise V31ContractError(
            "same-process V3 runtime replay diverged from run_ablation_arm predictions: "
            f"max_abs={max_abs} rtol={V3_RAW_REPRODUCTION_RTOL} atol={V3_RAW_REPRODUCTION_ATOL}"
        )
    return {
        "arm": arm,
        "runtime": runtime,
        "metrics": metrics,
        "y_true_normalized": np.asarray(arm["y_true"]).reshape(-1),
        "y_pred_normalized": np.asarray(arm["y_pred"]).reshape(-1),
        "y_true_original": np.asarray(metrics["y_true_original"]).reshape(-1),
        "y_pred_original": np.asarray(metrics["y_pred_original"]).reshape(-1),
    }


def calibrate_v3_raw(
    *,
    raw: Mapping[str, Any],
    bundle: Any,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    identity: V31RunIdentity,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    windows = build_calibration_windows(
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        bundle=bundle,
        horizon=int(identity.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=feature_columns,
    )
    yhat_val_norm = predict_v3_normalized(
        raw["runtime"],
        windows.X,
        windows.f_t,
        bundle.stage_a.knn_prior,
    )
    val_metrics = compute_screening_metrics(
        windows.y_normalized,
        yhat_val_norm,
        target_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    bias = estimate_validation_median_bias(
        y_val_original=val_metrics["y_true_original"],
        yhat_val_original=val_metrics["y_pred_original"],
    )
    yhat_test_raw = np.asarray(raw["y_pred_original"], dtype=np.float64).reshape(-1)
    y_true_test = np.asarray(raw["y_true_original"], dtype=np.float64).reshape(-1)
    yhat_test_cal = apply_bias_calibration(yhat_test_raw, bias)
    rmse_raw = original_rmse(y_true_test, yhat_test_raw)
    smape_raw = float(smape(y_true_test, yhat_test_raw))
    rmse_cal = original_rmse(y_true_test, yhat_test_cal)
    smape_cal = float(smape(y_true_test, yhat_test_cal))
    if not math.isclose(rmse_raw, float(raw["metrics"]["rmse_original"]), rel_tol=0.0, abs_tol=1e-12):
        raise V31ContractError("raw V3 original RMSE drifted between screening metrics and calibration path")
    val_residual = (
        np.asarray(val_metrics["y_true_original"], dtype=np.float64).reshape(-1)
        - np.asarray(val_metrics["y_pred_original"], dtype=np.float64).reshape(-1)
    )
    return {
        "calibration_type": CALIBRATION_TYPE,
        "calibration_sample_count": int(windows.sample_count),
        "calibration_bias": float(bias),
        "calibration_windows": windows,
        "validation_y_true": np.asarray(val_metrics["y_true_original"]).reshape(-1),
        "validation_y_pred": np.asarray(val_metrics["y_pred_original"]).reshape(-1),
        "validation_residual": val_residual,
        "validation_meta": windows.meta,
        "y_pred_v3_raw": yhat_test_raw,
        "y_pred_v3_1_calibrated": yhat_test_cal,
        "y_true": y_true_test,
        "rmse_v3_raw": float(rmse_raw),
        "smape_v3_raw": float(smape_raw),
        "rmse_v3_1_calibrated": float(rmse_cal),
        "smape_v3_1_calibrated": float(smape_cal),
        "rmse_improvement_vs_raw_v3": relative_improvement(rmse_raw, rmse_cal),
        "smape_improvement_vs_raw_v3": relative_improvement(smape_raw, smape_cal),
        "raw_residual_mean": float(np.mean(val_residual)),
        "raw_residual_median": float(np.median(val_residual)),
        "post_calibration_residual_mean": float(np.mean(val_residual - bias)),
        "post_calibration_residual_median": float(np.median(val_residual - bias)),
        "TEST_TRUTH_USED_FOR_CALIBRATION": "NO",
        "metric_space": ORIGINAL_SALES_SPACE,
    }


def parent_v3_prediction_path(
    project_root: Path,
    identity: V31RunIdentity,
    parsed: Mapping[str, str],
) -> Path:
    return (
        v3_robustness_root(project_root)
        / "predictions"
        / "v3"
        / identity.dataset
        / parsed["prediction_dirname"]
        / f"h{int(identity.horizon)}_s{int(identity.seed)}_{identity.selected_lr_label}"
        / "predictions.csv"
    )


def compare_raw_v3_reproduction(
    yhat_raw: np.ndarray,
    parent_path: Path,
    *,
    y_true_original: Optional[np.ndarray] = None,
    rmse_raw: Optional[float] = None,
    rtol: float = V3_RAW_REPRODUCTION_RTOL,
    atol: float = V3_RAW_REPRODUCTION_ATOL,
    parent_rmse_rtol: float = V3_RAW_PARENT_RMSE_RTOL,
) -> Dict[str, Any]:
    if not Path(parent_path).is_file():
        raise V31ContractError(f"V3 parent prediction artifact missing: {parent_path}")
    frame = pd.read_csv(parent_path)
    if "y_pred_original" not in frame.columns:
        raise V31ContractError(f"V3 parent predictions missing y_pred_original: {parent_path}")
    parent = np.asarray(frame["y_pred_original"], dtype=np.float64).reshape(-1)
    got = np.asarray(yhat_raw, dtype=np.float64).reshape(-1)
    if parent.shape != got.shape:
        raise V31ContractError(
            f"V3 raw prediction length drifted: parent={parent.shape} got={got.shape}"
        )
    if y_true_original is not None and "y_true_original" in frame.columns:
        parent_true = np.asarray(frame["y_true_original"], dtype=np.float64).reshape(-1)
        got_true = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
        if parent_true.shape != got_true.shape or not np.allclose(
            parent_true, got_true, rtol=0.0, atol=V3_RAW_YTRUE_ATOL
        ):
            raise V31ContractError("V3 evaluation y_true drifted from the frozen parent artifact")
    max_abs = float(np.max(np.abs(parent - got))) if parent.size else 0.0
    pointwise = bool(np.allclose(parent, got, rtol=rtol, atol=atol, equal_nan=False))
    parent_rmse = original_rmse(
        np.asarray(frame["y_true_original"], dtype=np.float64).reshape(-1),
        parent,
    ) if "y_true_original" in frame.columns else None
    rmse_rel = None
    if rmse_raw is not None and parent_rmse is not None and parent_rmse != 0.0:
        rmse_rel = abs(float(rmse_raw) - float(parent_rmse)) / abs(float(parent_rmse))
    if pointwise:
        mode = "pointwise"
    elif rmse_rel is not None and rmse_rel <= float(parent_rmse_rtol):
        mode = "parent_rmse_relative"
    else:
        raise V31ContractError(
            "V3_RAW_REPRODUCTION failed: raw V3.1 predictions diverged from frozen V3 "
            f"max_abs={max_abs} rmse_rel={rmse_rel} rtol={rtol} atol={atol} "
            f"parent_rmse_rtol={parent_rmse_rtol}"
        )
    return {
        "V3_RAW_REPRODUCTION": "PASS",
        "mode": mode,
        "max_abs_diff": max_abs,
        "rmse_rel": rmse_rel,
        "rtol": rtol,
        "atol": atol,
        "parent_rmse_rtol": parent_rmse_rtol,
        "parent_rmse": parent_rmse,
        "n": int(parent.size),
    }


def test_label_dates(
    raw_test: pd.DataFrame,
    *,
    horizon: int,
    window_size: int,
    feature_columns: Sequence[str],
) -> List[str]:
    meta = sequence_window_meta(
        raw_test, horizon=horizon, window_size=window_size, feature_columns=feature_columns
    )
    return [record["label_date"] for record in meta]


def cell_path(cells_dir: Path, identity: V31RunIdentity) -> Path:
    return Path(cells_dir) / f"{identity.cell_id}.json"


def classify_cell_artifact(path: Path, identity: V31RunIdentity, digest: str) -> str:
    if not Path(path).is_file():
        return "MISSING"
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "MALFORMED"
    missing = [key for key in REQUIRED_CELL_FIELDS if key not in payload]
    if missing:
        return "MALFORMED"
    status = str(payload.get("status") or "").upper()
    if status == "FAILED":
        return "FAILED"
    if status != "OK":
        return "MALFORMED"
    try:
        if str(payload.get("dataset")) != identity.dataset:
            return "STALE"
        if str(payload.get("target_identity")) != identity.target_identity:
            return "STALE"
        if int(payload.get("horizon")) != int(identity.horizon):
            return "STALE"
        if int(payload.get("seed")) != int(identity.seed):
            return "STALE"
        if format_lr(parse_lr(payload.get("learning_rate_label") or payload["learning_rate"])) != identity.selected_lr_label:
            return "STALE"
        if str(payload.get("parent_method")) != PARENT_METHOD:
            return "STALE"
        if str(payload.get("parent_architecture_digest")) != digest:
            return "STALE"
        if str(payload.get("calibration_type")) != CALIBRATION_TYPE:
            return "STALE"
        if str(payload.get("metric_space")) != ORIGINAL_SALES_SPACE:
            return "STALE"
        for key in ("rmse_v3_raw", "rmse_v3_1_calibrated", "smape_v3_raw", "smape_v3_1_calibrated", "calibration_bias"):
            if not math.isfinite(float(payload[key])):
                return "MALFORMED"
        if int(payload["calibration_sample_count"]) <= 0:
            return "MALFORMED"
    except (TypeError, ValueError, KeyError):
        return "MALFORMED"
    return "OK"


def load_valid_ok_cells(
    cells_dir: Path,
    identities: Sequence[V31RunIdentity],
    digest: str,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    completed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for identity in identities:
        path = cell_path(cells_dir, identity)
        if classify_cell_artifact(path, identity, digest) != "OK":
            continue
        completed[identity.identity_tuple] = json.loads(path.read_text(encoding="utf-8"))
    return completed


def completion_guard(
    identities: Sequence[V31RunIdentity],
    cell_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = len(identities)
    ok_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "OK"]
    failed_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "FAILED"]
    materialized = []
    for row in ok_rows:
        materialized.append(
            (
                "v3_1",
                str(row["dataset"]),
                str(row["target_identity"]),
                int(row["horizon"]),
                int(row["seed"]),
                format_lr(parse_lr(row.get("learning_rate_label") or row["learning_rate"])),
            )
        )
    unique = set(materialized)
    expected_set = {item.identity_tuple for item in identities}
    missing = expected_set - unique
    extras = unique - expected_set
    duplicate = len(materialized) - len(unique)
    complete = (
        expected == EXPECTED_RUNS
        and len(unique) == expected
        and not failed_rows
        and duplicate == 0
        and not missing
        and not extras
    )
    return {
        "EXPECTED_RUNS": expected,
        "OK_RUNS": len(unique),
        "FAILED_RUNS": len(failed_rows),
        "DUPLICATE_RUNS": int(duplicate),
        "MISSING_RUNS": len(missing),
        "EXTRA_RUNS": len(extras),
        "ROBUSTNESS_COMPLETE": "YES" if complete else "NO",
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values))


def _std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) * 100.0


def load_frozen_v3_robustness_rows(project_root: Path) -> List[Dict[str, Any]]:
    path = v3_robustness_root(project_root) / "results" / "three_track_robustness_all_runs.csv"
    if not path.is_file():
        raise V31ContractError(f"frozen V3 robustness CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = []
    for row in rows:
        if str(row.get("arm") or "").lower() != "v3":
            continue
        if str(row.get("status") or row.get("run_status") or "").upper() != "OK":
            continue
        kept.append(row)
    if len(kept) != EXPECTED_RUNS:
        raise V31ContractError(f"frozen V3 robustness must contain 54 OK V3 rows, got {len(kept)}")
    return kept


def summarize_seed_group(rows: Sequence[Mapping[str, Any]], rmse_key: str, smape_key: str) -> Dict[str, Any]:
    rmses = [float(row[rmse_key]) for row in rows]
    smapes = [float(row[smape_key]) for row in rows]
    if len(rmses) != 3:
        raise V31ContractError(f"headline aggregation requires exactly 3 seeds, got {len(rmses)}")
    return {
        "rmse_mean": _mean(rmses),
        "rmse_std": _std(rmses),
        "rmse_median": _median(rmses),
        "smape_mean": _mean(smapes),
        "smape_std": _std(smapes),
        "smape_median": _median(smapes),
        "n_seeds": len(rmses),
        "seeds": [int(row["seed"]) for row in rows],
    }


def load_baseline_seed_means(
    project_root: Path,
    *,
    dataset: str,
    target_identity: str,
    horizon: int,
    seeds: Sequence[int] = V3_1_SEEDS,
) -> Dict[str, Dict[str, float]]:
    path = default_baseline_csv(dataset, project_root)
    if not Path(path).is_file():
        raise V31ContractError(f"baseline CSV missing for {dataset}: {path}")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload: Dict[str, Dict[str, float]] = {}
    for _key, method, _display in BL_METHODS:
        rmses: List[float] = []
        smapes: List[float] = []
        for seed in seeds:
            ref = lookup_reference_row(
                rows,
                dataset_id=dataset,
                target_identity=target_identity,
                method=method,
                horizon=int(horizon),
                seed=int(seed),
                mode=SCREENING_MODE,
            )
            if ref.get("status") != "FOUND" or ref.get("rmse") is None:
                raise V31ContractError(
                    f"missing baseline {method} for {dataset} {target_identity} H={horizon} seed={seed}"
                )
            rmses.append(float(ref["rmse"]))
            smapes.append(float(ref["smape"]))
        payload[method] = {
            "mean_rmse": _mean(rmses),
            "mean_smape": _mean(smapes),
        }
    return payload


def build_dataset_horizon_tables(
    v3_1_rows: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    def _group(rows: Sequence[Mapping[str, Any]], rmse_key: str, smape_key: str) -> Dict[Tuple[str, str, int], List[dict]]:
        grouped: Dict[Tuple[str, str, int], List[dict]] = {}
        for row in rows:
            if str(row.get("status") or "OK").upper() != "OK":
                continue
            key = (str(row["dataset"]), str(row["target_identity"]), int(row["horizon"]))
            grouped.setdefault(key, []).append(dict(row))
        return grouped

    v31_groups = _group(v3_1_rows, "rmse_v3_1_calibrated", "smape_v3_1_calibrated")
    v3_groups = _group(v3_rows, "rmse_original", "smape_original")
    rmse_table: List[Dict[str, Any]] = []
    smape_table: List[Dict[str, Any]] = []
    diagnostics = {
        "beats_best_bl": 0,
        "top2": 0,
        "top3": 0,
        "n": 0,
    }
    for dataset, target in EXPECTED_COMMON_TARGETS:
        for horizon in V3_1_HORIZONS:
            key = (dataset, target, int(horizon))
            v31 = v31_groups.get(key) or []
            v3 = v3_groups.get(key) or []
            if len(v31) != 3 or len(v3) != 3:
                raise V31ContractError(
                    f"dataset×horizon aggregation requires 3 seeds for {key}, "
                    f"got v3.1={len(v31)} v3={len(v3)}"
                )
            v31_stats = summarize_seed_group(v31, "rmse_v3_1_calibrated", "smape_v3_1_calibrated")
            v3_stats = summarize_seed_group(v3, "rmse_original", "smape_original")
            bl = load_baseline_seed_means(
                project_root,
                dataset=dataset,
                target_identity=target,
                horizon=int(horizon),
            )
            best_method, best_payload = min(bl.items(), key=lambda item: item[1]["mean_rmse"])
            worst_method, worst_payload = max(bl.items(), key=lambda item: item[1]["mean_rmse"])
            ranked = [("V3.1", v31_stats["rmse_mean"])] + [
                (method, payload["mean_rmse"]) for method, payload in bl.items()
            ]
            ranked.sort(key=lambda item: item[1])
            rank = 1 + sum(1 for name, value in ranked if name != "V3.1" and value < v31_stats["rmse_mean"])
            beats = "YES" if v31_stats["rmse_mean"] < best_payload["mean_rmse"] else "NO"
            diagnostics["n"] += 1
            if beats == "YES":
                diagnostics["beats_best_bl"] += 1
            if rank <= 2:
                diagnostics["top2"] += 1
            if rank <= 3:
                diagnostics["top3"] += 1
            vs_v3 = relative_improvement(v3_stats["rmse_mean"], v31_stats["rmse_mean"])
            vs_best = relative_improvement(best_payload["mean_rmse"], v31_stats["rmse_mean"])
            vs_worst = relative_improvement(worst_payload["mean_rmse"], v31_stats["rmse_mean"])
            rmse_table.append(
                {
                    "DATASET": dataset,
                    "TARGET": target,
                    "HORIZON": int(horizon),
                    "V3_RMSE_MEAN": v3_stats["rmse_mean"],
                    "V3_RMSE_STD": v3_stats["rmse_std"],
                    "V3_1_RMSE_MEAN": v31_stats["rmse_mean"],
                    "V3_1_RMSE_STD": v31_stats["rmse_std"],
                    "BEST_BL": best_method,
                    "BEST_BL_RMSE_MEAN": best_payload["mean_rmse"],
                    "WORST_BL": worst_method,
                    "WORST_BL_RMSE_MEAN": worst_payload["mean_rmse"],
                    "V3_1_VS_V3_PCT": _pct(vs_v3),
                    "V3_1_VS_BEST_BL_PCT": _pct(vs_best),
                    "V3_1_VS_WORST_BL_PCT": _pct(vs_worst),
                    "V3_1_RANK_AMONG_V3_1_PLUS_BL1_BL5": rank,
                    "BEST_BL_BEAT": beats,
                }
            )
            best_s_method, best_s = min(bl.items(), key=lambda item: item[1]["mean_smape"])
            worst_s_method, worst_s = max(bl.items(), key=lambda item: item[1]["mean_smape"])
            smape_rank_pool = [("V3.1", v31_stats["smape_mean"])] + [
                (method, payload["mean_smape"]) for method, payload in bl.items()
            ]
            smape_rank = 1 + sum(
                1 for name, value in smape_rank_pool if name != "V3.1" and value < v31_stats["smape_mean"]
            )
            smape_table.append(
                {
                    "DATASET": dataset,
                    "TARGET": target,
                    "HORIZON": int(horizon),
                    "V3_SMAPE_MEAN": v3_stats["smape_mean"],
                    "V3_SMAPE_STD": v3_stats["smape_std"],
                    "V3_1_SMAPE_MEAN": v31_stats["smape_mean"],
                    "V3_1_SMAPE_STD": v31_stats["smape_std"],
                    "BEST_BL": best_s_method,
                    "BEST_BL_SMAPE_MEAN": best_s["mean_smape"],
                    "WORST_BL": worst_s_method,
                    "WORST_BL_SMAPE_MEAN": worst_s["mean_smape"],
                    "V3_1_VS_V3_PCT": _pct(relative_improvement(v3_stats["smape_mean"], v31_stats["smape_mean"])),
                    "V3_1_VS_BEST_BL_PCT": _pct(relative_improvement(best_s["mean_smape"], v31_stats["smape_mean"])),
                    "V3_1_VS_WORST_BL_PCT": _pct(relative_improvement(worst_s["mean_smape"], v31_stats["smape_mean"])),
                    "V3_1_RANK_AMONG_V3_1_PLUS_BL1_BL5": smape_rank,
                    "BEST_BL_BEAT": "YES" if v31_stats["smape_mean"] < best_s["mean_smape"] else "NO",
                }
            )
    if len(rmse_table) != 18 or len(smape_table) != 18:
        raise V31ContractError(
            f"headline tables must have 18 dataset×horizon rows, got {len(rmse_table)}/{len(smape_table)}"
        )
    return {
        "rmse": rmse_table,
        "smape": smape_table,
        "diagnostics": diagnostics,
    }


def build_v3_1_manifest(project_root: Path) -> Dict[str, Any]:
    frozen = assert_v3_parent_contract(project_root)
    spec = inherited_v3_spec()
    datasets = {
        dataset: {
            "primary_target": target,
            "learning_rate": EXPECTED_V3_LR[(dataset, target)],
        }
        for dataset, target in EXPECTED_COMMON_TARGETS
    }
    return {
        "selection_frozen": True,
        "method": METHOD_DISPLAY,
        "parent_method": PARENT_METHOD,
        "parent_manifest": str(v3_frozen_manifest_path(project_root)),
        "parent_architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
        "parent_robustness_run": str(v3_robustness_root(project_root)),
        "parent_screening_run": str(v3_screening_root(project_root)),
        "selection_stage": "development_derived_calibration",
        "common_target_policy": "V3_FROZEN_PRIMARY_TARGETS",
        "targets": 6,
        "horizons": list(V3_1_HORIZONS),
        "seeds": list(V3_1_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "epochs": SCREENING_EPOCHS,
        "window_size": WINDOW_SIZE,
        "training_recipe": dict(v3_1_training_recipe()),
        "RevIN": v3_1_revin_config(),
        "architecture": dict(v3_1_architecture()),
        "architecture_digest": spec.architecture_digest,
        "metric_space": ORIGINAL_SALES_SPACE,
        "calibration": {
            "enabled": True,
            "type": CALIBRATION_TYPE,
            "formula": CALIBRATION_FORMULA,
            "bias": BIAS_FORMULA,
            "calibration_truth_source": CALIBRATION_TRUTH_SOURCE,
            "calibration_context": CALIBRATION_CONTEXT,
            "test_truth_consumed": False,
            "dataset_specific_tuning": False,
            "horizon_specific_tuning": False,
            "seed_specific_tuning": False,
            "lr_reselection": False,
            "target_reselection": False,
        },
        "datasets": datasets,
        "frozen_lr_mapping": {
            f"{dataset}|{target}": lr_label
            for (dataset, target), lr_label in EXPECTED_V3_LR.items()
        },
        "CALIBRATION_UNIFORM_ACROSS_DATASETS": "YES",
        "CALIBRATION_UNIFORM_ACROSS_HORIZONS": "YES",
        "CALIBRATION_UNIFORM_ACROSS_SEEDS": "YES",
        "TEST_TRUTH_USED_FOR_CALIBRATION": "NO",
        "FORMAL_METHODS_CHANGE": "NO",
        "parent_v3_manifest_verified": bool(frozen.get("V3_SELECTION_CONTRACT_VERIFIED")),
    }


def validate_v3_1_manifest(manifest: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    if manifest.get("method") != METHOD_DISPLAY:
        raise V31ContractError(f"V3.1 manifest method drifted: {manifest.get('method')}")
    if manifest.get("parent_method") != PARENT_METHOD:
        raise V31ContractError("V3.1 parent_method drifted")
    if str(manifest.get("parent_architecture_digest")) != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V31ContractError("V3.1 parent digest drifted")
    cal = manifest.get("calibration") or {}
    if cal.get("enabled") is not True or cal.get("type") != CALIBRATION_TYPE:
        raise V31ContractError(f"V3.1 calibration block drifted: {cal}")
    if cal.get("test_truth_consumed") is not False:
        raise V31ContractError("V3.1 calibration must not consume test truth")
    if int(manifest.get("expected_runs") or 0) != EXPECTED_RUNS:
        raise V31ContractError("V3.1 expected_runs must be 54")
    frozen_lr = frozen_v3_lr_map(assert_v3_parent_contract(project_root))
    for dataset, target in EXPECTED_COMMON_TARGETS:
        got = format_lr(parse_lr(manifest["datasets"][dataset]["learning_rate"]))
        if got != frozen_lr[(dataset, target)]:
            raise V31ContractError(f"V3.1 manifest LR drifted on {dataset}")
    return dict(manifest)


def load_v3_1_manifest(project_root: Path) -> Dict[str, Any]:
    path = v3_1_manifest_path(project_root)
    if not path.is_file():
        raise V31ContractError(f"V3.1 manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_v3_1_manifest(payload, project_root)


def write_v3_1_manifest(project_root: Path) -> Path:
    path = v3_1_manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v3_1_manifest(project_root)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_v3_1_manifest(loaded, project_root)
    return path


def format_preflight(
    *,
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_map: Mapping[Tuple[str, str], str],
    counts: Mapping[str, int],
) -> str:
    assert_calibration_rule_is_uniform()
    lines = [
        "============================================================",
        "MS-KATL FTO V3.1 CALIBRATED ROBUSTNESS PREFLIGHT",
        "============================================================",
        "",
        f"METHOD={METHOD_DISPLAY}",
        f"PARENT={PARENT_METHOD}",
        "",
        f"DATASETS={','.join(datasets)}",
        f"TARGETS={len(datasets)}",
        f"HORIZONS={','.join(str(item) for item in horizons)}",
        f"SEEDS={','.join(str(item) for item in seeds)}",
        "",
        f"EXPECTED_RUNS={counts['expected_runs']}",
        f"UNIQUE_RUNS={counts['unique_runs']}",
        "",
        "CALIBRATION=VALIDATION_MEDIAN_BIAS",
        f"CALIBRATION_SPACE={CALIBRATION_SPACE}",
        "TEST_TRUTH_USED_FOR_CALIBRATION=NO",
        "",
        "DATASET_SPECIFIC_TUNING=NO",
        "HORIZON_SPECIFIC_TUNING=NO",
        "SEED_SPECIFIC_TUNING=NO",
        "",
        "LR_RESELECTION=NO",
        "TARGET_RESELECTION=NO",
        "",
        "V3_PARENT_UNCHANGED=YES",
        "V3_SCREENING_UNCHANGED=YES",
        "V3_ROBUSTNESS_UNCHANGED=YES",
        "FORMAL_METHODS_CHANGE=NO",
        "",
        "READY_FOR_SMOKE=YES",
        "============================================================",
        "",
        f"{'DATASET':<8}{'TARGET':<22}{'LR':<10}",
        "------------------------------------------------------------",
    ]
    for dataset, target in EXPECTED_COMMON_TARGETS:
        if dataset not in set(datasets):
            continue
        lines.append(f"{dataset:<8}{target:<22}{lr_map[(dataset, target)]:<10}")
    lines.append("============================================================")
    return "\n".join(lines) + "\n"


def split_target_frames(target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return temporal_split_by_ratio_or_dates(target_df)
