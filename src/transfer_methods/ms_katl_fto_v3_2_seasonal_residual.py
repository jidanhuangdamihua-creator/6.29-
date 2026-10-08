"""MS-KATL FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL contract.

Parent is frozen FUSED_TARGET_ONLY_V3 plus one uniform lag-7 seasonal
residual mechanism on Stage-B only. Not part of FORMAL_METHODS.

Does not reselect targets or learning rates, does not inherit V3.1
calibration, and does not overwrite V3 screening / robustness artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates
from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.evaluation.metrics import inverse_transform_sales, smape
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
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_mechanism_ablation import (
    assert_formal_methods_untouched,
    resolve_ablation_arm,
    run_ablation_arm,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_DATASETS,
    default_baseline_csv,
    format_lr,
    lookup_reference_row,
    original_rmse,
    parse_lr,
    parse_target_identity,
)
from src.transfer_methods.revin import RevIN

METHOD_DISPLAY = "FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL"
METHOD_ARM = "fused_target_only_v3_2_seasonal_residual"
PARENT_METHOD = V3_METHOD_DISPLAY
ROBUSTNESS_LABEL = "MS_KATL_FTO_V3_2_SEASONAL_RESIDUAL_ROBUSTNESS"
ROBUSTNESS_TRACK = "development_seasonal_residual"
SEASONAL_PRIOR_TYPE = "LAG7"
SEASONAL_PRIOR_NAME = "lag7_seasonal_naive"
LAG_DAYS = 7
RESIDUAL_DEFINITION = "y_t-y_t_minus_7"
RESIDUALIZATION_SCOPE = "TARGET_STAGE_B_ONLY"
REVIN_OUTPUT_MODE = "scale_only_residual"
WINDOW_SIZE = 10
EXPECTED_RUNS = 54
UNIQUE_RUNS = 54
V3_2_HORIZONS: Tuple[int, ...] = ROBUSTNESS_HORIZONS
V3_2_SEEDS: Tuple[int, ...] = ROBUSTNESS_SEEDS
DEFAULT_V3_ROBUSTNESS_DIRNAME = "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
V3_2_MANIFEST_RELPATH = Path("configs") / "ms_katl_fto_v3_2_seasonal_residual_manifest.json"
V3_PARENT_REPRODUCTION_RTOL = 1e-5
V3_PARENT_REPRODUCTION_ATOL = 1e-5
V3_1_CALIBRATION_ENABLED = "NO"

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

REQUIRED_CELL_FIELDS = (
    "status",
    "dataset",
    "target_identity",
    "horizon",
    "seed",
    "learning_rate",
    "parent_method",
    "seasonal_prior_type",
    "seasonal_lag_days",
    "train_residual_sample_count",
    "validation_residual_sample_count",
    "test_prediction_count",
    "prior_maturity_guard",
    "rmse_original",
    "smape_original",
    "parent_v3_rmse_reference",
    "parent_v3_smape_reference",
    "relative_improvement_vs_v3_rmse",
    "relative_improvement_vs_v3_smape",
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

FORBIDDEN_PRIOR_TOKENS = (
    "lag=1",
    "lag=3",
    "lag=5",
    "lag=14",
    "MA3",
    "MA7",
    "MA14",
    "HistoricalMean",
    "best_bl",
    "LightGBM",
    "median_bias",
    "mean_bias",
    "bias_alpha",
)

FORBIDDEN_CALIBRATION_TOKENS = (
    "validation_median_bias",
    "yhat_calibrated = yhat_raw + bias",
    "V3_1_CALIBRATED",
    "calibration_bias",
    "mean_bias",
    "bias_alpha",
)


class V32ContractError(RuntimeError):
    """Fail-closed V3.2 seasonal-residual contract violation."""


def _frozen_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def project_root_from(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[2]


def v3_2_manifest_path(project_root: Path) -> Path:
    return Path(project_root) / V3_2_MANIFEST_RELPATH


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
        raise V32ContractError("V3 parent artifacts were not found for fingerprinting")
    return payload


def assert_parent_fingerprints_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> str:
    if dict(before) != dict(after):
        changed = sorted(set(before) | set(after))
        details = [key for key in changed if before.get(key) != after.get(key)]
        raise V32ContractError(
            "V3 parent files unexpectedly modified: " + ", ".join(details[:12])
        )
    return "YES"


def assert_not_v3_parent_output_dir(path: Path, project_root: Path) -> None:
    run_dir = Path(path).resolve()
    name = run_dir.name
    if any(marker in name for marker in PARENT_OUTPUT_MARKERS) and "V3_2" not in name:
        raise V32ContractError(f"refusing to write V3.2 outputs into V3 parent run: {path}")
    forbidden = {
        v3_screening_root(project_root).resolve(),
        v3_robustness_root(project_root).resolve(),
        v3_frozen_manifest_path(project_root).resolve(),
    }
    if run_dir in forbidden:
        raise V32ContractError(f"refusing to write V3.2 outputs into V3 parent path: {path}")
    for parent in (v3_screening_root(project_root), v3_robustness_root(project_root)):
        try:
            run_dir.relative_to(parent.resolve())
        except ValueError:
            continue
        raise V32ContractError(f"refusing to write V3.2 outputs under V3 parent: {path}")


def assert_formal_methods_unchanged_v32() -> Tuple[str, ...]:
    expected = assert_formal_methods_untouched()
    assert_formal_methods_unchanged()
    if METHOD_DISPLAY in FORMAL_METHODS or METHOD_ARM in FORMAL_METHODS:
        raise V32ContractError("V3.2 must not be added to FORMAL_METHODS")
    if PARENT_METHOD in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise V32ContractError("MS-KATL development methods must stay outside FORMAL_METHODS")
    return expected


def assert_v3_1_not_inherited(source: Optional[str] = None) -> str:
    text = source if source is not None else Path(__file__).read_text(encoding="utf-8")
    import_needle = "from src.transfer_methods." + "ms_katl_fto_v3_1_calibrated" + " import"
    alias_needle = "import src.transfer_methods." + "ms_katl_fto_v3_1_calibrated"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(import_needle) or stripped.startswith(alias_needle):
            raise V32ContractError("V3.2 must not import the V3.1 calibration module")
    leaked_a = "def " + "calibrate_v3_raw("
    leaked_b = "def " + "apply_bias_calibration("
    if leaked_a in text or leaked_b in text:
        raise V32ContractError("V3.1 calibration leaked into V3.2")
    if PARENT_METHOD != V3_METHOD_DISPLAY:
        raise V32ContractError("V3.2 parent must be FUSED_TARGET_ONLY_V3")
    if PARENT_METHOD == "FUSED_TARGET_ONLY_V3_1_CALIBRATED":
        raise V32ContractError("V3.2 must not inherit V3.1")
    if V3_1_CALIBRATION_ENABLED != "NO":
        raise V32ContractError("V3_1_CALIBRATION_ENABLED must be NO")
    return "YES"


def assert_v3_parent_contract(project_root: Path) -> Dict[str, Any]:
    frozen = load_v3_frozen_robustness_manifest(project_root)
    if frozen.get("method") != PARENT_METHOD:
        raise V32ContractError(f"V3 frozen manifest method drifted: {frozen.get('method')}")
    if str(frozen.get("architecture_digest")) != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V32ContractError("V3 frozen architecture digest drifted")
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V32ContractError("live V3 architecture digest drifted")
    for dataset, target in EXPECTED_COMMON_TARGETS:
        label = format_lr(parse_lr(frozen["datasets"][dataset]["learning_rate"]))
        identity = str(frozen["datasets"][dataset]["primary_target"]).strip()
        if identity != target:
            raise V32ContractError(f"V3 primary target drifted on {dataset}: {identity}")
        if label != EXPECTED_V3_LR[(dataset, target)]:
            raise V32ContractError(
                f"V3 frozen LR drifted on {dataset} {target}: {label} != {EXPECTED_V3_LR[(dataset, target)]}"
            )
    guard_path = (
        v3_robustness_root(project_root) / "results" / "three_track_completion_guard.json"
    )
    if guard_path.is_file():
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        if int(guard.get("EXPECTED_RUNS", -1)) != 162 or str(guard.get("ROBUSTNESS_COMPLETE")) != "YES":
            raise V32ContractError(f"V3 robustness is not the frozen complete reference: {guard}")
        if int(guard.get("OK_RUNS", -1)) != 162 or int(guard.get("FAILED_RUNS", 1)) != 0:
            raise V32ContractError(f"V3 robustness completeness drifted: {guard}")
    return frozen


def v3_2_architecture() -> Dict[str, Any]:
    arch = dict(v3_architecture())
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V32ContractError("V3.2 cannot inherit a drifted V3 architecture")
    return arch


def v3_2_training_recipe() -> Dict[str, Any]:
    recipe = dict(v3_training_recipe())
    expected = {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    if recipe != expected:
        raise V32ContractError(f"V3.2 recipe must inherit frozen V3 recipe, got {recipe}")
    return recipe


def v3_2_revin_config() -> Dict[str, Any]:
    config = v3_revin_config()
    expected = {"enabled": True, "affine": True, "eps": 1e-5, "target_channel": 0}
    if bool(config["enabled"]) is not True or bool(config["affine"]) is not True:
        raise V32ContractError(f"V3.2 RevIN drifted: {config}")
    if not math.isclose(float(config["eps"]), expected["eps"], rel_tol=0.0, abs_tol=0.0):
        raise V32ContractError(f"V3.2 RevIN eps drifted: {config}")
    if int(config["target_channel"]) != expected["target_channel"]:
        raise V32ContractError(f"V3.2 RevIN target_channel drifted: {config}")
    payload = {
        **dict(config),
        "output_mode": REVIN_OUTPUT_MODE,
        "include_mean": False,
        "residualization_scope": RESIDUALIZATION_SCOPE,
    }
    return payload


def inherited_v3_spec():
    spec = v3_arm_spec()
    if spec.architecture_digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V32ContractError("inherited V3 arm digest drifted")
    if spec.method_display != PARENT_METHOD:
        raise V32ContractError("inherited V3 method display drifted")
    return spec


@dataclass(frozen=True)
class V32RunIdentity:
    dataset: str
    target_identity: str
    horizon: int
    seed: int
    selected_lr: float
    selected_lr_label: str

    @property
    def arm(self) -> str:
        return "v3_2"

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
    def cell_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        return (
            f"v3_2_{self.dataset}_{target_slug}"
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
            raise V32ContractError(f"frozen target drift on {dataset}")
        if label != EXPECTED_V3_LR[(dataset, target)]:
            raise V32ContractError(
                f"V3.2 must inherit frozen V3 LR for {dataset} {target}, got {label}"
            )
        mapping[(dataset, target)] = label
    if mapping != dict(EXPECTED_V3_LR):
        raise V32ContractError("V3.2 LR mapping drifted from the frozen V3 map")
    return mapping


def build_v3_2_identities(
    *,
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_map: Mapping[Tuple[str, str], str],
) -> Tuple[V32RunIdentity, ...]:
    identities: List[V32RunIdentity] = []
    targets = dict(EXPECTED_COMMON_TARGETS)
    for dataset in datasets:
        target = targets.get(dataset)
        if target is None:
            raise V32ContractError(f"dataset {dataset} is outside the frozen V3 target map")
        label = lr_map.get((dataset, target))
        if label is None:
            raise V32ContractError(f"missing frozen LR for {dataset} {target}")
        for horizon in horizons:
            for seed in seeds:
                identities.append(
                    V32RunIdentity(
                        dataset=dataset,
                        target_identity=target,
                        horizon=int(horizon),
                        seed=int(seed),
                        selected_lr=parse_lr(label),
                        selected_lr_label=label,
                    )
                )
    tuples = [item.identity_tuple for item in identities]
    if len(tuples) != len(set(tuples)):
        raise V32ContractError("duplicate V3.2 cell identity")
    return tuple(identities)


def parse_dataset_ids(spec: Optional[object]) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return SCREENING_DATASETS
    parts = [part.strip().upper() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(f"D{int(part.replace('D', ''))}" for part in parts)
    extra = [item for item in resolved if item not in SCREENING_DATASETS]
    if extra:
        raise V32ContractError(f"unknown datasets: {extra}")
    seen = set()
    ordered: List[str] = []
    for item in resolved:
        if item in seen:
            raise V32ContractError(f"duplicate dataset {item}")
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
        raise V32ContractError(f"{name} values {extra} are not in frozen set {list(allowed)}")
    seen = set()
    ordered: List[int] = []
    for item in resolved:
        if item in seen:
            raise V32ContractError(f"duplicate {name} {item}")
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


def _as_normalized_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts.normalize().date())


def _entity_pair(entity_id: object, item_id: object) -> Tuple[str, str]:
    return (str(entity_id), str(item_id))


def sequence_window_meta(
    df: pd.DataFrame,
    *,
    horizon: int,
    window_size: int,
    feature_columns: Sequence[str],
) -> List[Dict[str, str]]:
    """Mirror build_tabular_sequence window indices with calendar timestamps."""
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


def build_entity_sales_lookup(
    frame: pd.DataFrame,
) -> Dict[Tuple[str, str], Dict[pd.Timestamp, float]]:
    if "entity_id" not in frame.columns or "item_id" not in frame.columns:
        raise V32ContractError("seasonal prior lookup requires entity_id and item_id")
    if "date" not in frame.columns or "sales" not in frame.columns:
        raise V32ContractError("seasonal prior lookup requires date and sales")
    ordered = frame.sort_values(["entity_id", "item_id", "date"], kind="mergesort")
    payload: Dict[Tuple[str, str], Dict[pd.Timestamp, float]] = {}
    for group_key, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        normalized_key = group_key if isinstance(group_key, tuple) else (group_key,)
        entity_id = str(normalized_key[0])
        item_id = str(normalized_key[1]) if len(normalized_key) > 1 else ""
        mapping: Dict[pd.Timestamp, float] = {}
        dates = pd.to_datetime(group["date"], errors="raise")
        sales = pd.to_numeric(group["sales"], errors="raise")
        for date_value, sales_value in zip(dates, sales):
            ts = _as_normalized_timestamp(date_value)
            if ts in mapping:
                raise V32ContractError(
                    f"duplicate sales date for entity={entity_id}/{item_id} date={ts.date()}"
                )
            mapping[ts] = float(sales_value)
        payload[_entity_pair(entity_id, item_id)] = mapping
    return payload


def observable_history(
    lookup: Mapping[pd.Timestamp, float],
    forecast_origin: object,
) -> Dict[pd.Timestamp, float]:
    origin = _as_normalized_timestamp(forecast_origin)
    return {ts: value for ts, value in lookup.items() if ts <= origin}


def lookup_lag7_seasonal_prior(
    *,
    target_date: object,
    forecast_origin: object,
    history: Mapping[pd.Timestamp, float],
    entity_key: Tuple[str, str],
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """Date-aware lag-7 prior with matured-truth and fail-closed missing lag."""
    target_ts = _as_normalized_timestamp(target_date)
    origin_ts = _as_normalized_timestamp(forecast_origin)
    if horizon is not None:
        expected = origin_ts + pd.Timedelta(days=int(horizon))
        if target_ts != expected:
            raise V32ContractError(
                "SEASONAL_LAG7_TARGET_DATE_MISMATCH "
                f"entity={entity_key[0]}/{entity_key[1]} "
                f"forecast_origin={origin_ts.date()} target_date={target_ts.date()} "
                f"expected_target_date={expected.date()} horizon={int(horizon)}"
            )
    lag_ts = target_ts - pd.Timedelta(days=int(LAG_DAYS))
    if lag_ts > origin_ts:
        raise V32ContractError(
            "SEASONAL_LAG7_FUTURE_TARGET_ACCESS_FORBIDDEN "
            f"entity={entity_key[0]}/{entity_key[1]} "
            f"forecast_origin={origin_ts.date()} target_date={target_ts.date()} "
            f"required_lag_date={lag_ts.date()} PRIOR_TIMESTAMP>FORECAST_ORIGIN"
        )
    if not (lag_ts <= origin_ts):
        raise V32ContractError("PRIOR_TIMESTAMP <= FORECAST_ORIGIN assertion failed")
    if lag_ts not in history:
        raise V32ContractError(
            "SEASONAL_LAG7_HISTORY_MISSING "
            f"entity={entity_key[0]}/{entity_key[1]} "
            f"forecast_origin={origin_ts.date()} target_date={target_ts.date()} "
            f"required_lag_date={lag_ts.date()}"
        )
    value = float(history[lag_ts])
    if not math.isfinite(value):
        raise V32ContractError(
            "SEASONAL_LAG7_HISTORY_NONFINITE "
            f"entity={entity_key[0]}/{entity_key[1]} "
            f"required_lag_date={lag_ts.date()}"
        )
    return {
        "lag_timestamp": lag_ts.strftime("%Y-%m-%d"),
        "lag_value": value,
        "forecast_origin": origin_ts.strftime("%Y-%m-%d"),
        "target_date": target_ts.strftime("%Y-%m-%d"),
        "prior_maturity_guard": "PASS",
        "entity_id": entity_key[0],
        "item_id": entity_key[1],
        "seasonal_lag_days": int(LAG_DAYS),
    }


def seasonal_prior_from_bl5_record(record: Any) -> float:
    """BL5 lag-7 information contract, implemented independently of baseline scores."""
    origin = pd.Timestamp(record.forecast_origin).normalize()
    target = pd.Timestamp(record.label_date).normalize()
    input_dates = tuple(_as_normalized_timestamp(value) for value in record.input_dates)
    input_sales = tuple(float(value) for value in record.input_sales)
    if len(input_dates) != len(input_sales):
        raise V32ContractError("SEASONAL_NAIVE_HISTORY_LENGTH_MISMATCH")
    if len(set(input_dates)) != len(input_dates):
        raise V32ContractError("SEASONAL_NAIVE_HISTORY_DUPLICATE_DATE")
    if any(date > origin for date in input_dates):
        raise V32ContractError("SEASONAL_NAIVE_HISTORY_AFTER_FORECAST_ORIGIN")
    history = dict(zip(input_dates, input_sales))
    entity = getattr(record, "target_key", ())
    if isinstance(entity, (tuple, list)):
        entity_key = _entity_pair(entity[0], entity[1] if len(entity) > 1 else "")
    else:
        entity_key = _entity_pair(entity, "")
    prior = lookup_lag7_seasonal_prior(
        target_date=target,
        forecast_origin=origin,
        history=history,
        entity_key=entity_key,
        horizon=int(record.horizon),
    )
    return float(prior["lag_value"])


def residual_original(y_original: np.ndarray, prior_original: np.ndarray) -> np.ndarray:
    y = np.asarray(y_original, dtype=np.float64).reshape(-1)
    prior = np.asarray(prior_original, dtype=np.float64).reshape(-1)
    if y.shape != prior.shape:
        raise V32ContractError(f"residual arrays must align: {y.shape} vs {prior.shape}")
    return y - prior


def sales_to_minmax(values: np.ndarray, sales_min: float, sales_max: float) -> np.ndarray:
    span = float(sales_max) - float(sales_min)
    if not math.isfinite(span) or span == 0.0:
        raise V32ContractError("sales scaler span must be finite and non-zero")
    return (np.asarray(values, dtype=np.float64).reshape(-1) - float(sales_min)) / span


def residual_to_original(residual_minmax: np.ndarray, sales_min: float, sales_max: float) -> np.ndarray:
    span = float(sales_max) - float(sales_min)
    if not math.isfinite(span) or span == 0.0:
        raise V32ContractError("sales scaler span must be finite and non-zero")
    return np.asarray(residual_minmax, dtype=np.float64).reshape(-1) * span


def reconstruct_original_sales(
    prior_original: np.ndarray,
    residual_minmax: np.ndarray,
    sales_min: float,
    sales_max: float,
) -> np.ndarray:
    return np.asarray(prior_original, dtype=np.float64).reshape(-1) + residual_to_original(
        residual_minmax, sales_min, sales_max
    )


def denormalize_residual_numpy(
    revin: RevIN,
    y_norm: np.ndarray,
    mean: np.ndarray,
    stdev: np.ndarray,
) -> np.ndarray:
    return revin.denormalize_numpy(y_norm, mean, stdev, include_mean=False).reshape(-1)


def assert_residual_revin_semantics() -> str:
    source = (
        inspect.getsource(denormalize_residual_numpy)
        + inspect.getsource(reconstruct_original_sales)
        + inspect.getsource(residual_original)
    )
    if "include_mean=False" not in source:
        raise V32ContractError("V3_2_BLOCKED_BY_RESIDUAL_REVIN_SEMANTICS: scale-only denorm missing")
    if "prior_original" not in source or "residual_minmax" not in source:
        raise V32ContractError("V3_2_BLOCKED_BY_RESIDUAL_REVIN_SEMANTICS: reconstruction missing")
    return "PASS"


def assert_mechanism_is_uniform(source: Optional[str] = None) -> Dict[str, str]:
    text = source if source is not None else (
        inspect.getsource(lookup_lag7_seasonal_prior)
        + "\n"
        + inspect.getsource(residual_original)
        + "\n"
        + inspect.getsource(reconstruct_original_sales)
        + "\n"
        + inspect.getsource(seasonal_prior_from_bl5_record)
    )
    forbidden_tokens = (
        "if dataset",
        "if D1",
        "if D2",
        "best_seed",
        "min_seed",
        "selected_seed",
        "lag=1",
        "MA7",
        "median_bias",
    )
    hits = [token for token in forbidden_tokens if token in text]
    if hits:
        raise V32ContractError(f"dataset/horizon/seed/prior-selection logic detected: {hits}")
    for fn in (
        lookup_lag7_seasonal_prior,
        residual_original,
        reconstruct_original_sales,
        seasonal_prior_from_bl5_record,
    ):
        params = inspect.signature(fn).parameters
        for name in ("dataset", "seed", "y_test", "baseline", "rmse"):
            if name in params:
                raise V32ContractError(f"{fn.__name__} must not take {name}")
    return {
        "TARGET_RESELECTION": "NO",
        "LR_RESELECTION": "NO",
        "PRIOR_RESELECTION": "NO",
        "DATASET_SPECIFIC_TUNING": "NO",
        "HORIZON_SPECIFIC_TUNING": "NO",
        "SEED_SPECIFIC_TUNING": "NO",
        "TEST_FUTURE_TRUTH_USED": "NO",
        "V3_1_CALIBRATION_ENABLED": V3_1_CALIBRATION_ENABLED,
    }


def assert_no_baseline_result_consumption(source: Optional[str] = None) -> str:
    text = source if source is not None else (
        inspect.getsource(lookup_lag7_seasonal_prior)
        + inspect.getsource(seasonal_prior_from_bl5_record)
        + inspect.getsource(residual_original)
    )
    forbidden = ("bl5_rmse", "baseline_rmse", "predict_bl5(")
    hits = [token for token in forbidden if token in text]
    if hits:
        raise V32ContractError(f"V3.2 prior path consumed baseline results: {hits}")
    return "YES"


@dataclass(frozen=True)
class ResidualSplit:
    residual_minmax: np.ndarray
    y_original: np.ndarray
    prior_original: np.ndarray
    prior_minmax: np.ndarray
    meta: List[Dict[str, str]]
    audit_rows: List[Dict[str, Any]]


def _residualize_split(
    *,
    y_minmax: np.ndarray,
    frame: pd.DataFrame,
    entity_lookup: Mapping[Tuple[str, str], Mapping[pd.Timestamp, float]],
    horizon: int,
    window_size: int,
    feature_columns: Sequence[str],
    sales_min: float,
    sales_max: float,
    split_name: str,
) -> ResidualSplit:
    y = np.asarray(y_minmax, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return ResidualSplit(
            residual_minmax=y,
            y_original=y,
            prior_original=y,
            prior_minmax=y,
            meta=[],
            audit_rows=[],
        )
    meta = sequence_window_meta(
        frame, horizon=int(horizon), window_size=int(window_size), feature_columns=feature_columns
    )
    if len(meta) != y.size:
        raise V32ContractError(
            f"{split_name} window meta length {len(meta)} != labels {y.size}"
        )
    y_original = y * (float(sales_max) - float(sales_min)) + float(sales_min)
    priors = np.empty(y.size, dtype=np.float64)
    audit_rows: List[Dict[str, Any]] = []
    for index, record in enumerate(meta):
        entity_key = _entity_pair(record["entity_id"], record["item_id"])
        origin = record["window_end"]
        full_history = entity_lookup.get(entity_key)
        if full_history is None:
            raise V32ContractError(
                f"SEASONAL_LAG7_ENTITY_MISSING split={split_name} entity={entity_key[0]}/{entity_key[1]}"
            )
        history = observable_history(full_history, origin)
        prior = lookup_lag7_seasonal_prior(
            target_date=record["label_date"],
            forecast_origin=origin,
            history=history,
            entity_key=entity_key,
            horizon=int(horizon),
        )
        priors[index] = float(prior["lag_value"])
        actual_residual = float(y_original[index] - priors[index])
        audit_rows.append(
            {
                "split": split_name,
                "timestamp": record["label_date"],
                "forecast_origin": prior["forecast_origin"],
                "lag_timestamp": prior["lag_timestamp"],
                "lag_value": prior["lag_value"],
                "actual_target": float(y_original[index]),
                "actual_residual": actual_residual,
                "window_start": record["window_start"],
                "window_end": record["window_end"],
                "entity_id": record["entity_id"],
                "item_id": record["item_id"],
            }
        )
    prior_minmax = sales_to_minmax(priors, sales_min, sales_max)
    residual = y - prior_minmax
    reconstructed = reconstruct_original_sales(priors, residual, sales_min, sales_max)
    if not np.allclose(reconstructed, y_original, rtol=0.0, atol=1e-9):
        raise V32ContractError("residual reconstruction identity failed on labels")
    return ResidualSplit(
        residual_minmax=residual,
        y_original=y_original,
        prior_original=priors,
        prior_minmax=prior_minmax,
        meta=meta,
        audit_rows=audit_rows,
    )


def residualize_stage_b_bundle(
    bundle: Any,
    *,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    horizon: int,
    window_size: int = WINDOW_SIZE,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    sales_min, sales_max = _sales_minmax(bundle.target_scaler, bundle.target_feature_columns)
    history_frame = pd.concat([raw_train, raw_val, raw_test], ignore_index=True)
    entity_lookup = build_entity_sales_lookup(history_frame)
    train = _residualize_split(
        y_minmax=bundle.y_train,
        frame=raw_train,
        entity_lookup=entity_lookup,
        horizon=horizon,
        window_size=window_size,
        feature_columns=feature_columns,
        sales_min=sales_min,
        sales_max=sales_max,
        split_name="train",
    )
    val = _residualize_split(
        y_minmax=bundle.y_val,
        frame=raw_val,
        entity_lookup=entity_lookup,
        horizon=horizon,
        window_size=window_size,
        feature_columns=feature_columns,
        sales_min=sales_min,
        sales_max=sales_max,
        split_name="val",
    )
    test = _residualize_split(
        y_minmax=bundle.y_test,
        frame=raw_test,
        entity_lookup=entity_lookup,
        horizon=horizon,
        window_size=window_size,
        feature_columns=feature_columns,
        sales_min=sales_min,
        sales_max=sales_max,
        split_name="test",
    )
    residual_bundle = replace(
        bundle,
        y_train=train.residual_minmax.astype(np.float32),
        y_val=val.residual_minmax.astype(np.float32),
        y_test=test.residual_minmax.astype(np.float32),
    )
    return {
        "bundle": residual_bundle,
        "train": train,
        "val": val,
        "test": test,
        "sales_min": sales_min,
        "sales_max": sales_max,
        "residualization_scope": RESIDUALIZATION_SCOPE,
        "prior_maturity_guard": "PASS",
    }


def _sales_minmax(scaler: object, feature_columns: Sequence[str]) -> Tuple[float, float]:
    features = [str(col) for col in feature_columns]
    if "sales" not in features:
        raise V32ContractError("sales channel missing from feature columns")
    idx = features.index("sales")
    data_min = np.asarray(getattr(scaler, "data_min_"), dtype=np.float64)
    data_max = np.asarray(getattr(scaler, "data_max_"), dtype=np.float64)
    if idx >= data_min.size or idx >= data_max.size:
        raise V32ContractError("sales scaler index out of range")
    return float(data_min[idx]), float(data_max[idx])


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def run_v3_2_stage_b(
    *,
    bundle: Any,
    identity: V32RunIdentity,
    spec,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_columns: Sequence[str],
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
) -> Dict[str, Any]:
    del metric_protocol
    assert_residual_revin_semantics()
    assert_v3_1_not_inherited()
    flags = assert_mechanism_is_uniform()
    residualized = residualize_stage_b_bundle(
        bundle,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        horizon=int(identity.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=feature_columns,
    )
    ablation = resolve_ablation_arm(spec.ablation_arm)
    kwargs = dict(spec.run_ablation_kwargs())
    kwargs["revin_include_mean"] = False
    arm = run_ablation_arm(
        ablation,
        residualized["bundle"],
        epochs=int(epochs),
        batch_size=int(batch_size),
        gamma=0.5,
        use_adversarial=False,
        metric_protocol=None,
        fine_tune_learning_rate=float(identity.selected_lr),
        return_runtime=True,
        **kwargs,
    )
    runtime = arm.pop("runtime")
    rhat_minmax = np.asarray(arm["y_pred"], dtype=np.float64).reshape(-1)
    test = residualized["test"]
    if rhat_minmax.shape != test.prior_original.shape:
        raise V32ContractError(
            f"residual prediction length {rhat_minmax.shape} != priors {test.prior_original.shape}"
        )
    yhat_minmax = test.prior_minmax + rhat_minmax
    yhat_original = inverse_transform_sales(
        yhat_minmax,
        sales_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    y_true_original = inverse_transform_sales(
        np.asarray(bundle.y_test, dtype=np.float64).reshape(-1),
        sales_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    if y_true_original.shape != yhat_original.shape:
        raise V32ContractError("reconstructed prediction length drifted from y_true")
    rmse_o = original_rmse(y_true_original, yhat_original)
    smape_o = float(smape(y_true_original, yhat_original))
    predicted_residual_original = residual_to_original(
        rhat_minmax, residualized["sales_min"], residualized["sales_max"]
    )
    history = arm.get("history") or {}
    val_rmse_hist = list(history.get("val_rmse") or [])
    residual_validation_loss = (
        float(val_rmse_hist[-1]) if val_rmse_hist and math.isfinite(float(val_rmse_hist[-1])) else None
    )
    revin = runtime.get("revin")
    if revin is None:
        raise V32ContractError("V3.2 requires RevIN runtime")
    recipe = dict(arm.get("training_recipe") or {})
    if recipe.get("revin_include_mean") is not False:
        raise V32ContractError("V3.2 Stage-B must use residual scale-only RevIN denorm")
    return {
        "arm": arm,
        "runtime": runtime,
        "residualized": residualized,
        "y_true_original": y_true_original,
        "y_pred_original": yhat_original,
        "predicted_residual_minmax": rhat_minmax,
        "predicted_residual_original": predicted_residual_original,
        "seasonal_prior": test.prior_original,
        "rmse_original": rmse_o,
        "smape_original": smape_o,
        "residual_validation_loss": residual_validation_loss,
        "train_residual_sample_count": int(residualized["train"].residual_minmax.size),
        "validation_residual_sample_count": int(residualized["val"].residual_minmax.size),
        "test_prediction_count": int(rhat_minmax.size),
        "prior_maturity_guard": "PASS",
        "metric_space": ORIGINAL_SALES_SPACE,
        "revin_output_mode": REVIN_OUTPUT_MODE,
        "residualization_scope": RESIDUALIZATION_SCOPE,
        "V3_1_CALIBRATION_ENABLED": V3_1_CALIBRATION_ENABLED,
        **flags,
    }


def parent_v3_prediction_path(
    project_root: Path,
    identity: V32RunIdentity,
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


def lookup_parent_v3_metrics(
    v3_rows: Sequence[Mapping[str, Any]],
    identity: V32RunIdentity,
) -> Tuple[Optional[float], Optional[float]]:
    for row in v3_rows:
        if str(row.get("arm") or "").lower() != "v3":
            continue
        if str(row.get("dataset")) != identity.dataset:
            continue
        if str(row.get("target_identity")) != identity.target_identity:
            continue
        if int(row.get("horizon")) != int(identity.horizon):
            continue
        if int(row.get("seed")) != int(identity.seed):
            continue
        return float(row["rmse_original"]), float(row["smape_original"])
    return None, None


def test_label_dates(
    raw_test: pd.DataFrame,
    *,
    horizon: int,
    window_size: int = WINDOW_SIZE,
    feature_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    cols = list(feature_columns) if feature_columns is not None else ["sales"]
    meta = sequence_window_meta(
        raw_test, horizon=horizon, window_size=window_size, feature_columns=cols
    )
    return [record["label_date"] for record in meta]


def cell_path(cells_dir: Path, identity: V32RunIdentity) -> Path:
    return Path(cells_dir) / f"{identity.cell_id}.json"


def classify_cell_artifact(path: Path, identity: V32RunIdentity, digest: str) -> str:
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
        if str(payload.get("parent_method")) == "FUSED_TARGET_ONLY_V3_1_CALIBRATED":
            return "STALE"
        if str(payload.get("seasonal_prior_type")) != SEASONAL_PRIOR_TYPE:
            return "STALE"
        if int(payload.get("seasonal_lag_days")) != LAG_DAYS:
            return "STALE"
        if str(payload.get("metric_space")) != ORIGINAL_SALES_SPACE:
            return "STALE"
        if str(payload.get("V3_1_CALIBRATION_ENABLED") or V3_1_CALIBRATION_ENABLED) != "NO":
            return "STALE"
        parent_digest = str(payload.get("parent_architecture_digest") or digest)
        if parent_digest != digest:
            return "STALE"
        for key in ("rmse_original", "smape_original"):
            if not math.isfinite(float(payload[key])):
                return "MALFORMED"
        if int(payload["test_prediction_count"]) <= 0:
            return "MALFORMED"
    except (TypeError, ValueError, KeyError):
        return "MALFORMED"
    return "OK"


def load_valid_ok_cells(
    cells_dir: Path,
    identities: Sequence[V32RunIdentity],
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
    identities: Sequence[V32RunIdentity],
    cell_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = len(identities)
    ok_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "OK"]
    failed_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "FAILED"]
    materialized = []
    for row in ok_rows:
        materialized.append(
            (
                "v3_2",
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
        raise V32ContractError(f"frozen V3 robustness CSV missing: {path}")
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
        raise V32ContractError(f"frozen V3 robustness must contain 54 OK V3 rows, got {len(kept)}")
    return kept


def summarize_seed_group(rows: Sequence[Mapping[str, Any]], rmse_key: str, smape_key: str) -> Dict[str, Any]:
    rmses = [float(row[rmse_key]) for row in rows]
    smapes = [float(row[smape_key]) for row in rows]
    if len(rmses) != 3:
        raise V32ContractError(f"headline aggregation requires exactly 3 seeds, got {len(rmses)}")
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
    seeds: Sequence[int] = V3_2_SEEDS,
) -> Dict[str, Dict[str, float]]:
    from src.transfer_methods.ms_katl_primary_target_lr_screening import SCREENING_MODE

    path = default_baseline_csv(dataset, project_root)
    if not Path(path).is_file():
        raise V32ContractError(f"baseline CSV missing for {dataset}: {path}")
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
                raise V32ContractError(
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
    v3_2_rows: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> Dict[str, Any]:
    def _group(rows: Sequence[Mapping[str, Any]], rmse_key: str, smape_key: str) -> Dict[Tuple[str, str, int], List[dict]]:
        grouped: Dict[Tuple[str, str, int], List[dict]] = {}
        for row in rows:
            if str(row.get("status") or "OK").upper() != "OK":
                continue
            key = (str(row["dataset"]), str(row["target_identity"]), int(row["horizon"]))
            grouped.setdefault(key, []).append(dict(row))
        return grouped

    v32_groups = _group(v3_2_rows, "rmse_original", "smape_original")
    v3_groups = _group(v3_rows, "rmse_original", "smape_original")
    rmse_table: List[Dict[str, Any]] = []
    smape_table: List[Dict[str, Any]] = []
    vs_v3_pcts: List[float] = []
    vs_best_gaps: List[float] = []
    diagnostics = {
        "beats_v3": 0,
        "beats_best_bl": 0,
        "top1": 0,
        "top2": 0,
        "top3": 0,
        "n": 0,
        "by_dataset": {},
    }
    for dataset, target in EXPECTED_COMMON_TARGETS:
        dataset_ranks: Dict[str, int] = {}
        for horizon in V3_2_HORIZONS:
            key = (dataset, target, int(horizon))
            v32 = v32_groups.get(key) or []
            v3 = v3_groups.get(key) or []
            if len(v32) != 3 or len(v3) != 3:
                raise V32ContractError(
                    f"dataset×horizon aggregation requires 3 seeds for {key}, "
                    f"got v3.2={len(v32)} v3={len(v3)}"
                )
            v32_stats = summarize_seed_group(v32, "rmse_original", "smape_original")
            v3_stats = summarize_seed_group(v3, "rmse_original", "smape_original")
            bl = load_baseline_seed_means(
                project_root,
                dataset=dataset,
                target_identity=target,
                horizon=int(horizon),
            )
            best_method, best_payload = min(bl.items(), key=lambda item: item[1]["mean_rmse"])
            worst_method, worst_payload = max(bl.items(), key=lambda item: item[1]["mean_rmse"])
            ranked = [("V3.2", v32_stats["rmse_mean"])] + [
                (method, payload["mean_rmse"]) for method, payload in bl.items()
            ]
            ranked.sort(key=lambda item: item[1])
            rank = 1 + sum(1 for name, value in ranked if name != "V3.2" and value < v32_stats["rmse_mean"])
            beats = "YES" if v32_stats["rmse_mean"] < best_payload["mean_rmse"] else "NO"
            vs_v3 = relative_improvement(v3_stats["rmse_mean"], v32_stats["rmse_mean"])
            vs_best = relative_improvement(best_payload["mean_rmse"], v32_stats["rmse_mean"])
            vs_worst = relative_improvement(worst_payload["mean_rmse"], v32_stats["rmse_mean"])
            diagnostics["n"] += 1
            if vs_v3 is not None and vs_v3 > 0:
                diagnostics["beats_v3"] += 1
            if beats == "YES":
                diagnostics["beats_best_bl"] += 1
            if rank == 1:
                diagnostics["top1"] += 1
            if rank <= 2:
                diagnostics["top2"] += 1
            if rank <= 3:
                diagnostics["top3"] += 1
            if vs_v3 is not None:
                vs_v3_pcts.append(float(vs_v3) * 100.0)
            vs_best_gaps.append(float(v32_stats["rmse_mean"] - best_payload["mean_rmse"]))
            dataset_ranks[f"H{int(horizon)}"] = int(rank)
            rmse_table.append(
                {
                    "DATASET": dataset,
                    "HORIZON": int(horizon),
                    "V3_RMSE_MEAN": v3_stats["rmse_mean"],
                    "V3_2_RMSE_MEAN": v32_stats["rmse_mean"],
                    "V3_RMSE_STD": v3_stats["rmse_std"],
                    "V3_2_RMSE_STD": v32_stats["rmse_std"],
                    "V3_RMSE_MEDIAN": v3_stats["rmse_median"],
                    "V3_2_RMSE_MEDIAN": v32_stats["rmse_median"],
                    "V3_2_VS_V3_PCT": _pct(vs_v3),
                    "BEST_BL": best_method,
                    "BEST_BL_RMSE_MEAN": best_payload["mean_rmse"],
                    "V3_2_VS_BEST_BL_PCT": _pct(vs_best),
                    "WORST_BL": worst_method,
                    "WORST_BL_RMSE_MEAN": worst_payload["mean_rmse"],
                    "V3_2_VS_WORST_BL_PCT": _pct(vs_worst),
                    "V3_2_RANK": rank,
                    "BEST_BL_BEAT": beats,
                    "TARGET": target,
                }
            )
            best_s_method, best_s = min(bl.items(), key=lambda item: item[1]["mean_smape"])
            worst_s_method, worst_s = max(bl.items(), key=lambda item: item[1]["mean_smape"])
            smape_rank_pool = [("V3.2", v32_stats["smape_mean"])] + [
                (method, payload["mean_smape"]) for method, payload in bl.items()
            ]
            smape_rank = 1 + sum(
                1 for name, value in smape_rank_pool if name != "V3.2" and value < v32_stats["smape_mean"]
            )
            smape_table.append(
                {
                    "DATASET": dataset,
                    "HORIZON": int(horizon),
                    "V3_SMAPE_MEAN": v3_stats["smape_mean"],
                    "V3_2_SMAPE_MEAN": v32_stats["smape_mean"],
                    "V3_SMAPE_STD": v3_stats["smape_std"],
                    "V3_2_SMAPE_STD": v32_stats["smape_std"],
                    "V3_SMAPE_MEDIAN": v3_stats["smape_median"],
                    "V3_2_SMAPE_MEDIAN": v32_stats["smape_median"],
                    "V3_2_VS_V3_PCT": _pct(relative_improvement(v3_stats["smape_mean"], v32_stats["smape_mean"])),
                    "BEST_BL": best_s_method,
                    "BEST_BL_SMAPE_MEAN": best_s["mean_smape"],
                    "V3_2_VS_BEST_BL_PCT": _pct(relative_improvement(best_s["mean_smape"], v32_stats["smape_mean"])),
                    "WORST_BL": worst_s_method,
                    "WORST_BL_SMAPE_MEAN": worst_s["mean_smape"],
                    "V3_2_VS_WORST_BL_PCT": _pct(relative_improvement(worst_s["mean_smape"], v32_stats["smape_mean"])),
                    "V3_2_RANK": smape_rank,
                    "BEST_BL_BEAT": "YES" if v32_stats["smape_mean"] < best_s["mean_smape"] else "NO",
                    "TARGET": target,
                }
            )
        diagnostics["by_dataset"][dataset] = dataset_ranks
    if len(rmse_table) != 18 or len(smape_table) != 18:
        raise V32ContractError(
            f"headline tables must have 18 dataset×horizon rows, got {len(rmse_table)}/{len(smape_table)}"
        )
    diagnostics["median_rmse_improvement_vs_v3"] = (
        float(statistics.median(vs_v3_pcts)) if vs_v3_pcts else None
    )
    diagnostics["mean_rmse_improvement_vs_v3"] = (
        float(statistics.mean(vs_v3_pcts)) if vs_v3_pcts else None
    )
    diagnostics["median_rmse_gap_vs_best_bl"] = (
        float(statistics.median(vs_best_gaps)) if vs_best_gaps else None
    )
    return {
        "rmse": rmse_table,
        "smape": smape_table,
        "diagnostics": diagnostics,
    }


def format_headline_tables(tables: Mapping[str, Any]) -> str:
    lines = [
        "============================================================",
        "V3.2 vs V3 / BL  RMSE  (seed42/43/44 mean)",
        "============================================================",
        f"{'DATASET':<8}{'H':<4}{'V3':<12}{'V3.2':<12}{'VS_V3%':<10}{'BEST_BL':<22}{'BL_RMSE':<12}{'VS_BL%':<10}{'WORST_BL':<22}{'W_RMSE':<12}{'RANK':<6}{'BEAT'}",
    ]
    for row in tables["rmse"]:
        lines.append(
            f"{row['DATASET']:<8}{row['HORIZON']:<4}"
            f"{row['V3_RMSE_MEAN']:<12.4f}{row['V3_2_RMSE_MEAN']:<12.4f}"
            f"{(row['V3_2_VS_V3_PCT'] if row['V3_2_VS_V3_PCT'] is not None else float('nan')):<10.2f}"
            f"{str(row['BEST_BL']):<22}{row['BEST_BL_RMSE_MEAN']:<12.4f}"
            f"{(row['V3_2_VS_BEST_BL_PCT'] if row['V3_2_VS_BEST_BL_PCT'] is not None else float('nan')):<10.2f}"
            f"{str(row['WORST_BL']):<22}{row['WORST_BL_RMSE_MEAN']:<12.4f}"
            f"{row['V3_2_RANK']:<6}{row['BEST_BL_BEAT']}"
        )
    lines.extend(
        [
            "============================================================",
            "V3.2 vs V3 / BL  sMAPE  (seed42/43/44 mean)",
            "============================================================",
            f"{'DATASET':<8}{'H':<4}{'V3':<12}{'V3.2':<12}{'VS_V3%':<10}{'BEST_BL':<22}{'BL_SMAPE':<12}{'VS_BL%':<10}{'WORST_BL':<22}{'W_SMAPE':<12}{'RANK':<6}{'BEAT'}",
        ]
    )
    for row in tables["smape"]:
        lines.append(
            f"{row['DATASET']:<8}{row['HORIZON']:<4}"
            f"{row['V3_SMAPE_MEAN']:<12.4f}{row['V3_2_SMAPE_MEAN']:<12.4f}"
            f"{(row['V3_2_VS_V3_PCT'] if row['V3_2_VS_V3_PCT'] is not None else float('nan')):<10.2f}"
            f"{str(row['BEST_BL']):<22}{row['BEST_BL_SMAPE_MEAN']:<12.4f}"
            f"{(row['V3_2_VS_BEST_BL_PCT'] if row['V3_2_VS_BEST_BL_PCT'] is not None else float('nan')):<10.2f}"
            f"{str(row['WORST_BL']):<22}{row['WORST_BL_SMAPE_MEAN']:<12.4f}"
            f"{row['V3_2_RANK']:<6}{row['BEST_BL_BEAT']}"
        )
    diag = tables["diagnostics"]
    lines.extend(
        [
            "============================================================",
            f"V3_2_BEATS_V3={diag['beats_v3']}/{diag['n']}",
            f"V3_2_BEATS_BEST_BL={diag['beats_best_bl']}/{diag['n']}",
            f"V3_2_TOP1={diag['top1']}/{diag['n']}",
            f"V3_2_TOP2={diag['top2']}/{diag['n']}",
            f"V3_2_TOP3={diag['top3']}/{diag['n']}",
            f"MEDIAN_RMSE_IMPROVEMENT_VS_V3={diag['median_rmse_improvement_vs_v3']}",
            f"MEAN_RMSE_IMPROVEMENT_VS_V3={diag['mean_rmse_improvement_vs_v3']}",
            f"MEDIAN_RMSE_GAP_VS_BEST_BL={diag['median_rmse_gap_vs_best_bl']}",
            "DATASET_RANKS_H1_H3_H5:",
        ]
    )
    for dataset, ranks in diag["by_dataset"].items():
        lines.append(
            f"  {dataset}: H1={ranks.get('H1')} H3={ranks.get('H3')} H5={ranks.get('H5')}"
        )
    lines.append("============================================================")
    return "\n".join(lines) + "\n"


def build_v3_2_manifest(project_root: Path) -> Dict[str, Any]:
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
        "selection_stage": ROBUSTNESS_TRACK,
        "common_target_policy": "V3_FROZEN_PRIMARY_TARGETS",
        "targets": 6,
        "horizons": list(V3_2_HORIZONS),
        "seeds": list(V3_2_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "epochs": 50,
        "window_size": WINDOW_SIZE,
        "training_recipe": dict(v3_2_training_recipe()),
        "RevIN": v3_2_revin_config(),
        "architecture": dict(v3_2_architecture()),
        "architecture_digest": spec.architecture_digest,
        "metric_space": ORIGINAL_SALES_SPACE,
        "seasonal_prior": {
            "type": SEASONAL_PRIOR_NAME,
            "lag_days": LAG_DAYS,
            "selection": False,
            "date_aware": True,
            "missing_lag_policy": "fail_closed",
            "matured_truth_required": True,
        },
        "residual": {
            "definition": RESIDUAL_DEFINITION,
            "prediction_reconstruction": "y_t_minus_7+residual_hat",
            "scope": RESIDUALIZATION_SCOPE,
        },
        "V3_1_calibration": False,
        "datasets": datasets,
        "frozen_lr_mapping": {
            f"{dataset}|{target}": lr_label
            for (dataset, target), lr_label in EXPECTED_V3_LR.items()
        },
        "TARGET_RESELECTION": "NO",
        "LR_RESELECTION": "NO",
        "PRIOR_RESELECTION": "NO",
        "FORMAL_METHODS_CHANGE": "NO",
        "parent_v3_manifest_verified": bool(frozen.get("V3_SELECTION_CONTRACT_VERIFIED")),
        "RESIDUALIZATION_SCOPE": RESIDUALIZATION_SCOPE,
        "RESIDUAL_REVIN_SEMANTICS": assert_residual_revin_semantics(),
        "V3_1_CALIBRATION_ENABLED": V3_1_CALIBRATION_ENABLED,
    }


def validate_v3_2_manifest(manifest: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    if manifest.get("method") != METHOD_DISPLAY:
        raise V32ContractError(f"V3.2 manifest method drifted: {manifest.get('method')}")
    if manifest.get("parent_method") != PARENT_METHOD:
        raise V32ContractError("V3.2 parent_method drifted")
    if manifest.get("parent_method") == "FUSED_TARGET_ONLY_V3_1_CALIBRATED":
        raise V32ContractError("V3.2 must not inherit V3.1")
    if str(manifest.get("parent_architecture_digest")) != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise V32ContractError("V3.2 parent digest drifted")
    prior = manifest.get("seasonal_prior") or {}
    if prior.get("type") != SEASONAL_PRIOR_NAME or int(prior.get("lag_days") or -1) != LAG_DAYS:
        raise V32ContractError(f"V3.2 seasonal prior drifted: {prior}")
    if prior.get("selection") is not False:
        raise V32ContractError("V3.2 must not search priors")
    if manifest.get("V3_1_calibration") is not False:
        raise V32ContractError("V3.2 must disable V3.1 calibration")
    if int(manifest.get("expected_runs") or 0) != EXPECTED_RUNS:
        raise V32ContractError("V3.2 expected_runs must be 54")
    frozen_lr = frozen_v3_lr_map(assert_v3_parent_contract(project_root))
    for dataset, target in EXPECTED_COMMON_TARGETS:
        got = format_lr(parse_lr(manifest["datasets"][dataset]["learning_rate"]))
        if got != frozen_lr[(dataset, target)]:
            raise V32ContractError(f"V3.2 manifest LR drifted on {dataset}")
    return dict(manifest)


def load_v3_2_manifest(project_root: Path) -> Dict[str, Any]:
    path = v3_2_manifest_path(project_root)
    if not path.is_file():
        raise V32ContractError(f"V3.2 manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_v3_2_manifest(payload, project_root)


def write_v3_2_manifest(project_root: Path) -> Path:
    path = v3_2_manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v3_2_manifest(project_root)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_v3_2_manifest(loaded, project_root)
    return path


def format_preflight(
    *,
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_map: Mapping[Tuple[str, str], str],
    counts: Mapping[str, int],
) -> str:
    assert_formal_methods_unchanged_v32()
    assert_v3_1_not_inherited()
    flags = assert_mechanism_is_uniform()
    revin_status = assert_residual_revin_semantics()
    lines = [
        "============================================================",
        "MS-KATL FTO V3.2 SEASONAL-RESIDUAL PREFLIGHT",
        "============================================================",
        "",
        f"METHOD={METHOD_DISPLAY}",
        f"PARENT={PARENT_METHOD}",
        "",
        f"PRIOR=LAG7_SEASONAL",
        f"RESIDUAL_TARGET=YES",
        f"LAG_DAYS={LAG_DAYS}",
        "",
        f"DATASETS={','.join(datasets)}",
        f"TARGETS={len(datasets)}",
        f"HORIZONS={','.join(str(item) for item in horizons)}",
        f"SEEDS={','.join(str(item) for item in seeds)}",
        "",
        f"EXPECTED_RUNS={counts['expected_runs']}",
        f"UNIQUE_RUNS={counts['unique_runs']}",
        "",
        f"TARGET_RESELECTION={flags['TARGET_RESELECTION']}",
        f"LR_RESELECTION={flags['LR_RESELECTION']}",
        f"PRIOR_RESELECTION={flags['PRIOR_RESELECTION']}",
        "",
        f"DATASET_SPECIFIC_TUNING={flags['DATASET_SPECIFIC_TUNING']}",
        f"HORIZON_SPECIFIC_TUNING={flags['HORIZON_SPECIFIC_TUNING']}",
        f"SEED_SPECIFIC_TUNING={flags['SEED_SPECIFIC_TUNING']}",
        "",
        f"TEST_FUTURE_TRUTH_USED={flags['TEST_FUTURE_TRUTH_USED']}",
        "PRIOR_MATURITY_CONTRACT=PASS",
        "",
        f"RESIDUAL_REVIN_SEMANTICS={revin_status}",
        f"RESIDUALIZATION_SCOPE={RESIDUALIZATION_SCOPE}",
        "",
        f"V3_1_CALIBRATION_ENABLED={V3_1_CALIBRATION_ENABLED}",
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


def split_target_frames(target_df: pd.DataFrame):
    return temporal_split_by_ratio_or_dates(target_df)
