"""Metric-space repair for MS-KATL-Lite D4 A/B/C cross-target predictions.

Recomputes RMSE / sMAPE in original_sales_space from already-finished
normalized predictions. Does not retrain models, refit scalers, or overwrite
historical run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    temporal_split_by_ratio_or_dates,
)
from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE, SALES_VALUE_POLICY
from src.evaluation.metrics import inverse_transform_sales, smape
from src.transfer_methods.ms_katl_cross_target import (
    BL_METHODS,
    CROSS_TARGET_ARM_NAMES,
    D4_CROSS_TARGET_IDENTITIES,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_FORMAL_RELATIVE,
    FORMAL_MSML_METHOD,
    FORMAL_NO_TL_METHOD,
    NA,
    REFERENCE_MISSING,
    _beats,
    _csv_value,
    _delta,
    _fmt,
    _mean,
    _median,
    _relative_improvement,
    _win_record,
    decide_cross_target_verdict,
    load_baseline_references,
    load_formal_references,
    parse_d4_target_identity,
)

NORMALIZED_SPACE = "normalized_minmax_space"
SOURCE_RUN_NAME = "20260818_141311_897713_MS_KATL_D4_H1_S42_CROSS_TARGET"
ACCEPTED_FORMAL_GIT = "630ae4ef33a271187ed7225437d0609d8a13e941"
ARM_FILES = CROSS_TARGET_ARM_NAMES
SANITY_TARGET = "166/258"
SANITY_B_NORMALIZED_RMSE = 0.3371
SANITY_B_ORIGINAL_RMSE = 0.3708
SANITY_BL2_ORIGINAL_RMSE = 0.3452
AFFINE_RECOVERY_ATOL = 1e-5
ORIGINAL_RMSE_MATCH_ATOL = 1e-8
Y_TRUE_MATCH_ATOL = 1e-10

PREFLIGHT_TEXT = """================================================================================
MS-KATL D4 CROSS-TARGET METRIC-SPACE REPAIR PREFLIGHT
================================================================================

SOURCE_RUN=
outputs/runs/20260818_141311_897713_MS_KATL_D4_H1_S42_CROSS_TARGET

DATASET=D4
MODE=without
HORIZON=1
SEED=42

TARGETS=5
ARMS=3
PREDICTION_FILES_EXPECTED=15
PREDICTION_FILES_FOUND={prediction_files_found}

SOURCE_METRIC_SPACE=normalized
TARGET_METRIC_SPACE=original_sales_space

EXACT_TARGET_SCALERS_AVAILABLE={scalers_available}
SCALER_REFIT_REQUIRED=NO
MODEL_RETRAIN_REQUIRED=NO

EXTERNAL_REFERENCES_EXPECTED=35
EXTERNAL_REFERENCES_FOUND={external_references_found}

HISTORICAL_RESULTS_OVERWRITE=NO
ARCHITECTURE_CHANGE=NO
FORMAL_METHODS_CHANGE=NO
================================================================================
"""


class MetricSpaceMismatchError(ValueError):
    """Fail-closed guard for mixing normalized and original_sales_space metrics."""

    def __init__(
        self,
        *,
        target_identity: str,
        candidate_method: str,
        candidate_metric_space: object,
        reference_method: str,
        reference_metric_space: object,
    ) -> None:
        self.target_identity = target_identity
        self.candidate_method = candidate_method
        self.candidate_metric_space = candidate_metric_space
        self.reference_method = reference_method
        self.reference_metric_space = reference_metric_space
        super().__init__(
            "metric-space mismatch: cannot compare "
            f"target={target_identity} "
            f"candidate={candidate_method} space={candidate_metric_space} vs "
            f"reference={reference_method} space={reference_metric_space}"
        )


class MetricSpaceRepairError(RuntimeError):
    """Fail-closed repair error."""


class FrozenSalesInverseScaler:
    """Read-only sales min/max container compatible with inverse_transform_sales.

    Never supports fit / fit_transform. Stores only the sales-column inverse
    parameters recovered from the training-time identity.
    """

    def __init__(
        self,
        sales_min: float,
        sales_max: float,
        *,
        n_features: int = 1,
        sales_index: int = 0,
    ) -> None:
        data_min = np.zeros(int(n_features), dtype=np.float64)
        data_max = np.ones(int(n_features), dtype=np.float64)
        data_min[int(sales_index)] = float(sales_min)
        data_max[int(sales_index)] = float(sales_max)
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.sales_min = float(sales_min)
        self.sales_max = float(sales_max)
        self.sales_index = int(sales_index)

    def fit(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("metric-space repair forbids scaler.fit")

    def fit_transform(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("metric-space repair forbids scaler.fit_transform")


def source_run_dir(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "runs" / SOURCE_RUN_NAME


def repair_root(run_dir: Path) -> Path:
    return Path(run_dir) / "metric_space_repair"


def historical_protected_paths(run_dir: Path) -> List[Path]:
    root = Path(run_dir)
    paths = [
        root / "run_config.json",
        root / "results" / "d4_h1_s42_cross_target_results.csv",
        root / "results" / "d4_h1_s42_cross_target_summary.csv",
        root / "results" / "d4_h1_s42_cross_target_diagnostics.json",
    ]
    pred_root = root / "predictions"
    for identity in D4_CROSS_TARGET_IDENTITIES:
        dirname = parse_d4_target_identity(identity)["prediction_dirname"]
        for arm in ARM_FILES:
            paths.append(pred_root / dirname / f"{arm}.csv")
    return paths


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "mtime_ns": None, "size": None}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
        "size": len(data),
    }


def snapshot_historical_run(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {str(path): _file_fingerprint(path) for path in historical_protected_paths(run_dir)}


def assert_historical_run_untouched(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    for path, previous in before.items():
        current = after.get(path) or _file_fingerprint(Path(path))
        if previous.get("sha256") != current.get("sha256"):
            raise MetricSpaceRepairError(f"historical artifact overwritten: {path}")
        if previous.get("mtime_ns") != current.get("mtime_ns"):
            raise MetricSpaceRepairError(f"historical artifact rewritten: {path}")


def require_same_metric_space(
    *,
    target_identity: str,
    candidate_method: str,
    candidate_metric_space: object,
    reference_method: str,
    reference_metric_space: object,
) -> None:
    candidate = str(candidate_metric_space or "").strip()
    reference = str(reference_metric_space or "").strip()
    if candidate != reference:
        raise MetricSpaceMismatchError(
            target_identity=target_identity,
            candidate_method=candidate_method,
            candidate_metric_space=candidate_metric_space,
            reference_method=reference_method,
            reference_metric_space=reference_metric_space,
        )


def original_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise ValueError(f"RMSE arrays must match and be non-empty: {true.shape}, {pred.shape}")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def recover_sales_minmax_from_identity(
    y_normalized: np.ndarray,
    y_original: np.ndarray,
    *,
    atol: float = AFFINE_RECOVERY_ATOL,
) -> Tuple[float, float]:
    """Recover exact sales min/max from the training-time MinMax identity.

    This is algebraic inversion of
        y_original = y_normalized * (max - min) + min
    using paired values already produced by the training scaler.
    It does not call MinMaxScaler.fit / fit_transform.
    """
    y_norm = np.asarray(y_normalized, dtype=np.float64).reshape(-1)
    y_orig = np.asarray(y_original, dtype=np.float64).reshape(-1)
    if y_norm.shape != y_orig.shape or y_norm.size == 0:
        raise MetricSpaceRepairError(
            f"scaler identity recovery requires aligned non-empty arrays: {y_norm.shape}, {y_orig.shape}"
        )
    if not np.isfinite(y_norm).all() or not np.isfinite(y_orig).all():
        raise MetricSpaceRepairError("scaler identity recovery requires finite arrays")
    i = int(np.argmin(y_norm))
    j = int(np.argmax(y_norm))
    denom = float(y_norm[j] - y_norm[i])
    if abs(denom) < 1e-12:
        raise MetricSpaceRepairError("cannot recover target scaler: y_true_normalized has no range")
    span = float(y_orig[j] - y_orig[i]) / denom
    sales_min = float(y_orig[i] - y_norm[i] * span)
    sales_max = float(sales_min + span)
    reconstructed = y_norm * span + sales_min
    residual = float(np.max(np.abs(reconstructed - y_orig)))
    if residual > atol:
        raise MetricSpaceRepairError(
            f"INVERSE_TRANSFORM_REPRODUCIBLE=NO residual={residual} atol={atol}"
        )
    return sales_min, sales_max


def extract_sequence_label_dates(
    df: pd.DataFrame,
    *,
    horizon: int,
    window_size: int,
) -> List[str]:
    ordered = df.sort_values(["entity_id", "item_id", "date"], kind="mergesort").reset_index(drop=True)
    dates: List[str] = []
    for _key, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        g = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        group_dates = pd.to_datetime(g["date"], errors="raise")
        n = len(g)
        max_end = n - horizon
        for end_idx in range(window_size - 1, max_end):
            target_idx = end_idx + horizon
            if target_idx >= n:
                continue
            dates.append(pd.Timestamp(group_dates.iloc[target_idx]).strftime("%Y-%m-%d"))
    return dates


def load_raw_test_windows(
    target_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    horizon: int,
    window_size: int,
) -> Dict[str, Any]:
    _train_df, _val_df, test_df = temporal_split_by_ratio_or_dates(target_df)
    _x_test, y_test = build_tabular_sequence(
        test_df,
        horizon=horizon,
        window_size=window_size,
        feature_columns=list(feature_columns),
        reuse_context=None,
    )
    dates = extract_sequence_label_dates(test_df, horizon=horizon, window_size=window_size)
    y_original = np.asarray(y_test, dtype=np.float64).reshape(-1)
    if len(dates) != int(y_original.size):
        raise MetricSpaceRepairError(
            f"label date count {len(dates)} != raw y_test count {y_original.size}"
        )
    train_sales = np.asarray(_train_df["sales"], dtype=np.float64).reshape(-1)
    return {
        "y_true_original": y_original,
        "dates": dates,
        "train_sales_min": float(np.min(train_sales)) if train_sales.size else None,
        "train_sales_max": float(np.max(train_sales)) if train_sales.size else None,
        "n_test": int(y_original.size),
    }


def expected_prediction_files(run_dir: Path) -> List[Path]:
    paths = []
    pred_root = Path(run_dir) / "predictions"
    for identity in D4_CROSS_TARGET_IDENTITIES:
        dirname = parse_d4_target_identity(identity)["prediction_dirname"]
        for arm in ARM_FILES:
            paths.append(pred_root / dirname / f"{arm}.csv")
    return paths


def load_prediction_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MetricSpaceRepairError(f"prediction file missing: {path}")
    frame = pd.read_csv(path)
    if "y_true" not in frame.columns or "y_pred" not in frame.columns:
        raise MetricSpaceRepairError(f"prediction file missing y_true/y_pred: {path}")
    return frame


def load_diagnostics(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "results" / "d4_h1_s42_cross_target_diagnostics.json"
    if not path.exists():
        raise MetricSpaceRepairError(f"diagnostics missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_scaler_digest(diagnostics: Mapping[str, Any], target_identity: str, arm: str) -> str:
    split = diagnostics["targets"][target_identity]["arms"][arm].get("target_split_digest") or {}
    digest = split.get("target_scaler")
    if not digest:
        raise MetricSpaceRepairError(
            f"TARGET_SCALER_AVAILABLE=NO for {target_identity} {arm}"
        )
    return str(digest)


def _arm_split_digest(diagnostics: Mapping[str, Any], target_identity: str, arm: str) -> Dict[str, Any]:
    return dict(diagnostics["targets"][target_identity]["arms"][arm].get("target_split_digest") or {})


def _arm_original_scale_rmse(diagnostics: Mapping[str, Any], target_identity: str, arm: str) -> float:
    protocol = diagnostics["targets"][target_identity]["arms"][arm]["metric_protocol"]
    value = protocol.get("original_scale_rmse")
    if value is None:
        raise MetricSpaceRepairError(
            f"missing original_scale_rmse for {target_identity} {arm}"
        )
    return float(value)


def audit_target_abc_consistency(
    *,
    target_identity: str,
    prediction_frames: Mapping[str, pd.DataFrame],
    diagnostics: Mapping[str, Any],
) -> Dict[str, str]:
    digests = [_arm_scaler_digest(diagnostics, target_identity, arm) for arm in ARM_FILES]
    splits = [_arm_split_digest(diagnostics, target_identity, arm) for arm in ARM_FILES]
    y_true = [
        np.asarray(prediction_frames[arm]["y_true"], dtype=np.float64).reshape(-1)
        for arm in ARM_FILES
    ]
    scaler_ok = all(value == digests[0] for value in digests)
    split_ok = all(value == splits[0] for value in splits)
    y_ok = all(np.allclose(value, y_true[0], atol=Y_TRUE_MATCH_ATOL, rtol=0.0) for value in y_true)
    flags = {
        "A/B/C_TARGET_SCALER_MATCH": "YES" if scaler_ok else "NO",
        "A/B/C_TARGET_SPLIT_MATCH": "YES" if split_ok else "NO",
        "A/B/C_Y_TRUE_MATCH": "YES" if y_ok else "NO",
    }
    if any(flag != "YES" for flag in flags.values()):
        raise MetricSpaceRepairError(
            f"fail closed for {target_identity}: {flags}"
        )
    return flags


def count_external_references(
    *,
    project_root: Path,
    formal_csv: Optional[Path] = None,
    baseline_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    formal_path = Path(formal_csv) if formal_csv else Path(project_root) / DEFAULT_FORMAL_RELATIVE
    baseline_path = Path(baseline_csv) if baseline_csv else Path(project_root) / DEFAULT_BASELINE_RELATIVE
    found = 0
    missing: List[str] = []
    records = []
    for identity in D4_CROSS_TARGET_IDENTITIES:
        formal = load_formal_references(formal_path, target_identity=identity)
        baselines = load_baseline_references(baseline_path, target_identity=identity)
        items = [
            ("No-TL", formal["no_tl"]),
            ("MSML-TL", formal["msml"]),
        ]
        for key, method, _display in BL_METHODS:
            items.append((method, baselines[key]))
        for method, row in items:
            status = row.get("status")
            records.append({"target_identity": identity, "method": method, "status": status})
            if status == "FOUND" and row.get("rmse") is not None:
                found += 1
            else:
                missing.append(f"{identity} {method}")
    return {
        "expected": 35,
        "found": found,
        "missing": missing,
        "label": f"{found}/35",
        "formal_csv": str(formal_path),
        "baseline_csv": str(baseline_path),
        "records": records,
    }


def print_repair_preflight(
    *,
    prediction_files_found: int,
    scalers_available: str,
    external_references_found: int,
) -> str:
    text = PREFLIGHT_TEXT.format(
        prediction_files_found=prediction_files_found,
        scalers_available=scalers_available,
        external_references_found=external_references_found,
    )
    print(text, end="")
    return text


def _reference_metric_space(row: Mapping[str, Any]) -> str:
    return str(
        row.get("rmse_metric_space")
        or row.get("metric_space")
        or row.get("smape_metric_space")
        or ""
    ).strip()


def _guard_reference(
    *,
    target_identity: str,
    candidate_method: str,
    reference_row: Mapping[str, Any],
) -> None:
    if reference_row.get("status") != "FOUND":
        raise MetricSpaceRepairError(
            f"external reference missing for {target_identity} {reference_row.get('method')}"
        )
    actual = _reference_metric_space(reference_row)
    require_same_metric_space(
        target_identity=target_identity,
        candidate_method=candidate_method,
        candidate_metric_space=ORIGINAL_SALES_SPACE,
        reference_method=str(reference_row.get("method")),
        reference_metric_space=actual,
    )


def inverse_prediction_frame(
    frame: pd.DataFrame,
    *,
    scaler: FrozenSalesInverseScaler,
    feature_columns: Sequence[str],
    dates: Sequence[str],
) -> pd.DataFrame:
    y_true_norm = np.asarray(frame["y_true"], dtype=np.float64).reshape(-1)
    y_pred_norm = np.asarray(frame["y_pred"], dtype=np.float64).reshape(-1)
    y_true_orig = inverse_transform_sales(
        y_true_norm,
        sales_scaler=scaler,
        feature_columns=feature_columns,
        apply_sales_value_policy=True,
    )
    y_pred_orig = inverse_transform_sales(
        y_pred_norm,
        sales_scaler=scaler,
        feature_columns=feature_columns,
        apply_sales_value_policy=True,
    )
    extra_cols = [col for col in frame.columns if col not in {"y_true", "y_pred"}]
    out = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true_norm), dtype=int),
            "date": list(dates),
            "y_true_normalized": y_true_norm,
            "y_pred_normalized": y_pred_norm,
            "y_true_original": y_true_orig,
            "y_pred_original": y_pred_orig,
            "metric_space": ORIGINAL_SALES_SPACE,
        }
    )
    for col in extra_cols:
        out[col] = frame[col].to_numpy()
    return out


def repair_one_arm(
    *,
    target_identity: str,
    arm: str,
    prediction_frame: pd.DataFrame,
    scaler: FrozenSalesInverseScaler,
    feature_columns: Sequence[str],
    dates: Sequence[str],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    repaired = inverse_prediction_frame(
        prediction_frame,
        scaler=scaler,
        feature_columns=feature_columns,
        dates=dates,
    )
    y_true_norm = repaired["y_true_normalized"].to_numpy(dtype=np.float64)
    y_pred_norm = repaired["y_pred_normalized"].to_numpy(dtype=np.float64)
    y_true_orig = repaired["y_true_original"].to_numpy(dtype=np.float64)
    y_pred_orig = repaired["y_pred_original"].to_numpy(dtype=np.float64)
    rmse_norm = original_rmse(y_true_norm, y_pred_norm)
    smape_norm = float(smape(y_true_norm, y_pred_norm))
    rmse_orig = original_rmse(y_true_orig, y_pred_orig)
    smape_orig = float(smape(y_true_orig, y_pred_orig))
    expected_orig = _arm_original_scale_rmse(diagnostics, target_identity, arm)
    if not math.isclose(rmse_orig, expected_orig, rel_tol=0.0, abs_tol=1e-6):
        raise MetricSpaceRepairError(
            f"repaired original RMSE mismatch for {target_identity} {arm}: "
            f"got={rmse_orig} expected_from_diagnostics={expected_orig}"
        )
    if abs(rmse_orig - rmse_norm) < 1e-12 and abs(rmse_norm - SANITY_B_NORMALIZED_RMSE) < 1e-3:
        raise MetricSpaceRepairError(
            f"inverse-transform repair did not change RMSE for {target_identity} {arm}"
        )
    return {
        "target_identity": target_identity,
        "arm": arm,
        "repaired_frame": repaired,
        "rmse_normalized": rmse_norm,
        "smape_normalized": smape_norm,
        "rmse_original": rmse_orig,
        "smape_original": smape_orig,
        "rmse": rmse_orig,
        "smape": smape_orig,
        "comparison_rmse": rmse_orig,
        "comparison_smape": smape_orig,
        "comparison_metric_space": ORIGINAL_SALES_SPACE,
        "metric_space": ORIGINAL_SALES_SPACE,
        "y_true_original": y_true_orig,
        "y_pred_original": y_pred_orig,
        "y_true_normalized": y_true_norm,
        "y_pred_normalized": y_pred_norm,
        "target_scaler_digest": _arm_scaler_digest(diagnostics, target_identity, arm),
        "target_split_digest": _arm_split_digest(diagnostics, target_identity, arm),
        "sales_min": scaler.sales_min,
        "sales_max": scaler.sales_max,
        "n_samples": int(y_true_orig.size),
    }


def repair_one_target(
    *,
    target_identity: str,
    prediction_frames: Mapping[str, pd.DataFrame],
    raw_windows: Mapping[str, Any],
    feature_columns: Sequence[str],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    flags = audit_target_abc_consistency(
        target_identity=target_identity,
        prediction_frames=prediction_frames,
        diagnostics=diagnostics,
    )
    y_true_norm = np.asarray(prediction_frames["fused_target_only"]["y_true"], dtype=np.float64).reshape(-1)
    y_true_raw = np.asarray(raw_windows["y_true_original"], dtype=np.float64).reshape(-1)
    if y_true_norm.shape != y_true_raw.shape:
        raise MetricSpaceRepairError(
            f"{target_identity} eval sample identity mismatch: "
            f"prediction={y_true_norm.shape} raw_test={y_true_raw.shape}"
        )
    sales_min, sales_max = recover_sales_minmax_from_identity(y_true_norm, y_true_raw)
    train_min = raw_windows.get("train_sales_min")
    train_max = raw_windows.get("train_sales_max")
    if train_min is not None and train_max is not None:
        if not (
            math.isclose(sales_min, float(train_min), rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(sales_max, float(train_max), rel_tol=0.0, abs_tol=1e-6)
        ):
            raise MetricSpaceRepairError(
                f"{target_identity} recovered sales min/max {sales_min}/{sales_max} "
                f"!= train min/max {train_min}/{train_max}"
            )
    feature_list = [str(col) for col in feature_columns]
    if "sales" not in feature_list:
        raise MetricSpaceRepairError(f"{target_identity} feature_columns missing sales")
    scaler = FrozenSalesInverseScaler(
        sales_min,
        sales_max,
        n_features=len(feature_list),
        sales_index=feature_list.index("sales"),
    )
    dates = list(raw_windows["dates"])
    arm_results: Dict[str, Any] = {}
    for arm in ARM_FILES:
        arm_results[arm] = repair_one_arm(
            target_identity=target_identity,
            arm=arm,
            prediction_frame=prediction_frames[arm],
            scaler=scaler,
            feature_columns=feature_list,
            dates=dates,
            diagnostics=diagnostics,
        )
    y_true_orig_values = [arm_results[arm]["y_true_original"] for arm in ARM_FILES]
    if not all(
        np.allclose(value, y_true_orig_values[0], atol=Y_TRUE_MATCH_ATOL, rtol=0.0)
        for value in y_true_orig_values
    ):
        raise MetricSpaceRepairError(f"{target_identity} original y_true mismatch across A/B/C")
    init_effect = float(arm_results["raw_target_only"]["rmse_original"] - arm_results["fused_target_only"]["rmse_original"])
    transfer_effect = float(arm_results["fused_target_only"]["rmse_original"] - arm_results["full_current"]["rmse_original"])
    init_effect_norm = float(arm_results["raw_target_only"]["rmse_normalized"] - arm_results["fused_target_only"]["rmse_normalized"])
    transfer_effect_norm = float(arm_results["fused_target_only"]["rmse_normalized"] - arm_results["full_current"]["rmse_normalized"])
    return {
        "target_identity": target_identity,
        "flags": flags,
        "sales_min": sales_min,
        "sales_max": sales_max,
        "target_scaler_digest": arm_results["fused_target_only"]["target_scaler_digest"],
        "feature_columns": feature_list,
        "sales_index": feature_list.index("sales"),
        "arm_results": arm_results,
        "INIT_EFFECT_ORIGINAL": init_effect,
        "TRANSFER_EFFECT_ORIGINAL": transfer_effect,
        "INIT_EFFECT": init_effect_norm,
        "TRANSFER_EFFECT": transfer_effect_norm,
    }


def build_original_space_comparison(
    *,
    target_repair: Mapping[str, Any],
    formal_refs: Mapping[str, Mapping[str, Any]],
    baseline_refs: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = target_repair["target_identity"]
    arms = target_repair["arm_results"]
    b = arms["fused_target_only"]
    b_rmse = float(b["rmse_original"])
    no_tl = formal_refs["no_tl"]
    msml = formal_refs["msml"]
    _guard_reference(target_identity=identity, candidate_method="B_FUSED_TARGET_ONLY", reference_row=no_tl)
    _guard_reference(target_identity=identity, candidate_method="B_FUSED_TARGET_ONLY", reference_row=msml)
    for key, method, _display in BL_METHODS:
        _guard_reference(
            target_identity=identity,
            candidate_method="B_FUSED_TARGET_ONLY",
            reference_row=baseline_refs[key],
        )
    best_bl = baseline_refs["best"]
    parsed = parse_d4_target_identity(identity)
    comparison = {
        "target_identity": parsed["target_identity"],
        "store_id": parsed["store_id"],
        "product_id": parsed["product_id"],
        "entity_key": parsed["entity_key"],
        "metric_space": ORIGINAL_SALES_SPACE,
        "comparison_metric_space": ORIGINAL_SALES_SPACE,
        "arm_results": arms,
        "flags": target_repair["flags"],
        "effects": {
            "INIT_EFFECT_ORIGINAL": target_repair["INIT_EFFECT_ORIGINAL"],
            "TRANSFER_EFFECT_ORIGINAL": target_repair["TRANSFER_EFFECT_ORIGINAL"],
            "INIT_EFFECT": target_repair["INIT_EFFECT"],
            "TRANSFER_EFFECT": target_repair["TRANSFER_EFFECT"],
        },
        "formal": formal_refs,
        "baselines": baseline_refs,
        "A_rmse": float(arms["full_current"]["rmse_original"]),
        "A_smape": float(arms["full_current"]["smape_original"]),
        "A_rmse_normalized": float(arms["full_current"]["rmse_normalized"]),
        "A_smape_normalized": float(arms["full_current"]["smape_normalized"]),
        "B_rmse": b_rmse,
        "B_smape": float(b["smape_original"]),
        "B_rmse_normalized": float(b["rmse_normalized"]),
        "B_smape_normalized": float(b["smape_normalized"]),
        "C_rmse": float(arms["raw_target_only"]["rmse_original"]),
        "C_smape": float(arms["raw_target_only"]["smape_original"]),
        "C_rmse_normalized": float(arms["raw_target_only"]["rmse_normalized"]),
        "C_smape_normalized": float(arms["raw_target_only"]["smape_normalized"]),
        "no_tl_rmse": no_tl.get("rmse"),
        "no_tl_smape": no_tl.get("smape"),
        "no_tl_status": no_tl.get("status"),
        "msml_rmse": msml.get("rmse"),
        "msml_smape": msml.get("smape"),
        "msml_status": msml.get("status"),
        "best_bl_method": best_bl.get("method"),
        "best_bl_rmse": best_bl.get("rmse"),
        "best_bl_smape": best_bl.get("smape"),
        "best_bl_status": best_bl.get("status"),
        "B_beats_A": bool(b_rmse < float(arms["full_current"]["rmse_original"])),
        "B_beats_C": bool(b_rmse < float(arms["raw_target_only"]["rmse_original"])),
        "B_beats_A_normalized": bool(float(b["rmse_normalized"]) < float(arms["full_current"]["rmse_normalized"])),
        "B_beats_C_normalized": bool(float(b["rmse_normalized"]) < float(arms["raw_target_only"]["rmse_normalized"])),
        "B_beats_NoTL": _beats(b_rmse, no_tl.get("rmse")),
        "B_beats_MSML": _beats(b_rmse, msml.get("rmse")),
        "B_beats_BL4": _beats(b_rmse, baseline_refs["bl4"].get("rmse")),
        "B_beats_best_BL": _beats(b_rmse, best_bl.get("rmse")),
        "relative_improvement_vs_no_tl": _relative_improvement(no_tl.get("rmse"), b_rmse),
        "relative_improvement_vs_msml": _relative_improvement(msml.get("rmse"), b_rmse),
        "relative_improvement_vs_bl4": _relative_improvement(baseline_refs["bl4"].get("rmse"), b_rmse),
        "relative_improvement_vs_best_bl": _relative_improvement(best_bl.get("rmse"), b_rmse),
        "delta_rmse_vs_no_tl": _delta(b_rmse, no_tl.get("rmse")),
        "delta_rmse_vs_msml": _delta(b_rmse, msml.get("rmse")),
        "delta_rmse_vs_best_bl": _delta(b_rmse, best_bl.get("rmse")),
        "sales_min": target_repair["sales_min"],
        "sales_max": target_repair["sales_max"],
        "target_scaler_digest": target_repair["target_scaler_digest"],
    }
    for key, _method, _display in BL_METHODS:
        comparison[f"{key}_rmse"] = baseline_refs[key].get("rmse")
        comparison[f"{key}_smape"] = baseline_refs[key].get("smape")
        comparison[f"{key}_status"] = baseline_refs[key].get("status")
        comparison[f"delta_rmse_vs_{key}"] = _delta(b_rmse, baseline_refs[key].get("rmse"))
        comparison[f"B_beats_{key.upper()}"] = _beats(b_rmse, baseline_refs[key].get("rmse"))
    return comparison


def aggregate_original_space(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(comparisons) != 5:
        raise ValueError(f"aggregate summary requires 5 targets, got {len(comparisons)}")
    init_effects = [item["effects"]["INIT_EFFECT_ORIGINAL"] for item in comparisons]
    transfer_effects = [item["effects"]["TRANSFER_EFFECT_ORIGINAL"] for item in comparisons]
    return {
        "B_vs_A": _win_record([item["B_beats_A"] for item in comparisons]),
        "B_vs_C": _win_record([item["B_beats_C"] for item in comparisons]),
        "B_vs_A_normalized": _win_record([item["B_beats_A_normalized"] for item in comparisons]),
        "B_vs_C_normalized": _win_record([item["B_beats_C_normalized"] for item in comparisons]),
        "B_vs_NoTL": _win_record([item["B_beats_NoTL"] for item in comparisons]),
        "B_vs_MSML": _win_record([item["B_beats_MSML"] for item in comparisons]),
        "B_vs_BL1": _win_record([item["B_beats_BL1"] for item in comparisons]),
        "B_vs_BL2": _win_record([item["B_beats_BL2"] for item in comparisons]),
        "B_vs_BL3": _win_record([item["B_beats_BL3"] for item in comparisons]),
        "B_vs_BL4": _win_record([item["B_beats_BL4"] for item in comparisons]),
        "B_vs_BL5": _win_record([item["B_beats_BL5"] for item in comparisons]),
        "B_vs_bestBL": _win_record([item["B_beats_best_BL"] for item in comparisons]),
        "median_init_effect": _median(init_effects),
        "mean_init_effect": _mean(init_effects),
        "median_transfer_effect": _median(transfer_effects),
        "mean_transfer_effect": _mean(transfer_effects),
        "median_B_delta_vs_NoTL": _median([item["delta_rmse_vs_no_tl"] for item in comparisons]),
        "mean_B_delta_vs_NoTL": _mean([item["delta_rmse_vs_no_tl"] for item in comparisons]),
        "median_B_delta_vs_MSML": _median([item["delta_rmse_vs_msml"] for item in comparisons]),
        "mean_B_delta_vs_MSML": _mean([item["delta_rmse_vs_msml"] for item in comparisons]),
        "median_B_delta_vs_BL4": _median([item["delta_rmse_vs_bl4"] for item in comparisons]),
        "mean_B_delta_vs_BL4": _mean([item["delta_rmse_vs_bl4"] for item in comparisons]),
        "median_B_delta_vs_bestBL": _median([item["delta_rmse_vs_best_bl"] for item in comparisons]),
        "mean_B_delta_vs_bestBL": _mean([item["delta_rmse_vs_best_bl"] for item in comparisons]),
        "median_relative_improvement_vs_no_tl": _median(
            [item["relative_improvement_vs_no_tl"] for item in comparisons]
        ),
        "mean_relative_improvement_vs_no_tl": _mean(
            [item["relative_improvement_vs_no_tl"] for item in comparisons]
        ),
        "median_relative_improvement_vs_msml": _median(
            [item["relative_improvement_vs_msml"] for item in comparisons]
        ),
        "mean_relative_improvement_vs_msml": _mean(
            [item["relative_improvement_vs_msml"] for item in comparisons]
        ),
        "median_relative_improvement_vs_bl4": _median(
            [item["relative_improvement_vs_bl4"] for item in comparisons]
        ),
        "mean_relative_improvement_vs_bl4": _mean(
            [item["relative_improvement_vs_bl4"] for item in comparisons]
        ),
        "median_relative_improvement_vs_best_bl": _median(
            [item["relative_improvement_vs_best_bl"] for item in comparisons]
        ),
        "mean_relative_improvement_vs_best_bl": _mean(
            [item["relative_improvement_vs_best_bl"] for item in comparisons]
        ),
    }


def sanity_check_166_258(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    match = next((item for item in comparisons if item["target_identity"] == SANITY_TARGET), None)
    if match is None:
        raise MetricSpaceRepairError("166/258 comparison missing")
    b_norm = float(match["B_rmse_normalized"])
    b_orig = float(match["B_rmse"])
    bl2 = float(match["bl2_rmse"])
    if math.isclose(b_orig, b_norm, rel_tol=0.0, abs_tol=1e-6):
        raise MetricSpaceRepairError("166/258 original RMSE still equals normalized RMSE; repair failed")
    if not math.isclose(b_norm, SANITY_B_NORMALIZED_RMSE, rel_tol=0.0, abs_tol=5e-4):
        raise MetricSpaceRepairError(
            f"166/258 B normalized RMSE={b_norm} expected≈{SANITY_B_NORMALIZED_RMSE}"
        )
    if not math.isclose(b_orig, SANITY_B_ORIGINAL_RMSE, rel_tol=0.0, abs_tol=5e-4):
        raise MetricSpaceRepairError(
            f"166/258 B original RMSE={b_orig} expected≈{SANITY_B_ORIGINAL_RMSE}"
        )
    if not math.isclose(bl2, SANITY_BL2_ORIGINAL_RMSE, rel_tol=0.0, abs_tol=5e-4):
        raise MetricSpaceRepairError(
            f"166/258 BL2 original RMSE={bl2} expected={SANITY_BL2_ORIGINAL_RMSE}"
        )
    beats_bl2 = bool(b_orig < bl2)
    if beats_bl2:
        raise MetricSpaceRepairError("166/258 B original RMSE incorrectly beats BL2")
    return {
        "B_normalized_rmse": b_norm,
        "B_original_rmse": b_orig,
        "BL2_original_rmse": bl2,
        "B_beats_BL2": False,
        "B_beats_best_BL": bool(match["B_beats_best_BL"]),
    }


def format_original_space_tables(
    comparisons: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    verdict: str,
) -> str:
    lines = [
        "==============================================================================================",
        "D4 H1 S42 CROSS-TARGET — ORIGINAL SALES SPACE",
        "==============================================================================================",
        "",
        f"{'TARGET':<11}{'A_FULL':<10}{'B_FUSED':<10}{'C_RAW':<10}{'NoTL':<10}{'MSML':<10}{'BL1':<10}{'BL2':<10}{'BL3':<10}{'BL4':<10}{'BL5':<10}{'BEST_BL'}",
        "----------------------------------------------------------------------------------------------------------------------",
    ]
    for item in comparisons:
        best_label = str(item.get("best_bl_method") or "")
        if best_label.startswith("BL") and "_" in best_label:
            best_label = best_label.split("_", 1)[0]
        lines.append(
            f"{item['target_identity']:<11}"
            f"{_fmt(item['A_rmse']):<10}"
            f"{_fmt(item['B_rmse']):<10}"
            f"{_fmt(item['C_rmse']):<10}"
            f"{_fmt(item['no_tl_rmse']):<10}"
            f"{_fmt(item['msml_rmse']):<10}"
            f"{_fmt(item['bl1_rmse']):<10}"
            f"{_fmt(item['bl2_rmse']):<10}"
            f"{_fmt(item['bl3_rmse']):<10}"
            f"{_fmt(item['bl4_rmse']):<10}"
            f"{_fmt(item['bl5_rmse']):<10}"
            f"{best_label}"
        )
    lines.extend(
        [
            "----------------------------------------------------------------------------------------------------------------------",
            "METRIC_SPACE=original_sales_space",
            "==============================================================================================",
            "",
            "================================================================================",
            "ORIGINAL-SPACE MECHANISM EFFECTS",
            "================================================================================",
            "",
            f"{'TARGET':<13}{'INIT_EFFECT_ORIG':<22}{'TRANSFER_EFFECT_ORIG'}",
            "-------------------------------------------------------",
        ]
    )
    for item in comparisons:
        lines.append(
            f"{item['target_identity']:<13}"
            f"{_fmt(item['effects']['INIT_EFFECT_ORIGINAL'], 6):<22}"
            f"{_fmt(item['effects']['TRANSFER_EFFECT_ORIGINAL'], 6)}"
        )
    lines.extend(
        [
            "-------------------------------------------------------",
            "",
            f"MEDIAN INIT_EFFECT_ORIG        {_fmt(aggregates['median_init_effect'], 6)}",
            f"MEAN   INIT_EFFECT_ORIG        {_fmt(aggregates['mean_init_effect'], 6)}",
            "",
            f"MEDIAN TRANSFER_EFFECT_ORIG    {_fmt(aggregates['median_transfer_effect'], 6)}",
            f"MEAN   TRANSFER_EFFECT_ORIG    {_fmt(aggregates['mean_transfer_effect'], 6)}",
            "================================================================================",
            "",
            "================================================================================",
            "B FUSED_TARGET_ONLY — ORIGINAL-SPACE WIN RATES",
            "================================================================================",
            "",
            f"B vs A             {aggregates['B_vs_A']['label']}",
            f"B vs C             {aggregates['B_vs_C']['label']}",
            f"B vs No-TL         {aggregates['B_vs_NoTL']['label']}",
            f"B vs MSML-TL       {aggregates['B_vs_MSML']['label']}",
            f"B vs BL4           {aggregates['B_vs_BL4']['label']}",
            f"B vs BEST BL       {aggregates['B_vs_bestBL']['label']}",
            "",
            f"B vs BL1           {aggregates['B_vs_BL1']['label']}",
            f"B vs BL2           {aggregates['B_vs_BL2']['label']}",
            f"B vs BL3           {aggregates['B_vs_BL3']['label']}",
            f"B vs BL5           {aggregates['B_vs_BL5']['label']}",
            "",
            f"B vs A (normalized diagnostic) {aggregates['B_vs_A_normalized']['label']}",
            f"B vs C (normalized diagnostic) {aggregates['B_vs_C_normalized']['label']}",
            "",
            f"VERDICT={verdict}",
            "================================================================================",
        ]
    )
    return "\n".join(lines)


def write_repair_outputs(
    *,
    run_dir: Path,
    comparisons: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    audit: Mapping[str, Any],
    verdict: str,
) -> Dict[str, Path]:
    root = repair_root(run_dir)
    pred_root = root / "predictions_original"
    results_dir = root / "results"
    pred_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    summary_rows = []
    for comparison in comparisons:
        parsed = parse_d4_target_identity(comparison["target_identity"])
        arm_dir = pred_root / parsed["prediction_dirname"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        for arm in ARM_FILES:
            arm_result = comparison["arm_results"][arm]
            frame = arm_result["repaired_frame"]
            frame.to_csv(arm_dir / f"{arm}.csv", index=False, encoding="utf-8")
            result_rows.append(
                {
                    "dataset": "D4",
                    "target_identity": comparison["target_identity"],
                    "store_id": comparison["store_id"],
                    "product_id": comparison["product_id"],
                    "mode": "without",
                    "horizon": 1,
                    "seed": 42,
                    "arm": arm,
                    "rmse": arm_result["rmse_original"],
                    "smape": arm_result["smape_original"],
                    "rmse_original": arm_result["rmse_original"],
                    "smape_original": arm_result["smape_original"],
                    "rmse_normalized": arm_result["rmse_normalized"],
                    "smape_normalized": arm_result["smape_normalized"],
                    "comparison_rmse": arm_result["rmse_original"],
                    "comparison_smape": arm_result["smape_original"],
                    "comparison_metric_space": ORIGINAL_SALES_SPACE,
                    "metric_space": ORIGINAL_SALES_SPACE,
                    "init_effect_original": comparison["effects"]["INIT_EFFECT_ORIGINAL"] if arm == "fused_target_only" else NA,
                    "transfer_effect_original": comparison["effects"]["TRANSFER_EFFECT_ORIGINAL"] if arm == "fused_target_only" else NA,
                    "target_scaler_digest": comparison["target_scaler_digest"],
                    "sales_min": comparison["sales_min"],
                    "sales_max": comparison["sales_max"],
                }
            )
        summary_rows.append(
            {
                "target_identity": comparison["target_identity"],
                "metric_space": ORIGINAL_SALES_SPACE,
                "A_rmse_original": comparison["A_rmse"],
                "B_rmse_original": comparison["B_rmse"],
                "C_rmse_original": comparison["C_rmse"],
                "A_rmse_normalized": comparison["A_rmse_normalized"],
                "B_rmse_normalized": comparison["B_rmse_normalized"],
                "C_rmse_normalized": comparison["C_rmse_normalized"],
                "A_smape_original": comparison["A_smape"],
                "B_smape_original": comparison["B_smape"],
                "C_smape_original": comparison["C_smape"],
                "formal_no_tl_rmse": comparison["no_tl_rmse"],
                "msml_rmse": comparison["msml_rmse"],
                "bl1_rmse": comparison["bl1_rmse"],
                "bl2_rmse": comparison["bl2_rmse"],
                "bl3_rmse": comparison["bl3_rmse"],
                "bl4_rmse": comparison["bl4_rmse"],
                "bl5_rmse": comparison["bl5_rmse"],
                "best_bl_method": comparison["best_bl_method"],
                "best_bl_rmse": comparison["best_bl_rmse"],
                "init_effect_original": comparison["effects"]["INIT_EFFECT_ORIGINAL"],
                "transfer_effect_original": comparison["effects"]["TRANSFER_EFFECT_ORIGINAL"],
                "B_beats_A": _csv_value(comparison["B_beats_A"]),
                "B_beats_C": _csv_value(comparison["B_beats_C"]),
                "B_beats_NoTL": _csv_value(comparison["B_beats_NoTL"]),
                "B_beats_MSML": _csv_value(comparison["B_beats_MSML"]),
                "B_beats_BL4": _csv_value(comparison["B_beats_BL4"]),
                "B_beats_best_BL": _csv_value(comparison["B_beats_best_BL"]),
                "delta_rmse_vs_no_tl_original": comparison["delta_rmse_vs_no_tl"],
                "delta_rmse_vs_msml_original": comparison["delta_rmse_vs_msml"],
                "delta_rmse_vs_bl4_original": comparison["delta_rmse_vs_bl4"],
                "delta_rmse_vs_best_bl_original": comparison["delta_rmse_vs_best_bl"],
                "relative_improvement_vs_no_tl": comparison["relative_improvement_vs_no_tl"],
                "relative_improvement_vs_msml": comparison["relative_improvement_vs_msml"],
                "relative_improvement_vs_bl4": comparison["relative_improvement_vs_bl4"],
                "relative_improvement_vs_best_bl": comparison["relative_improvement_vs_best_bl"],
            }
        )

    results_csv = results_dir / "d4_h1_s42_cross_target_original_results.csv"
    summary_csv = results_dir / "d4_h1_s42_cross_target_original_summary.csv"
    audit_json = results_dir / "metric_space_repair_audit.json"
    pd.DataFrame(result_rows).to_csv(results_csv, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8")
    payload = dict(audit)
    payload["verdict"] = verdict
    payload["aggregates"] = json.loads(json.dumps(aggregates, default=str))
    audit_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tables_path = results_dir / "original_space_tables.txt"
    tables_path.write_text(
        format_original_space_tables(comparisons, aggregates, verdict=verdict) + "\n",
        encoding="utf-8",
    )
    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "audit_json": audit_json,
        "tables_txt": tables_path,
        "predictions_dir": pred_root,
    }
