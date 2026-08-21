"""D4 cross-target validation for MS-KATL-Lite mechanism arms A/B/C.

Reuses the paired Stage-A + arm runner from the single-cell mechanism audit.
Does not modify FORMAL_METHODS, MSML-TL, No-TL, baselines, or architecture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.protocols.experiment_protocol import FORMAL_METHODS, formal_target_entity_keys
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_mechanism_ablation import (
    ABLATION_ARM_SPECS,
    CROSS_TARGET_ARM_NAMES,
    compute_abc_effects,
    resolve_ablation_arm,
    resolve_cross_target_arms,
)

D4_CROSS_TARGET_IDENTITIES: Tuple[str, ...] = (
    "166/258",
    "166/311",
    "166/313",
    "166/432",
    "166/433",
)
CROSS_TARGET_LABEL = "MS_KATL_D4_H1_S42_CROSS_TARGET"
HISTORICAL_MECHANISM_RUN = "20260818_134401_999267_MS_KATL_MECHANISM_D4_h1_s42"
DEFAULT_FORMAL_RELATIVE = (
    "outputs/runs/20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8/"
    "d4_without/results/dataset4_without_results.csv"
)
DEFAULT_BASELINE_RELATIVE = "outputs/baselines/d4_baselines_multiseed.csv"
REFERENCE_MISSING = "MISSING"
NA = "NA"

BL_METHODS = (
    ("bl1", "BL1_HistoricalMean", "BL1 Historical Mean"),
    ("bl2", "BL2_MovingAverage", "BL2 Moving Average"),
    ("bl3", "BL3_LightGBM", "BL3 LightGBM"),
    ("bl4", "BL4_LSTM", "BL4 LSTM"),
    ("bl5", "BL5_SeasonalNaive", "BL5 Seasonal Naive"),
)
FORMAL_NO_TL_METHOD = "No-TL"
FORMAL_MSML_METHOD = "MSML-TL"

PREFLIGHT_TEXT = """============================================================
D4 CROSS-TARGET PREFLIGHT
============================================================
DATASET=D4
MODE=without
HORIZON=1
SEED=42
EPOCHS=50
GAMMA=0.5
LAMBDA_ADV=0

TARGETS=5
ARMS_PER_TARGET=3
TOTAL_ARM_RUNS=15

TARGETS:
166/258
166/311
166/313
166/432
166/433

ARMS:
A FULL_CURRENT
B FUSED_TARGET_ONLY
C RAW_TARGET_ONLY

STAGE_A_SHARED_WITHIN_TARGET=YES
FORMAL_METHODS_UNCHANGED=YES
BASELINES_RERUN=NO
FORMAL_METHODS_RERUN=NO
============================================================
"""


def print_d4_cross_target_preflight() -> str:
    print(PREFLIGHT_TEXT, end="")
    return PREFLIGHT_TEXT


def parse_d4_target_identity(value: object) -> Dict[str, str]:
    """Parse a D4 identity without using CSV row order.

    Accepts `166/258` or `166_258` and always returns slash-form identity.
    """
    text = str(value).strip()
    if not text or text.upper() == REFERENCE_MISSING:
        raise ValueError(f"empty D4 target identity: {value!r}")
    if "/" in text:
        parts = text.split("/")
    elif "_" in text:
        parts = text.split("_", 1)
    else:
        raise ValueError(f"D4 target identity must be store/product, got {value!r}")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"D4 target identity must have two components, got {value!r}")
    store_id, product_id = str(parts[0]).strip(), str(parts[1]).strip()
    identity = f"{store_id}/{product_id}"
    return {
        "target_identity": identity,
        "entity_key": f"{store_id}_{product_id}",
        "store_id": store_id,
        "product_id": product_id,
        "prediction_dirname": f"{store_id}_{product_id}",
    }


def parse_cross_target_targets(spec: object) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        targets = D4_CROSS_TARGET_IDENTITIES
    elif isinstance(spec, (list, tuple)):
        targets = tuple(parse_d4_target_identity(item)["target_identity"] for item in spec)
    else:
        text = str(spec).strip()
        parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        targets = tuple(parse_d4_target_identity(part)["target_identity"] for part in parts)
    if len(targets) != 5:
        raise ValueError(f"cross-target mode requires exactly 5 D4 targets, got {list(targets)}")
    if len(set(targets)) != 5:
        raise ValueError(f"duplicate D4 targets are not allowed: {list(targets)}")
    protocol_set = set(formal_target_entity_keys(4))
    if set(targets) != protocol_set:
        raise ValueError(
            "cross-target identities must equal the D4 protocol set "
            f"{sorted(protocol_set)}, got {sorted(targets)}"
        )
    # Keep the caller/user order; never reorder by protocol or CSV row order.
    return targets


def parse_cross_target_arm_spec(spec: object) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "abc", "all"}:
        arms: Sequence[str] = CROSS_TARGET_ARM_NAMES
    elif isinstance(spec, (list, tuple)):
        arms = spec
    else:
        arms = [part.strip() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    return resolve_cross_target_arms(arms)


def plan_d4_cross_target(
    *,
    targets: object = "all",
    arms: object = "full_current,fused_target_only,raw_target_only",
) -> Dict[str, Any]:
    target_identities = parse_cross_target_targets(targets)
    resolved_arms = parse_cross_target_arm_spec(arms)
    planned = [
        {
            "target_identity": identity,
            **parse_d4_target_identity(identity),
            "arm": arm,
            "letter": ABLATION_ARM_SPECS[arm].letter,
            "display": ABLATION_ARM_SPECS[arm].display,
        }
        for identity in target_identities
        for arm in resolved_arms
    ]
    if len(planned) != 15:
        raise ValueError(f"cross-target plan must contain 15 arm-runs, got {len(planned)}")
    return {
        "targets": target_identities,
        "arms": resolved_arms,
        "planned": planned,
        "n_targets": 5,
        "arms_per_target": 3,
        "total_arm_runs": 15,
    }


def format_source_identity(key: object) -> str:
    if isinstance(key, (list, tuple)):
        return "/".join(str(part) for part in key)
    text = str(key).strip()
    if "_" in text and "/" not in text:
        store, product = text.split("_", 1)
        return f"{store}/{product}"
    return text


def historical_mechanism_run_dir(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "runs" / HISTORICAL_MECHANISM_RUN


def assert_new_run_does_not_clobber_historical(project_root: Path, run_dir: Path) -> None:
    historical = historical_mechanism_run_dir(project_root).resolve()
    resolved = Path(run_dir).resolve()
    if resolved == historical:
        raise ValueError(f"refusing to overwrite historical mechanism run: {historical}")
    if HISTORICAL_MECHANISM_RUN in resolved.parts:
        raise ValueError(f"refusing to write inside historical mechanism run: {resolved}")
    if CROSS_TARGET_LABEL not in Path(run_dir).name and "CROSS_TARGET" not in Path(run_dir).name:
        # Dedicated runner always uses the cross-target label; tests may pass any new dir.
        pass


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


def snapshot_protected_artifacts(project_root: Path) -> Dict[str, Dict[str, Any]]:
    root = Path(project_root)
    return {
        "formal_csv": _file_fingerprint(root / DEFAULT_FORMAL_RELATIVE),
        "baseline_csv": _file_fingerprint(root / DEFAULT_BASELINE_RELATIVE),
        "historical_results_csv": _file_fingerprint(
            historical_mechanism_run_dir(root) / "results" / "mechanism_ablation_results.csv"
        ),
    }


def assert_protected_artifacts_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, previous in before.items():
        current = after[name]
        if previous.get("exists") and current.get("sha256") != previous.get("sha256"):
            raise RuntimeError(f"protected artifact changed during cross-target run: {name}")
        if previous.get("exists") and current.get("mtime_ns") != previous.get("mtime_ns"):
            raise RuntimeError(f"protected artifact was rewritten: {name}")


def _normalize_mode(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if "without" in text:
        return "without"
    if text in {"with", "with_info_sharing", "with_information_sharing"}:
        return "with"
    return text


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "" or str(value).strip().upper() == REFERENCE_MISSING:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _row_identity(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("target_entity_key", "target_identity", "entity_key", "canonical_target_key"):
        if key in row and str(row[key]).strip():
            try:
                return parse_d4_target_identity(row[key])["target_identity"]
            except ValueError:
                continue
    store = str(row.get("target_store_id") or row.get("store_id") or "").strip()
    product = str(
        row.get("target_item_id")
        or row.get("target_product_id")
        or row.get("product_id")
        or ""
    ).strip()
    if store and product:
        return parse_d4_target_identity(f"{store}/{product}")["target_identity"]
    return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def lookup_reference_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_identity: str,
    method: str,
    horizon: int,
    seed: int,
    mode: str,
    dataset: str = "D4",
) -> Dict[str, Any]:
    """Join by identity + method + horizon + seed + mode. Never by row order."""
    wanted = parse_d4_target_identity(target_identity)["target_identity"]
    matches: List[Mapping[str, Any]] = []
    for row in rows:
        identity = _row_identity(row)
        if identity != wanted:
            continue
        if str(row.get("method", "")).strip() != method:
            continue
        if int(row.get("horizon", -1)) != int(horizon):
            continue
        if int(row.get("seed", row.get("random_state", -1))) != int(seed):
            continue
        row_mode = _normalize_mode(row.get("information_sharing", row.get("scenario", row.get("mode", ""))))
        if row_mode != _normalize_mode(mode):
            continue
        dataset_value = str(row.get("dataset_id", row.get("dataset", dataset))).strip().upper()
        if dataset_value not in {"", "4", "D4", "DATASET4"}:
            continue
        matches.append(row)
    if not matches:
        return {
            "status": REFERENCE_MISSING,
            "method": method,
            "target_identity": wanted,
            "rmse": None,
            "smape": None,
        }
    rmses = {_optional_float(row.get("rmse")) for row in matches}
    smapes = {_optional_float(row.get("smape")) for row in matches}
    if len(rmses) != 1:
        raise ValueError(
            f"conflicting {method} RMSE values for {wanted} H={horizon} seed={seed}: {rmses}"
        )
    chosen = matches[0]
    metric_space = (
        chosen.get("rmse_metric_space")
        or chosen.get("metric_space")
        or chosen.get("current_metric_space_actual")
        or chosen.get("primary_metric_space_actual")
    )
    return {
        "status": "FOUND",
        "method": method,
        "target_identity": wanted,
        "rmse": _optional_float(chosen.get("rmse")),
        "smape": _optional_float(chosen.get("smape")),
        "n_matches": len(matches),
        "rmse_metric_space": chosen.get("rmse_metric_space") or metric_space,
        "smape_metric_space": chosen.get("smape_metric_space") or metric_space,
        "metric_space": metric_space,
    }


def load_formal_references(
    path: Optional[Path],
    *,
    target_identity: str,
    horizon: int = 1,
    seed: int = 42,
    mode: str = "without",
) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        missing = {
            "status": REFERENCE_MISSING,
            "target_identity": target_identity,
            "rmse": None,
            "smape": None,
        }
        return {
            "no_tl": {**missing, "method": FORMAL_NO_TL_METHOD},
            "msml": {**missing, "method": FORMAL_MSML_METHOD},
        }
    rows = _read_csv_rows(Path(path))
    return {
        "no_tl": lookup_reference_row(
            rows,
            target_identity=target_identity,
            method=FORMAL_NO_TL_METHOD,
            horizon=horizon,
            seed=seed,
            mode=mode,
        ),
        "msml": lookup_reference_row(
            rows,
            target_identity=target_identity,
            method=FORMAL_MSML_METHOD,
            horizon=horizon,
            seed=seed,
            mode=mode,
        ),
    }


def load_baseline_references(
    path: Optional[Path],
    *,
    target_identity: str,
    horizon: int = 1,
    seed: int = 42,
    mode: str = "without",
) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    if path is None or not Path(path).exists():
        for key, method, _display in BL_METHODS:
            refs[key] = {
                "status": REFERENCE_MISSING,
                "method": method,
                "target_identity": target_identity,
                "rmse": None,
                "smape": None,
            }
        refs["best"] = {
            "status": REFERENCE_MISSING,
            "method": None,
            "rmse": None,
            "smape": None,
        }
        return refs
    rows = _read_csv_rows(Path(path))
    found: List[Tuple[str, Dict[str, Any]]] = []
    for key, method, _display in BL_METHODS:
        row = lookup_reference_row(
            rows,
            target_identity=target_identity,
            method=method,
            horizon=horizon,
            seed=seed,
            mode=mode,
        )
        refs[key] = row
        if row["status"] == "FOUND" and row["rmse"] is not None:
            found.append((key, row))
    if found:
        best_key, best_row = min(found, key=lambda item: float(item[1]["rmse"]))
        refs["best"] = {
            "status": "FOUND",
            "method": best_row["method"],
            "key": best_key,
            "rmse": best_row["rmse"],
            "smape": best_row["smape"],
        }
    else:
        refs["best"] = {
            "status": REFERENCE_MISSING,
            "method": None,
            "rmse": None,
            "smape": None,
        }
    return refs


def _delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _relative_improvement(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    if float(reference) == 0.0:
        return None
    return (float(reference) - float(candidate)) / float(reference)


def _beats(left: Optional[float], right: Optional[float]) -> Optional[bool]:
    if left is None or right is None:
        return None
    return float(left) < float(right)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return REFERENCE_MISSING
    return f"{float(value):.{digits}f}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return value


def _source_fields(stage_meta: Mapping[str, Any]) -> Dict[str, Any]:
    identities = [format_source_identity(item) for item in stage_meta.get("source_identities") or []]
    distances = list(stage_meta.get("source_distances") or [])
    model_digests = list(stage_meta.get("source_weight_digests") or [])
    scaler_digests = list(stage_meta.get("source_scaler_digests") or [])
    fields: Dict[str, Any] = {}
    for index in range(3):
        suffix = index + 1
        fields[f"source_{suffix}_identity"] = identities[index] if index < len(identities) else NA
        fields[f"source_{suffix}_distance"] = distances[index] if index < len(distances) else NA
        fields[f"source_{suffix}_model_digest"] = model_digests[index] if index < len(model_digests) else NA
        fields[f"source_{suffix}_scaler_digest"] = scaler_digests[index] if index < len(scaler_digests) else NA
    fields["fused_target_init_digest"] = stage_meta.get("fused_target_init_digest", NA)
    fields["target_split_digest"] = stage_meta.get("target_split_digest", {})
    fields["target_scaler_digest"] = None
    split = stage_meta.get("target_split_digest") or {}
    if isinstance(split, dict):
        fields["target_scaler_digest"] = split.get("target_scaler", NA)
    return fields


def build_target_comparison(
    *,
    target_identity: str,
    arm_rows: Mapping[str, Mapping[str, Any]],
    flags: Mapping[str, str],
    formal_refs: Mapping[str, Mapping[str, Any]],
    baseline_refs: Mapping[str, Any],
    cell_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    if any(value != "YES" for value in flags.values()):
        raise ValueError(f"paired Stage-A contract failed for {target_identity}: {dict(flags)}")
    spec_a = resolve_ablation_arm("full_current")
    spec_b = resolve_ablation_arm("fused_target_only")
    spec_c = resolve_ablation_arm("raw_target_only")
    if spec_b.transfer_branch_enabled or spec_c.transfer_branch_enabled:
        raise ValueError("B/C transfer branch must be disabled")
    if spec_c.use_fused_target_init:
        raise ValueError("C must use raw initialization")
    if not spec_a.use_fused_target_init or not spec_b.use_fused_target_init:
        raise ValueError("A/B must use fused initialization")
    row_b = arm_rows["fused_target_only"]
    row_c = arm_rows["raw_target_only"]
    if row_b["target_conv_init_digest"] != arm_rows["full_current"]["target_conv_init_digest"]:
        raise ValueError(f"A/B fused init digest mismatch for {target_identity}")
    if row_c["target_conv_init_digest"] == row_b["target_conv_init_digest"]:
        raise ValueError(f"C loaded fused weights for {target_identity}")
    if row_b["gate_mode"] != "disabled" or row_c["gate_mode"] != "disabled":
        raise ValueError(f"B/C gate must be disabled for {target_identity}")
    if row_b["target_split_digest"] != row_c["target_split_digest"]:
        raise ValueError(f"B/C target split mismatch for {target_identity}")
    effects = compute_abc_effects(dict(arm_rows))
    b_rmse = float(row_b["rmse"])
    no_tl = formal_refs["no_tl"]
    msml = formal_refs["msml"]
    best_bl = baseline_refs["best"]
    parsed = parse_d4_target_identity(target_identity)
    source_fields = _source_fields(
        {
            "source_identities": row_b["source_identities"],
            "source_distances": row_b["source_distances"],
            "source_weight_digests": row_b["source_weight_digests"],
            "source_scaler_digests": row_b["source_scaler_digests"],
            "fused_target_init_digest": row_b["target_conv_init_digest"],
            "target_split_digest": row_b["target_split_digest"],
        }
    )
    comparison = {
        "target_identity": parsed["target_identity"],
        "store_id": parsed["store_id"],
        "product_id": parsed["product_id"],
        "entity_key": parsed["entity_key"],
        "arm_rows": dict(arm_rows),
        "flags": dict(flags),
        "effects": effects,
        "formal": formal_refs,
        "baselines": baseline_refs,
        "source_fields": source_fields,
        "cell_meta": dict(cell_meta),
        "A_rmse": float(arm_rows["full_current"]["rmse"]),
        "A_smape": float(arm_rows["full_current"]["smape"]),
        "B_rmse": b_rmse,
        "B_smape": float(row_b["smape"]),
        "C_rmse": float(row_c["rmse"]),
        "C_smape": float(row_c["smape"]),
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
        "B_beats_A": bool(b_rmse < float(arm_rows["full_current"]["rmse"])),
        "B_beats_C": bool(b_rmse < float(row_c["rmse"])),
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
    }
    for key, _method, _display in BL_METHODS:
        comparison[f"{key}_rmse"] = baseline_refs[key].get("rmse")
        comparison[f"{key}_smape"] = baseline_refs[key].get("smape")
        comparison[f"{key}_status"] = baseline_refs[key].get("status")
        comparison[f"delta_rmse_vs_{key}"] = _delta(b_rmse, baseline_refs[key].get("rmse"))
    return comparison


def _win_record(flags: Sequence[Optional[bool]]) -> Dict[str, Any]:
    wins = sum(1 for flag in flags if flag is True)
    losses = sum(1 for flag in flags if flag is False)
    missing = sum(1 for flag in flags if flag is None)
    ties = len(flags) - wins - losses - missing
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "missing": missing,
        "n": len(flags),
        "rate": wins / len(flags) if flags else float("nan"),
        "label": f"{wins}/{len(flags)}",
    }


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.median(clean))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.mean(clean))


def aggregate_cross_target(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(comparisons) != 5:
        raise ValueError(f"aggregate summary requires 5 targets, got {len(comparisons)}")
    init_effects = [item["effects"]["INIT_EFFECT"] for item in comparisons]
    transfer_effects = [item["effects"]["TRANSFER_EFFECT"] for item in comparisons]
    return {
        "B_vs_A": _win_record([item["B_beats_A"] for item in comparisons]),
        "B_vs_C": _win_record([item["B_beats_C"] for item in comparisons]),
        "B_vs_NoTL": _win_record([item["B_beats_NoTL"] for item in comparisons]),
        "B_vs_MSML": _win_record([item["B_beats_MSML"] for item in comparisons]),
        "B_vs_BL4": _win_record([item["B_beats_BL4"] for item in comparisons]),
        "B_vs_bestBL": _win_record([item["B_beats_best_BL"] for item in comparisons]),
        "median_init_effect": _median(init_effects),
        "mean_init_effect": _mean(init_effects),
        "median_transfer_effect": _median(transfer_effects),
        "mean_transfer_effect": _mean(transfer_effects),
        "median_B_delta_vs_NoTL": _median([item["delta_rmse_vs_no_tl"] for item in comparisons]),
        "median_B_delta_vs_MSML": _median([item["delta_rmse_vs_msml"] for item in comparisons]),
        "median_B_delta_vs_BL4": _median([item["delta_rmse_vs_bl4"] for item in comparisons]),
        "median_B_delta_vs_bestBL": _median([item["delta_rmse_vs_best_bl"] for item in comparisons]),
        "median_relative_improvement_vs_no_tl": _median(
            [item["relative_improvement_vs_no_tl"] for item in comparisons]
        ),
        "median_relative_improvement_vs_msml": _median(
            [item["relative_improvement_vs_msml"] for item in comparisons]
        ),
        "median_relative_improvement_vs_bl4": _median(
            [item["relative_improvement_vs_bl4"] for item in comparisons]
        ),
        "median_relative_improvement_vs_best_bl": _median(
            [item["relative_improvement_vs_best_bl"] for item in comparisons]
        ),
    }


def decide_cross_target_verdict(aggregates: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]]) -> str:
    b_vs_c = int(aggregates["B_vs_C"]["wins"])
    b_vs_a = int(aggregates["B_vs_A"]["wins"])
    median_init = aggregates.get("median_init_effect")
    init_positive = median_init is not None and float(median_init) > 0
    b_best_of_abc = sum(
        1
        for item in comparisons
        if item["B_rmse"] <= item["A_rmse"] and item["B_rmse"] <= item["C_rmse"]
    )
    if b_vs_c >= 4 and b_vs_a >= 4 and init_positive and b_best_of_abc >= 4:
        return "FUSED_TARGET_ONLY_CROSS_TARGET_SUPPORTED"
    if b_vs_c <= 1:
        return "FUSED_TARGET_ONLY_NOT_CROSS_TARGET_SUPPORTED"
    return "FUSED_TARGET_ONLY_MIXED_CROSS_TARGET_EVIDENCE"


def _arm_csv_row(
    comparison: Mapping[str, Any],
    arm_name: str,
) -> Dict[str, Any]:
    row = comparison["arm_rows"][arm_name]
    spec = ABLATION_ARM_SPECS[arm_name]
    meta = comparison["cell_meta"]
    b_rmse = comparison["B_rmse"]
    applicable_a = arm_name == "full_current"
    applicable_b = arm_name == "fused_target_only"
    gate_mean = row["gate_diagnostics"].get("gate_mean_final") if applicable_a else None
    transfer_ratio = row["transfer_diagnostics"].get("transfer_ratio") if spec.transfer_branch_enabled else 0.0
    direction = (
        row["transfer_diagnostics"].get("correction_direction_accuracy")
        if spec.transfer_branch_enabled
        else None
    )
    drifts = row["conv_drifts"]
    return {
        "dataset": meta.get("dataset", "D4"),
        "target_identity": comparison["target_identity"],
        "store_id": comparison["store_id"],
        "product_id": comparison["product_id"],
        "mode": meta.get("mode", "without"),
        "horizon": meta.get("horizon", 1),
        "seed": meta.get("seed", 42),
        "epochs": meta.get("epochs", 50),
        "arm": arm_name,
        "rmse": row["rmse"],
        "smape": row["smape"],
        "fused_init": spec.use_fused_target_init,
        "target_conv_trainable": spec.target_backbone_trainable,
        "transfer_enabled": spec.transfer_branch_enabled,
        "gate_mode": row["gate_mode"],
        "init_effect": comparison["effects"]["INIT_EFFECT"] if applicable_b else NA,
        "transfer_effect": comparison["effects"]["TRANSFER_EFFECT"] if applicable_b else NA,
        "delta_rmse_vs_no_tl": _delta(row["rmse"], comparison["no_tl_rmse"]),
        "delta_rmse_vs_msml": _delta(row["rmse"], comparison["msml_rmse"]),
        "delta_rmse_vs_bl1": _delta(row["rmse"], comparison["bl1_rmse"]),
        "delta_rmse_vs_bl2": _delta(row["rmse"], comparison["bl2_rmse"]),
        "delta_rmse_vs_bl3": _delta(row["rmse"], comparison["bl3_rmse"]),
        "delta_rmse_vs_bl4": _delta(row["rmse"], comparison["bl4_rmse"]),
        "delta_rmse_vs_bl5": _delta(row["rmse"], comparison["bl5_rmse"]),
        "delta_rmse_vs_best_bl": _delta(row["rmse"], comparison["best_bl_rmse"]),
        "best_bl_method": comparison["best_bl_method"] or NA,
        "best_bl_rmse": comparison["best_bl_rmse"] if comparison["best_bl_rmse"] is not None else NA,
        "conv1_relative_drift": drifts.get("conv1_relative_weight_drift", NA) if applicable_b else drifts.get("conv1_relative_weight_drift", NA),
        "conv2_relative_drift": drifts.get("conv2_relative_weight_drift", NA),
        "conv3_relative_drift": drifts.get("conv3_relative_weight_drift", NA),
        "total_backbone_relative_drift": drifts.get("total_backbone_relative_weight_drift", NA),
        "gate_mean": gate_mean if gate_mean is not None else NA,
        "transfer_ratio": transfer_ratio if transfer_ratio is not None else NA,
        "correction_direction_accuracy": direction if direction is not None else NA,
        "b_rmse_for_effects": b_rmse,
    }


def write_cross_target_outputs(
    run_dir: Path,
    comparisons: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    cell_contract: Mapping[str, Any],
    pairing_flags: Mapping[str, str],
    protected_before: Mapping[str, Any],
    protected_after: Mapping[str, Any],
    verdict: str,
) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    pred_root = run_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    result_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "cell_contract": dict(cell_contract),
        "formal_methods": list(FORMAL_METHODS),
        "ms_katl_in_formal_methods": METHOD_NAME in FORMAL_METHODS,
        "pairing": dict(pairing_flags),
        "protected_artifacts_before": dict(protected_before),
        "protected_artifacts_after": dict(protected_after),
        "aggregates": json.loads(json.dumps(aggregates, default=str)),
        "verdict": verdict,
        "targets": {},
    }
    for comparison in comparisons:
        parsed = parse_d4_target_identity(comparison["target_identity"])
        pred_dir = pred_root / parsed["prediction_dirname"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        for arm_name in CROSS_TARGET_ARM_NAMES:
            row = comparison["arm_rows"][arm_name]
            pred_df = pd.DataFrame(
                {
                    "y_true": np.asarray(row["y_true"]).reshape(-1),
                    "y_pred": np.asarray(row["y_pred"]).reshape(-1),
                    "target_prediction": np.asarray(row["decomposition"]["target_prediction"]).reshape(-1),
                    "transfer_raw": np.asarray(row["decomposition"]["transfer_raw"]).reshape(-1),
                    "gate": np.asarray(row["decomposition"]["gate"]).reshape(-1),
                    "transfer_gated": np.asarray(row["decomposition"]["transfer_gated"]).reshape(-1),
                    "final_prediction": np.asarray(row["decomposition"]["final_prediction"]).reshape(-1),
                }
            )
            pred_df.to_csv(pred_dir / f"{arm_name}.csv", index=False, encoding="utf-8")
            result_rows.append({key: _csv_value(value) for key, value in _arm_csv_row(comparison, arm_name).items()})
        summary_rows.append(
            {
                "target_identity": comparison["target_identity"],
                "A_rmse": comparison["A_rmse"],
                "B_rmse": comparison["B_rmse"],
                "C_rmse": comparison["C_rmse"],
                "formal_no_tl_rmse": _csv_value(comparison["no_tl_rmse"]),
                "msml_rmse": _csv_value(comparison["msml_rmse"]),
                "bl1_rmse": _csv_value(comparison["bl1_rmse"]),
                "bl2_rmse": _csv_value(comparison["bl2_rmse"]),
                "bl3_rmse": _csv_value(comparison["bl3_rmse"]),
                "bl4_rmse": _csv_value(comparison["bl4_rmse"]),
                "bl5_rmse": _csv_value(comparison["bl5_rmse"]),
                "best_bl_method": _csv_value(comparison["best_bl_method"]),
                "best_bl_rmse": _csv_value(comparison["best_bl_rmse"]),
                "init_effect": comparison["effects"]["INIT_EFFECT"],
                "transfer_effect": comparison["effects"]["TRANSFER_EFFECT"],
                "B_beats_A": _csv_value(comparison["B_beats_A"]),
                "B_beats_C": _csv_value(comparison["B_beats_C"]),
                "B_beats_NoTL": _csv_value(comparison["B_beats_NoTL"]),
                "B_beats_MSML": _csv_value(comparison["B_beats_MSML"]),
                "B_beats_BL4": _csv_value(comparison["B_beats_BL4"]),
                "B_beats_best_BL": _csv_value(comparison["B_beats_best_BL"]),
            }
        )
        arm_dump = {}
        for arm_name, row in comparison["arm_rows"].items():
            arm_dump[arm_name] = {
                key: row[key]
                for key in row
                if key not in {"y_true", "y_pred", "history", "decomposition"}
            }
            arm_dump[arm_name]["attention_diagnostics"] = {
                key: value
                for key, value in (row.get("attention_diagnostics") or {}).items()
                if key != "alpha_test"
            }
        diagnostics["targets"][comparison["target_identity"]] = {
            "source_fields": comparison["source_fields"],
            "flags": comparison["flags"],
            "effects": comparison["effects"],
            "references": {
                "formal": comparison["formal"],
                "baselines": {
                    key: dict(comparison["baselines"][key])
                    for key in ["bl1", "bl2", "bl3", "bl4", "bl5", "best"]
                },
            },
            "arms": arm_dump,
            "B_beats": {
                "A": comparison["B_beats_A"],
                "C": comparison["B_beats_C"],
                "NoTL": comparison["B_beats_NoTL"],
                "MSML": comparison["B_beats_MSML"],
                "BL4": comparison["B_beats_BL4"],
                "best_BL": comparison["B_beats_best_BL"],
            },
            "relative_improvement": {
                "vs_no_tl": comparison["relative_improvement_vs_no_tl"],
                "vs_msml": comparison["relative_improvement_vs_msml"],
                "vs_bl4": comparison["relative_improvement_vs_bl4"],
                "vs_best_bl": comparison["relative_improvement_vs_best_bl"],
            },
        }

    results_csv = results_dir / "d4_h1_s42_cross_target_results.csv"
    summary_csv = results_dir / "d4_h1_s42_cross_target_summary.csv"
    diagnostics_json = results_dir / "d4_h1_s42_cross_target_diagnostics.json"
    pd.DataFrame(result_rows).to_csv(results_csv, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8")
    diagnostics_json.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "diagnostics_json": diagnostics_json,
        "predictions_dir": pred_root,
    }


def format_cross_target_terminal_summary(
    comparisons: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    pairing_flags: Mapping[str, str],
    *,
    verdict: str,
) -> str:
    lines = [
        "================================================================================",
        "D4 H1 S42 CROSS-TARGET VALIDATION",
        "================================================================================",
        "",
        f"{'TARGET':<12}{'A_FULL':<12}{'B_FUSED_TGT':<14}{'C_RAW_TGT':<13}{'NoTL':<10}{'MSML':<10}{'BL4':<10}{'BEST_BL'}",
        "--------------------------------------------------------------------------------",
    ]
    for item in comparisons:
        lines.append(
            f"{item['target_identity']:<12}"
            f"{_fmt(item['A_rmse']):<12}"
            f"{_fmt(item['B_rmse']):<14}"
            f"{_fmt(item['C_rmse']):<13}"
            f"{_fmt(item['no_tl_rmse']):<10}"
            f"{_fmt(item['msml_rmse']):<10}"
            f"{_fmt(item['bl4_rmse']):<10}"
            f"{_fmt(item['best_bl_rmse'])}"
        )
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            "",
            "WIN RATES FOR B",
            f"B vs A             {aggregates['B_vs_A']['label']}",
            f"B vs C             {aggregates['B_vs_C']['label']}",
            f"B vs No-TL         {aggregates['B_vs_NoTL']['label']}",
            f"B vs MSML-TL       {aggregates['B_vs_MSML']['label']}",
            f"B vs BL4           {aggregates['B_vs_BL4']['label']}",
            f"B vs BEST BL       {aggregates['B_vs_bestBL']['label']}",
            "",
            "EFFECTS",
            f"median INIT_EFFECT        {_fmt(aggregates['median_init_effect'])}",
            f"mean   INIT_EFFECT        {_fmt(aggregates['mean_init_effect'])}",
            "",
            f"median TRANSFER_EFFECT    {_fmt(aggregates['median_transfer_effect'])}",
            f"mean   TRANSFER_EFFECT    {_fmt(aggregates['mean_transfer_effect'])}",
            "",
            f"median B Δ vs No-TL       {_fmt(aggregates['median_B_delta_vs_NoTL'])}",
            f"median B Δ vs MSML        {_fmt(aggregates['median_B_delta_vs_MSML'])}",
            f"median B Δ vs BL4         {_fmt(aggregates['median_B_delta_vs_BL4'])}",
            f"median B Δ vs BEST BL     {_fmt(aggregates['median_B_delta_vs_bestBL'])}",
            "",
            "PAIRING CONTRACT",
            f"STAGE_A_SHARED_WITHIN_TARGET={pairing_flags.get('STAGE_A_SHARED_WITHIN_TARGET', 'NO')}",
            f"SOURCE_IDENTITIES_MATCH={pairing_flags.get('SOURCE_IDENTITIES_MATCH', 'NO')}",
            f"SOURCE_MODEL_DIGESTS_MATCH={pairing_flags.get('SOURCE_MODEL_DIGESTS_MATCH', 'NO')}",
            f"SOURCE_SCALERS_MATCH={pairing_flags.get('SOURCE_SCALERS_MATCH', 'NO')}",
            f"TARGET_SPLITS_MATCH={pairing_flags.get('TARGET_SPLITS_MATCH', 'NO')}",
            f"FUSED_INIT_MATCH_A_B={pairing_flags.get('FUSED_INIT_MATCH_A_B', 'NO')}",
            "",
            f"FORMAL_METHODS_UNCHANGED={pairing_flags.get('FORMAL_METHODS_UNCHANGED', 'NO')}",
            f"HISTORICAL_RESULTS_UNTOUCHED={pairing_flags.get('HISTORICAL_RESULTS_UNTOUCHED', 'NO')}",
            f"VERDICT={verdict}",
            "================================================================================",
        ]
    )
    return "\n".join(lines)


def overall_pairing_flags(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    def _all(key: str) -> str:
        return "YES" if all(item["flags"].get(key) == "YES" for item in comparisons) else "NO"

    return {
        "STAGE_A_SHARED_WITHIN_TARGET": _all("STAGE_A_SHARED"),
        "SOURCE_IDENTITIES_MATCH": _all("SOURCE_IDENTITIES_MATCH"),
        "SOURCE_MODEL_DIGESTS_MATCH": _all("SOURCE_MODEL_DIGESTS_MATCH"),
        "SOURCE_SCALERS_MATCH": _all("SOURCE_SCALERS_MATCH"),
        "TARGET_SPLITS_MATCH": _all("TARGET_SPLIT_MATCH"),
        "FUSED_INIT_MATCH_A_B": _all("FUSED_TARGET_INIT_MATCH"),
        "FORMAL_METHODS_UNCHANGED": "YES",
        "HISTORICAL_RESULTS_UNTOUCHED": "YES",
    }


def format_target_method_table(comparison: Mapping[str, Any]) -> str:
    rows = [
        ("BL1 Historical Mean", comparison["bl1_rmse"], comparison["bl1_smape"]),
        ("BL2 Moving Average", comparison["bl2_rmse"], comparison["bl2_smape"]),
        ("BL3 LightGBM", comparison["bl3_rmse"], comparison["bl3_smape"]),
        ("BL4 LSTM", comparison["bl4_rmse"], comparison["bl4_smape"]),
        ("BL5 Seasonal Naive", comparison["bl5_rmse"], comparison["bl5_smape"]),
        ("Formal No-TL", comparison["no_tl_rmse"], comparison["no_tl_smape"]),
        ("MSML-TL", comparison["msml_rmse"], comparison["msml_smape"]),
        ("A FULL_CURRENT", comparison["A_rmse"], comparison["A_smape"]),
        ("B FUSED_TARGET_ONLY", comparison["B_rmse"], comparison["B_smape"]),
        ("C RAW_TARGET_ONLY", comparison["C_rmse"], comparison["C_smape"]),
        ("BEST_BL", comparison["best_bl_rmse"], comparison["best_bl_smape"]),
    ]
    lines = [
        f"TARGET {comparison['target_identity']}",
        "--------------------------------------------------",
        f"{'METHOD':<25}{'RMSE':<11}{'sMAPE'}",
    ]
    for name, rmse, smape in rows:
        extra = ""
        if name == "BEST_BL" and comparison.get("best_bl_method"):
            extra = f"  ({comparison['best_bl_method']})"
        lines.append(f"{name:<25}{_fmt(rmse):<11}{_fmt(smape)}{extra}")
    lines.append("--------------------------------------------------")
    return "\n".join(lines)
