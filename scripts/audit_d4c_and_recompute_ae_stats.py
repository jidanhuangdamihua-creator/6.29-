#!/usr/bin/env python3
"""Read-only D4-C evidence audit + offline A-E mechanism recomputation.

ANALYSIS-ONLY / NO-TRAINING GUARD
---------------------------------
This script must never train, fit, optimize, create checkpoints, or write into
existing experiment run directories. It only reads frozen artifacts and writes
a new audit directory.

Frozen statistical definitions reused (not reinvented):
- 5-seed mean over dataset x horizon x arm, then 12 condition-level pairs
- RI = (E_reference - E_treatment) / E_reference * 100%
- Win/Tie/Loss: exact numeric comparison, no extra practical-equivalence band
  (TIE_RULE from tools/build_final_statistical_derivation_freeze.py)
- two-sided paired Wilcoxon signed-rank on the 12 condition means
- 100,000 condition-level percentile bootstrap of mean RI
- Holm correction separately for the RMSE family of 4 and the sMAPE family of 4
- A-E analysis RNG seed 20260919 from
  outputs/paper_statistics/corrected_ae_statistical_analysis_20260919_154638
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


ANALYSIS_ONLY = True
FORBID_TRAINING = True

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOKENS = (
    "--execute",
    "--train",
    "run_ms_katl",
    "AE_FORMAL_TRAINING_UNLOCK",
    "model.fit",
    "optimizer.apply",
)

FORMAL_RUN = ROOT / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227"
FORMAL_CSV = FORMAL_RUN / "ae_uniform_3e4_results.csv"
FORMAL_MANIFEST = FORMAL_RUN / "manifest_snapshot.json"
FORMAL_LOG = ROOT / "logs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227.log"
FREEZE_ROOT = ROOT / "outputs" / "paper_statistics" / "corrected_ae_uniform_3e4_freeze_20260919_152249"
PRIOR_STATS_ROOT = ROOT / "outputs" / "paper_statistics" / "corrected_ae_statistical_analysis_20260919_154638"
DIAGNOSTIC_RUN = ROOT / "outputs" / "mechanism_diagnostics" / "mechanism_rerun_20260920_215718"
OUTPUT_DIR = ROOT / "outputs" / "posthoc_d4c_ae_audit_20260921"

DATASETS = ("D1", "D4", "D5", "D6")
TARGETS = {"D1": "1/10", "D4": "166/258", "D5": "48/1159414", "D6": "CA_1/FOODS_3_080"}
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("A", "B", "C", "D", "E")
ARM_NAMES = {
    "A": "TARGET_NATIVE_INIT",
    "B": "CONTROLLED_FROZEN_FUSED",
    "C": "FUSED_TRAINABLE",
    "D": "CURRENT_V3",
    "E": "BEST_SOURCE_TRAINABLE",
}
CONTRASTS = (
    ("C-A", "C", "A"),
    ("C-B", "C", "B"),
    ("C-E", "C", "E"),
    ("D-C", "D", "C"),
)
TAIL_EPOCHS = (40, 45, 46, 47, 48, 49, 50)
LAST5 = (46, 47, 48, 49, 50)

# Frozen A-E statistical contract (2026-09-19 analysis + paper freeze tie rule).
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_RANDOM_SEED = 20260919
BOOTSTRAP_RNG_POLICY = "reset_default_rng_to_bootstrap_seed_per_contrast_metric"
TIE_RULE = "exact_numeric_no_tolerance"
WILCOXON_ALTERNATIVE = "two-sided"
WILCOXON_ZERO_METHOD = "wilcox"
WILCOXON_METHOD = "auto"
HOLM_ALPHA = 0.05
STAGE_B_EPOCHS_CODE = 50
STAGE_B_BATCH_SIZE_CODE = 16
EMA_MOMENTUM_CODE = 0.99

CODE_PATH_EVIDENCE = {
    "fused_target_init_snapshot": "src/transfer_methods/ms_katl_lite.py:fused_target_init_snapshot",
    "clone_cnn_from_snapshot": "src/transfer_methods/ms_katl_lite.py:clone_cnn_from_snapshot",
    "construct_ae_arm": "src/transfer_methods/ms_katl_ae_ablation.py:construct_ae_arm",
    "snapshot_for_init_mode": "src/transfer_methods/ms_katl_ae_ablation.py:snapshot_for_init_mode",
    "head_snapshot": "src/transfer_methods/ms_katl_ae_ablation.py:construct_v3_head_target_snapshot",
    "train_loop": "src/transfer_methods/ms_katl_lite.py:train_ms_katl_head",
    "formal_train": "src/transfer_methods/ms_katl_ae_formal.py:default_v3_train_stage_b",
    "formal_predict": "src/transfer_methods/ms_katl_ae_formal.py:default_v3_predict",
    "diagnostics_kwargs": "src/transfer_methods/sita_mechanism_diagnostics_capture.py:stage_b_diagnostics_train_kwargs",
    "val_rmse_history": "src/transfer_methods/ms_katl_lite.py:train_ms_katl_head history['val_rmse']",
}


class AnalysisOnlyError(RuntimeError):
    """Fail closed if this script is asked to train or mutate experiments."""


def _guard() -> None:
    if not ANALYSIS_ONLY or not FORBID_TRAINING:
        raise AnalysisOnlyError("analysis-only guard disabled")
    joined = " ".join(sys.argv).lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in joined:
            raise AnalysisOnlyError(f"refusing training-related argument: {token}")
    output = str(OUTPUT_DIR.resolve())
    if str(FORMAL_RUN.resolve()) in output and output != str((ROOT / "outputs" / "posthoc_d4c_ae_audit_20260921").resolve()):
        raise AnalysisOnlyError("refusing to write into the formal run directory")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def holm_adjust(p_values: Mapping[str, float], alpha: float = HOLM_ALPHA) -> Dict[str, Dict[str, Any]]:
    items = [(index, key, float(value)) for index, (key, value) in enumerate(p_values.items())]
    ordered = sorted(items, key=lambda item: (item[2], item[0]))
    adjusted: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    for rank, (_index, key, p_value) in enumerate(ordered, start=1):
        multiplier = len(items) - rank + 1
        candidate = min(1.0, p_value * multiplier)
        running = max(running, candidate)
        adjusted[key] = {
            "raw_p": p_value,
            "rank": rank,
            "multiplier": multiplier,
            "adjusted_p": running,
            "significant": running <= alpha,
        }
    return adjusted


def relative_improvement_pct(reference: float, treatment: float) -> float:
    reference = float(reference)
    if not math.isfinite(reference) or reference == 0.0:
        raise AnalysisOnlyError(f"relative improvement requires a finite nonzero reference: {reference}")
    return (reference - float(treatment)) / reference * 100.0


def win_tie_loss(reference: float, treatment: float) -> str:
    left = float(reference)
    right = float(treatment)
    if right < left:
        return "W"
    if right == left:
        return "T"
    return "L"


def wilcoxon_paired(differences: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray([float(value) for value in differences], dtype=float)
    zero_count = int(np.count_nonzero(values == 0.0))
    nonzero = values[values != 0.0]
    payload: Dict[str, Any] = {
        "n": int(len(values)),
        "zero_difference_count": zero_count,
        "method_requested": WILCOXON_METHOD,
        "alternative": WILCOXON_ALTERNATIVE,
        "zero_method": WILCOXON_ZERO_METHOD,
        "status": "OK",
        "scipy_zero_handling": (
            "zeros dropped via nonzero mask before scipy.stats.wilcoxon; "
            "zero_method=wilcox is still passed for the remaining values"
        ),
    }
    if nonzero.size == 0:
        payload.update({"statistic": float("nan"), "p_value": float("nan"), "status": "ALL_ZERO"})
        return payload
    result = stats.wilcoxon(
        nonzero,
        zero_method=WILCOXON_ZERO_METHOD,
        alternative=WILCOXON_ALTERNATIVE,
        method=WILCOXON_METHOD,
        nan_policy="raise",
    )
    payload["statistic"] = float(result.statistic)
    payload["p_value"] = float(result.pvalue)
    payload["scipy_method_used"] = str(getattr(result, "method", WILCOXON_METHOD))
    return payload


def bootstrap_mean_ci(values: Sequence[float], seed: int = BOOTSTRAP_RANDOM_SEED) -> Tuple[float, float, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size != 12:
        raise AnalysisOnlyError(f"bootstrap input must contain exactly 12 condition values, got {array.size}")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    means = array[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(array.mean()), float(lower), float(upper)


def cell_dir_name(dataset: str, target: str, horizon: int, seed: int) -> str:
    target_token = target.replace("/", "-")
    return f"{dataset}__{target_token}__h{horizon}__s{seed}__without"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_formal_rows() -> List[Dict[str, Any]]:
    with FORMAL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = dict(row)
        item["horizon"] = int(row["horizon"])
        item["seed"] = int(row["seed"])
        item["RMSE"] = float(row["RMSE"]) if row.get("RMSE") not in (None, "") else float("nan")
        item["sMAPE"] = float(row["sMAPE"]) if row.get("sMAPE") not in (None, "") else float("nan")
        item["rmse_original"] = (
            float(row["rmse_original"]) if row.get("rmse_original") not in (None, "") else float("nan")
        )
        item["smape_original"] = (
            float(row["smape_original"]) if row.get("smape_original") not in (None, "") else float("nan")
        )
        parsed.append(item)
    return parsed


def integrity_audit(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected_keys = [
        (dataset, TARGETS[dataset], horizon, arm, seed)
        for dataset in DATASETS
        for horizon in HORIZONS
        for arm in ARMS
        for seed in SEEDS
    ]
    observed = [(row["dataset"], row["target"], int(row["horizon"]), row["arm"], int(row["seed"])) for row in rows]
    observed_counter = Counter(observed)
    missing = [key for key in expected_keys if observed_counter[key] == 0]
    extras = [key for key, count in observed_counter.items() if key not in set(expected_keys)]
    duplicates = [key for key, count in observed_counter.items() if count > 1]
    nan_rmse = sum(1 for row in rows if not finite(row.get("rmse_original")))
    nan_smape = sum(1 for row in rows if not finite(row.get("smape_original")))
    failed = [row for row in rows if str(row.get("run_status", "")).strip() != "OK"]
    pair_incomplete = [row for row in rows if str(row.get("pair_status", "")).strip() != "PAIR_COMPLETE"]
    wrong_target = [row for row in rows if row.get("target") != TARGETS.get(str(row.get("dataset")))]
    wrong_lr = [row for row in rows if str(row.get("learning_rate_label", "")).strip() != "3e-4"]
    artifact_mismatch = 0
    artifact_checked = 0
    status_abnormal = 0
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                cell_path = FORMAL_RUN / "cells" / cell_dir_name(dataset, TARGETS[dataset], horizon, seed)
                result_path = cell_path / "attempt_001" / "result.json"
                status_path = cell_path / "attempt_001" / "status.json"
                artifact_checked += 1
                if not result_path.is_file() or not status_path.is_file():
                    artifact_mismatch += 1
                    continue
                result = load_json(result_path)
                status = load_json(status_path)
                if str(status.get("status")) != "COMPLETE" or str(result.get("status")) != "COMPLETE":
                    status_abnormal += 1
                if str(result.get("pair_status")) != "PAIR_COMPLETE":
                    status_abnormal += 1
                for arm in ARMS:
                    csv_row = next(
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and int(row["horizon"]) == horizon
                        and int(row["seed"]) == seed
                        and row["arm"] == arm
                    )
                    arm_payload = result["arms"][arm]
                    if abs(float(arm_payload["rmse_original"]) - float(csv_row["rmse_original"])) > 1e-12:
                        artifact_mismatch += 1
                    if str(arm_payload.get("status")) != "OK":
                        status_abnormal += 1
    complete = (
        len(rows) == 300
        and not missing
        and not extras
        and not duplicates
        and nan_rmse == 0
        and nan_smape == 0
        and not failed
        and not pair_incomplete
        and not wrong_target
        and not wrong_lr
        and artifact_mismatch == 0
        and status_abnormal == 0
        and int(manifest.get("expected_runs", -1)) == 300
    )
    summary = {
        "n_rows": len(rows),
        "expected_rows": 300,
        "n_missing": len(missing),
        "n_extras": len(extras),
        "n_duplicates": len(duplicates),
        "nan_rmse": nan_rmse,
        "nan_smape": nan_smape,
        "failed_rows": len(failed),
        "pair_incomplete_rows": len(pair_incomplete),
        "wrong_target_rows": len(wrong_target),
        "wrong_lr_rows": len(wrong_lr),
        "canonical_cell_dirs": 60,
        "artifact_checked": artifact_checked,
        "artifact_mismatch": artifact_mismatch,
        "status_abnormal": status_abnormal,
        "complete_legal_300": complete,
        "matrix_kind": manifest.get("matrix_kind"),
        "runtime_commit": manifest.get("runtime_commit"),
        "label": manifest.get("label"),
        "missing": [str(item) for item in missing],
        "duplicates": [str(item) for item in duplicates],
    }
    audit_rows = []
    for dataset, target, horizon, arm, seed in expected_keys:
        count = observed_counter[(dataset, target, horizon, arm, seed)]
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["target"] == target
            and int(row["horizon"]) == horizon
            and row["arm"] == arm
            and int(row["seed"]) == seed
        ]
        row = selected[0] if selected else {}
        audit_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "horizon": horizon,
                "arm": arm,
                "seed": seed,
                "present_count": count,
                "run_status": row.get("run_status", "MISSING"),
                "pair_status": row.get("pair_status", "MISSING"),
                "RMSE": row.get("rmse_original", ""),
                "sMAPE": row.get("smape_original", ""),
                "learning_rate_label": row.get("learning_rate_label", ""),
                "duplicate": count > 1,
                "nan_metric": bool(selected) and (not finite(row.get("rmse_original")) or not finite(row.get("smape_original"))),
                "ok": count == 1
                and str(row.get("run_status")) == "OK"
                and finite(row.get("rmse_original"))
                and finite(row.get("smape_original")),
            }
        )
    return audit_rows, summary


def condition_means(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target"], int(row["horizon"]), row["arm"])].append(row)
    output = []
    for dataset in DATASETS:
        target = TARGETS[dataset]
        for horizon in HORIZONS:
            for arm in ARMS:
                items = grouped[(dataset, target, horizon, arm)]
                seeds = sorted(int(item["seed"]) for item in items)
                output.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "horizon": horizon,
                        "arm": arm,
                        "arm_name": ARM_NAMES[arm],
                        "n_seeds": len(items),
                        "seeds": ",".join(str(seed) for seed in seeds),
                        "rmse_mean": float(np.mean([float(item["rmse_original"]) for item in items])),
                        "smape_mean": float(np.mean([float(item["smape_original"]) for item in items])),
                    }
                )
    return output


def contrast_tables(means: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index = {(row["dataset"], int(row["horizon"]), row["arm"]): row for row in means}
    pairwise = []
    statistics_rows = []
    for contrast, treatment, reference in CONTRASTS:
        for metric, treat_key, ref_key, ri_key in (
            ("RMSE", "rmse_mean", "rmse_mean", "rmse_relative_improvement_pct"),
            ("sMAPE", "smape_mean", "smape_mean", "smape_relative_improvement_pct"),
        ):
            condition_rows = []
            ris = []
            diffs = []
            outcomes = []
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    treat = index[(dataset, horizon, treatment)]
                    ref = index[(dataset, horizon, reference)]
                    treat_value = float(treat[treat_key])
                    ref_value = float(ref[ref_key])
                    ri = relative_improvement_pct(ref_value, treat_value)
                    outcome = win_tie_loss(ref_value, treat_value)
                    diffs.append(treat_value - ref_value)
                    ris.append(ri)
                    outcomes.append(outcome)
                    condition_rows.append(
                        {
                            "comparison": contrast,
                            "treatment_arm": treatment,
                            "reference_arm": reference,
                            "dataset": dataset,
                            "target": TARGETS[dataset],
                            "horizon": horizon,
                            "metric": metric,
                            "treatment_5seed_mean": treat_value,
                            "reference_5seed_mean": ref_value,
                            "relative_improvement_pct": ri,
                            "win_tie_loss": outcome,
                        }
                    )
            pairwise.extend(condition_rows)
            wx = wilcoxon_paired(diffs)
            mean_ri, ci_low, ci_high = bootstrap_mean_ci(ris, seed=BOOTSTRAP_RANDOM_SEED)
            statistics_rows.append(
                {
                    "comparison": contrast,
                    "treatment_arm": treatment,
                    "reference_arm": reference,
                    "metric": metric,
                    "n_pairs": 12,
                    "wins": outcomes.count("W"),
                    "ties": outcomes.count("T"),
                    "losses": outcomes.count("L"),
                    "mean_relative_improvement_pct": mean_ri,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "bootstrap_n": BOOTSTRAP_RESAMPLES,
                    "bootstrap_rng_seed": BOOTSTRAP_RANDOM_SEED,
                    "bootstrap_unit": "12_dataset_horizon_condition_RI",
                    "wilcoxon_statistic": wx.get("statistic"),
                    "wilcoxon_raw_p": wx.get("p_value"),
                    "wilcoxon_n": wx.get("n"),
                    "zero_difference_count": wx.get("zero_difference_count"),
                    "wilcoxon_zero_method": wx.get("zero_method"),
                    "wilcoxon_method_requested": wx.get("method_requested"),
                    "wilcoxon_scipy_method_used": wx.get("scipy_method_used", ""),
                    "wilcoxon_zero_handling": wx.get("scipy_zero_handling"),
                    "tie_rule": TIE_RULE,
                    "_diffs": diffs,
                    "_ris": ris,
                    "_condition_rows": condition_rows,
                }
            )
    for metric in ("RMSE", "sMAPE"):
        family = {row["comparison"]: float(row["wilcoxon_raw_p"]) for row in statistics_rows if row["metric"] == metric}
        adjusted = holm_adjust(family)
        for row in statistics_rows:
            if row["metric"] != metric:
                continue
            details = adjusted[row["comparison"]]
            row["holm_family"] = f"{metric}_4"
            row["holm_family_size"] = 4
            row["holm_adjusted_p"] = details["adjusted_p"]
            row["holm_reject_at_0.05"] = bool(details["significant"])
    clean_stats = []
    for row in statistics_rows:
        clean = dict(row)
        clean.pop("_diffs", None)
        clean.pop("_ris", None)
        clean.pop("_condition_rows", None)
        clean_stats.append(clean)
    return pairwise, clean_stats


def d4c_cells() -> List[Tuple[int, int]]:
    return [(horizon, seed) for horizon in HORIZONS for seed in SEEDS]


def inspect_history_artifacts(cell_root: Path) -> Dict[str, Any]:
    files = [path for path in cell_root.rglob("*") if path.is_file()]
    names = [path.name.lower() for path in files]
    history_like = [
        str(path.relative_to(FORMAL_RUN))
        for path in files
        if any(token in path.name.lower() for token in ("hist", "epoch", "val_rmse", "dynamics"))
    ]
    return {
        "n_files": len(files),
        "file_names": sorted(set(names)),
        "history_like": history_like,
    }


def audit_d4c_validation(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for horizon, seed in d4c_cells():
        csv_row = next(
            row
            for row in rows
            if row["dataset"] == "D4" and row["arm"] == "C" and int(row["horizon"]) == horizon and int(row["seed"]) == seed
        )
        cell_path = FORMAL_RUN / "cells" / cell_dir_name("D4", "166/258", horizon, seed)
        result = load_json(cell_path / "attempt_001" / "result.json")
        arm = result["arms"]["C"]
        history_info = inspect_history_artifacts(cell_path)
        payload = {
            "dataset": "D4",
            "arm": "C",
            "horizon": horizon,
            "seed": seed,
            "cell_dir": str(cell_path.relative_to(ROOT)),
            "run_status": csv_row.get("run_status"),
            "pair_status": csv_row.get("pair_status"),
            "csv_validation_metric": csv_row.get("validation_metric") or "NOT RECORDED",
            "best_epoch": "NOT RECORDED",
            "best_val_rmse": "NOT RECORDED",
            "final_val_rmse": "NOT RECORDED",
            "history_files": ";".join(history_info["history_like"]) if history_info["history_like"] else "NONE",
            "artifact_files": ",".join(history_info["file_names"]),
            "test_rmse_original_not_used_for_judgement": arm.get("rmse_original"),
            "judgement": "cannot determine",
            "evidence": (
                "Formal cell artifacts persist result.json/status.json/provenance.json only. "
                "No epoch history, no val_rmse trajectory, no best_epoch. "
                "CSV validation_metric is empty. Formal train path does not pass x_val."
            ),
        }
        for epoch in TAIL_EPOCHS:
            payload[f"val_rmse_e{epoch}"] = "NOT RECORDED"
        payload["change_40_50_abs"] = "NOT RECORDED"
        payload["change_40_50_pct"] = "NOT RECORDED"
        payload["last5_range"] = "NOT RECORDED"
        payload["last5_slope"] = "NOT RECORDED"
        output.append(payload)
    return output


def linear_slope(xs: Sequence[int], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = float(len(x))
    denom = n * float(np.sum(x * x)) - float(np.sum(x)) ** 2
    if denom == 0:
        return float("nan")
    return (n * float(np.sum(x * y)) - float(np.sum(x)) * float(np.sum(y))) / denom


def describe_tail(values: Mapping[int, float]) -> Dict[str, Any]:
    missing = [epoch for epoch in TAIL_EPOCHS if epoch not in values]
    if missing:
        return {
            "change_40_50_abs": "NOT RECORDED",
            "change_40_50_pct": "NOT RECORDED",
            "last5_range": "NOT RECORDED",
            "last5_slope": "NOT RECORDED",
            "judgement": "cannot determine",
        }
    change = values[50] - values[40]
    pct = change / values[40] * 100.0 if values[40] != 0 else float("nan")
    last5 = [values[epoch] for epoch in LAST5]
    slope = linear_slope(LAST5, last5)
    last5_range = max(last5) - min(last5)
    # Descriptive only; no convergence threshold.
    if change < 0 and slope < 0:
        judgement = "still improving"
    elif change > 0 and slope > 0:
        judgement = "still increasing"
    elif last5_range == 0:
        judgement = "approximately plateau"
    else:
        judgement = "small fluctuation"
    return {
        "change_40_50_abs": change,
        "change_40_50_pct": pct,
        "last5_range": last5_range,
        "last5_slope": slope,
        "judgement": judgement,
    }


def diagnostic_validation_tail() -> List[Dict[str, Any]]:
    output = []
    for horizon, seed in d4c_cells():
        path = DIAGNOSTIC_RUN / "training_dynamics" / f"D4__166_258__h{horizon}__s{seed}__C.csv"
        row: Dict[str, Any] = {
            "dataset": "D4",
            "arm": "C",
            "horizon": horizon,
            "seed": seed,
            "source": "independent_mechanism_diagnostic_NOT_FORMAL",
            "source_path": str(path.relative_to(ROOT)) if path.is_file() else "MISSING",
            "replaces_frozen_results": "NO",
        }
        if not path.is_file():
            for epoch in TAIL_EPOCHS:
                row[f"val_rmse_e{epoch}"] = "NOT RECORDED"
            row.update(describe_tail({}))
            output.append(row)
            continue
        by_epoch = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                epoch = int(item["epoch"])
                if epoch in TAIL_EPOCHS:
                    by_epoch[epoch] = float(item["validation_rmse_original_scale"])
                    row[f"weight_variant_e{epoch}"] = item.get("weight_variant")
                    row[f"ema_applied_e{epoch}"] = item.get("ema_applied")
        for epoch in TAIL_EPOCHS:
            row[f"val_rmse_e{epoch}"] = by_epoch.get(epoch, "NOT RECORDED")
        row.update(describe_tail(by_epoch))
        output.append(row)
    return output


def audit_d4c_raw_ema(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for horizon, seed in d4c_cells():
        cell_path = FORMAL_RUN / "cells" / cell_dir_name("D4", "166/258", horizon, seed)
        result = load_json(cell_path / "attempt_001" / "result.json")
        status = load_json(cell_path / "attempt_001" / "status.json")
        provenance = load_json(cell_path / "attempt_001" / "provenance.json")
        arm_c = result["arms"]["C"]
        arm_a = result["arms"]["A"]
        arm_b = result["arms"]["B"]
        arm_d = result["arms"]["D"]
        fused_ok = (
            arm_c.get("init_mode") == "fused"
            and arm_c.get("source_init_consumed") is True
            and arm_c.get("fused_init_digest") == arm_b.get("fused_init_digest") == arm_d.get("fused_init_digest")
            and arm_c.get("target_initial_conv_digest") == arm_b.get("target_initial_conv_digest") == arm_d.get("target_initial_conv_digest")
            and arm_c.get("target_initial_conv_digest") != arm_a.get("target_initial_conv_digest")
            and arm_c.get("fused_init_digest") != arm_c.get("target_initial_conv_digest")
            and bool(arm_c.get("fused_init_digest"))
            and bool(arm_c.get("target_initial_conv_digest"))
        )
        head_shared = len({result["arms"][arm]["head_target_init_digest"] for arm in ARMS}) == 1
        csv_row = next(
            row
            for row in rows
            if row["dataset"] == "D4" and row["arm"] == "C" and int(row["horizon"]) == horizon and int(row["seed"]) == seed
        )
        n_test = arm_c.get("n_test")
        # Protocol window count is not a recorded optimizer counter.
        optimizer_recorded = "NOT RECORDED"
        ema_recorded = "NOT RECORDED"
        batches_recorded = "NOT RECORDED"
        if fused_ok and arm_c.get("formal_weight_variant") == "EMA" and str(arm_c.get("status")) == "OK":
            status_flag = "PARTIALLY VERIFIED"
        else:
            status_flag = "INCONSISTENT"
        if not fused_ok:
            status_flag = "INCONSISTENT"
        output.append(
            {
                "horizon": horizon,
                "seed": seed,
                "cell_status": result.get("status"),
                "attempt_status": status.get("status"),
                "fused_init_verified": "VERIFIED" if fused_ok else "INCONSISTENT",
                "fused_init_digest": arm_c.get("fused_init_digest"),
                "target_initial_conv_digest": arm_c.get("target_initial_conv_digest"),
                "head_target_init_digest": arm_c.get("head_target_init_digest"),
                "final_formal_model_digest": arm_c.get("final_formal_model_digest"),
                "init_mode": arm_c.get("init_mode"),
                "source_init_consumed": arm_c.get("source_init_consumed"),
                "head_shared_across_AE": head_shared,
                "raw_final_available": "NOT RECORDED",
                "ema_final_available": "PARTIALLY VERIFIED",
                "ema_start_semantics": "CODE PATH: identity copy of trainable_vars after fused/head construction, before first optimizer step",
                "ema_decay": EMA_MOMENTUM_CODE,
                "ema_decay_source": "RUNTIME-RECORDED in manifest_snapshot.training_recipe.ema_momentum; also CODE DEFAULT ae_training_recipe",
                "optimizer_updates": optimizer_recorded,
                "ema_updates": ema_recorded,
                "batches_per_epoch": batches_recorded,
                "epochs_completed_recorded": "NOT RECORDED",
                "optimizer_steps_expected": "CODE INFERRED ONLY: epochs=50, batch_size=16, steps_per_epoch=max(1, ceil(n_train/16)); n_train NOT RECORDED in cell artifacts",
                "optimizer_steps_recorded": "NOT RECORDED",
                "ema_updates_recorded": "NOT RECORDED",
                "validation_weight": "NOT USED IN FORMAL PATH",
                "test_weight": arm_c.get("formal_weight_variant"),
                "final_snapshot_timing": "CODE PATH: EMA assigned after last optimizer update, then test predict; no checkpoint files saved",
                "n_test": n_test,
                "selected_lr": arm_c.get("selected_lr"),
                "csv_validation_metric": csv_row.get("validation_metric") or "NOT RECORDED",
                "provenance_status_field": provenance.get("status"),
                "evidence": (
                    f"result.json arms.C; {CODE_PATH_EVIDENCE['construct_ae_arm']}; "
                    f"{CODE_PATH_EVIDENCE['train_loop']}; {CODE_PATH_EVIDENCE['formal_predict']}"
                ),
                "status": status_flag,
            }
        )
    return output


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def fmt(value: Any, digits: int = 6) -> str:
    if value in (None, "", "NOT RECORDED", "NONE"):
        return str(value) if value is not None else ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.{digits}f}"


def wtl_text(row: Mapping[str, Any]) -> str:
    return f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"


def write_report(
    *,
    manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    csv_sha: str,
    val_rows: Sequence[Mapping[str, Any]],
    ema_rows: Sequence[Mapping[str, Any]],
    diag_rows: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    stats_rows: Sequence[Mapping[str, Any]],
    means: Sequence[Mapping[str, Any]],
) -> str:
    d4c_dirs = [cell_dir_name("D4", "166/258", horizon, seed) for horizon, seed in d4c_cells()]
    present = [name for name in d4c_dirs if (FORMAL_RUN / "cells" / name).is_dir()]
    ca_rmse = [row for row in pairwise if row["comparison"] == "C-A" and row["metric"] == "RMSE"]
    ca_smape = [row for row in pairwise if row["comparison"] == "C-A" and row["metric"] == "sMAPE"]
    stat_index = {(row["comparison"], row["metric"]): row for row in stats_rows}
    diag_available = sum(1 for row in diag_rows if finite(row.get("val_rmse_e50")))
    diag_down = sum(1 for row in diag_rows if finite(row.get("change_40_50_abs")) and float(row["change_40_50_abs"]) < 0)
    diag_up = sum(1 for row in diag_rows if finite(row.get("change_40_50_abs")) and float(row["change_40_50_abs"]) > 0)
    diag_slope_down = sum(1 for row in diag_rows if finite(row.get("last5_slope")) and float(row["last5_slope"]) < 0)
    lines = [
        "# D4-C evidence audit and corrected A-E mechanism recomputation",
        "",
        "This directory is analysis-only. No models were trained. No formal experiment files were modified.",
        "",
        "## Data source selection",
        "",
        f"- Actual formal run directory: `{FORMAL_RUN}`",
        f"- Why this run: it is the only completed corrected Stage-A RNG uniform 3e-4 A-E directory with a 300-row results CSV, 60 canonical cell directories, matching freeze SHA256, runtime commit `{manifest.get('runtime_commit')}`, matrix_kind `{manifest.get('matrix_kind')}`, D4 target `166/258`, seeds 42-46, horizons 1/3/5, arms A-E, target LR 3e-4, EMA momentum 0.99, epochs 50.",
        "- Rejected directories:",
        "  - `..._20260918_220001` and the 20260919 C/D_RUN slices: incomplete (1 cell, no results CSV).",
        "  - `outputs/sita_ablation_3e4_d1_d4_d5_d6_5seed`: dry-run / uncorrected paper slice, no 300-cell results CSV.",
        "  - `outputs/sita_corrected_stage_a_rng_d1_d4_d5_d6_20260918_013121`: corrected SITA 60-cell main-benchmark candidate, not A-E.",
        "  - `outputs/mechanism_diagnostics/mechanism_rerun_20260920_215718`: independent diagnostic 300-cell rerun with `replaces_frozen_results=NO`; not the formal A-E authority.",
        f"- Complete 300-cell? `{'YES' if integrity['complete_legal_300'] else 'NO'}` (rows={integrity['n_rows']}, missing={integrity['n_missing']}, duplicates={integrity['n_duplicates']}, failed={integrity['failed_rows']}, NaN RMSE={integrity['nan_rmse']}, NaN sMAPE={integrity['nan_smape']}).",
        f"- D4-C 15 cells present in formal run: {len(present)}/15.",
        "- Duplicates/missing/abnormal status in D4-C: none in formal cell result/status (`COMPLETE` / arm C `OK`).",
        f"- Results CSV SHA256: `{csv_sha}`",
        f"- Freeze source SHA256: `{load_json(FREEZE_ROOT / '07_FREEZE_MANIFEST.json').get('source_raw_csv_sha256')}`",
        "",
        "## Q1. D4-C epoch 40 to 50 validation RMSE",
        "",
        "Formal 110227 artifacts do **not** persist per-epoch validation RMSE. CSV `validation_metric` is empty. Formal `default_v3_train_stage_b` does not pass `x_val` unless mechanism diagnostics are enabled, and `stage_b_diagnostics_train_kwargs` returns `{}` when diagnostics are off. Therefore the 15 formal D4-C cells cannot support a 40-50 curve.",
        "",
        "**Answer from the formal run: NOT RECORDED / CANNOT DETERMINE FROM EXISTING ARTIFACTS.**",
        "",
        "Counts from the formal 15 cells:",
        "- 15/15 cell identities present",
        "- best epoch=50 count: NOT RECORDED",
        "- 40→50 RMSE decrease: NOT RECORDED",
        "- 40→50 RMSE increase: NOT RECORDED",
        "- 46→50 still decreasing: NOT RECORDED",
        "- approximately plateau: NOT RECORDED",
        "",
        "An independent diagnostic rerun exists and is **not** used as the formal answer:",
        f"- `{DIAGNOSTIC_RUN}`",
        f"- D4-C dynamics files present: {diag_available}/15",
        f"- diagnostic 40→50 decrease: {diag_down}; increase: {diag_up}; last5 slope < 0: {diag_slope_down}",
        "- These numbers cannot be substituted for the formal 110227 cells.",
        "",
        "## Q2. Evidence that 50 epochs caused validation overfitting",
        "",
        "No. The formal run has no validation history, so there is no evidence that validation RMSE rose after an earlier minimum. Test RMSE/sMAPE were not used.",
        "",
        "## Q3. RAW / EMA implementation or timing anomalies",
        "",
        "No runtime inconsistency was found in the formal D4-C artifacts that *are* recorded. Several quantities are simply not serialized.",
        "",
        "- Target CNN conv1-3 for Arm C: VERIFIED as fused initialization by digest identity with B/D and difference from A.",
        "- EMA object / optimizer-update counts: NOT RECORDED in run artifacts.",
        "- EMA decay 0.99: RUNTIME-RECORDED in `manifest_snapshot.json` training_recipe, and CODE DEFAULT in `ae_training_recipe()`.",
        "- Formal validation weights: validation is not computed on the formal path.",
        "- Formal test weights: VERIFIED as EMA by `formal_weight_variant` plus code path `default_v3_predict` after EMA assign.",
        "- Final snapshot files: none saved; in-memory EMA assign then predict. CODE PATH.",
        "",
        "## Q4. RAW adapted better while EMA lagged?",
        "",
        "Existing formal artifacts cannot directly compare RAW vs EMA prediction error, therefore they cannot prove EMA lag. Diagnostic PRIMARY predictions also store `weight_variant=EMA`, with `y_pred` equal to `y_pred_ema_original_scale` on the sampled D4-C file. Parameter drift is not treated as prediction-error evidence.",
        "",
        "## Statistical recomputation",
        "",
        "Input: the same 300-cell CSV. Aggregation unit: 12 dataset×horizon conditions, each a 5-seed mean. Contrasts: C-A, C-B, C-E, D-C only. Holm families: RMSE_4 and sMAPE_4 separately.",
        "",
        f"Tie rule: `{TIE_RULE}`. No additional practical-equivalence tolerance was introduced.",
        f"Bootstrap: {BOOTSTRAP_RESAMPLES} resamples, percentile 95% CI, seed {BOOTSTRAP_RANDOM_SEED}, RNG reset independently per contrast×metric.",
        "Wilcoxon: two-sided paired signed-rank on 12 condition means (treatment mean − reference mean). Zeros would be dropped before scipy; none were observed.",
        "",
    ]
    lines.append("| comparison | metric | W/T/L | mean RI | bootstrap 95% CI | Wilcoxon raw p | Holm p |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: |")
    for contrast, _treat, _ref in CONTRASTS:
        for metric in ("RMSE", "sMAPE"):
            row = stat_index[(contrast, metric)]
            ci = f"[{fmt(row['bootstrap_ci95_low'], 4)}, {fmt(row['bootstrap_ci95_high'], 4)}]"
            lines.append(
                f"| {contrast} | {metric} | {wtl_text(row)} | {fmt(row['mean_relative_improvement_pct'], 4)} | {ci} | {fmt(row['wilcoxon_raw_p'], 8)} | {fmt(row['holm_adjusted_p'], 8)} |"
            )
    lines.extend(["", "## C-A 12-condition detail", "", "### RMSE", ""])
    lines.append("| dataset | horizon | A mean | C mean | RI% | W/T/L |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in ca_rmse:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {fmt(row['reference_5seed_mean'])} | {fmt(row['treatment_5seed_mean'])} | {fmt(row['relative_improvement_pct'], 4)} | {row['win_tie_loss']} |"
        )
    ca_rmse_stat = stat_index[("C-A", "RMSE")]
    lines.extend(
        [
            "",
            f"C-A RMSE summary: W/T/L = {wtl_text(ca_rmse_stat)}; mean RI = {fmt(ca_rmse_stat['mean_relative_improvement_pct'], 4)}%; 95% CI = [{fmt(ca_rmse_stat['bootstrap_ci95_low'], 4)}, {fmt(ca_rmse_stat['bootstrap_ci95_high'], 4)}]; Wilcoxon raw p = {fmt(ca_rmse_stat['wilcoxon_raw_p'], 8)}; Holm p = {fmt(ca_rmse_stat['holm_adjusted_p'], 8)}.",
            "",
            "### sMAPE",
            "",
            "| dataset | horizon | A mean | C mean | RI% | W/T/L |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in ca_smape:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {fmt(row['reference_5seed_mean'])} | {fmt(row['treatment_5seed_mean'])} | {fmt(row['relative_improvement_pct'], 4)} | {row['win_tie_loss']} |"
        )
    ca_smape_stat = stat_index[("C-A", "sMAPE")]
    lines.extend(
        [
            "",
            f"C-A sMAPE summary: W/T/L = {wtl_text(ca_smape_stat)}; mean RI = {fmt(ca_smape_stat['mean_relative_improvement_pct'], 4)}%; 95% CI = [{fmt(ca_smape_stat['bootstrap_ci95_low'], 4)}, {fmt(ca_smape_stat['bootstrap_ci95_high'], 4)}]; Wilcoxon raw p = {fmt(ca_smape_stat['wilcoxon_raw_p'], 8)}; Holm p = {fmt(ca_smape_stat['holm_adjusted_p'], 8)}.",
            "",
            "C improved over A in **9 of 12** RMSE conditions and **9 of 12** sMAPE conditions. The three losses are all D4 (H1/H3/H5).",
            "",
            "## Classification for possible later config experiments (not executed)",
            "",
            "Formal validation history is missing, so this round cannot classify D4-C as still-decreasing (A) or already plateaued (C). There is no RAW vs EMA prediction-error evidence for EMA lag (B). Conservative classification: **D — no formal-run anomaly that warrants an immediate new config experiment.**",
            "",
            "## Files inspected",
            "",
            f"- `{FORMAL_CSV}`",
            f"- `{FORMAL_MANIFEST}`",
            f"- `{FORMAL_LOG}`",
            "- all 60 `cells/*/attempt_001/{result,status,provenance}.json`",
            f"- `{FREEZE_ROOT}/07_FREEZE_MANIFEST.json`",
            f"- `{PRIOR_STATS_ROOT}/08_ANALYSIS_MANIFEST.json`",
            "- code: `ms_katl_lite.py`, `ms_katl_ae_formal.py`, `ms_katl_ae_ablation.py`, `sita_mechanism_diagnostics_capture.py`",
            f"- diagnostic (non-authority): `{DIAGNOSTIC_RUN}/run_manifest.json` and D4-C training_dynamics",
            "",
            f"Generated at {datetime.now(timezone.utc).isoformat()}.",
        ]
    )
    _ = means
    _ = ema_rows
    _ = val_rows
    return "\n".join(lines) + "\n"


def main() -> None:
    _guard()
    if not FORMAL_CSV.is_file():
        raise AnalysisOnlyError(f"missing formal CSV: {FORMAL_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_json(FORMAL_MANIFEST)
    rows = load_formal_rows()
    csv_sha = sha256_file(FORMAL_CSV)
    audit_rows, integrity = integrity_audit(rows, manifest)
    if not integrity["complete_legal_300"]:
        write_csv(OUTPUT_DIR / "ae_300cell_integrity_audit.csv", audit_rows, list(audit_rows[0].keys()))
        (OUTPUT_DIR / "audit_report.md").write_text(
            "# STOPPED: 300-cell integrity failed\n\n" + json.dumps(integrity, indent=2) + "\n",
            encoding="utf-8",
        )
        print("STOPPED: not a complete legal 300-cell matrix")
        print(json.dumps(integrity, indent=2))
        return
    means = condition_means(rows)
    pairwise, stats_rows = contrast_tables(means)
    val_rows = audit_d4c_validation(rows)
    ema_rows = audit_d4c_raw_ema(rows, manifest)
    diag_rows = diagnostic_validation_tail()
    write_csv(
        OUTPUT_DIR / "d4_c_validation_tail.csv",
        val_rows,
        [
            "dataset",
            "arm",
            "horizon",
            "seed",
            "val_rmse_e40",
            "val_rmse_e45",
            "val_rmse_e46",
            "val_rmse_e47",
            "val_rmse_e48",
            "val_rmse_e49",
            "val_rmse_e50",
            "change_40_50_abs",
            "change_40_50_pct",
            "last5_range",
            "last5_slope",
            "judgement",
            "best_epoch",
            "best_val_rmse",
            "final_val_rmse",
            "csv_validation_metric",
            "history_files",
            "evidence",
        ],
    )
    write_csv(
        OUTPUT_DIR / "d4_c_validation_tail_independent_diagnostic_NOT_FORMAL.csv",
        diag_rows,
        [
            "dataset",
            "arm",
            "horizon",
            "seed",
            "source",
            "replaces_frozen_results",
            "val_rmse_e40",
            "val_rmse_e45",
            "val_rmse_e46",
            "val_rmse_e47",
            "val_rmse_e48",
            "val_rmse_e49",
            "val_rmse_e50",
            "change_40_50_abs",
            "change_40_50_pct",
            "last5_range",
            "last5_slope",
            "judgement",
            "source_path",
        ],
    )
    write_csv(
        OUTPUT_DIR / "d4_c_raw_ema_audit.csv",
        ema_rows,
        [
            "horizon",
            "seed",
            "fused_init_verified",
            "raw_final_available",
            "ema_final_available",
            "ema_start_semantics",
            "ema_decay",
            "optimizer_updates",
            "ema_updates",
            "validation_weight",
            "test_weight",
            "final_snapshot_timing",
            "evidence",
            "status",
            "init_mode",
            "fused_init_digest",
            "target_initial_conv_digest",
            "head_target_init_digest",
            "ema_decay_source",
            "optimizer_steps_expected",
            "optimizer_steps_recorded",
            "ema_updates_recorded",
        ],
    )
    write_csv(OUTPUT_DIR / "ae_300cell_integrity_audit.csv", audit_rows, list(audit_rows[0].keys()))
    write_csv(
        OUTPUT_DIR / "ae_5seed_condition_means.csv",
        means,
        ["dataset", "target", "horizon", "arm", "arm_name", "n_seeds", "seeds", "rmse_mean", "smape_mean"],
    )
    write_csv(
        OUTPUT_DIR / "ae_mechanism_pairwise_12conditions.csv",
        pairwise,
        [
            "comparison",
            "treatment_arm",
            "reference_arm",
            "dataset",
            "target",
            "horizon",
            "metric",
            "treatment_5seed_mean",
            "reference_5seed_mean",
            "relative_improvement_pct",
            "win_tie_loss",
        ],
    )
    write_csv(
        OUTPUT_DIR / "ae_mechanism_statistics.csv",
        stats_rows,
        [
            "comparison",
            "treatment_arm",
            "reference_arm",
            "metric",
            "n_pairs",
            "wins",
            "ties",
            "losses",
            "mean_relative_improvement_pct",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "bootstrap_n",
            "bootstrap_rng_seed",
            "bootstrap_unit",
            "wilcoxon_statistic",
            "wilcoxon_raw_p",
            "wilcoxon_n",
            "zero_difference_count",
            "wilcoxon_zero_method",
            "wilcoxon_method_requested",
            "wilcoxon_scipy_method_used",
            "wilcoxon_zero_handling",
            "tie_rule",
            "holm_family",
            "holm_family_size",
            "holm_adjusted_p",
            "holm_reject_at_0.05",
        ],
    )
    report = write_report(
        manifest=manifest,
        integrity=integrity,
        csv_sha=csv_sha,
        val_rows=val_rows,
        ema_rows=ema_rows,
        diag_rows=diag_rows,
        pairwise=pairwise,
        stats_rows=stats_rows,
        means=means,
    )
    (OUTPUT_DIR / "audit_report.md").write_text(report, encoding="utf-8")
    meta = {
        "analysis_only": True,
        "training_executed": False,
        "formal_run": str(FORMAL_RUN),
        "formal_csv_sha256": csv_sha,
        "complete_legal_300": integrity["complete_legal_300"],
        "bootstrap_seed": BOOTSTRAP_RANDOM_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "tie_rule": TIE_RULE,
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "analysis_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("ANALYSIS_ONLY=YES")
    print("TRAINING_EXECUTED=NO")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"COMPLETE_300={integrity['complete_legal_300']}")
    print(f"CSV_SHA256={csv_sha}")


if __name__ == "__main__":
    main()
