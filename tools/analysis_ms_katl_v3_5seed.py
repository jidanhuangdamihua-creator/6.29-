#!/usr/bin/env python3
"""Fail-closed paper tables for frozen V3-without seeds 42-46."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASETS = ("D1", "D4", "D5", "D6")
TARGETS = {
    "D1": "1/10",
    "D4": "166/311",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
HORIZONS = (1, 3, 5)
OLD_SEEDS = (42, 43, 44)
NEW_SEEDS = (45, 46)
SEEDS = OLD_SEEDS + NEW_SEEDS
V3_METHOD = "FUSED_TARGET_ONLY_V3"
BASELINES = (
    "BL1_HistoricalMean",
    "BL2_MovingAverage",
    "BL3_LightGBM",
    "BL4_LSTM",
    "BL5_SeasonalNaive",
)
LEARNING_BASELINES = ("BL3_LightGBM", "BL4_LSTM")
METHODS = BASELINES + (V3_METHOD,)
DISPLAY = {
    "BL1_HistoricalMean": "Historical Mean",
    "BL2_MovingAverage": "Moving Average",
    "BL3_LightGBM": "LightGBM",
    "BL4_LSTM": "LSTM",
    "BL5_SeasonalNaive": "Seasonal Naive",
    V3_METHOD: "V3",
}
KEY = ["dataset", "target", "horizon", "seed"]
METHOD_KEY = KEY + ["method"]
TIE_RELATIVE = 0.001
EXPECTED_OLD = 36
EXPECTED_NEW = 24
EXPECTED_FINAL = 60
EXPECTED_BASELINES = 300


class FailClosed(RuntimeError):
    pass


def require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FailClosed(f"{label} missing columns: {missing}")


def expected_keys(seeds: tuple[int, ...]) -> set[tuple[Any, ...]]:
    return {
        (dataset, TARGETS[dataset], horizon, seed)
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in seeds
    }


def expected_baseline_identities() -> set[tuple[Any, ...]]:
    return {
        (*key, method)
        for key in expected_keys(SEEDS)
        for method in BASELINES
    }


def normalize_v3(frame: pd.DataFrame, seeds: tuple[int, ...], label: str) -> pd.DataFrame:
    require(
        frame,
        [
            "status", "arm", "dataset", "target_identity", "mode", "horizon", "seed",
            "method", "rmse_original", "smape_original", "metric_space", "run_status",
            "sales_min", "sales_max", "architecture_digest", "training_recipe",
        ],
        label,
    )
    out = frame.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip().str.upper()
    out["target"] = out["target_identity"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["mode"] = out["mode"].astype(str).str.strip().str.lower()
    out["arm"] = out["arm"].astype(str).str.strip().str.lower()
    out["status"] = out["status"].astype(str).str.strip().str.upper()
    out["run_status"] = out["run_status"].astype(str).str.strip().str.upper()
    out["metric_space"] = out["metric_space"].astype(str).str.strip()
    for column in ("horizon", "seed"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    for column in ("rmse_original", "smape_original", "sales_min", "sales_max"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    scope = out[
        out["dataset"].isin(DATASETS)
        & out["horizon"].isin(HORIZONS)
        & out["seed"].isin(seeds)
        & (out["method"] == V3_METHOD)
        & (out["arm"] == "v3")
    ].copy()
    if len(scope) != len(expected_keys(seeds)):
        raise FailClosed(f"{label} expected {len(expected_keys(seeds))} V3 rows, got {len(scope)}")
    if set(scope["mode"]) != {"without"}:
        raise FailClosed(f"{label} contains non-without mode: {sorted(set(scope['mode']))}")
    if set(scope["status"]) != {"OK"} or set(scope["run_status"]) != {"OK"}:
        raise FailClosed(f"{label} contains non-OK cells")
    if set(scope["metric_space"]) != {"original_sales_space"}:
        raise FailClosed(f"{label} metric-space mismatch")
    scope["expected_target"] = scope["dataset"].map(TARGETS)
    if (scope["target"] != scope["expected_target"]).any():
        raise FailClosed(f"{label} target mismatch")
    if scope.duplicated(KEY).any() or set(map(tuple, scope[KEY].itertuples(index=False, name=None))) != expected_keys(seeds):
        raise FailClosed(f"{label} missing or duplicate identities")
    values = scope[["rmse_original", "smape_original", "sales_min", "sales_max"]].to_numpy(float)
    if not np.isfinite(values).all() or (scope["sales_max"] <= scope["sales_min"]).any():
        raise FailClosed(f"{label} contains non-finite metrics or invalid scaler range")
    return scope


def load_v3(repo: Path, new_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_path = (
        repo / "outputs/runs/20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
        / "results/three_track_robustness_all_runs.csv"
    )
    new_path = new_root / "results/three_track_robustness_all_runs.csv"
    if not old_path.is_file() or not new_path.is_file():
        raise FailClosed(f"V3 source missing: old={old_path.is_file()} new={new_path.is_file()}")
    old = normalize_v3(pd.read_csv(old_path), OLD_SEEDS, "old V3")
    raw_new = pd.read_csv(new_path)
    if "mode" in raw_new and (raw_new["mode"].astype(str).str.lower() == "with").any():
        raise FailClosed("NEW_WITH_CELLS is not zero")
    new = normalize_v3(raw_new, NEW_SEEDS, "new V3")
    for column in ("source_mode", "fused_init_source_mode", "source_authority_identity"):
        require(new, [column], "new V3")
    if set(new["source_mode"].astype(str).str.lower()) != {"without"}:
        raise FailClosed("SOURCE_MODE is not uniformly without")
    if set(new["fused_init_source_mode"].astype(str).str.lower()) != {"without"}:
        raise FailClosed("FUSED_INIT_SOURCE_MODE is not uniformly without")
    if new["source_authority_identity"].astype(str).str.strip().eq("").any():
        raise FailClosed("new V3 source authority is blank")
    if set(old["architecture_digest"]) != set(new["architecture_digest"]):
        raise FailClosed("V3 architecture digest changed between old and new seeds")
    if set(old["training_recipe"].astype(str)) != set(new["training_recipe"].astype(str)):
        raise FailClosed("V3 training recipe changed between old and new seeds")
    combined = pd.concat([old, new], ignore_index=True)
    if len(combined) != EXPECTED_FINAL or combined.duplicated(KEY).any():
        raise FailClosed("final V3 matrix is not exactly 60 unique cells")
    ranges = combined.groupby("dataset")[["sales_min", "sales_max"]].nunique(dropna=False)
    if (ranges != 1).any().any():
        raise FailClosed("target scaler range changes within a dataset")
    combined["rmse"] = combined["rmse_original"].astype(float)
    combined["smape"] = combined["smape_original"].astype(float)
    combined["nrmse_range_percent"] = combined["rmse"] / (
        combined["sales_max"] - combined["sales_min"]
    ) * 100.0
    combined["method"] = V3_METHOD
    return old, new, combined


def load_baselines(repo: Path, v3: pd.DataFrame) -> pd.DataFrame:
    path = repo / "outputs/baselines/d1_d6_bl1_bl5_baselines_multiseed.csv"
    raw = pd.read_csv(path)
    require(
        raw,
        [
            "dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed",
            "rmse", "smape", "result_status", "rmse_metric_space", "smape_metric_space",
        ],
        "baseline",
    )
    b = raw.copy()
    b["dataset"] = b["dataset_id"].astype(str).str.strip().str.upper()
    b["target"] = b["target_entity_key"].astype(str).str.strip()
    b["scenario"] = b["scenario"].astype(str).str.strip().str.lower()
    b["method"] = b["method"].astype(str).str.strip()
    for column in ("horizon", "seed"):
        b[column] = pd.to_numeric(b[column], errors="coerce").astype("Int64")
    for column in ("rmse", "smape"):
        b[column] = pd.to_numeric(b[column], errors="coerce")
    b = b[
        b["dataset"].isin(DATASETS)
        & b["horizon"].isin(HORIZONS)
        & b["seed"].isin(SEEDS)
        & b["method"].isin(BASELINES)
        & (b["scenario"] == "without")
    ].copy()
    rows_before_target_filter = len(b)
    b["expected_target"] = b["dataset"].map(TARGETS)
    b = b[b["target"] == b["expected_target"]].copy()
    print(f"BASELINE_ROWS_BEFORE_TARGET_FILTER={rows_before_target_filter}")
    print(f"BASELINE_ROWS_AFTER_TARGET_FILTER={len(b)}")
    if len(b) != EXPECTED_BASELINES:
        raise FailClosed(f"baseline exact scope mismatch: rows={len(b)}")
    actual_identities = set(map(tuple, b[METHOD_KEY].itertuples(index=False, name=None)))
    if (
        b.duplicated(METHOD_KEY).any()
        or actual_identities != expected_baseline_identities()
        or set(b["result_status"].astype(str)) != {"confirmed_baseline"}
    ):
        raise FailClosed("baseline duplicate or status mismatch")
    per_dataset = b.groupby("dataset", sort=True).size().to_dict()
    if per_dataset != {dataset: 75 for dataset in DATASETS}:
        raise FailClosed(f"baseline per-dataset scope mismatch: {per_dataset}")
    method_counts = b.groupby(["dataset", "horizon", "seed"])["method"].nunique()
    if len(method_counts) != len(DATASETS) * len(HORIZONS) * len(SEEDS) or not (method_counts == 5).all():
        raise FailClosed("baseline method completeness mismatch")
    if set(b["rmse_metric_space"].astype(str)) != {"original_sales_space"}:
        raise FailClosed("baseline RMSE metric space mismatch")
    if set(b["smape_metric_space"].astype(str)) != {"original_sales_space"}:
        raise FailClosed("baseline sMAPE metric space mismatch")
    if not np.isfinite(b[["rmse", "smape"]].to_numpy(float)).all():
        raise FailClosed("baseline contains non-finite metrics")
    ranges = v3.groupby("dataset")[["sales_min", "sales_max"]].first()
    b = b.join(ranges, on="dataset")
    b["nrmse_range_percent"] = b["rmse"] / (b["sales_max"] - b["sales_min"]) * 100.0
    return b


def compare_label(v3: float, baseline: float) -> str:
    relative = abs(baseline - v3) / abs(baseline) if baseline != 0 else np.inf
    if relative < TIE_RELATIVE:
        return "TIE"
    return "WIN" if v3 < baseline else "LOSS"


def gain(v3: float, baseline: float) -> float:
    return (baseline - v3) / baseline * 100.0 if baseline != 0 else np.nan


def summary_stats(all_runs: pd.DataFrame) -> pd.DataFrame:
    grouped = all_runs.groupby(["dataset", "target", "horizon", "method"], sort=True)
    out = grouped.agg(
        seed_count=("seed", "nunique"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        rmse_median=("rmse", "median"), rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
        smape_mean=("smape", "mean"), smape_sd=("smape", "std"),
        smape_median=("smape", "median"), smape_min=("smape", "min"), smape_max=("smape", "max"),
        nrmse_range_percent_mean=("nrmse_range_percent", "mean"),
        nrmse_range_percent_sd=("nrmse_range_percent", "std"),
    ).reset_index()
    if set(out["seed_count"]) != {5}:
        raise FailClosed("a dataset x horizon x method group does not contain five seeds")
    return out


def condition_comparisons(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    ix = stats.set_index(["dataset", "target", "horizon", "method"])
    rows = []
    for dataset in DATASETS:
        target = TARGETS[dataset]
        for horizon in HORIZONS:
            v3 = float(ix.loc[(dataset, target, horizon, V3_METHOD), f"{metric}_mean"])
            values = {
                method: float(ix.loc[(dataset, target, horizon, method), f"{metric}_mean"])
                for method in BASELINES
            }
            for method, baseline in values.items():
                rows.append(
                    {
                        "dataset": dataset, "target": target, "horizon": horizon,
                        "metric": metric.upper(), "comparison": method,
                        "baseline_mean": baseline, "v3_mean": v3,
                        "gain_percent": gain(v3, baseline),
                        "outcome": compare_label(v3, baseline),
                    }
                )
            for name, pool in (
                ("BEST_LEARNING_BASELINE", LEARNING_BASELINES),
                ("BEST_ALL_BASELINE", BASELINES),
            ):
                best_method = min(pool, key=lambda method: values[method])
                baseline = values[best_method]
                rows.append(
                    {
                        "dataset": dataset, "target": target, "horizon": horizon,
                        "metric": metric.upper(), "comparison": name,
                        "best_method": best_method, "baseline_mean": baseline, "v3_mean": v3,
                        "gain_percent": gain(v3, baseline),
                        "outcome": compare_label(v3, baseline),
                    }
                )
    return pd.DataFrame(rows)


def win_loss_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, comparison), group in comparisons.groupby(["metric", "comparison"], sort=True):
        rows.append(
            {
                "metric": metric, "comparison": comparison,
                "v3_win": int((group["outcome"] == "WIN").sum()),
                "tie": int((group["outcome"] == "TIE").sum()),
                "v3_loss": int((group["outcome"] == "LOSS").sum()),
                "conditions": len(group), "tie_threshold_relative": TIE_RELATIVE,
            }
        )
    return pd.DataFrame(rows)


def ranks(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("rmse", "smape"):
        part = stats.copy()
        part["rank"] = part.groupby(["dataset", "horizon"])[f"{metric}_mean"].rank(
            method="average", ascending=True
        )
        for method, group in part.groupby("method", sort=False):
            rows.append(
                {
                    "method": method, "metric": metric.upper(),
                    "average_rank": float(group["rank"].mean()),
                    "median_rank": float(group["rank"].median()),
                    "rank1_conditions": int((group["rank"] == 1).sum()),
                    "conditions": len(group),
                }
            )
    return pd.DataFrame(rows)


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.6g} ± {sd:.6g}"


def paper_table(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    ix = stats.set_index(["dataset", "horizon", "method"])
    for dataset in DATASETS:
        for horizon in HORIZONS:
            row = {"Dataset": dataset, "Horizon": horizon}
            for method in METHODS:
                values = ix.loc[(dataset, horizon, method)]
                row[DISPLAY[method]] = format_mean_sd(
                    float(values[f"{metric}_mean"]), float(values[f"{metric}_sd"])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def learning_table(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ix = stats.set_index(["dataset", "horizon", "method"])
    for dataset in DATASETS:
        for horizon in HORIZONS:
            values = {
                method: ix.loc[(dataset, horizon, method)] for method in (*LEARNING_BASELINES, V3_METHOD)
            }
            best_rmse = min(LEARNING_BASELINES, key=lambda method: float(values[method]["rmse_mean"]))
            best_smape = min(LEARNING_BASELINES, key=lambda method: float(values[method]["smape_mean"]))
            rows.append(
                {
                    "Dataset": dataset, "H": horizon,
                    "LightGBM RMSE": format_mean_sd(float(values[LEARNING_BASELINES[0]]["rmse_mean"]), float(values[LEARNING_BASELINES[0]]["rmse_sd"])),
                    "LSTM RMSE": format_mean_sd(float(values[LEARNING_BASELINES[1]]["rmse_mean"]), float(values[LEARNING_BASELINES[1]]["rmse_sd"])),
                    "V3 RMSE": format_mean_sd(float(values[V3_METHOD]["rmse_mean"]), float(values[V3_METHOD]["rmse_sd"])),
                    "Best learning baseline RMSE": DISPLAY[best_rmse],
                    "V3 vs best learning baseline RMSE gain %": gain(float(values[V3_METHOD]["rmse_mean"]), float(values[best_rmse]["rmse_mean"])),
                    "LightGBM sMAPE": format_mean_sd(float(values[LEARNING_BASELINES[0]]["smape_mean"]), float(values[LEARNING_BASELINES[0]]["smape_sd"])),
                    "LSTM sMAPE": format_mean_sd(float(values[LEARNING_BASELINES[1]]["smape_mean"]), float(values[LEARNING_BASELINES[1]]["smape_sd"])),
                    "V3 sMAPE": format_mean_sd(float(values[V3_METHOD]["smape_mean"]), float(values[V3_METHOD]["smape_sd"])),
                    "Best learning baseline sMAPE": DISPLAY[best_smape],
                    "V3 vs best learning baseline sMAPE gain %": gain(float(values[V3_METHOD]["smape_mean"]), float(values[best_smape]["smape_mean"])),
                }
            )
    return pd.DataFrame(rows)


def stability(v3: pd.DataFrame, baselines: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[dict[str, Any]]]:
    rows = []
    anomalies = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            five = v3[(v3["dataset"] == dataset) & (v3["horizon"] == horizon)]
            three = five[five["seed"].isin(OLD_SEEDS)]
            row: dict[str, Any] = {"dataset": dataset, "target": TARGETS[dataset], "horizon": horizon}
            changed = False
            for metric in ("rmse", "smape"):
                mean3, mean5 = float(three[metric].mean()), float(five[metric].mean())
                row[f"3seed_{metric}_mean"] = mean3
                row[f"5seed_{metric}_mean"] = mean5
                row[f"{metric}_mean_shift_percent"] = (mean5 - mean3) / mean3 * 100.0
                for method in LEARNING_BASELINES:
                    b = baselines[
                        (baselines["dataset"] == dataset)
                        & (baselines["horizon"] == horizon)
                        & (baselines["method"] == method)
                    ]
                    label3 = compare_label(mean3, float(b[b["seed"].isin(OLD_SEEDS)][metric].mean()))
                    label5 = compare_label(mean5, float(b[metric].mean()))
                    row[f"3seed_{metric}_vs_{DISPLAY[method]}"] = label3
                    row[f"5seed_{metric}_vs_{DISPLAY[method]}"] = label5
                    changed = changed or label3 != label5
                values = five[metric].to_numpy(float)
                for seed in NEW_SEEDS:
                    value = float(five.loc[five["seed"] == seed, metric].iloc[0])
                    others = five.loc[five["seed"] != seed, metric].to_numpy(float)
                    sd = float(np.std(others, ddof=1))
                    score = abs(value - float(np.mean(others))) / sd if sd > 0 else (np.inf if value != others[0] else 0.0)
                    if score > 3.0:
                        anomalies.append({"dataset": dataset, "horizon": horizon, "metric": metric, "seed": seed, "leave_one_out_z": score})
            row["conclusion_changed"] = "YES" if changed else "NO"
            rows.append(row)
    frame = pd.DataFrame(rows)
    grouped = v3.groupby(["dataset", "horizon"])[["rmse", "smape"]].agg(["mean", "std"])
    rmse_cv = grouped[("rmse", "std")] / grouped[("rmse", "mean")] * 100.0
    smape_cv = grouped[("smape", "std")] / grouped[("smape", "mean")] * 100.0
    cv = {
        "rmse_mean_cv_percent": float(rmse_cv.mean()),
        "rmse_median_cv_percent": float(rmse_cv.median()),
        "rmse_max_cv_percent": float(rmse_cv.max()),
        "smape_mean_cv_percent": float(smape_cv.mean()),
        "smape_median_cv_percent": float(smape_cv.median()),
        "smape_max_cv_percent": float(smape_cv.max()),
    }
    return frame, cv, anomalies


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_tree_sha256(root: Path) -> tuple[str, int]:
    paths = sorted({*root.glob("cells/v3_*.json"), *root.glob("predictions/v3/**/*")})
    files = [path for path in paths if path.is_file()]
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def verify_frozen(new_root: Path) -> dict[str, Any]:
    record_path = new_root / "frozen_protection.json"
    if not record_path.is_file():
        raise FailClosed(f"missing frozen protection record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    selection = Path(record["selection_manifest"])
    robustness = Path(record["robustness_manifest"])
    old_root = Path(record["existing_v3_result_root"])
    tree_sha, tree_files = frozen_tree_sha256(old_root)
    checks = {
        "selection_manifest_unchanged": sha256_file(selection) == record["selection_manifest_sha256"],
        "robustness_manifest_unchanged": sha256_file(robustness) == record["robustness_manifest_sha256"],
        "seed42_44_results_unchanged": tree_sha == record["existing_v3_result_tree_sha256"],
        "tree_file_count_unchanged": tree_files == record["existing_v3_result_tree_file_count"],
    }
    if not all(checks.values()):
        raise FailClosed(f"frozen protection failed: {checks}")
    return {**checks, "current_tree_sha256": tree_sha, "current_tree_file_count": tree_files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper/v3_5seed"))
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    new_root = args.new_root if args.new_root.is_absolute() else repo / args.new_root
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    try:
        protected = verify_frozen(new_root)
        old, new, v3 = load_v3(repo, new_root)
        baselines = load_baselines(repo, v3)
        v3_out = v3[KEY + ["method", "rmse", "smape", "nrmse_range_percent", "sales_min", "sales_max", "architecture_digest"]].copy()
        all_runs = pd.concat(
            [v3_out, baselines[KEY + ["method", "rmse", "smape", "nrmse_range_percent", "sales_min", "sales_max"]]],
            ignore_index=True,
        )
        stats = summary_stats(all_runs)
        v3_summary = stats[stats["method"] == V3_METHOD].reset_index(drop=True)
        rmse_cmp = condition_comparisons(stats, "rmse")
        smape_cmp = condition_comparisons(stats, "smape")
        comparisons = pd.concat([rmse_cmp, smape_cmp], ignore_index=True)
        learning = comparisons[comparisons["comparison"].isin((*LEARNING_BASELINES, "BEST_LEARNING_BASELINE"))].copy()
        relative = comparisons.copy()
        wtl = win_loss_table(comparisons)
        rank = ranks(stats)
        shifts, cv, anomalies = stability(v3, baselines)
        main_rmse = paper_table(stats, "rmse")
        main_smape = paper_table(stats, "smape")
        learning_main = learning_table(stats)
        if out.exists() and any((out / name).exists() for name in ("v3_5seed_all_runs.csv", "v3_5seed_summary.json")):
            raise FailClosed(f"refusing to overwrite an existing paper result set: {out}")
        out.mkdir(parents=True, exist_ok=True)
        v3_out.sort_values(KEY).to_csv(out / "v3_5seed_all_runs.csv", index=False)
        v3_summary.to_csv(out / "v3_5seed_summary.csv", index=False)
        rmse_cmp.to_csv(out / "v3_vs_all_baselines_rmse.csv", index=False)
        smape_cmp.to_csv(out / "v3_vs_all_baselines_smape.csv", index=False)
        learning.to_csv(out / "v3_vs_learning_baselines.csv", index=False)
        relative.to_csv(out / "v3_relative_improvement.csv", index=False)
        wtl.to_csv(out / "v3_win_tie_loss.csv", index=False)
        rank.to_csv(out / "v3_average_rank.csv", index=False)
        shifts.to_csv(out / "v3_3seed_vs_5seed_stability.csv", index=False)
        main_rmse.to_csv(out / "v3_paper_main_rmse.csv", index=False)
        main_smape.to_csv(out / "v3_paper_main_smape.csv", index=False)
        learning_main.to_csv(out / "v3_paper_learning_baseline_table.csv", index=False)
        deterministic = {}
        for method in ("BL1_HistoricalMean", "BL2_MovingAverage", "BL5_SeasonalNaive"):
            unique = baselines[baselines["method"] == method].groupby(["dataset", "horizon"])[["rmse", "smape"]].nunique()
            deterministic[method] = bool((unique == 1).all().all())
        payload = {
            "contract": {
                "method": V3_METHOD, "mode": "without", "datasets": DATASETS,
                "horizons": HORIZONS, "seeds": SEEDS, "new_seeds": NEW_SEEDS,
                "lr_rescreening": False, "target_reselection": False,
                "with_runs": 0, "rmse_space": "original_sales_space",
                "smape_space": "original_sales_space", "std_ddof": 1,
                "tie_threshold_relative": TIE_RELATIVE,
                "cross_dataset_raw_rmse_averaging": False,
            },
            "completeness": {
                "expected_new_cells": EXPECTED_NEW, "actual_new_cells": len(new),
                "expected_final_v3_cells": EXPECTED_FINAL, "final_v3_cells": len(v3),
                "final_unique_v3_cells": int(v3[KEY].drop_duplicates().shape[0]),
                "missing_v3_cells": 0, "duplicate_v3_cells": 0,
                "expected_baseline_rows": EXPECTED_BASELINES,
                "baseline_rows": len(baselines),
                "baseline_unique_rows": int(baselines[METHOD_KEY].drop_duplicates().shape[0]),
                "baseline_duplicate_rows": int(baselines.duplicated(METHOD_KEY).sum()),
                "baseline_target_scope_match": True,
                "baseline_method_completeness": True,
            },
            "frozen_protection": protected,
            "cv": cv,
            "conclusion_changes": int((shifts["conclusion_changed"] == "YES").sum()),
            "new_seed_anomalies_leave_one_out_z_gt_3": anomalies,
            "deterministic_baseline_seed_records_repeated": deterministic,
            "win_tie_loss": wtl.to_dict("records"),
            "average_rank": rank.to_dict("records"),
        }
        (out / "v3_5seed_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=lambda value: value.item()) + "\n",
            encoding="utf-8",
        )
        print(f"ACTUAL_NEW_CELLS={len(new)}")
        print(f"FINAL_V3_CELLS={len(v3)}")
        print(f"FINAL_UNIQUE_V3_CELLS={v3[KEY].drop_duplicates().shape[0]}")
        print("MISSING_V3_CELLS=0")
        print("DUPLICATE_V3_CELLS=0")
        print("NEW_WITH_CELLS=0")
        print(f"BASELINE_ROWS={len(baselines)}")
        print(f"BASELINE_UNIQUE_ROWS={baselines[METHOD_KEY].drop_duplicates().shape[0]}")
        print("BASELINE_TARGET_SCOPE_MATCH=YES")
        print("BASELINE_METHOD_COMPLETENESS=PASS")
        print(f"PAPER_OUTPUT_DIR={out}")
        return 0
    except FailClosed as exc:
        print("FAIL_CLOSED=YES")
        print(f"FAIL_CLOSED_REASON={exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
