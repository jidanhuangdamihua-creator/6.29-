#!/usr/bin/env python3
"""Offline, fail-closed report builder for the frozen MS-KATL V3 benchmark."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


FINAL_METHOD = "FUSED_TARGET_ONLY_V3"
DATASETS = ["D1", "D2", "D3", "D4", "D5", "D6"]
TARGETS = {
    "D1": "1/10",
    "D2": "1/10",
    "D3": "10",
    "D4": "166/311",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
HORIZONS = [1, 3, 5]
SEEDS = [42, 43, 44]
BASELINES = [
    "BL1_HistoricalMean",
    "BL2_MovingAverage",
    "BL3_LightGBM",
    "BL4_LSTM",
    "BL5_SeasonalNaive",
]
METHODS = [FINAL_METHOD, *BASELINES]
METHOD_ORDER = {method: i for i, method in enumerate(METHODS)}
KEY = ["dataset", "target", "horizon", "seed"]
METHOD_KEY = [*KEY, "method"]
SCENARIO = "without"
TIE_TOLERANCE = 1e-6
EXPECTED_V3_CELLS = 54
EXPECTED_BASELINE_ROWS = 270


class FailClosed(RuntimeError):
    pass


def text_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def keyset(frame: pd.DataFrame, columns: Sequence[str]) -> set[Tuple[object, ...]]:
    return set(map(tuple, frame[list(columns)].itertuples(index=False, name=None)))


def key_string(key: Tuple[object, ...]) -> str:
    return "|".join(str(x) for x in key)


def require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FailClosed(f"{label} missing columns: {missing}")


def expected_keys() -> pd.DataFrame:
    return pd.DataFrame(
        [(d, TARGETS[d], h, s) for d in DATASETS for h in HORIZONS for s in SEEDS],
        columns=KEY,
    )


def find_baseline_source(directory: Path) -> Tuple[Path, List[Path]]:
    required = {"target_entity_key", "scenario", "horizon", "seed", "method", "rmse", "smape"}
    candidates = []
    for path in sorted(directory.glob("*multiseed*.csv")):
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
            if not required.issubset(header):
                continue
            dataset_column = "dataset" if "dataset" in header else "dataset_id" if "dataset_id" in header else None
            if dataset_column is None:
                continue
            pairs = pd.read_csv(path, usecols=[dataset_column, "target_entity_key"])
            if set(text_col(pairs[dataset_column]).str.upper()) == set(DATASETS):
                candidates.append(path)
        except Exception:
            continue
    if len(candidates) != 1:
        print("BASELINE_SOURCE_AMBIGUOUS=YES")
        print("BASELINE_SOURCE_CANDIDATES=" + json.dumps([str(p) for p in candidates], ensure_ascii=False))
        raise FailClosed(f"formal baseline source is not unique: {candidates}")
    return candidates[0], candidates


def load_sources(repo: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    v3_root = repo / "outputs/runs/20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
    v3_path = v3_root / "results/three_track_robustness_all_runs.csv"
    paired_path = v3_root / "results/three_track_paired_by_cell.csv"
    baseline_path, candidates = find_baseline_source(repo / "outputs/baselines")
    if not v3_path.is_file():
        raise FailClosed(f"missing frozen V3 result source: {v3_path}")
    v3_raw = pd.read_csv(v3_path)
    require(v3_raw, ["status", "dataset", "target_identity", "mode", "horizon", "seed", "method", "rmse_original", "smape_original", "metric_space", "run_status"], "V3 source")
    v3 = v3_raw.copy()
    v3["dataset"] = text_col(v3["dataset"]).str.upper()
    v3["target"] = text_col(v3["target_identity"])
    v3["mode"] = text_col(v3["mode"])
    v3["method"] = text_col(v3["method"])
    v3["status"] = text_col(v3["status"])
    v3["run_status"] = text_col(v3["run_status"])
    v3["metric_space"] = text_col(v3["metric_space"])
    v3["horizon"] = pd.to_numeric(v3["horizon"], errors="coerce").astype("Int64")
    v3["seed"] = pd.to_numeric(v3["seed"], errors="coerce").astype("Int64")
    v3 = v3[
        (v3["method"] == FINAL_METHOD)
        & (v3["status"] == "OK")
        & (v3["run_status"] == "OK")
        & (v3["mode"] == SCENARIO)
        & v3["dataset"].isin(DATASETS)
        & v3["horizon"].isin(HORIZONS)
        & v3["seed"].isin(SEEDS)
    ].copy()
    v3["expected_target"] = v3["dataset"].map(TARGETS)
    if (v3["target"] != v3["expected_target"]).any():
        raise FailClosed("V3 target mismatch")
    v3["rmse"] = pd.to_numeric(v3["rmse_original"], errors="coerce")
    v3["smape"] = pd.to_numeric(v3["smape_original"], errors="coerce")
    if len(v3) != EXPECTED_V3_CELLS:
        raise FailClosed(f"V3_EXPECTED_CELLS={EXPECTED_V3_CELLS} but selected {len(v3)}")
    if v3.duplicated(KEY).any():
        raise FailClosed("V3 duplicate keys: " + json.dumps(v3.loc[v3.duplicated(KEY, keep=False), KEY].to_dict("records")))
    if set(v3["metric_space"]) != {"original_sales_space"}:
        raise FailClosed(f"V3 metric-space mismatch: {sorted(set(v3['metric_space']))}")
    if not np.isfinite(v3[["rmse", "smape"]].to_numpy(dtype=float)).all():
        raise FailClosed("V3 non-finite metric")

    baseline_raw = pd.read_csv(baseline_path)
    require(baseline_raw, ["target_entity_key", "scenario", "horizon", "seed", "method", "rmse", "smape", "result_status", "rmse_metric_space", "smape_metric_space"], "baseline source")
    baseline = baseline_raw.copy()
    dataset_column = "dataset" if "dataset" in baseline.columns else "dataset_id" if "dataset_id" in baseline.columns else None
    if dataset_column is None:
        raise FailClosed("baseline source missing dataset or dataset_id")
    baseline["dataset"] = text_col(baseline[dataset_column]).str.upper()
    baseline["target"] = text_col(baseline["target_entity_key"])
    baseline["scenario"] = text_col(baseline["scenario"])
    baseline["method"] = text_col(baseline["method"])
    baseline["result_status"] = text_col(baseline["result_status"])
    baseline["rmse_metric_space"] = text_col(baseline["rmse_metric_space"])
    baseline["smape_metric_space"] = text_col(baseline["smape_metric_space"])
    baseline["horizon"] = pd.to_numeric(baseline["horizon"], errors="coerce").astype("Int64")
    baseline["seed"] = pd.to_numeric(baseline["seed"], errors="coerce").astype("Int64")
    baseline["rmse"] = pd.to_numeric(baseline["rmse"], errors="coerce")
    baseline["smape"] = pd.to_numeric(baseline["smape"], errors="coerce")
    b = baseline[
        baseline["dataset"].isin(DATASETS)
        & baseline["target"].isin(list(TARGETS.values()))
        & baseline["horizon"].isin(HORIZONS)
        & baseline["seed"].isin(SEEDS)
        & (baseline["scenario"] == SCENARIO)
        & baseline["method"].isin(BASELINES)
    ].copy()
    b["expected_target"] = b["dataset"].map(TARGETS)
    if (b["target"] != b["expected_target"]).any():
        raise FailClosed("baseline target mismatch in exact scope")
    if len(b) != EXPECTED_BASELINE_ROWS:
        raise FailClosed(f"MATCHED_BASELINE_ROWS expected {EXPECTED_BASELINE_ROWS}, source has {len(b)}")
    if b.duplicated(METHOD_KEY).any():
        raise FailClosed("baseline duplicate method keys")
    if set(b["method"]) != set(BASELINES):
        raise FailClosed("baseline method missing")
    if set(b["result_status"]) != {"confirmed_baseline"}:
        raise FailClosed(f"baseline status mismatch: {sorted(set(b['result_status']))}")
    if set(b["rmse_metric_space"]) != {"original_sales_space"} or set(b["smape_metric_space"]) != {"original_sales_space"}:
        raise FailClosed("baseline metric-space mismatch")
    if not np.isfinite(b[["rmse", "smape"]].to_numpy(dtype=float)).all():
        raise FailClosed("baseline non-finite metric")

    v3_keys = keyset(v3, KEY)
    b_keys = keyset(b, KEY)
    missing = sorted(v3_keys - b_keys)
    extra = sorted(b_keys - v3_keys)
    if missing or extra:
        raise FailClosed(f"exact key mismatch; missing={missing}; extra={extra}")
    expected_method_keys = {
        (d, TARGETS[d], h, s, m)
        for d, _target, h, s in expected_keys().itertuples(index=False, name=None)
        for m in BASELINES
    }
    actual_method_keys = keyset(b, METHOD_KEY)
    missing_methods = sorted(expected_method_keys - actual_method_keys)
    if missing_methods:
        raise FailClosed("method missing: " + json.dumps([key_string(k) for k in missing_methods]))

    v3_out = v3[["dataset", "target", "horizon", "seed", "rmse", "smape"]].copy()
    v3_out["method"] = FINAL_METHOD
    v3_out["metric_space"] = "original_sales_space"
    v3_out["scenario"] = SCENARIO
    v3_out["source_type"] = "frozen_v3"
    v3_out["result_status"] = "confirmed_v3_frozen"
    b_out = b[["dataset", "target", "horizon", "seed", "method", "rmse", "smape"]].copy()
    b_out["metric_space"] = "original_sales_space"
    b_out["scenario"] = SCENARIO
    b_out["source_type"] = "formal_baseline"
    b_out["result_status"] = "confirmed_baseline"
    matched = pd.concat([v3_out, b_out], ignore_index=True)
    matched["horizon"] = matched["horizon"].astype(int)
    matched["seed"] = matched["seed"].astype(int)
    matched["order"] = matched["method"].map(METHOD_ORDER)
    matched = matched.sort_values(KEY + ["order"]).drop(columns=["order"]).reset_index(drop=True)
    integrity = {
        "v3_frozen_root": str(v3_root),
        "v3_results_file": str(v3_path),
        "v3_paired_file_present": paired_path.is_file(),
        "v3_source_rows_read": len(v3_raw),
        "v3_selected_rows": len(v3),
        "v3_expected_cells": EXPECTED_V3_CELLS,
        "baseline_source_file": str(baseline_path),
        "baseline_source_candidates_considered": [str(p) for p in candidates],
        "baseline_source_rows_read": len(baseline_raw),
        "baseline_scope_rows": len(b),
        "baseline_expected_rows": EXPECTED_BASELINE_ROWS,
        "matched_v3_cells": len(v3_keys),
        "matched_baseline_rows": len(b),
        "missing_keys": [key_string(k) for k in missing],
        "extra_keys": [key_string(k) for k in extra],
        "duplicate_v3_keys": int(v3.duplicated(KEY).sum()),
        "duplicate_baseline_method_keys": int(b.duplicated(METHOD_KEY).sum()),
    }
    return matched, integrity


def stats_by_condition(matched: pd.DataFrame) -> pd.DataFrame:
    out = matched.groupby(["dataset", "target", "horizon", "method"])[["rmse", "smape"]].agg(["mean", "std", "median"]).reset_index()
    out.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else str(c) for c in out.columns]
    return out


def improve(base: float, v3: float) -> float:
    return (base - v3) / base * 100 if base != 0 else float("nan")


def label(value: float) -> str:
    return "TIE" if abs(value) <= TIE_TOLERANCE * 100 else ("WIN" if value > 0 else "LOSS")


def main_tables(stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ix = stats.set_index(["dataset", "target", "horizon", "method"])
    rmse_rows, smape_rows, ranks = [], [], []
    for d in DATASETS:
        for h in HORIZONS:
            t = TARGETS[d]
            vals = {}
            for m in METHODS:
                r = ix.loc[(d, t, h, m)]
                vals[m] = {f"{metric}_{k}": float(r[f"{metric}_{k}"]) for metric in ["rmse", "smape"] for k in ["mean", "std", "median"]}
            for metric in ["rmse", "smape"]:
                ordered = sorted(METHODS, key=lambda m: (vals[m][f"{metric}_mean"], METHOD_ORDER[m]))
                for rank, m in enumerate(ordered, 1):
                    ranks.append({"DATASET": d, "TARGET": t, "HORIZON": h, "METRIC": metric.upper(), "METHOD": m, "VALUE_MEAN": vals[m][f"{metric}_mean"], "RANK": rank})
            rb = sorted(BASELINES, key=lambda m: (vals[m]["rmse_mean"], METHOD_ORDER[m]))
            sb = sorted(BASELINES, key=lambda m: (vals[m]["smape_mean"], METHOD_ORDER[m]))
            rr = {"DATASET": d, "TARGET": t, "HORIZON": h}
            sr = {"DATASET": d, "TARGET": t, "HORIZON": h}
            for m in METHODS:
                short = "V3" if m == FINAL_METHOD else m.split("_")[0]
                for metric, row in [("RMSE", rr), ("SMAPE", sr)]:
                    for k in ["MEAN", "STD", "MEDIAN"]:
                        row[f"{short}_{metric}_{k}"] = vals[m][f"{metric.lower()}_{k.lower()}"]
            rbest, rworst = vals[rb[0]]["rmse_mean"], vals[rb[-1]]["rmse_mean"]
            sbest, sworst = vals[sb[0]]["smape_mean"], vals[sb[-1]]["smape_mean"]
            rr.update({"BEST_BL": rb[0], "BEST_BL_RMSE_MEAN": rbest, "SECOND_BEST_BL": rb[1], "SECOND_BEST_BL_RMSE_MEAN": vals[rb[1]]["rmse_mean"], "WORST_BL": rb[-1], "WORST_BL_RMSE_MEAN": rworst, "V3_VS_BEST_BL_PCT": improve(rbest, vals[FINAL_METHOD]["rmse_mean"]), "V3_VS_WORST_BL_PCT": improve(rworst, vals[FINAL_METHOD]["rmse_mean"]), "V3_RANK_AMONG_6": [*sorted(METHODS, key=lambda m: (vals[m]["rmse_mean"], METHOD_ORDER[m]))].index(FINAL_METHOD) + 1})
            rr["V3_BEATS_BEST_BL"] = "YES" if label(rr["V3_VS_BEST_BL_PCT"]) == "WIN" else "NO"
            sr.update({"BEST_BL": sb[0], "BEST_BL_SMAPE_MEAN": sbest, "SECOND_BEST_BL": sb[1], "SECOND_BEST_BL_SMAPE_MEAN": vals[sb[1]]["smape_mean"], "WORST_BL": sb[-1], "WORST_BL_SMAPE_MEAN": sworst, "V3_VS_BEST_BL_PCT": improve(sbest, vals[FINAL_METHOD]["smape_mean"]), "V3_VS_WORST_BL_PCT": improve(sworst, vals[FINAL_METHOD]["smape_mean"]), "V3_RANK_AMONG_6": [*sorted(METHODS, key=lambda m: (vals[m]["smape_mean"], METHOD_ORDER[m]))].index(FINAL_METHOD) + 1})
            sr["V3_BEATS_BEST_BL"] = "YES" if label(sr["V3_VS_BEST_BL_PCT"]) == "WIN" else "NO"
            rmse_rows.append(rr)
            smape_rows.append(sr)
    return pd.DataFrame(rmse_rows), pd.DataFrame(smape_rows), pd.DataFrame(ranks)


def comparisons(matched: pd.DataFrame, metric: str) -> pd.DataFrame:
    v3 = matched[matched.method == FINAL_METHOD].set_index(KEY)[metric]
    rows = []
    for m in BASELINES:
        base = matched[matched.method == m].set_index(KEY)[metric]
        pair = pd.concat([v3.rename("v3"), base.rename("base")], axis=1).sort_index()
        vals = ((pair["base"] - pair["v3"]) / pair["base"] * 100).to_numpy()
        labels = np.array([label(x) for x in vals])
        rows.append({"BASELINE": m, "N": len(vals), "WINS": int((labels == "WIN").sum()), "TIES": int((labels == "TIE").sum()), "LOSSES": int((labels == "LOSS").sum()), "WIN_RATE": float((labels == "WIN").mean()), "MEDIAN_IMPROVEMENT_PCT": float(np.median(vals)), "MEAN_IMPROVEMENT_PCT": float(np.mean(vals)), "TIE_TOLERANCE_RELATIVE": TIE_TOLERANCE})
    return pd.DataFrame(rows)


def rank_summary(ranks: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = ranks[ranks.METRIC == metric.upper()]
    rows = []
    for m in METHODS:
        r = sub[sub.METHOD == m].RANK.to_numpy(dtype=int)
        rows.append({"METHOD": m, "MEAN_RANK": float(r.mean()), "MEDIAN_RANK": float(np.median(r)), **{f"RANK{i}_COUNT": int((r == i).sum()) for i in range(1, 7)}, "TOP1_RATE": float((r <= 1).mean()), "TOP2_RATE": float((r <= 2).mean()), "TOP3_RATE": float((r <= 3).mean()), "N_CONDITIONS": len(r)})
    return pd.DataFrame(rows)


def stability(matched: pd.DataFrame) -> pd.DataFrame:
    v = matched[matched.method == FINAL_METHOD]
    rows = []
    for d in DATASETS:
        for h in HORIZONS:
            s = v[(v.dataset == d) & (v.horizon == h)]
            rm, rs, sm, ss = s.rmse.mean(), s.rmse.std(), s.smape.mean(), s.smape.std()
            rows.append({"DATASET": d, "TARGET": TARGETS[d], "HORIZON": h, "RMSE_MEAN": rm, "RMSE_STD": rs, "RMSE_CV": rs / rm, "SMAPE_MEAN": sm, "SMAPE_STD": ss, "SMAPE_CV": ss / sm, "SEEDS_USED": "42,43,44"})
    return pd.DataFrame(rows)


def dataset_table(rmse: pd.DataFrame, ranks: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    vr = ranks[(ranks.METRIC == "RMSE") & (ranks.METHOD == FINAL_METHOD)]
    rows = []
    for d in DATASETS:
        c = rmse[rmse.DATASET == d].sort_values("HORIZON")
        r = vr[vr.DATASET == d].sort_values("HORIZON").RANK.to_list()
        def wins(base: str) -> int:
            p = matched[(matched.dataset == d) & (matched.method.isin([FINAL_METHOD, base]))].pivot_table(index=KEY, columns="method", values="rmse", aggfunc="first")
            return int(sum(label(x) == "WIN" for x in ((p[base] - p[FINAL_METHOD]) / p[base] * 100).to_numpy()))
        cv = matched[(matched.dataset == d) & (matched.method == FINAL_METHOD)].groupby("horizon").rmse.agg(["mean", "std"])
        rows.append({"DATASET": d, "TARGET": TARGETS[d], "V3_RANK_H1": r[0], "V3_RANK_H3": r[1], "V3_RANK_H5": r[2], "V3_BEATS_BEST_BL_COUNT_OUT_OF_3": int((c.V3_BEATS_BEST_BL == "YES").sum()), "V3_TOP2_COUNT": int(sum(x <= 2 for x in r)), "V3_TOP3_COUNT": int(sum(x <= 3 for x in r)), "MEDIAN_RELATIVE_GAP_TO_BEST_BL_PCT": float(c.V3_VS_BEST_BL_PCT.median()), "V3_VS_BL3_WIN_COUNT_OUT_OF_9": wins("BL3_LightGBM"), "V3_VS_BL4_WIN_COUNT_OUT_OF_9": wins("BL4_LSTM"), "SEED_RMSE_CV_MEDIAN": float((cv["std"] / cv["mean"]).median())})
    return pd.DataFrame(rows)


def horizon_table(rmse: pd.DataFrame, ranks: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    vr = ranks[(ranks.METRIC == "RMSE") & (ranks.METHOD == FINAL_METHOD)]
    rows = []
    for h in HORIZONS:
        c = rmse[rmse.HORIZON == h]
        r = vr[vr.HORIZON == h].RANK.to_numpy(dtype=int)
        row = {"HORIZON": h, "V3_TOP1": int((r == 1).sum()), "V3_TOP2": int((r <= 2).sum()), "V3_TOP3": int((r <= 3).sum()), "V3_BEATS_BEST_BL_COUNT_OUT_OF_6": int((c.V3_BEATS_BEST_BL == "YES").sum()), "MEDIAN_GAP_TO_BEST_BL_PCT": float(c.V3_VS_BEST_BL_PCT.median())}
        for m in BASELINES:
            v = matched[(matched.horizon == h) & (matched.method == FINAL_METHOD)].set_index(KEY).rmse
            b = matched[(matched.horizon == h) & (matched.method == m)].set_index(KEY).rmse
            row[f"V3_VS_{m.split('_')[0]}_WIN_RATE"] = float(np.mean([label(x) == "WIN" for x in ((b - v) / b * 100).to_numpy()]))
        rows.append(row)
    return pd.DataFrame(rows)


def gap_table(rmse: pd.DataFrame) -> pd.DataFrame:
    out = rmse[["DATASET", "TARGET", "HORIZON", "BEST_BL", "BEST_BL_RMSE_MEAN", "V3_RMSE_MEAN", "V3_VS_BEST_BL_PCT", "V3_BEATS_BEST_BL"]].copy()
    out["ABS_GAP_PCT"] = out.V3_VS_BEST_BL_PCT.abs()
    for x in [0.5, 1.0, 2.0, 5.0]:
        out[f"WITHIN_{str(x).replace('.', '_')}_PERCENT"] = out.ABS_GAP_PCT <= x
    return out.sort_values(["ABS_GAP_PCT", "DATASET", "HORIZON"]).reset_index(drop=True)


def paired_stats(matched: pd.DataFrame) -> pd.DataFrame:
    cols = ["BASELINE", "METRIC", "N", "MEDIAN_PAIRED_DIFFERENCE_V3_MINUS_BL", "WILCOXON_STATISTIC", "RAW_P_VALUE", "HOLM_ADJUSTED_P_VALUE", "NOTE"]
    if not SCIPY_AVAILABLE:
        return pd.DataFrame(columns=cols)
    rows = []
    for metric in ["rmse", "smape"]:
        v = matched[matched.method == FINAL_METHOD].set_index(KEY)[metric]
        for m in BASELINES:
            b = matched[matched.method == m].set_index(KEY)[metric]
            p = pd.concat([v.rename("v3"), b.rename("base")], axis=1).dropna().sort_index()
            diff = (p.v3 - p.base).to_numpy(dtype=float)
            stat = pv = float("nan")
            note = "two-sided Wilcoxon; paired difference is V3 - baseline"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
                    stat, pv = float(result.statistic), float(result.pvalue)
                except Exception as exc:
                    note += "; unavailable: " + str(exc)
            rows.append({"BASELINE": m, "METRIC": metric.upper(), "N": len(diff), "MEDIAN_PAIRED_DIFFERENCE_V3_MINUS_BL": float(np.median(diff)), "WILCOXON_STATISTIC": stat, "RAW_P_VALUE": pv, "HOLM_ADJUSTED_P_VALUE": float("nan"), "NOTE": note})
    out = pd.DataFrame(rows)
    order = sorted(out.index[out.RAW_P_VALUE.notna()], key=lambda i: float(out.loc[i, "RAW_P_VALUE"]))
    prev = 0.0
    for rank, i in enumerate(order, 1):
        adj = min(1.0, max(prev, float(out.loc[i, "RAW_P_VALUE"]) * len(order) / rank))
        out.loc[i, "HOLM_ADJUSTED_P_VALUE"] = adj
        prev = adj
    return out


def output_dir(repo: Path) -> Path:
    root = repo / "outputs/analysis"
    base = root / "ms_katl_v3_final_benchmark"
    root.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        base.mkdir()
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = root / f"ms_katl_v3_final_benchmark_{stamp}"
    i = 1
    while p.exists():
        p = root / f"ms_katl_v3_final_benchmark_{stamp}_{i:02d}"
        i += 1
    p.mkdir()
    return p


def md(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False, floatfmt=".6f")
    except Exception:
        return frame.to_csv(index=False)


def write_report(out: Path, integrity: Mapping[str, object], rmse: pd.DataFrame, smape: pd.DataFrame, rc: pd.DataFrame, sc: pd.DataFrame, rr: pd.DataFrame, sr: pd.DataFrame, dt: pd.DataFrame, ht: pd.DataFrame, st: pd.DataFrame, gt: pd.DataFrame, ps: pd.DataFrame) -> None:
    rcols = ["DATASET", "TARGET", "HORIZON", "V3_RMSE_MEAN", "V3_RMSE_STD", "V3_RMSE_MEDIAN", "BL1_RMSE_MEAN", "BL2_RMSE_MEAN", "BL3_RMSE_MEAN", "BL4_RMSE_MEAN", "BL5_RMSE_MEAN", "BEST_BL", "BEST_BL_RMSE_MEAN", "SECOND_BEST_BL", "SECOND_BEST_BL_RMSE_MEAN", "WORST_BL", "WORST_BL_RMSE_MEAN", "V3_VS_BEST_BL_PCT", "V3_VS_WORST_BL_PCT", "V3_RANK_AMONG_6", "V3_BEATS_BEST_BL"]
    scols = [x.replace("RMSE", "SMAPE") for x in rcols]
    lines = [
        "# MS-KATL V3 Final Benchmark Report",
        "",
        "## 1. Scope",
        "",
        f"Final method: {FINAL_METHOD}. MS_KATL_V3_FINAL_FROZEN_MODEL=YES.",
        "Targets: D1=1/10; D2=1/10; D3=10; D4=166/311; D5=48/1159414; D6=CA_1/FOODS_3_080.",
        "Horizons: H1/H3/H5. Seeds: 42/43/44. Primary unit: dataset x horizon.",
        "Baselines: BL1_HistoricalMean, BL2_MovingAverage, BL3_LightGBM, BL4_LSTM, BL5_SeasonalNaive.",
        "",
        "TRAINING_STARTED=NO",
        "V3_RERUN=NO",
        "BASELINE_RERUN=NO",
        "TARGET_RESELECTION=NO",
        "LR_RESELECTION=NO",
        "MODEL_VERSION_SELECTION_DURING_BENCHMARK=NO",
        "",
        "## 2. Data integrity",
        "",
        f"V3 cells: {integrity['matched_v3_cells']}/{integrity['v3_expected_cells']}; baseline rows: {integrity['matched_baseline_rows']}/{integrity['baseline_expected_rows']}; matched rows: 324.",
        f"V3 source: {integrity['v3_results_file']}",
        f"Baseline source: {integrity['baseline_source_file']}",
        "Exact key: dataset x target x horizon x seed. Each V3 key has five baseline rows; no silent inner-join dropping.",
        f"Duplicate V3 keys: {integrity['duplicate_v3_keys']}; duplicate baseline method keys: {integrity['duplicate_baseline_method_keys']}; missing: {integrity['missing_keys']}; extra: {integrity['extra_keys']}.",
        "Metric space: original sales space. SEED_AGGREGATION=42_43_44_ALL. BEST_SEED_SELECTION=NO.",
        "",
        "## 3. RMSE main table",
        "",
        md(rmse[rcols]),
        "",
        "## 4. sMAPE main table",
        "",
        md(smape[scols]),
        "",
        "## 5. V3 vs each baseline",
        "",
        "RMSE:",
        md(rc),
        "",
        "sMAPE:",
        md(sc),
        "",
        "## 6. Learning-baseline comparison",
        "",
        "All 54 matched cells are used for LightGBM and LSTM comparisons.",
        md(rc[rc.BASELINE.isin(["BL3_LightGBM", "BL4_LSTM"])]),
        md(sc[sc.BASELINE.isin(["BL3_LightGBM", "BL4_LSTM"])]),
        "",
        "## 7. Rank distribution",
        "",
        "Ranks are among six methods over 18 dataset x horizon seed-mean conditions.",
        "RMSE:",
        md(rr),
        "sMAPE:",
        md(sr),
        "",
        "## 8. Dataset-level behavior",
        "",
        md(dt),
        "",
        "## 9. Horizon-level behavior",
        "",
        md(ht),
        "",
        "## 10. Seed stability",
        "",
        md(st),
        f"MEDIAN_RMSE_CV={st.RMSE_CV.median():.6f}; MAX_RMSE_CV={st.RMSE_CV.max():.6f}; MEDIAN_SMAPE_CV={st.SMAPE_CV.median():.6f}; MAX_SMAPE_CV={st.SMAPE_CV.max():.6f}. These CVs describe seed variation only.",
        "",
        "## 11. Gap to strongest baseline",
        "",
        md(gt),
        f"WITHIN_0_5_PERCENT={int(gt.WITHIN_0_5_PERCENT.sum())}; WITHIN_1_PERCENT={int(gt.WITHIN_1_0_PERCENT.sum())}; WITHIN_2_PERCENT={int(gt.WITHIN_2_0_PERCENT.sum())}; WITHIN_5_PERCENT={int(gt.WITHIN_5_0_PERCENT.sum())}; BEATS_BEST_BL={int((gt.V3_BEATS_BEST_BL == 'YES').sum())}.",
        "",
        "## 12. Paired statistics",
        "",
        "Wilcoxon signed-rank is included because SciPy was available. Paired difference is V3 - baseline; negative favors V3. Raw and Holm-adjusted p-values are retained.",
        md(ps),
        "",
        "## 13. Final factual conclusions",
        "",
        f"V3 beats the strongest baseline in {int((rmse.V3_BEATS_BEST_BL == 'YES').sum())}/18 RMSE conditions and is rank 1 in {int(rr.loc[rr.METHOD == FINAL_METHOD, 'RANK1_COUNT'].iloc[0])}/18. This does not support a universal-best claim.",
        f"V3 RMSE mean rank={float(rr.loc[rr.METHOD == FINAL_METHOD, 'MEAN_RANK'].iloc[0]):.3f}; median rank={float(rr.loc[rr.METHOD == FINAL_METHOD, 'MEDIAN_RANK'].iloc[0]):.3f}. Cross-dataset raw RMSE averaging was not performed.",
        f"LightGBM RMSE: {int(rc.loc[rc.BASELINE == 'BL3_LightGBM', 'WINS'].iloc[0])} wins, {int(rc.loc[rc.BASELINE == 'BL3_LightGBM', 'LOSSES'].iloc[0])} losses, median improvement {float(rc.loc[rc.BASELINE == 'BL3_LightGBM', 'MEDIAN_IMPROVEMENT_PCT'].iloc[0]):.3f}%. LSTM RMSE: {int(rc.loc[rc.BASELINE == 'BL4_LSTM', 'WINS'].iloc[0])} wins, {int(rc.loc[rc.BASELINE == 'BL4_LSTM', 'LOSSES'].iloc[0])} losses, median improvement {float(rc.loc[rc.BASELINE == 'BL4_LSTM', 'MEDIAN_IMPROVEMENT_PCT'].iloc[0]):.3f}%.",
        "Win rate, top rank, closeness, stability, and statistical significance are reported as distinct claims.",
        "CROSS_DATASET_RAW_RMSE_AVERAGING=NO.",
        "POST_HOC_DATASET_SELECTION=NO.",
        "POST_HOC_HORIZON_SELECTION=NO.",
        "",
        "## 14. Experimental branches note",
        "",
        "V3.1 = calibration experiment. V3.2 = seasonal residual experiment. Neither enters the formal benchmark rank or any max/selection operation.",
    ]
    (out / "FINAL_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    print("ANALYSIS_ONLY=YES")
    print("TRAINING_STARTED=NO")
    print("V3_RERUN=NO")
    print("BASELINE_RERUN=NO")
    print(f"FINAL_METHOD={FINAL_METHOD}")
    try:
        matched, integrity = load_sources(repo)
        print(f"V3_EXPECTED_CELLS={EXPECTED_V3_CELLS}")
        print(f"MATCHED_V3_CELLS={integrity['matched_v3_cells']}")
        print(f"MATCHED_BASELINE_ROWS={integrity['matched_baseline_rows']}")
        st = stats_by_condition(matched)
        rmse, smape, ranks = main_tables(st)
        rc, sc = comparisons(matched, "rmse"), comparisons(matched, "smape")
        rr, sr = rank_summary(ranks, "rmse"), rank_summary(ranks, "smape")
        dt, ht, stab, gap, paired = dataset_table(rmse, ranks, matched), horizon_table(rmse, ranks, matched), stability(matched), gap_table(rmse), paired_stats(matched)
        out = output_dir(repo)
        matched[["dataset", "target", "horizon", "seed", "method", "rmse", "smape", "metric_space", "scenario", "source_type", "result_status"]].to_csv(out / "01_matched_seed_level.csv", index=False)
        rmse.to_csv(out / "02_rmse_by_dataset_horizon.csv", index=False)
        smape.to_csv(out / "03_smape_by_dataset_horizon.csv", index=False)
        rc.to_csv(out / "04_v3_vs_each_baseline_rmse.csv", index=False)
        sc.to_csv(out / "05_v3_vs_each_baseline_smape.csv", index=False)
        rr.to_csv(out / "06_rank_summary_rmse.csv", index=False)
        sr.to_csv(out / "07_rank_summary_smape.csv", index=False)
        dt.to_csv(out / "08_dataset_summary.csv", index=False)
        ht.to_csv(out / "09_horizon_summary.csv", index=False)
        stab.to_csv(out / "10_seed_stability.csv", index=False)
        gap.to_csv(out / "11_best_baseline_gap.csv", index=False)
        paired.to_csv(out / "12_paired_statistics.csv", index=False)
        config = {
            "final_method": FINAL_METHOD,
            "v3_model_frozen": True,
            "v3_frozen_root": integrity["v3_frozen_root"],
            "v3_results_file": integrity["v3_results_file"],
            "baseline_source_file": integrity["baseline_source_file"],
            "baseline_source_candidates_considered": integrity["baseline_source_candidates_considered"],
            "target_mapping": TARGETS,
            "horizons": HORIZONS,
            "seeds": SEEDS,
            "methods": METHODS,
            "scenario": SCENARIO,
            "tie_tolerance_relative": TIE_TOLERANCE,
            "metric_definitions": {"rmse": "original sales space", "smape": "original sales space", "improvement_pct": "(baseline - V3) / baseline * 100", "paired_difference": "V3 - baseline"},
            "aggregation_rules": {"primary_unit": "dataset x horizon", "seed_aggregation": "42_43_44_ALL", "best_seed_selection": False, "std_ddof": 1, "raw_rmse_cross_dataset_averaging": False},
            "training_started": False,
            "target_reselection": False,
            "lr_reselection": False,
            "model_version_selection": False,
            "scipy_available": SCIPY_AVAILABLE,
            "matched_v3_cells": integrity["matched_v3_cells"],
            "matched_baseline_rows": integrity["matched_baseline_rows"],
            "timestamp": datetime.now().astimezone().isoformat(),
            "output_directory": str(out),
        }
        (out / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(out, integrity, rmse, smape, rc, sc, rr, sr, dt, ht, stab, gap, paired)
        print("SEED_AGGREGATION=42_43_44_ALL")
        print("BEST_SEED_SELECTION=NO")
        print("CROSS_DATASET_RAW_RMSE_AVERAGING=NO")
        print("POST_HOC_DATASET_SELECTION=NO")
        print("POST_HOC_HORIZON_SELECTION=NO")
        print(f"REPORT_DIR={out}")
        print("============================================================")
        print("MS-KATL V3 FINAL BENCHMARK COMPLETE")
        print("============================================================")
        return 0
    except FailClosed as exc:
        print("FAIL_CLOSED=YES")
        print(f"FAIL_CLOSED_REASON={exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
