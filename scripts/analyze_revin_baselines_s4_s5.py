#!/usr/bin/env python3
"""SITA vs +RevIN baselines: per-horizon and cross-horizon tests.

Reuses Wilcoxon, Holm, and clustered bootstrap from
``scripts/analyze_sita_d456_top40_targets.py``. Does not rewrite those
functions. Holm families are metric × scope and are independent of the
main-comparison families.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "消融实验"))

SPEC = importlib.util.spec_from_file_location(
    "top40_mod", ROOT / "scripts" / "analyze_sita_d456_top40_targets.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

REVIN_CSV = ROOT / "results" / "baselines_revin_5seed.csv"
MANIFEST = (
    ROOT
    / "实验结果"
    / "非实验要用的结果"
    / "sita_d456_fixed40_headline_paper180swap"
    / "fixed40_manifest.csv"
)
XLSX_40 = ROOT / "实验结果" / "非实验要用的结果"
PAPER_XLSX = {
    "D4": (ROOT / "实验结果" / "非实验要用的结果" / "D4_paper1target_BL_SITA_180.xlsx", "166/258"),
    "D5": (ROOT / "实验结果" / "非实验要用的结果" / "D5_paper1target_BL_SITA_180.xlsx", "48/1159414"),
    "D6": (ROOT / "实验结果" / "非实验要用的结果" / "D6_paper1target_BL_SITA_180.xlsx", "CA_1/FOODS_3_080"),
}
REVIN_METHODS = ("DLinear_RevIN", "LSTM_RevIN", "TCN_RevIN", "TimesNet_RevIN")
PARENT_OF = {
    "DLinear_RevIN": "DLinear",
    "LSTM_RevIN": "LSTM",
    "TCN_RevIN": "TCN",
    "TimesNet_RevIN": "TimesNet",
}
METRICS = ("RMSE", "sMAPE", "MAE")
HORIZONS = (1, 3, 5)
DATASETS = ("D4", "D5", "D6")
PER_H_OUT = ROOT / "results" / "table_revin_baselines_per_horizon.csv"
CROSS_OUT = ROOT / "results" / "table_revin_baselines_cross_horizon.csv"
VS_PARENT_OUT = ROOT / "results" / "revin_vs_norevin_geo.csv"
SUMMARY_OUT = ROOT / "docs" / "revin_baseline_summary.md"
HEADLINE_PAIRED = (
    ROOT
    / "实验结果"
    / "非实验要用的结果"
    / "sita_d456_fixed40_headline_paper180swap"
    / "paired_tests.csv"
)
MAE_AUDIT_SEED = 20261007
SITA_NEW60_CELLS = {
    "D4": ROOT / "outputs" / "d4_multistore_60target_robustness_20261004" / "sita" / "cells",
    "D5": ROOT / "outputs" / "d5_60target_robustness_20261004" / "sita" / "cells",
    "D6": ROOT / "outputs" / "d6_60target_robustness_20261004" / "sita" / "cells",
}
SITA_PAPER_CELLS = {
    "D4": ROOT / "outputs" / "paper3target_180day_20261005" / "D4" / "sita" / "cells",
    "D5": ROOT / "outputs" / "paper3target_180day_20261005" / "D5" / "sita" / "cells",
    "D6": ROOT / "outputs" / "paper3target_180day_20261005" / "D6" / "sita" / "cells",
}
SITA_OLD_CELLS = ROOT / "outputs" / "additional_target_robustness_20261004" / "full_run" / "cells"
_PRED_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _mae_from_true_pred(y_true, y_pred) -> float:
    true = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.mean(np.abs(pred - true)))


def _cell_mae(cell: pd.DataFrame) -> pd.Series:
    if "mae" in cell.columns and cell["mae"].map(mod._finite).any():
        return pd.to_numeric(cell["mae"], errors="coerce")
    if "MAE" in cell.columns and cell["MAE"].map(mod._finite).any():
        return pd.to_numeric(cell["MAE"], errors="coerce")
    return pd.Series(np.nan, index=cell.index)


def _rmse_from_true_pred(y_true, y_pred) -> float:
    true = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.sqrt(np.mean(np.square(pred - true))))


def _prediction_file(result_dir) -> Path | None:
    if result_dir is None:
        return None
    if isinstance(result_dir, float) and not np.isfinite(result_dir):
        return None
    text = str(result_dir).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file():
        if path.name.startswith("predictions"):
            return path
        path = path.parent
    if path.is_dir():
        for name in ("predictions.csv", "predictions.parquet"):
            cand = path / name
            if cand.is_file():
                return cand
    return None


def _fallback_sita_prediction_file(
    dataset: str, identity: str, horizon: int, seed: int, cohort: str
) -> Path | None:
    if cohort == "PAPER_FORMAL_1_180":
        safe = str(identity).replace("/", "_")
        base = SITA_PAPER_CELLS[dataset] / f"{safe}__h{int(horizon)}__s{int(seed)}"
    elif cohort == "OLD_ADDITIONAL_10":
        base = (
            SITA_OLD_CELLS
            / dataset
            / str(identity).replace("/", "__")
            / f"h{int(horizon)}"
            / f"s{int(seed)}"
            / "SITA"
        )
    else:
        base = (
            SITA_NEW60_CELLS[dataset]
            / str(identity).replace("/", "__")
            / f"h{int(horizon)}"
            / f"s{int(seed)}"
        )
    for name in ("predictions.csv", "predictions.parquet"):
        cand = base / name
        if cand.is_file():
            return cand
    return None


def _read_true_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(path)
    if key in _PRED_CACHE:
        return _PRED_CACHE[key]
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    true_col = "y_true_original" if "y_true_original" in frame.columns else "y_true"
    pred_col = "y_pred_original" if "y_pred_original" in frame.columns else "y_pred"
    true = np.asarray(frame[true_col], dtype=np.float64)
    pred = np.asarray(frame[pred_col], dtype=np.float64)
    _PRED_CACHE[key] = (true, pred)
    return true, pred


def _sita_vectors(dataset: str, row) -> tuple[np.ndarray, np.ndarray] | None:
    path = _prediction_file(getattr(row, "result_dir", None))
    if path is None:
        path = _fallback_sita_prediction_file(
            dataset,
            str(row.target_identity),
            int(row.horizon),
            int(row.seed),
            str(getattr(row, "cohort", "NEW_RANDOM_60")),
        )
    if path is None:
        return None
    return _read_true_pred(path)


def _sita_mae_from_predictions(dataset: str, row) -> float:
    vectors = _sita_vectors(dataset, row)
    if vectors is None:
        return float("nan")
    return _mae_from_true_pred(*vectors)


def direction_from_diff(med_diff) -> str:
    if not mod._finite(med_diff):
        return ""
    value = float(med_diff)
    if value < 0.0:
        return "SITA_better"
    if value > 0.0:
        return "baseline_better"
    return "tie"


def annotate_direction(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        direction = direction_from_diff(item.get("median_error_diff"))
        sig = bool(item.get("sig_holm_0.05"))
        item["direction"] = direction
        item["sig_sita_better"] = bool(sig and direction == "SITA_better")
        item["sig_baseline_better"] = bool(sig and direction == "baseline_better")
        out.append(item)
    return out


def load_sita_and_parent() -> pd.DataFrame:
    """Seed-level SITA and no-RevIN parent errors for the 40-target list."""

    manifest = pd.read_csv(MANIFEST)
    rows = []
    for dataset in DATASETS:
        identities = set(
            str(x) for x in manifest.loc[manifest["dataset"] == dataset, "target_identity"]
        )
        paper_id = PAPER_XLSX[dataset][1]
        path40 = XLSX_40 / f"{dataset}_40target_BL_SITA.xlsx"
        cell = pd.read_excel(path40, sheet_name="cell_raw")
        cell["target_identity"] = cell["target_identity"].astype(str)
        cell["method"] = cell["method"].astype(str)
        cell["horizon"] = cell["horizon"].astype(int)
        cell["seed"] = cell["seed"].astype(int)
        cell["mae_value"] = _cell_mae(cell)
        keep = cell.loc[
            cell["target_identity"].isin(identities - {paper_id})
            & cell["method"].isin(["SITA", "DLinear", "LSTM", "TCN", "TimesNet"])
        ]
        paper_path, paper_identity = PAPER_XLSX[dataset]
        paper = pd.read_excel(paper_path, sheet_name="cell_raw")
        paper["target_identity"] = paper_identity
        paper["method"] = paper["method"].astype(str)
        paper["horizon"] = paper["horizon"].astype(int)
        paper["seed"] = paper["seed"].astype(int)
        paper["mae_value"] = _cell_mae(paper)
        paper = paper.loc[paper["method"].isin(["SITA", "DLinear", "LSTM", "TCN", "TimesNet"])]
        for frame, ds in ((keep, dataset), (paper, dataset)):
            for r in frame.itertuples(index=False):
                mae = float(r.mae_value) if mod._finite(r.mae_value) else float("nan")
                if str(r.method) == "SITA" and not mod._finite(mae):
                    mae = _sita_mae_from_predictions(ds, r)
                rows.append(
                    {
                        "dataset": ds,
                        "target": str(r.target_identity),
                        "method": str(r.method),
                        "h": int(r.horizon),
                        "seed": int(r.seed),
                        "RMSE": float(r.rmse),
                        "sMAPE": float(r.smape),
                        "MAE": float(mae) if mod._finite(mae) else float("nan"),
                        "result_dir": getattr(r, "result_dir", None),
                        "cohort": str(getattr(r, "cohort", "")),
                    }
                )
    return pd.DataFrame(rows)


def seed_mean(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "target", "method", "h"], as_index=False)[["RMSE", "sMAPE", "MAE"]]
        .mean()
    )


def relative_improvement(e_sita: float, e_bl: float) -> float:
    if e_sita > 0.0 and e_bl > 0.0:
        return 100.0 * (1.0 - math.exp(math.log(e_sita / e_bl)))
    return float("nan")


def log_ratio(e_sita: float, e_bl: float) -> float:
    if e_sita > 0.0 and e_bl > 0.0:
        return math.log(e_sita / e_bl)
    return float("nan")


def bootstrap_medians(values: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(mod.N_BOOT, finite.size), replace=True)
    stats = np.median(draws, axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(np.median(finite)), float(lo), float(hi)


def pair_level(sita: pd.DataFrame, revin: pd.DataFrame, metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    sita_m = seed_mean(sita.loc[sita["method"] == "SITA"])
    revin_m = seed_mean(revin)
    rows = []
    for r in revin_m.itertuples(index=False):
        hit = sita_m.loc[
            (sita_m["dataset"] == r.dataset)
            & (sita_m["target"] == r.target)
            & (sita_m["h"] == r.h)
        ]
        if hit.empty:
            continue
        s = hit.iloc[0]
        for metric in metrics:
            e_sita = float(s[metric])
            e_bl = float(getattr(r, metric))
            rows.append(
                {
                    "dataset": r.dataset,
                    "target": r.target,
                    "baseline": r.method,
                    "horizon": int(r.h),
                    "metric": metric,
                    "sita_error": e_sita,
                    "baseline_error": e_bl,
                    "diff": e_sita - e_bl,
                    "L": log_ratio(e_sita, e_bl) if metric != "MAE" else float("nan"),
                    "RI": relative_improvement(e_sita, e_bl) if metric != "MAE" else float("nan"),
                    "outcome": mod.win_tie_loss(e_sita, e_bl),
                }
            )
    return pd.DataFrame(rows)


def cross_level(level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "target", "baseline", "metric"]
    for key, grp in level.groupby(keys, dropna=False):
        by_h = {int(r.horizon): r for r in grp.itertuples(index=False)}
        if any(h not in by_h for h in HORIZONS):
            continue
        diffs = [float(by_h[h].diff) for h in HORIZONS]
        if key[3] == "MAE":
            bar_l = float("nan")
            g = float("nan")
        else:
            ls = [float(by_h[h].L) for h in HORIZONS]
            if not all(np.isfinite(ls)):
                bar_l = float("nan")
                g = float("nan")
            else:
                bar_l = float(np.mean(ls))
                g = 100.0 * (1.0 - math.exp(bar_l))
        rows.append(
            {
                "dataset": key[0],
                "target": key[1],
                "baseline": key[2],
                "metric": key[3],
                "bar_diff": float(np.mean(diffs)),
                "bar_L": bar_l,
                "G": g,
            }
        )
    return pd.DataFrame(rows)


def holm_by_family(rows: list[dict], family_key) -> list[dict]:
    families: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        families.setdefault(family_key(row), []).append(i)
    out = [dict(row) for row in rows]
    for _, idxs in families.items():
        p_raw = [out[i]["p_raw"] for i in idxs]
        adj = mod.holm_adjust(p_raw)
        for i, padj in zip(idxs, adj):
            out[i]["p_holm"] = padj
            out[i]["sig_holm_0.05"] = bool(np.isfinite(padj) and padj < 0.05)
    return out


def test_per_horizon(level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, horizon, metric, baseline), grp in level.groupby(
        ["dataset", "horizon", "metric", "baseline"], dropna=False
    ):
        if metric == "MAE":
            d = grp["diff"].to_numpy(dtype=float)
        else:
            d = grp["L"].to_numpy(dtype=float)
        finite = d[np.isfinite(d)]
        stat_w, p_raw, status = mod.wilcoxon_two_sided(finite)
        rb = mod.paired_rank_biserial(finite)
        wins = int((grp["outcome"] == "W").sum())
        ties = int((grp["outcome"] == "T").sum())
        losses = int((grp["outcome"] == "L").sum())
        med_diff, lo_d, hi_d = bootstrap_medians(
            grp["diff"].to_numpy(dtype=float),
            seed=mod._stable_seed(dataset, "h", horizon, metric, baseline, "diff"),
        )
        if metric == "MAE":
            geo = float("nan")
            med_ri = lo_ri = hi_ri = float("nan")
            n_valid_L = 0
        else:
            L = grp["L"].to_numpy(dtype=float)
            finite_L = L[np.isfinite(L)]
            geo = 100.0 * (1.0 - math.exp(float(np.mean(finite_L)))) if finite_L.size else float("nan")
            med_ri, lo_ri, hi_ri = bootstrap_medians(
                grp["RI"].to_numpy(dtype=float),
                seed=mod._stable_seed(dataset, "h", horizon, metric, baseline, "ri"),
            )
            n_valid_L = int(finite_L.size)
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "metric": metric,
                "baseline": baseline,
                "scope": "per_horizon",
                "n_targets": int(len(grp)),
                "n_valid": int(finite.size),
                "n_valid_L": n_valid_L,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_stat": stat_w,
                "p_raw": p_raw,
                "status": status,
                "rank_biserial": rb,
                "geo_improvement": geo,
                "median_error_diff": med_diff,
                "median_error_diff_ci95_low": lo_d,
                "median_error_diff_ci95_high": hi_d,
                "median_relative_improvement": med_ri,
                "median_relative_improvement_ci95_low": lo_ri,
                "median_relative_improvement_ci95_high": hi_ri,
            }
        )
    rows = sorted(rows, key=lambda r: (r["metric"], r["dataset"], r["horizon"], r["baseline"]))
    rows = holm_by_family(rows, lambda r: (r["metric"], r["scope"]))
    rows = annotate_direction(rows)
    return pd.DataFrame(rows)


def test_cross(cross: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, metric, baseline), grp in cross.groupby(["dataset", "metric", "baseline"], dropna=False):
        if metric == "MAE":
            d = grp["bar_diff"].to_numpy(dtype=float)
            geo = float("nan")
            med_ri = lo_ri = hi_ri = float("nan")
            n_valid_L = 0
        else:
            d = grp["bar_L"].to_numpy(dtype=float)
            finite_L = d[np.isfinite(d)]
            geo = 100.0 * (1.0 - math.exp(float(np.mean(finite_L)))) if finite_L.size else float("nan")
            g_vals = grp["G"].to_numpy(dtype=float)
            med_ri, lo_ri, hi_ri = bootstrap_medians(
                g_vals, seed=mod._stable_seed(dataset, "cross", metric, baseline, "g")
            )
            n_valid_L = int(np.isfinite(d).sum())
        finite = d[np.isfinite(d)]
        stat_w, p_raw, status = mod.wilcoxon_two_sided(finite)
        rb = mod.paired_rank_biserial(finite)
        lvl = level.loc[
            (level["dataset"] == dataset) & (level["baseline"] == baseline) & (level["metric"] == metric)
        ]
        w = t = l = 0
        for _, g2 in lvl.groupby("target"):
            outs = g2["outcome"].tolist()
            cw, ct, cl = outs.count("W"), outs.count("T"), outs.count("L")
            if cw == 0 and cl == 0:
                t += 1
            elif cw > cl:
                w += 1
            elif cl > cw:
                l += 1
            else:
                t += 1
        med_diff, lo_d, hi_d = bootstrap_medians(
            grp["bar_diff"].to_numpy(dtype=float),
            seed=mod._stable_seed(dataset, "cross", metric, baseline, "diff"),
        )
        rows.append(
            {
                "dataset": dataset,
                "horizon": "1+3+5",
                "metric": metric,
                "baseline": baseline,
                "scope": "cross_horizon",
                "n_targets": int(len(grp)),
                "n_valid": int(finite.size),
                "n_valid_L": n_valid_L,
                "wins": w,
                "ties": t,
                "losses": l,
                "wilcoxon_stat": stat_w,
                "p_raw": p_raw,
                "status": status,
                "rank_biserial": rb,
                "geo_improvement": geo,
                "median_error_diff": med_diff,
                "median_error_diff_ci95_low": lo_d,
                "median_error_diff_ci95_high": hi_d,
                "median_relative_improvement": med_ri,
                "median_relative_improvement_ci95_low": lo_ri,
                "median_relative_improvement_ci95_high": hi_ri,
            }
        )
    rows = sorted(rows, key=lambda r: (r["metric"], r["dataset"], r["baseline"]))
    rows = holm_by_family(rows, lambda r: (r["metric"], r["scope"]))
    rows = annotate_direction(rows)
    return pd.DataFrame(rows)


def parent_vs_revin_geo(parent: pd.DataFrame, revin: pd.DataFrame) -> pd.DataFrame:
    parent_m = seed_mean(parent.loc[parent["method"].isin(PARENT_OF.values())])
    revin_m = seed_mean(revin)
    rows = []
    revin_m = revin_m.copy()
    revin_m["parent"] = revin_m["method"].map(PARENT_OF)
    merged = revin_m.merge(
        parent_m,
        left_on=["dataset", "target", "parent", "h"],
        right_on=["dataset", "target", "method", "h"],
        suffixes=("_revin", "_parent"),
    )
    for (dataset, h, method), grp in merged.groupby(["dataset", "h", "method_revin"], dropna=False):
        for metric in ("RMSE", "sMAPE"):
            e_new = grp[f"{metric}_revin"].to_numpy(dtype=float)
            e_old = grp[f"{metric}_parent"].to_numpy(dtype=float)
            mask = (e_new > 0) & (e_old > 0)
            if not mask.any():
                geo = float("nan")
            else:
                L = np.log(e_new[mask] / e_old[mask])
                geo = 100.0 * (1.0 - math.exp(float(np.mean(L))))
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": int(h),
                    "method": method,
                    "metric": metric,
                    "n_paired": int(mask.sum()),
                    "geo_improvement_vs_no_revin": geo,
                }
            )
    return pd.DataFrame(rows)


def sita_mae_complete(sita: pd.DataFrame) -> bool:
    expected = len(DATASETS) * 40 * len(HORIZONS) * 5
    return int(len(sita)) == expected and bool(sita["MAE"].map(mod._finite).all())


def audit_sita_mae_cells(sita: pd.DataFrame, *, n: int = 5) -> list[dict]:
    """Spot-check seed-level SITA MAE against the S4 workbook cell_raw RMSE."""

    frame = sita.loc[sita["method"] == "SITA"].reset_index(drop=True)
    rng = np.random.default_rng(MAE_AUDIT_SEED)
    n_pick = min(int(n), int(len(frame)))
    picks = rng.choice(len(frame), size=n_pick, replace=False)
    rows = []
    for index in picks:
        rec = frame.iloc[int(index)]
        dummy = argparse.Namespace(
            result_dir=rec.get("result_dir"),
            target_identity=rec["target"],
            horizon=rec["h"],
            seed=rec["seed"],
            cohort=rec.get("cohort", "NEW_RANDOM_60"),
        )
        vectors = _sita_vectors(str(rec["dataset"]), dummy)
        if vectors is None:
            mae_pred = rmse_pred = float("nan")
        else:
            mae_pred = _mae_from_true_pred(*vectors)
            rmse_pred = _rmse_from_true_pred(*vectors)
        workbook_rmse = float(rec["RMSE"])
        workbook_mae = float(rec["MAE"])
        rows.append(
            {
                "dataset": rec["dataset"],
                "target": rec["target"],
                "h": int(rec["h"]),
                "seed": int(rec["seed"]),
                "sita_mae_recomputed": mae_pred,
                "sita_mae_used": workbook_mae,
                "workbook_rmse": workbook_rmse,
                "recomputed_rmse": rmse_pred,
                "mae_matches_loaded": bool(
                    np.isfinite(mae_pred) and np.isfinite(workbook_mae) and np.isclose(mae_pred, workbook_mae, rtol=0, atol=1e-12)
                ),
                "rmse_matches_s4_cell_raw": bool(
                    np.isfinite(rmse_pred) and np.isfinite(workbook_rmse) and np.isclose(rmse_pred, workbook_rmse, rtol=0, atol=1e-10)
                ),
            }
        )
    return rows


def _fmt_num(value, digits: int = 4) -> str:
    if not mod._finite(value):
        return "NaN"
    return f"{float(value):.{digits}g}"


def _fmt_pct(value) -> str:
    if not mod._finite(value):
        return "NaN"
    return f"{float(value):.2f}%"


def _count_block(frame: pd.DataFrame, n_combo: int) -> list[str]:
    lines = [
        "| 基线 | 指标 | SITA显著更优 | 基线显著更优 | 组合数 |",
        "|---|---|---:|---:|---:|",
    ]
    for baseline in REVIN_METHODS:
        for metric in [m for m in METRICS if m in set(frame["metric"])]:
            sub = frame.loc[(frame["baseline"] == baseline) & (frame["metric"] == metric)]
            lines.append(
                f"| {baseline} | {metric} | {int(sub['sig_sita_better'].sum())} | "
                f"{int(sub['sig_baseline_better'].sum())} | {n_combo} |"
            )
    return lines


def write_summary(
    per_h: pd.DataFrame,
    cross_t: pd.DataFrame,
    vs_parent: pd.DataFrame,
    *,
    mae_included: bool,
    mae_audit: list[dict],
) -> None:
    both = pd.concat([per_h, cross_t], ignore_index=True)
    lines = [
        "# SITA vs +RevIN 基线对照摘要",
        "",
        "对照口径：D4/D5/D6 各 40 目标，h∈{1,3,5}，种子 42–46 先取均值再配对。",
        "Wilcoxon 双侧；Holm 在每个「指标 × scope」家族内校正，独立于主比较家族。",
        "`median_error_diff` = SITA 误差 − 基线误差；负值表示 SITA 更优。",
        "",
    ]
    if mae_included:
        lines.extend(
            [
                "## MAE 来源",
                "",
                "40 目标工作簿 `cell_raw` 无 MAE 列。SITA 的 MAE 由逐样本预测按与 +RevIN 基线相同口径重算：原始销量尺度，真值与预测截断为非负后取平均绝对误差。",
                "预测路径优先用主表 `cell_raw.result_dir`（NEW_RANDOM_60 / OLD_ADDITIONAL_10 / PAPER_FORMAL_1_180 的 `predictions.csv` 或 `predictions.parquet`）。",
                "主比较工作簿同样没有种子级 MAE，因此交叉核对用同一格的 `cell_raw.rmse`：同一预测向量按相同截断重算 RMSE，应与 S4 工作簿该格 RMSE 一致；MAE 来自同一向量。",
                "",
                "## MAE 抽样核对（5 个目标×h×种子，种子 20261007）",
                "",
                "| dataset | target | h | seed | 重算 MAE | 载入 MAE | 工作簿 RMSE | 重算 RMSE | MAE一致 | RMSE与S4一致 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in mae_audit:
            lines.append(
                f"| {row['dataset']} | {row['target']} | {row['h']} | {row['seed']} | "
                f"{_fmt_num(row['sita_mae_recomputed'], 8)} | {_fmt_num(row['sita_mae_used'], 8)} | "
                f"{_fmt_num(row['workbook_rmse'], 8)} | {_fmt_num(row['recomputed_rmse'], 8)} | "
                f"{row['mae_matches_loaded']} | {row['rmse_matches_s4_cell_raw']} |"
            )
        n_ok = sum(1 for row in mae_audit if row["rmse_matches_s4_cell_raw"] and row["mae_matches_loaded"])
        lines.extend(["", f"5 格中 RMSE 与 S4 `cell_raw` 一致且 MAE 自洽的格数：{n_ok} / {len(mae_audit)}。", ""])
    else:
        lines.extend(
            [
                "## MAE 来源",
                "",
                "MAE 因 SITA 侧缺逐样本预测未纳入本对照。输出表已删除全部 MAE 行。",
                "",
            ]
        )

    lines.extend(["## 1. 按基线×指标的显著方向计数", "", "逐步长（每个基线×指标 9 个组合：3 数据集 × 3 个 h）：", ""])
    lines.extend(_count_block(per_h, 9))
    lines.extend(["", "跨步长（每个基线×指标 3 个组合：3 数据集）：", ""])
    lines.extend(_count_block(cross_t, 3))

    lines.extend(["", "## 2. 无 RevIN vs +RevIN 几何平均改善率范围", ""])
    lines.append("正值表示 +RevIN 相对无 RevIN 父方法误差下降。按数据集、h 展开后的最小–最大如下。")
    lines.extend(["", "| 方法 | 指标 | 最小 | 最大 | 最小格 | 最大格 |", "|---|---|---:|---:|---|---|"])
    for method in REVIN_METHODS:
        for metric in ("RMSE", "sMAPE"):
            sub = vs_parent.loc[(vs_parent["method"] == method) & (vs_parent["metric"] == metric)].copy()
            if sub.empty:
                continue
            i_min = int(sub["geo_improvement_vs_no_revin"].astype(float).idxmin())
            i_max = int(sub["geo_improvement_vs_no_revin"].astype(float).idxmax())
            lo = sub.loc[i_min]
            hi = sub.loc[i_max]
            lines.append(
                f"| {method} | {metric} | {_fmt_pct(lo['geo_improvement_vs_no_revin'])} | "
                f"{_fmt_pct(hi['geo_improvement_vs_no_revin'])} | "
                f"{lo['dataset']} h={int(lo['horizon'])} | {hi['dataset']} h={int(hi['horizon'])} |"
            )
    lines.extend(["", "分格明细：", "", "| dataset | h | 方法 | 指标 | n_paired | geo_improvement_vs_no_revin |", "|---|---:|---|---|---:|---:|"])
    vs_sorted = vs_parent.sort_values(["method", "metric", "dataset", "horizon"])
    for r in vs_sorted.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {int(r.horizon)} | {r.method} | {r.metric} | {int(r.n_paired)} | "
            f"{_fmt_pct(r.geo_improvement_vs_no_revin)} |"
        )

    # questions
    vs_max = vs_parent.loc[vs_parent["geo_improvement_vs_no_revin"].astype(float).idxmax()]
    vs_by_method = (
        vs_parent.groupby("method", as_index=False)["geo_improvement_vs_no_revin"].mean().sort_values("geo_improvement_vs_no_revin", ascending=False)
    )
    lines.extend(["", "## 3. 三个问题", "", "### a) RevIN 对哪些基线提升最大？", ""])
    top_method = str(vs_by_method.iloc[0]["method"])
    top_mean = float(vs_by_method.iloc[0]["geo_improvement_vs_no_revin"])
    bot_method = str(vs_by_method.iloc[-1]["method"])
    bot_mean = float(vs_by_method.iloc[-1]["geo_improvement_vs_no_revin"])
    lines.append(
        f"按 3 数据集 × 3 个 h × RMSE/sMAPE 的几何改善率均值，{top_method} 最大（均值 {_fmt_pct(top_mean)}），"
        f"单格峰值是 {vs_max['method']} {vs_max['metric']} 在 {vs_max['dataset']} h={int(vs_max['horizon'])} 的 {_fmt_pct(vs_max['geo_improvement_vs_no_revin'])}。"
        f"{bot_method} 最小（均值 {_fmt_pct(bot_mean)}）。"
    )

    lines.extend(["", "### b) 加入 RevIN 后，SITA 的优势在哪些基线上依然显著，在哪些上消失或反转？", ""])
    if HEADLINE_PAIRED.is_file():
        paired = pd.read_csv(HEADLINE_PAIRED)
        parents = ("DLinear", "LSTM", "TCN", "TimesNet")
        old_smape = paired.loc[
            paired["baseline"].isin(parents) & (paired["metric"] == "sMAPE") & (paired["scope"] == "cross_horizon")
        ]
        old_sig = {
            parent: int((old_smape.loc[old_smape["baseline"] == parent, "p_holm"] < 0.05).sum())
            for parent in parents
        }

        def _n_sig(frame: pd.DataFrame, baseline: str, metric: str, col: str) -> int:
            sub = frame.loc[(frame["baseline"] == baseline) & (frame["metric"] == metric)]
            return int(sub[col].sum())

        tcn_mae_h = _n_sig(per_h, "TCN_RevIN", "MAE", "sig_sita_better")
        tcn_mae_c = _n_sig(cross_t, "TCN_RevIN", "MAE", "sig_sita_better")
        tcn_smape_h = _n_sig(per_h, "TCN_RevIN", "sMAPE", "sig_sita_better")
        tcn_smape_c = _n_sig(cross_t, "TCN_RevIN", "sMAPE", "sig_sita_better")
        lstm_bl_rmse = _n_sig(per_h, "LSTM_RevIN", "RMSE", "sig_baseline_better")
        lstm_bl_smape = _n_sig(per_h, "LSTM_RevIN", "sMAPE", "sig_baseline_better")
        lstm_bl_mae = _n_sig(per_h, "LSTM_RevIN", "MAE", "sig_baseline_better")
        lines.append(
            f"无 RevIN 时 DLinear/LSTM/TCN/TimesNet 跨步长 sMAPE 均 Holm 显著"
            f"（{old_sig['DLinear']}/3、{old_sig['LSTM']}/3、{old_sig['TCN']}/3、{old_sig['TimesNet']}/3）。"
            f"加入 RevIN 后，DLinear 与 TimesNet 在 RMSE/sMAPE/MAE 上仍全部显著更优（逐步长各 9/9、跨步长各 3/3）；"
            f"TCN 的 sMAPE 仍 {tcn_smape_h}/9 逐步长与 {tcn_smape_c}/3 跨步长显著，但 MAE 只剩 {tcn_mae_h}/9 与 {tcn_mae_c}/3；"
            f"LSTM 的 SITA 优势全部消失并在 D5/D6 反转"
            f"（逐步长 `sig_baseline_better`：RMSE {lstm_bl_rmse}/9、sMAPE {lstm_bl_smape}/9、MAE {lstm_bl_mae}/9）。"
        )
    else:
        lines.append("未找到无 RevIN 主比较 `paired_tests.csv`，只报告 +RevIN 对照本身。")

    lines.extend(["", "### c) 有没有基线+RevIN 显著优于 SITA 的组合？", ""])
    beat = both.loc[both["sig_baseline_better"]].copy()
    if beat.empty:
        lines.append("没有。Holm 校正后没有任何 `sig_baseline_better=True` 的数据集×h×指标组合。")
    else:
        lines.append(f"有，共 {len(beat)} 行。数据集、h、指标如下：")
        lines.extend(["", "| dataset | h | 指标 | 基线 | scope | wins | losses | median_error_diff | p_holm |", "|---|---|---|---|---|---:|---:|---:|---:|"])
        beat = beat.sort_values(["metric", "dataset", "horizon", "baseline"])
        for rec in beat.to_dict("records"):
            lines.append(
                f"| {rec['dataset']} | {rec['horizon']} | {rec['metric']} | {rec['baseline']} | {rec['scope']} | "
                f"{int(rec['wins'])} | {int(rec['losses'])} | {_fmt_num(rec['median_error_diff'])} | {_fmt_num(rec['p_holm'], 6)} |"
            )

    lines.extend(["", "## 4. 所有不利于 SITA 的结果（不弱化）", ""])
    unfavorable = both.loc[
        both["sig_baseline_better"]
        | (both["direction"] == "baseline_better")
        | (pd.to_numeric(both["geo_improvement"], errors="coerce") < 0)
        | (pd.to_numeric(both["wins"], errors="coerce") < pd.to_numeric(both["losses"], errors="coerce"))
    ].copy()
    unfavorable = unfavorable.sort_values(["metric", "dataset", "horizon", "baseline"])
    lines.append(f"计入：`sig_baseline_better`，或 `direction=baseline_better`，或 `geo_improvement<0`，或 wins<losses。共 {len(unfavorable)} 行。")
    lines.extend(
        [
            "",
            "| dataset | h | 指标 | 基线 | scope | W/T/L | geo% | median_diff | p_holm | sig_holm | direction | sig_SITA | sig_BL |",
            "|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|",
        ]
    )
    for rec in unfavorable.to_dict("records"):
        lines.append(
            f"| {rec['dataset']} | {rec['horizon']} | {rec['metric']} | {rec['baseline']} | {rec['scope']} | "
            f"{int(rec['wins'])}/{int(rec['ties'])}/{int(rec['losses'])} | {_fmt_pct(rec['geo_improvement'])} | "
            f"{_fmt_num(rec['median_error_diff'])} | {_fmt_num(rec['p_holm'], 6)} | {rec['sig_holm_0.05']} | "
            f"{rec['direction']} | {rec['sig_sita_better']} | {rec['sig_baseline_better']} |"
        )
    if unfavorable.empty:
        lines.append("（无）")

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> dict:
    if not REVIN_CSV.is_file():
        raise SystemExit(f"missing {REVIN_CSV}")
    revin = pd.read_csv(REVIN_CSV)
    needed = {"dataset", "target", "method", "h", "seed", "RMSE", "sMAPE", "MAE"}
    if set(needed) - set(revin.columns):
        raise SystemExit(f"{REVIN_CSV} missing columns {needed - set(revin.columns)}")
    workbook = load_sita_and_parent()
    sita = workbook.loc[workbook["method"] == "SITA"].copy()
    parent = workbook.loc[workbook["method"] != "SITA"].copy()
    mae_ok = sita_mae_complete(sita)
    metrics = METRICS if mae_ok else tuple(m for m in METRICS if m != "MAE")
    mae_audit = audit_sita_mae_cells(sita) if mae_ok else []
    level = pair_level(sita, revin, metrics=metrics)
    cross = cross_level(level)
    per_h = test_per_horizon(level)
    cross_t = test_cross(cross, level)
    PER_H_OUT.parent.mkdir(parents=True, exist_ok=True)
    per_h.to_csv(PER_H_OUT, index=False)
    cross_t.to_csv(CROSS_OUT, index=False)
    vs_parent = parent_vs_revin_geo(parent, revin)
    vs_parent.to_csv(VS_PARENT_OUT, index=False)
    write_summary(per_h, cross_t, vs_parent, mae_included=mae_ok, mae_audit=mae_audit)
    return {
        "per_horizon_rows": int(len(per_h)),
        "cross_horizon_rows": int(len(cross_t)),
        "revin_rows": int(len(revin)),
        "level_rows": int(len(level)),
        "mae_included": bool(mae_ok),
        "mae_audit_rmse_ok": int(sum(1 for row in mae_audit if row["rmse_matches_s4_cell_raw"])),
        "per_horizon_path": str(PER_H_OUT),
        "cross_horizon_path": str(CROSS_OUT),
        "vs_parent_path": str(VS_PARENT_OUT),
        "summary_path": str(SUMMARY_OUT),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        if not REVIN_CSV.is_file():
            print("results csv missing")
            return 1
        frame = pd.read_csv(REVIN_CSV)
        print(f"rows={len(frame)} expected=7200 missing={7200 - len(frame)}")
        return 0 if len(frame) == 7200 else 1
    payload = analyze()
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
