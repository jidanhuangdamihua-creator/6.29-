#!/usr/bin/env python3
"""Main SITA vs SITA_SALES_ONLY: target-level stats after 5-seed means."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("top40_mod", ROOT / "scripts" / "analyze_sita_d456_top40_targets.py")
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

sys.path.insert(0, str(ROOT / "scripts"))
import run_sita_sales_only_ablation as so  # noqa: E402

DATASETS = so.DATASETS
OUT = so.OUT
WORKBOOKS = so.WORKBOOKS

SALES_CSV = OUT / "metrics_seed_level.csv"
S4 = ROOT / "results" / "sales_only_ablation_table_s4_source.csv"
STATS = OUT / "sales_only_vs_full_stats.csv"
METRICS = ("RMSE", "MAE", "sMAPE")
HORIZONS = (1, 3, 5)


def main_seed_table() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        cell = pd.read_excel(WORKBOOKS[ds], sheet_name="cell_raw")
        sita = cell.loc[cell["method"].astype(str) == "SITA"]
        for _, row in sita.iterrows():
            ident = str(row["target_identity"])
            horizon = int(row["horizon"])
            seed = int(row["seed"])
            pred = so.load_main_predictions(str(row["result_dir"]))
            y_true = pred["y_true"].to_numpy(dtype=np.float64)
            y_pred = pred["y_pred"].to_numpy(dtype=np.float64)
            rows.append({
                "dataset": ds,
                "target": ident,
                "h": horizon,
                "seed": seed,
                "RMSE": float(row["rmse"]),
                "sMAPE": float(row["smape"]),
                "MAE": so.mae_original(y_true, y_pred),
            })
    return pd.DataFrame(rows)


def seed_mean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "target", "h"], as_index=False)[list(METRICS)].mean()


def bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(mod.N_BOOT, finite.size), replace=True)
    stats = np.median(draws, axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(np.median(finite)), float(lo), float(hi)


def wtl(full: np.ndarray, sales: np.ndarray) -> tuple[int, int, int]:
    w = t = l = 0
    for a, b in zip(full, sales):
        tag = mod.win_tie_loss(float(a), float(b))
        if tag == "W":
            w += 1
        elif tag == "T":
            t += 1
        else:
            l += 1
    return w, t, l


def analyze() -> pd.DataFrame:
    sales = pd.read_csv(SALES_CSV)
    full = main_seed_table()
    s_mean = seed_mean(sales)
    f_mean = seed_mean(full)
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            for metric in METRICS:
                a = f_mean.loc[(f_mean.dataset == ds) & (f_mean.h == h), ["target", metric]].rename(columns={metric: "full"})
                b = s_mean.loc[(s_mean.dataset == ds) & (s_mean.h == h), ["target", metric]].rename(columns={metric: "sales"})
                merged = a.merge(b, on="target")
                diff = merged["sales"] - merged["full"]
                rel = np.where(merged["sales"].to_numpy(dtype=float) == 0.0, np.nan, diff / merged["sales"])
                ri = 100.0 * rel
                d = diff.to_numpy(dtype=float)
                stat, p_raw, status = mod.wilcoxon_two_sided(d)
                med, lo, hi = bootstrap_median(d, seed=mod._stable_seed(ds, h, metric, "sales-full"))
                med_rel, lo_rel, hi_rel = bootstrap_median(rel, seed=mod._stable_seed(ds, h, metric, "rel"))
                med_ri, lo_ri, hi_ri = bootstrap_median(ri, seed=mod._stable_seed(ds, h, metric, "ri"))
                wins, ties, losses = wtl(merged["full"].to_numpy(dtype=float), merged["sales"].to_numpy(dtype=float))
                n_paired = int(len(merged))
                rows.append({
                    "dataset": ds,
                    "horizon": h,
                    "metric": metric,
                    "n_paired_targets": n_paired,
                    "Full SITA": float(np.median(merged["full"])) if n_paired else float("nan"),
                    "Sales-only SITA": float(np.median(merged["sales"])) if n_paired else float("nan"),
                    "mean_full": float(np.mean(merged["full"])) if n_paired else float("nan"),
                    "mean_sales_only": float(np.mean(merged["sales"])) if n_paired else float("nan"),
                    "absolute_difference": med,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "median_relative_difference": med_rel,
                    "rel_ci95_low": lo_rel,
                    "rel_ci95_high": hi_rel,
                    "mean_RI_cov": float(np.nanmean(ri)) if n_paired else float("nan"),
                    "median_RI_cov": med_ri,
                    "RI_ci95_low": lo_ri,
                    "RI_ci95_high": hi_ri,
                    "W": wins,
                    "T": ties,
                    "L": losses,
                    "wilcoxon_stat": stat,
                    "p_raw": p_raw,
                    "status": status,
                })
    families: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        families.setdefault((row["dataset"], row["metric"]), []).append(i)
    for idxs in families.values():
        adj = mod.holm_adjust([rows[i]["p_raw"] for i in idxs])
        for i, padj in zip(idxs, adj):
            rows[i]["p_holm"] = padj
    stats = pd.DataFrame(rows)
    s4 = stats.rename(columns={"p_raw": "raw Wilcoxon p", "p_holm": "Holm-adjusted p"})
    s4 = s4[[
        "dataset", "horizon", "metric", "Full SITA", "Sales-only SITA",
        "absolute_difference", "median_relative_difference", "ci95_low", "ci95_high",
        "W", "T", "L", "raw Wilcoxon p", "Holm-adjusted p",
        "mean_RI_cov", "median_RI_cov", "RI_ci95_low", "RI_ci95_high", "n_paired_targets",
    ]]
    s4 = s4.rename(columns={"ci95_low": "95% CI low", "ci95_high": "95% CI high"})
    S4.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(STATS, index=False)
    s4.to_csv(S4, index=False)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if not SALES_CSV.is_file():
        print(f"missing {SALES_CSV}")
        return 1
    n = len(pd.read_csv(SALES_CSV))
    print(f"sales-only rows={n} expected=1800")
    if n < 1800 and not args.allow_partial:
        print("incomplete: wait for the 1800-cell run to finish, then rerun this script.")
        print("do not use any previously written S4 csv from a 3-row canary.")
        return 2
    analyze()
    print(f"wrote {STATS} and {S4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
