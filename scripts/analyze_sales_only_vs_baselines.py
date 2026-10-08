#!/usr/bin/env python3
"""Sales-only SITA vs the eight main baselines on the Table S1 40-target set.

Reuses the Table S4/S9 protocol from analyze_revin_baselines_s4_s5.py:
paired Wilcoxon, 100k bootstrap median CIs, Holm families = (metric, scope).
This Holm family is independent of Full SITA vs baselines and of Full vs sales-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "revin_s4_mod", ROOT / "scripts" / "analyze_revin_baselines_s4_s5.py"
)
revin = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = revin
SPEC.loader.exec_module(revin)

import run_sita_sales_only_ablation as so  # noqa: E402

DATASETS = so.DATASETS
BASELINES = (
    "Seasonal Naive",
    "DLinear",
    "LSTM",
    "LightGBM",
    "MSML-TL",
    "PatchTST",
    "TCN",
    "TimesNet",
)
METRICS = ("RMSE", "sMAPE")
HORIZONS = (1, 3, 5)
EXPECTED_TARGETS = 40
EXPECTED_SALES_CELLS = 1800
MANIFEST = so.MANIFEST
WORKBOOKS = so.WORKBOOKS
SALES_CSV = so.OUT / "metrics_seed_level.csv"
OUT_DIR = ROOT / "results"
PER_H = OUT_DIR / "sales_only_vs_baselines_per_horizon.csv"
CROSS = OUT_DIR / "sales_only_vs_baselines_cross_horizon.csv"
XLSX = OUT_DIR / "sales_only_vs_baselines_S4S9.xlsx"
META = so.OUT / "sales_only_vs_baselines_meta.json"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_sales_only() -> pd.DataFrame:
    if not SALES_CSV.is_file():
        raise FileNotFoundError(SALES_CSV)
    frame = pd.read_csv(SALES_CSV)
    if len(frame) != EXPECTED_SALES_CELLS:
        raise RuntimeError(f"sales-only cells {len(frame)} != {EXPECTED_SALES_CELLS}")
    frame["dataset"] = frame["dataset"].astype(str)
    frame["target"] = frame["target"].astype(str)
    frame["method"] = "SITA"
    frame["h"] = frame["h"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    if "MAE" not in frame.columns:
        frame["MAE"] = np.nan
    return frame[["dataset", "target", "method", "h", "seed", "RMSE", "sMAPE", "MAE"]]


def load_baselines(identities: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        seed = pd.read_excel(WORKBOOKS[ds], sheet_name="seed_mean")
        seed["target_identity"] = seed["target_identity"].astype(str)
        seed["method"] = seed["method"].astype(str)
        keep = identities[ds]
        got = [x for x in keep if x in set(seed["target_identity"])]
        if got != keep:
            missing = sorted(set(keep) - set(got))
            extra = sorted(set(seed.loc[seed["method"].isin(BASELINES), "target_identity"]) - set(keep))
            raise RuntimeError(f"{ds} seed_mean targets != Table S1 missing={missing[:8]} extra={extra[:8]}")
        sub = seed.loc[seed["method"].isin(BASELINES) & seed["target_identity"].isin(keep)].copy()
        expected = EXPECTED_TARGETS * len(BASELINES) * len(HORIZONS)
        if len(sub) != expected:
            raise RuntimeError(f"{ds} baseline seed_mean rows {len(sub)} != {expected}")
        if not (sub["n_test"].astype(int) == 180).all():
            raise RuntimeError(f"{ds} baseline n_test is not all 180")
        if not (sub["n_seeds"].astype(int) == 5).all():
            raise RuntimeError(f"{ds} baseline n_seeds is not all 5")
        for r in sub.itertuples(index=False):
            rows.append(
                {
                    "dataset": ds,
                    "target": str(r.target_identity),
                    "method": str(r.method),
                    "h": int(r.horizon),
                    "seed": 0,
                    "RMSE": float(r.rmse_mean),
                    "sMAPE": float(r.smape_mean),
                    "MAE": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def sita_table_format(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["W/T/L"] = [f"{int(w)}/{int(t)}/{int(l)}" for w, t, l in zip(out["wins"], out["ties"], out["losses"])]
    out["holm_family"] = [
        f"{m}|{s}|sales_only_vs_8baselines" for m, s in zip(out["metric"], out["scope"])
    ]
    cols = [
        "dataset", "horizon", "metric", "baseline", "scope", "n_targets",
        "geo_improvement", "median_error_diff", "median_error_diff_ci95_low",
        "median_error_diff_ci95_high", "median_relative_improvement",
        "median_relative_improvement_ci95_low", "median_relative_improvement_ci95_high",
        "W/T/L", "wins", "ties", "losses", "p_raw", "p_holm", "sig_holm_0.05",
        "direction", "sig_sita_better", "sig_baseline_better",
        "wilcoxon_stat", "rank_biserial", "status", "holm_family",
    ]
    return out[[c for c in cols if c in out.columns]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    started = _utc()
    _, _, meta = so.frozen()
    identities = {ds: [str(x) for x in meta["identities"][ds]] for ds in DATASETS}
    for ds, ids in identities.items():
        if len(ids) != EXPECTED_TARGETS:
            raise RuntimeError(f"{ds} has {len(ids)} targets")

    sales = load_sales_only()
    by = sales.groupby("dataset")["target"].nunique().to_dict()
    if by != {ds: EXPECTED_TARGETS for ds in DATASETS} and not args.allow_partial:
        raise RuntimeError(f"sales-only target counts {by}")

    baselines = load_baselines(identities)
    print(f"[{_utc()}] pairing sales-only vs 8 baselines", flush=True)
    level = revin.pair_level(sales, baselines, metrics=METRICS)
    expected_level = EXPECTED_TARGETS * 3 * len(DATASETS) * len(BASELINES) * len(METRICS)
    if len(level) != expected_level:
        raise RuntimeError(f"paired rows {len(level)} != {expected_level}")
    unpaired = []
    for ds, ids in identities.items():
        have = set(level.loc[level.dataset == ds, "target"])
        miss = [x for x in ids if x not in have]
        if miss:
            unpaired.append(f"{ds}:{miss}")
    if unpaired:
        raise RuntimeError(f"unpaired targets {unpaired}")

    print(f"[{_utc()}] per-horizon tests", flush=True)
    per_h = revin.test_per_horizon(level)
    print(f"[{_utc()}] cross-horizon tests", flush=True)
    cross = revin.test_cross(revin.cross_level(level), level)
    per_h_out = sita_table_format(per_h)
    cross_out = sita_table_format(cross)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_h_out.to_csv(PER_H, index=False)
    cross_out.to_csv(CROSS, index=False)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "protocol": "Table S4/S9: 5-seed mean then target unit; Wilcoxon on L (RMSE/sMAPE); "
                    "bootstrap median CI N_BOOT=100000; Holm family=(metric, scope), independent of Full SITA families",
                    "sales_only_method": "SITA_SALES_ONLY",
                    "baselines": ", ".join(BASELINES),
                    "targets": "Table S1 / paper180swap 120",
                    "n_per_horizon_tests": int(len(per_h_out)),
                    "n_cross_horizon_tests": int(len(cross_out)),
                    "started_utc": started,
                    "finished_utc": _utc(),
                }
            ]
        ).to_excel(writer, sheet_name="protocol", index=False)
        per_h_out.to_excel(writer, sheet_name="per_horizon_S4", index=False)
        cross_out.to_excel(writer, sheet_name="cross_horizon_S5", index=False)
        level.to_excel(writer, sheet_name="target_level_pairs", index=False)
    payload = {
        "status": "SALES_ONLY_VS_8BASELINES_OK",
        "n_sales_cells": int(len(sales)),
        "n_paired_level_rows": int(len(level)),
        "n_per_horizon": int(len(per_h_out)),
        "n_cross_horizon": int(len(cross_out)),
        "holm_family": "(metric, scope) independent of Full SITA vs baselines",
        "per_horizon_csv": str(PER_H.relative_to(ROOT)),
        "cross_horizon_csv": str(CROSS.relative_to(ROOT)),
        "xlsx": str(XLSX.relative_to(ROOT)),
        "finished_at": _utc(),
    }
    META.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
