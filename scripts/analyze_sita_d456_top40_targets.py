#!/usr/bin/env python3
"""SITA D4/D5/D6 Top-40 target screening and statistical analysis.

Implements 实验结果 companion docs:
- SITA_top40统计分析_可执行指导.md
- SITA_top40选择与统计分析_方法论.md

Read-only on input workbooks and sealed sales; writes a new output directory.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "消融实验"
for path in (ROOT, ABLATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from additional_target_robustness.protocol import CALENDARS  # noqa: E402
from additional_target_robustness.target_data import load_target_series  # noqa: E402
from sita_ablation.metrics import evaluate_original_sales  # noqa: E402

DATASETS = ("D4", "D5", "D6")
METHODS = (
    "SITA",
    "Seasonal Naive",
    "DLinear",
    "LSTM",
    "LightGBM",
    "MSML-TL",
    "PatchTST",
    "TCN",
    "TimesNet",
)
BASELINES = tuple(m for m in METHODS if m != "SITA")
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("sMAPE", "RMSE")
TIE_TOL = 0.001
N_BOOT = 100_000
BOOT_SEED = 20261005
SCRIPT_VERSION = "analyze_sita_d456_top40_targets.py@2026-10-05-on40"

INPUT_DIR = ROOT / "实验结果"
DEFAULT_OUT = ROOT / "实验结果" / "sita_d456_top40_analysis_on40"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def instance_id(dataset: str, cohort: str, universe: str, identity: str) -> str:
    return f"{dataset}|{cohort}|{universe}|{identity}"


def workbook_path(dataset: str) -> Path:
    return INPUT_DIR / f"{dataset}_70target_BL_SITA.xlsx"


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def win_tie_loss(e_sita: float, e_bl: float) -> str:
    if e_bl > 0.0:
        if abs(e_sita / e_bl - 1.0) <= TIE_TOL:
            return "T"
        return "W" if e_sita < e_bl else "L"
    if e_sita == 0.0 and e_bl == 0.0:
        return "T"
    return "W" if e_sita < e_bl else ("T" if e_sita == e_bl else "L")


def bounded_q(e_sita: float, e_bl: float) -> float:
    if e_sita == 0.0 and e_bl == 0.0:
        return 0.0
    if e_sita == 0.0:
        return 1.0
    if e_bl == 0.0:
        return -1.0
    return (e_bl - e_sita) / (e_bl + e_sita)


def average_ranks_desc(values: np.ndarray) -> np.ndarray:
    """Larger value -> better (rank 1). Ties get average ranks."""
    n = len(values)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (math.inf if not np.isfinite(p_values[i]) else p_values[i], i))
    adj = [math.nan] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        p = p_values[idx]
        if not np.isfinite(p):
            adj[idx] = math.nan
            continue
        val = min(1.0, (m - rank + 1) * p)
        running = max(running, val)
        adj[idx] = running
    return adj


def paired_rank_biserial(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    nonzero = d[d != 0.0]
    if nonzero.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())
    denom = r_plus + r_minus
    if denom == 0.0:
        return float("nan")
    return (r_plus - r_minus) / denom


def wilcoxon_two_sided(differences: np.ndarray) -> tuple[float, float, str]:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    nonzero = d[d != 0.0]
    if nonzero.size == 0:
        return float("nan"), float("nan"), "ALL_ZERO"
    try:
        res = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue), "OK"
    except ValueError as exc:
        return float("nan"), float("nan"), f"ERROR:{exc}"


@dataclass
class AuditResult:
    ok: bool = True
    rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def add(self, check: str, status: str, detail: str = "", **extra: Any) -> None:
        row = {"check": check, "status": status, "detail": detail, **extra}
        self.rows.append(row)
        if status != "PASS":
            self.ok = False
            self.issues.append(f"{check}: {detail or status}")


def load_sheets(dataset: str) -> dict[str, pd.DataFrame]:
    path = workbook_path(dataset)
    return {
        "path": path,  # type: ignore[dict-item]
        "cell_raw": pd.read_excel(path, sheet_name="cell_raw"),
        "seed_mean": pd.read_excel(path, sheet_name="seed_mean"),
        "coverage": pd.read_excel(path, sheet_name="coverage"),
        "targets": pd.read_excel(path, sheet_name="targets"),
    }


def prepare_frames(dataset: str, sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = sheets["targets"].copy()
    targets["dataset"] = dataset
    targets["source_universe_id"] = targets["source_universe_id"].astype(str)
    targets["target_identity"] = targets["target_identity"].astype(str)
    targets["cohort"] = targets["cohort"].astype(str)
    targets["target_instance_id"] = [
        instance_id(dataset, c, u, i)
        for c, u, i in zip(targets["cohort"], targets["source_universe_id"], targets["target_identity"])
    ]

    cell = sheets["cell_raw"].copy()
    cell["dataset"] = dataset
    cell["source_universe_id"] = cell["source_universe_id"].astype(str)
    cell["target_identity"] = cell["target_identity"].astype(str)
    cell["cohort"] = cell["cohort"].astype(str)
    cell["method"] = cell["method"].astype(str)
    cell["horizon"] = cell["horizon"].astype(int)
    cell["seed"] = cell["seed"].astype(int)
    cell["target_instance_id"] = [
        instance_id(dataset, c, u, i)
        for c, u, i in zip(cell["cohort"], cell["source_universe_id"], cell["target_identity"])
    ]

    seed = sheets["seed_mean"].copy()
    seed["method"] = seed["method"].astype(str)
    seed["horizon"] = seed["horizon"].astype(int)
    seed["target_identity"] = seed["target_identity"].astype(str)
    seed["cohort"] = seed["cohort"].astype(str)
    seed = seed.merge(
        targets[["cohort", "target_identity", "source_universe_id", "dataset", "target_instance_id"]],
        on=["cohort", "target_identity"],
        how="left",
        validate="many_to_one",
    )
    return targets, cell, seed, sheets["coverage"].copy()


def audit_dataset(dataset: str, targets: pd.DataFrame, cell: pd.DataFrame, seed: pd.DataFrame, coverage: pd.DataFrame) -> AuditResult:
    audit = AuditResult()
    audit.add("input_workbook", "PASS", str(workbook_path(dataset)))

    # coverage cohort x method
    if "complete" not in coverage.columns:
        audit.add("coverage_complete_field", "FAIL", "missing complete column")
    else:
        incomplete = coverage.loc[~coverage["complete"].astype(bool)]
        audit.add(
            "coverage_cohort_method",
            "PASS" if incomplete.empty else "FAIL",
            f"rows={len(coverage)}; incomplete={len(incomplete)}",
        )
        expected_cov = 2 * 9
        present_methods = {(str(r.cohort), str(r.method)) for r in coverage.itertuples()}
        missing_cov = []
        for cohort in ("OLD_ADDITIONAL_10", "NEW_RANDOM_60"):
            for method in METHODS:
                if (cohort, method) not in present_methods:
                    missing_cov.append(f"{cohort}|{method}")
        audit.add(
            "coverage_method_presence",
            "PASS" if not missing_cov and len(coverage) == expected_cov else "FAIL",
            f"coverage_rows={len(coverage)} missing={missing_cov[:20]}",
        )

    n_inst = targets["target_instance_id"].nunique()
    dup_inst = int(targets["target_instance_id"].duplicated().sum())
    audit.add(
        "target_instance_count",
        "PASS" if n_inst == 70 and dup_inst == 0 and len(targets) == 70 else "FAIL",
        f"n={len(targets)} unique={n_inst} dup={dup_inst}",
    )

    # cell_raw seed coverage
    cell_ok = cell.loc[cell["status"].astype(str).isin(["COMPLETED", "SKIP"])].copy()
    key_cols = ["target_instance_id", "method", "horizon", "seed"]
    seed_counts = cell_ok.groupby(["target_instance_id", "method", "horizon"], dropna=False)["seed"].agg(
        ["count", "nunique", lambda s: sorted(set(int(x) for x in s))]
    )
    seed_counts.columns = ["count", "nunique", "seeds"]
    bad_seed = seed_counts.loc[(seed_counts["count"] != 5) | (seed_counts["nunique"] != 5)]
    expected_cells = 70 * 9 * 3
    audit.add(
        "cell_raw_seed_coverage",
        "PASS" if bad_seed.empty and len(seed_counts) == expected_cells else "FAIL",
        f"combos={len(seed_counts)} bad={len(bad_seed)}",
    )

    # exact seed set
    wrong_seed_set = 0
    for seeds in seed_counts["seeds"]:
        if tuple(seeds) != SEEDS:
            wrong_seed_set += 1
    audit.add("cell_raw_seed_values", "PASS" if wrong_seed_set == 0 else "FAIL", f"wrong_sets={wrong_seed_set}")

    # seed_mean uniqueness and mean match
    if seed["source_universe_id"].isna().any() or seed["target_instance_id"].isna().any():
        audit.add("seed_mean_join", "FAIL", "failed join to targets for universe/instance")
    else:
        audit.add("seed_mean_join", "PASS", "joined")

    gcols = ["target_instance_id", "method", "horizon"]
    dup_seed_mean = int(seed.duplicated(gcols).sum())
    audit.add(
        "seed_mean_unique",
        "PASS" if dup_seed_mean == 0 and len(seed) == expected_cells else "FAIL",
        f"rows={len(seed)} dups={dup_seed_mean}",
    )

    cell_means = (
        cell_ok.groupby(gcols, dropna=False)[["smape", "rmse"]]
        .mean()
        .rename(columns={"smape": "smape_from_raw", "rmse": "rmse_from_raw"})
        .reset_index()
    )
    merged = seed.merge(cell_means, on=gcols, how="outer", indicator=True)
    mismatch = merged.loc[
        (merged["_merge"] != "both")
        | ~np.isclose(merged["smape_mean"], merged["smape_from_raw"], rtol=0, atol=1e-10, equal_nan=False)
        | ~np.isclose(merged["rmse_mean"], merged["rmse_from_raw"], rtol=0, atol=1e-10, equal_nan=False)
    ]
    audit.add(
        "seed_mean_matches_cell_raw",
        "PASS" if mismatch.empty else "FAIL",
        f"mismatches={len(mismatch)}",
    )

    finite_ok = seed["smape_mean"].map(_finite).all() and seed["rmse_mean"].map(_finite).all()
    audit.add("metrics_finite", "PASS" if finite_ok else "FAIL", f"finite={finite_ok}")

    methods_ok = set(seed["method"].unique()) == set(METHODS)
    audit.add("methods_set", "PASS" if methods_ok else "FAIL", f"found={sorted(set(seed['method'].unique()))}")

    # same test dates across methods per instance
    date_var = (
        cell_ok.groupby("target_instance_id")[["first_target_date", "last_target_date"]]
        .nunique()
        .max(axis=1)
    )
    audit.add(
        "shared_test_dates",
        "PASS" if (date_var <= 1).all() else "FAIL",
        f"instances_with_date_conflict={(date_var > 1).sum()}",
    )

    # SITA source distance stability
    sita = cell_ok.loc[cell_ok["method"] == "SITA", [
        "target_instance_id", "horizon", "seed",
        "source_1", "source_1_distance", "source_2", "source_2_distance", "source_3", "source_3_distance",
    ]].copy()
    src_cols = ["source_1", "source_1_distance", "source_2", "source_2_distance", "source_3", "source_3_distance"]
    unstable = 0
    missing_src = 0
    for tid, grp in sita.groupby("target_instance_id"):
        if grp[src_cols].isna().any().any():
            missing_src += 1
        dedup = grp[src_cols].drop_duplicates()
        if len(dedup) != 1:
            unstable += 1
    audit.add(
        "sita_source_distance_stable",
        "PASS" if unstable == 0 and missing_src == 0 else "FAIL",
        f"unstable={unstable} missing_src={missing_src}",
    )
    return audit


def build_target_level(dataset: str, seed: pd.DataFrame) -> pd.DataFrame:
    wide = {}
    for metric, col in (("sMAPE", "smape_mean"), ("RMSE", "rmse_mean")):
        piv = seed.pivot_table(
            index=["dataset", "cohort", "source_universe_id", "target_identity", "target_instance_id", "horizon"],
            columns="method",
            values=col,
            aggfunc="first",
        )
        wide[metric] = piv

    rows: list[dict[str, Any]] = []
    for metric, piv in wide.items():
        for idx, row in piv.iterrows():
            dataset_i, cohort, universe, identity, tid, horizon = idx
            e_sita = float(row["SITA"])
            for baseline in BASELINES:
                e_bl = float(row[baseline])
                if e_sita > 0.0 and e_bl > 0.0:
                    l_val = math.log(e_sita / e_bl)
                    ri = 100.0 * (1.0 - math.exp(l_val))
                else:
                    l_val = float("nan")
                    ri = float("nan")
                rows.append(
                    {
                        "dataset": dataset_i,
                        "cohort": cohort,
                        "source_universe_id": universe,
                        "target_identity": identity,
                        "target_instance_id": tid,
                        "baseline": baseline,
                        "horizon": int(horizon),
                        "metric": metric,
                        "sita_error": e_sita,
                        "baseline_error": e_bl,
                        "L": l_val,
                        "RI": ri,
                        "outcome": win_tie_loss(e_sita, e_bl),
                        "Q": bounded_q(e_sita, e_bl),
                    }
                )
    return pd.DataFrame(rows)


def build_cross_horizon(level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "cohort", "source_universe_id", "target_identity", "target_instance_id", "baseline", "metric"]
    for key, grp in level.groupby(keys, dropna=False):
        by_h = {int(r.horizon): r for r in grp.itertuples(index=False)}
        Ls = []
        valid = True
        for h in HORIZONS:
            if h not in by_h:
                valid = False
                break
            val = by_h[h].L
            if not np.isfinite(val):
                valid = False
                break
            Ls.append(float(val))
        if valid:
            bar_l = float(np.mean(Ls))
            g = 100.0 * (1.0 - math.exp(bar_l))
        else:
            bar_l = float("nan")
            g = float("nan")
        # descriptive cross-horizon outcome from mean Q? use outcomes on available horizons via errors
        # Keep G/bar_L only; win rate uses per-horizon later.
        rows.append(
            {
                "dataset": key[0],
                "cohort": key[1],
                "source_universe_id": key[2],
                "target_identity": key[3],
                "target_instance_id": key[4],
                "baseline": key[5],
                "metric": key[6],
                "L_h1": by_h.get(1).L if 1 in by_h else float("nan"),
                "L_h3": by_h.get(3).L if 3 in by_h else float("nan"),
                "L_h5": by_h.get(5).L if 5 in by_h else float("nan"),
                "bar_L": bar_l,
                "G": g,
                "n_valid_horizons": int(sum(np.isfinite(by_h[h].L) for h in HORIZONS if h in by_h)),
            }
        )
    return pd.DataFrame(rows)


def composite_scores(level: pd.DataFrame) -> pd.DataFrame:
    # one row per instance with 48 P scores and C
    instances = (
        level[["dataset", "cohort", "source_universe_id", "target_identity", "target_instance_id"]]
        .drop_duplicates()
        .sort_values("target_instance_id")
        .reset_index(drop=True)
    )
    assert len(instances) == 70
    p_cols: dict[str, np.ndarray] = {}
    for baseline in BASELINES:
        for horizon in HORIZONS:
            for metric in METRICS:
                sub = level.loc[
                    (level["baseline"] == baseline)
                    & (level["horizon"] == horizon)
                    & (level["metric"] == metric)
                ].set_index("target_instance_id")
                q = instances["target_instance_id"].map(sub["Q"]).to_numpy(dtype=float)
                ranks = average_ranks_desc(q)
                p = 1.0 - (ranks - 1.0) / 69.0
                col = f"P__{baseline}__h{horizon}__{metric}"
                p_cols[col] = p
    p_mat = np.column_stack([p_cols[c] for c in p_cols])
    c_vals = p_mat.mean(axis=1)
    out = instances.copy()
    for col, arr in p_cols.items():
        out[col] = arr
    out["C"] = c_vals
    # rank by C desc; ties by target_instance_id
    order = sorted(range(len(out)), key=lambda i: (-out.loc[i, "C"], out.loc[i, "target_instance_id"]))
    rank = np.empty(len(out), dtype=int)
    group = np.empty(len(out), dtype=object)
    for pos, i in enumerate(order, start=1):
        rank[i] = pos
        if pos <= 40:
            group[i] = "Top40"
        elif pos > 60:
            group[i] = "Bottom10"
        else:
            group[i] = "Middle20"
    out["rank_in_dataset"] = rank
    out["group"] = group
    return out.sort_values(["rank_in_dataset", "target_instance_id"]).reset_index(drop=True)


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return BOOT_SEED + (sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 1_000_000)


def cluster_bootstrap_bundle(
    ri: np.ndarray,
    win_ind: np.ndarray,
    log_ratio: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, tuple[float, float, float]]:
    """One clustered resample stream -> CIs for mean RI, win-rate, and geo(L)."""
    ri = np.asarray(ri, dtype=float)
    win_ind = np.asarray(win_ind, dtype=float)
    log_ratio = np.asarray(log_ratio, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    empty = (float("nan"), float("nan"), float("nan"))
    if ri.size == 0:
        return {"ri": empty, "winrate": empty, "geo": empty}

    cl_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        cl_to_idx[c].append(i)
    cl_list = list(cl_to_idx)
    n_cl = len(cl_list)
    lengths = np.array([len(cl_to_idx[c]) for c in cl_list], dtype=int)
    max_n = int(lengths.max())
    idx_mat = np.full((n_cl, max_n), -1, dtype=int)
    for i, c in enumerate(cl_list):
        idxs = cl_to_idx[c]
        idx_mat[i, : len(idxs)] = idxs

    rng = np.random.default_rng(seed)
    # chunk to limit memory
    chunk = 5000
    ri_stats = np.empty(n_boot, dtype=float)
    wr_stats = np.empty(n_boot, dtype=float)
    geo_stats = np.empty(n_boot, dtype=float)
    pos = 0
    while pos < n_boot:
        n = min(chunk, n_boot - pos)
        chosen = rng.integers(0, n_cl, size=(n, n_cl))
        take = idx_mat[chosen]  # (n, n_cl, max_n)
        flat_idx = take.reshape(n, -1)
        valid = flat_idx >= 0
        # gather with invalid -> nan
        ri_take = np.where(valid, ri[np.clip(flat_idx, 0, len(ri) - 1)], np.nan)
        wr_take = np.where(valid, win_ind[np.clip(flat_idx, 0, len(win_ind) - 1)], np.nan)
        l_take = np.where(valid, log_ratio[np.clip(flat_idx, 0, len(log_ratio) - 1)], np.nan)
        with np.errstate(all="ignore"):
            ri_stats[pos : pos + n] = np.nanmean(ri_take, axis=1)
            wr_stats[pos : pos + n] = np.nanmean(wr_take, axis=1)
            mean_l = np.nanmean(l_take, axis=1)
            geo_stats[pos : pos + n] = 100.0 * (1.0 - np.exp(mean_l))
        pos += n

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return empty
        lo, hi = np.quantile(finite, [0.025, 0.975])
        return float(np.mean(finite)), float(lo), float(hi)

    return {"ri": _ci(ri_stats), "winrate": _ci(wr_stats), "geo": _ci(geo_stats)}


def summarize_horizon(level: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return descriptive horizon summaries and bootstrap intervals."""
    id_to_product = targets.set_index("target_instance_id")["target_identity"].to_dict()
    summary_rows = []
    boot_rows = []
    for (dataset, horizon, metric, baseline), grp in level.groupby(
        ["dataset", "horizon", "metric", "baseline"], dropna=False
    ):
        n = len(grp)
        wins = int((grp["outcome"] == "W").sum())
        ties = int((grp["outcome"] == "T").sum())
        losses = int((grp["outcome"] == "L").sum())
        ri = grp["RI"].to_numpy(dtype=float)
        L = grp["L"].to_numpy(dtype=float)
        finite_L = L[np.isfinite(L)]
        geo = 100.0 * (1.0 - math.exp(float(np.mean(finite_L)))) if finite_L.size else float("nan")
        # average rank of SITA among 9 methods needs seed_mean; skip here and fill later
        summary_rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "metric": metric,
                "baseline": baseline,
                "n_targets": n,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / n if n else float("nan"),
                "ri_mean": float(np.nanmean(ri)) if n else float("nan"),
                "ri_median": float(np.nanmedian(ri)) if n else float("nan"),
                "geo_improvement": geo,
                "n_valid_L": int(np.isfinite(L).sum()),
            }
        )
        clusters = grp["target_instance_id"].map(id_to_product).to_numpy()
        win_ind = (grp["outcome"] == "W").astype(float).to_numpy()
        bundle = cluster_bootstrap_bundle(
            ri,
            win_ind,
            L,
            clusters,
            seed=_stable_seed(dataset, "h", horizon, metric, baseline),
        )
        mean_ri, lo_ri, hi_ri = bundle["ri"]
        mean_wr, lo_wr, hi_wr = bundle["winrate"]
        mean_g, lo_g, hi_g = bundle["geo"]
        boot_rows.append(
            {
                "dataset": dataset,
                "scope": "by_horizon",
                "horizon": int(horizon),
                "metric": metric,
                "baseline": baseline,
                "ri_mean_boot": mean_ri,
                "ri_ci95_low": lo_ri,
                "ri_ci95_high": hi_ri,
                "win_rate_boot": mean_wr,
                "win_rate_ci95_low": lo_wr,
                "win_rate_ci95_high": hi_wr,
                "geo_boot": mean_g,
                "geo_ci95_low": lo_g,
                "geo_ci95_high": hi_g,
                "n_boot": N_BOOT,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(boot_rows)


def summarize_cross(cross: pd.DataFrame, level: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_to_product = targets.set_index("target_instance_id")["target_identity"].to_dict()
    summary_rows = []
    boot_rows = []
    test_rows = []
    # cross-horizon win: majority? guide: report G median and win rate where bar_L<0 among valid
    for (dataset, metric, baseline), grp in cross.groupby(["dataset", "metric", "baseline"], dropna=False):
        valid = grp.loc[np.isfinite(grp["bar_L"])]
        n_valid = len(valid)
        g_med = float(valid["G"].median()) if n_valid else float("nan")
        win_valid = int((valid["bar_L"] < 0).sum()) if n_valid else 0
        bar_l = valid["bar_L"].to_numpy(dtype=float)
        geo = 100.0 * (1.0 - math.exp(float(np.mean(bar_l)))) if n_valid else float("nan")
        summary_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "baseline": baseline,
                "n_targets": len(grp),
                "n_valid": n_valid,
                "g_median": g_med,
                "barL_win_count": win_valid,
                "barL_win_rate": win_valid / n_valid if n_valid else float("nan"),
                "geo_improvement": geo,
            }
        )
        if n_valid:
            clusters = valid["target_instance_id"].map(id_to_product).to_numpy()
            win_ind = (valid["bar_L"] < 0).astype(float).to_numpy()
            g_vals = valid["G"].to_numpy(dtype=float)
            bundle = cluster_bootstrap_bundle(
                g_vals,
                win_ind,
                bar_l,
                clusters,
                seed=_stable_seed(dataset, "cross", metric, baseline),
            )
            mean_g, lo_g, hi_g = bundle["geo"]
            mean_wr, lo_wr, hi_wr = bundle["winrate"]
        else:
            mean_g = lo_g = hi_g = mean_wr = lo_wr = hi_wr = float("nan")
        boot_rows.append(
            {
                "dataset": dataset,
                "scope": "cross_horizon",
                "horizon": "1+3+5",
                "metric": metric,
                "baseline": baseline,
                "geo_boot": mean_g,
                "geo_ci95_low": lo_g,
                "geo_ci95_high": hi_g,
                "win_rate_boot": mean_wr,
                "win_rate_ci95_low": lo_wr,
                "win_rate_ci95_high": hi_wr,
                "n_boot": N_BOOT,
                "n_valid": n_valid,
            }
        )

    # main tests: cross-horizon sMAPE only
    smape_cross = cross.loc[cross["metric"] == "sMAPE"].copy()
    raw_tests = []
    for (dataset, baseline), grp in smape_cross.groupby(["dataset", "baseline"], dropna=False):
        d = grp["bar_L"].to_numpy(dtype=float)
        # SITA better => bar_L < 0; Wilcoxon on bar_L (two-sided)
        finite = d[np.isfinite(d)]
        stat_w, p_raw, status = wilcoxon_two_sided(finite)
        rb = paired_rank_biserial(finite)
        # W/T/L on cross: use per-instance outcomes across horizons from level? use bar_L sign with tie tol on exp?
        # Count from three-horizon majority using level outcomes on sMAPE
        lvl = level.loc[
            (level["dataset"] == dataset)
            & (level["baseline"] == baseline)
            & (level["metric"] == "sMAPE")
        ]
        # instance-level: if all three outcomes T -> T; elif wins>losses W; elif losses>wins L; else T
        w = t = l = 0
        for tid, g2 in lvl.groupby("target_instance_id"):
            outs = g2["outcome"].tolist()
            cw = outs.count("W")
            ct = outs.count("T")
            cl = outs.count("L")
            if cw == 0 and cl == 0:
                t += 1
            elif cw > cl:
                w += 1
            elif cl > cw:
                l += 1
            else:
                t += 1
        raw_tests.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "metric": "sMAPE",
                "scope": "cross_horizon",
                "n_valid": int(np.isfinite(d).sum()),
                "wilcoxon_stat": stat_w,
                "p_raw": p_raw,
                "status": status,
                "rank_biserial": rb,
                "wins": w,
                "ties": t,
                "losses": l,
                "mean_bar_L": float(np.nanmean(d)),
                "geo_improvement": 100.0 * (1.0 - math.exp(float(np.nanmean(d)))) if np.isfinite(d).any() else float("nan"),
            }
        )
    # Holm over 24
    # ensure order dataset x baseline fixed
    raw_tests = sorted(raw_tests, key=lambda r: (r["dataset"], r["baseline"]))
    p_adj = holm_adjust([r["p_raw"] for r in raw_tests])
    for row, padj in zip(raw_tests, p_adj):
        row["p_holm"] = padj
        test_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(boot_rows), pd.DataFrame(test_rows)


def method_ranks(seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, horizon), grp in seed.groupby(["dataset", "horizon"], dropna=False):
        for metric, col in (("sMAPE", "smape_mean"), ("RMSE", "rmse_mean")):
            piv = grp.pivot_table(index="target_instance_id", columns="method", values=col, aggfunc="first")
            # lower error better
            ranks = piv.rank(axis=1, method="average", ascending=True)
            mean_ranks = ranks.mean(axis=0)
            for method, val in mean_ranks.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "metric": metric,
                        "method": method,
                        "mean_rank": float(val),
                        "n_targets": int(len(piv)),
                    }
                )
    return pd.DataFrame(rows)


def load_sales_features(dataset: str, targets: pd.DataFrame) -> pd.DataFrame:
    """Load sealed target sales via protocol helper (handles D5 authority path)."""
    rows = []
    # Cache by identity: D4 may reuse the same product identity across instances.
    cache: dict[str, pd.DataFrame] = {}
    for r in targets.itertuples(index=False):
        tid = r.target_instance_id
        identity = r.target_identity
        try:
            if identity not in cache:
                cache[identity] = load_target_series(dataset, identity)
            frame = cache[identity]
            start = pd.Timestamp(r.train_start)
            end = pd.Timestamp(r.val_end)
            pre = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
            expected = pd.date_range(start, end, freq="D")
            if len(expected) != 30:
                raise ValueError(f"expected 30 calendar days, got {len(expected)}")
            if len(pre) != 30 or not pre["date"].reset_index(drop=True).equals(pd.Series(expected)):
                raise ValueError(
                    f"{dataset} {identity} pre-forecast span not exactly 30 ordered calendar days "
                    f"(got {len(pre)})"
                )
            y = pre["sales"].to_numpy(dtype=float)
            mean_y = float(np.mean(y))
            sd_y = float(np.std(y, ddof=0))
            zero_share = float(np.mean(y == 0.0))
            if mean_y == 0.0:
                cv = float("nan")
                all_zero = True
            else:
                cv = sd_y / mean_y
                all_zero = False
            test = frame.loc[
                (frame["date"] >= pd.Timestamp(r.test_start)) & (frame["date"] <= pd.Timestamp(r.test_end))
            ]
            if len(test) != int(r.n_test_days):
                raise ValueError(f"test days {len(test)} != {r.n_test_days}")
            y_test = test["sales"].to_numpy(dtype=float)
            test_zero = float(np.mean(y_test == 0.0))
            pair = evaluate_original_sales(y_test, y_test)
            rows.append(
                {
                    "dataset": dataset,
                    "cohort": r.cohort,
                    "source_universe_id": r.source_universe_id,
                    "target_identity": identity,
                    "target_instance_id": tid,
                    "pre30_mean_sales": mean_y,
                    "pre30_zero_share": zero_share,
                    "CV_sales": cv,
                    "pre30_all_zero": all_zero,
                    "test_zero_share": test_zero,
                    "smape_zero_denominator_count_both_zero_truth": pair.smape_zero_denominator_count,
                    "feature_status": "OK",
                    "feature_detail": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - collect per-target feature failures
            rows.append(
                {
                    "dataset": dataset,
                    "cohort": r.cohort,
                    "source_universe_id": r.source_universe_id,
                    "target_identity": identity,
                    "target_instance_id": tid,
                    "pre30_mean_sales": float("nan"),
                    "pre30_zero_share": float("nan"),
                    "CV_sales": float("nan"),
                    "pre30_all_zero": False,
                    "test_zero_share": float("nan"),
                    "smape_zero_denominator_count_both_zero_truth": float("nan"),
                    "feature_status": "MISSING",
                    "feature_detail": f"{type(exc).__name__}: {exc}",
                }
            )
    _ = CALENDARS[dataset]
    return pd.DataFrame(rows)


def attach_sources_and_cv(features: pd.DataFrame, cell: pd.DataFrame, seed: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    sita = cell.loc[cell["method"] == "SITA"].copy()
    src = (
        sita.groupby("target_instance_id", as_index=False)
        .agg(
            source_1=("source_1", "first"),
            source_1_distance=("source_1_distance", "first"),
            source_2=("source_2", "first"),
            source_2_distance=("source_2_distance", "first"),
            source_3=("source_3", "first"),
            source_3_distance=("source_3_distance", "first"),
        )
    )
    src["nearest_source_distance"] = src[
        ["source_1_distance", "source_2_distance", "source_3_distance"]
    ].min(axis=1)

    # CV_seed_smape by instance x horizon
    sita_seed = seed.loc[seed["method"] == "SITA", ["target_instance_id", "horizon", "smape_mean", "smape_std"]].copy()
    # Prefer std/mean from seed_mean; if mean==0 -> NA
    sita_seed["CV_seed_smape"] = np.where(
        sita_seed["smape_mean"].to_numpy(dtype=float) == 0.0,
        np.nan,
        sita_seed["smape_std"].to_numpy(dtype=float) / sita_seed["smape_mean"].to_numpy(dtype=float),
    )
    cv_wide = sita_seed.pivot(index="target_instance_id", columns="horizon", values="CV_seed_smape")
    cv_wide = cv_wide.rename(columns={1: "CV_seed_smape_h1", 3: "CV_seed_smape_h3", 5: "CV_seed_smape_h5"}).reset_index()

    out = features.merge(src, on="target_instance_id", how="left")
    out = out.merge(cv_wide, on="target_instance_id", how="left")
    score_cols = ["target_instance_id", "C", "rank_in_dataset", "group"]
    out = out.merge(scores[score_cols], on="target_instance_id", how="left")
    out["baseline_coverage_complete"] = True
    return out


def feature_group_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Descriptive feature summary on the analysis set (Top40)."""
    cols = [
        "pre30_mean_sales",
        "pre30_zero_share",
        "CV_sales",
        "nearest_source_distance",
        "source_1_distance",
        "source_2_distance",
        "source_3_distance",
        "test_zero_share",
        "C",
    ]
    rows = []
    for dataset, grp in features.groupby("dataset", dropna=False):
        for col in cols:
            s = pd.to_numeric(grp[col], errors="coerce")
            finite = s[np.isfinite(s)]
            rows.append(
                {
                    "dataset": dataset,
                    "group": "Top40",
                    "feature": col,
                    "n_non_missing": int(finite.shape[0]),
                    "median": float(finite.median()) if len(finite) else float("nan"),
                    "q1": float(finite.quantile(0.25)) if len(finite) else float("nan"),
                    "q3": float(finite.quantile(0.75)) if len(finite) else float("nan"),
                }
            )
        for col in cols:
            if col == "C":
                continue
            x = pd.to_numeric(grp[col], errors="coerce")
            y = pd.to_numeric(grp["C"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 3:
                rho, p = stats.spearmanr(x[mask], y[mask])
            else:
                rho, p = float("nan"), float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "group": "Top40_spearman_vs_C",
                    "feature": col,
                    "n_non_missing": int(mask.sum()),
                    "median": float(rho) if np.isfinite(rho) else float("nan"),
                    "q1": float(p) if np.isfinite(p) else float("nan"),
                    "q3": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_xlsx(tables: dict[str, pd.DataFrame], path: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in tables.items():
        ws = wb.create_sheet(name[:31])
        if df is None or df.empty:
            ws.append(["empty"])
            continue
        ws.append(list(df.columns))
        for row in df.itertuples(index=False, name=None):
            out = []
            for v in row:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out.append(None)
                else:
                    out.append(v)
            ws.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def analyze_dataset(dataset: str, out_dir: Path) -> dict[str, Any]:
    sheets = load_sheets(dataset)
    targets, cell, seed, coverage = prepare_frames(dataset, sheets)
    audit = audit_dataset(dataset, targets, cell, seed, coverage)
    audit_df = pd.DataFrame(audit.rows)
    audit_df.insert(0, "dataset", dataset)
    write_csv(audit_df, out_dir / f"data_audit_{dataset}.csv")
    if not audit.ok:
        return {"dataset": dataset, "ok": False, "issues": audit.issues, "audit": audit_df}

    # 1) Pool all 70 instances (OLD_ADDITIONAL_10 + NEW_RANDOM_60) only to select Top40.
    level_all = build_target_level(dataset, seed)
    scores_all = composite_scores(level_all)
    top40 = scores_all.loc[scores_all["group"] == "Top40"].copy()
    bottom10 = scores_all.loc[scores_all["group"] == "Bottom10"].copy()
    middle20 = scores_all.loc[scores_all["group"] == "Middle20"].copy()
    top_ids = set(top40["target_instance_id"].tolist())
    if len(top_ids) != 40:
        raise RuntimeError(f"{dataset}: expected 40 Top40 ids, got {len(top_ids)}")

    # 2) All headline / descriptive statistical analysis uses only those 40 targets.
    targets40 = targets.loc[targets["target_instance_id"].isin(top_ids)].copy()
    cell40 = cell.loc[cell["target_instance_id"].isin(top_ids)].copy()
    seed40 = seed.loc[seed["target_instance_id"].isin(top_ids)].copy()
    level = level_all.loc[level_all["target_instance_id"].isin(top_ids)].copy()
    cross = build_cross_horizon(level)
    scores = top40.copy()

    horizon_summary, boot_h = summarize_horizon(level, targets40)
    cross_summary, boot_c, tests = summarize_cross(cross, level, targets40)
    ranks = method_ranks(seed40)

    features = load_sales_features(dataset, targets40)
    features = attach_sources_and_cv(features, cell40, seed40, scores_all)
    feat_summary = feature_group_summary(features)

    write_csv(scores_all, out_dir / f"target_composite_scores_{dataset}.csv")
    write_csv(top40, out_dir / f"top40_targets_{dataset}.csv")
    write_csv(bottom10, out_dir / f"bottom10_targets_{dataset}.csv")
    write_csv(middle20, out_dir / f"middle20_targets_{dataset}.csv")
    write_csv(level, out_dir / f"target_level_by_horizon_{dataset}.csv")
    write_csv(cross, out_dir / f"target_cross_horizon_summary_{dataset}.csv")
    write_csv(features, out_dir / f"target_features_{dataset}.csv")
    write_csv(feat_summary, out_dir / f"feature_group_summary_{dataset}.csv")
    write_csv(tests, out_dir / f"paired_tests_{dataset}.csv")
    write_csv(pd.concat([boot_h, boot_c], ignore_index=True), out_dir / f"bootstrap_intervals_{dataset}.csv")
    write_csv(ranks, out_dir / f"method_ranks_{dataset}.csv")
    write_csv(horizon_summary, out_dir / f"horizon_summary_{dataset}.csv")
    write_csv(cross_summary, out_dir / f"cross_horizon_dataset_summary_{dataset}.csv")

    return {
        "dataset": dataset,
        "ok": True,
        "issues": [],
        "audit": audit_df,
        "level": level,
        "cross": cross,
        "scores": scores,
        "scores_all": scores_all,
        "top40": top40,
        "bottom10": bottom10,
        "middle20": middle20,
        "features": features,
        "feature_summary": feat_summary,
        "tests": tests,
        "bootstrap": pd.concat([boot_h, boot_c], ignore_index=True),
        "ranks": ranks,
        "horizon_summary": horizon_summary,
        "cross_summary": cross_summary,
    }


def write_report(results: list[dict[str, Any]], out_dir: Path, started: str) -> None:
    lines = [
        "# SITA D4/D5/D6 Top-40 Target Analysis Report",
        "",
        f"- started_utc: {started}",
        f"- finished_utc: {utc_now()}",
        f"- script: {SCRIPT_VERSION}",
        f"- inputs: {INPUT_DIR}/D{{4,5,6}}_70target_BL_SITA.xlsx",
        f"- selection: pool all 70 targets (OLD_ADDITIONAL_10 + NEW_RANDOM_60, no cohort split)",
        f"- analysis set: Top40 only; headline Wilcoxon/Holm/bootstrap/ranks/features all on n=40",
        f"- tie_tolerance: {TIE_TOL}",
        f"- bootstrap: {N_BOOT} cluster resamples by target_identity within Top40",
        f"- main tests: cross-horizon sMAPE on Top40, 24 Wilcoxon + Holm",
        "",
    ]
    for res in results:
        ds = res["dataset"]
        lines.append(f"## {ds}")
        if not res["ok"]:
            lines.append("Audit FAILED; Top-40 selection stopped for this dataset.")
            for issue in res["issues"]:
                lines.append(f"- {issue}")
            lines.append("")
            continue
        scores_all = res.get("scores_all", res["scores"])
        top = res["top40"]
        lines.append(
            f"- screened from 70 pooled instances; analysis n={len(top)}; "
            f"selection counts Top40/Middle20/Bottom10 = "
            f"{(scores_all.group=='Top40').sum()}/{(scores_all.group=='Middle20').sum()}/{(scores_all.group=='Bottom10').sum()}"
        )
        lines.append(f"- Top40 C range: [{top.C.min():.4f}, {top.C.max():.4f}]")
        lines.append("- Top5 by C:")
        for r in top.head(5).itertuples(index=False):
            lines.append(f"  - `{r.target_instance_id}` C={r.C:.4f}")
        tests = res["tests"]
        sig = tests.loc[tests["p_holm"] < 0.05]
        lines.append(f"- Significant Top40 cross-horizon sMAPE tests after Holm: {len(sig)} / {len(tests)}")
        if not tests.empty:
            lines.append("- Per-baseline geo% / W-T-L / Holm p:")
            for r in tests.sort_values("baseline").itertuples(index=False):
                lines.append(
                    f"  - {r.baseline}: geo={r.geo_improvement:.2f}% "
                    f"W/T/L={r.wins}/{r.ties}/{r.losses} p_holm={r.p_holm:.3g}"
                )
        feat = res["features"]
        lines.append(
            f"- Top40 feature OK: {(feat.feature_status=='OK').sum()} / {len(feat)}; "
            f"pre30 zero-share median={feat.pre30_zero_share.median():.4f}; "
            f"test zero-share median={feat.test_zero_share.median():.4f}"
        )
        lines.append("")
    (out_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = utc_now()
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": started,
        "script_version": SCRIPT_VERSION,
        "inputs": {ds: str(workbook_path(ds)) for ds in DATASETS},
        "selection_pool": "all_70_pooled_no_cohort_split",
        "analysis_set": "top40_only",
        "n_boot": N_BOOT,
        "tie_tolerance": TIE_TOL,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results = []
    for ds in DATASETS:
        print(f"[{utc_now()}] analyzing {ds}", flush=True)
        results.append(analyze_dataset(ds, out_dir))

    # combined deliveries
    audits = [r["audit"] for r in results if r.get("audit") is not None]
    write_csv(pd.concat(audits, ignore_index=True), out_dir / "data_audit.csv")

    ok_results = [r for r in results if r["ok"]]
    if not ok_results:
        write_report(results, out_dir, started)
        print("ALL DATASETS FAILED AUDIT", file=sys.stderr)
        return 2

    def cat(key: str) -> pd.DataFrame:
        return pd.concat([r[key] for r in ok_results], ignore_index=True)

    tables = {
        "data_audit": pd.concat(audits, ignore_index=True),
        "target_level_by_horizon": cat("level"),
        "target_cross_horizon": cat("cross"),
        "target_composite_scores": cat("scores_all"),
        "top40_targets": cat("top40"),
        "bottom10_targets": cat("bottom10"),
        "middle20_targets": cat("middle20"),
        "target_features": cat("features"),
        "feature_group_summary": cat("feature_summary"),
        "paired_tests": cat("tests"),
        "bootstrap_intervals": cat("bootstrap"),
        "method_ranks": cat("ranks"),
    }
    for name, df in tables.items():
        write_csv(df, out_dir / f"{name}.csv")
    write_xlsx(tables, out_dir / "SITA_D456_target_analysis.xlsx")
    write_report(results, out_dir, started)

    for r in ok_results:
        sc_all = r["scores_all"]
        assert len(sc_all) == 70
        assert (sc_all["group"] == "Top40").sum() == 40
        assert (sc_all["group"] == "Bottom10").sum() == 10
        assert (sc_all["group"] == "Middle20").sum() == 20
        assert len(r["top40"]) == 40
        assert r["level"]["target_instance_id"].nunique() == 40
        assert set(r["horizon_summary"]["n_targets"].unique()) == {40}
        assert int(r["tests"]["n_valid"].max()) <= 40
        assert len(r["features"]) == 40
    print(f"[{utc_now()}] done -> {out_dir}")
    for r in results:
        print(f"  {r['dataset']}: {'OK' if r['ok'] else 'FAIL'}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
