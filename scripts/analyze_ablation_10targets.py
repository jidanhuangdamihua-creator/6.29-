#!/usr/bin/env python3
"""Target-level stats for the 10-target F1-F4/P1/P2 ablation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "top40_mod", ROOT / "scripts" / "analyze_sita_d456_top40_targets.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

CSV = ROOT / "results" / "ablation_10targets_5seed.csv"
OUT = ROOT / "results" / "ablation_10targets_stats.csv"
SUMMARY = ROOT / "docs" / "ablation_10targets_summary.md"
CONTRASTS = (("F2", "F1"), ("F4", "F2"), ("F4", "F3"), ("F4", "P1"), ("F4", "P2"))
METRICS = ("RMSE", "sMAPE", "MAE")
HORIZONS = (1, 3, 5)
DATASETS = ("D4", "D5", "D6")


def seed_mean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "target", "condition", "h"], as_index=False)[list(METRICS)].mean()


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


def analyze() -> pd.DataFrame:
    raw = pd.read_csv(CSV)
    means = seed_mean(raw)
    rows = []
    for dataset in DATASETS:
        for h in HORIZONS:
            for metric in METRICS:
                for cand, ctrl in CONTRASTS:
                    a = means.loc[(means.dataset == dataset) & (means.h == h) & (means.condition == cand)]
                    b = means.loc[(means.dataset == dataset) & (means.h == h) & (means.condition == ctrl)]
                    merged = a.merge(b, on="target", suffixes=("_cand", "_ctrl"))
                    diff = merged[f"{metric}_cand"] - merged[f"{metric}_ctrl"]
                    d = diff.to_numpy(dtype=float)
                    stat, p_raw, status = mod.wilcoxon_two_sided(d)
                    med, lo, hi = bootstrap_median(d, seed=mod._stable_seed(dataset, h, metric, cand, ctrl))
                    rows.append({
                        "dataset": dataset, "h": h, "metric": metric,
                        "contrast": f"{cand}-{ctrl}", "candidate": cand, "control": ctrl,
                        "n_targets": int(len(merged)), "median_diff": med,
                        "ci95_low": lo, "ci95_high": hi, "wilcoxon_stat": stat,
                        "p_raw": p_raw, "status": status,
                    })
    # Holm within dataset x metric x horizon (5 contrasts)
    families: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        families.setdefault((row["dataset"], row["metric"], row["h"]), []).append(i)
    for idxs in families.values():
        adj = mod.holm_adjust([rows[i]["p_raw"] for i in idxs])
        for i, padj in zip(idxs, adj):
            rows[i]["p_holm"] = padj
            rows[i]["sig_holm_0.05"] = bool(np.isfinite(padj) and padj < 0.05)
    stats = pd.DataFrame(rows)

    # stability: seed SD then median across targets
    sd = (
        raw.groupby(["dataset", "target", "condition", "h"], as_index=False)[list(METRICS)]
        .std(ddof=1)
    )
    stab_rows = []
    for dataset in DATASETS:
        for h in HORIZONS:
            for cond in ("F1", "F2", "F3", "F4", "P1", "P2"):
                sub = sd.loc[(sd.dataset == dataset) & (sd.h == h) & (sd.condition == cond)]
                for metric in METRICS:
                    stab_rows.append({
                        "dataset": dataset, "h": h, "condition": cond, "metric": metric,
                        "median_seed_sd": float(np.nanmedian(sub[metric].to_numpy(dtype=float))) if len(sub) else float("nan"),
                        "n_targets": int(len(sub)),
                    })
    stab = pd.DataFrame(stab_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT, index=False)
    stab.to_csv(ROOT / "results" / "ablation_10targets_stability.csv", index=False)
    write_summary(stats, stab, means)
    return stats


def _cell(stats: pd.DataFrame, dataset: str, h: int, metric: str, contrast: str) -> pd.Series:
    hit = stats.loc[
        (stats.dataset == dataset) & (stats.h == h) & (stats.metric == metric) & (stats.contrast == contrast)
    ]
    return hit.iloc[0]


def write_summary(stats: pd.DataFrame, stab: pd.DataFrame, means: pd.DataFrame) -> None:
    lines = [
        "# 10-target 消融摘要",
        "",
        "统计单位是目标：先对种子 42–46 取均值，再在 10 个目标上配对。",
        "Holm 在每个「数据集 × 指标 × h」内对 5 项对比校正。不改写论文原结论。",
        "",
        "## 稳定性（种子 SD 的目标中位数）",
        "",
        stab.to_csv(index=False),
        "",
        "## 配对结果要点",
        "",
    ]
    paper_claims = []
    for dataset in DATASETS:
        for h in HORIZONS:
            f2f1 = _cell(stats, dataset, h, "RMSE", "F2-F1")
            f4f2 = _cell(stats, dataset, h, "RMSE", "F4-F2")
            f4f3 = _cell(stats, dataset, h, "RMSE", "F4-F3")
            f4p1 = _cell(stats, dataset, h, "RMSE", "F4-P1")
            f4p2 = _cell(stats, dataset, h, "RMSE", "F4-P2")
            if int(f2f1["n_targets"]) < 2 or not np.isfinite(float(f2f1.median_diff)):
                lines.append(f"- {dataset} h={h} RMSE：配对格子不完整，跳过")
                continue
            s_f2f1 = bool(f2f1["sig_holm_0.05"])
            s_f4f2 = bool(f4f2["sig_holm_0.05"])
            s_f4p1 = bool(f4p1["sig_holm_0.05"])
            s_f4p2 = bool(f4p2["sig_holm_0.05"])
            lines.append(
                f"- {dataset} h={h} RMSE：F2-F1 中位差 {f2f1.median_diff:.4g} (Holm p={f2f1.p_holm:.3g}, sig={s_f2f1}); "
                f"F4-F2 {f4f2.median_diff:.4g} (p={f4f2.p_holm:.3g}, sig={s_f4f2}); "
                f"F4-F3 {f4f3.median_diff:.4g}; F4-P1 {f4p1.median_diff:.4g} (sig={s_f4p1}); "
                f"F4-P2 {f4p2.median_diff:.4g} (sig={s_f4p2})"
            )
            complete = int(f2f1["n_targets"]) >= 10
            if not complete:
                paper_claims.append(f"{dataset} h={h} 目标数={int(f2f1['n_targets'])} < 10，暂不对照论文结论")
                continue
            if dataset in {"D4", "D6"} and s_f4p1:
                paper_claims.append(f"{dataset} h={h} F4 vs P1 RMSE 显著，与「D4/D6 无差异」不一致")
            if dataset == "D5" and not s_f4p1:
                paper_claims.append(f"D5 h={h} F4 vs P1 RMSE 不显著，与「D5 有差异」不一致")
            if (not s_f2f1) or float(f2f1.median_diff) >= 0:
                paper_claims.append(f"{dataset} h={h} RevIN (F2-F1) RMSE 未显示 F2 更低")
            if not s_f4f2:
                paper_claims.append(f"{dataset} h={h} F4 相对 F2 的额外降幅不显著")
            if (not s_f4p2) or float(f4p2.median_diff) >= 0:
                paper_claims.append(f"{dataset} h={h} P2 未显著劣于 F4")
    lines += ["", "## 与论文原结论不一致的项（原样列出，不改写论文）", ""]
    if paper_claims:
        lines.extend(f"- {item}" for item in paper_claims)
    else:
        lines.append("- 本次 10-target 结果未出现与论文 4.3 方向相反的显著结论。")
    lines += ["", "对照论文：RevIN 降幅大；F4 相对 F2 有额外小幅降幅；P1 在 D4/D6 无差异而 D5 有差异；P2 劣于 F4。"]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not CSV.is_file():
        print(f"missing {CSV}")
        return 1
    n = len(pd.read_csv(CSV))
    print(f"rows={n} expected=2700 missing={2700-n}")
    if n < 2700:
        print("incomplete; stats still written from available rows")
    analyze()
    print(f"wrote {OUT} and {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
