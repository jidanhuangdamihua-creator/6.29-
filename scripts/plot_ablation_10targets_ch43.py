#!/usr/bin/env python3
"""Chapter 4.3 10-target ablation tables (S6a/S6b/S6d/S7) and Figures 3–4.

Reads existing 10-target CSVs only. Does not modify original result files.
Does not rerun experiments. S7 / S6c 10-target gaps are documented, not filled.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ablation10_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "ablation10_xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "results" / "ablation_10targets_5seed.csv"
STATS_CSV = ROOT / "results" / "ablation_10targets_stats.csv"
STAB_CSV = ROOT / "results" / "ablation_10targets_stability.csv"
S7_SOURCE = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/论文证据/第四章数据/第四章表格输出/table3_supplement.csv"
)
ALT_LR_SOURCE = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/论文证据/第四章数据/第四章表格输出/table3_ablation_alt_lr.csv"
)
OUT_DIR = ROOT / "figures" / "ablation_10targets"
DOCS_DIR = ROOT / "docs"

DATASETS = ("D4", "D5", "D6")
HORIZONS = (1, 3, 5)
CONDITIONS_MAIN = ("F1", "F2", "F3", "F4", "P1", "P2")
FIG3_CONDS = ("F1", "F2", "F3", "F4")
FIG4_CONDS = ("P1", "P2", "F2", "F3")
S6B_CONDS = ("F2", "F3", "P1", "P2")
METRICS_TABLE = ("sMAPE", "RMSE")
N_BOOT = 100_000
BOOT_SEED = 20261005
PANEL_LABELS = {"D4": "(a)  D4", "D5": "(b)  D5", "D6": "(c)  D6"}

# 2×2 encoding: grey = random init, blue = source init; open = no RevIN, filled = RevIN.
# Palette aligned with plot_paper_fig2_fig3.py (serif + forest blue) and Okabe–Ito.
STYLE = {
    "F1": dict(color="#7A7A7A", facecolor="white", marker="o", label="F1 random init, no RevIN"),
    "F2": dict(color="#4D4D4D", facecolor="#4D4D4D", marker="o", label="F2 random init, RevIN"),
    "F3": dict(color="#0072B2", facecolor="white", marker="o", label="F3 source init, no RevIN"),
    "F4": dict(color="#1B4F72", facecolor="#1B4F72", marker="o", label="F4 source init, RevIN"),
    "P1": dict(color="#E69F00", facecolor="#E69F00", marker="s", label="P1 frozen conv."),
    "P2": dict(color="#009E73", facecolor="#009E73", marker="D", label="P2 single source"),
}

S7_LABELS = {
    "P3": "P3 uniform fusion weights",
    "F4_random3": "F4 random-3 sources",
    "F4_bottom3": "F4 farthest-3 sources",
    "F4_k5": "F4 K=5",
    "F4_ell15": "F4 comparison window ℓ=15",
}
S7_LABELS_ZH = {
    "P3": "P3 均匀融合权重",
    "F4_random3": "F4 随机选源（Random-3）",
    "F4_bottom3": "F4 最远选源（Bottom-3）",
    "F4_k5": "F4 K=5",
    "F4_ell15": "F4 比较窗口 15 天",
}


def apply_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": True,
            "mathtext.fontset": "stix",
        }
    )


def _stable_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return BOOT_SEED + (sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 1_000_000)


def bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(N_BOOT, finite.size), replace=True)
    stats = np.median(draws, axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(np.median(finite)), float(lo), float(hi)


def seed_means(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["dataset", "target", "condition", "h"], as_index=False)[["sMAPE", "RMSE", "MAE"]].mean()


def fmt_pm(mean: float, sd: float) -> str:
    if not (np.isfinite(mean) and np.isfinite(sd)):
        return "—"
    return f"{mean:.2f} ± {sd:.2f}"


def fmt_iqr(med: float, q1: float, q3: float) -> str:
    if not (np.isfinite(med) and np.isfinite(q1) and np.isfinite(q3)):
        return "—"
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return f"{p:.3g}"
    return f"{p:.4f}"


def fmt_num(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax < 0.01 or ax >= 1000):
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def direction_of(diff: float) -> str:
    if not np.isfinite(diff) or diff == 0:
        return "equal to F4"
    return "higher than F4" if diff > 0 else "lower than F4"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_wide_table(rows: list[dict], conditions: tuple[str, ...], value_key: str, caption: str) -> str:
    header = "| Dataset | h | " + " | ".join(conditions) + " |"
    rule = "| --- | ---: | " + " | ".join("---" for _ in conditions) + " |"
    lookup = {(r["dataset"], int(r["h"])): r for r in rows}
    body = []
    for dataset in DATASETS:
        for h in HORIZONS:
            rec = lookup[(dataset, h)]
            cells = [rec[f"{value_key}_{c}"] for c in conditions]
            body.append("| " + " | ".join([dataset, str(h), *cells]) + " |")
    return "\n".join([caption, "", header, rule, *body, ""])


def build_s6a(means: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    for dataset in DATASETS:
        for h in HORIZONS:
            wide = {"dataset": dataset, "h": h, "n_targets": 10, "aggregation": "target-level after 5-seed mean"}
            for cond in CONDITIONS_MAIN:
                sub = means.loc[
                    (means.dataset == dataset) & (means.h == h) & (means.condition == cond)
                ]
                for metric in METRICS_TABLE:
                    vals = sub[metric].to_numpy(dtype=float)
                    mean = float(np.mean(vals))
                    sd = float(np.std(vals, ddof=1))
                    med = float(np.median(vals))
                    q1, q3 = np.quantile(vals, [0.25, 0.75])
                    long_rows.append(
                        {
                            "table": "S6a",
                            "dataset": dataset,
                            "h": h,
                            "condition": cond,
                            "metric": metric,
                            "n_targets": int(len(vals)),
                            "mean": mean,
                            "sd_across_targets": sd,
                            "mean_pm_sd": fmt_pm(mean, sd),
                            "median": med,
                            "q1": float(q1),
                            "q3": float(q3),
                            "median_q1_q3": fmt_iqr(med, float(q1), float(q3)),
                            "learning_rate": "3e-5" if dataset != "D5" else "3e-4",
                        }
                    )
                    if metric == "sMAPE":
                        wide[f"mean_sd_{cond}"] = fmt_pm(mean, sd)
                        wide[f"median_iqr_{cond}"] = fmt_iqr(med, float(q1), float(q3))
            wide_rows.append(wide)
    return long_rows, wide_rows


def build_s6b(stats: pd.DataFrame) -> list[dict]:
    """Rearrange stats.csv to (condition − F4). Do not recompute tests."""
    rows = []
    hit = stats.loc[
        (stats.control == "F4")
        | ((stats.candidate == "F4") & (stats.control.isin(S6B_CONDS)))
    ]
    for _, rec in hit.iterrows():
        if rec["metric"] not in METRICS_TABLE:
            continue
        if rec["candidate"] == "F4" and rec["control"] in S6B_CONDS:
            condition = str(rec["control"])
            median = -float(rec["median_diff"])
            lo = -float(rec["ci95_high"])
            hi = -float(rec["ci95_low"])
            source_contrast = rec["contrast"]
            sign_flipped = True
        elif rec["control"] == "F4" and rec["candidate"] in S6B_CONDS:
            condition = str(rec["candidate"])
            median = float(rec["median_diff"])
            lo = float(rec["ci95_low"])
            hi = float(rec["ci95_high"])
            source_contrast = rec["contrast"]
            sign_flipped = False
        else:
            continue
        rows.append(
            {
                "table": "S6b",
                "dataset": rec["dataset"],
                "h": int(rec["h"]),
                "metric": rec["metric"],
                "condition": condition,
                "contrast": f"{condition}-F4",
                "n_targets": int(rec["n_targets"]),
                "median_diff": median,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_holm": float(rec["p_holm"]),
                "p_raw": float(rec["p_raw"]),
                "wilcoxon_stat": rec["wilcoxon_stat"],
                "sig_holm_0.05": bool(rec["sig_holm_0.05"]),
                "direction": direction_of(median),
                "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0)),
                "source_contrast": source_contrast,
                "sign_flipped_from_source": sign_flipped,
                "note": "median and CI sign-flipped from stats.csv F4−condition; two-sided p unchanged",
            }
        )
    order = {(d, h, m, c): i for i, (d, h, m, c) in enumerate(
        (d, h, m, c) for d in DATASETS for h in HORIZONS for m in METRICS_TABLE for c in S6B_CONDS
    )}
    rows.sort(key=lambda r: order.get((r["dataset"], r["h"], r["metric"], r["condition"]), 10**6))
    return rows


def build_s6d(stab: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for h in HORIZONS:
            for metric in METRICS_TABLE:
                by_cond = {}
                for cond in CONDITIONS_MAIN:
                    sub = stab.loc[
                        (stab.dataset == dataset)
                        & (stab.h == h)
                        & (stab.condition == cond)
                        & (stab.metric == metric)
                    ]
                    sd = float(sub["median_seed_sd"].iloc[0]) if len(sub) else float("nan")
                    n = int(sub["n_targets"].iloc[0]) if len(sub) else 0
                    by_cond[cond] = (sd, n)
                f2 = by_cond["F2"][0]
                f4 = by_cond["F4"][0]
                ratio = f4 / f2 if (np.isfinite(f2) and np.isfinite(f4) and f2 != 0) else float("nan")
                for cond in CONDITIONS_MAIN:
                    rows.append(
                        {
                            "table": "S6d",
                            "dataset": dataset,
                            "h": h,
                            "condition": cond,
                            "metric": metric,
                            "median_seed_sd": by_cond[cond][0],
                            "n_targets": by_cond[cond][1],
                            "F4_over_F2_sd_ratio": ratio if cond in {"F2", "F4"} else "",
                        }
                    )
    return rows


def s6d_ratio_wide(stab: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for h in HORIZONS:
            rec = {"dataset": dataset, "h": h}
            for metric in METRICS_TABLE:
                cells = {}
                for cond in CONDITIONS_MAIN:
                    sub = stab.loc[
                        (stab.dataset == dataset)
                        & (stab.h == h)
                        & (stab.condition == cond)
                        & (stab.metric == metric)
                    ]
                    cells[cond] = float(sub["median_seed_sd"].iloc[0]) if len(sub) else float("nan")
                rec[f"{metric}_F4_over_F2"] = (
                    cells["F4"] / cells["F2"] if cells["F2"] not in (0, np.nan) and np.isfinite(cells["F2"]) else float("nan")
                )
                for cond in CONDITIONS_MAIN:
                    rec[f"{metric}_{cond}"] = cells[cond]
            rows.append(rec)
    return rows


def build_s7(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    src = pd.read_csv(path)
    rows = []
    for _, rec in src.iterrows():
        code = str(rec["condition_code"])
        rows.append(
            {
                "table": "S7",
                "coverage": "single-target only",
                "condition_code": code,
                "condition": S7_LABELS.get(code, code),
                "condition_zh": S7_LABELS_ZH.get(code, rec.get("condition", code)),
                "protocol": rec["protocol"],
                "dataset": rec["dataset"],
                "h": int(rec["horizon"]),
                "target_learning_rate": rec["target_learning_rate"],
                "smape_mean": float(rec["smape_mean"]),
                "smape_std": float(rec["smape_std"]),
                "smape_mean_pm_std": rec["smape_mean_pm_std"],
                "delta_vs_F4": float(rec["delta_vs_F4"]),
                "n_targets": 1,
                "n_seeds": 5,
                "note": "仅单目标；5-seed mean ± seed SD; not a 10-target summary",
            }
        )
    return rows


def write_s6a_md(wide: list[dict], path: Path) -> None:
    lines = [
        "# Table S6a. Ablation sMAPE at the main learning rate (10 targets)",
        "",
        "Each cell is the **target-level mean ± SD across 10 targets**, after averaging seeds 42–46 within each target.",
        "Datasets D4/D6 use target learning rate 3e-5; D5 uses 3e-4.",
        "Metrics are on the original sales scale. No 10-target swapped-learning-rate (S6c) results are included.",
        "",
        md_wide_table(wide, CONDITIONS_MAIN, "mean_sd", "### Mean ± SD across targets"),
        md_wide_table(wide, CONDITIONS_MAIN, "median_iqr", "### Median [Q1, Q3] across targets (backup)"),
        "Source: `results/ablation_10targets_5seed.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_s6b_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# Table S6b. Paired target-level differences versus F4 (condition − F4)",
        "",
        "Values are rearranged from `results/ablation_10targets_stats.csv` (no re-testing).",
        "Original contrasts are `F4−condition`; median and bootstrap 95% CI are sign-flipped to `condition−F4`.",
        "Two-sided Wilcoxon / Holm *p* values are unchanged. Holm was computed within each dataset × metric × *h* over the original five contrasts.",
        "Positive median: the condition has higher error than F4. Direction follows the sign of the median.",
        "",
    ]
    for metric in METRICS_TABLE:
        lines += [f"## {metric}", ""]
        header = (
            "| Dataset | h | Condition | Median (condition−F4) | Bootstrap 95% CI | "
            "Holm *p* | Direction | CI excludes 0 |"
        )
        rule = "| --- | ---: | --- | ---: | --- | ---: | --- | --- |"
        lines += [header, rule]
        for r in rows:
            if r["metric"] != metric:
                continue
            ci = f"[{fmt_num(r['ci95_low'])}, {fmt_num(r['ci95_high'])}]"
            lines.append(
                "| {dataset} | {h} | {condition} | {med} | {ci} | {p} | {d} | {ex} |".format(
                    dataset=r["dataset"],
                    h=r["h"],
                    condition=r["condition"],
                    med=fmt_num(r["median_diff"]),
                    ci=ci,
                    p=fmt_p(r["p_holm"]),
                    d=r["direction"],
                    ex="yes" if r["ci_excludes_zero"] else "no",
                )
            )
        lines.append("")
    lines += [
        "F1 versus F4 is not in `ablation_10targets_stats.csv` and is not invented here.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_s6d_md(ratio_rows: list[dict], path: Path) -> None:
    lines = [
        "# Table S6d. Seed-to-seed stability (median across 10 targets)",
        "",
        "For each target, compute the sample SD over seeds 42–46; then take the **median of those 10 SDs**.",
        "The F4/F2 column is the ratio of those median seed SDs. Source: `results/ablation_10targets_stability.csv`.",
        "",
    ]
    for metric in METRICS_TABLE:
        lines += [f"## {metric}", ""]
        header = "| Dataset | h | " + " | ".join(CONDITIONS_MAIN) + " | F4 / F2 |"
        rule = "| --- | ---: | " + " | ".join("---" for _ in CONDITIONS_MAIN) + " | ---: |"
        lines += [header, rule]
        for r in ratio_rows:
            cells = [fmt_num(r[f"{metric}_{c}"], nd=3) for c in CONDITIONS_MAIN]
            ratio = fmt_num(r[f"{metric}_F4_over_F2"], nd=3)
            lines.append("| " + " | ".join([r["dataset"], str(r["h"]), *cells, ratio]) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_s7_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# Table S7. Other source settings — **single-target only**",
        "",
        "**Coverage label: 仅单目标.** These numbers are the original one-target-per-dataset ablation (D1/D4/D5/D6, seeds 42–46).",
        "They are **not** 10-target summaries. No 10-target results exist in `results/` for these settings; nothing was re-run.",
        "",
        "Cells: 5-seed mean ± seed SD of test sMAPE. Δ = condition mean − main F4 mean (same dataset and *h*).",
        "",
    ]
    codes = list(dict.fromkeys(r["condition_code"] for r in rows))
    for code in codes:
        label = S7_LABELS.get(code, code)
        label_zh = S7_LABELS_ZH.get(code, code)
        lines += [f"## {label} / {label_zh}", ""]
        header = "| Dataset | h | sMAPE mean ± SD | Δ vs F4 | Learning rate |"
        rule = "| --- | ---: | --- | ---: | --- |"
        lines += [header, rule]
        for r in rows:
            if r["condition_code"] != code:
                continue
            delta = r["delta_vs_F4"]
            dtxt = f"{delta:+.4g}"
            lines.append(
                f"| {r['dataset']} | {r['h']} | {r['smape_mean_pm_std']} | {dtxt} | {r['target_learning_rate']} |"
            )
        lines.append("")
    lines += [
        "Source (read-only): `论文文稿相关/论文证据/第四章数据/第四章表格输出/table3_supplement.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def inventory_gaps() -> dict:
    raw_names = [p.name for p in (ROOT / "results").glob("*")]
    keywords = ("p3", "random3", "bottom", "k5", "ell15", "alt_lr", "uniform", "farthest")
    hits = [n for n in raw_names if any(k in n.lower() for k in keywords)]
    ten_target_conditions = {"F1", "F2", "F3", "F4", "P1", "P2"}
    s7_settings = ("P3", "F4_random3", "F4_bottom3", "F4_k5", "F4_ell15")
    n_ds, n_tgt, n_h, n_seed = 3, 10, 3, 5
    s7_full = len(s7_settings) * n_ds * n_tgt * n_h * n_seed
    s7_new9 = len(s7_settings) * n_ds * 9 * n_h * n_seed
    s6c_conds = ("F1", "F2", "F3", "F4")
    s6c_full = len(s6c_conds) * n_ds * n_tgt * n_h * n_seed
    s6c_new9 = len(s6c_conds) * n_ds * 9 * n_h * n_seed
    return {
        "results_keyword_hits": hits,
        "ten_target_conditions": sorted(ten_target_conditions),
        "s7_settings": s7_settings,
        "s7_full_cells": s7_full,
        "s7_new9_cells": s7_new9,
        "s6c_full_cells": s6c_full,
        "s6c_new9_cells": s6c_new9,
        "alt_lr_single_target_present": ALT_LR_SOURCE.is_file(),
        "s7_single_target_present": S7_SOURCE.is_file(),
    }


def write_status_doc(gaps: dict, path: Path) -> None:
    hits = gaps["results_keyword_hits"] or ["(none)"]
    lines = [
        "# 10-target 消融：表 S7 / S6c 覆盖缺口",
        "",
        "盘点范围：`results/` 下是否已有 **每个数据集 10 个目标** 的 S7 与互换学习率（S6c）结果。",
        "未补跑任何实验，未改写任何原始结果文件。",
        "",
        "## 已有的 10 目标结果",
        "",
        "- 文件：`results/ablation_10targets_5seed.csv`（2700 行，完整）。",
        "- 数据集：D4、D5、D6（不含 D1）。",
        "- 条件：F1、F2、F3、F4、P1、P2。",
        "- 协议：主学习率（D4/D6 = 3e-5，D5 = 3e-4），h ∈ {1,3,5}，种子 42–46，每数据集 10 个目标。",
        "- 配套：`results/ablation_10targets_stats.csv`、`results/ablation_10targets_stability.csv`。",
        "",
        "## 表 S7（其他源设置）",
        "",
        "需要的 10 目标条件：均匀融合权重（P3）、随机选源（Random-3）、最远选源（Bottom-3）、K=5、比较窗口 15 天。",
        "",
        f"`results/` 中与这些关键词匹配的文件：{', '.join(hits)}。",
        "",
        "**结论：没有这些条件在 10 个目标上的结果。**",
        "",
        "单目标版本存在，来源为第四章原表 `table3_supplement.csv`（D1/D4/D5/D6 各 1 个目标、5 个种子）。",
        "本次输出的表 S7 沿用该单目标表，并在标题与表注中标注 **「仅单目标」**。",
        "",
        "若要把 S7 扩到与主消融相同的 10 目标（D4/D5/D6）：",
        "",
        f"- 完整格子：5 个条件 × 3 数据集 × 10 目标 × 3 个 h × 5 种子 = **{gaps['s7_full_cells']}**。",
        f"- 当前 10 目标结果中可复用：**0**。",
        f"- 原单目标 † 格（D4/D5/D6 各 1 个目标）已在 `消融实验/runs/` 的 P3、F4/random3、F4/bottom3、F4/k5、F4/ell15 中；若允许复用这 225 格，仍需新跑 9 个新目标：**{gaps['s7_new9_cells']}** 格。",
        "- D1 不在本次 10 目标范围内；原 S7 含 D1，扩到 10 目标时需另行决定是否纳入。",
        "",
        "## 表 S6c（互换学习率）",
        "",
        "互换学习率 = 原 S1：F1–F4 在「另一个学习率」下重跑（D4/D6 用 3e-4，D5 用 3e-5）。",
        "",
        "**结论：没有 10 目标的互换学习率结果，本次不伪造 S6c。**",
        "",
        "单目标 alt_lr 仍在原表 `table3_ablation_alt_lr.csv`（D1/D4/D5/D6，F1–F4；P1/P2 本来就没有 alt_lr）。",
        "",
        "若扩到 10 目标（D4/D5/D6，F1–F4）：",
        "",
        f"- 完整格子：4 个条件 × 3 数据集 × 10 目标 × 3 个 h × 5 种子 = **{gaps['s6c_full_cells']}**。",
        f"- 当前 10 目标结果中可复用：**0**。",
        f"- 若复用 `消融实验/runs/`{{F1,F2,F3,F4}}/alt_lr 的 † 格（180 格），仍需新跑：**{gaps['s6c_new9_cells']}** 格。",
        "",
        "## 不在本次图表中的内容",
        "",
        "- 图 3、图 4、表 S6a/S6b/S6d 只用主学习率、10 目标、F1–F4/P1/P2。",
        "- 图 4 按 10 目标数据只画 P1、P2、F2、F3 相对 F4 的差值；不把 S7 的单目标条件画进去。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig3_summaries(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for h in HORIZONS:
            for cond in FIG3_CONDS:
                vals = means.loc[
                    (means.dataset == dataset) & (means.h == h) & (means.condition == cond),
                    "sMAPE",
                ].to_numpy(dtype=float)
                med, lo, hi = bootstrap_median(vals, seed=_stable_seed(dataset, h, "sMAPE", cond, "fig3"))
                rows.append(
                    {
                        "dataset": dataset,
                        "h": h,
                        "condition": cond,
                        "n_targets": int(len(vals)),
                        "median": med,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def jitter_x(dataset: str, target: str, scale: float = 0.028) -> float:
    seed = _stable_seed("jitter", dataset, target)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-scale, scale))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)


def plot_fig3(means: pd.DataFrame, summary: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 3.62), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.24, wspace=0.22)
    x_h = {1: 0.0, 3: 1.0, 5: 2.0}
    offsets = dict(zip(FIG3_CONDS, (-0.27, -0.09, 0.09, 0.27)))
    plotted = means.loc[means.condition.isin(FIG3_CONDS), "sMAPE"].to_numpy(dtype=float)
    y_lo = max(12.0, float(np.nanmin(plotted)) / 1.20)
    y_hi = float(np.nanmax(plotted)) * 1.18

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(PANEL_LABELS[dataset], loc="left", pad=6)
        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["h = 1", "h = 3", "h = 5"])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks([20, 30, 50, 100, 200])
        ax.set_yticklabels(["20", "30", "50", "100", "200"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A8A8A")
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.tick_params(axis="x", length=0, pad=3)
        ax.grid(axis="y", color="#F0F0F0", linewidth=0.6, zorder=0, which="major")
        ax.set_axisbelow(True)

        for cond in FIG3_CONDS:
            st = STYLE[cond]
            xs, ys = [], []
            for h in HORIZONS:
                sub = means.loc[
                    (means.dataset == dataset) & (means.h == h) & (means.condition == cond)
                ]
                base = x_h[h] + offsets[cond]
                for _, row in sub.iterrows():
                    ax.scatter(
                        base + jitter_x(dataset, row["target"]),
                        row["sMAPE"],
                        s=11,
                        marker=st["marker"],
                        facecolors=st["facecolor"],
                        edgecolors=st["color"],
                        linewidths=0.55,
                        alpha=0.38,
                        zorder=2,
                        clip_on=False,
                    )
                rec = summary.loc[
                    (summary.dataset == dataset) & (summary.h == h) & (summary.condition == cond)
                ].iloc[0]
                xs.append(base)
                ys.append(float(rec["median"]))
                yerr = np.array(
                    [
                        [float(rec["median"] - rec["ci95_low"])],
                        [float(rec["ci95_high"] - rec["median"])],
                    ]
                )
                ax.errorbar(
                    base,
                    rec["median"],
                    yerr=yerr,
                    fmt="none",
                    ecolor=st["color"],
                    elinewidth=1.15,
                    capsize=2.2,
                    capthick=1.15,
                    zorder=3,
                )
            ax.plot(
                xs,
                ys,
                color=st["color"],
                linewidth=1.35,
                zorder=3,
                solid_capstyle="round",
            )
            ax.scatter(
                xs,
                ys,
                s=38,
                marker=st["marker"],
                facecolors=st["facecolor"],
                edgecolors=st["color"],
                linewidths=1.05,
                zorder=4,
                clip_on=False,
            )

        ax.set_xlabel("Forecast horizon")
        if dataset == "D4":
            ax.set_ylabel("Test sMAPE (%, log scale)")

    handles = []
    for cond in FIG3_CONDS:
        st = STYLE[cond]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=st["marker"],
                color=st["color"],
                markerfacecolor=st["facecolor"],
                markeredgecolor=st["color"],
                markeredgewidth=1.0,
                markersize=6.2,
                linewidth=1.2,
                label=st["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.01),
        columnspacing=1.3,
        handletextpad=0.45,
        fontsize=8,
    )
    save_figure(fig, stem)
    plt.close(fig)


def plot_fig4(means: pd.DataFrame, stats_s6b: list[dict], stem: Path) -> None:
    s6b = pd.DataFrame(stats_s6b)
    s6b = s6b.loc[s6b.metric == "sMAPE"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 3.62), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.24, wspace=0.22)
    x_h = {1: 0.0, 3: 1.0, 5: 2.0}
    offsets = dict(zip(FIG4_CONDS, (-0.27, -0.09, 0.09, 0.27)))

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(PANEL_LABELS[dataset], loc="left", pad=6)
        ax.axhline(0.0, color="#B5B5B5", linestyle=(0, (3, 2.5)), linewidth=0.8, zorder=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["h = 1", "h = 3", "h = 5"])
        ax.set_xlim(-0.55, 2.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8A8A8A")
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.tick_params(axis="x", length=0, pad=3)
        ax.grid(axis="y", color="#F0F0F0", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        f4 = means.loc[(means.dataset == dataset) & (means.condition == "F4"), ["target", "h", "sMAPE"]]
        f4 = f4.rename(columns={"sMAPE": "sMAPE_F4"})

        for cond in FIG4_CONDS:
            st = STYLE[cond]
            sub = means.loc[(means.dataset == dataset) & (means.condition == cond), ["target", "h", "sMAPE"]]
            merged = sub.merge(f4, on=["target", "h"], how="inner")
            merged["delta"] = merged["sMAPE"] - merged["sMAPE_F4"]
            xs, ys = [], []
            for h in HORIZONS:
                block = merged.loc[merged.h == h]
                base = x_h[h] + offsets[cond]
                for _, row in block.iterrows():
                    ax.scatter(
                        base + jitter_x(dataset, row["target"]),
                        row["delta"],
                        s=11,
                        marker=st["marker"],
                        facecolors=st["facecolor"],
                        edgecolors=st["color"],
                        linewidths=0.55,
                        alpha=0.38,
                        zorder=2,
                        clip_on=False,
                    )
                rec = s6b.loc[
                    (s6b.dataset == dataset) & (s6b.h == h) & (s6b.condition == cond)
                ].iloc[0]
                xs.append(base)
                ys.append(float(rec["median_diff"]))
                yerr = np.array(
                    [
                        [float(rec["median_diff"] - rec["ci95_low"])],
                        [float(rec["ci95_high"] - rec["median_diff"])],
                    ]
                )
                ax.errorbar(
                    base,
                    rec["median_diff"],
                    yerr=yerr,
                    fmt="none",
                    ecolor=st["color"],
                    elinewidth=1.15,
                    capsize=2.2,
                    capthick=1.15,
                    zorder=3,
                )
            ax.plot(xs, ys, color=st["color"], linewidth=1.35, zorder=3, solid_capstyle="round")
            ax.scatter(
                xs,
                ys,
                s=38,
                marker=st["marker"],
                facecolors=st["facecolor"],
                edgecolors=st["color"],
                linewidths=1.05,
                zorder=4,
                clip_on=False,
            )

        ax.set_xlabel("Forecast horizon")
        if dataset == "D4":
            ax.set_ylabel("sMAPE (condition − F4)")

    handles = []
    for cond in FIG4_CONDS:
        st = STYLE[cond]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=st["marker"],
                color=st["color"],
                markerfacecolor=st["facecolor"],
                markeredgecolor=st["color"],
                markeredgewidth=1.0,
                markersize=6.2,
                linewidth=1.2,
                label=st["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.01),
        columnspacing=1.1,
        handletextpad=0.4,
        fontsize=8,
    )
    save_figure(fig, stem)
    plt.close(fig)


def write_sources(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Chapter 4.3 10-target figure/table sources",
                "",
                "| Output | Source |",
                "| --- | --- |",
                "| Table S6a | `results/ablation_10targets_5seed.csv` (5-seed means, then 10-target mean±SD / median IQR) |",
                "| Table S6b | `results/ablation_10targets_stats.csv` (rearranged to condition−F4; tests not recomputed) |",
                "| Table S6d | `results/ablation_10targets_stability.csv` (median seed SD across 10 targets; F4/F2 ratio derived) |",
                "| Table S7 | original single-target `table3_supplement.csv` (labelled 仅单目标); no 10-target file |",
                "| Figure 3 | `results/ablation_10targets_5seed.csv`; median bootstrap CI computed here (N_BOOT=100000) |",
                "| Figure 4 points | `results/ablation_10targets_5seed.csv` (target-level 5-seed means, paired vs F4) |",
                "| Figure 4 median/CI | `results/ablation_10targets_stats.csv` via Table S6b (sMAPE, sign-flipped) |",
                "",
                "Original result files were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    apply_journal_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for required in (RAW_CSV, STATS_CSV, STAB_CSV):
        if not required.is_file():
            print(f"MISSING {required}")
            return 1

    raw = pd.read_csv(RAW_CSV)
    stats = pd.read_csv(STATS_CSV)
    stab = pd.read_csv(STAB_CSV)
    if len(raw) != 2700:
        print(f"WARNING expected 2700 rows in 5seed csv, got {len(raw)}")
    means = seed_means(raw)

    s6a_long, s6a_wide = build_s6a(means)
    s6b = build_s6b(stats)
    s6d = build_s6d(stab)
    s6d_wide = s6d_ratio_wide(stab)
    s7 = build_s7(S7_SOURCE)
    gaps = inventory_gaps()

    write_csv(
        OUT_DIR / "table_S6a_smape_long.csv",
        s6a_long,
        [
            "table",
            "dataset",
            "h",
            "condition",
            "metric",
            "n_targets",
            "mean",
            "sd_across_targets",
            "mean_pm_sd",
            "median",
            "q1",
            "q3",
            "median_q1_q3",
            "learning_rate",
        ],
    )
    write_csv(
        OUT_DIR / "table_S6a_smape_mean_sd.csv",
        s6a_wide,
        ["dataset", "h", "n_targets", *[f"mean_sd_{c}" for c in CONDITIONS_MAIN]],
    )
    write_csv(
        OUT_DIR / "table_S6a_smape_median_iqr.csv",
        s6a_wide,
        ["dataset", "h", "n_targets", *[f"median_iqr_{c}" for c in CONDITIONS_MAIN]],
    )
    write_s6a_md(s6a_wide, OUT_DIR / "table_S6a.md")

    write_csv(
        OUT_DIR / "table_S6b_vs_F4.csv",
        s6b,
        [
            "table",
            "dataset",
            "h",
            "metric",
            "condition",
            "contrast",
            "n_targets",
            "median_diff",
            "ci95_low",
            "ci95_high",
            "p_holm",
            "p_raw",
            "wilcoxon_stat",
            "sig_holm_0.05",
            "direction",
            "ci_excludes_zero",
            "source_contrast",
            "sign_flipped_from_source",
            "note",
        ],
    )
    write_s6b_md(s6b, OUT_DIR / "table_S6b.md")

    write_csv(
        OUT_DIR / "table_S6d_stability.csv",
        s6d,
        [
            "table",
            "dataset",
            "h",
            "condition",
            "metric",
            "median_seed_sd",
            "n_targets",
            "F4_over_F2_sd_ratio",
        ],
    )
    write_s6d_md(s6d_wide, OUT_DIR / "table_S6d.md")

    if s7:
        write_csv(
            OUT_DIR / "table_S7_single_target.csv",
            s7,
            [
                "table",
                "coverage",
                "condition_code",
                "condition",
                "condition_zh",
                "protocol",
                "dataset",
                "h",
                "target_learning_rate",
                "smape_mean",
                "smape_std",
                "smape_mean_pm_std",
                "delta_vs_F4",
                "n_targets",
                "n_seeds",
                "note",
            ],
        )
        write_s7_md(s7, OUT_DIR / "table_S7_single_target.md")
    else:
        print("WARNING S7 source missing; status doc will still list the gap")

    write_status_doc(gaps, DOCS_DIR / "ablation_10targets_s7_status.md")
    write_sources(OUT_DIR / "SOURCES.md")

    fig3_sum = fig3_summaries(means)
    fig3_sum.to_csv(OUT_DIR / "fig3_median_bootstrap.csv", index=False)
    means.loc[means.condition.isin(FIG3_CONDS), ["dataset", "target", "condition", "h", "sMAPE"]].to_csv(
        OUT_DIR / "fig3_target_points.csv", index=False
    )
    plot_fig3(means, fig3_sum, OUT_DIR / "fig3_revin_init_ablation_10targets")

    # Figure 4 target-level deltas for audit
    f4 = means.loc[means.condition == "F4", ["dataset", "target", "h", "sMAPE"]].rename(columns={"sMAPE": "sMAPE_F4"})
    fig4_pts = means.loc[means.condition.isin(FIG4_CONDS)].merge(f4, on=["dataset", "target", "h"])
    fig4_pts = fig4_pts.assign(delta=fig4_pts["sMAPE"] - fig4_pts["sMAPE_F4"])
    fig4_pts[["dataset", "target", "condition", "h", "sMAPE", "sMAPE_F4", "delta"]].to_csv(
        OUT_DIR / "fig4_target_deltas.csv", index=False
    )
    plot_fig4(means, s6b, OUT_DIR / "fig4_vs_F4_10targets")

    print(f"WROTE_DIR={OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"WROTE={p}")
    print(f"WROTE={DOCS_DIR / 'ablation_10targets_s7_status.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
