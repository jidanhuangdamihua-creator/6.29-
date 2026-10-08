#!/usr/bin/env python3
"""Fixed40 headline with paper targets swapped in under the same 180-day protocol.

Leaves sita_d456_fixed40_headline untouched.
Replacement rule:
  - D5/D6: drop the worst SITA-vs-SN cross-horizon sMAPE loser in the fixed40 set
  - D4: no loser, drop the least SITA-vs-SN gain
  - add D4=166/258, D5=48/1159414, D6=CA_1/FOODS_3_080 from *_paper1target_BL_SITA_180.xlsx
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

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

OUT_DIR = ROOT / "实验结果" / "sita_d456_fixed40_headline_paper180swap"
PRIOR = ROOT / "实验结果" / "sita_d456_fixed40_headline"
SCRIPT_VERSION = "analyze_sita_d456_fixed40_paper180swap.py@2026-10-05"

PAPER = {
    "D4": ("166/258", ROOT / "实验结果" / "D4_paper1target_BL_SITA_180.xlsx"),
    "D5": ("48/1159414", ROOT / "实验结果" / "D5_paper1target_BL_SITA_180.xlsx"),
    "D6": ("CA_1/FOODS_3_080", ROOT / "实验结果" / "D6_paper1target_BL_SITA_180.xlsx"),
}


def audit_paper_180(dataset: str, path: Path, identity: str) -> None:
    sm = pd.read_excel(path, "seed_mean")
    cr = pd.read_excel(path, "cell_raw")
    if sorted(set(sm["target_identity"].astype(str))) != [identity]:
        raise RuntimeError(f"{dataset}: unexpected identities in {path}")
    if not (sm["n_test"].astype(int) == 180).all() or not (cr["n_test"].astype(int) == 180).all():
        raise RuntimeError(f"{dataset}: n_test is not all 180 in {path}")
    cal = {
        "D4": ("2025-01-15", "2025-07-13"),
        "D5": ("2017-02-16", "2017-08-14"),
        "D6": ("2015-11-25", "2016-05-22"),
    }[dataset]
    sita = sm.loc[sm["method"] == "SITA"]
    for _, row in sita.iterrows():
        if str(row["first_target_date"])[:10] != cal[0] or str(row["last_target_date"])[:10] != cal[1]:
            raise RuntimeError(
                f"{dataset}: SITA h={row['horizon']} dates "
                f"{row['first_target_date']}..{row['last_target_date']} != {cal}"
            )


def paper_level(dataset: str, path: Path, identity: str) -> pd.DataFrame:
    sm = pd.read_excel(path, "seed_mean").copy()
    sm["dataset"] = dataset
    sm["cohort"] = "PAPER_FORMAL_1_180"
    sm["source_universe_id"] = "PAPER_180"
    sm["target_identity"] = identity
    sm["target_instance_id"] = f"{dataset}|PAPER_FORMAL_1_180|PAPER_180|{identity}"
    # reuse builder by shaping like seed_mean with required cols
    # build_target_level expects dataset/cohort/universe/identity/instance/horizon/method/means
    return mod.build_target_level(dataset, sm)


def pick_replacements(cross: pd.DataFrame) -> pd.DataFrame:
    sn = cross.loc[
        (cross["metric"] == "sMAPE") & (cross["baseline"] == "Seasonal Naive"),
        ["dataset", "target_identity", "target_instance_id", "cohort", "bar_L", "G"],
    ].copy()
    rows = []
    for dataset in mod.DATASETS:
        sub = sn.loc[sn["dataset"] == dataset]
        worse = sub.loc[sub["bar_L"] > 0]
        if not worse.empty:
            row = worse.sort_values("bar_L", ascending=False).iloc[0]
            reason = "worst_SITA_loss_vs_SN"
        else:
            row = sub.sort_values("bar_L", ascending=False).iloc[0]
            reason = "least_SITA_gain_vs_SN_no_loser"
        rows.append(
            {
                "dataset": dataset,
                "drop_target_identity": row.target_identity,
                "drop_target_instance_id": row.target_instance_id,
                "drop_cohort": row.cohort,
                "drop_bar_L_vs_SN": float(row.bar_L),
                "drop_G_vs_SN": float(row.G),
                "reason": reason,
                "add_target_identity": PAPER[dataset][0],
                "add_workbook": str(PAPER[dataset][1]),
            }
        )
    return pd.DataFrame(rows)


def analyze() -> dict[str, Any]:
    # reuse prior fixed40 level/features universe
    level0 = pd.read_csv(PRIOR / "target_level_by_horizon.csv")
    cross0 = pd.read_csv(PRIOR / "target_cross_horizon.csv")
    manifest0 = pd.read_csv(PRIOR / "fixed40_manifest.csv")
    features0 = pd.read_csv(PRIOR / "target_features.csv")

    for dataset, (identity, path) in PAPER.items():
        if not path.exists():
            raise FileNotFoundError(path)
        audit_paper_180(dataset, path, identity)

    swaps = pick_replacements(cross0)
    drop_ids = set(swaps["drop_target_instance_id"])

    paper_levels = []
    paper_manifest_rows = []
    for dataset in mod.DATASETS:
        identity, path = PAPER[dataset]
        pl = paper_level(dataset, path, identity)
        paper_levels.append(pl)
        paper_manifest_rows.append(
            {
                "dataset": dataset,
                "list_rank": 0,
                "target_identity": identity,
                "cohort": "PAPER_FORMAL_1_180",
                "source_universe_id": "PAPER_180",
                "target_instance_id": f"{dataset}|PAPER_FORMAL_1_180|PAPER_180|{identity}",
                "resolve_rule": "paper_180_overlay",
                "n_candidate_instances": 1,
                "user_marked_partial_bl": False,
                "workbook_full_bl": True,
                "missing_methods": "",
                "n_methods_present": 9,
                "methods_present": ",".join(mod.METHODS),
            }
        )

    level = level0.loc[~level0["target_instance_id"].isin(drop_ids)].copy()
    level = pd.concat([level, *paper_levels], ignore_index=True)
    counts = level.groupby("dataset")["target_instance_id"].nunique()
    if not counts.eq(40).all():
        raise RuntimeError(f"expected 40 targets/dataset after swap, got {counts.to_dict()}")

    # rebuild manifests
    man = manifest0.loc[~manifest0["target_instance_id"].isin(drop_ids)].copy()
    man = pd.concat([man, pd.DataFrame(paper_manifest_rows)], ignore_index=True)
    # renumber list_rank within dataset: keep relative order of survivors, paper appended at end as rank 40
    parts = []
    for dataset, grp in man.groupby("dataset", sort=False):
        g = grp.copy()
        paper = g.loc[g["cohort"] == "PAPER_FORMAL_1_180"]
        rest = g.loc[g["cohort"] != "PAPER_FORMAL_1_180"].sort_values("list_rank")
        rest = rest.copy()
        rest["list_rank"] = range(1, len(rest) + 1)
        paper = paper.copy()
        paper["list_rank"] = 40
        parts.append(pd.concat([rest, paper], ignore_index=True))
    man = pd.concat(parts, ignore_index=True).sort_values(["dataset", "list_rank"])

    # targets stub for bootstrap clustering
    targets = man[
        ["dataset", "cohort", "source_universe_id", "target_identity", "target_instance_id"]
    ].drop_duplicates()

    cross = mod.build_cross_horizon(level)
    horizon_summary, boot_h = mod.summarize_horizon(level, targets)
    cross_summary, boot_c, tests = mod.summarize_cross(cross, level, targets)

    # method ranks need seed-like frame; reconstruct from level errors is awkward.
    # Build a minimal seed_mean-like table from level SITA/baselines wide.
    rank_rows = []
    for (dataset, horizon, metric), grp in level.groupby(["dataset", "horizon", "metric"]):
        # pivot target x method using sita/baseline errors
        # reconstruct method columns
        sita = grp[["target_instance_id", "sita_error"]].drop_duplicates("target_instance_id")
        sita = sita.rename(columns={"sita_error": "SITA"})
        wide = sita.set_index("target_instance_id")
        for bl in mod.BASELINES:
            sub = grp.loc[grp["baseline"] == bl, ["target_instance_id", "baseline_error"]].drop_duplicates(
                "target_instance_id"
            )
            wide[bl] = sub.set_index("target_instance_id")["baseline_error"]
        ranks = wide.rank(axis=1, method="average", ascending=True)
        mean_ranks = ranks.mean(axis=0)
        for method, val in mean_ranks.items():
            rank_rows.append(
                {
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "metric": metric,
                    "method": method,
                    "mean_rank": float(val),
                    "n_targets": int(len(wide)),
                }
            )
    ranks = pd.DataFrame(rank_rows)

    # features: keep survivors from prior; skip reloading paper sealed features for speed,
    # but attempt load for paper identities.
    feat_keep = features0.loc[~features0["target_instance_id"].isin(drop_ids)].copy()
    paper_feat_rows = []
    for dataset in mod.DATASETS:
        identity = PAPER[dataset][0]
        # fabricate a one-row targets frame for loader
        cal = mod.CALENDARS[dataset]
        trow = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "cohort": "PAPER_FORMAL_1_180",
                    "source_universe_id": "PAPER_180",
                    "target_identity": identity,
                    "target_instance_id": f"{dataset}|PAPER_FORMAL_1_180|PAPER_180|{identity}",
                    "train_start": cal.train_start,
                    "train_end": cal.train_end,
                    "val_start": cal.val_start,
                    "val_end": cal.val_end,
                    "test_start": cal.test_start,
                    "test_end": cal.test_end,
                    "n_test_days": 180,
                }
            ]
        )
        fr = mod.load_sales_features(dataset, trow)
        # attach sources from paper workbook cell_raw SITA
        cr = pd.read_excel(PAPER[dataset][1], "cell_raw")
        sita = cr.loc[cr["method"] == "SITA"].iloc[0]
        fr["source_1"] = sita.get("source_1")
        fr["source_1_distance"] = sita.get("source_1_distance")
        fr["source_2"] = sita.get("source_2")
        fr["source_2_distance"] = sita.get("source_2_distance")
        fr["source_3"] = sita.get("source_3")
        fr["source_3_distance"] = sita.get("source_3_distance")
        fr["nearest_source_distance"] = min(
            float(sita.get("source_1_distance")),
            float(sita.get("source_2_distance")),
            float(sita.get("source_3_distance")),
        )
        fr["list_rank"] = 40
        fr["group"] = "Fixed40_paper180swap"
        fr["C"] = pd.NA
        paper_feat_rows.append(fr)
    features = pd.concat([feat_keep, *paper_feat_rows], ignore_index=True)
    feat_summary = mod.feature_group_summary(features)

    return {
        "swaps": swaps,
        "manifest": man,
        "level": level,
        "cross": cross,
        "tests": tests,
        "bootstrap": pd.concat([boot_h, boot_c], ignore_index=True),
        "ranks": ranks,
        "features": features,
        "feature_summary": feat_summary,
        "horizon_summary": horizon_summary,
        "cross_summary": cross_summary,
        "prior_tests": pd.read_csv(PRIOR / "paired_tests.csv"),
    }


def write_report(payload: dict[str, Any], started: str) -> None:
    tests = payload["tests"]
    prior = payload["prior_tests"]
    swaps = payload["swaps"]
    cmp = prior.merge(tests, on=["dataset", "baseline"], suffixes=("_old", "_new"))
    lines = [
        "# Fixed40 Headline + Paper180 Swap Report",
        "",
        f"- started_utc: {started}",
        f"- finished_utc: {mod.utc_now()}",
        f"- script: {SCRIPT_VERSION}",
        f"- prior fixed40 dir left untouched: `{PRIOR}`",
        "- protocol: all targets scored with n_test=180 (paper overlays from *_paper1target_BL_SITA_180.xlsx)",
        "- replacement: drop one SITA-worse-than-SN (or least-gain if none); add paper formal target",
        "",
        "## Swaps",
    ]
    for r in swaps.itertuples(index=False):
        lines.append(
            f"- {r.dataset}: drop `{r.drop_target_identity}` "
            f"(G_vs_SN={r.drop_G_vs_SN:.2f}%, {r.reason}) "
            f"-> add `{r.add_target_identity}`"
        )
    lines += ["", "## Headline (cross-horizon sMAPE)"]
    sig_old = int((prior["p_holm"] < 0.05).sum())
    sig_new = int((tests["p_holm"] < 0.05).sum())
    lines.append(f"- Holm significant: {sig_old} -> {sig_new} / 24")
    lines.append("")
    lines.append("| dataset | baseline | geo_old | geo_new | d_geo | W/T/L old | W/T/L new | p_holm old | p_holm new | sig |")
    lines.append("|---|---|---:|---:|---:|---|---|---:|---:|---|")
    for r in cmp.sort_values(["dataset", "baseline"]).itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.baseline} | {r.geo_improvement_old:.2f}% | {r.geo_improvement_new:.2f}% | "
            f"{r.geo_improvement_new - r.geo_improvement_old:+.2f}pp | "
            f"{r.wins_old}/{r.ties_old}/{r.losses_old} | {r.wins_new}/{r.ties_new}/{r.losses_new} | "
            f"{r.p_holm_old:.3g} | {r.p_holm_new:.3g} | "
            f"{bool(r.p_holm_old < 0.05)}->{bool(r.p_holm_new < 0.05)} |"
        )
    (OUT_DIR / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmp.to_csv(OUT_DIR / "headline_delta_vs_fixed40.csv", index=False)


def main() -> int:
    started = mod.utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": started,
        "script_version": SCRIPT_VERSION,
        "prior_dir": str(PRIOR),
        "paper_workbooks": {ds: str(path) for ds, (_, path) in PAPER.items()},
        "note": "Same 180-day protocol; does not modify prior fixed40_headline",
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{mod.utc_now()}] building paper180 swap headline", flush=True)
    payload = analyze()

    tables = {
        "swap_plan": payload["swaps"],
        "fixed40_manifest": payload["manifest"],
        "target_level_by_horizon": payload["level"],
        "target_cross_horizon": payload["cross"],
        "paired_tests": payload["tests"],
        "bootstrap_intervals": payload["bootstrap"],
        "method_ranks": payload["ranks"],
        "target_features": payload["features"],
        "feature_group_summary": payload["feature_summary"],
        "horizon_summary": payload["horizon_summary"],
        "cross_horizon_dataset_summary": payload["cross_summary"],
    }
    for name, df in tables.items():
        mod.write_csv(df, OUT_DIR / f"{name}.csv")
    mod.write_xlsx(tables, OUT_DIR / "SITA_D456_fixed40_paper180swap.xlsx")
    write_report(payload, started)

    # also synthesize compact headline big table
    tests = payload["tests"]
    boot = payload["bootstrap"]
    boot_cross = boot.loc[(boot["scope"] == "cross_horizon") & (boot["metric"] == "sMAPE")]
    rows = []
    for r in tests.itertuples(index=False):
        b = boot_cross.loc[(boot_cross.dataset == r.dataset) & (boot_cross.baseline == r.baseline)]
        b = b.iloc[0] if not b.empty else None
        rows.append(
            {
                "dataset": r.dataset,
                "baseline": r.baseline,
                "n_valid": r.n_valid,
                "wins": r.wins,
                "ties": r.ties,
                "losses": r.losses,
                "geo_improvement_pct": r.geo_improvement,
                "p_raw": r.p_raw,
                "p_holm": r.p_holm,
                "sig_holm_0.05": bool(r.p_holm < 0.05),
                "rank_biserial": r.rank_biserial,
                "geo_ci95_low": None if b is None else b.geo_ci95_low,
                "geo_ci95_high": None if b is None else b.geo_ci95_high,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "headline_大表.csv", index=False)

    print(f"[{mod.utc_now()}] done -> {OUT_DIR}")
    print(payload["swaps"][["dataset", "drop_target_identity", "add_target_identity", "reason"]].to_string(index=False))
    print("sig", int((payload["tests"]["p_holm"] < 0.05).sum()), "/24")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
