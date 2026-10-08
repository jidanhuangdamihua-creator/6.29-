#!/usr/bin/env python3
"""Headline stats on a fixed Top-40 target list (does not modify prior C-selected runs).

Selection is by target_identity from the user list. D4 identities that appear in both
OLD_ADDITIONAL_10 and NEW_RANDOM_60 resolve to NEW_RANDOM_60 when both exist.
Analysis uses whatever baselines are present in the workbook for each target; user-marked
partial-BL identities are flagged even if the workbook later gained full coverage.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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

OUT_DIR = ROOT / "实验结果" / "sita_d456_fixed40_headline"
SCRIPT_VERSION = "analyze_sita_d456_fixed40_headline.py@2026-10-05"

FIXED40: dict[str, list[str]] = {
    "D4": [
        "139/104", "139/209", "139/277", "139/295", "139/334", "139/340", "139/453",
        "155/187", "155/209", "155/275", "155/277", "155/334", "161/192", "166/133",
        "166/244", "166/266", "166/510", "166/530", "576/104", "576/160", "576/192",
        "155/167", "161/506", "166/242", "166/246", "166/307", "139/245", "166/229",
        "576/310", "155/508", "274/184", "576/209", "161/184", "161/295", "139/184",
        "576/433", "155/506", "274/128", "274/190", "274/506",
    ],
    "D5": [
        "48/1037845", "48/1091365", "48/162066", "48/164036", "48/1975558", "48/315277",
        "48/364834", "48/575863", "48/964753", "48/1975512", "48/2010885", "48/2026501",
        "48/2048300", "48/265237", "48/1975514", "48/502067", "48/1012787", "48/1969929",
        "48/372924", "48/848765", "48/913968", "48/1354384", "48/179274", "48/2010329",
        "48/326951", "48/559870", "48/1146795", "48/115611", "48/1176562", "48/1937662",
        "48/261700", "48/355090", "48/414750", "48/559873", "48/1084881", "48/111223",
        "48/1354711", "48/1109390", "48/1354380", "48/2048271",
    ],
    "D6": [
        "CA_1/FOODS_3_074", "CA_1/FOODS_3_099", "CA_1/FOODS_3_135", "CA_1/FOODS_3_139",
        "CA_1/FOODS_3_181", "CA_1/FOODS_3_306", "CA_1/FOODS_3_406", "CA_1/FOODS_3_434",
        "CA_1/FOODS_3_460", "CA_1/FOODS_3_497", "CA_1/FOODS_3_562", "CA_1/FOODS_3_570",
        "CA_1/FOODS_3_596", "CA_1/FOODS_3_640", "CA_1/FOODS_3_704", "CA_1/FOODS_3_716",
        "CA_1/FOODS_3_820", "CA_1/FOODS_3_106", "CA_1/FOODS_3_021", "CA_1/FOODS_3_167",
        "CA_1/FOODS_3_239", "CA_1/FOODS_3_380", "CA_1/FOODS_3_448", "CA_1/FOODS_3_769",
        "CA_1/FOODS_3_784", "CA_1/FOODS_3_195", "CA_1/FOODS_3_255", "CA_1/FOODS_3_119",
        "CA_1/FOODS_3_323", "CA_1/FOODS_3_647", "CA_1/FOODS_3_773", "CA_1/FOODS_3_214",
        "CA_1/FOODS_3_283", "CA_1/FOODS_3_407", "CA_1/FOODS_3_454", "CA_1/FOODS_3_633",
        "CA_1/FOODS_3_057", "CA_1/FOODS_3_137", "CA_1/FOODS_3_198", "CA_1/FOODS_3_286",
    ],
}

USER_PARTIAL_BL = {
    "D5": {"48/1091365", "48/162066", "48/575863", "48/1354380"},
    "D6": {
        "CA_1/FOODS_3_181",
        "CA_1/FOODS_3_784",
        "CA_1/FOODS_3_773",
        "CA_1/FOODS_3_407",
        "CA_1/FOODS_3_057",
        "CA_1/FOODS_3_198",
    },
}
PARTIAL_OK_METHODS = {"SITA", "Seasonal Naive", "PatchTST", "TimesNet"}
FULL_METHODS = set(mod.METHODS)


def resolve_instances(dataset: str, targets: pd.DataFrame, identities: list[str]) -> pd.DataFrame:
    rows = []
    for rank, identity in enumerate(identities, start=1):
        hits = targets.loc[targets["target_identity"].astype(str) == identity].copy()
        if hits.empty:
            raise RuntimeError(f"{dataset}: identity {identity} not in targets")
        if len(hits) == 1:
            chosen = hits.iloc[0]
            rule = "unique"
        else:
            new = hits.loc[hits["cohort"] == "NEW_RANDOM_60"]
            if not new.empty:
                chosen = new.iloc[0]
                rule = "prefer_NEW_RANDOM_60"
            else:
                chosen = hits.sort_values("cohort").iloc[0]
                rule = f"fallback_{chosen.cohort}"
        rows.append(
            {
                "dataset": dataset,
                "list_rank": rank,
                "target_identity": identity,
                "cohort": chosen.cohort,
                "source_universe_id": chosen.source_universe_id,
                "target_instance_id": chosen.target_instance_id,
                "resolve_rule": rule,
                "n_candidate_instances": int(len(hits)),
                "user_marked_partial_bl": identity in USER_PARTIAL_BL.get(dataset, set()),
            }
        )
    out = pd.DataFrame(rows)
    if out["target_instance_id"].nunique() != 40:
        raise RuntimeError(f"{dataset}: resolved to {out['target_instance_id'].nunique()} instances")
    return out


def coverage_flags(seed: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in resolved.itertuples(index=False):
        sub = seed.loc[seed["target_instance_id"] == r.target_instance_id]
        methods = set(sub["method"].astype(str))
        horizons = set(int(h) for h in sub["horizon"])
        rows.append(
            {
                "target_instance_id": r.target_instance_id,
                "target_identity": r.target_identity,
                "n_methods_present": len(methods),
                "methods_present": ",".join(sorted(methods)),
                "workbook_full_bl": methods == FULL_METHODS and horizons == {1, 3, 5},
                "missing_methods": ",".join(sorted(FULL_METHODS - methods)),
                "user_marked_partial_bl": r.user_marked_partial_bl,
            }
        )
    return pd.DataFrame(rows)


def analyze_one(dataset: str, out_dir: Path) -> dict[str, Any]:
    sheets = mod.load_sheets(dataset)
    targets, cell, seed, coverage = mod.prepare_frames(dataset, sheets)
    identities = FIXED40[dataset]
    if len(identities) != 40 or len(set(identities)) != 40:
        raise RuntimeError(f"{dataset}: fixed list must contain 40 unique identities")

    resolved = resolve_instances(dataset, targets, identities)
    cov = coverage_flags(seed, resolved)
    resolved = resolved.merge(cov, on=["target_instance_id", "target_identity", "user_marked_partial_bl"], how="left")

    top_ids = set(resolved["target_instance_id"])
    targets40 = targets.loc[targets["target_instance_id"].isin(top_ids)].copy()
    cell40 = cell.loc[cell["target_instance_id"].isin(top_ids)].copy()
    seed40 = seed.loc[seed["target_instance_id"].isin(top_ids)].copy()

    # Keep only method rows that exist; headline uses available pairs.
    level = mod.build_target_level(dataset, seed40)
    # Drop baseline rows where baseline error is missing already handled by pivot;
    # for safety, require finite baseline and SITA errors.
    level = level.loc[level["sita_error"].map(mod._finite) & level["baseline_error"].map(mod._finite)].copy()
    cross = mod.build_cross_horizon(level)
    horizon_summary, boot_h = mod.summarize_horizon(level, targets40)
    cross_summary, boot_c, tests = mod.summarize_cross(cross, level, targets40)
    ranks = mod.method_ranks(seed40)
    features = mod.load_sales_features(dataset, targets40)
    # attach without C-selection scores: synthesize minimal score frame
    score_stub = resolved[["dataset", "cohort", "source_universe_id", "target_identity", "target_instance_id"]].copy()
    score_stub["C"] = pd.NA
    score_stub["rank_in_dataset"] = resolved["list_rank"]
    score_stub["group"] = "Fixed40"
    features = mod.attach_sources_and_cv(features, cell40, seed40, score_stub)
    features = features.merge(
        resolved[
            [
                "target_instance_id",
                "list_rank",
                "user_marked_partial_bl",
                "workbook_full_bl",
                "missing_methods",
                "resolve_rule",
            ]
        ],
        on="target_instance_id",
        how="left",
    )
    feat_summary = mod.feature_group_summary(features)

    mod.write_csv(resolved, out_dir / f"fixed40_manifest_{dataset}.csv")
    mod.write_csv(level, out_dir / f"target_level_by_horizon_{dataset}.csv")
    mod.write_csv(cross, out_dir / f"target_cross_horizon_summary_{dataset}.csv")
    mod.write_csv(horizon_summary, out_dir / f"horizon_summary_{dataset}.csv")
    mod.write_csv(cross_summary, out_dir / f"cross_horizon_dataset_summary_{dataset}.csv")
    mod.write_csv(tests, out_dir / f"paired_tests_{dataset}.csv")
    mod.write_csv(pd.concat([boot_h, boot_c], ignore_index=True), out_dir / f"bootstrap_intervals_{dataset}.csv")
    mod.write_csv(ranks, out_dir / f"method_ranks_{dataset}.csv")
    mod.write_csv(features, out_dir / f"target_features_{dataset}.csv")
    mod.write_csv(feat_summary, out_dir / f"feature_group_summary_{dataset}.csv")

    return {
        "dataset": dataset,
        "resolved": resolved,
        "level": level,
        "cross": cross,
        "tests": tests,
        "bootstrap": pd.concat([boot_h, boot_c], ignore_index=True),
        "ranks": ranks,
        "features": features,
        "feature_summary": feat_summary,
        "horizon_summary": horizon_summary,
        "cross_summary": cross_summary,
    }


def write_report(results: list[dict[str, Any]], out_dir: Path, started: str) -> None:
    lines = [
        "# SITA D4/D5/D6 Fixed-40 Headline Report",
        "",
        f"- started_utc: {started}",
        f"- finished_utc: {mod.utc_now()}",
        f"- script: {SCRIPT_VERSION}",
        "- prior C-selected run left untouched",
        "- analysis set: fixed user list, 40 target_identity per dataset",
        "- D4 duplicate identities: prefer NEW_RANDOM_60 instance",
        "- headline: cross-horizon sMAPE, 3 datasets × 8 baselines, Wilcoxon + Holm",
        "- bootstrap: 100000 cluster resamples by target_identity within the fixed 40",
        "",
    ]
    for res in results:
        ds = res["dataset"]
        man = res["resolved"]
        tests = res["tests"]
        lines.append(f"## {ds}")
        lines.append(f"- n_targets: {len(man)}")
        lines.append(
            f"- workbook_full_bl: {int(man.workbook_full_bl.sum())}/40; "
            f"user_marked_partial_bl: {int(man.user_marked_partial_bl.sum())}/40"
        )
        if man.user_marked_partial_bl.any():
            marked = man.loc[man.user_marked_partial_bl, ["target_identity", "workbook_full_bl", "missing_methods"]]
            lines.append("- user-marked partial-BL status in current workbook:")
            for r in marked.itertuples(index=False):
                lines.append(
                    f"  - `{r.target_identity}` workbook_full_bl={bool(r.workbook_full_bl)} "
                    f"missing=[{r.missing_methods}]"
                )
        dups = man.loc[man.n_candidate_instances > 1]
        if not dups.empty:
            lines.append(f"- D4 multi-instance resolutions: {len(dups)}")
            for r in dups.itertuples(index=False):
                lines.append(f"  - `{r.target_identity}` -> {r.cohort} ({r.resolve_rule})")
        sig = tests.loc[tests["p_holm"] < 0.05]
        lines.append(f"- Significant headline tests after Holm: {len(sig)} / {len(tests)}")
        lines.append("- Per-baseline geo% / W-T-L / n_valid / Holm p:")
        for r in tests.sort_values("baseline").itertuples(index=False):
            lines.append(
                f"  - {r.baseline}: geo={r.geo_improvement:.2f}% "
                f"W/T/L={r.wins}/{r.ties}/{r.losses} n_valid={r.n_valid} p_holm={r.p_holm:.3g}"
            )
        lines.append("")
    (out_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = mod.utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": started,
        "script_version": SCRIPT_VERSION,
        "note": "Does not modify sita_d456_top40_analysis_on40",
        "fixed_lists": {ds: FIXED40[ds] for ds in mod.DATASETS},
        "user_partial_bl": {k: sorted(v) for k, v in USER_PARTIAL_BL.items()},
        "d4_resolve_rule": "prefer_NEW_RANDOM_60_when_duplicate_identity",
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    results = []
    for ds in mod.DATASETS:
        print(f"[{mod.utc_now()}] fixed40 headline {ds}", flush=True)
        results.append(analyze_one(ds, OUT_DIR))

    def cat(key: str) -> pd.DataFrame:
        return pd.concat([r[key] for r in results], ignore_index=True)

    tables = {
        "fixed40_manifest": cat("resolved"),
        "target_level_by_horizon": cat("level"),
        "target_cross_horizon": cat("cross"),
        "paired_tests": cat("tests"),
        "bootstrap_intervals": cat("bootstrap"),
        "method_ranks": cat("ranks"),
        "target_features": cat("features"),
        "feature_group_summary": cat("feature_summary"),
    }
    for name, df in tables.items():
        mod.write_csv(df, OUT_DIR / f"{name}.csv")
    mod.write_xlsx(tables, OUT_DIR / "SITA_D456_fixed40_headline.xlsx")
    write_report(results, OUT_DIR, started)

    for r in results:
        assert len(r["resolved"]) == 40
        assert r["level"]["target_instance_id"].nunique() == 40
        assert set(r["horizon_summary"]["n_targets"].unique()) == {40}
    print(f"[{mod.utc_now()}] done -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
