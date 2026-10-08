#!/usr/bin/env python3
"""Aggregate the completed SITA-TCN formal 60-cell matrix vs BL7 harness rerun.

Writes only under results/sita_tcn/. Does not modify freeze workbooks.
Statistical units are 12 condition means, not 60 seed-runs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sita_tcn_protocol import (
    BL7_METHOD_NAME,
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_RESAMPLES,
    DATASETS,
    FORMAL_OUTPUT_ROOT,
    HORIZONS,
    METHOD_NAME,
    MODE,
    SEEDS,
    TARGETS,
    TIE_RULE,
    assert_isolated_output_dir,
    bootstrap_mean_ci,
    json_dump,
    now_utc,
    relative_improvement_pct,
    wilcoxon_paired,
    win_tie_loss,
)

SITA_FREEZE_XLSX = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)
SITA_SHEET = "03_V3_Raw_5Seed"
SITA_CNN_12CONDITION_FALLBACK = (
    ROOT / "results" / "formal_bl8_msml_tl" / "BL8_SITA_VS_MSML_TL_12CONDITION_COMPARISON.csv"
)
EXPECTED_TARGET_CELLS = 60
EXPECTED_TOTAL_CELLS = 120
EXPECTED_CONDITIONS = 12


def _load_cells(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "cells").glob("*/cell_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": str(payload["dataset"]),
                "target": str(payload.get("target_id") or payload.get("target")),
                "horizon": int(payload["horizon"]),
                "seed": int(payload.get("target_seed") or payload.get("seed")),
                "mode": str(payload.get("mode", MODE)),
                "method": str(payload["model_name"]),
                "init_mode": str(payload["init_mode"]),
                "rmse": float(payload["rmse"]),
                "smape": float(payload["smape"]),
                "n": int(payload.get("n", 0)),
                "metric_space": str(payload.get("metric_space", "")),
                "result_status": str(payload.get("result_status", "")),
                "cell_dir": str(path.parent),
                "source_1_id": payload.get("source_1_id", ""),
                "source_2_id": payload.get("source_2_id", ""),
                "source_3_id": payload.get("source_3_id", ""),
                "weight_1": payload.get("weight_1"),
                "weight_2": payload.get("weight_2"),
                "weight_3": payload.get("weight_3"),
                "weight_sum": payload.get("weight_sum"),
                "epoch0_train_loss": payload.get("epoch0_train_loss"),
                "epoch0_val_loss": payload.get("epoch0_val_loss"),
                "best_val_epoch": payload.get("best_val_epoch"),
                "best_val_loss": payload.get("best_val_loss"),
                "target_head_init_hash": payload.get("target_head_init_hash", ""),
                "shared_source_init_hash": payload.get("shared_source_init_hash", ""),
                "fused_backbone_hash": payload.get("fused_backbone_hash", ""),
                "ORIGINAL_SALES_SPACE": payload.get("ORIGINAL_SALES_SPACE", ""),
                "TARGET_ONLY_INFERENCE": payload.get("TARGET_ONLY_INFERENCE", ""),
                "PAIRED_HEAD_INITIALIZATION": payload.get("PAIRED_HEAD_INITIALIZATION", ""),
                "code_commit": payload.get("code_commit", ""),
            }
        )
    if not rows:
        raise RuntimeError(f"no cell manifests under {output_root / 'cells'}")
    return pd.DataFrame(rows)


def _assert_completeness(frame: pd.DataFrame) -> dict[str, Any]:
    expected_random = {
        (dataset, TARGETS[dataset], horizon, seed, "random")
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    }
    expected_sita = {
        (dataset, TARGETS[dataset], horizon, seed, "sita")
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    }
    actual = set(
        zip(
            frame["dataset"].astype(str),
            frame["target"].astype(str),
            frame["horizon"].astype(int),
            frame["seed"].astype(int),
            frame["init_mode"].astype(str),
        )
    )
    missing = sorted((expected_random | expected_sita) - actual)
    extra = sorted(actual - (expected_random | expected_sita))
    if missing or extra or len(frame) != EXPECTED_TOTAL_CELLS:
        raise RuntimeError(
            f"formal completeness failed: n={len(frame)} missing={missing[:8]} extra={extra[:8]}"
        )
    if (frame["metric_space"].astype(str) != "original_sales_space").any():
        raise RuntimeError("formal cells left original_sales_space")
    methods = set(frame["method"].astype(str))
    if methods != {BL7_METHOD_NAME, METHOD_NAME}:
        raise RuntimeError(f"unexpected methods: {methods}")
    return {
        "expected_cells": EXPECTED_TOTAL_CELLS,
        "actual_cells": int(len(frame)),
        "missing": 0,
        "extra": 0,
        "ORIGINAL_SALES_SPACE": "YES",
    }


def _aggregate(frame: pd.DataFrame, init_mode: str, method: str) -> pd.DataFrame:
    subset = frame[frame["init_mode"].astype(str) == init_mode].copy()
    records = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            group = subset[
                (subset["dataset"] == dataset)
                & (subset["target"] == TARGETS[dataset])
                & (subset["horizon"].astype(int) == int(horizon))
            ]
            seeds = sorted(int(value) for value in group["seed"].tolist())
            if seeds != list(SEEDS):
                raise RuntimeError(f"{dataset}/H{horizon}/{init_mode} seeds={seeds}")
            rmse = [float(value) for value in group["rmse"].tolist()]
            smape = [float(value) for value in group["smape"].tolist()]
            records.append(
                {
                    "method": method,
                    "init_mode": init_mode,
                    "dataset": dataset,
                    "target": TARGETS[dataset],
                    "horizon": int(horizon),
                    "mode": MODE,
                    "n_seeds": 5,
                    "rmse_mean": float(statistics.mean(rmse)),
                    "rmse_std": float(statistics.stdev(rmse)),
                    "smape_mean": float(statistics.mean(smape)),
                    "smape_std": float(statistics.stdev(smape)),
                    "n_mean": float(statistics.mean(float(value) for value in group["n"].tolist())),
                }
            )
    return pd.DataFrame.from_records(records)


def _wtl_counts(labels: list[str]) -> dict[str, int]:
    return {
        "W": int(sum(label == "W" for label in labels)),
        "T": int(sum(label == "T" for label in labels)),
        "L": int(sum(label == "L" for label in labels)),
    }


def _try_sita_cnn_reference() -> pd.DataFrame | None:
    if SITA_FREEZE_XLSX.exists():
        try:
            frame = pd.read_excel(SITA_FREEZE_XLSX, sheet_name=SITA_SHEET)
            frame.columns = [str(column).strip() for column in frame.columns]
            if "method" in frame.columns:
                frame = frame[frame["method"].astype(str).isin(["SITA", "MS-KATL-V3", "FUSED_TARGET_ONLY_V3"])]
            target_col = "target" if "target" in frame.columns else "target_entity_key"
            rmse_col = "rmse_original" if "rmse_original" in frame.columns else "rmse"
            smape_col = "smape_original" if "smape_original" in frame.columns else "smape"
            keep = pd.DataFrame(
                {
                    "dataset": frame["dataset"].astype(str),
                    "target": frame[target_col].astype(str),
                    "horizon": frame["horizon"].astype(int),
                    "seed": frame["seed"].astype(int),
                    "rmse": frame[rmse_col].astype(float),
                    "smape": frame[smape_col].astype(float),
                }
            )
            if "mode" in frame.columns:
                keep = keep.assign(mode=frame["mode"].astype(str))
                keep = keep[keep["mode"].isin(["without", "without_information_sharing", MODE])]
            records = []
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    group = keep[
                        (keep["dataset"] == dataset)
                        & (keep["target"] == TARGETS[dataset])
                        & (keep["horizon"].astype(int) == int(horizon))
                    ]
                    if len(group) != 5:
                        records = []
                        break
                    records.append(
                        {
                            "dataset": dataset,
                            "target": TARGETS[dataset],
                            "horizon": int(horizon),
                            "sita_cnn_rmse_mean": float(group["rmse"].mean()),
                            "sita_cnn_rmse_std": float(group["rmse"].std(ddof=1)),
                            "sita_cnn_smape_mean": float(group["smape"].mean()),
                            "sita_cnn_smape_std": float(group["smape"].std(ddof=1)),
                            "sita_cnn_source": str(SITA_FREEZE_XLSX),
                        }
                    )
                if not records:
                    break
            if records and len(records) == EXPECTED_CONDITIONS:
                return pd.DataFrame.from_records(records)
        except Exception:
            pass
    if not SITA_CNN_12CONDITION_FALLBACK.exists():
        return None
    frame = pd.read_csv(SITA_CNN_12CONDITION_FALLBACK)
    keep = pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str),
            "target": frame["target"].astype(str),
            "horizon": frame["horizon"].astype(int),
            "sita_cnn_rmse_mean": frame["sita_rmse_mean"].astype(float),
            "sita_cnn_rmse_std": frame["sita_rmse_std"].astype(float),
            "sita_cnn_smape_mean": frame["sita_smape_mean"].astype(float),
            "sita_cnn_smape_std": frame["sita_smape_std"].astype(float),
            "sita_cnn_source": str(SITA_CNN_12CONDITION_FALLBACK),
        }
    )
    expected = {(dataset, TARGETS[dataset], horizon) for dataset in DATASETS for horizon in HORIZONS}
    actual = set(zip(keep["dataset"], keep["target"], keep["horizon"]))
    if actual != expected:
        return None
    return keep


def finalize(output_root: Path = FORMAL_OUTPUT_ROOT) -> dict[str, Any]:
    destination = assert_isolated_output_dir(output_root)
    cells = _load_cells(destination)
    completeness = _assert_completeness(cells)
    bl7 = _aggregate(cells, "random", BL7_METHOD_NAME)
    sita_tcn = _aggregate(cells, "sita", METHOD_NAME)
    condition_rows = []
    rmse_ri = []
    smape_ri = []
    rmse_wtl = []
    smape_wtl = []
    rmse_diff = []
    smape_diff = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            left = bl7[(bl7["dataset"] == dataset) & (bl7["horizon"] == horizon)].iloc[0]
            right = sita_tcn[(sita_tcn["dataset"] == dataset) & (sita_tcn["horizon"] == horizon)].iloc[0]
            ri_rmse = relative_improvement_pct(float(left["rmse_mean"]), float(right["rmse_mean"]))
            ri_smape = relative_improvement_pct(float(left["smape_mean"]), float(right["smape_mean"]))
            wtl_rmse = win_tie_loss(float(left["rmse_mean"]), float(right["rmse_mean"]))
            wtl_smape = win_tie_loss(float(left["smape_mean"]), float(right["smape_mean"]))
            rmse_ri.append(ri_rmse)
            smape_ri.append(ri_smape)
            rmse_wtl.append(wtl_rmse)
            smape_wtl.append(wtl_smape)
            rmse_diff.append(float(left["rmse_mean"]) - float(right["rmse_mean"]))
            smape_diff.append(float(left["smape_mean"]) - float(right["smape_mean"]))
            condition_rows.append(
                {
                    "dataset": dataset,
                    "target": TARGETS[dataset],
                    "horizon": int(horizon),
                    "mode": MODE,
                    "bl7_rmse_mean": float(left["rmse_mean"]),
                    "bl7_rmse_std": float(left["rmse_std"]),
                    "sita_tcn_rmse_mean": float(right["rmse_mean"]),
                    "sita_tcn_rmse_std": float(right["rmse_std"]),
                    "rmse_relative_improvement_pct": ri_rmse,
                    "rmse_wtl": wtl_rmse,
                    "bl7_smape_mean": float(left["smape_mean"]),
                    "bl7_smape_std": float(left["smape_std"]),
                    "sita_tcn_smape_mean": float(right["smape_mean"]),
                    "sita_tcn_smape_std": float(right["smape_std"]),
                    "smape_relative_improvement_pct": ri_smape,
                    "smape_wtl": wtl_smape,
                    "n_seeds": 5,
                    "statistical_unit": "condition_mean",
                    "tie_rule": TIE_RULE,
                }
            )
    conditions = pd.DataFrame.from_records(condition_rows)
    reference = _try_sita_cnn_reference()
    if reference is not None:
        conditions = conditions.merge(reference, on=["dataset", "target", "horizon"], how="left")
        conditions["REFERENCE_ONLY_DIFFERENT_BACKBONE"] = "YES"
    else:
        conditions["REFERENCE_ONLY_DIFFERENT_BACKBONE"] = "YES_SITA_CNN_AUTHORITY_MISSING"
    rmse_ci = bootstrap_mean_ci(rmse_ri)
    smape_ci = bootstrap_mean_ci(smape_ri)
    rmse_w = _wtl_counts(rmse_wtl)
    smape_w = _wtl_counts(smape_wtl)
    rmse_wx = wilcoxon_paired(rmse_diff)
    smape_wx = wilcoxon_paired(smape_diff)
    summary = {
        "comparison": "BL7 TCN+RevIN vs SITA-TCN+RevIN",
        "unique_experimental_variable": "TCN backbone initialization strategy",
        "n_conditions": EXPECTED_CONDITIONS,
        "n_seed_runs_per_arm": EXPECTED_TARGET_CELLS,
        "bootstrap_unit": "12 condition means",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_random_seed": BOOTSTRAP_RANDOM_SEED,
        "tie_rule": TIE_RULE,
        "seed_level_wilcoxon": "FORBIDDEN",
        "sita_cnn_role": "REFERENCE_ONLY_DIFFERENT_BACKBONE=YES",
        "rmse_mean_relative_improvement_pct": float(np.mean(rmse_ri)),
        "rmse_median_relative_improvement_pct": float(np.median(rmse_ri)),
        "rmse_bootstrap_ci_2.5": rmse_ci[0],
        "rmse_bootstrap_ci_97.5": rmse_ci[1],
        "rmse_W": rmse_w["W"],
        "rmse_T": rmse_w["T"],
        "rmse_L": rmse_w["L"],
        "rmse_wilcoxon": rmse_wx,
        "smape_mean_relative_improvement_pct": float(np.mean(smape_ri)),
        "smape_median_relative_improvement_pct": float(np.median(smape_ri)),
        "smape_bootstrap_ci_2.5": smape_ci[0],
        "smape_bootstrap_ci_97.5": smape_ci[1],
        "smape_W": smape_w["W"],
        "smape_T": smape_w["T"],
        "smape_L": smape_w["L"],
        "smape_wilcoxon": smape_wx,
        "completeness": completeness,
        "created_at": now_utc(),
    }
    cells.to_csv(destination / "SITA_TCN_FORMAL_120CELL_RAW.csv", index=False)
    pd.concat([bl7, sita_tcn], ignore_index=True).to_csv(
        destination / "SITA_TCN_FORMAL_5SEED_AGGREGATED.csv", index=False
    )
    conditions.to_csv(destination / "SITA_TCN_VS_BL7_CONDITION_STATS.csv", index=False)
    json_dump(destination / "SITA_TCN_VS_BL7_STATISTICAL_SUMMARY.json", summary)
    _write_markdown(destination / "SITA_TCN_VS_BL7_STATISTICAL_SUMMARY.md", conditions, summary)
    audit_copy = destination.parent / "audits" / "SITA_TCN_VS_BL7_STATISTICAL_SUMMARY.md"
    audit_copy.parent.mkdir(parents=True, exist_ok=True)
    audit_copy.write_text(
        (destination / "SITA_TCN_VS_BL7_STATISTICAL_SUMMARY.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k not in {"rmse_wilcoxon", "smape_wilcoxon"}}, indent=2, default=str))
    return summary


def _write_markdown(path: Path, conditions: pd.DataFrame, summary: dict[str, Any]) -> None:
    wx_rmse = summary["rmse_wilcoxon"]
    wx_smape = summary["smape_wilcoxon"]
    lines = [
        "# SITA-TCN vs BL7 TCN+RevIN Statistical Summary",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "Comparison is **BL7 TCN+RevIN (random init)** vs **SITA-TCN+RevIN (source-informed init)**.",
        "The only experimental variable is TCN backbone initialization.",
        "SITA-CNN is reference-only and is not used in the significance test.",
        "",
        "```text",
        f"N_CONDITIONS={summary['n_conditions']}",
        f"BOOTSTRAP_UNIT={summary['bootstrap_unit']}",
        f"TIE_RULE={summary['tie_rule']}",
        f"SEED_LEVEL_WILCOXON={summary['seed_level_wilcoxon']}",
        f"RMSE_MEAN_RI%={summary['rmse_mean_relative_improvement_pct']:.6f}",
        f"RMSE_MEDIAN_RI%={summary['rmse_median_relative_improvement_pct']:.6f}",
        f"RMSE_BOOTSTRAP_CI=[{summary['rmse_bootstrap_ci_2.5']:.6f}, {summary['rmse_bootstrap_ci_97.5']:.6f}]",
        f"RMSE_W/T/L={summary['rmse_W']}/{summary['rmse_T']}/{summary['rmse_L']}",
        f"RMSE_WILCOXON_P={wx_rmse.get('p_value')}",
        f"RMSE_WILCOXON_STATUS={wx_rmse.get('status')}",
        f"RMSE_RANK_BISERIAL={wx_rmse.get('rank_biserial')}",
        f"RMSE_COHENS_D={wx_rmse.get('cohens_d_paired')}",
        f"sMAPE_MEAN_RI%={summary['smape_mean_relative_improvement_pct']:.6f}",
        f"sMAPE_MEDIAN_RI%={summary['smape_median_relative_improvement_pct']:.6f}",
        f"sMAPE_BOOTSTRAP_CI=[{summary['smape_bootstrap_ci_2.5']:.6f}, {summary['smape_bootstrap_ci_97.5']:.6f}]",
        f"sMAPE_W/T/L={summary['smape_W']}/{summary['smape_T']}/{summary['smape_L']}",
        f"sMAPE_WILCOXON_P={wx_smape.get('p_value')}",
        f"sMAPE_WILCOXON_STATUS={wx_smape.get('status')}",
        f"sMAPE_RANK_BISERIAL={wx_smape.get('rank_biserial')}",
        f"sMAPE_COHENS_D={wx_smape.get('cohens_d_paired')}",
        "```",
        "",
        "W means SITA-TCN has strictly lower error than BL7 on that condition mean.",
        "Positive relative improvement means SITA-TCN reduced error versus BL7.",
        "Wilcoxon is paired on the 12 condition-mean differences (BL7 − SITA-TCN).",
        "",
        "## Condition table",
        "",
        "| Dataset | H | BL7 RMSE | SITA-TCN RMSE | RMSE RI% | RMSE | BL7 sMAPE | SITA-TCN sMAPE | sMAPE RI% | sMAPE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in conditions.iterrows():
        lines.append(
            "| {dataset} | {horizon} | {bl7_rmse_mean:.6f}±{bl7_rmse_std:.6f} | {sita_tcn_rmse_mean:.6f}±{sita_tcn_rmse_std:.6f} | {rmse_relative_improvement_pct:.3f} | {rmse_wtl} | {bl7_smape_mean:.4f}±{bl7_smape_std:.4f} | {sita_tcn_smape_mean:.4f}±{sita_tcn_smape_std:.4f} | {smape_relative_improvement_pct:.3f} | {smape_wtl} |".format(
                **row
            )
        )
    if "sita_cnn_rmse_mean" in conditions.columns:
        lines.extend(
            [
                "",
                "## SITA-CNN reference only",
                "",
                "`REFERENCE_ONLY_DIFFERENT_BACKBONE=YES`. Not used for Wilcoxon or W/T/L.",
                "",
            ]
        )
        for _, row in conditions.iterrows():
            lines.append(
                f"- {row['dataset']} H{int(row['horizon'])}: SITA-CNN RMSE={row['sita_cnn_rmse_mean']:.6f}, sMAPE={row['sita_cnn_smape_mean']:.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    finalize()
