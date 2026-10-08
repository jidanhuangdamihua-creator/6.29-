#!/usr/bin/env python3
"""Read-only comparison of SITA vs original and RevIN-strengthened DLinear/TCN.

Loads frozen SITA / DLinear / TCN results without modifying them, then compares
against robustness CSVs written under results/robustness/revin_strengthened_baselines/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baselines.revin_strengthened_protocol import (
    DEFAULT_FREEZE,
    DEFAULT_OUTPUT_DIR,
    PAPER_HORIZONS,
    PAPER_SEEDS,
    PAPER_TARGETS,
    aggregate_five_seed,
    relative_error_change,
    relative_improvement,
    sha256_file,
)
from tools.build_final_statistical_derivation_freeze import (
    FORMAL_SHEETS,
    _normalize_main_rows,
    holm_adjust,
    read_sheet_rows,
    wilcoxon_exact,
)

SITA_SHEET = FORMAL_SHEETS["main"]
BASELINE_SHEET = FORMAL_SHEETS["baseline"]
FROZEN_PARENTS = ("DLinear", "TCN")
STRENGTHENED = {"DLinear": "DLinear_RevIN", "TCN": "TCN_RevIN"}


def _load_frozen(path: Path) -> pd.DataFrame:
    main = _normalize_main_rows(read_sheet_rows(path, SITA_SHEET), SITA_SHEET)
    baseline = _normalize_main_rows(read_sheet_rows(path, BASELINE_SHEET), BASELINE_SHEET)
    frame = pd.DataFrame(main + baseline)
    keep = frame["method"].isin(("SITA",) + FROZEN_PARENTS)
    selected = frame.loc[keep].copy()
    selected["dataset"] = selected["dataset"].astype(str).str.upper()
    selected["target"] = selected["target"].astype(str)
    selected["mode"] = "without"
    return selected


def _load_strengthened(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "dataset", "target", "horizon", "seed", "rmse", "smape"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"strengthened result missing columns: {missing}")
    if "mode" not in frame.columns:
        frame["mode"] = frame.get("scenario", "without")
    return frame


def _condition_map(frame: pd.DataFrame, method: str) -> dict[tuple[str, str, int], dict[str, float]]:
    subset = frame.loc[frame["method"].astype(str) == method].copy()
    if "mode" not in subset.columns:
        subset["mode"] = "without"
    aggregated = aggregate_five_seed(subset) if "rmse_mean" not in subset.columns else subset
    output = {}
    for _, row in aggregated.iterrows():
        key = (str(row["dataset"]).upper(), str(row["target"]), int(row["horizon"]))
        output[key] = {
            "rmse": float(row["rmse_mean"] if "rmse_mean" in row else row["rmse"]),
            "smape": float(row["smape_mean"] if "smape_mean" in row else row["smape"]),
            "n_seeds": int(row.get("n_seeds", len(PAPER_SEEDS))),
        }
    return output


def _seed_map(frame: pd.DataFrame, method: str) -> dict[tuple[str, str, int, int], dict[str, float]]:
    subset = frame.loc[frame["method"].astype(str) == method]
    output = {}
    for _, row in subset.iterrows():
        key = (
            str(row.get("dataset", row.get("dataset_id"))).upper(),
            str(row.get("target", row.get("target_entity_key"))),
            int(row["horizon"]),
            int(row["seed"]),
        )
        output[key] = {"rmse": float(row["rmse"]), "smape": float(row["smape"])}
    return output


def build_comparison(
    frozen: pd.DataFrame,
    strengthened: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sita = _condition_map(frozen, "SITA")
    effect_rows = []
    versus_rows = []
    for parent, strengthened_name in STRENGTHENED.items():
        parent_map = _condition_map(frozen, parent)
        strengthened_map = _condition_map(strengthened, strengthened_name)
        for dataset, target in PAPER_TARGETS.items():
            for horizon in PAPER_HORIZONS:
                key = (dataset, target, horizon)
                if key not in parent_map or key not in strengthened_map or key not in sita:
                    raise ValueError(f"missing comparison condition {key} for {parent}")
                before = parent_map[key]
                after = strengthened_map[key]
                proposed = sita[key]
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "horizon": horizon,
                        "mode": "without",
                        "parent_method": parent,
                        "strengthened_method": strengthened_name,
                        "parent_rmse": before["rmse"],
                        "strengthened_rmse": after["rmse"],
                        "rmse_absolute_difference": after["rmse"] - before["rmse"],
                        "rmse_relative_change_pct": relative_error_change(before["rmse"], after["rmse"]),
                        "parent_smape": before["smape"],
                        "strengthened_smape": after["smape"],
                        "smape_absolute_difference": after["smape"] - before["smape"],
                        "smape_relative_change_pct": relative_error_change(before["smape"], after["smape"]),
                    }
                )
                for metric in ("rmse", "smape"):
                    versus_rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "horizon": horizon,
                            "mode": "without",
                            "metric": metric.upper() if metric == "rmse" else "sMAPE",
                            "sita": proposed[metric],
                            "parent_method": parent,
                            "parent": before[metric],
                            "strengthened_method": strengthened_name,
                            "strengthened": after[metric],
                            "sita_vs_parent_relative_improvement_pct": relative_improvement(
                                before[metric], proposed[metric]
                            ),
                            "sita_vs_strengthened_relative_improvement_pct": relative_improvement(
                                after[metric], proposed[metric]
                            ),
                            "sita_still_better_than_strengthened": proposed[metric] < after[metric],
                        }
                    )
    effect = pd.DataFrame(effect_rows)
    versus = pd.DataFrame(versus_rows)
    stats_rows = []
    for parent, strengthened_name in STRENGTHENED.items():
        for metric, column in (("RMSE", "rmse"), ("sMAPE", "smape")):
            differences = []
            improvements = []
            wins = 0
            for dataset, target in PAPER_TARGETS.items():
                for horizon in PAPER_HORIZONS:
                    key = (dataset, target, horizon)
                    sita_value = sita[key][column]
                    strengthened_value = _condition_map(strengthened, strengthened_name)[key][column]
                    differences.append(sita_value - strengthened_value)
                    improvements.append(relative_improvement(strengthened_value, sita_value))
                    wins += int(sita_value < strengthened_value)
            try:
                test = wilcoxon_exact(differences)
                statistic = test["statistic"]
                p_value = test["p_value"]
                note = ""
            except ValueError as exc:
                statistic = float("nan")
                p_value = float("nan")
                note = str(exc)
            stats_rows.append(
                {
                    "comparison": f"SITA_vs_{strengthened_name}",
                    "metric": metric,
                    "n_conditions": 12,
                    "sita_wins": wins,
                    "mean_relative_improvement_pct": float(sum(improvements) / len(improvements)),
                    "wilcoxon_W": statistic,
                    "wilcoxon_raw_p": p_value,
                    "wilcoxon_alternative": "two-sided",
                    "wilcoxon_method": "exact",
                    "wilcoxon_note": note,
                }
            )
    raw_p = {
        f"{row['comparison']}|{row['metric']}": float(row["wilcoxon_raw_p"])
        for row in stats_rows
    }
    adjusted = holm_adjust(
        {
            key: float(p_value)
            for key, p_value in raw_p.items()
            if np.isfinite(p_value)
        }
    ) if any(np.isfinite(value) for value in raw_p.values()) else {}
    for row in stats_rows:
        key = f"{row['comparison']}|{row['metric']}"
        if key in adjusted:
            row["holm_adjusted_p"] = adjusted[key]["adjusted_p"]
            row["holm_significant_0.05"] = bool(adjusted[key]["adjusted_p"] < 0.05)
        else:
            row["holm_adjusted_p"] = float("nan")
            row["holm_significant_0.05"] = False
    stats = pd.DataFrame(stats_rows)
    return effect, versus, stats


def write_comparison(
    *,
    freeze_path: Path,
    strengthened_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    freeze = Path(freeze_path).resolve()
    source = Path(strengthened_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen(freeze)
    strengthened = _load_strengthened(source)
    effect, versus, stats = build_comparison(frozen, strengthened)
    effect_path = destination / "REVIN_EFFECT_ON_BASELINE.csv"
    versus_path = destination / "SITA_VS_REVIN_STRENGTHENED_BASELINES.csv"
    stats_path = destination / "SITA_VS_REVIN_STRENGTHENED_STATISTICS.csv"
    effect.to_csv(effect_path, index=False, encoding="utf-8")
    versus.to_csv(versus_path, index=False, encoding="utf-8")
    stats.to_csv(stats_path, index=False, encoding="utf-8")
    freeze_sha = sha256_file(freeze)
    payload = {
        "freeze_path": str(freeze),
        "freeze_sha256": freeze_sha,
        "freeze_modified": False,
        "strengthened_path": str(source),
        "output_dir": str(destination),
        "n_effect_rows": int(len(effect)),
        "n_versus_rows": int(len(versus)),
        "n_stat_rows": int(len(stats)),
        "sita_win_counts": {
            f"{row.comparison}|{row.metric}": int(row.sita_wins)
            for row in stats.itertuples()
        },
    }
    manifest_path = destination / "REVIN_STRENGTHENED_COMPARISON_MANIFEST.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sha256_file(freeze) != freeze_sha:
        raise RuntimeError("comparison mutated the freeze workbook")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument(
        "--strengthened",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "REVIN_STRENGTHENED_BASELINES_5SEED_RAW.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_comparison(
        freeze_path=args.freeze,
        strengthened_path=args.strengthened,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
