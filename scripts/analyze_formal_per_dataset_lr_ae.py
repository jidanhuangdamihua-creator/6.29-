#!/usr/bin/env python3
"""Audit and statistically analyze the completed formal per-dataset-LR A–E run.

Analysis only: this script reads the formal output and writes a new statistics
directory. It never trains, mutates experiment roots, or overwrites the
retained uniform-3e-4 statistics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("D1", "D4", "D5", "D6")
TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
LR_LABELS = {"D1": "3e-5", "D4": "3e-5", "D5": "3e-4", "D6": "3e-5"}
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("A", "B", "C", "D", "E")
MATRIX_KIND = "FORMAL_PER_DATASET_LR_AE"
CONTRASTS = (("C-A", "C", "A"), ("C-B", "C", "B"), ("C-E", "C", "E"), ("D-C", "D", "C"))


def _load_frozen_statistical_helpers():
    source = Path(__file__).resolve().parent / "audit_d4c_and_recompute_ae_stats.py"
    spec = importlib.util.spec_from_file_location("frozen_ae_statistical_helpers", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen statistical helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_formal_rows(run_root: Path) -> List[Dict[str, Any]]:
    path = Path(run_root) / "ae_formal_per_dataset_lr_results.csv"
    rows: List[Dict[str, Any]] = []
    for raw in _read_csv(path):
        row = dict(raw)
        row["horizon"] = int(raw["horizon"])
        row["seed"] = int(raw["seed"])
        for key in ("RMSE", "sMAPE", "rmse_original", "smape_original"):
            row[key] = float(raw[key]) if raw.get(key) not in (None, "") else float("nan")
        rows.append(row)
    return rows


def _latest_attempt(cell_dir: Path) -> Path | None:
    attempts = sorted(cell_dir.glob("attempt_*/result.json"))
    return attempts[-1] if attempts else None


def integrity_audit_formal_rows(rows: Sequence[Mapping[str, Any]], run_root: Path) -> Dict[str, Any]:
    expected = [
        (dataset, TARGETS[dataset], horizon, seed, arm)
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
        for arm in ARMS
    ]
    observed = [
        (str(row.get("dataset")), str(row.get("target")), int(row.get("horizon")), int(row.get("seed")), str(row.get("arm")))
        for row in rows
    ]
    counts = Counter(observed)
    expected_set = set(expected)
    missing = [key for key in expected if counts[key] == 0]
    duplicates = [key for key, count in counts.items() if count > 1]
    extras = [key for key in counts if key not in expected_set]
    failed = [row for row in rows if row.get("run_status") != "OK" or row.get("pair_status") != "PAIR_COMPLETE"]
    nan_rows = [row for row in rows if not finite(row.get("rmse_original")) or not finite(row.get("smape_original"))]
    wrong_target = [row for row in rows if row.get("target") != TARGETS.get(str(row.get("dataset")))]
    wrong_lr = [row for row in rows if row.get("learning_rate_label") != LR_LABELS.get(str(row.get("dataset")))]
    wrong_kind = [row for row in rows if row.get("matrix_kind") != MATRIX_KIND]
    artifact_mismatch = 0
    status_abnormal = 0
    runtime_values: Dict[str, set] = {dataset: set() for dataset in DATASETS}
    reused_rows = 0
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                cell_dir = Path(run_root) / "cells" / f"{dataset}__{TARGETS[dataset].replace('/', '-')}__h{horizon}__s{seed}__without"
                result_path = _latest_attempt(cell_dir)
                if result_path is None:
                    artifact_mismatch += 1
                    continue
                status_path = result_path.parent / "status.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
                if status.get("status") != "COMPLETE" or result.get("status") != "COMPLETE" or result.get("pair_status") != "PAIR_COMPLETE":
                    status_abnormal += 1
                for arm in ARMS:
                    arm_payload = (result.get("arms") or {}).get(arm) or {}
                    runtime_values[dataset].add(float(arm_payload.get("selected_lr", float("nan"))))
                    if result.get("reused_in_formal_root") and dataset == "D5":
                        reused_rows += 1
                    matching = [
                        row for row in rows
                        if row.get("dataset") == dataset
                        and int(row.get("horizon")) == horizon
                        and int(row.get("seed")) == seed
                        and row.get("arm") == arm
                    ]
                    if len(matching) != 1:
                        continue
                    if not finite(arm_payload.get("rmse_original")) or abs(float(arm_payload["rmse_original"]) - float(matching[0]["rmse_original"])) > 1e-12:
                        artifact_mismatch += 1
                    if arm_payload.get("status") != "OK":
                        status_abnormal += 1
    expected_runtime = {"D1": {3e-5}, "D4": {3e-5}, "D5": {3e-4}, "D6": {3e-5}}
    runtime_lr = {dataset: LR_LABELS[dataset] if values == expected_runtime[dataset] else sorted(values) for dataset, values in runtime_values.items()}
    complete = (
        len(rows) == 300
        and not missing
        and not duplicates
        and not extras
        and not failed
        and not nan_rows
        and not wrong_target
        and not wrong_lr
        and not wrong_kind
        and artifact_mismatch == 0
        and status_abnormal == 0
        and all(runtime_values[dataset] == expected_runtime[dataset] for dataset in DATASETS)
    )
    return {
        "verdict": "PASS" if complete else "FAIL",
        "n_rows": len(rows),
        "expected_rows": 300,
        "n_missing": len(missing),
        "n_duplicates": len(duplicates),
        "n_extras": len(extras),
        "n_failed": len(failed),
        "n_nan": len(nan_rows),
        "n_wrong_target": len(wrong_target),
        "n_wrong_lr": len(wrong_lr),
        "n_wrong_matrix_kind": len(wrong_kind),
        "artifact_mismatch": artifact_mismatch,
        "status_abnormal": status_abnormal,
        "runtime_lr": runtime_lr,
        "d5_reused_rows": reused_rows,
        "d5_trained_rows": 75 - reused_rows,
        "non_d5_trained_rows": len(rows) - reused_rows,
        "matrix_kind": MATRIX_KIND,
        "missing": [str(key) for key in missing],
        "duplicates": [str(key) for key in duplicates],
        "complete_300": complete,
    }


def compute_formal_statistics(rows: Sequence[Mapping[str, Any]]):
    helpers = _load_frozen_statistical_helpers()
    means = helpers.condition_means(rows)
    pairwise, statistics = helpers.contrast_tables(means)
    return means, pairwise, statistics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def _comparison_rows(statistics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    uniform_path = ROOT / "outputs" / "paper_statistics" / "corrected_ae_statistical_analysis_20260919_154638" / "03_AE_STATISTICAL_TESTS.csv"
    uniform = _read_csv(uniform_path)
    uniform_index = {(row.get("contrast"), row.get("metric")): row for row in uniform}
    output = []
    for row in statistics:
        key = (row["comparison"], row["metric"])
        old = uniform_index.get(key, {})
        output.append(
            {
                "comparison": row["comparison"],
                "metric": row["metric"],
                "uniform_wins": old.get("wins", ""),
                "uniform_ties": old.get("ties", ""),
                "uniform_losses": old.get("losses", ""),
                "formal_wins": row.get("wins"),
                "formal_ties": row.get("ties"),
                "formal_losses": row.get("losses"),
                "uniform_mean_RI": old.get("mean_relative_improvement_pct", ""),
                "formal_mean_RI": row.get("mean_relative_improvement_pct"),
                "uniform_bootstrap_low": old.get("bootstrap_ci95_low", ""),
                "uniform_bootstrap_high": old.get("bootstrap_ci95_high", ""),
                "formal_bootstrap_low": row.get("bootstrap_ci95_low"),
                "formal_bootstrap_high": row.get("bootstrap_ci95_high"),
                "uniform_raw_p": old.get("raw_p", ""),
                "formal_raw_p": row.get("wilcoxon_raw_p"),
                "uniform_holm_p": old.get("holm_adjusted_p", ""),
                "formal_holm_p": row.get("holm_adjusted_p"),
                "uniform_reject": old.get("reject_at_0_05", ""),
                "formal_reject": row.get("holm_reject_at_0.05"),
            }
        )
    return output


def _write_report(output_dir: Path, audit: Mapping[str, Any], statistics: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]], d4_ca: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# FORMAL_PER_DATASET_LR_AE statistical audit",
        "",
        f"- Integrity verdict: `{audit['verdict']}`",
        f"- Rows: `{audit['n_rows']}/300`; missing `{audit['n_missing']}`; duplicate `{audit['n_duplicates']}`; failed `{audit['n_failed']}`; NaN `{audit['n_nan']}`.",
        f"- Runtime target LR: `{audit['runtime_lr']}`.",
        f"- D5 reuse: `{audit['d5_reused_rows']}` rows; D5 trained rows: `{audit['d5_trained_rows']}`; D1/D4/D6 trained rows: `{audit['non_d5_trained_rows']}`.",
        "",
        "## Formal C–A",
        "",
    ]
    ca = [row for row in statistics if row["comparison"] == "C-A"]
    for row in ca:
        lines.append(
            f"- {row['metric']}: W/T/L `{row['wins']}/{row['ties']}/{row['losses']}`, mean RI `{float(row['mean_relative_improvement_pct']):.6f}%`, bootstrap CI `[{float(row['bootstrap_ci95_low']):.6f}, {float(row['bootstrap_ci95_high']):.6f}]`, Wilcoxon p `{float(row['wilcoxon_raw_p']):.9g}`, Holm p `{float(row['holm_adjusted_p']):.9g}`."
        )
    lines.extend(["", "### D4 C–A", ""])
    for row in d4_ca:
        lines.append(
            f"- H{row['horizon']} {row['metric']}: treatment `{row['treatment_5seed_mean']}`, reference `{row['reference_5seed_mean']}`, RI `{row['relative_improvement_pct']:.6f}%`, `{row['win_tie_loss']}`."
        )
    lines.extend(["", "## Formal vs retained UNIFORM_3E4_AE", ""])
    for row in comparison:
        lines.append(
            f"- {row['comparison']} {row['metric']}: uniform W/T/L `{row['uniform_wins']}/{row['uniform_ties']}/{row['uniform_losses']}` vs formal `{row['formal_wins']}/{row['formal_ties']}/{row['formal_losses']}`; mean RI `{row['uniform_mean_RI']}` -> `{float(row['formal_mean_RI']):.6f}`; Holm p `{row['uniform_holm_p']}` -> `{float(row['formal_holm_p']):.9g}`."
        )
    (output_dir / "06_FORMAL_PER_DATASET_LR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(run_root: Path, output_dir: Path) -> Dict[str, Any]:
    run_root = Path(run_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing statistics output: {output_dir}")
    rows = load_formal_rows(run_root)
    audit = integrity_audit_formal_rows(rows, run_root)
    if audit["verdict"] != "PASS":
        raise RuntimeError(f"formal integrity audit failed: {audit}")
    means, pairwise, statistics = compute_formal_statistics(rows)
    d4_ca = [row for row in pairwise if row["comparison"] == "C-A" and row["dataset"] == "D4"]
    comparison = _comparison_rows(statistics)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "00_INTEGRITY_AUDIT.json").write_text(json.dumps(_clean_json(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "01_FORMAL_CONDITION_MEANS.csv", means)
    _write_csv(output_dir / "02_FORMAL_CONDITION_CONTRASTS.csv", pairwise)
    _write_csv(output_dir / "03_FORMAL_STATISTICAL_TESTS.csv", statistics)
    _write_csv(output_dir / "04_D4_C_A_DETAIL.csv", d4_ca)
    _write_csv(output_dir / "05_UNIFORM_VS_FORMAL_COMPARISON.csv", comparison)
    manifest = {
        "label": "FORMAL_PER_DATASET_LR_AE",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "integrity": audit,
        "condition_mean_unit": "dataset x horizon x arm after 5-seed mean",
        "condition_count": 12,
        "relative_improvement": "(E_reference - E_treatment) / E_reference * 100%",
        "wilcoxon": {"alternative": "two-sided", "n": 12, "zero_method": "wilcox", "method": "auto"},
        "bootstrap": {"resamples": 100000, "unit": "12 condition-level RI values", "seed": 20260919, "type": "percentile"},
        "holm": {"families": ["RMSE_4", "sMAPE_4"], "alpha": 0.05},
        "uniform_comparison_source": str(ROOT / "outputs" / "paper_statistics" / "corrected_ae_statistical_analysis_20260919_154638"),
    }
    (output_dir / "STATISTICS_MANIFEST.json").write_text(json.dumps(_clean_json(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output_dir, audit, statistics, comparison, d4_ca)
    return {"output_dir": str(output_dir), "audit": audit, "statistics": statistics, "comparison": comparison}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze formal per-dataset-LR A–E results only.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (ROOT / "outputs" / "paper_statistics" / f"formal_per_dataset_lr_ae_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    result = analyze(args.run_root, output_dir)
    audit = result["audit"]
    print(f"RUN_ROOT={args.run_root}")
    print(f"OUTPUT_DIR={result['output_dir']}")
    print(f"INTEGRITY_VERDICT={audit['verdict']}")
    print(f"ROWS={audit['n_rows']}")
    print(f"MISSING={audit['n_missing']}")
    print(f"DUPLICATES={audit['n_duplicates']}")
    print(f"FAILED={audit['n_failed']}")
    print(f"NAN={audit['n_nan']}")
    print(f"D5_REUSED_ROWS={audit['d5_reused_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
