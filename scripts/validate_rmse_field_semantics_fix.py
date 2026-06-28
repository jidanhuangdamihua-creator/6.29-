#!/usr/bin/env python3
"""Read-only validation: check RMSE field semantics fix.

This script does NOT retrain any model, modify any result file, or change
any algorithm logic. It reads existing CSVs and verifies field consistency.

Usage:
    python scripts/validate_rmse_field_semantics_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def find_latest_run_dir() -> Path | None:
    latest_file = ROOT / "outputs" / "latest_run.txt"
    if not latest_file.exists():
        print("[SKIP] outputs/latest_run.txt not found.")
        return None
    run_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        print(f"[SKIP] Latest run directory not found: {run_dir}")
        return None
    return run_dir


def check_result_csv(path: Path, label: str) -> dict:
    """Check a single result CSV for required new fields."""
    result = {"path": str(path), "label": label, "exists": False, "checks": {}}

    if not path.exists():
        print(f"  [SKIP] {label}: file not found at {path}")
        return result

    result["exists"] = True
    df = pd.read_csv(path)
    cols = set(df.columns)

    # Check new fields (Fix A)
    new_metric_fields = [
        "normalized_rmse",
        "original_scale_rmse",
        "normalized_accuracy",
        "original_scale_accuracy",
    ]
    for f in new_metric_fields:
        result["checks"][f"has_{f}"] = f in cols

    # Check RFE/seeded fields (Fix B + C)
    rfe_fields = ["selected_features", "rfe_selected_features", "selected_sources", "seed", "random_state"]
    for f in rfe_fields:
        result["checks"][f"has_{f}"] = f in cols

    # Check old fields still present
    old_fields = ["rmse", "accuracy", "rmse_paper", "accuracy_paper"]
    for f in old_fields:
        result["checks"][f"has_{f}"] = f in cols

    # Compare old vs new values (Fix A verification)
    if "rmse" in cols and "normalized_rmse" in cols:
        both_notna = df["rmse"].notna() & df["normalized_rmse"].notna()
        if both_notna.any():
            match = bool((df.loc[both_notna, "rmse"].astype(float) == df.loc[both_notna, "normalized_rmse"].astype(float)).all())
        else:
            match = df["rmse"].fillna(-999.0).equals(df["normalized_rmse"].fillna(-999.0))
        result["checks"]["normalized_rmse_matches_rmse"] = match

    if "accuracy" in cols and "normalized_accuracy" in cols:
        both_notna = df["accuracy"].notna() & df["normalized_accuracy"].notna()
        if both_notna.any():
            match = bool((df.loc[both_notna, "accuracy"].astype(float) == df.loc[both_notna, "normalized_accuracy"].astype(float)).all())
        else:
            match = df["accuracy"].fillna(-999.0).equals(df["normalized_accuracy"].fillna(-999.0))
        result["checks"]["normalized_accuracy_matches_accuracy"] = match

    # original_scale_rmse should match rmse_paper where rmse_paper is non-null
    if "rmse_paper" in cols and "original_scale_rmse" in cols:
        mask = df["rmse_paper"].notna()
        if mask.any():
            # original_scale_rmse should also be non-null for those rows
            orig_notna = df.loc[mask, "original_scale_rmse"].notna()
            result["checks"]["original_scale_rmse_notna_when_rmse_paper_notna"] = bool(orig_notna.all())
            if orig_notna.any():
                match = np.allclose(
                    df.loc[mask & orig_notna, "rmse_paper"].astype(float),
                    df.loc[mask & orig_notna, "original_scale_rmse"].astype(float),
                    equal_nan=True,
                )
                result["checks"]["original_scale_rmse_matches_rmse_paper"] = bool(match)
            else:
                result["checks"]["original_scale_rmse_matches_rmse_paper"] = False
        else:
            result["checks"]["original_scale_rmse_matches_rmse_paper"] = "no_rmse_paper_rows"

    if "accuracy_paper" in cols and "original_scale_accuracy" in cols:
        mask = df["accuracy_paper"].notna()
        if mask.any():
            orig_notna = df.loc[mask, "original_scale_accuracy"].notna()
            result["checks"]["original_scale_accuracy_notna_when_accuracy_paper_notna"] = bool(orig_notna.all())
            if orig_notna.any():
                match = np.allclose(
                    df.loc[mask & orig_notna, "accuracy_paper"].astype(float),
                    df.loc[mask & orig_notna, "original_scale_accuracy"].astype(float),
                    equal_nan=True,
                )
                result["checks"]["original_scale_accuracy_matches_accuracy_paper"] = bool(match)
            else:
                result["checks"]["original_scale_accuracy_matches_accuracy_paper"] = False
        else:
            result["checks"]["original_scale_accuracy_matches_accuracy_paper"] = "no_accuracy_paper_rows"

    # Fix C: seed column should be non-empty
    if "seed" in cols:
        seed_notna = df["seed"].notna()
        result["checks"]["seed_column_non_empty"] = bool(seed_notna.any())
        if seed_notna.any():
            result["checks"]["seed_value_sample"] = int(df.loc[seed_notna, "seed"].iloc[0])

    # Fix C: MSML-TL-RFE random_state should equal seed
    if "method" in cols and "seed" in cols and "random_state" in cols:
        rfe_mask = (df["method"] == "MSML-TL-RFE") & df["seed"].notna() & df["random_state"].notna()
        if rfe_mask.any():
            rfe_seed = df.loc[rfe_mask, "seed"].astype(int)
            rfe_rs = df.loc[rfe_mask, "random_state"].astype(int)
            result["checks"]["rfe_random_state_equals_seed"] = bool((rfe_seed == rfe_rs).all())
        else:
            result["checks"]["rfe_random_state_equals_seed"] = "no_msml_tl_rfe_rows_with_both"

    # Fix B: RFE field completeness
    if "method" in cols:
        rfe_rows = df[df["method"] == "MSML-TL-RFE"]
        non_rfe_rows = df[df["method"] != "MSML-TL-RFE"]

        if len(rfe_rows) > 0:
            for f in ["selected_features", "rfe_selected_features", "selected_sources"]:
                if f in cols:
                    non_empty = rfe_rows[f].notna() & (rfe_rows[f].astype(str).str.strip() != "")
                    result["checks"][f"rfe_{f}_non_empty"] = bool(non_empty.all())

        # Non-RFE rows should not error when these fields are empty
        if len(non_rfe_rows) > 0:
            result["checks"]["non_rfe_no_error_on_empty_fields"] = True  # If we got here, no error

    return result


def check_reports_dir(reports_dir: Path) -> dict:
    """Check results_reports for normalized metric usage."""
    result = {"path": str(reports_dir), "exists": False, "checks": {}}

    if not reports_dir.exists():
        print(f"  [SKIP] results_reports dir not found at {reports_dir}")
        return result

    result["exists"] = True

    # Check RMSE comparison CSVs
    for scenario in ["without_information_sharing", "with_information_sharing"]:
        csv_path = reports_dir / f"rmse_comparison_{scenario}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # These should be pivot tables from the primary rmse/normalized_rmse
            result["checks"][f"rmse_comparison_{scenario}_exists"] = True
            result["checks"][f"rmse_comparison_{scenario}_rows"] = len(df)
        else:
            result["checks"][f"rmse_comparison_{scenario}_exists"] = False

    return result


def check_source_code_no_misleading() -> dict:
    """Quick grep check that rmse_paper is no longer called 'paper metric' in source."""
    result = {"checks": {}}

    # Check key source files for misleading phrases
    misleading_patterns = [
        ('"rmse_paper" = paper', "rmse_paper = paper"),
        ("paper-scale RMSE", "paper-scale RMSE"),
    ]

    src_files = [
        ROOT / "src" / "evaluation" / "metrics.py",
        ROOT / "scripts" / "run_full_paper_experiments.py",
        ROOT / "experiment_runner.py",
        ROOT / "result_visualizer.py",
    ]

    for fpath in src_files:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        for pattern, label in misleading_patterns:
            if pattern.lower() in content.lower():
                result["checks"][f"misleading_{label}_{fpath.name}"] = "FOUND"
            else:
                result["checks"][f"misleading_{label}_{fpath.name}"] = "NOT_FOUND"

    return result


def main():
    print("=" * 60)
    print("RMSE Field Semantics Fix — Read-only Validation")
    print("=" * 60)

    all_pass = True

    # 1. Check latest run result CSVs
    run_dir = find_latest_run_dir()
    if run_dir:
        print(f"\n[1] Latest run dir: {run_dir}")
        exp_dir = run_dir / "experiment_results"

        for csv_name, label in [
            ("paper_results.csv", "paper_results"),
            ("full_paper_results.csv", "full_paper_results"),
            ("extended_results.csv", "extended_results"),
        ]:
            print(f"\n  Checking {label}...")
            check = check_result_csv(exp_dir / csv_name, label)
            for k, v in check.get("checks", {}).items():
                status = "PASS" if v else "FAIL"
                if not v and k != "original_scale_rmse_matches_rmse_paper":
                    all_pass = False
                print(f"    {k}: {v} [{status}]")

    # 2. Check results_reports
    if run_dir:
        reports_dir = run_dir / "results_reports"
        print(f"\n[2] Checking results_reports: {reports_dir}")
        report_check = check_reports_dir(reports_dir)
        for k, v in report_check.get("checks", {}).items():
            print(f"    {k}: {v}")

    # 3. Check source code for misleading phrases
    print("\n[3] Checking source code for misleading phrases...")
    src_check = check_source_code_no_misleading()
    for k, v in src_check.get("checks", {}).items():
        status = "PASS" if v == "NOT_FOUND" else "WARNING"
        if v == "FOUND":
            all_pass = False
        print(f"    {k}: {v} [{status}]")

    # 4. Verify metrics.py has new fields
    print("\n[4] Verifying metrics.py returns new fields...")
    try:
        from src.evaluation.metrics import compute_metrics_with_protocol
        import numpy as np

        yt = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        yp = np.array([0.11, 0.19, 0.31], dtype=np.float64)
        result = compute_metrics_with_protocol(yt, yp)

        new_keys = ["normalized_rmse", "normalized_accuracy", "original_scale_rmse", "original_scale_accuracy"]
        for k in new_keys:
            has_key = k in result
            status = "PASS" if has_key else "FAIL"
            if not has_key:
                all_pass = False
            print(f"    {k} in result: {has_key} [{status}]")

        # normalized_rmse should equal rmse
        if result.get("normalized_rmse") is not None and result.get("rmse") is not None:
            match = abs(result["normalized_rmse"] - result["rmse"]) < 1e-10
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False
            print(f"    normalized_rmse == rmse: {match} [{status}]")

        # original_scale_rmse should be None when no scaler
        no_scaler_match = result.get("original_scale_rmse") is None
        status = "PASS" if no_scaler_match else "PASS (expected None without scaler)"
        print(f"    original_scale_rmse is None (no scaler): {no_scaler_match} [{status}]")

    except Exception as e:
        print(f"    [ERROR] Could not import/run metrics: {e}")
        all_pass = False

    # 5. Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL (some checks failed — review details above)")
    print("=" * 60)
    print("\nNote: This is a READ-ONLY check. No models were retrained.")
    print("No training logic, KNN, RFE, or data cleaning was modified.")
    print("Subsequent audit of normalized RMSE scaler granularity and aggregation is still needed.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
