"""Recompute D4 A/B/C metrics in original_sales_space from an existing run.

Does not retrain models, refit scalers, or overwrite historical results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.transfer_methods.ms_katl_cross_target import (
    D4_CROSS_TARGET_IDENTITIES,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_FORMAL_RELATIVE,
    load_baseline_references,
    load_formal_references,
    parse_d4_target_identity,
)
from src.transfer_methods.ms_katl_metric_space_repair import (
    ARM_FILES,
    SOURCE_RUN_NAME,
    aggregate_original_space,
    assert_historical_run_untouched,
    build_original_space_comparison,
    count_external_references,
    decide_cross_target_verdict,
    expected_prediction_files,
    format_original_space_tables,
    load_diagnostics,
    load_prediction_csv,
    load_raw_test_windows,
    print_repair_preflight,
    repair_one_target,
    repair_root,
    sanity_check_166_258,
    snapshot_historical_run,
    source_run_dir,
    write_repair_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_cross_target_runner():
    path = PROJECT_ROOT / "scripts" / "run_ms_katl_d4_cross_target.py"
    spec = importlib.util.spec_from_file_location("run_ms_katl_d4_cross_target_for_repair", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-KATL D4 metric-space repair (no retraining).")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--formal-csv", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else source_run_dir(PROJECT_ROOT)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    formal_csv = args.formal_csv or (PROJECT_ROOT / DEFAULT_FORMAL_RELATIVE)
    baseline_csv = args.baseline_csv or (PROJECT_ROOT / DEFAULT_BASELINE_RELATIVE)

    pred_files = expected_prediction_files(run_dir)
    pred_found = sum(1 for path in pred_files if path.exists())
    diagnostics = load_diagnostics(run_dir)
    scaler_ok = "YES"
    for identity in D4_CROSS_TARGET_IDENTITIES:
        for arm in ARM_FILES:
            split = diagnostics["targets"][identity]["arms"][arm].get("target_split_digest") or {}
            if not split.get("target_scaler"):
                scaler_ok = "NO"
    refs = count_external_references(
        project_root=PROJECT_ROOT,
        formal_csv=formal_csv,
        baseline_csv=baseline_csv,
    )
    print_repair_preflight(
        prediction_files_found=pred_found,
        scalers_available=scaler_ok,
        external_references_found=refs["found"],
    )
    if args.preflight:
        return
    if pred_found != 15:
        missing = [str(path) for path in pred_files if not path.exists()]
        raise RuntimeError(f"PREDICTION_FILES_FOUND={pred_found}/15 missing={missing}")
    if scaler_ok != "YES":
        raise RuntimeError("EXACT_TARGET_SCALERS_AVAILABLE=NO")
    if refs["found"] != 35:
        raise RuntimeError(f"EXTERNAL_REFERENCES_FOUND={refs['found']}/35 missing={refs['missing']}")

    before = snapshot_historical_run(run_dir)
    runner = _load_cross_target_runner()
    config = runner._build_config(50)
    pool = runner._load_d4_without_pool(config)

    target_repairs: Dict[str, Any] = []
    scaler_provenance = {}
    processed = 0
    for identity in D4_CROSS_TARGET_IDENTITIES:
        parsed = parse_d4_target_identity(identity)
        pred_dir = run_dir / "predictions" / parsed["prediction_dirname"]
        frames = {arm: load_prediction_csv(pred_dir / f"{arm}.csv") for arm in ARM_FILES}
        cell = runner.prepare_d4_without_target_cell(
            config,
            entity_key=parsed["entity_key"],
            knn_data=pool["knn_data"],
            source_df=pool["source_df"],
            target_df=pool["target_df"],
            feature_cols=pool["feature_cols"],
            group_cols=pool["group_cols"],
        )
        raw_windows = load_raw_test_windows(
            cell["target_model_df"],
            feature_columns=cell["model_feature_cols"],
            horizon=int(config["horizon"]),
            window_size=int(config["window_size"]),
        )
        repaired = repair_one_target(
            target_identity=identity,
            prediction_frames=frames,
            raw_windows=raw_windows,
            feature_columns=cell["model_feature_cols"],
            diagnostics=diagnostics,
        )
        scaler_provenance[identity] = {
            "target_scaler_digest": repaired["target_scaler_digest"],
            "sales_min": repaired["sales_min"],
            "sales_max": repaired["sales_max"],
            "sales_index": repaired["sales_index"],
            "feature_columns": repaired["feature_columns"],
            "recovery": "training_time_minmax_identity_from_paired_y_true",
            "scaler_refit": False,
            "n_test": raw_windows["n_test"],
        }
        target_repairs.append(repaired)
        processed += 3
        print(
            f"[repair] {identity} sales_min={repaired['sales_min']:.6f} "
            f"sales_max={repaired['sales_max']:.6f} "
            f"B_original_rmse={repaired['arm_results']['fused_target_only']['rmse_original']:.6f}"
        )

    comparisons = []
    for repaired in target_repairs:
        identity = repaired["target_identity"]
        formal = load_formal_references(formal_csv, target_identity=identity)
        baselines = load_baseline_references(baseline_csv, target_identity=identity)
        comparisons.append(
            build_original_space_comparison(
                target_repair=repaired,
                formal_refs=formal,
                baseline_refs=baselines,
            )
        )
    aggregates = aggregate_original_space(comparisons)
    verdict = decide_cross_target_verdict(
        {
            "B_vs_A": aggregates["B_vs_A"],
            "B_vs_C": aggregates["B_vs_C"],
            "median_init_effect": aggregates["median_init_effect"],
        },
        [
            {
                "B_rmse": item["B_rmse"],
                "A_rmse": item["A_rmse"],
                "C_rmse": item["C_rmse"],
            }
            for item in comparisons
        ],
    )
    sanity = sanity_check_166_258(comparisons)
    audit = {
        "source_run_root": str(run_dir),
        "source_prediction_files": [str(path) for path in pred_files],
        "source_metric_space": "normalized",
        "target_metric_space": ORIGINAL_SALES_SPACE,
        "inverse_transform_contract": {
            "helper": "src.evaluation.metrics.inverse_transform_sales",
            "sales_value_policy": "clip_negative_to_zero_v1",
            "smape": "src.evaluation.metrics.smape",
            "scaler_refit": False,
        },
        "target_scaler_provenance": scaler_provenance,
        "prediction_files_processed": processed,
        "prediction_files_successful": processed,
        "external_reference_source": {
            "formal_csv": str(formal_csv),
            "baseline_csv": str(baseline_csv),
            "accepted_git": "630ae4ef33a271187ed7225437d0609d8a13e941",
        },
        "external_reference_join_count": refs["label"],
        "external_references_found": refs["found"],
        "external_references_missing": refs["missing"],
        "historical_results_overwritten": False,
        "models_retrained": False,
        "architecture_modified": False,
        "sanity_166_258": sanity,
        "CURRENT_PREDICTION_SPACE": "normalized",
        "TARGET_SCALER_AVAILABLE": "YES",
        "INVERSE_TRANSFORM_REPRODUCIBLE": "YES",
        "A/B/C_TARGET_SCALER_MATCH": "YES",
        "A/B/C_TARGET_SPLIT_MATCH": "YES",
        "A/B/C_Y_TRUE_MATCH": "YES",
    }
    outputs = write_repair_outputs(
        run_dir=run_dir,
        comparisons=comparisons,
        aggregates=aggregates,
        audit=audit,
        verdict=verdict,
    )
    after = snapshot_historical_run(run_dir)
    assert_historical_run_untouched(before, after)
    if repair_root(run_dir) not in outputs["results_csv"].parents:
        raise RuntimeError("repair outputs must live under metric_space_repair/")
    print()
    print(format_original_space_tables(comparisons, aggregates, verdict=verdict))
    print()
    print(f"WROTE {outputs['results_csv']}")
    print(f"WROTE {outputs['summary_csv']}")
    print(f"WROTE {outputs['audit_json']}")


if __name__ == "__main__":
    main()
