#!/usr/bin/env python3
"""Arm A vs Arm C learning-rate sensitivity screening for frozen SITA / V3 A–E.

This wrapper does not change the SITA model, A–E ablation definitions, or the
formal 270-run results. It only varies Stage-B learning rate for arms A and C.

Usage:
    python scripts/run_ms_katl_sita_ac_lr_sensitivity.py --dry-run
    SITA_AC_LR_SCREENING_UNLOCK=YES python scripts/run_ms_katl_sita_ac_lr_sensitivity.py --execute
    python scripts/run_ms_katl_sita_ac_lr_sensitivity.py --analyze --output-dir outputs/sita_ablation_lr_sensitivity_ac_seed42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_sita_ac_lr_sensitivity import (
    EXPECTED_TOTAL_RUNS,
    UNLOCK_ENV,
    LRSensitivityContractError,
    LRSensitivityTrainingBlocked,
    analyze_screening_results,
    assert_output_isolated,
    default_output_dir,
    enumerate_lr_sensitivity_runs,
    run_lr_sensitivity_dry_run,
    run_lr_sensitivity_execute,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen SITA/V3 Arm A vs Arm C Stage-B learning-rate sensitivity screening."
    )
    parser.add_argument("--dry-run", "--preflight", action="store_true", dest="dry_run")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run the 108-run screening. Requires {UNLOCK_ENV}=YES. Does not overwrite formal A–E results.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze completed 108-run artifacts. Does not train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Independent output root. Default: outputs/sita_ablation_lr_sensitivity_ac_seed42",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Retrain even if an OK cell artifact already exists. Default is skip-existing / resume.",
    )
    parser.add_argument(
        "--skip-real-cell-resolve",
        action="store_true",
        help="Dry-run without loading KNN authorities. Default dry-run resolves real D1–D6 routing.",
    )
    return parser.parse_args()


def _resolve_output_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        return default_output_dir(PROJECT_ROOT)
    return assert_output_isolated(requested, project_root=PROJECT_ROOT)


def main() -> int:
    args = _parse_args()
    selected = [args.dry_run, args.execute, args.analyze]
    if sum(bool(item) for item in selected) != 1:
        print("EXPLICIT_MODE_REQUIRED=YES")
        print("Pass exactly one of --dry-run, --execute, --analyze")
        print("TRAINING_STARTED=NO")
        return 2
    output_dir = _resolve_output_dir(args.output_dir)
    try:
        if args.dry_run:
            result = run_lr_sensitivity_dry_run(
                output_dir=output_dir,
                project_root=PROJECT_ROOT,
                resolve_real_cells=not args.skip_real_cell_resolve,
            )
            print(result["text"], end="")
            print(f"OUTPUT_DIR={result['output_root']}")
            print(f"EXPECTED_TOTAL_RUNS={EXPECTED_TOTAL_RUNS}")
            print(f"DRY_RUN_TOTAL_RUNS={result['DRY_RUN_TOTAL_RUNS']}")
            print(f"READY_FOR_LR_SCREENING={result['READY_FOR_LR_SCREENING']}")
            print("TRAINING_STARTED=NO")
            if result["READY_FOR_LR_SCREENING"] != "YES" or int(result["DRY_RUN_TOTAL_RUNS"]) != EXPECTED_TOTAL_RUNS:
                return 2
            return 0
        if args.analyze:
            identities = enumerate_lr_sensitivity_runs(PROJECT_ROOT)
            payload = analyze_screening_results(output_dir, identities)
            print(f"OUTPUT_DIR={output_dir}")
            print(f"ANALYSIS_COMPLETE={payload.get('ANALYSIS_COMPLETE')}")
            print(f"OK_RUNS={payload.get('OK_RUNS')}")
            print("TRAINING_STARTED=NO")
            print("TEST_BASED_LR_SELECTION=NO")
            print("FORMAL_RESULTS_OVERWRITTEN=NO")
            return 0 if payload.get("ANALYSIS_COMPLETE") == "YES" else 2
        result = run_lr_sensitivity_execute(
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
            skip_existing=not args.no_skip_existing,
        )
        print(f"OUTPUT_DIR={result['output_root']}")
        print(f"OK_RUNS={result['OK_RUNS']}")
        print(f"FAILED_RUNS={result['FAILED_RUNS']}")
        print(f"PENDING_AFTER={result['PENDING_AFTER']}")
        print(f"TRAINING_STARTED={result['TRAINING_STARTED']}")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        print("TEST_BASED_LR_SELECTION=NO")
        if int(result["PENDING_AFTER"]) > 0:
            print("SCREENING_COMPLETE=NO")
            return 2
        print("SCREENING_COMPLETE=YES")
        return 0
    except LRSensitivityTrainingBlocked as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("FORMAL_TRAINING_BLOCKED=YES")
        return 2
    except LRSensitivityContractError as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("READY_FOR_LR_SCREENING=NO")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
