#!/usr/bin/env python3
"""Uniform LR=3e-4 paper-target A–E ablation (D1/D4/D5/D6, 5 seeds).

This wrapper does not change the SITA model, A–E ablation definitions, or the
frozen 270-run / extra-target results. It only applies a global Stage-B LR of
3e-4 on the specified paper targets.

Usage:
    python scripts/run_ms_katl_sita_ablation_3e4.py --dry-run
    AE_FORMAL_TRAINING_UNLOCK=YES python scripts/run_ms_katl_sita_ablation_3e4.py --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_FORMAL_EXECUTION_ENABLED,
    AE_ARMS,
    UNIFORM_3E4_RUNS,
    AEContractError,
    AETrainingBlocked,
)
from src.transfer_methods.ms_katl_sita_ablation_3e4 import (
    UNLOCK_ENV,
    Uniform3e4ContractError,
    assert_output_isolated,
    default_output_dir,
    run_uniform_3e4_plan_only,
    run_uniform_3e4_dry_run,
    run_uniform_3e4_execute,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen SITA/V3 A–E uniform LR=3e-4 paper-target 5-seed ablation."
    )
    parser.add_argument("--dry-run", "--preflight", action="store_true", dest="dry_run")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run the 300-run slice. Requires {UNLOCK_ENV}=YES. Does not overwrite frozen A–E results.",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        type=Path,
        dest="output_root",
        default=None,
        help="New corrected output root; required with --execute unless --resume is used.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing corrected root to resume after fail-closed digest validation.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Retrain even if a PAIR_COMPLETE cell bundle already exists. Default is skip-existing / resume.",
    )
    parser.add_argument(
        "--skip-real-cell-resolve",
        action="store_true",
        help="Dry-run without loading KNN authorities. Default dry-run resolves real D1/D4/D5/D6 routing.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Validate matrix and selector without writing files.")
    parser.add_argument("--dataset", default=None, help="Formal matrix dataset, e.g. D1 or D4.")
    parser.add_argument("--horizon", type=int, default=None, help="Formal matrix horizon: 1, 3, or 5.")
    parser.add_argument("--seed", type=int, default=None, help="Formal matrix seed: 42, 43, 44, 45, or 46.")
    parser.add_argument("--arm", choices=AE_ARMS, default=None, help="Formal A-E arm selector.")
    return parser.parse_args(argv)


def _resolve_output_dir(requested: Optional[Path], resume: Optional[Path], *, dry_run: bool) -> Optional[Path]:
    if requested is not None and resume is not None:
        raise Uniform3e4ContractError("provide either --output-root or --resume, not both")
    if resume is not None:
        return assert_output_isolated(resume, project_root=PROJECT_ROOT)
    if requested is not None:
        return assert_output_isolated(requested, project_root=PROJECT_ROOT)
    if dry_run:
        return assert_output_isolated(
            PROJECT_ROOT / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_dry_run",
            project_root=PROJECT_ROOT,
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    enabled = "YES" if AE_FORMAL_EXECUTION_ENABLED else "NO"
    try:
        output_dir = _resolve_output_dir(args.output_root, args.resume, dry_run=args.dry_run)
    except Uniform3e4ContractError as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("READY_FOR_FULL_RUN=NO")
        return 2
    selector_values = (args.dataset, args.horizon, args.seed, args.arm)
    if any(value is not None for value in selector_values) and not all(value is not None for value in selector_values):
        print("single-cell selector requires --dataset --horizon --seed --arm")
        print("TRAINING_STARTED=NO")
        return 2
    if args.execute and args.dry_run:
        print("pass exactly one of --dry-run or --execute")
        print("TRAINING_STARTED=NO")
        return 2
    if args.execute:
        try:
            result = run_uniform_3e4_execute(
                output_dir=output_dir if args.resume is None else None,
                resume_root=args.resume,
                project_root=PROJECT_ROOT,
                skip_existing=not args.no_skip_existing,
                selected_dataset=args.dataset,
                selected_horizon=args.horizon,
                selected_seed=args.seed,
                selected_arm=args.arm,
            )
        except AETrainingBlocked as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            print("FORMAL_TRAINING_BLOCKED=YES")
            print("FORMAL_EXECUTION_BLOCKED=YES")
            print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
            print("ACTUAL_EXECUTION_REQUESTED=YES")
            print("READY_FOR_FULL_RUN=NO")
            return 2
        except (AEContractError, Uniform3e4ContractError) as exc:
            print(str(exc))
            print("FORMAL_COMPLETION_FAILED=YES")
            print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
            print("ACTUAL_EXECUTION_REQUESTED=YES")
            print("READY_FOR_FULL_RUN=NO")
            return 2
        print(f"OUTPUT_ROOT={result.get('output_root')}")
        print(f"MATRIX_KIND={result.get('MATRIX_KIND')}")
        print(f"EXPECTED_RUNS={result.get('EXPECTED_RUNS')}")
        print(f"PAIR_COMPLETE={result.get('PAIR_COMPLETE')}")
        print(f"PAIR_INCOMPLETE={result.get('PAIR_INCOMPLETE')}")
        print(f"TRAINING_STARTED={result.get('TRAINING_STARTED')}")
        if result.get("results_csv"):
            print(f"RESULTS_CSV={result['results_csv']}")
        pair_incomplete = int(result.get("PAIR_INCOMPLETE") or 0)
        if pair_incomplete > 0:
            print("FORMAL_COMPLETION_FAILED=YES")
            return 2
        print("FORMAL_COMPLETION_FAILED=NO")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        return 0
    if not args.dry_run:
        try:
            plan = run_uniform_3e4_plan_only(
                dataset=args.dataset,
                horizon=args.horizon,
                seed=args.seed,
                arm=args.arm,
            )
        except (AEContractError, Uniform3e4ContractError) as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            return 2
        print("PLAN_ONLY=YES")
        print("TRAINING_STARTED=NO")
        print("SINGLE_CELL_GATE_CLI_READY=PASS")
        print("SINGLE_CELL_GATE_USES_FORMAL_RUNTIME=YES")
        print(f"DRY_RUN_TOTAL_RUNS={plan['DRY_RUN_TOTAL_RUNS']}")
        print(f"EXPECTED_TOTAL_RUNS={plan['EXPECTED_TOTAL_RUNS']}")
        if plan.get("selected_cell"):
            print(f"SELECTED_STAGE_A_KEY={plan['selected_cell']['stage_a_key']}")
            print(f"SELECTED_TARGET={plan['selected_cell']['target']}")
            print(f"SELECTED_ARM={plan['selected_cell']['arm']}")
        return 0
    try:
        result = run_uniform_3e4_dry_run(
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
            resolve_real_cells=not args.skip_real_cell_resolve,
        )
    except (AEContractError, Uniform3e4ContractError) as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("READY_FOR_FULL_RUN=NO")
        return 2
    print(result["text"], end="" if str(result["text"]).endswith("\n") else "\n")
    print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
    print("ACTUAL_EXECUTION_REQUESTED=NO")
    print(f"CODE_CHANGE_REQUIRED={result['CODE_CHANGE_REQUIRED']}")
    print(f"DRY_RUN_TOTAL_RUNS={result['DRY_RUN_TOTAL_RUNS']}")
    print(f"EXPECTED_TOTAL_RUNS={result['EXPECTED_TOTAL_RUNS']}")
    print(f"SLICE_MANIFEST={result['slice_manifest']}")
    print(f"OUTPUT_ROOT={result['output_root']}")
    if int(result["DRY_RUN_TOTAL_RUNS"]) != UNIFORM_3E4_RUNS:
        print("READY_FOR_FULL_RUN=NO")
        return 2
    print(f"READY_FOR_FULL_RUN={result['READY_FOR_FULL_RUN']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
