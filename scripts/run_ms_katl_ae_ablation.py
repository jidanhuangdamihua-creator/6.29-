#!/usr/bin/env python3
"""MS-KATL V3 A–E core mechanism ablation runner.

Level 1: --preflight  (contracts / matrix, no real-cell load required)
Level 2: --dry-run    (real D1–D6 routing, mocked Stage A, no training)
Level 3: --execute    (formal 270-run). Dual-locked: ENABLED + --execute + UNLOCK.

Paper-candidate extra-target slice (does not mutate the frozen 270-run):
    python scripts/run_ms_katl_ae_ablation.py --dataset D4 --target 166/258
    AE_FORMAL_TRAINING_UNLOCK=YES python scripts/run_ms_katl_ae_ablation.py --execute --dataset D4 --target 166/258
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    EXTRA_TARGET_CANONICAL_CELLS,
    EXTRA_TARGET_MATRIX_KIND,
    EXTRA_TARGET_RUNS,
    HORIZONS,
    SEEDS,
    AEContractError,
    AETrainingBlocked,
    resolve_extra_target_lr,
    run_ae_formal_matrix,
    run_ae_preflight,
)
from src.transfer_methods.ms_katl_ae_execution import (
    enumerate_extra_target_cell_plans,
    enumerate_real_arm_plans,
    run_ae_dry_run,
)
from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_preflight
from src.transfer_methods.ms_katl_primary_target_lr_screening import parse_target_identity


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="MS-KATL V3 A–E controlled ablation (fail-closed).")
    parser.add_argument("--preflight", action="store_true", help="Level 1: contracts and matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Level 2: real cell routing, no training.")
    parser.add_argument(
        "--formal-preflight",
        action="store_true",
        help="Check frozen V3 authorities and launch gates. Does not train.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Explicit run root to resume. Never auto-discovers latest.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Level 3: formal 270-run or extra-target slice. Requires AE_FORMAL_TRAINING_UNLOCK=YES and ENABLED=YES.",
    )
    parser.add_argument("--dataset", default=None, help="Extra-target dataset, e.g. D4. Requires --target.")
    parser.add_argument("--target", default=None, help="Extra-target identity, e.g. 166/258. Requires --dataset.")
    parser.add_argument("--write-plan", type=Path, default=None, help="Write planned 270-run JSON.")
    parser.add_argument(
        "--skip-construction",
        action="store_true",
        help="Skip synthetic/fake model construction during preflight/dry-run.",
    )
    return parser, parser.parse_args(argv)


def _resolve_extra_target(args: argparse.Namespace) -> Optional[Tuple[str, str]]:
    if args.dataset is None and args.target is None:
        return None
    if args.dataset is None or args.target is None:
        raise AEContractError("extra-target A–E requires both --dataset and --target")
    parsed = parse_target_identity(args.dataset, args.target)
    return parsed["dataset"], parsed["target_identity"]


def _print_extra_target_plan(dataset: str, target: str) -> None:
    lr_value, lr_label = resolve_extra_target_lr(dataset, target)
    print("MATRIX_KIND=EXTRA_TARGET_SLICE")
    print(f"DATASET={dataset}")
    print(f"TARGET={target}")
    print(f"SELECTED_LR={lr_label}")
    print(f"SELECTED_LR_VALUE={lr_value}")
    print(f"HORIZONS={list(HORIZONS)}")
    print(f"SEEDS={list(SEEDS)}")
    print(f"ARMS={list(AE_ARMS)}")
    print(f"CANONICAL_CELLS={EXTRA_TARGET_CANONICAL_CELLS}")
    print(f"EXPECTED_RUNS={EXTRA_TARGET_RUNS}")
    print("SCENARIO=without")
    print("LR_POLICY=DATASET_INHERITED_V3_LR")
    print("FROZEN_270_UNTOUCHED=YES")
    print("FROZEN_PRIMARY_D4_TARGET=166/311")
    print("TRAINING_STARTED=NO")


def _run_extra_target_execute(dataset: str, target: str, resume_root: Optional[Path]) -> dict:
    cells = enumerate_extra_target_cell_plans(dataset, target, project_root=PROJECT_ROOT)
    arm_plans = enumerate_real_arm_plans(cells, expected_runs=EXTRA_TARGET_RUNS)
    lr_label = cells[0].selected_lr_label
    print("MATRIX_KIND=EXTRA_TARGET_SLICE")
    print(f"DATASET={dataset}")
    print(f"TARGET={target}")
    print(f"SELECTED_LR={lr_label}")
    print(f"CANONICAL_CELLS={len(cells)}")
    print(f"EXPECTED_RUNS={len(arm_plans)}")
    print("FROZEN_270_UNTOUCHED=YES")
    print("ACTUAL_EXECUTION_REQUESTED=YES")
    return run_ae_formal_matrix(
        execute=True,
        resume_root=resume_root,
        cells=cells,
        matrix_kind=EXTRA_TARGET_MATRIX_KIND,
    )


def main(argv: list[str] | None = None) -> int:
    parser, args = _parse_args(argv)
    enabled = "YES" if AE_FORMAL_EXECUTION_ENABLED else "NO"
    try:
        extra = _resolve_extra_target(args)
    except AEContractError as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        return 2
    selected = [args.preflight, args.dry_run, args.formal_preflight, args.execute]
    if extra is not None:
        if args.dry_run:
            print("extra-target A–E does not use --dry-run; pass --execute or omit mode flags")
            print("TRAINING_STARTED=NO")
            return 2
        if args.formal_preflight:
            print("extra-target A–E does not use --formal-preflight; frozen 270 preflight is separate")
            print("TRAINING_STARTED=NO")
            return 2
        if args.execute:
            try:
                result = _run_extra_target_execute(extra[0], extra[1], args.resume)
            except AETrainingBlocked as exc:
                print(str(exc))
                print("TRAINING_STARTED=NO")
                print("FORMAL_TRAINING_BLOCKED=YES")
                print("FORMAL_EXECUTION_BLOCKED=YES")
                print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
                print("ACTUAL_EXECUTION_REQUESTED=YES")
                return 2
            except AEContractError as exc:
                print(str(exc))
                print("FORMAL_COMPLETION_FAILED=YES")
                print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
                print("ACTUAL_EXECUTION_REQUESTED=YES")
                return 2
            print(f"OUTPUT_ROOT={result.get('output_root')}")
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
            return 0
        _print_extra_target_plan(extra[0], extra[1])
        print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
        print("ACTUAL_EXECUTION_REQUESTED=NO")
        return 0
    if not any(selected) and args.write_plan is None:
        parser.print_help()
        print("\nEXPLICIT_MODE_REQUIRED=YES")
        print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
        print("ACTUAL_EXECUTION_REQUESTED=NO")
        print("TRAINING_STARTED=NO")
        return 0
    if args.execute:
        try:
            result = run_ae_formal_matrix(execute=True, resume_root=args.resume)
        except AETrainingBlocked as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            print("FORMAL_TRAINING_BLOCKED=YES")
            print("FORMAL_EXECUTION_BLOCKED=YES")
            print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
            print("ACTUAL_EXECUTION_REQUESTED=YES")
            return 2
        except AEContractError as exc:
            print(str(exc))
            print("FORMAL_COMPLETION_FAILED=YES")
            print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
            print("ACTUAL_EXECUTION_REQUESTED=YES")
            return 2
        text = result.get("text")
        if text:
            print(text, end="" if str(text).endswith("\n") else "\n")
        pair_incomplete = int(result.get("PAIR_INCOMPLETE") or 0)
        if pair_incomplete > 0:
            print(f"PAIR_COMPLETE={result.get('PAIR_COMPLETE')}")
            print(f"PAIR_INCOMPLETE={pair_incomplete}")
            print("FORMAL_COMPLETION_FAILED=YES")
            return 2
        return 0
    if args.formal_preflight:
        result = run_ae_formal_preflight(project_root=PROJECT_ROOT)
        print(result["text"], end="")
        return 0 if result["FORMAL_PREFLIGHT"] == "PASS" else 2
    if args.dry_run:
        result = run_ae_dry_run(
            project_root=PROJECT_ROOT,
            write_artifacts=True,
            include_construction=not args.skip_construction,
        )
        print(result["text"], end="")
        print(f"AE_CONTRACT_READY={result['AE_CONTRACT_READY']}")
        print(f"AE_EXECUTION_WIRING_READY={result['AE_EXECUTION_WIRING_READY']}")
        print(f"READY_FOR_FORMAL_AE_EXECUTION_ENABLE={result['READY_FOR_FORMAL_AE_EXECUTION_ENABLE']}")
        print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
        print("ACTUAL_EXECUTION_REQUESTED=NO")
        print(f"TRAINING_STARTED={result['TRAINING_STARTED']}")
        return 0
    result = run_ae_preflight(
        project_root=PROJECT_ROOT,
        write_plan_path=args.write_plan,
        include_construction_contracts=not args.skip_construction,
    )
    print(result["text"], end="")
    print("AE_CONTRACT_READY=YES")
    print(f"AE_FORMAL_EXECUTION_ENABLED={enabled}")
    print("ACTUAL_EXECUTION_REQUESTED=NO")
    print(f"TRAINING_STARTED={result['TRAINING_STARTED']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
