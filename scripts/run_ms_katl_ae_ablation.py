#!/usr/bin/env python3
"""MS-KATL V3 A–E core mechanism ablation runner.

Level 1: --preflight  (contracts / matrix, no real-cell load required)
Level 2: --dry-run    (real D1–D6 routing, mocked Stage A, no training)
Level 3: --execute    (formal 270-run). Dual-locked: ENABLED + --execute + UNLOCK.

Usage:
    python scripts/run_ms_katl_ae_ablation.py --preflight
    python scripts/run_ms_katl_ae_ablation.py --dry-run
    python scripts/run_ms_katl_ae_ablation.py --formal-preflight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_FORMAL_EXECUTION_ENABLED,
    AEContractError,
    AETrainingBlocked,
    run_ae_formal_matrix,
    run_ae_preflight,
)
from src.transfer_methods.ms_katl_ae_execution import run_ae_dry_run
from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_preflight


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
        help="Level 3: formal 270-run. Requires AE_FORMAL_TRAINING_UNLOCK=YES and ENABLED=YES.",
    )
    parser.add_argument("--write-plan", type=Path, default=None, help="Write planned 270-run JSON.")
    parser.add_argument(
        "--skip-construction",
        action="store_true",
        help="Skip synthetic/fake model construction during preflight/dry-run.",
    )
    return parser, parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser, args = _parse_args(argv)
    enabled = "YES" if AE_FORMAL_EXECUTION_ENABLED else "NO"
    selected = [args.preflight, args.dry_run, args.formal_preflight, args.execute]
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
