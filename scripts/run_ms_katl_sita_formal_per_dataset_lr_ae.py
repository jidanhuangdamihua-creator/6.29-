#!/usr/bin/env python3
"""Run the formal corrected SITA A–E slice under per-dataset target LRs.

The retained uniform-3e-4 slice is never used as the formal output root. D5
is reused only after the exact read-only parity audit; otherwise all 300
arm-runs are executed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    AETrainingBlocked,
)
from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
    FORMAL_PER_DATASET_LR_MATRIX_KIND,
    FormalPerDatasetLRContractError,
    run_formal_per_dataset_lr_dry_run,
    run_formal_per_dataset_lr_execute,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal corrected SITA A–E ablation under the frozen per-dataset target-LR protocol."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resolve-real-cells", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if sum(bool(item) for item in (args.dry_run, args.execute, args.plan_only)) > 1:
        print("pass at most one of --dry-run, --execute, or --plan-only")
        print("TRAINING_STARTED=NO")
        return 2
    if args.output_root is not None and args.resume is not None:
        print("provide either --output-root or --resume, not both")
        print("TRAINING_STARTED=NO")
        return 2
    if args.execute:
        try:
            result = run_formal_per_dataset_lr_execute(
                output_dir=args.output_root,
                resume_root=args.resume,
                project_root=PROJECT_ROOT,
            )
        except AETrainingBlocked as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            print("FORMAL_TRAINING_BLOCKED=YES")
            print(f"AE_FORMAL_EXECUTION_ENABLED={'YES' if AE_FORMAL_EXECUTION_ENABLED else 'NO'}")
            return 2
        except FormalPerDatasetLRContractError as exc:
            print(str(exc))
            print("FORMAL_COMPLETION_FAILED=YES")
            print("TRAINING_STARTED=NO")
            return 2
        print(f"OUTPUT_ROOT={result.get('output_root')}")
        print(f"MATRIX_KIND={result.get('MATRIX_KIND', FORMAL_PER_DATASET_LR_MATRIX_KIND)}")
        print(f"EXPECTED_RUNS={result.get('EXPECTED_RUNS')}")
        print(f"PAIR_COMPLETE={result.get('PAIR_COMPLETE')}")
        print(f"PAIR_INCOMPLETE={result.get('PAIR_INCOMPLETE')}")
        print(f"D5_REUSED={result.get('D5_REUSED')}")
        print(f"TRAINING_STARTED={result.get('TRAINING_STARTED')}")
        if result.get("results_csv"):
            print(f"RESULTS_CSV={result['results_csv']}")
        if int(result.get("PAIR_INCOMPLETE") or 0) > 0:
            print("FORMAL_COMPLETION_FAILED=YES")
            return 2
        print("FORMAL_COMPLETION_FAILED=NO")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        return 0

    result = run_formal_per_dataset_lr_dry_run(
        project_root=PROJECT_ROOT,
        resolve_real_cells=args.resolve_real_cells,
    )
    print(f"MATRIX_KIND={result['MATRIX_KIND']}")
    print(f"EXPECTED_TOTAL_RUNS={result['EXPECTED_TOTAL_RUNS']}")
    print(f"EXPECTED_CANONICAL_CELLS={result['EXPECTED_CANONICAL_CELLS']}")
    print(f"D1_LR={result['D1_LR']}")
    print(f"D4_LR={result['D4_LR']}")
    print(f"D5_LR={result['D5_LR']}")
    print(f"D6_LR={result['D6_LR']}")
    print(f"ARMS={','.join(AE_ARMS)}")
    print(f"D5_REUSE_AUDIT={result['D5_REUSE_AUDIT']['status']}")
    print(f"D5_REUSE_MISMATCHES={len(result['D5_REUSE_AUDIT']['mismatches'])}")
    print("TRAINING_STARTED=NO")
    print("READY_FOR_FULL_RUN=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
