#!/usr/bin/env python3
"""SITA mechanism diagnostic rerun.

Plan-only by default. `--run` starts the selected diagnostic execution loop into a
timestamped diagnostic root. It does not write frozen A–E results.

    PYTHONPATH="$PWD" .venv/bin/python scripts/run_mechanism_diagnostic_rerun.py --plan-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SITA mechanism diagnostic 42-cell rerun.")
    parser.add_argument(
        "--matrix",
        choices=("diagnostic-42", "formal-300"),
        default="diagnostic-42",
        help="Diagnostic universe. Default preserves the existing 42-cell behavior.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Enumerate the selected matrix and audit wiring. No training. Does not create a run directory.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Create a timestamped run root and execute the selected diagnostic loop.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory. Required with --resume. Forbidden for a new --run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run. Must be paired with --run-dir.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Alias of --run. Blocked until plan-only reports EXECUTE_READY=YES.",
    )
    parser.add_argument(
        "--only-cell",
        type=str,
        default=None,
        help=(
            "Run one cell from the selected official matrix, e.g. D1-H1-seed42-B."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from src.transfer_methods.sita_mechanism_diagnostic_rerun import (
        format_plan,
        matrix_digest_for_cells,
        mechanism_rerun_config_digest,
        plan_mechanism_diagnostic_rerun,
        resolve_diagnostic_only_cell,
    )

    want_run = bool(args.run or args.execute)
    only_cell = str(args.only_cell).strip() if args.only_cell else ""
    if only_cell and not (want_run or args.resume or args.plan_only):
        print("usage: scripts/run_mechanism_diagnostic_rerun.py --run --only-cell D1-H1-seed42-B")
        return 2
    if args.plan_only or not (want_run or args.resume):
        if not args.plan_only and not want_run and not args.resume:
            print("usage: scripts/run_mechanism_diagnostic_rerun.py --plan-only")
            return 2
        report = plan_mechanism_diagnostic_rerun(args.matrix)
        sys.stdout.write(format_plan(report))
        if only_cell:
            cell = resolve_diagnostic_only_cell(only_cell, matrix_mode=args.matrix)
            print(f"ONLY_CELL={cell.cell_key}")
            if args.matrix == "diagnostic-42":
                print("DEFAULT_42_CELL_MATRIX_UNCHANGED=YES")
        return 0

    report = plan_mechanism_diagnostic_rerun(args.matrix)
    sys.stdout.write(format_plan(report))
    if not report.get("final_execute_ready", report["execute_ready"]):
        print("EXECUTE_BLOCKED: plan-only gaps remain. Frozen A–E results were not touched.")
        return 2

    from src.transfer_methods.ms_katl_ae_ablation import AETrainingBlocked, assert_formal_training_permitted
    from src.transfer_methods.sita_mechanism_diagnostic_execution import (
        mechanism_rerun_code_digest,
        run_mechanism_diagnostic_cell_loop,
    )
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        prepare_run_root,
        resolve_run_directory,
    )

    smoke_cells = None
    try:
        assert_formal_training_permitted(execute=True)
        smoke_cells = None
        allow_subset = False
        config_digest = str(report.get("config_digest") or mechanism_rerun_config_digest())
        matrix_digest = str(report["matrix_digest"])
        if only_cell:
            smoke = resolve_diagnostic_only_cell(only_cell, matrix_mode=args.matrix)
            smoke_cells = (smoke,)
            allow_subset = True
            config_digest = mechanism_rerun_config_digest(smoke_cells, matrix_mode=args.matrix)
            matrix_digest = matrix_digest_for_cells(smoke_cells)
        run_dir = resolve_run_directory(
            plan_only=False,
            resume=bool(args.resume),
            run_dir=args.run_dir,
            matrix_mode=args.matrix,
        )
        allowed_root = (PROJECT_ROOT / "outputs" / "mechanism_diagnostics").resolve()
        resolved_run_dir = Path(run_dir).resolve()
        if resolved_run_dir == allowed_root or allowed_root not in resolved_run_dir.parents:
            raise RuntimeError(
                "diagnostic outputs must stay under outputs/mechanism_diagnostics; "
                "frozen formal result roots are not valid run directories"
            )
        code_digest = mechanism_rerun_code_digest()
        prepare_run_root(
            run_dir,
            config_digest=config_digest,
            matrix_digest=matrix_digest,
            code_digest=code_digest,
            resume=bool(args.resume),
            matrix_mode=args.matrix,
        )
        loop = run_mechanism_diagnostic_cell_loop(
            run_dir,
            cells=smoke_cells,
            resume=bool(args.resume),
            config_digest=config_digest,
            matrix_digest=matrix_digest,
            code_digest=code_digest,
            allow_cell_subset=allow_subset,
            matrix_mode=args.matrix,
        )
    except AETrainingBlocked as exc:
        print(f"EXECUTE_BLOCKED: {exc}. Frozen A–E results were not touched.")
        return 2
    except Exception as exc:
        print(f"EXECUTE_BLOCKED: {exc}")
        return 2
    print(f"run_dir={loop['run_dir']}")
    print(f"CELL_LOOP_STARTED=YES")
    if only_cell:
        print(f"ONLY_CELL={smoke_cells[0].cell_key}")
        if args.matrix == "diagnostic-42":
            print("DEFAULT_42_CELL_MATRIX_UNCHANGED=YES")
    print(f"n_cells={loop['n_cells']}")
    print(f"n_ran={loop['n_ran']}")
    print(f"n_skipped={loop['n_skipped']}")
    print(f"n_failed={loop['n_failed']}")
    print(f"finalized={loop['finalized']}")
    return 0 if int(loop["n_failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
