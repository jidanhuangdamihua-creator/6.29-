#!/usr/bin/env python3
"""Arm F Random-3 and Arm G Bottom-3 source-selection ablation.

Does not modify frozen Arm A–E or the uniform 3e-4 cell bundles.

Usage:
    python scripts/run_ms_katl_sita_source_selection_ablation.py --dry-run
    python scripts/run_ms_katl_sita_source_selection_ablation.py --build-manifest
    AE_FORMAL_TRAINING_UNLOCK=YES python scripts/run_ms_katl_sita_source_selection_ablation.py --execute --stage stage1
    AE_FORMAL_TRAINING_UNLOCK=YES python scripts/run_ms_katl_sita_source_selection_ablation.py --execute --stage full
    python scripts/run_ms_katl_sita_source_selection_ablation.py --analyze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import AETrainingBlocked
from src.transfer_methods.ms_katl_sita_ablation_3e4 import UNLOCK_ENV
from src.transfer_methods.ms_katl_sita_source_selection_ablation import (
    ARM_F,
    ARM_G,
    SOURCE_SET_IDS,
    STAGE1_SOURCE_SETS,
    SourceSelectionAblationError,
    analyze_results,
    assert_output_isolated,
    default_output_dir,
    freeze_source_selection,
    run_dry_run,
    run_execute,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SITA Arm F Random-3 / Arm G Bottom-3 source-selection ablation."
    )
    parser.add_argument("--dry-run", "--preflight", action="store_true", dest="dry_run")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("stage1", "full"),
        default="stage1",
        help="stage1 = Arm F R1–R3 + Arm G; full = Arm F R1–R5 + Arm G.",
    )
    parser.add_argument(
        "--arms",
        default="F,G",
        help="Comma-separated arms to run. Default: F,G",
    )
    parser.add_argument(
        "--source-sets",
        default="",
        help="Comma-separated Random-3 sets, e.g. R1,R2,R3. Default depends on --stage.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Retrain even if an OK run manifest already exists.",
    )
    return parser.parse_args(argv)


def _resolve_output_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        return default_output_dir(PROJECT_ROOT)
    return assert_output_isolated(requested, project_root=PROJECT_ROOT)


def _parse_arms(raw: str) -> Tuple[str, ...]:
    arms = tuple(part.strip().upper() for part in str(raw).split(",") if part.strip())
    unknown = [arm for arm in arms if arm not in {ARM_F, ARM_G}]
    if unknown:
        raise SourceSelectionAblationError(f"unknown arms {unknown}")
    if not arms:
        raise SourceSelectionAblationError("at least one arm is required")
    return arms


def _parse_source_sets(raw: str, stage: str) -> Tuple[str, ...]:
    if str(raw).strip():
        sets = tuple(part.strip().upper() for part in str(raw).split(",") if part.strip())
    elif stage == "full":
        sets = SOURCE_SET_IDS
    else:
        sets = STAGE1_SOURCE_SETS
    unknown = [item for item in sets if item not in SOURCE_SET_IDS]
    if unknown:
        raise SourceSelectionAblationError(f"unknown source sets {unknown}")
    return sets


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = _resolve_output_dir(args.output_dir)
        arms = _parse_arms(args.arms)
        source_sets = _parse_source_sets(args.source_sets, args.stage)
    except SourceSelectionAblationError as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("READY_FOR_FULL_RUN=NO")
        return 2
    selected = 0
    if args.dry_run:
        selected += 1
    if args.build_manifest:
        selected += 1
    if args.execute:
        selected += 1
    if args.analyze:
        selected += 1
    if selected != 1:
        print("pass exactly one of --dry-run, --build-manifest, --execute, --analyze")
        print("TRAINING_STARTED=NO")
        return 2
    if args.build_manifest:
        try:
            result = freeze_source_selection(project_root=PROJECT_ROOT)
        except SourceSelectionAblationError as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            print("READY_FOR_FULL_RUN=NO")
            return 2
        print("SOURCE_SELECTION_MANIFEST_BUILT=YES")
        print(f"SOURCE_SELECTION_MANIFEST={result['csv']}")
        print(f"RANKING_JSON={result['ranking_json']}")
        print(f"CONFIG_MANIFEST={result['config_manifest']}")
        print("TRAINING_STARTED=NO")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        return 0
    if args.analyze:
        try:
            summary = analyze_results(output_dir=output_dir, project_root=PROJECT_ROOT)
        except SourceSelectionAblationError as exc:
            print(str(exc))
            return 2
        print(f"N_COMPLETE_CONDITIONS={summary['n_complete_conditions']}")
        print(f"TOP3_BEATS_RANDOM_SETS_RMSE={summary['top3_beats_random_sets_RMSE']}")
        print(f"CORRECTION_FAMILY={summary['correction_family']}")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        return 0
    if args.execute:
        try:
            result = run_execute(
                output_dir=output_dir,
                project_root=PROJECT_ROOT,
                arms=arms,
                source_sets=source_sets,
                skip_existing=not args.no_skip_existing,
            )
        except AETrainingBlocked as exc:
            print(str(exc))
            print("TRAINING_STARTED=NO")
            print("FORMAL_TRAINING_BLOCKED=YES")
            print("READY_FOR_FULL_RUN=NO")
            return 2
        except SourceSelectionAblationError as exc:
            print(str(exc))
            print("FORMAL_COMPLETION_FAILED=YES")
            print("READY_FOR_FULL_RUN=NO")
            return 2
        print(f"OUTPUT_ROOT={result['output_root']}")
        print(f"OK_RUNS={result['OK_RUNS']}")
        print(f"FAILED_RUNS={result['FAILED_RUNS']}")
        print(f"EXPECTED_RUNS={result['EXPECTED_RUNS']}")
        print(f"TRAINING_STARTED={result['TRAINING_STARTED']}")
        print("FORMAL_RESULTS_OVERWRITTEN=NO")
        if int(result["FAILED_RUNS"]) > 0 or int(result["OK_RUNS"]) != int(result["EXPECTED_RUNS"]):
            print("FORMAL_COMPLETION_FAILED=YES")
            return 2
        print("FORMAL_COMPLETION_FAILED=NO")
        return 0
    try:
        result = run_dry_run(
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
            arms=arms,
            source_sets=source_sets,
        )
    except SourceSelectionAblationError as exc:
        print(str(exc))
        print("TRAINING_STARTED=NO")
        print("READY_FOR_FULL_RUN=NO")
        return 2
    print(result["text"], end="" if str(result["text"]).endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
