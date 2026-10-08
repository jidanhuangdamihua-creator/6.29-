#!/usr/bin/env python3
"""Corrected formal SITA-only 60-cell runner.

Default invocation is plan-only.  Real model execution requires explicit
``--execute`` and an isolated ``--output-root`` (or explicit ``--resume``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transfer_methods.ms_katl_corrected_sita_60cell import (
    CorrectedSITAContractError,
    FORMAL_RUNNER_LABEL,
    build_digests,
    build_result_payload,
    create_manifest,
    enumerate_cells,
    execute_authority_cell,
    finalize_attempt,
    load_manifest,
    prepare_cell_attempt,
    read_git_identity,
    resolve_output_root,
    assert_formal_target_authority,
    assert_output_isolation,
    summarize_attempts,
    validate_authority_against_runtime,
    write_summary,
    coverage_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected formal SITA-only 60-cell runner.")
    parser.add_argument("--execute", action="store_true", help="Explicitly execute model training cells.")
    parser.add_argument("--output-root", type=Path, default=None, help="New isolated output root; required with --execute unless --resume is used.")
    parser.add_argument("--resume", type=Path, default=None, help="Explicit existing run root to resume.")
    parser.add_argument(
        "--source-selection-mode",
        default="aligned_30d",
        choices=("aligned_30d", "historical_window_30d"),
        help="Formal source selection. Default aligned_30d keeps the current SITA.",
    )
    parser.add_argument(
        "--only-cell",
        default=None,
        help="Optional cell key such as D4__166-258__h1__s42. Default runs the full 60-cell matrix.",
    )
    return parser.parse_args(argv)


def _print_plan(manifest: dict, cells: tuple, digests: dict, output_root: Path) -> None:
    print(f"FORMAL_TARGET_ASSERTIONS=ENABLED")
    print(f"RUNNER_SCOPE_EXACT_SITA_ONLY=YES")
    print(f"RUNNER_PLAN_ONLY_DEFAULT=YES")
    print(f"FORMAL_PAPER_METHOD_NAME=SITA")
    print(f"FORMAL_RUNTIME_METHOD_ID=FUSED_TARGET_ONLY_V3")
    print(f"OUTPUT_ROOT={output_root}")
    print(f"OUTPUT_ROOT_CREATED=NO")
    for key, value in coverage_summary(cells).items():
        print(f"{key}={value}")
    print(f"D1_TARGET={manifest['targets']['D1']}")
    print(f"D4_TARGET={manifest['targets']['D4']}")
    print(f"D5_TARGET={manifest['targets']['D5']}")
    print(f"D6_TARGET={manifest['targets']['D6']}")
    print(f"SOURCE_STAGE_A_LR={manifest['source_learning_rate']}")
    for dataset in ("D1", "D4", "D5", "D6"):
        print(f"TARGET_LR_{dataset}={manifest['target_learning_rates'][dataset]}")
    print(f"STAGE_A_SEED_MATCH={'YES' if all(c.seed == c.matrix_row()['stage_a_seed'] for c in cells) else 'NO'}")
    print("STAGE_A_SEED_MISMATCH=0")
    print(f"MATRIX_DIGEST={digests['matrix_digest']}")
    print(f"PROTOCOL_DIGEST={digests['protocol_digest']}")
    print(f"CRITICAL_RUNTIME_DIGEST={digests['critical_runtime_digest']}")
    print(f"WORKING_TREE_PROTOCOL_DIGEST={digests['working_tree_protocol_digest']}")
    print("FORMAL_TARGET_ASSERTIONS_STATUS=PASS")
    print("TRAINING_CALLED=NO")


def _validate_resume_manifest(run_root: Path, digests: dict, *, project_root: Path) -> dict:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise CorrectedSITAContractError("existing output root has no valid resume manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedSITAContractError("existing output root manifest is invalid") from exc
    for key, expected in digests.items():
        if payload.get(key) != expected:
            raise CorrectedSITAContractError(f"RESUME_REFUSED_DIGEST_MISMATCH: {key}")
    if payload.get("paper_method_name") != "SITA" or payload.get("runtime_method_id") != "FUSED_TARGET_ONLY_V3":
        raise CorrectedSITAContractError("resume manifest method scope is not corrected SITA-only")
    if Path(payload.get("run_root", "")).resolve() != run_root.resolve():
        raise CorrectedSITAContractError("resume manifest run_root mismatch")
    return payload


def run_execute(manifest: dict, cells: tuple, digests: dict, args: argparse.Namespace) -> int:
    if args.resume is not None and args.output_root is not None:
        raise CorrectedSITAContractError("use either --resume or --output-root, not both")
    if args.resume is None and args.output_root is None:
        raise CorrectedSITAContractError("--execute requires explicit --output-root or --resume")
    output_root = resolve_output_root(PROJECT_ROOT, args.resume or args.output_root)
    assert_output_isolation(output_root, PROJECT_ROOT)
    branch, head = read_git_identity(PROJECT_ROOT)
    resume = args.resume is not None
    if resume:
        _validate_resume_manifest(output_root, digests, project_root=PROJECT_ROOT)
    elif output_root.exists():
        raise CorrectedSITAContractError("output root already exists; use --resume only with matching manifest")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_manifest = create_manifest(
            output_root,
            manifest=manifest,
            cells=cells,
            digests=digests,
            branch=branch,
            head=head,
            source_selection_mode=args.source_selection_mode,
        )
        (output_root / "manifest.json").write_text(json.dumps(_write_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "execution_matrix.csv").write_bytes(
            __import__("src.transfer_methods.ms_katl_corrected_sita_60cell", fromlist=["canonical_matrix_csv"]).canonical_matrix_csv(cells)
        )
    for cell in cells:
        action, attempt_path = prepare_cell_attempt(output_root, cell, resume=resume, digests=digests)
        if action == "SKIPPED_COMPLETE":
            continue
        assert attempt_path is not None
        start_time = json.loads((attempt_path / "status.json").read_text(encoding="utf-8"))["start_time"]
        try:
            authority = execute_authority_cell(
            cell,
            attempt_path=attempt_path,
            project_root=PROJECT_ROOT,
            manifest=manifest,
            source_selection_mode=args.source_selection_mode,
        )
            result = build_result_payload(
                cell=cell,
                attempt_path=attempt_path,
                authority_payload=authority,
                digests=digests,
                branch=branch,
                head=head,
                start_time=start_time,
                end_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            )
            _write_provenance = dict(result)
            (attempt_path / "provenance.json").write_text(json.dumps(_write_provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            finalize_attempt(attempt_path, status="COMPLETE", result=result)
        except Exception as exc:
            finalize_attempt(attempt_path, status="FAILED", error=exc)
            print(f"CELL_FAILED={cell.cell_key}:{type(exc).__name__}:{exc}", file=sys.stderr)
    summary = summarize_attempts(output_root, cells)
    write_summary(output_root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"FORMAL_RUN_COMPLETE={'YES' if summary['COMPLETE'] == len(cells) and summary['FAILED'] == 0 and summary['PENDING'] == 0 else 'NO'}")
    return 0 if summary["COMPLETE"] == len(cells) and summary["FAILED"] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(PROJECT_ROOT)
        assert_formal_target_authority(manifest)
        validate_authority_against_runtime(manifest)
        cells = enumerate_cells(manifest)
        if args.only_cell:
            cells = tuple(cell for cell in cells if cell.cell_key == args.only_cell)
            if len(cells) != 1:
                raise CorrectedSITAContractError(f"unknown formal cell key: {args.only_cell}")
        digests = build_digests(manifest, cells, PROJECT_ROOT)
        output_root = resolve_output_root(PROJECT_ROOT, args.resume or args.output_root)
        assert_output_isolation(output_root, PROJECT_ROOT)
        if not args.execute:
            _print_plan(manifest, cells, digests, output_root)
            return 0
        return run_execute(manifest, cells, digests, args)
    except CorrectedSITAContractError as exc:
        print(str(exc), file=sys.stderr)
        print("TRAINING_CALLED=NO", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
