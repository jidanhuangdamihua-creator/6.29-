#!/usr/bin/env python3
"""Formal corrected SITA for the three paper targets only (D4/D5/D6).

45 cells = 3 datasets × 3 horizons × 5 seeds.
Does not run D1. Does not touch formal freeze trees or 60/10-target outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sita_corrected_60cell import PROJECT_ROOT, _print_plan, run_execute
from src.transfer_methods.ms_katl_corrected_sita_60cell import (
    CorrectedSITAContractError,
    assert_formal_target_authority,
    assert_output_isolation,
    build_digests,
    enumerate_cells,
    load_manifest,
    resolve_output_root,
    validate_authority_against_runtime,
)

PAPER_DATASETS = ("D4", "D5", "D6")
DEFAULT_OUT = Path("outputs") / "paper3target_sita_isolated_20261005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-target SITA full matrix for D4/D5/D6 only.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--source-selection-mode",
        default="aligned_30d",
        choices=("aligned_30d", "historical_window_30d"),
    )
    parser.add_argument("--dataset", choices=("D4", "D5", "D6", "all"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(PROJECT_ROOT)
        assert_formal_target_authority(manifest)
        validate_authority_against_runtime(manifest)
        cells = enumerate_cells(manifest)
        wanted = PAPER_DATASETS if args.dataset == "all" else (args.dataset,)
        cells = tuple(cell for cell in cells if cell.dataset in wanted)
        if not cells:
            raise CorrectedSITAContractError(f"no cells for datasets={wanted}")
        digests = build_digests(manifest, cells, PROJECT_ROOT)
        # reuse corrected runner helpers by shaping args
        class _Args:
            execute = args.execute
            output_root = None if args.resume is not None else args.output_root
            resume = args.resume
            source_selection_mode = args.source_selection_mode
            only_cell = None

        ns = _Args()
        output_root = resolve_output_root(PROJECT_ROOT, ns.resume or ns.output_root)
        assert_output_isolation(output_root, PROJECT_ROOT)
        print(f"PAPER_DATASETS={','.join(wanted)}")
        print(f"N_CELLS={len(cells)}")
        target_pairs = sorted({(c.dataset, c.target) for c in cells}, key=lambda item: PAPER_DATASETS.index(item[0]) if item[0] in PAPER_DATASETS else 99)
        print("TARGETS=" + ";".join(f"{dataset}:{target}" for dataset, target in target_pairs))
        if not args.execute:
            _print_plan(manifest, cells, digests, output_root)
            return 0
        return run_execute(manifest, cells, digests, ns)
    except CorrectedSITAContractError as exc:
        print(str(exc), file=sys.stderr)
        print("TRAINING_CALLED=NO", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
