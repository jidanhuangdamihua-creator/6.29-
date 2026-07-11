#!/usr/bin/env python3
"""Run and combine the formal 5-horizon x 5-seed protocol matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_SEEDS
from src.utils.result_validation import promote_complete_baseline_groups


@dataclass(frozen=True)
class MatrixTask:
    dataset: str
    scenario: str
    horizon: int
    seed: int
    output_dir: Path
    command: Tuple[str, ...]
    result_path: Path


def build_matrix_tasks(
    *,
    dataset: str,
    scenario: str,
    output_dir: Path,
) -> Tuple[MatrixTask, ...]:
    normalized = str(dataset).strip().lower()
    if normalized not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError("dataset must be d1 through d6")
    if scenario not in {"without", "with"}:
        raise ValueError("scenario must be without or with")
    dataset_id = int(normalized[1:])
    tasks = []
    for horizon in FORMAL_HORIZONS:
        for seed in FORMAL_SEEDS:
            cell_dir = Path(output_dir) / "cells" / f"h{horizon}_s{seed}"
            if dataset_id <= 3:
                command = (
                    sys.executable,
                    "scripts/run_full_paper_experiments.py",
                    "--only-dataset",
                    f"dataset{dataset_id}",
                    "--strict-paper-mode",
                    "--info-sharing",
                    scenario,
                    "--horizon",
                    str(horizon),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(cell_dir),
                )
            else:
                command = (
                    sys.executable,
                    f"scripts/run_d{dataset_id}_experiment.py",
                    "--info-sharing",
                    scenario,
                    "--horizon",
                    str(horizon),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(cell_dir),
                )
            result_path = (
                cell_dir
                / "results"
                / f"dataset{dataset_id}_{scenario}_results.csv"
            )
            tasks.append(
                MatrixTask(
                    normalized,
                    scenario,
                    horizon,
                    seed,
                    cell_dir,
                    command,
                    result_path,
                )
            )
    return tuple(tasks)


def combine_result_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("formal matrix produced no result frames")
    combined = pd.concat([frame.copy() for frame in frames], ignore_index=True, sort=False)
    promoted = promote_complete_baseline_groups(combined)
    incomplete = promoted["result_status"].astype(str) != "confirmed_baseline"
    if incomplete.any():
        counts = promoted.loc[incomplete, "result_status"].astype(str).value_counts().to_dict()
        raise ValueError(
            "formal matrix contains failed, invalid, legacy, or incomplete groups: "
            f"{counts}"
        )
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[f"d{i}" for i in range(1, 7)])
    parser.add_argument("--scenario", required=True, choices=("without", "with"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = build_matrix_tasks(
        dataset=args.dataset,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )
    if args.dry_run:
        for task in tasks:
            print(" ".join(task.command))
        return

    frames = []
    for index, task in enumerate(tasks, start=1):
        print(
            f"[{index}/{len(tasks)}] {task.dataset}/{task.scenario} "
            f"horizon={task.horizon} seed={task.seed}"
        )
        completed = subprocess.run(task.command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"matrix cell failed: horizon={task.horizon} seed={task.seed} "
                f"returncode={completed.returncode}"
            )
        if not task.result_path.is_file():
            raise FileNotFoundError(f"matrix cell result missing: {task.result_path}")
        frames.append(pd.read_csv(task.result_path))

    combined = combine_result_frames(frames)
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = int(args.dataset[1:])
    output = results_dir / f"dataset{dataset_id}_{args.scenario}_results.csv"
    combined.to_csv(output, index=False, encoding="utf-8")
    confirmed = int((combined["result_status"].astype(str) == "confirmed_baseline").sum())
    print(f"Saved {len(combined)} rows ({confirmed} confirmed) to {output}")


if __name__ == "__main__":
    main()
