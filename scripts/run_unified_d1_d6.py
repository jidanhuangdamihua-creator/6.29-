#!/usr/bin/env python3
"""Unified D1-D6 runner.

Design notes from the referenced scripts:
1. D1-D3 source count k is controlled by paper protocol/strict mode, not a
   direct CLI argument. Passing --strict-paper-mode makes multi-source methods
   use top-k=3; No-TL and SS-TL keep their existing semantics.
2. D1-D3 lr/epochs/dropout/clipnorm are injected by FORMAL_* constants through
   _apply_formal_config_overrides() in scripts/run_full_paper_experiments.py.
   D1-D3 seed is not injectable from that runner.
3. scripts/run_full_paper_experiments.py has no --config argument and no
   parameter to restrict INFO_SHARING_SCENARIOS to just one scenario.
4. D4-D6 argparse exposes only info_sharing, smoke, target_limit, and
   source_limit. lr/epochs/source_count/random_state live in each runner's
   config dict; dropout/clipnorm come from model defaults.
5. D4-D6 run_dir format is:
   outputs/runs/{YYYYmmdd_HHMMSS}_D{dataset_id}_{source_history_days}d_{info_sharing}
6. scripts/aggregate_d1_d6_results.py uses hardcoded SOURCE_CSVS, so this
   runner reports this run's CSVs but does not call the aggregate script.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.run_utils import create_run_dir, reserve_new_output_dir


RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
UNIFIED_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
UNIFIED_RUN_DIR = RUNS_DIR / UNIFIED_RUN_ID

FIXED_LR = 1e-4
FIXED_EPOCHS = 50
FIXED_CLIPNORM = None
FIXED_DROPOUT = 0.1
FIXED_K = 3
FIXED_SEED = 42

VALID_DATASETS = ("d1", "d2", "d3", "d4", "d5", "d6")
D1_D3_DATASET_ARGS = {
    "d1": "dataset1",
    "d2": "dataset2",
    "d3": "dataset3",
}
D4_D6_SCENARIOS = ("without", "with")


@dataclass(frozen=True)
class Task:
    dataset_token: str
    dataset_id: int
    label: str
    scenario: str
    cmd: List[str]
    config_check: str
    result_filename: str
    expected_result_path: Optional[Path] = None
    result_paths: List[Path] = field(default_factory=list)
    returncode: Optional[int] = None
    elapsed_seconds: Optional[float] = None


def _split_tokens(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return list(VALID_DATASETS)

    tokens: List[str] = []
    for value in values:
        for part in str(value).split(","):
            token = part.strip().lower()
            if token:
                tokens.append(token)
    return tokens or list(VALID_DATASETS)


def expand_only_tokens(values: Optional[Iterable[str]]) -> List[str]:
    requested = _split_tokens(values)
    unknown = [token for token in requested if token not in VALID_DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset id(s): {unknown}. Valid values: {list(VALID_DATASETS)}"
        )

    requested_set = set(requested)
    return [token for token in VALID_DATASETS if token in requested_set]


def _fixed_values_text() -> str:
    return (
        f"lr={FIXED_LR:g} epochs={FIXED_EPOCHS} k={FIXED_K} "
        f"seed={FIXED_SEED} dropout={FIXED_DROPOUT:g} clipnorm={FIXED_CLIPNORM}"
    )


def _d1_d3_config_check() -> str:
    return (
        "[CONFIG CHECK] "
        f"{_fixed_values_text()} - "
        "D1-D3 use FORMAL_* overrides for lr/epochs/dropout/clipnorm; "
        "--strict-paper-mode enforces k=3 for multi-source methods; "
        "seed=42 is noted but not injectable in run_full_paper_experiments.py."
    )


def _d4_d6_config_check() -> str:
    return (
        "[CONFIG CHECK] "
        f"{_fixed_values_text()} - "
        "D4-D6 values come from each runner config/model defaults; "
        "no hyperparameter CLI injection is available or needed."
    )


def _build_d1_d3_task(
    dataset_token: str,
    run_dir: Path,
    info_sharing: Optional[str] = None,
) -> Task:
    dataset_id = int(dataset_token[1:])
    dataset_arg = D1_D3_DATASET_ARGS[dataset_token]
    cmd = [
        sys.executable,
        "scripts/run_full_paper_experiments.py",
        "--only-dataset",
        dataset_arg,
        "--strict-paper-mode",
        "--output-dir",
        str(run_dir),
    ]
    if info_sharing is None:
        label = f"D{dataset_id}"
        scenario = "with+without"
        result_filename = f"dataset{dataset_id}_results.csv"
    else:
        cmd.extend(["--info-sharing", info_sharing])
        label = f"D{dataset_id}-{info_sharing}"
        scenario = info_sharing
        result_filename = f"dataset{dataset_id}_{info_sharing}_results.csv"
    return Task(
        dataset_token=dataset_token,
        dataset_id=dataset_id,
        label=label,
        scenario=scenario,
        cmd=cmd,
        config_check=_d1_d3_config_check(),
        result_filename=result_filename,
        expected_result_path=run_dir / "results" / result_filename,
    )


def _build_d4_d6_task(dataset_token: str, scenario: str, smoke: bool, run_dir: Path) -> Task:
    dataset_id = int(dataset_token[1:])
    cmd = [
        sys.executable,
        f"scripts/run_d{dataset_id}_experiment.py",
        "--info-sharing",
        scenario,
        "--output-dir",
        str(run_dir),
    ]
    if smoke:
        cmd.append("--smoke")
    result_filename = f"dataset{dataset_id}_{scenario}_results.csv"
    return Task(
        dataset_token=dataset_token,
        dataset_id=dataset_id,
        label=f"D{dataset_id}-{scenario}",
        scenario=scenario,
        cmd=cmd,
        config_check=_d4_d6_config_check(),
        result_filename=result_filename,
        expected_result_path=run_dir / "results" / result_filename,
    )


def build_tasks(
    only: Optional[Iterable[str]],
    smoke: bool,
    run_dir: Optional[Path] = None,
    info_sharing: Optional[str] = None,
) -> List[Task]:
    run_dir = UNIFIED_RUN_DIR if run_dir is None else Path(run_dir)
    if info_sharing is not None and info_sharing not in D4_D6_SCENARIOS:
        raise ValueError("--info-sharing must be one of: without, with")
    tasks: List[Task] = []
    for dataset_token in expand_only_tokens(only):
        scenarios = (info_sharing,) if info_sharing is not None else D4_D6_SCENARIOS
        for scenario in scenarios:
            task_run_dir = run_dir / f"{dataset_token}_{scenario}"
            if dataset_token in D1_D3_DATASET_ARGS:
                tasks.append(_build_d1_d3_task(dataset_token, task_run_dir, info_sharing=scenario))
            else:
                tasks.append(_build_d4_d6_task(dataset_token, scenario, smoke=smoke, run_dir=task_run_dir))
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified D1-D6 full experiments.")
    parser.add_argument(
        "--only",
        action="append",
        help="Dataset id(s), comma-separated or repeated. Valid values: d1,d2,d3,d4,d5,d6.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass --smoke to selected D4-D6 runners only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered execution plan without running subprocesses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional shared D1-D6 run directory. Defaults to outputs/runs/<timestamp>.",
    )
    parser.add_argument(
        "--info-sharing",
        choices=["without", "with"],
        default=None,
        help="Run only one information-sharing mode for selected datasets.",
    )
    return parser.parse_args()


def _format_cmd(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def print_dry_run(tasks: Sequence[Task]) -> None:
    for index, task in enumerate(tasks, start=1):
        print(f"{index}. [{task.label}] {_format_cmd(task.cmd)}")
        print(f"   {task.config_check}")
        if task.expected_result_path is not None:
            print(f"   expected result: {task.expected_result_path}")


def _snapshot_run_dirs() -> set[Path]:
    if not RUNS_DIR.exists():
        return set()
    return {path for path in RUNS_DIR.iterdir() if path.is_dir()}


def _resolve_result_path(text: str, filename: str) -> List[Path]:
    escaped = re.escape(filename)
    patterns = [
        rf"Results saved to\s+(.+?{escaped})",
        rf"Saved Dataset\d+ results:\s+(.+?{escaped})",
    ]
    paths: List[Path] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(1).strip()
            path = Path(raw)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path not in paths:
                paths.append(path)
    return paths


def _fallback_scan_result_paths(before_dirs: set[Path], filename: str) -> List[Path]:
    if not RUNS_DIR.exists():
        return []

    after_dirs = {path for path in RUNS_DIR.iterdir() if path.is_dir()}
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.name)
    paths: List[Path] = []
    for run_dir in new_dirs:
        candidate = run_dir / "results" / filename
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return paths


def _tail(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _task_with_result(
    task: Task,
    *,
    result_paths: Sequence[Path],
    returncode: int,
    elapsed_seconds: float,
) -> Task:
    return Task(
        dataset_token=task.dataset_token,
        dataset_id=task.dataset_id,
        label=task.label,
        scenario=task.scenario,
        cmd=list(task.cmd),
        config_check=task.config_check,
        result_filename=task.result_filename,
        expected_result_path=task.expected_result_path,
        result_paths=list(result_paths),
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
    )


def run_task(task: Task) -> Task:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[D{task.dataset_id}] 开始 {task.scenario}，时间戳 {timestamp}")
    print(f"Command: {_format_cmd(task.cmd)}")
    print(task.config_check)
    print(f"[orchestrator] parent python: {sys.executable}", flush=True)
    print(f"[orchestrator] command: {_format_cmd(task.cmd)}", flush=True)

    before_dirs = _snapshot_run_dirs()
    start = time.perf_counter()
    completed = subprocess.run(
        task.cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    result_paths = _resolve_result_path(combined_output, task.result_filename)
    if not result_paths:
        result_paths = _fallback_scan_result_paths(before_dirs, task.result_filename)

    run_dirs = sorted({path.parent.parent for path in result_paths})
    run_dir_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in run_dirs) or "未找到"
    print(
        f"[D{task.dataset_id}] 完成 {task.scenario}: "
        f"耗时 {elapsed:.1f}s, returncode={completed.returncode}, 输出目录路径={run_dir_text}"
    )

    if completed.returncode != 0:
        print(f"[D{task.dataset_id}] 子进程失败但继续后续任务。")
        if completed.stdout:
            print("[stdout tail]")
            print(_tail(completed.stdout))
        if completed.stderr:
            print("[stderr tail]")
            print(_tail(completed.stderr))

    return _task_with_result(
        task,
        result_paths=result_paths,
        returncode=int(completed.returncode),
        elapsed_seconds=float(elapsed),
    )


def _csv_row_count(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def print_result_summary(tasks: Sequence[Task]) -> None:
    print("\n数据集 | scenario | 结果路径 | 行数")
    print("--- | --- | --- | ---")
    for task in tasks:
        if not task.result_paths:
            print(f"D{task.dataset_id} | {task.scenario} | missing | missing")
            continue
        for path in task.result_paths:
            row_count = _csv_row_count(path)
            row_text = "missing" if row_count is None else str(row_count)
            try:
                display_path = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                display_path = str(path)
            print(f"D{task.dataset_id} | {task.scenario} | {display_path} | {row_text}")


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        run_dir = UNIFIED_RUN_DIR if args.output_dir is None else Path(args.output_dir)
    elif args.output_dir is None:
        run_dir = create_run_dir(PROJECT_ROOT, "unified")
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        reserve_new_output_dir(run_dir)
    try:
        tasks = build_tasks(
            only=args.only,
            smoke=bool(args.smoke),
            run_dir=run_dir,
            info_sharing=args.info_sharing,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print_dry_run(tasks)
        return

    completed_tasks = [run_task(task) for task in tasks]
    print_result_summary(completed_tasks)
    failed_tasks = [task for task in completed_tasks if task.returncode not in (None, 0)]
    if failed_tasks:
        labels = ", ".join(f"{task.label}(returncode={task.returncode})" for task in failed_tasks)
        print(f"[orchestrator] 子进程失败，统一 runner 返回非零状态: {labels}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# Minimal smoke verification command:
# .venv/bin/python scripts/run_unified_d1_d6.py --only d4 --smoke --dry-run
# Expected output: prints 2 commands, d4_without and d4_with, and runs no subprocesses.
