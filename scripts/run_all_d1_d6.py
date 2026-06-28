"""Orchestrate D4-D6 fixed-parquet experiment tasks.

This runner intentionally exposes only the concrete D4-D6 task ids plus the
safe aliases documented in AGENTS.md.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VALID_TASK_IDS = (
    "d4_without",
    "d4_with",
    "d5_without",
    "d5_with",
    "d6_without",
    "d6_with",
)

TASK_ALIASES = {
    "d4": ("d4_without", "d4_with"),
    "d5": ("d5_without", "d5_with"),
    "d6": ("d6_without", "d6_with"),
    "d4_d6": VALID_TASK_IDS,
}


def _split_task_tokens(values: Iterable[str] | None) -> List[str]:
    if not values:
        return list(VALID_TASK_IDS)
    tokens: List[str] = []
    for value in values:
        for part in str(value).split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tokens or list(VALID_TASK_IDS)


def expand_task_ids(values: Iterable[str] | None) -> List[str]:
    """Expand safe aliases before validating unknown task ids."""
    expanded: List[str] = []
    for token in _split_task_tokens(values):
        for task_id in TASK_ALIASES.get(token, (token,)):
            if task_id not in expanded:
                expanded.append(task_id)

    unknown = [task_id for task_id in expanded if task_id not in VALID_TASK_IDS]
    if unknown:
        raise ValueError(
            f"Unknown task id(s): {unknown}. Valid task ids: {list(VALID_TASK_IDS)}. "
            f"Aliases: {sorted(TASK_ALIASES)}"
        )
    return expanded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D4-D6 fixed-parquet experiment tasks.")
    parser.add_argument(
        "--only",
        action="append",
        help=(
            "Task id(s), comma-separated or repeated. Valid ids: "
            "d4_without,d4_with,d5_without,d5_with,d6_without,d6_with. "
            "Safe aliases: d4,d5,d6,d4_d6."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to each selected D4-D6 runner.")
    parser.add_argument("--target-limit", type=int, default=None, help="Optional smoke target entity limit.")
    parser.add_argument("--source-limit", type=int, default=None, help="Optional smoke source entity limit.")
    return parser.parse_args()


def _task_to_module_and_scenario(task_id: str) -> tuple[str, str]:
    dataset_token, scenario = task_id.split("_", 1)
    return f"scripts.run_{dataset_token}_experiment", scenario


def run_task(task_id: str, smoke: bool = False, target_limit: int | None = None, source_limit: int | None = None) -> None:
    module_name, scenario = _task_to_module_and_scenario(task_id)
    module = importlib.import_module(module_name)

    argv = [module_name, "--info-sharing", scenario]
    if smoke:
        argv.append("--smoke")
    if target_limit is not None:
        argv.extend(["--target-limit", str(int(target_limit))])
    if source_limit is not None:
        argv.extend(["--source-limit", str(int(source_limit))])

    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = _parse_args()
    task_ids = expand_task_ids(args.only)
    print(f"Selected D4-D6 tasks: {','.join(task_ids)}")
    for task_id in task_ids:
        print(f"Running task: {task_id}")
        run_task(
            task_id,
            smoke=bool(args.smoke),
            target_limit=args.target_limit,
            source_limit=args.source_limit,
        )


if __name__ == "__main__":
    main()
