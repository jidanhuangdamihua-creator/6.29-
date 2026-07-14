#!/usr/bin/env python3
"""Single-owner D1-D6 formal matrix orchestrator."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_strict_protocol_baseline import (
    build_matrix_tasks,
    build_mode_expected_contract,
)
from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_METHODS, FORMAL_SEEDS
from src.utils.result_acceptance import (
    AcceptanceScope,
    AggregateProfile,
    ExpectedResultContract,
    ResultAcceptanceError,
)
from src.utils.run_artifacts import (
    CodeIdentity,
    discover_code_identity,
    discover_input_identity,
    publish_global_aggregate,
    publish_mode_matrix,
    resumable_formal_cell,
    verify_formal_cell_artifact,
    verify_formal_mode_artifact,
    write_or_validate_run_plan,
)
from src.utils.run_layout import RunLayout
from src.utils.run_utils import create_run_dir, reserve_new_output_dir
from src.utils.result_schema import (
    RESULT_SCHEMA_REGISTRY_VERSION,
    result_schema_registry_digest,
)


RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
UNIFIED_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
UNIFIED_RUN_DIR = RUNS_DIR / UNIFIED_RUN_ID
VALID_DATASETS = tuple(f"d{number}" for number in range(1, 7))
VALID_MODES = ("without", "with")


def discover_formal_input_identity(project_root: Path) -> dict[str, dict[str, object]]:
    root = Path(project_root)
    paths = [
        root / "configs" / "default_config.json",
        root / "configs" / "dataset_paths.json",
        root / "configs" / "matrix_config.json",
    ]
    paths.extend(sorted((root / "configs" / "solidified" / "knn").glob("**/*.json")))
    for dataset_id in range(1, 7):
        paths.extend(
            [
                root / "数据集" / "固化数据" / f"dataset{dataset_id}-source.parquet",
                root / "数据集" / "固化数据" / f"dataset{dataset_id}-target.parquet",
            ]
        )
    d5_raw = root / "数据集" / "原始数据" / "Dataset 5Favorita"
    paths.extend(
        d5_raw / filename
        for filename in (
            "train.csv",
            "transactions.csv",
            "items.csv",
            "stores.csv",
            "oil.csv",
            "holidays_events.csv",
        )
    )
    return discover_input_identity(root, paths)


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
    horizon: int = 1
    seed: int = 42
    result_paths: List[Path] = field(default_factory=list)
    returncode: Optional[int] = None
    elapsed_seconds: Optional[float] = None


def _split_tokens(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return list(VALID_DATASETS)
    tokens: List[str] = []
    for value in values:
        tokens.extend(part.strip().lower() for part in str(value).split(",") if part.strip())
    return tokens or list(VALID_DATASETS)


def expand_only_tokens(values: Optional[Iterable[str]]) -> List[str]:
    requested = _split_tokens(values)
    unknown = [token for token in requested if token not in VALID_DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset id(s): {unknown}. Valid values: {list(VALID_DATASETS)}")
    selected = set(requested)
    return [token for token in VALID_DATASETS if token in selected]


def build_tasks(
    only: Optional[Iterable[str]],
    smoke: bool,
    run_dir: Optional[Path] = None,
    info_sharing: Optional[str] = None,
) -> List[Task]:
    if smoke:
        raise ValueError(
            "unified --smoke is disabled because diagnostic cells are not formal or resumable"
        )
    run_root = UNIFIED_RUN_DIR if run_dir is None else Path(run_dir)
    if info_sharing is not None and info_sharing not in VALID_MODES:
        raise ValueError("--info-sharing must be one of: without, with")
    layout = RunLayout(run_root)
    tasks: List[Task] = []
    modes = (info_sharing,) if info_sharing is not None else VALID_MODES
    for dataset_token in expand_only_tokens(only):
        dataset_id = int(dataset_token[1:])
        for mode in modes:
            matrix = build_matrix_tasks(
                dataset=dataset_token,
                scenario=mode,
                output_dir=layout.mode_dir(dataset_id, mode),
            )
            for cell in matrix:
                command = list(cell.command)
                if smoke and dataset_id >= 4:
                    command.append("--smoke")
                tasks.append(
                    Task(
                        dataset_token=dataset_token,
                        dataset_id=dataset_id,
                        label=f"D{dataset_id}-{mode}-h{cell.horizon}-s{cell.seed}",
                        scenario=mode,
                        cmd=command,
                        config_check="[FORMAL CELL] strict_paper; six methods; exact horizon/seed",
                        result_filename=cell.result_path.name,
                        expected_result_path=cell.result_path,
                        horizon=cell.horizon,
                        seed=cell.seed,
                    )
                )
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("standalone", "prepare", "mode-worker", "aggregate"),
        default="standalone",
        help="Internal formal lifecycle operation; standalone preserves the direct runner.",
    )
    parser.add_argument("--only", action="append", help="d1..d6, comma-separated or repeated")
    parser.add_argument("--info-sharing", choices=VALID_MODES, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse only accepted cells with matching hashes and code identity")
    parser.add_argument("--smoke", action="store_true", help="Diagnostic D4-D6 cells; never promotable")
    return parser.parse_args()


def _format_cmd(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def print_dry_run(tasks: Sequence[Task]) -> None:
    for index, task in enumerate(tasks, start=1):
        print(f"{index}. [{task.label}] {_format_cmd(task.cmd)}")
        print(f"   expected result: {task.expected_result_path}")
    print(f"[FORMAL PLAN] cells={len(tasks)} unique={len({task.expected_result_path for task in tasks})}")


def _task_with_result(
    task: Task,
    *,
    result_paths: Sequence[Path],
    returncode: int,
    elapsed_seconds: float,
) -> Task:
    return replace(
        task,
        result_paths=list(result_paths),
        returncode=int(returncode),
        elapsed_seconds=float(elapsed_seconds),
    )


def run_task(task: Task) -> Task:
    print(f"[{task.label}] {_format_cmd(task.cmd)}", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(
        task.cmd,
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    expected = task.expected_result_path
    result_paths = [expected] if expected is not None and expected.is_file() else []
    return _task_with_result(
        task,
        result_paths=result_paths,
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
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
    print("\ncell | result | rows")
    for task in tasks:
        path = task.expected_result_path
        count = None if path is None else _csv_row_count(path)
        print(f"{task.label} | {path or 'missing'} | {count if count is not None else 'missing'}")


def _mode_groups(tasks: Sequence[Task]) -> list[tuple[tuple[int, str], list[Task]]]:
    grouped: dict[tuple[int, str], list[Task]] = {}
    for task in tasks:
        grouped.setdefault((task.dataset_id, task.scenario), []).append(task)
    return list(grouped.items())


def _global_contract(mode_contracts: Sequence[ExpectedResultContract]) -> ExpectedResultContract:
    dataset_ids = tuple(sorted({contract.dataset_ids[0] for contract in mode_contracts}))
    modes = tuple(mode for mode in VALID_MODES if any(mode in contract.modes for contract in mode_contracts))
    targets = {
        key: value
        for contract in mode_contracts
        for key, value in contract.targets_by_dataset_mode.items()
    }
    full = set(targets) == {
        (dataset_id, mode) for dataset_id in range(1, 7) for mode in VALID_MODES
    }
    return ExpectedResultContract(
        scope=AcceptanceScope.GLOBAL_AGGREGATE,
        formal=True,
        dataset_ids=dataset_ids,
        modes=modes,
        protocol_tracks=("strict_paper",),
        targets_by_dataset_mode=targets,
        methods=FORMAL_METHODS,
        horizons=FORMAL_HORIZONS,
        seeds=FORMAL_SEEDS,
        confirmation_eligible=True,
        aggregate_profile=(
            AggregateProfile.FULL_D1_D6_BASELINE
            if full
            else AggregateProfile.RUN_SELECTION_AGGREGATE
        ),
    )


def _task_plan_entry(task: Task) -> dict[str, object]:
    if task.expected_result_path is None:
        raise RuntimeError(f"formal task has no expected result path: {task.label}")
    return {
        "dataset_id": task.dataset_id,
        "mode": task.scenario,
        "horizon": task.horizon,
        "seed": task.seed,
        "command": list(task.cmd),
        "result_path": str(task.expected_result_path),
    }


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_plan(
    run_root: Path,
    *,
    code_identity: CodeIdentity,
    input_identity: dict[str, dict[str, object]],
) -> dict[str, object]:
    root = Path(run_root).resolve()
    tasks = build_tasks(None, smoke=False, run_dir=root)
    cells = [_task_plan_entry(task) for task in tasks]
    result_paths = [str(cell["result_path"]) for cell in cells]
    if len(cells) != 300 or len(set(result_paths)) != 300:
        raise RuntimeError("formal run plan must contain exactly 300 unique cells")

    payload: dict[str, object] = {
        "run_plan_version": "formal_d1_d6_run_plan_v2",
        "code_identity": code_identity.to_dict(),
        "schema_registry_version": RESULT_SCHEMA_REGISTRY_VERSION,
        "schema_registry_digest": result_schema_registry_digest(),
        "input_identity": input_identity,
        "methods": list(FORMAL_METHODS),
        "horizons": list(FORMAL_HORIZONS),
        "seeds": list(FORMAL_SEEDS),
        "cells": cells,
    }
    return {**payload, "run_identity": _canonical_digest(payload)}


def prepare_formal_run(run_root: Path, *, resume: bool) -> dict[str, object]:
    root = Path(run_root).resolve()
    code_identity = discover_code_identity(PROJECT_ROOT)
    if code_identity.dirty:
        raise RuntimeError("formal execution requires a clean git worktree")
    input_identity = discover_formal_input_identity(PROJECT_ROOT)
    plan = build_run_plan(
        root,
        code_identity=code_identity,
        input_identity=input_identity,
    )

    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"resume run root does not exist: {root}")
    else:
        reserve_new_output_dir(root)
    write_or_validate_run_plan(root / "run_plan.json", plan, resume=resume)
    return plan


def load_validated_run_plan(
    run_root: Path,
) -> tuple[dict[str, object], CodeIdentity]:
    root = Path(run_root).resolve()
    plan_path = root / "run_plan.json"
    try:
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"formal run plan is unreadable: {plan_path}") from exc
    if not isinstance(stored, dict):
        raise RuntimeError(f"formal run plan must be a JSON object: {plan_path}")

    code_identity = discover_code_identity(PROJECT_ROOT)
    if code_identity.dirty:
        raise RuntimeError("formal execution requires a clean git worktree")
    current = build_run_plan(
        root,
        code_identity=code_identity,
        input_identity=discover_formal_input_identity(PROJECT_ROOT),
    )
    if stored != current:
        raise RuntimeError("current formal plan or identity does not match run_plan.json")
    return current, code_identity


def _require_tasks_match_plan(
    tasks: Sequence[Task],
    plan: Mapping[str, object],
    *,
    dataset_id: int,
    mode: str,
) -> None:
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list):
        raise RuntimeError("formal run plan has no cell list")
    planned = [
        cell
        for cell in raw_cells
        if isinstance(cell, dict)
        and cell.get("dataset_id") == dataset_id
        and cell.get("mode") == mode
    ]
    actual = [_task_plan_entry(task) for task in tasks]
    if len(planned) != 25 or len(actual) != 25 or planned != actual:
        raise RuntimeError(
            f"mode worker plan mismatch for d{dataset_id}_{mode}: "
            f"planned={len(planned)} actual={len(actual)}"
        )


def _cell_contract(task: Task, mode_expected: ExpectedResultContract) -> ExpectedResultContract:
    return replace(
        mode_expected,
        scope=AcceptanceScope.CELL,
        horizons=(task.horizon,),
        seeds=(task.seed,),
    )


def execute_mode_worker(
    mode_dir: Path,
    dataset: str,
    mode: str,
    *,
    resume: bool,
) -> Path:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown formal dataset: {dataset}")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown formal mode: {mode}")
    dataset_id = int(dataset[1:])
    requested_mode_dir = Path(mode_dir).resolve()
    run_root = requested_mode_dir.parent
    if not (run_root / "run_plan.json").is_file():
        raise ValueError(
            f"mode worker output is not below a prepared canonical mode directory: "
            f"{requested_mode_dir}"
        )
    layout = RunLayout(run_root)
    canonical_mode_dir = layout.mode_dir(dataset_id, mode).resolve()
    if requested_mode_dir != canonical_mode_dir:
        raise ValueError(
            f"mode worker output is not the canonical mode directory: {requested_mode_dir}"
        )

    plan, code_identity = load_validated_run_plan(run_root)
    tasks = build_tasks(
        [dataset],
        smoke=False,
        run_dir=run_root,
        info_sharing=mode,
    )
    _require_tasks_match_plan(
        tasks,
        plan,
        dataset_id=dataset_id,
        mode=mode,
    )
    expected = build_mode_expected_contract(dataset=dataset, scenario=mode)
    output = layout.mode_result(dataset_id, mode)
    all_cells_reused = bool(resume)

    for task in tasks:
        if task.expected_result_path is None:
            raise RuntimeError(f"formal task has no expected result path: {task.label}")
        cell_expected = _cell_contract(task, expected)
        if resume:
            try:
                verify_formal_cell_artifact(
                    task.expected_result_path,
                    acceptance_path=task.expected_result_path.with_suffix(
                        ".acceptance.json"
                    ),
                    expected=cell_expected,
                    code_identity=code_identity,
                )
                print(f"[RESUME] accepted {task.label}")
                continue
            except ResultAcceptanceError:
                all_cells_reused = False

        completed = run_task(task)
        if completed.returncode != 0 or not completed.result_paths:
            print_result_summary([completed])
            raise RuntimeError(
                f"formal cell failed: {task.label} returncode={completed.returncode}"
            )
        verify_formal_cell_artifact(
            task.expected_result_path,
            acceptance_path=task.expected_result_path.with_suffix(".acceptance.json"),
            expected=cell_expected,
            code_identity=code_identity,
        )

    cell_paths = [
        task.expected_result_path
        for task in tasks
        if task.expected_result_path is not None
    ]
    if len(cell_paths) != 25:
        raise RuntimeError("mode worker did not resolve exactly 25 cell paths")

    if all_cells_reused:
        try:
            verify_formal_mode_artifact(
                output,
                acceptance_path=layout.mode_acceptance_report(dataset_id, mode),
                cell_paths=cell_paths,
                expected=expected,
                code_identity=code_identity,
            )
            print(f"[RESUME] accepted mode=d{dataset_id}_{mode}")
            return output
        except ResultAcceptanceError:
            pass

    load_validated_run_plan(run_root)
    publish_mode_matrix(
        cell_paths,
        stable_path=output,
        expected=expected,
        code_identity=code_identity,
    )
    verify_formal_mode_artifact(
        output,
        acceptance_path=layout.mode_acceptance_report(dataset_id, mode),
        cell_paths=cell_paths,
        expected=expected,
        code_identity=code_identity,
    )
    print(f"[ACCEPTED] mode=d{dataset_id}_{mode} result={output}")
    return output


def aggregate_prepared_run(run_root: Path) -> Path:
    root = Path(run_root).resolve()
    plan, code_identity = load_validated_run_plan(root)
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 300:
        raise RuntimeError("global publication requires the full 300-cell formal plan")

    layout = RunLayout(root)
    mode_paths: list[Path] = []
    mode_contracts: list[ExpectedResultContract] = []
    for dataset_id in range(1, 7):
        dataset = f"d{dataset_id}"
        for mode in VALID_MODES:
            tasks = build_tasks(
                [dataset],
                smoke=False,
                run_dir=root,
                info_sharing=mode,
            )
            _require_tasks_match_plan(
                tasks,
                plan,
                dataset_id=dataset_id,
                mode=mode,
            )
            cell_paths = [
                task.expected_result_path
                for task in tasks
                if task.expected_result_path is not None
            ]
            if len(cell_paths) != 25:
                raise RuntimeError(
                    f"global publication requires 25 cells for d{dataset_id}_{mode}"
                )
            expected = build_mode_expected_contract(
                dataset=dataset,
                scenario=mode,
            )
            output = layout.mode_result(dataset_id, mode)
            verify_formal_mode_artifact(
                output,
                acceptance_path=layout.mode_acceptance_report(dataset_id, mode),
                cell_paths=cell_paths,
                expected=expected,
                code_identity=code_identity,
            )
            mode_paths.append(output)
            mode_contracts.append(expected)

    load_validated_run_plan(root)
    publish_global_aggregate(
        mode_paths,
        stable_path=layout.aggregate_result,
        expected=_global_contract(mode_contracts),
        code_identity=code_identity,
    )
    print(f"[ACCEPTED] aggregate={layout.aggregate_result}")
    return layout.aggregate_result


def _resolve_run_root(args: argparse.Namespace) -> Path:
    output_dir = getattr(args, "output_dir", None)
    if getattr(args, "dry_run", False):
        return UNIFIED_RUN_DIR if output_dir is None else Path(output_dir)
    if output_dir is None:
        return create_run_dir(PROJECT_ROOT, "unified")
    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if getattr(args, "resume", False):
        if not path.is_dir():
            raise FileNotFoundError(f"resume run root does not exist: {path}")
        return path
    return reserve_new_output_dir(path)


def main() -> None:
    args = _parse_args()
    operation = getattr(args, "operation", "standalone")
    if operation != "standalone":
        if getattr(args, "smoke", False) or getattr(args, "dry_run", False):
            raise SystemExit(
                f"--operation {operation} does not allow --smoke or --dry-run"
            )
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            raise SystemExit(f"--operation {operation} requires --output-dir")
        output_path = Path(output_dir)

        if operation == "prepare":
            if getattr(args, "only", None) or getattr(args, "info_sharing", None):
                raise SystemExit("prepare does not allow --only or --info-sharing")
            plan = prepare_formal_run(
                output_path,
                resume=bool(getattr(args, "resume", False)),
            )
            print(
                f"[PREPARED] run_root={output_path} "
                f"run_identity={plan['run_identity']} cells={len(plan['cells'])}"
            )
            return

        if operation == "mode-worker":
            selected = expand_only_tokens(getattr(args, "only", None))
            mode = getattr(args, "info_sharing", None)
            if len(selected) != 1 or mode not in VALID_MODES:
                raise SystemExit(
                    "mode-worker requires exactly one --only dN and one --info-sharing mode"
                )
            execute_mode_worker(
                output_path,
                selected[0],
                mode,
                resume=bool(getattr(args, "resume", False)),
            )
            return

        if operation == "aggregate":
            if (
                getattr(args, "only", None)
                or getattr(args, "info_sharing", None)
                or getattr(args, "resume", False)
            ):
                raise SystemExit(
                    "aggregate does not allow --only, --info-sharing, or --resume"
                )
            aggregate_prepared_run(output_path)
            return

        raise SystemExit(f"unknown formal lifecycle operation: {operation}")

    if args.smoke:
        raise SystemExit(
            "unified --smoke is disabled; run an individual dataset diagnostic runner instead"
        )
    run_root = _resolve_run_root(args)
    try:
        tasks = build_tasks(
            only=args.only,
            smoke=bool(args.smoke),
            run_dir=run_root,
            info_sharing=args.info_sharing,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print_dry_run(tasks)
        return

    code_identity = discover_code_identity(PROJECT_ROOT)
    if code_identity.dirty:
        raise SystemExit(
            "formal execution requires a clean git worktree; commit or stash code changes first"
        )
    input_identity = discover_formal_input_identity(PROJECT_ROOT)
    run_plan = {
        "run_plan_version": "formal_d1_d6_run_plan_v1",
        "code_identity": code_identity.to_dict(),
        "schema_registry_version": RESULT_SCHEMA_REGISTRY_VERSION,
        "schema_registry_digest": result_schema_registry_digest(),
        "input_identity": input_identity,
        "methods": list(FORMAL_METHODS),
        "horizons": list(FORMAL_HORIZONS),
        "seeds": list(FORMAL_SEEDS),
        "cells": [
            {
                "dataset_id": task.dataset_id,
                "mode": task.scenario,
                "horizon": task.horizon,
                "seed": task.seed,
                "command": list(task.cmd),
                "result_path": str(task.expected_result_path),
            }
            for task in tasks
        ],
    }
    write_or_validate_run_plan(
        run_root / "run_plan.json",
        run_plan,
        resume=bool(getattr(args, "resume", False)),
    )
    completed_tasks: list[Task] = []
    for task in tasks:
        if getattr(args, "resume", False) and not args.smoke and task.expected_result_path is not None:
            mode_expected = build_mode_expected_contract(
                dataset=task.dataset_token,
                scenario=task.scenario,
            )
            cell_expected = replace(
                mode_expected,
                scope=AcceptanceScope.CELL,
                horizons=(task.horizon,),
                seeds=(task.seed,),
            )
            if resumable_formal_cell(
                stable_path=task.expected_result_path,
                manifest_path=task.expected_result_path.with_suffix(".manifest.json"),
                expected=cell_expected,
                code_identity=code_identity,
            ):
                completed_tasks.append(
                    _task_with_result(task, result_paths=[task.expected_result_path], returncode=0, elapsed_seconds=0.0)
                )
                print(f"[RESUME] accepted {task.label}")
                continue
        completed = run_task(task)
        completed_tasks.append(completed)
        if completed.returncode != 0 or not completed.result_paths:
            print_result_summary(completed_tasks)
            raise SystemExit(1)

    print_result_summary(completed_tasks)
    if args.smoke:
        print("[DIAGNOSTIC] smoke outputs are not acceptance-eligible")
        return

    if discover_formal_input_identity(PROJECT_ROOT) != input_identity:
        raise SystemExit("formal inputs changed while the run was executing")

    layout = RunLayout(run_root)
    mode_paths: list[Path] = []
    mode_contracts: list[ExpectedResultContract] = []
    for (dataset_id, mode), group in _mode_groups(completed_tasks):
        expected = build_mode_expected_contract(dataset=f"d{dataset_id}", scenario=mode)
        output = layout.mode_result(dataset_id, mode)
        publish_mode_matrix(
            [task.expected_result_path for task in group if task.expected_result_path is not None],
            stable_path=output,
            expected=expected,
            code_identity=code_identity,
        )
        mode_paths.append(output)
        mode_contracts.append(expected)

    publish_global_aggregate(
        mode_paths,
        stable_path=layout.aggregate_result,
        expected=_global_contract(mode_contracts),
        code_identity=code_identity,
    )
    print(f"[ACCEPTED] aggregate={layout.aggregate_result}")


if __name__ == "__main__":
    main()
