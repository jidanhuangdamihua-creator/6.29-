#!/usr/bin/env python3
"""Single-owner D1-D6 formal matrix orchestrator."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime
import getpass
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
from src.protocols.artifact_schemas import (
    EVALUATED_PREDICTION_TRACE_SCHEMA_NAME,
    WORKER_PREDICTION_TRACE_SCHEMA_NAME,
    artifact_schema_registry_digest,
    get_artifact_schema,
)
from src.protocols.feature_schema import get_predictor_schema
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
    finalize_failed_scheduler_attempt,
    publish_global_aggregate,
    publish_mode_matrix,
    resumable_formal_cell,
    verify_formal_cell_artifact,
    verify_formal_mode_artifact,
    write_or_validate_run_plan,
)
from src.utils.run_layout import RunLayout
from src.utils.run_recovery import ActorIdentity, RunRecovery, RunState
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


def _recovery_actor() -> ActorIdentity:
    command = "\0".join(sys.argv).encode("utf-8")
    return ActorIdentity(
        subject=getpass.getuser(),
        subject_type="os_user",
        auth_context_id="local-cli",
        command_digest=hashlib.sha256(command).hexdigest(),
    )


def _current_fencing_token(run_root: Path) -> int:
    state_path = RunLayout(Path(run_root)).state
    if not state_path.is_file():
        return 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state["fencing_token"])


def _formal_parquet_dir(project_root: Path, dataset_id: int) -> Path:
    root = Path(project_root)
    return root / "数据集" / "固化数据" / "d1_d6_sealed_v1" / f"dataset{int(dataset_id)}"


def discover_formal_input_identity(project_root: Path) -> dict[str, dict[str, object]]:
    root = Path(project_root)
    paths = [
        root / "configs" / "default_config.json",
        root / "configs" / "dataset_paths.json",
        root / "configs" / "matrix_config.json",
    ]
    paths.extend(sorted((root / "configs" / "solidified" / "knn").glob("**/*.json")))
    common_artifacts = (
        "source.parquet",
        "target.parquet",
        "manifest.json",
        "validation_report.json",
        "source_schema.json",
        "target_schema.json",
        "predictor_schema.json",
        "knn_schema.json",
        "calendarization_audit.json",
        "source_sales_canonicalization.json",
        "provenance.json",
    )
    for dataset_id in range(1, 7):
        parquet_dir = _formal_parquet_dir(root, dataset_id)
        paths.extend(parquet_dir / name for name in common_artifacts)
        if dataset_id >= 3:
            paths.append(parquet_dir / "adopt_validation_report.json")
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
    horizons: tuple[int, ...] = FORMAL_HORIZONS
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
                        label=f"D{dataset_id}-{mode}-s{cell.seed}",
                        scenario=mode,
                        cmd=command,
                        config_check="[FORMAL SEED BUNDLE] strict_paper; six methods; h1-h5; exact seed",
                        result_filename=cell.result_path.name,
                        expected_result_path=cell.result_path,
                        horizons=cell.horizons,
                        seed=cell.seed,
                    )
                )
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("standalone", "prepare", "mode-worker", "aggregate", "scheduler-finalize"),
        default="standalone",
        help="Internal formal lifecycle operation; standalone preserves the direct runner.",
    )
    parser.add_argument("--only", action="append", help="d1..d6, comma-separated or repeated")
    parser.add_argument("--info-sharing", choices=VALID_MODES, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse only accepted cells with matching hashes and code identity")
    parser.add_argument("--smoke", action="store_true", help="Diagnostic D4-D6 cells; never promotable")
    parser.add_argument("--scheduler-outcome", choices=("partial_failed",))
    parser.add_argument("--scheduler-reason")
    parser.add_argument("--scheduler-task-status", action="append", default=[])
    return parser.parse_args()


def _parse_scheduler_task_statuses(values: Sequence[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for value in values:
        task, separator, status = str(value).partition("=")
        if not separator or not task or not status:
            raise ValueError(
                "--scheduler-task-status must use the form dN_mode=status"
            )
        if task in statuses:
            raise ValueError(f"duplicate scheduler task status: {task}")
        statuses[task] = status
    return statuses


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
        "horizons": list(task.horizons),
        "seed": task.seed,
        "command": list(task.cmd),
        "result_path": str(task.expected_result_path),
    }


def _input_subset_identity(
    input_identity: Mapping[str, Mapping[str, object]],
    *,
    dataset_id: int,
    filename: str,
) -> dict[str, object]:
    suffix = f"d1_d6_sealed_v1/dataset{dataset_id}/{filename}"
    matches = [
        (path, dict(identity))
        for path, identity in input_identity.items()
        if str(path).replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        return {"path": suffix, "status": "unresolved"}
    path, identity = matches[0]
    return {"path": path, **identity}


def _pin_task_identities(
    entry: dict[str, object],
    *,
    input_identity: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    dataset_id = int(entry["dataset_id"])
    mode = str(entry["mode"])
    seed = int(entry["seed"])
    worker_schema = get_artifact_schema(WORKER_PREDICTION_TRACE_SCHEMA_NAME)
    evaluated_schema = get_artifact_schema(EVALUATED_PREDICTION_TRACE_SCHEMA_NAME)
    predictor_schema = get_predictor_schema(f"D{dataset_id}")
    mode_cache_payload = {
        "dataset_id": dataset_id,
        "mode": mode,
        "target": _input_subset_identity(
            input_identity, dataset_id=dataset_id, filename="target.parquet"
        ),
    }
    trace_base = {
        "dataset_id": dataset_id,
        "mode": mode,
        "seed": seed,
        "horizons": list(FORMAL_HORIZONS),
    }
    return {
        **entry,
        "mode_cache_identity": {
            **mode_cache_payload,
            "digest": _canonical_digest(mode_cache_payload),
        },
        "artifact_schema_registry_identity": {
            "version": "artifact_schema_registry_v1",
            "digest": artifact_schema_registry_digest(),
        },
        "predictor_feature_schema_identity": {
            "version": predictor_schema.version,
            "digest": predictor_schema.digest,
            "dimension": predictor_schema.dimension,
        },
        "source_repair_identity": _input_subset_identity(
            input_identity,
            dataset_id=dataset_id,
            filename="source_sales_canonicalization.json",
        ),
        "expected_trace_identities": {
            "worker": {
                **trace_base,
                "schema_name": worker_schema.schema_name,
                "schema_version": worker_schema.schema_version,
                "schema_digest": worker_schema.schema_digest,
            },
            "evaluated": {
                **trace_base,
                "schema_name": evaluated_schema.schema_name,
                "schema_version": evaluated_schema.schema_version,
                "schema_digest": evaluated_schema.schema_digest,
            },
        },
    }


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_legacy_run_plan(
    plan_path: Path,
    *,
    stored: object | None = None,
) -> None:
    path = Path(plan_path)
    if stored is None:
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"formal run plan is unreadable: {path}") from exc
    if not isinstance(stored, dict):
        return
    cells = stored.get("cells")
    version = str(stored.get("run_plan_version", ""))
    legacy_cells = isinstance(cells, list) and (
        len(cells) == 300
        or any(isinstance(cell, dict) and "horizon" in cell for cell in cells)
    )
    if legacy_cells or version in {
        "formal_d1_d6_run_plan_v1",
        "formal_d1_d6_run_plan_v2",
    }:
        raise RuntimeError(
            "legacy 300-cell run plans cannot resume under the seed-bundle protocol"
        )


def build_run_plan(
    run_root: Path,
    *,
    code_identity: CodeIdentity,
    input_identity: dict[str, dict[str, object]],
) -> dict[str, object]:
    root = Path(run_root).resolve()
    tasks = build_tasks(None, smoke=False, run_dir=root)
    cells = [
        _pin_task_identities(_task_plan_entry(task), input_identity=input_identity)
        for task in tasks
    ]
    result_paths = [str(cell["result_path"]) for cell in cells]
    if len(cells) != 60 or len(set(result_paths)) != 60:
        raise RuntimeError("formal run plan must contain exactly 60 unique seed bundles")

    payload: dict[str, object] = {
        "run_plan_version": "formal_d1_d6_seed_bundle_plan_v3",
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
    if resume:
        _reject_legacy_run_plan(root / "run_plan.json")
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
    recovery = RunRecovery(root)
    if resume:
        state = recovery.load_state()
        recovery.resume(
            _recovery_actor(),
            expected_fencing_token=int(state["fencing_token"]),
            reason="authenticated CLI resume",
        )
    else:
        recovery.create(
            _recovery_actor(),
            run_identity=str(plan["run_identity"]),
        )
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
    _reject_legacy_run_plan(plan_path, stored=stored)

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
    planned_core = [
        {key: cell.get(key) for key in actual[0]}
        for cell in planned
    ] if actual else []
    if len(planned) != 5 or len(actual) != 5 or planned_core != actual:
        raise RuntimeError(
            f"mode worker plan mismatch for d{dataset_id}_{mode}: "
            f"planned={len(planned)} actual={len(actual)}"
        )


def _cell_contract(task: Task, mode_expected: ExpectedResultContract) -> ExpectedResultContract:
    return replace(
        mode_expected,
        scope=AcceptanceScope.CELL,
        horizons=task.horizons,
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
    if len(cell_paths) != 5:
        raise RuntimeError("mode worker did not resolve exactly five seed bundle paths")

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
        fencing_token=_current_fencing_token(run_root),
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
    if not isinstance(raw_cells, list) or len(raw_cells) != 60:
        raise RuntimeError("global publication requires the full 60-bundle formal plan")

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
            if len(cell_paths) != 5:
                raise RuntimeError(
                    f"global publication requires five bundles for d{dataset_id}_{mode}"
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
        fencing_token=_current_fencing_token(root),
    )
    recovery = RunRecovery(root)
    state = recovery.load_state()
    if state["run_state"] == RunState.RUNNING.value:
        recovery.finish_attempt(
            {
                "status": RunState.COMPLETE_UNSEALED.value,
                "task_statuses": {
                    f"d{dataset_id}_{mode}": "succeeded"
                    for dataset_id in range(1, 7)
                    for mode in VALID_MODES
                },
            },
            fencing_token=int(state["fencing_token"]),
        )
        recovery.transition(
            RunState.COMPLETE_UNSEALED,
            actor=_recovery_actor(),
            fencing_token=int(state["fencing_token"]),
            reason="all twelve modes aggregated; awaiting final sealing gate",
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

        if operation == "scheduler-finalize":
            if (
                getattr(args, "only", None)
                or getattr(args, "info_sharing", None)
                or getattr(args, "resume", False)
                or getattr(args, "scheduler_outcome", None) != "partial_failed"
                or not getattr(args, "scheduler_reason", None)
            ):
                raise SystemExit(
                    "scheduler-finalize requires partial_failed, a reason, and task statuses"
                )
            try:
                statuses = _parse_scheduler_task_statuses(
                    getattr(args, "scheduler_task_status", [])
                )
                result = finalize_failed_scheduler_attempt(
                    output_path,
                    task_statuses=statuses,
                    reason=str(args.scheduler_reason),
                    actor=_recovery_actor(),
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"[PARTIAL_FAILED] attempt_result={result}")
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
    plan_cells = [
        _pin_task_identities(_task_plan_entry(task), input_identity=input_identity)
        for task in tasks
    ]
    plan_payload = {
        "run_plan_version": "formal_d1_d6_seed_bundle_plan_v3",
        "code_identity": code_identity.to_dict(),
        "schema_registry_version": RESULT_SCHEMA_REGISTRY_VERSION,
        "schema_registry_digest": result_schema_registry_digest(),
        "input_identity": input_identity,
        "methods": list(FORMAL_METHODS),
        "horizons": list(FORMAL_HORIZONS),
        "seeds": list(FORMAL_SEEDS),
        "cells": plan_cells,
    }
    run_plan = {**plan_payload, "run_identity": _canonical_digest(plan_payload)}
    if bool(getattr(args, "resume", False)):
        _reject_legacy_run_plan(run_root / "run_plan.json")
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
                horizons=task.horizons,
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
