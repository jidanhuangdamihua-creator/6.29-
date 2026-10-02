"""Schedule ablation cells, then stop at the frozen gates.

Preflight runs before any training task, sealed-table read, or official
result file. OPEN-1, OPEN-2, and GAP-TCN-SOURCE-FIT are closed when
``tcn_source_fit`` is complete. An incomplete contract still refuses the
formal entry. Dry-run returns the gate result and the in-memory plan only.

Cell outcomes use the frozen tokens ``planned``, ``SKIP``, ``COMPLETED``,
``SKIPPED_INSUFFICIENT_CANDIDATES``, ``HOLD_HASH_MISMATCH``, ``HOLD_FAILED``,
and ``BLOCKED_GAP_TCN_SOURCE_FIT``. The phase audit file stays on schema v1:
``missing``, ``hash_mismatch``, ``failed``, and ``insufficient_candidates``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from sita_ablation import adaptation as adaptation_mod
from sita_ablation import io as io_mod
from sita_ablation import metrics as metrics_mod
from sita_ablation import stage_a as stage_a_mod
from sita_ablation.arms import (
    BLOCKED_GAP_TCN_SOURCE_FIT,
    RunSpec,
    expand_runs,
    run_directory,
    stage_a_key_for,
)
from sita_ablation.data import ConditionFrames, ContractError, DateSpan, DatasetId, load_dataset_contract
from sita_ablation.formal_context import assemble_formal_context
from sita_ablation.io import RunPayload, cell_id, require_clean_workspace_for_formal_run
from sita_ablation.source_selection import STATUS_SKIPPED
from sita_ablation.stage_a import StageAKey

OPEN_1_STATUS = "CLOSED"
OPEN_2_STATUS = "CLOSED"
PROTOCOL_VERSION = "PROTOCOL_AUDIT_FREEZE"
PLANNED = "planned"
_PHASES = (1, 2, 3)
_DATASETS = {"D1", "D4", "D5", "D6"}
_HORIZONS = {"1", "3", "5"}
_SEEDS = {"42", "43", "44", "45", "46"}
_FROZEN_NAMES = ("features_frozen.json", "protocol_frozen.json", "tcn_frozen.json")
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_CONFIG_DIR = _PACKAGE_ROOT / "configs"
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")
_TERMINAL_SKIP = "SKIP"
_AUDIT_BUCKETS = {
    "HOLD_HASH_MISMATCH": "hash_mismatch",
    "HOLD_FAILED": "failed",
    "SKIPPED_INSUFFICIENT_CANDIDATES": "insufficient_candidates",
    PLANNED: "missing",
}
_OMITTED_FROM_AUDIT = {"COMPLETED", _TERMINAL_SKIP, BLOCKED_GAP_TCN_SOURCE_FIT}

Clock = Callable[[], datetime]
Inspect = Callable[[Path], str]
WriteRun = Callable[[Path, RunPayload], object]
WriteFailed = Callable[[Path, int], None]
Adapt = Callable[[RunSpec], object]
Evaluate = Callable[[object], object]
Select = Callable[[RunSpec], object]


@dataclass(frozen=True)
class PreflightResult:
    """Gate result. ``accepted`` is false while a formal run must not start."""

    accepted: bool
    workspace_dirty: bool
    open_1: str
    open_2: str
    gap_tcn_source_fit: str
    frozen_config_sha256: str | None
    protocol_version: str | None
    schema_version: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlannedCell:
    spec: RunSpec
    cell_id: str
    directory: str
    stage_a_key_sha256: str | None
    status: str


@dataclass(frozen=True)
class DryRunResult:
    preflight: PreflightResult
    plan: tuple[PlannedCell, ...]
    phase: int
    workers: int


@dataclass(frozen=True)
class AttemptStamp:
    started_at: str
    finished_at: str
    seed: int


@dataclass(frozen=True)
class CellOutcome:
    spec: RunSpec
    cell_id: str
    status: str
    attempts: tuple[AttemptStamp, ...]


@dataclass(frozen=True)
class ExecutionResult:
    preflight: PreflightResult
    plan: tuple[PlannedCell, ...]
    outcomes: tuple[CellOutcome, ...]
    audit_path: Path | None


@dataclass(frozen=True)
class DispatchResult:
    outcomes: tuple[CellOutcome, ...]
    audit_path: Path


@dataclass
class CellContext:
    """Caller-supplied cell inputs. Missing windows never fall through to sealed tables."""

    clock: Clock
    inspect: Inspect
    write_run: WriteRun
    write_failed: WriteFailed
    logs_root: Path
    adapt: Adapt | None = None
    evaluate: Evaluate | None = None
    select: Select | None = None
    windows: object = None
    target_scaler: object = None
    init_template: object = None
    stage_a: object = None
    y_true: object = None
    frames: dict[str, ConditionFrames] | None = None
    conditions: dict[tuple[str, int], object] | None = None


def probe_workspace_dirty(repo_root: Path | None = None) -> bool:
    """Return whether ``git status --porcelain`` reports any worktree change."""
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("git workspace state is required and could not be read") from exc
    return bool(porcelain.strip())


def utc_timestamp_text(value: datetime) -> str:
    """Schema v1 UTC text with six microsecond digits and a ``+00:00`` suffix."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("timestamps reject a naive datetime")
    text = value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise ContractError("timestamps must be UTC with microseconds and +00:00")
    return text


def preflight(
    *,
    workspace_dirty: bool | None = None,
    repo_root: Path | None = None,
) -> PreflightResult:
    """Check the formal-run gates. This function does not create or modify files."""
    dirty = probe_workspace_dirty(repo_root) if workspace_dirty is None else workspace_dirty
    if type(dirty) is not bool:
        raise ContractError("workspace_dirty must stay a boolean")
    try:
        require_clean_workspace_for_formal_run(dirty)
    except ContractError:
        return PreflightResult(
            accepted=False,
            workspace_dirty=True,
            open_1=OPEN_1_STATUS,
            open_2=OPEN_2_STATUS,
            gap_tcn_source_fit=_gap_status(),
            frozen_config_sha256=None,
            protocol_version=None,
            schema_version=None,
            reasons=("workspace_dirty",),
        )
    identity = _frozen_identity()
    reasons: list[str] = []
    if OPEN_1_STATUS == "UNRESOLVED":
        reasons.append("OPEN-1")
    if OPEN_2_STATUS == "UNRESOLVED":
        reasons.append("OPEN-2")
    gap_status = _gap_status()
    if gap_status == "UNRESOLVED":
        reasons.append("GAP-TCN-SOURCE-FIT")
    if identity["file_sha256"] != identity["reported_sha256"]:
        reasons.append("frozen_config_sha256")
    protocol_version = identity["protocol_version"]
    if protocol_version is None:
        reasons.append("protocol_version")
    if identity["schema_version"] != 1:
        reasons.append("schema_v1")
    return PreflightResult(
        accepted=not reasons,
        workspace_dirty=False,
        open_1=OPEN_1_STATUS,
        open_2=OPEN_2_STATUS,
        gap_tcn_source_fit=gap_status,
        frozen_config_sha256=identity["file_sha256"],
        protocol_version=protocol_version,
        schema_version=identity["schema_version"],
        reasons=tuple(reasons),
    )


def dry_run(
    phase: int,
    *,
    workers: int,
    arms: Sequence[str] | None = None,
    only_cell: str | None = None,
    workspace_dirty: bool | None = None,
    repo_root: Path | None = None,
) -> DryRunResult:
    """Return preflight and the cell plan. Does not read sealed parquet, fit, or write runs."""
    _require_workers(workers)
    _require_phase(phase)
    gate = preflight(workspace_dirty=workspace_dirty, repo_root=repo_root)
    if gate.workspace_dirty:
        return DryRunResult(preflight=gate, plan=(), phase=phase, workers=workers)
    plan = _plan(phase, arms=arms, only_cell=only_cell)
    return DryRunResult(preflight=gate, plan=plan, phase=phase, workers=workers)


def execute(
    phase: int,
    *,
    workers: int,
    runs_root: Path,
    arms: Sequence[str] | None = None,
    only_cell: str | None = None,
    workspace_dirty: bool | None = None,
    repo_root: Path | None = None,
    context: CellContext | None = None,
    audit_path: Path | None = None,
    planned: Sequence[RunSpec] | None = None,
) -> ExecutionResult:
    """Formal entry. Unresolved gates return before any cell runs or any result file is written."""
    preview = dry_run(
        phase,
        workers=workers,
        arms=arms,
        only_cell=only_cell,
        workspace_dirty=workspace_dirty,
        repo_root=repo_root,
    )
    if not preview.preflight.accepted:
        return ExecutionResult(preview.preflight, preview.plan, (), None)
    if context is None or audit_path is None:
        raise ContractError("formal run inputs are not in memory; sealed tables are not read")
    dispatched = dispatch_cells(
        tuple(cell.spec for cell in preview.plan),
        phase=phase,
        runs_root=Path(runs_root),
        context=context,
        audit_path=Path(audit_path),
        planned=planned,
    )
    return ExecutionResult(
        preview.preflight,
        preview.plan,
        dispatched.outcomes,
        dispatched.audit_path,
    )


def dispatch_cells(
    specs: Sequence[RunSpec],
    *,
    phase: int,
    runs_root: Path,
    context: CellContext,
    audit_path: Path,
    planned: Sequence[RunSpec] | None = None,
) -> DispatchResult:
    """Run the resume, retry, and skip rules, then write a schema v1 phase audit.

    Hooks supply adaptation, metrics, and IO. Absent windows raise before
    ``adapt_target`` or ``evaluate_test``, and this function does not read
    sealed tables. ``DONE`` is written only by ``write_run``.
    """
    _require_phase(phase)
    outcomes = tuple(_dispatch_one(spec, Path(runs_root), context) for spec in specs)
    destination = Path(audit_path)
    _write_audit(phase, outcomes, destination, planned=planned)
    return DispatchResult(outcomes=outcomes, audit_path=destination)


def _gap_status() -> str:
    return "CLOSED" if stage_a_mod.tcn_source_fit_is_complete() else "UNRESOLVED"


def _tcn_source_blocked(spec: RunSpec) -> bool:
    return _sourced_tcn(spec) and not stage_a_mod.tcn_source_fit_is_complete()


def _dispatch_one(spec: RunSpec, runs_root: Path, context: CellContext) -> CellOutcome:
    if _tcn_source_blocked(spec):
        return CellOutcome(spec, cell_id(spec), BLOCKED_GAP_TCN_SOURCE_FIT, ())
    directory = run_directory(spec, runs_root)
    decision = context.inspect(directory)
    if decision == _TERMINAL_SKIP:
        return CellOutcome(spec, cell_id(spec), _TERMINAL_SKIP, ())
    if decision == "HOLD_HASH_MISMATCH":
        return CellOutcome(spec, cell_id(spec), "HOLD_HASH_MISMATCH", ())
    if decision == "HOLD_FAILED":
        return CellOutcome(spec, cell_id(spec), "HOLD_FAILED", ())
    if decision == "RETRY_ONCE":
        return _attempt_loop(spec, directory, context, first_attempt=2)
    if decision == "RUN":
        return _attempt_loop(spec, directory, context, first_attempt=1)
    raise ContractError(f"resume decision is not defined: {decision}")


def _attempt_loop(spec: RunSpec, directory: Path, context: CellContext, *, first_attempt: int) -> CellOutcome:
    attempt_number = first_attempt
    stamps: list[AttemptStamp] = []
    while True:
        started = context.clock()
        try:
            outcome = _run_attempt(spec, directory, context, started)
        except ContractError as exc:
            if "HOLD_HASH_MISMATCH" in str(exc) or "HOLD_FAILED" in str(exc):
                return CellOutcome(spec, cell_id(spec), _hold_token(str(exc)), tuple(stamps))
            outcome = None
            error: Exception = exc
        except Exception as exc:
            outcome = None
            error = exc
        else:
            return outcome
        finished = context.clock()
        stamps.append(AttemptStamp(utc_timestamp_text(started), utc_timestamp_text(finished), int(spec.seed)))
        _append_log(spec, context.logs_root, f"{type(error).__name__}: {error}")
        context.write_failed(directory, attempt_number)
        if attempt_number >= 2:
            return CellOutcome(spec, cell_id(spec), "HOLD_FAILED", tuple(stamps))
        attempt_number = 2


def _run_attempt(spec: RunSpec, directory: Path, context: CellContext, started: datetime) -> CellOutcome:
    selection = None
    key = None
    if spec.selection_rule != "none":
        key = _stage_key(spec)
        selection = _selection_for(spec, context)
        if getattr(selection, "status", None) == STATUS_SKIPPED:
            finished = context.clock()
            context.write_run(
                directory,
                RunPayload(
                    spec=spec,
                    started_at=started,
                    finished_at=finished,
                    stage_a_key=key,
                    selection=selection,
                ),
            )
            return _stamped(spec, STATUS_SKIPPED, started, finished)
    result = _adapt(spec, context)
    if getattr(result, "status", None) == BLOCKED_GAP_TCN_SOURCE_FIT:
        return CellOutcome(spec, cell_id(spec), BLOCKED_GAP_TCN_SOURCE_FIT, ())
    metrics = _metrics(result, context)
    finished = context.clock()
    context.write_run(
        directory,
        RunPayload(
            spec=spec,
            started_at=started,
            finished_at=finished,
            adaptation=result,
            metrics=metrics,
            stage_a_key=key,
            selection=selection,
        ),
    )
    return _stamped(spec, "COMPLETED", started, finished)


def _adapt(spec: RunSpec, context: CellContext) -> object:
    if context.adapt is not None:
        return context.adapt(spec)
    if context.windows is None or context.target_scaler is None or context.init_template is None:
        raise ContractError("target windows are not in memory; sealed tables are not read")
    return adaptation_mod.adapt_target(
        spec,
        context.windows,
        context.target_scaler,
        context.init_template,
        context.stage_a,
    )


def _metrics(result: object, context: CellContext) -> object:
    if context.evaluate is not None:
        return context.evaluate(result)
    if context.y_true is None:
        raise ContractError("original sales are not in memory; sealed tables are not read")
    return metrics_mod.evaluate_test(result, context.y_true)


def _selection_for(spec: RunSpec, context: CellContext) -> object:
    if context.select is None:
        raise ContractError("runner does not rank sources or read sealed tables")
    return context.select(spec)


def _plan(phase: int, *, arms: Sequence[str] | None, only_cell: str | None) -> tuple[PlannedCell, ...]:
    selected = _filter_specs(_expand(phase), arms=arms, only_cell=only_cell)
    frames: dict[str, ConditionFrames] = {}
    cells = []
    for spec in selected:
        digest = None
        if spec.selection_rule != "none":
            digest = stage_a_mod.stage_a_key_sha256(_stage_key(spec, frames))
        cells.append(
            PlannedCell(
                spec=spec,
                cell_id=cell_id(spec),
                directory=run_directory(spec).as_posix(),
                stage_a_key_sha256=digest,
                status=BLOCKED_GAP_TCN_SOURCE_FIT if _tcn_source_blocked(spec) else PLANNED,
            )
        )
    return tuple(cells)


def _expand(phase: int) -> tuple[RunSpec, ...]:
    if phase == 1:
        return expand_runs(1)
    if phase == 2:
        return expand_runs(2)
    if phase == 3:
        return expand_runs(3)
    raise ContractError(f"phase must be 1, 2, or 3, got {phase!r}")


def _filter_specs(
    specs: Sequence[RunSpec],
    *,
    arms: Sequence[str] | None,
    only_cell: str | None,
) -> tuple[RunSpec, ...]:
    chosen = tuple(specs)
    if arms is not None:
        known = {spec.arm for spec in chosen}
        unknown = [arm for arm in arms if arm not in known]
        if unknown:
            raise ContractError(f"arms are not in this phase: {unknown}")
        allowed = set(arms)
        chosen = tuple(spec for spec in chosen if spec.arm in allowed)
    if only_cell is not None:
        dataset, horizon, seed = _parse_only_cell(only_cell)
        chosen = tuple(
            spec for spec in chosen if spec.dataset == dataset and spec.horizon == horizon and spec.seed == seed
        )
    return chosen


def _stage_key(spec: RunSpec, frames: dict[str, ConditionFrames] | None = None) -> StageAKey:
    cache = {} if frames is None else frames
    dataset = spec.dataset
    if dataset not in cache:
        cache[dataset] = _frames_from_contract(dataset)
    key = stage_a_key_for(spec, cache[dataset])
    if key is None:
        raise ContractError("a sourced arm is missing its Stage A key")
    return key


def _frames_from_contract(dataset: DatasetId) -> ConditionFrames:
    """Date and column identity from the frozen contract. Rows stay empty."""
    contract = load_dataset_contract(dataset)
    return ConditionFrames(
        dataset=dataset,
        source_rows=pd.DataFrame(),
        target_rows=pd.DataFrame(),
        key_fields=contract.key_fields,
        target_key=contract.target_key,
        model_features=contract.model_features,
        similarity_variables=contract.similarity_variables,
        source_history=contract.source_history,
        target_train=contract.target_train,
        target_val=contract.target_val,
        target_test=contract.target_test,
        comparison_30=DateSpan(contract.target_train.start, contract.target_val.end),
        comparison_15=DateSpan(contract.target_train.start, contract.target_train.end),
        data_sha256={},
        grouping_field=contract.grouping_field,
    )


def _frozen_identity() -> dict[str, object]:
    digest = hashlib.sha256()
    versions: list[str] = []
    for name in _FROZEN_NAMES:
        path = _CONFIG_DIR / name
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"frozen config {name} could not be read") from exc
        digest.update(raw)
        status = payload.get("status")
        versions.append(status if type(status) is str else "")
    protocol = versions[0] if versions and all(item == PROTOCOL_VERSION for item in versions) else None
    schema = io_mod.SCHEMA_VERSION
    if type(schema) is not int:
        schema = -1
    return {
        "file_sha256": digest.hexdigest(),
        "reported_sha256": stage_a_mod.frozen_config_sha256(),
        "protocol_version": protocol,
        "schema_version": schema,
    }


def _write_audit(
    phase: int,
    outcomes: Sequence[CellOutcome],
    path: Path,
    *,
    planned: Sequence[RunSpec] | None,
) -> None:
    buckets = {name: [] for name in io_mod.PHASE_AUDIT_CATEGORIES}
    seen = {item.cell_id for item in outcomes}
    for item in outcomes:
        bucket = _audit_bucket(item.status)
        if bucket is not None:
            buckets[bucket].append(item.cell_id)
    if planned is not None:
        for spec in planned:
            identity = cell_id(spec)
            if identity in seen or _sourced_tcn(spec):
                continue
            buckets["missing"].append(identity)
            seen.add(identity)
    io_mod.write_phase_audit(
        path,
        phase=phase,
        missing=buckets["missing"],
        hash_mismatch=buckets["hash_mismatch"],
        failed=buckets["failed"],
        insufficient_candidates=buckets["insufficient_candidates"],
    )


def _audit_bucket(status: str) -> str | None:
    if status in _OMITTED_FROM_AUDIT:
        return None
    bucket = _AUDIT_BUCKETS.get(status)
    if bucket is None:
        raise ContractError(f"phase audit status is not defined: {status}")
    return bucket


def _sourced_tcn(spec: RunSpec) -> bool:
    return spec.backbone == "tcn" and spec.selection_rule != "none"


def _stamped(spec: RunSpec, status: str, started: datetime, finished: datetime) -> CellOutcome:
    stamp = AttemptStamp(utc_timestamp_text(started), utc_timestamp_text(finished), int(spec.seed))
    return CellOutcome(spec, cell_id(spec), status, (stamp,))


def _hold_token(message: str) -> str:
    if "HOLD_HASH_MISMATCH" in message:
        return "HOLD_HASH_MISMATCH"
    return "HOLD_FAILED"


def _append_log(spec: RunSpec, logs_root: Path, text: str) -> None:
    path = io_mod.log_path(spec, logs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _parse_only_cell(text: str) -> tuple[str, int, int]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ContractError("only-cell must be dataset:horizon:seed")
    dataset, horizon_text, seed_text = parts
    if dataset not in _DATASETS or horizon_text not in _HORIZONS or seed_text not in _SEEDS:
        raise ContractError("only-cell is outside the frozen grid")
    return dataset, int(horizon_text), int(seed_text)


def _require_workers(workers: int) -> None:
    if type(workers) is not int or workers < 1:
        raise ContractError("workers must be an integer greater than or equal to 1")


def _require_phase(phase: int) -> None:
    if type(phase) is not int or phase not in _PHASES:
        raise ContractError(f"phase must be 1, 2, or 3, got {phase!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Dry-run stops after preflight. A formal run loads inputs only after the gates pass."""
    import argparse

    parser = argparse.ArgumentParser(description="Ablation runner gates and dry-run plan")
    parser.add_argument("--phase", type=int, required=True, choices=_PHASES)
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-cell", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dry_run:
        result = dry_run(args.phase, workers=args.workers, arms=args.arms, only_cell=args.only_cell)
        print(json.dumps({"accepted": result.preflight.accepted, "reasons": list(result.preflight.reasons), "plan": len(result.plan)}))
        return 0 if result.preflight.accepted else 2
    preview = dry_run(args.phase, workers=args.workers, arms=args.arms, only_cell=args.only_cell)
    if not preview.preflight.accepted:
        print(json.dumps({"accepted": preview.preflight.accepted, "reasons": list(preview.preflight.reasons)}))
        return 2
    context = assemble_formal_context(tuple(cell.spec for cell in preview.plan))
    audit_path = _REPO_ROOT / io_mod.phase_audit_path(args.phase)
    result = execute(
        args.phase,
        workers=args.workers,
        runs_root=_PACKAGE_ROOT / "runs",
        arms=args.arms,
        only_cell=args.only_cell,
        context=context,
        audit_path=audit_path,
    )
    print(json.dumps({"accepted": result.preflight.accepted, "reasons": list(result.preflight.reasons)}))
    return 0 if result.preflight.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
