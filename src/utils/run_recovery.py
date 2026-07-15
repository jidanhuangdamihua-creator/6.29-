"""Append-only formal-run recovery with heartbeat leases and fencing.

Scheduler events are authoritative.  ``state.json`` and ``lease.json`` are
atomic, rebuildable indexes used to make local decisions efficiently.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Callable, Iterator, Mapping, Optional
from uuid import uuid4

from src.utils.run_layout import RunLayout


class RunState(str, Enum):
    RUNNING = "running"
    PARTIAL_FAILED = "partial_failed"
    COMPLETE_UNSEALED = "complete_unsealed"
    SEALED_SUCCESS = "sealed_success"
    SEALED_FAILED = "sealed_failed"


class CellState(str, Enum):
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    ACCEPTED = "accepted"
    FAILED = "failed"
    ORPHANED = "orphaned"


RUN_TRANSITIONS = {
    RunState.RUNNING: frozenset({RunState.PARTIAL_FAILED, RunState.COMPLETE_UNSEALED}),
    RunState.PARTIAL_FAILED: frozenset({RunState.RUNNING}),
    RunState.COMPLETE_UNSEALED: frozenset(
        {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}
    ),
    RunState.SEALED_SUCCESS: frozenset(),
    RunState.SEALED_FAILED: frozenset(),
}

CELL_TRANSITIONS = {
    CellState.QUEUED: frozenset({CellState.IN_FLIGHT, CellState.FAILED}),
    CellState.IN_FLIGHT: frozenset(
        {CellState.ACCEPTED, CellState.FAILED, CellState.ORPHANED}
    ),
    CellState.ACCEPTED: frozenset(),
    CellState.FAILED: frozenset({CellState.IN_FLIGHT}),
    CellState.ORPHANED: frozenset({CellState.IN_FLIGHT}),
}

REUSE_IDENTITY_FIELDS = frozenset(
    {
        "run_plan_digest",
        "code_digest",
        "input_digest",
        "protocol_digest",
        "cache_digest",
        "schema_digest",
        "content_digest",
    }
)


class RecoveryError(RuntimeError):
    pass


class LeaseConflictError(RecoveryError):
    pass


class StaleFencingTokenError(RecoveryError):
    pass


class IdentityMismatchError(RecoveryError):
    pass


@dataclass(frozen=True)
class ActorIdentity:
    subject: str
    subject_type: str
    auth_context_id: str
    command_digest: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"actor {name} must be a non-empty string")
        if len(self.command_digest) != 64:
            raise ValueError("actor command_digest must be a SHA-256 hex digest")


@dataclass(frozen=True)
class Lease:
    attempt_id: str
    fencing_token: int
    heartbeat_at: str
    expires_at: str
    hostname: str
    pid: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> datetime:
    result = _utc_now() if value is None else value
    if result.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and destination.exists():
            raise FileExistsError(f"append-only file already exists: {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RecoveryError(f"recovery JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"recovery JSON must be an object: {path}")
    return value


class RunRecovery:
    def __init__(self, run_root: Path, *, lease_ttl_seconds: int = 60) -> None:
        if int(lease_ttl_seconds) <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self.layout = RunLayout(Path(run_root))
        self.lease_ttl_seconds = int(lease_ttl_seconds)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.layout.run_root.mkdir(parents=True, exist_ok=True)
        with self.layout.recovery_lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _attempt_id(now: datetime) -> str:
        return now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex

    def _new_lease(self, attempt_id: str, token: int, now: datetime) -> Lease:
        return Lease(
            attempt_id=attempt_id,
            fencing_token=token,
            heartbeat_at=_iso(now),
            expires_at=_iso(now + timedelta(seconds=self.lease_ttl_seconds)),
            hostname=socket.gethostname(),
            pid=os.getpid(),
        )

    def _write_lease(self, lease: Lease) -> None:
        _atomic_json(self.layout.lease, asdict(lease))

    def _require_token(self, state: Mapping[str, Any], fencing_token: int) -> None:
        if int(fencing_token) != int(state.get("fencing_token", -1)):
            raise StaleFencingTokenError(
                f"stale fencing token {fencing_token}; current={state.get('fencing_token')}"
            )

    def _attempt_scaffold(
        self,
        attempt_id: str,
        *,
        actor: ActorIdentity,
        run_identity: str,
        token: int,
        now: datetime,
        reason: str,
    ) -> None:
        attempt_dir = self.layout.attempt_dir(attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        self.layout.attempt_events_dir(attempt_id).mkdir()
        self.layout.attempt_worker_logs(attempt_id).mkdir()
        self.layout.attempt_cells_dir(attempt_id).mkdir()
        manifest = {
            "attempt_id": attempt_id,
            "run_identity": run_identity,
            "fencing_token": token,
            "created_at": _iso(now),
            "reason": reason,
            "actor": asdict(actor),
            "hostname": socket.gethostname(),
            "os_user": getpass.getuser(),
            "pid": os.getpid(),
        }
        _atomic_json(self.layout.attempt_manifest(attempt_id), manifest, exclusive=True)
        _fsync_directory(attempt_dir)
        _fsync_directory(self.layout.attempts_dir)

    def create(
        self,
        actor: ActorIdentity,
        *,
        run_identity: str,
        now: Optional[datetime] = None,
    ) -> Lease:
        timestamp = _as_utc(now)
        with self._locked():
            if self.layout.state.exists() or self.layout.attempts_dir.exists():
                raise FileExistsError("formal recovery state already exists")
            attempt_id = self._attempt_id(timestamp)
            token = 1
            self._attempt_scaffold(
                attempt_id,
                actor=actor,
                run_identity=run_identity,
                token=token,
                now=timestamp,
                reason="initial attempt",
            )
            state: dict[str, Any] = {
                "run_identity": run_identity,
                "run_state": RunState.RUNNING.value,
                "fencing_token": token,
                "current_attempt_id": attempt_id,
                "event_sequence": 0,
                "last_event_sha256": None,
                "cells": {},
                "artifact_binding_set": None,
                "downstream_scheduling_started": False,
            }
            state = self._append_event(
                state,
                event_type="run_created",
                actor=actor,
                reason="initial attempt",
                before_state=None,
                after_state=RunState.RUNNING.value,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            lease = self._new_lease(attempt_id, token, timestamp)
            self._write_lease(lease)
            return lease

    def _event_files(self) -> list[Path]:
        if not self.layout.attempts_dir.is_dir():
            return []
        return sorted(self.layout.attempts_dir.glob("*/scheduler_events/*.json"))

    def read_events(self) -> list[dict[str, Any]]:
        events = [_read_json(path) for path in self._event_files()]
        events.sort(key=lambda value: int(value["sequence"]))
        previous = None
        for expected, event in enumerate(events, start=1):
            if event.get("sequence") != expected:
                raise RecoveryError("event sequence is not continuous")
            event_sha = event.get("event_sha256")
            unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
            if hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != event_sha:
                raise RecoveryError("event hash validation failed")
            if event.get("previous_event_sha256") != previous:
                raise RecoveryError("event hash chain validation failed")
            previous = event_sha
        return events

    def _append_event(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        actor: ActorIdentity,
        reason: str,
        before_state: Optional[str],
        after_state: Optional[str],
        now: datetime,
        subject_id: Optional[str] = None,
        identities: Optional[Mapping[str, str]] = None,
        published_artifacts: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        sequence = int(state.get("event_sequence", 0)) + 1
        attempt_id = str(state["current_attempt_id"])
        event: dict[str, Any] = {
            "sequence": sequence,
            "event_type": event_type,
            "subject_id": subject_id,
            "actor_subject": actor.subject,
            "actor_subject_type": actor.subject_type,
            "auth_context_id": actor.auth_context_id,
            "hostname": socket.gethostname(),
            "os_user": getpass.getuser(),
            "pid": os.getpid(),
            "command_digest": actor.command_digest,
            "attempt_id": attempt_id,
            "fencing_token": int(state["fencing_token"]),
            "timestamp": _iso(now),
            "reason": str(reason),
            "before_state": before_state,
            "after_state": after_state,
            "identities": None if identities is None else dict(identities),
            "published_artifacts": (
                None if published_artifacts is None else dict(published_artifacts)
            ),
            "previous_event_sha256": state.get("last_event_sha256"),
        }
        event["event_sha256"] = hashlib.sha256(_canonical_bytes(event)).hexdigest()
        event_path = self.layout.attempt_events_dir(attempt_id) / f"{sequence:020d}.json"
        _atomic_json(event_path, event, exclusive=True)
        updated = dict(state)
        updated["event_sequence"] = sequence
        updated["last_event_sha256"] = event["event_sha256"]
        return updated

    def load_state(self) -> dict[str, Any]:
        return _read_json(self.layout.state)

    def heartbeat(self, fencing_token: int, *, now: Optional[datetime] = None) -> Lease:
        timestamp = _as_utc(now)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            if RunState(state["run_state"]) is not RunState.RUNNING:
                raise LeaseConflictError("heartbeat requires a running attempt")
            lease = self._new_lease(
                str(state["current_attempt_id"]), int(fencing_token), timestamp
            )
            self._write_lease(lease)
            return lease

    def _lease_expired(self, now: datetime) -> bool:
        if not self.layout.lease.is_file():
            return True
        lease = _read_json(self.layout.lease)
        return now >= _parse_time(str(lease["expires_at"]))

    def resume(
        self,
        actor: ActorIdentity,
        *,
        expected_fencing_token: int,
        reason: str,
        now: Optional[datetime] = None,
    ) -> Lease:
        timestamp = _as_utc(now)
        with self._locked():
            state = self.load_state()
            if int(expected_fencing_token) != int(state.get("fencing_token", -1)):
                raise LeaseConflictError(
                    "resume_lease_conflict: fencing compare-and-swap failed"
                )
            if RunState(state["run_state"]) is not RunState.PARTIAL_FAILED:
                raise LeaseConflictError("resume_lease_conflict: run is not partial_failed")
            if not self._lease_expired(timestamp):
                raise LeaseConflictError("resume_lease_conflict: heartbeat lease is active")

            cells = dict(state.get("cells", {}))
            for cell_id, record in sorted(cells.items()):
                if record.get("state") == CellState.IN_FLIGHT.value:
                    state = self._record_cell_transition(
                        state,
                        cell_id=cell_id,
                        after=CellState.ORPHANED,
                        actor=actor,
                        identities=record.get("identities") or {},
                        reason="expired attempt left cell in flight",
                        now=timestamp,
                    )

            old_state = RunState(state["run_state"])
            new_token = int(state["fencing_token"]) + 1
            attempt_id = self._attempt_id(timestamp)
            self._attempt_scaffold(
                attempt_id,
                actor=actor,
                run_identity=str(state["run_identity"]),
                token=new_token,
                now=timestamp,
                reason=reason,
            )
            state["fencing_token"] = new_token
            state["current_attempt_id"] = attempt_id
            state["run_state"] = RunState.RUNNING.value
            state["artifact_binding_set"] = None
            state["downstream_scheduling_started"] = False
            state = self._append_event(
                state,
                event_type="run_transition",
                actor=actor,
                reason=reason,
                before_state=old_state.value,
                after_state=RunState.RUNNING.value,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            lease = self._new_lease(attempt_id, new_token, timestamp)
            self._write_lease(lease)
            return lease

    def transition(
        self,
        after: RunState,
        *,
        actor: ActorIdentity,
        fencing_token: int,
        reason: str,
        now: Optional[datetime] = None,
    ) -> RunState:
        timestamp = _as_utc(now)
        target = RunState(after)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            before = RunState(state["run_state"])
            if before in {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}:
                raise ValueError(f"run state {before.value} is terminal")
            if target not in RUN_TRANSITIONS[before]:
                raise ValueError(
                    f"illegal run state transition: {before.value}->{target.value}"
                )
            if before is RunState.PARTIAL_FAILED and target is RunState.RUNNING:
                raise ValueError("partial_failed->running requires resume CAS")
            state["run_state"] = target.value
            state = self._append_event(
                state,
                event_type="run_transition",
                actor=actor,
                reason=reason,
                before_state=before.value,
                after_state=target.value,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            if target is RunState.SEALED_SUCCESS:
                # The seal marker is intentionally the final publication.
                _atomic_json(
                    self.layout.sealed_success,
                    {
                        "run_identity": state["run_identity"],
                        "attempt_id": state["current_attempt_id"],
                        "fencing_token": state["fencing_token"],
                        "event_sha256": state["last_event_sha256"],
                        "sealed_at": _iso(timestamp),
                    },
                    exclusive=True,
                )
            if target in {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}:
                self.layout.lease.unlink(missing_ok=True)
                _fsync_directory(self.layout.run_root)
            return target

    @staticmethod
    def _validate_identities(identities: Mapping[str, str]) -> dict[str, str]:
        normalized = dict(identities)
        missing = REUSE_IDENTITY_FIELDS - normalized.keys()
        extra = normalized.keys() - REUSE_IDENTITY_FIELDS
        if missing or extra:
            raise ValueError(
                f"cell identities must be exact; missing={sorted(missing)} extra={sorted(extra)}"
            )
        if any(not isinstance(value, str) or not value for value in normalized.values()):
            raise ValueError("cell identity values must be non-empty strings")
        return normalized

    def _record_cell_transition(
        self,
        state: dict[str, Any],
        *,
        cell_id: str,
        after: CellState,
        actor: ActorIdentity,
        identities: Mapping[str, str],
        reason: str,
        now: datetime,
        published_artifacts: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        cells = dict(state.get("cells", {}))
        record = dict(cells.get(cell_id, {}))
        before_value = record.get("state", CellState.QUEUED.value)
        before = CellState(before_value)
        if after not in CELL_TRANSITIONS[before]:
            raise ValueError(
                f"illegal cell state transition: {before.value}->{after.value}"
            )
        record.update(
            {
                "state": after.value,
                "attempt_id": state["current_attempt_id"],
                "fencing_token": state["fencing_token"],
                "identities": dict(identities),
                "published_artifacts": (
                    None if published_artifacts is None else dict(published_artifacts)
                ),
                "updated_at": _iso(now),
            }
        )
        cells[cell_id] = record
        state["cells"] = cells
        return self._append_event(
            state,
            event_type="cell_transition",
            actor=actor,
            reason=reason,
            before_state=before.value,
            after_state=after.value,
            now=now,
            subject_id=cell_id,
            identities=identities,
            published_artifacts=published_artifacts,
        )

    def set_cell_state(
        self,
        cell_id: str,
        after: CellState,
        *,
        actor: ActorIdentity,
        fencing_token: int,
        identities: Mapping[str, str],
        reason: str,
        now: Optional[datetime] = None,
    ) -> CellState:
        timestamp = _as_utc(now)
        target = CellState(after)
        exact_identities = self._validate_identities(identities)
        if target is CellState.ACCEPTED:
            raise ValueError("accepted requires accept_cell after atomic artifact publication")
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            if RunState(state["run_state"]) is not RunState.RUNNING:
                raise RecoveryError("cell transition requires a running attempt")
            state = self._record_cell_transition(
                state,
                cell_id=str(cell_id),
                after=target,
                actor=actor,
                identities=exact_identities,
                reason=reason,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            return target

    def accept_cell(
        self,
        cell_id: str,
        *,
        actor: ActorIdentity,
        fencing_token: int,
        identities: Mapping[str, str],
        published_artifacts: Mapping[str, Any],
        reason: str,
        now: Optional[datetime] = None,
    ) -> CellState:
        if not published_artifacts:
            raise ValueError("accepted cell requires published artifact identities")
        timestamp = _as_utc(now)
        exact_identities = self._validate_identities(identities)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            if RunState(state["run_state"]) is not RunState.RUNNING:
                raise RecoveryError("cell acceptance requires a running attempt")
            state = self._record_cell_transition(
                state,
                cell_id=str(cell_id),
                after=CellState.ACCEPTED,
                actor=actor,
                identities=exact_identities,
                published_artifacts=published_artifacts,
                reason=reason,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            return CellState.ACCEPTED

    def publish_cell_directory(
        self,
        cell_id: str,
        candidate_dir: Path,
        *,
        validator: Callable[[Path], Mapping[str, Any]],
        actor: ActorIdentity,
        fencing_token: int,
        identities: Mapping[str, str],
        reason: str,
        now: Optional[datetime] = None,
    ) -> Path:
        """Validate and atomically rename a complete candidate directory."""
        timestamp = _as_utc(now)
        exact_identities = self._validate_identities(identities)
        candidate = Path(candidate_dir)
        if not candidate.is_dir():
            raise FileNotFoundError(f"candidate cell directory missing: {candidate}")
        artifacts = dict(validator(candidate))
        if not artifacts:
            raise RecoveryError("cell validator returned no artifact identities")
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            attempt_id = str(state["current_attempt_id"])
            destination = self.layout.attempt_cells_dir(attempt_id) / str(cell_id)
            if destination.exists():
                raise FileExistsError(f"accepted cell directory already exists: {destination}")
            if candidate.parent.resolve() != destination.parent.resolve():
                raise RecoveryError("candidate must share the accepted cell parent for atomic rename")
            for path in sorted(candidate.rglob("*")):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            _fsync_directory(candidate)
            os.replace(candidate, destination)
            _fsync_directory(destination.parent)
            state = self._record_cell_transition(
                state,
                cell_id=str(cell_id),
                after=CellState.ACCEPTED,
                actor=actor,
                identities=exact_identities,
                published_artifacts=artifacts,
                reason=reason,
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)
            return destination

    def cell_state(self, cell_id: str) -> CellState:
        record = self.load_state().get("cells", {}).get(str(cell_id))
        return CellState.QUEUED if record is None else CellState(record["state"])

    def is_cell_reusable(self, cell_id: str, *, identities: Mapping[str, str]) -> bool:
        exact = self._validate_identities(identities)
        record = self.load_state().get("cells", {}).get(str(cell_id))
        return bool(
            record
            and record.get("state") == CellState.ACCEPTED.value
            and record.get("identities") == exact
            and record.get("published_artifacts")
        )

    def require_reusable_cell(
        self, cell_id: str, *, identities: Mapping[str, str]
    ) -> dict[str, Any]:
        exact = self._validate_identities(identities)
        record = self.load_state().get("cells", {}).get(str(cell_id))
        if not record or record.get("state") != CellState.ACCEPTED.value:
            raise RecoveryError(f"cell is not accepted and reusable: {cell_id}")
        if record.get("identities") != exact:
            raise IdentityMismatchError(f"accepted cell identity mismatch: {cell_id}")
        if not record.get("published_artifacts"):
            raise RecoveryError(f"accepted cell has no published artifacts: {cell_id}")
        return dict(record)

    def finish_attempt(
        self, payload: Mapping[str, Any], *, fencing_token: int
    ) -> Path:
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            path = self.layout.attempt_result(str(state["current_attempt_id"]))
            _atomic_json(
                path,
                {**dict(payload), "attempt_id": state["current_attempt_id"], "fencing_token": fencing_token},
                exclusive=True,
            )
            return path

    def publish_artifact_binding_set(
        self,
        payload: Mapping[str, Any],
        *,
        binding_set_digest: str,
        actor: ActorIdentity,
        fencing_token: int,
        reason: str = "validated artifact authority frozen",
        now: Optional[datetime] = None,
    ) -> Path:
        """CAS-publish the only downstream artifact authority for an attempt."""
        timestamp = _as_utc(now)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            run_state = RunState(state["run_state"])
            if run_state in {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}:
                raise RecoveryError("artifact bindings cannot change in a sealed terminal state")
            if run_state is not RunState.RUNNING:
                raise RecoveryError("artifact bindings require a running attempt")
            if state.get("downstream_scheduling_started"):
                raise RecoveryError("artifact bindings must freeze before downstream scheduling")
            if state.get("artifact_binding_set") is not None:
                raise FileExistsError("artifact binding set is already frozen for this attempt")
            attempt_id = str(state["current_attempt_id"])
            if payload.get("attempt_id") != attempt_id:
                raise RecoveryError("binding set attempt does not match current attempt")
            if int(payload.get("fencing_token", -1)) != int(fencing_token):
                raise StaleFencingTokenError("binding set fencing token mismatch")
            unsigned = dict(payload)
            claimed_digest = unsigned.pop("binding_set_digest", None)
            computed_digest = "sha256:" + hashlib.sha256(
                _canonical_bytes(unsigned)
            ).hexdigest()
            if (
                claimed_digest != str(binding_set_digest)
                or str(binding_set_digest) != computed_digest
            ):
                raise RecoveryError("artifact binding set digest validation failed")
            path = self.layout.artifact_binding_set(attempt_id)
            _atomic_json(path, dict(payload), exclusive=True)
            relative_path = path.relative_to(self.layout.run_root).as_posix()
            authority = {
                "attempt_id": attempt_id,
                "fencing_token": int(fencing_token),
                "path": relative_path,
                "digest": str(binding_set_digest),
            }
            state["artifact_binding_set"] = authority
            state = self._append_event(
                state,
                event_type="artifact_binding_set_frozen",
                actor=actor,
                reason=reason,
                before_state=None,
                after_state="frozen",
                now=timestamp,
                published_artifacts=authority,
            )
            _atomic_json(self.layout.state, state)
            return path

    def record_artifact_rehydration(
        self,
        *,
        logical_artifact_id: str,
        old_artifact_sha256: str,
        new_artifact_sha256: str,
        physical_path: str,
        condition: str,
        actor: ActorIdentity,
        fencing_token: int,
        reason: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Append an authenticated audit event without making it binding authority."""
        timestamp = _as_utc(now)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            if RunState(state["run_state"]) is not RunState.RUNNING:
                raise RecoveryError("artifact rehydration requires a running attempt")
            if state.get("artifact_binding_set") is not None:
                raise RecoveryError("artifact rehydration cannot follow binding freeze")
            if state.get("downstream_scheduling_started"):
                raise RecoveryError("artifact rehydration cannot follow downstream scheduling")
            artifact_event = {
                "logical_artifact_id": str(logical_artifact_id),
                "old_artifact_sha256": str(old_artifact_sha256),
                "new_artifact_sha256": str(new_artifact_sha256),
                "physical_path": str(physical_path),
                "condition": str(condition),
                "fit_call_count": 0,
                "predict_call_count": 0,
            }
            state = self._append_event(
                state,
                event_type="artifact_rehydrated",
                actor=actor,
                reason=reason,
                before_state=str(condition),
                after_state="rehydrated",
                now=timestamp,
                subject_id=str(logical_artifact_id),
                published_artifacts=artifact_event,
            )
            _atomic_json(self.layout.state, state)

    def mark_downstream_scheduling_started(
        self,
        *,
        actor: ActorIdentity,
        fencing_token: int,
        reason: str = "downstream scheduling started",
        now: Optional[datetime] = None,
    ) -> None:
        timestamp = _as_utc(now)
        with self._locked():
            state = self.load_state()
            self._require_token(state, fencing_token)
            if RunState(state["run_state"]) is not RunState.RUNNING:
                raise RecoveryError("downstream scheduling requires a running attempt")
            if state.get("artifact_binding_set") is None:
                raise RecoveryError("artifact binding set must freeze before downstream scheduling")
            if state.get("downstream_scheduling_started"):
                raise RecoveryError("downstream scheduling was already started")
            state["downstream_scheduling_started"] = True
            state = self._append_event(
                state,
                event_type="downstream_scheduling_started",
                actor=actor,
                reason=reason,
                before_state=None,
                after_state="started",
                now=timestamp,
            )
            _atomic_json(self.layout.state, state)

    def rebuild_state(self) -> dict[str, Any]:
        events = self.read_events()
        if not events or events[0].get("event_type") != "run_created":
            raise RecoveryError("cannot rebuild state without run_created event")
        first_manifest = _read_json(
            self.layout.attempt_manifest(str(events[0]["attempt_id"]))
        )
        state: dict[str, Any] = {
            "run_identity": first_manifest["run_identity"],
            "run_state": RunState.RUNNING.value,
            "fencing_token": int(events[0]["fencing_token"]),
            "current_attempt_id": events[0]["attempt_id"],
            "event_sequence": 0,
            "last_event_sha256": None,
            "cells": {},
            "artifact_binding_set": None,
            "downstream_scheduling_started": False,
        }
        for event in events:
            state["event_sequence"] = event["sequence"]
            state["last_event_sha256"] = event["event_sha256"]
            state["fencing_token"] = event["fencing_token"]
            state["current_attempt_id"] = event["attempt_id"]
            if event["event_type"] in {"run_created", "run_transition"}:
                state["run_state"] = event["after_state"]
                if (
                    event["event_type"] == "run_transition"
                    and event["before_state"] == RunState.PARTIAL_FAILED.value
                    and event["after_state"] == RunState.RUNNING.value
                ):
                    state["artifact_binding_set"] = None
                    state["downstream_scheduling_started"] = False
            elif event["event_type"] == "cell_transition":
                state["cells"][event["subject_id"]] = {
                    "state": event["after_state"],
                    "attempt_id": event["attempt_id"],
                    "fencing_token": event["fencing_token"],
                    "identities": event["identities"],
                    "published_artifacts": event["published_artifacts"],
                    "updated_at": event["timestamp"],
                }
            elif event["event_type"] == "artifact_binding_set_frozen":
                state["artifact_binding_set"] = event["published_artifacts"]
            elif event["event_type"] == "downstream_scheduling_started":
                state["downstream_scheduling_started"] = True
        _atomic_json(self.layout.state, state)
        return state
