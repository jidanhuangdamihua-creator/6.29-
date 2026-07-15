from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from src.utils.run_recovery import (
    ActorIdentity,
    CellState,
    IdentityMismatchError,
    LeaseConflictError,
    RunRecovery,
    RunState,
    StaleFencingTokenError,
)


UTC = timezone.utc


def _actor() -> ActorIdentity:
    return ActorIdentity(
        subject="test-operator",
        subject_type="human",
        auth_context_id="test-context",
        command_digest="a" * 64,
    )


def _identities(value: str = "a") -> dict[str, str]:
    return {
        "run_plan_digest": value * 64,
        "code_digest": value * 64,
        "input_digest": value * 64,
        "protocol_digest": value * 64,
        "cache_digest": value * 64,
        "schema_digest": value * 64,
        "content_digest": value * 64,
    }


def test_attempts_events_and_results_are_append_only(tmp_path: Path) -> None:
    recovery = RunRecovery(tmp_path / "run", lease_ttl_seconds=30)
    now = datetime(2026, 7, 15, 1, 0, tzinfo=UTC)

    first = recovery.create(_actor(), run_identity="r" * 64, now=now)
    recovery.transition(
        RunState.PARTIAL_FAILED,
        actor=_actor(),
        fencing_token=first.fencing_token,
        reason="worker failure",
        now=now + timedelta(seconds=1),
    )
    recovery.finish_attempt(
        {"status": "partial_failed"},
        fencing_token=first.fencing_token,
    )
    second = recovery.resume(
        _actor(),
        expected_fencing_token=first.fencing_token,
        reason="retry",
        now=now + timedelta(seconds=31),
    )

    assert first.attempt_id != second.attempt_id
    assert second.fencing_token == first.fencing_token + 1
    assert recovery.layout.attempt_manifest(first.attempt_id).is_file()
    assert recovery.layout.attempt_result(first.attempt_id).is_file()
    assert recovery.layout.attempt_manifest(second.attempt_id).is_file()
    events = recovery.read_events()
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["previous_event_sha256"] is None
    assert all(
        events[index]["previous_event_sha256"] == events[index - 1]["event_sha256"]
        for index in range(1, len(events))
    )


def test_only_declared_run_transitions_are_allowed_and_seals_are_terminal(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)

    with pytest.raises(ValueError, match="illegal run state transition"):
        recovery.transition(
            RunState.SEALED_SUCCESS,
            actor=_actor(),
            fencing_token=lease.fencing_token,
            reason="too early",
        )

    recovery.transition(
        RunState.COMPLETE_UNSEALED,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        reason="all artifacts verified",
    )
    recovery.transition(
        RunState.SEALED_SUCCESS,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        reason="final gate passed",
    )
    with pytest.raises(ValueError, match="terminal"):
        recovery.transition(
            RunState.PARTIAL_FAILED,
            actor=_actor(),
            fencing_token=lease.fencing_token,
            reason="cannot reopen",
        )
    assert recovery.layout.sealed_success.is_file()
    assert recovery.read_events()[-1]["after_state"] == "sealed_success"


def test_lease_expiry_uses_heartbeat_not_pid_and_resume_is_fenced(tmp_path: Path) -> None:
    recovery = RunRecovery(tmp_path / "run", lease_ttl_seconds=10)
    now = datetime(2026, 7, 15, 2, 0, tzinfo=UTC)
    first = recovery.create(_actor(), run_identity="r" * 64, now=now)
    recovery.transition(
        RunState.PARTIAL_FAILED,
        actor=_actor(),
        fencing_token=first.fencing_token,
        reason="interrupted",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(LeaseConflictError, match="resume_lease_conflict"):
        recovery.resume(
            _actor(),
            expected_fencing_token=first.fencing_token,
            reason="too soon",
            now=now + timedelta(seconds=9),
        )

    second = recovery.resume(
        _actor(),
        expected_fencing_token=first.fencing_token,
        reason="expired owner",
        now=now + timedelta(seconds=12),
    )
    with pytest.raises(StaleFencingTokenError):
        recovery.heartbeat(first.fencing_token, now=now + timedelta(seconds=13))
    with pytest.raises((LeaseConflictError, StaleFencingTokenError)):
        recovery.resume(
            _actor(),
            expected_fencing_token=first.fencing_token,
            reason="CAS loser",
            now=now + timedelta(seconds=25),
        )
    assert second.fencing_token == 2


def test_expired_attempt_orphans_in_flight_cells_and_only_exact_accepted_reuses(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run", lease_ttl_seconds=5)
    now = datetime(2026, 7, 15, 3, 0, tzinfo=UTC)
    first = recovery.create(_actor(), run_identity="r" * 64, now=now)
    recovery.set_cell_state(
        "cell-a",
        CellState.IN_FLIGHT,
        actor=_actor(),
        fencing_token=first.fencing_token,
        identities=_identities(),
        reason="dispatched",
        now=now + timedelta(seconds=1),
    )
    recovery.set_cell_state(
        "cell-b",
        CellState.IN_FLIGHT,
        actor=_actor(),
        fencing_token=first.fencing_token,
        identities=_identities(),
        reason="dispatched",
        now=now + timedelta(seconds=1),
    )
    recovery.accept_cell(
        "cell-b",
        actor=_actor(),
        fencing_token=first.fencing_token,
        identities=_identities(),
        published_artifacts={"manifest_sha256": "b" * 64},
        reason="validated atomic publication",
        now=now + timedelta(seconds=2),
    )
    recovery.transition(
        RunState.PARTIAL_FAILED,
        actor=_actor(),
        fencing_token=first.fencing_token,
        reason="crash",
        now=now + timedelta(seconds=2),
    )
    second = recovery.resume(
        _actor(),
        expected_fencing_token=first.fencing_token,
        reason="recover",
        now=now + timedelta(seconds=8),
    )

    assert recovery.cell_state("cell-a") is CellState.ORPHANED
    assert recovery.cell_state("cell-b") is CellState.ACCEPTED
    assert recovery.is_cell_reusable("cell-b", identities=_identities())
    assert not recovery.is_cell_reusable("cell-a", identities=_identities())
    with pytest.raises(IdentityMismatchError):
        recovery.require_reusable_cell("cell-b", identities=_identities("c"))
    assert second.fencing_token == 2


def test_stale_token_cannot_publish_and_state_json_rebuilds_from_events(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run", lease_ttl_seconds=1)
    now = datetime(2026, 7, 15, 4, 0, tzinfo=UTC)
    first = recovery.create(_actor(), run_identity="r" * 64, now=now)
    recovery.transition(
        RunState.PARTIAL_FAILED,
        actor=_actor(),
        fencing_token=first.fencing_token,
        reason="stop",
        now=now,
    )
    recovery.resume(
        _actor(),
        expected_fencing_token=first.fencing_token,
        reason="new owner",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(StaleFencingTokenError):
        recovery.set_cell_state(
            "cell-a",
            CellState.IN_FLIGHT,
            actor=_actor(),
            fencing_token=first.fencing_token,
            identities=_identities(),
            reason="stale publication",
        )

    expected = recovery.load_state()
    recovery.layout.state.write_text("{}", encoding="utf-8")
    rebuilt = recovery.rebuild_state()
    assert rebuilt == expected
    assert json.loads(recovery.layout.state.read_text(encoding="utf-8")) == expected


def test_cell_directory_is_visible_only_after_validation_and_atomic_rename(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)
    recovery.set_cell_state(
        "cell-a",
        CellState.IN_FLIGHT,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        identities=_identities(),
        reason="start",
    )
    parent = recovery.layout.attempt_cells_dir(lease.attempt_id)
    candidate = parent / ".cell-a.candidate"
    candidate.mkdir()
    (candidate / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid candidate"):
        recovery.publish_cell_directory(
            "cell-a",
            candidate,
            validator=lambda _: (_ for _ in ()).throw(RuntimeError("invalid candidate")),
            actor=_actor(),
            fencing_token=lease.fencing_token,
            identities=_identities(),
            reason="validate",
        )
    assert candidate.is_dir()
    assert not (parent / "cell-a").exists()

    published = recovery.publish_cell_directory(
        "cell-a",
        candidate,
        validator=lambda _: {"manifest_sha256": "b" * 64},
        actor=_actor(),
        fencing_token=lease.fencing_token,
        identities=_identities(),
        reason="validated",
    )
    assert published == parent / "cell-a"
    assert published.is_dir()
    assert not candidate.exists()
    assert recovery.cell_state("cell-a") is CellState.ACCEPTED
