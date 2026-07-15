from __future__ import annotations

import pytest

from src.utils.result_validation import CELL_TRANSITIONS, validate_cell_transition
from src.utils.run_recovery import CELL_TRANSITIONS as RECOVERY_CELL_TRANSITIONS
from src.utils.run_recovery import RUN_TRANSITIONS, CellState, RunState


def test_cell_state_machine_allows_only_declared_transitions() -> None:
    assert CELL_TRANSITIONS == {
        "planned": frozenset({"claimed", "failed"}),
        "claimed": frozenset({"running", "failed"}),
        "running": frozenset({"candidate", "failed"}),
        "candidate": frozenset({"accepted_cell", "failed"}),
        "accepted_cell": frozenset(),
        "failed": frozenset(),
    }

    for current, allowed in CELL_TRANSITIONS.items():
        for candidate in CELL_TRANSITIONS:
            if candidate in allowed:
                assert validate_cell_transition(current, candidate) == candidate
            else:
                with pytest.raises(ValueError, match="illegal cell state transition"):
                    validate_cell_transition(current, candidate)


def test_cell_state_machine_rejects_unknown_states() -> None:
    with pytest.raises(ValueError, match="unknown cell state"):
        validate_cell_transition("missing", "failed")
    with pytest.raises(ValueError, match="unknown cell state"):
        validate_cell_transition("planned", "missing")


def test_formal_recovery_state_sets_are_frozen() -> None:
    assert RUN_TRANSITIONS == {
        RunState.RUNNING: frozenset(
            {RunState.PARTIAL_FAILED, RunState.COMPLETE_UNSEALED}
        ),
        RunState.PARTIAL_FAILED: frozenset({RunState.RUNNING}),
        RunState.COMPLETE_UNSEALED: frozenset(
            {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}
        ),
        RunState.SEALED_SUCCESS: frozenset(),
        RunState.SEALED_FAILED: frozenset(),
    }
    assert set(RECOVERY_CELL_TRANSITIONS) == set(CellState)
    assert RECOVERY_CELL_TRANSITIONS[CellState.ACCEPTED] == frozenset()
