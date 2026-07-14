from __future__ import annotations

import pytest

from src.utils.result_validation import CELL_TRANSITIONS, validate_cell_transition


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
