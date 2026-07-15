from __future__ import annotations

import pytest

from src.utils.truth_isolation import (
    SealedTruthAccessError,
    TruthAccessTripwire,
    run_truth_free_fit,
)


def test_tripwire_records_before_raising() -> None:
    tripwire = TruthAccessTripwire()

    with pytest.raises(SealedTruthAccessError):
        tripwire.access("evaluator_truth_frame", "row 1")

    assert tripwire.attempted_access_count == 1
    assert tripwire.access_log[0]["resource"] == "evaluator_truth_frame"


def test_caught_truth_access_still_fails_the_fit_contract() -> None:
    tripwire = TruthAccessTripwire()

    def fit() -> str:
        try:
            tripwire.access("evaluator_truth_frame")
        except SealedTruthAccessError:
            pass
        return "fitted"

    with pytest.raises(AssertionError, match="attempted_access_count"):
        run_truth_free_fit(fit, tripwire=tripwire)


def test_clean_fit_has_no_truth_or_evaluator_loader_access() -> None:
    tripwire = TruthAccessTripwire()

    assert run_truth_free_fit(lambda: "fitted", tripwire=tripwire) == "fitted"
    assert tripwire.attempted_access_count == 0
    assert tripwire.evaluator_loader_call_count == 0
