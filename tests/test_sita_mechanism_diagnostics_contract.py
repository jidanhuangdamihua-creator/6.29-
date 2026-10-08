"""Contract tests 13–15 for SITA mechanism diagnostics pairing and isolation.

No training. These tests only exercise fail-closed pairing, EMA/RAW isolation,
and C_INIT → B_FINAL comparability decisions.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismCheckpoint,
    MechanismDiagnosticsContractError,
    PrimaryDiagnosticArtifact,
    assert_c_init_captured_before_first_target_optimizer_update,
    generate_c_init_b_final_frozen_reference,
    isolate_primary_weight_variant,
    pair_c_init_c_final,
    refuse_forced_c_init_b_final_comparison,
)


def _c_init(**overrides) -> MechanismCheckpoint:
    base = MechanismCheckpoint(
        arm="C",
        stage="INIT",
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        architecture="SITA_CNN_V3",
        reference_input_digest="ref-input-1",
        source_informed_init_digest="fused-init-1",
        source_identities=("src-a", "src-b", "src-c"),
        source_weights_digest="weights-1",
        backbone_digest="backbone-init-1",
        weight_variant="EMA",
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        init_mode="fused",
        backbone_trainable=True,
    )
    return replace(base, **overrides)


def _c_final(**overrides) -> MechanismCheckpoint:
    values = {
        "stage": "FINAL",
        "optimizer_update_count": 12,
        "captured_before_first_target_optimizer_update": False,
        "backbone_digest": "backbone-final-c",
    }
    values.update(overrides)
    return _c_init(**values)


def _b_init(**overrides) -> MechanismCheckpoint:
    values = {
        "arm": "B",
        "backbone_trainable": False,
    }
    values.update(overrides)
    return _c_init(**values)


def _b_final(**overrides) -> MechanismCheckpoint:
    values = {
        "arm": "B",
        "stage": "FINAL",
        "optimizer_update_count": 12,
        "captured_before_first_target_optimizer_update": False,
        "backbone_trainable": False,
    }
    values.update(overrides)
    return _c_init(**values)


def _primary(kind: str, weight_variant: str = "EMA") -> PrimaryDiagnosticArtifact:
    return PrimaryDiagnosticArtifact(
        kind=kind,
        weight_variant=weight_variant,
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        architecture="SITA_CNN_V3",
        reference_input_digest="ref-input-1",
        comparison=COMPARISON_C_INIT_C_FINAL,
    )


def _primary_set(weight_variant: str = "EMA"):
    return (
        _primary("prediction", weight_variant),
        _primary("parameter_drift", weight_variant),
        _primary("CKA", weight_variant),
    )


# ---------------------------------------------------------------------------
# 13. C_INIT → C_FINAL pairing
# ---------------------------------------------------------------------------


def test_c_init_c_final_pairs_when_identity_and_timing_match():
    decision = pair_c_init_c_final(_c_init(), _c_final())
    assert decision.status == "GENERATED"
    assert decision.comparison == COMPARISON_C_INIT_C_FINAL
    assert decision.identity == {
        "dataset": "D1",
        "target": "1/10",
        "horizon": 1,
        "seed": 42,
        "architecture": "SITA_CNN_V3",
        "reference_input_digest": "ref-input-1",
    }
    assert decision.payload["before_first_target_optimizer_update"] is True


@pytest.mark.parametrize(
    "field, value",
    [
        ("dataset", "D6"),
        ("target", "CA_1/FOODS_3_080"),
        ("horizon", 5),
        ("seed", 46),
        ("architecture", "SITA_TCN"),
        ("reference_input_digest", "ref-input-OTHER"),
    ],
)
def test_c_init_c_final_rejects_identity_mismatch(field, value):
    with pytest.raises(MechanismDiagnosticsContractError, match=f"{field} mismatch"):
        pair_c_init_c_final(_c_init(), _c_final(**{field: value}))


def test_c_init_must_occur_before_first_target_optimizer_update():
    late = _c_init(
        optimizer_update_count=1,
        captured_before_first_target_optimizer_update=False,
    )
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="after a target optimizer update",
    ):
        assert_c_init_captured_before_first_target_optimizer_update(late)
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="after a target optimizer update",
    ):
        pair_c_init_c_final(late, _c_final())


def test_c_init_rejects_zero_updates_without_before_flag():
    unmarked = _c_init(captured_before_first_target_optimizer_update=False)
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="not marked as captured before the first",
    ):
        pair_c_init_c_final(unmarked, _c_final())


def test_c_final_must_be_after_an_optimizer_update():
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="after at least one target optimizer update",
    ):
        pair_c_init_c_final(
            _c_init(),
            _c_final(
                optimizer_update_count=0,
                captured_before_first_target_optimizer_update=False,
            ),
        )


def test_c_final_cannot_carry_before_first_update_flag():
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="C_FINAL cannot be marked as captured before",
    ):
        pair_c_init_c_final(
            _c_init(),
            _c_final(captured_before_first_target_optimizer_update=True),
        )


# ---------------------------------------------------------------------------
# 14. EMA / RAW isolation
# ---------------------------------------------------------------------------


def test_formal_evaluation_is_ema():
    assert FORMAL_EVALUATION_WEIGHT_VARIANT == "EMA"


def test_primary_prediction_drift_cka_must_share_weight_variant():
    shared = isolate_primary_weight_variant(_primary_set("EMA"))
    assert shared == "EMA"


def test_primary_all_raw_is_rejected_when_formal_evaluation_is_ema():
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="PRIMARY outputs must use EMA",
    ):
        isolate_primary_weight_variant(
            _primary_set("RAW"),
            formal_evaluation_weight_variant="EMA",
        )


def test_primary_rejects_raw_ema_mixture_when_formal_evaluation_is_ema():
    mixed = (
        _primary("prediction", "EMA"),
        _primary("parameter_drift", "RAW"),
        _primary("CKA", "EMA"),
    )
    with pytest.raises(
        MechanismDiagnosticsContractError,
        match="weight_variant mismatch",
    ):
        isolate_primary_weight_variant(
            mixed,
            formal_evaluation_weight_variant="EMA",
        )


def test_primary_all_raw_is_legal_only_when_formal_evaluation_is_raw():
    shared = isolate_primary_weight_variant(
        _primary_set("RAW"),
        formal_evaluation_weight_variant="RAW",
    )
    assert shared == "RAW"


def test_primary_requires_all_three_artifacts():
    with pytest.raises(MechanismDiagnosticsContractError, match="missing"):
        isolate_primary_weight_variant(
            (_primary("prediction", "EMA"), _primary("CKA", "EMA"))
        )


# ---------------------------------------------------------------------------
# 15. C_INIT → B_FINAL comparability
# ---------------------------------------------------------------------------


def test_c_init_b_final_generated_only_when_init_identity_is_comparable():
    decision = generate_c_init_b_final_frozen_reference(
        _c_init(),
        _b_final(),
        b_init=_b_init(),
    )
    assert decision.status == "GENERATED"
    assert decision.comparison == COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert decision.payload["frozen_backbone_confirmed"] is True
    assert decision.reason is None


def test_c_init_b_final_skips_when_source_informed_init_digest_differs():
    decision = generate_c_init_b_final_frozen_reference(
        _c_init(),
        _b_final(source_informed_init_digest="fused-init-OTHER"),
        b_init=_b_init(),
    )
    assert decision.status == "SKIPPED"
    assert decision.comparison == COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert "source-informed initialization identity not comparable" in decision.reason
    assert "source_informed_init_digest mismatch" in decision.reason
    assert decision.generated is False


def test_c_init_b_final_skips_when_source_identities_differ():
    decision = generate_c_init_b_final_frozen_reference(
        _c_init(),
        _b_final(source_identities=("src-x", "src-y", "src-z")),
        b_init=_b_init(),
    )
    assert decision.status == "SKIPPED"
    assert "source_identities mismatch" in decision.reason


def test_c_init_b_final_skips_without_b_init_evidence():
    decision = generate_c_init_b_final_frozen_reference(_c_init(), _b_final())
    assert decision.status == "SKIPPED"
    assert "B_INIT evidence is required" in decision.reason


def test_c_init_b_final_skips_when_c_init_backbone_differs_from_b_init():
    decision = generate_c_init_b_final_frozen_reference(
        _c_init(backbone_digest="c-only"),
        _b_final(),
        b_init=_b_init(),
    )
    assert decision.status == "SKIPPED"
    assert "C_INIT backbone digest does not match B_INIT" in decision.reason


def test_c_init_b_final_skips_when_b_backbone_is_not_frozen():
    drifted = generate_c_init_b_final_frozen_reference(
        _c_init(),
        _b_final(backbone_digest="b-drifted"),
        b_init=_b_init(),
    )
    assert drifted.status == "SKIPPED"
    assert "drifted from B_INIT" in drifted.reason

    trainable = generate_c_init_b_final_frozen_reference(
        _c_init(),
        _b_final(backbone_trainable=True),
        b_init=_b_init(),
    )
    assert trainable.status == "SKIPPED"
    assert "marked trainable" in trainable.reason


def test_c_init_b_final_never_emits_when_force_requested():
    forced = refuse_forced_c_init_b_final_comparison(
        _c_init(),
        _b_final(source_informed_init_digest="not-comparable"),
        b_init=_b_init(),
        force=True,
    )
    assert forced.status == "SKIPPED"
    assert forced.comparison == COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert forced.generated is False
    assert "source-informed initialization identity not comparable" in forced.reason
