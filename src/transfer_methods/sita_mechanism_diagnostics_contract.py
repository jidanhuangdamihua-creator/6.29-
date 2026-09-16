"""Fail-closed pairing / isolation / comparability contracts for SITA mechanism diagnostics.

These helpers do not train models and do not rewrite frozen A–E numbers.
They only decide whether a diagnostic comparison is legal to emit.

Formal A–E / SITA Stage-B evaluation uses EMA weights. PRIMARY diagnostics
must therefore stay on a single weight_variant and must not mix RAW with EMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

WEIGHT_VARIANTS = ("EMA", "RAW")
PRIMARY_ARTIFACT_KINDS = ("prediction", "parameter_drift", "CKA")
CHECKPOINT_STAGES = ("INIT", "FINAL")
FORMAL_EVALUATION_WEIGHT_VARIANT = "EMA"

COMPARISON_C_INIT_C_FINAL = "C_INIT_C_FINAL"
COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT = "C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT"
COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE = "C_INIT_B_FINAL_FROZEN_REFERENCE"
COMPARISON_B_C_MATCHED_SEED = "B_C_MATCHED_SEED"
COMPARISON_A_C_MATCHED_SEED = "A_C_MATCHED_SEED"
COMPARISON_B_B_CROSS_SEED = "B_B_CROSS_SEED"
COMPARISON_C_C_CROSS_SEED = "C_C_CROSS_SEED"
CANONICAL_CKA_COMPARISONS = (
    COMPARISON_B_C_MATCHED_SEED,
    COMPARISON_A_C_MATCHED_SEED,
    COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_B_B_CROSS_SEED,
    COMPARISON_C_C_CROSS_SEED,
)

PAIRING_IDENTITY_FIELDS = (
    "dataset",
    "target",
    "horizon",
    "seed",
    "architecture",
    "reference_input_digest",
)

SOURCE_INFORMED_INIT_IDENTITY_FIELDS = (
    "dataset",
    "target",
    "horizon",
    "seed",
    "architecture",
    "init_mode",
    "source_informed_init_digest",
    "source_identities",
    "source_weights_digest",
)


class MechanismDiagnosticsContractError(ValueError):
    """Raised when a diagnostic pairing or isolation contract is violated."""


@dataclass(frozen=True)
class MechanismCheckpoint:
    arm: str
    stage: str
    dataset: str
    target: str
    horizon: int
    seed: int
    architecture: str
    reference_input_digest: str
    source_informed_init_digest: str
    source_identities: Tuple[str, ...]
    source_weights_digest: str
    backbone_digest: str
    weight_variant: str
    optimizer_update_count: int
    captured_before_first_target_optimizer_update: bool
    init_mode: str = "fused"
    backbone_trainable: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.arm not in {"A", "B", "C"}:
            raise MechanismDiagnosticsContractError(f"unsupported diagnostic arm={self.arm!r}")
        if self.stage not in CHECKPOINT_STAGES:
            raise MechanismDiagnosticsContractError(f"unsupported checkpoint stage={self.stage!r}")
        if self.weight_variant not in WEIGHT_VARIANTS:
            raise MechanismDiagnosticsContractError(
                f"unsupported weight_variant={self.weight_variant!r}"
            )
        if int(self.optimizer_update_count) < 0:
            raise MechanismDiagnosticsContractError("optimizer_update_count must be >= 0")


@dataclass(frozen=True)
class PrimaryDiagnosticArtifact:
    kind: str
    weight_variant: str
    dataset: str
    target: str
    horizon: int
    seed: int
    architecture: str
    reference_input_digest: str
    comparison: str
    role: str = "PRIMARY"

    def __post_init__(self) -> None:
        if self.kind not in PRIMARY_ARTIFACT_KINDS:
            raise MechanismDiagnosticsContractError(
                f"unsupported PRIMARY artifact kind={self.kind!r}"
            )
        if self.weight_variant not in WEIGHT_VARIANTS:
            raise MechanismDiagnosticsContractError(
                f"unsupported weight_variant={self.weight_variant!r}"
            )
        if self.role != "PRIMARY":
            raise MechanismDiagnosticsContractError(
                f"PRIMARY artifact role must be 'PRIMARY', got {self.role!r}"
            )


@dataclass(frozen=True)
class DiagnosticDecision:
    status: str
    comparison: str
    reason: Optional[str] = None
    identity: Optional[Mapping[str, Any]] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def generated(self) -> bool:
        return self.status == "GENERATED"


def _identity_payload(record: MechanismCheckpoint) -> Dict[str, Any]:
    return {
        "dataset": record.dataset,
        "target": record.target,
        "horizon": int(record.horizon),
        "seed": int(record.seed),
        "architecture": record.architecture,
        "reference_input_digest": record.reference_input_digest,
    }


def _source_informed_init_payload(record: MechanismCheckpoint) -> Dict[str, Any]:
    return {
        "dataset": record.dataset,
        "target": record.target,
        "horizon": int(record.horizon),
        "seed": int(record.seed),
        "architecture": record.architecture,
        "init_mode": record.init_mode,
        "source_informed_init_digest": record.source_informed_init_digest,
        "source_identities": tuple(record.source_identities),
        "source_weights_digest": record.source_weights_digest,
    }


def _field_mismatch(left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]) -> Optional[str]:
    for name in fields:
        if left[name] != right[name]:
            return (
                f"{name} mismatch: {left[name]!r} != {right[name]!r}"
            )
    return None


def assert_c_init_captured_before_first_target_optimizer_update(
    checkpoint: MechanismCheckpoint,
) -> None:
    """C_INIT must be taken before the first target optimizer update."""
    if checkpoint.arm != "C" or checkpoint.stage != "INIT":
        raise MechanismDiagnosticsContractError(
            "C_INIT timing check requires arm='C' and stage='INIT'"
        )
    if int(checkpoint.optimizer_update_count) != 0:
        raise MechanismDiagnosticsContractError(
            "C_INIT checkpoint occurred after a target optimizer update: "
            f"optimizer_update_count={checkpoint.optimizer_update_count}"
        )
    if not checkpoint.captured_before_first_target_optimizer_update:
        raise MechanismDiagnosticsContractError(
            "C_INIT checkpoint is not marked as captured before the first "
            "target optimizer update"
        )


def pair_c_init_c_final(
    c_init: MechanismCheckpoint,
    c_final: MechanismCheckpoint,
) -> DiagnosticDecision:
    """Pair C_INIT with C_FINAL only when identity and timing contracts hold."""
    if c_init.arm != "C" or c_init.stage != "INIT":
        raise MechanismDiagnosticsContractError(
            f"left checkpoint must be C_INIT, got arm={c_init.arm!r} stage={c_init.stage!r}"
        )
    if c_final.arm != "C" or c_final.stage != "FINAL":
        raise MechanismDiagnosticsContractError(
            f"right checkpoint must be C_FINAL, got arm={c_final.arm!r} stage={c_final.stage!r}"
        )
    mismatch = _field_mismatch(
        _identity_payload(c_init),
        _identity_payload(c_final),
        PAIRING_IDENTITY_FIELDS,
    )
    if mismatch is not None:
        raise MechanismDiagnosticsContractError(
            f"C_INIT → C_FINAL pairing rejected: {mismatch}"
        )
    assert_c_init_captured_before_first_target_optimizer_update(c_init)
    if c_final.captured_before_first_target_optimizer_update:
        raise MechanismDiagnosticsContractError(
            "C_FINAL cannot be marked as captured before the first target optimizer update"
        )
    if int(c_final.optimizer_update_count) <= 0:
        raise MechanismDiagnosticsContractError(
            "C_FINAL must occur after at least one target optimizer update, "
            f"got optimizer_update_count={c_final.optimizer_update_count}"
        )
    return DiagnosticDecision(
        status="GENERATED",
        comparison=COMPARISON_C_INIT_C_FINAL,
        identity=_identity_payload(c_init),
        payload={
            "c_init_backbone_digest": c_init.backbone_digest,
            "c_final_backbone_digest": c_final.backbone_digest,
            "c_init_optimizer_update_count": int(c_init.optimizer_update_count),
            "c_final_optimizer_update_count": int(c_final.optimizer_update_count),
            "before_first_target_optimizer_update": True,
        },
    )


def isolate_primary_weight_variant(
    artifacts: Sequence[PrimaryDiagnosticArtifact],
    *,
    formal_evaluation_weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
) -> str:
    """Require PRIMARY prediction / parameter drift / CKA to share one weight_variant."""
    if formal_evaluation_weight_variant not in WEIGHT_VARIANTS:
        raise MechanismDiagnosticsContractError(
            f"unsupported formal evaluation weight_variant="
            f"{formal_evaluation_weight_variant!r}"
        )
    by_kind = {kind: None for kind in PRIMARY_ARTIFACT_KINDS}
    for artifact in artifacts:
        if artifact.role != "PRIMARY":
            raise MechanismDiagnosticsContractError(
                f"non-PRIMARY artifact leaked into isolation check: {artifact.role!r}"
            )
        if by_kind[artifact.kind] is not None:
            raise MechanismDiagnosticsContractError(
                f"duplicate PRIMARY artifact kind={artifact.kind!r}"
            )
        by_kind[artifact.kind] = artifact
    missing = [kind for kind, artifact in by_kind.items() if artifact is None]
    if missing:
        raise MechanismDiagnosticsContractError(
            f"PRIMARY artifacts missing {missing}; required={list(PRIMARY_ARTIFACT_KINDS)}"
        )

    variants = {artifact.weight_variant for artifact in by_kind.values()}
    if len(variants) != 1:
        detail = {
            kind: artifact.weight_variant for kind, artifact in by_kind.items()
        }
        raise MechanismDiagnosticsContractError(
            "PRIMARY prediction / parameter drift / CKA weight_variant mismatch: "
            f"{detail}"
        )
    shared = next(iter(variants))
    if formal_evaluation_weight_variant == "EMA" and shared != "EMA":
        raise MechanismDiagnosticsContractError(
            "formal evaluation is EMA, so PRIMARY outputs must use EMA; "
            f"got PRIMARY weight_variant={shared!r}"
        )
    if formal_evaluation_weight_variant == "EMA":
        mixed = [
            kind
            for kind, artifact in by_kind.items()
            if artifact.weight_variant != "EMA"
        ]
        if mixed:
            raise MechanismDiagnosticsContractError(
                "formal evaluation is EMA; PRIMARY must not mix RAW/EMA. "
                f"RAW leaked in {mixed}"
            )
    return shared


def source_informed_initialization_comparable(
    left: MechanismCheckpoint,
    right: MechanismCheckpoint,
) -> Optional[str]:
    """Return an explicit incomparability reason, or None when comparable."""
    mismatch = _field_mismatch(
        _source_informed_init_payload(left),
        _source_informed_init_payload(right),
        SOURCE_INFORMED_INIT_IDENTITY_FIELDS,
    )
    if mismatch is not None:
        return f"source-informed initialization identity not comparable: {mismatch}"
    if left.init_mode != "fused" or right.init_mode != "fused":
        return (
            "source-informed initialization identity not comparable: "
            f"init_mode left={left.init_mode!r} right={right.init_mode!r}"
        )
    return None


def generate_c_init_b_final_frozen_reference(
    c_init: MechanismCheckpoint,
    b_final: MechanismCheckpoint,
    *,
    b_init: Optional[MechanismCheckpoint] = None,
) -> DiagnosticDecision:
    """Emit C_INIT_B_FINAL_FROZEN_REFERENCE only when B/C init identity is comparable.

    Otherwise skip with an explicit reason. Never force the comparison.
    """
    if c_init.arm != "C" or c_init.stage != "INIT":
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "left checkpoint is not C_INIT "
                f"(arm={c_init.arm!r}, stage={c_init.stage!r})"
            ),
        )
    if b_final.arm != "B" or b_final.stage != "FINAL":
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "right checkpoint is not B_FINAL "
                f"(arm={b_final.arm!r}, stage={b_final.stage!r})"
            ),
        )

    try:
        assert_c_init_captured_before_first_target_optimizer_update(c_init)
    except MechanismDiagnosticsContractError as exc:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=str(exc),
        )

    incomparable = source_informed_initialization_comparable(c_init, b_final)
    if incomparable is not None:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=incomparable,
            identity=_identity_payload(c_init),
        )

    if b_init is None:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "B_INIT evidence is required to confirm B/C source-informed "
                "initialization identity on the compared backbone objects"
            ),
            identity=_identity_payload(c_init),
        )
    if b_init.arm != "B" or b_init.stage != "INIT":
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "supplied B_INIT checkpoint is not arm='B' stage='INIT' "
                f"(arm={b_init.arm!r}, stage={b_init.stage!r})"
            ),
            identity=_identity_payload(c_init),
        )

    init_incomparable = source_informed_initialization_comparable(c_init, b_init)
    if init_incomparable is not None:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=init_incomparable,
            identity=_identity_payload(c_init),
        )
    if c_init.backbone_digest != b_init.backbone_digest:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "C_INIT backbone digest does not match B_INIT backbone digest; "
                "source-informed initialization objects are not comparable"
            ),
            identity=_identity_payload(c_init),
        )
    if b_final.backbone_digest != b_init.backbone_digest:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "B_FINAL backbone digest drifted from B_INIT; frozen-backbone "
                "reference is not comparable"
            ),
            identity=_identity_payload(c_init),
        )
    if b_final.backbone_trainable is True:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason="B_FINAL backbone is marked trainable, so it is not a frozen reference",
            identity=_identity_payload(c_init),
        )
    if c_init.backbone_digest != b_final.backbone_digest:
        return DiagnosticDecision(
            status="SKIPPED",
            comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            reason=(
                "C_INIT backbone digest does not match B_FINAL backbone digest "
                "even though B was expected to remain at the shared fused initialization"
            ),
            identity=_identity_payload(c_init),
        )

    return DiagnosticDecision(
        status="GENERATED",
        comparison=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
        identity=_identity_payload(c_init),
        payload={
            "c_init_backbone_digest": c_init.backbone_digest,
            "b_init_backbone_digest": b_init.backbone_digest,
            "b_final_backbone_digest": b_final.backbone_digest,
            "source_informed_init_digest": c_init.source_informed_init_digest,
            "frozen_backbone_confirmed": True,
        },
    )


def refuse_forced_c_init_b_final_comparison(
    c_init: MechanismCheckpoint,
    b_final: MechanismCheckpoint,
    *,
    b_init: Optional[MechanismCheckpoint] = None,
    force: bool = False,
) -> DiagnosticDecision:
    """Ignore force=True. Incomparable B/C init identity must skip, never emit."""
    del force
    return generate_c_init_b_final_frozen_reference(c_init, b_final, b_init=b_init)
