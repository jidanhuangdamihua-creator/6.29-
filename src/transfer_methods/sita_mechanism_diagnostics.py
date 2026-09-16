"""Mechanism diagnostics: linear CKA, parameter drift, and PRIMARY assembly.

This module does not train models and does not rewrite frozen A–E numbers.
It only computes diagnostics from already captured checkpoints/representations
and refuses illegal pairings via the contract layer.

Formal evaluation is EMA. PRIMARY prediction / parameter drift / CKA must
share that weight_variant.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    DiagnosticDecision,
    MechanismCheckpoint,
    MechanismDiagnosticsContractError,
    PrimaryDiagnosticArtifact,
    generate_c_init_b_final_frozen_reference,
    isolate_primary_weight_variant,
    pair_c_init_c_final,
)

WEIGHT_DRIFT_EPS = 1e-12
CKA_DEGENERATE_EPS = 1e-12
CKA_NUMERICAL_TOL = 1e-8


def digest_arrays(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def reference_input_digest(inputs: np.ndarray) -> str:
    return digest_arrays([np.asarray(inputs)])


def _as_weight_list(weights: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(weight) for weight in weights)
    if not arrays:
        raise MechanismDiagnosticsContractError("weight snapshot is empty")
    return arrays


def parameter_l2_distance(
    initial_weights: Sequence[np.ndarray],
    final_weights: Sequence[np.ndarray],
) -> float:
    initial = _as_weight_list(initial_weights)
    final = _as_weight_list(final_weights)
    if len(initial) != len(final):
        raise MechanismDiagnosticsContractError(
            "parameter drift requires matching snapshot lengths: "
            f"{len(initial)} vs {len(final)}"
        )
    delta_sq = 0.0
    for left, right in zip(initial, final):
        if left.shape != right.shape:
            raise MechanismDiagnosticsContractError(
                f"parameter drift shape mismatch: {left.shape} vs {right.shape}"
            )
        delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
        delta_sq += float(np.sum(delta * delta))
    return float(np.sqrt(delta_sq))


def relative_parameter_drift(
    initial_weights: Sequence[np.ndarray],
    final_weights: Sequence[np.ndarray],
    eps: float = WEIGHT_DRIFT_EPS,
) -> float:
    initial = _as_weight_list(initial_weights)
    l2 = parameter_l2_distance(initial, final_weights)
    init_sq = 0.0
    for left in initial:
        values = np.asarray(left, dtype=np.float64)
        init_sq += float(np.sum(values * values))
    return float(l2 / (np.sqrt(init_sq) + float(eps)))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA on sample-aligned representations, columns centered.

    X, Y have shape (n_samples, n_features). Samples must be the same
    reference inputs used for the paired INIT/FINAL forward.
    """
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise MechanismDiagnosticsContractError(
            f"CKA requires 2D representation matrices, got {left.ndim}D and {right.ndim}D"
        )
    if left.shape[0] != right.shape[0]:
        raise MechanismDiagnosticsContractError(
            "CKA sample axis mismatch: representations were not computed on the "
            f"same reference inputs ({left.shape[0]} vs {right.shape[0]})"
        )
    if left.shape[0] < 2:
        raise MechanismDiagnosticsContractError("CKA requires at least 2 reference samples")

    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = right.T @ left
    gram_x = left.T @ left
    gram_y = right.T @ right
    numerator = float(np.sum(cross * cross))
    denom_x = float(np.sqrt(np.sum(gram_x * gram_x)))
    denom_y = float(np.sqrt(np.sum(gram_y * gram_y)))
    if denom_x <= CKA_DEGENERATE_EPS or denom_y <= CKA_DEGENERATE_EPS:
        raise MechanismDiagnosticsContractError(
            "CKA is degenerate: at least one centered representation has near-zero gram"
        )
    value = numerator / (denom_x * denom_y)
    if value < -CKA_NUMERICAL_TOL or value > 1.0 + CKA_NUMERICAL_TOL:
        raise MechanismDiagnosticsContractError(f"CKA outside [0, 1]: {value}")
    return float(min(1.0, max(0.0, value)))


def _require_shared_weight_variant(
    *records: MechanismCheckpoint,
    weight_variant: str,
) -> None:
    for record in records:
        if record.weight_variant != weight_variant:
            raise MechanismDiagnosticsContractError(
                "checkpoint weight_variant does not match PRIMARY "
                f"weight_variant={weight_variant!r}: "
                f"arm={record.arm} stage={record.stage} "
                f"weight_variant={record.weight_variant!r}"
            )


def _primary_artifact(
    kind: str,
    checkpoint: MechanismCheckpoint,
    *,
    comparison: str,
    weight_variant: str,
) -> PrimaryDiagnosticArtifact:
    return PrimaryDiagnosticArtifact(
        kind=kind,
        weight_variant=weight_variant,
        dataset=checkpoint.dataset,
        target=checkpoint.target,
        horizon=int(checkpoint.horizon),
        seed=int(checkpoint.seed),
        architecture=checkpoint.architecture,
        reference_input_digest=checkpoint.reference_input_digest,
        comparison=comparison,
    )


@dataclass(frozen=True)
class PrimaryDiagnosticBundle:
    pairing: DiagnosticDecision
    weight_variant: str
    parameter_l2: float
    relative_parameter_drift: float
    cka: float
    prediction: Mapping[str, Any]
    artifacts: Tuple[PrimaryDiagnosticArtifact, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.pairing.status,
            "comparison": self.pairing.comparison,
            "weight_variant": self.weight_variant,
            "parameter_l2": self.parameter_l2,
            "relative_parameter_drift": self.relative_parameter_drift,
            "cka": self.cka,
            "prediction": dict(self.prediction),
            "identity": dict(self.pairing.identity or {}),
        }


def assemble_c_init_c_final_primary(
    c_init: MechanismCheckpoint,
    c_final: MechanismCheckpoint,
    *,
    init_backbone_weights: Sequence[np.ndarray],
    final_backbone_weights: Sequence[np.ndarray],
    init_representations: np.ndarray,
    final_representations: np.ndarray,
    prediction: Mapping[str, Any],
    weight_variant: str,
    formal_evaluation_weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
) -> PrimaryDiagnosticBundle:
    """Build PRIMARY prediction / drift / CKA for a legal C_INIT → C_FINAL pair."""
    pairing = pair_c_init_c_final(c_init, c_final)
    _require_shared_weight_variant(c_init, c_final, weight_variant=weight_variant)
    artifacts = (
        _primary_artifact("prediction", c_init, comparison=COMPARISON_C_INIT_C_FINAL, weight_variant=weight_variant),
        _primary_artifact(
            "parameter_drift", c_init, comparison=COMPARISON_C_INIT_C_FINAL, weight_variant=weight_variant
        ),
        _primary_artifact("CKA", c_init, comparison=COMPARISON_C_INIT_C_FINAL, weight_variant=weight_variant),
    )
    isolate_primary_weight_variant(
        artifacts,
        formal_evaluation_weight_variant=formal_evaluation_weight_variant,
    )
    if digest_arrays(init_backbone_weights) != c_init.backbone_digest:
        raise MechanismDiagnosticsContractError(
            "C_INIT backbone weights do not match the checkpoint backbone_digest"
        )
    if digest_arrays(final_backbone_weights) != c_final.backbone_digest:
        raise MechanismDiagnosticsContractError(
            "C_FINAL backbone weights do not match the checkpoint backbone_digest"
        )
    return PrimaryDiagnosticBundle(
        pairing=pairing,
        weight_variant=weight_variant,
        parameter_l2=parameter_l2_distance(init_backbone_weights, final_backbone_weights),
        relative_parameter_drift=relative_parameter_drift(
            init_backbone_weights, final_backbone_weights
        ),
        cka=linear_cka(init_representations, final_representations),
        prediction=dict(prediction),
        artifacts=artifacts,
    )


def assemble_c_init_b_final_frozen_reference(
    c_init: MechanismCheckpoint,
    b_final: MechanismCheckpoint,
    *,
    b_init: Optional[MechanismCheckpoint] = None,
    init_backbone_weights: Optional[Sequence[np.ndarray]] = None,
    b_final_backbone_weights: Optional[Sequence[np.ndarray]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate the frozen-B reference only when B/C init identity is comparable."""
    del force
    decision = generate_c_init_b_final_frozen_reference(c_init, b_final, b_init=b_init)
    if not decision.generated:
        return {
            "status": "SKIPPED",
            "comparison": COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
            "reason": decision.reason,
            "identity": dict(decision.identity or {}),
        }
    payload: Dict[str, Any] = {
        "status": "GENERATED",
        "comparison": COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
        "reason": None,
        "identity": dict(decision.identity or {}),
        "payload": dict(decision.payload),
    }
    if init_backbone_weights is not None and b_final_backbone_weights is not None:
        payload["parameter_l2"] = parameter_l2_distance(
            init_backbone_weights, b_final_backbone_weights
        )
        payload["relative_parameter_drift"] = relative_parameter_drift(
            init_backbone_weights, b_final_backbone_weights
        )
    return payload
