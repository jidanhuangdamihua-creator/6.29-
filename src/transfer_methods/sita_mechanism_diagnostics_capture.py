"""Optional AE Stage-B capture for SITA mechanism diagnostics.

Default off. Formal A–E numbers are unchanged unless
``enable_mechanism_diagnostics`` is set on the mutable Stage-B context.

When enabled, B/C INIT snapshots are taken by the
``on_before_first_target_optimizer_update`` hook; FINAL snapshots are taken
after ``train_ms_katl_head`` returns (EMA already assigned). PRIMARY
C_INIT→C_FINAL and optional C_INIT→B_FINAL are assembled through the
fail-closed contract layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from src.transfer_methods.sita_mechanism_diagnostics import (
    assemble_c_init_b_final_frozen_reference,
    assemble_c_init_c_final_primary,
    digest_arrays,
    reference_input_digest,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    INITIAL_WEIGHT_VARIANT,
    MechanismCheckpoint,
    MechanismDiagnosticsContractError,
)

ENABLE_MECHANISM_DIAGNOSTICS_KEY = "enable_mechanism_diagnostics"
MECHANISM_DIAGNOSTICS_SESSION_KEY = "mechanism_diagnostics_session"
ACTIVE_DIAGNOSTIC_ARMS_KEY = "mechanism_diagnostics_arms"
FORMAL_DIAGNOSTIC_ARCHITECTURE = "SITA_CNN_V3"
DIAGNOSTIC_ARMS = ("A", "B", "C")
FORMAL_DIAGNOSTIC_ARMS = ("A", "B", "C", "D", "E")
MIN_REFERENCE_SAMPLES = 2
FORMAL_TEST_REFERENCE_SPLIT = "formal_test_inputs"
CKA_CONDITION_REFERENCE_STORE_KEY = "cka_condition_references"
CKA_REPRESENTATION_LAYERS = ("flatten", "conv1", "conv2", "conv3")
CKA_ACTIVATION_RESHAPE_RULE = "reshape(n_samples, -1)"


def mechanism_diagnostics_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping):
        return False
    return bool(context.get(ENABLE_MECHANISM_DIAGNOSTICS_KEY))


def active_diagnostic_arms(context: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    if not isinstance(context, Mapping):
        return DIAGNOSTIC_ARMS
    configured = context.get(ACTIVE_DIAGNOSTIC_ARMS_KEY, DIAGNOSTIC_ARMS)
    arms = tuple(str(arm).strip().upper() for arm in configured)
    if not arms or any(arm not in FORMAL_DIAGNOSTIC_ARMS for arm in arms):
        raise MechanismDiagnosticsContractError(f"invalid active diagnostic arms: {configured!r}")
    return arms


def flatten_named_conv(named: Mapping[str, Sequence[np.ndarray]]) -> Tuple[np.ndarray, ...]:
    arrays = []
    for name in sorted(named):
        for weight in named[name]:
            arrays.append(np.asarray(weight))
    if not arrays:
        raise MechanismDiagnosticsContractError("named conv snapshot is empty")
    return tuple(arrays)


def as_representation_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        raise MechanismDiagnosticsContractError(
            f"representation must be at least 2D, got shape {array.shape}"
        )
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def source_identity_strings(identities: Sequence[Any]) -> Tuple[str, ...]:
    labels = []
    for item in identities:
        if isinstance(item, (tuple, list)):
            labels.append("/".join(str(part) for part in item))
        else:
            labels.append(str(item))
    return tuple(labels)


def source_weights_digest(weights: Sequence[Any]) -> str:
    return digest_arrays([np.asarray(weights, dtype=np.float64)])


def select_reference_inputs(x_train: np.ndarray) -> np.ndarray:
    """Deprecated train-prefix helper. PRIMARY CKA must not call this."""
    raise MechanismDiagnosticsContractError(
        "PRIMARY CKA reference forbids X_train_prefix; use select_formal_test_reference_inputs"
    )


def _bundle_x_test(bundle: Any) -> np.ndarray:
    x_test = getattr(bundle, "X_test", None)
    if x_test is None and isinstance(bundle, Mapping):
        x_test = bundle.get("X_test")
    if x_test is None:
        raise MechanismDiagnosticsContractError(
            "CKA reference requires bundle.X_test (formal_test_inputs); X_train is not allowed"
        )
    return np.asarray(x_test)


def select_formal_test_reference_inputs(bundle: Any) -> np.ndarray:
    """Full formal test windows actually fed to the model. No train/val/prefix substitute."""
    array = _bundle_x_test(bundle)
    if array.ndim < 1 or int(array.shape[0]) < MIN_REFERENCE_SAMPLES:
        raise MechanismDiagnosticsContractError(
            "formal test CKA reference needs at least "
            f"{MIN_REFERENCE_SAMPLES} samples, got {0 if array.ndim < 1 else array.shape[0]}"
        )
    return np.array(array, copy=True)


def condition_reference_id(cell: Any) -> str:
    return f"{cell.dataset}|{cell.target}|h{int(cell.horizon)}"


def snapshot_backbone_from_constructed(constructed: Any) -> Tuple[np.ndarray, ...]:
    live_cnn = getattr(constructed, "target_cnn", None)
    if live_cnn is not None and hasattr(live_cnn, "get_layer"):
        from src.transfer_methods.ms_katl_lite import CONV_LAYER_NAMES, snapshot_named_layer_weights

        named = snapshot_named_layer_weights(live_cnn, CONV_LAYER_NAMES)
        return flatten_named_conv(named)
    named = getattr(constructed, "target_initial_conv", None)
    if named:
        return flatten_named_conv(named)
    raise MechanismDiagnosticsContractError("cannot snapshot constructed-arm backbone weights")


def forward_reference_from_constructed(
    constructed: Any,
    reference_inputs: np.ndarray,
) -> np.ndarray:
    from src.transfer_methods.ms_katl_lite import extract_features_maybe_revin

    z, _, _ = extract_features_maybe_revin(
        getattr(constructed, "target_extractor", None),
        reference_inputs,
        revin=getattr(constructed, "revin", None),
        training=False,
    )
    return as_representation_matrix(np.asarray(z))


def forward_named_representations_from_constructed(
    constructed: Any,
    reference_inputs: np.ndarray,
    *,
    flatten_rep: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Capture flatten + conv1/conv2/conv3 activations on the fixed test reference.

    Conv activations with ndim > 2 are reshaped with ``activation.reshape(n_samples, -1)``.
    The primary flatten representation continues to come from
    ``extract_features_maybe_revin`` / ``forward_reference_from_constructed``.
    """
    flatten = (
        as_representation_matrix(np.asarray(flatten_rep))
        if flatten_rep is not None
        else forward_reference_from_constructed(constructed, reference_inputs)
    )
    named: Dict[str, np.ndarray] = {"flatten": flatten}
    target_cnn = getattr(constructed, "target_cnn", None)
    if target_cnn is None or not hasattr(target_cnn, "layers"):
        return named
    import tensorflow as tf

    x_tf = tf.constant(np.asarray(reference_inputs), dtype=tf.float32)
    revin = getattr(constructed, "revin", None)
    if revin is not None:
        x_tf, _, _ = revin.normalize(x_tf)
    hidden = x_tf
    captured: Dict[str, np.ndarray] = {}
    for layer in target_cnn.layers:
        layer_name = str(getattr(layer, "name", "") or "")
        class_name = layer.__class__.__name__
        if class_name == "InputLayer" or layer_name in {"", "input"}:
            continue
        hidden = layer(hidden, training=False)
        if layer_name in ("conv1", "conv2", "conv3"):
            array = hidden.numpy() if hasattr(hidden, "numpy") else np.asarray(hidden)
            captured[layer_name] = as_representation_matrix(array)
    missing = [name for name in ("conv1", "conv2", "conv3") if name not in captured]
    if missing:
        raise MechanismDiagnosticsContractError(
            f"conv activation capture missing layers: {missing}"
        )
    named.update(captured)
    return named


def snapshot_named_layers_from_constructed(
    constructed: Any,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    if constructed is None:
        raise MechanismDiagnosticsContractError("cannot snapshot named layers without constructed arm")
    target_cnn = getattr(constructed, "target_cnn", None)
    head = getattr(constructed, "head", None)
    if target_cnn is None:
        raise MechanismDiagnosticsContractError("constructed arm is missing target_cnn")
    if head is None:
        raise MechanismDiagnosticsContractError("constructed arm is missing head")

    def _get_layer(model: Any, name: str) -> Any:
        getter = getattr(model, "get_layer", None)
        if callable(getter):
            try:
                layer = getter(name)
            except Exception as exc:
                raise MechanismDiagnosticsContractError(
                    f"constructed target_cnn is missing layer {name!r}"
                ) from exc
            if layer is not None:
                return layer
        for layer in getattr(model, "layers", ()) or ():
            if getattr(layer, "name", None) == name:
                return layer
        raise MechanismDiagnosticsContractError(
            f"constructed target_cnn is missing layer {name!r}"
        )

    def _copy_layer_weights(layer: Any, name: str) -> Tuple[np.ndarray, ...]:
        getter = getattr(layer, "get_weights", None)
        if not callable(getter):
            raise MechanismDiagnosticsContractError(
                f"constructed layer {name!r} cannot provide weights"
            )
        try:
            weights = getter()
        except Exception as exc:
            raise MechanismDiagnosticsContractError(
                f"constructed layer {name!r} cannot provide weights"
            ) from exc
        if not weights:
            raise MechanismDiagnosticsContractError(
                f"constructed layer {name!r} has no weights"
            )
        return tuple(np.array(weight, copy=True) for weight in weights)

    named: Dict[str, Tuple[np.ndarray, ...]] = {}
    for name in ("conv1", "conv2", "conv3"):
        named[name] = _copy_layer_weights(_get_layer(target_cnn, name), name)
    head_target = getattr(head, "head_target", None)
    if head_target is None:
        raise MechanismDiagnosticsContractError("constructed head is missing head_target")
    named["head_target"] = _copy_layer_weights(head_target, "head_target")
    return named


@dataclass
class MechanismDiagnosticsSession:
    dataset: str
    target: str
    horizon: int
    seed: int
    diagnostic_arms: Tuple[str, ...] = DIAGNOSTIC_ARMS
    architecture: str = FORMAL_DIAGNOSTIC_ARCHITECTURE
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT
    source_informed_init_digest: str = ""
    source_identities: Tuple[str, ...] = ()
    source_weights_digest: str = ""
    reference_inputs: Optional[np.ndarray] = None
    reference_input_digest: str = ""
    checkpoints: Dict[Tuple[str, str], MechanismCheckpoint] = field(default_factory=dict)
    backbone_weights: Dict[Tuple[str, str], Tuple[np.ndarray, ...]] = field(default_factory=dict)
    named_layer_weights: Dict[
        Tuple[str, str], Dict[str, Tuple[np.ndarray, ...]]
    ] = field(default_factory=dict)
    representations: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    named_representations: Dict[Tuple[str, str], Dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    primary: Optional[Any] = None
    frozen_reference: Optional[Dict[str, Any]] = None

    def capture_stage(
        self,
        *,
        arm: str,
        stage: str,
        backbone_weights: Sequence[np.ndarray],
        representations: np.ndarray,
        optimizer_update_count: int,
        captured_before_first_target_optimizer_update: bool,
        init_mode: str,
        backbone_trainable: Optional[bool],
        weight_variant: Optional[str] = None,
        ema_update_count: int = 0,
        named_representations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> MechanismCheckpoint:
        if arm not in self.diagnostic_arms:
            raise MechanismDiagnosticsContractError(f"diagnostics capture rejects arm={arm!r}")
        if self.reference_input_digest == "":
            raise MechanismDiagnosticsContractError("reference_input_digest is required before capture")
        weights = tuple(np.asarray(weight) for weight in backbone_weights)
        reps = as_representation_matrix(representations)
        if self.reference_inputs is not None and reps.shape[0] != int(self.reference_inputs.shape[0]):
            raise MechanismDiagnosticsContractError(
                "representation sample count does not match reference inputs: "
                f"{reps.shape[0]} vs {self.reference_inputs.shape[0]}"
            )
        key = (arm, stage)
        checkpoint = MechanismCheckpoint(
            arm=arm,
            stage=stage,
            dataset=self.dataset,
            target=self.target,
            horizon=int(self.horizon),
            seed=int(self.seed),
            architecture=self.architecture,
            reference_input_digest=self.reference_input_digest,
            source_informed_init_digest=self.source_informed_init_digest,
            source_identities=tuple(self.source_identities),
            source_weights_digest=self.source_weights_digest,
            backbone_digest=digest_arrays(weights),
            weight_variant=str(weight_variant or self.weight_variant),
            optimizer_update_count=int(optimizer_update_count),
            captured_before_first_target_optimizer_update=bool(
                captured_before_first_target_optimizer_update
            ),
            init_mode=init_mode,
            backbone_trainable=backbone_trainable,
            ema_update_count=int(ema_update_count),
        )
        self.checkpoints[key] = checkpoint
        self.backbone_weights[key] = weights
        self.representations[key] = reps
        if named_representations is not None:
            self.named_representations[key] = {
                str(layer): as_representation_matrix(np.asarray(array))
                for layer, array in named_representations.items()
            }
        else:
            self.named_representations[key] = {"flatten": reps}
        return checkpoint

    def maybe_assemble(self) -> None:
        c_init = self.checkpoints.get(("C", "INIT"))
        c_final = self.checkpoints.get(("C", "FINAL"))
        if c_init is not None and c_final is not None:
            self.primary = assemble_c_init_c_final_primary(
                c_init,
                c_final,
                init_backbone_weights=self.backbone_weights[("C", "INIT")],
                final_backbone_weights=self.backbone_weights[("C", "FINAL")],
                init_representations=self.representations[("C", "INIT")],
                final_representations=self.representations[("C", "FINAL")],
                prediction={
                    "weight_variant": self.weight_variant,
                    "source": "mechanism_diagnostics_capture",
                    "n_reference_samples": int(self.representations[("C", "INIT")].shape[0]),
                },
                weight_variant=self.weight_variant,
            )
        c_init = self.checkpoints.get(("C", "INIT"))
        b_init = self.checkpoints.get(("B", "INIT"))
        b_final = self.checkpoints.get(("B", "FINAL"))
        if c_init is not None and b_final is not None:
            self.frozen_reference = assemble_c_init_b_final_frozen_reference(
                c_init,
                b_final,
                b_init=b_init,
                init_backbone_weights=self.backbone_weights.get(("C", "INIT")),
                b_final_backbone_weights=self.backbone_weights.get(("B", "FINAL")),
            )

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": True,
            "dataset": self.dataset,
            "target": self.target,
            "horizon": int(self.horizon),
            "seed": int(self.seed),
            "diagnostic_arms": list(self.diagnostic_arms),
            "architecture": self.architecture,
            "weight_variant": self.weight_variant,
            "reference_input_digest": self.reference_input_digest,
            "source_informed_init_digest": self.source_informed_init_digest,
            "source_identities": list(self.source_identities),
            "source_weights_digest": self.source_weights_digest,
            "checkpoints": {
                f"{arm}_{stage}": {
                    "arm": checkpoint.arm,
                    "stage": checkpoint.stage,
                    "backbone_digest": checkpoint.backbone_digest,
                    "weight_variant": checkpoint.weight_variant,
                    "optimizer_update_count": checkpoint.optimizer_update_count,
                    "ema_update_count": checkpoint.ema_update_count,
                    "captured_before_first_target_optimizer_update": (
                        checkpoint.captured_before_first_target_optimizer_update
                    ),
                    "init_mode": checkpoint.init_mode,
                    "backbone_trainable": checkpoint.backbone_trainable,
                }
                for (arm, stage), checkpoint in sorted(self.checkpoints.items())
            },
            "primary": None if self.primary is None else self.primary.as_dict(),
            "frozen_reference": None if self.frozen_reference is None else dict(self.frozen_reference),
        }
        return payload


def session_from_cell(
    cell: Any,
    stage_a: Any,
    *,
    reference_inputs: np.ndarray,
    architecture: str = FORMAL_DIAGNOSTIC_ARCHITECTURE,
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
    diagnostic_arms: Sequence[str] = DIAGNOSTIC_ARMS,
) -> MechanismDiagnosticsSession:
    refs = np.asarray(reference_inputs)
    return MechanismDiagnosticsSession(
        dataset=str(cell.dataset),
        target=str(cell.target),
        horizon=int(cell.horizon),
        seed=int(cell.seed),
        diagnostic_arms=tuple(str(arm).strip().upper() for arm in diagnostic_arms),
        architecture=architecture,
        weight_variant=weight_variant,
        source_informed_init_digest=str(getattr(stage_a, "fused_target_digest", "") or ""),
        source_identities=source_identity_strings(getattr(stage_a, "source_identities", ())),
        source_weights_digest=source_weights_digest(getattr(stage_a, "source_weights", ())),
        reference_inputs=refs,
        reference_input_digest=reference_input_digest(refs),
    )


def ensure_session(
    context: MutableMapping[str, Any],
    *,
    cell: Any,
    stage_a: Any,
    bundle: Any,
) -> MechanismDiagnosticsSession:
    existing = context.get(MECHANISM_DIAGNOSTICS_SESSION_KEY)
    if isinstance(existing, MechanismDiagnosticsSession):
        return existing
    refs = select_formal_test_reference_inputs(bundle)
    digest = reference_input_digest(refs)
    store = context.setdefault(CKA_CONDITION_REFERENCE_STORE_KEY, {})
    cid = condition_reference_id(cell)
    prior = store.get(cid)
    if prior is not None:
        if str(prior.get("digest")) != digest:
            raise MechanismDiagnosticsContractError(
                f"one condition -> exactly one reference_input_digest violated for {cid}: "
                f"{prior.get('digest')!r} vs {digest!r}"
            )
        refs = np.asarray(prior["inputs"])
        digest = str(prior["digest"])
    else:
        store[cid] = {
            "inputs": refs,
            "digest": digest,
            "split": FORMAL_TEST_REFERENCE_SPLIT,
        }
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        assert_one_condition_one_digest,
        persist_condition_reference_if_needed,
    )

    assert_one_condition_one_digest(
        {key: [str(value["digest"])] for key, value in store.items()}
    )
    persist_condition_reference_if_needed(context, cell=cell, x_test=refs, digest=digest)
    session = session_from_cell(
        cell,
        stage_a,
        reference_inputs=refs,
        diagnostic_arms=active_diagnostic_arms(context),
    )
    if session.reference_input_digest != digest:
        raise MechanismDiagnosticsContractError("session digest drifted from condition store")
    context[MECHANISM_DIAGNOSTICS_SESSION_KEY] = session
    return session


def _arm_init_mode(constructed: Any, arm: str) -> str:
    spec = getattr(constructed, "spec", None)
    if spec is not None and getattr(spec, "init_mode", None):
        return str(spec.init_mode)
    return "fused" if arm in DIAGNOSTIC_ARMS else "native"


def _arm_backbone_trainable(constructed: Any, arm: str) -> Optional[bool]:
    spec = getattr(constructed, "spec", None)
    if spec is not None and hasattr(spec, "target_backbone_trainable"):
        return bool(spec.target_backbone_trainable)
    if arm == "B":
        return False
    if arm == "C":
        return True
    return None


def capture_live_stage(
    session: MechanismDiagnosticsSession,
    *,
    arm: str,
    stage: str,
    constructed: Any,
    optimizer_update_count: int,
    captured_before_first_target_optimizer_update: bool,
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
    ema_update_count: int = 0,
) -> MechanismCheckpoint:
    if session.reference_inputs is None:
        raise MechanismDiagnosticsContractError("session missing reference_inputs")
    named_layer_weights = snapshot_named_layers_from_constructed(constructed)
    flatten_rep = forward_reference_from_constructed(constructed, session.reference_inputs)
    named_reps = forward_named_representations_from_constructed(
        constructed,
        session.reference_inputs,
        flatten_rep=flatten_rep,
    )
    checkpoint = session.capture_stage(
        arm=arm,
        stage=stage,
        backbone_weights=snapshot_backbone_from_constructed(constructed),
        representations=flatten_rep,
        optimizer_update_count=optimizer_update_count,
        captured_before_first_target_optimizer_update=captured_before_first_target_optimizer_update,
        init_mode=_arm_init_mode(constructed, arm),
        backbone_trainable=_arm_backbone_trainable(constructed, arm),
        weight_variant=weight_variant,
        ema_update_count=int(ema_update_count),
        named_representations=named_reps,
    )
    session.named_layer_weights[(arm, stage)] = named_layer_weights
    return checkpoint


def stage_b_diagnostics_train_kwargs(
    context: Optional[Mapping[str, Any]],
    *,
    cell: Any = None,
    arm: str,
    stage_a: Any = None,
    constructed: Any = None,
    bundle: Any = None,
) -> Dict[str, Any]:
    """Return extra train_ms_katl_head kwargs. Empty when diagnostics are off."""
    if not mechanism_diagnostics_enabled(context) or arm not in active_diagnostic_arms(context):
        return {}
    if not isinstance(context, MutableMapping):
        raise MechanismDiagnosticsContractError("diagnostics capture requires a mutable context")
    if cell is None or stage_a is None or constructed is None or bundle is None:
        raise MechanismDiagnosticsContractError("enabled diagnostics require cell/stage_a/constructed/bundle")
    session = ensure_session(context, cell=cell, stage_a=stage_a, bundle=bundle)
    rows: list = context.setdefault("training_dynamics_rows", [])

    def _on_before(payload: Mapping[str, object]) -> None:
        capture_live_stage(
            session,
            arm=arm,
            stage="INIT",
            constructed=constructed,
            optimizer_update_count=int(payload.get("optimizer_update_count", 0)),
            captured_before_first_target_optimizer_update=bool(
                payload.get("captured_before_first_target_optimizer_update", False)
            ),
            weight_variant=INITIAL_WEIGHT_VARIANT,
            ema_update_count=int(payload.get("ema_update_count", 0)),
        )

    def _on_epoch(payload: Mapping[str, object]) -> None:
        from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
            record_diagnostic_epoch_eval,
        )

        record_diagnostic_epoch_eval(
            rows,
            payload=payload,
            constructed=constructed,
            bundle=bundle,
            cell=cell,
            arm=arm,
        )

    def _on_before_ema(payload: Mapping[str, object]) -> None:
        capture_live_stage(
            session,
            arm=arm,
            stage="RAW_FINAL",
            constructed=constructed,
            optimizer_update_count=int(payload.get("optimizer_update_count", 0)),
            captured_before_first_target_optimizer_update=False,
            weight_variant="RAW",
            ema_update_count=int(payload.get("ema_update_count", 0)),
        )

    extra: Dict[str, Any] = {
        "on_before_first_target_optimizer_update": _on_before,
        "on_diagnostic_epoch_eval": _on_epoch,
        "on_before_ema_assign": _on_before_ema,
    }
    x_val = getattr(bundle, "X_val", None)
    y_val = getattr(bundle, "y_val", None)
    f_t = getattr(bundle, "f_t", None)
    f_t_val = None if f_t is None else (f_t.get("val") if isinstance(f_t, Mapping) else None)
    if x_val is not None and y_val is not None and f_t_val is not None and len(np.asarray(y_val)) > 0:
        extra["x_val"] = x_val
        extra["y_val"] = y_val
        extra["f_t_val"] = f_t_val
    return extra


def finalize_stage_b_diagnostics(
    context: Optional[Mapping[str, Any]],
    *,
    arm: str,
    constructed: Any = None,
) -> None:
    if not mechanism_diagnostics_enabled(context) or arm not in active_diagnostic_arms(context):
        return
    if not isinstance(context, MutableMapping):
        raise MechanismDiagnosticsContractError("diagnostics finalize requires a mutable context")
    session = context.get(MECHANISM_DIAGNOSTICS_SESSION_KEY)
    if not isinstance(session, MechanismDiagnosticsSession):
        raise MechanismDiagnosticsContractError("diagnostics session missing at FINAL capture")
    if constructed is None:
        raise MechanismDiagnosticsContractError("FINAL capture requires the constructed arm")
    if (arm, "INIT") not in session.checkpoints:
        raise MechanismDiagnosticsContractError(
            f"{arm}_INIT was not captured before the first target optimizer update"
        )
    if (arm, "RAW_FINAL") not in session.checkpoints:
        raise MechanismDiagnosticsContractError(
            f"{arm}_RAW_FINAL was not captured before EMA assignment"
        )
    raw_final = session.checkpoints[(arm, "RAW_FINAL")]
    capture_live_stage(
        session,
        arm=arm,
        stage="FINAL",
        constructed=constructed,
        optimizer_update_count=int(raw_final.optimizer_update_count),
        captured_before_first_target_optimizer_update=False,
        ema_update_count=int(raw_final.ema_update_count),
    )
    session.maybe_assemble()
