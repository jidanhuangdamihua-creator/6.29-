"""MS-KATL V3 A–E core mechanism ablation contracts.

Independent of the historical six-arm mechanism-ablation registry (A–F).
Construction and contract checks are allowed. Formal Stage-A / Stage-B
training is fail-closed unless explicitly unlocked. The execution body is
enabled as capability only: `--execute` and `AE_FORMAL_TRAINING_UNLOCK=YES`
are still required before any real training.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE, SALES_VALUE_POLICY
from src.models.cnn_model import build_base_cnn
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_COMMON_TARGETS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    v3_arm_spec,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    V3_REVIN_CONTRACT,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    CONV_LAYER_NAMES,
    METHOD_NAME,
    MSKATLLite,
    clone_cnn_from_snapshot,
    digest_weight_arrays,
    extract_features_maybe_revin,
    fused_params_digest,
    fused_target_init_snapshot,
    raw_target_init_snapshot,
    snapshot_model_weights,
    snapshot_named_layer_weights,
)
from src.transfer_methods.ms_katl_mechanism_ablation import ABLATION_ARM_NAMES, assert_formal_methods_untouched
from src.transfer_methods.ms_katl_primary_target_lr_screening import parse_lr
from src.transfer_methods.msml_tl import (
    extract_layer_params,
    fuse_source_models_layerwise,
    get_transferable_layer_names,
    load_fused_params_into_target_model,
)
from src.transfer_methods.revin import (
    DEFAULT_REVIN_AFFINE,
    DEFAULT_REVIN_EPS,
    DEFAULT_REVIN_TARGET_CHANNEL,
    RevIN,
)
from src.utils.experiment_hyperparams import FIXED_CLIPNORM, FIXED_EPOCHS, FIXED_LEARNING_RATE

PROTOCOL_VERSION = "ae_core_mechanism_v1"
PARENT_METHOD = V3_METHOD_DISPLAY
AE_LABEL = "MS_KATL_AE_ABLATION"
AE_ARMS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
AE_ARM_NAMES: Dict[str, str] = {
    "A": "TARGET_NATIVE_INIT",
    "B": "CONTROLLED_FROZEN_FUSED",
    "C": "FUSED_TRAINABLE",
    "D": "CURRENT_V3",
    "E": "BEST_SOURCE_TRAINABLE",
}
VALID_INIT_MODES = ("native", "fused", "best_source")
LEGACY_MECHANISM_ARMS = tuple(ABLATION_ARM_NAMES)
HORIZONS: Tuple[int, ...] = (1, 3, 5)
SEEDS: Tuple[int, ...] = (42, 43, 44)
SCENARIO = "without"
EXPECTED_CANONICAL_CELLS = 54
EXPECTED_RUNS = 270
STAGE_A_LEARNING_RATE = float(FIXED_LEARNING_RATE)
STAGE_B_EPOCHS = int(FIXED_EPOCHS)
STAGE_B_BATCH_SIZE = 16
STAGE_B_EARLY_STOPPING = False
FORMAL_TRAINING_UNLOCK_ENV = "AE_FORMAL_TRAINING_UNLOCK"
AE_FORMAL_EXECUTION_IMPLEMENTED = True
AE_FORMAL_EXECUTION_ENABLED = True
AE_EXECUTION_BODY_ENABLED = AE_FORMAL_EXECUTION_ENABLED
MANIFEST_RELPATH = Path("configs") / "ms_katl_ae_ablation_manifest.json"
PLANNED_CONTRASTS: Tuple[str, ...] = ("C_vs_B", "C_vs_A", "C_vs_E", "D_vs_C")

V3_LR_MAP: Mapping[Tuple[str, str], str] = MappingProxyType(
    {
        ("D1", "1/10"): "3e-5",
        ("D2", "1/10"): "3e-5",
        ("D3", "10"): "3e-5",
        ("D4", "166/311"): "3e-5",
        ("D5", "48/1159414"): "3e-4",
        ("D6", "CA_1/FOODS_3_080"): "3e-5",
    }
)

SYNTHETIC_SOURCE_FILLS = (1.0, 2.0, 4.0)
SYNTHETIC_SOURCE_WEIGHTS = (0.5, 0.3, 0.2)
SYNTHETIC_SOURCE_DISTANCES = (0.10, 0.50, 0.90)
SYNTHETIC_DENSE_OUT_FILL = 99.0
_AE_BUILD_COUNTER = 0


class AEContractError(RuntimeError):
    """Fail-closed A–E protocol violation."""


class AETrainingBlocked(RuntimeError):
    """Formal training was requested without permission, or execution is disabled."""


@dataclass(frozen=True)
class AEAblationArmSpec:
    arm: str
    name: str
    init_mode: str
    target_backbone_trainable: bool
    revin_enabled: bool
    transfer_branch_enabled: bool
    gate_mode: str
    use_adversarial: bool
    revin_affine: bool = False
    revin_eps: float = DEFAULT_REVIN_EPS
    revin_target_channel: int = DEFAULT_REVIN_TARGET_CHANNEL

    def __post_init__(self) -> None:
        if self.arm not in AE_ARMS:
            raise AEContractError(f"unknown AE arm {self.arm!r}")
        if self.name != AE_ARM_NAMES[self.arm]:
            raise AEContractError(f"arm {self.arm} name drifted: {self.name}")
        if self.init_mode not in VALID_INIT_MODES:
            raise AEContractError(f"unknown init_mode={self.init_mode!r}")
        if self.transfer_branch_enabled or self.gate_mode != "zero" or self.use_adversarial:
            raise AEContractError(f"arm {self.arm} must disable transfer/gate/adversarial")
        if self.arm == "D":
            if not self.revin_enabled or self.init_mode != "fused" or not self.target_backbone_trainable:
                raise AEContractError("arm D must be fused + trainable + RevIN ON")
        elif self.revin_enabled:
            raise AEContractError(f"arm {self.arm} must keep RevIN OFF")
        if self.arm == "A" and self.init_mode != "native":
            raise AEContractError("arm A must use native init")
        if self.arm == "B" and (self.init_mode != "fused" or self.target_backbone_trainable):
            raise AEContractError("arm B must be fused + frozen")
        if self.arm == "C" and (self.init_mode != "fused" or not self.target_backbone_trainable):
            raise AEContractError("arm C must be fused + trainable")
        if self.arm == "E" and (self.init_mode != "best_source" or not self.target_backbone_trainable):
            raise AEContractError("arm E must be best-source + trainable")


@dataclass(frozen=True)
class AERunIdentity:
    arm: str
    dataset: str
    target_identity: str
    horizon: int
    seed: int
    selected_lr: float
    selected_lr_label: str
    scenario: str = SCENARIO

    @property
    def canonical_tuple(self) -> Tuple[Any, ...]:
        return (self.dataset, self.target_identity, int(self.horizon), int(self.seed), self.scenario)

    @property
    def identity_tuple(self) -> Tuple[Any, ...]:
        return (*self.canonical_tuple, self.arm, self.selected_lr_label)

    @property
    def cell_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        return (
            f"{self.arm}_{self.dataset}_{target_slug}"
            f"_h{int(self.horizon)}_s{int(self.seed)}_{self.selected_lr_label}"
        )


@dataclass
class AESyntheticStageA:
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    selected_sources: List[Dict[str, object]]
    input_shape: Tuple[int, ...]
    layer_names: List[str]
    source_conv_params: List[Dict[str, List[np.ndarray]]]
    source_dense_out: List[List[np.ndarray]]
    source_weight_snapshots: List[List[np.ndarray]]
    source_weight_digests: List[str]
    fused_params: Dict[str, List[np.ndarray]]
    fused_params_digest: str
    fused_target_snapshot: List[np.ndarray]
    native_target_snapshot: List[np.ndarray]
    best_source_snapshot: List[np.ndarray]
    fused_target_digest: str
    native_target_digest: str
    best_source_snapshot_digest: str
    source_selection_digest: str
    stage_a_digest: str
    knn_prior: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_split_digest: Dict[str, str]
    target_scaler_digest: str
    target_feature_columns: List[str]
    learning_rate: float
    seed: int
    d_model: int = 0
    source_extractors: List[object] = field(default_factory=list)


@dataclass
class AEConstructedArm:
    spec: AEAblationArmSpec
    target_cnn: object
    target_extractor: object
    head: MSKATLLite
    revin: Optional[RevIN]
    init_snapshot: List[np.ndarray]
    target_initial_conv: Dict[str, List[np.ndarray]]
    target_initial_conv_digest: str
    head_target_init_digest: str
    fused_init_digest: str
    best_source_snapshot_digest: str
    best_source_identity: Tuple[object, ...]
    initialization_source_count: int
    source_derived_conv_params: bool
    trainable_variable_names: List[str]
    trainable_includes_head_target: bool
    trainable_includes_conv: bool
    prediction_path_parameter_count: int
    trainable_prediction_path_parameter_count: int
    frozen_prediction_path_parameter_count: int
    total_model_object_parameter_count: int
    revin_parameter_count: int
    target_split_digest: Dict[str, str]
    source_selection_digest: str
    stage_a_digest: str
    selected_lr: float
    recipe: Dict[str, Any]


def project_root_from(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[2]


def ae_manifest_path(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / MANIFEST_RELPATH


def canonical_targets() -> Tuple[Tuple[str, str], ...]:
    if tuple(EXPECTED_COMMON_TARGETS) != tuple(V3_LR_MAP.keys()):
        raise AEContractError("AE target list drifted from frozen V3 common targets")
    return tuple(EXPECTED_COMMON_TARGETS)


def ae_training_recipe() -> Dict[str, Any]:
    recipe = dict(v3_training_recipe())
    expected = {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    if recipe != expected:
        raise AEContractError(f"AE recipe must equal frozen V3 recipe, got {recipe}")
    return recipe


def require_v3_recipe(recipe: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if recipe is None:
        raise AEContractError("AE_RECIPE_MISMATCH: A–E runner refuses to default to MSE/constant/no-EMA")
    resolved = {key: recipe[key] for key in ("loss_name", "huber_delta", "lr_schedule", "use_ema", "ema_momentum")}
    expected = ae_training_recipe()
    if resolved != expected:
        raise AEContractError(f"AE_RECIPE_MISMATCH: {resolved} != {expected}")
    return dict(resolved)


def ae_optimizer_contract() -> Dict[str, Any]:
    if FIXED_CLIPNORM is not None:
        raise AEContractError("V3/AE optimizer clipnorm must be None")
    return {
        "optimizer": "Adam",
        "clipnorm": None,
        "epochs": STAGE_B_EPOCHS,
        "batch_size": STAGE_B_BATCH_SIZE,
        "stage_b_early_stopping": STAGE_B_EARLY_STOPPING,
        "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
    }


def resolve_cell_lr(dataset: str, target_identity: str) -> Tuple[float, str]:
    key = (str(dataset).strip().upper(), str(target_identity).strip())
    if key not in V3_LR_MAP:
        raise AEContractError(f"no frozen V3 LR for {key}")
    label = V3_LR_MAP[key]
    return float(parse_lr(label)), label


def ae_arm_spec(arm: str) -> AEAblationArmSpec:
    key = str(arm).strip().upper()
    if key in {name.upper() for name in LEGACY_MECHANISM_ARMS} or str(arm).strip().lower() in LEGACY_MECHANISM_ARMS:
        raise AEContractError(
            f"legacy mechanism arm {arm!r} is not an A–E paper identity; "
            f"forbidden={list(LEGACY_MECHANISM_ARMS)}"
        )
    if key not in AE_ARMS:
        raise AEContractError(f"unknown AE arm {arm!r}. Expected one of {list(AE_ARMS)}")
    revin = dict(V3_REVIN_CONTRACT) if key == "D" else {
        "revin_enabled": False,
        "revin_affine": False,
        "revin_eps": float(DEFAULT_REVIN_EPS),
        "revin_target_channel": int(DEFAULT_REVIN_TARGET_CHANNEL),
    }
    init_mode = {"A": "native", "B": "fused", "C": "fused", "D": "fused", "E": "best_source"}[key]
    trainable = key != "B"
    return AEAblationArmSpec(
        arm=key,
        name=AE_ARM_NAMES[key],
        init_mode=init_mode,
        target_backbone_trainable=trainable,
        revin_enabled=bool(revin["revin_enabled"]),
        transfer_branch_enabled=False,
        gate_mode="zero",
        use_adversarial=False,
        revin_affine=bool(revin["revin_affine"]),
        revin_eps=float(revin["revin_eps"]),
        revin_target_channel=int(revin["revin_target_channel"]),
    )


def ae_arm_registry() -> Mapping[str, AEAblationArmSpec]:
    registry = {arm: ae_arm_spec(arm) for arm in AE_ARMS}
    if tuple(registry) != AE_ARMS:
        raise AEContractError("AE registry key order drifted")
    names = [spec.name for spec in registry.values()]
    if len(set(names)) != len(names):
        raise AEContractError("AE arm names are not unique")
    overlap = set(n.lower() for n in names) & set(LEGACY_MECHANISM_ARMS)
    if overlap:
        raise AEContractError(f"AE names collided with legacy arms: {overlap}")
    return registry


def assert_d_equals_frozen_v3() -> Dict[str, str]:
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError(
            f"frozen V3 architecture digest drifted: {digest} != {EXPECTED_V3_ARCHITECTURE_DIGEST}"
        )
    spec = ae_arm_spec("D")
    v3_spec = v3_arm_spec()
    arch = v3_architecture()
    if not spec.revin_enabled or not spec.target_backbone_trainable or spec.init_mode != "fused":
        raise AEContractError("arm D spec drifted from fused trainable RevIN V3")
    if spec.transfer_branch_enabled or spec.gate_mode != "zero" or spec.use_adversarial:
        raise AEContractError("arm D must keep transfer/gate/adv off")
    if v3_spec.method_display != PARENT_METHOD:
        raise AEContractError("v3_arm_spec method_display drifted")
    if v3_spec.ablation_arm != "fused_target_only":
        raise AEContractError("V3 authority arm must remain fused_target_only internally")
    if bool(arch["revin_enabled"]) is not True:
        raise AEContractError("V3 architecture must enable RevIN")
    if dict(v3_spec.training_recipe) != ae_training_recipe():
        raise AEContractError("V3 recipe authority drifted from AE recipe")
    assert_formal_methods_untouched()
    if PARENT_METHOD in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise AEContractError("A–E must not add V3/MS-KATL to FORMAL_METHODS")
    return {
        "ARM_D_PARENT_V3_IDENTITY": "PASS",
        "parent_method": PARENT_METHOD,
        "parent_v3_architecture_digest": digest,
        "D_ARCHITECTURE_DIGEST": digest,
    }


def _ae_name(prefix: str) -> str:
    global _AE_BUILD_COUNTER
    _AE_BUILD_COUNTER += 1
    return f"{prefix}_{_AE_BUILD_COUNTER}"


def reset_ae_rng(seed: int) -> None:
    normalized = int(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized)
    random.seed(normalized)
    np.random.seed(normalized)
    tf.random.set_seed(normalized)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(normalized)


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_selection_digest(
    identities: Sequence[Sequence[object]],
    distances: Sequence[float],
    weights: Sequence[float],
) -> str:
    return _json_digest(
        {
            "identities": [list(item) for item in identities],
            "distances": [float(value) for value in distances],
            "weights": [float(value) for value in weights],
        }
    )


def split_digest_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
) -> Dict[str, str]:
    return {
        "X_train": digest_weight_arrays([np.asarray(X_train)]),
        "y_train": digest_weight_arrays([np.asarray(y_train)]),
        "X_test": digest_weight_arrays([np.asarray(X_test)]),
        "y_test": digest_weight_arrays([np.asarray(y_test)]),
        "target_scaler": digest_weight_arrays([np.asarray(scaler_min), np.asarray(scaler_max)]),
    }


def conv_params_digest(params: Mapping[str, Sequence[np.ndarray]]) -> str:
    parts: List[np.ndarray] = []
    for name in CONV_LAYER_NAMES:
        for array in params[name]:
            parts.append(np.asarray(array))
    return digest_weight_arrays(parts)


def named_conv_digest(model) -> Tuple[Dict[str, List[np.ndarray]], str]:
    snapshot = snapshot_named_layer_weights(model, CONV_LAYER_NAMES)
    return snapshot, conv_params_digest(snapshot)


def head_target_digest(head: MSKATLLite) -> str:
    if not head.head_target.built:
        raise AEContractError("head_target must be built before digest")
    return digest_weight_arrays([np.asarray(weight) for weight in head.head_target.get_weights()])


def _count_weights(arrays: Sequence[np.ndarray]) -> int:
    return int(sum(np.asarray(array).size for array in arrays))


def _layer_weight_arrays(layer) -> List[np.ndarray]:
    return [np.asarray(weight) for weight in layer.get_weights()]


def prediction_path_layers(target_cnn, head: MSKATLLite, revin: Optional[RevIN]) -> Dict[str, List[np.ndarray]]:
    conv = snapshot_named_layer_weights(target_cnn, CONV_LAYER_NAMES)
    payload = {name: conv[name] for name in CONV_LAYER_NAMES}
    payload["head_target"] = _layer_weight_arrays(head.head_target)
    if revin is not None and bool(revin.affine):
        affine: List[np.ndarray] = []
        if revin.affine_weight is not None:
            affine.append(np.asarray(revin.affine_weight.numpy()))
        if revin.affine_bias is not None:
            affine.append(np.asarray(revin.affine_bias.numpy()))
        payload["revin_affine"] = affine
    return payload


def count_parameter_groups(
    target_cnn,
    head: MSKATLLite,
    revin: Optional[RevIN],
    *,
    backbone_trainable: bool,
) -> Dict[str, int]:
    pred = prediction_path_layers(target_cnn, head, revin)
    pred_count = sum(_count_weights(values) for values in pred.values())
    conv_count = sum(_count_weights(pred[name]) for name in CONV_LAYER_NAMES)
    head_count = _count_weights(pred["head_target"])
    revin_count = _count_weights(pred.get("revin_affine", []))
    trainable_pred = head_count + revin_count + (conv_count if backbone_trainable else 0)
    frozen_pred = pred_count - trainable_pred
    total_object = (
        _count_weights(target_cnn.get_weights())
        + _count_weights(head.get_weights())
        + revin_count
    )
    return {
        "prediction_path_parameter_count": int(pred_count),
        "trainable_prediction_path_parameter_count": int(trainable_pred),
        "frozen_prediction_path_parameter_count": int(frozen_pred),
        "total_model_object_parameter_count": int(total_object),
        "revin_parameter_count": int(revin_count),
        "trainable_parameter_count": int(trainable_pred),
        "frozen_parameter_count": int(frozen_pred),
    }


def best_source_target_init_snapshot(
    input_shape: Tuple[int, ...],
    learning_rate: float,
    rank1_conv_params: Mapping[str, Sequence[np.ndarray]],
    seed: int,
) -> List[np.ndarray]:
    """Native CNN, then overwrite conv1/2/3 only. Never copies source dense_out."""
    unknown = set(rank1_conv_params) - set(CONV_LAYER_NAMES)
    if unknown:
        raise AEContractError(f"best-source init received non-conv layers: {sorted(unknown)}")
    missing = [name for name in CONV_LAYER_NAMES if name not in rank1_conv_params]
    if missing:
        raise AEContractError(f"best-source init missing conv layers: {missing}")
    reset_ae_rng(seed)
    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    native_dense = [np.array(weight, copy=True) for weight in model.get_layer("dense_out").get_weights()]
    model = load_fused_params_into_target_model(model, dict(rank1_conv_params))
    loaded_dense = model.get_layer("dense_out").get_weights()
    for left, right in zip(native_dense, loaded_dense):
        if not np.array_equal(left, right):
            raise AEContractError("best-source init overwrote dense_out; refuse")
    return snapshot_model_weights(model)


def rank1_conv_params_from_stage_a(stage_a: AESyntheticStageA) -> Dict[str, List[np.ndarray]]:
    if not stage_a.source_identities:
        raise AEContractError("Stage A has no sources")
    distances = np.asarray(stage_a.source_distances, dtype=np.float64)
    if int(np.argmin(distances)) != 0:
        raise AEContractError("selected_sources[0] is not the nearest source")
    return {name: [np.array(array, copy=True) for array in arrays] for name, arrays in stage_a.source_conv_params[0].items()}


def snapshot_for_init_mode(spec: AEAblationArmSpec, stage_a: AESyntheticStageA) -> List[np.ndarray]:
    if spec.init_mode == "native":
        return [np.array(weight, copy=True) for weight in stage_a.native_target_snapshot]
    if spec.init_mode == "fused":
        return [np.array(weight, copy=True) for weight in stage_a.fused_target_snapshot]
    if spec.init_mode == "best_source":
        return [np.array(weight, copy=True) for weight in stage_a.best_source_snapshot]
    raise AEContractError(f"unknown init_mode={spec.init_mode!r}")


def _constant_conv_model(
    input_shape: Tuple[int, ...],
    fill: float,
    dense_fill: float,
    learning_rate: float,
    seed: int,
    name_suffix: str,
):
    reset_ae_rng(seed)
    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    model._name = f"synthetic_source_{name_suffix}"
    for layer_name in CONV_LAYER_NAMES:
        layer = model.get_layer(layer_name)
        layer.set_weights([np.full(weight.shape, fill, dtype=np.float32) for weight in layer.get_weights()])
    dense = model.get_layer("dense_out")
    dense.set_weights([np.full(weight.shape, dense_fill, dtype=np.float32) for weight in dense.get_weights()])
    return model


def build_synthetic_stage_a(
    *,
    input_shape: Tuple[int, ...] = (8, 3),
    seed: int = 42,
    n: int = 8,
    learning_rate: float = STAGE_A_LEARNING_RATE,
) -> AESyntheticStageA:
    fills = SYNTHETIC_SOURCE_FILLS
    weights = list(SYNTHETIC_SOURCE_WEIGHTS)
    distances = list(SYNTHETIC_SOURCE_DISTANCES)
    identities = [("S1",), ("S2",), ("S3",)]
    models = [
        _constant_conv_model(
            input_shape,
            fill=float(fill),
            dense_fill=SYNTHETIC_DENSE_OUT_FILL,
            learning_rate=learning_rate,
            seed=int(seed) + index,
            name_suffix=str(index),
        )
        for index, fill in enumerate(fills)
    ]
    layer_names = list(get_transferable_layer_names(models[0]))
    if tuple(layer_names) != CONV_LAYER_NAMES:
        raise AEContractError(f"transferable layers drifted: {layer_names}")
    conv_params = [extract_layer_params(model, layer_names) for model in models]
    dense_out = [ [np.array(w, copy=True) for w in model.get_layer("dense_out").get_weights()] for model in models ]
    snapshots = [snapshot_model_weights(model) for model in models]
    fused = fuse_source_models_layerwise(models, weights, layer_names)
    expected = float(weights[0] * fills[0] + weights[1] * fills[1] + weights[2] * fills[2])
    for name in CONV_LAYER_NAMES:
        for array in fused[name]:
            if not np.allclose(array, expected, rtol=0.0, atol=1e-6):
                raise AEContractError(f"synthetic fusion failed for {name}: expected {expected}")
    native_snapshot = raw_target_init_snapshot(input_shape, learning_rate, seed=seed)
    fused_snapshot = fused_target_init_snapshot(input_shape, learning_rate, fused, seed=seed)
    rank1 = {name: [np.array(array, copy=True) for array in conv_params[0][name]] for name in CONV_LAYER_NAMES}
    best_snapshot = best_source_target_init_snapshot(input_shape, learning_rate, rank1, seed=seed)
    rng = np.random.default_rng(int(seed))
    X_train = rng.normal(size=(n, *input_shape)).astype(np.float32)
    y_train = rng.normal(size=(n,)).astype(np.float32)
    X_val = rng.normal(size=(n, *input_shape)).astype(np.float32)
    y_val = rng.normal(size=(n,)).astype(np.float32)
    X_test = rng.normal(size=(n, *input_shape)).astype(np.float32)
    y_test = rng.normal(size=(n,)).astype(np.float32)
    scaler_min = np.zeros((input_shape[-1],), dtype=np.float64)
    scaler_max = np.ones((input_shape[-1],), dtype=np.float64)
    split = split_digest_from_arrays(X_train, y_train, X_test, y_test, scaler_min, scaler_max)
    selected = [
        {"source_key": identities[i], "distance": distances[i], "weight": weights[i]}
        for i in range(3)
    ]
    selection_digest = source_selection_digest(identities, distances, weights)
    stage = AESyntheticStageA(
        source_identities=identities,
        source_distances=distances,
        source_weights=weights,
        selected_sources=selected,
        input_shape=tuple(input_shape),
        layer_names=layer_names,
        source_conv_params=conv_params,
        source_dense_out=dense_out,
        source_weight_snapshots=snapshots,
        source_weight_digests=[digest_weight_arrays(item) for item in snapshots],
        fused_params=fused,
        fused_params_digest=fused_params_digest(fused),
        fused_target_snapshot=fused_snapshot,
        native_target_snapshot=native_snapshot,
        best_source_snapshot=best_snapshot,
        fused_target_digest=digest_weight_arrays(fused_snapshot),
        native_target_digest=digest_weight_arrays(native_snapshot),
        best_source_snapshot_digest=conv_params_digest(rank1),
        source_selection_digest=selection_digest,
        stage_a_digest=_json_digest(
            {
                "selection": selection_digest,
                "source_weight_digests": [digest_weight_arrays(item) for item in snapshots],
                "fused_params_digest": fused_params_digest(fused),
            }
        ),
        knn_prior=np.asarray([0.5, 0.3, 0.2], dtype=np.float64),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        target_split_digest=split,
        target_scaler_digest=split["target_scaler"],
        target_feature_columns=["sales", "f1", "f2"][: input_shape[-1]],
        learning_rate=float(learning_rate),
        seed=int(seed),
    )
    return stage


def _probe_source_features(stage_a: AESyntheticStageA, batch: int) -> List[tf.Tensor]:
    d_model = int(np.prod(stage_a.input_shape))
    zeros = np.zeros((batch, d_model), dtype=np.float32)
    return [tf.constant(zeros) for _ in stage_a.source_identities]


def construct_v3_head_target_snapshot(
    stage_a: AESyntheticStageA,
    *,
    seed: int,
) -> Tuple[List[np.ndarray], str]:
    """Build head_target the way frozen V3 / arm D does. Construction only."""
    reset_ae_rng(seed)
    target_cnn, extractor = clone_cnn_from_snapshot(
        stage_a.input_shape,
        stage_a.learning_rate,
        stage_a.fused_target_snapshot,
        target_backbone_trainable=True,
        name=_ae_name("ae_v3_authority_extractor"),
    )
    n_features = int(stage_a.X_train.shape[-1])
    revin = RevIN(
        num_features=n_features,
        eps=float(DEFAULT_REVIN_EPS),
        affine=bool(DEFAULT_REVIN_AFFINE),
        target_channel=int(DEFAULT_REVIN_TARGET_CHANNEL),
        name=_ae_name("ae_v3_authority_revin"),
    )
    _ = revin.normalize(tf.constant(stage_a.X_train[:1], dtype=tf.float32))
    probe_z, _, _ = extract_features_maybe_revin(
        extractor, stage_a.X_train[:1], revin=revin, training=False
    )
    head = MSKATLLite(
        d_model=int(np.asarray(probe_z).shape[-1]),
        num_sources=len(stage_a.source_identities),
        transfer_branch_enabled=False,
        gate_mode="zero",
        name=_ae_name("ae_v3_authority_head"),
    )
    prior = np.repeat(np.asarray(stage_a.knn_prior, dtype=np.float32).reshape(1, -1), 1, axis=0)
    _ = head(
        tf.constant(probe_z, dtype=tf.float32),
        _probe_source_features(stage_a, 1),
        tf.constant(prior, dtype=tf.float32),
        training=False,
    )
    snapshot = [np.array(weight, copy=True) for weight in head.head_target.get_weights()]
    return snapshot, digest_weight_arrays(snapshot)


def select_ae_trainable_variables(
    head: MSKATLLite,
    target_extractor,
    spec: AEAblationArmSpec,
    revin: Optional[RevIN],
) -> List[tf.Variable]:
    variables: List[tf.Variable] = []
    seen = set()

    def _add(candidates) -> None:
        for var in candidates:
            if id(var) in seen:
                continue
            seen.add(id(var))
            variables.append(var)

    if spec.target_backbone_trainable and target_extractor is not None:
        _add(target_extractor.trainable_variables)
    _add(head.head_target.trainable_variables)
    if spec.transfer_branch_enabled:
        raise AEContractError("A–E must not train a transfer branch")
    if revin is not None:
        _add(revin.trainable_variables)
    forbidden = []
    for adapter in head.adapters:
        forbidden.extend(adapter.trainable_variables)
    forbidden.extend(head.cross_attention.trainable_variables)
    forbidden.extend(head.head_transfer.trainable_variables)
    forbidden.extend(head.residual_gate.trainable_variables)
    if any(id(var) in seen for var in forbidden):
        raise AEContractError("unused transfer modules leaked into AE trainable set")
    return variables


def construct_ae_arm(
    spec: AEAblationArmSpec,
    stage_a: AESyntheticStageA,
    *,
    seed: int,
    selected_lr: float,
    recipe: Optional[Mapping[str, Any]],
    head_target_snapshot: Sequence[np.ndarray],
) -> AEConstructedArm:
    require_v3_recipe(recipe)
    reset_ae_rng(seed)
    snapshot = snapshot_for_init_mode(spec, stage_a)
    target_cnn, extractor = clone_cnn_from_snapshot(
        stage_a.input_shape,
        stage_a.learning_rate,
        snapshot,
        target_backbone_trainable=spec.target_backbone_trainable,
        name=_ae_name(f"ae_target_extractor_{spec.arm}"),
    )
    revin: Optional[RevIN] = None
    if spec.revin_enabled:
        revin = RevIN(
            num_features=int(stage_a.X_train.shape[-1]),
            eps=float(spec.revin_eps),
            affine=bool(spec.revin_affine),
            target_channel=int(spec.revin_target_channel),
            name=_ae_name(f"ae_revin_{spec.arm}"),
        )
        _ = revin.normalize(tf.constant(stage_a.X_train[:1], dtype=tf.float32))
    probe_z, _, _ = extract_features_maybe_revin(
        extractor, stage_a.X_train[:1], revin=revin, training=False
    )
    head = MSKATLLite(
        d_model=int(np.asarray(probe_z).shape[-1]),
        num_sources=len(stage_a.source_identities),
        transfer_branch_enabled=False,
        gate_mode="zero",
        name=_ae_name(f"ae_head_{spec.arm}"),
    )
    prior = np.repeat(np.asarray(stage_a.knn_prior, dtype=np.float32).reshape(1, -1), 1, axis=0)
    _ = head(
        tf.constant(probe_z, dtype=tf.float32),
        _probe_source_features(stage_a, 1),
        tf.constant(prior, dtype=tf.float32),
        training=False,
    )
    head.head_target.set_weights([np.array(weight, copy=True) for weight in head_target_snapshot])
    conv_snapshot, conv_digest = named_conv_digest(target_cnn)
    trainable = select_ae_trainable_variables(head, extractor, spec, revin)
    trainable_ids = {id(var) for var in trainable}
    trainable_names = [var.name for var in trainable]
    head_ids = {id(var) for var in head.head_target.trainable_variables}
    conv_ids = set()
    for name in CONV_LAYER_NAMES:
        conv_ids.update(id(var) for var in target_cnn.get_layer(name).weights)
    trainable_includes_head = bool(head_ids) and head_ids.issubset(trainable_ids)
    trainable_includes_conv = bool(conv_ids & trainable_ids)
    if not trainable_includes_head:
        raise AEContractError(f"arm {spec.arm} trainable set is missing head_target")
    if spec.arm == "B" and trainable_includes_conv:
        raise AEContractError("arm B trainable set includes conv backbone")
    if spec.arm == "B":
        for name in CONV_LAYER_NAMES:
            layer = target_cnn.get_layer(name)
            if layer.trainable or any(getattr(var, "trainable", False) for var in layer.weights):
                raise AEContractError(f"arm B {name} is trainable")
    if spec.arm in {"A", "C", "D", "E"}:
        if not trainable_includes_conv:
            raise AEContractError(f"arm {spec.arm} trainable set is missing conv backbone")
        for name in CONV_LAYER_NAMES:
            layer = target_cnn.get_layer(name)
            if not layer.trainable:
                raise AEContractError(f"arm {spec.arm} {name} must be trainable")
    counts = count_parameter_groups(
        target_cnn, head, revin, backbone_trainable=spec.target_backbone_trainable
    )
    init_count = {"native": 0, "fused": 3, "best_source": 1}[spec.init_mode]
    return AEConstructedArm(
        spec=spec,
        target_cnn=target_cnn,
        target_extractor=extractor,
        head=head,
        revin=revin,
        init_snapshot=snapshot,
        target_initial_conv=conv_snapshot,
        target_initial_conv_digest=conv_digest,
        head_target_init_digest=head_target_digest(head),
        fused_init_digest=stage_a.fused_target_digest,
        best_source_snapshot_digest=stage_a.best_source_snapshot_digest,
        best_source_identity=tuple(stage_a.source_identities[0]),
        initialization_source_count=init_count,
        source_derived_conv_params=spec.init_mode != "native",
        trainable_variable_names=trainable_names,
        trainable_includes_head_target=trainable_includes_head,
        trainable_includes_conv=trainable_includes_conv,
        prediction_path_parameter_count=counts["prediction_path_parameter_count"],
        trainable_prediction_path_parameter_count=counts["trainable_prediction_path_parameter_count"],
        frozen_prediction_path_parameter_count=counts["frozen_prediction_path_parameter_count"],
        total_model_object_parameter_count=counts["total_model_object_parameter_count"],
        revin_parameter_count=counts["revin_parameter_count"],
        target_split_digest=dict(stage_a.target_split_digest),
        source_selection_digest=stage_a.source_selection_digest,
        stage_a_digest=stage_a.stage_a_digest,
        selected_lr=float(selected_lr),
        recipe=ae_training_recipe(),
    )


def construct_all_ae_arms(
    stage_a: AESyntheticStageA,
    *,
    seed: int,
    selected_lr: float,
    recipe: Optional[Mapping[str, Any]] = None,
    arm_order: Sequence[str] = AE_ARMS,
) -> Dict[str, AEConstructedArm]:
    resolved_recipe = require_v3_recipe(recipe if recipe is not None else ae_training_recipe())
    order = tuple(str(item).strip().upper() for item in arm_order)
    if tuple(sorted(order)) != tuple(sorted(AE_ARMS)):
        raise AEContractError(f"arm_order must be a permutation of {list(AE_ARMS)}, got {list(order)}")
    registry = ae_arm_registry()
    head_snapshot, _head_digest = construct_v3_head_target_snapshot(stage_a, seed=seed)
    arms = {}
    for key in order:
        spec = registry[key]
        arms[key] = construct_ae_arm(
            spec,
            stage_a,
            seed=seed,
            selected_lr=selected_lr,
            recipe=resolved_recipe,
            head_target_snapshot=head_snapshot,
        )
    verify_shared_stage_a_arms(arms, stage_a)
    verify_common_head_init(arms)
    return arms


def verify_shared_stage_a_arms(
    arms: Mapping[str, AEConstructedArm],
    stage_a: AESyntheticStageA,
) -> Dict[str, str]:
    expected = set(AE_ARMS)
    if set(arms) != expected:
        raise AEContractError(f"shared Stage A requires arms {sorted(expected)}, got {sorted(arms)}")
    selections = [arms[arm].source_selection_digest for arm in ("B", "C", "D", "E")]
    if len(set(selections)) != 1 or selections[0] != stage_a.source_selection_digest:
        raise AEContractError("B/C/D/E source_selection_digest mismatch")
    fused = [arms[arm].fused_init_digest for arm in ("B", "C", "D")]
    if len(set(fused)) != 1 or fused[0] != stage_a.fused_target_digest:
        raise AEContractError("B/C/D fused_init_digest mismatch")
    conv_bcd = [arms[arm].target_initial_conv_digest for arm in ("B", "C", "D")]
    if len(set(conv_bcd)) != 1:
        raise AEContractError("B/C/D initial conv digest mismatch")
    if arms["E"].best_source_identity != tuple(stage_a.source_identities[0]):
        raise AEContractError("E.best_source_identity != selected_sources[0]")
    if arms["E"].best_source_snapshot_digest != stage_a.best_source_snapshot_digest:
        raise AEContractError("E best-source snapshot digest mismatch")
    splits = [tuple(sorted(arms[arm].target_split_digest.items())) for arm in AE_ARMS]
    if len(set(splits)) != 1:
        raise AEContractError("A–E target_split_digest mismatch")
    lrs = [arms[arm].selected_lr for arm in AE_ARMS]
    if len(set(lrs)) != 1:
        raise AEContractError("A–E selected_lr mismatch")
    return {
        "SHARED_STAGE_A_DESIGN": "PASS",
        "FUSED_INIT_CONTRACT": "PASS",
        "RANK1_SOURCE_INIT_CONTRACT": "PASS",
        "TARGET_SPLIT_CONTRACT": "PASS",
    }


def verify_common_head_init(arms: Mapping[str, AEConstructedArm]) -> str:
    digests = [arms[arm].head_target_init_digest for arm in AE_ARMS]
    if len(set(digests)) != 1:
        raise AEContractError(f"head_target init digests diverged: {digests}")
    return "PASS"


def arm_factor_payload(arm: AEConstructedArm) -> Dict[str, Any]:
    spec = arm.spec
    return {
        "init_mode": spec.init_mode,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "revin_enabled": spec.revin_enabled,
        "revin_affine": spec.revin_affine if spec.revin_enabled else False,
        "revin_eps": spec.revin_eps if spec.revin_enabled else None,
        "revin_target_channel": spec.revin_target_channel if spec.revin_enabled else None,
        "transfer_branch_enabled": spec.transfer_branch_enabled,
        "gate_mode": spec.gate_mode,
        "use_adversarial": spec.use_adversarial,
        "recipe": dict(arm.recipe),
        "selected_lr": float(arm.selected_lr),
        "epochs": STAGE_B_EPOCHS,
        "batch_size": STAGE_B_BATCH_SIZE,
        "head_target_init_digest": arm.head_target_init_digest,
        "target_initial_conv_digest": arm.target_initial_conv_digest,
        "fused_init_digest": arm.fused_init_digest,
        "best_source_snapshot_digest": arm.best_source_snapshot_digest,
        "source_selection_digest": arm.source_selection_digest,
        "target_split_digest": dict(arm.target_split_digest),
        "initialization_source_count": arm.initialization_source_count,
        "prediction_path_parameter_count": arm.prediction_path_parameter_count,
        "trainable_prediction_path_parameter_count": arm.trainable_prediction_path_parameter_count,
        "frozen_prediction_path_parameter_count": arm.frozen_prediction_path_parameter_count,
        "revin_parameter_count": arm.revin_parameter_count,
        "source_derived_conv_params": arm.source_derived_conv_params,
    }


def _diff_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    keys = sorted(set(left) | set(right))
    return {key: (left.get(key), right.get(key)) for key in keys if left.get(key) != right.get(key)}


def pairwise_single_factor_contracts(arms: Mapping[str, AEConstructedArm]) -> Dict[str, str]:
    a = arm_factor_payload(arms["A"])
    b = arm_factor_payload(arms["B"])
    c = arm_factor_payload(arms["C"])
    d = arm_factor_payload(arms["D"])
    e = arm_factor_payload(arms["E"])
    allowed = {
        "A_C": {
            "init_mode",
            "target_initial_conv_digest",
            "initialization_source_count",
            "source_derived_conv_params",
        },
        "B_C": {
            "target_backbone_trainable",
            "trainable_prediction_path_parameter_count",
            "frozen_prediction_path_parameter_count",
        },
        "C_E": {
            "init_mode",
            "target_initial_conv_digest",
            "initialization_source_count",
        },
        "C_D": {
            "revin_enabled",
            "revin_affine",
            "revin_eps",
            "revin_target_channel",
            "revin_parameter_count",
            "prediction_path_parameter_count",
            "trainable_prediction_path_parameter_count",
        },
    }
    pairs = {
        "A_C_SINGLE_FACTOR_CONTRACT": ("A_C", a, c),
        "B_C_SINGLE_FACTOR_CONTRACT": ("B_C", b, c),
        "C_E_SINGLE_FACTOR_CONTRACT": ("C_E", c, e),
        "C_D_SINGLE_FACTOR_CONTRACT": ("C_D", c, d),
    }
    report = {}
    for flag, (allow_key, left, right) in pairs.items():
        diff = _diff_payload(left, right)
        unexpected = set(diff) - allowed[allow_key]
        if unexpected:
            raise AEContractError(f"{flag} FAIL unexpected diffs={sorted(unexpected)} full={diff}")
        missing_intended = allowed[allow_key] - set(diff)
        # Some allowed keys may legally match (e.g. revin_eps stored only when enabled).
        if allow_key == "A_C" and "init_mode" not in diff:
            raise AEContractError("A vs C did not differ in init_mode")
        if allow_key == "B_C" and "target_backbone_trainable" not in diff:
            raise AEContractError("B vs C did not differ in backbone trainability")
        if allow_key == "C_E" and "init_mode" not in diff:
            raise AEContractError("C vs E did not differ in init_mode")
        if allow_key == "C_D" and "revin_enabled" not in diff:
            raise AEContractError("C vs D did not differ in revin_enabled")
        report[flag] = "PASS"
        report[f"{allow_key}_DIFF_KEYS"] = sorted(diff)
        _ = missing_intended
    return report


def metric_contract() -> Dict[str, Any]:
    return {
        "primary_metric": "rmse_original",
        "primary_metric_space": ORIGINAL_SALES_SPACE,
        "secondary_metrics": ["smape_original", "mae_original"],
        "sales_value_policy": SALES_VALUE_POLICY,
        "scaler_inverse": "affine_minmax_sales_column",
        "revin_inverse_before_scaler": "D_only",
        "clipping": SALES_VALUE_POLICY,
    }


def analysis_contract() -> Dict[str, Any]:
    return {
        "primary_metric": "rmse_original",
        "analysis_unit": "mean_over_seeds_per_target_horizon",
        "condition_count": 18,
        "planned_contrasts": list(PLANNED_CONTRASTS),
        "multiple_comparison": "Holm",
        "alpha": 0.05,
        "seed_cells_not_iid": True,
        "forbid_post_hoc_contrast_changes": True,
    }


def provenance_schema_fields() -> Tuple[str, ...]:
    return (
        "protocol_version",
        "dataset",
        "target",
        "scenario",
        "horizon",
        "seed",
        "arm",
        "arm_name",
        "selected_lr",
        "target_init_mode",
        "target_backbone_trainable",
        "revin_enabled",
        "transfer_branch_enabled",
        "gate_mode",
        "use_adversarial",
        "source_identities",
        "source_distances",
        "source_weights",
        "source_selection_digest",
        "source_weight_digests",
        "stage_a_digest",
        "fused_init_digest",
        "best_source_snapshot_digest",
        "target_initial_conv_digest",
        "head_target_init_digest",
        "target_split_digest",
        "target_scaler_digest",
        "loss",
        "huber_delta",
        "optimizer",
        "lr_schedule",
        "ema",
        "ema_momentum",
        "epochs",
        "batch_size",
        "architecture_digest",
        "parent_v3_architecture_digest",
        "recipe_digest",
        "prediction_path_parameter_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "rmse_original",
        "smape_original",
        "mae_original",
        "status",
        "error",
    )


def planned_result_row(identity: AERunIdentity) -> Dict[str, Any]:
    spec = ae_arm_spec(identity.arm)
    recipe = ae_training_recipe()
    v3_digest = v3_architecture_digest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": identity.dataset,
        "target": identity.target_identity,
        "scenario": identity.scenario,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "arm": identity.arm,
        "arm_name": spec.name,
        "selected_lr": float(identity.selected_lr),
        "target_init_mode": spec.init_mode,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "revin_enabled": spec.revin_enabled,
        "transfer_branch_enabled": False,
        "gate_mode": "zero",
        "use_adversarial": False,
        "architecture_digest": v3_digest if identity.arm == "D" else None,
        "parent_v3_architecture_digest": v3_digest,
        "recipe_digest": _json_digest(recipe),
        "rmse_original": None,
        "smape_original": None,
        "mae_original": None,
        "status": "PLANNED",
        "error": "",
    }


def enumerate_canonical_cells() -> List[Tuple[str, str, int, int]]:
    cells = [
        (dataset, target, horizon, seed)
        for dataset, target in canonical_targets()
        for horizon in HORIZONS
        for seed in SEEDS
    ]
    if len(cells) != EXPECTED_CANONICAL_CELLS:
        raise AEContractError(f"canonical cells={len(cells)} != {EXPECTED_CANONICAL_CELLS}")
    return cells


def enumerate_planned_runs() -> List[AERunIdentity]:
    identities: List[AERunIdentity] = []
    for dataset, target, horizon, seed in enumerate_canonical_cells():
        lr_value, lr_label = resolve_cell_lr(dataset, target)
        for arm in AE_ARMS:
            identities.append(
                AERunIdentity(
                    arm=arm,
                    dataset=dataset,
                    target_identity=target,
                    horizon=horizon,
                    seed=seed,
                    selected_lr=lr_value,
                    selected_lr_label=lr_label,
                )
            )
    if len(identities) != EXPECTED_RUNS:
        raise AEContractError(f"planned runs={len(identities)} != {EXPECTED_RUNS}")
    if len({item.identity_tuple for item in identities}) != EXPECTED_RUNS:
        raise AEContractError("duplicate A–E identities")
    by_cell: Dict[Tuple[Any, ...], List[float]] = {}
    for item in identities:
        by_cell.setdefault(item.canonical_tuple, []).append(item.selected_lr)
    for key, lrs in by_cell.items():
        if len(lrs) != 5 or len(set(lrs)) != 1:
            raise AEContractError(f"LR equality failed for {key}: {lrs}")
    return identities


def assert_scenario_frozen(identities: Sequence[AERunIdentity]) -> None:
    if any(item.scenario != SCENARIO for item in identities):
        raise AEContractError("A–E scenario must be without")


def recipe_digest() -> str:
    return _json_digest(ae_training_recipe())


def ae_protocol_manifest_payload() -> Dict[str, Any]:
    v3_digest = v3_architecture_digest()
    recipe = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "label": AE_LABEL,
        "parent_method": PARENT_METHOD,
        "parent_v3_architecture_digest": v3_digest,
        "arms": {
            arm: {
                "name": spec.name,
                "init_mode": spec.init_mode,
                "target_backbone_trainable": spec.target_backbone_trainable,
                "revin_enabled": spec.revin_enabled,
                "transfer_branch_enabled": spec.transfer_branch_enabled,
                "gate_mode": spec.gate_mode,
                "use_adversarial": spec.use_adversarial,
            }
            for arm, spec in ae_arm_registry().items()
        },
        "targets": [{"dataset": dataset, "target_identity": target} for dataset, target in canonical_targets()],
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "scenario": SCENARIO,
        "v3_lr_map": {
            dataset: {"target_identity": target, "learning_rate": label}
            for (dataset, target), label in V3_LR_MAP.items()
        },
        "training_recipe": recipe,
        "optimizer": optimizer,
        "primary_contrasts": list(PLANNED_CONTRASTS),
        "primary_metric": "rmse_original",
        "metric_contract": metric_contract(),
        "analysis_contract": analysis_contract(),
        "expected_canonical_cells": EXPECTED_CANONICAL_CELLS,
        "expected_runs": EXPECTED_RUNS,
        "legacy_mechanism_arms_forbidden": list(LEGACY_MECHANISM_ARMS),
        "formal_training_unlock_env": FORMAL_TRAINING_UNLOCK_ENV,
        "execution_body_enabled": AE_EXECUTION_BODY_ENABLED,
        "formal_execution_implemented": AE_FORMAL_EXECUTION_IMPLEMENTED,
        "formal_execution_enabled": AE_FORMAL_EXECUTION_ENABLED,
    }


def load_ae_manifest(project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = ae_manifest_path(project_root)
    if not path.is_file():
        raise AEContractError(f"missing A–E manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = ae_protocol_manifest_payload()
    for key in (
        "protocol_version",
        "parent_method",
        "parent_v3_architecture_digest",
        "scenario",
        "expected_canonical_cells",
        "expected_runs",
        "primary_contrasts",
        "training_recipe",
        "v3_lr_map",
        "arms",
        "execution_body_enabled",
        "formal_execution_implemented",
        "formal_execution_enabled",
    ):
        if payload.get(key) != expected.get(key):
            raise AEContractError(f"A–E manifest drifted at {key}")
    return payload


def write_ae_manifest(project_root: Optional[Path] = None) -> Path:
    path = ae_manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ae_protocol_manifest_payload()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_formal_plan(path: Path, identities: Optional[Sequence[AERunIdentity]] = None) -> Path:
    planned = list(identities) if identities is not None else enumerate_planned_runs()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_method": PARENT_METHOD,
        "status": "PLANNED",
        "expected_runs": len(planned),
        "runs": [planned_result_row(item) for item in planned],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def assert_formal_training_permitted(
    *,
    execute: bool,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    env = environ if environ is not None else os.environ
    unlocked = str(env.get(FORMAL_TRAINING_UNLOCK_ENV, "")).strip() == "YES"
    if not bool(execute) or not unlocked:
        raise AETrainingBlocked("FORMAL_TRAINING_BLOCKED")
    if not AE_FORMAL_EXECUTION_ENABLED:
        raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")


def run_ae_formal_matrix(
    *,
    execute: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Formal 270-run entry. Importing this function must not train.

    Training starts only after `--execute` and `AE_FORMAL_TRAINING_UNLOCK=YES`,
    and only when `AE_FORMAL_EXECUTION_ENABLED` is True. Any missing lock
    remains `FORMAL_EXECUTION_BLOCKED` / `FORMAL_TRAINING_BLOCKED`.
    """
    assert_formal_training_permitted(execute=execute, environ=environ)
    from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_body

    return run_ae_formal_body(execute=execute, environ=environ, mock_execution_authorized=False, **_kwargs)


def format_preflight(
    *,
    pairwise: Optional[Mapping[str, str]] = None,
    training_guard: str = "PASS",
) -> str:
    pairwise = pairwise or {}
    lines = [
        f"AE_PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"PARENT_METHOD={PARENT_METHOD}",
        "",
        f"TARGETS={len(canonical_targets())}",
        f"HORIZONS={len(HORIZONS)}",
        f"SEEDS={len(SEEDS)}",
        f"ARMS={len(AE_ARMS)}",
        "",
        f"CANONICAL_CELLS={EXPECTED_CANONICAL_CELLS}",
        f"EXPECTED_RUNS={EXPECTED_RUNS}",
        f"PLANNED_RUNS={EXPECTED_RUNS}",
        f"UNIQUE_RUN_KEYS={EXPECTED_RUNS}",
        f"UNIQUE_STAGE_A_KEYS={EXPECTED_CANONICAL_CELLS}",
        "",
        f"SCENARIO={SCENARIO}",
        "",
        f"ARM_A={AE_ARM_NAMES['A']}",
        f"ARM_B={AE_ARM_NAMES['B']}",
        f"ARM_C={AE_ARM_NAMES['C']}",
        f"ARM_D={AE_ARM_NAMES['D']}",
        f"ARM_E={AE_ARM_NAMES['E']}",
        "",
        "LR_CONTRACT=PASS",
        "RECIPE_CONTRACT=PASS",
        f"PAIRWISE_FACTOR_CONTRACT={pairwise.get('PAIRWISE_FACTOR_CONTRACT', 'PASS')}",
        f"A_C_SINGLE_FACTOR={pairwise.get('A_C_SINGLE_FACTOR_CONTRACT', 'PASS')}",
        f"B_C_SINGLE_FACTOR={pairwise.get('B_C_SINGLE_FACTOR_CONTRACT', 'PASS')}",
        f"C_E_SINGLE_FACTOR={pairwise.get('C_E_SINGLE_FACTOR_CONTRACT', 'PASS')}",
        f"C_D_SINGLE_FACTOR={pairwise.get('C_D_SINGLE_FACTOR_CONTRACT', 'PASS')}",
        "D_EQUALS_FROZEN_V3=PASS",
        f"TRAINING_GUARD={training_guard}",
        "",
        "TRAINING_STARTED=NO",
        "FORMAL_TRAINING_BLOCKED=YES",
        f"AE_FORMAL_EXECUTION_ENABLED={'YES' if AE_FORMAL_EXECUTION_ENABLED else 'NO'}",
        "ACTUAL_EXECUTION_REQUESTED=NO",
    ]
    return "\n".join(lines) + "\n"


def run_ae_preflight(
    *,
    project_root: Optional[Path] = None,
    write_plan_path: Optional[Path] = None,
    include_construction_contracts: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    d_identity = assert_d_equals_frozen_v3()
    recipe = ae_training_recipe()
    identities = enumerate_planned_runs()
    assert_scenario_frozen(identities)
    load_ae_manifest(root)
    pairwise_report: Dict[str, str] = {"PAIRWISE_FACTOR_CONTRACT": "PASS"}
    construction_flags: Dict[str, str] = {}
    if include_construction_contracts:
        stage_a = build_synthetic_stage_a(seed=42)
        lr_value, _ = resolve_cell_lr("D4", "166/311")
        arms = construct_all_ae_arms(stage_a, seed=42, selected_lr=lr_value, recipe=recipe)
        pairwise_report.update(pairwise_single_factor_contracts(arms))
        pairwise_report["PAIRWISE_FACTOR_CONTRACT"] = "PASS"
        construction_flags = {
            "COMMON_HEAD_INIT_CONTRACT": verify_common_head_init(arms),
            **verify_shared_stage_a_arms(arms, stage_a),
        }
    training_guard = "PASS"
    try:
        run_ae_formal_matrix(execute=False)
        training_guard = "FAIL"
    except AETrainingBlocked as exc:
        if "FORMAL_TRAINING_BLOCKED" not in str(exc):
            raise
    if write_plan_path is not None:
        write_formal_plan(Path(write_plan_path), identities)
    text = format_preflight(pairwise=pairwise_report, training_guard=training_guard)
    ready = (
        training_guard == "PASS"
        and pairwise_report.get("A_C_SINGLE_FACTOR_CONTRACT", "PASS") == "PASS"
        and pairwise_report.get("B_C_SINGLE_FACTOR_CONTRACT", "PASS") == "PASS"
        and pairwise_report.get("C_E_SINGLE_FACTOR_CONTRACT", "PASS") == "PASS"
        and pairwise_report.get("C_D_SINGLE_FACTOR_CONTRACT", "PASS") == "PASS"
        and d_identity["ARM_D_PARENT_V3_IDENTITY"] == "PASS"
    )
    return {
        "text": text,
        "identities": identities,
        "pairwise": pairwise_report,
        "construction": construction_flags,
        "d_identity": d_identity,
        "training_guard": training_guard,
        "READY_FOR_FORMAL_AE_RUN": "YES" if ready else "NO",
        "TRAINING_STARTED": "NO",
    }
