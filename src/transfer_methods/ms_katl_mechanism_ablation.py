"""MS-KATL-Lite single-cell mechanism ablation (extended / diagnostic track).

Not part of FORMAL_METHODS. Does not change the default MS-KATL-Lite architecture.
Six paired arms share one frozen Stage-A artifact bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.evaluation.metrics import compute_metrics_with_protocol, smape
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_lite import (
    CONV_LAYER_NAMES,
    DECOMPOSITION_TOLERANCE,
    DEFAULT_GAMMA,
    GATE_BIAS_INIT,
    METHOD_NAME,
    MSKATLLite,
    MSKATLStageAArtifacts,
    TIE_ABS_EPS,
    WEIGHT_DRIFT_EPS,
    _transform_with_scaler,
    clone_cnn_from_snapshot,
    digest_weight_arrays,
    extract_features_maybe_revin,
    fused_target_init_snapshot,
    model_weight_digest,
    predict_ms_katl_decomposed,
    prepare_ms_katl_stage_a,
    raw_target_init_snapshot,
    relative_weight_drift,
    snapshot_model_weights,
    snapshot_named_layer_weights,
    train_ms_katl_head,
)
from src.transfer_methods.revin import (
    DEFAULT_REVIN_AFFINE,
    DEFAULT_REVIN_EPS,
    DEFAULT_REVIN_TARGET_CHANNEL,
    RevIN,
)

LOGGER = logging.getLogger("experiment")
_ARM_BUILD_COUNTER = 0

# Optional Stage-B training-recipe overrides (v2 screening arm). None or an
# empty mapping keeps the frozen v1 behaviour (MSE + constant LR + no EMA).
TRAINING_RECIPE_KEYS = ("loss_name", "huber_delta", "lr_schedule", "use_ema", "ema_momentum")


def validate_training_recipe(recipe: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if recipe is None:
        return {}
    unknown = [key for key in recipe if key not in TRAINING_RECIPE_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown training_recipe keys {unknown}. Allowed: {list(TRAINING_RECIPE_KEYS)}"
        )
    return {key: recipe[key] for key in TRAINING_RECIPE_KEYS if key in recipe}


def _unique_name(prefix: str) -> str:
    global _ARM_BUILD_COUNTER
    _ARM_BUILD_COUNTER += 1
    return f"{prefix}_{_ARM_BUILD_COUNTER}"

ABLATION_ARM_NAMES = (
    "full_current",
    "fused_target_only",
    "raw_target_only",
    "raw_plus_transfer",
    "fused_gate_one",
    "fused_frozen_backbone",
)
CROSS_TARGET_ARM_NAMES = (
    "full_current",
    "fused_target_only",
    "raw_target_only",
)
CROSS_TARGET_FORBIDDEN_ARMS = (
    "raw_plus_transfer",
    "fused_gate_one",
    "fused_frozen_backbone",
)
FUSED_INIT_ARMS = (
    "full_current",
    "fused_target_only",
    "fused_gate_one",
    "fused_frozen_backbone",
)
RAW_INIT_ARMS = (
    "raw_target_only",
    "raw_plus_transfer",
)

# Prompt approximations retained only as historical labels.
# DO NOT USE FOR FORMAL OR CROSS-METHOD COMPARISON.
# These values are not identity-joined accepted metrics. Comparison must use
# exact identity join against accepted formal/baseline CSVs instead.
DEPRECATED_DISPLAY_ONLY_REFERENCE_APPROXIMATIONS = {
    "BL4_LSTM": {"label": "BL4_LSTM", "rmse": 1.07, "smape": 115.0, "usable_for_comparison": False},
    "FORMAL_NO_TL": {"label": "No-TL", "rmse": 0.68, "smape": 37.5, "usable_for_comparison": False},
    "MSML_TL": {"label": "MSML-TL", "rmse": 1.19, "smape": 141.4, "usable_for_comparison": False},
}
REFERENCE_METRICS = DEPRECATED_DISPLAY_ONLY_REFERENCE_APPROXIMATIONS

FIXED_CELL = {
    "dataset_id": 4,
    "dataset": "D4",
    "target_store_id": "166",
    "target_product_id": "258",
    "target_identity": "166/258",
    "entity_key": "166_258",
    "mode": "without",
    "horizon": 1,
    "seed": 42,
    "gamma": 0.5,
    "lambda_adv": 0.0,
}


@dataclass(frozen=True)
class AblationSpec:
    arm: str
    letter: str
    display: str
    use_fused_target_init: bool
    target_backbone_trainable: bool
    transfer_branch_enabled: bool
    gate_mode: str


ABLATION_ARM_SPECS: Dict[str, AblationSpec] = {
    "full_current": AblationSpec(
        arm="full_current",
        letter="A",
        display="FULL_CURRENT",
        use_fused_target_init=True,
        target_backbone_trainable=True,
        transfer_branch_enabled=True,
        gate_mode="learnable",
    ),
    "fused_target_only": AblationSpec(
        arm="fused_target_only",
        letter="B",
        display="FUSED_TARGET_ONLY",
        use_fused_target_init=True,
        target_backbone_trainable=True,
        transfer_branch_enabled=False,
        gate_mode="zero",
    ),
    "raw_target_only": AblationSpec(
        arm="raw_target_only",
        letter="C",
        display="RAW_TARGET_ONLY",
        use_fused_target_init=False,
        target_backbone_trainable=True,
        transfer_branch_enabled=False,
        gate_mode="zero",
    ),
    "raw_plus_transfer": AblationSpec(
        arm="raw_plus_transfer",
        letter="D",
        display="RAW_PLUS_TRANSFER",
        use_fused_target_init=False,
        target_backbone_trainable=True,
        transfer_branch_enabled=True,
        gate_mode="learnable",
    ),
    "fused_gate_one": AblationSpec(
        arm="fused_gate_one",
        letter="E",
        display="FUSED_GATE_ONE",
        use_fused_target_init=True,
        target_backbone_trainable=True,
        transfer_branch_enabled=True,
        gate_mode="one",
    ),
    "fused_frozen_backbone": AblationSpec(
        arm="fused_frozen_backbone",
        letter="F",
        display="FUSED_FROZEN_BACKBONE",
        use_fused_target_init=True,
        target_backbone_trainable=False,
        transfer_branch_enabled=True,
        gate_mode="learnable",
    ),
}


def resolve_ablation_arm(arm: str) -> AblationSpec:
    key = str(arm).strip().lower()
    if key not in ABLATION_ARM_SPECS:
        raise ValueError(
            f"Unknown ablation_arm={arm!r}. Expected one of: {list(ABLATION_ARM_NAMES)}"
        )
    return ABLATION_ARM_SPECS[key]


def resolve_cross_target_arms(arms: Sequence[str]) -> Tuple[str, ...]:
    """Fail closed unless the caller requests exactly A/B/C in cross-target mode."""
    resolved = tuple(resolve_ablation_arm(arm).arm for arm in arms)
    forbidden = [arm for arm in resolved if arm in CROSS_TARGET_FORBIDDEN_ARMS]
    if forbidden:
        raise ValueError(
            "cross-target mode forbids D/E/F arms "
            f"{list(CROSS_TARGET_FORBIDDEN_ARMS)}; got {forbidden}"
        )
    if set(resolved) != set(CROSS_TARGET_ARM_NAMES) or len(resolved) != len(CROSS_TARGET_ARM_NAMES):
        raise ValueError(
            "cross-target mode requires exactly "
            f"{list(CROSS_TARGET_ARM_NAMES)}, got {list(resolved)}"
        )
    return tuple(arm for arm in CROSS_TARGET_ARM_NAMES)


def assert_formal_methods_untouched() -> Tuple[str, ...]:
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    if FORMAL_METHODS != expected:
        raise RuntimeError(
            "FORMAL_METHODS was modified; mechanism ablation must not touch the formal six-method contract. "
            f"got={FORMAL_METHODS!r}"
        )
    if METHOD_NAME in FORMAL_METHODS:
        raise RuntimeError("MS-KATL-Lite must remain outside FORMAL_METHODS")
    return expected


@dataclass
class MSKATLSharedBundle:
    stage_a: MSKATLStageAArtifacts
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    f_t: Dict[str, List[Optional[np.ndarray]]]
    f_s_pools: List[np.ndarray]
    target_scaler: object
    target_feature_columns: List[str]
    fused_target_snapshot: List[np.ndarray]
    raw_target_snapshot: List[np.ndarray]
    fused_target_digest: str
    raw_target_digest: str
    target_split_digest: Dict[str, str]
    d_model: int
    learning_rate: float
    seed: int


def _extract_numpy(extractor, tensor: np.ndarray) -> np.ndarray:
    return extractor(tf.constant(tensor, dtype=tf.float32), training=False).numpy()


def array_digest(array: np.ndarray) -> str:
    return digest_weight_arrays([np.asarray(array)])


def prepare_shared_target_bundle(
    stage_a: MSKATLStageAArtifacts,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    horizon: int,
    window_size: int,
    learning_rate: float,
    seed: int,
    transformation_reuse_context=None,
) -> MSKATLSharedBundle:
    """Build target windows and frozen source features once for all arms."""
    raw_train_df, raw_val_df, raw_test_df = temporal_split_by_ratio_or_dates(target_df)
    raw_frames = {
        "train": raw_train_df.copy(),
        "val": raw_val_df.copy(),
        "test": raw_test_df.copy(),
    }
    target_train_df, target_val_df, target_test_df, target_scaler, target_feature_columns = normalize_features(
        raw_train_df, raw_val_df, raw_test_df, feature_columns=feature_cols,
        reuse_context=transformation_reuse_context,
    )

    def _windows(df: pd.DataFrame, use_reuse: bool) -> Tuple[np.ndarray, np.ndarray]:
        X, y = build_tabular_sequence(
            df, horizon=horizon, window_size=window_size, feature_columns=feature_cols,
            reuse_context=transformation_reuse_context if use_reuse else None,
        )
        return X, y

    X_train, y_train = _windows(target_train_df, use_reuse=True)
    X_val, y_val = _windows(target_val_df, use_reuse=True)
    X_test, y_test = _windows(target_test_df, use_reuse=True)
    if len(y_train) == 0:
        raise ValueError("Target train split produced zero windows; adjust window_size/horizon.")
    if len(y_test) == 0:
        raise ValueError("Target test split produced zero windows; adjust window_size/horizon.")
    X_train = to_cnn_tensor(X_train)
    X_val = to_cnn_tensor(X_val) if len(y_val) else X_val
    X_test = to_cnn_tensor(X_test)

    f_t: Dict[str, List[Optional[np.ndarray]]] = {"train": [], "val": [], "test": []}
    for scaler_i, extractor_i in zip(stage_a.source_scalers, stage_a.source_extractors):
        for split_name in ("train", "val", "test"):
            raw_frame = raw_frames[split_name]
            scaled = _transform_with_scaler(raw_frame, scaler_i, target_feature_columns)
            X_scaled, y_scaled = build_tabular_sequence(
                scaled, horizon=horizon, window_size=window_size,
                feature_columns=feature_cols, reuse_context=None,
            )
            if split_name == "val" and len(y_scaled) == 0:
                f_t[split_name].append(None)
                continue
            f_t[split_name].append(_extract_numpy(extractor_i, to_cnn_tensor(X_scaled)))

    f_s_pools = [
        _extract_numpy(extractor_i, tensor_i)
        for extractor_i, tensor_i in zip(stage_a.source_extractors, stage_a.source_train_tensors)
    ]
    probe = _extract_numpy(stage_a.source_extractors[0], X_train[:1])
    d_model = int(probe.shape[-1])
    fused_snapshot = fused_target_init_snapshot(
        stage_a.input_shape, learning_rate, stage_a.fused_params, seed=seed
    )
    raw_snapshot = raw_target_init_snapshot(stage_a.input_shape, learning_rate, seed=seed)
    split_digest = {
        "X_train": array_digest(X_train),
        "y_train": array_digest(np.asarray(y_train)),
        "X_test": array_digest(X_test),
        "y_test": array_digest(np.asarray(y_test)),
        "target_scaler": hashlib.sha256(
            np.ascontiguousarray(getattr(target_scaler, "data_min_")).tobytes()
            + np.ascontiguousarray(getattr(target_scaler, "data_max_")).tobytes()
        ).hexdigest(),
    }
    return MSKATLSharedBundle(
        stage_a=stage_a,
        X_train=X_train,
        y_train=np.asarray(y_train),
        X_val=X_val,
        y_val=np.asarray(y_val),
        X_test=X_test,
        y_test=np.asarray(y_test),
        f_t=f_t,
        f_s_pools=f_s_pools,
        target_scaler=target_scaler,
        target_feature_columns=list(target_feature_columns),
        fused_target_snapshot=fused_snapshot,
        raw_target_snapshot=raw_snapshot,
        fused_target_digest=digest_weight_arrays(fused_snapshot),
        raw_target_digest=digest_weight_arrays(raw_snapshot),
        target_split_digest=split_digest,
        d_model=d_model,
        learning_rate=float(learning_rate),
        seed=int(seed),
    )


def select_stage_b_trainable_variables(
    model: MSKATLLite,
    target_extractor,
    spec: AblationSpec,
    revin: Optional[RevIN] = None,
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
    _add(model.head_target.trainable_variables)
    if spec.transfer_branch_enabled:
        for adapter in model.adapters:
            _add(adapter.trainable_variables)
        _add(model.cross_attention.trainable_variables)
        _add(model.head_transfer.trainable_variables)
        if spec.gate_mode == "learnable":
            _add(model.residual_gate.trainable_variables)
    if revin is not None:
        _add(revin.trainable_variables)
    return variables


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.floating, np.integer)):
        number = value.item()
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return number
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _attention_entropy(alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(alpha, dtype=np.float64), eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _correction_diagnostics(
    y_true: np.ndarray,
    target_prediction: np.ndarray,
    transfer_gated: np.ndarray,
    tie_eps: float = TIE_ABS_EPS,
) -> Dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    target = np.asarray(target_prediction, dtype=np.float64).reshape(-1)
    gated = np.asarray(transfer_gated, dtype=np.float64).reshape(-1)
    error = true - target
    abs_target = float(np.mean(np.abs(target)))
    abs_raw_placeholder = None
    abs_gated = float(np.mean(np.abs(gated)))
    transfer_ratio = abs_gated / (abs_target + 1e-12)
    mask = (np.abs(error) > tie_eps) & (np.abs(gated) > tie_eps)
    tie_count = int((~mask).sum())
    if int(mask.sum()) == 0:
        direction_acc = float("nan")
    else:
        direction_acc = float(np.mean(np.sign(gated[mask]) == np.sign(error[mask])))
    corr = float("nan")
    if true.size >= 3 and np.std(gated) > 0 and np.std(error) > 0:
        corr = float(np.corrcoef(gated, error)[0, 1])
    return {
        "mean_abs_target_prediction": abs_target,
        "mean_abs_transfer_gated": abs_gated,
        "transfer_ratio": float(transfer_ratio),
        "correction_direction_accuracy": direction_acc,
        "correction_tie_count": tie_count,
        "correction_evaluated_count": int(mask.sum()),
        "correction_tie_eps": float(tie_eps),
        "corr_transfer_gated_vs_target_only_error": corr,
        "mean_abs_transfer_raw": abs_raw_placeholder,
    }


def _layer_drifts(initial: Dict[str, List[np.ndarray]], final: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
    drifts = {}
    all_init: List[np.ndarray] = []
    all_final: List[np.ndarray] = []
    for name in CONV_LAYER_NAMES:
        drifts[f"{name}_relative_weight_drift"] = relative_weight_drift(initial[name], final[name])
        all_init.extend(initial[name])
        all_final.extend(final[name])
    drifts["total_backbone_relative_weight_drift"] = relative_weight_drift(all_init, all_final)
    return drifts


def run_ablation_arm(
    spec: AblationSpec,
    bundle: MSKATLSharedBundle,
    *,
    epochs: int,
    batch_size: int,
    gamma: float,
    use_adversarial: bool,
    metric_protocol: Optional[Dict[str, Any]] = None,
    fine_tune_learning_rate: Optional[float] = None,
    training_recipe: Optional[Mapping[str, Any]] = None,
    revin_enabled: bool = False,
    revin_affine: bool = DEFAULT_REVIN_AFFINE,
    revin_eps: float = DEFAULT_REVIN_EPS,
    revin_target_channel: int = DEFAULT_REVIN_TARGET_CHANNEL,
    revin_include_mean: bool = True,
    return_runtime: bool = False,
) -> Dict[str, Any]:
    recipe_kwargs = validate_training_recipe(training_recipe)
    snapshot = (
        bundle.fused_target_snapshot if spec.use_fused_target_init else bundle.raw_target_snapshot
    )
    tf.random.set_seed(int(bundle.seed))
    target_cnn, target_extractor = clone_cnn_from_snapshot(
        bundle.stage_a.input_shape,
        bundle.learning_rate,
        snapshot,
        target_backbone_trainable=spec.target_backbone_trainable,
        name=_unique_name(f"target_feature_extractor_{spec.arm}"),
    )
    train_learning_rate = (
        float(fine_tune_learning_rate)
        if fine_tune_learning_rate is not None
        else float(bundle.learning_rate)
    )
    initial_conv = snapshot_named_layer_weights(target_cnn, CONV_LAYER_NAMES)
    initial_target_digest = model_weight_digest(target_cnn)
    initial_conv_digest = digest_weight_arrays(
        [w for name in CONV_LAYER_NAMES for w in initial_conv[name]]
    )

    head = MSKATLLite(
        d_model=bundle.d_model,
        num_sources=bundle.stage_a.effective_k,
        gamma=gamma,
        transfer_branch_enabled=spec.transfer_branch_enabled,
        gate_mode=spec.gate_mode,
        name=_unique_name(f"ms_katl_lite_{spec.arm}"),
    )
    revin: Optional[RevIN] = None
    if revin_enabled:
        n_features = int(bundle.X_train.shape[-1])
        revin = RevIN(
            num_features=n_features,
            eps=float(revin_eps),
            affine=bool(revin_affine),
            target_channel=int(revin_target_channel),
            name=_unique_name(f"revin_{spec.arm}"),
        )
        _ = revin.normalize(tf.constant(bundle.X_train[:1], dtype=tf.float32))
    has_val = len(bundle.y_val) > 0 and all(f is not None for f in bundle.f_t["val"])
    probe_z, _, _ = extract_features_maybe_revin(
        target_extractor, bundle.X_train[:1], revin=revin, training=False
    )
    _ = head(
        tf.constant(probe_z, dtype=tf.float32),
        [tf.constant(f[:1], dtype=tf.float32) for f in bundle.f_t["train"]],
        tf.constant(np.repeat(bundle.stage_a.knn_prior.reshape(1, -1), 1, axis=0), dtype=tf.float32),
        training=False,
    )
    trainable_vars = select_stage_b_trainable_variables(head, target_extractor, spec, revin=revin)
    disc_init = (
        [snapshot_model_weights(disc) for disc in head.discriminators]
        if all(disc.built for disc in head.discriminators)
        else None
    )
    initial_gate_weights = (
        [np.array(weight, copy=True) for weight in head.residual_gate.get_weights()]
        if spec.gate_mode == "learnable" and head.residual_gate.built
        else None
    )

    training_summary = train_ms_katl_head(
        head,
        x_train=bundle.X_train,
        f_t_train=bundle.f_t["train"],
        y_train=bundle.y_train,
        f_s_pools=bundle.f_s_pools,
        knn_prior=bundle.stage_a.knn_prior,
        target_extractor=target_extractor,
        x_val=bundle.X_val if has_val else None,
        f_t_val=bundle.f_t["val"] if has_val else None,
        y_val=bundle.y_val if has_val else None,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=train_learning_rate,
        seed=bundle.seed,
        use_adversarial=bool(use_adversarial),
        trainable_variables=trainable_vars,
        collect_gate_grads=spec.gate_mode == "learnable",
        revin=revin,
        revin_include_mean=bool(revin_include_mean),
        **recipe_kwargs,
    )

    z_t_test, test_mean, test_stdev = extract_features_maybe_revin(
        target_extractor, bundle.X_test, revin=revin, training=False
    )
    decomposed = predict_ms_katl_decomposed(
        head, z_t_test, bundle.f_t["test"], bundle.stage_a.knn_prior
    )
    y_pred_model = np.asarray(decomposed["y_pred"]).reshape(-1)
    if revin is not None:
        y_pred = revin.denormalize_numpy(
            y_pred_model,
            test_mean.numpy(),
            test_stdev.numpy(),
            include_mean=bool(revin_include_mean),
        ).reshape(-1)
        target_prediction_model = np.asarray(decomposed["target_prediction"]).reshape(-1)
        target_prediction = revin.denormalize_numpy(
            target_prediction_model,
            test_mean.numpy(),
            test_stdev.numpy(),
            include_mean=bool(revin_include_mean),
        ).reshape(-1)
    else:
        y_pred = y_pred_model
        target_prediction = np.asarray(decomposed["target_prediction"]).reshape(-1)
    y_true = np.asarray(bundle.y_test).reshape(-1)
    identity_gap = np.max(np.abs(
        decomposed["y_pred"].reshape(-1)
        - decomposed["target_prediction"].reshape(-1)
        - decomposed["transfer_gated"].reshape(-1)
    ))
    if float(identity_gap) >= DECOMPOSITION_TOLERANCE:
        raise ValueError(
            f"prediction decomposition identity failed for arm={spec.arm}: "
            f"max_abs={identity_gap} tolerance={DECOMPOSITION_TOLERANCE}"
        )

    protocol_metrics = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    rmse = float(protocol_metrics["rmse"])
    smape_value = float(protocol_metrics["smape"])

    final_conv = snapshot_named_layer_weights(target_cnn, CONV_LAYER_NAMES)
    drifts = _layer_drifts(initial_conv, final_conv)
    if not spec.target_backbone_trainable:
        for name in CONV_LAYER_NAMES:
            if drifts[f"{name}_relative_weight_drift"] > 1e-12:
                raise ValueError(
                    f"frozen backbone invariant failed for {spec.arm}: "
                    f"{name} drift={drifts[f'{name}_relative_weight_drift']}"
                )

    gate = np.asarray(decomposed["gate"]).reshape(-1)
    transfer_raw = np.asarray(decomposed["transfer_raw"]).reshape(-1)
    transfer_gated = np.asarray(decomposed["transfer_gated"]).reshape(-1)
    contrib = _correction_diagnostics(y_true, target_prediction, transfer_gated)
    contrib["mean_abs_transfer_raw"] = float(np.mean(np.abs(transfer_raw)))

    alpha = np.asarray(decomposed["alpha"])
    prior = np.asarray(bundle.stage_a.knn_prior, dtype=np.float64).reshape(1, -1)
    attention = {}
    if spec.transfer_branch_enabled:
        mean_alpha = [float(v) for v in np.mean(alpha, axis=0)]
        attention = {
            "knn_prior_pi": [float(v) for v in bundle.stage_a.knn_prior],
            "mean_alpha": mean_alpha,
            "mean_alpha_source_1": mean_alpha[0] if len(mean_alpha) > 0 else float("nan"),
            "mean_alpha_source_2": mean_alpha[1] if len(mean_alpha) > 1 else float("nan"),
            "mean_alpha_source_3": mean_alpha[2] if len(mean_alpha) > 2 else float("nan"),
            "attention_entropy_mean": float(np.mean(_attention_entropy(alpha))),
            "mean_abs_alpha_minus_prior": float(np.mean(np.abs(alpha - prior))),
            "alpha_test": alpha.tolist(),
        }

    gate_diag = dict(training_summary.get("gate_diagnostics") or {})
    if spec.gate_mode == "learnable":
        gate_diag.update({
            "gate_mode": "learnable",
            "gate_mean_final": float(np.mean(gate)),
            "gate_std_final": float(np.std(gate)),
            "gate_min_final": float(np.min(gate)),
            "gate_max_final": float(np.max(gate)),
            "gate_bias_initial": GATE_BIAS_INIT,
        })
        if head.residual_gate.built:
            bias = float(head.residual_gate.gate_proj.bias.numpy().reshape(-1)[0])
            kernel = head.residual_gate.gate_proj.kernel.numpy()
            gate_diag["gate_bias_final"] = bias
            gate_diag["gate_kernel_l2"] = float(np.linalg.norm(kernel))
            if initial_gate_weights is not None:
                final_gate_weights = [
                    np.array(weight, copy=True) for weight in head.residual_gate.get_weights()
                ]
                delta_sq = 0.0
                for initial, final in zip(initial_gate_weights, final_gate_weights):
                    delta = np.asarray(final) - np.asarray(initial)
                    delta_sq += float(np.sum(delta * delta))
                gate_diag["parameter_delta_norm"] = float(math.sqrt(delta_sq))
        gate_diag["gradient_norm"] = gate_diag.get("gate_grad_norm_mean")
    elif spec.gate_mode == "one":
        if not np.allclose(gate, 1.0, atol=1e-6):
            raise ValueError(f"fused_gate_one invariant failed: gate not identically 1, max={gate.max()} min={gate.min()}")
        gate_diag = {
            "gate_mode": "constant_one",
            "gate_mean_final": 1.0,
            "gate_std_final": 0.0,
            "gate_min_final": 1.0,
            "gate_max_final": 1.0,
        }
    else:
        if not np.allclose(gate, 0.0, atol=1e-6):
            raise ValueError(f"{spec.arm} expected disabled gate==0")
        if not np.allclose(transfer_gated, 0.0, atol=1e-6):
            raise ValueError(f"{spec.arm} expected transfer branch disabled")
        gate_diag = {"gate_mode": "disabled"}

    disc_unchanged = None
    if disc_init is not None and head.discriminators[0].built and not use_adversarial:
        disc_unchanged = all(
            digest_weight_arrays(initial) == digest_weight_arrays(snapshot_model_weights(disc))
            for initial, disc in zip(disc_init, head.discriminators)
        )

    inference_gate_off = None
    if spec.arm == "full_current":
        y_off = target_prediction
        inference_gate_off = {
            "rmse_same_weights_gate_off": _rmse(y_true, y_off),
            "smape_same_weights_gate_off": float(smape(y_true, y_off)),
            "rmse_full": _rmse(y_true, y_pred),
            "smape_full": float(smape(y_true, y_pred)),
        }

    result = {
        "arm": spec.arm,
        "letter": spec.letter,
        "display": spec.display,
        "fused_init": spec.use_fused_target_init,
        "target_conv_trainable": spec.target_backbone_trainable,
        "transfer_enabled": spec.transfer_branch_enabled,
        "gate_mode": spec.gate_mode if spec.transfer_branch_enabled else "disabled",
        "rmse": rmse,
        "smape": smape_value,
        "rmse_model_space": _rmse(y_true, y_pred),
        "smape_model_space": float(smape(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "decomposition": {
            "target_prediction": target_prediction,
            "transfer_raw": transfer_raw,
            "gate": gate,
            "transfer_gated": transfer_gated,
            "final_prediction": y_pred,
            "identity_max_abs": float(identity_gap),
        },
        "transfer_diagnostics": contrib,
        "gate_diagnostics": gate_diag,
        "attention_diagnostics": attention,
        "conv_drifts": drifts,
        "source_identities": [tuple(k) for k in bundle.stage_a.source_identities],
        "source_distances": list(bundle.stage_a.source_distances),
        "source_weight_digests": list(bundle.stage_a.source_weight_digests),
        "source_scaler_digests": list(bundle.stage_a.source_scaler_digests),
        "fused_params_digest": bundle.stage_a.fused_params_digest,
        "target_init_digest": initial_target_digest,
        "target_conv_init_digest": initial_conv_digest,
        "target_split_digest": dict(bundle.target_split_digest),
        "inference_gate_off": inference_gate_off,
        "lambda_adv": 0.0 if not use_adversarial else None,
        "use_adversarial": bool(use_adversarial),
        "fine_tune_learning_rate": train_learning_rate,
        "bundle_learning_rate": float(bundle.learning_rate),
        "training_recipe": dict(training_summary.get("training_recipe") or {}),
        "revin_enabled": bool(revin_enabled),
        "revin_affine": bool(revin.affine) if revin is not None else False,
        "revin_eps": float(revin.eps) if revin is not None else None,
        "revin_target_channel": int(revin.target_channel) if revin is not None else None,
        "revin_include_mean": bool(revin_include_mean) if revin is not None else None,
        "discriminator_weights_unchanged": disc_unchanged,
        "metric_protocol": protocol_metrics,
        "history": training_summary["history"],
    }
    if return_runtime:
        result["runtime"] = {
            "head": head,
            "target_extractor": target_extractor,
            "target_cnn": target_cnn,
            "revin": revin,
        }
    return result


def compute_core_effects(arm_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    rmse = {name: float(arm_rows[name]["rmse"]) for name in ABLATION_ARM_NAMES}
    return {
        "INIT_EFFECT": rmse["raw_target_only"] - rmse["fused_target_only"],
        "TRANSFER_EFFECT": rmse["fused_target_only"] - rmse["full_current"],
        "RAW_TRANSFER_EFFECT": rmse["raw_target_only"] - rmse["raw_plus_transfer"],
        "GATE_PROTECTION_EFFECT": rmse["fused_gate_one"] - rmse["full_current"],
        "UNFREEZE_EFFECT": rmse["fused_frozen_backbone"] - rmse["full_current"],
    }


def compute_abc_effects(arm_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """INIT/TRANSFER effects using the same definitions as the six-arm audit."""
    required = CROSS_TARGET_ARM_NAMES
    missing = [name for name in required if name not in arm_rows]
    if missing:
        raise ValueError(f"ABC effects require arms {list(required)}, missing {missing}")
    rmse_a = float(arm_rows["full_current"]["rmse"])
    rmse_b = float(arm_rows["fused_target_only"]["rmse"])
    rmse_c = float(arm_rows["raw_target_only"]["rmse"])
    return {
        "INIT_EFFECT": rmse_c - rmse_b,
        "TRANSFER_EFFECT": rmse_b - rmse_a,
    }


def verify_shared_stage_a_contract(
    arm_rows: Sequence[Dict[str, Any]],
    *,
    expected_arms: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    expected = tuple(expected_arms) if expected_arms is not None else ABLATION_ARM_NAMES
    got_arms = [row["arm"] for row in arm_rows]
    if len(arm_rows) != len(expected) or set(got_arms) != set(expected):
        raise ValueError(f"expected {len(expected)} arms {list(expected)}, got {got_arms}")
    identities = [tuple(row["source_identities"]) for row in arm_rows]
    orders = identities
    digests = [tuple(row["source_weight_digests"]) for row in arm_rows]
    scalers = [tuple(row["source_scaler_digests"]) for row in arm_rows]
    splits = [tuple(sorted(row["target_split_digest"].items())) for row in arm_rows]
    fused = [row["fused_params_digest"] for row in arm_rows]

    def _flag(values) -> str:
        return "YES" if all(v == values[0] for v in values) else "NO"

    flags = {
        "STAGE_A_SHARED": _flag(digests) if _flag(digests) == "YES" and _flag(scalers) == "YES" else "NO",
        "SOURCE_IDENTITIES_MATCH": _flag(identities),
        "SOURCE_ORDER_MATCH": _flag(orders),
        "SOURCE_MODEL_DIGESTS_MATCH": _flag(digests),
        "SOURCE_SCALERS_MATCH": _flag(scalers),
        "TARGET_SPLIT_MATCH": _flag(splits),
        "FUSED_PARAMS_DIGEST_MATCH": _flag(fused),
    }
    if any(v != "YES" for v in flags.values()):
        raise ValueError(f"shared Stage-A contract failed: {flags}")

    fused_init_digests = [
        row["target_conv_init_digest"]
        for row in arm_rows
        if row["arm"] in FUSED_INIT_ARMS
    ]
    raw_init_digests = [
        row["target_conv_init_digest"]
        for row in arm_rows
        if row["arm"] in RAW_INIT_ARMS
    ]
    if fused_init_digests:
        if len(set(fused_init_digests)) != 1:
            raise ValueError("fused target conv initialization digests do not match")
        flags["FUSED_TARGET_INIT_MATCH"] = "YES"
    if raw_init_digests:
        if len(set(raw_init_digests)) != 1:
            raise ValueError("raw target conv initialization digests do not match")
        flags["RAW_TARGET_INIT_MATCH"] = "YES"
    if fused_init_digests and raw_init_digests:
        if fused_init_digests[0] == raw_init_digests[0]:
            raise ValueError(
                "raw target init digest matches fused init; C must not load fused weights"
            )
        flags["RAW_INIT_DISTINCT_FROM_FUSED"] = "YES"
    return flags


def format_terminal_summary(
    arm_rows: Dict[str, Dict[str, Any]],
    effects: Dict[str, float],
    flags: Dict[str, str],
    *,
    dataset: str,
    target_identity: str,
    mode: str,
    horizon: int,
    seed: int,
) -> str:
    lines = [
        "======================================================================",
        "MS-KATL-Lite Mechanism Ablation",
        f"{dataset} | target={target_identity} | {mode} | H={horizon} | seed={seed}",
        "======================================================================",
        "",
        f"{'ARM':<28}{'RMSE':<11}{'sMAPE':<11}{'GATE':<10}{'TR_RATIO':<10}",
        "----------------------------------------------------------------------",
    ]
    for name in ABLATION_ARM_NAMES:
        row = arm_rows[name]
        spec = ABLATION_ARM_SPECS[name]
        gate = row["gate_diagnostics"].get("gate_mean_final", float("nan"))
        tr_ratio = row["transfer_diagnostics"].get("transfer_ratio", float("nan"))
        gate_s = f"{gate:.4f}" if isinstance(gate, (int, float)) and math.isfinite(float(gate)) else str(gate)
        lines.append(
            f"{spec.letter + ' ' + spec.display:<28}"
            f"{row['rmse']:<11.4f}{row['smape']:<11.4f}{gate_s:<10}{tr_ratio:<10.4f}"
        )
    lines.append("----------------------------------------------------------------------")
    for key in (
        "INIT_EFFECT",
        "TRANSFER_EFFECT",
        "RAW_TRANSFER_EFFECT",
        "GATE_PROTECTION_EFFECT",
        "UNFREEZE_EFFECT",
    ):
        lines.append(f"{key:<28}{effects[key]:.4f}")
    lines.append("----------------------------------------------------------------------")
    lines.append("REFERENCE")
    lines.append(f"{'BL4_LSTM':<28}{1.07:<11.2f}{115.0:<11.1f}")
    lines.append(f"{'FORMAL_NO_TL':<28}{0.68:<11.2f}{37.5:<11.1f}")
    lines.append(f"{'MSML_TL':<28}{1.19:<11.2f}{141.4:<11.1f}")
    lines.append("----------------------------------------------------------------------")
    for key, value in flags.items():
        lines.append(f"{key}={value}")
    a_off = arm_rows["full_current"].get("inference_gate_off") or {}
    if a_off:
        lines.append(
            "A_GATE_OFF_INFERENCE "
            f"rmse={a_off.get('rmse_same_weights_gate_off')} "
            f"smape={a_off.get('smape_same_weights_gate_off')}"
        )
    return "\n".join(lines)


def write_ablation_outputs(
    run_dir: Path,
    arm_rows: Dict[str, Dict[str, Any]],
    effects: Dict[str, float],
    flags: Dict[str, str],
    *,
    cell_meta: Dict[str, Any],
) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    pred_dir = run_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    diagnostics: Dict[str, Any] = {
        "cell": cell_meta,
        "formal_methods": list(FORMAL_METHODS),
        "shared_contract": flags,
        "effects": effects,
        "reference": {
            "usable_for_comparison": False,
            "note": (
                "DO NOT USE FOR FORMAL OR CROSS-METHOD COMPARISON. "
                "Hardcoded prompt approximations only; use exact identity join."
            ),
            "deprecated_display_only_reference_approximations": (
                DEPRECATED_DISPLAY_ONLY_REFERENCE_APPROXIMATIONS
            ),
        },
        "arms": {},
    }
    for name in ABLATION_ARM_NAMES:
        row = arm_rows[name]
        spec = ABLATION_ARM_SPECS[name]
        pred_path = pred_dir / f"{name}.csv"
        pred_df = pd.DataFrame({
            "y_true": np.asarray(row["y_true"]).reshape(-1),
            "y_pred": np.asarray(row["y_pred"]).reshape(-1),
            "target_prediction": np.asarray(row["decomposition"]["target_prediction"]).reshape(-1),
            "transfer_raw": np.asarray(row["decomposition"]["transfer_raw"]).reshape(-1),
            "gate": np.asarray(row["decomposition"]["gate"]).reshape(-1),
            "transfer_gated": np.asarray(row["decomposition"]["transfer_gated"]).reshape(-1),
            "final_prediction": np.asarray(row["decomposition"]["final_prediction"]).reshape(-1),
        })
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")
        csv_rows.append({
            "arm": name,
            "letter": spec.letter,
            "display": spec.display,
            "dataset": cell_meta.get("dataset"),
            "target_identity": cell_meta.get("target_identity"),
            "mode": cell_meta.get("mode"),
            "horizon": cell_meta.get("horizon"),
            "seed": cell_meta.get("seed"),
            "epochs": cell_meta.get("epochs"),
            "gamma": cell_meta.get("gamma"),
            "lambda_adv": cell_meta.get("lambda_adv"),
            "rmse": row["rmse"],
            "smape": row["smape"],
            "fused_init": spec.use_fused_target_init,
            "target_conv_trainable": spec.target_backbone_trainable,
            "transfer_enabled": spec.transfer_branch_enabled,
            "gate_mode": row["gate_mode"],
            "gate_mean": row["gate_diagnostics"].get("gate_mean_final", np.nan),
            "transfer_ratio": row["transfer_diagnostics"].get("transfer_ratio", np.nan),
            "correction_direction_accuracy": row["transfer_diagnostics"].get(
                "correction_direction_accuracy", np.nan
            ),
            "conv1_relative_drift": row["conv_drifts"]["conv1_relative_weight_drift"],
            "conv2_relative_drift": row["conv_drifts"]["conv2_relative_weight_drift"],
            "conv3_relative_drift": row["conv_drifts"]["conv3_relative_weight_drift"],
            "delta_rmse_vs_full": float(row["rmse"] - arm_rows["full_current"]["rmse"]),
            # Hardcoded REFERENCE_METRICS must not enter comparison. Leave NA.
            "delta_rmse_vs_no_tl": None,
            "delta_rmse_vs_msml": None,
            "delta_rmse_vs_bl4": None,
            "delta_smape_vs_full": float(row["smape"] - arm_rows["full_current"]["smape"]),
            "delta_smape_vs_no_tl": None,
            "delta_smape_vs_msml": None,
            "delta_smape_vs_bl4": None,
        })
        diagnostics["arms"][name] = {
            key: _to_jsonable(value)
            for key, value in row.items()
            if key not in {"history"}
        }
        diagnostics["arms"][name]["history_tail"] = _to_jsonable(
            {k: (v[-3:] if isinstance(v, list) else v) for k, v in row["history"].items()}
        )

    csv_path = results_dir / "mechanism_ablation_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
    json_path = results_dir / "mechanism_ablation_diagnostics.json"
    json_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "results_csv": csv_path,
        "diagnostics_json": json_path,
        "predictions_dir": pred_dir,
    }


def run_selected_ablation_arms(
    bundle: MSKATLSharedBundle,
    arms: Sequence[str],
    *,
    epochs: int,
    batch_size: int = 16,
    gamma: float = DEFAULT_GAMMA,
    lambda_adv: float = 0.0,
    metric_protocol: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
    cell_meta: Optional[Dict[str, Any]] = None,
    write_six_arm_outputs: bool = False,
) -> Dict[str, Any]:
    assert_formal_methods_untouched()
    if float(lambda_adv) != 0.0:
        raise ValueError("mechanism ablation requires lambda_adv=0 (fail closed)")
    resolved_arms = tuple(resolve_ablation_arm(arm).arm for arm in arms)
    if len(set(resolved_arms)) != len(resolved_arms):
        raise ValueError(f"duplicate ablation arms are not allowed: {list(resolved_arms)}")
    arm_rows: Dict[str, Dict[str, Any]] = {}
    for name in resolved_arms:
        spec = resolve_ablation_arm(name)
        LOGGER.info("[mechanism_ablation] start arm=%s", name)
        arm_rows[name] = run_ablation_arm(
            spec,
            bundle,
            epochs=epochs,
            batch_size=batch_size,
            gamma=gamma,
            use_adversarial=False,
            metric_protocol=metric_protocol,
        )
    flags = verify_shared_stage_a_contract(
        list(arm_rows.values()),
        expected_arms=resolved_arms,
    )
    if set(resolved_arms) == set(ABLATION_ARM_NAMES):
        effects = compute_core_effects(arm_rows)
    elif set(resolved_arms) == set(CROSS_TARGET_ARM_NAMES):
        effects = compute_abc_effects(arm_rows)
    else:
        effects = {}
    meta = dict(cell_meta or {})
    meta.setdefault("epochs", int(epochs))
    meta.setdefault("gamma", float(gamma))
    meta.setdefault("lambda_adv", 0.0)
    output_paths = {}
    if run_dir is not None and write_six_arm_outputs:
        output_paths = write_ablation_outputs(
            Path(run_dir), arm_rows, effects, flags, cell_meta=meta
        )
    summary = ""
    if set(resolved_arms) == set(ABLATION_ARM_NAMES):
        summary = format_terminal_summary(
            arm_rows,
            effects,
            flags,
            dataset=str(meta.get("dataset", "D4")),
            target_identity=str(meta.get("target_identity", "166/258")),
            mode=str(meta.get("mode", "without")),
            horizon=int(meta.get("horizon", 1)),
            seed=int(meta.get("seed", 42)),
        )
    return {
        "arm_rows": arm_rows,
        "effects": effects,
        "flags": flags,
        "summary": summary,
        "output_paths": output_paths,
        "resolved_arms": resolved_arms,
    }


def run_six_arm_mechanism_ablation(
    bundle: MSKATLSharedBundle,
    *,
    epochs: int,
    batch_size: int = 16,
    gamma: float = DEFAULT_GAMMA,
    lambda_adv: float = 0.0,
    metric_protocol: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
    cell_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return run_selected_ablation_arms(
        bundle,
        ABLATION_ARM_NAMES,
        epochs=epochs,
        batch_size=batch_size,
        gamma=gamma,
        lambda_adv=lambda_adv,
        metric_protocol=metric_protocol,
        run_dir=run_dir,
        cell_meta=cell_meta,
        write_six_arm_outputs=True,
    )


def run_abc_mechanism_ablation(
    bundle: MSKATLSharedBundle,
    *,
    epochs: int,
    batch_size: int = 16,
    gamma: float = DEFAULT_GAMMA,
    lambda_adv: float = 0.0,
    metric_protocol: Optional[Dict[str, Any]] = None,
    cell_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    arms = resolve_cross_target_arms(CROSS_TARGET_ARM_NAMES)
    return run_selected_ablation_arms(
        bundle,
        arms,
        epochs=epochs,
        batch_size=batch_size,
        gamma=gamma,
        lambda_adv=lambda_adv,
        metric_protocol=metric_protocol,
        run_dir=None,
        cell_meta=cell_meta,
        write_six_arm_outputs=False,
    )


def print_preflight(
    *,
    dataset: str = "D4",
    target: str = "166/258",
    mode: str = "without",
    horizon: int = 1,
    seed: int = 42,
    arms: int = 6,
) -> str:
    text = (
        f"dataset={dataset}\n"
        f"target={target}\n"
        f"mode={mode}\n"
        f"horizon={horizon}\n"
        f"seed={seed}\n"
        f"arms={arms}\n"
        f"shared_stage_a=yes\n"
    )
    print(text, end="")
    return text
