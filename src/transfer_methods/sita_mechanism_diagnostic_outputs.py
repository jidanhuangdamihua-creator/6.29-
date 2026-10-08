"""Orchestration outputs for the independent SITA mechanism diagnostic rerun.

Numpy-first helpers: formal-test CKA references, training-dynamics rows,
pointwise predictions, sales-bin summaries, CKA comparison matrix,
layer_role drift schema, historical-reference comparison, and resume.
Does not train models and does not rewrite frozen A–E numbers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import smape
from src.transfer_methods.sita_mechanism_diagnostics import (
    linear_cka,
    parameter_frobenius_norm,
    parameter_l2_distance,
    reference_input_digest,
    relative_parameter_drift,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_A_C_MATCHED_SEED,
    COMPARISON_B_B_CROSS_SEED,
    COMPARISON_B_C_MATCHED_SEED,
    COMPARISON_C_C_CROSS_SEED,
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
    COMPARISON_C_INIT_C_RAW_FINAL,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismDiagnosticsContractError,
)

FORMAL_TEST_REFERENCE_SPLIT = "formal_test_inputs"
TRANSFERRED_BACKBONE_LAYERS = ("conv1", "conv2", "conv3")
PREDICTION_HEAD_LAYER = "head_target"
LAYER_ROLES = (
    "transferred_backbone",
    "prediction_head",
    "non_transferred_layer",
    "other",
)
CONSISTENCY_PASS_PCT = 0.5
CONSISTENCY_REVIEW_PCT = 2.0
LOW_REFERENCE_SAMPLE_COUNT = 100
FORMAL_AE_UNIFORM_3E4_CSV = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果/ae_uniform_3e4_results.csv"
)
FORMAL_PER_DATASET_LR_AUTHORITY_ROOT = Path(
    "/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/sita_ablation_formal_per_dataset_lr_ae_20260921_185137"
)
FORMAL_PER_DATASET_LR_CSV = (
    FORMAL_PER_DATASET_LR_AUTHORITY_ROOT / "ae_formal_per_dataset_lr_results.csv"
)
OUTPUT_ROOT_PARENT = Path("outputs") / "mechanism_diagnostics"
FORMAL_300_OUTPUT_PREFIX = "mechanism_formal_aligned_300"
DIAGNOSTIC_42_OUTPUT_PREFIX = "mechanism_rerun"
CKA_REPRESENTATION_LAYERS = ("flatten", "conv1", "conv2", "conv3")
CKA_ACTIVATION_RESHAPE_RULE = "reshape(n_samples, -1)"
EXACT_NUMERIC_MATCH = "EXACT_NUMERIC_MATCH"
TOLERANCE_MATCH = "TOLERANCE_MATCH"
MISMATCH = "MISMATCH"
IDENTITY_ABS_TOL = 1e-12
IDENTITY_REL_TOL = 1e-12
CELL_STATUS_PENDING = "PENDING"
CELL_STATUS_RUNNING = "RUNNING"
CELL_STATUS_COMPLETE = "COMPLETE"
CELL_STATUS_FAILED = "FAILED"
ERROR_BIN_IDENTIFIER_FIELDS = ("dataset", "target", "horizon", "seed", "arm", "bin")
CELL_JSON_ARTIFACT_NAMES = ("result.json", "status.json", "provenance.json")
HISTORICAL_COMPARISON_FILENAME = "historical_frozen_comparison.csv"


def condition_id(dataset: str, target: str, horizon: int) -> str:
    return f"{dataset}|{target}|h{int(horizon)}"


def cell_id(dataset: str, target: str, horizon: int, seed: int, arm: str) -> str:
    return f"{dataset}|{target}|h{int(horizon)}|s{int(seed)}|{arm}"


def canonical_cell_id(dataset: str, target: str, horizon: int, seed: int) -> str:
    return f"{dataset}|{target}|h{int(horizon)}|s{int(seed)}"


def filesystem_safe_id(value: str) -> str:
    return str(value).replace("\\", "_").replace("/", "_").replace("|", "__")


def classify_layer_role(layer_name: str) -> str:
    name = str(layer_name)
    if name in TRANSFERRED_BACKBONE_LAYERS:
        return "transferred_backbone"
    if name == PREDICTION_HEAD_LAYER:
        return "prediction_head"
    if name.startswith("adapter") or name in {
        "residual_gate",
        "head_transfer",
        "cross_attention",
        "dropout",
        "dense_out",
    }:
        return "non_transferred_layer"
    return "other"


def primary_transferred_backbone_weights(
    named_weights: Mapping[str, Sequence[np.ndarray]],
) -> Tuple[np.ndarray, ...]:
    arrays: List[np.ndarray] = []
    for name in sorted(named_weights):
        role = classify_layer_role(name)
        if name == PREDICTION_HEAD_LAYER or role == "prediction_head":
            continue
        if role != "transferred_backbone":
            continue
        for weight in named_weights[name]:
            arrays.append(np.asarray(weight))
    if not arrays:
        raise MechanismDiagnosticsContractError("PRIMARY drift has no transferred_backbone weights")
    return tuple(arrays)


def select_formal_test_reference_inputs(bundle: Any) -> np.ndarray:
    x_test = getattr(bundle, "X_test", None)
    if x_test is None and isinstance(bundle, Mapping):
        x_test = bundle.get("X_test")
    if x_test is None:
        raise MechanismDiagnosticsContractError("formal_test_inputs require bundle.X_test")
    array = np.array(np.asarray(x_test), copy=True)
    if array.ndim < 1 or int(array.shape[0]) < 2:
        raise MechanismDiagnosticsContractError("formal test reference is empty")
    return array


def reference_manifest_rows(
    x_test: np.ndarray,
    *,
    sample_ids: Optional[Sequence[Any]] = None,
    dates: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    n = int(np.asarray(x_test).shape[0])
    rows = []
    for i in range(n):
        sample_id = sample_ids[i] if sample_ids is not None and i < len(sample_ids) else i
        date_value = dates[i] if dates is not None and i < len(dates) else i
        rows.append(
            {
                "sample_id": sample_id,
                "date_or_prediction_index": date_value,
                "split": FORMAL_TEST_REFERENCE_SPLIT,
                "input_shape": ",".join(str(int(v)) for v in np.asarray(x_test).shape[1:]),
            }
        )
    return rows


def assert_one_condition_one_digest(digest_by_condition: Mapping[str, Sequence[str]]) -> None:
    for cid, digests in digest_by_condition.items():
        unique = {str(item) for item in digests}
        if len(unique) != 1:
            raise MechanismDiagnosticsContractError(
                "one condition -> exactly one reference_input_digest "
                f"violated for {cid}: {sorted(unique)}"
            )


def save_condition_reference(
    run_dir: Path,
    cid: str,
    x_test: np.ndarray,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    ref_dir = Path(run_dir) / "cka" / "reference_inputs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    cid_fs = filesystem_safe_id(cid)
    npz_path = ref_dir / f"{cid_fs}.npz"
    csv_path = ref_dir / f"{cid_fs}_manifest.csv"
    digest = reference_input_digest(np.asarray(x_test))
    np.savez_compressed(npz_path, x=np.asarray(x_test), digest=np.asarray(digest))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "date_or_prediction_index", "split", "input_shape"]
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return {
        "npz": str(npz_path),
        "manifest": str(csv_path),
        "digest": digest,
        "split": FORMAL_TEST_REFERENCE_SPLIT,
    }


def original_scale_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def original_scale_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(smape(y_true, y_pred))


def invert_sales_minmax(
    values: np.ndarray,
    *,
    data_min: float,
    data_max: float,
) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * (float(data_max) - float(data_min)) + float(data_min)


def record_diagnostic_epoch_eval(
    rows: List[Dict[str, Any]],
    *,
    payload: Mapping[str, object],
    constructed: Any,
    bundle: Any,
    cell: Any,
    arm: str,
) -> Dict[str, Any]:
    """Record epoch 0 / epoch 1...E diagnostic metrics. Live model eval is optional."""
    row: Dict[str, Any] = {
        "dataset": getattr(cell, "dataset", None),
        "target": getattr(cell, "target", None),
        "horizon": int(getattr(cell, "horizon", 0) or 0),
        "seed": int(getattr(cell, "seed", 0) or 0),
        "arm": arm,
        "epoch": int(payload.get("epoch", 0)),
        "before_first_target_optimizer_update": bool(
            payload.get("before_first_target_optimizer_update", False)
        ),
        "weight_variant": str(payload.get("weight_variant") or FORMAL_EVALUATION_WEIGHT_VARIANT),
        "train_objective_loss": payload.get("train_objective_loss"),
        "validation_objective_loss": payload.get("validation_objective_loss"),
        "validation_rmse_original_scale": payload.get("validation_rmse_original_scale"),
        "validation_smape_original_scale": payload.get("validation_smape_original_scale"),
        "learning_rate": payload.get("learning_rate"),
        "gradient_norm": payload.get("gradient_norm"),
        "parameter_norm": payload.get("parameter_norm"),
        "ema_parameter_norm": payload.get("ema_parameter_norm"),
        "ema_applied": payload.get("ema_applied"),
        "optimizer_update_count": payload.get("optimizer_update_count"),
        "ema_update_count": payload.get("ema_update_count"),
    }
    if row["validation_rmse_original_scale"] is None:
        y_true = payload.get("y_true_original")
        y_pred = payload.get("y_pred_original")
        if y_true is not None and y_pred is not None:
            row["validation_rmse_original_scale"] = original_scale_rmse(y_true, y_pred)
            row["validation_smape_original_scale"] = original_scale_smape(y_true, y_pred)
    if row["validation_rmse_original_scale"] is None and _can_live_eval(constructed, bundle):
        live = _live_original_val_metrics(constructed, bundle)
        row.update(live)
    rows.append(row)
    return row


def _can_live_eval(constructed: Any, bundle: Any) -> bool:
    head = getattr(constructed, "head", None) if constructed is not None else None
    return (
        head is not None
        and callable(getattr(head, "call", None))
        and getattr(bundle, "X_val", None) is not None
        and getattr(bundle, "y_val", None) is not None
        and getattr(bundle, "f_t", None) is not None
        and getattr(bundle, "target_scaler", None) is not None
    )


def _live_original_val_metrics(constructed: Any, bundle: Any) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_lite import extract_features_maybe_revin, predict_ms_katl
    from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

    x_val = np.asarray(bundle.X_val)
    y_val = np.asarray(bundle.y_val)
    f_t_val = bundle.f_t["val"]
    z_val, mean, stdev = extract_features_maybe_revin(
        constructed.target_extractor, x_val, revin=constructed.revin, training=False
    )
    y_pred = predict_ms_katl(constructed.head, z_val, f_t_val, bundle.stage_a.knn_prior)
    if constructed.revin is not None and mean is not None:
        y_pred = constructed.revin.denormalize_numpy(
            y_pred,
            mean.numpy() if hasattr(mean, "numpy") else mean,
            stdev.numpy() if hasattr(stdev, "numpy") else stdev,
            include_mean=True,
        )
    metrics = compute_screening_metrics(
        y_val,
        y_pred,
        target_scaler=bundle.target_scaler,
        feature_columns=list(bundle.target_feature_columns),
    )
    residual = np.asarray(y_val, dtype=np.float64).reshape(-1) - np.asarray(y_pred, dtype=np.float64).reshape(-1)
    val_obj = float(np.mean(residual * residual))
    return {
        "validation_objective_loss": val_obj,
        "validation_rmse_original_scale": float(metrics["rmse_original"]),
        "validation_smape_original_scale": float(metrics["smape_original"]),
    }


def pointwise_prediction_rows(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    arm: str,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
    sample_ids: Optional[Sequence[Any]] = None,
    dates: Optional[Sequence[Any]] = None,
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
    y_pred_ema_original: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    true = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred_original, dtype=np.float64).reshape(-1)
    pred_ema = pred if y_pred_ema_original is None else np.asarray(y_pred_ema_original, dtype=np.float64).reshape(-1)
    if pred_ema.shape != pred.shape:
        raise MechanismDiagnosticsContractError("pointwise EMA prediction length mismatch")
    if true.shape != pred.shape:
        raise MechanismDiagnosticsContractError("pointwise y_true/y_pred length mismatch")
    rows = []
    for i in range(int(true.shape[0])):
        signed = float(pred[i] - true[i])
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "horizon": int(horizon),
                "seed": int(seed),
                "arm": arm,
                "sample_id": sample_ids[i] if sample_ids is not None else i,
                "sample_index": i,
                "date_or_prediction_index": dates[i] if dates is not None else i,
                "y_true": float(true[i]),
                "y_pred": float(pred[i]),
                "y_pred_ema_original_scale": float(pred_ema[i]),
                "signed_error": signed,
                "absolute_error": abs(signed),
                "squared_error": signed * signed,
                "weight_variant": weight_variant,
            }
        )
    return rows


def assert_paired_sample_ids(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> None:
    left_ids = [row["sample_id"] for row in left]
    right_ids = [row["sample_id"] for row in right]
    if left_ids != right_ids:
        raise MechanismDiagnosticsContractError("B/C prediction sample IDs are not aligned")


def error_by_sales_bin(
    rows: Sequence[Mapping[str, Any]],
    *,
    notes: str = "",
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    identifiers = {
        field: rows[0].get(field)
        for field in ("dataset", "target", "horizon", "seed", "arm")
    }
    if any(value is None or str(value).strip() == "" for value in identifiers.values()):
        raise MechanismDiagnosticsContractError(
            "error_by_sales_bin requires non-empty dataset/target/horizon/seed/arm identifiers"
        )
    for row in rows:
        for field, expected in identifiers.items():
            value = row.get(field)
            if value is None or str(value).strip() == "" or str(value) != str(expected):
                raise MechanismDiagnosticsContractError(
                    f"error_by_sales_bin identifier mismatch or blank: {field}={value!r}"
                )
    y_true = np.asarray([row["y_true"] for row in rows], dtype=np.float64)
    sq = np.asarray([row["squared_error"] for row in rows], dtype=np.float64)
    abs_err = np.asarray([row["absolute_error"] for row in rows], dtype=np.float64)
    total_sse = float(np.sum(sq))
    qs = np.quantile(y_true, [1.0 / 3.0, 2.0 / 3.0]) if y_true.size else np.array([])
    unique_edges = np.unique(qs) if y_true.size else np.array([])
    if unique_edges.size < 2:
        labels = ["all"]
        masks = [np.ones(y_true.shape[0], dtype=bool)]
        note = notes or "could not form 3 tertile bins; using actual available bin"
    else:
        edges = np.unique(np.concatenate([[y_true.min() - 1e-12], unique_edges, [y_true.max() + 1e-12]]))
        n_bins = max(1, len(edges) - 1)
        labels = ["low", "middle", "high"][:n_bins]
        masks = []
        for i in range(len(edges) - 1):
            if i == len(edges) - 2:
                masks.append((y_true >= edges[i]) & (y_true <= edges[i + 1]))
            else:
                masks.append((y_true >= edges[i]) & (y_true < edges[i + 1]))
        note = notes
        if len(masks) != 3:
            note = note or "could not form 3 tertile bins; using actual available bins"
    out = []
    for label, mask in zip(labels, masks):
        n = int(np.sum(mask))
        if n == 0:
            continue
        sse = float(np.sum(sq[mask]))
        out.append(
            {
                **identifiers,
                "bin": label,
                "n": n,
                "MAE": float(np.mean(abs_err[mask])),
                "RMSE": float(np.sqrt(np.mean(sq[mask]))),
                "mean_squared_error": float(np.mean(sq[mask])),
                "sum_squared_error": sse,
                "squared_error_contribution_pct": (100.0 * sse / total_sse) if total_sse else 0.0,
                "notes": note,
            }
        )
    return out


def unordered_seed_pairs(seeds: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    ordered = sorted(int(s) for s in seeds)
    return tuple((ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered)))


def cka_row(
    *,
    comparison_type: str,
    left: np.ndarray,
    right: np.ndarray,
    weight_variant: str,
    reference_input_digest_value: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    left_m = np.asarray(left)
    right_m = np.asarray(right)
    n_samples = int(left_m.shape[0])
    payload = {
        "comparison_type": comparison_type,
        "cka": linear_cka(left_m, right_m),
        "n_samples": n_samples,
        "n_features_left": int(left_m.shape[1]),
        "n_features_right": int(right_m.shape[1]),
        "weight_variant": weight_variant,
        "reference_input_digest": reference_input_digest_value,
        "low_reference_sample_count": n_samples < LOW_REFERENCE_SAMPLE_COUNT,
        "flag": "LOW_REFERENCE_SAMPLE_COUNT" if n_samples < LOW_REFERENCE_SAMPLE_COUNT else "",
    }
    if extra:
        payload.update(dict(extra))
    return payload


def generate_cka_comparison_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_scale_available: bool,
    include_c_init_b_final: bool = True,
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
) -> List[Dict[str, Any]]:
    """records: arm, seed, stage, representations, reference_input_digest."""
    by_key = {
        (str(item["arm"]), int(item["seed"]), str(item["stage"])): item
        for item in records
    }
    seeds = sorted({int(item["seed"]) for item in records})
    digest_values = [str(item["reference_input_digest"]) for item in records]
    if digest_values and len(set(digest_values)) != 1:
        raise MechanismDiagnosticsContractError("CKA matrix mixed reference_input_digest within a condition")
    digest = digest_values[0] if digest_values else ""
    layer_values = [str(item.get("layer") or "") for item in records if item.get("layer")]
    layer = layer_values[0] if layer_values and len(set(layer_values)) == 1 else ""
    rows: List[Dict[str, Any]] = []

    def _rep(arm: str, seed: int, stage: str) -> Optional[np.ndarray]:
        item = by_key.get((arm, int(seed), stage))
        if item is None:
            return None
        return np.asarray(item["representations"])

    for seed in seeds:
        b_final = _rep("B", seed, "FINAL")
        c_final = _rep("C", seed, "FINAL")
        c_init = _rep("C", seed, "INIT")
        a_final = _rep("A", seed, "FINAL")
        extra = {"seed": seed, "reference_scale_available": reference_scale_available}
        if layer:
            extra["layer"] = layer
        if b_final is not None and c_final is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_B_C_MATCHED_SEED,
                    left=b_final,
                    right=c_final,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
        if c_init is not None and c_final is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
                    left=c_init,
                    right=c_final,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
        c_raw_final = _rep("C", seed, "RAW_FINAL")
        if c_init is not None and c_raw_final is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_C_INIT_C_RAW_FINAL,
                    left=c_init,
                    right=c_raw_final,
                    weight_variant="RAW",
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
        if a_final is not None and c_final is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_A_C_MATCHED_SEED,
                    left=a_final,
                    right=c_final,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
        if include_c_init_b_final and c_init is not None and b_final is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
                    left=c_init,
                    right=b_final,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )

    for left_seed, right_seed in unordered_seed_pairs(seeds):
        extra = {
            "seed_left": left_seed,
            "seed_right": right_seed,
            "reference_scale_available": reference_scale_available,
        }
        if layer:
            extra["layer"] = layer
        b_left = _rep("B", left_seed, "FINAL")
        b_right = _rep("B", right_seed, "FINAL")
        if b_left is not None and b_right is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_B_B_CROSS_SEED,
                    left=b_left,
                    right=b_right,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
        c_left = _rep("C", left_seed, "FINAL")
        c_right = _rep("C", right_seed, "FINAL")
        if c_left is not None and c_right is not None:
            rows.append(
                cka_row(
                    comparison_type=COMPARISON_C_C_CROSS_SEED,
                    left=c_left,
                    right=c_right,
                    weight_variant=weight_variant,
                    reference_input_digest_value=digest,
                    extra=extra,
                )
            )
    return rows


def _rel_diff_pct(formal: float, rerun: float) -> Tuple[float, float, str]:
    abs_diff = abs(float(rerun) - float(formal))
    if abs(float(formal)) < 1e-12:
        return abs_diff, abs_diff, "formal metric near 0; using absolute difference"
    rel = 100.0 * abs_diff / abs(float(formal))
    return abs_diff, rel, ""


def audit_status_from_diffs(rel_rmse_pct: float, rel_smape_pct: float) -> str:
    if rel_rmse_pct < CONSISTENCY_PASS_PCT and rel_smape_pct < CONSISTENCY_PASS_PCT:
        return "PASS"
    if rel_rmse_pct > CONSISTENCY_REVIEW_PCT or rel_smape_pct > CONSISTENCY_REVIEW_PCT:
        return "FAIL"
    return "REVIEW"


def consistency_audit_row(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    arm: str,
    formal_rmse: float,
    rerun_rmse: float,
    formal_smape: float,
    rerun_smape: float,
    weight_variant: str = FORMAL_EVALUATION_WEIGHT_VARIANT,
    same_protocol_metadata: bool = True,
    extra_notes: str = "",
    comparison_purpose: str = "historical_reference_only",
) -> Dict[str, Any]:
    abs_rmse, rel_rmse, note_rmse = _rel_diff_pct(formal_rmse, rerun_rmse)
    abs_smape, rel_smape, note_smape = _rel_diff_pct(formal_smape, rerun_smape)
    notes = "; ".join(
        item for item in (note_rmse, note_smape, extra_notes, "formal CSV rounding may apply") if item
    )
    status = audit_status_from_diffs(rel_rmse, rel_smape)
    return {
        "dataset": dataset,
        "target": target,
        "horizon": int(horizon),
        "seed": int(seed),
        "arm": arm,
        "formal_rmse": float(formal_rmse),
        "rerun_rmse": float(rerun_rmse),
        "abs_rmse_diff": abs_rmse,
        "rel_rmse_diff_pct": rel_rmse,
        "formal_smape": float(formal_smape),
        "rerun_smape": float(rerun_smape),
        "abs_smape_diff": abs_smape,
        "rel_smape_diff_pct": rel_smape,
        "weight_variant": weight_variant,
        "same_protocol_metadata": bool(same_protocol_metadata),
        "audit_status": status,
        "notes": notes,
        "replaces_frozen_results": False,
        "comparison_purpose": comparison_purpose,
        "blocks_execution": False,
    }


def timestamped_run_root(
    now: Optional[datetime] = None,
    *,
    matrix_mode: str = "diagnostic-42",
) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    prefix = (
        FORMAL_300_OUTPUT_PREFIX
        if str(matrix_mode).strip().lower() == "formal-300"
        else DIAGNOSTIC_42_OUTPUT_PREFIX
    )
    return OUTPUT_ROOT_PARENT / f"{prefix}_{stamp}"


def create_run_root(
    base: Optional[Path] = None,
    now: Optional[datetime] = None,
    *,
    matrix_mode: str = "diagnostic-42",
) -> Path:
    named = timestamped_run_root(now, matrix_mode=matrix_mode)
    root = Path(base) / named.name if base is not None else named
    if root.exists():
        raise MechanismDiagnosticsContractError(f"run root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def formal_reference_result_digest(path: Path = FORMAL_AE_UNIFORM_3E4_CSV) -> str:
    reference = Path(path)
    if not reference.is_file():
        raise MechanismDiagnosticsContractError(f"formal reference result missing: {reference}")
    return hashlib.sha256(reference.read_bytes()).hexdigest()


def cell_artifact_directory(run_dir: Path, cid: str) -> Path:
    return Path(run_dir) / "cells" / filesystem_safe_id(cid)


def canonical_stage_a_path(run_dir: Path, canonical_id: str) -> Path:
    return Path(run_dir) / "stage_a" / f"{filesystem_safe_id(canonical_id)}.json"


def write_canonical_stage_a_artifact(
    run_dir: Path,
    cell: Any,
    payload: Mapping[str, Any],
) -> Path:
    """Write one immutable Stage-A record for a canonical cell.

    Repeated arm calls may verify the existing record, but never replace it.
    """
    canonical_id = str(
        getattr(cell, "canonical_cell_key", "")
        or canonical_cell_id(cell.dataset, cell.target, int(cell.horizon), int(cell.seed))
    )
    path = canonical_stage_a_path(run_dir, canonical_id)
    normalized = {"canonical_cell_id": canonical_id, **dict(payload)}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        identity_fields = (
            "canonical_cell_id",
            "stage_a_identifier",
            "stage_a_digest",
            "selected_source_identities",
            "source_distances",
            "source_weights",
            "source_model_digests",
            "fused_conv1_digest",
            "fused_conv2_digest",
            "fused_conv3_digest",
            "fused_params_digest",
            "target_split_digest",
            "reference_input_digest",
        )
        for field in identity_fields:
            if existing.get(field) != normalized.get(field):
                raise MechanismDiagnosticsContractError(
                    f"Stage-A shared artifact drifted for {canonical_id}: {field}"
                )
        return path
    write_json_atomic(path, normalized)
    return path


def write_cell_json_artifacts(
    run_dir: Path,
    cid: str,
    *,
    result: Mapping[str, Any],
    status: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    root = cell_artifact_directory(run_dir, cid)
    paths = {
        "result": root / "result.json",
        "status": root / "status.json",
        "provenance": root / "provenance.json",
    }
    write_json_atomic(paths["result"], result)
    write_json_atomic(paths["status"], status)
    write_json_atomic(paths["provenance"], provenance)
    return paths


def write_cell_parameter_snapshots(
    run_dir: Path,
    cid: str,
    named_snapshots: Mapping[Tuple[str, str], Mapping[str, Sequence[np.ndarray]]],
) -> Path:
    path = Path(run_dir) / "parameter_snapshots" / f"{filesystem_safe_id(cid)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for (arm, stage), named in sorted(named_snapshots.items()):
        for layer, weights in sorted(named.items()):
            for index, weight in enumerate(weights):
                arrays[f"{arm}_{stage}_{layer}_{index}"] = np.asarray(weight)
    if not arrays:
        raise MechanismDiagnosticsContractError("parameter snapshot payload is empty")
    np.savez_compressed(path, **arrays)
    return path


def write_cell_parameter_drift_summary(
    run_dir: Path,
    cid: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    backbone = [
        row for row in rows if str(row.get("layer_role")) == "transferred_backbone"
    ]
    values = [
        float(row["relative_parameter_drift"])
        for row in backbone
        if row.get("relative_parameter_drift") not in (None, "")
    ]
    by_layer = {str(row.get("layer")): row for row in rows}
    def _value(layer: str, field: str) -> float:
        value = by_layer.get(layer, {}).get(field, float("nan"))
        return float(value)

    backbone_l2 = [
        _value(layer, "parameter_l2")
        for layer in TRANSFERRED_BACKBONE_LAYERS
        if layer in by_layer
    ]
    backbone_rel = [
        _value(layer, "relative_parameter_drift")
        for layer in TRANSFERRED_BACKBONE_LAYERS
        if layer in by_layer
    ]
    payload = {
        "cell_id": cid,
        "n_layers": len(rows),
        "transferred_backbone_layers": [str(row.get("layer")) for row in backbone],
        "primary_relative_l2_drift": float(np.mean(values)) if values else float("nan"),
        "backbone_zero_drift": bool(values) and all(abs(value) <= 1e-12 for value in values),
        "conv1_l2": _value("conv1", "parameter_l2"),
        "conv2_l2": _value("conv2", "parameter_l2"),
        "conv3_l2": _value("conv3", "parameter_l2"),
        "head_target_l2": _value("head_target", "parameter_l2"),
        "whole_backbone_l2": float(np.sqrt(np.sum(np.square(backbone_l2)))) if backbone_l2 else float("nan"),
        "conv1_relative_parameter_drift": _value("conv1", "relative_parameter_drift"),
        "conv2_relative_parameter_drift": _value("conv2", "relative_parameter_drift"),
        "conv3_relative_parameter_drift": _value("conv3", "relative_parameter_drift"),
        "head_target_relative_parameter_drift": _value("head_target", "relative_parameter_drift"),
        "whole_backbone_relative_parameter_drift": float(np.sqrt(np.sum(np.square(backbone_rel)))) if backbone_rel else float("nan"),
    }
    path = cell_artifact_directory(run_dir, cid) / "parameter_drift_summary.json"
    write_json_atomic(path, payload)
    return path


def mark_cell_complete(run_dir: Path, cid: str, payload: Mapping[str, Any]) -> None:
    status_dir = Path(run_dir) / "status"
    safe = filesystem_safe_id(cid)
    write_json_atomic(status_dir / f"{safe}.json", dict(payload, status=CELL_STATUS_COMPLETE, cell_id=cid))
    complete = status_dir / f"{safe}.COMPLETE"
    tmp = complete.with_name(complete.name + ".tmp")
    tmp.write_text("COMPLETE\n", encoding="utf-8")
    os.replace(tmp, complete)


def mark_cell_failed(run_dir: Path, cid: str, payload: Mapping[str, Any]) -> None:
    write_json_atomic(
        Path(run_dir) / "status" / f"{filesystem_safe_id(cid)}.json",
        dict(payload, status=CELL_STATUS_FAILED, cell_id=cid),
    )


def resume_cell_decision(
    run_dir: Path,
    cid: str,
    *,
    expected_config_digest: str,
    expected_matrix_digest: str,
    stored_config_digest: Optional[str],
    stored_matrix_digest: Optional[str],
    expected_code_digest: Optional[str] = None,
    stored_code_digest: Optional[str] = None,
) -> str:
    if stored_config_digest is not None and stored_config_digest != expected_config_digest:
        raise MechanismDiagnosticsContractError("config digest mismatch")
    if stored_matrix_digest is not None and stored_matrix_digest != expected_matrix_digest:
        raise MechanismDiagnosticsContractError("matrix digest mismatch")
    if expected_code_digest is not None and stored_code_digest is not None:
        if stored_code_digest != expected_code_digest:
            raise MechanismDiagnosticsContractError("code digest mismatch")
    status_path = Path(run_dir) / "status" / f"{filesystem_safe_id(cid)}.json"
    if required_cell_outputs_complete(run_dir, cid):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("status") == CELL_STATUS_COMPLETE:
            return "SKIP"
    if status_path.is_file():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("status") == CELL_STATUS_FAILED:
            return "RERUN"
        if payload.get("status") == CELL_STATUS_COMPLETE and not required_cell_outputs_complete(run_dir, cid):
            return "RERUN"
        if payload.get("status") in {CELL_STATUS_PENDING, CELL_STATUS_RUNNING}:
            return "RERUN"
    return "RUN"


def load_run_digests(run_dir: Path) -> Dict[str, str]:
    manifest = Path(run_dir) / "config_digest.json"
    if not manifest.is_file():
        raise MechanismDiagnosticsContractError("resume run dir missing config_digest.json")
    return json.loads(manifest.read_text(encoding="utf-8"))


def persist_condition_reference_if_needed(
    context: Mapping[str, Any],
    *,
    cell: Any,
    x_test: np.ndarray,
    digest: str,
    sample_ids: Optional[Sequence[Any]] = None,
    dates: Optional[Sequence[Any]] = None,
) -> Optional[Dict[str, str]]:
    run_dir = context.get("mechanism_diagnostics_run_dir") if isinstance(context, Mapping) else None
    if not run_dir:
        return None
    cid = condition_id(str(cell.dataset), str(cell.target), int(cell.horizon))
    store = context.setdefault("cka_condition_references", {}) if isinstance(context, dict) else {}
    prior = store.get(cid, {}) if isinstance(store, dict) else {}
    if prior.get("persisted"):
        if str(prior.get("digest")) != str(digest):
            raise MechanismDiagnosticsContractError(
                f"one condition -> exactly one reference_input_digest violated for {cid}"
            )
        return {"digest": str(prior["digest"]), "split": FORMAL_TEST_REFERENCE_SPLIT}
    saved = save_condition_reference(
        Path(run_dir),
        cid,
        np.asarray(x_test),
        reference_manifest_rows(np.asarray(x_test), sample_ids=sample_ids, dates=dates),
    )
    if saved["digest"] != digest:
        raise MechanismDiagnosticsContractError("persisted CKA reference digest drifted")
    if isinstance(store, dict):
        store.setdefault(cid, {})
        store[cid]["persisted"] = True
        store[cid]["digest"] = digest
        store[cid]["split"] = FORMAL_TEST_REFERENCE_SPLIT
    return saved


def parameter_drift_rows(
    named_init: Mapping[str, Sequence[np.ndarray]],
    named_final: Mapping[str, Sequence[np.ndarray]],
    *,
    extra: Optional[Mapping[str, Any]] = None,
    final_variant: str = "EMA_FINAL",
) -> List[Dict[str, Any]]:
    names = sorted(set(named_init) | set(named_final))
    rows: List[Dict[str, Any]] = []
    for name in names:
        role = classify_layer_role(name)
        init_w = list(named_init.get(name, ()))
        final_w = list(named_final.get(name, ()))
        if init_w and final_w:
            l2 = parameter_l2_distance(init_w, final_w)
            rel = relative_parameter_drift(init_w, final_w)
            init_norm = parameter_frobenius_norm(init_w)
            final_norm = parameter_frobenius_norm(final_w)
        else:
            l2 = rel = init_norm = final_norm = float("nan")
        row = {
            "layer": name,
            "layer_role": role,
            "parameter_l2": l2,
            "relative_parameter_drift": rel,
            "absolute_drift": l2,
            "relative_drift": rel,
            "init_norm": init_norm,
            "final_norm": final_norm,
            "final_variant": final_variant,
        }
        if extra:
            row.update(dict(extra))
        rows.append(row)
    return rows


def primary_drift_from_named(
    named_init: Mapping[str, Sequence[np.ndarray]],
    named_final: Mapping[str, Sequence[np.ndarray]],
) -> Tuple[np.ndarray, ...]:
    init_w = primary_transferred_backbone_weights(named_init)
    _ = primary_transferred_backbone_weights(named_final)
    for name in list(named_init) + list(named_final):
        if classify_layer_role(name) == "prediction_head" or name == PREDICTION_HEAD_LAYER:
            if name in TRANSFERRED_BACKBONE_LAYERS:
                raise MechanismDiagnosticsContractError("head_target mixed into transferred_backbone")
    return init_w


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_pointwise_predictions(run_dir: Path, cid: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "predictions" / f"{filesystem_safe_id(cid)}.csv"
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "sample_id",
            "sample_index",
            "date_or_prediction_index",
            "y_true",
            "y_pred",
            "y_pred_ema_original_scale",
            "signed_error",
            "absolute_error",
            "squared_error",
            "weight_variant",
        ],
    )
    return path


def write_training_dynamics(run_dir: Path, cid: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "training_dynamics" / f"{filesystem_safe_id(cid)}.csv"
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "epoch",
            "before_first_target_optimizer_update",
            "train_objective_loss",
            "validation_objective_loss",
            "validation_rmse_original_scale",
            "validation_smape_original_scale",
            "learning_rate",
            "gradient_norm",
            "parameter_norm",
            "ema_parameter_norm",
            "ema_applied",
            "weight_variant",
            "optimizer_update_count",
            "ema_update_count",
        ],
    )
    return path


def write_error_by_sales_bin(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "error_by_sales_bin.csv"
    assert_error_bin_identifiers(rows)
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "bin",
            "n",
            "MAE",
            "RMSE",
            "mean_squared_error",
            "sum_squared_error",
            "squared_error_contribution_pct",
            "notes",
        ],
    )
    return path


def write_pointwise_summary(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "pointwise_summary.csv"
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "n_samples",
            "weight_variant",
            "reconstructed_rmse",
            "reconstructed_smape",
            "max_abs_y_pred_minus_y_pred_ema",
            "y_pred_equals_y_pred_ema",
            "prediction_digest",
        ],
    )
    return path


def assert_error_bin_identifiers(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed after summary generation if any identifier is missing."""
    for row in rows:
        identifiers = {field: row.get(field) for field in ERROR_BIN_IDENTIFIER_FIELDS}
        if any(value is None or str(value).strip() == "" for value in identifiers.values()):
            raise MechanismDiagnosticsContractError(
                f"error_by_sales_bin identifiers.notna().all() failed: {identifiers!r}"
            )
    # Keep the contract explicit without requiring pandas for this CSV writer.
    identifiers_notna = [
        all(row.get(field) is not None and str(row.get(field)).strip() != "" for field in ERROR_BIN_IDENTIFIER_FIELDS)
        for row in rows
    ]
    assert all(identifiers_notna), "error_by_sales_bin identifiers.notna().all() failed"


def write_historical_frozen_comparison(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist optional historical comparison outside the diagnostic result set."""
    path = Path(run_dir) / "summaries" / HISTORICAL_COMPARISON_FILENAME
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "formal_rmse",
            "rerun_rmse",
            "abs_rmse_diff",
            "rel_rmse_diff_pct",
            "formal_smape",
            "rerun_smape",
            "abs_smape_diff",
            "rel_smape_diff_pct",
            "weight_variant",
            "same_protocol_metadata",
            "audit_status",
            "notes",
            "comparison_purpose",
            "blocks_execution",
        ],
    )
    return path


# Kept as an import-level compatibility alias; it intentionally writes the
# renamed historical-reference file and is never a run-success gate.
write_consistency_audit = write_historical_frozen_comparison


def write_cell_status_summary(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "cell_status.csv"
    write_csv(
        path,
        rows,
        ["cell_id", "dataset", "target", "horizon", "seed", "arm", "status", "notes"],
    )
    return path


def write_cell_checkpoints(run_dir: Path, cid: str, payload: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / "checkpoints" / f"{filesystem_safe_id(cid)}.json"
    write_json_atomic(path, payload)
    return path


def write_cell_representations(
    run_dir: Path,
    cid: str,
    *,
    init_rep: np.ndarray,
    final_rep: np.ndarray,
    reference_input_digest_value: str,
    raw_final_rep: Optional[np.ndarray] = None,
) -> Path:
    path = Path(run_dir) / "representations" / f"{filesystem_safe_id(cid)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "INIT": np.asarray(init_rep),
        "FINAL": np.asarray(final_rep),
        "reference_input_digest": np.asarray(reference_input_digest_value),
    }
    if raw_final_rep is not None:
        arrays["RAW_FINAL"] = np.asarray(raw_final_rep)
    np.savez_compressed(path, **arrays)
    return path


def write_layer_representations(
    run_dir: Path,
    cid: str,
    layer: str,
    *,
    arrays_by_stage: Mapping[str, np.ndarray],
    reference_input_digest_value: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = Path(run_dir) / "representations" / str(layer) / f"{filesystem_safe_id(cid)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(stage): np.asarray(array)
        for stage, array in arrays_by_stage.items()
    }
    payload["reference_input_digest"] = np.asarray(reference_input_digest_value)
    payload["layer"] = np.asarray(str(layer))
    payload["reshape_rule"] = np.asarray(CKA_ACTIVATION_RESHAPE_RULE)
    if metadata:
        payload["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez_compressed(path, **payload)
    return path


def write_parameter_drift_variant_summary(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    filename: str,
) -> Path:
    path = Path(run_dir) / "summaries" / filename
    write_csv(
        path,
        rows,
        [
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "layer",
            "layer_role",
            "final_variant",
            "absolute_drift",
            "relative_drift",
            "init_norm",
            "final_norm",
        ],
    )
    return path


def write_formal_consistency(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = Path(run_dir) / "summaries" / "formal_consistency.csv"
    fieldnames = [
        "dataset",
        "target",
        "horizon",
        "seed",
        "arm",
        "formal_selected_lr",
        "diagnostic_selected_lr",
        "formal_rmse",
        "diagnostic_rmse",
        "abs_rmse_diff",
        "rel_rmse_diff",
        "formal_smape",
        "diagnostic_smape",
        "abs_smape_diff",
        "rel_smape_diff",
        "prediction_digest_match",
        "final_formal_model_digest_match",
        "stage_a_digest_match",
        "fused_init_digest_match",
        "identity_status",
        "severity_status",
        "formal_prediction_digest",
        "diagnostic_prediction_digest",
        "formal_final_formal_model_digest",
        "diagnostic_final_formal_model_digest",
        "formal_stage_a_digest",
        "diagnostic_stage_a_digest",
        "formal_fused_init_digest",
        "diagnostic_fused_init_digest",
        "authority_root",
        "authority_kind",
    ]
    write_csv(path, rows, fieldnames)
    return path


def write_cell_parameter_drift(run_dir: Path, cid: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "parameter_drift" / f"{filesystem_safe_id(cid)}.csv"
    write_csv(
        path,
        rows,
        [
            "layer",
            "layer_role",
            "parameter_l2",
            "relative_parameter_drift",
            "absolute_drift",
            "relative_drift",
            "init_norm",
            "final_norm",
            "final_variant",
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
        ],
    )
    return path


def write_training_dynamics_summary(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "training_dynamics_summary.csv"
    write_csv(
        path,
        rows,
        [
            "cell_id",
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "n_epochs",
            "epoch0_present",
            "weight_variant",
        ],
    )
    return path


def write_parameter_drift_summary(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "parameter_drift_summary.csv"
    write_csv(
        path,
        rows,
        [
            "cell_id",
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "n_layers",
            "primary_relative_l2_drift",
            "conv1_l2",
            "conv2_l2",
            "conv3_l2",
            "head_target_l2",
            "whole_backbone_l2",
            "conv1_relative_parameter_drift",
            "conv2_relative_parameter_drift",
            "conv3_relative_parameter_drift",
            "head_target_relative_parameter_drift",
            "whole_backbone_relative_parameter_drift",
        ],
    )
    return path


def write_layerwise_parameter_drift(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "layerwise_parameter_drift.csv"
    write_csv(
        path,
        rows,
        [
            "cell_id",
            "dataset",
            "target",
            "horizon",
            "seed",
            "arm",
            "layer",
            "layer_role",
            "parameter_l2",
            "relative_parameter_drift",
        ],
    )
    return path


def write_cka_pairwise_matrix(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "cka_pairwise_matrix.csv"
    fieldnames = sorted({key for row in rows for key in row}) or [
        "comparison_type",
        "cka",
        "weight_variant",
        "reference_input_digest",
    ]
    write_csv(path, rows, fieldnames)
    return path


def write_cka_condition_summary(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "cka_condition_summary.csv"
    write_csv(
        path,
        rows,
        [
            "condition_id",
            "layer",
            "n_comparisons",
            "reference_input_digest",
            "weight_variant",
        ],
    )
    return path


def required_cell_output_paths(run_dir: Path, cid: str) -> Tuple[Path, ...]:
    root = Path(run_dir)
    safe = filesystem_safe_id(cid)
    return (
        cell_artifact_directory(root, cid) / "result.json",
        cell_artifact_directory(root, cid) / "status.json",
        cell_artifact_directory(root, cid) / "provenance.json",
        cell_artifact_directory(root, cid) / "parameter_drift_summary.json",
        root / "status" / f"{safe}.json",
        root / "status" / f"{safe}.COMPLETE",
        root / "predictions" / f"{safe}.csv",
        root / "training_dynamics" / f"{safe}.csv",
        root / "checkpoints" / f"{safe}.json",
        root / "representations" / f"{safe}.npz",
        root / "parameter_snapshots" / f"{safe}.npz",
        root / "parameter_drift" / f"{safe}.csv",
        root / "representations" / "flatten" / f"{safe}.npz",
        root / "representations" / "conv1" / f"{safe}.npz",
        root / "representations" / "conv2" / f"{safe}.npz",
        root / "representations" / "conv3" / f"{safe}.npz",
        canonical_stage_a_path(root, str(cid).rsplit("|", 1)[0]),
    )


def required_cell_outputs_complete(run_dir: Path, cid: str) -> bool:
    return all(path.is_file() for path in required_cell_output_paths(run_dir, cid))


def reference_scale_available(dataset: str) -> bool:
    return str(dataset).upper() == "D6"


def d6_cross_seed_pair_count(n_seeds: int = 5) -> int:
    return int(n_seeds * (n_seeds - 1) / 2)


def parse_formal_ae_uniform_universe(
    path: Path = FORMAL_AE_UNIFORM_3E4_CSV,
) -> Dict[str, Any]:
    """Parse formal seed provenance from the frozen CSV. Do not hardcode seed values."""
    if not Path(path).is_file():
        raise MechanismDiagnosticsContractError(f"formal results CSV missing: {path}")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seeds_by_dataset: Dict[str, set] = {}
    seeds_by_condition_arm: Dict[Tuple[str, str, int, str], set] = {}
    conditions: set = set()
    cell_keys = []
    run_status = set()
    pair_status = set()
    for row in rows:
        dataset = str(row["dataset"])
        target = str(row["target"])
        horizon = int(row["horizon"])
        seed = int(row["seed"])
        arm = str(row["arm"])
        seeds_by_dataset.setdefault(dataset, set()).add(seed)
        seeds_by_condition_arm.setdefault((dataset, target, horizon, arm), set()).add(seed)
        conditions.add((dataset, target, horizon))
        cell_keys.append((dataset, target, horizon, seed, arm))
        run_status.add(str(row.get("run_status") or ""))
        pair_status.add(str(row.get("pair_status") or ""))
    seed_counts = [len(values) for values in seeds_by_condition_arm.values()]
    n_unique = len(set(cell_keys))
    return {
        "n_rows": len(rows),
        "n_unique_cells": n_unique,
        "extra_or_duplicate_rows": len(rows) != n_unique,
        "n_conditions": len(conditions),
        "n_condition_arms": len(seeds_by_condition_arm),
        "seeds_by_dataset": {
            dataset: sorted(int(seed) for seed in seeds)
            for dataset, seeds in sorted(seeds_by_dataset.items())
        },
        "seed_count_per_condition_arm_min": min(seed_counts) if seed_counts else 0,
        "seed_count_per_condition_arm_max": max(seed_counts) if seed_counts else 0,
        "run_status_values": sorted(run_status),
        "pair_status_values": sorted(pair_status),
        "path": str(path),
    }


def assert_formal_universe_schema(universe: Mapping[str, Any]) -> None:
    if universe["extra_or_duplicate_rows"]:
        raise MechanismDiagnosticsContractError("formal CSV has duplicate or extra status rows")
    if int(universe["n_rows"]) != 300 or int(universe["n_unique_cells"]) != 300:
        raise MechanismDiagnosticsContractError(
            f"formal CSV row count drifted: n_rows={universe['n_rows']} unique={universe['n_unique_cells']}"
        )
    if int(universe["n_conditions"]) != 12:
        raise MechanismDiagnosticsContractError(
            f"formal CSV condition count drifted: {universe['n_conditions']}"
        )
    if int(universe["seed_count_per_condition_arm_min"]) != 5:
        raise MechanismDiagnosticsContractError(
            "formal condition-arm unique seed count is not 5: "
            f"{universe['seed_count_per_condition_arm_min']}"
        )
    if int(universe["seed_count_per_condition_arm_max"]) != 5:
        raise MechanismDiagnosticsContractError(
            "formal condition-arm unique seed count is not 5: "
            f"{universe['seed_count_per_condition_arm_max']}"
        )


def diagnostic_selected_seeds(cells: Sequence[Any]) -> Dict[str, List[int]]:
    def _seeds(dataset: str, arms: Sequence[str]) -> List[int]:
        return sorted(
            {
                int(cell.seed)
                for cell in cells
                if cell.dataset == dataset and cell.arm in set(arms)
            }
        )

    return {
        "D1 A/B/C": _seeds("D1", ("A", "B", "C")),
        "D4 A/B/C": _seeds("D4", ("A", "B", "C")),
        "D5 A/B/C": _seeds("D5", ("A", "B", "C")),
        "D6 A": _seeds("D6", ("A",)),
        "D6 B/C": _seeds("D6", ("B", "C")),
    }


def assert_diagnostic_seeds_are_formal_subset(
    universe: Mapping[str, Any],
    selected: Mapping[str, Sequence[int]],
) -> None:
    formal = universe["seeds_by_dataset"]
    mapping = {
        "D1 A/B/C": "D1",
        "D4 A/B/C": "D4",
        "D5 A/B/C": "D5",
        "D6 A": "D6",
        "D6 B/C": "D6",
    }
    for label, dataset in mapping.items():
        chosen = [int(seed) for seed in selected[label]]
        available = [int(seed) for seed in formal[dataset]]
        if not set(chosen).issubset(set(available)):
            raise MechanismDiagnosticsContractError(
                f"{label} diagnostic seeds {chosen} are not a subset of formal {dataset} seeds {available}"
            )
        if dataset in {"D1", "D4", "D5"} and set(chosen) == set(available):
            raise MechanismDiagnosticsContractError(
                f"{label} must remain a deliberate subset of formal {dataset} seeds, not the full universe"
            )


def load_formal_ae_uniform_3e4(
    path: Path = FORMAL_AE_UNIFORM_3E4_CSV,
) -> Dict[Tuple[str, str, int, int, str], Dict[str, float]]:
    if not Path(path).is_file():
        raise MechanismDiagnosticsContractError(f"formal results CSV missing: {path}")
    out: Dict[Tuple[str, str, int, int, str], Dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                str(row["dataset"]),
                str(row["target"]),
                int(row["horizon"]),
                int(row["seed"]),
                str(row["arm"]),
            )
            out[key] = {
                "formal_rmse": float(row["RMSE"]),
                "formal_smape": float(row["sMAPE"]),
            }
    return out


def _relative_diff(abs_diff: float, baseline: float) -> float:
    denom = abs(float(baseline))
    if denom <= IDENTITY_ABS_TOL:
        return float(abs_diff)
    return float(abs_diff) / denom


def identity_match_status(
    *,
    abs_rmse: float,
    abs_smape: float,
    rel_rmse: float,
    rel_smape: float,
    digest_match: bool,
) -> str:
    exact_numbers = float(abs_rmse) == 0.0 and float(abs_smape) == 0.0
    if exact_numbers and digest_match:
        return EXACT_NUMERIC_MATCH
    rmse_tol = float(abs_rmse) <= IDENTITY_ABS_TOL or float(rel_rmse) <= IDENTITY_REL_TOL
    smape_tol = float(abs_smape) <= IDENTITY_ABS_TOL or float(rel_smape) <= IDENTITY_REL_TOL
    if rmse_tol and smape_tol:
        return TOLERANCE_MATCH
    return MISMATCH


def formal_authority_attempt_dir(
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    *,
    root: Path = FORMAL_PER_DATASET_LR_AUTHORITY_ROOT,
) -> Path:
    safe_target = str(target).replace("/", "-")
    return (
        Path(root)
        / "cells"
        / f"{dataset}__{safe_target}__h{int(horizon)}__s{int(seed)}__without"
        / "attempt_001"
    )


def load_formal_per_dataset_lr_authority(
    root: Path = FORMAL_PER_DATASET_LR_AUTHORITY_ROOT,
) -> Dict[Tuple[str, str, int, int, str], Dict[str, Any]]:
    csv_path = Path(root) / "ae_formal_per_dataset_lr_results.csv"
    if not csv_path.is_file():
        raise MechanismDiagnosticsContractError(
            f"formal per-dataset-LR authority CSV missing: {csv_path}"
        )
    out: Dict[Tuple[str, str, int, int, str], Dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row["dataset"]),
                str(row["target"]),
                int(row["horizon"]),
                int(row["seed"]),
                str(row["arm"]),
            )
            lr_raw = row.get("learning_rate") or row.get("learning_rate_label") or ""
            try:
                selected_lr = float(lr_raw)
            except (TypeError, ValueError):
                selected_lr = float("nan")
            out[key] = {
                "formal_rmse": float(row.get("rmse_original") or row["RMSE"]),
                "formal_smape": float(row.get("smape_original") or row["sMAPE"]),
                "formal_selected_lr": selected_lr,
                "formal_selected_lr_label": str(row.get("learning_rate_label") or lr_raw),
            }
    for (dataset, target, horizon, seed, arm), payload in list(out.items()):
        attempt = formal_authority_attempt_dir(dataset, target, horizon, seed, root=root)
        result_path = attempt / "result.json"
        if not result_path.is_file():
            continue
        blob = json.loads(result_path.read_text(encoding="utf-8"))
        arm_payload = (blob.get("arms") or {}).get(arm, {})
        if not isinstance(arm_payload, Mapping):
            continue
        payload.update(
            {
                "prediction_digest": arm_payload.get("prediction_digest"),
                "final_formal_model_digest": arm_payload.get("final_formal_model_digest"),
                "stage_a_digest": arm_payload.get("stage_a_digest"),
                "fused_init_digest": arm_payload.get("fused_init_digest"),
                "selected_lr": arm_payload.get("selected_lr", payload.get("formal_selected_lr")),
            }
        )
        if arm_payload.get("selected_lr") not in (None, ""):
            payload["formal_selected_lr"] = float(arm_payload["selected_lr"])
    return out


def audit_formal_per_dataset_lr_authority(
    root: Path = FORMAL_PER_DATASET_LR_AUTHORITY_ROOT,
) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
        FORMAL_PER_DATASET_LR_LABELS,
        FORMAL_PER_DATASET_LR_TARGETS,
        FORMAL_PER_DATASET_LR_VALUES,
    )

    snapshot_path = Path(root) / "manifest_snapshot.json"
    if not snapshot_path.is_file():
        raise MechanismDiagnosticsContractError(
            f"formal authority manifest_snapshot.json missing: {snapshot_path}"
        )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    live_labels = dict(FORMAL_PER_DATASET_LR_LABELS)
    snap_labels = dict(snapshot.get("target_learning_rates") or {})
    if snap_labels != live_labels:
        raise MechanismDiagnosticsContractError(
            f"formal authority LR map drifted: {snap_labels} != {live_labels}"
        )
    if dict(snapshot.get("paper_targets") or snapshot.get("dataset_target_map") or {}) != dict(
        FORMAL_PER_DATASET_LR_TARGETS
    ):
        raise MechanismDiagnosticsContractError("formal authority paper targets drifted")
    if str(snapshot.get("matrix_kind")) != "FORMAL_PER_DATASET_LR_AE":
        raise MechanismDiagnosticsContractError(
            f"formal authority matrix_kind drifted: {snapshot.get('matrix_kind')!r}"
        )
    table = load_formal_per_dataset_lr_authority(root)
    by_dataset: Dict[str, set] = {}
    for (dataset, _target, _horizon, _seed, _arm), payload in table.items():
        by_dataset.setdefault(dataset, set()).add(float(payload["formal_selected_lr"]))
    for dataset, expected in FORMAL_PER_DATASET_LR_VALUES.items():
        observed = by_dataset.get(dataset, set())
        if observed != {float(expected)}:
            raise MechanismDiagnosticsContractError(
                f"formal authority selected_lr for {dataset} drifted: {observed} != {{{expected}}}"
            )
    return {
        "root": str(Path(root).resolve()),
        "csv": str((Path(root) / "ae_formal_per_dataset_lr_results.csv").resolve()),
        "matrix_kind": snapshot.get("matrix_kind"),
        "target_learning_rates": live_labels,
        "n_rows": len(table),
        "source_lr": snapshot.get("source_lr"),
        "ema_decay": 0.99,
    }


def resolve_run_directory(
    *,
    plan_only: bool,
    resume: bool,
    run_dir: Optional[str],
    now: Optional[datetime] = None,
    matrix_mode: str = "diagnostic-42",
) -> Optional[Path]:
    if plan_only:
        return None
    if resume:
        if not run_dir:
            raise MechanismDiagnosticsContractError("--resume requires explicit --run-dir")
        path = Path(run_dir)
        if not path.is_dir():
            raise MechanismDiagnosticsContractError(f"--run-dir does not exist: {path}")
        return path
    if run_dir:
        raise MechanismDiagnosticsContractError("new --run must not reuse --run-dir; resume requires both flags")
    return create_run_root(now=now, matrix_mode=matrix_mode)


def prepare_run_root(
    run_dir: Path,
    *,
    config_digest: str,
    matrix_digest: str,
    resume: bool,
    code_digest: Optional[str] = None,
    matrix_mode: str = "diagnostic-42",
) -> Path:
    run_dir = Path(run_dir)
    manifest = run_dir / "config_digest.json"
    payload = {
        "config_digest": config_digest,
        "matrix_digest": matrix_digest,
        "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
        "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
        "sita_ablation_3e4_manifest_runtime_authority": "NO",
        "matrix_mode": str(matrix_mode),
    }
    if code_digest is not None:
        payload["code_digest"] = code_digest
    if resume:
        stored = load_run_digests(run_dir)
        if stored.get("config_digest") != config_digest:
            raise MechanismDiagnosticsContractError("config digest mismatch")
        if stored.get("matrix_digest") != matrix_digest:
            raise MechanismDiagnosticsContractError("matrix digest mismatch")
        if code_digest is not None and stored.get("code_digest") not in {None, ""}:
            if stored.get("code_digest") != code_digest:
                raise MechanismDiagnosticsContractError("code digest mismatch")
        return run_dir
    if manifest.exists():
        raise MechanismDiagnosticsContractError(f"run root already initialized: {run_dir}")
    write_json_atomic(manifest, payload)
    write_json_atomic(run_dir / "run_manifest.json", dict(payload))
    return run_dir


def latest_run_dir_forbidden() -> str:
    return "auto-reuse of the latest run directory is forbidden"


def epoch0_precedes_first_optimizer_update(lite_source: str) -> bool:
    marker = '"epoch": 0'
    apply_marker = "optimizer.apply_gradients(finite_grads)"
    idx_epoch = lite_source.find(marker)
    idx_apply = lite_source.find(apply_marker)
    return 0 <= idx_epoch < idx_apply


def diagnostics_disabled_keeps_formal_train_kwargs(kwargs: Mapping[str, Any]) -> bool:
    forbidden = {
        "on_before_first_target_optimizer_update",
        "on_diagnostic_epoch_eval",
        "on_before_ema_assign",
    }
    return not any(key in kwargs for key in forbidden)
