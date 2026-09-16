"""Orchestration outputs for the 42-cell SITA mechanism diagnostic rerun.

Numpy-first helpers: formal-test CKA references, training-dynamics rows,
pointwise predictions, sales-bin summaries, CKA comparison matrix,
layer_role drift schema, formal-result consistency audit, and resume.
Does not train models and does not rewrite frozen A–E numbers.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import smape
from src.transfer_methods.sita_mechanism_diagnostics import linear_cka, reference_input_digest
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_A_C_MATCHED_SEED,
    COMPARISON_B_B_CROSS_SEED,
    COMPARISON_B_C_MATCHED_SEED,
    COMPARISON_C_C_CROSS_SEED,
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
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
OUTPUT_ROOT_PARENT = Path("outputs") / "mechanism_diagnostics"
CELL_STATUS_PENDING = "PENDING"
CELL_STATUS_RUNNING = "RUNNING"
CELL_STATUS_COMPLETE = "COMPLETE"
CELL_STATUS_FAILED = "FAILED"


def condition_id(dataset: str, target: str, horizon: int) -> str:
    return f"{dataset}|{target}|h{int(horizon)}"


def cell_id(dataset: str, target: str, horizon: int, seed: int, arm: str) -> str:
    return f"{dataset}|{target}|h{int(horizon)}|s{int(seed)}|{arm}"


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
) -> List[Dict[str, Any]]:
    true = np.asarray(y_true_original, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred_original, dtype=np.float64).reshape(-1)
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
                "date_or_prediction_index": dates[i] if dates is not None else i,
                "y_true": float(true[i]),
                "y_pred": float(pred[i]),
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
    }


def timestamped_run_root(now: Optional[datetime] = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT_PARENT / f"mechanism_rerun_{stamp}"


def create_run_root(base: Optional[Path] = None, now: Optional[datetime] = None) -> Path:
    named = timestamped_run_root(now)
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
) -> List[Dict[str, Any]]:
    from src.transfer_methods.sita_mechanism_diagnostics import (
        parameter_l2_distance,
        relative_parameter_drift,
    )

    names = sorted(set(named_init) | set(named_final))
    rows: List[Dict[str, Any]] = []
    for name in names:
        role = classify_layer_role(name)
        init_w = list(named_init.get(name, ()))
        final_w = list(named_final.get(name, ()))
        if init_w and final_w:
            l2 = parameter_l2_distance(init_w, final_w)
            rel = relative_parameter_drift(init_w, final_w)
        else:
            l2 = rel = float("nan")
        row = {
            "layer": name,
            "layer_role": role,
            "parameter_l2": l2,
            "relative_parameter_drift": rel,
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
            "date_or_prediction_index",
            "y_true",
            "y_pred",
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
            "weight_variant",
        ],
    )
    return path


def write_error_by_sales_bin(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "error_by_sales_bin.csv"
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


def write_consistency_audit(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / "summaries" / "formal_result_consistency.csv"
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
        ],
    )
    return path


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
) -> Path:
    path = Path(run_dir) / "representations" / f"{filesystem_safe_id(cid)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        INIT=np.asarray(init_rep),
        FINAL=np.asarray(final_rep),
        reference_input_digest=np.asarray(reference_input_digest_value),
    )
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
        root / "status" / f"{safe}.json",
        root / "status" / f"{safe}.COMPLETE",
        root / "predictions" / f"{safe}.csv",
        root / "training_dynamics" / f"{safe}.csv",
        root / "checkpoints" / f"{safe}.json",
        root / "representations" / f"{safe}.npz",
        root / "parameter_drift" / f"{safe}.csv",
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


def resolve_run_directory(
    *,
    plan_only: bool,
    resume: bool,
    run_dir: Optional[str],
    now: Optional[datetime] = None,
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
    return create_run_root(now=now)


def prepare_run_root(
    run_dir: Path,
    *,
    config_digest: str,
    matrix_digest: str,
    resume: bool,
    code_digest: Optional[str] = None,
) -> Path:
    run_dir = Path(run_dir)
    manifest = run_dir / "config_digest.json"
    payload = {
        "config_digest": config_digest,
        "matrix_digest": matrix_digest,
        "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
        "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
        "sita_ablation_3e4_manifest_runtime_authority": "NO",
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
    }
    return not any(key in kwargs for key in forbidden)
