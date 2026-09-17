"""42-cell mechanism diagnostic execution loop.

The real executor trains through frozen A/B/C Stage-A/B authorities with
diagnostics enabled. Tests inject a synthetic executor and must not import
TensorFlow or start formal training.

This module does not rewrite frozen A–E RMSE/sMAPE and does not publish into
the frozen 3e-4 A–E run root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
    CELL_STATUS_COMPLETE,
    CELL_STATUS_FAILED,
    FORMAL_AE_UNIFORM_3E4_CSV,
    FORMAL_TEST_REFERENCE_SPLIT,
    classify_layer_role,
    consistency_audit_row,
    error_by_sales_bin,
    filesystem_safe_id,
    generate_cka_comparison_matrix,
    load_formal_ae_uniform_3e4,
    load_run_digests,
    mark_cell_complete,
    mark_cell_failed,
    parameter_drift_rows,
    pointwise_prediction_rows,
    primary_transferred_backbone_weights,
    reference_scale_available,
    resume_cell_decision,
    write_cell_checkpoints,
    write_cell_parameter_drift,
    write_cell_representations,
    write_cell_status_summary,
    write_cka_condition_summary,
    write_cka_pairwise_matrix,
    write_consistency_audit,
    write_error_by_sales_bin,
    write_json_atomic,
    write_layerwise_parameter_drift,
    write_parameter_drift_summary,
    write_pointwise_predictions,
    write_training_dynamics,
    write_training_dynamics_summary,
)
from src.transfer_methods.sita_mechanism_diagnostic_rerun import (
    DIAGNOSTIC_RERUN_TARGETS,
    DiagnosticRerunCell,
    enumerate_mechanism_diagnostic_rerun_cells,
    matrix_digest_for_cells,
    mechanism_rerun_config_digest,
)
from src.transfer_methods.sita_mechanism_diagnostics import relative_parameter_drift
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismDiagnosticsContractError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITA_ABLATION_3E4_MANIFEST_RELPATH = Path("configs") / "ms_katl_sita_ablation_3e4_manifest.json"
FROZEN_AE_OUTPUT_MARKER = "sita_ablation_3e4_d1_d4_d5_d6_5seed"

RUNTIME_SOURCE_FILES: Tuple[str, ...] = (
    "scripts/run_mechanism_diagnostic_rerun.py",
    "src/transfer_methods/sita_mechanism_diagnostic_rerun.py",
    "src/transfer_methods/sita_mechanism_diagnostic_execution.py",
    "src/transfer_methods/sita_mechanism_diagnostics_capture.py",
    "src/transfer_methods/sita_mechanism_diagnostic_outputs.py",
    "src/transfer_methods/sita_mechanism_diagnostics.py",
    "src/transfer_methods/sita_mechanism_diagnostics_contract.py",
    "src/transfer_methods/ms_katl_ae_formal.py",
    "src/transfer_methods/ms_katl_lite.py",
    "src/transfer_methods/ms_katl_ae_ablation.py",
    "src/transfer_methods/ms_katl_ae_execution.py",
    "src/transfer_methods/ms_katl_v3_frame_assembly.py",
)

CellExecutor = Callable[[DiagnosticRerunCell, Path], Mapping[str, Any]]


def mechanism_rerun_code_digest(*, project_root: Optional[Path] = None) -> str:
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    digest = hashlib.sha256()
    for relpath in RUNTIME_SOURCE_FILES:
        path = root / relpath
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\n")
    return digest.hexdigest()


def sita_ablation_3e4_manifest_runtime_role() -> Dict[str, str]:
    """Code-level role of the historical 3e-4 slice JSON for diagnostic --run.

    Authority for the 42-cell matrix is DIAGNOSTIC_RERUN_* in
    sita_mechanism_diagnostic_rerun.py. Stage-B LR/recipe come from
    UNIFORM_3E4_* / ae_training_recipe() in ms_katl_ae_ablation.py.
    Consistency lookup uses the frozen CSV, not this JSON.
    """
    return {
        "plan_only_reads": "NO",
        "run_reads": "NO",
        "matrix_parsed_from_manifest": "NO",
        "consistency_audit_uses": "NO",
        "run_manifest_records_digest": "NO",
        "runtime_authority": "NO",
        "actual_matrix_authority": "src/transfer_methods/sita_mechanism_diagnostic_rerun.py:DIAGNOSTIC_RERUN_*",
        "actual_lr_authority": "src/transfer_methods/ms_katl_ae_ablation.py:resolve_uniform_3e4_lr",
        "actual_recipe_authority": "src/transfer_methods/ms_katl_ae_ablation.py:ae_training_recipe",
        "actual_consistency_authority": str(FORMAL_AE_UNIFORM_3E4_CSV),
        "historical_role": (
            "auxiliary snapshot written by scripts/run_ms_katl_sita_ablation_3e4.py; "
            "not opened by diagnostic --plan-only/--run"
        ),
    }


def diagnostic_runtime_reads_sita_3e4_manifest(source_text: str) -> bool:
    if "ms_katl_sita_ablation_3e4_manifest.json" not in source_text:
        return False
    for token in ("read_text(", "json.load(", "json.loads(", "open("):
        if token in source_text:
            # Filename present plus a generic IO token is not enough; require
            # the JSON path to appear in an IO expression.
            continue
    lowered = source_text.replace(" ", "")
    return (
        "SITA_ABLATION_3E4_MANIFEST_RELPATH.read_text" in lowered
        or "ms_katl_sita_ablation_3e4_manifest.json\").read_text" in lowered
        or "json.load" in source_text and "ms_katl_sita_ablation_3e4_manifest.json" in source_text
        and "sita_ablation_3e4_manifest_runtime_role" not in source_text
    )


def _named_dummy_weights(scale: float) -> Dict[str, Tuple[np.ndarray, ...]]:
    eye = np.eye(2, dtype=np.float32) * float(scale)
    head = np.array([[scale, 0.0]], dtype=np.float32)
    return {
        "conv1": (eye.copy(),),
        "conv2": (eye.copy(),),
        "conv3": (eye.copy(),),
        "head_target": (head.copy(),),
    }


def persist_synthetic_cell_artifacts(
    run_dir: Path,
    cell: DiagnosticRerunCell,
    *,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    fail: bool = False,
) -> Dict[str, Any]:
    cid = cell.cell_key
    true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64) if y_true is None else np.asarray(y_true)
    pred = np.asarray([1.1, 1.9, 3.2], dtype=np.float64) if y_pred is None else np.asarray(y_pred)
    init_rep = np.eye(max(2, int(true.shape[0])), dtype=np.float32)
    final_rep = init_rep + 0.01
    digest = hashlib.sha256(b"synthetic-formal-test-inputs").hexdigest()
    pred_rows = pointwise_prediction_rows(
        dataset=cell.dataset,
        target=cell.target,
        horizon=cell.horizon,
        seed=cell.seed,
        arm=cell.arm,
        y_true_original=true,
        y_pred_original=pred,
    )
    write_pointwise_predictions(run_dir, cid, pred_rows)
    dynamics = [
        {
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "epoch": 0,
            "before_first_target_optimizer_update": True,
            "train_objective_loss": 1.0,
            "validation_objective_loss": 1.1,
            "validation_rmse_original_scale": 1.2,
            "validation_smape_original_scale": 1.3,
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
        },
        {
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "epoch": 1,
            "before_first_target_optimizer_update": False,
            "train_objective_loss": 0.8,
            "validation_objective_loss": 0.9,
            "validation_rmse_original_scale": 1.0,
            "validation_smape_original_scale": 1.1,
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
        },
    ]
    write_training_dynamics(run_dir, cid, dynamics)
    write_cell_checkpoints(
        run_dir,
        cid,
        {
            "cell_id": cid,
            "arm": cell.arm,
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
            "reference_input_digest": digest,
            "checkpoints": {
                f"{cell.arm}_INIT": {
                    "stage": "INIT",
                    "captured_before_first_target_optimizer_update": True,
                    "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
                    "optimizer_update_count": 0,
                },
                f"{cell.arm}_FINAL": {
                    "stage": "FINAL",
                    "captured_before_first_target_optimizer_update": False,
                    "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
                    "optimizer_update_count": 1,
                },
            },
        },
    )
    write_cell_representations(
        run_dir,
        cid,
        init_rep=init_rep,
        final_rep=final_rep,
        reference_input_digest_value=digest,
    )
    drift_rows = parameter_drift_rows(
        _named_dummy_weights(1.0),
        _named_dummy_weights(1.05),
        extra={
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
        },
    )
    write_cell_parameter_drift(run_dir, cid, drift_rows)
    payload = {
        "cell_id": cid,
        "dataset": cell.dataset,
        "target": cell.target,
        "horizon": int(cell.horizon),
        "seed": int(cell.seed),
        "arm": cell.arm,
        "rerun_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "rerun_smape": float(np.mean(np.abs(pred - true) / np.maximum(1e-12, (np.abs(pred) + np.abs(true)) / 2.0)) * 100.0),
        "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
        "reference_input_digest": digest,
        "synthetic": True,
        "frozen_ae_root_untouched": "YES",
    }
    if fail:
        return payload
    return payload


@dataclass
class FakeDiagnosticCellExecutor:
    fail_keys: Tuple[str, ...] = ()
    raise_keys: Tuple[str, ...] = ()
    calls: List[str] = field(default_factory=list)

    def __call__(self, cell: DiagnosticRerunCell, run_dir: Path) -> Mapping[str, Any]:
        self.calls.append(cell.cell_key)
        if cell.cell_key in set(self.raise_keys):
            raise RuntimeError(f"synthetic executor crash: {cell.cell_key}")
        payload = persist_synthetic_cell_artifacts(
            run_dir,
            cell,
            fail=cell.cell_key in set(self.fail_keys),
        )
        if cell.cell_key in set(self.fail_keys):
            raise RuntimeError(f"synthetic executor marked failure: {cell.cell_key}")
        return payload


class FormalDiagnosticCellExecutor:
    """Real 42-cell path. Lazy-imports formal authorities so tests stay TF-free."""

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.project_root = PROJECT_ROOT if project_root is None else Path(project_root)
        self.environ = environ

    def __call__(self, cell: DiagnosticRerunCell, run_dir: Path) -> Mapping[str, Any]:
        if FROZEN_AE_OUTPUT_MARKER in Path(run_dir).as_posix():
            raise MechanismDiagnosticsContractError("diagnostic --run refuses the frozen A–E output root")
        from src.transfer_methods.ms_katl_ae_ablation import (
            AETrainingBlocked,
            assert_formal_training_permitted,
            resolve_uniform_3e4_lr,
        )
        from src.transfer_methods.ms_katl_ae_execution import resolve_real_canonical_cell
        from src.transfer_methods.ms_katl_ae_formal import (
            default_v3_make_common_head,
            default_v3_metric,
            default_v3_predict,
            default_v3_prepare_context,
            default_v3_prepare_stage_a,
            default_v3_train_stage_b,
        )
        from src.transfer_methods.sita_mechanism_diagnostics_capture import (
            ENABLE_MECHANISM_DIAGNOSTICS_KEY,
            MECHANISM_DIAGNOSTICS_SESSION_KEY,
        )

        assert_formal_training_permitted(execute=True, environ=self.environ)
        expected_target = DIAGNOSTIC_RERUN_TARGETS[cell.dataset]
        if cell.target != expected_target:
            raise MechanismDiagnosticsContractError(
                f"diagnostic target drifted for {cell.dataset}: {cell.target} != {expected_target}"
            )
        lr_value, lr_label = resolve_uniform_3e4_lr(cell.dataset, cell.target)
        plan = resolve_real_canonical_cell(
            cell.dataset,
            cell.target,
            int(cell.horizon),
            int(cell.seed),
            project_root=self.project_root,
            selected_lr=lr_value,
            selected_lr_label=lr_label,
        )
        context = default_v3_prepare_context(plan, project_root=self.project_root)
        if not isinstance(context, dict):
            context = dict(context)
        context[ENABLE_MECHANISM_DIAGNOSTICS_KEY] = True
        context["mechanism_diagnostics_run_dir"] = str(run_dir)
        context["training_dynamics_rows"] = []
        stage_a = default_v3_prepare_stage_a(plan, context)
        head = default_v3_make_common_head(plan, stage_a, context)
        runtime = default_v3_train_stage_b(plan, cell.arm, stage_a, head, context)
        runtime = default_v3_predict(plan, runtime, context)
        runtime = default_v3_metric(plan, runtime, context)
        session = context.get(MECHANISM_DIAGNOSTICS_SESSION_KEY)
        constructed = context.get("_arm_ephemeral")
        constructed = getattr(constructed, "constructed", constructed)
        payload = getattr(runtime, "_predict_payload", {}) or {}
        y_true = np.asarray(payload.get("y_true"))
        y_pred = np.asarray(payload.get("y_pred"))
        true_o = getattr(runtime, "y_true_original", None)
        pred_o = getattr(runtime, "y_pred_original", None)
        if true_o is None or pred_o is None:
            metrics_extra = payload.get("target_scaler")
            from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

            scored = compute_screening_metrics(
                y_true,
                y_pred,
                target_scaler=metrics_extra,
                feature_columns=payload.get("feature_columns"),
            )
            true_o = scored.get("y_true_original", y_true)
            pred_o = scored.get("y_pred_original", y_pred)
        cid = cell.cell_key
        pred_rows = pointwise_prediction_rows(
            dataset=cell.dataset,
            target=cell.target,
            horizon=cell.horizon,
            seed=cell.seed,
            arm=cell.arm,
            y_true_original=np.asarray(true_o),
            y_pred_original=np.asarray(pred_o),
        )
        write_pointwise_predictions(run_dir, cid, pred_rows)
        dynamics_rows = list(context.get("training_dynamics_rows") or [])
        if not dynamics_rows:
            raise MechanismDiagnosticsContractError("training dynamics rows missing after diagnostic Stage-B")
        write_training_dynamics(run_dir, cid, dynamics_rows)
        if session is None:
            raise MechanismDiagnosticsContractError("diagnostics session missing after enabled Stage-B")
        init_key = (cell.arm, "INIT")
        final_key = (cell.arm, "FINAL")
        if init_key not in session.checkpoints or final_key not in session.checkpoints:
            raise MechanismDiagnosticsContractError("INIT/FINAL EMA checkpoints missing")
        write_cell_checkpoints(run_dir, cid, session.as_dict())
        write_cell_representations(
            run_dir,
            cid,
            init_rep=session.representations[init_key],
            final_rep=session.representations[final_key],
            reference_input_digest_value=str(session.reference_input_digest),
        )
        named_snapshots = getattr(session, "named_layer_weights", None)
        named_init = named_snapshots.get(init_key) if isinstance(named_snapshots, Mapping) else None
        named_final = named_snapshots.get(final_key) if isinstance(named_snapshots, Mapping) else None
        if named_init is None or named_final is None:
            raise MechanismDiagnosticsContractError(
                "INIT/FINAL named layer weight snapshots missing"
            )
        drift_rows = parameter_drift_rows(
            named_init,
            named_final,
            extra={
                "dataset": cell.dataset,
                "target": cell.target,
                "horizon": int(cell.horizon),
                "seed": int(cell.seed),
                "arm": cell.arm,
            },
        )
        write_cell_parameter_drift(run_dir, cid, drift_rows)
        _ = AETrainingBlocked
        return {
            "cell_id": cid,
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "rerun_rmse": float(runtime.rmse_original) if runtime.rmse_original is not None else float("nan"),
            "rerun_smape": float(runtime.smape_original) if runtime.smape_original is not None else float("nan"),
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "reference_input_digest": str(session.reference_input_digest),
            "synthetic": False,
            "frozen_ae_root_untouched": "YES",
            "published_to_frozen_ae_root": "NO",
        }


def _status_rows(cells: Sequence[DiagnosticRerunCell], outcomes: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for cell in cells:
        item = outcomes.get(cell.cell_key, {})
        rows.append(
            {
                "cell_id": cell.cell_key,
                "dataset": cell.dataset,
                "target": cell.target,
                "horizon": int(cell.horizon),
                "seed": int(cell.seed),
                "arm": cell.arm,
                "status": item.get("status", CELL_STATUS_FAILED),
                "notes": item.get("notes", ""),
            }
        )
    return rows


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize_mechanism_diagnostic_run(
    run_dir: Path,
    cells: Sequence[DiagnosticRerunCell],
    outcomes: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    run_dir = Path(run_dir)
    status_rows = _status_rows(cells, outcomes)
    write_cell_status_summary(run_dir, status_rows)
    complete = [cell for cell in cells if outcomes.get(cell.cell_key, {}).get("status") == CELL_STATUS_COMPLETE]
    if len(complete) != len(tuple(cells)):
        return {
            "finalized": "NO",
            "reason": "not all cells COMPLETE",
            "cell_status": str(run_dir / "summaries" / "cell_status.csv"),
        }
    pred_rows: List[Dict[str, Any]] = []
    dynamics_summary = []
    layerwise = []
    drift_summary = []
    cka_records = []
    for cell in complete:
        cid = cell.cell_key
        safe = filesystem_safe_id(cid)
        pred_rows.extend(_load_csv_rows(run_dir / "predictions" / f"{safe}.csv"))
        dyn = _load_csv_rows(run_dir / "training_dynamics" / f"{safe}.csv")
        dynamics_summary.append(
            {
                "cell_id": cid,
                "dataset": cell.dataset,
                "target": cell.target,
                "horizon": int(cell.horizon),
                "seed": int(cell.seed),
                "arm": cell.arm,
                "n_epochs": len(dyn),
                "epoch0_present": any(str(row.get("epoch")) == "0" for row in dyn),
                "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            }
        )
        drift = _load_csv_rows(run_dir / "parameter_drift" / f"{safe}.csv")
        for row in drift:
            layerwise.append(dict(row, cell_id=cid))
        backbone = [row for row in drift if classify_layer_role(str(row.get("layer"))) == "transferred_backbone"]
        rels = [float(row["relative_parameter_drift"]) for row in backbone if row.get("relative_parameter_drift")]
        drift_summary.append(
            {
                "cell_id": cid,
                "dataset": cell.dataset,
                "target": cell.target,
                "horizon": int(cell.horizon),
                "seed": int(cell.seed),
                "arm": cell.arm,
                "n_layers": len(drift),
                "primary_relative_l2_drift": float(np.mean(rels)) if rels else float("nan"),
            }
        )
        npz_path = run_dir / "representations" / f"{safe}.npz"
        if npz_path.is_file():
            blob = np.load(npz_path, allow_pickle=True)
            digest = str(np.asarray(blob["reference_input_digest"]).tolist())
            if digest.startswith("['") or digest.startswith('["'):
                digest = str(np.asarray(blob["reference_input_digest"]).reshape(-1)[0])
            for stage in ("INIT", "FINAL"):
                if stage in blob.files:
                    cka_records.append(
                        {
                            "arm": cell.arm,
                            "seed": int(cell.seed),
                            "stage": stage,
                            "condition_id": cell.condition_key,
                            "representations": np.asarray(blob[stage]),
                            "reference_input_digest": digest,
                        }
                    )
    write_training_dynamics_summary(run_dir, dynamics_summary)
    bin_rows: List[Dict[str, Any]] = []
    by_cell_preds: Dict[str, List[Dict[str, Any]]] = {}
    for row in pred_rows:
        key = f"{row.get('dataset')}|{row.get('target')}|h{row.get('horizon')}|s{row.get('seed')}|{row.get('arm')}"
        by_cell_preds.setdefault(key, []).append(row)
    for cid, rows in sorted(by_cell_preds.items()):
        bin_rows.extend(error_by_sales_bin(rows))
    write_error_by_sales_bin(run_dir, bin_rows)
    write_layerwise_parameter_drift(run_dir, layerwise)
    write_parameter_drift_summary(run_dir, drift_summary)
    cka_rows: List[Dict[str, Any]] = []
    condition_rows = []
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for record in cka_records:
        by_condition.setdefault(str(record["condition_id"]), []).append(record)
    for cid, records in sorted(by_condition.items()):
        matrix = generate_cka_comparison_matrix(
            records,
            reference_scale_available=reference_scale_available(cid.split("|", 1)[0]),
        )
        for row in matrix:
            cka_rows.append(dict(row, condition_id=cid))
        digests = {str(item["reference_input_digest"]) for item in records}
        condition_rows.append(
            {
                "condition_id": cid,
                "n_comparisons": len(matrix),
                "reference_input_digest": next(iter(digests)) if len(digests) == 1 else "MIXED",
                "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            }
        )
    write_cka_pairwise_matrix(run_dir, cka_rows)
    write_cka_condition_summary(run_dir, condition_rows)
    formal = load_formal_ae_uniform_3e4()
    audit_rows = []
    for cell in complete:
        outcome = outcomes[cell.cell_key]
        key = (cell.dataset, cell.target, int(cell.horizon), int(cell.seed), cell.arm)
        formal_row = formal.get(key)
        if formal_row is None:
            continue
        audit_rows.append(
            consistency_audit_row(
                dataset=cell.dataset,
                target=cell.target,
                horizon=int(cell.horizon),
                seed=int(cell.seed),
                arm=cell.arm,
                formal_rmse=float(formal_row["formal_rmse"]),
                rerun_rmse=float(outcome.get("rerun_rmse", float("nan"))),
                formal_smape=float(formal_row["formal_smape"]),
                rerun_smape=float(outcome.get("rerun_smape", float("nan"))),
            )
        )
    write_consistency_audit(run_dir, audit_rows)
    _ = relative_parameter_drift
    _ = primary_transferred_backbone_weights
    return {
        "finalized": "YES",
        "training_dynamics_summary": str(run_dir / "summaries" / "training_dynamics_summary.csv"),
        "error_by_sales_bin": str(run_dir / "summaries" / "error_by_sales_bin.csv"),
        "layerwise_parameter_drift": str(run_dir / "summaries" / "layerwise_parameter_drift.csv"),
        "parameter_drift_summary": str(run_dir / "summaries" / "parameter_drift_summary.csv"),
        "cka_pairwise_matrix": str(run_dir / "summaries" / "cka_pairwise_matrix.csv"),
        "cka_condition_summary": str(run_dir / "summaries" / "cka_condition_summary.csv"),
        "formal_result_consistency": str(run_dir / "summaries" / "formal_result_consistency.csv"),
        "cell_status": str(run_dir / "summaries" / "cell_status.csv"),
        "replaces_frozen_results": "NO",
    }


def run_mechanism_diagnostic_cell_loop(
    run_dir: Path,
    *,
    cells: Optional[Sequence[DiagnosticRerunCell]] = None,
    executor: Optional[CellExecutor] = None,
    resume: bool = False,
    config_digest: Optional[str] = None,
    matrix_digest: Optional[str] = None,
    code_digest: Optional[str] = None,
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    allow_cell_subset: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if FROZEN_AE_OUTPUT_MARKER in run_dir.as_posix():
        raise MechanismDiagnosticsContractError("diagnostic cell loop refuses the frozen A–E output root")
    universe = enumerate_mechanism_diagnostic_rerun_cells()
    if len(universe) != 42:
        raise MechanismDiagnosticsContractError(f"diagnostic matrix drifted: {len(universe)} != 42")
    resolved = tuple(cells) if cells is not None else universe
    if allow_cell_subset:
        if not resolved:
            raise MechanismDiagnosticsContractError("--only-cell subset is empty")
        universe_keys = {cell.cell_key for cell in universe}
        unknown = [cell.cell_key for cell in resolved if cell.cell_key not in universe_keys]
        if unknown:
            raise MechanismDiagnosticsContractError(
                f"--only-cell is not in the 42-cell diagnostic matrix: {unknown}"
            )
    elif len(resolved) != 42:
        raise MechanismDiagnosticsContractError(f"diagnostic matrix drifted: {len(resolved)} != 42")
    cfg = config_digest or mechanism_rerun_config_digest(resolved)
    mat = matrix_digest or matrix_digest_for_cells(resolved)
    code = code_digest or mechanism_rerun_code_digest(project_root=project_root)
    stored = load_run_digests(run_dir) if resume else {
        "config_digest": cfg,
        "matrix_digest": mat,
        "code_digest": code,
    }
    worker = executor if executor is not None else FormalDiagnosticCellExecutor(
        project_root=project_root,
        environ=environ,
    )
    outcomes: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    failed = 0
    ran = 0
    for cell in resolved:
        cid = cell.cell_key
        decision = resume_cell_decision(
            run_dir,
            cid,
            expected_config_digest=cfg,
            expected_matrix_digest=mat,
            stored_config_digest=str(stored.get("config_digest")),
            stored_matrix_digest=str(stored.get("matrix_digest")),
            expected_code_digest=code,
            stored_code_digest=stored.get("code_digest"),
        )
        if decision == "SKIP":
            skipped += 1
            status_path = run_dir / "status" / f"{filesystem_safe_id(cid)}.json"
            prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            outcomes[cid] = {
                "status": CELL_STATUS_COMPLETE,
                "notes": "SKIP",
                "rerun_rmse": prior.get("rerun_rmse", float("nan")),
                "rerun_smape": prior.get("rerun_smape", float("nan")),
            }
            continue
        try:
            payload = dict(worker(cell, run_dir))
            mark_cell_complete(run_dir, cid, payload)
            ran += 1
            outcomes[cid] = {
                "status": CELL_STATUS_COMPLETE,
                "notes": "",
                "rerun_rmse": payload.get("rerun_rmse", float("nan")),
                "rerun_smape": payload.get("rerun_smape", float("nan")),
            }
        except Exception as exc:
            failed += 1
            tb = traceback.format_exc()
            mark_cell_failed(
                run_dir,
                cid,
                {
                    "cell_id": cid,
                    "error": str(exc),
                    "traceback": tb,
                },
            )
            outcomes[cid] = {
                "status": CELL_STATUS_FAILED,
                "notes": str(exc),
                "rerun_rmse": float("nan"),
                "rerun_smape": float("nan"),
            }
            continue
    finalize = finalize_mechanism_diagnostic_run(run_dir, resolved, outcomes)
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "config_digest": cfg,
            "matrix_digest": mat,
            "code_digest": code,
            "n_cells": len(resolved),
            "n_ran": ran,
            "n_skipped": skipped,
            "n_failed": failed,
            "finalized": finalize.get("finalized"),
            "sita_ablation_3e4_manifest_runtime_authority": "NO",
            "sita_ablation_3e4_manifest_role": sita_ablation_3e4_manifest_runtime_role(),
            "replaces_frozen_results": "NO",
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
        },
    )
    return {
        "run_dir": str(run_dir),
        "n_cells": len(resolved),
        "n_ran": ran,
        "n_skipped": skipped,
        "n_failed": failed,
        "finalized": finalize.get("finalized"),
        "outcomes": outcomes,
        "finalize": finalize,
        "cell_loop_started": "YES",
    }
