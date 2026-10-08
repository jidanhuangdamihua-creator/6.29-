"""SITA mechanism diagnostic execution loop.

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
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
    CELL_STATUS_COMPLETE,
    CELL_STATUS_FAILED,
    CKA_REPRESENTATION_LAYERS,
    CKA_ACTIVATION_RESHAPE_RULE,
    EXACT_NUMERIC_MATCH,
    FORMAL_AE_UNIFORM_3E4_CSV,
    FORMAL_PER_DATASET_LR_AUTHORITY_ROOT,
    FORMAL_PER_DATASET_LR_CSV,
    FORMAL_TEST_REFERENCE_SPLIT,
    MISMATCH,
    TOLERANCE_MATCH,
    classify_layer_role,
    formal_reference_result_digest,
    error_by_sales_bin,
    assert_formal_universe_schema,
    filesystem_safe_id,
    generate_cka_comparison_matrix,
    identity_match_status,
    load_formal_ae_uniform_3e4,
    load_formal_per_dataset_lr_authority,
    parse_formal_ae_uniform_universe,
    load_run_digests,
    mark_cell_complete,
    mark_cell_failed,
    parameter_drift_rows,
    pointwise_prediction_rows,
    primary_transferred_backbone_weights,
    reference_scale_available,
    required_cell_outputs_complete,
    resume_cell_decision,
    write_cell_checkpoints,
    write_cell_json_artifacts,
    write_cell_parameter_drift,
    write_cell_parameter_drift_summary,
    write_cell_parameter_snapshots,
    write_canonical_stage_a_artifact,
    write_cell_representations,
    write_layer_representations,
    write_cell_status_summary,
    write_cka_condition_summary,
    write_cka_pairwise_matrix,
    write_formal_consistency,
    write_historical_frozen_comparison,
    write_csv,
    write_error_by_sales_bin,
    write_json_atomic,
    write_layerwise_parameter_drift,
    write_parameter_drift_summary,
    write_parameter_drift_variant_summary,
    write_pointwise_predictions,
    write_pointwise_summary,
    write_training_dynamics,
    write_training_dynamics_summary,
    original_scale_rmse,
    original_scale_smape,
)
from src.transfer_methods.sita_mechanism_diagnostic_rerun import (
    DEFAULT_MATRIX_MODE,
    EXPECTED_DIAGNOSTIC_RERUN_CELLS,
    EXPECTED_FORMAL_300_CELLS,
    EXPECTED_FORMAL_300_ARM_RUNS,
    EXPECTED_FORMAL_300_CANONICAL_CELLS,
    FORMAL_300_ARMS,
    FORMAL_300_MATRIX_MODE,
    DIAGNOSTIC_RERUN_TARGETS,
    DiagnosticRerunCell,
    cells_for_matrix,
    canonical_cells_for_matrix,
    assert_formal_300_cell_structure,
    enumerate_mechanism_diagnostic_rerun_cells,
    matrix_digest_for_cells,
    mechanism_rerun_config_digest,
)
from src.transfer_methods.sita_mechanism_diagnostics import digest_arrays, relative_parameter_drift
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    INITIAL_WEIGHT_VARIANT,
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


def resolve_diagnostic_target_lr(
    dataset: str,
    target: str,
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> Tuple[float, str]:
    mode = str(matrix_mode).strip().lower()
    if mode == FORMAL_300_MATRIX_MODE:
        from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
            FORMAL_PER_DATASET_LR_LABELS,
            FORMAL_PER_DATASET_LR_TARGETS,
            FORMAL_PER_DATASET_LR_VALUES,
        )

        expected = FORMAL_PER_DATASET_LR_TARGETS[str(dataset)]
        if str(target) != expected:
            raise MechanismDiagnosticsContractError(
                f"formal-300 target drifted for {dataset}: {target} != {expected}"
            )
        return float(FORMAL_PER_DATASET_LR_VALUES[str(dataset)]), str(
            FORMAL_PER_DATASET_LR_LABELS[str(dataset)]
        )
    from src.transfer_methods.ms_katl_ae_ablation import resolve_uniform_3e4_lr

    return resolve_uniform_3e4_lr(dataset, target)


def _write_synthetic_layer_representations(
    run_dir: Path,
    cid: str,
    *,
    init_rep: np.ndarray,
    raw_rep: np.ndarray,
    final_rep: np.ndarray,
    digest: str,
) -> None:
    for layer in CKA_REPRESENTATION_LAYERS:
        write_layer_representations(
            run_dir,
            cid,
            layer,
            arrays_by_stage={
                "INIT": init_rep,
                "RAW_FINAL": raw_rep,
                "FINAL": final_rep,
            },
            reference_input_digest_value=digest,
            metadata={
                "layer": layer,
                "reshape_rule": CKA_ACTIVATION_RESHAPE_RULE,
                "n_samples": int(init_rep.shape[0]),
                "feature_shape": list(init_rep.shape[1:]),
            },
        )


def persist_synthetic_cell_artifacts(
    run_dir: Path,
    cell: DiagnosticRerunCell,
    *,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    fail: bool = False,
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> Dict[str, Any]:
    cid = cell.cell_key
    true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64) if y_true is None else np.asarray(y_true)
    pred = np.asarray([1.1, 1.9, 3.2], dtype=np.float64) if y_pred is None else np.asarray(y_pred)
    init_rep = np.eye(max(2, int(true.shape[0])), dtype=np.float32)
    raw_rep = init_rep + 0.005
    final_rep = init_rep + 0.01
    digest = hashlib.sha256(b"synthetic-formal-test-inputs").hexdigest()
    selected_lr, selected_lr_label = resolve_diagnostic_target_lr(
        cell.dataset, cell.target, matrix_mode
    )
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
            "epoch": epoch,
            "before_first_target_optimizer_update": epoch == 0,
            "train_objective_loss": 1.0 if epoch == 0 else 0.8,
            "validation_objective_loss": 1.1 if epoch == 0 else 0.9,
            "validation_rmse_original_scale": 1.2 if epoch == 0 else 1.0,
            "validation_smape_original_scale": 1.3 if epoch == 0 else 1.1,
            "learning_rate": selected_lr,
            "gradient_norm": 0.0 if epoch == 0 else 0.1,
            "parameter_norm": 1.0 if epoch == 0 else 1.05,
            "ema_parameter_norm": 1.0 if epoch == 0 else 1.02,
            "ema_applied": epoch > 0,
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "optimizer_update_count": 0 if epoch == 0 else 50,
            "ema_update_count": 0 if epoch == 0 else 50,
        }
        for epoch in (0, 1)
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
                    "weight_variant": INITIAL_WEIGHT_VARIANT,
                    "optimizer_update_count": 0,
                    "ema_update_count": 0,
                },
                f"{cell.arm}_FINAL": {
                    "stage": "FINAL",
                    "captured_before_first_target_optimizer_update": False,
                    "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
                    "optimizer_update_count": 50,
                    "ema_update_count": 50,
                },
                f"{cell.arm}_RAW_FINAL": {
                    "stage": "RAW_FINAL",
                    "captured_before_first_target_optimizer_update": False,
                    "weight_variant": "RAW",
                    "optimizer_update_count": 50,
                    "ema_update_count": 50,
                },
            },
        },
    )
    write_cell_representations(
        run_dir,
        cid,
        init_rep=init_rep,
        final_rep=final_rep,
        raw_final_rep=raw_rep,
        reference_input_digest_value=digest,
    )
    _write_synthetic_layer_representations(
        run_dir,
        cid,
        init_rep=init_rep,
        raw_rep=raw_rep,
        final_rep=final_rep,
        digest=digest,
    )
    init_named = _named_dummy_weights(1.0)
    raw_named = init_named if cell.arm == "B" else _named_dummy_weights(1.04)
    final_named = init_named if cell.arm == "B" else _named_dummy_weights(1.05)
    extra = {
        "dataset": cell.dataset,
        "target": cell.target,
        "horizon": int(cell.horizon),
        "seed": int(cell.seed),
        "arm": cell.arm,
    }
    ema_drift_rows = parameter_drift_rows(
        init_named, final_named, extra=extra, final_variant="EMA_FINAL"
    )
    raw_drift_rows = parameter_drift_rows(
        init_named, raw_named, extra=extra, final_variant="RAW_FINAL"
    )
    if cell.arm == "B":
        backbone_rows = [
            row for row in ema_drift_rows if row.get("layer_role") == "transferred_backbone"
        ]
        if not backbone_rows or any(
            abs(float(row.get("parameter_l2", float("nan")))) > 1e-12
            for row in backbone_rows
        ):
            raise MechanismDiagnosticsContractError(
                "synthetic B invariant failed"
            )
    write_cell_parameter_drift(run_dir, cid, ema_drift_rows + raw_drift_rows)
    write_cell_parameter_drift_summary(run_dir, cid, ema_drift_rows)
    write_cell_parameter_snapshots(
        run_dir,
        cid,
        {
            (cell.arm, "INIT"): init_named,
            (cell.arm, "RAW_FINAL"): raw_named,
            (cell.arm, "FINAL"): final_named,
        },
    )
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
        "selected_lr": selected_lr,
        "selected_lr_label": selected_lr_label,
        "reference_input_digest": digest,
        "canonical_cell_id": cell.canonical_cell_key,
        "stage_a_shared": {
            "canonical_cell_id": cell.canonical_cell_key,
            "stage_a_identifier": f"stage-a:{cell.canonical_cell_key}",
            "stage_a_digest": f"synthetic-stage-a:{cell.canonical_cell_key}",
            "selected_source_identities": [["synthetic", "source-1"], ["synthetic", "source-2"], ["synthetic", "source-3"]],
            "source_distances": [0.1, 0.5, 0.9],
            "source_weights": [0.5, 0.3, 0.2],
            "source_model_digests": ["synthetic-source-1", "synthetic-source-2", "synthetic-source-3"],
            "source_model_order": ["synthetic-source-1", "synthetic-source-2", "synthetic-source-3"],
            "fused_conv1_digest": "synthetic-fused-conv1",
            "fused_conv2_digest": "synthetic-fused-conv2",
            "fused_conv3_digest": "synthetic-fused-conv3",
            "fused_params_digest": "synthetic-fused-params",
            "target_split_digest": "synthetic-target-split",
            "reference_input_digest": digest,
            "rng_metadata": {
                "cell_seed": int(cell.seed),
                "stage_a_seed": int(cell.seed),
                "stage_a_reset_once_at_entry": True,
                "per_arm_reset_location": "before_arm_training",
                "source_training_order": ["synthetic-source-1", "synthetic-source-2", "synthetic-source-3"],
            },
        },
        "synthetic": True,
        "frozen_ae_root_untouched": "YES",
    }
    if fail:
        return payload
    write_canonical_stage_a_artifact(run_dir, cell, payload["stage_a_shared"])
    write_cell_json_artifacts(
        run_dir,
        cid,
        result=payload,
        status={"status": "COMPLETE", "cell_id": cid, "synthetic": True},
        provenance={
            "status": "COMPLETE",
            "cell_id": cid,
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "diagnostic_rng_neutral": True,
            "canonical_cell_id": cell.canonical_cell_key,
            "stage_a_shared": payload["stage_a_shared"],
            "formal_reference_result": {
                "path": str(
                    FORMAL_PER_DATASET_LR_CSV
                    if str(matrix_mode).strip().lower() == FORMAL_300_MATRIX_MODE
                    else FORMAL_AE_UNIFORM_3E4_CSV
                ),
                "digest": "synthetic",
                "authority_kind": (
                    "FORMAL_PER_DATASET_LR_AE"
                    if str(matrix_mode).strip().lower() == FORMAL_300_MATRIX_MODE
                    else "HISTORICAL_UNIFORM_3E4"
                ),
            },
        },
    )
    return payload


@dataclass
class FakeDiagnosticCellExecutor:
    fail_keys: Tuple[str, ...] = ()
    raise_keys: Tuple[str, ...] = ()
    calls: List[str] = field(default_factory=list)
    matrix_mode: str = DEFAULT_MATRIX_MODE

    def __call__(self, cell: DiagnosticRerunCell, run_dir: Path) -> Mapping[str, Any]:
        self.calls.append(cell.cell_key)
        if cell.cell_key in set(self.raise_keys):
            raise RuntimeError(f"synthetic executor crash: {cell.cell_key}")
        payload = persist_synthetic_cell_artifacts(
            run_dir,
            cell,
            fail=cell.cell_key in set(self.fail_keys),
            matrix_mode=self.matrix_mode,
        )
        if cell.cell_key in set(self.fail_keys):
            raise RuntimeError(f"synthetic executor marked failure: {cell.cell_key}")
        return payload


def diagnostic_variant_flags(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Read the two C-arm structural switches for a diagnostic rerun.

    Both stay off unless the process environment asks for them, so an ordinary
    formal-300 run keeps the previous fusion and the previous Stage-B loss.
    """
    env = os.environ if environ is None else environ
    align_raw = str(env.get("SITA_ALIGN_FUSION_FILTERS", "")).strip().upper()
    align = align_raw in {"1", "YES", "TRUE"}
    lambda_raw = str(env.get("SITA_ARM_C_INIT_ANCHOR_LAMBDA", "")).strip()
    if not lambda_raw:
        anchor = 0.0
    else:
        try:
            anchor = float(lambda_raw)
        except ValueError as exc:
            raise MechanismDiagnosticsContractError(
                f"SITA_ARM_C_INIT_ANCHOR_LAMBDA is not a number: {lambda_raw!r}"
            ) from exc
    if not np.isfinite(anchor) or anchor < 0.0:
        raise MechanismDiagnosticsContractError(
            f"SITA_ARM_C_INIT_ANCHOR_LAMBDA must be finite and >= 0, got {anchor}"
        )
    return {
        "align_fusion_filters": bool(align),
        "ae_arm_c_init_anchor_lambda": float(anchor),
    }


class FormalDiagnosticCellExecutor:
    """Real diagnostic path over the selected matrix.

    The numerical path is deliberately the formal A–E Stage-A/Stage-B/predict/
    metric chain; this class only enables passive capture and writes diagnostics.
    """

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        environ: Optional[Mapping[str, str]] = None,
        matrix_mode: str = DEFAULT_MATRIX_MODE,
        config_digest: str = "",
        matrix_digest: str = "",
        code_digest: str = "",
    ) -> None:
        self.project_root = PROJECT_ROOT if project_root is None else Path(project_root)
        self.environ = environ
        self.matrix_mode = str(matrix_mode).strip().lower()
        self.config_digest = str(config_digest)
        self.matrix_digest = str(matrix_digest)
        self.code_digest = str(code_digest)
        self._canonical_contexts: Dict[str, Dict[str, Any]] = {}

    def _prepare_canonical_context(
        self,
        cell: DiagnosticRerunCell,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Prepare one context/Stage-A realization for all five arms."""
        from src.transfer_methods.ms_katl_ae_execution import resolve_real_canonical_cell
        from src.transfer_methods.ms_katl_ae_formal import (
            default_v3_make_common_head,
            default_v3_prepare_context,
            default_v3_prepare_stage_a,
        )
        from src.transfer_methods.sita_mechanism_diagnostics_capture import (
            ACTIVE_DIAGNOSTIC_ARMS_KEY,
            ENABLE_MECHANISM_DIAGNOSTICS_KEY,
            MECHANISM_DIAGNOSTICS_SESSION_KEY,
            ensure_session,
        )

        expected_target = DIAGNOSTIC_RERUN_TARGETS[cell.dataset]
        if cell.target != expected_target:
            raise MechanismDiagnosticsContractError(
                f"diagnostic target drifted for {cell.dataset}: {cell.target} != {expected_target}"
            )
        lr_value, lr_label = resolve_diagnostic_target_lr(
            cell.dataset, cell.target, self.matrix_mode
        )
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
        context[ACTIVE_DIAGNOSTIC_ARMS_KEY] = (
            FORMAL_300_ARMS if self.matrix_mode == FORMAL_300_MATRIX_MODE else ("A", "B", "C")
        )
        context["mechanism_diagnostics_matrix_mode"] = self.matrix_mode
        context["mechanism_diagnostics_config_digest"] = self.config_digest
        context["mechanism_diagnostics_matrix_digest"] = self.matrix_digest
        context["mechanism_diagnostics_code_digest"] = self.code_digest
        context["mechanism_diagnostics_run_dir"] = str(run_dir)
        context["mechanism_diagnostics_stage_a_call_count"] = 1
        variant = diagnostic_variant_flags(self.environ)
        stage_a_kwargs = context.get("stage_a_kwargs")
        if not isinstance(stage_a_kwargs, dict):
            raise MechanismDiagnosticsContractError("diagnostic Stage-A kwargs are missing")
        stage_a_kwargs["align_fusion_filters"] = bool(variant["align_fusion_filters"])
        context["ae_arm_c_init_anchor_lambda"] = float(variant["ae_arm_c_init_anchor_lambda"])
        context["sita_variant_flags"] = dict(variant)
        stage_a = default_v3_prepare_stage_a(plan, context)
        head_digest = default_v3_make_common_head(plan, stage_a, context)
        bundle = context.get("bundle")
        if bundle is None:
            raise MechanismDiagnosticsContractError("shared Stage-A context is missing bundle")
        # Create the fixed reference set once, before any arm training.
        session = ensure_session(context, cell=cell, stage_a=stage_a, bundle=bundle)
        stage_a_payload = self._stage_a_payload(cell, stage_a, context, session)
        write_canonical_stage_a_artifact(run_dir, cell, stage_a_payload)
        return {
            "plan": plan,
            "context": context,
            "stage_a": stage_a,
            "head_digest": head_digest,
            "stage_a_payload": stage_a_payload,
            "session": session,
            "run_dir": str(Path(run_dir).resolve()),
            "selected_lr": float(lr_value),
            "selected_lr_label": str(lr_label),
        }

    def _stage_a_payload(
        self,
        cell: DiagnosticRerunCell,
        stage_a: Any,
        context: Mapping[str, Any],
        session: Any,
    ) -> Dict[str, Any]:
        from src.transfer_methods.sita_mechanism_diagnostics import digest_arrays

        bundle = context.get("bundle")
        artifacts = context.get("stage_a_artifacts")
        fused_params = getattr(artifacts, "fused_params", None)
        fused_conv_digests = {}
        if isinstance(fused_params, Mapping):
            for layer in ("conv1", "conv2", "conv3"):
                arrays = fused_params.get(layer)
                if arrays is not None:
                    fused_conv_digests[layer] = digest_arrays([np.asarray(item) for item in arrays])
        identities = [list(item) for item in getattr(stage_a, "source_identities", ())]
        source_digests = list(getattr(stage_a, "source_weight_digests", ()) or ())
        source_order = [str(item) for item in source_digests]
        return {
            "canonical_cell_id": cell.canonical_cell_key,
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "stage_a_identifier": str(getattr(stage_a, "stage_a_key", cell.canonical_cell_key)),
            "stage_a_digest": str(getattr(stage_a, "stage_a_digest", "") or getattr(stage_a, "fused_params_digest", "")),
            "selected_source_identities": identities,
            "source_distances": [float(item) for item in getattr(stage_a, "source_distances", ())],
            "source_weights": [float(item) for item in getattr(stage_a, "source_weights", ())],
            "source_model_digests": source_digests,
            "source_model_order": source_order,
            "fused_conv1_digest": fused_conv_digests.get("conv1", ""),
            "fused_conv2_digest": fused_conv_digests.get("conv2", ""),
            "fused_conv3_digest": fused_conv_digests.get("conv3", ""),
            "fused_params_digest": str(getattr(stage_a, "fused_params_digest", "")),
            "align_fusion_filters": bool((context.get("sita_variant_flags") or {}).get("align_fusion_filters", False)),
            "arm_c_init_anchor_lambda": float((context.get("sita_variant_flags") or {}).get("ae_arm_c_init_anchor_lambda", 0.0)),
            "target_split_digest": getattr(stage_a, "target_split_digest", {}),
            "reference_input_digest": str(getattr(session, "reference_input_digest", "")),
            "rng_metadata": {
                "cell_seed": int(cell.seed),
                "stage_a_seed": int(getattr(stage_a, "stage_a_seed", cell.seed) or cell.seed),
                "stage_a_reset_once_at_entry": True,
                "source_training_order": identities,
                "source_model_order": source_order,
                "per_arm_reset_location": "before_arm_training",
                "target_model_construction_seed": int(cell.seed),
                "target_model_construction_reset_location": "construct_ae_arm",
                "diagnostic_rng_neutral": True,
            },
            "code_config_runtime": {
                "config_digest": self.config_digest,
                "matrix_digest": self.matrix_digest,
                "code_digest": self.code_digest,
                "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
                "stage_a_bundle_digest": str(getattr(bundle, "stage_a_digest", "") or getattr(stage_a, "stage_a_digest", "")),
            },
        }

    def __call__(self, cell: DiagnosticRerunCell, run_dir: Path) -> Mapping[str, Any]:
        if FROZEN_AE_OUTPUT_MARKER in Path(run_dir).as_posix():
            raise MechanismDiagnosticsContractError("diagnostic --run refuses the frozen A–E output root")
        from src.transfer_methods.ms_katl_ae_ablation import (
            AETrainingBlocked,
            assert_formal_training_permitted,
            reset_ae_rng,
        )
        from src.transfer_methods.ms_katl_ae_formal import (
            default_v3_metric,
            default_v3_predict,
            default_v3_train_stage_b,
        )
        from src.transfer_methods.sita_mechanism_diagnostics_capture import (
            MECHANISM_DIAGNOSTICS_SESSION_KEY,
        )

        assert_formal_training_permitted(execute=True, environ=self.environ)
        canonical_key = cell.canonical_cell_key
        prepared = self._canonical_contexts.get(canonical_key)
        if prepared is None:
            prepared = self._prepare_canonical_context(cell, run_dir)
            self._canonical_contexts[canonical_key] = prepared
        elif prepared["run_dir"] != str(Path(run_dir).resolve()):
            raise MechanismDiagnosticsContractError(
                f"canonical cell {canonical_key} attempted in multiple run roots"
            )
        plan = prepared["plan"]
        context = prepared["context"]
        stage_a = prepared["stage_a"]
        head = prepared["head_digest"]
        context["training_dynamics_rows"] = []
        # The reset belongs to the explicitly defined arm execution contract;
        # diagnostic capture itself does not call any RNG API.
        reset_ae_rng(int(cell.seed))
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
        raw_final_key = (cell.arm, "RAW_FINAL")
        final_key = (cell.arm, "FINAL")
        if (
            init_key not in session.checkpoints
            or raw_final_key not in session.checkpoints
            or final_key not in session.checkpoints
        ):
            raise MechanismDiagnosticsContractError("INIT/RAW_FINAL/FINAL checkpoints missing")
        write_cell_checkpoints(run_dir, cid, session.as_dict())
        write_cell_representations(
            run_dir,
            cid,
            init_rep=session.representations[init_key],
            final_rep=session.representations[final_key],
            raw_final_rep=session.representations[raw_final_key],
            reference_input_digest_value=str(session.reference_input_digest),
        )
        named_snapshots = getattr(session, "named_layer_weights", None)
        named_init = named_snapshots.get(init_key) if isinstance(named_snapshots, Mapping) else None
        named_raw = named_snapshots.get(raw_final_key) if isinstance(named_snapshots, Mapping) else None
        named_final = named_snapshots.get(final_key) if isinstance(named_snapshots, Mapping) else None
        if named_init is None or named_raw is None or named_final is None:
            raise MechanismDiagnosticsContractError(
                "INIT/RAW_FINAL/FINAL named layer weight snapshots missing"
            )
        extra = {
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
        }
        ema_drift_rows = parameter_drift_rows(
            named_init, named_final, extra=extra, final_variant="EMA_FINAL"
        )
        raw_drift_rows = parameter_drift_rows(
            named_init, named_raw, extra=extra, final_variant="RAW_FINAL"
        )
        drift_rows = ema_drift_rows + raw_drift_rows
        if cell.arm == "B":
            backbone_rows = [
                row
                for row in ema_drift_rows + raw_drift_rows
                if row.get("layer_role") == "transferred_backbone"
            ]
            if not backbone_rows or any(
                abs(float(row.get("parameter_l2", float("nan")))) > 1e-12
                or abs(float(row.get("relative_parameter_drift", float("nan")))) > 1e-12
                for row in backbone_rows
            ):
                raise MechanismDiagnosticsContractError(
                    "B invariant failed: controlled frozen backbone drift is not zero"
                )
        write_cell_parameter_drift(run_dir, cid, drift_rows)
        write_cell_parameter_drift_summary(run_dir, cid, ema_drift_rows)
        snapshot_path = write_cell_parameter_snapshots(
            run_dir,
            cid,
            getattr(session, "named_layer_weights", {}),
        )
        named_reps = getattr(session, "named_representations", {})
        for layer in CKA_REPRESENTATION_LAYERS:
            layer_arrays = {}
            for stage, key in (("INIT", init_key), ("RAW_FINAL", raw_final_key), ("FINAL", final_key)):
                layer_map = named_reps.get(key) if isinstance(named_reps, Mapping) else None
                if not isinstance(layer_map, Mapping) or layer not in layer_map:
                    if layer == "flatten":
                        layer_arrays[stage] = session.representations[key]
                    else:
                        raise MechanismDiagnosticsContractError(
                            f"{layer} representation missing for {cell.arm} {stage}"
                        )
                else:
                    layer_arrays[stage] = layer_map[layer]
            write_layer_representations(
                run_dir,
                cid,
                layer,
                arrays_by_stage=layer_arrays,
                reference_input_digest_value=str(session.reference_input_digest),
                metadata={
                    "dataset": cell.dataset,
                    "target": cell.target,
                    "horizon": int(cell.horizon),
                    "seed": int(cell.seed),
                    "arm": cell.arm,
                    "layer": layer,
                    "n_samples": int(np.asarray(layer_arrays["FINAL"]).shape[0]),
                    "feature_shape": list(np.asarray(layer_arrays["FINAL"]).shape[1:]),
                    "reference_input_digest": str(session.reference_input_digest),
                    "representation_digest": digest_arrays([np.asarray(layer_arrays["FINAL"])]),
                    "reshape_rule": CKA_ACTIVATION_RESHAPE_RULE,
                },
            )
        from src.transfer_methods.ms_katl_ae_formal import serialize_formal_arm_runtime

        runtime_provenance = serialize_formal_arm_runtime(runtime)
        init_ckpt = session.checkpoints[init_key]
        raw_ckpt = session.checkpoints[raw_final_key]
        final_ckpt = session.checkpoints[final_key]
        if int(init_ckpt.optimizer_update_count) != 0:
            raise MechanismDiagnosticsContractError("INIT optimizer_update_count must be 0")
        if int(final_ckpt.optimizer_update_count) != int(raw_ckpt.optimizer_update_count):
            raise MechanismDiagnosticsContractError(
                "FINAL optimizer_update_count must equal RAW_FINAL optimizer_update_count"
            )
        if int(raw_ckpt.optimizer_update_count) <= 0:
            raise MechanismDiagnosticsContractError("RAW_FINAL optimizer_update_count must be > 0")
        formal_reference_path = (
            Path(FORMAL_PER_DATASET_LR_CSV).resolve()
            if self.matrix_mode == FORMAL_300_MATRIX_MODE
            else Path(FORMAL_AE_UNIFORM_3E4_CSV).resolve()
        )
        result_payload = {
            "cell_id": cid,
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "rerun_rmse": float(runtime.rmse_original) if runtime.rmse_original is not None else float("nan"),
            "rerun_smape": float(runtime.smape_original) if runtime.smape_original is not None else float("nan"),
            "selected_lr": float(prepared["selected_lr"]),
            "selected_lr_label": str(prepared["selected_lr_label"]),
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "reference_input_digest": str(session.reference_input_digest),
            "raw_final_backbone_digest": session.checkpoints[raw_final_key].backbone_digest,
            "ema_final_backbone_digest": session.checkpoints[final_key].backbone_digest,
            "prediction_digest": runtime.prediction_digest,
            "final_formal_model_digest": runtime.final_formal_model_digest,
            "stage_a_digest": runtime.stage_a_digest,
            "fused_init_digest": runtime.fused_init_digest,
            "synthetic": False,
            "frozen_ae_root_untouched": "YES",
            "published_to_frozen_ae_root": "NO",
            "matrix_mode": self.matrix_mode,
            "canonical_cell_id": canonical_key,
            "stage_a_shared": prepared["stage_a_payload"],
            "stage_a_shared_across_arms": True,
            "per_arm_rng_reset": True,
            "parameter_snapshot_path": str(snapshot_path),
            "runtime": runtime_provenance,
            "ema_audit": {
                "ema_enabled": True,
                "ema_decay": 0.99,
                "ema_update_timing": "after_optimizer_update",
                "ema_assign_timing": "after_RAW_FINAL_capture",
                "formal_prediction_weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
                "optimizer_update_count": int(raw_ckpt.optimizer_update_count),
                "ema_update_count": int(raw_ckpt.ema_update_count),
            },
        }
        provenance = {
            "status": "COMPLETE",
            "cell_id": cid,
            "matrix_mode": self.matrix_mode,
            "dataset": cell.dataset,
            "target": cell.target,
            "horizon": int(cell.horizon),
            "seed": int(cell.seed),
            "arm": cell.arm,
            "code_digest": self.code_digest,
            "config_digest": self.config_digest,
            "matrix_digest": self.matrix_digest,
            "rng_seed_semantics": {
                "cell_seed": int(cell.seed),
                "stage_a_seed": int(cell.seed),
                "target_adaptation_seed": int(cell.seed),
                "stage_a_reset_once_at_entry": True,
                "stage_a_reset_before_source_model_creation": True,
                "stage_a_per_source_reseed": False,
                "diagnostic_rng_neutral": True,
            },
            "source_selection": {
                "source_identities": [list(item) for item in runtime.source_identities],
                "source_distances": list(runtime.source_distances),
                "source_weights": list(runtime.source_weights),
                "source_selection_digest": runtime.source_selection_digest,
                "source_model_digests": list(runtime.source_model_digests),
            },
            "source_parameter_fusion": {
                "fused_params_digest": runtime.fused_params_digest,
                "fused_init_digest": runtime.fused_init_digest,
                "target_initial_conv_digest": runtime.target_initial_conv_digest,
                "best_source_identity": list(runtime.best_source_identity),
                "best_source_snapshot_digest": runtime.best_source_snapshot_digest,
                "stage_a_digest": runtime.stage_a_digest,
            },
            "target_adaptation": {
                "init_snapshot": "source-informed initialization before first target optimizer update",
                "final_snapshot": "final EMA backbone after target adaptation",
                "backbone_layers": ["conv1", "conv2", "conv3"],
            },
            "ema_variant": runtime.formal_weight_variant or FORMAL_EVALUATION_WEIGHT_VARIANT,
            "selected_lr": float(prepared["selected_lr"]),
            "selected_lr_label": str(prepared["selected_lr_label"]),
            "formal_reference_result": {
                "path": str(formal_reference_path),
                "digest": formal_reference_result_digest(formal_reference_path)
                if Path(formal_reference_path).is_file()
                else "MISSING",
                "authority_kind": (
                    "FORMAL_PER_DATASET_LR_AE"
                    if self.matrix_mode == FORMAL_300_MATRIX_MODE
                    else "HISTORICAL_UNIFORM_3E4"
                ),
            },
            "b_backbone_zero_drift_invariant": (
                "PASS" if cell.arm == "B" else "NOT_APPLICABLE"
            ),
            "reference_input_digest": str(session.reference_input_digest),
            "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
            "frozen_ae_root_untouched": "YES",
            "canonical_cell_id": canonical_key,
            "stage_a_shared": prepared["stage_a_payload"],
            "stage_a_shared_across_arms": True,
            "per_arm_rng_reset": True,
        }
        write_cell_json_artifacts(
            run_dir,
            cid,
            result=result_payload,
            status={"status": "COMPLETE", "cell_id": cid, "matrix_mode": self.matrix_mode},
            provenance=provenance,
        )
        _ = AETrainingBlocked
        return result_payload


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _audit_cell_artifacts(run_dir: Path, cell: DiagnosticRerunCell) -> List[str]:
    """Return completeness/semantic errors for one completed arm."""
    errors: List[str] = []
    cid = cell.cell_key
    safe = filesystem_safe_id(cid)
    if not required_cell_outputs_complete(run_dir, cid):
        errors.append("required_cell_outputs_missing")
        return errors
    result = _load_json(run_dir / "cells" / safe / "result.json")
    status = _load_json(run_dir / "cells" / safe / "status.json")
    provenance = _load_json(run_dir / "cells" / safe / "provenance.json")
    if status.get("status") != CELL_STATUS_COMPLETE:
        errors.append("cell_status_not_COMPLETE")
    for payload_name, payload in (("result", result), ("provenance", provenance)):
        if payload.get("canonical_cell_id") != cell.canonical_cell_key:
            errors.append(f"{payload_name}_canonical_cell_id_missing_or_mismatched")
        if not payload.get("stage_a_shared"):
            errors.append(f"{payload_name}_stage_a_shared_missing")
    if not _finite(result.get("rerun_rmse")):
        errors.append("rmse_nonfinite")
    if not _finite(result.get("rerun_smape")):
        errors.append("smape_nonfinite")
    pred = _load_csv_rows(run_dir / "predictions" / f"{safe}.csv")
    if not pred:
        errors.append("predictions_missing")
    else:
        for row in pred:
            for field in ("dataset", "target", "horizon", "seed", "arm", "y_true", "y_pred", "absolute_error", "squared_error"):
                if row.get(field) in (None, ""):
                    errors.append(f"prediction_{field}_missing")
            for field in ("y_true", "y_pred", "absolute_error", "squared_error"):
                if not _finite(row.get(field)):
                    errors.append(f"prediction_{field}_nonfinite")
    dynamics = _load_csv_rows(run_dir / "training_dynamics" / f"{safe}.csv")
    epochs = {int(row["epoch"]) for row in dynamics if row.get("epoch") not in (None, "")}
    if 0 not in epochs or not dynamics or max(epochs) < 1:
        errors.append("training_dynamics_missing_epoch0_or_final")
    for row in dynamics:
        for field in (
            "train_objective_loss",
            "validation_objective_loss",
            "validation_rmse_original_scale",
            "validation_smape_original_scale",
            "learning_rate",
            "gradient_norm",
            "parameter_norm",
            "ema_parameter_norm",
        ):
            if row.get(field) not in (None, "") and not _finite(row.get(field)):
                errors.append(f"training_dynamics_{field}_nonfinite")
    checkpoint = _load_json(run_dir / "checkpoints" / f"{safe}.json")
    checkpoints = checkpoint.get("checkpoints", {})
    by_stage = {str(item.get("stage")): item for item in checkpoints.values()}
    if "INIT" not in by_stage:
        errors.append("INIT_checkpoint_missing")
    else:
        if int(by_stage["INIT"].get("optimizer_update_count", -1)) != 0:
            errors.append("INIT_optimizer_update_count_not_zero")
        if str(by_stage["INIT"].get("weight_variant")) not in {INITIAL_WEIGHT_VARIANT, FORMAL_EVALUATION_WEIGHT_VARIANT}:
            errors.append("INIT_weight_variant_invalid")
    if "RAW_FINAL" not in by_stage:
        errors.append("RAW_FINAL_checkpoint_missing")
    if "FINAL" not in by_stage:
        errors.append("FINAL_checkpoint_missing")
    if "RAW_FINAL" in by_stage and "FINAL" in by_stage:
        if int(by_stage["FINAL"].get("optimizer_update_count", -1)) != int(
            by_stage["RAW_FINAL"].get("optimizer_update_count", -2)
        ):
            errors.append("FINAL_optimizer_update_count_ne_RAW_FINAL")
        if int(by_stage["RAW_FINAL"].get("optimizer_update_count", 0)) <= 0:
            errors.append("RAW_FINAL_optimizer_update_count_not_positive")
    drift = _load_csv_rows(run_dir / "parameter_drift" / f"{safe}.csv")
    variants = {str(row.get("final_variant") or "EMA_FINAL") for row in drift}
    if "EMA_FINAL" not in variants and not any(str(row.get("final_variant") or "") == "" for row in drift):
        if not any(str(row.get("layer")) == "conv1" for row in drift):
            errors.append("drift_rows_missing")
    layers_ema = {
        str(row.get("layer")): row
        for row in drift
        if str(row.get("final_variant") or "EMA_FINAL") in {"EMA_FINAL", ""}
    }
    for layer in ("conv1", "conv2", "conv3", "head_target"):
        if layer not in layers_ema:
            errors.append(f"drift_{layer}_missing")
        else:
            for field in ("parameter_l2", "relative_parameter_drift"):
                if not _finite(layers_ema[layer].get(field)):
                    errors.append(f"drift_{layer}_{field}_nonfinite")
    if cell.arm == "B":
        for layer in ("conv1", "conv2", "conv3"):
            if layer in layers_ema and (
                abs(float(layers_ema[layer]["parameter_l2"])) > 1e-12
                or abs(float(layers_ema[layer]["relative_parameter_drift"])) > 1e-12
            ):
                errors.append(f"B_{layer}_drift_not_zero")
    for layer in CKA_REPRESENTATION_LAYERS:
        layer_path = run_dir / "representations" / layer / f"{safe}.npz"
        if not layer_path.is_file() or layer_path.stat().st_size == 0:
            errors.append(f"{layer}_representation_missing")
            continue
        try:
            blob = np.load(layer_path, allow_pickle=True)
            if not {"INIT", "RAW_FINAL", "FINAL"} <= set(blob.files):
                errors.append(f"{layer}_representation_stages_missing")
        except Exception:
            errors.append(f"{layer}_representation_unreadable")
    try:
        blob = np.load(run_dir / "representations" / f"{safe}.npz", allow_pickle=True)
        if (
            "INIT" not in blob.files
            or "RAW_FINAL" not in blob.files
            or "FINAL" not in blob.files
            or "reference_input_digest" not in blob.files
        ):
            errors.append("representation_or_reference_digest_missing")
    except Exception:
        errors.append("representation_unreadable")
    return errors


def _audit_stage_a_shared(
    run_dir: Path,
    cells: Sequence[DiagnosticRerunCell],
    *,
    require_five_arms: bool,
) -> List[str]:
    errors: List[str] = []
    groups: Dict[str, List[DiagnosticRerunCell]] = {}
    for cell in cells:
        groups.setdefault(cell.canonical_cell_key, []).append(cell)
    if require_five_arms:
        try:
            assert_formal_300_cell_structure(cells)
        except MechanismDiagnosticsContractError as exc:
            errors.append(str(exc))
    for canonical_id, group in groups.items():
        digests = set()
        source_digests = set()
        reference_digests = set()
        stage_path = run_dir / "stage_a" / f"{filesystem_safe_id(canonical_id)}.json"
        stage_a = _load_json(stage_path)
        if not stage_a:
            errors.append(f"{canonical_id}:stage_a_shared_missing")
            continue
        for cell in group:
            payload = _load_json(run_dir / "cells" / filesystem_safe_id(cell.cell_key) / "provenance.json")
            shared = payload.get("stage_a_shared", {})
            digests.add(str(shared.get("stage_a_digest", "")))
            source_digests.add(str(shared.get("fused_params_digest", "")))
            reference_digests.add(str(shared.get("reference_input_digest", "")))
        if len(group) == 5 and {cell.arm for cell in group} != set(FORMAL_300_ARMS):
            errors.append(f"{canonical_id}:arms_are_not_A_to_E")
        if len(digests) != 1 or "" in digests:
            errors.append(f"{canonical_id}:stage_a_digest_not_shared")
        if len(source_digests) != 1 or "" in source_digests:
            errors.append(f"{canonical_id}:fused_params_digest_not_shared")
        if len(reference_digests) != 1 or "" in reference_digests:
            errors.append(f"{canonical_id}:reference_input_digest_not_shared")
        if str(stage_a.get("stage_a_digest", "")) not in digests:
            errors.append(f"{canonical_id}:stage_a_artifact_digest_mismatch")
    return errors


def finalize_mechanism_diagnostic_run(
    run_dir: Path,
    cells: Sequence[DiagnosticRerunCell],
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    matrix_mode: str = DEFAULT_MATRIX_MODE,
    allow_cell_subset: bool = False,
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
    if matrix_mode == FORMAL_300_MATRIX_MODE and not allow_cell_subset:
        if len(tuple(cells)) != EXPECTED_FORMAL_300_ARM_RUNS:
            raise MechanismDiagnosticsContractError(
                f"formal-300 finalize received {len(tuple(cells))} arm runs, "
                f"expected {EXPECTED_FORMAL_300_ARM_RUNS}"
            )
    artifact_errors = {
        cell.cell_key: _audit_cell_artifacts(run_dir, cell)
        for cell in complete
    }
    artifact_errors = {key: value for key, value in artifact_errors.items() if value}
    if artifact_errors:
        return {
            "finalized": "NO",
            "reason": "completeness/semantic audit failed",
            "artifact_errors": json.dumps(artifact_errors, sort_keys=True),
            "cell_status": str(run_dir / "summaries" / "cell_status.csv"),
        }
    shared_errors = _audit_stage_a_shared(
        run_dir,
        complete,
        require_five_arms=matrix_mode == FORMAL_300_MATRIX_MODE and not allow_cell_subset,
    )
    if shared_errors:
        return {
            "finalized": "NO",
            "reason": "Stage-A sharing audit failed",
            "stage_a_errors": json.dumps(shared_errors, sort_keys=True),
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
        ema_only = [
            row for row in drift if str(row.get("final_variant") or "EMA_FINAL") in {"EMA_FINAL", ""}
        ]
        backbone = [row for row in ema_only if classify_layer_role(str(row.get("layer"))) == "transferred_backbone"]
        rels = [float(row["relative_parameter_drift"]) for row in backbone if row.get("relative_parameter_drift")]
        by_layer = {str(row.get("layer")): row for row in ema_only}
        def _drift_value(layer: str, field: str) -> float:
            return float(by_layer[layer][field]) if layer in by_layer else float("nan")
        backbone_l2 = [_drift_value(layer, "parameter_l2") for layer in ("conv1", "conv2", "conv3") if layer in by_layer]
        backbone_rel = [_drift_value(layer, "relative_parameter_drift") for layer in ("conv1", "conv2", "conv3") if layer in by_layer]
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
                "conv1_l2": _drift_value("conv1", "parameter_l2"),
                "conv2_l2": _drift_value("conv2", "parameter_l2"),
                "conv3_l2": _drift_value("conv3", "parameter_l2"),
                "head_target_l2": _drift_value("head_target", "parameter_l2"),
                "whole_backbone_l2": float(np.sqrt(np.sum(np.square(backbone_l2)))) if backbone_l2 else float("nan"),
                "conv1_relative_parameter_drift": _drift_value("conv1", "relative_parameter_drift"),
                "conv2_relative_parameter_drift": _drift_value("conv2", "relative_parameter_drift"),
                "conv3_relative_parameter_drift": _drift_value("conv3", "relative_parameter_drift"),
                "head_target_relative_parameter_drift": _drift_value("head_target", "relative_parameter_drift"),
                "whole_backbone_relative_parameter_drift": float(np.sqrt(np.sum(np.square(backbone_rel)))) if backbone_rel else float("nan"),
            }
        )
        for layer in CKA_REPRESENTATION_LAYERS:
            layer_path = run_dir / "representations" / layer / f"{safe}.npz"
            if not layer_path.is_file():
                continue
            blob = np.load(layer_path, allow_pickle=True)
            digest = str(np.asarray(blob["reference_input_digest"]).reshape(-1)[0])
            for stage in ("INIT", "RAW_FINAL", "FINAL"):
                if stage in blob.files:
                    cka_records.append(
                        {
                            "arm": cell.arm,
                            "seed": int(cell.seed),
                            "stage": stage,
                            "condition_id": cell.condition_key,
                            "layer": layer,
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
    pointwise_summary_rows: List[Dict[str, Any]] = []
    for cid, rows in sorted(by_cell_preds.items()):
        y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
        y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
        y_pred_ema = np.asarray(
            [float(row.get("y_pred_ema_original_scale", row["y_pred"])) for row in rows],
            dtype=np.float64,
        )
        max_abs = float(np.max(np.abs(y_pred - y_pred_ema))) if y_pred.size else 0.0
        first = rows[0] if rows else {}
        result_blob = _load_json(
            run_dir / "cells" / filesystem_safe_id(str(cid)) / "result.json"
        ) if (run_dir / "cells" / filesystem_safe_id(str(cid)) / "result.json").is_file() else {}
        pointwise_summary_rows.append(
            {
                "dataset": first.get("dataset"),
                "target": first.get("target"),
                "horizon": first.get("horizon"),
                "seed": first.get("seed"),
                "arm": first.get("arm"),
                "n_samples": len(rows),
                "weight_variant": first.get("weight_variant", FORMAL_EVALUATION_WEIGHT_VARIANT),
                "reconstructed_rmse": original_scale_rmse(y_true, y_pred),
                "reconstructed_smape": original_scale_smape(y_true, y_pred),
                "max_abs_y_pred_minus_y_pred_ema": max_abs,
                "y_pred_equals_y_pred_ema": bool(max_abs == 0.0),
                "prediction_digest": result_blob.get("prediction_digest")
                or (result_blob.get("runtime") or {}).get("prediction_digest")
                or "",
            }
        )
    write_pointwise_summary(run_dir, pointwise_summary_rows)
    write_layerwise_parameter_drift(run_dir, layerwise)
    write_parameter_drift_summary(run_dir, drift_summary)
    raw_drift_rows = [
        row for row in layerwise if str(row.get("final_variant") or "") == "RAW_FINAL"
    ]
    ema_drift_rows = [
        row for row in layerwise if str(row.get("final_variant") or "EMA_FINAL") in {"EMA_FINAL", ""}
    ]
    write_parameter_drift_variant_summary(run_dir, raw_drift_rows, filename="parameter_drift_raw.csv")
    write_parameter_drift_variant_summary(run_dir, ema_drift_rows, filename="parameter_drift_ema.csv")
    cka_rows: List[Dict[str, Any]] = []
    condition_rows = []
    by_condition_layer: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in cka_records:
        by_condition_layer.setdefault(
            (str(record["condition_id"]), str(record.get("layer") or "flatten")),
            [],
        ).append(record)
    for (cid, layer), records in sorted(by_condition_layer.items()):
        digests = {str(item["reference_input_digest"]) for item in records}
        if len(digests) != 1:
            raise MechanismDiagnosticsContractError(
                f"one condition -> exactly one reference_input_digest violated for {cid}: {sorted(digests)}"
            )
        matrix = generate_cka_comparison_matrix(
            records,
            reference_scale_available=reference_scale_available(cid.split("|", 1)[0]),
        )
        for row in matrix:
            cka_rows.append(dict(row, condition_id=cid, layer=layer))
        condition_rows.append(
            {
                "condition_id": cid,
                "layer": layer,
                "n_comparisons": len(matrix),
                "reference_input_digest": next(iter(digests)) if len(digests) == 1 else "MIXED",
                "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            }
        )
    write_cka_pairwise_matrix(run_dir, cka_rows)
    write_cka_condition_summary(run_dir, condition_rows)
    for layer in CKA_REPRESENTATION_LAYERS:
        layer_rows = [row for row in cka_rows if str(row.get("layer")) == layer]
        write_csv(
            run_dir / "summaries" / f"cka_{layer}.csv",
            layer_rows,
            sorted({key for row in layer_rows for key in row}) or ["comparison_type", "cka", "layer"],
        )
    historical_rows: List[Dict[str, Any]] = []
    if matrix_mode != FORMAL_300_MATRIX_MODE and Path(FORMAL_AE_UNIFORM_3E4_CSV).is_file():
        historical = load_formal_ae_uniform_3e4(FORMAL_AE_UNIFORM_3E4_CSV)
        for cell in complete:
            key = (cell.dataset, cell.target, int(cell.horizon), int(cell.seed), cell.arm)
            formal_row = historical.get(key)
            if formal_row is None:
                continue
            outcome = outcomes[cell.cell_key]
            from src.transfer_methods.sita_mechanism_diagnostic_outputs import consistency_audit_row

            historical_rows.append(
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
                    comparison_purpose="historical_reference_only",
                )
            )
        write_historical_frozen_comparison(run_dir, historical_rows)
    consistency_rows: List[Dict[str, Any]] = []
    identity_counts = {EXACT_NUMERIC_MATCH: 0, TOLERANCE_MATCH: 0, MISMATCH: 0}
    if matrix_mode == FORMAL_300_MATRIX_MODE:
        authority = load_formal_per_dataset_lr_authority(FORMAL_PER_DATASET_LR_AUTHORITY_ROOT)
        from src.transfer_methods.sita_mechanism_diagnostic_outputs import _relative_diff

        for cell in complete:
            key = (cell.dataset, cell.target, int(cell.horizon), int(cell.seed), cell.arm)
            formal_row = authority.get(key, {})
            result = _load_json(run_dir / "cells" / filesystem_safe_id(cell.cell_key) / "result.json")
            runtime = result.get("runtime") if isinstance(result.get("runtime"), Mapping) else {}
            diagnostic_rmse = float(outcomes[cell.cell_key].get("rerun_rmse", result.get("rerun_rmse", float("nan"))))
            diagnostic_smape = float(outcomes[cell.cell_key].get("rerun_smape", result.get("rerun_smape", float("nan"))))
            formal_rmse = float(formal_row.get("formal_rmse", float("nan")))
            formal_smape = float(formal_row.get("formal_smape", float("nan")))
            abs_rmse = abs(diagnostic_rmse - formal_rmse)
            abs_smape = abs(diagnostic_smape - formal_smape)
            rel_rmse = _relative_diff(abs_rmse, formal_rmse)
            rel_smape = _relative_diff(abs_smape, formal_smape)
            digest_pairs = {
                "prediction_digest": (
                    formal_row.get("prediction_digest"),
                    result.get("prediction_digest") or runtime.get("prediction_digest"),
                ),
                "final_formal_model_digest": (
                    formal_row.get("final_formal_model_digest"),
                    result.get("final_formal_model_digest") or runtime.get("final_formal_model_digest"),
                ),
                "stage_a_digest": (
                    formal_row.get("stage_a_digest"),
                    result.get("stage_a_digest") or runtime.get("stage_a_digest"),
                ),
                "fused_init_digest": (
                    formal_row.get("fused_init_digest"),
                    result.get("fused_init_digest") or runtime.get("fused_init_digest"),
                ),
            }
            comparable = [
                left == right
                for left, right in digest_pairs.values()
                if left not in (None, "") and right not in (None, "")
            ]
            digest_match = bool(comparable) and all(comparable)
            identity = identity_match_status(
                abs_rmse=abs_rmse,
                abs_smape=abs_smape,
                rel_rmse=rel_rmse,
                rel_smape=rel_smape,
                digest_match=digest_match,
            )
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
            from src.transfer_methods.sita_mechanism_diagnostic_outputs import audit_status_from_diffs

            consistency_rows.append(
                {
                    "dataset": cell.dataset,
                    "target": cell.target,
                    "horizon": int(cell.horizon),
                    "seed": int(cell.seed),
                    "arm": cell.arm,
                    "formal_selected_lr": formal_row.get("formal_selected_lr"),
                    "diagnostic_selected_lr": result.get("selected_lr") or runtime.get("selected_lr"),
                    "formal_rmse": formal_rmse,
                    "diagnostic_rmse": diagnostic_rmse,
                    "abs_rmse_diff": abs_rmse,
                    "rel_rmse_diff": rel_rmse,
                    "formal_smape": formal_smape,
                    "diagnostic_smape": diagnostic_smape,
                    "abs_smape_diff": abs_smape,
                    "rel_smape_diff": rel_smape,
                    "prediction_digest_match": digest_pairs["prediction_digest"][0] == digest_pairs["prediction_digest"][1],
                    "final_formal_model_digest_match": digest_pairs["final_formal_model_digest"][0]
                    == digest_pairs["final_formal_model_digest"][1],
                    "stage_a_digest_match": digest_pairs["stage_a_digest"][0] == digest_pairs["stage_a_digest"][1],
                    "fused_init_digest_match": digest_pairs["fused_init_digest"][0] == digest_pairs["fused_init_digest"][1],
                    "identity_status": identity,
                    "severity_status": audit_status_from_diffs(rel_rmse * 100.0, rel_smape * 100.0),
                    "formal_prediction_digest": digest_pairs["prediction_digest"][0],
                    "diagnostic_prediction_digest": digest_pairs["prediction_digest"][1],
                    "formal_final_formal_model_digest": digest_pairs["final_formal_model_digest"][0],
                    "diagnostic_final_formal_model_digest": digest_pairs["final_formal_model_digest"][1],
                    "formal_stage_a_digest": digest_pairs["stage_a_digest"][0],
                    "diagnostic_stage_a_digest": digest_pairs["stage_a_digest"][1],
                    "formal_fused_init_digest": digest_pairs["fused_init_digest"][0],
                    "diagnostic_fused_init_digest": digest_pairs["fused_init_digest"][1],
                    "authority_root": str(FORMAL_PER_DATASET_LR_AUTHORITY_ROOT),
                    "authority_kind": "FORMAL_PER_DATASET_LR_AE",
                }
            )
        write_formal_consistency(run_dir, consistency_rows)
    write_json_atomic(
        run_dir / "summaries" / "mechanism_readiness.json",
        {
            "n_cells": len(tuple(cells)),
            "n_complete": len(complete),
            "n_canonical_cells": len({cell.canonical_cell_key for cell in complete}),
            "cka_layers": list(CKA_REPRESENTATION_LAYERS),
            "parameter_drift_raw_rows": len(raw_drift_rows),
            "parameter_drift_ema_rows": len(ema_drift_rows),
            "cka_rows": len(cka_rows),
            "identity_counts": identity_counts,
            "authority_kind": (
                "FORMAL_PER_DATASET_LR_AE" if matrix_mode == FORMAL_300_MATRIX_MODE else "HISTORICAL_UNIFORM_3E4"
            ),
        },
    )
    _ = relative_parameter_drift
    _ = primary_transferred_backbone_weights
    return {
        "finalized": "YES",
        "training_dynamics_summary": str(run_dir / "summaries" / "training_dynamics_summary.csv"),
        "error_by_sales_bin": str(run_dir / "summaries" / "error_by_sales_bin.csv"),
        "layerwise_parameter_drift": str(run_dir / "summaries" / "layerwise_parameter_drift.csv"),
        "parameter_drift_summary": str(run_dir / "summaries" / "parameter_drift_summary.csv"),
        "parameter_drift_raw": str(run_dir / "summaries" / "parameter_drift_raw.csv"),
        "parameter_drift_ema": str(run_dir / "summaries" / "parameter_drift_ema.csv"),
        "cka_pairwise_matrix": str(run_dir / "summaries" / "cka_pairwise_matrix.csv"),
        "cka_condition_summary": str(run_dir / "summaries" / "cka_condition_summary.csv"),
        "formal_consistency": str(run_dir / "summaries" / "formal_consistency.csv"),
        "pointwise_summary": str(run_dir / "summaries" / "pointwise_summary.csv"),
        "historical_frozen_comparison": str(
            run_dir / "summaries" / "historical_frozen_comparison.csv"
        ),
        "cell_status": str(run_dir / "summaries" / "cell_status.csv"),
        "replaces_frozen_results": "NO",
        "identity_counts": identity_counts,
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
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if FROZEN_AE_OUTPUT_MARKER in run_dir.as_posix():
        raise MechanismDiagnosticsContractError("diagnostic cell loop refuses the frozen A–E output root")
    mode = str(matrix_mode).strip().lower()
    universe = cells_for_matrix(mode)
    expected_n = (
        EXPECTED_FORMAL_300_CELLS if mode == FORMAL_300_MATRIX_MODE else EXPECTED_DIAGNOSTIC_RERUN_CELLS
    )
    if mode == DEFAULT_MATRIX_MODE:
        if len(universe) != 42:
            raise MechanismDiagnosticsContractError(f"diagnostic matrix drifted: {len(universe)} != 42")
    elif len(universe) != expected_n:
        raise MechanismDiagnosticsContractError(
            f"{mode} diagnostic matrix drifted: {len(universe)} != {expected_n}"
        )
    resolved = tuple(cells) if cells is not None else universe
    if allow_cell_subset:
        if not resolved:
            raise MechanismDiagnosticsContractError("--only-cell subset is empty")
        universe_keys = {cell.cell_key for cell in universe}
        unknown = [cell.cell_key for cell in resolved if cell.cell_key not in universe_keys]
        if unknown:
            raise MechanismDiagnosticsContractError(
                f"--only-cell is not in the {mode} diagnostic matrix: {unknown}"
            )
    elif mode == DEFAULT_MATRIX_MODE:
        if len(resolved) != 42:
            raise MechanismDiagnosticsContractError(f"diagnostic matrix drifted: {len(resolved)} != 42")
    elif len(resolved) != expected_n:
        raise MechanismDiagnosticsContractError(
            f"{mode} diagnostic matrix drifted: {len(resolved)} != {expected_n}"
        )
    cfg = config_digest or mechanism_rerun_config_digest(resolved, matrix_mode=mode)
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
        matrix_mode=mode,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    if hasattr(worker, "matrix_mode"):
        worker.matrix_mode = mode
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
            stage_a_shared = payload.get("stage_a_shared")
            if isinstance(stage_a_shared, Mapping):
                write_canonical_stage_a_artifact(run_dir, cell, stage_a_shared)
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
    finalize = finalize_mechanism_diagnostic_run(
        run_dir,
        resolved,
        outcomes,
        matrix_mode=mode,
        allow_cell_subset=allow_cell_subset,
    )
    write_json_atomic(
        run_dir / "run_manifest.json",
        {
            "experiment_type": "independent_mechanism_diagnostic",
            "matrix_mode": mode,
            "execution_unit": "canonical_cell",
            "config_digest": cfg,
            "matrix_digest": mat,
            "code_digest": code,
            "n_cells": len(resolved),
            "n_arm_runs": len(resolved),
            "n_canonical_cells": len({cell.canonical_cell_key for cell in resolved}),
            "n_completed": ran + skipped,
            "n_ran": ran,
            "n_skipped": skipped,
            "n_failed": failed,
            "finalized": finalize.get("finalized"),
            "all_finalized": finalize.get("finalized") == "YES",
            "datasets": sorted({cell.dataset for cell in resolved}),
            "targets": sorted({cell.target for cell in resolved}),
            "horizons": sorted({int(cell.horizon) for cell in resolved}),
            "seeds": sorted({int(cell.seed) for cell in resolved}),
            "arms": sorted({cell.arm for cell in resolved}),
            "stage_a_shared_within_canonical_cell": True,
            "per_arm_rng_reset": True,
            "diagnostic_rng_neutral": True,
            "passive_diagnostics": True,
            "historical_frozen_reproduction_required": False,
            "historical_trend_match_required": False,
            "sita_ablation_3e4_manifest_runtime_authority": "NO",
            "sita_ablation_3e4_manifest_role": sita_ablation_3e4_manifest_runtime_role(),
            "replaces_frozen_results": "NO",
            "weight_variant": FORMAL_EVALUATION_WEIGHT_VARIANT,
            "reference_split": FORMAL_TEST_REFERENCE_SPLIT,
            "cka_layers": list(CKA_REPRESENTATION_LAYERS),
            "cka_activation_reshape_rule": CKA_ACTIVATION_RESHAPE_RULE,
            "formal_reference_result": {
                "path": str(
                    Path(FORMAL_PER_DATASET_LR_CSV).resolve()
                    if mode == FORMAL_300_MATRIX_MODE
                    else FORMAL_AE_UNIFORM_3E4_CSV.resolve()
                ),
                "digest": (
                    formal_reference_result_digest(
                        FORMAL_PER_DATASET_LR_CSV if mode == FORMAL_300_MATRIX_MODE else FORMAL_AE_UNIFORM_3E4_CSV
                    )
                    if Path(
                        FORMAL_PER_DATASET_LR_CSV if mode == FORMAL_300_MATRIX_MODE else FORMAL_AE_UNIFORM_3E4_CSV
                    ).is_file()
                    else "MISSING"
                ),
                "comparison_purpose": (
                    "formal_per_dataset_lr_ae_authority"
                    if mode == FORMAL_300_MATRIX_MODE
                    else "historical_reference_only"
                ),
                "authority_kind": (
                    "FORMAL_PER_DATASET_LR_AE"
                    if mode == FORMAL_300_MATRIX_MODE
                    else "HISTORICAL_UNIFORM_3E4"
                ),
                "blocks_execution": False,
            },
            "finalize": finalize,
        },
    )
    return {
        "run_dir": str(run_dir),
        "n_cells": len(resolved),
        "n_arm_runs": len(resolved),
        "n_canonical_cells": len({cell.canonical_cell_key for cell in resolved}),
        "n_completed": ran + skipped,
        "n_ran": ran,
        "n_skipped": skipped,
        "n_failed": failed,
        "finalized": finalize.get("finalized"),
        "outcomes": outcomes,
        "finalize": finalize,
        "cell_loop_started": "YES",
    }
