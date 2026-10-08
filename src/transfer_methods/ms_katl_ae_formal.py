"""Formal A–E execution body bound to frozen V3 Stage-A / Stage-B / metric authorities.

Real training remains unreachable unless AE_FORMAL_EXECUTION_ENABLED is True
and both CLI `--execute` and AE_FORMAL_TRAINING_UNLOCK=YES are present.
Tests may inject mock authorities with mock_execution_authorized=True.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    EXPECTED_CANONICAL_CELLS,
    EXPECTED_RUNS,
    EXTRA_TARGET_CANONICAL_CELLS,
    EXTRA_TARGET_MATRIX_KIND,
    EXTRA_TARGET_RUNS,
    FORMAL_TRAINING_UNLOCK_ENV,
    FROZEN_MATRIX_KIND,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    UNIFORM_3E4_CANONICAL_CELLS,
    UNIFORM_3E4_DATASETS,
    UNIFORM_3E4_MATRIX_KIND,
    UNIFORM_3E4_RUNS,
    UNIFORM_3E4_SEEDS,
    UNIFORM_3E4_TARGETS,
    FORMAL_PER_DATASET_LR_MATRIX_KIND,
    FORMAL_PER_DATASET_LR_RUNS,
    FORMAL_PER_DATASET_LR_CANONICAL_CELLS,
    AEConstructedArm,
    AEContractError,
    AETrainingBlocked,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_protocol_manifest_payload,
    ae_training_recipe,
    assert_d_equals_frozen_v3,
    assert_formal_training_permitted,
    build_ae_stage_a_from_shared_bundle,
    construct_all_ae_arms,
    load_ae_manifest,
    project_root_from,
    reset_ae_rng,
    run_ae_preflight,
    select_ae_trainable_variables,
)
from src.transfer_methods.ms_katl_ae_execution import (
    AE_RUN_LABEL,
    PREFLIGHT_RELPATH,
    RealArmPlan,
    RealCanonicalCellPlan,
    assert_ae_output_root,
    assert_extra_target_slice,
    assert_formal_run_root,
    enumerate_real_arm_plans,
    enumerate_real_canonical_cell_plans,
    enumerate_uniform_3e4_cell_plans,
    formal_manifest_snapshot_payload,
    planned_extra_target_run_root,
    planned_formal_run_root,
    resolve_all_formal_stage_a_input_plans,
    validate_resume_root,
    write_planned_matrix_csv,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    v3_arm_spec,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    digest_weight_arrays,
    extract_features_maybe_revin,
    prepare_ms_katl_stage_a,
    predict_ms_katl_decomposed,
    train_ms_katl_head,
)
from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle
from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics
from src.transfer_methods.ms_katl_v3_frame_assembly import WINDOW_SIZE
from src.transfer_methods.sita_mechanism_diagnostics_capture import (
    ENABLE_MECHANISM_DIAGNOSTICS_KEY,
    MECHANISM_DIAGNOSTICS_SESSION_KEY,
    finalize_stage_b_diagnostics,
    stage_b_diagnostics_train_kwargs,
)

AE_STAGE_A_AUTHORITY = prepare_ms_katl_stage_a
AE_STAGE_B_AUTHORITY = train_ms_katl_head
AE_PREDICTION_AUTHORITY = predict_ms_katl_decomposed
AE_METRIC_AUTHORITY = compute_screening_metrics
MOCK_STATUS = "MOCK_ACCEPTANCE_ONLY"
MOCK_METRIC = "MOCK_METRIC"
REAL_METRIC_KIND = "rmse_original"
AE_CONSTRUCTED_ARMS_KEY = "ae_constructed_arms"
AE_STAGE_A_VIEW_KEY = "ae_stage_a"
AE_SHARED_BUNDLE_KEY = "bundle"
AE_HEAD_DIGEST_KEY = "head_target_init_digest"
AE_EPHEMERAL_KEYS = (
    AE_CONSTRUCTED_ARMS_KEY,
    AE_STAGE_A_VIEW_KEY,
    "stage_a_artifacts",
    "head",
    "target_extractor",
    "trainable_variables",
    "revin",
    "_predict_payload",
    ENABLE_MECHANISM_DIAGNOSTICS_KEY,
    MECHANISM_DIAGNOSTICS_SESSION_KEY,
)


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


ARM_D_REASON = (
    "uniform ablation protocol uses target LR=3e-4 and is not numerically identical "
    "to the formal SITA main configuration"
)
FORMAL_HARDENING_RUNTIME_FILES = (
    "configs/ms_katl_ae_ablation_manifest.json",
    "configs/ms_katl_sita_ablation_3e4_manifest.json",
    "scripts/run_ms_katl_ae_ablation.py",
    "scripts/run_ms_katl_sita_ablation_3e4.py",
    "src/transfer_methods/ms_katl_ae_ablation.py",
    "src/transfer_methods/ms_katl_ae_execution.py",
    "src/transfer_methods/ms_katl_ae_formal.py",
    "src/transfer_methods/ms_katl_ae_live_context.py",
    "src/transfer_methods/ms_katl_lite.py",
    "src/transfer_methods/msml_tl.py",
    "src/transfer_methods/ms_katl_sita_ablation_3e4.py",
)
FORMAL_RESUME_DIGEST_KEYS = (
    "matrix_digest",
    "protocol_digest",
    "critical_runtime_digest",
    "working_tree_protocol_digest",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_output(project_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(project_root), stderr=subprocess.STDOUT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AEContractError(f"git provenance failed: {' '.join(args)}") from exc


def _critical_runtime_snapshot(project_root: Path) -> Dict[str, str]:
    root = Path(project_root)
    snapshot: Dict[str, str] = {}
    for relative in FORMAL_HARDENING_RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            raise AEContractError(f"critical runtime file missing: {path}")
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _critical_runtime_digest(project_root: Path) -> str:
    return _json_digest(_critical_runtime_snapshot(Path(project_root)))


def _working_tree_protocol_digest(project_root: Path) -> str:
    root = Path(project_root)
    head = _git_output(root, ["rev-parse", "HEAD"])
    diff = _git_output(root, ["diff", "--no-ext-diff", "--binary", "--", *FORMAL_HARDENING_RUNTIME_FILES])
    untracked = _git_output(root, ["ls-files", "--others", "--exclude-standard", "--", *FORMAL_HARDENING_RUNTIME_FILES])
    untracked_digests = {
        relative: _critical_runtime_snapshot(root)[relative]
        for relative in untracked.splitlines()
        if relative.strip()
    }
    return _json_digest(
        {
            "HEAD": head,
            "critical_file_bytes": _critical_runtime_snapshot(root),
            "critical_git_diff": diff,
            "critical_untracked_file_digests": untracked_digests,
        }
    )


def formal_protocol_digest(*, matrix_kind: str, expected_runs: int) -> str:
    uniform = str(matrix_kind).strip().upper().replace("-", "_") == UNIFORM_3E4_MATRIX_KIND
    formal_per_dataset = (
        str(matrix_kind).strip().upper().replace("-", "_")
        == FORMAL_PER_DATASET_LR_MATRIX_KIND
    )
    payload: Dict[str, Any] = {
        "matrix_kind": str(matrix_kind),
        "expected_runs": int(expected_runs),
        "ae_protocol": ae_protocol_manifest_payload(),
        "arm_d_is_formal_main_sita": False,
        "arm_d_reason": ARM_D_REASON,
    }
    if uniform:
        payload["uniform_protocol"] = {
            "datasets": list(UNIFORM_3E4_DATASETS),
            "targets": dict(UNIFORM_3E4_TARGETS),
            "seeds": list(UNIFORM_3E4_SEEDS),
            "target_lr": "3e-4",
            "source_lr": "1e-4",
            "canonical_cells": UNIFORM_3E4_CANONICAL_CELLS,
            "runs": UNIFORM_3E4_RUNS,
        }
    if formal_per_dataset:
        from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
            FORMAL_PER_DATASET_LR_DATASETS,
            FORMAL_PER_DATASET_LR_LABELS,
            FORMAL_PER_DATASET_LR_TARGETS,
            FORMAL_PER_DATASET_LR_SEEDS,
        )

        payload["formal_per_dataset_lr_protocol"] = {
            "datasets": list(FORMAL_PER_DATASET_LR_DATASETS),
            "targets": dict(FORMAL_PER_DATASET_LR_TARGETS),
            "seeds": list(FORMAL_PER_DATASET_LR_SEEDS),
            "target_learning_rates": dict(FORMAL_PER_DATASET_LR_LABELS),
            "source_lr": "1e-4",
            "canonical_cells": FORMAL_PER_DATASET_LR_CANONICAL_CELLS,
            "runs": FORMAL_PER_DATASET_LR_RUNS,
        }
    return _json_digest(payload)


def build_formal_run_provenance(
    project_root: Path,
    *,
    matrix_digest: str,
    matrix_kind: str,
    expected_runs: int,
    cells: Sequence[RealCanonicalCellPlan],
) -> Dict[str, Any]:
    root = Path(project_root)
    branch = _git_output(root, ["branch", "--show-current"])
    head = _git_output(root, ["rev-parse", "HEAD"])
    targets: Dict[str, str] = {}
    for cell in cells:
        targets[str(cell.dataset)] = str(cell.target)
    normalized_kind = str(matrix_kind).strip().upper().replace("-", "_")
    uniform = normalized_kind == UNIFORM_3E4_MATRIX_KIND
    formal_per_dataset = normalized_kind == FORMAL_PER_DATASET_LR_MATRIX_KIND
    if uniform:
        targets = dict(UNIFORM_3E4_TARGETS)
    return {
        "paper_method_name": "SITA",
        "ablation_protocol_name": (
            "FORMAL_AE_PER_DATASET_TARGET_LR"
            if formal_per_dataset
            else "CORRECTED_AE_UNIFORM_3E4"
        ),
        "matrix_kind": str(matrix_kind),
        "branch": branch,
        "HEAD": head,
        "runtime_commit": head,
        "matrix_digest": str(matrix_digest),
        "protocol_digest": formal_protocol_digest(matrix_kind=matrix_kind, expected_runs=expected_runs),
        "critical_runtime_digest": _critical_runtime_digest(root),
        "working_tree_protocol_digest": _working_tree_protocol_digest(root),
        "dataset_target_map": targets,
        "horizons": sorted({int(cell.horizon) for cell in cells}),
        "seeds": sorted({int(cell.seed) for cell in cells}),
        "arms": list(AE_ARMS),
        "source_lr": "1e-4",
        "uniform_target_lr": "3e-4" if uniform else None,
        "stage_a_rng_contract": {
            "stage_a_seed_equals_cell_seed": True,
            "reset_once_at_entry": True,
            "reset_before_source_model_creation": True,
            "per_source_reseed": False,
            "sequential_source_stream": True,
            "historical_source_weight_reuse": False,
        },
        "historical_source_weight_reuse": False,
        "ARM_D_IS_FORMAL_MAIN_SITA": False,
        "ARM_D_REASON": ARM_D_REASON,
        "created_at": _utc_now(),
        "expected_runs": int(expected_runs),
        "canonical_stage_a_keys": sorted({str(cell.stage_a_key) for cell in cells}),
    }
    if formal_per_dataset:
        from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
            FORMAL_PER_DATASET_LR_LABELS,
        )

        payload["target_learning_rates"] = dict(FORMAL_PER_DATASET_LR_LABELS)


def _attempt_cell_dir(run_root: Path, cell: RealCanonicalCellPlan) -> Path:
    safe_key = str(cell.stage_a_key).replace("|", "__").replace("/", "-")
    return Path(run_root) / "cells" / safe_key


def _attempt_dir(run_root: Path, cell: RealCanonicalCellPlan, number: int) -> Path:
    return _attempt_cell_dir(run_root, cell) / f"attempt_{int(number):03d}"


def _attempt_numbers(cell_dir: Path) -> List[int]:
    numbers: List[int] = []
    for candidate in Path(cell_dir).glob("attempt_*"):
        try:
            numbers.append(int(candidate.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(numbers)


def _attempt_payload_base(
    cell: RealCanonicalCellPlan,
    attempt_id: str,
    digests: Mapping[str, str],
    *,
    branch: str,
    head: str,
) -> Dict[str, Any]:
    return {
        "dataset": cell.dataset,
        "target": cell.target,
        "horizon": int(cell.horizon),
        "seed": int(cell.seed),
        "cell_key": str(cell.stage_a_key),
        "canonical_stage_a_key": str(cell.stage_a_key),
        "stage_a_seed": int(cell.seed),
        "attempt_id": attempt_id,
        "runtime_commit": head,
        "branch": branch,
        **dict(digests),
    }


def _assert_attempt_digests(attempt: Path, digests: Mapping[str, str]) -> None:
    for name in ("status.json", "result.json", "provenance.json"):
        path = Path(attempt) / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in FORMAL_RESUME_DIGEST_KEYS:
            if payload.get(key) != digests.get(key):
                raise AEContractError(f"resume digest mismatch: {key}")


def _attempt_is_complete(attempt: Path, cell: RealCanonicalCellPlan, digests: Mapping[str, str]) -> bool:
    status_path = Path(attempt) / "status.json"
    result_path = Path(attempt) / "result.json"
    if not status_path.is_file() or not result_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        status.get("status") == "COMPLETE"
        and result.get("status") == "COMPLETE"
        and result.get("cell_key") == cell.stage_a_key
        and int(result.get("stage_a_seed")) == int(cell.seed)
        and all(result.get(key) == value for key, value in digests.items())
    )


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def prepare_formal_cell_attempt(
    run_root: Path,
    cell: RealCanonicalCellPlan,
    *,
    resume: bool,
    digests: Mapping[str, str],
    branch: str,
    head: str,
) -> Tuple[str, Optional[Path]]:
    cell_dir = _attempt_cell_dir(Path(run_root), cell)
    latest_number = _attempt_numbers(cell_dir)[-1] if _attempt_numbers(cell_dir) else None
    latest = _attempt_dir(run_root, cell, latest_number) if latest_number is not None else None
    if latest is not None and resume:
        _assert_attempt_digests(latest, digests)
        if _attempt_is_complete(latest, cell, digests):
            return "SKIPPED_COMPLETE", latest
    if latest is not None and not resume:
        raise AEContractError(f"existing cell attempts require explicit resume: {cell.stage_a_key}")
    if latest is not None:
        status_path = latest / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if str(status.get("status", "")).upper() == "RUNNING":
                if _pid_alive(status.get("pid")):
                    raise AEContractError(f"resume active attempt: {cell.stage_a_key}")
                status.update(
                    {
                        "status": "INTERRUPTED",
                        "interrupted_at": _utc_now(),
                        "interrupted_reason": "resume_observed_dead_process",
                    }
                )
                atomic_write_json(status_path, status)
                atomic_write_json(
                    latest / "result.json",
                    {
                        **_attempt_payload_base(cell, latest.name, digests, branch=branch, head=head),
                        "status": "INTERRUPTED",
                    },
                )
    next_number = (latest_number + 1) if latest_number is not None else 1
    attempt = _attempt_dir(run_root, cell, next_number)
    attempt.mkdir(parents=True, exist_ok=False)
    base = _attempt_payload_base(cell, attempt.name, digests, branch=branch, head=head)
    atomic_write_json(
        attempt / "status.json",
        {**base, "status": "RUNNING", "pid": os.getpid(), "start_time": _utc_now()},
    )
    atomic_write_json(attempt / "result.json", {**base, "status": "RUNNING"})
    atomic_write_json(attempt / "provenance.json", {**base, "status": "RUNNING"})
    return "RUN", attempt


def finalize_formal_cell_attempt(
    attempt: Path,
    *,
    status: str,
    result: Optional[Mapping[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    status_path = Path(attempt) / "status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload.update({"status": str(status), "end_time": _utc_now()})
    if error is not None:
        status_payload["exception_type"] = type(error).__name__
        status_payload["exception_message"] = str(error)
    atomic_write_json(status_path, status_payload)
    provenance_path = Path(attempt) / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    result_payload = {**provenance, **dict(result or {})}
    result_payload.setdefault("status", str(status))
    if error is not None:
        result_payload["exception_type"] = type(error).__name__
        result_payload["exception_message"] = str(error)
    atomic_write_json(Path(attempt) / "result.json", result_payload)


def validate_formal_resume_provenance(run_root: Path, expected: Mapping[str, Any]) -> None:
    manifest_path = Path(run_root) / "manifest_snapshot.json"
    if not manifest_path.is_file():
        raise AEContractError("resume root missing manifest_snapshot.json")
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in FORMAL_RESUME_DIGEST_KEYS:
        if snapshot.get(key) != expected.get(key):
            raise AEContractError(f"resume digest mismatch: {key}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic JSON publish: temp write + fsync + os.replace (run_artifacts convention)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

@dataclass
class FormalStageAView:
    stage_a_key: str
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    source_weight_digests: List[str]
    fused_params_digest: str
    fused_target_digest: str
    best_source_identity: Tuple[object, ...]
    best_source_conv_digest: str
    target_split_digest: str
    requested_k: int = 3
    effective_k: int = 3
    stage_a_seed: Optional[int] = None
    source_selection_digest: Optional[str] = None
    stage_a_digest: Optional[str] = None


@dataclass
class FormalArmRuntime:
    arm: str
    spec: Any
    init_mode: str
    source_init_consumed: bool
    stage_a_shared_for_pairing: bool
    head_target_init_digest: str
    target_initial_conv_digest: str
    fused_init_digest: str
    best_source_identity: Tuple[object, ...]
    best_source_snapshot_digest: str
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    selected_lr: float
    recipe_digest: str
    target_split_digest: str
    parent_method: str
    parent_v3_architecture_digest: str
    prediction_path_parameter_count: int
    trainable_prediction_path_parameter_count: int
    frozen_prediction_path_parameter_count: int
    status: str = MOCK_STATUS
    error: str = ""
    rmse_original: Optional[float] = None
    smape_original: Optional[float] = None
    mae_original: Optional[float] = None
    metric_kind: Optional[str] = None
    source_model_digests: List[str] = field(default_factory=list)
    source_selection_digest: Optional[str] = None
    stage_a_digest: Optional[str] = None
    fused_params_digest: Optional[str] = None
    final_formal_model_digest: Optional[str] = None
    formal_weight_variant: Optional[str] = None
    prediction_digest: Optional[str] = None
    prediction_digest_space: Optional[str] = None
    n_test: Optional[int] = None


@dataclass
class FormalCallLedger:
    context_prep: List[str] = field(default_factory=list)
    stage_a: List[str] = field(default_factory=list)
    stage_b: List[Tuple[str, str]] = field(default_factory=list)
    predict: List[Tuple[str, str]] = field(default_factory=list)
    metric: List[Tuple[str, str]] = field(default_factory=list)
    bundle_validations: List[str] = field(default_factory=list)
    publish: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        arms = {arm: sum(1 for _, item in self.stage_b if item == arm) for arm in AE_ARMS}
        return {
            "CONTEXT_PREP_CALLS": len(self.context_prep),
            "STAGE_A_CALLS": len(self.stage_a),
            "STAGE_B_CALLS": len(self.stage_b),
            "PREDICT_CALLS": len(self.predict),
            "METRIC_CALLS": len(self.metric),
            "BUNDLE_VALIDATIONS": len(self.bundle_validations),
            "PUBLISH_ATTEMPTS": len(self.publish),
            "A_CALLS": arms["A"],
            "B_CALLS": arms["B"],
            "C_CALLS": arms["C"],
            "D_CALLS": arms["D"],
            "E_CALLS": arms["E"],
        }


@dataclass
class AEFormalAuthorities:
    prepare_context: Callable[[RealCanonicalCellPlan], Dict[str, Any]]
    prepare_stage_a: Callable[[RealCanonicalCellPlan, Dict[str, Any]], FormalStageAView]
    make_common_head: Callable[..., str]
    train_stage_b: Callable[..., FormalArmRuntime]
    predict: Callable[..., FormalArmRuntime]
    metric: Callable[..., FormalArmRuntime]
    publish: Callable[[RealCanonicalCellPlan, Dict[str, FormalArmRuntime], str, Path], None]
    reset_rng: Callable[[int], None]


@dataclass
class FormalArmEphemeralRuntime:
    """In-memory per-arm construction objects. Never serialized into cell bundles."""

    constructed: AEConstructedArm
    trainable_variables: List[Any]


def _context_arms(context: Mapping[str, Any]) -> Dict[str, AEConstructedArm]:
    arms = context.get(AE_CONSTRUCTED_ARMS_KEY)
    if not isinstance(arms, dict) or set(arms) != set(AE_ARMS):
        raise AEContractError("formal context missing constructed A–E arm runtimes")
    return arms  # type: ignore[return-value]


def _bind_arm_trainables(context: Dict[str, Any], arm: str) -> FormalArmEphemeralRuntime:
    constructed = _context_arms(context)[arm]
    trainable = select_ae_trainable_variables(
        constructed.head,
        constructed.target_extractor,
        constructed.spec,
        constructed.revin,
    )
    if not trainable:
        raise AEContractError(f"arm {arm} trainable_variables empty after construction")
    ephemeral = FormalArmEphemeralRuntime(constructed=constructed, trainable_variables=trainable)
    context["head"] = constructed.head
    context["target_extractor"] = constructed.target_extractor
    context["trainable_variables"] = trainable
    context["revin"] = constructed.revin
    context["_arm_ephemeral"] = ephemeral
    return ephemeral


def materialize_formal_ae_arms(
    context: Dict[str, Any],
    cell: RealCanonicalCellPlan,
    *,
    stage_a_artifacts: Any,
) -> Tuple[str, Dict[str, AEConstructedArm]]:
    """Build shared Stage-B bundle + A–E arms once per canonical cell (construction only)."""
    live = context.get("live")
    kwargs = dict(context.get("stage_a_kwargs") or {})
    if live is None and not kwargs:
        raise AEContractError("materialize_formal_ae_arms requires live context or stage_a_kwargs")
    target_df = getattr(live, "target_df", None) if live is not None else kwargs.get("target_df")
    feature_cols = (
        list(getattr(live, "feature_cols", ()) or ())
        if live is not None
        else list(kwargs.get("feature_cols") or [])
    )
    if target_df is None or not feature_cols:
        raise AEContractError("shared target bundle requires target_df and feature_cols")
    shared = prepare_shared_target_bundle(
        stage_a_artifacts,
        target_df,
        feature_cols,
        horizon=int(cell.horizon),
        window_size=int(kwargs.get("window_size") or WINDOW_SIZE),
        learning_rate=float(kwargs.get("learning_rate") or cell.selected_lr),
        seed=int(cell.seed),
        transformation_reuse_context=kwargs.get("transformation_reuse_context"),
    )
    ae_stage = build_ae_stage_a_from_shared_bundle(shared)
    arms = construct_all_ae_arms(
        ae_stage,
        seed=int(cell.seed),
        selected_lr=float(cell.selected_lr),
        recipe=ae_training_recipe(),
    )
    head_digest = str(arms["A"].head_target_init_digest)
    context[AE_SHARED_BUNDLE_KEY] = shared
    context[AE_STAGE_A_VIEW_KEY] = ae_stage
    context[AE_CONSTRUCTED_ARMS_KEY] = arms
    context[AE_HEAD_DIGEST_KEY] = head_digest
    context["stage_a_artifacts"] = stage_a_artifacts
    return head_digest, arms


def assert_no_old_v3_result_injection(source_text: str) -> str:
    start = source_text.find("def assert_no_old_v3_result_injection")
    rest = source_text[start:] if start >= 0 else source_text
    end = rest.find("\ndef ", 1)
    runtime_source = (source_text[:start] + rest[end:]) if start >= 0 and end > 0 else source_text
    forbidden = (
        "v3_screening_target_best_lr",
        "screening_target_best_lr.csv",
        "load_valid_ok_cells",
        "DEFAULT_V3_RUN_DIRNAME",
        "wilcoxon",
        "find latest",
        "latest output",
        "fused_frozen_backbone",
    )
    lowered = runtime_source.lower()
    for token in forbidden:
        if token.lower() in lowered:
            raise AEContractError(f"D_OLD_RESULT_REUSE forbidden token present: {token}")
    return "NO"


def planned_matrix_digest(arm_plans: Sequence[RealArmPlan]) -> str:
    payload = [
        {
            "dataset": item.cell.dataset,
            "target": item.cell.target,
            "horizon": item.cell.horizon,
            "seed": item.cell.seed,
            "scenario": item.cell.scenario,
            "arm": item.arm,
            "lr": item.cell.selected_lr_label,
            "init_mode": item.init_mode,
        }
        for item in arm_plans
    ]
    return _json_digest({"runs": payload, "n": len(payload)})


_MANIFEST_IDENTITY_KEYS = (
    "protocol_version",
    "parent_v3_architecture_digest",
    "expected_runs",
    "planned_matrix_digest",
    "matrix_digest",
    "protocol_digest",
    "critical_runtime_digest",
    "working_tree_protocol_digest",
    "runtime_commit",
)


def bootstrap_fresh_formal_run_metadata(
    run_root: Path,
    arm_plans: Sequence[RealArmPlan],
    planned_matrix_digest_value: str,
    *,
    expected_runs: int = EXPECTED_RUNS,
    extra_snapshot: Optional[Mapping[str, Any]] = None,
) -> None:
    """Initialize immutable formal run metadata before any Stage-A / Stage-B training.

    Fresh only: create once. Existing identity mismatch fails closed; matching
    identity is left untouched (never rewritten).
    """
    root = assert_ae_output_root(Path(run_root))
    snapshot_kind = FROZEN_MATRIX_KIND
    if extra_snapshot and extra_snapshot.get("matrix_kind"):
        snapshot_kind = str(extra_snapshot["matrix_kind"])
    elif expected_runs == EXTRA_TARGET_RUNS:
        snapshot_kind = EXTRA_TARGET_MATRIX_KIND
    elif expected_runs == UNIFORM_3E4_RUNS:
        snapshot_kind = UNIFORM_3E4_MATRIX_KIND
    elif expected_runs != EXPECTED_RUNS:
        raise AEContractError(f"bootstrap expected_runs={expected_runs} has no matrix kind")
    assert_formal_run_root(root, snapshot_kind)
    if "results" in root.parts and "preflight" not in root.parts and "audits" not in root.parts:
        raise AEContractError("refusing formal bootstrap under results/")
    if "preflight" in root.parts and "formal_mock" in root.as_posix():
        raise AEContractError("refusing formal bootstrap under preflight mock root")

    root.mkdir(parents=True, exist_ok=True)
    (root / "cell_bundles").mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(parents=True, exist_ok=True)

    snapshot = formal_manifest_snapshot_payload(planned_matrix_digest_value)
    snapshot["expected_runs"] = int(expected_runs)
    snapshot["expected_canonical_cells"] = int(expected_runs) // len(AE_ARMS)
    snapshot["matrix_kind"] = snapshot_kind
    if extra_snapshot:
        snapshot.update(dict(extra_snapshot))
    if snapshot.get("protocol_version") != PROTOCOL_VERSION:
        raise AEContractError("bootstrap protocol_version drifted")
    if snapshot.get("expected_runs") != int(expected_runs):
        raise AEContractError("bootstrap expected_runs drifted")
    if snapshot.get("planned_matrix_digest") != planned_matrix_digest_value:
        raise AEContractError("bootstrap planned_matrix_digest drifted")

    manifest_path = root / "manifest_snapshot.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in _MANIFEST_IDENTITY_KEYS:
            if existing.get(key) != snapshot.get(key):
                raise AEContractError(f"fresh run manifest identity mismatch: {key}")
    else:
        atomic_write_json(manifest_path, snapshot)

    plan_path = root / "plans" / "planned_matrix.csv"
    if plan_path.is_file():
        with plan_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(expected_runs):
            raise AEContractError(
                f"existing planned matrix rows={len(rows)} != {expected_runs}"
            )
    else:
        write_planned_matrix_csv(plan_path, arm_plans, expected_runs=int(expected_runs))


def serialize_formal_arm_runtime(runtime: FormalArmRuntime) -> Dict[str, Any]:
    """JSON-safe FormalArmRuntime provenance (no TF/numpy payloads)."""
    source_identities = [list(item) for item in runtime.source_identities]
    source_model_digests = list(runtime.source_model_digests)
    if source_model_digests and len(source_model_digests) != len(source_identities):
        raise AEContractError(
            "source_model_digests must align one-to-one with source_identities"
        )
    return {
        "arm": runtime.arm,
        "status": runtime.status,
        "error": runtime.error,
        "metric_kind": runtime.metric_kind,
        "rmse_original": runtime.rmse_original,
        "smape_original": runtime.smape_original,
        "mae_original": runtime.mae_original,
        "init_mode": runtime.init_mode,
        "source_init_consumed": runtime.source_init_consumed,
        "stage_a_shared_for_pairing": runtime.stage_a_shared_for_pairing,
        "head_target_init_digest": runtime.head_target_init_digest,
        "target_initial_conv_digest": runtime.target_initial_conv_digest,
        "fused_init_digest": runtime.fused_init_digest,
        "fused_params_digest": runtime.fused_params_digest,
        "best_source_identity": list(runtime.best_source_identity),
        "best_source_snapshot_digest": runtime.best_source_snapshot_digest,
        "source_identities": source_identities,
        "source_model_digests": source_model_digests,
        "source_distances": list(runtime.source_distances),
        "source_weights": list(runtime.source_weights),
        "source_selection_digest": runtime.source_selection_digest,
        "stage_a_digest": runtime.stage_a_digest,
        "selected_lr": runtime.selected_lr,
        "recipe_digest": runtime.recipe_digest,
        "target_split_digest": runtime.target_split_digest,
        "parent_method": runtime.parent_method,
        "parent_v3_architecture_digest": runtime.parent_v3_architecture_digest,
        "prediction_path_parameter_count": runtime.prediction_path_parameter_count,
        "trainable_prediction_path_parameter_count": runtime.trainable_prediction_path_parameter_count,
        "frozen_prediction_path_parameter_count": runtime.frozen_prediction_path_parameter_count,
        "final_formal_model_digest": runtime.final_formal_model_digest,
        "formal_weight_variant": runtime.formal_weight_variant,
        "prediction_digest": runtime.prediction_digest,
        "prediction_digest_space": runtime.prediction_digest_space,
        "n_test": runtime.n_test,
    }


def formal_inference_model_digest(constructed: AEConstructedArm) -> str:
    """Digest the exact post-training weights used by the formal inference path."""
    weights = list(constructed.target_cnn.get_weights())
    weights.extend(constructed.head.get_weights())
    if constructed.revin is not None:
        weights.extend(constructed.revin.get_weights())
    return digest_weight_arrays(weights)


def cell_bundle_path(run_root: Path, stage_a_key: str) -> Path:
    root = Path(run_root)
    if root.name.startswith("attempt_"):
        return root / "result.json"
    return root / "cell_bundles" / f"{stage_a_key.replace('/', '-')}.json"


def _is_hardened_attempt_root(path: Path) -> bool:
    root = Path(path)
    if not root.name.startswith("attempt_"):
        return False
    formal_root = root.parent.parent.parent
    return (formal_root / "manifest_snapshot.json").is_file() and (formal_root / "plans").is_dir()


def stage_a_view_from_v3(
    artifacts: Any,
    cell: RealCanonicalCellPlan,
    fused_target_digest: str = "",
    *,
    digest_source: Optional[Any] = None,
) -> FormalStageAView:
    identities = [tuple(item) for item in artifacts.source_identities]
    existing_stage_a = digest_source if digest_source is not None else artifacts
    return FormalStageAView(
        stage_a_key=cell.stage_a_key,
        source_identities=identities,
        source_distances=[float(x) for x in artifacts.source_distances],
        source_weights=[float(x) for x in artifacts.source_weights],
        source_weight_digests=list(artifacts.source_weight_digests),
        fused_params_digest=str(artifacts.fused_params_digest),
        fused_target_digest=str(fused_target_digest or artifacts.fused_params_digest),
        best_source_identity=identities[0],
        best_source_conv_digest=str(artifacts.source_weight_digests[0]),
        target_split_digest=cell.target_split_digest,
        requested_k=3,
        effective_k=int(artifacts.effective_k),
        stage_a_seed=int(getattr(artifacts, "stage_a_seed", cell.seed) or cell.seed),
        source_selection_digest=str(getattr(existing_stage_a, "source_selection_digest", "")) or None,
        stage_a_digest=str(getattr(existing_stage_a, "stage_a_digest", "")) or None,
    )


def _require_enabled_real_path() -> None:
    if not AE_FORMAL_EXECUTION_ENABLED:
        raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")


def default_v3_prepare_context(
    cell: RealCanonicalCellPlan,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    _require_enabled_real_path()
    from src.transfer_methods.ms_katl_ae_live_context import build_live_ae_cell_context

    live = build_live_ae_cell_context(cell, project_root=project_root)
    if live.stage_a_key != cell.stage_a_key or live.target_bundle_key != cell.stage_a_key:
        raise AEContractError(
            f"live context key mismatch: stage_a={live.stage_a_key} bundle={live.target_bundle_key}"
        )
    return live.as_formal_context()


def default_v3_prepare_stage_a(cell: RealCanonicalCellPlan, context: Mapping[str, Any]) -> FormalStageAView:
    _require_enabled_real_path()
    mutable = dict(context) if not isinstance(context, dict) else context
    live = mutable.get("live")
    if live is not None and str(getattr(live, "stage_a_key", "")) != cell.stage_a_key:
        raise AEContractError("Stage A cell does not match live context")
    kwargs = dict(mutable.get("stage_a_kwargs") or {})
    if not kwargs:
        raise AEContractError("missing stage_a_kwargs for prepare_ms_katl_stage_a")
    if kwargs.get("seed") is None:
        raise AEContractError("missing stage_a seed for prepare_ms_katl_stage_a")
    if int(kwargs["seed"]) != int(cell.seed):
        raise AEContractError(
            f"stage_a seed {kwargs['seed']} != cell.seed {cell.seed} for {cell.stage_a_key}"
        )
    artifacts = AE_STAGE_A_AUTHORITY(**kwargs)
    head_digest, _arms = materialize_formal_ae_arms(mutable, cell, stage_a_artifacts=artifacts)
    shared = mutable[AE_SHARED_BUNDLE_KEY]
    fused_digest = str(getattr(shared, "fused_target_digest", "") or "")
    view = stage_a_view_from_v3(
        artifacts,
        cell,
        fused_target_digest=fused_digest,
        digest_source=mutable.get(AE_STAGE_A_VIEW_KEY),
    )
    mutable[AE_HEAD_DIGEST_KEY] = head_digest
    if not isinstance(context, dict):
        raise AEContractError("formal Stage-A context must be a mutable dict")
    return view


def default_v3_make_common_head(
    cell: RealCanonicalCellPlan,
    stage_a: FormalStageAView,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    _require_enabled_real_path()
    _ = (cell, stage_a)
    if context is None or AE_HEAD_DIGEST_KEY not in context:
        raise AEContractError("common head digest requires prior A–E arm materialization")
    arms = _context_arms(context)
    digests = {arm: arms[arm].head_target_init_digest for arm in AE_ARMS}
    if len(set(digests.values())) != 1:
        raise AEContractError(f"A–E head_target_init_digest diverged: {digests}")
    return str(context[AE_HEAD_DIGEST_KEY])


def default_v3_train_stage_b(
    cell: RealCanonicalCellPlan,
    arm: str,
    stage_a: FormalStageAView,
    head_digest: str,
    context: Mapping[str, Any],
) -> FormalArmRuntime:
    _require_enabled_real_path()
    if not isinstance(context, dict):
        raise AEContractError("Stage-B requires mutable formal context")
    spec = ae_arm_spec(arm)
    if arm == "D":
        _ = v3_arm_spec()
        _ = v3_architecture()
        _ = v3_training_recipe()
    ephemeral = _bind_arm_trainables(context, arm)
    constructed = ephemeral.constructed
    bundle = context.get(AE_SHARED_BUNDLE_KEY)
    if bundle is None:
        raise AEContractError("Stage-B V3 authority requires shared bundle and trainable variables")
    if constructed.head_target_init_digest != head_digest:
        raise AEContractError("Stage-B arm head digest drifted from common snapshot")
    opt = ae_optimizer_contract()
    recipe = ae_training_recipe()
    train_kwargs = dict(recipe)
    train_kwargs.update(
        stage_b_diagnostics_train_kwargs(
            context,
            cell=cell,
            arm=arm,
            stage_a=stage_a,
            constructed=constructed,
            bundle=bundle,
        )
    )
    AE_STAGE_B_AUTHORITY(
        constructed.head,
        x_train=bundle.X_train,
        f_t_train=bundle.f_t["train"],
        y_train=bundle.y_train,
        f_s_pools=bundle.f_s_pools,
        knn_prior=bundle.stage_a.knn_prior,
        target_extractor=constructed.target_extractor,
        epochs=int(opt["epochs"]),
        batch_size=int(opt["batch_size"]),
        learning_rate=float(cell.selected_lr),
        seed=int(cell.seed),
        use_adversarial=False,
        trainable_variables=ephemeral.trainable_variables,
        revin=constructed.revin,
        # 初始化锚定正则：仅 C 臂（FUSED_TRAINABLE）使用，保护融合初始化先验
        # 默认 0.0（从 context 或全局配置读取；不传时退化为原有行为）
        init_anchor_lambda=float(context.get("ae_arm_c_init_anchor_lambda", 0.0)) if arm == "C" else 0.0,
        **train_kwargs,
    )
    finalize_stage_b_diagnostics(context, arm=arm, constructed=constructed)
    return FormalArmRuntime(
        arm=arm,
        spec=spec,
        init_mode=spec.init_mode,
        source_init_consumed=spec.init_mode != "native",
        stage_a_shared_for_pairing=True,
        head_target_init_digest=head_digest,
        target_initial_conv_digest=str(constructed.target_initial_conv_digest),
        fused_init_digest=stage_a.fused_target_digest,
        fused_params_digest=stage_a.fused_params_digest,
        best_source_identity=stage_a.best_source_identity,
        best_source_snapshot_digest=stage_a.best_source_conv_digest,
        source_identities=list(stage_a.source_identities),
        source_model_digests=list(stage_a.source_weight_digests),
        source_distances=list(stage_a.source_distances),
        source_weights=list(stage_a.source_weights),
        source_selection_digest=stage_a.source_selection_digest,
        stage_a_digest=stage_a.stage_a_digest,
        selected_lr=float(cell.selected_lr),
        recipe_digest=_json_digest(ae_training_recipe()),
        target_split_digest=stage_a.target_split_digest,
        parent_method=PARENT_METHOD,
        parent_v3_architecture_digest=v3_architecture_digest(),
        prediction_path_parameter_count=int(constructed.prediction_path_parameter_count),
        trainable_prediction_path_parameter_count=int(
            constructed.trainable_prediction_path_parameter_count
        ),
        frozen_prediction_path_parameter_count=int(constructed.frozen_prediction_path_parameter_count),
        status="OK",
    )


def default_v3_predict(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
    _require_enabled_real_path()
    if context is None or not isinstance(context, dict):
        raise AEContractError("prediction requires mutable formal context with arm runtimes")
    ephemeral = _bind_arm_trainables(context, runtime.arm)
    constructed = ephemeral.constructed
    bundle = context[AE_SHARED_BUNDLE_KEY]
    z_t_test, test_mean, test_stdev = extract_features_maybe_revin(
        constructed.target_extractor,
        bundle.X_test,
        revin=constructed.revin,
        training=False,
    )
    decomposed = AE_PREDICTION_AUTHORITY(
        constructed.head,
        z_t_test,
        bundle.f_t["test"],
        bundle.stage_a.knn_prior,
    )
    y_pred = decomposed["y_pred"]
    if runtime.spec.revin_enabled:
        if constructed.revin is None:
            raise AEContractError("arm D prediction requires RevIN inverse")
        mean_np = test_mean.numpy() if hasattr(test_mean, "numpy") else test_mean
        stdev_np = test_stdev.numpy() if hasattr(test_stdev, "numpy") else test_stdev
        y_pred = constructed.revin.denormalize_numpy(
            y_pred,
            mean_np,
            stdev_np,
            include_mean=True,
        )
    runtime.final_formal_model_digest = formal_inference_model_digest(constructed)
    runtime.formal_weight_variant = "EMA" if ae_training_recipe()["use_ema"] else "RAW"
    payload = {
        "head": constructed.head,
        "target_extractor": constructed.target_extractor,
        "revin": constructed.revin,
        "z_t_test": z_t_test,
        "f_t_test": bundle.f_t["test"],
        "knn_prior": bundle.stage_a.knn_prior,
        "y_pred": y_pred,
        "y_true": bundle.y_test,
        "target_scaler": bundle.target_scaler,
        "feature_columns": list(bundle.target_feature_columns),
        "test_mean": test_mean,
        "test_stdev": test_stdev,
    }
    runtime.metric_kind = "pending_v3_metric"
    runtime._predict_payload = payload  # type: ignore[attr-defined]
    _ = cell
    return runtime


def default_v3_metric(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
    _require_enabled_real_path()
    payload = getattr(runtime, "_predict_payload", None)
    if not payload:
        raise AEContractError("metric authority requires prediction payload")
    metrics = AE_METRIC_AUTHORITY(
        payload["y_true"],
        payload["y_pred"],
        target_scaler=payload["target_scaler"],
        feature_columns=payload["feature_columns"],
    )
    runtime.rmse_original = float(metrics["rmse_original"])
    runtime.smape_original = float(metrics["smape_original"])
    prediction_original = metrics.get("y_pred_original")
    if prediction_original is None:
        raise AEContractError("metric authority did not return original-space predictions")
    runtime.prediction_digest = digest_weight_arrays([prediction_original])
    runtime.prediction_digest_space = str(metrics.get("metric_space") or "original_sales_space")
    runtime.n_test = int(prediction_original.size)
    if "y_true_original" in metrics and "y_pred_original" in metrics:
        import numpy as np

        true_o = np.asarray(metrics["y_true_original"], dtype=np.float64).reshape(-1)
        pred_o = np.asarray(metrics["y_pred_original"], dtype=np.float64).reshape(-1)
        runtime.mae_original = float(np.mean(np.abs(true_o - pred_o)))
    runtime.metric_kind = REAL_METRIC_KIND
    runtime.status = "OK"
    _ = (cell, context)
    return runtime


def _validate_real_complete_arm_evidence(runtime: FormalArmRuntime) -> None:
    """Fail closed on publication-only evidence required for formal C/D arms."""
    source_identities = runtime.source_identities
    source_model_digests = runtime.source_model_digests
    if (
        not isinstance(source_identities, (list, tuple))
        or len(source_identities) != 3
        or not isinstance(source_model_digests, (list, tuple))
        or len(source_model_digests) != len(source_identities)
        or any(
            not isinstance(digest, str) or not digest.strip()
            for digest in source_model_digests
        )
    ):
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "source_model_digests must contain exactly 3 non-empty digests "
            "aligned with source_identities"
        )

    if (
        not isinstance(runtime.source_selection_digest, str)
        or not runtime.source_selection_digest.strip()
    ):
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "source_selection_digest is required"
        )

    if not isinstance(runtime.stage_a_digest, str) or not runtime.stage_a_digest.strip():
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "stage_a_digest is required"
        )

    if (
        not isinstance(runtime.final_formal_model_digest, str)
        or not runtime.final_formal_model_digest.strip()
    ):
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "final_formal_model_digest is required"
        )

    if (
        not isinstance(runtime.prediction_digest, str)
        or not runtime.prediction_digest.strip()
    ):
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "prediction_digest is required"
        )

    if isinstance(runtime.n_test, bool) or not isinstance(runtime.n_test, int) or runtime.n_test <= 0:
        raise AEContractError(
            f"REAL_COMPLETE_EVIDENCE_FAILED arm={runtime.arm}: "
            "n_test must be a positive integer"
        )


def default_v3_publish(
    cell: RealCanonicalCellPlan,
    runtimes: Mapping[str, FormalArmRuntime],
    pair_status: str,
    output_root: Path,
) -> None:
    _require_enabled_real_path()
    root = assert_ae_output_root(Path(output_root))
    from src.transfer_methods.ms_katl_ae_execution import _formal_output_label_ok

    if not _formal_output_label_ok(root) and not _is_hardened_attempt_root(root):
        raise AEContractError("real publisher requires MS_KATL_AE_CORE_ABLATION or sita_ablation_3e4 run root")
    if "results" in root.parts and "preflight" not in root.parts and "audits" not in root.parts:
        raise AEContractError("real publisher refuses results/ roots")
    if "preflight" in root.parts:
        raise AEContractError("real publisher refuses preflight mock roots")
    if not set(runtimes).issubset(set(AE_ARMS)) or not runtimes:
        raise AEContractError("real publisher requires a non-empty A–E runtime subset")
    if pair_status == "PAIR_COMPLETE" and set(runtimes) != set(AE_ARMS):
        raise AEContractError("PAIR_COMPLETE requires full A–E runtime bundle")
    if pair_status in {"PAIR_COMPLETE", "SINGLE_ARM_COMPLETE"}:
        for arm in ("C", "D"):
            runtime = runtimes.get(arm)
            if runtime is not None:
                _validate_real_complete_arm_evidence(runtime)

    bundle_dir = root / "cell_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "FORMAL",
        "pair_status": pair_status,
        "stage_a_key": cell.stage_a_key,
        "dataset": cell.dataset,
        "target": cell.target,
        "horizon": int(cell.horizon),
        "seed": int(cell.seed),
        "stage_a_seed": int(cell.seed),
        "scenario": cell.scenario,
        "selected_lr": float(cell.selected_lr),
        "arms": {arm: serialize_formal_arm_runtime(runtimes[arm]) for arm in runtimes},
    }
    if pair_status == "PAIR_COMPLETE":
        for arm in AE_ARMS:
            arm_payload = payload["arms"][arm]
            if arm_payload.get("status") == MOCK_STATUS or arm_payload.get("metric_kind") == MOCK_METRIC:
                raise AEContractError("real publisher forbids MOCK markers on PAIR_COMPLETE")
    atomic_write_json(cell_bundle_path(root, cell.stage_a_key), payload)

def default_formal_authorities() -> AEFormalAuthorities:
    return AEFormalAuthorities(
        prepare_context=default_v3_prepare_context,
        prepare_stage_a=default_v3_prepare_stage_a,
        make_common_head=default_v3_make_common_head,
        train_stage_b=default_v3_train_stage_b,
        predict=default_v3_predict,
        metric=default_v3_metric,
        publish=default_v3_publish,
        reset_rng=reset_ae_rng,
    )


def assert_authorities_are_frozen_v3() -> Dict[str, str]:
    if AE_STAGE_A_AUTHORITY is not prepare_ms_katl_stage_a:
        raise AEContractError("AE_STAGE_A_AUTHORITY drifted from prepare_ms_katl_stage_a")
    if AE_STAGE_B_AUTHORITY is not train_ms_katl_head:
        raise AEContractError("AE_STAGE_B_AUTHORITY drifted from train_ms_katl_head")
    if AE_PREDICTION_AUTHORITY is not predict_ms_katl_decomposed:
        raise AEContractError("AE_PREDICTION_AUTHORITY drifted from predict_ms_katl_decomposed")
    if AE_METRIC_AUTHORITY is not compute_screening_metrics:
        raise AEContractError("AE_METRIC_AUTHORITY drifted from compute_screening_metrics")
    d_flags = assert_d_equals_frozen_v3()
    if v3_architecture_digest() != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError("V3 digest drifted")
    _ = v3_arm_spec()
    _ = v3_architecture()
    _ = v3_training_recipe()
    return {
        "REAL_STAGE_A_AUTHORITY": "PASS",
        "REAL_STAGE_B_AUTHORITY": "PASS",
        "REAL_PREDICTION_AUTHORITY": "PASS",
        "REAL_METRIC_AUTHORITY": "PASS",
        **d_flags,
    }


def _arm_init_digest(cell: RealCanonicalCellPlan, arm: str, stage_a: FormalStageAView) -> str:
    spec = ae_arm_spec(arm)
    if spec.init_mode == "native":
        return _json_digest({"mode": "native", "seed": cell.seed, "key": cell.stage_a_key})
    if spec.init_mode == "fused":
        return stage_a.fused_target_digest
    return stage_a.best_source_conv_digest


def _parameter_counts(arm: str) -> Tuple[int, int, int]:
    spec = ae_arm_spec(arm)
    prediction = 100 + (2 if spec.revin_enabled else 0)
    trainable = 40 if spec.target_backbone_trainable else 10
    if spec.revin_enabled:
        trainable += 2
    frozen = prediction - trainable
    return prediction, trainable, frozen


def build_mock_stage_a(cell: RealCanonicalCellPlan) -> FormalStageAView:
    identities = [("S1", cell.dataset), ("S2", cell.dataset), ("S3", cell.dataset)]
    return FormalStageAView(
        stage_a_key=cell.stage_a_key,
        source_identities=identities,
        source_distances=[0.10, 0.50, 0.90],
        source_weights=[0.5, 0.3, 0.2],
        source_weight_digests=["d1", "d2", "d3"],
        fused_params_digest=_json_digest({"fused": cell.stage_a_key}),
        fused_target_digest=_json_digest({"fused_target": cell.stage_a_key}),
        best_source_identity=identities[0],
        best_source_conv_digest=_json_digest({"rank1": identities[0], "key": cell.stage_a_key}),
        target_split_digest=cell.target_split_digest,
        requested_k=3,
        effective_k=3,
        stage_a_seed=int(cell.seed),
        source_selection_digest=_json_digest({"selection": cell.stage_a_key}),
        stage_a_digest=_json_digest({"stage_a": cell.stage_a_key}),
    )


def make_mock_authorities(
    *,
    ledger: FormalCallLedger,
    fail_arm: Optional[str] = None,
    publish_root: Optional[Path] = None,
    live_contexts: Optional[Mapping[str, Any]] = None,
) -> AEFormalAuthorities:
    def prepare_context(cell: RealCanonicalCellPlan) -> Dict[str, Any]:
        if cell.stage_a_key in ledger.context_prep:
            raise AEContractError(f"context prepared twice for {cell.stage_a_key}")
        ledger.context_prep.append(cell.stage_a_key)
        if live_contexts is None:
            return {"cell": cell.stage_a_key, "status": MOCK_STATUS}
        live = live_contexts.get(cell.stage_a_key)
        if live is None:
            raise AEContractError(f"missing real live context for {cell.stage_a_key}")
        if getattr(live, "source_df", None) is None or getattr(live, "target_df", None) is None:
            raise AEContractError("fake context forbidden in real-context mock acceptance")
        if str(getattr(live, "status", "")) != "CONTEXT_READY":
            raise AEContractError(f"live context not CONTEXT_READY: {cell.stage_a_key}")
        return live.as_formal_context()

    def prepare_stage_a(cell: RealCanonicalCellPlan, context: Mapping[str, Any]) -> FormalStageAView:
        live = context.get("live") if isinstance(context, Mapping) else None
        if live_contexts is not None:
            if live is None:
                raise AEContractError("real-context mock body forbids missing live context")
            if getattr(live, "source_df", None) is None:
                raise AEContractError("synthetic dataframe forbidden")
        _ = context
        if cell.stage_a_key in ledger.stage_a:
            raise AEContractError(f"Stage A prepared twice for {cell.stage_a_key}")
        ledger.stage_a.append(cell.stage_a_key)
        return build_mock_stage_a(cell)

    def make_common_head(
        cell: RealCanonicalCellPlan,
        stage_a: FormalStageAView,
        context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        _ = context
        return _json_digest({"head": "v3_d_authority", "key": cell.stage_a_key, "fused": stage_a.fused_target_digest})

    def train_stage_b(
        cell: RealCanonicalCellPlan,
        arm: str,
        stage_a: FormalStageAView,
        head_digest: str,
        context: Mapping[str, Any],
    ) -> FormalArmRuntime:
        _ = context
        spec = ae_arm_spec(arm)
        ledger.stage_b.append((cell.stage_a_key, arm))
        if fail_arm == arm:
            raise RuntimeError(f"mock Stage-B failure on {arm}")
        pred_count, trainable, frozen = _parameter_counts(arm)
        recipe = ae_training_recipe()
        return FormalArmRuntime(
            arm=arm,
            spec=spec,
            init_mode=spec.init_mode,
            source_init_consumed=spec.init_mode != "native",
            stage_a_shared_for_pairing=True,
            head_target_init_digest=head_digest,
            target_initial_conv_digest=_arm_init_digest(cell, arm, stage_a),
            fused_init_digest=stage_a.fused_target_digest,
            fused_params_digest=stage_a.fused_params_digest,
            best_source_identity=stage_a.best_source_identity,
            best_source_snapshot_digest=stage_a.best_source_conv_digest,
            source_identities=list(stage_a.source_identities),
            source_model_digests=list(stage_a.source_weight_digests),
            source_distances=list(stage_a.source_distances),
            source_weights=list(stage_a.source_weights),
            source_selection_digest=stage_a.source_selection_digest,
            stage_a_digest=stage_a.stage_a_digest,
            selected_lr=float(cell.selected_lr),
            recipe_digest=_json_digest(recipe),
            target_split_digest=stage_a.target_split_digest,
            parent_method=PARENT_METHOD,
            parent_v3_architecture_digest=v3_architecture_digest(),
            prediction_path_parameter_count=pred_count,
            trainable_prediction_path_parameter_count=trainable,
            frozen_prediction_path_parameter_count=frozen,
            status=MOCK_STATUS,
            metric_kind=MOCK_METRIC,
        )

    def predict(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
        _ = context
        ledger.predict.append((cell.stage_a_key, runtime.arm))
        runtime.metric_kind = MOCK_METRIC
        runtime.rmse_original = None
        runtime.smape_original = None
        runtime.mae_original = None
        return runtime

    def metric(cell: RealCanonicalCellPlan, runtime: FormalArmRuntime, context: Optional[Mapping[str, Any]] = None) -> FormalArmRuntime:
        _ = context
        ledger.metric.append((cell.stage_a_key, runtime.arm))
        runtime.metric_kind = MOCK_METRIC
        runtime.status = MOCK_STATUS
        runtime.rmse_original = None
        return runtime

    def publish(
        cell: RealCanonicalCellPlan,
        runtimes: Mapping[str, FormalArmRuntime],
        pair_status: str,
        output_root: Path,
    ) -> None:
        ledger.publish.append(cell.stage_a_key)
        root = publish_root if publish_root is not None else output_root
        bundle_dir = Path(root) / "cell_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": MOCK_STATUS,
            "pair_status": pair_status,
            "stage_a_key": cell.stage_a_key,
            "arms": {
                arm: {
                    "status": runtime.status,
                    "rmse_original": None,
                    "metric_kind": MOCK_METRIC,
                    "init_mode": runtime.init_mode,
                    "source_init_consumed": runtime.source_init_consumed,
                }
                for arm, runtime in runtimes.items()
            },
        }
        (bundle_dir / f"{cell.stage_a_key.replace('/', '-')}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    return AEFormalAuthorities(
        prepare_context=prepare_context,
        prepare_stage_a=prepare_stage_a,
        make_common_head=make_common_head,
        train_stage_b=train_stage_b,
        predict=predict,
        metric=metric,
        publish=publish,
        reset_rng=reset_ae_rng,
    )


def validate_paired_runtime_bundle(
    cell: RealCanonicalCellPlan,
    stage_a: FormalStageAView,
    runtimes: Mapping[str, FormalArmRuntime],
) -> str:
    if set(runtimes) != set(AE_ARMS):
        return "PAIR_INCOMPLETE"
    if any(item.status not in {MOCK_STATUS, "OK"} or item.error for item in runtimes.values()):
        return "PAIR_INCOMPLETE"
    bcd = [runtimes[arm] for arm in ("B", "C", "D", "E")]
    if len({tuple(item.source_identities) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source identities diverged")
    if len({tuple(item.source_distances) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source distances diverged")
    if len({tuple(item.source_weights) for item in bcd}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: source weights diverged")
    if len({runtimes[arm].fused_init_digest for arm in ("B", "C", "D")}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: fused init digest diverged")
    if runtimes["E"].best_source_identity != tuple(stage_a.source_identities[0]):
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: E rank-1 identity mismatch")
    if runtimes["E"].best_source_snapshot_digest != stage_a.best_source_conv_digest:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: E rank-1 digest mismatch")
    if len({item.head_target_init_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: head init digest diverged")
    if len({item.target_split_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: target split digest diverged")
    if len({item.selected_lr for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: LR diverged")
    if len({item.recipe_digest for item in runtimes.values()}) != 1:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: recipe digest diverged")
    if runtimes["A"].source_init_consumed:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: arm A consumed source init")
    if runtimes["D"].parent_v3_architecture_digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: D parent digest drifted")
    if runtimes["B"].target_initial_conv_digest != runtimes["C"].target_initial_conv_digest:
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: B/C fused conv digest diverged")
    if runtimes["A"].init_mode != "native" or runtimes["E"].init_mode != "best_source":
        raise AEContractError("PAIR_PROTOCOL_VIOLATION: init modes drifted")
    _ = cell
    return "PAIR_COMPLETE"


def load_pair_statuses(run_root: Path) -> Dict[str, str]:
    path = Path(run_root) / "cell_bundles"
    statuses: Dict[str, str] = {}
    if not path.is_dir():
        return statuses
    for file in path.glob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8"))
        statuses[str(payload["stage_a_key"])] = str(payload.get("pair_status") or "PAIR_INCOMPLETE")
    return statuses


def run_ae_formal_body(
    *,
    mock_execution_authorized: bool = False,
    execute: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    authorities: Optional[AEFormalAuthorities] = None,
    arm_order: Sequence[str] = AE_ARMS,
    resume_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    ledger: Optional[FormalCallLedger] = None,
    cells: Optional[Sequence[RealCanonicalCellPlan]] = None,
    matrix_kind: str = FROZEN_MATRIX_KIND,
    hardened_attempts: bool = False,
    selected_stage_a_key: Optional[str] = None,
    selected_arm: Optional[str] = None,
    reused_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Canonical formal orchestration. Real trainers stay blocked without dual lock."""
    if resume_root is not None and output_root is None:
        output_root = Path(resume_root)
    if not mock_execution_authorized:
        assert_formal_training_permitted(execute=execute, environ=environ)
        if not AE_FORMAL_EXECUTION_ENABLED:
            raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")
        authorities = authorities or default_formal_authorities()
    elif authorities is None:
        raise AEContractError("mock formal body requires injected authorities")
    order = tuple(str(item).strip().upper() for item in arm_order)
    if selected_arm is not None:
        selected_arm = str(selected_arm).strip().upper()
        if selected_arm not in AE_ARMS:
            raise AEContractError(f"selected_arm must be one of {list(AE_ARMS)}")
        if selected_stage_a_key is None:
            raise AEContractError("selected_arm requires selected_stage_a_key")
        order = (selected_arm,)
    elif tuple(sorted(order)) != tuple(sorted(AE_ARMS)):
        raise AEContractError(f"arm_order must be a permutation of {list(AE_ARMS)}")
    kind = str(matrix_kind).strip().upper().replace("-", "_")
    if kind == "FROZEN270":
        kind = FROZEN_MATRIX_KIND
    allowed_kinds = {
        FROZEN_MATRIX_KIND,
        EXTRA_TARGET_MATRIX_KIND,
        UNIFORM_3E4_MATRIX_KIND,
        FORMAL_PER_DATASET_LR_MATRIX_KIND,
    }
    if kind not in allowed_kinds:
        raise AEContractError(f"unknown matrix_kind={matrix_kind!r}")
    if reused_payloads and not hardened_attempts:
        raise AEContractError("reused formal cells require hardened attempts")

    root = project_root_from(project_root)
    load_ae_manifest(root)
    extra_identity: Optional[Tuple[str, str]] = None
    if kind == EXTRA_TARGET_MATRIX_KIND:
        if cells is None:
            raise AEContractError("EXTRA_TARGET_SLICE requires explicit cells")
        cell_plans = list(cells)
        arm_plans = enumerate_real_arm_plans(cell_plans, expected_runs=EXTRA_TARGET_RUNS)
        extra_identity = assert_extra_target_slice(cell_plans, arm_plans)
        expected_cells = EXTRA_TARGET_CANONICAL_CELLS
        expected_runs = EXTRA_TARGET_RUNS
    elif kind == UNIFORM_3E4_MATRIX_KIND:
        from src.transfer_methods.ms_katl_ae_execution import assert_uniform_3e4_slice

        cell_plans = list(cells) if cells is not None else enumerate_uniform_3e4_cell_plans(project_root=root)
        arm_plans = enumerate_real_arm_plans(cell_plans, expected_runs=UNIFORM_3E4_RUNS)
        assert_uniform_3e4_slice(cell_plans, arm_plans)
        expected_cells = UNIFORM_3E4_CANONICAL_CELLS
        expected_runs = UNIFORM_3E4_RUNS
    elif kind == FORMAL_PER_DATASET_LR_MATRIX_KIND:
        from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
            assert_formal_per_dataset_lr_slice,
            enumerate_formal_per_dataset_lr_cell_plans,
        )

        cell_plans = (
            list(cells)
            if cells is not None
            else enumerate_formal_per_dataset_lr_cell_plans(project_root=root)
        )
        arm_plans = enumerate_real_arm_plans(cell_plans, expected_runs=FORMAL_PER_DATASET_LR_RUNS)
        assert_formal_per_dataset_lr_slice(cell_plans, arm_plans)
        expected_cells = FORMAL_PER_DATASET_LR_CANONICAL_CELLS
        expected_runs = FORMAL_PER_DATASET_LR_RUNS
    else:
        cell_plans = list(cells) if cells is not None else enumerate_real_canonical_cell_plans(project_root=root)
        arm_plans = enumerate_real_arm_plans(cell_plans)
        if len(cell_plans) != EXPECTED_CANONICAL_CELLS or len(arm_plans) != EXPECTED_RUNS:
            raise AEContractError("formal body matrix drifted")
        expected_cells = EXPECTED_CANONICAL_CELLS
        expected_runs = EXPECTED_RUNS
    matrix_hash = planned_matrix_digest(arm_plans)
    provenance: Optional[Dict[str, Any]] = None
    if not mock_execution_authorized:
        provenance = build_formal_run_provenance(
            root,
            matrix_digest=matrix_hash,
            matrix_kind=kind,
            expected_runs=expected_runs,
            cells=cell_plans,
        )
    if resume_root is not None:
        validate_resume_root(
            Path(resume_root),
            project_root=root,
            matrix_kind=kind,
            extra_target=extra_identity,
        )
        snapshot = json.loads((Path(resume_root) / "manifest_snapshot.json").read_text(encoding="utf-8"))
        current = ae_protocol_manifest_payload()
        if snapshot.get("protocol_version") != PROTOCOL_VERSION:
            raise AEContractError("resume protocol_version mismatch")
        if snapshot.get("parent_v3_architecture_digest") != current["parent_v3_architecture_digest"]:
            raise AEContractError("resume parent V3 digest mismatch")
        if snapshot.get("planned_matrix_digest") not in {None, matrix_hash} and snapshot.get("planned_matrix_digest") != matrix_hash:
            raise AEContractError("resume planned matrix digest mismatch")
        if not mock_execution_authorized:
            if provenance is None:
                raise AEContractError("resume provenance is unavailable")
            validate_formal_resume_provenance(Path(resume_root), provenance)

    if output_root is not None:
        out = Path(output_root)
    elif mock_execution_authorized:
        out = root / PREFLIGHT_RELPATH / "formal_mock"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if kind == EXTRA_TARGET_MATRIX_KIND and extra_identity is not None:
            out = root / planned_extra_target_run_root(timestamp, extra_identity[0], extra_identity[1])
        elif kind == UNIFORM_3E4_MATRIX_KIND:
            from src.transfer_methods.ms_katl_ae_ablation import UNIFORM_3E4_OUTPUT_RELPATH

            out = root / UNIFORM_3E4_OUTPUT_RELPATH
        else:
            out = root / planned_formal_run_root(timestamp)
    if mock_execution_authorized:
        if "outputs" in out.parts and "runs" in out.parts:
            raise AEContractError("mock formal artifacts must not enter outputs/runs")
        if "results" in out.parts and "preflight" not in out.parts and "audits" not in out.parts:
            raise AEContractError("mock formal artifacts must not enter results/")
    else:
        assert_formal_run_root(out, kind)
        if kind == EXTRA_TARGET_MATRIX_KIND and "EXTRA_" not in out.name:
            raise AEContractError("extra-target run root must include EXTRA_")
    out.mkdir(parents=True, exist_ok=True)

    if not mock_execution_authorized:
        if resume_root is None:
            extra_snapshot = None
            if kind == EXTRA_TARGET_MATRIX_KIND and extra_identity is not None:
                extra_snapshot = {
                    "matrix_kind": EXTRA_TARGET_MATRIX_KIND,
                    "extra_target": {
                        "dataset": extra_identity[0],
                        "target_identity": extra_identity[1],
                    },
                    "frozen_270_untouched": True,
                }
            elif kind == UNIFORM_3E4_MATRIX_KIND:
                extra_snapshot = {
                    "matrix_kind": UNIFORM_3E4_MATRIX_KIND,
                    "uniform_learning_rate": "3e-4",
                    "paper_targets": dict(UNIFORM_3E4_TARGETS),
                    "frozen_270_untouched": True,
                    "extra_target_slice_untouched": True,
                    "only_global_lr_changed": True,
                }
            elif kind == FORMAL_PER_DATASET_LR_MATRIX_KIND:
                from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
                    FORMAL_PER_DATASET_LR_LABELS,
                    FORMAL_PER_DATASET_LR_TARGETS,
                )

                extra_snapshot = {
                    "matrix_kind": FORMAL_PER_DATASET_LR_MATRIX_KIND,
                    "target_learning_rates": dict(FORMAL_PER_DATASET_LR_LABELS),
                    "paper_targets": dict(FORMAL_PER_DATASET_LR_TARGETS),
                    "uniform_slice_untouched": True,
                    "d5_reuse_audited": bool(reused_payloads),
                    "d5_reused": bool(reused_payloads),
                }
            if provenance is not None:
                extra_snapshot = dict(extra_snapshot or {})
                extra_snapshot.update(provenance)
                extra_snapshot["run_root"] = str(out.resolve())
            bootstrap_fresh_formal_run_metadata(
                out,
                arm_plans,
                matrix_hash,
                expected_runs=expected_runs,
                extra_snapshot=extra_snapshot,
            )
        # Resume path: validate_resume_root already ran; never rewrite manifest/matrix.

    completed = {} if hardened_attempts else load_pair_statuses(out)
    active_ledger = ledger if ledger is not None else FormalCallLedger()
    bundles: Dict[str, str] = dict(completed)
    identity_index: Dict[str, Dict[str, Any]] = {}
    selected_only = selected_stage_a_key is not None
    reused_count = 0

    for cell in cell_plans:
        if selected_only and cell.stage_a_key != selected_stage_a_key:
            continue
        attempt_path: Optional[Path] = None
        if reused_payloads and cell.stage_a_key in reused_payloads:
            if provenance is None:
                raise AEContractError("reused formal cells require formal provenance")
            action, attempt_path = prepare_formal_cell_attempt(
                out,
                cell,
                resume=resume_root is not None,
                digests={key: provenance[key] for key in FORMAL_RESUME_DIGEST_KEYS},
                branch=str(provenance["branch"]),
                head=str(provenance["HEAD"]),
            )
            if action == "SKIPPED_COMPLETE":
                bundles[cell.stage_a_key] = "PAIR_COMPLETE"
                reused_count += 1
                continue
            reused_payload = dict(reused_payloads[cell.stage_a_key])
            reused_payload.update(
                {
                    "status": "COMPLETE",
                    "pair_status": "PAIR_COMPLETE",
                    "dataset": cell.dataset,
                    "target": cell.target,
                    "horizon": int(cell.horizon),
                    "seed": int(cell.seed),
                    "cell_key": cell.stage_a_key,
                    "canonical_stage_a_key": cell.stage_a_key,
                    "stage_a_seed": int(cell.seed),
                    "attempt_id": attempt_path.name,
                    "matrix_digest": provenance["matrix_digest"],
                    "protocol_digest": provenance["protocol_digest"],
                    "critical_runtime_digest": provenance["critical_runtime_digest"],
                    "working_tree_protocol_digest": provenance["working_tree_protocol_digest"],
                    "runtime_commit": provenance["runtime_commit"],
                    "branch": provenance["branch"],
                    "reused_in_formal_root": True,
                }
            )
            finalize_formal_cell_attempt(attempt_path, status="COMPLETE", result=reused_payload)
            bundles[cell.stage_a_key] = "PAIR_COMPLETE"
            reused_count += 1
            if kind in {
                EXTRA_TARGET_MATRIX_KIND,
                UNIFORM_3E4_MATRIX_KIND,
                FORMAL_PER_DATASET_LR_MATRIX_KIND,
            } and not mock_execution_authorized:
                print(f"AE_CELL_REUSED {cell.stage_a_key} PAIR_COMPLETE", flush=True)
            continue
        if hardened_attempts:
            if provenance is None:
                raise AEContractError("hardened attempts require formal provenance")
            action, attempt_path = prepare_formal_cell_attempt(
                out,
                cell,
                resume=resume_root is not None,
                digests={key: provenance[key] for key in FORMAL_RESUME_DIGEST_KEYS},
                branch=str(provenance["branch"]),
                head=str(provenance["HEAD"]),
            )
            if action == "SKIPPED_COMPLETE":
                bundles[cell.stage_a_key] = "PAIR_COMPLETE"
                continue
        elif completed.get(cell.stage_a_key) == "PAIR_COMPLETE":
            continue
        if kind in {
            EXTRA_TARGET_MATRIX_KIND,
            UNIFORM_3E4_MATRIX_KIND,
            FORMAL_PER_DATASET_LR_MATRIX_KIND,
        } and not mock_execution_authorized:
            print(f"AE_CELL_START {cell.stage_a_key}", flush=True)
        context = authorities.prepare_context(cell)
        if not isinstance(context, dict):
            raise AEContractError("prepare_context must return a mutable dict")
        if cell.stage_a_key not in active_ledger.context_prep:
            active_ledger.context_prep.append(cell.stage_a_key)
        live = context.get("live") if isinstance(context, Mapping) else None
        if live is not None and str(getattr(live, "stage_a_key", "")) != cell.stage_a_key:
            raise AEContractError("live context/stage-A cell mismatch")
        if live is not None and str(getattr(live, "target_bundle_key", "")) != cell.stage_a_key:
            raise AEContractError("live context/target-bundle cell mismatch")
        stage_a = authorities.prepare_stage_a(cell, context)
        head_digest = authorities.make_common_head(cell, stage_a, context)
        runtimes: Dict[str, FormalArmRuntime] = {}
        failed = False
        for arm in order:
            authorities.reset_rng(int(cell.seed))
            try:
                runtime = authorities.train_stage_b(cell, arm, stage_a, head_digest, context)
                runtime = authorities.predict(cell, runtime, context)
                runtime = authorities.metric(cell, runtime, context)
                runtimes[arm] = runtime
            except Exception as exc:
                failed = True
                spec = ae_arm_spec(arm)
                failure_metric_kind = MOCK_METRIC if mock_execution_authorized else None
                runtimes[arm] = FormalArmRuntime(
                    arm=arm,
                    spec=spec,
                    init_mode=spec.init_mode,
                    source_init_consumed=spec.init_mode != "native",
                    stage_a_shared_for_pairing=True,
                    head_target_init_digest=head_digest,
                    target_initial_conv_digest="",
                    fused_init_digest=stage_a.fused_target_digest,
                    best_source_identity=stage_a.best_source_identity,
                    best_source_snapshot_digest=stage_a.best_source_conv_digest,
                    source_identities=list(stage_a.source_identities),
                    source_distances=list(stage_a.source_distances),
                    source_weights=list(stage_a.source_weights),
                    selected_lr=float(cell.selected_lr),
                    recipe_digest=_json_digest(ae_training_recipe()),
                    target_split_digest=stage_a.target_split_digest,
                    parent_method=PARENT_METHOD,
                    parent_v3_architecture_digest=v3_architecture_digest(),
                    prediction_path_parameter_count=0,
                    trainable_prediction_path_parameter_count=0,
                    frozen_prediction_path_parameter_count=0,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                    metric_kind=failure_metric_kind,
                )
        active_ledger.bundle_validations.append(cell.stage_a_key)
        if selected_only:
            pair_status = "SINGLE_ARM_INCOMPLETE" if failed else "SINGLE_ARM_COMPLETE"
        elif failed or set(runtimes) != set(AE_ARMS):
            pair_status = "PAIR_INCOMPLETE"
        else:
            pair_status = validate_paired_runtime_bundle(cell, stage_a, runtimes)
        if pair_status == "PAIR_COMPLETE":
            identity_index[cell.stage_a_key] = {
                arm: {
                    "init_mode": runtimes[arm].init_mode,
                    "head": runtimes[arm].head_target_init_digest,
                    "conv": runtimes[arm].target_initial_conv_digest,
                    "stage_a": tuple(runtimes[arm].source_identities),
                    "best": runtimes[arm].best_source_identity,
                }
                for arm in AE_ARMS
            }
        publish_root = attempt_path if attempt_path is not None else out
        authorities.publish(cell, runtimes, pair_status, publish_root)
        if attempt_path is not None and provenance is not None:
            result_path = attempt_path / "result.json"
            raw_result: Dict[str, Any] = {}
            if result_path.is_file():
                raw_result = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload = {
                **raw_result,
                "status": "COMPLETE" if pair_status in {"PAIR_COMPLETE", "SINGLE_ARM_COMPLETE"} else "FAILED",
                "pair_status": pair_status,
                "dataset": cell.dataset,
                "target": cell.target,
                "horizon": int(cell.horizon),
                "seed": int(cell.seed),
                "cell_key": cell.stage_a_key,
                "canonical_stage_a_key": cell.stage_a_key,
                "stage_a_seed": int(cell.seed),
                "attempt_id": attempt_path.name,
                "matrix_digest": provenance["matrix_digest"],
                "protocol_digest": provenance["protocol_digest"],
                "critical_runtime_digest": provenance["critical_runtime_digest"],
                "working_tree_protocol_digest": provenance["working_tree_protocol_digest"],
                "runtime_commit": provenance["runtime_commit"],
                "branch": provenance["branch"],
                "arms": {arm: serialize_formal_arm_runtime(runtime) for arm, runtime in runtimes.items()},
            }
            finalize_formal_cell_attempt(
                attempt_path,
                status=result_payload["status"],
                result=result_payload,
            )
        bundles[cell.stage_a_key] = pair_status
        if kind in {
            EXTRA_TARGET_MATRIX_KIND,
            UNIFORM_3E4_MATRIX_KIND,
            FORMAL_PER_DATASET_LR_MATRIX_KIND,
        } and not mock_execution_authorized:
            print(f"AE_CELL_DONE {cell.stage_a_key} {pair_status}", flush=True)

    counts = active_ledger.counts()
    pair_complete = sum(1 for status in bundles.values() if status == "PAIR_COMPLETE")
    pair_incomplete = sum(1 for status in bundles.values() if status != "PAIR_COMPLETE")
    selected_complete = sum(
        1 for status in bundles.values() if status in {"SINGLE_ARM_COMPLETE", "PAIR_COMPLETE"}
    )
    result = {
        "status": MOCK_STATUS if mock_execution_authorized else "FORMAL",
        "MATRIX_KIND": kind,
        "CANONICAL_CELLS": len(cell_plans),
        "EXPECTED_RUNS": expected_runs,
        "CONTEXT_PREP_CALLS": counts["CONTEXT_PREP_CALLS"],
        "STAGE_A_CALLS": counts["STAGE_A_CALLS"],
        "STAGE_B_CALLS": counts["STAGE_B_CALLS"],
        "PREDICT_CALLS": counts["PREDICT_CALLS"],
        "METRIC_CALLS": counts["METRIC_CALLS"],
        "BUNDLE_VALIDATIONS": counts["BUNDLE_VALIDATIONS"],
        "PUBLISH_ATTEMPTS": counts["PUBLISH_ATTEMPTS"],
        "A_CALLS": counts["A_CALLS"],
        "B_CALLS": counts["B_CALLS"],
        "C_CALLS": counts["C_CALLS"],
        "D_CALLS": counts["D_CALLS"],
        "E_CALLS": counts["E_CALLS"],
        "PAIR_COMPLETE": pair_complete,
        "PAIR_INCOMPLETE": pair_incomplete,
        "REUSED_CELLS": reused_count,
        "SINGLE_ARM_COMPLETE": selected_complete if selected_only else 0,
        "bundles": bundles,
        "identity_index": identity_index,
        "planned_matrix_digest": matrix_hash,
        "ledger": active_ledger,
        "output_root": str(out),
        "TRAINING_STARTED": "NO" if mock_execution_authorized else "YES",
        "AE_FORMAL_EXECUTION_ENABLED": AE_FORMAL_EXECUTION_ENABLED,
    }
    if extra_identity is not None:
        result["EXTRA_DATASET"] = extra_identity[0]
        result["EXTRA_TARGET"] = extra_identity[1]
    if not mock_execution_authorized:
        if selected_only:
            selected_status = bundles.get(str(selected_stage_a_key))
            if selected_status not in {"SINGLE_ARM_COMPLETE", "PAIR_COMPLETE"}:
                raise AEContractError(
                    f"FORMAL_COMPLETION_FAILED: selected cell status={selected_status}"
                )
        elif pair_incomplete > 0 or pair_complete != expected_cells:
            raise AEContractError(
                "FORMAL_COMPLETION_FAILED: "
                f"PAIR_COMPLETE={pair_complete} PAIR_INCOMPLETE={pair_incomplete} "
                f"canonical_cells={expected_cells}"
            )
        result["TRAINING_STARTED"] = "YES"
        result["status"] = "FORMAL_COMPLETE"
        if selected_only:
            return result
        if kind == EXTRA_TARGET_MATRIX_KIND:
            from src.transfer_methods.ms_katl_ae_execution import write_extra_target_results_table

            result["results_csv"] = str(write_extra_target_results_table(out))
        elif kind == UNIFORM_3E4_MATRIX_KIND:
            from src.transfer_methods.ms_katl_sita_ablation_3e4 import write_uniform_3e4_results_table

            result["results_csv"] = str(write_uniform_3e4_results_table(out))
        elif kind == FORMAL_PER_DATASET_LR_MATRIX_KIND:
            from src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae import (
                write_formal_per_dataset_lr_results_table,
            )

            result["results_csv"] = str(write_formal_per_dataset_lr_results_table(out))
    return result


def git_provenance(project_root: Optional[Path] = None) -> Dict[str, str]:
    root = project_root_from(project_root)

    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr).strip()

    porcelain = _git("status", "--porcelain")
    ae_core = (
        "src/transfer_methods/ms_katl_ae_ablation.py",
        "src/transfer_methods/ms_katl_ae_execution.py",
        "src/transfer_methods/ms_katl_ae_formal.py",
        "src/transfer_methods/ms_katl_ae_live_context.py",
        "src/transfer_methods/ms_katl_v3_frame_assembly.py",
        "scripts/run_ms_katl_ae_ablation.py",
        "tests/test_ms_katl_ae_ablation_contract.py",
        "tests/test_ms_katl_ae_formal_body.py",
        "configs/ms_katl_ae_ablation_manifest.json",
    )
    dirty = []
    for line in porcelain.splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if path in ae_core:
            dirty.append(path)
    freeze_required = "YES" if dirty or porcelain else "NO"
    if dirty:
        freeze_required = "YES"
    return {
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": porcelain,
        "ae_core_untracked_or_dirty": ", ".join(dirty),
        "FORMAL_GIT_FREEZE_REQUIRED": freeze_required,
    }


def run_ae_formal_preflight(
    *,
    project_root: Optional[Path] = None,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    load_ae_manifest(root)
    authorities = assert_authorities_are_frozen_v3()
    cells = enumerate_real_canonical_cell_plans(project_root=root)
    arms = enumerate_real_arm_plans(cells)
    git_info = git_provenance(root)
    source = Path(__file__).read_text(encoding="utf-8")
    old_reuse = assert_no_old_v3_result_injection(source)
    preflight = run_ae_preflight(project_root=root, include_construction_contracts=True)
    pairwise = preflight["pairwise"]
    kwargs_report = resolve_all_formal_stage_a_input_plans(cells, project_root=root)
    from src.transfer_methods.ms_katl_ae_live_context import (
        construct_all_live_ae_contexts,
        representative_trace,
    )

    live_report = construct_all_live_ae_contexts(cells, project_root=root)
    traces = {}
    d4_trace = {}
    for ctx in live_report["contexts"]:
        if int(ctx.horizon) == 1 and int(ctx.seed) == 42:
            traces[ctx.dataset] = representative_trace(ctx)
        if ctx.dataset == "D4" and int(ctx.horizon) == 1 and int(ctx.seed) == 42:
            d4_trace = representative_trace(ctx)
    training_guard = "PASS"
    try:
        assert_formal_training_permitted(execute=True, environ={})
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    try:
        assert_formal_training_permitted(
            execute=False,
            environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"},
        )
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    enabled = "YES" if AE_FORMAL_EXECUTION_ENABLED else "NO"
    unique_run_keys = len({item.run_key for item in arms})
    unique_stage_a = len({cell.stage_a_key for cell in cells})
    payload = {
        "AE_PROTOCOL_VERSION": PROTOCOL_VERSION,
        "REAL_STAGE_A_AUTHORITY": authorities["REAL_STAGE_A_AUTHORITY"],
        "REAL_STAGE_B_AUTHORITY": authorities["REAL_STAGE_B_AUTHORITY"],
        "REAL_PREDICTION_AUTHORITY": authorities["REAL_PREDICTION_AUTHORITY"],
        "REAL_METRIC_AUTHORITY": authorities["REAL_METRIC_AUTHORITY"],
        "CANONICAL_CELLS": len(cells),
        "PLANNED_RUNS": len(arms),
        "UNIQUE_RUN_KEYS": unique_run_keys,
        "UNIQUE_STAGE_A_KEYS": unique_stage_a,
        "STAGE_A_AUTHORITY_NAME": AE_STAGE_A_AUTHORITY.__name__,
        "STAGE_B_AUTHORITY_NAME": AE_STAGE_B_AUTHORITY.__name__,
        "PREDICTION_AUTHORITY_NAME": AE_PREDICTION_AUTHORITY.__name__,
        "METRIC_AUTHORITY_NAME": AE_METRIC_AUTHORITY.__name__,
        "AE_FORMAL_EXECUTION_ENABLED": enabled,
        "FORMAL_EXECUTION_CAPABILITY": "ENABLED" if AE_FORMAL_EXECUTION_ENABLED else "DISABLED",
        "ACTUAL_EXECUTION_REQUESTED": "NO",
        "TRAINING_GUARD": training_guard,
        "D_OLD_RESULT_REUSE": old_reuse,
        "D_EQUALS_FROZEN_V3": preflight["d_identity"]["ARM_D_PARENT_V3_IDENTITY"],
        "A_C_SINGLE_FACTOR": pairwise.get("A_C_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "B_C_SINGLE_FACTOR": pairwise.get("B_C_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "C_E_SINGLE_FACTOR": pairwise.get("C_E_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "C_D_SINGLE_FACTOR": pairwise.get("C_D_SINGLE_FACTOR_CONTRACT", "FAIL"),
        "LR_CONTRACT": "PASS",
        "RECIPE_CONTRACT": "PASS",
        "FORMAL_GIT_FREEZE_REQUIRED": git_info["FORMAL_GIT_FREEZE_REQUIRED"],
        "git": git_info,
        "planned_formal_run_root_pattern": str(planned_formal_run_root("<timestamp>")),
        "REAL_DATA_FORMAL_PRECHECK": kwargs_report["REAL_DATA_FORMAL_PRECHECK"],
        "REAL_STAGE_A_KWARGS_RESOLUTION": kwargs_report["REAL_STAGE_A_KWARGS_RESOLUTION"],
        "REAL_STAGE_A_KWARGS_FAILURES": kwargs_report["failures"],
        "LIVE_CONTEXT_CONSTRUCTION": live_report["LIVE_CONTEXT_CONSTRUCTION"],
        "REAL_STAGE_A_KWARGS_OBJECTS": live_report["REAL_STAGE_A_KWARGS_OBJECTS"],
        "STAGE_A_SIGNATURE_COMPATIBILITY": live_report["STAGE_A_SIGNATURE_COMPATIBILITY"],
        "REAL_SHARED_TARGET_BUNDLES": live_report["REAL_SHARED_TARGET_BUNDLES"],
        "D1_LIVE_CONTEXT": live_report["dataset_status"]["D1"],
        "D2_LIVE_CONTEXT": live_report["dataset_status"]["D2"],
        "D3_LIVE_CONTEXT": live_report["dataset_status"]["D3"],
        "D4_LIVE_CONTEXT": live_report["dataset_status"]["D4"],
        "D5_LIVE_CONTEXT": live_report["dataset_status"]["D5"],
        "D6_LIVE_CONTEXT": live_report["dataset_status"]["D6"],
        "LIVE_CONTEXT_MATRIX": live_report["matrix"],
        "D4_REPRESENTATIVE_TRACE": d4_trace,
        "DATASET_REPRESENTATIVE_TRACES": traces,
        "LIVE_CONTEXT_FAILURES": live_report["failures"],
        "FORMAL_STAGE_A_WILL_RECOMPUTE": "YES",
        "TRAINING_STARTED": "NO",
    }
    artifact = {
        key: value
        for key, value in payload.items()
        if key not in {"LIVE_CONTEXT_FAILURES", "REAL_STAGE_A_KWARGS_FAILURES"} or value
    }
    if write_artifacts:
        out = root / PREFLIGHT_RELPATH
        out.mkdir(parents=True, exist_ok=True)
        (out / "formal_preflight_report.json").write_text(
            json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (out / "live_context_matrix.json").write_text(
            json.dumps({"rows": live_report["matrix"], "status": "CONTEXT_READY"}, indent=2) + "\n",
            encoding="utf-8",
        )
    ok = (
        training_guard == "PASS"
        and old_reuse == "NO"
        and payload["CANONICAL_CELLS"] == EXPECTED_CANONICAL_CELLS
        and payload["PLANNED_RUNS"] == EXPECTED_RUNS
        and unique_run_keys == EXPECTED_RUNS
        and unique_stage_a == EXPECTED_CANONICAL_CELLS
        and payload["A_C_SINGLE_FACTOR"] == "PASS"
        and payload["B_C_SINGLE_FACTOR"] == "PASS"
        and payload["C_E_SINGLE_FACTOR"] == "PASS"
        and payload["C_D_SINGLE_FACTOR"] == "PASS"
        and payload["D_EQUALS_FROZEN_V3"] == "PASS"
        and kwargs_report["REAL_DATA_FORMAL_PRECHECK"] == "PASS"
        and live_report["ok"]
        and AE_FORMAL_EXECUTION_ENABLED
    )
    lines = [
        f"AE_PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"REAL_STAGE_A_AUTHORITY={payload['REAL_STAGE_A_AUTHORITY']}",
        f"REAL_STAGE_B_AUTHORITY={payload['REAL_STAGE_B_AUTHORITY']}",
        f"REAL_PREDICTION_AUTHORITY={payload['REAL_PREDICTION_AUTHORITY']}",
        f"REAL_METRIC_AUTHORITY={payload['REAL_METRIC_AUTHORITY']}",
        f"CANONICAL_CELLS={payload['CANONICAL_CELLS']}",
        f"PLANNED_RUNS={payload['PLANNED_RUNS']}",
        f"UNIQUE_RUN_KEYS={payload['UNIQUE_RUN_KEYS']}",
        f"UNIQUE_STAGE_A_KEYS={payload['UNIQUE_STAGE_A_KEYS']}",
        f"LR_CONTRACT={payload['LR_CONTRACT']}",
        f"RECIPE_CONTRACT={payload['RECIPE_CONTRACT']}",
        f"A_C_SINGLE_FACTOR={payload['A_C_SINGLE_FACTOR']}",
        f"B_C_SINGLE_FACTOR={payload['B_C_SINGLE_FACTOR']}",
        f"C_E_SINGLE_FACTOR={payload['C_E_SINGLE_FACTOR']}",
        f"C_D_SINGLE_FACTOR={payload['C_D_SINGLE_FACTOR']}",
        f"D_EQUALS_FROZEN_V3={payload['D_EQUALS_FROZEN_V3']}",
        f"FORMAL_EXECUTION_CAPABILITY={payload['FORMAL_EXECUTION_CAPABILITY']}",
        f"ACTUAL_EXECUTION_REQUESTED={payload['ACTUAL_EXECUTION_REQUESTED']}",
        f"AE_FORMAL_EXECUTION_ENABLED={enabled}",
        f"TRAINING_GUARD={training_guard}",
        f"D_OLD_RESULT_REUSE={old_reuse}",
        f"FORMAL_GIT_FREEZE_REQUIRED={payload['FORMAL_GIT_FREEZE_REQUIRED']}",
        f"REAL_DATA_FORMAL_PRECHECK={payload['REAL_DATA_FORMAL_PRECHECK']}",
        f"LIVE_CONTEXT_CONSTRUCTION={payload['LIVE_CONTEXT_CONSTRUCTION']}",
        f"REAL_STAGE_A_KWARGS_OBJECTS={payload['REAL_STAGE_A_KWARGS_OBJECTS']}",
        f"STAGE_A_SIGNATURE_COMPATIBILITY={payload['STAGE_A_SIGNATURE_COMPATIBILITY']}",
        f"REAL_SHARED_TARGET_BUNDLES={payload['REAL_SHARED_TARGET_BUNDLES']}",
        f"D1_LIVE_CONTEXT={payload['D1_LIVE_CONTEXT']}",
        f"D2_LIVE_CONTEXT={payload['D2_LIVE_CONTEXT']}",
        f"D3_LIVE_CONTEXT={payload['D3_LIVE_CONTEXT']}",
        f"D4_LIVE_CONTEXT={payload['D4_LIVE_CONTEXT']}",
        f"D5_LIVE_CONTEXT={payload['D5_LIVE_CONTEXT']}",
        f"D6_LIVE_CONTEXT={payload['D6_LIVE_CONTEXT']}",
        f"FORMAL_STAGE_A_WILL_RECOMPUTE={payload['FORMAL_STAGE_A_WILL_RECOMPUTE']}",
        f"planned_formal_run_root={payload['planned_formal_run_root_pattern']}",
        "TRAINING_STARTED=NO",
        f"FORMAL_PREFLIGHT={'PASS' if ok else 'FAIL'}",
    ]
    if d4_trace:
        lines.extend(
            [
                "",
                "D4_REPRESENTATIVE_TRACE",
                f"sealed_source_path={d4_trace.get('sealed_source_path')}",
                f"sealed_target_path={d4_trace.get('sealed_target_path')}",
                f"source_df_shape={d4_trace.get('source_df_shape')}",
                f"target_df_shape={d4_trace.get('target_df_shape')}",
                f"target_identity={d4_trace.get('target_identity')}",
                f"group_cols={d4_trace.get('group_cols')}",
                f"selected_lr={d4_trace.get('selected_lr')}",
                f"stage_a_key={d4_trace.get('stage_a_key')}",
                f"target_bundle_key={d4_trace.get('target_bundle_key')}",
                "STOP_BEFORE=prepare_ms_katl_stage_a",
            ]
        )
    payload["text"] = "\n".join(lines) + "\n"
    payload["FORMAL_PREFLIGHT"] = "PASS" if ok else "FAIL"
    return payload
