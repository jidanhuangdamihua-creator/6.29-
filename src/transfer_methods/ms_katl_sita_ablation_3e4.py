"""Uniform LR=3e-4 paper-target A–E ablation slice (D1/D4/D5/D6, 5 seeds).

This wrapper does not change the frozen SITA / FUSED_TARGET_ONLY_V3 model,
the A–E arm definitions, or the 270-run / extra-target formal results.
The only intended deltas versus the frozen 270-run are:

1. Stage-B learning_rate is globally 3e-4
2. Paper target mapping (D4 uses 166/258 instead of 166/311)
3. Full seed set {42,43,44,45,46}
4. D2 and D3 are excluded
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    EXPECTED_RUNS,
    HORIZONS,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    SCENARIO,
    SEEDS,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EARLY_STOPPING,
    STAGE_B_EPOCHS,
    UNIFORM_3E4_CANONICAL_CELLS,
    UNIFORM_3E4_DATASETS,
    UNIFORM_3E4_LEARNING_RATE_LABEL,
    UNIFORM_3E4_MANIFEST_RELPATH,
    UNIFORM_3E4_MATRIX_KIND,
    UNIFORM_3E4_OUTPUT_MARKER,
    UNIFORM_3E4_OUTPUT_RELPATH,
    UNIFORM_3E4_RUNS,
    UNIFORM_3E4_RUNS_PER_DATASET,
    UNIFORM_3E4_RUNS_PER_DATASET_HORIZON,
    UNIFORM_3E4_SEEDS,
    UNIFORM_3E4_TARGETS,
    V3_LR_MAP,
    AEContractError,
    AERunIdentity,
    AETrainingBlocked,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_training_recipe,
    assert_d_equals_frozen_v3,
    canonical_targets,
    load_ae_manifest,
    project_root_from,
    resolve_uniform_3e4_lr,
)
from src.transfer_methods.ms_katl_ae_execution import (
    REQUESTED_K,
    enumerate_real_arm_plans,
    enumerate_uniform_3e4_cell_plans,
    stage_a_key,
    write_planned_matrix_csv,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_lite import CONV_LAYER_NAMES
from src.transfer_methods.ms_katl_primary_target_lr_screening import format_lr, parse_lr
from src.transfer_methods.ms_katl_v3_frame_assembly import WINDOW_SIZE
from src.utils.experiment_hyperparams import FIXED_LEARNING_RATE

SLICE_LABEL = "SITA_ABLATION_3E4_PAPER_5SEED"
FORMAL_ABLATION_RUNNER = "scripts/run_ms_katl_ae_ablation.py"
FORMAL_ABLATION_MODULE = "src/transfer_methods/ms_katl_ae_ablation.py"
FORMAL_CONFIG = PARENT_METHOD
FORMAL_MANIFEST_RELPATH = Path("configs") / "ms_katl_ae_ablation_manifest.json"
WRAPPER_RUNNER = "scripts/run_ms_katl_sita_ablation_3e4.py"
UNLOCK_ENV = "AE_FORMAL_TRAINING_UNLOCK"
WEIGHT_MODE = "inverse_distance"
PROTECTED_OUTPUT_MARKERS = (
    "MS_KATL_AE_CORE_ABLATION",
    "MS_KATL_AE_ABLATION",
    "FTO_THREE_TRACK",
    "FTO_V3_SCREENING",
    "PRIMARY_TARGET_LR_SCREENING",
    "SITA_ALL_TARGETS",
    "PAPER_CANDIDATE_BASELINE",
    "sita_ablation_lr_sensitivity",
)
RESULT_FIELDNAMES = (
    "dataset",
    "target",
    "horizon",
    "seed",
    "arm",
    "arm_name",
    "learning_rate",
    "learning_rate_label",
    "RMSE",
    "sMAPE",
    "rmse_original",
    "smape_original",
    "validation_metric",
    "run_status",
    "config_identity",
    "pair_status",
    "scenario",
    "matrix_kind",
    "stage_a_key",
)


class Uniform3e4ContractError(AEContractError):
    """Fail-closed uniform 3e-4 slice violation."""


def select_uniform_3e4_cell(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    arm: str,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one formal matrix row without loading data or training models."""
    dataset_key = str(dataset).strip().upper()
    arm_key = str(arm).strip().upper()
    if dataset_key not in UNIFORM_3E4_TARGETS:
        raise Uniform3e4ContractError(f"dataset {dataset_key} is outside corrected uniform A-E matrix")
    if int(horizon) not in HORIZONS:
        raise Uniform3e4ContractError(f"horizon must be one of {HORIZONS}")
    if int(seed) not in UNIFORM_3E4_SEEDS:
        raise Uniform3e4ContractError(f"seed must be one of {UNIFORM_3E4_SEEDS}")
    if arm_key not in AE_ARMS:
        raise Uniform3e4ContractError(f"arm must be one of {AE_ARMS}")
    resolved_target = str(UNIFORM_3E4_TARGETS[dataset_key])
    if target is not None and str(target) != resolved_target:
        if dataset_key == "D4":
            raise Uniform3e4ContractError("D4 target must be 166/258")
        raise Uniform3e4ContractError(
            f"target for {dataset_key} must be {resolved_target}, not {target}"
        )
    lr_value, lr_label = resolve_uniform_3e4_lr(dataset_key, resolved_target)
    return {
        "dataset": dataset_key,
        "target": resolved_target,
        "horizon": int(horizon),
        "seed": int(seed),
        "arm": arm_key,
        "selected_lr": float(lr_value),
        "selected_lr_label": lr_label,
        "source_lr": float(STAGE_A_LEARNING_RATE),
        "stage_a_key": stage_a_key(dataset_key, resolved_target, int(horizon), int(seed)),
    }


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def default_output_dir(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / UNIFORM_3E4_OUTPUT_RELPATH


def assert_output_isolated(run_dir: Path, project_root: Optional[Path] = None) -> Path:
    root = project_root_from(project_root)
    resolved = Path(run_dir)
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    posix = resolved.as_posix()
    name = resolved.name
    protected_default = (root / UNIFORM_3E4_OUTPUT_RELPATH).resolve()
    if resolved == protected_default:
        raise Uniform3e4ContractError(
            f"refusing corrected execution in historical uniform root: {resolved}"
        )
    if resolved.parent == (root / "outputs").resolve() and not name.startswith(
        "sita_ablation_3e4_corrected_stage_a_rng_uniform_"
    ):
        raise Uniform3e4ContractError(
            "corrected formal root must be timestamped sita_ablation_3e4_corrected_stage_a_rng_uniform_<timestamp>"
        )
    if UNIFORM_3E4_OUTPUT_MARKER not in posix:
        raise Uniform3e4ContractError(f"uniform 3e-4 output must include {UNIFORM_3E4_OUTPUT_MARKER}: {resolved}")
    for marker in PROTECTED_OUTPUT_MARKERS:
        if marker in name:
            raise Uniform3e4ContractError(
                f"refusing to write uniform 3e-4 outputs into protected path: {resolved}"
            )
    runs_root = (root / "outputs" / "runs").resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError:
        pass
    else:
        raise Uniform3e4ContractError("refusing to write uniform 3e-4 outputs under outputs/runs/")
    frozen_manifest = (root / FORMAL_MANIFEST_RELPATH).resolve()
    if resolved == frozen_manifest or resolved == frozen_manifest.parent:
        raise Uniform3e4ContractError("refusing to overwrite the frozen A–E manifest")
    return resolved


def arm_config_payload(arm: str) -> Dict[str, Any]:
    spec = ae_arm_spec(arm)
    optimizer = ae_optimizer_contract()
    recipe = ae_training_recipe()
    return {
        "arm": spec.arm,
        "arm_name": spec.name,
        "init_mode": spec.init_mode,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "revin_enabled": spec.revin_enabled,
        "transfer_branch_enabled": spec.transfer_branch_enabled,
        "gate_mode": spec.gate_mode,
        "use_adversarial": spec.use_adversarial,
        "recipe": recipe,
        "optimizer_frozen": {
            "optimizer": optimizer["optimizer"],
            "clipnorm": optimizer["clipnorm"],
            "epochs": optimizer["epochs"],
            "batch_size": optimizer["batch_size"],
            "stage_b_early_stopping": optimizer["stage_b_early_stopping"],
            "stage_a_learning_rate": optimizer["stage_a_learning_rate"],
        },
        "k": REQUESTED_K,
        "window_size": WINDOW_SIZE,
        "weight_mode": WEIGHT_MODE,
        "conv_transfer_scope": list(CONV_LAYER_NAMES),
        "scenario": SCENARIO,
        "parent_method": PARENT_METHOD,
        "parent_v3_architecture_digest": v3_architecture_digest(),
        "protocol_version": PROTOCOL_VERSION,
        "stage_b_learning_rate": UNIFORM_3E4_LEARNING_RATE_LABEL,
        "matrix_kind": UNIFORM_3E4_MATRIX_KIND,
    }


def config_identity(arm: str) -> str:
    return _json_digest(arm_config_payload(arm))


def assert_formal_ae_identity_frozen(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root_from(project_root)
    manifest = load_ae_manifest(root)
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise Uniform3e4ContractError("frozen V3 architecture digest drifted")
    d_flags = assert_d_equals_frozen_v3()
    if tuple(manifest["arms"]) != AE_ARMS:
        raise Uniform3e4ContractError("formal A–E arm set drifted")
    if tuple(manifest["horizons"]) != HORIZONS:
        raise Uniform3e4ContractError("formal horizons drifted from {1,3,5}")
    if tuple(int(seed) for seed in manifest["seeds"]) != SEEDS or SEEDS != (42, 43, 44):
        raise Uniform3e4ContractError("frozen formal seed set drifted from {42,43,44}")
    if manifest["scenario"] != SCENARIO or SCENARIO != "without":
        raise Uniform3e4ContractError("formal scenario must remain without")
    if int(manifest["expected_runs"]) != EXPECTED_RUNS or EXPECTED_RUNS != 270:
        raise Uniform3e4ContractError("frozen 270-run size drifted")
    if dict(V3_LR_MAP) != {
        ("D1", "1/10"): "3e-5",
        ("D2", "1/10"): "3e-5",
        ("D3", "10"): "3e-5",
        ("D4", "166/311"): "3e-5",
        ("D5", "48/1159414"): "3e-4",
        ("D6", "CA_1/FOODS_3_080"): "3e-5",
    }:
        raise Uniform3e4ContractError("frozen V3_LR_MAP drifted")
    frozen_targets = canonical_targets()
    if ("D4", "166/311") not in frozen_targets:
        raise Uniform3e4ContractError("frozen primary D4 target must remain 166/311")
    if ("D4", "166/258") in frozen_targets:
        raise Uniform3e4ContractError("166/258 must not enter the frozen 270-run target list")
    recipe = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    if recipe["loss_name"] != "huber" or recipe["lr_schedule"] != "cosine" or not recipe["use_ema"]:
        raise Uniform3e4ContractError("frozen AE recipe drifted")
    if optimizer["epochs"] != STAGE_B_EPOCHS or STAGE_B_EPOCHS != 50:
        raise Uniform3e4ContractError("Stage-B epochs drifted")
    if optimizer["batch_size"] != STAGE_B_BATCH_SIZE or STAGE_B_BATCH_SIZE != 16:
        raise Uniform3e4ContractError("Stage-B batch size drifted")
    if bool(optimizer["stage_b_early_stopping"]) is not False or STAGE_B_EARLY_STOPPING:
        raise Uniform3e4ContractError("Stage-B early stopping must remain OFF")
    if float(optimizer["stage_a_learning_rate"]) != float(STAGE_A_LEARNING_RATE) or float(
        STAGE_A_LEARNING_RATE
    ) != float(FIXED_LEARNING_RATE):
        raise Uniform3e4ContractError("Stage-A LR must remain frozen at 1e-4")
    if REQUESTED_K != 3:
        raise Uniform3e4ContractError("K must remain 3")
    if tuple(CONV_LAYER_NAMES) != ("conv1", "conv2", "conv3"):
        raise Uniform3e4ContractError("conv transfer scope drifted")
    if FORMAL_METHODS != (
        "No-TL",
        "SS-TL",
        "MSWA-TL",
        "MSSB-TL",
        "MSML-TL",
        "MSML-TL-RFE",
    ):
        raise Uniform3e4ContractError("FORMAL_METHODS drifted")
    specs = {arm: ae_arm_spec(arm) for arm in AE_ARMS}
    if specs["A"].init_mode != "native" or not specs["A"].target_backbone_trainable or specs["A"].revin_enabled:
        raise Uniform3e4ContractError("arm A identity drifted")
    if specs["B"].init_mode != "fused" or specs["B"].target_backbone_trainable or specs["B"].revin_enabled:
        raise Uniform3e4ContractError("arm B identity drifted")
    if specs["C"].init_mode != "fused" or not specs["C"].target_backbone_trainable or specs["C"].revin_enabled:
        raise Uniform3e4ContractError("arm C identity drifted")
    if specs["D"].init_mode != "fused" or not specs["D"].target_backbone_trainable or not specs["D"].revin_enabled:
        raise Uniform3e4ContractError("arm D identity drifted")
    if specs["E"].init_mode != "best_source" or not specs["E"].target_backbone_trainable or specs["E"].revin_enabled:
        raise Uniform3e4ContractError("arm E identity drifted")
    return {
        "FORMAL_ABLATION_RUNNER": FORMAL_ABLATION_RUNNER,
        "FORMAL_CONFIG": FORMAL_CONFIG,
        "FORMAL_MANIFEST": str((root / FORMAL_MANIFEST_RELPATH).resolve()),
        "protocol_version": PROTOCOL_VERSION,
        "parent_v3_architecture_digest": digest,
        "ARM_D_PARENT_V3_IDENTITY": d_flags["ARM_D_PARENT_V3_IDENTITY"],
        "arm_payloads": {arm: arm_config_payload(arm) for arm in AE_ARMS},
        "config_identities": {arm: config_identity(arm) for arm in AE_ARMS},
        "frozen_targets": [{"dataset": dataset, "target_identity": target} for dataset, target in frozen_targets],
        "paper_targets": dict(UNIFORM_3E4_TARGETS),
        "formal_v3_lr_map": {
            dataset: {"target_identity": target, "learning_rate": label}
            for (dataset, target), label in V3_LR_MAP.items()
        },
    }


def enumerate_uniform_3e4_identities() -> List[AERunIdentity]:
    identities: List[AERunIdentity] = []
    for dataset in UNIFORM_3E4_DATASETS:
        target = UNIFORM_3E4_TARGETS[dataset]
        lr_value, lr_label = resolve_uniform_3e4_lr(dataset, target)
        for horizon in HORIZONS:
            for seed in UNIFORM_3E4_SEEDS:
                for arm in AE_ARMS:
                    identities.append(
                        AERunIdentity(
                            arm=arm,
                            dataset=dataset,
                            target_identity=target,
                            horizon=int(horizon),
                            seed=int(seed),
                            selected_lr=float(lr_value),
                            selected_lr_label=lr_label,
                        )
                    )
    return identities


def validate_uniform_3e4_matrix(
    identities: Sequence[AERunIdentity],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    _ = project_root
    if len(identities) != UNIFORM_3E4_RUNS:
        raise Uniform3e4ContractError(
            f"DRY_RUN_TOTAL_RUNS={len(identities)} != EXPECTED_TOTAL_RUNS={UNIFORM_3E4_RUNS}"
        )
    unique = {item.identity_tuple for item in identities}
    if len(unique) != UNIFORM_3E4_RUNS:
        raise Uniform3e4ContractError("duplicate uniform 3e-4 identities")
    datasets = {item.dataset for item in identities}
    if datasets != set(UNIFORM_3E4_DATASETS) or "D2" in datasets or "D3" in datasets:
        raise Uniform3e4ContractError(f"dataset set drifted: {sorted(datasets)}")
    targets = {(item.dataset, item.target_identity) for item in identities}
    expected_targets = {(dataset, UNIFORM_3E4_TARGETS[dataset]) for dataset in UNIFORM_3E4_DATASETS}
    if targets != expected_targets:
        raise Uniform3e4ContractError(f"paper targets drifted: {sorted(targets)}")
    extra_targets = {
        (item.dataset, item.target_identity)
        for item in identities
        if (item.dataset, item.target_identity) not in expected_targets
    }
    if extra_targets:
        raise Uniform3e4ContractError(f"unexpected targets present: {sorted(extra_targets)}")
    if {int(item.horizon) for item in identities} != set(HORIZONS):
        raise Uniform3e4ContractError("horizons drifted from {1,3,5}")
    if {int(item.seed) for item in identities} != set(UNIFORM_3E4_SEEDS):
        raise Uniform3e4ContractError("seeds drifted from {42,43,44,45,46}")
    if {item.arm for item in identities} != set(AE_ARMS):
        raise Uniform3e4ContractError("arms drifted from A–E")
    lr_labels = {item.selected_lr_label for item in identities}
    if lr_labels != {UNIFORM_3E4_LEARNING_RATE_LABEL}:
        raise Uniform3e4ContractError(f"learning_rate labels drifted: {sorted(lr_labels)}")
    expected_lr = float(parse_lr(UNIFORM_3E4_LEARNING_RATE_LABEL))
    if any(float(item.selected_lr) != expected_lr for item in identities):
        raise Uniform3e4ContractError("learning_rate values drifted from 3e-4")
    if any(format_lr(item.selected_lr) != UNIFORM_3E4_LEARNING_RATE_LABEL for item in identities):
        raise Uniform3e4ContractError("learning_rate failed format_lr(3e-4) round-trip")
    by_dataset: Dict[str, int] = {}
    by_dh: Dict[Tuple[str, int], int] = {}
    by_dhs: Dict[Tuple[str, int, int], set] = {}
    for item in identities:
        by_dataset[item.dataset] = by_dataset.get(item.dataset, 0) + 1
        dh = (item.dataset, int(item.horizon))
        by_dh[dh] = by_dh.get(dh, 0) + 1
        dhs = (item.dataset, int(item.horizon), int(item.seed))
        by_dhs.setdefault(dhs, set()).add(item.arm)
    if any(count != UNIFORM_3E4_RUNS_PER_DATASET for count in by_dataset.values()):
        raise Uniform3e4ContractError(f"per-dataset runs drifted: {by_dataset}")
    if any(count != UNIFORM_3E4_RUNS_PER_DATASET_HORIZON for count in by_dh.values()):
        raise Uniform3e4ContractError(f"per-dataset×horizon runs drifted: {by_dh}")
    if any(arms != set(AE_ARMS) for arms in by_dhs.values()):
        raise Uniform3e4ContractError("each dataset×horizon×seed must contain arms A–E")
    if len(by_dhs) != UNIFORM_3E4_CANONICAL_CELLS:
        raise Uniform3e4ContractError("canonical cell count drifted")
    return {
        "DATASETS": 4,
        "HORIZONS": "1,3,5",
        "SEEDS": "42,43,44,45,46",
        "ARMS": "A,B,C,D,E",
        "LEARNING_RATE": UNIFORM_3E4_LEARNING_RATE_LABEL,
        "EXPECTED_D1_RUNS": UNIFORM_3E4_RUNS_PER_DATASET,
        "EXPECTED_D4_RUNS": UNIFORM_3E4_RUNS_PER_DATASET,
        "EXPECTED_D5_RUNS": UNIFORM_3E4_RUNS_PER_DATASET,
        "EXPECTED_D6_RUNS": UNIFORM_3E4_RUNS_PER_DATASET,
        "EXPECTED_TOTAL_RUNS": UNIFORM_3E4_RUNS,
        "DRY_RUN_TOTAL_RUNS": len(identities),
        "UNIQUE_RUN_KEYS": len(unique),
        "UNIQUE_STAGE_A_KEYS": UNIFORM_3E4_CANONICAL_CELLS,
        "NO_D2": "YES",
        "NO_D3": "YES",
        "NO_OTHER_TARGET": "YES",
        "NO_DUPLICATE_CELL": "YES",
        "NO_MISSING_CELL": "YES",
        "ALL_LR_3E4": "YES",
        "MATRIX_OK": "YES" if len(identities) == UNIFORM_3E4_RUNS else "NO",
    }


def slice_manifest_payload(
    identities: Sequence[AERunIdentity],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    flags = assert_formal_ae_identity_frozen(project_root)
    report = validate_uniform_3e4_matrix(identities, project_root=project_root)
    return {
        "label": SLICE_LABEL,
        "matrix_kind": UNIFORM_3E4_MATRIX_KIND,
        "protocol_version": PROTOCOL_VERSION,
        "formal_ablation_runner": FORMAL_ABLATION_RUNNER,
        "formal_ablation_module": FORMAL_ABLATION_MODULE,
        "formal_config": FORMAL_CONFIG,
        "parent_method": PARENT_METHOD,
        "parent_v3_architecture_digest": flags["parent_v3_architecture_digest"],
        "frozen_270_untouched": True,
        "extra_target_slice_untouched": True,
        "sita_model_changed": False,
        "ablation_definition_changed": False,
        "data_protocol_changed": False,
        "target_mapping_changed": True,
        "only_global_lr_changed": True,
        "learning_rate": UNIFORM_3E4_LEARNING_RATE_LABEL,
        "datasets": list(UNIFORM_3E4_DATASETS),
        "paper_targets": dict(UNIFORM_3E4_TARGETS),
        "horizons": list(HORIZONS),
        "seeds": list(UNIFORM_3E4_SEEDS),
        "arms": list(AE_ARMS),
        "arm_names": dict(AE_ARM_NAMES),
        "scenario": SCENARIO,
        "k": REQUESTED_K,
        "window_size": WINDOW_SIZE,
        "weight_mode": WEIGHT_MODE,
        "conv_transfer_scope": list(CONV_LAYER_NAMES),
        "recipe": ae_training_recipe(),
        "optimizer": ae_optimizer_contract(),
        "stage_a_learning_rate": float(STAGE_A_LEARNING_RATE),
        "expected_canonical_cells": UNIFORM_3E4_CANONICAL_CELLS,
        "expected_runs": UNIFORM_3E4_RUNS,
        "runs_per_dataset": UNIFORM_3E4_RUNS_PER_DATASET,
        "runs_per_dataset_horizon": UNIFORM_3E4_RUNS_PER_DATASET_HORIZON,
        "planned_matrix_digest": _json_digest(
            {
                "runs": [
                    {
                        "dataset": item.dataset,
                        "target": item.target_identity,
                        "horizon": int(item.horizon),
                        "seed": int(item.seed),
                        "arm": item.arm,
                        "lr": item.selected_lr_label,
                    }
                    for item in identities
                ],
                "n": len(identities),
                "label": SLICE_LABEL,
            }
        ),
        "matrix_report": report,
        "config_identities": flags["config_identities"],
        "formal_v3_lr_map_untouched": flags["formal_v3_lr_map"],
        "output_relpath": str(UNIFORM_3E4_OUTPUT_RELPATH),
    }


def write_slice_manifest(project_root: Optional[Path] = None) -> Path:
    root = project_root_from(project_root)
    identities = enumerate_uniform_3e4_identities()
    payload = slice_manifest_payload(identities, project_root=root)
    path = root / UNIFORM_3E4_MANIFEST_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def format_dry_run(
    *,
    identities: Sequence[AERunIdentity],
    report: Mapping[str, Any],
    flags: Mapping[str, Any],
    run_dir: Path,
    ready: str,
) -> str:
    lines = [
        "============================================================",
        "SITA / V3 A–E UNIFORM LR=3e-4 PAPER-TARGET 5-SEED DRY-RUN",
        "============================================================",
        "",
        f"FORMAL_ABLATION_RUNNER={FORMAL_ABLATION_RUNNER}",
        f"FORMAL_CONFIG={FORMAL_CONFIG}",
        f"FORMAL_PROTOCOL_VERSION={PROTOCOL_VERSION}",
        f"FORMAL_MANIFEST={flags['FORMAL_MANIFEST']}",
        f"PARENT_V3_ARCHITECTURE_DIGEST={flags['parent_v3_architecture_digest']}",
        f"MATRIX_KIND={UNIFORM_3E4_MATRIX_KIND}",
        "",
        "D1_TARGET=1/10",
        "D4_TARGET=166/258",
        "D5_TARGET=48/1159414",
        "D6_TARGET=CA_1/FOODS_3_080",
        "",
        "DATASET_COUNT=4",
        "HORIZONS=1,3,5",
        "SEEDS=42,43,44,45,46",
        "ARMS=A,B,C,D,E",
        f"LEARNING_RATE={UNIFORM_3E4_LEARNING_RATE_LABEL}",
        "",
        f"EXPECTED_D1_RUNS={UNIFORM_3E4_RUNS_PER_DATASET}",
        f"EXPECTED_D4_RUNS={UNIFORM_3E4_RUNS_PER_DATASET}",
        f"EXPECTED_D5_RUNS={UNIFORM_3E4_RUNS_PER_DATASET}",
        f"EXPECTED_D6_RUNS={UNIFORM_3E4_RUNS_PER_DATASET}",
        "",
        f"EXPECTED_TOTAL_RUNS={UNIFORM_3E4_RUNS}",
        f"DRY_RUN_TOTAL_RUNS={report['DRY_RUN_TOTAL_RUNS']}",
        "",
        "CODE_CHANGE_REQUIRED=YES",
        "SITA_MODEL_CHANGED=NO",
        "ABLATION_DEFINITION_CHANGED=NO",
        "DATA_PROTOCOL_CHANGED=NO",
        "TARGET_MAPPING_CHANGED=YES",
        "ONLY_GLOBAL_LR_CHANGED=YES",
        "FORMAL_RESULTS_OVERWRITTEN=NO",
        "FROZEN_270_UNTOUCHED=YES",
        "EXTRA_TARGET_SLICE_UNTOUCHED=YES",
        "NO_D2=YES",
        "NO_D3=YES",
        "NO_OTHER_TARGET=YES",
        "NO_DUPLICATE_CELL=YES",
        "NO_MISSING_CELL=YES",
        "ALL_LR_3E4=YES",
        "",
        f"OUTPUT_DIR={run_dir}",
        f"TRAINING_STARTED=NO",
        f"READY_FOR_FULL_RUN={ready}",
        "",
        f"UNIQUE_RUN_KEYS={report['UNIQUE_RUN_KEYS']}",
        f"UNIQUE_STAGE_A_KEYS={report['UNIQUE_STAGE_A_KEYS']}",
        f"ARM_D_PARENT_V3_IDENTITY={flags['ARM_D_PARENT_V3_IDENTITY']}",
    ]
    _ = identities
    return "\n".join(lines) + "\n"


def _is_dry_run_snapshot(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("label") or "") == SLICE_LABEL


def relocate_dry_run_snapshot(run_dir: Path) -> bool:
    """Move a dry-run snapshot out of the formal resume path.

    Dry-run uses a different planned_matrix_digest schema than the formal
    A–E body. Occupying manifest_snapshot.json would fail execute with
    'resume planned matrix digest mismatch'.
    """
    manifest_path = Path(run_dir) / "manifest_snapshot.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _is_dry_run_snapshot(payload):
        return False
    destination = Path(run_dir) / "dry_run_manifest.json"
    if destination.is_file():
        manifest_path.unlink()
        return True
    os.replace(manifest_path, destination)
    return True


def bootstrap_slice_root(
    run_dir: Path,
    identities: Sequence[AERunIdentity],
    *,
    project_root: Optional[Path] = None,
) -> Path:
    out = assert_output_isolated(run_dir, project_root=project_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "cell_bundles").mkdir(parents=True, exist_ok=True)
    relocate_dry_run_snapshot(out)
    snapshot = slice_manifest_payload(identities, project_root=project_root)
    manifest_path = out / "dry_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("label", "matrix_kind", "expected_runs", "parent_v3_architecture_digest"):
            if existing.get(key) != snapshot.get(key):
                raise Uniform3e4ContractError(f"dry-run manifest identity mismatch: {key}")
    else:
        atomic_write_json(manifest_path, snapshot)
    return out


def prepare_uniform_3e4_execute_root(
    run_dir: Path,
    *,
    skip_existing: bool = True,
    project_root: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    out = assert_output_isolated(run_dir, project_root=project_root)
    out.mkdir(parents=True, exist_ok=True)
    relocated = relocate_dry_run_snapshot(out)
    _ = relocated
    resume_root: Optional[Path] = None
    snapshot_path = out / "manifest_snapshot.json"
    if skip_existing and snapshot_path.is_file():
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if _is_dry_run_snapshot(payload):
            raise Uniform3e4ContractError("dry-run snapshot still occupies formal resume path")
        resume_root = out
    return out, resume_root


def write_uniform_3e4_results_table(run_root: Path) -> Path:
    root = assert_output_isolated(run_root)
    bundle_dir = root / "cell_bundles"
    bundle_paths = list(bundle_dir.glob("*.json"))
    bundle_paths.extend(root.glob("cells/*/attempt_*/result.json"))
    rows: List[Dict[str, Any]] = []
    identities = {(item.dataset, item.target_identity, int(item.horizon), int(item.seed), item.arm): item for item in enumerate_uniform_3e4_identities()}
    for path in sorted(set(bundle_paths)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arms = payload.get("arms") or {}
        for arm in AE_ARMS:
            arm_payload = arms.get(arm) or {}
            dataset = str(payload.get("dataset"))
            target = str(payload.get("target"))
            horizon = int(payload.get("horizon"))
            seed = int(payload.get("seed"))
            identity = identities.get((dataset, target, horizon, seed, arm))
            lr_label = UNIFORM_3E4_LEARNING_RATE_LABEL
            if identity is not None:
                lr_label = identity.selected_lr_label
            status = str(arm_payload.get("status") or payload.get("pair_status") or "UNKNOWN")
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "seed": seed,
                    "arm": arm,
                    "arm_name": AE_ARM_NAMES.get(arm, arm),
                    "learning_rate": lr_label,
                    "learning_rate_label": lr_label,
                    "RMSE": arm_payload.get("rmse_original"),
                    "sMAPE": arm_payload.get("smape_original"),
                    "rmse_original": arm_payload.get("rmse_original"),
                    "smape_original": arm_payload.get("smape_original"),
                    "validation_metric": arm_payload.get("validation_metric")
                    if arm_payload.get("validation_metric") is not None
                    else "",
                    "run_status": status,
                    "config_identity": config_identity(arm),
                    "pair_status": payload.get("pair_status"),
                    "scenario": payload.get("scenario", SCENARIO),
                    "matrix_kind": UNIFORM_3E4_MATRIX_KIND,
                    "stage_a_key": payload.get("stage_a_key"),
                }
            )
    destination = root / "ae_uniform_3e4_results.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def run_uniform_3e4_dry_run(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    resolve_real_cells: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    flags = assert_formal_ae_identity_frozen(root)
    identities = enumerate_uniform_3e4_identities()
    report = validate_uniform_3e4_matrix(identities, project_root=root)
    dry_total = int(report["DRY_RUN_TOTAL_RUNS"])
    ready = "YES" if dry_total == UNIFORM_3E4_RUNS else "NO"
    if dry_total != UNIFORM_3E4_RUNS:
        raise Uniform3e4ContractError(
            f"DRY_RUN_TOTAL_RUNS={dry_total} != 300; stopping, must not start training"
        )
    run_dir = bootstrap_slice_root(output_dir or default_output_dir(root), identities, project_root=root)
    manifest_path = write_slice_manifest(root)
    if resolve_real_cells:
        cells = enumerate_uniform_3e4_cell_plans(project_root=root)
        arm_plans = enumerate_real_arm_plans(cells, expected_runs=UNIFORM_3E4_RUNS)
        write_planned_matrix_csv(run_dir / "plans" / "planned_matrix.csv", arm_plans, expected_runs=UNIFORM_3E4_RUNS)
        report["REAL_CELL_RESOLVE"] = "PASS"
        report["CANONICAL_CELLS_RESOLVED"] = len(cells)
    else:
        report["REAL_CELL_RESOLVE"] = "SKIPPED"
        report["CANONICAL_CELLS_RESOLVED"] = 0
    text = format_dry_run(identities=identities, report=report, flags=flags, run_dir=run_dir, ready=ready)
    (run_dir / "logs" / "dry_run.txt").write_text(text, encoding="utf-8")
    atomic_write_json(
        run_dir / "dry_run_report.json",
        {
            "flags": {
                key: value
                for key, value in flags.items()
                if key not in {"arm_payloads"}
            },
            "report": report,
            "ready_for_full_run": ready,
            "slice_manifest": str(manifest_path),
        },
    )
    return {
        "text": text,
        "report": report,
        "flags": flags,
        "output_root": str(run_dir),
        "slice_manifest": str(manifest_path),
        "DRY_RUN_TOTAL_RUNS": dry_total,
        "EXPECTED_TOTAL_RUNS": UNIFORM_3E4_RUNS,
        "READY_FOR_FULL_RUN": ready,
        "TRAINING_STARTED": "NO",
        "CODE_CHANGE_REQUIRED": "YES",
    }


def run_uniform_3e4_plan_only(
    *,
    dataset: Optional[str] = None,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
    arm: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the complete uniform matrix without writing artifacts or training."""
    identities = enumerate_uniform_3e4_identities()
    report = validate_uniform_3e4_matrix(identities)
    selector_values = (dataset, horizon, seed, arm)
    if any(value is not None for value in selector_values) and not all(value is not None for value in selector_values):
        raise Uniform3e4ContractError("single-cell selector requires dataset, horizon, seed, and arm")
    selected = None
    if all(value is not None for value in selector_values):
        selected = select_uniform_3e4_cell(
            dataset=str(dataset), horizon=int(horizon), seed=int(seed), arm=str(arm)
        )
    return {
        **report,
        "PLAN_ONLY": "YES",
        "TRAINING_STARTED": "NO",
        "SINGLE_CELL_GATE_CLI_READY": "PASS",
        "SINGLE_CELL_GATE_USES_FORMAL_RUNTIME": "YES",
        "selected_cell": selected,
    }


def run_uniform_3e4_execute(
    *,
    output_dir: Optional[Path] = None,
    resume_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    skip_existing: bool = True,
    selected_dataset: Optional[str] = None,
    selected_horizon: Optional[int] = None,
    selected_seed: Optional[int] = None,
    selected_arm: Optional[str] = None,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    if output_dir is not None and resume_root is not None:
        raise Uniform3e4ContractError("provide either --output-root or --resume, not both")
    selector_values = (selected_dataset, selected_horizon, selected_seed, selected_arm)
    if any(value is not None for value in selector_values) and not all(value is not None for value in selector_values):
        raise Uniform3e4ContractError("single-cell selector requires dataset, horizon, seed, and arm")
    if not skip_existing:
        raise Uniform3e4ContractError("COMPLETE attempts are always skipped; --no-skip-existing is forbidden")
    identities = enumerate_uniform_3e4_identities()
    report = validate_uniform_3e4_matrix(identities, project_root=root)
    if int(report["DRY_RUN_TOTAL_RUNS"]) != UNIFORM_3E4_RUNS:
        raise Uniform3e4ContractError("refusing execute: matrix is not 300 runs")
    if output_dir is None and resume_root is None:
        raise Uniform3e4ContractError(
            "corrected execution requires explicit --output-root <new-root> or --resume <existing-root>"
        )
    requested_root = Path(resume_root if resume_root is not None else output_dir)
    run_dir = assert_output_isolated(requested_root, project_root=root)
    if resume_root is None and run_dir.exists():
        raise Uniform3e4ContractError(
            f"new corrected output root already exists; use --resume explicitly: {run_dir}"
        )
    all_cells = enumerate_uniform_3e4_cell_plans(project_root=root)
    selected_key: Optional[str] = None
    if all(value is not None for value in selector_values):
        selected = select_uniform_3e4_cell(
            dataset=str(selected_dataset),
            horizon=int(selected_horizon),
            seed=int(selected_seed),
            arm=str(selected_arm),
        )
        selected_key = str(selected["stage_a_key"])
        if sum(cell.stage_a_key == selected_key for cell in all_cells) != 1:
            raise Uniform3e4ContractError(f"selected cell not found in formal matrix: {selected_key}")
    from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_body

    result = run_ae_formal_body(
        execute=True,
        environ=environ,
        cells=all_cells,
        matrix_kind=UNIFORM_3E4_MATRIX_KIND,
        output_root=run_dir,
        resume_root=resume_root,
        project_root=root,
        hardened_attempts=True,
        selected_stage_a_key=selected_key,
        selected_arm=str(selected_arm).upper() if selected_arm is not None else None,
    )
    result["DRY_RUN_TOTAL_RUNS"] = int(report["DRY_RUN_TOTAL_RUNS"])
    result["EXPECTED_TOTAL_RUNS"] = UNIFORM_3E4_RUNS
    result["PLAN_ONLY"] = "NO"
    result["SINGLE_CELL_SELECTOR"] = "YES" if selected_key is not None else "NO"
    result["SELECTED_STAGE_A_KEY"] = selected_key
    return result
