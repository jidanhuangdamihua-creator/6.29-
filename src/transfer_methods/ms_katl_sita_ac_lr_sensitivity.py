"""Arm A vs Arm C Stage-B learning-rate sensitivity screening.

This wrapper does not change the frozen SITA / FUSED_TARGET_ONLY_V3 model,
the A–E ablation definitions, or the 270-run formal results. It only sweeps
Stage-B target-training learning rates for arms A and C on the frozen
canonical targets, seed 42, horizons {1,3,5}.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    HORIZONS as AE_FORMAL_HORIZONS,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    SCENARIO,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EARLY_STOPPING,
    STAGE_B_EPOCHS,
    V3_LR_MAP,
    AEContractError,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_training_recipe,
    canonical_targets,
    load_ae_manifest,
    project_root_from,
    resolve_cell_lr,
)
from src.transfer_methods.ms_katl_ae_execution import (
    REQUESTED_K,
    RealCanonicalCellPlan,
    resolve_real_canonical_cell,
    stage_a_key,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_lite import CONV_LAYER_NAMES
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    LR_GRID_LABELS,
    format_lr,
    parse_lr,
)
from src.transfer_methods.ms_katl_v3_frame_assembly import WINDOW_SIZE
from src.utils.experiment_hyperparams import FIXED_LEARNING_RATE

SCREENING_LABEL = "SITA_ABLATION_LR_SENSITIVITY_AC"
FORMAL_ABLATION_RUNNER = "scripts/run_ms_katl_ae_ablation.py"
FORMAL_ABLATION_MODULE = "src/transfer_methods/ms_katl_ae_ablation.py"
FORMAL_CONFIG = PARENT_METHOD
FORMAL_MANIFEST_RELPATH = Path("configs") / "ms_katl_ae_ablation_manifest.json"
FORMAL_PROTOCOL_VERSION = PROTOCOL_VERSION
SCREENING_ARMS: Tuple[str, ...] = ("A", "C")
SCREENING_HORIZONS: Tuple[int, ...] = (1, 3, 5)
SCREENING_SEEDS: Tuple[int, ...] = (42,)
SCREENING_LR_LABELS: Tuple[str, ...] = LR_GRID_LABELS
SCREENING_LR_VALUES: Tuple[float, ...] = LR_GRID
SCREENING_DATASETS: Tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")
EXPECTED_DATASETS = 6
EXPECTED_HORIZONS = 3
EXPECTED_SEEDS = 1
EXPECTED_ARMS = 2
EXPECTED_LEARNING_RATES = 3
EXPECTED_TOTAL_RUNS = 108
RUNS_PER_DATASET = 18
RUNS_PER_DATASET_HORIZON = 6
UNIQUE_STAGE_A_CELLS = 18
DEFAULT_OUTPUT_RELPATH = Path("outputs") / "sita_ablation_lr_sensitivity_ac_seed42"
UNLOCK_ENV = "SITA_AC_LR_SCREENING_UNLOCK"
VALIDATION_METRIC_NAME = "validation_rmse_original"
STAGE_B_LR_SCOPE = "STAGE_B_TARGET_TRAINING"
WEIGHT_MODE = "inverse_distance"
RMSE_TIE_ABS = 1e-12
PROTECTED_OUTPUT_MARKERS = (
    "MS_KATL_AE_CORE_ABLATION",
    "MS_KATL_AE_ABLATION",
    "FTO_THREE_TRACK",
    "FTO_V3_SCREENING",
    "PRIMARY_TARGET_LR_SCREENING",
    "SITA_ALL_TARGETS",
    "PAPER_CANDIDATE_BASELINE",
)
FORBIDDEN_ARMS = ("B", "D", "E")
RESULT_FIELDNAMES = (
    "dataset",
    "target",
    "horizon",
    "seed",
    "arm",
    "arm_name",
    "learning_rate",
    "learning_rate_label",
    "formal_lr_label",
    "validation_metric",
    "validation_rmse_original",
    "validation_smape_original",
    "RMSE",
    "sMAPE",
    "rmse_original",
    "smape_original",
    "run_status",
    "config_identity",
    "init_mode",
    "target_backbone_trainable",
    "revin_enabled",
    "scenario",
    "mode",
    "stage_a_key",
    "run_id",
    "error",
)


class LRSensitivityContractError(AEContractError):
    """Fail-closed A/C LR-sensitivity protocol violation."""


class LRSensitivityTrainingBlocked(RuntimeError):
    """Execute was requested without the screening unlock."""


@dataclass(frozen=True)
class LRSensitivityRun:
    dataset: str
    target_identity: str
    horizon: int
    seed: int
    arm: str
    learning_rate: float
    learning_rate_label: str
    formal_lr_label: str
    scenario: str = SCENARIO

    @property
    def identity_tuple(self) -> Tuple[Any, ...]:
        return (
            self.dataset,
            self.target_identity,
            int(self.horizon),
            int(self.seed),
            self.arm,
            self.learning_rate_label,
        )

    @property
    def stage_a_tuple(self) -> Tuple[Any, ...]:
        return (self.dataset, self.target_identity, int(self.horizon), int(self.seed), self.scenario)

    @property
    def condition_tuple(self) -> Tuple[Any, ...]:
        return (self.dataset, self.target_identity, int(self.horizon))

    @property
    def run_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        return (
            f"{self.arm}_{self.dataset}_{target_slug}"
            f"_h{int(self.horizon)}_s{int(self.seed)}_{self.learning_rate_label}"
        )

    @property
    def stage_a_key(self) -> str:
        return stage_a_key(self.dataset, self.target_identity, self.horizon, self.seed, self.scenario)


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


def frozen_target_mapping(project_root: Optional[Path] = None) -> Tuple[Tuple[str, str], ...]:
    root = project_root_from(project_root)
    manifest = load_ae_manifest(root)
    manifest_targets = tuple(
        (str(item["dataset"]), str(item["target_identity"])) for item in manifest["targets"]
    )
    code_targets = canonical_targets()
    if manifest_targets != code_targets:
        raise LRSensitivityContractError(
            f"AE manifest targets drifted from code canonical_targets: {manifest_targets} != {code_targets}"
        )
    if code_targets != tuple(V3_LR_MAP.keys()):
        raise LRSensitivityContractError("canonical targets drifted from V3_LR_MAP")
    return code_targets


def formal_lr_label(dataset: str, target_identity: str) -> str:
    _, label = resolve_cell_lr(dataset, target_identity)
    if label not in SCREENING_LR_LABELS:
        raise LRSensitivityContractError(
            f"frozen formal LR {label} for {(dataset, target_identity)} is outside the screening grid"
        )
    return label


def arm_config_payload(arm: str) -> Dict[str, Any]:
    key = str(arm).strip().upper()
    if key not in SCREENING_ARMS:
        raise LRSensitivityContractError(f"screening forbids arm {arm!r}; allowed={list(SCREENING_ARMS)}")
    spec = ae_arm_spec(key)
    if key == "A":
        if spec.init_mode != "native" or not spec.target_backbone_trainable or spec.revin_enabled:
            raise LRSensitivityContractError("arm A identity drifted from native + trainable + RevIN OFF")
    if key == "C":
        if spec.init_mode != "fused" or not spec.target_backbone_trainable or spec.revin_enabled:
            raise LRSensitivityContractError("arm C identity drifted from fused + trainable + RevIN OFF")
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
        "stage_b_lr_scope": STAGE_B_LR_SCOPE,
    }


def config_identity(arm: str) -> str:
    return _json_digest(arm_config_payload(arm))


def assert_formal_ae_identity_frozen(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root_from(project_root)
    manifest = load_ae_manifest(root)
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise LRSensitivityContractError("frozen V3 architecture digest drifted")
    if manifest["parent_method"] != PARENT_METHOD or PARENT_METHOD != FORMAL_CONFIG:
        raise LRSensitivityContractError("formal parent method drifted")
    if tuple(manifest["arms"]) != AE_ARMS:
        raise LRSensitivityContractError("formal A–E arm set drifted")
    if tuple(manifest["horizons"]) != AE_FORMAL_HORIZONS or AE_FORMAL_HORIZONS != SCREENING_HORIZONS:
        raise LRSensitivityContractError("formal horizons drifted from {1,3,5}")
    if tuple(int(seed) for seed in manifest["seeds"]) != (42, 43, 44):
        raise LRSensitivityContractError("formal seed set drifted; screening still uses seed 42 only")
    if manifest["scenario"] != SCENARIO or SCENARIO != "without":
        raise LRSensitivityContractError("formal scenario must remain without")
    recipe = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    if recipe["loss_name"] != "huber" or recipe["lr_schedule"] != "cosine" or not recipe["use_ema"]:
        raise LRSensitivityContractError("frozen AE recipe drifted")
    if optimizer["epochs"] != STAGE_B_EPOCHS or STAGE_B_EPOCHS != 50:
        raise LRSensitivityContractError("Stage-B epochs drifted")
    if optimizer["batch_size"] != STAGE_B_BATCH_SIZE or STAGE_B_BATCH_SIZE != 16:
        raise LRSensitivityContractError("Stage-B batch size drifted")
    if bool(optimizer["stage_b_early_stopping"]) is not False or STAGE_B_EARLY_STOPPING:
        raise LRSensitivityContractError("Stage-B early stopping must remain OFF")
    if float(optimizer["stage_a_learning_rate"]) != float(STAGE_A_LEARNING_RATE) or float(
        STAGE_A_LEARNING_RATE
    ) != float(FIXED_LEARNING_RATE):
        raise LRSensitivityContractError("Stage-A LR must remain frozen at 1e-4")
    if REQUESTED_K != 3:
        raise LRSensitivityContractError("K must remain 3")
    if tuple(CONV_LAYER_NAMES) != ("conv1", "conv2", "conv3"):
        raise LRSensitivityContractError("conv transfer scope drifted")
    a_payload = arm_config_payload("A")
    c_payload = arm_config_payload("C")
    if a_payload["init_mode"] == c_payload["init_mode"]:
        raise LRSensitivityContractError("A vs C must differ in initialization")
    if a_payload["target_backbone_trainable"] is not True or c_payload["target_backbone_trainable"] is not True:
        raise LRSensitivityContractError("A and C must keep the target backbone trainable")
    if a_payload["revin_enabled"] or c_payload["revin_enabled"]:
        raise LRSensitivityContractError("A and C must keep RevIN OFF")
    return {
        "FORMAL_ABLATION_RUNNER": FORMAL_ABLATION_RUNNER,
        "FORMAL_CONFIG": FORMAL_CONFIG,
        "FORMAL_MANIFEST": str((root / FORMAL_MANIFEST_RELPATH).resolve()),
        "protocol_version": PROTOCOL_VERSION,
        "parent_v3_architecture_digest": digest,
        "arm_a": a_payload,
        "arm_c": c_payload,
        "targets": [{"dataset": dataset, "target_identity": target} for dataset, target in frozen_target_mapping(root)],
        "formal_v3_lr_map": {
            dataset: {"target_identity": target, "learning_rate": label}
            for (dataset, target), label in V3_LR_MAP.items()
        },
        "A_CONFIG_IDENTITY": config_identity("A"),
        "C_CONFIG_IDENTITY": config_identity("C"),
    }


def default_output_dir(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / DEFAULT_OUTPUT_RELPATH


def assert_output_isolated(run_dir: Path, project_root: Optional[Path] = None) -> Path:
    root = project_root_from(project_root)
    resolved = Path(run_dir)
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    posix = resolved.as_posix()
    name = resolved.name
    for marker in PROTECTED_OUTPUT_MARKERS:
        if marker in name or marker in posix:
            raise LRSensitivityContractError(
                f"refusing to write LR-sensitivity outputs into protected path: {resolved}"
            )
    runs_root = (root / "outputs" / "runs").resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError:
        pass
    else:
        raise LRSensitivityContractError("refusing to write LR-sensitivity outputs under outputs/runs/")
    formal_manifest = (root / FORMAL_MANIFEST_RELPATH).resolve()
    if resolved == formal_manifest or resolved == formal_manifest.parent:
        raise LRSensitivityContractError("refusing to overwrite the frozen A–E manifest")
    return resolved


def enumerate_lr_sensitivity_runs(project_root: Optional[Path] = None) -> List[LRSensitivityRun]:
    if SCREENING_LR_LABELS != ("3e-5", "1e-4", "3e-4"):
        raise LRSensitivityContractError("LR grid labels drifted")
    if SCREENING_LR_VALUES != (3e-5, 1e-4, 3e-4):
        raise LRSensitivityContractError("LR grid values drifted")
    if tuple(parse_lr(label) for label in SCREENING_LR_LABELS) != SCREENING_LR_VALUES:
        raise LRSensitivityContractError("LR grid parse drifted")
    if SCREENING_ARMS != ("A", "C"):
        raise LRSensitivityContractError("screening arms must remain A,C")
    if SCREENING_HORIZONS != (1, 3, 5):
        raise LRSensitivityContractError("screening horizons must remain 1,3,5")
    if SCREENING_SEEDS != (42,):
        raise LRSensitivityContractError("screening seed must remain 42")
    identities: List[LRSensitivityRun] = []
    for dataset, target in frozen_target_mapping(project_root):
        frozen_label = formal_lr_label(dataset, target)
        for horizon in SCREENING_HORIZONS:
            for seed in SCREENING_SEEDS:
                for arm in SCREENING_ARMS:
                    for lr_label in SCREENING_LR_LABELS:
                        identities.append(
                            LRSensitivityRun(
                                dataset=dataset,
                                target_identity=target,
                                horizon=int(horizon),
                                seed=int(seed),
                                arm=arm,
                                learning_rate=float(parse_lr(lr_label)),
                                learning_rate_label=lr_label,
                                formal_lr_label=frozen_label,
                            )
                        )
    return identities


def validate_lr_sensitivity_matrix(
    identities: Sequence[LRSensitivityRun],
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    targets = frozen_target_mapping(project_root)
    if len(identities) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError(
            f"planned runs={len(identities)} != EXPECTED_TOTAL_RUNS={EXPECTED_TOTAL_RUNS}"
        )
    unique = {item.identity_tuple for item in identities}
    if len(unique) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError("duplicate A/C LR-sensitivity identities")
    datasets = tuple(dict.fromkeys(item.dataset for item in identities))
    if datasets != SCREENING_DATASETS:
        raise LRSensitivityContractError(f"datasets drifted: {datasets}")
    if {item.horizon for item in identities} != set(SCREENING_HORIZONS):
        raise LRSensitivityContractError("extra or missing horizons")
    if {item.seed for item in identities} != set(SCREENING_SEEDS):
        raise LRSensitivityContractError("extra or missing seeds")
    if {item.arm for item in identities} != set(SCREENING_ARMS):
        raise LRSensitivityContractError("arms drifted from A,C")
    forbidden = {item.arm for item in identities} & set(FORBIDDEN_ARMS)
    if forbidden:
        raise LRSensitivityContractError(f"matrix contains forbidden arms: {sorted(forbidden)}")
    if {item.learning_rate_label for item in identities} != set(SCREENING_LR_LABELS):
        raise LRSensitivityContractError("extra or missing learning rates")
    observed_targets = tuple(dict.fromkeys((item.dataset, item.target_identity) for item in identities))
    if observed_targets != targets:
        raise LRSensitivityContractError("target mapping drifted from frozen A–E ablation")
    by_dataset: Dict[str, int] = {}
    by_condition: Dict[Tuple[str, int], int] = {}
    for item in identities:
        by_dataset[item.dataset] = by_dataset.get(item.dataset, 0) + 1
        by_condition[(item.dataset, int(item.horizon))] = by_condition.get((item.dataset, int(item.horizon)), 0) + 1
    if any(count != RUNS_PER_DATASET for count in by_dataset.values()) or len(by_dataset) != EXPECTED_DATASETS:
        raise LRSensitivityContractError(f"per-dataset run counts drifted: {by_dataset}")
    if any(count != RUNS_PER_DATASET_HORIZON for count in by_condition.values()) or len(by_condition) != 18:
        raise LRSensitivityContractError(f"per-dataset×horizon run counts drifted: {by_condition}")
    stage_a_keys = {item.stage_a_key for item in identities}
    if len(stage_a_keys) != UNIQUE_STAGE_A_CELLS:
        raise LRSensitivityContractError(f"Stage-A cells={len(stage_a_keys)} != {UNIQUE_STAGE_A_CELLS}")
    for item in identities:
        if format_lr(item.learning_rate) != item.learning_rate_label:
            raise LRSensitivityContractError(f"LR label mismatch: {item.run_id}")
        if item.scenario != SCENARIO:
            raise LRSensitivityContractError("scenario must remain without")
    return {
        "DATASETS": EXPECTED_DATASETS,
        "HORIZONS": EXPECTED_HORIZONS,
        "SEEDS": EXPECTED_SEEDS,
        "ARMS": EXPECTED_ARMS,
        "LEARNING_RATES": EXPECTED_LEARNING_RATES,
        "EXPECTED_TOTAL_RUNS": EXPECTED_TOTAL_RUNS,
        "DRY_RUN_TOTAL_RUNS": len(identities),
        "UNIQUE_RUN_KEYS": len(unique),
        "UNIQUE_STAGE_A_KEYS": len(stage_a_keys),
        "RUNS_PER_DATASET": dict(by_dataset),
        "RUNS_PER_DATASET_HORIZON": {f"{dataset}_h{horizon}": count for (dataset, horizon), count in by_condition.items()},
        "FORBIDDEN_ARMS_PRESENT": "NO",
        "EXTRA_SEED_PRESENT": "NO",
        "EXTRA_HORIZON_PRESENT": "NO",
        "EXTRA_LR_PRESENT": "NO",
        "TARGET_MAPPING_MATCHES_FORMAL_AE": "YES",
        "MATRIX_OK": "YES",
    }


def planned_matrix_digest(identities: Sequence[LRSensitivityRun]) -> str:
    payload = [
        {
            "dataset": item.dataset,
            "target": item.target_identity,
            "horizon": int(item.horizon),
            "seed": int(item.seed),
            "arm": item.arm,
            "lr": item.learning_rate_label,
        }
        for item in identities
    ]
    return _json_digest({"runs": payload, "n": len(payload), "label": SCREENING_LABEL})


def screening_manifest_payload(
    identities: Sequence[LRSensitivityRun],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    flags = assert_formal_ae_identity_frozen(project_root)
    report = validate_lr_sensitivity_matrix(identities, project_root=project_root)
    return {
        "label": SCREENING_LABEL,
        "matrix_kind": SCREENING_LABEL,
        "formal_ablation_runner": FORMAL_ABLATION_RUNNER,
        "formal_ablation_module": FORMAL_ABLATION_MODULE,
        "formal_config": FORMAL_CONFIG,
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "parent_v3_architecture_digest": flags["parent_v3_architecture_digest"],
        "targets": flags["targets"],
        "arms": list(SCREENING_ARMS),
        "arm_names": {arm: AE_ARM_NAMES[arm] for arm in SCREENING_ARMS},
        "horizons": list(SCREENING_HORIZONS),
        "seeds": list(SCREENING_SEEDS),
        "learning_rates": list(SCREENING_LR_LABELS),
        "scenario": SCENARIO,
        "stage_a_learning_rate": float(STAGE_A_LEARNING_RATE),
        "stage_b_lr_scope": STAGE_B_LR_SCOPE,
        "k": REQUESTED_K,
        "window_size": WINDOW_SIZE,
        "weight_mode": WEIGHT_MODE,
        "recipe": ae_training_recipe(),
        "optimizer": ae_optimizer_contract(),
        "validation_metric": VALIDATION_METRIC_NAME,
        "lr_selection_uses_test": False,
        "expected_runs": EXPECTED_TOTAL_RUNS,
        "planned_matrix_digest": planned_matrix_digest(identities),
        "A_CONFIG_IDENTITY": flags["A_CONFIG_IDENTITY"],
        "C_CONFIG_IDENTITY": flags["C_CONFIG_IDENTITY"],
        "formal_v3_lr_map": flags["formal_v3_lr_map"],
        "SITA_MODEL_CHANGED": "NO",
        "ABLATION_DEFINITION_CHANGED": "NO",
        "DATA_PROTOCOL_CHANGED": "NO",
        "ONLY_LEARNING_RATE_VARIED": "YES",
        "TEST_BASED_LR_SELECTION": "NO",
        "FORMAL_RESULTS_OVERWRITTEN": "NO",
        **report,
    }


def write_planned_matrix_csv(path: Path, identities: Sequence[LRSensitivityRun]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(identities) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError("planned matrix size drifted")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "target",
                "horizon",
                "seed",
                "arm",
                "arm_name",
                "learning_rate",
                "formal_lr_label",
                "stage_a_key",
                "run_id",
                "config_identity",
                "status",
            ),
        )
        writer.writeheader()
        for item in identities:
            spec = ae_arm_spec(item.arm)
            writer.writerow(
                {
                    "dataset": item.dataset,
                    "target": item.target_identity,
                    "horizon": int(item.horizon),
                    "seed": int(item.seed),
                    "arm": item.arm,
                    "arm_name": spec.name,
                    "learning_rate": item.learning_rate_label,
                    "formal_lr_label": item.formal_lr_label,
                    "stage_a_key": item.stage_a_key,
                    "run_id": item.run_id,
                    "config_identity": config_identity(item.arm),
                    "status": "PLANNED",
                }
            )
    return destination


def bootstrap_screening_root(
    run_dir: Path,
    identities: Sequence[LRSensitivityRun],
    *,
    project_root: Optional[Path] = None,
) -> Path:
    out = assert_output_isolated(run_dir, project_root=project_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    snapshot = screening_manifest_payload(identities, project_root=project_root)
    manifest_path = out / "manifest_snapshot.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("label", "planned_matrix_digest", "expected_runs", "parent_v3_architecture_digest"):
            if existing.get(key) != snapshot.get(key):
                raise LRSensitivityContractError(f"resume manifest identity mismatch: {key}")
    else:
        atomic_write_json(manifest_path, snapshot)
    plan_path = out / "plans" / "planned_matrix.csv"
    if not plan_path.is_file():
        write_planned_matrix_csv(plan_path, identities)
    return out


def planned_row(item: LRSensitivityRun) -> Dict[str, Any]:
    spec = ae_arm_spec(item.arm)
    return {
        "dataset": item.dataset,
        "target": item.target_identity,
        "horizon": int(item.horizon),
        "seed": int(item.seed),
        "arm": item.arm,
        "arm_name": spec.name,
        "learning_rate": item.learning_rate_label,
        "learning_rate_label": item.learning_rate_label,
        "formal_lr_label": item.formal_lr_label,
        "validation_metric": None,
        "validation_rmse_original": None,
        "validation_smape_original": None,
        "RMSE": None,
        "sMAPE": None,
        "rmse_original": None,
        "smape_original": None,
        "run_status": "PLANNED",
        "config_identity": config_identity(item.arm),
        "init_mode": spec.init_mode,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "revin_enabled": spec.revin_enabled,
        "scenario": item.scenario,
        "mode": SCENARIO,
        "stage_a_key": item.stage_a_key,
        "run_id": item.run_id,
        "error": "",
    }


def cell_artifact_path(run_dir: Path, item: LRSensitivityRun) -> Path:
    return Path(run_dir) / "cells" / f"{item.run_id}.json"


def load_completed_ok_runs(run_dir: Path, identities: Sequence[LRSensitivityRun]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    completed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in identities:
        path = cell_artifact_path(run_dir, item)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("run_status") or "").upper() == "OK":
            completed[item.identity_tuple] = payload
    return completed


def resolve_screening_cell_plan(
    item: LRSensitivityRun,
    *,
    knn_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    project_root: Optional[Path] = None,
) -> RealCanonicalCellPlan:
    return resolve_real_canonical_cell(
        item.dataset,
        item.target_identity,
        item.horizon,
        item.seed,
        knn_cache=knn_cache,
        project_root=project_root,
        selected_lr=float(item.learning_rate),
        selected_lr_label=item.learning_rate_label,
    )


def resolve_all_screening_cell_plans(
    identities: Sequence[LRSensitivityRun],
    *,
    project_root: Optional[Path] = None,
) -> List[RealCanonicalCellPlan]:
    knn_cache: Dict[str, Dict[str, Any]] = {}
    seen: Dict[Tuple[Any, ...], RealCanonicalCellPlan] = {}
    plans: List[RealCanonicalCellPlan] = []
    for item in identities:
        key = (*item.stage_a_tuple, item.learning_rate_label)
        if key not in seen:
            seen[key] = resolve_screening_cell_plan(item, knn_cache=knn_cache, project_root=project_root)
        plans.append(seen[key])
    if len(plans) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError("resolved cell plans drifted from 108")
    return plans


def format_dry_run(
    *,
    identities: Sequence[LRSensitivityRun],
    report: Mapping[str, Any],
    flags: Mapping[str, Any],
    run_dir: Path,
) -> str:
    target_lines = [
        f"{dataset:<4} {target}" for dataset, target in ((row["dataset"], row["target_identity"]) for row in flags["targets"])
    ]
    lines = [
        "============================================================",
        "SITA / V3 ARM A vs ARM C LEARNING-RATE SENSITIVITY DRY-RUN",
        "============================================================",
        "",
        f"FORMAL_ABLATION_RUNNER={FORMAL_ABLATION_RUNNER}",
        f"FORMAL_CONFIG={FORMAL_CONFIG}",
        f"FORMAL_PROTOCOL_VERSION={FORMAL_PROTOCOL_VERSION}",
        f"FORMAL_MANIFEST={flags['FORMAL_MANIFEST']}",
        f"PARENT_V3_ARCHITECTURE_DIGEST={flags['parent_v3_architecture_digest']}",
        "",
        "ARM_A_NAME=TARGET_NATIVE_INIT",
        "ARM_A_INIT=native",
        "ARM_A_BACKBONE=trainable",
        "ARM_A_REVIN=OFF",
        f"ARM_A_CONFIG_IDENTITY={flags['A_CONFIG_IDENTITY']}",
        "",
        "ARM_C_NAME=FUSED_TRAINABLE",
        "ARM_C_INIT=fused",
        "ARM_C_BACKBONE=trainable",
        "ARM_C_REVIN=OFF",
        f"ARM_C_CONFIG_IDENTITY={flags['C_CONFIG_IDENTITY']}",
        "",
        "TARGET_MAPPING=",
        *target_lines,
        "",
        f"DATASETS={report['DATASETS']}",
        f"HORIZONS={report['HORIZONS']}",
        f"SEEDS={report['SEEDS']}",
        f"ARMS={report['ARMS']}",
        f"LEARNING_RATES={report['LEARNING_RATES']}",
        "",
        f"EXPECTED_TOTAL_RUNS={report['EXPECTED_TOTAL_RUNS']}",
        f"DRY_RUN_TOTAL_RUNS={report['DRY_RUN_TOTAL_RUNS']}",
        f"UNIQUE_RUN_KEYS={report['UNIQUE_RUN_KEYS']}",
        f"UNIQUE_STAGE_A_KEYS={report['UNIQUE_STAGE_A_KEYS']}",
        "",
        "NO_DUPLICATES=YES",
        "NO_MISSING=YES",
        f"FORBIDDEN_ARMS_PRESENT={report['FORBIDDEN_ARMS_PRESENT']}",
        f"EXTRA_SEED_PRESENT={report['EXTRA_SEED_PRESENT']}",
        f"EXTRA_HORIZON_PRESENT={report['EXTRA_HORIZON_PRESENT']}",
        f"EXTRA_LR_PRESENT={report['EXTRA_LR_PRESENT']}",
        f"TARGET_MAPPING_MATCHES_FORMAL_AE={report['TARGET_MAPPING_MATCHES_FORMAL_AE']}",
        "",
        "SITA_MODEL_CHANGED=NO",
        "ABLATION_DEFINITION_CHANGED=NO",
        "DATA_PROTOCOL_CHANGED=NO",
        "ONLY_LEARNING_RATE_VARIED=YES",
        f"LEARNING_RATE_SCOPE={STAGE_B_LR_SCOPE}",
        "STAGE_A_LEARNING_RATE_FROZEN=1e-4",
        "TEST_BASED_LR_SELECTION=NO",
        "FORMAL_RESULTS_OVERWRITTEN=NO",
        "",
        "ARMS=A,C",
        "LEARNING_RATES=3e-5,1e-4,3e-4",
        "HORIZONS=1,3,5",
        "SEED=42",
        "",
        f"OUTPUT_DIR={run_dir}",
        f"READY_FOR_LR_SCREENING={'YES' if report['MATRIX_OK'] == 'YES' and report['DRY_RUN_TOTAL_RUNS'] == EXPECTED_TOTAL_RUNS else 'NO'}",
        "TRAINING_STARTED=NO",
        "============================================================",
    ]
    return "\n".join(lines) + "\n"


def run_lr_sensitivity_dry_run(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    resolve_real_cells: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    flags = assert_formal_ae_identity_frozen(root)
    identities = enumerate_lr_sensitivity_runs(root)
    report = validate_lr_sensitivity_matrix(identities, project_root=root)
    if int(report["DRY_RUN_TOTAL_RUNS"]) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError("dry-run is not 108 runs; stopping")
    run_dir = assert_output_isolated(output_dir or default_output_dir(root), project_root=root)
    bootstrap_screening_root(run_dir, identities, project_root=root)
    if resolve_real_cells:
        plans = resolve_all_screening_cell_plans(identities, project_root=root)
        if len({(plan.dataset, plan.target, int(plan.horizon), int(plan.seed), plan.selected_lr_label) for plan in plans}) != 54:
            raise LRSensitivityContractError("resolved LR cells drifted from 54 unique dataset×horizon×seed×lr")
        if {plan.scenario for plan in plans} != {SCENARIO}:
            raise LRSensitivityContractError("resolved cells drifted from without")
    write_planned_matrix_csv(run_dir / "plans" / "planned_matrix.csv", identities)
    text = format_dry_run(identities=identities, report=report, flags=flags, run_dir=run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "dry_run.txt").write_text(text, encoding="utf-8")
    return {
        "identities": identities,
        "report": report,
        "flags": flags,
        "output_root": run_dir,
        "text": text,
        "TRAINING_STARTED": "NO",
        "READY_FOR_LR_SCREENING": "YES" if report["DRY_RUN_TOTAL_RUNS"] == EXPECTED_TOTAL_RUNS else "NO",
        "DRY_RUN_TOTAL_RUNS": report["DRY_RUN_TOTAL_RUNS"],
    }


def assert_execute_unlocked(environ: Optional[Mapping[str, str]] = None) -> None:
    env = environ if environ is not None else os.environ
    if str(env.get(UNLOCK_ENV, "")).strip() != "YES":
        raise LRSensitivityTrainingBlocked(
            f"LR screening execute requires {UNLOCK_ENV}=YES"
        )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _relative_improvement(baseline: float, challenger: float) -> float:
    if not math.isfinite(baseline) or baseline == 0.0:
        raise LRSensitivityContractError("relative improvement requires finite non-zero baseline")
    return (float(baseline) - float(challenger)) / float(baseline)


def _win_label(a_value: float, c_value: float) -> str:
    if abs(float(a_value) - float(c_value)) <= RMSE_TIE_ABS:
        return "TIE"
    if float(c_value) < float(a_value):
        return "C"
    return "A"


def select_lr_by_validation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok_rows = [dict(row) for row in rows if str(row.get("run_status", "")).upper() == "OK"]
    if len(ok_rows) != len(SCREENING_LR_LABELS):
        raise LRSensitivityContractError("validation LR selection requires all three LR runs OK")
    for row in ok_rows:
        if not _finite(row.get(VALIDATION_METRIC_NAME)):
            raise LRSensitivityContractError("validation LR selection requires finite validation RMSE")
        if "rmse_original" in row and row.get("rmse_original") is not None:
            # Selection key must remain validation; test RMSE may exist but is ignored here.
            pass
    ranked = sorted(
        ok_rows,
        key=lambda row: (float(row[VALIDATION_METRIC_NAME]), parse_lr(row["learning_rate_label"])),
    )
    chosen = ranked[0]
    return {
        **chosen,
        "best_validation_lr": chosen["learning_rate_label"],
        "best_validation_metric": float(chosen[VALIDATION_METRIC_NAME]),
    }


def _rows_by_arm_lr(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["arm"]), str(row["learning_rate_label"]))
        if key in index:
            raise LRSensitivityContractError(f"duplicate arm/LR row for {row}")
        index[key] = dict(row)
    return index


def summarize_fixed_lr_comparison(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_condition: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["target"]), int(row["horizon"]))
        by_condition.setdefault(key, []).append(dict(row))
    summaries: List[Dict[str, Any]] = []
    for lr_label in SCREENING_LR_LABELS:
        wins = losses = ties = 0
        rel_rmse: List[float] = []
        rel_smape: List[float] = []
        for condition_rows in by_condition.values():
            index = _rows_by_arm_lr(condition_rows)
            a_row = index[("A", lr_label)]
            c_row = index[("C", lr_label)]
            if str(a_row.get("run_status")) != "OK" or str(c_row.get("run_status")) != "OK":
                continue
            winner = _win_label(float(a_row["rmse_original"]), float(c_row["rmse_original"]))
            if winner == "C":
                wins += 1
            elif winner == "A":
                losses += 1
            else:
                ties += 1
            rel_rmse.append(_relative_improvement(float(a_row["rmse_original"]), float(c_row["rmse_original"])))
            rel_smape.append(_relative_improvement(float(a_row["smape_original"]), float(c_row["smape_original"])))
        summaries.append(
            {
                "learning_rate": lr_label,
                "conditions_compared": wins + losses + ties,
                "c_wins": wins,
                "a_wins": losses,
                "ties": ties,
                "mean_relative_rmse_improvement": statistics.mean(rel_rmse) if rel_rmse else None,
                "median_relative_rmse_improvement": statistics.median(rel_rmse) if rel_rmse else None,
                "mean_relative_smape_improvement": statistics.mean(rel_smape) if rel_smape else None,
                "median_relative_smape_improvement": statistics.median(rel_smape) if rel_smape else None,
            }
        )
    return summaries


def summarize_validation_selected(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["target"]), int(row["horizon"]))
        by_condition.setdefault(key, []).append(dict(row))
    c_wins = a_wins = ties = 0
    rel_rmse: List[float] = []
    rel_smape: List[float] = []
    dataset_direction: Dict[str, List[str]] = {}
    condition_rows: List[Dict[str, Any]] = []
    a_formal_not_val_opt = 0
    a_improved = 0
    c_improved = 0
    direction_flips = 0
    for key, condition_rows_raw in sorted(by_condition.items()):
        index = _rows_by_arm_lr(condition_rows_raw)
        a_group = [index[("A", lr_label)] for lr_label in SCREENING_LR_LABELS if ("A", lr_label) in index]
        c_group = [index[("C", lr_label)] for lr_label in SCREENING_LR_LABELS if ("C", lr_label) in index]
        a_best = select_lr_by_validation(a_group)
        c_best = select_lr_by_validation(c_group)
        winner = _win_label(float(a_best["rmse_original"]), float(c_best["rmse_original"]))
        if winner == "C":
            c_wins += 1
        elif winner == "A":
            a_wins += 1
        else:
            ties += 1
        rel_rmse.append(_relative_improvement(float(a_best["rmse_original"]), float(c_best["rmse_original"])))
        rel_smape.append(_relative_improvement(float(a_best["smape_original"]), float(c_best["smape_original"])))
        dataset_direction.setdefault(key[0], []).append(winner)
        formal_lr = str(a_group[0]["formal_lr_label"])
        a_formal = index[("A", formal_lr)]
        c_formal = index[("C", formal_lr)]
        if a_best["best_validation_lr"] != formal_lr:
            a_formal_not_val_opt += 1
        if float(a_best["rmse_original"]) < float(a_formal["rmse_original"]) - RMSE_TIE_ABS:
            a_improved += 1
        if float(c_best["rmse_original"]) < float(c_formal["rmse_original"]) - RMSE_TIE_ABS:
            c_improved += 1
        formal_winner = _win_label(float(a_formal["rmse_original"]), float(c_formal["rmse_original"]))
        if winner != formal_winner:
            direction_flips += 1
        condition_rows.append(
            {
                "dataset": key[0],
                "target": key[1],
                "horizon": key[2],
                "A_best_validation_lr": a_best["best_validation_lr"],
                "C_best_validation_lr": c_best["best_validation_lr"],
                "A_formal_lr": formal_lr,
                "A_val_rmse": a_best["best_validation_metric"],
                "C_val_rmse": c_best["best_validation_metric"],
                "A_test_rmse": float(a_best["rmse_original"]),
                "C_test_rmse": float(c_best["rmse_original"]),
                "A_test_smape": float(a_best["smape_original"]),
                "C_test_smape": float(c_best["smape_original"]),
                "winner": winner,
                "formal_winner": formal_winner,
            }
        )
    dataset_summary = {
        dataset: max(set(winners), key=winners.count) if winners else "NA"
        for dataset, winners in dataset_direction.items()
    }
    return {
        "conditions": 18,
        "c_wins": c_wins,
        "a_wins": a_wins,
        "ties": ties,
        "mean_relative_rmse_improvement": statistics.mean(rel_rmse) if rel_rmse else None,
        "median_relative_rmse_improvement": statistics.median(rel_rmse) if rel_rmse else None,
        "mean_relative_smape_improvement": statistics.mean(rel_smape) if rel_smape else None,
        "median_relative_smape_improvement": statistics.median(rel_smape) if rel_smape else None,
        "dataset_level_direction": dataset_summary,
        "arm_a_formal_lr_not_validation_optimal": a_formal_not_val_opt,
        "arm_a_improved_after_val_lr": a_improved,
        "arm_c_improved_after_val_lr": c_improved,
        "c_vs_a_direction_flips": direction_flips,
        "condition_rows": condition_rows,
    }


def classify_sensitivity_case(validation_summary: Mapping[str, Any], fixed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    a_not_opt = int(validation_summary.get("arm_a_formal_lr_not_validation_optimal") or 0)
    a_improved = int(validation_summary.get("arm_a_improved_after_val_lr") or 0)
    c_improved = int(validation_summary.get("arm_c_improved_after_val_lr") or 0)
    flips = int(validation_summary.get("c_vs_a_direction_flips") or 0)
    formal_lr_rows = [item for item in fixed if item["learning_rate"] in SCREENING_LR_LABELS]
    selected_mean = validation_summary.get("mean_relative_rmse_improvement")
    # Compare selected-mean against the mean of the three fixed-LR comparisons as a coarse robustness check.
    fixed_means = [item.get("mean_relative_rmse_improvement") for item in formal_lr_rows if item.get("mean_relative_rmse_improvement") is not None]
    case = "INDETERMINATE"
    if a_improved >= 10 and c_improved >= 10 and flips <= 3:
        case = "CASE_1"
    elif a_improved >= 10 and a_improved > c_improved + 2 and (selected_mean is not None and selected_mean < (statistics.mean(fixed_means) if fixed_means else selected_mean)):
        case = "CASE_2"
    elif c_improved >= a_improved and (selected_mean is not None and selected_mean >= 0):
        case = "CASE_3"
    elif a_improved < 9 and c_improved < 9 and flips <= 3:
        case = "CASE_1"
    return {
        "case": case,
        "arm_a_formal_lr_not_validation_optimal_conditions": a_not_opt,
        "arm_a_improved_conditions": a_improved,
        "arm_c_improved_conditions": c_improved,
        "c_vs_a_direction_flips": flips,
        "notes": (
            "CASE_1=both arms move similarly and C vs A is stable; "
            "CASE_2=A is more LR-sensitive and C vs A shrinks; "
            "CASE_3=C is equally or more sensitive and C vs A holds or expands."
        ),
    }


def load_result_rows(run_dir: Path, identities: Sequence[LRSensitivityRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in identities:
        path = cell_artifact_path(run_dir, item)
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append(planned_row(item))
    return rows


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in RESULT_FIELDNAMES})
    return destination


def analyze_screening_results(run_dir: Path, identities: Sequence[LRSensitivityRun]) -> Dict[str, Any]:
    rows = load_result_rows(run_dir, identities)
    ok_rows = [row for row in rows if str(row.get("run_status", "")).upper() == "OK"]
    if len(ok_rows) != EXPECTED_TOTAL_RUNS:
        return {
            "ANALYSIS_COMPLETE": "NO",
            "OK_RUNS": len(ok_rows),
            "EXPECTED_TOTAL_RUNS": EXPECTED_TOTAL_RUNS,
            "reason": "analysis requires all 108 OK runs",
        }
    fixed = summarize_fixed_lr_comparison(ok_rows)
    selected = summarize_validation_selected(ok_rows)
    case = classify_sensitivity_case(selected, fixed)
    payload = {
        "ANALYSIS_COMPLETE": "YES",
        "fixed_lr_comparison": fixed,
        "validation_selected": selected,
        "case_assessment": case,
        "TEST_BASED_LR_SELECTION": "NO",
        "FORMAL_RESULTS_OVERWRITTEN": "NO",
    }
    results_dir = Path(run_dir) / "results"
    write_results_csv(results_dir / "all_runs.csv", rows)
    atomic_write_json(results_dir / "fixed_lr_comparison.json", {"rows": fixed})
    atomic_write_json(results_dir / "validation_selected_comparison.json", selected)
    atomic_write_json(results_dir / "case_assessment.json", case)
    return payload


def evaluate_split_metrics(
    constructed: Any,
    bundle: Any,
    split: str,
) -> Dict[str, float]:
    from src.transfer_methods.ms_katl_lite import extract_features_maybe_revin, predict_ms_katl_decomposed
    from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics

    x_map = {"train": bundle.X_train, "val": bundle.X_val, "test": bundle.X_test}
    y_map = {"train": bundle.y_train, "val": bundle.y_val, "test": bundle.y_test}
    x = x_map[split]
    y_true = y_map[split]
    features = bundle.f_t.get(split) or []
    if x is None or len(y_true) == 0 or any(item is None for item in features):
        raise LRSensitivityContractError(f"{split} split is empty or missing source features")
    z, mean, stdev = extract_features_maybe_revin(
        constructed.target_extractor,
        x,
        revin=constructed.revin,
        training=False,
    )
    decomposed = predict_ms_katl_decomposed(
        constructed.head,
        z,
        features,
        bundle.stage_a.knn_prior,
    )
    y_pred = decomposed["y_pred"]
    if constructed.revin is not None:
        raise LRSensitivityContractError("A and C must evaluate with RevIN OFF")
    _ = (mean, stdev)
    metrics = compute_screening_metrics(
        y_true,
        y_pred,
        target_scaler=bundle.target_scaler,
        feature_columns=list(bundle.target_feature_columns),
    )
    return {
        "rmse_original": float(metrics["rmse_original"]),
        "smape_original": float(metrics["smape_original"]),
    }


def train_one_screening_run(
    item: LRSensitivityRun,
    *,
    ae_stage: Any,
    bundle: Any,
    head_snapshot: Sequence[Any],
) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_ae_ablation import (
        construct_ae_arm,
        reset_ae_rng,
        select_ae_trainable_variables,
    )
    from src.transfer_methods.ms_katl_lite import train_ms_katl_head

    spec = ae_arm_spec(item.arm)
    reset_ae_rng(int(item.seed))
    constructed = construct_ae_arm(
        spec,
        ae_stage,
        seed=int(item.seed),
        selected_lr=float(item.learning_rate),
        recipe=ae_training_recipe(),
        head_target_snapshot=head_snapshot,
    )
    if constructed.spec.revin_enabled:
        raise LRSensitivityContractError(f"arm {item.arm} must keep RevIN OFF")
    trainable = select_ae_trainable_variables(
        constructed.head,
        constructed.target_extractor,
        constructed.spec,
        constructed.revin,
    )
    opt = ae_optimizer_contract()
    recipe = ae_training_recipe()
    train_ms_katl_head(
        constructed.head,
        x_train=bundle.X_train,
        f_t_train=bundle.f_t["train"],
        y_train=bundle.y_train,
        f_s_pools=bundle.f_s_pools,
        knn_prior=bundle.stage_a.knn_prior,
        target_extractor=constructed.target_extractor,
        epochs=int(opt["epochs"]),
        batch_size=int(opt["batch_size"]),
        learning_rate=float(item.learning_rate),
        seed=int(item.seed),
        use_adversarial=False,
        trainable_variables=trainable,
        revin=constructed.revin,
        **recipe,
    )
    test_metrics = evaluate_split_metrics(constructed, bundle, "test")
    val_metrics = evaluate_split_metrics(constructed, bundle, "val")
    row = planned_row(item)
    row.update(
        {
            "validation_metric": float(val_metrics["rmse_original"]),
            "validation_rmse_original": float(val_metrics["rmse_original"]),
            "validation_smape_original": float(val_metrics["smape_original"]),
            "RMSE": float(test_metrics["rmse_original"]),
            "sMAPE": float(test_metrics["smape_original"]),
            "rmse_original": float(test_metrics["rmse_original"]),
            "smape_original": float(test_metrics["smape_original"]),
            "run_status": "OK",
            "error": "",
        }
    )
    return row


def run_lr_sensitivity_execute(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    assert_execute_unlocked(environ)
    root = project_root_from(project_root)
    flags = assert_formal_ae_identity_frozen(root)
    identities = enumerate_lr_sensitivity_runs(root)
    report = validate_lr_sensitivity_matrix(identities, project_root=root)
    if int(report["DRY_RUN_TOTAL_RUNS"]) != EXPECTED_TOTAL_RUNS:
        raise LRSensitivityContractError("refusing execute: matrix is not 108 runs")
    run_dir = bootstrap_screening_root(output_dir or default_output_dir(root), identities, project_root=root)
    completed = load_completed_ok_runs(run_dir, identities) if skip_existing else {}
    pending = [item for item in identities if item.identity_tuple not in completed]
    from collections import defaultdict

    from src.transfer_methods.ms_katl_ae_ablation import build_ae_stage_a_from_shared_bundle, construct_v3_head_target_snapshot
    from src.transfer_methods.ms_katl_ae_live_context import build_live_ae_cell_context
    from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
    from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle

    grouped: Dict[Tuple[Any, ...], List[LRSensitivityRun]] = defaultdict(list)
    for item in pending:
        grouped[item.stage_a_tuple].append(item)

    knn_cache: Dict[str, Dict[str, Any]] = {}
    failed = 0
    for stage_key, arm_runs in grouped.items():
        print(f"AE_LR_CELL_START {stage_key}", flush=True)
        template = arm_runs[0]
        cell = resolve_screening_cell_plan(template, knn_cache=knn_cache, project_root=root)
        try:
            live = build_live_ae_cell_context(cell, project_root=root)
            artifacts = prepare_ms_katl_stage_a(**live.stage_a_kwargs)
            bundle = prepare_shared_target_bundle(
                artifacts,
                live.target_df,
                list(live.feature_cols),
                horizon=int(cell.horizon),
                window_size=int(live.stage_a_kwargs["window_size"]),
                learning_rate=float(STAGE_A_LEARNING_RATE),
                seed=int(cell.seed),
                transformation_reuse_context=live.stage_a_kwargs.get("transformation_reuse_context"),
            )
            ae_stage = build_ae_stage_a_from_shared_bundle(bundle)
            head_snapshot, _digest = construct_v3_head_target_snapshot(ae_stage, seed=int(template.seed))
            for item in arm_runs:
                path = cell_artifact_path(run_dir, item)
                if skip_existing and item.identity_tuple in completed:
                    print(f"SKIP_OK {item.run_id}", flush=True)
                    continue
                try:
                    payload = train_one_screening_run(
                        item,
                        ae_stage=ae_stage,
                        bundle=bundle,
                        head_snapshot=head_snapshot,
                    )
                    print(
                        f"RUN_OK {item.run_id} RMSE={payload['RMSE']:.6g} "
                        f"VAL={payload['validation_metric']:.6g}",
                        flush=True,
                    )
                except Exception as exc:
                    failed += 1
                    payload = planned_row(item)
                    payload["run_status"] = "FAILED"
                    payload["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"RUN_FAILED {item.run_id} reason={exc}", flush=True)
                atomic_write_json(path, payload)
                if payload["run_status"] == "OK":
                    completed[item.identity_tuple] = payload
        except Exception as exc:
            for item in arm_runs:
                payload = planned_row(item)
                payload["run_status"] = "FAILED"
                payload["error"] = f"{type(exc).__name__}: {exc}"
                atomic_write_json(cell_artifact_path(run_dir, item), payload)
                failed += 1
            print(f"STAGE_A_FAILED {stage_key} reason={exc}", flush=True)

    all_rows = load_result_rows(run_dir, identities)
    write_results_csv(run_dir / "results" / "all_runs.csv", all_rows)
    ok_count = sum(1 for row in all_rows if str(row.get("run_status")) == "OK")
    analysis = None
    if ok_count == EXPECTED_TOTAL_RUNS:
        analysis = analyze_screening_results(run_dir, identities)
    return {
        "output_root": run_dir,
        "TRAINING_STARTED": "YES",
        "OK_RUNS": ok_count,
        "FAILED_RUNS": failed,
        "PENDING_AFTER": EXPECTED_TOTAL_RUNS - ok_count,
        "FORMAL_RESULTS_OVERWRITTEN": "NO",
        "flags": flags,
        "analysis": analysis,
    }
