"""Formal corrected SITA A–E slice under the frozen per-dataset target LR map.

This module is intentionally separate from the retained uniform-3e-4 slice.
It owns the corrected paper-target matrix contract and the read-only D5 reuse
audit. Real execution is delegated to the shared formal A–E authorities by the
runner added alongside this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    HORIZONS,
    SCENARIO,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EPOCHS,
    AERunIdentity,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_training_recipe,
)
from src.transfer_methods.ms_katl_fto_v3_screening import V3_REVIN_CONTRACT


FORMAL_PER_DATASET_LR_MATRIX_KIND = "FORMAL_PER_DATASET_LR_AE"
FORMAL_PER_DATASET_LR_DATASETS: Tuple[str, ...] = ("D1", "D4", "D5", "D6")
FORMAL_PER_DATASET_LR_TARGETS: Mapping[str, str] = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
FORMAL_PER_DATASET_LR_LABELS: Mapping[str, str] = {
    "D1": "3e-5",
    "D4": "3e-5",
    "D5": "3e-4",
    "D6": "3e-5",
}
FORMAL_PER_DATASET_LR_VALUES: Mapping[str, float] = {
    dataset: float(label)
    for dataset, label in FORMAL_PER_DATASET_LR_LABELS.items()
}
FORMAL_PER_DATASET_LR_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
FORMAL_PER_DATASET_LR_CANONICAL_CELLS = (
    len(FORMAL_PER_DATASET_LR_DATASETS)
    * len(HORIZONS)
    * len(FORMAL_PER_DATASET_LR_SEEDS)
)
FORMAL_PER_DATASET_LR_RUNS = FORMAL_PER_DATASET_LR_CANONICAL_CELLS * len(AE_ARMS)
FORMAL_PER_DATASET_LR_RUNS_PER_DATASET = len(HORIZONS) * len(FORMAL_PER_DATASET_LR_SEEDS) * len(AE_ARMS)
FORMAL_PER_DATASET_LR_OUTPUT_MARKER = "sita_ablation_formal_per_dataset_lr_ae"
FORMAL_PER_DATASET_LR_OLD_UNIFORM_MARKER = "sita_ablation_3e4_corrected_stage_a_rng_uniform"


class FormalPerDatasetLRContractError(RuntimeError):
    """Fail-closed formal per-dataset target-LR contract violation."""


def _stage_a_key(dataset: str, target: str, horizon: int, seed: int) -> str:
    return f"{dataset}|{target}|h{int(horizon)}|s{int(seed)}|{SCENARIO}"


def enumerate_formal_per_dataset_lr_identities() -> List[AERunIdentity]:
    identities: List[AERunIdentity] = []
    for dataset in FORMAL_PER_DATASET_LR_DATASETS:
        target = FORMAL_PER_DATASET_LR_TARGETS[dataset]
        label = FORMAL_PER_DATASET_LR_LABELS[dataset]
        value = FORMAL_PER_DATASET_LR_VALUES[dataset]
        for horizon in HORIZONS:
            for seed in FORMAL_PER_DATASET_LR_SEEDS:
                for arm in AE_ARMS:
                    identities.append(
                        AERunIdentity(
                            arm=arm,
                            dataset=dataset,
                            target_identity=target,
                            horizon=int(horizon),
                            seed=int(seed),
                            selected_lr=value,
                            selected_lr_label=label,
                        )
                    )
    return identities


def validate_formal_per_dataset_lr_matrix(
    identities: Sequence[AERunIdentity],
) -> Dict[str, Any]:
    if len(identities) != FORMAL_PER_DATASET_LR_RUNS:
        raise FormalPerDatasetLRContractError(
            f"formal per-dataset LR runs={len(identities)} != {FORMAL_PER_DATASET_LR_RUNS}"
        )
    keys = [item.identity_tuple for item in identities]
    if len(set(keys)) != FORMAL_PER_DATASET_LR_RUNS:
        raise FormalPerDatasetLRContractError("duplicate formal per-dataset LR identities")
    observed_targets = {(item.dataset, item.target_identity) for item in identities}
    expected_targets = set(FORMAL_PER_DATASET_LR_TARGETS.items())
    if observed_targets != expected_targets:
        raise FormalPerDatasetLRContractError(
            f"formal corrected target map drifted: {sorted(observed_targets)}"
        )
    if {int(item.horizon) for item in identities} != set(HORIZONS):
        raise FormalPerDatasetLRContractError("formal horizons must be {1,3,5}")
    if {int(item.seed) for item in identities} != set(FORMAL_PER_DATASET_LR_SEEDS):
        raise FormalPerDatasetLRContractError("formal seeds must be {42,43,44,45,46}")
    if {item.arm for item in identities} != set(AE_ARMS):
        raise FormalPerDatasetLRContractError("formal arms must be A–E")
    for item in identities:
        expected_label = FORMAL_PER_DATASET_LR_LABELS[item.dataset]
        expected_value = FORMAL_PER_DATASET_LR_VALUES[item.dataset]
        if item.selected_lr_label != expected_label or float(item.selected_lr) != expected_value:
            raise FormalPerDatasetLRContractError(
                f"target LR drifted for {item.dataset}: {item.selected_lr_label}"
            )
    return {
        "MATRIX_KIND": FORMAL_PER_DATASET_LR_MATRIX_KIND,
        "EXPECTED_TOTAL_RUNS": FORMAL_PER_DATASET_LR_RUNS,
        "EXPECTED_CANONICAL_CELLS": FORMAL_PER_DATASET_LR_CANONICAL_CELLS,
        "D1_LR": FORMAL_PER_DATASET_LR_LABELS["D1"],
        "D4_LR": FORMAL_PER_DATASET_LR_LABELS["D4"],
        "D5_LR": FORMAL_PER_DATASET_LR_LABELS["D5"],
        "D6_LR": FORMAL_PER_DATASET_LR_LABELS["D6"],
        "DATASETS": list(FORMAL_PER_DATASET_LR_DATASETS),
        "HORIZONS": list(HORIZONS),
        "SEEDS": list(FORMAL_PER_DATASET_LR_SEEDS),
        "ARMS": list(AE_ARMS),
    }


def enumerate_formal_per_dataset_lr_cell_plans(
    *,
    project_root: Optional[Path] = None,
) -> List[Any]:
    """Resolve the 60 real cells through the shared KNN/data authority."""
    from src.transfer_methods.ms_katl_ae_execution import resolve_real_canonical_cell

    cache: Dict[str, Dict[str, Any]] = {}
    cells = []
    for dataset in FORMAL_PER_DATASET_LR_DATASETS:
        target = FORMAL_PER_DATASET_LR_TARGETS[dataset]
        for horizon in HORIZONS:
            for seed in FORMAL_PER_DATASET_LR_SEEDS:
                cells.append(
                    resolve_real_canonical_cell(
                        dataset,
                        target,
                        int(horizon),
                        int(seed),
                        knn_cache=cache,
                        project_root=project_root,
                        selected_lr=FORMAL_PER_DATASET_LR_VALUES[dataset],
                        selected_lr_label=FORMAL_PER_DATASET_LR_LABELS[dataset],
                    )
                )
    assert_formal_per_dataset_lr_slice(cells)
    return cells


def assert_formal_per_dataset_lr_slice(
    cells: Sequence[Any],
    arm_plans: Optional[Sequence[Any]] = None,
) -> None:
    if len(cells) != FORMAL_PER_DATASET_LR_CANONICAL_CELLS:
        raise FormalPerDatasetLRContractError(
            f"formal per-dataset LR cells={len(cells)} != {FORMAL_PER_DATASET_LR_CANONICAL_CELLS}"
        )
    observed = {(cell.dataset, cell.target) for cell in cells}
    expected = set(FORMAL_PER_DATASET_LR_TARGETS.items())
    if observed != expected:
        raise FormalPerDatasetLRContractError(f"formal target map drifted: {sorted(observed)}")
    keys = [cell.stage_a_key for cell in cells]
    if len(set(keys)) != len(keys):
        raise FormalPerDatasetLRContractError("duplicate formal per-dataset Stage-A keys")
    if {int(cell.horizon) for cell in cells} != set(HORIZONS):
        raise FormalPerDatasetLRContractError("formal horizons drifted")
    if {int(cell.seed) for cell in cells} != set(FORMAL_PER_DATASET_LR_SEEDS):
        raise FormalPerDatasetLRContractError("formal seeds drifted")
    for cell in cells:
        if cell.scenario != SCENARIO:
            raise FormalPerDatasetLRContractError("formal scenario must be without")
        expected_label = FORMAL_PER_DATASET_LR_LABELS[cell.dataset]
        if cell.target != FORMAL_PER_DATASET_LR_TARGETS[cell.dataset]:
            raise FormalPerDatasetLRContractError(f"formal target drifted for {cell.dataset}")
        if cell.selected_lr_label != expected_label:
            raise FormalPerDatasetLRContractError(f"formal LR label drifted for {cell.dataset}")
        if float(cell.selected_lr) != FORMAL_PER_DATASET_LR_VALUES[cell.dataset]:
            raise FormalPerDatasetLRContractError(f"formal LR value drifted for {cell.dataset}")
    if arm_plans is not None:
        if len(arm_plans) != FORMAL_PER_DATASET_LR_RUNS:
            raise FormalPerDatasetLRContractError("formal arm-run count drifted")
        if {item.arm for item in arm_plans} != set(AE_ARMS):
            raise FormalPerDatasetLRContractError("formal arm set drifted")


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def formal_arm_config_identity(arm: str, learning_rate_label: str) -> str:
    spec = ae_arm_spec(arm)
    return _json_digest(
        {
            "arm": spec.arm,
            "arm_name": spec.name,
            "init_mode": spec.init_mode,
            "target_backbone_trainable": spec.target_backbone_trainable,
            "revin_enabled": spec.revin_enabled,
            "transfer_branch_enabled": spec.transfer_branch_enabled,
            "gate_mode": spec.gate_mode,
            "use_adversarial": spec.use_adversarial,
            "stage_b_learning_rate": str(learning_rate_label),
            "stage_a_learning_rate": "1e-4",
            "stage_b_epochs": 50,
            "batch_size": 16,
            "ema_decay": 0.99,
            "matrix_kind": FORMAL_PER_DATASET_LR_MATRIX_KIND,
        }
    )


def write_formal_per_dataset_lr_results_table(run_root: Path) -> Path:
    root = Path(run_root)
    result_paths = list((root / "cell_bundles").glob("*.json"))
    result_paths.extend(root.glob("cells/*/attempt_*/result.json"))
    identity_map = {
        (item.dataset, item.target_identity, int(item.horizon), int(item.seed), item.arm): item
        for item in enumerate_formal_per_dataset_lr_identities()
    }
    rows: List[Dict[str, Any]] = []
    for path in sorted(set(result_paths)):
        payload = _json(path)
        arms = payload.get("arms") or {}
        for arm in AE_ARMS:
            arm_payload = arms.get(arm) or {}
            key = (
                str(payload.get("dataset")),
                str(payload.get("target")),
                int(payload.get("horizon")),
                int(payload.get("seed")),
                arm,
            )
            identity = identity_map.get(key)
            if identity is None:
                raise FormalPerDatasetLRContractError(f"result row outside formal matrix: {key}")
            rows.append(
                {
                    "dataset": identity.dataset,
                    "target": identity.target_identity,
                    "horizon": identity.horizon,
                    "seed": identity.seed,
                    "arm": arm,
                    "arm_name": AE_ARM_NAMES[arm],
                    "learning_rate": identity.selected_lr_label,
                    "learning_rate_label": identity.selected_lr_label,
                    "RMSE": arm_payload.get("rmse_original"),
                    "sMAPE": arm_payload.get("smape_original"),
                    "rmse_original": arm_payload.get("rmse_original"),
                    "smape_original": arm_payload.get("smape_original"),
                    "validation_metric": arm_payload.get("validation_metric", ""),
                    "run_status": arm_payload.get("status") or payload.get("status"),
                    "config_identity": formal_arm_config_identity(arm, identity.selected_lr_label),
                    "pair_status": payload.get("pair_status"),
                    "scenario": payload.get("scenario", SCENARIO),
                    "matrix_kind": FORMAL_PER_DATASET_LR_MATRIX_KIND,
                    "stage_a_key": payload.get("stage_a_key") or payload.get("cell_key"),
                    "reuse_status": "REUSED_D5_UNIFORM_3E4_EXACT"
                    if payload.get("reused_in_formal_root")
                    else "TRAINED_FORMAL_PER_DATASET_LR",
                    "source_root": payload.get("reused_from_root", ""),
                }
            )
    destination = root / "ae_formal_per_dataset_lr_results.csv"
    fieldnames = [
        "dataset", "target", "horizon", "seed", "arm", "arm_name",
        "learning_rate", "learning_rate_label", "RMSE", "sMAPE",
        "rmse_original", "smape_original", "validation_metric", "run_status",
        "config_identity", "pair_status", "scenario", "matrix_kind", "stage_a_key",
        "reuse_status", "source_root",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def formal_per_dataset_lr_manifest_payload() -> Dict[str, Any]:
    optimizer = ae_optimizer_contract()
    recipe = ae_training_recipe()
    return {
        "paper_method_name": "SITA",
        "ablation_protocol_name": "FORMAL A–E ABLATION UNDER THE FROZEN PER-DATASET TARGET-LR PROTOCOL",
        "matrix_kind": FORMAL_PER_DATASET_LR_MATRIX_KIND,
        "datasets": list(FORMAL_PER_DATASET_LR_DATASETS),
        "target_mapping": dict(FORMAL_PER_DATASET_LR_TARGETS),
        "target_learning_rates": dict(FORMAL_PER_DATASET_LR_LABELS),
        "horizons": list(HORIZONS),
        "seeds": list(FORMAL_PER_DATASET_LR_SEEDS),
        "arms": {arm: AE_ARM_NAMES[arm] for arm in AE_ARMS},
        "source_selection": {"k": 3, "weight_mode": "inverse_distance"},
        "stage_a_learning_rate": "1e-4",
        "stage_a_epochs": 50,
        "target_epochs": 50,
        "batch_size": 16,
        "optimizer": optimizer,
        "recipe": recipe,
        "ema_decay": 0.99,
        "revin": dict(V3_REVIN_CONTRACT),
        "scenario": SCENARIO,
        "expected_canonical_cells": FORMAL_PER_DATASET_LR_CANONICAL_CELLS,
        "expected_runs": FORMAL_PER_DATASET_LR_RUNS,
        "d5_reuse_allowed_only_after_audit": True,
        "uniform_slice_retained": True,
    }


def assert_formal_per_dataset_lr_output_isolated(
    path: Path,
    *,
    project_root: Optional[Path] = None,
) -> Path:
    root = Path(path).resolve()
    project = Path(project_root or Path.cwd()).resolve()
    if root == project or project not in root.parents:
        raise FormalPerDatasetLRContractError("formal output must be inside project root")
    if "outputs" not in root.parts:
        raise FormalPerDatasetLRContractError("formal per-dataset LR output must be under outputs/")
    if FORMAL_PER_DATASET_LR_OLD_UNIFORM_MARKER in root.name or "sita_ablation_3e4" in root.name:
        raise FormalPerDatasetLRContractError("uniform 3e-4 output cannot be used as formal per-dataset LR output")
    if FORMAL_PER_DATASET_LR_OUTPUT_MARKER not in root.name:
        raise FormalPerDatasetLRContractError(
            f"formal output name must include {FORMAL_PER_DATASET_LR_OUTPUT_MARKER}"
        )
    if "outputs/runs" in root.as_posix().replace("\\", "/"):
        raise FormalPerDatasetLRContractError("formal per-dataset LR output must not use outputs/runs")
    return root


def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalPerDatasetLRContractError(f"invalid JSON artifact: {path}: {exc}") from exc


def audit_existing_d5_reuse(uniform_root: Path, *, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Audit D5 without mutating the old uniform root."""
    root = Path(uniform_root).resolve()
    mismatches: List[str] = []
    expected_target = FORMAL_PER_DATASET_LR_TARGETS["D5"]
    expected_seeds = set(FORMAL_PER_DATASET_LR_SEEDS)
    expected_horizons = set(HORIZONS)
    expected_arms = set(AE_ARMS)
    manifest_path = root / "manifest_snapshot.json"
    uniform_manifest_path = root / "manifest_snapshot.json"
    if not root.is_dir():
        mismatches.append(f"missing uniform root: {root}")
    if root.name.find("sita_ablation_3e4") < 0:
        mismatches.append("source root is not the retained uniform-3e-4 output")
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        if manifest.get("matrix_kind") != "UNIFORM_3E4_PAPER_5SEED":
            mismatches.append("source matrix_kind is not UNIFORM_3E4_PAPER_5SEED")
        if manifest.get("uniform_target_lr") != "3e-4":
            mismatches.append("source uniform target LR is not 3e-4")
        if manifest.get("source_lr") != "1e-4":
            mismatches.append("source Stage-A LR is not 1e-4")
        rng = manifest.get("stage_a_rng_contract") or {}
        expected_rng = {
            "reset_once_at_entry": True,
            "reset_before_source_model_creation": True,
            "sequential_source_stream": True,
            "per_source_reseed": False,
            "stage_a_seed_equals_cell_seed": True,
            "historical_source_weight_reuse": False,
        }
        if any(rng.get(key) != value for key, value in expected_rng.items()):
            mismatches.append("current corrected Stage-A RNG contract mismatch")
    else:
        mismatches.append("source manifest_snapshot.json missing")

    result_paths = sorted((root / "cells").glob("D5__*/attempt_*/result.json"))
    if len(result_paths) != 15:
        mismatches.append(f"D5 result cell count={len(result_paths)} != 15")
    rows = []
    seen = set()
    source_signatures = set()
    recipe_digests = set()
    architecture_digests = set()
    for result_path in result_paths:
        payload = _json(result_path)
        cell = (payload.get("dataset"), payload.get("target"), int(payload.get("horizon", -1)), int(payload.get("seed", -1)))
        if cell in seen:
            mismatches.append(f"duplicate D5 cell: {cell}")
        seen.add(cell)
        if payload.get("status") != "COMPLETE" or payload.get("pair_status") != "PAIR_COMPLETE":
            mismatches.append(f"D5 incomplete result: {cell}")
        if payload.get("dataset") != "D5" or payload.get("target") != expected_target:
            mismatches.append(f"D5 target identity mismatch: {cell}")
        if int(payload.get("horizon", -1)) not in expected_horizons or int(payload.get("seed", -1)) not in expected_seeds:
            mismatches.append(f"D5 horizon/seed mismatch: {cell}")
        if float(payload.get("selected_lr", -1)) != 3e-4:
            mismatches.append(f"D5 target LR mismatch: {cell}")
        arms = payload.get("arms") or {}
        if set(arms) != expected_arms:
            mismatches.append(f"D5 arm set mismatch: {cell}")
        cell_split_digests = set()
        for arm in AE_ARMS:
            arm_payload = arms.get(arm) or {}
            spec = ae_arm_spec(arm)
            if arm_payload.get("status") != "OK":
                mismatches.append(f"D5 arm status mismatch: {cell}/{arm}")
            if arm_payload.get("init_mode") != spec.init_mode:
                mismatches.append(f"D5 init mode mismatch: {cell}/{arm}")
            if bool(arm_payload.get("source_init_consumed")) != (spec.init_mode != "native"):
                mismatches.append(f"D5 source-init semantics mismatch: {cell}/{arm}")
            if float(arm_payload.get("selected_lr", -1)) != 3e-4:
                mismatches.append(f"D5 arm LR mismatch: {cell}/{arm}")
            if arm_payload.get("target_split_digest"):
                cell_split_digests.add(arm_payload["target_split_digest"])
            source_signatures.add(
                (
                    tuple(tuple(item) for item in arm_payload.get("source_identities", [])),
                    tuple(float(value) for value in arm_payload.get("source_weights", [])),
                    tuple(float(value) for value in arm_payload.get("source_distances", [])),
                    arm_payload.get("source_selection_digest"),
                )
            )
            if arm_payload.get("recipe_digest"):
                recipe_digests.add(arm_payload["recipe_digest"])
            if arm_payload.get("parent_v3_architecture_digest"):
                architecture_digests.add(arm_payload["parent_v3_architecture_digest"])
        if len(cell_split_digests) != 1:
            mismatches.append(f"D5 split is not shared across A-E: {cell}")
        rows.append(payload)

    expected_cells = {
        ("D5", expected_target, horizon, seed)
        for horizon in HORIZONS
        for seed in FORMAL_PER_DATASET_LR_SEEDS
    }
    if seen != expected_cells:
        mismatches.append("D5 cell grid differs from 3 horizons × 5 seeds")
    csv_path = root / "ae_uniform_3e4_results.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = [row for row in csv.DictReader(handle) if row.get("dataset") == "D5"]
        if len(csv_rows) != 75:
            mismatches.append(f"D5 CSV arm rows={len(csv_rows)} != 75")
        if any(row.get("learning_rate") != "3e-4" for row in csv_rows):
            mismatches.append("D5 CSV target LR label mismatch")
    else:
        mismatches.append("D5 CSV result table missing")

    optimizer = ae_optimizer_contract()
    recipe = ae_training_recipe()
    static_parity = {
        "stage_a_lr": "PASS" if float(STAGE_A_LEARNING_RATE) == 1e-4 else "FAIL",
        "epochs": "PASS" if int(STAGE_B_EPOCHS) == 50 and int(optimizer["epochs"]) == 50 else "FAIL",
        "batch_size": "PASS" if int(STAGE_B_BATCH_SIZE) == 16 and int(optimizer["batch_size"]) == 16 else "FAIL",
        "optimizer": "PASS" if optimizer.get("optimizer") == "Adam" else "FAIL",
        "ema": "PASS" if recipe.get("ema_momentum") == 0.99 else "FAIL",
        "revin": "PASS" if bool(V3_REVIN_CONTRACT.get("revin_enabled")) else "FAIL",
        "arm_definitions": "PASS" if all(ae_arm_spec(arm).name == AE_ARM_NAMES[arm] for arm in AE_ARMS) else "FAIL",
        "target_source_identifiers": "PASS" if len(source_signatures) == 1 else "FAIL",
        "current_corrected_stage_a_rng": "PASS" if not any("RNG contract" in item for item in mismatches) else "FAIL",
    }
    for name, status in static_parity.items():
        if status != "PASS":
            mismatches.append(f"{name} parity failed")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "source_root": str(root),
        "arm_runs": len(rows) * len(AE_ARMS),
        "target_lr": "3e-4",
        "stage_a_lr": "1e-4",
        "source_selection": static_parity["target_source_identifiers"],
        "current_corrected_stage_a_rng": static_parity["current_corrected_stage_a_rng"],
        "mismatches": mismatches,
        "static_parity": static_parity,
        "recipe_digest_count": len(recipe_digests),
        "architecture_digest_count": len(architecture_digests),
        "provenance_running_is_legacy_start_record": True,
        "old_uniform_root_modified": False,
    }


def _load_d5_reuse_payloads(uniform_root: Path) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(uniform_root).glob("cells/D5__*/attempt_*/result.json")):
        payload = _json(path)
        key = str(payload.get("stage_a_key") or payload.get("cell_key") or "")
        if not key:
            raise FormalPerDatasetLRContractError(f"D5 reuse result has no stage_a_key: {path}")
        payload["reused_from_root"] = str(Path(uniform_root).resolve())
        payload["reuse_status"] = "REUSED_D5_UNIFORM_3E4_EXACT"
        payloads[key] = payload
    return payloads


def run_formal_per_dataset_lr_dry_run(
    *,
    project_root: Optional[Path] = None,
    resolve_real_cells: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    identities = enumerate_formal_per_dataset_lr_identities()
    report = validate_formal_per_dataset_lr_matrix(identities)
    resolved_cells = 0
    if resolve_real_cells:
        resolved_cells = len(enumerate_formal_per_dataset_lr_cell_plans(project_root=root))
    d5_root = root / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227"
    d5_audit = audit_existing_d5_reuse(d5_root, project_root=root)
    return {
        **report,
        "PLAN_ONLY": "YES",
        "TRAINING_STARTED": "NO",
        "READY_FOR_FULL_RUN": "YES" if resolved_cells in {0, 60} else "NO",
        "RESOLVED_CELLS": resolved_cells,
        "D5_REUSE_AUDIT": d5_audit,
        "OUTPUT_LABEL": "FORMAL_PER_DATASET_LR_AE",
    }


def run_formal_per_dataset_lr_execute(
    *,
    output_dir: Optional[Path] = None,
    resume_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    if output_dir is not None and resume_root is not None:
        raise FormalPerDatasetLRContractError("provide either --output-root or --resume, not both")
    if output_dir is None and resume_root is None:
        raise FormalPerDatasetLRContractError(
            "formal per-dataset LR execution requires explicit --output-root or --resume"
        )
    requested = Path(resume_root if resume_root is not None else output_dir).resolve()
    assert_formal_per_dataset_lr_output_isolated(requested, project_root=root)
    if resume_root is None and requested.exists():
        raise FormalPerDatasetLRContractError(
            f"new formal output root already exists; use --resume explicitly: {requested}"
        )
    audit = audit_existing_d5_reuse(
        root / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227",
        project_root=root,
    )
    reuse_payloads = _load_d5_reuse_payloads(Path(audit["source_root"])) if audit["status"] == "PASS" else None
    from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_body

    cells = enumerate_formal_per_dataset_lr_cell_plans(project_root=root)
    result = run_ae_formal_body(
        execute=True,
        environ=environ,
        cells=cells,
        matrix_kind=FORMAL_PER_DATASET_LR_MATRIX_KIND,
        output_root=requested,
        resume_root=resume_root,
        project_root=root,
        hardened_attempts=True,
        reused_payloads=reuse_payloads,
    )
    result["D5_REUSE_AUDIT"] = audit
    result["D5_REUSED"] = "YES" if reuse_payloads else "NO"
    result["EXPECTED_TOTAL_RUNS"] = FORMAL_PER_DATASET_LR_RUNS
    result["OUTPUT_LABEL"] = "FORMAL_PER_DATASET_LR_AE"
    return result
