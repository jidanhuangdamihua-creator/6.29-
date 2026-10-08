"""D4–D6 all-candidate-target enumeration for frozen SITA / FUSED_TARGET_ONLY_V3.

This module only enumerates the confirmatory matrix. It does not change the
SITA model, Stage-A/B training recipe, data protocol, or frozen primary-target
definitions. Paper mode remains without-information-sharing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_cross_target import D4_CROSS_TARGET_IDENTITIES
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_COMMON_TARGETS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    RunIdentity,
    RobustnessContractError,
    assert_formal_methods_unchanged,
    common_targets_from_frozen_manifest,
    load_v3_frozen_robustness_manifest,
    v3_arm_spec,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    V3_REVIN_CONTRACT,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_EPOCHS,
    SCREENING_MODE,
    canonical_candidate_targets,
    format_lr,
    parse_lr,
    parse_target_identity,
)
from src.transfer_methods.ms_katl_v3_frame_assembly import v3_knn_authority_path
from src.transfer_methods.ms_katl_v3_with_probe import FROZEN_WITHOUT_RESULTS_RELPATH

ALL_TARGETS_LABEL = "SITA_ALL_TARGETS_D4_D6_5SEED"
SITA_FORMAL_CONFIG = "FUSED_TARGET_ONLY_V3"
SITA_FORMAL_RUNNER = "scripts/run_ms_katl_fto_three_track_robustness.py"
SITA_FORMAL_MANIFEST_RELPATH = Path("configs") / "ms_katl_fto_v3_robustness_manifest.json"
PAPER_SITA_MODE = SCREENING_MODE
ALL_TARGETS_DATASETS: Tuple[str, ...] = ("D4", "D5", "D6")
ALL_TARGETS_HORIZONS: Tuple[int, ...] = (1, 3, 5)
ALL_TARGETS_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
CELLS_PER_TARGET = len(ALL_TARGETS_HORIZONS) * len(ALL_TARGETS_SEEDS)
EXPECTED_TARGETS_PER_DATASET = 5
EXPECTED_DATASET_CELLS = 75
EXPECTED_TOTAL_CELLS = 225
MAX_JOBS = 1
D5_MAX_JOBS = 1
WORKERS = 1
DEFAULT_OUTPUT_RELPATH = Path("outputs") / "sita_all_targets_d4_d6_5seed"
LR_POLICY = "DATASET_FROZEN_PRIMARY_LR"
TARGET_POLICY = "D4_D6_ALL_CANONICAL_CANDIDATES"
V3_SEED45_46_COMPLETION_RELPATH = (
    Path("outputs") / "runs" / "20260823_MS_KATL_V3_WITHOUT_SEED45_46_COMPLETION"
)
V3_WITH_PROBE_RELPATH = Path("outputs") / "runs" / "20260823_MS_KATL_V3_WITH_4_CELL_PROBE"
DEFAULT_V3_SCREENING_DIRNAME = "20260818_180250_355272_MS_KATL_FTO_V3_SCREENING"

DATASET_INHERITED_LR: Dict[str, str] = {
    "D4": "3e-5",
    "D5": "3e-4",
    "D6": "3e-5",
}

REQUESTED_TARGET_SPECS: Dict[str, Tuple[str, ...]] = {
    "D4": ("166/258", "166/311", "166/313", "166/432", "166/433"),
    "D5": ("48/1159414", "48/1159415", "48/1349808", "48/320682", "48/364606"),
    "D6": (
        "CA_1/FOODS_3_080",
        "CA_1/FOODS_3_377",
        "CA_1/FOODS_3_555",
        "CA_1/FOODS_3_586",
        "CA_1/FOODS_3_668",
    ),
}

PROTECTED_OUTPUT_MARKERS = (
    "MS_KATL_FTO_THREE_TRACK_ROBUSTNESS",
    "FTO_V3_SCREENING",
    "FTO_V2_SCREENING",
    "PRIMARY_TARGET_LR_SCREENING",
    "V3_WITHOUT_SEED45_46_COMPLETION",
    "MS_KATL_V3_WITH_4_CELL_PROBE",
    "V3_1_CALIBRATED",
    "V3_2_SEASONAL_RESIDUAL",
)


class AllTargetsContractError(RobustnessContractError):
    """Fail-closed all-target enumeration contract violation."""


def canonical_targets_for_dataset(dataset: str) -> Tuple[str, ...]:
    requested = REQUESTED_TARGET_SPECS[dataset]
    resolved = tuple(
        parse_target_identity(dataset, value)["target_identity"] for value in requested
    )
    inventory = set(canonical_candidate_targets(dataset))
    if set(resolved) != inventory:
        raise AllTargetsContractError(
            f"{dataset} requested targets {list(resolved)} must equal canonical inventory "
            f"{sorted(inventory)}"
        )
    if len(resolved) != EXPECTED_TARGETS_PER_DATASET or len(set(resolved)) != EXPECTED_TARGETS_PER_DATASET:
        raise AllTargetsContractError(
            f"{dataset} must have {EXPECTED_TARGETS_PER_DATASET} unique targets, got {resolved}"
        )
    if dataset == "D4" and resolved != D4_CROSS_TARGET_IDENTITIES:
        raise AllTargetsContractError("D4 target order drifted from canonical five-target order")
    return resolved


def all_target_inventory() -> Dict[str, Tuple[str, ...]]:
    return {dataset: canonical_targets_for_dataset(dataset) for dataset in ALL_TARGETS_DATASETS}


def dataset_inherited_lr_map(project_root: Path) -> Dict[str, str]:
    frozen = load_v3_frozen_robustness_manifest(project_root)
    mapping: Dict[str, str] = {}
    for dataset in ALL_TARGETS_DATASETS:
        label = format_lr(parse_lr(frozen["datasets"][dataset]["learning_rate"]))
        expected = DATASET_INHERITED_LR[dataset]
        if label != expected:
            raise AllTargetsContractError(
                f"{dataset} frozen SITA LR drifted: {label} != {expected}"
            )
        mapping[dataset] = label
    return mapping


def assert_frozen_primary_targets_untouched(project_root: Path) -> Tuple[Tuple[str, str], ...]:
    frozen = load_v3_frozen_robustness_manifest(project_root)
    observed = tuple(
        (item.dataset, item.target_identity)
        for item in common_targets_from_frozen_manifest(frozen)
    )
    if observed != EXPECTED_COMMON_TARGETS:
        raise AllTargetsContractError(
            f"frozen primary targets drifted: {observed} != {EXPECTED_COMMON_TARGETS}"
        )
    return observed


def assert_sita_identity_frozen(project_root: Path) -> Dict[str, Any]:
    assert_formal_methods_unchanged()
    if V3_METHOD_DISPLAY in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise AllTargetsContractError("SITA must remain outside FORMAL_METHODS")
    spec = v3_arm_spec()
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AllTargetsContractError(
            f"SITA architecture digest drifted: {digest} != {EXPECTED_V3_ARCHITECTURE_DIGEST}"
        )
    if spec.architecture_digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AllTargetsContractError("v3 arm digest drifted from frozen SITA identity")
    arch = v3_architecture()
    recipe = v3_training_recipe()
    expected_flags = {
        "use_fused_target_init": True,
        "target_backbone_trainable": True,
        "transfer_branch_enabled": False,
        "gate_mode": "zero",
        "lambda_adv": 0.0,
        "attention_used_in_prediction": False,
        "adapter_used_in_prediction": False,
        "revin_enabled": True,
        "revin_affine": True,
    }
    for key, value in expected_flags.items():
        if arch[key] != value:
            raise AllTargetsContractError(f"SITA architecture flag drifted: {key}={arch[key]!r}")
    if dict(V3_REVIN_CONTRACT)["revin_enabled"] is not True:
        raise AllTargetsContractError("RevIN contract drifted")
    if dict(recipe) != {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }:
        raise AllTargetsContractError(f"SITA training recipe drifted: {recipe}")
    if PAPER_SITA_MODE != "without":
        raise AllTargetsContractError("paper SITA mode must remain without")
    frozen = load_v3_frozen_robustness_manifest(project_root)
    if frozen.get("method") != SITA_FORMAL_CONFIG:
        raise AllTargetsContractError(f"frozen method drifted: {frozen.get('method')}")
    if str(frozen.get("architecture_digest")) != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AllTargetsContractError("frozen manifest architecture digest drifted")
    if str(frozen.get("common_target_policy")) != "V3_FROZEN_PRIMARY_TARGETS":
        raise AllTargetsContractError("frozen primary-target policy drifted")
    assert_frozen_primary_targets_untouched(project_root)
    return {
        "method": SITA_FORMAL_CONFIG,
        "architecture_digest": digest,
        "mode": PAPER_SITA_MODE,
        "training_recipe": dict(recipe),
        "architecture": dict(arch),
        "epochs": SCREENING_EPOCHS,
        "formal_methods": list(FORMAL_METHODS),
    }


def protected_result_roots(project_root: Path) -> Tuple[Path, ...]:
    root = Path(project_root).resolve()
    return (
        (root / FROZEN_WITHOUT_RESULTS_RELPATH).resolve(),
        (root / "outputs" / "runs" / DEFAULT_V3_SCREENING_DIRNAME).resolve(),
        (root / V3_SEED45_46_COMPLETION_RELPATH).resolve(),
        (root / V3_WITH_PROBE_RELPATH).resolve(),
        (root / SITA_FORMAL_MANIFEST_RELPATH).resolve(),
        (root / "outputs" / "paper").resolve(),
        (root / "configs").resolve(),
    )


def default_output_dir(project_root: Path) -> Path:
    return Path(project_root) / DEFAULT_OUTPUT_RELPATH


def assert_output_does_not_clobber_frozen(run_dir: Path, project_root: Path) -> Path:
    resolved = Path(run_dir).resolve()
    name = resolved.name
    for marker in PROTECTED_OUTPUT_MARKERS:
        if marker in name:
            raise AllTargetsContractError(
                f"refusing to write all-target outputs into frozen SITA result name: {run_dir}"
            )
    for forbidden in protected_result_roots(project_root):
        if resolved == forbidden:
            raise AllTargetsContractError(f"refusing to overwrite frozen path: {forbidden}")
        try:
            resolved.relative_to(forbidden)
        except ValueError:
            continue
        raise AllTargetsContractError(f"refusing to write under frozen path: {forbidden}")
    return resolved


def source_authority_path(dataset: str, project_root: Path) -> Path:
    path = v3_knn_authority_path(dataset, PAPER_SITA_MODE)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def build_all_target_identities(
    *,
    project_root: Path,
    datasets: Sequence[str] = ALL_TARGETS_DATASETS,
    horizons: Sequence[int] = ALL_TARGETS_HORIZONS,
    seeds: Sequence[int] = ALL_TARGETS_SEEDS,
) -> Tuple[RunIdentity, ...]:
    extra = [item for item in datasets if item not in ALL_TARGETS_DATASETS]
    if extra:
        raise AllTargetsContractError(f"D1 and non-D4/D5/D6 datasets are forbidden: {extra}")
    if any(item in {"D1", "D2", "D3"} for item in datasets):
        raise AllTargetsContractError("this matrix excludes D1/D2/D3")
    extra_h = [item for item in horizons if int(item) not in ALL_TARGETS_HORIZONS]
    extra_s = [item for item in seeds if int(item) not in ALL_TARGETS_SEEDS]
    if extra_h or extra_s:
        raise AllTargetsContractError(
            f"horizon/seed outside frozen SITA sets: horizons={extra_h} seeds={extra_s}"
        )
    identity_flags = assert_sita_identity_frozen(project_root)
    if PAPER_SITA_MODE != "without" or SCREENING_MODE != "without":
        raise AllTargetsContractError("paper SITA mode must remain without")
    lr_map = dataset_inherited_lr_map(project_root)
    inventory = all_target_inventory()
    identities: List[RunIdentity] = []
    for dataset in datasets:
        lr_label = lr_map[dataset]
        lr_value = parse_lr(lr_label)
        for target in inventory[dataset]:
            parsed = parse_target_identity(dataset, target)
            if parsed["target_identity"] != target:
                raise AllTargetsContractError(
                    f"target key format drifted: {parsed['target_identity']} != {target}"
                )
            for horizon in horizons:
                for seed in seeds:
                    identities.append(
                        RunIdentity(
                            arm="v3",
                            dataset=dataset,
                            target_identity=target,
                            horizon=int(horizon),
                            seed=int(seed),
                            selected_lr=float(lr_value),
                            selected_lr_label=format_lr(lr_value),
                            mode=PAPER_SITA_MODE,
                        )
                    )
    tuples = [item.identity_tuple for item in identities]
    if len(tuples) != len(set(tuples)):
        raise AllTargetsContractError("duplicate all-target cell identity")
    return tuple(identities)


def validate_all_target_matrix(
    identities: Sequence[RunIdentity],
) -> Dict[str, Any]:
    rows = list(identities)
    datasets = [item.dataset for item in rows]
    if any(item in {"D1", "D2", "D3"} for item in datasets):
        raise AllTargetsContractError("matrix contains D1/D2/D3")
    if set(datasets) - set(ALL_TARGETS_DATASETS):
        raise AllTargetsContractError(f"unexpected datasets: {sorted(set(datasets))}")
    extra_targets: List[Tuple[str, str]] = []
    inventory = all_target_inventory()
    by_target: Dict[Tuple[str, str], List[RunIdentity]] = {}
    for item in rows:
        key = (item.dataset, item.target_identity)
        allowed = inventory.get(item.dataset, ())
        if item.target_identity not in allowed:
            extra_targets.append(key)
        by_target.setdefault(key, []).append(item)
        if item.mode != PAPER_SITA_MODE:
            raise AllTargetsContractError(f"non-paper mode: {item.mode}")
        if item.arm != "v3":
            raise AllTargetsContractError(f"non-SITA arm: {item.arm}")
        if int(item.horizon) not in ALL_TARGETS_HORIZONS:
            raise AllTargetsContractError(f"illegal horizon {item.horizon}")
        if int(item.seed) not in ALL_TARGETS_SEEDS:
            raise AllTargetsContractError(f"illegal seed {item.seed}")
        if item.selected_lr_label != DATASET_INHERITED_LR[item.dataset]:
            raise AllTargetsContractError(
                f"LR re-search is forbidden: {item.dataset} {item.target_identity} "
                f"{item.selected_lr_label} != {DATASET_INHERITED_LR[item.dataset]}"
            )
    if extra_targets:
        raise AllTargetsContractError(f"extra targets: {extra_targets}")
    missing_targets = []
    short_targets = []
    for dataset, targets in inventory.items():
        for target in targets:
            cells = by_target.get((dataset, target), [])
            if not cells:
                missing_targets.append((dataset, target))
            elif len(cells) != CELLS_PER_TARGET:
                short_targets.append(
                    {"dataset": dataset, "target": target, "cells": len(cells)}
                )
    counts = {
        dataset: len({item.target_identity for item in rows if item.dataset == dataset})
        for dataset in ALL_TARGETS_DATASETS
    }
    cell_counts = {
        dataset: len([item for item in rows if item.dataset == dataset])
        for dataset in ALL_TARGETS_DATASETS
    }
    unique = len({item.identity_tuple for item in rows})
    payload = {
        "D4_TARGETS": counts["D4"],
        "D5_TARGETS": counts["D5"],
        "D6_TARGETS": counts["D6"],
        "HORIZONS": len({int(item.horizon) for item in rows}),
        "SEEDS": len({int(item.seed) for item in rows}),
        "HORIZON_VALUES": list(ALL_TARGETS_HORIZONS),
        "SEED_VALUES": list(ALL_TARGETS_SEEDS),
        "EXPECTED_D4_CELLS": EXPECTED_DATASET_CELLS,
        "EXPECTED_D5_CELLS": EXPECTED_DATASET_CELLS,
        "EXPECTED_D6_CELLS": EXPECTED_DATASET_CELLS,
        "EXPECTED_TOTAL_CELLS": EXPECTED_TOTAL_CELLS,
        "DRY_RUN_CELL_COUNT": unique,
        "D4_CELL_COUNT": cell_counts["D4"],
        "D5_CELL_COUNT": cell_counts["D5"],
        "D6_CELL_COUNT": cell_counts["D6"],
        "DUPLICATE_CELLS": len(rows) - unique,
        "MISSING_TARGETS": missing_targets,
        "SHORT_TARGETS": short_targets,
        "EXTRA_TARGETS": extra_targets,
        "HAS_D1": any(item == "D1" for item in datasets),
        "MODE": PAPER_SITA_MODE,
        "METHOD": SITA_FORMAL_CONFIG,
        "LR_POLICY": LR_POLICY,
        "TARGET_POLICY": TARGET_POLICY,
        "MAX_JOBS": MAX_JOBS,
        "D5_MAX_JOBS": D5_MAX_JOBS,
        "WORKERS": WORKERS,
    }
    ready = (
        unique == EXPECTED_TOTAL_CELLS
        and len(rows) == EXPECTED_TOTAL_CELLS
        and payload["DUPLICATE_CELLS"] == 0
        and not missing_targets
        and not short_targets
        and not extra_targets
        and payload["D4_TARGETS"] == 5
        and payload["D5_TARGETS"] == 5
        and payload["D6_TARGETS"] == 5
        and payload["HORIZONS"] == 3
        and payload["SEEDS"] == 5
        and cell_counts["D4"] == EXPECTED_DATASET_CELLS
        and cell_counts["D5"] == EXPECTED_DATASET_CELLS
        and cell_counts["D6"] == EXPECTED_DATASET_CELLS
        and not payload["HAS_D1"]
    )
    payload["READY_FOR_FULL_RUN"] = "YES" if ready else "NO"
    if not ready:
        raise AllTargetsContractError(
            "all-target matrix is not 225 unique legal cells: "
            + json.dumps(
                {
                    "DRY_RUN_CELL_COUNT": unique,
                    "MISSING_TARGETS": missing_targets,
                    "SHORT_TARGETS": short_targets,
                    "EXTRA_TARGETS": extra_targets,
                    "DUPLICATE_CELLS": payload["DUPLICATE_CELLS"],
                },
                default=str,
            )
        )
    return payload


def cell_plan_rows(
    identities: Sequence[RunIdentity],
    *,
    project_root: Path,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    from src.transfer_methods.ms_katl_fto_three_track_robustness import cell_path

    rows = []
    for item in identities:
        parsed = parse_target_identity(item.dataset, item.target_identity)
        rows.append(
            {
                "dataset": item.dataset,
                "target": item.target_identity,
                "target_identity": item.target_identity,
                "horizon": int(item.horizon),
                "seed": int(item.seed),
                "mode": item.mode,
                "method": SITA_FORMAL_CONFIG,
                "arm": item.arm,
                "learning_rate": item.selected_lr,
                "learning_rate_label": item.selected_lr_label,
                "entity_key": parsed["entity_key"],
                "cell_id": item.cell_id,
                "source_authority": str(source_authority_path(item.dataset, project_root)),
                "cell_path": str(cell_path(Path(run_dir) / "cells", item)),
            }
        )
    return rows


def format_dry_run(
    *,
    identities: Sequence[RunIdentity],
    report: Mapping[str, Any],
    run_dir: Path,
    identity_flags: Mapping[str, Any],
) -> str:
    lines = [
        "============================================================",
        "SITA / FUSED_TARGET_ONLY_V3 ALL-TARGET D4-D6 DRY-RUN",
        "============================================================",
        "",
        f"SITA_FORMAL_RUNNER={SITA_FORMAL_RUNNER}",
        f"SITA_FORMAL_CONFIG={SITA_FORMAL_CONFIG}",
        f"SITA_FORMAL_MANIFEST={SITA_FORMAL_MANIFEST_RELPATH.as_posix()}",
        f"SITA_ARCHITECTURE_DIGEST={identity_flags['architecture_digest']}",
        f"PAPER_SITA_MODE={PAPER_SITA_MODE}",
        f"TARGET_POLICY={TARGET_POLICY}",
        f"LR_POLICY={LR_POLICY}",
        "ROBUSTNESS_RESELECTION=NO",
        "SITA_MODEL_CHANGED=NO",
        "SITA_TRAINING_PROTOCOL_CHANGED=NO",
        "DATA_PROTOCOL_CHANGED=NO",
        "FROZEN_RESULTS_OVERWRITTEN=NO",
        "",
        f"D4_TARGETS={report['D4_TARGETS']}",
        f"D5_TARGETS={report['D5_TARGETS']}",
        f"D6_TARGETS={report['D6_TARGETS']}",
        "",
        f"HORIZONS={report['HORIZONS']}",
        f"SEEDS={report['SEEDS']}",
        "HORIZON_VALUES=1,3,5",
        "SEED_VALUES=42,43,44,45,46",
        "",
        f"EXPECTED_D4_CELLS={report['EXPECTED_D4_CELLS']}",
        f"EXPECTED_D5_CELLS={report['EXPECTED_D5_CELLS']}",
        f"EXPECTED_D6_CELLS={report['EXPECTED_D6_CELLS']}",
        f"EXPECTED_TOTAL_CELLS={report['EXPECTED_TOTAL_CELLS']}",
        f"DRY_RUN_CELL_COUNT={report['DRY_RUN_CELL_COUNT']}",
        "",
        f"MAX_JOBS={MAX_JOBS}",
        f"D5_MAX_JOBS={D5_MAX_JOBS}",
        f"WORKERS={WORKERS}",
        "CONCURRENCY_POLICY=SEQUENTIAL_INHERIT_THREE_TRACK",
        "",
        f"OUTPUT_DIR={run_dir}",
        f"READY_FOR_FULL_RUN={report['READY_FOR_FULL_RUN']}",
        "============================================================",
        "",
        f"{'DATASET':<8}{'TARGET':<22}{'LR':<8}{'CELLS':<8}",
        "------------------------------------------------------------",
    ]
    inventory = all_target_inventory()
    lr_map = {item.dataset: item.selected_lr_label for item in identities}
    for dataset in ALL_TARGETS_DATASETS:
        for target in inventory[dataset]:
            n_cells = len(
                [
                    item
                    for item in identities
                    if item.dataset == dataset and item.target_identity == target
                ]
            )
            lines.append(f"{dataset:<8}{target:<22}{lr_map[dataset]:<8}{n_cells:<8}")
    lines.append("============================================================")
    return "\n".join(lines) + "\n"


def completion_guard_225(
    identities: Sequence[RunIdentity],
    cell_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = len(identities)
    expected_set = {item.identity_tuple for item in identities}
    ok_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "OK"]
    failed_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "FAILED"]
    materialized = []
    for row in ok_rows:
        materialized.append(
            (
                str(row["arm"]).lower(),
                str(row["dataset"]),
                str(row["target_identity"]),
                str(row.get("mode") or PAPER_SITA_MODE).strip().lower(),
                int(row["horizon"]),
                int(row["seed"]),
                format_lr(parse_lr(row.get("learning_rate_label") or row["learning_rate"])),
            )
        )
    unique = set(materialized)
    missing = expected_set - unique
    extras = unique - expected_set
    return {
        "EXPECTED_TOTAL_CELLS": EXPECTED_TOTAL_CELLS,
        "EXPECTED_RUNS": expected,
        "OK_RUNS": len(unique),
        "FAILED_RUNS": len(failed_rows),
        "DUPLICATE_RUNS": len(materialized) - len(unique),
        "MISSING_RUNS": len(missing),
        "EXTRA_RUNS": len(extras),
        "SITA_ALL_TARGETS_COMPLETE": (
            "YES"
            if expected == EXPECTED_TOTAL_CELLS
            and len(unique) == expected
            and not failed_rows
            and not missing
            and not extras
            and (len(materialized) - len(unique)) == 0
            else "NO"
        ),
    }


def result_row_with_required_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(payload)
    row.setdefault("target", row.get("target_identity"))
    row.setdefault("RMSE", row.get("rmse_original"))
    row.setdefault("sMAPE", row.get("smape_original"))
    row.setdefault("method", SITA_FORMAL_CONFIG)
    row.setdefault("mode", PAPER_SITA_MODE)
    row.setdefault("run_status", row.get("status"))
    return row
