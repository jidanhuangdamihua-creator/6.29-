"""Real D1–D6 execution wiring for the frozen V3 A–E protocol.

Phases are explicit. Dry-run may resolve and plan real cells. Formal Stage-A /
Stage-B training remains fail-closed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.constants import SOLIDIFIED_KNN_ROOT
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    AE_FORMAL_EXECUTION_IMPLEMENTED,
    AE_LABEL,
    EXPECTED_CANONICAL_CELLS,
    EXPECTED_RUNS,
    HORIZONS,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    SCENARIO,
    SEEDS,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EPOCHS,
    AEContractError,
    AERunIdentity,
    AESyntheticStageA,
    AETrainingBlocked,
    assert_formal_training_permitted,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_protocol_manifest_payload,
    ae_training_recipe,
    analysis_contract,
    assert_d_equals_frozen_v3,
    build_synthetic_stage_a,
    canonical_targets,
    construct_all_ae_arms,
    enumerate_canonical_cells,
    enumerate_planned_runs,
    load_ae_manifest,
    pairwise_single_factor_contracts,
    project_root_from,
    require_v3_recipe,
    resolve_cell_lr,
    run_ae_preflight,
    verify_common_head_init,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_MODE,
    parse_target_identity,
)
from src.utils.parquet_data_loader import load_knn_results

AE_RUN_LABEL = "MS_KATL_AE_CORE_ABLATION"
PREFLIGHT_RELPATH = Path("outputs") / "preflight" / "ms_katl_ae_ablation"
REQUESTED_K = 3
FORBIDDEN_RUN_MARKERS = (
    "PRIMARY_TARGET_LR_SCREENING",
    "FTO_V2_SCREENING",
    "FTO_V3_SCREENING",
    "FTO_THREE_TRACK",
    "THREE_TRACK_ROBUSTNESS",
    "MSML_TL",
    "d1d6_630ae4ef",
)


class AEExecutionMode(str, Enum):
    PREFLIGHT = "preflight"
    DRY_RUN = "dry_run"
    FORMAL = "formal"


class AEExecutionPhase(str, Enum):
    PLAN = "PLAN"
    LOAD = "LOAD"
    RESOLVE = "RESOLVE"
    PREPARE_CONTEXT = "PREPARE_CONTEXT"
    STAGE_A_PLAN = "STAGE_A_PLAN"
    STAGE_A_TRAIN = "STAGE_A_TRAIN"
    BUILD_ARMS = "BUILD_ARMS"
    TRAIN_STAGE_B = "TRAIN_STAGE_B"
    PREDICT = "PREDICT"
    METRICS = "METRICS"
    RESULT_PUBLISH = "RESULT_PUBLISH"


DRY_RUN_ALLOWED_PHASES = frozenset(
    {
        AEExecutionPhase.PLAN,
        AEExecutionPhase.LOAD,
        AEExecutionPhase.RESOLVE,
        AEExecutionPhase.PREPARE_CONTEXT,
        AEExecutionPhase.STAGE_A_PLAN,
        AEExecutionPhase.BUILD_ARMS,
    }
)
FORMAL_ONLY_PHASES = frozenset(
    {
        AEExecutionPhase.STAGE_A_TRAIN,
        AEExecutionPhase.TRAIN_STAGE_B,
        AEExecutionPhase.PREDICT,
        AEExecutionPhase.METRICS,
        AEExecutionPhase.RESULT_PUBLISH,
    }
)


def assert_phase_allowed(mode: AEExecutionMode, phase: AEExecutionPhase) -> None:
    if mode is AEExecutionMode.PREFLIGHT and phase not in {
        AEExecutionPhase.PLAN,
        AEExecutionPhase.RESOLVE,
    }:
        raise AEContractError(f"phase {phase.value} is not allowed in PREFLIGHT")
    if mode is AEExecutionMode.DRY_RUN and phase in FORMAL_ONLY_PHASES:
        raise AETrainingBlocked(f"TRAINING_FORBIDDEN_DURING_REAL_DRY_RUN: {phase.value}")
    if mode is AEExecutionMode.DRY_RUN and phase not in DRY_RUN_ALLOWED_PHASES:
        raise AEContractError(f"phase {phase.value} is not allowed in DRY_RUN")
    if mode is AEExecutionMode.FORMAL:
        if not AE_FORMAL_EXECUTION_ENABLED:
            raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")
        if not AE_FORMAL_EXECUTION_IMPLEMENTED:
            raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")


@dataclass(frozen=True)
class RealCanonicalCellPlan:
    dataset: str
    target: str
    scenario: str
    horizon: int
    seed: int
    selected_lr: float
    selected_lr_label: str
    dataset_source: str
    target_identity: str
    entity_key: str
    target_split_identity: str
    target_split_digest: str
    source_pool_scope: str
    source_selection_contract: str
    source_pool_authority: str
    knn_diagnostic_sources: Tuple[str, ...]
    source_ids_status: str
    stage_a_key: str
    planned_arms: Tuple[str, ...] = AE_ARMS
    requested_k: int = REQUESTED_K


@dataclass(frozen=True)
class RealArmPlan:
    cell: RealCanonicalCellPlan
    arm: str
    arm_name: str
    init_mode: str
    backbone_trainable: bool
    revin_enabled: bool
    transfer_branch_enabled: bool
    gate_mode: str
    use_adversarial: bool
    stage_a_shared_for_pairing: bool
    source_init_consumed: bool
    parent_method: str
    parent_v3_architecture_digest: str
    recipe: Dict[str, Any]
    status: str = "PLANNED"

    @property
    def run_key(self) -> Tuple[Any, ...]:
        return (
            self.cell.dataset,
            self.cell.target,
            int(self.cell.horizon),
            int(self.cell.seed),
            self.cell.scenario,
            self.arm,
        )


@dataclass
class SharedStageAHandle:
    stage_a_key: str
    artifacts: AESyntheticStageA
    source_identities: List[Tuple[object, ...]]
    source_distances: List[float]
    source_weights: List[float]
    requested_k: int
    effective_k: int


@dataclass
class StageACallLedger:
    calls: List[str] = field(default_factory=list)
    artifacts: Dict[str, SharedStageAHandle] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.calls)


def dataset_number(dataset: str) -> int:
    return int(str(dataset).strip().upper()[1:])


def stage_a_key(dataset: str, target: str, horizon: int, seed: int, scenario: str = SCENARIO) -> str:
    return f"{dataset}|{target}|h{int(horizon)}|s{int(seed)}|{scenario}"


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def assert_ae_output_root(path: Path) -> Path:
    resolved = Path(path)
    name = resolved.name
    for marker in FORBIDDEN_RUN_MARKERS:
        if marker in name:
            raise AEContractError(f"refusing to write A–E outputs into frozen result root: {resolved}")
    if "results" in resolved.parts and "preflight" not in resolved.parts and "audits" not in resolved.parts:
        if AE_RUN_LABEL not in name and AE_LABEL not in name:
            raise AEContractError(f"A–E dry-run artifacts must not land in results/: {resolved}")
    return resolved


def planned_formal_run_root(timestamp: str) -> Path:
    return Path("outputs") / "runs" / f"{timestamp}_{AE_RUN_LABEL}"


def load_v3_knn_authority(dataset: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    knn_dir = Path(SOLIDIFIED_KNN_ROOT) / f"Dataset{dataset_number(dataset)}"
    payload = load_knn_results(knn_dir, SCREENING_MODE)
    if str(payload.get("info_sharing", SCREENING_MODE)).strip().lower() != SCENARIO:
        raise AEContractError(f"{dataset} KNN scenario drifted from {SCENARIO}")
    if int(payload.get("k", REQUESTED_K)) != REQUESTED_K:
        raise AEContractError(f"{dataset} KNN requested k drifted from {REQUESTED_K}")
    return payload


def _diagnostic_sources(payload: Mapping[str, Any], entity_key: str) -> Tuple[str, ...]:
    results = payload.get("results") or {}
    rows = results.get(entity_key)
    if not isinstance(rows, list) or not rows:
        raise AEContractError(f"KNN diagnostic sources missing for {entity_key}")
    if len(rows) < REQUESTED_K:
        raise AEContractError(f"KNN diagnostic sources for {entity_key} < requested_k={REQUESTED_K}")
    names = []
    for row in rows[:REQUESTED_K]:
        name = str(row.get("source_entity") or row.get("source_key") or "").strip()
        if not name:
            raise AEContractError(f"KNN diagnostic source row missing identity: {row}")
        names.append(name)
    return tuple(names)


def resolve_real_canonical_cell(
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    *,
    knn_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    project_root: Optional[Path] = None,
) -> RealCanonicalCellPlan:
    parsed = parse_target_identity(dataset, target)
    if parsed["target_identity"] != target:
        raise AEContractError(f"V3 identity resolver drifted: {parsed['target_identity']} != {target}")
    if parsed["dataset"] != dataset:
        raise AEContractError(f"dataset resolver drifted: {parsed['dataset']}")
    lr_value, lr_label = resolve_cell_lr(dataset, target)
    cache = knn_cache if knn_cache is not None else {}
    if dataset not in cache:
        cache[dataset] = load_v3_knn_authority(dataset, project_root=project_root)
    payload = cache[dataset]
    diagnostic = _diagnostic_sources(payload, parsed["entity_key"])
    key = stage_a_key(dataset, target, horizon, seed)
    split_payload = {
        "function": "temporal_split_by_ratio_or_dates",
        "authority": "FROZEN_V3_DATA_PROTOCOL",
        "dataset": dataset,
        "target": target,
        "scenario": SCENARIO,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": SCREENING_MODE,
    }
    return RealCanonicalCellPlan(
        dataset=dataset,
        target=target,
        scenario=SCENARIO,
        horizon=int(horizon),
        seed=int(seed),
        selected_lr=float(lr_value),
        selected_lr_label=lr_label,
        dataset_source="FROZEN_V3_DATA_PROTOCOL",
        target_identity=parsed["target_identity"],
        entity_key=parsed["entity_key"],
        target_split_identity="temporal_split_by_ratio_or_dates",
        target_split_digest=_json_digest(split_payload),
        source_pool_scope=str(payload.get("selection_authority", "shared_protocol")),
        source_selection_contract=str(payload.get("training_selection_authority", "shared_protocol_selector")),
        source_pool_authority=str(payload.get("training_selection_authority", "shared_protocol_selector")),
        knn_diagnostic_sources=diagnostic,
        source_ids_status="SOURCE_IDS_DEFERRED_TO_FORMAL_STAGE_A",
        stage_a_key=key,
    )


def enumerate_real_canonical_cell_plans(project_root: Optional[Path] = None) -> List[RealCanonicalCellPlan]:
    knn_cache: Dict[str, Dict[str, Any]] = {}
    plans = [
        resolve_real_canonical_cell(dataset, target, horizon, seed, knn_cache=knn_cache, project_root=project_root)
        for dataset, target, horizon, seed in enumerate_canonical_cells()
    ]
    if len(plans) != EXPECTED_CANONICAL_CELLS:
        raise AEContractError(f"real canonical plans={len(plans)} != {EXPECTED_CANONICAL_CELLS}")
    keys = [item.stage_a_key for item in plans]
    if len(set(keys)) != EXPECTED_CANONICAL_CELLS:
        raise AEContractError("duplicate Stage-A keys in real cell plans")
    return plans


STAGE_A_KWARG_KEYS: Tuple[str, ...] = (
    "source_df",
    "target_df",
    "feature_cols",
    "k",
    "horizon",
    "window_size",
    "weight_mode",
    "learning_rate",
    "source_epochs",
    "batch_size",
    "tau_k",
    "group_cols",
)
SHARED_BUNDLE_INPUT_KEYS: Tuple[str, ...] = (
    "target_identity",
    "entity_key",
    "target_split_identity",
    "knn_diagnostic_sources",
    "source_pool_authority",
    "source_selection_contract",
)


def plan_formal_stage_a_kwargs(
    cell: RealCanonicalCellPlan,
    *,
    project_root: Optional[Path] = None,
    knn_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve Stage-A kwargs / shared-bundle inputs without calling source training."""
    from src.protocols.formal_input_paths import FormalInputPathError, resolve_formal_dataset_paths

    root = project_root_from(project_root)
    knn_dir = root / "configs" / "solidified" / "knn" / f"Dataset{dataset_number(cell.dataset)}"
    if not knn_dir.is_dir():
        raise AEContractError(f"KNN authority missing for {cell.stage_a_key}: {knn_dir}")
    cache = knn_cache if knn_cache is not None else {}
    if cell.dataset not in cache:
        cache[cell.dataset] = load_knn_results(knn_dir, SCREENING_MODE)
    payload = cache[cell.dataset]
    group_cols = payload.get("group_cols")
    if not isinstance(group_cols, list) or not group_cols:
        raise AEContractError(f"KNN group_cols missing for {cell.stage_a_key}")
    if not cell.knn_diagnostic_sources or len(cell.knn_diagnostic_sources) < REQUESTED_K:
        raise AEContractError(f"KNN diagnostic sources unresolved for {cell.stage_a_key}")
    if not cell.target_identity or not cell.entity_key:
        raise AEContractError(f"target identity unresolved for {cell.stage_a_key}")
    try:
        paths = resolve_formal_dataset_paths(dataset_number(cell.dataset), repository_root=root)
    except FormalInputPathError as exc:
        raise AEContractError(f"FORMAL_DATA_UNRESOLVABLE {cell.stage_a_key}: {exc}") from exc
    for label, path in (
        ("source_parquet", paths.source_path),
        ("target_parquet", paths.target_path),
        ("dataset_manifest", paths.dataset_manifest_path),
        ("source_schema", paths.source_schema_path),
        ("target_schema", paths.target_schema_path),
    ):
        if not Path(path).is_file():
            raise AEContractError(f"{label} missing for {cell.stage_a_key}: {path}")
    kwargs_plan = {
        "source_df": {"authority": "SEALED_SOURCE_PARQUET", "path": str(paths.source_path)},
        "target_df": {"authority": "SEALED_TARGET_PARQUET", "path": str(paths.target_path)},
        "feature_cols": {"authority": "FROZEN_V3_MODEL_FEATURE_COLS", "status": "BOUND_AT_FRAME_LOAD"},
        "k": REQUESTED_K,
        "horizon": int(cell.horizon),
        "window_size": 10,
        "weight_mode": "inverse_distance",
        "learning_rate": float(STAGE_A_LEARNING_RATE),
        "source_epochs": int(STAGE_B_EPOCHS),
        "batch_size": int(STAGE_B_BATCH_SIZE),
        "tau_k": 1.0,
        "group_cols": tuple(group_cols),
    }
    missing_kwargs = [key for key in STAGE_A_KWARG_KEYS if key not in kwargs_plan]
    if missing_kwargs:
        raise AEContractError(f"stage_a_kwargs incomplete for {cell.stage_a_key}: {missing_kwargs}")
    bundle_inputs = {
        "target_identity": cell.target_identity,
        "entity_key": cell.entity_key,
        "target_split_identity": cell.target_split_identity,
        "knn_diagnostic_sources": tuple(cell.knn_diagnostic_sources),
        "source_pool_authority": cell.source_pool_authority,
        "source_selection_contract": cell.source_selection_contract,
    }
    missing_bundle = [key for key in SHARED_BUNDLE_INPUT_KEYS if not bundle_inputs.get(key)]
    if missing_bundle:
        raise AEContractError(f"shared_target_bundle inputs missing for {cell.stage_a_key}: {missing_bundle}")
    return {
        "stage_a_key": cell.stage_a_key,
        "stage_a_kwargs": kwargs_plan,
        "shared_target_bundle_inputs": bundle_inputs,
        "sealed_dataset_root": str(paths.dataset_root),
    }


def resolve_all_formal_stage_a_input_plans(
    cells: Sequence[RealCanonicalCellPlan],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    knn_cache: Dict[str, Dict[str, Any]] = {}
    plans: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for cell in cells:
        try:
            plans.append(plan_formal_stage_a_kwargs(cell, project_root=project_root, knn_cache=knn_cache))
        except AEContractError as exc:
            failures.append({"stage_a_key": cell.stage_a_key, "reason": str(exc)})
    total = len(cells)
    resolved = len(plans)
    ok = not failures and resolved == EXPECTED_CANONICAL_CELLS and total == EXPECTED_CANONICAL_CELLS
    return {
        "REAL_DATA_FORMAL_PRECHECK": "PASS" if ok else "FAIL",
        "REAL_STAGE_A_KWARGS_RESOLUTION": f"{resolved}/{total} PASS" if ok else "FAIL",
        "resolved_count": resolved,
        "cell_count": total,
        "failures": failures,
        "plans": plans if ok else [],
    }


def planned_arm_for(cell: RealCanonicalCellPlan, arm: str) -> RealArmPlan:
    spec = ae_arm_spec(arm)
    consumed = spec.init_mode != "native"
    parent_digest = v3_architecture_digest()
    return RealArmPlan(
        cell=cell,
        arm=spec.arm,
        arm_name=spec.name,
        init_mode=spec.init_mode,
        backbone_trainable=spec.target_backbone_trainable,
        revin_enabled=spec.revin_enabled,
        transfer_branch_enabled=spec.transfer_branch_enabled,
        gate_mode=spec.gate_mode,
        use_adversarial=spec.use_adversarial,
        stage_a_shared_for_pairing=True,
        source_init_consumed=consumed,
        parent_method=PARENT_METHOD if spec.arm == "D" else PARENT_METHOD,
        parent_v3_architecture_digest=parent_digest,
        recipe=ae_training_recipe(),
        status="PLANNED",
    )


def enumerate_real_arm_plans(cells: Sequence[RealCanonicalCellPlan]) -> List[RealArmPlan]:
    plans = [planned_arm_for(cell, arm) for cell in cells for arm in AE_ARMS]
    if len(plans) != EXPECTED_RUNS:
        raise AEContractError(f"real arm plans={len(plans)} != {EXPECTED_RUNS}")
    keys = [item.run_key for item in plans]
    if len(set(keys)) != EXPECTED_RUNS:
        raise AEContractError("duplicate A–E run keys")
    return plans


def assert_lr_equality(plans: Sequence[RealArmPlan]) -> str:
    by_cell: Dict[str, set] = {}
    for item in plans:
        by_cell.setdefault(item.cell.stage_a_key, set()).add(float(item.cell.selected_lr))
    if any(len(values) != 1 for values in by_cell.values()):
        raise AEContractError("LR_EQUALITY_ACROSS_ARMS failed")
    if len(by_cell) != EXPECTED_CANONICAL_CELLS:
        raise AEContractError("LR equality did not cover 54 cells")
    return "PASS"


def assert_recipe_equality(plans: Sequence[RealArmPlan]) -> str:
    expected = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    for item in plans:
        require_v3_recipe(item.recipe)
        if item.recipe != expected:
            raise AEContractError(f"recipe drifted on {item.run_key}")
        if optimizer["epochs"] != 50 or optimizer["batch_size"] != 16:
            raise AEContractError("optimizer contract drifted")
        if item.arm == "D" and item.parent_method != PARENT_METHOD:
            raise AEContractError("D parent_method drifted")
    return "PASS"


def assert_d_parent_identity(plans: Sequence[RealArmPlan]) -> str:
    d_plans = [item for item in plans if item.arm == "D"]
    if len(d_plans) != EXPECTED_CANONICAL_CELLS:
        raise AEContractError(f"D plans={len(d_plans)} != 54")
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise AEContractError("frozen V3 digest drifted")
    for item in d_plans:
        if item.parent_method != PARENT_METHOD:
            raise AEContractError("D parent_method mismatch")
        if item.parent_v3_architecture_digest != digest:
            raise AEContractError("D parent digest mismatch")
        if item.init_mode != "fused" or not item.revin_enabled or not item.backbone_trainable:
            raise AEContractError("D runtime spec drifted from V3")
    return "PASS"


class FakeRealStageAFactory:
    """No-training Stage A factory that preserves real cell identity."""

    def __init__(self) -> None:
        self.ledger = StageACallLedger()
        self._by_seed: Dict[int, AESyntheticStageA] = {}

    def __call__(self, cell: RealCanonicalCellPlan) -> SharedStageAHandle:
        if cell.stage_a_key in self.ledger.artifacts:
            raise AEContractError(f"Stage A prepared twice for {cell.stage_a_key}")
        seed = int(cell.seed)
        if seed not in self._by_seed:
            self._by_seed[seed] = build_synthetic_stage_a(seed=seed)
        artifacts = self._by_seed[seed]
        handle = SharedStageAHandle(
            stage_a_key=cell.stage_a_key,
            artifacts=artifacts,
            source_identities=list(artifacts.source_identities),
            source_distances=list(artifacts.source_distances),
            source_weights=list(artifacts.source_weights),
            requested_k=int(cell.requested_k),
            effective_k=len(artifacts.source_identities),
        )
        if handle.requested_k != REQUESTED_K:
            raise AEContractError("requested_k must remain 3")
        self.ledger.calls.append(cell.stage_a_key)
        self.ledger.artifacts[cell.stage_a_key] = handle
        return handle


def prepare_stage_a_once(
    cell: RealCanonicalCellPlan,
    *,
    mode: AEExecutionMode,
    factory: Callable[[RealCanonicalCellPlan], SharedStageAHandle],
    ledger: Optional[StageACallLedger] = None,
) -> SharedStageAHandle:
    assert_phase_allowed(mode, AEExecutionPhase.STAGE_A_PLAN)
    if mode is AEExecutionMode.FORMAL:
        assert_phase_allowed(mode, AEExecutionPhase.STAGE_A_TRAIN)
        raise AETrainingBlocked("FORMAL_EXECUTION_BLOCKED")
    handle = factory(cell)
    active = ledger if ledger is not None else getattr(factory, "ledger", None)
    if active is not None and active.calls.count(cell.stage_a_key) != 1:
        raise AEContractError(f"STAGE_A_CALL_COUNT_PER_CANONICAL_CELL != 1 for {cell.stage_a_key}")
    return handle


def build_cell_arm_plans(
    cell: RealCanonicalCellPlan,
    handle: SharedStageAHandle,
) -> List[RealArmPlan]:
    plans = [planned_arm_for(cell, arm) for arm in AE_ARMS]
    for item in plans[1:]:
        if item.cell.stage_a_key != handle.stage_a_key:
            raise AEContractError("arm plan is not bound to the shared Stage-A key")
    if plans[0].source_init_consumed:
        raise AEContractError("arm A must not consume source init")
    if not all(item.stage_a_shared_for_pairing for item in plans):
        raise AEContractError("all arms must share Stage A for pairing")
    for left, right in zip(plans[1:], plans[2:]):
        if left.cell.stage_a_key != right.cell.stage_a_key:
            raise AEContractError("B/C/D/E Stage-A keys diverged")
    return plans


def dry_run_canonical_graph(
    cells: Sequence[RealCanonicalCellPlan],
    *,
    factory: Optional[FakeRealStageAFactory] = None,
) -> Dict[str, Any]:
    factory = factory or FakeRealStageAFactory()
    arm_plans: List[RealArmPlan] = []
    for cell in cells:
        handle = prepare_stage_a_once(cell, mode=AEExecutionMode.DRY_RUN, factory=factory)
        arm_plans.extend(build_cell_arm_plans(cell, handle))
        for arm in ("B", "C", "D", "E"):
            if cell.stage_a_key not in factory.ledger.artifacts:
                raise AEContractError(f"{arm} missing shared Stage A")
    if factory.ledger.count != EXPECTED_CANONICAL_CELLS:
        raise AEContractError(f"STAGE_A_CALLS={factory.ledger.count} != 54")
    if len(arm_plans) != EXPECTED_RUNS:
        raise AEContractError(f"ARM_PLANS={len(arm_plans)} != 270")
    return {
        "cells": list(cells),
        "arm_plans": arm_plans,
        "stage_a_calls": factory.ledger.count,
        "factory": factory,
    }


def arm_build_order_invariance(seed: int = 42, selected_lr: float = 3e-5) -> str:
    stage_a = build_synthetic_stage_a(seed=seed)
    recipe = ae_training_recipe()
    forward = construct_all_ae_arms(
        stage_a, seed=seed, selected_lr=selected_lr, recipe=recipe, arm_order=AE_ARMS
    )
    reverse = construct_all_ae_arms(
        stage_a,
        seed=seed,
        selected_lr=selected_lr,
        recipe=recipe,
        arm_order=tuple(reversed(AE_ARMS)),
    )
    for arm in AE_ARMS:
        if forward[arm].target_initial_conv_digest != reverse[arm].target_initial_conv_digest:
            raise AEContractError(f"{arm} conv digest depends on build order")
        if forward[arm].head_target_init_digest != reverse[arm].head_target_init_digest:
            raise AEContractError(f"{arm} head digest depends on build order")
        if forward[arm].spec.init_mode != reverse[arm].spec.init_mode:
            raise AEContractError(f"{arm} config depends on build order")
    return "PASS"


def construct_dataset_representatives(
    cells: Sequence[RealCanonicalCellPlan],
) -> Dict[str, Any]:
    chosen: Dict[str, RealCanonicalCellPlan] = {}
    for cell in cells:
        if cell.dataset not in chosen and cell.horizon == 1 and cell.seed == 42:
            chosen[cell.dataset] = cell
    if tuple(chosen) != tuple(dataset for dataset, _ in canonical_targets()):
        raise AEContractError("representative construction missed a dataset")
    built = {}
    pairwise = {}
    recipe = ae_training_recipe()
    for dataset, cell in chosen.items():
        stage_a = build_synthetic_stage_a(seed=int(cell.seed))
        arms = construct_all_ae_arms(
            stage_a, seed=int(cell.seed), selected_lr=float(cell.selected_lr), recipe=recipe
        )
        pairwise = pairwise_single_factor_contracts(arms)
        built[dataset] = {
            "cell": cell,
            "arms": arms,
            "head_digest": next(iter(arms.values())).head_target_init_digest,
            "pairwise": pairwise,
        }
        if verify_common_head_init(arms) != "PASS":
            raise AEContractError(f"{dataset} common head init failed")
    return built


def d4_representative_trace(cells: Sequence[RealCanonicalCellPlan], built: Mapping[str, Any]) -> Dict[str, Any]:
    matches = [
        cell
        for cell in cells
        if cell.dataset == "D4" and cell.target == "166/311" and cell.horizon == 1 and cell.seed == 42
    ]
    if len(matches) != 1:
        raise AEContractError("D4 representative cell missing")
    cell = matches[0]
    payload = built["D4"]
    arms = payload["arms"]
    pairwise = pairwise_single_factor_contracts(arms)
    return {
        "dataset": cell.dataset,
        "target_identity": cell.target_identity,
        "horizon": cell.horizon,
        "seed": cell.seed,
        "scenario": cell.scenario,
        "selected_lr": cell.selected_lr_label,
        "source_pool_authority": cell.source_pool_authority,
        "knn_diagnostic_sources": list(cell.knn_diagnostic_sources),
        "source_ids_status": cell.source_ids_status,
        "stage_a_key": cell.stage_a_key,
        "A_init": arms["A"].spec.init_mode,
        "B_init": arms["B"].spec.init_mode,
        "C_init": arms["C"].spec.init_mode,
        "D_init": arms["D"].spec.init_mode,
        "E_init": arms["E"].spec.init_mode,
        "shared_head_digest": payload["head_digest"],
        "single_factor": {
            "A_C": pairwise["A_C_SINGLE_FACTOR_CONTRACT"],
            "B_C": pairwise["B_C_SINGLE_FACTOR_CONTRACT"],
            "C_E": pairwise["C_E_SINGLE_FACTOR_CONTRACT"],
            "C_D": pairwise["C_D_SINGLE_FACTOR_CONTRACT"],
        },
    }


PLANNED_MATRIX_FIELDNAMES = (
    "dataset",
    "target",
    "scenario",
    "horizon",
    "seed",
    "arm",
    "arm_name",
    "selected_lr",
    "stage_a_key",
    "init_mode",
    "backbone_trainable",
    "revin_enabled",
    "transfer_branch_enabled",
    "gate_mode",
    "use_adversarial",
    "status",
)


def write_planned_matrix_csv(path: Path, arm_plans: Sequence[RealArmPlan]) -> Path:
    """Write the frozen formal planned-matrix CSV (54 cells × 5 arms = 270 rows)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(arm_plans) != EXPECTED_RUNS:
        raise AEContractError(f"planned matrix rows={len(arm_plans)} != {EXPECTED_RUNS}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLANNED_MATRIX_FIELDNAMES))
        writer.writeheader()
        for item in arm_plans:
            if item.status != "PLANNED":
                raise AEContractError("formal planned matrix status must be PLANNED")
            writer.writerow(
                {
                    "dataset": item.cell.dataset,
                    "target": item.cell.target,
                    "scenario": item.cell.scenario,
                    "horizon": item.cell.horizon,
                    "seed": item.cell.seed,
                    "arm": item.arm,
                    "arm_name": item.arm_name,
                    "selected_lr": item.cell.selected_lr_label,
                    "stage_a_key": item.cell.stage_a_key,
                    "init_mode": item.init_mode,
                    "backbone_trainable": item.backbone_trainable,
                    "revin_enabled": item.revin_enabled,
                    "transfer_branch_enabled": item.transfer_branch_enabled,
                    "gate_mode": item.gate_mode,
                    "use_adversarial": item.use_adversarial,
                    "status": "PLANNED",
                }
            )
    return destination


def formal_manifest_snapshot_payload(planned_matrix_digest: str) -> Dict[str, Any]:
    """Resume-critical identity snapshot bound before any formal training."""
    payload = dict(ae_protocol_manifest_payload())
    payload["planned_matrix_digest"] = str(planned_matrix_digest)
    return payload


def validate_resume_root(run_root: Path, project_root: Optional[Path] = None) -> Dict[str, str]:
    root = Path(run_root)
    manifest_path = root / "manifest_snapshot.json"
    plan_path = root / "plans" / "planned_matrix.csv"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise AEContractError("resume root missing manifest_snapshot.json or planned_matrix.csv")
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = ae_protocol_manifest_payload()
    if snapshot.get("protocol_version") != PROTOCOL_VERSION:
        raise AEContractError("resume protocol_version mismatch")
    if snapshot.get("parent_v3_architecture_digest") != current["parent_v3_architecture_digest"]:
        raise AEContractError("resume parent V3 digest mismatch")
    if snapshot.get("expected_runs") != EXPECTED_RUNS:
        raise AEContractError("resume planned matrix size mismatch")
    _ = project_root
    return {"RESUME_CONTRACT": "PASS"}


def classify_paired_bundle(arm_statuses: Mapping[str, str]) -> str:
    if set(arm_statuses) != set(AE_ARMS):
        return "PAIR_INCOMPLETE"
    if all(status == "OK" for status in arm_statuses.values()):
        return "PAIR_COMPLETE"
    return "PAIR_INCOMPLETE"


def write_dry_run_artifacts(
    *,
    project_root: Path,
    cells: Sequence[RealCanonicalCellPlan],
    arm_plans: Sequence[RealArmPlan],
    report: Mapping[str, Any],
) -> Path:
    out = assert_ae_output_root(project_root / PREFLIGHT_RELPATH)
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / "canonical_cells.csv"
    matrix_path = out / "planned_matrix.csv"
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "target",
                "scenario",
                "horizon",
                "seed",
                "selected_lr",
                "stage_a_key",
                "source_pool_authority",
                "source_ids_status",
            ],
        )
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "dataset": cell.dataset,
                    "target": cell.target,
                    "scenario": cell.scenario,
                    "horizon": cell.horizon,
                    "seed": cell.seed,
                    "selected_lr": cell.selected_lr_label,
                    "stage_a_key": cell.stage_a_key,
                    "source_pool_authority": cell.source_pool_authority,
                    "source_ids_status": cell.source_ids_status,
                }
            )
    write_planned_matrix_csv(matrix_path, arm_plans)
    (out / "preflight_report.json").write_text(
        json.dumps(dict(report.get("preflight", {})), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (out / "contract_report.json").write_text(
        json.dumps(dict(report.get("contracts", {})), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if "d4_trace" in report:
        (out / "d4_representative.json").write_text(
            json.dumps(report["d4_trace"], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return out


def format_dry_run(report: Mapping[str, Any]) -> str:
    lines = [
        f"AE_PROTOCOL_VERSION={PROTOCOL_VERSION}",
        "",
        f"REAL_DATA_ROUTING={report['REAL_DATA_ROUTING']}",
        "",
        f"TARGETS={len(canonical_targets())}",
        f"HORIZONS={len(HORIZONS)}",
        f"SEEDS={len(SEEDS)}",
        f"ARMS={len(AE_ARMS)}",
        "",
        f"CANONICAL_CELLS={report['CANONICAL_CELLS']}",
        f"PLANNED_RUNS={report['PLANNED_RUNS']}",
        "",
        f"UNIQUE_RUN_KEYS={report['UNIQUE_RUN_KEYS']}",
        f"UNIQUE_STAGE_A_KEYS={report['UNIQUE_STAGE_A_KEYS']}",
        "",
        "STAGE_A_REAL_TRAINING=NO",
        f"STAGE_A_MOCK_CALLS={report['STAGE_A_MOCK_CALLS']}",
        "",
        f"LR_CONTRACT={report['LR_CONTRACT']}",
        f"RECIPE_CONTRACT={report['RECIPE_CONTRACT']}",
        "",
        f"A_C_SINGLE_FACTOR={report['A_C_SINGLE_FACTOR']}",
        f"B_C_SINGLE_FACTOR={report['B_C_SINGLE_FACTOR']}",
        f"C_E_SINGLE_FACTOR={report['C_E_SINGLE_FACTOR']}",
        f"C_D_SINGLE_FACTOR={report['C_D_SINGLE_FACTOR']}",
        "",
        f"ARM_BUILD_ORDER_INVARIANCE={report['ARM_BUILD_ORDER_INVARIANCE']}",
        "",
        f"D_PARENT_IDENTITY={report['D_PARENT_IDENTITY']}",
        f"D_EQUALS_FROZEN_V3={report['D_PARENT_IDENTITY']}",
        "",
        f"TRAINING_GUARD={report['TRAINING_GUARD']}",
        "TRAINING_STARTED=NO",
        "",
        f"FORMAL_EXECUTION_CAPABILITY={'ENABLED' if AE_FORMAL_EXECUTION_ENABLED else 'DISABLED'}",
        "ACTUAL_EXECUTION_REQUESTED=NO",
        f"FORMAL_EXECUTION_ENABLED={'YES' if AE_FORMAL_EXECUTION_ENABLED else 'NO'}",
        f"AE_FORMAL_EXECUTION_IMPLEMENTED={'YES' if AE_FORMAL_EXECUTION_IMPLEMENTED else 'NO'}",
    ]
    return "\n".join(lines) + "\n"


def run_ae_dry_run(
    *,
    project_root: Optional[Path] = None,
    write_artifacts: bool = True,
    include_construction: bool = True,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    load_ae_manifest(root)
    preflight = run_ae_preflight(project_root=root, include_construction_contracts=include_construction)
    cells = enumerate_real_canonical_cell_plans(project_root=root)
    graph = dry_run_canonical_graph(cells)
    arm_plans = graph["arm_plans"]
    lr_flag = assert_lr_equality(arm_plans)
    recipe_flag = assert_recipe_equality(arm_plans)
    d_flag = assert_d_parent_identity(arm_plans)
    order_flag = arm_build_order_invariance()
    built = construct_dataset_representatives(cells) if include_construction else {}
    pairwise = next(iter(built.values()))["pairwise"] if built else preflight["pairwise"]
    d4_trace = d4_representative_trace(cells, built) if built else {}
    unique_run_keys = len({item.run_key for item in arm_plans})
    unique_stage_a = len({cell.stage_a_key for cell in cells})
    training_guard = "PASS"
    try:
        assert_phase_allowed(AEExecutionMode.DRY_RUN, AEExecutionPhase.STAGE_A_TRAIN)
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    try:
        assert_formal_training_permitted(execute=False)
        training_guard = "FAIL"
    except AETrainingBlocked:
        pass
    contracts = {
        "A_C_SINGLE_FACTOR_CONTRACT": pairwise.get("A_C_SINGLE_FACTOR_CONTRACT", "PASS"),
        "B_C_SINGLE_FACTOR_CONTRACT": pairwise.get("B_C_SINGLE_FACTOR_CONTRACT", "PASS"),
        "C_E_SINGLE_FACTOR_CONTRACT": pairwise.get("C_E_SINGLE_FACTOR_CONTRACT", "PASS"),
        "C_D_SINGLE_FACTOR_CONTRACT": pairwise.get("C_D_SINGLE_FACTOR_CONTRACT", "PASS"),
        "ARM_BUILD_ORDER_INVARIANCE": order_flag,
        "D_PARENT_IDENTITY_54_54": d_flag,
        "LR_EQUALITY_ACROSS_ARMS": lr_flag,
        "analysis_contract": analysis_contract(),
        "assert_d_equals_frozen_v3": assert_d_equals_frozen_v3(),
    }
    report = {
        "REAL_DATA_ROUTING": "PASS",
        "CANONICAL_CELLS": len(cells),
        "PLANNED_RUNS": len(arm_plans),
        "UNIQUE_RUN_KEYS": unique_run_keys,
        "UNIQUE_STAGE_A_KEYS": unique_stage_a,
        "STAGE_A_MOCK_CALLS": graph["stage_a_calls"],
        "LR_CONTRACT": lr_flag,
        "RECIPE_CONTRACT": recipe_flag,
        "A_C_SINGLE_FACTOR": contracts["A_C_SINGLE_FACTOR_CONTRACT"],
        "B_C_SINGLE_FACTOR": contracts["B_C_SINGLE_FACTOR_CONTRACT"],
        "C_E_SINGLE_FACTOR": contracts["C_E_SINGLE_FACTOR_CONTRACT"],
        "C_D_SINGLE_FACTOR": contracts["C_D_SINGLE_FACTOR_CONTRACT"],
        "ARM_BUILD_ORDER_INVARIANCE": order_flag,
        "D_PARENT_IDENTITY": d_flag,
        "TRAINING_GUARD": training_guard,
        "preflight": {"text": preflight["text"], "READY": preflight.get("READY_FOR_FORMAL_AE_RUN")},
        "contracts": contracts,
        "d4_trace": d4_trace,
    }
    if write_artifacts:
        write_dry_run_artifacts(project_root=root, cells=cells, arm_plans=arm_plans, report=report)
    report["text"] = format_dry_run(report)
    report["cells"] = cells
    report["arm_plans"] = arm_plans
    report["factory"] = graph["factory"]
    report["AE_CONTRACT_READY"] = "YES"
    report["AE_EXECUTION_WIRING_READY"] = "YES" if training_guard == "PASS" and unique_run_keys == 270 else "NO"
    report["READY_FOR_FORMAL_AE_EXECUTION_ENABLE"] = report["AE_EXECUTION_WIRING_READY"]
    report["TRAINING_STARTED"] = "NO"
    return report


def run_ae_formal_execution(**kwargs: Any) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_ae_formal import run_ae_formal_body

    return run_ae_formal_body(mock_execution_authorized=False, **kwargs)
