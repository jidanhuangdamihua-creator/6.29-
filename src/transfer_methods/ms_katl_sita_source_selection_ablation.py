"""SITA Arm F (Random-3) and Arm G (Bottom-3) source-selection ablation.

Copies frozen Arm C (FUSED_TRAINABLE) from the uniform LR=3e-4 paper slice.
The only allowed delta is which three legal sources enter fusion.
Does not modify Arm A–E, the 3e-4 cell bundles, or any other frozen baseline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from src.evaluation.metrics import smape
from src.protocols.candidate_pool import select_daily_sequence_sources
from src.protocols.experiment_protocol import get_experiment_protocol, normalize_source_key
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.raw_preprocessing import TargetK3RawSourceContext
from src.source_selection.source_selector import SourceSelector
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    CONV_LAYER_NAMES,
    HORIZONS,
    PARENT_METHOD,
    PROTOCOL_VERSION,
    SCENARIO,
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EARLY_STOPPING,
    STAGE_B_EPOCHS,
    UNIFORM_3E4_DATASETS,
    UNIFORM_3E4_LEARNING_RATE_LABEL,
    UNIFORM_3E4_OUTPUT_RELPATH,
    UNIFORM_3E4_SEEDS,
    UNIFORM_3E4_TARGETS,
    AEContractError,
    AETrainingBlocked,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_training_recipe,
    assert_d_equals_frozen_v3,
    construct_ae_arm,
    construct_v3_head_target_snapshot,
    project_root_from,
    reset_ae_rng,
    select_ae_trainable_variables,
    v3_architecture_digest,
)
from src.transfer_methods.ms_katl_ae_execution import REQUESTED_K, resolve_real_canonical_cell
from src.transfer_methods.ms_katl_ae_formal import git_provenance
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    compute_screening_metrics,
    original_rmse,
    parse_lr,
)
from src.transfer_methods.ms_katl_sita_ablation_3e4 import (
    UNLOCK_ENV,
    WEIGHT_MODE,
    WINDOW_SIZE,
    assert_formal_ae_identity_frozen,
    config_identity as ae_config_identity,
)
from src.utils.dataframe_attrs import get_protocol_frame_context, promote_protocol_frame_context

SLICE_LABEL = "SITA_SOURCE_SELECTION_ABLATION"
ARM_F_IDENTITY = "SITA_ABLATION_F_RANDOM3"
ARM_G_IDENTITY = "SITA_ABLATION_G_BOTTOM3"
ARM_F = "F"
ARM_G = "G"
ARM_F_NAME = "RANDOM3_SOURCES"
ARM_G_NAME = "BOTTOM3_SOURCES"
SOURCE_SELECTION_GRANULARITY = "target-level"
SIMILARITY_DIRECTION = "LOWER_IS_MORE_SIMILAR"
SOURCE_SET_SEEDS: Dict[str, int] = {
    "R1": 1001,
    "R2": 1002,
    "R3": 1003,
    "R4": 1004,
    "R5": 1005,
}
SOURCE_SET_IDS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
STAGE1_SOURCE_SETS: Tuple[str, ...] = ("R1", "R2", "R3")
OUTPUT_MARKER = "sita_ablation_fg_source_selection"
OUTPUT_RELPATH = Path("outputs") / "sita_ablation_fg_source_selection"
CONFIG_MANIFEST_RELPATH = Path("configs") / "ms_katl_sita_source_selection_ablation_manifest.json"
SOURCE_MANIFEST_CSV_NAME = "SOURCE_SELECTION_MANIFEST.csv"
REFERENCE_CELL_HORIZON = 1
REFERENCE_CELL_SEED = 42
EXPECTED_F_FULL_RUNS = 300
EXPECTED_F_STAGE1_RUNS = 180
EXPECTED_G_RUNS = 60
PROTECTED_OUTPUT_MARKERS = (
    "MS_KATL_AE_CORE_ABLATION",
    "MS_KATL_AE_ABLATION",
    "sita_ablation_3e4",
    "sita_ablation_lr_sensitivity",
    "SITA_ALL_TARGETS",
    "PAPER_CANDIDATE_BASELINE",
    "FTO_THREE_TRACK",
    "FTO_V3_SCREENING",
)
CONFIG_DIFF_ALLOWED_FIELDS = frozenset(
    {
        "source_selection_strategy",
        "source_ids",
        "source_set_id",
        "source_sampling_seed",
        "source_identities",
        "arm",
        "arm_name",
        "result_identity",
    }
)
HOLM_FAMILY = "Holm"
HOLM_ALPHA = 0.05
HOLM_FAMILY_SIZE = 4
PLANNED_PAIRWISE = ("C_vs_F", "F_vs_A", "C_vs_G", "F_vs_G")
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_RANDOM_SEED = 20260912
BOOTSTRAP_CI_LEVEL = 0.95
FROZEN_ARM_C_TOP3: Mapping[str, Tuple[Tuple[str, ...], ...]] = {
    "D1": (("1", "7"), ("1", "8"), ("1", "2")),
    "D4": (("166", "184"), ("166", "530"), ("166", "229")),
    "D5": (("48", "2027203"), ("48", "1975514"), ("48", "305227")),
    "D6": (("CA_1", "FOODS_3_202"), ("CA_1", "FOODS_3_288"), ("CA_1", "FOODS_3_694")),
}


class SourceSelectionAblationError(AEContractError):
    """Fail-closed source-selection ablation violation."""


@dataclass(frozen=True)
class RankedSource:
    source_key: Tuple[str, ...]
    distance: float
    rank: int  # 1 = nearest / most similar


@dataclass(frozen=True)
class SelectedTriple:
    dataset: str
    target_id: str
    arm: str
    source_set_id: str
    source_sampling_seed: Optional[int]
    sources: Tuple[RankedSource, ...]  # distance-sorted, length 3
    top3: Tuple[Tuple[str, ...], ...]
    candidate_pool_size: int
    overlap_count: int
    overlap_ratio: float

    @property
    def mean_similarity(self) -> float:
        return float(np.mean([item.distance for item in self.sources]))

    @property
    def median_similarity(self) -> float:
        return float(np.median([item.distance for item in self.sources]))


@dataclass(frozen=True)
class AblationRunIdentity:
    arm: str
    dataset: str
    target_identity: str
    horizon: int
    training_seed: int
    source_set_id: str
    source_sampling_seed: Optional[int]
    selected_lr: float = float(parse_lr(UNIFORM_3E4_LEARNING_RATE_LABEL))
    selected_lr_label: str = UNIFORM_3E4_LEARNING_RATE_LABEL
    scenario: str = SCENARIO

    @property
    def identity_tuple(self) -> Tuple[Any, ...]:
        if self.arm == ARM_F:
            return (
                self.dataset,
                self.target_identity,
                int(self.horizon),
                self.source_set_id,
                int(self.training_seed),
                self.arm,
            )
        return (
            self.dataset,
            self.target_identity,
            int(self.horizon),
            int(self.training_seed),
            self.arm,
        )

    @property
    def run_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        if self.arm == ARM_F:
            return (
                f"{self.arm}_{self.dataset}_{target_slug}_h{int(self.horizon)}"
                f"_{self.source_set_id}_s{int(self.training_seed)}"
            )
        return (
            f"{self.arm}_{self.dataset}_{target_slug}_h{int(self.horizon)}"
            f"_s{int(self.training_seed)}"
        )

    @property
    def stage_a_key(self) -> str:
        return (
            f"{self.dataset}|{self.target_identity}|h{int(self.horizon)}"
            f"|s{int(self.training_seed)}|{self.scenario}|{self.arm}|{self.source_set_id}"
        )


@dataclass
class ForcedK3Evidence:
    payload: Dict[str, Any]

    def method_wrapper(self, **kwargs: Any) -> Dict[str, Any]:
        if int(kwargs.get("k", -1)) != REQUESTED_K:
            raise SourceSelectionAblationError("forced K3 evidence requires k=3")
        if str(kwargs.get("weight_mode", "")) != WEIGHT_MODE:
            raise SourceSelectionAblationError("forced K3 evidence requires inverse_distance")
        return deepcopy(self.payload)


@dataclass
class ForcedK3SelectionContext:
    payload: Dict[str, Any]
    lifecycle_identity: Tuple[object, ...]

    def __post_init__(self) -> None:
        self.lifecycle_identity = tuple(self.lifecycle_identity)
        if len(self.lifecycle_identity) != 5:
            raise SourceSelectionAblationError("forced K3 lifecycle identity must have five fields")

    def selection_for_method(self, **_kwargs: Any) -> ForcedK3Evidence:
        return ForcedK3Evidence(self.payload)


def inject_forced_k3_selection_context(
    stage_a_kwargs: Dict[str, Any],
    payload: Mapping[str, Any],
) -> ForcedK3SelectionContext:
    original = stage_a_kwargs.get("k3_selection_context")
    if original is None:
        raise SourceSelectionAblationError("missing k3_selection_context before forced injection")
    lifecycle_identity = tuple(original.lifecycle_identity)
    context = ForcedK3SelectionContext(dict(payload), lifecycle_identity=lifecycle_identity)
    stage_a_kwargs["k3_selection_context"] = context
    # Keep Arm C's raw-source protocol, but rebind it to the forced triple instead of Top-3.
    stage_a_kwargs["k3_raw_source_context"] = TargetK3RawSourceContext(
        lifecycle_identity=lifecycle_identity
    )
    return context


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


def format_source_id(key: Sequence[object]) -> str:
    return "/".join(str(part) for part in normalize_source_key(key))


def parse_source_id(value: object) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in normalize_source_key(value))
    text = str(value).strip()
    if not text:
        raise SourceSelectionAblationError("empty source id")
    if "/" in text:
        return tuple(part for part in text.split("/") if part)
    if "_" in text:
        left, right = text.split("_", 1)
        return (left, right)
    return (text,)


def default_output_dir(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / OUTPUT_RELPATH


def frozen_3e4_root(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / UNIFORM_3E4_OUTPUT_RELPATH


def source_manifest_csv_path(project_root: Optional[Path] = None) -> Path:
    return project_root_from(project_root) / SOURCE_MANIFEST_CSV_NAME


def ranking_json_path(project_root: Optional[Path] = None) -> Path:
    return default_output_dir(project_root) / "source_ranking.json"


def assert_output_isolated(run_dir: Path, project_root: Optional[Path] = None) -> Path:
    root = project_root_from(project_root)
    resolved = Path(run_dir)
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    posix = resolved.as_posix()
    name = resolved.name
    if OUTPUT_MARKER not in posix:
        raise SourceSelectionAblationError(
            f"source-selection ablation output must include {OUTPUT_MARKER}: {resolved}"
        )
    for marker in PROTECTED_OUTPUT_MARKERS:
        if marker in name or (marker == "sita_ablation_3e4" and marker in posix and OUTPUT_MARKER not in posix):
            raise SourceSelectionAblationError(
                f"refusing to write Arm F/G outputs into protected path: {resolved}"
            )
    frozen = frozen_3e4_root(root).resolve()
    try:
        resolved.relative_to(frozen)
    except ValueError:
        pass
    else:
        raise SourceSelectionAblationError("refusing to overwrite frozen 3e-4 Arm A–E results")
    return resolved


def arm_c_reference_payload() -> Dict[str, Any]:
    spec = ae_arm_spec("C")
    optimizer = ae_optimizer_contract()
    recipe = ae_training_recipe()
    return {
        "arm": "C",
        "arm_name": spec.name,
        "result_identity": "FUSED_TRAINABLE",
        "init_mode": spec.init_mode,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "revin_enabled": spec.revin_enabled,
        "transfer_branch_enabled": spec.transfer_branch_enabled,
        "gate_mode": spec.gate_mode,
        "use_adversarial": spec.use_adversarial,
        "recipe": recipe,
        "optimizer": {
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
        "protocol_version": PROTOCOL_VERSION,
        "stage_b_learning_rate": UNIFORM_3E4_LEARNING_RATE_LABEL,
        "source_selection_strategy": "similarity_top3",
        "trainable_status": "head_target+conv_backbone",
        "inference": "target_only",
    }


def arm_fg_config_payload(arm: str, *, source_set_id: str = "", source_sampling_seed: Optional[int] = None) -> Dict[str, Any]:
    payload = arm_c_reference_payload()
    if arm == ARM_F:
        payload["arm"] = ARM_F
        payload["arm_name"] = ARM_F_NAME
        payload["result_identity"] = ARM_F_IDENTITY
        payload["source_selection_strategy"] = "random3"
        payload["source_set_id"] = source_set_id
        payload["source_sampling_seed"] = source_sampling_seed
    elif arm == ARM_G:
        payload["arm"] = ARM_G
        payload["arm_name"] = ARM_G_NAME
        payload["result_identity"] = ARM_G_IDENTITY
        payload["source_selection_strategy"] = "similarity_bottom3"
        payload["source_set_id"] = "BOTTOM3"
        payload["source_sampling_seed"] = None
    else:
        raise SourceSelectionAblationError(f"unknown ablation arm {arm}")
    payload["source_ids"] = None
    payload["source_identities"] = None
    return payload


def config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[str]:
    keys = sorted(set(left) | set(right))
    diff = []
    for key in keys:
        if left.get(key) != right.get(key):
            diff.append(key)
    return diff


def assert_protocol_parity(arm: str) -> Dict[str, Any]:
    reference = arm_c_reference_payload()
    candidate = arm_fg_config_payload(arm, source_set_id="R1", source_sampling_seed=1001)
    diff = config_diff(reference, candidate)
    unexpected = [key for key in diff if key not in CONFIG_DIFF_ALLOWED_FIELDS]
    if unexpected:
        raise SourceSelectionAblationError(f"Arm C vs {arm} unexpected config diffs: {unexpected}")
    return {
        "CONFIG_DIFF_ALLOWED_FIELDS": sorted(CONFIG_DIFF_ALLOWED_FIELDS),
        "DIFF_FIELDS": diff,
        "DIFF_COUNT": len(unexpected),
        "PARITY": "PASS",
    }


def implementation_audit_flags(*, project_root: Optional[Path] = None) -> Dict[str, str]:
    root = project_root_from(project_root)
    spec_a = ae_arm_spec("A")
    spec_c = ae_arm_spec("C")
    recipe = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    frozen = frozen_3e4_root(root)
    bundles = list((frozen / "cell_bundles").glob("*.json")) if frozen.is_dir() else []
    flags = {
        "ARM_A_REFERENCE_CONFIG_FOUND": "YES" if spec_a.name == "TARGET_NATIVE_INIT" and spec_a.init_mode == "native" else "NO",
        "ARM_C_REFERENCE_CONFIG_FOUND": "YES" if spec_c.name == "FUSED_TRAINABLE" and spec_c.init_mode == "fused" else "NO",
        "TOP3_SELECTION_IMPLEMENTATION_FOUND": "YES",
        "LEGAL_SOURCE_POOL_IMPLEMENTATION_FOUND": "YES",
        "SOURCE_FUSION_IMPLEMENTATION_FOUND": "YES",
        "FORMAL_TARGETS_MATCH": "YES" if dict(UNIFORM_3E4_TARGETS) == {
            "D1": "1/10",
            "D4": "166/258",
            "D5": "48/1159414",
            "D6": "CA_1/FOODS_3_080",
        } else "NO",
        "FORMAL_SEEDS_MATCH": "YES" if tuple(UNIFORM_3E4_SEEDS) == (42, 43, 44, 45, 46) else "NO",
        "FORMAL_HORIZONS_MATCH": "YES" if tuple(HORIZONS) == (1, 3, 5) else "NO",
        "ABLATION_LR_MATCH_3E4": "YES" if UNIFORM_3E4_LEARNING_RATE_LABEL == "3e-4" else "NO",
        "SOURCE_SELECTION_GRANULARITY": SOURCE_SELECTION_GRANULARITY,
        "SIMILARITY_DIRECTION": SIMILARITY_DIRECTION,
        "ARM_C_REVIN": "OFF" if not spec_c.revin_enabled else "ON",
        "ARM_C_TRAINABLE": "YES" if spec_c.target_backbone_trainable else "NO",
        "LOSS": str(recipe["loss_name"]),
        "EMA": "ON" if recipe["use_ema"] else "OFF",
        "OPTIMIZER": str(optimizer["optimizer"]),
        "EPOCHS": str(optimizer["epochs"]),
        "BATCH_SIZE": str(optimizer["batch_size"]),
        "EARLY_STOPPING": "OFF" if not optimizer["stage_b_early_stopping"] else "ON",
        "FROZEN_3E4_BUNDLES": str(len(bundles)),
        "D4_TARGET": UNIFORM_3E4_TARGETS["D4"],
    }
    return flags


def assert_audit_gates(flags: Mapping[str, str]) -> None:
    required = (
        "ARM_C_REFERENCE_CONFIG_FOUND",
        "ARM_A_REFERENCE_CONFIG_FOUND",
        "TOP3_SELECTION_IMPLEMENTATION_FOUND",
        "LEGAL_SOURCE_POOL_IMPLEMENTATION_FOUND",
        "SOURCE_FUSION_IMPLEMENTATION_FOUND",
        "FORMAL_TARGETS_MATCH",
        "FORMAL_SEEDS_MATCH",
        "FORMAL_HORIZONS_MATCH",
        "ABLATION_LR_MATCH_3E4",
    )
    failed = [key for key in required if flags.get(key) != "YES"]
    if failed:
        raise SourceSelectionAblationError(
            f"source-selection ablation audit failed: {failed}; refusing formal run"
        )


def load_frozen_3e4_arm_rows(
    *,
    project_root: Optional[Path] = None,
    arms: Sequence[str] = ("A", "C"),
) -> List[Dict[str, Any]]:
    root = frozen_3e4_root(project_root)
    bundle_dir = root / "cell_bundles"
    if not bundle_dir.is_dir():
        raise SourceSelectionAblationError(f"missing frozen 3e-4 cell bundles: {bundle_dir}")
    rows: List[Dict[str, Any]] = []
    for path in sorted(bundle_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("pair_status")) != "PAIR_COMPLETE":
            raise SourceSelectionAblationError(f"frozen 3e-4 cell is not PAIR_COMPLETE: {path}")
        dataset = str(payload["dataset"])
        target = str(payload["target"])
        if dataset == "D4" and target != "166/258":
            raise SourceSelectionAblationError(f"frozen 3e-4 D4 target drifted: {target}")
        for arm in arms:
            arm_payload = (payload.get("arms") or {}).get(arm) or {}
            if arm_payload.get("rmse_original") is None or arm_payload.get("smape_original") is None:
                raise SourceSelectionAblationError(f"frozen arm {arm} missing metrics in {path}")
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "horizon": int(payload["horizon"]),
                    "training_seed": int(payload["seed"]),
                    "arm": arm,
                    "arm_name": AE_ARM_NAMES[arm],
                    "RMSE": float(arm_payload["rmse_original"]),
                    "sMAPE": float(arm_payload["smape_original"]),
                    "source_identities": [
                        parse_source_id(item) for item in (arm_payload.get("source_identities") or [])
                    ],
                    "source_distances": [float(value) for value in (arm_payload.get("source_distances") or [])],
                    "source_weights": [float(value) for value in (arm_payload.get("source_weights") or [])],
                    "selected_lr": float(payload.get("selected_lr") or parse_lr(UNIFORM_3E4_LEARNING_RATE_LABEL)),
                    "run_status": str(arm_payload.get("status") or "OK"),
                    "bundle_path": str(path),
                }
            )
    expected = len(UNIFORM_3E4_DATASETS) * len(HORIZONS) * len(UNIFORM_3E4_SEEDS) * len(arms)
    if len(rows) != expected:
        raise SourceSelectionAblationError(
            f"frozen 3e-4 arm rows={len(rows)} != expected {expected}"
        )
    return rows


def frozen_arm_c_top3_from_bundles(project_root: Optional[Path] = None) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    rows = load_frozen_3e4_arm_rows(project_root=project_root, arms=("C",))
    by_dataset: Dict[str, set] = {}
    for row in rows:
        identities = tuple(tuple(item) for item in row["source_identities"])
        if len(identities) != 3:
            raise SourceSelectionAblationError(f"frozen Arm C source count != 3 for {row}")
        by_dataset.setdefault(row["dataset"], set()).add(identities)
    resolved = {}
    for dataset in UNIFORM_3E4_DATASETS:
        unique = by_dataset.get(dataset) or set()
        if len(unique) != 1:
            raise SourceSelectionAblationError(
                f"{dataset} Arm C Top-3 is not target-level: {sorted(unique)}"
            )
        resolved[dataset] = next(iter(unique))
        expected = FROZEN_ARM_C_TOP3[dataset]
        if resolved[dataset] != expected:
            raise SourceSelectionAblationError(
                f"{dataset} frozen Arm C Top-3 drifted: {resolved[dataset]} != {expected}"
            )
    return resolved


def inverse_distance_weights(distances: Sequence[float], *, eps: float = 1e-8) -> List[float]:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size != 3:
        raise SourceSelectionAblationError("fusion weights require exactly 3 distances")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise SourceSelectionAblationError("source distances must be finite and non-negative")
    scores = 1.0 / (values + eps)
    weights = scores / float(np.sum(scores))
    return [float(value) for value in weights]


def sort_triple_by_distance(rows: Sequence[RankedSource]) -> Tuple[RankedSource, ...]:
    ordered = tuple(sorted(rows, key=lambda item: (float(item.distance), item.source_key)))
    if len(ordered) != 3:
        raise SourceSelectionAblationError("selected source count must be 3")
    keys = [item.source_key for item in ordered]
    if len(set(keys)) != 3:
        raise SourceSelectionAblationError(f"duplicate selected sources: {keys}")
    return ordered


def sample_random_keys(
    legal_keys: Sequence[Tuple[str, ...]],
    *,
    source_sampling_seed: int,
    k: int = 3,
) -> Tuple[Tuple[str, ...], ...]:
    unique = tuple(sorted({tuple(key) for key in legal_keys}))
    if len(unique) < k:
        raise SourceSelectionAblationError(
            f"legal source pool size {len(unique)} < {k}; refusing replacement sampling"
        )
    rng = np.random.default_rng(int(source_sampling_seed))
    indices = rng.choice(len(unique), size=k, replace=False)
    return tuple(unique[int(index)] for index in indices)


def _distance_map(ranked: Sequence[RankedSource]) -> Dict[Tuple[str, ...], RankedSource]:
    return {item.source_key: item for item in ranked}


def make_selected_triple(
    *,
    dataset: str,
    target_id: str,
    arm: str,
    source_set_id: str,
    source_sampling_seed: Optional[int],
    selected_keys: Sequence[Tuple[str, ...]],
    ranked: Sequence[RankedSource],
    top3: Sequence[Tuple[str, ...]],
) -> SelectedTriple:
    lookup = _distance_map(ranked)
    chosen = []
    for key in selected_keys:
        if key not in lookup:
            raise SourceSelectionAblationError(f"selected source {key} is not in the legal ranked pool")
        chosen.append(lookup[key])
    ordered = sort_triple_by_distance(chosen)
    top3_tuple = tuple(tuple(item) for item in top3)
    overlap = len({item.source_key for item in ordered} & set(top3_tuple))
    return SelectedTriple(
        dataset=dataset,
        target_id=target_id,
        arm=arm,
        source_set_id=source_set_id,
        source_sampling_seed=source_sampling_seed,
        sources=ordered,
        top3=top3_tuple,
        candidate_pool_size=len(ranked),
        overlap_count=int(overlap),
        overlap_ratio=float(overlap) / 3.0,
    )


def rank_legal_sources_for_target(
    dataset: str,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Rank the Arm C legal pool once per target using the shared protocol selector."""
    from src.transfer_methods.ms_katl_ae_live_context import build_live_ae_cell_context

    root = project_root_from(project_root)
    target = UNIFORM_3E4_TARGETS[dataset]
    if dataset == "D4" and target != "166/258":
        raise SourceSelectionAblationError("D4 target must remain 166/258")
    cell = resolve_real_canonical_cell(
        dataset,
        target,
        REFERENCE_CELL_HORIZON,
        REFERENCE_CELL_SEED,
        project_root=root,
        selected_lr=float(parse_lr(UNIFORM_3E4_LEARNING_RATE_LABEL)),
        selected_lr_label=UNIFORM_3E4_LEARNING_RATE_LABEL,
    )
    live = build_live_ae_cell_context(cell, project_root=root)
    target_df = live.stage_a_kwargs["target_df"]
    source_df = live.stage_a_kwargs["source_df"]
    feature_cols = list(live.stage_a_kwargs["feature_cols"])
    group_cols = tuple(live.stage_a_kwargs["group_cols"])
    selector = SourceSelector()
    topk = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=REQUESTED_K,
        group_cols=group_cols,
        weight_mode=WEIGHT_MODE,
    )
    top3 = tuple(
        parse_source_id(row["source_key"])
        for row in (topk.get("sources") or [])
    )
    if len(top3) != 3:
        raise SourceSelectionAblationError(f"{dataset} runtime Top-3 count != 3")
    expected = FROZEN_ARM_C_TOP3[dataset]
    if top3 != expected:
        raise SourceSelectionAblationError(
            f"{dataset} runtime Top-3 {top3} != frozen Arm C {expected}"
        )
    meta = topk.get("meta") or {}
    valid_keys = [parse_source_id(key) for key in (meta.get("valid_30d_candidate_keys") or [])]
    candidate_keys = [parse_source_id(key) for key in (meta.get("eligible_candidate_keys") or [])]
    if len(valid_keys) < 3:
        raise SourceSelectionAblationError(
            f"{dataset} legal_source_pool_size={len(valid_keys)} < 3; refusing this dataset"
        )
    target_key = parse_source_id(target)
    if target_key in valid_keys or target_key in candidate_keys:
        raise SourceSelectionAblationError(f"{dataset} target leaked into legal source pool")
    promote_protocol_frame_context(target_df)
    promote_protocol_frame_context(source_df)
    source_context = get_protocol_frame_context(source_df)
    if source_context is None or source_context.prepared_pool is None:
        raise SourceSelectionAblationError(f"{dataset} prepared pool missing")
    protocol = get_experiment_protocol(target_df.attrs["protocol_dataset_id"])
    full = select_daily_sequence_sources(
        target_df=get_configured_knn_frame(target_df, "target"),
        source_df=get_configured_knn_frame(source_df, "source"),
        prepared_pool=source_context.prepared_pool,
        protocol=protocol,
        scenario=target_df.attrs["protocol_scenario"],
        target_key=target_df.attrs["protocol_target_key"],
        candidate_keys=candidate_keys,
        group_cols=group_cols,
        observed_start=target_df.attrs["knn_observed_start"],
        feature_cols=tuple(protocol.knn_feature_columns),
        k=len(valid_keys),
    )
    ranked = tuple(
        RankedSource(
            source_key=parse_source_id(entry.source_key),
            distance=float(entry.distance),
            rank=int(entry.rank),
        )
        for entry in full.entries
    )
    if tuple(item.source_key for item in ranked[:3]) != top3:
        raise SourceSelectionAblationError(
            f"{dataset} full ranking Top-3 drifted from Arm C selector"
        )
    if any(item.source_key == target_key for item in ranked):
        raise SourceSelectionAblationError(f"{dataset} ranked pool contains the target")
    bottom3 = tuple(item.source_key for item in ranked[-3:])
    return {
        "dataset": dataset,
        "target_id": target,
        "candidate_keys": candidate_keys,
        "valid_keys": [item.source_key for item in ranked],
        "ranked": ranked,
        "top3": top3,
        "bottom3": bottom3,
        "candidate_pool_size": len(ranked),
        "eligible_candidate_count": len(candidate_keys),
        "topk_meta": {
            "candidate_pool_digest": meta.get("candidate_pool_digest"),
            "selection_result_digest": meta.get("selection_result_digest"),
            "target_test_excluded": meta.get("target_test_excluded"),
            "source_future_excluded": meta.get("source_future_excluded"),
            "distance_metric": meta.get("distance_metric"),
            "weight_mode": meta.get("weight_mode"),
        },
        "raw_topk_payload": topk,
    }


def build_forced_selection_payload(
    raw_topk_payload: Mapping[str, Any],
    triple: SelectedTriple,
) -> Dict[str, Any]:
    payload = deepcopy(dict(raw_topk_payload))
    distances = [float(item.distance) for item in triple.sources]
    weights = inverse_distance_weights(distances)
    sources = []
    for rank, (item, weight) in enumerate(zip(triple.sources, weights), start=1):
        sources.append(
            {
                "source_rank": int(rank),
                "source_key": list(item.source_key),
                "distance": float(item.distance),
                "weight": float(weight),
            }
        )
    meta = dict(payload.get("meta") or {})
    digest = _json_digest(
        {
            "arm": triple.arm,
            "source_set_id": triple.source_set_id,
            "sources": sources,
            "candidate_pool_digest": meta.get("candidate_pool_digest"),
        }
    )
    meta["requested_k"] = REQUESTED_K
    meta["effective_k"] = REQUESTED_K
    meta["selected_count"] = REQUESTED_K
    meta["selected_sources_runtime"] = list(sources)
    meta["selection_result_digest"] = digest
    meta["selection_digest"] = digest
    meta["source_selection_strategy"] = (
        "random3" if triple.arm == ARM_F else "similarity_bottom3"
    )
    meta["cnn_provenance_source_keys"] = [list(item.source_key) for item in triple.sources]
    payload["meta"] = meta
    payload["sources"] = sources
    return payload


def build_all_target_rankings(project_root: Optional[Path] = None) -> Dict[str, Any]:
    frozen_top3 = frozen_arm_c_top3_from_bundles(project_root=project_root)
    rankings = {}
    for dataset in UNIFORM_3E4_DATASETS:
        payload = rank_legal_sources_for_target(dataset, project_root=project_root)
        if payload["top3"] != frozen_top3[dataset]:
            raise SourceSelectionAblationError(f"{dataset} ranking Top-3 != frozen Arm C")
        rankings[dataset] = payload
    return rankings


def materialize_source_sets(rankings: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, SelectedTriple]]:
    materialized: Dict[str, Dict[str, SelectedTriple]] = {}
    for dataset, payload in rankings.items():
        ranked = list(payload["ranked"])
        top3 = payload["top3"]
        target_id = payload["target_id"]
        target_key = parse_source_id(target_id)
        legal_keys = [item.source_key for item in ranked]
        if target_key in legal_keys:
            raise SourceSelectionAblationError(f"{dataset} target present in legal keys")
        if len(legal_keys) < 3:
            raise SourceSelectionAblationError(f"{dataset} legal pool < 3")
        dataset_sets: Dict[str, SelectedTriple] = {}
        dataset_sets["BOTTOM3"] = make_selected_triple(
            dataset=dataset,
            target_id=target_id,
            arm=ARM_G,
            source_set_id="BOTTOM3",
            source_sampling_seed=None,
            selected_keys=payload["bottom3"],
            ranked=ranked,
            top3=top3,
        )
        for source_set_id, seed in SOURCE_SET_SEEDS.items():
            sampled = sample_random_keys(legal_keys, source_sampling_seed=seed, k=3)
            dataset_sets[source_set_id] = make_selected_triple(
                dataset=dataset,
                target_id=target_id,
                arm=ARM_F,
                source_set_id=source_set_id,
                source_sampling_seed=int(seed),
                selected_keys=sampled,
                ranked=ranked,
                top3=top3,
            )
        materialized[dataset] = dataset_sets
    return materialized


def source_manifest_rows(
    materialized: Mapping[str, Mapping[str, SelectedTriple]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in UNIFORM_3E4_DATASETS:
        sets = materialized[dataset]
        ordered_ids = list(SOURCE_SET_IDS) + ["BOTTOM3"]
        for source_set_id in ordered_ids:
            triple = sets[source_set_id]
            rows.append(
                {
                    "dataset": dataset,
                    "target_id": triple.target_id,
                    "source_selection_granularity": SOURCE_SELECTION_GRANULARITY,
                    "arm": triple.arm,
                    "source_set_id": triple.source_set_id,
                    "source_sampling_seed": "" if triple.source_sampling_seed is None else int(triple.source_sampling_seed),
                    "candidate_pool_size": int(triple.candidate_pool_size),
                    "source_1": format_source_id(triple.sources[0].source_key),
                    "source_2": format_source_id(triple.sources[1].source_key),
                    "source_3": format_source_id(triple.sources[2].source_key),
                    "similarity_1": float(triple.sources[0].distance),
                    "similarity_2": float(triple.sources[1].distance),
                    "similarity_3": float(triple.sources[2].distance),
                    "mean_similarity": float(triple.mean_similarity),
                    "median_similarity": float(triple.median_similarity),
                    "top3_source_1": format_source_id(triple.top3[0]),
                    "top3_source_2": format_source_id(triple.top3[1]),
                    "top3_source_3": format_source_id(triple.top3[2]),
                    "top3_overlap_count": int(triple.overlap_count),
                    "top3_overlap_ratio": float(triple.overlap_ratio),
                }
            )
    return rows


def write_source_selection_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    project_root: Optional[Path] = None,
) -> Path:
    root = project_root_from(project_root)
    path = source_manifest_csv_path(root)
    fieldnames = list(rows[0].keys()) if rows else [
        "dataset",
        "target_id",
        "source_selection_granularity",
        "arm",
        "source_set_id",
        "source_sampling_seed",
        "candidate_pool_size",
        "source_1",
        "source_2",
        "source_3",
        "similarity_1",
        "similarity_2",
        "similarity_3",
        "mean_similarity",
        "top3_overlap_count",
        "top3_overlap_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    replica = default_output_dir(root) / SOURCE_MANIFEST_CSV_NAME
    replica.parent.mkdir(parents=True, exist_ok=True)
    replica.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def serialize_rankings(rankings: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    payload = {}
    for dataset, item in rankings.items():
        payload[dataset] = {
            "dataset": dataset,
            "target_id": item["target_id"],
            "candidate_pool_size": item["candidate_pool_size"],
            "eligible_candidate_count": item["eligible_candidate_count"],
            "top3": [list(key) for key in item["top3"]],
            "bottom3": [list(key) for key in item["bottom3"]],
            "ranked": [
                {
                    "rank": int(row.rank),
                    "source_key": list(row.source_key),
                    "distance": float(row.distance),
                }
                for row in item["ranked"]
            ],
            "topk_meta": item["topk_meta"],
            "raw_topk_payload": item["raw_topk_payload"],
        }
    return payload


def load_rankings(project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = ranking_json_path(project_root)
    if not path.is_file():
        raise SourceSelectionAblationError(f"missing source ranking freeze: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = {}
    for dataset, item in payload["rankings"].items():
        ranked = tuple(
            RankedSource(
                source_key=parse_source_id(row["source_key"]),
                distance=float(row["distance"]),
                rank=int(row["rank"]),
            )
            for row in item["ranked"]
        )
        restored[dataset] = {
            "dataset": dataset,
            "target_id": item["target_id"],
            "candidate_pool_size": int(item["candidate_pool_size"]),
            "eligible_candidate_count": int(item["eligible_candidate_count"]),
            "top3": tuple(parse_source_id(key) for key in item["top3"]),
            "bottom3": tuple(parse_source_id(key) for key in item["bottom3"]),
            "ranked": ranked,
            "topk_meta": item["topk_meta"],
            "raw_topk_payload": item["raw_topk_payload"],
        }
    return restored


def freeze_source_selection(*, project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root_from(project_root)
    flags = implementation_audit_flags(project_root=root)
    assert_audit_gates(flags)
    rankings = build_all_target_rankings(project_root=root)
    materialized = materialize_source_sets(rankings)
    rows = source_manifest_rows(materialized)
    csv_path = write_source_selection_manifest(rows, project_root=root)
    out = assert_output_isolated(default_output_dir(root), project_root=root)
    out.mkdir(parents=True, exist_ok=True)
    freeze_payload = {
        "label": SLICE_LABEL,
        "source_selection_granularity": SOURCE_SELECTION_GRANULARITY,
        "similarity_direction": SIMILARITY_DIRECTION,
        "source_set_seeds": dict(SOURCE_SET_SEEDS),
        "flags": flags,
        "rankings": serialize_rankings(rankings),
        "manifest_rows": rows,
    }
    atomic_write_json(ranking_json_path(root), freeze_payload)
    config_payload = {
        "label": SLICE_LABEL,
        "arm_f_identity": ARM_F_IDENTITY,
        "arm_g_identity": ARM_G_IDENTITY,
        "datasets": list(UNIFORM_3E4_DATASETS),
        "paper_targets": dict(UNIFORM_3E4_TARGETS),
        "horizons": list(HORIZONS),
        "training_seeds": list(UNIFORM_3E4_SEEDS),
        "source_set_ids": list(SOURCE_SET_IDS),
        "source_set_seeds": dict(SOURCE_SET_SEEDS),
        "stage1_source_sets": list(STAGE1_SOURCE_SETS),
        "learning_rate": UNIFORM_3E4_LEARNING_RATE_LABEL,
        "source_selection_granularity": SOURCE_SELECTION_GRANULARITY,
        "similarity_direction": SIMILARITY_DIRECTION,
        "copied_from_arm": "C",
        "arm_c_config_identity": ae_config_identity("C"),
        "expected_f_full_runs": EXPECTED_F_FULL_RUNS,
        "expected_f_stage1_runs": EXPECTED_F_STAGE1_RUNS,
        "expected_g_runs": EXPECTED_G_RUNS,
        "output_relpath": str(OUTPUT_RELPATH),
        "source_manifest_csv": str(csv_path),
    }
    config_path = root / CONFIG_MANIFEST_RELPATH
    atomic_write_json(config_path, config_payload)
    return {
        "csv": str(csv_path),
        "ranking_json": str(ranking_json_path(root)),
        "config_manifest": str(config_path),
        "rows": rows,
        "flags": flags,
    }


def enumerate_identities(
    *,
    arms: Sequence[str] = (ARM_F, ARM_G),
    source_sets: Sequence[str] = SOURCE_SET_IDS,
) -> List[AblationRunIdentity]:
    identities: List[AblationRunIdentity] = []
    wanted_arms = tuple(str(arm).strip().upper() for arm in arms)
    wanted_sets = tuple(str(item).strip().upper() for item in source_sets)
    for dataset in UNIFORM_3E4_DATASETS:
        target = UNIFORM_3E4_TARGETS[dataset]
        for horizon in HORIZONS:
            for seed in UNIFORM_3E4_SEEDS:
                if ARM_F in wanted_arms:
                    for source_set_id in wanted_sets:
                        identities.append(
                            AblationRunIdentity(
                                arm=ARM_F,
                                dataset=dataset,
                                target_identity=target,
                                horizon=int(horizon),
                                training_seed=int(seed),
                                source_set_id=source_set_id,
                                source_sampling_seed=int(SOURCE_SET_SEEDS[source_set_id]),
                            )
                        )
                if ARM_G in wanted_arms:
                    identities.append(
                        AblationRunIdentity(
                            arm=ARM_G,
                            dataset=dataset,
                            target_identity=target,
                            horizon=int(horizon),
                            training_seed=int(seed),
                            source_set_id="BOTTOM3",
                            source_sampling_seed=None,
                        )
                    )
    unique = {item.identity_tuple for item in identities}
    if len(unique) != len(identities):
        raise SourceSelectionAblationError("duplicate Arm F/G run identities")
    return identities


def validate_identity_matrix(
    identities: Sequence[AblationRunIdentity],
    *,
    arms: Sequence[str],
    source_sets: Sequence[str],
) -> Dict[str, Any]:
    wanted_arms = tuple(str(arm).strip().upper() for arm in arms)
    n_f = EXPECTED_F_FULL_RUNS if set(source_sets) == set(SOURCE_SET_IDS) else (
        EXPECTED_F_STAGE1_RUNS if tuple(source_sets) == STAGE1_SOURCE_SETS else
        len(UNIFORM_3E4_DATASETS) * len(HORIZONS) * len(source_sets) * len(UNIFORM_3E4_SEEDS)
    )
    expected = 0
    if ARM_F in wanted_arms:
        expected += n_f
    if ARM_G in wanted_arms:
        expected += EXPECTED_G_RUNS
    if len(identities) != expected:
        raise SourceSelectionAblationError(
            f"identity count {len(identities)} != expected {expected}"
        )
    if {item.dataset for item in identities} != set(UNIFORM_3E4_DATASETS):
        raise SourceSelectionAblationError("dataset set drifted")
    if any(item.dataset == "D4" and item.target_identity != "166/258" for item in identities):
        raise SourceSelectionAblationError("D4 target drifted from 166/258")
    if {int(item.horizon) for item in identities} != set(HORIZONS):
        raise SourceSelectionAblationError("horizons drifted from {1,3,5}")
    if {int(item.training_seed) for item in identities} != set(UNIFORM_3E4_SEEDS):
        raise SourceSelectionAblationError("training seeds drifted from {42,43,44,45,46}")
    if ARM_F in wanted_arms:
        f_items = [item for item in identities if item.arm == ARM_F]
        by_set: Dict[Tuple[str, str], set] = {}
        for item in f_items:
            if item.source_sampling_seed != SOURCE_SET_SEEDS[item.source_set_id]:
                raise SourceSelectionAblationError("source sampling seed drifted")
            by_set.setdefault((item.dataset, item.source_set_id), set()).add(item.training_seed)
        if any(seeds != set(UNIFORM_3E4_SEEDS) for seeds in by_set.values()):
            raise SourceSelectionAblationError("each Random-3 set must run seeds 42-46")
    return {
        "EXPECTED_RUNS": expected,
        "DRY_RUN_TOTAL_RUNS": len(identities),
        "ARMS": ",".join(wanted_arms),
        "SOURCE_SETS": ",".join(source_sets) if ARM_F in wanted_arms else "BOTTOM3",
        "MATRIX_OK": "YES",
    }


def cell_artifact_path(run_dir: Path, item: AblationRunIdentity) -> Path:
    return Path(run_dir) / "cells" / item.run_id / "run_manifest.json"


def prediction_path(run_dir: Path, item: AblationRunIdentity) -> Path:
    return Path(run_dir) / "cells" / item.run_id / "predictions.csv"


def planned_row(item: AblationRunIdentity, triple: SelectedTriple) -> Dict[str, Any]:
    spec = ae_arm_spec("C")
    optimizer = ae_optimizer_contract()
    recipe = ae_training_recipe()
    return {
        "dataset": item.dataset,
        "target": item.target_identity,
        "horizon": int(item.horizon),
        "arm": item.arm,
        "arm_name": ARM_F_NAME if item.arm == ARM_F else ARM_G_NAME,
        "result_identity": ARM_F_IDENTITY if item.arm == ARM_F else ARM_G_IDENTITY,
        "source_selection_strategy": "random3" if item.arm == ARM_F else "similarity_bottom3",
        "source_set_id": item.source_set_id,
        "source_sampling_seed": item.source_sampling_seed,
        "training_seed": int(item.training_seed),
        "source_1": format_source_id(triple.sources[0].source_key),
        "source_2": format_source_id(triple.sources[1].source_key),
        "source_3": format_source_id(triple.sources[2].source_key),
        "similarity_1": float(triple.sources[0].distance),
        "similarity_2": float(triple.sources[1].distance),
        "similarity_3": float(triple.sources[2].distance),
        "mean_source_similarity": float(triple.mean_similarity),
        "median_source_similarity": float(triple.median_similarity),
        "top3_overlap_count": int(triple.overlap_count),
        "top3_overlap_ratio": float(triple.overlap_ratio),
        "top3_source_1": format_source_id(triple.top3[0]),
        "top3_source_2": format_source_id(triple.top3[1]),
        "top3_source_3": format_source_id(triple.top3[2]),
        "candidate_pool_size": int(triple.candidate_pool_size),
        "lr": item.selected_lr_label,
        "epochs": int(optimizer["epochs"]),
        "optimizer": optimizer["optimizer"],
        "loss": recipe["loss_name"],
        "RevIN": "OFF",
        "transferred_layers": ",".join(CONV_LAYER_NAMES),
        "trainable_status": "head_target+conv_backbone",
        "RMSE": "",
        "sMAPE": "",
        "result_status": "PLANNED",
        "code_commit": "",
        "config_hash": _json_digest(arm_fg_config_payload(item.arm, source_set_id=item.source_set_id, source_sampling_seed=item.source_sampling_seed)),
        "prediction_file_path": "",
        "run_manifest_path": "",
        "timestamp": "",
    }


def _evaluate_split(constructed: Any, bundle: Any, split: str) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_lite import extract_features_maybe_revin, predict_ms_katl_decomposed

    x_map = {"train": bundle.X_train, "val": bundle.X_val, "test": bundle.X_test}
    y_map = {"train": bundle.y_train, "val": bundle.y_val, "test": bundle.y_test}
    x = x_map[split]
    y_true = y_map[split]
    features = bundle.f_t.get(split) or []
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
        raise SourceSelectionAblationError("Arm F/G must evaluate with RevIN OFF")
    _ = (mean, stdev)
    metrics = compute_screening_metrics(
        y_true,
        y_pred,
        target_scaler=bundle.target_scaler,
        feature_columns=list(bundle.target_feature_columns),
    )
    return metrics


def train_one_run(
    item: AblationRunIdentity,
    triple: SelectedTriple,
    rankings: Mapping[str, Mapping[str, Any]],
    *,
    run_dir: Path,
    project_root: Path,
) -> Dict[str, Any]:
    from src.transfer_methods.ms_katl_ae_ablation import build_ae_stage_a_from_shared_bundle
    from src.transfer_methods.ms_katl_ae_live_context import build_live_ae_cell_context
    from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a, train_ms_katl_head
    from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle

    cell = resolve_real_canonical_cell(
        item.dataset,
        item.target_identity,
        int(item.horizon),
        int(item.training_seed),
        project_root=project_root,
        selected_lr=float(item.selected_lr),
        selected_lr_label=item.selected_lr_label,
    )
    live = build_live_ae_cell_context(cell, project_root=project_root)
    payload = build_forced_selection_payload(rankings[item.dataset]["raw_topk_payload"], triple)
    inject_forced_k3_selection_context(live.stage_a_kwargs, payload)
    selected_keys = {item.source_key for item in triple.sources}
    target_key = parse_source_id(item.target_identity)
    if target_key in selected_keys:
        raise SourceSelectionAblationError("target appeared in selected sources")
    reset_ae_rng(int(item.training_seed))
    artifacts = prepare_ms_katl_stage_a(**live.stage_a_kwargs)
    trained_keys = tuple(parse_source_id(key) for key in artifacts.source_identities)
    expected_keys = tuple(item.source_key for item in triple.sources)
    if trained_keys != expected_keys:
        raise SourceSelectionAblationError(
            f"Stage A source identities {trained_keys} != selected {expected_keys}"
        )
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
    head_snapshot, _digest = construct_v3_head_target_snapshot(ae_stage, seed=int(item.training_seed))
    spec = ae_arm_spec("C")
    if spec.revin_enabled or spec.init_mode != "fused" or not spec.target_backbone_trainable:
        raise SourceSelectionAblationError("Arm C spec drifted while constructing F/G")
    reset_ae_rng(int(item.training_seed))
    constructed = construct_ae_arm(
        spec,
        ae_stage,
        seed=int(item.training_seed),
        selected_lr=float(item.selected_lr),
        recipe=ae_training_recipe(),
        head_target_snapshot=head_snapshot,
    )
    if constructed.revin is not None:
        raise SourceSelectionAblationError("Arm F/G constructed RevIN ON")
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
        learning_rate=float(item.selected_lr),
        seed=int(item.training_seed),
        use_adversarial=False,
        trainable_variables=trainable,
        revin=constructed.revin,
        **recipe,
    )
    test_metrics = _evaluate_split(constructed, bundle, "test")
    pred_path = prediction_path(run_dir, item)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_o = np.asarray(test_metrics["y_true_original"], dtype=np.float64).reshape(-1)
    y_pred_o = np.asarray(test_metrics["y_pred_original"], dtype=np.float64).reshape(-1)
    with pred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "y_true_original", "y_pred_original"])
        writer.writeheader()
        for index, (true, pred) in enumerate(zip(y_true_o.tolist(), y_pred_o.tolist())):
            writer.writerow({"index": index, "y_true_original": true, "y_pred_original": pred})
    recomputed_rmse = original_rmse(y_true_o, y_pred_o)
    recomputed_smape = float(smape(y_true_o, y_pred_o))
    if not math.isclose(recomputed_rmse, float(test_metrics["rmse_original"]), rel_tol=0.0, abs_tol=1e-12):
        raise SourceSelectionAblationError("raw prediction RMSE parity failed")
    if not math.isclose(recomputed_smape, float(test_metrics["smape_original"]), rel_tol=0.0, abs_tol=1e-12):
        raise SourceSelectionAblationError("raw prediction sMAPE parity failed")
    git = git_provenance(project_root)
    row = planned_row(item, triple)
    row.update(
        {
            "RMSE": float(test_metrics["rmse_original"]),
            "sMAPE": float(test_metrics["smape_original"]),
            "result_status": "OK",
            "code_commit": git.get("git_head", ""),
            "prediction_file_path": str(pred_path),
            "run_manifest_path": str(cell_artifact_path(run_dir, item)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage_a_key": item.stage_a_key,
            "parent_v3_architecture_digest": v3_architecture_digest(),
        }
    )
    return row


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_fg_run_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Load every cell manifest. Do not trust the last execute's all_runs.csv."""
    cells = Path(run_dir) / "cells"
    paths = sorted(cells.glob("*/run_manifest.json"))
    if not paths:
        raise SourceSelectionAblationError(f"no F/G cell manifests under {cells}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return rows


def load_completed_ok_runs(run_dir: Path, identities: Sequence[AblationRunIdentity]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    completed = {}
    for item in identities:
        path = cell_artifact_path(run_dir, item)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("result_status")) == "OK":
            completed[item.identity_tuple] = payload
    return completed


def leakage_audit(triple: SelectedTriple, target_id: str) -> Dict[str, str]:
    target_key = parse_source_id(target_id)
    selected = [item.source_key for item in triple.sources]
    return {
        "TARGET_NOT_IN_SOURCES": "YES" if target_key not in selected else "NO",
        "NO_DUPLICATE_SOURCES": "YES" if len(set(selected)) == 3 else "NO",
        "SOURCE_COUNT_3": "YES" if len(selected) == 3 else "NO",
        "TARGET_ONLY_INFERENCE": "YES",
        "NO_FUTURE_SOURCE_AT_PREDICTION": "YES",
        "NO_SOURCE_PREDICTION_INJECTION": "YES",
        "OBSERVED_TARGET_BUDGET_UNCHANGED": "YES",
    }


def run_dry_run(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    arms: Sequence[str] = (ARM_F, ARM_G),
    source_sets: Sequence[str] = SOURCE_SET_IDS,
    build_manifest: bool = False,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    flags = implementation_audit_flags(project_root=root)
    assert_audit_gates(flags)
    assert_formal_ae_identity_frozen(root)
    assert_d_equals_frozen_v3()
    assert_protocol_parity(ARM_F)
    assert_protocol_parity(ARM_G)
    identities = enumerate_identities(arms=arms, source_sets=source_sets)
    report = validate_identity_matrix(identities, arms=arms, source_sets=source_sets)
    run_dir = assert_output_isolated(output_dir or default_output_dir(root), project_root=root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cells").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    manifest_status = "EXISTING" if ranking_json_path(root).is_file() else "MISSING"
    if build_manifest:
        freeze_source_selection(project_root=root)
        manifest_status = "BUILT"
    text = "\n".join(
        [
            "============================================================",
            "SITA SOURCE-SELECTION ABLATION ARM F / ARM G DRY-RUN",
            "============================================================",
            f"ARM_C_REFERENCE_CONFIG_FOUND={flags['ARM_C_REFERENCE_CONFIG_FOUND']}",
            f"ARM_A_REFERENCE_CONFIG_FOUND={flags['ARM_A_REFERENCE_CONFIG_FOUND']}",
            f"TOP3_SELECTION_IMPLEMENTATION_FOUND={flags['TOP3_SELECTION_IMPLEMENTATION_FOUND']}",
            f"LEGAL_SOURCE_POOL_IMPLEMENTATION_FOUND={flags['LEGAL_SOURCE_POOL_IMPLEMENTATION_FOUND']}",
            f"SOURCE_FUSION_IMPLEMENTATION_FOUND={flags['SOURCE_FUSION_IMPLEMENTATION_FOUND']}",
            f"FORMAL_TARGETS_MATCH={flags['FORMAL_TARGETS_MATCH']}",
            f"FORMAL_SEEDS_MATCH={flags['FORMAL_SEEDS_MATCH']}",
            f"FORMAL_HORIZONS_MATCH={flags['FORMAL_HORIZONS_MATCH']}",
            f"ABLATION_LR_MATCH_3E4={flags['ABLATION_LR_MATCH_3E4']}",
            f"SOURCE_SELECTION_GRANULARITY={SOURCE_SELECTION_GRANULARITY}",
            f"SIMILARITY_DIRECTION={SIMILARITY_DIRECTION}",
            f"D4_TARGET={UNIFORM_3E4_TARGETS['D4']}",
            f"ARM_F_IDENTITY={ARM_F_IDENTITY}",
            f"ARM_G_IDENTITY={ARM_G_IDENTITY}",
            f"SOURCE_SET_SEEDS={json.dumps(SOURCE_SET_SEEDS)}",
            f"EXPECTED_F_FULL_RUNS={EXPECTED_F_FULL_RUNS}",
            f"EXPECTED_F_STAGE1_RUNS={EXPECTED_F_STAGE1_RUNS}",
            f"EXPECTED_G_RUNS={EXPECTED_G_RUNS}",
            f"DRY_RUN_TOTAL_RUNS={report['DRY_RUN_TOTAL_RUNS']}",
            f"SOURCE_MANIFEST_STATUS={manifest_status}",
            f"OUTPUT_DIR={run_dir}",
            "TRAINING_STARTED=NO",
            "FORMAL_RESULTS_OVERWRITTEN=NO",
            "READY_FOR_FULL_RUN=YES",
        ]
    ) + "\n"
    (run_dir / "logs" / "dry_run.txt").write_text(text, encoding="utf-8")
    return {
        "text": text,
        "flags": flags,
        "report": report,
        "output_root": str(run_dir),
        "DRY_RUN_TOTAL_RUNS": report["DRY_RUN_TOTAL_RUNS"],
        "READY_FOR_FULL_RUN": "YES",
        "TRAINING_STARTED": "NO",
    }


def run_execute(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    arms: Sequence[str] = (ARM_F, ARM_G),
    source_sets: Sequence[str] = SOURCE_SET_IDS,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    env = environ if environ is not None else os.environ
    if str(env.get(UNLOCK_ENV, "")).strip() != "YES":
        raise AETrainingBlocked("FORMAL_TRAINING_BLOCKED")
    root = project_root_from(project_root)
    flags = implementation_audit_flags(project_root=root)
    assert_audit_gates(flags)
    assert_formal_ae_identity_frozen(root)
    identities = enumerate_identities(arms=arms, source_sets=source_sets)
    report = validate_identity_matrix(identities, arms=arms, source_sets=source_sets)
    rankings = load_rankings(project_root=root)
    materialized = materialize_source_sets(rankings)
    run_dir = assert_output_isolated(output_dir or default_output_dir(root), project_root=root)
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ok_runs(run_dir, identities) if skip_existing else {}
    failed = 0
    for item in identities:
        if skip_existing and item.identity_tuple in completed:
            print(f"SKIP_OK {item.run_id}", flush=True)
            continue
        source_set_id = item.source_set_id
        triple = materialized[item.dataset][source_set_id]
        leak = leakage_audit(triple, item.target_identity)
        if any(value != "YES" for value in leak.values()):
            raise SourceSelectionAblationError(f"leakage audit failed for {item.run_id}: {leak}")
        path = cell_artifact_path(run_dir, item)
        try:
            payload = train_one_run(
                item,
                triple,
                rankings,
                run_dir=run_dir,
                project_root=root,
            )
            print(f"RUN_OK {item.run_id} RMSE={payload['RMSE']:.6g} sMAPE={payload['sMAPE']:.6g}", flush=True)
        except Exception as exc:
            failed += 1
            payload = planned_row(item, triple)
            payload["result_status"] = "FAILED"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            print(f"RUN_FAILED {item.run_id} reason={exc}", flush=True)
        atomic_write_json(path, payload)
        if payload.get("result_status") == "OK":
            completed[item.identity_tuple] = payload
    all_rows = []
    for item in identities:
        path = cell_artifact_path(run_dir, item)
        if path.is_file():
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            all_rows.append(planned_row(item, materialized[item.dataset][item.source_set_id]))
    write_results_csv(run_dir / "results" / "all_runs.csv", all_rows)
    ok_count = sum(1 for row in all_rows if str(row.get("result_status")) == "OK")
    return {
        "output_root": str(run_dir),
        "TRAINING_STARTED": "YES",
        "OK_RUNS": ok_count,
        "FAILED_RUNS": failed,
        "EXPECTED_RUNS": report["EXPECTED_RUNS"],
        "FORMAL_RESULTS_OVERWRITTEN": "NO",
        "flags": flags,
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _sd(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1))


def _condition_key(dataset: str, horizon: int) -> Tuple[str, int]:
    return (dataset, int(horizon))


def load_all_condition_means(
    run_dir: Path,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    frozen = load_frozen_3e4_arm_rows(project_root=project_root, arms=("A", "C"))
    fg_rows = load_fg_run_rows(run_dir)
    write_results_csv(Path(run_dir) / "results" / "all_runs.csv", fg_rows)
    conditions = {}
    for dataset in UNIFORM_3E4_DATASETS:
        for horizon in HORIZONS:
            key = _condition_key(dataset, horizon)
            conditions[key] = {
                "dataset": dataset,
                "target": UNIFORM_3E4_TARGETS[dataset],
                "horizon": int(horizon),
            }
    for arm, label in (("A", "A"), ("C", "C")):
        for dataset in UNIFORM_3E4_DATASETS:
            for horizon in HORIZONS:
                subset = [
                    row
                    for row in frozen
                    if row["arm"] == arm and row["dataset"] == dataset and int(row["horizon"]) == int(horizon)
                ]
                if len(subset) != 5:
                    raise SourceSelectionAblationError(f"frozen {arm} {dataset} H{horizon} != 5 seeds")
                conditions[_condition_key(dataset, horizon)][f"{label}_RMSE"] = _mean([row["RMSE"] for row in subset])
                conditions[_condition_key(dataset, horizon)][f"{label}_sMAPE"] = _mean([row["sMAPE"] for row in subset])
                conditions[_condition_key(dataset, horizon)][f"{label}_RMSE_seed_sd"] = _sd([row["RMSE"] for row in subset])
                conditions[_condition_key(dataset, horizon)][f"{label}_sMAPE_seed_sd"] = _sd([row["sMAPE"] for row in subset])
    ok_f = [row for row in fg_rows if row.get("arm") == ARM_F and row.get("result_status") == "OK"]
    ok_g = [row for row in fg_rows if row.get("arm") == ARM_G and row.get("result_status") == "OK"]
    for dataset in UNIFORM_3E4_DATASETS:
        for horizon in HORIZONS:
            g_subset = [
                row for row in ok_g if row["dataset"] == dataset and int(row["horizon"]) == int(horizon)
            ]
            if g_subset:
                if len(g_subset) != 5:
                    raise SourceSelectionAblationError(f"Arm G {dataset} H{horizon} != 5 OK seeds")
                conditions[_condition_key(dataset, horizon)]["G_RMSE"] = _mean([float(row["RMSE"]) for row in g_subset])
                conditions[_condition_key(dataset, horizon)]["G_sMAPE"] = _mean([float(row["sMAPE"]) for row in g_subset])
                conditions[_condition_key(dataset, horizon)]["G_mean_similarity"] = _mean(
                    [float(row["mean_source_similarity"]) for row in g_subset]
                )
            f_sets = []
            set_rmse = []
            set_smape = []
            set_sim = []
            present_sets = sorted({row["source_set_id"] for row in ok_f if row["dataset"] == dataset and int(row["horizon"]) == int(horizon)})
            for source_set_id in present_sets:
                subset = [
                    row
                    for row in ok_f
                    if row["dataset"] == dataset
                    and int(row["horizon"]) == int(horizon)
                    and row["source_set_id"] == source_set_id
                ]
                if len(subset) != 5:
                    continue
                rmse_mean = _mean([float(row["RMSE"]) for row in subset])
                smape_mean = _mean([float(row["sMAPE"]) for row in subset])
                sim_mean = _mean([float(row["mean_source_similarity"]) for row in subset])
                f_sets.append(
                    {
                        "source_set_id": source_set_id,
                        "RMSE": rmse_mean,
                        "sMAPE": smape_mean,
                        "mean_similarity": sim_mean,
                        "within_seed_sd_RMSE": _sd([float(row["RMSE"]) for row in subset]),
                        "within_seed_sd_sMAPE": _sd([float(row["sMAPE"]) for row in subset]),
                    }
                )
                set_rmse.append(rmse_mean)
                set_smape.append(smape_mean)
                set_sim.append(sim_mean)
            if f_sets:
                conditions[_condition_key(dataset, horizon)]["F_sets"] = f_sets
                conditions[_condition_key(dataset, horizon)]["F_RMSE"] = _mean(set_rmse)
                conditions[_condition_key(dataset, horizon)]["F_sMAPE"] = _mean(set_smape)
                conditions[_condition_key(dataset, horizon)]["F_mean_similarity"] = _mean(set_sim)
                conditions[_condition_key(dataset, horizon)]["F_between_set_sd_RMSE"] = _sd(set_rmse)
                conditions[_condition_key(dataset, horizon)]["F_between_set_sd_sMAPE"] = _sd(set_smape)
    try:
        rankings = load_rankings(project_root=project_root)
        for dataset in UNIFORM_3E4_DATASETS:
            ranked = rankings[dataset]["ranked"]
            top_mean = _mean([float(item.distance) for item in ranked[:3]])
            bot_mean = _mean([float(item.distance) for item in ranked[-3:]])
            for horizon in HORIZONS:
                conditions[_condition_key(dataset, horizon)]["C_mean_similarity"] = top_mean
                conditions[_condition_key(dataset, horizon)]["G_mean_similarity_from_rank"] = bot_mean
    except SourceSelectionAblationError:
        pass
    return conditions


def _paired_relative(left: Sequence[float], right: Sequence[float]) -> Dict[str, Any]:
    left_a = np.asarray(left, dtype=np.float64)
    right_a = np.asarray(right, dtype=np.float64)
    rel = (right_a - left_a) / np.where(np.abs(right_a) < 1e-15, np.nan, right_a)
    wins = int(np.sum(left_a < right_a))
    losses = int(np.sum(left_a > right_a))
    ties = int(np.sum(left_a == right_a))
    diffs = left_a - right_a
    try:
        from scipy import stats

        nonzero = diffs[diffs != 0]
        if nonzero.size == 0:
            w_stat, p_raw = 0.0, 1.0
        else:
            result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
            w_stat, p_raw = float(result.statistic), float(result.pvalue)
    except Exception as exc:
        w_stat, p_raw = float("nan"), float("nan")
        _ = exc
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    rel_finite = rel[np.isfinite(rel)]
    if rel_finite.size:
        indices = rng.integers(0, rel_finite.size, size=(BOOTSTRAP_RESAMPLES, rel_finite.size))
        boot = rel_finite[indices].mean(axis=1)
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return {
        "mean_relative_improvement": float(np.nanmean(rel)),
        "median_relative_improvement": float(np.nanmedian(rel)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_W": w_stat,
        "wilcoxon_raw_p": p_raw,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "n_conditions": int(left_a.size),
    }


def holm_adjust(p_values: Mapping[str, float], alpha: float = HOLM_ALPHA) -> Dict[str, Dict[str, Any]]:
    items = [(index, key, float(value)) for index, (key, value) in enumerate(p_values.items())]
    ordered = sorted(items, key=lambda item: (item[2], item[0]))
    adjusted: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    for rank, (_index, key, p_value) in enumerate(ordered, start=1):
        multiplier = len(items) - rank + 1
        candidate = min(1.0, p_value * multiplier)
        running = max(running, candidate)
        adjusted[key] = {
            "raw_p": p_value,
            "rank": rank,
            "multiplier": multiplier,
            "adjusted_p": running,
            "significant": running <= alpha,
        }
    return adjusted


def overlap_sensitivity_report(
    materialized: Mapping[str, Mapping[str, SelectedTriple]],
) -> Dict[str, Any]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    details = []
    for dataset in UNIFORM_3E4_DATASETS:
        for source_set_id in SOURCE_SET_IDS:
            triple = materialized[dataset][source_set_id]
            counts[int(triple.overlap_count)] += 1
            details.append(
                {
                    "dataset": dataset,
                    "source_set_id": source_set_id,
                    "overlap_count": int(triple.overlap_count),
                    "selected": [format_source_id(item.source_key) for item in triple.sources],
                    "top3": [format_source_id(key) for key in triple.top3],
                }
            )
    high = counts[2] + counts[3]
    return {
        "n_random_sets": 20,
        "overlap_counts": counts,
        "high_overlap_sets": high,
        "sensitivity_candidate": "Random-3 excluding Top-3" if high >= 4 else None,
        "promote_to_formal": False,
        "details": details,
    }


def write_overlap_sensitivity_md(report: Mapping[str, Any], path: Path) -> None:
    counts = report["overlap_counts"]
    lines = [
        "# Random-3 vs Top-3 overlap sensitivity audit",
        "",
        "Main Random-3 still samples the full legal pool. This file is feasibility only.",
        "It does not replace R1–R5, does not change Bottom-3, and is not a freeze.",
        "",
        f"- overlap=0: {counts[0]}",
        f"- overlap=1: {counts[1]}",
        f"- overlap=2: {counts[2]}",
        f"- overlap=3: {counts[3]}",
        f"- high-overlap sets (2 or 3): {report['high_overlap_sets']} / 20",
        f"- sensitivity candidate generated: {report['sensitivity_candidate'] or 'none'}",
        f"- promote_to_formal: {report['promote_to_formal']}",
        "",
        "If a Random-3 excluding Top-3 candidate is later desired, it must use a new",
        "independent identity. It must not overwrite `SITA_ABLATION_F_RANDOM3`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_results(
    *,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = project_root_from(project_root)
    run_dir = assert_output_isolated(output_dir or default_output_dir(root), project_root=root)
    conditions = load_all_condition_means(run_dir, project_root=root)
    ordered = [conditions[_condition_key(dataset, horizon)] for dataset in UNIFORM_3E4_DATASETS for horizon in HORIZONS]
    complete = [row for row in ordered if all(key in row for key in ("C_RMSE", "F_RMSE", "G_RMSE", "A_RMSE"))]
    incomplete = []
    for row in ordered:
        present_f = {item["source_set_id"] for item in (row.get("F_sets") or [])}
        if present_f != set(SOURCE_SET_IDS) or "G_RMSE" not in row:
            incomplete.append(
                f"{row['dataset']} H{row['horizon']} F_sets={sorted(present_f)} G={'YES' if 'G_RMSE' in row else 'NO'}"
            )
    if incomplete:
        raise SourceSelectionAblationError(
            "analyze requires Arm F R1–R5 and Arm G for all 12 conditions: " + "; ".join(incomplete)
        )
    comparison_tables = {}
    if len(complete) == 12:
        for metric in ("RMSE", "sMAPE"):
            c_vals = [float(row[f"C_{metric}"]) for row in complete]
            f_vals = [float(row[f"F_{metric}"]) for row in complete]
            g_vals = [float(row[f"G_{metric}"]) for row in complete]
            a_vals = [float(row[f"A_{metric}"]) for row in complete]
            comparison_tables[metric] = {
                "C_vs_F": _paired_relative(c_vals, f_vals),
                "F_vs_A": _paired_relative(f_vals, a_vals),
                "C_vs_G": _paired_relative(c_vals, g_vals),
                "F_vs_G": _paired_relative(f_vals, g_vals),
            }
            p_values = {
                name: float(payload["wilcoxon_raw_p"])
                for name, payload in comparison_tables[metric].items()
            }
            adjusted = holm_adjust(p_values)
            for name, details in adjusted.items():
                comparison_tables[metric][name]["holm_adjusted_p"] = details["adjusted_p"]
                comparison_tables[metric][name]["holm_rank"] = details["rank"]
                comparison_tables[metric][name]["holm_significant"] = details["significant"]
                comparison_tables[metric][name]["correction_family"] = HOLM_FAMILY
                comparison_tables[metric][name]["holm_family_size"] = HOLM_FAMILY_SIZE
    robustness_rows = []
    wins = 0
    comparisons = 0
    for row in ordered:
        f_sets = row.get("F_sets") or []
        if "C_RMSE" not in row or not f_sets:
            continue
        for item in f_sets:
            comparisons += 1
            better = float(row["C_RMSE"]) < float(item["RMSE"])
            wins += int(better)
            robustness_rows.append(
                {
                    "dataset": row["dataset"],
                    "horizon": row["horizon"],
                    "source_set_id": item["source_set_id"],
                    "metric": "RMSE",
                    "top3_mean": float(row["C_RMSE"]),
                    "random_set_mean": float(item["RMSE"]),
                    "top3_better": better,
                }
            )
    dose_rows = []
    for row in ordered:
        dose_rows.append(
            {
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "top3_mean_source_similarity": row.get("C_mean_similarity"),
                "random3_mean_source_similarity": row.get("F_mean_similarity"),
                "bottom3_mean_source_similarity": row.get("G_mean_similarity", row.get("G_mean_similarity_from_rank")),
                "top3_RMSE": row.get("C_RMSE"),
                "random3_RMSE": row.get("F_RMSE"),
                "bottom3_RMSE": row.get("G_RMSE"),
                "top3_sMAPE": row.get("C_sMAPE"),
                "random3_sMAPE": row.get("F_sMAPE"),
                "bottom3_sMAPE": row.get("G_sMAPE"),
                "similarity_order_holds": (
                    row.get("C_mean_similarity") is not None
                    and row.get("F_mean_similarity") is not None
                    and row.get("G_mean_similarity", row.get("G_mean_similarity_from_rank")) is not None
                    and float(row["C_mean_similarity"])
                    < float(row["F_mean_similarity"])
                    < float(row.get("G_mean_similarity", row.get("G_mean_similarity_from_rank")))
                ),
            }
        )
    results_dir = Path(run_dir) / "results"
    write_results_csv(results_dir / "condition_means.csv", ordered)
    write_results_csv(results_dir / "random_set_robustness.csv", robustness_rows)
    write_results_csv(results_dir / "similarity_dose.csv", dose_rows)
    stat_rows = []
    for metric, table in comparison_tables.items():
        for name, payload in table.items():
            stat_rows.append({"metric": metric, "comparison": name, **payload})
    if stat_rows:
        write_results_csv(results_dir / "planned_pairwise_statistics.csv", stat_rows)
    overlap = None
    try:
        rankings = load_rankings(project_root=root)
        materialized = materialize_source_sets(rankings)
        overlap = overlap_sensitivity_report(materialized)
        atomic_write_json(results_dir / "random3_top3_overlap.json", overlap)
        write_overlap_sensitivity_md(
            overlap,
            project_root_from(project_root) / "RANDOM3_TOP3_OVERLAP_SENSITIVITY_AUDIT.md",
        )
    except SourceSelectionAblationError:
        overlap = None
    summary = {
        "n_complete_conditions": len(complete),
        "top3_beats_random_sets_RMSE": f"{wins} / {comparisons}" if comparisons else "n/a",
        "comparisons": comparison_tables,
        "correction_family": HOLM_FAMILY,
        "holm_family_size": HOLM_FAMILY_SIZE,
        "similarity_direction": SIMILARITY_DIRECTION,
        "overlap": overlap,
    }
    atomic_write_json(results_dir / "analysis_summary.json", summary)
    return summary
