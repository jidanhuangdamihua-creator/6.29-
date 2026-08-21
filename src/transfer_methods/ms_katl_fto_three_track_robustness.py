"""MS-KATL FUSED_TARGET_ONLY three-track paired robustness contract.

Confirmatory evaluation of frozen V1 / V2 / V3 on the V3 primary-target set.
Does not reselect targets or learning rates, does not modify FORMAL_METHODS,
MSML, No-TL, baselines, or screening outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_fto_v2_screening import (
    DEFAULT_V1_RUN_DIRNAME,
    DEFAULT_V2_RUN_DIRNAME,
    v2_architecture,
    v2_architecture_digest,
    v2_training_recipe,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_ARM,
    V3_METHOD_DISPLAY,
    V3_REVIN_CONTRACT,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    DEFAULT_EMA_MOMENTUM,
    DEFAULT_HUBER_DELTA,
    DEFAULT_LOSS_NAME,
    DEFAULT_LR_SCHEDULE,
    DEFAULT_USE_EMA,
    METHOD_NAME,
)
from src.transfer_methods.ms_katl_mechanism_ablation import assert_formal_methods_untouched
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    LR_GRID_LABELS,
    METHOD_ARM as V1_METHOD_ARM,
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    RMSE_TIE_RELATIVE,
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_HORIZON,
    SCREENING_SEED,
    ScreeningContractError,
    architecture_digest as v1_architecture_digest,
    format_lr,
    fused_target_only_architecture,
    parse_lr,
    parse_target_identity,
    select_target_best_lr,
)
from src.transfer_methods.ms_katl_fto_v2_screening import V2_METHOD_ARM, V2_METHOD_DISPLAY
from src.transfer_methods.revin import (
    DEFAULT_REVIN_AFFINE,
    DEFAULT_REVIN_EPS,
    DEFAULT_REVIN_TARGET_CHANNEL,
)

ROBUSTNESS_LABEL = "MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
ROBUSTNESS_TRACK = "confirmatory_robustness"
ROBUSTNESS_ARMS: Tuple[str, ...] = ("v1", "v2", "v3")
ROBUSTNESS_HORIZONS: Tuple[int, ...] = (1, 3, 5)
ROBUSTNESS_SEEDS: Tuple[int, ...] = (42, 43, 44)
EXPECTED_RUNS = 162
PAIRED_CELLS = 54
RUNS_PER_ARM = 54
COMMON_TARGET_POLICY = "V3_FROZEN_PRIMARY_TARGETS"
LR_POLICY = "ARM_SPECIFIC_FROZEN_TARGET_LR"
DEFAULT_V3_RUN_DIRNAME = "20260818_180250_355272_MS_KATL_FTO_V3_SCREENING"
V3_FROZEN_MANIFEST_RELPATH = Path("configs") / "ms_katl_fto_v3_robustness_manifest.json"
EXPECTED_V3_ARCHITECTURE_DIGEST = (
    "ac07f0863906ec9a3b4ed23ea56189ecae70e2b80ac2485cf85f5c7ee270b760"
)
V1_TARGET_BEST_CSV = "screening_target_best_lr.csv"
V1_ALL_RUNS_CSV = "screening_all_runs.csv"
V2_TARGET_BEST_CSV = "v2_screening_target_best_lr.csv"
V2_ALL_RUNS_CSV = "v2_screening_all_runs.csv"
V3_TARGET_BEST_CSV = "v3_screening_target_best_lr.csv"
V3_ALL_RUNS_CSV = "v3_screening_all_runs.csv"
V3_SELECTION_MANIFEST_NAME = "v3_selection_manifest.json"
SCREENING_DIRNAME_MARKERS = (
    "PRIMARY_TARGET_LR_SCREENING",
    "FTO_V2_SCREENING",
    "FTO_V3_SCREENING",
)
REQUIRED_CELL_FIELDS = (
    "status",
    "arm",
    "dataset",
    "target_identity",
    "horizon",
    "seed",
    "learning_rate",
    "learning_rate_label",
    "rmse_original",
    "smape_original",
    "metric_space",
    "architecture_digest",
)
V1_TRAINING_RECIPE_PROVENANCE: Dict[str, Any] = {
    "loss_name": DEFAULT_LOSS_NAME,
    "huber_delta": DEFAULT_HUBER_DELTA,
    "lr_schedule": DEFAULT_LR_SCHEDULE,
    "use_ema": DEFAULT_USE_EMA,
    "ema_momentum": DEFAULT_EMA_MOMENTUM,
}

EXPECTED_COMMON_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("D1", "1/10"),
    ("D2", "1/10"),
    ("D3", "10"),
    ("D4", "166/311"),
    ("D5", "48/1159414"),
    ("D6", "CA_1/FOODS_3_080"),
)


class RobustnessContractError(RuntimeError):
    """Fail-closed three-track robustness contract violation."""


def _frozen_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def v1_training_recipe_provenance() -> Dict[str, Any]:
    """Reported V1 recipe. V1 screening passes no recipe kwargs (library defaults)."""
    recipe = dict(V1_TRAINING_RECIPE_PROVENANCE)
    if recipe["loss_name"] != "mse" or recipe["lr_schedule"] != "constant" or recipe["use_ema"]:
        raise RobustnessContractError(f"V1 recipe provenance drifted: {recipe}")
    return recipe


def v1_revin_config() -> Dict[str, Any]:
    return {
        "enabled": False,
        "affine": bool(DEFAULT_REVIN_AFFINE),
        "eps": float(DEFAULT_REVIN_EPS),
        "target_channel": int(DEFAULT_REVIN_TARGET_CHANNEL),
    }


def v2_revin_config() -> Dict[str, Any]:
    return v1_revin_config()


def v3_revin_config() -> Dict[str, Any]:
    arch = v3_architecture()
    config = {
        "enabled": bool(arch["revin_enabled"]),
        "affine": bool(arch["revin_affine"]),
        "eps": float(arch["revin_eps"]),
        "target_channel": int(arch["revin_target_channel"]),
    }
    expected = {
        "enabled": True,
        "affine": True,
        "eps": 1e-5,
        "target_channel": 0,
    }
    if config["enabled"] != expected["enabled"] or config["affine"] != expected["affine"]:
        raise RobustnessContractError(f"V3 RevIN drifted: {config}")
    if not math.isclose(config["eps"], expected["eps"], rel_tol=0.0, abs_tol=0.0):
        raise RobustnessContractError(f"V3 RevIN eps drifted: {config}")
    if config["target_channel"] != expected["target_channel"]:
        raise RobustnessContractError(f"V3 RevIN target_channel drifted: {config}")
    if dict(V3_REVIN_CONTRACT) != {
        "revin_enabled": True,
        "revin_affine": True,
        "revin_eps": float(DEFAULT_REVIN_EPS),
        "revin_target_channel": int(DEFAULT_REVIN_TARGET_CHANNEL),
    }:
        raise RobustnessContractError(f"V3_REVIN_CONTRACT drifted: {V3_REVIN_CONTRACT}")
    return config


@dataclass(frozen=True)
class RobustnessArmSpec:
    key: str
    label: str
    method_display: str
    ablation_arm: str
    architecture: Mapping[str, Any]
    architecture_digest: str
    training_recipe: Mapping[str, Any]
    pass_training_recipe: bool
    revin_enabled: bool
    revin_affine: bool
    revin_eps: float
    revin_target_channel: int
    screening_run_dirname: str
    target_best_csv_name: str
    all_runs_csv_name: str
    selection_manifest_name: Optional[str] = None

    def run_ablation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.pass_training_recipe:
            kwargs["training_recipe"] = dict(self.training_recipe)
        if self.revin_enabled:
            kwargs["revin_enabled"] = True
            kwargs["revin_affine"] = bool(self.revin_affine)
            kwargs["revin_eps"] = float(self.revin_eps)
            kwargs["revin_target_channel"] = int(self.revin_target_channel)
        return kwargs

    def revin_provenance(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.revin_enabled),
            "affine": bool(self.revin_affine),
            "eps": float(self.revin_eps),
            "target_channel": int(self.revin_target_channel),
        }


def v1_arm_spec() -> RobustnessArmSpec:
    arch = fused_target_only_architecture()
    if arch != v2_architecture():
        raise RobustnessContractError("V1 FUSED_TARGET_ONLY core drifted from V2")
    revin = v1_revin_config()
    return RobustnessArmSpec(
        key="v1",
        label="V1",
        method_display=V1_METHOD_DISPLAY,
        ablation_arm=V1_METHOD_ARM,
        architecture=_frozen_mapping(arch),
        architecture_digest=v1_architecture_digest(),
        training_recipe=_frozen_mapping(v1_training_recipe_provenance()),
        pass_training_recipe=False,
        revin_enabled=False,
        revin_affine=bool(revin["affine"]),
        revin_eps=float(revin["eps"]),
        revin_target_channel=int(revin["target_channel"]),
        screening_run_dirname=DEFAULT_V1_RUN_DIRNAME,
        target_best_csv_name=V1_TARGET_BEST_CSV,
        all_runs_csv_name=V1_ALL_RUNS_CSV,
        selection_manifest_name="primary_target_lr_selection_manifest.json",
    )


def v2_arm_spec() -> RobustnessArmSpec:
    recipe = v2_training_recipe()
    expected = {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    if dict(recipe) != expected:
        raise RobustnessContractError(f"V2 training recipe drifted: {recipe}")
    revin = v2_revin_config()
    return RobustnessArmSpec(
        key="v2",
        label="V2",
        method_display=V2_METHOD_DISPLAY,
        ablation_arm=V2_METHOD_ARM,
        architecture=_frozen_mapping(v2_architecture()),
        architecture_digest=v2_architecture_digest(),
        training_recipe=_frozen_mapping(recipe),
        pass_training_recipe=True,
        revin_enabled=False,
        revin_affine=bool(revin["affine"]),
        revin_eps=float(revin["eps"]),
        revin_target_channel=int(revin["target_channel"]),
        screening_run_dirname=DEFAULT_V2_RUN_DIRNAME,
        target_best_csv_name=V2_TARGET_BEST_CSV,
        all_runs_csv_name=V2_ALL_RUNS_CSV,
        selection_manifest_name="v2_selection_manifest.json",
    )


def v3_arm_spec() -> RobustnessArmSpec:
    recipe = v3_training_recipe()
    if dict(recipe) != dict(v2_training_recipe()):
        raise RobustnessContractError("V3 recipe must stay identical to frozen V2")
    arch = v3_architecture()
    digest = v3_architecture_digest()
    if digest != EXPECTED_V3_ARCHITECTURE_DIGEST:
        raise RobustnessContractError(
            f"V3 architecture digest drifted: {digest} != {EXPECTED_V3_ARCHITECTURE_DIGEST}"
        )
    revin = v3_revin_config()
    return RobustnessArmSpec(
        key="v3",
        label="V3",
        method_display=V3_METHOD_DISPLAY,
        ablation_arm=V3_METHOD_ARM,
        architecture=_frozen_mapping(arch),
        architecture_digest=digest,
        training_recipe=_frozen_mapping(recipe),
        pass_training_recipe=True,
        revin_enabled=True,
        revin_affine=bool(revin["affine"]),
        revin_eps=float(revin["eps"]),
        revin_target_channel=int(revin["target_channel"]),
        screening_run_dirname=DEFAULT_V3_RUN_DIRNAME,
        target_best_csv_name=V3_TARGET_BEST_CSV,
        all_runs_csv_name=V3_ALL_RUNS_CSV,
        selection_manifest_name=V3_SELECTION_MANIFEST_NAME,
    )


def robustness_arm_registry() -> Mapping[str, RobustnessArmSpec]:
    specs = (v1_arm_spec(), v2_arm_spec(), v3_arm_spec())
    registry = {spec.key: spec for spec in specs}
    if tuple(registry) != ROBUSTNESS_ARMS:
        raise RobustnessContractError(f"arm registry keys drifted: {tuple(registry)}")
    if len(registry) != 3:
        raise RobustnessContractError("exactly 3 arms are required")
    return registry


def assert_stage_a_is_arm_independent() -> str:
    """Stage-A does not take recipe/RevIN; all arms share fused_target_only core."""
    registry = robustness_arm_registry()
    cores = []
    for spec in registry.values():
        if spec.ablation_arm != "fused_target_only":
            raise RobustnessContractError(
                f"{spec.key} ablation arm must be fused_target_only, got {spec.ablation_arm}"
            )
        core = {key: spec.architecture[key] for key in fused_target_only_architecture()}
        cores.append(core)
        kwargs = spec.run_ablation_kwargs()
        if "k" in kwargs or "source_epochs" in kwargs or "horizon" in kwargs:
            raise RobustnessContractError(f"{spec.key} Stage-B kwargs leak Stage-A fields")
    if any(core != cores[0] for core in cores):
        raise RobustnessContractError("V1/V2/V3 FUSED_TARGET_ONLY core diverged; refuse Stage-A reuse")
    return "PASS"


@dataclass(frozen=True)
class CommonTarget:
    dataset: str
    target_identity: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.dataset, self.target_identity)


@dataclass(frozen=True)
class RunIdentity:
    arm: str
    dataset: str
    target_identity: str
    horizon: int
    seed: int
    selected_lr: float
    selected_lr_label: str

    @property
    def identity_tuple(self) -> Tuple[Any, ...]:
        return (
            self.arm,
            self.dataset,
            self.target_identity,
            int(self.horizon),
            int(self.seed),
            self.selected_lr_label,
        )

    @property
    def paired_tuple(self) -> Tuple[Any, ...]:
        return (self.dataset, self.target_identity, int(self.horizon), int(self.seed))

    @property
    def cell_id(self) -> str:
        target_slug = self.target_identity.replace("/", "-")
        return (
            f"{self.arm}_{self.dataset}_{target_slug}"
            f"_h{int(self.horizon)}_s{int(self.seed)}_{self.selected_lr_label}"
        )


def project_root_from(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[2]


def screening_run_root(arm: RobustnessArmSpec, project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "runs" / arm.screening_run_dirname


def screening_results_dir(arm: RobustnessArmSpec, project_root: Path) -> Path:
    return screening_run_root(arm, project_root) / "results"


def target_best_csv_path(arm: RobustnessArmSpec, project_root: Path) -> Path:
    return screening_results_dir(arm, project_root) / arm.target_best_csv_name


def all_runs_csv_path(arm: RobustnessArmSpec, project_root: Path) -> Path:
    return screening_results_dir(arm, project_root) / arm.all_runs_csv_name


def v3_screening_manifest_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / "runs"
        / DEFAULT_V3_RUN_DIRNAME
        / "results"
        / V3_SELECTION_MANIFEST_NAME
    )


def v3_frozen_manifest_path(project_root: Path) -> Path:
    return Path(project_root) / V3_FROZEN_MANIFEST_RELPATH


def assert_not_screening_output_dir(path: Path) -> None:
    name = Path(path).name
    for marker in SCREENING_DIRNAME_MARKERS:
        if marker in name and ROBUSTNESS_LABEL not in name:
            raise RobustnessContractError(
                f"refusing to write robustness outputs into screening run: {path}"
            )


def assert_formal_methods_unchanged() -> Tuple[str, ...]:
    expected = assert_formal_methods_untouched()
    if V1_METHOD_DISPLAY in FORMAL_METHODS or V2_METHOD_DISPLAY in FORMAL_METHODS:
        raise RobustnessContractError("robustness must not add FUSED_TARGET_ONLY to FORMAL_METHODS")
    if V3_METHOD_DISPLAY in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise RobustnessContractError("robustness must not add MS-KATL methods to FORMAL_METHODS")
    return expected


def parse_arm_keys(spec: Optional[object]) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return ROBUSTNESS_ARMS
    parts = [part.strip().lower() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved: List[str] = []
    registry = robustness_arm_registry()
    for part in parts:
        key = part if part in registry else part.replace("fused_target_only_", "v")
        if key not in registry:
            raise RobustnessContractError(f"unknown arm {part!r}; expected {list(ROBUSTNESS_ARMS)}")
        if key in resolved:
            raise RobustnessContractError(f"duplicate arm {key!r}")
        resolved.append(key)
    return tuple(resolved)


def parse_dataset_ids(spec: Optional[object]) -> Tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return SCREENING_DATASETS
    parts = [part.strip().upper() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(f"D{int(part.replace('D', ''))}" for part in parts)
    extra = [item for item in resolved if item not in SCREENING_DATASETS]
    if extra:
        raise RobustnessContractError(f"unknown datasets: {extra}")
    seen = set()
    ordered: List[str] = []
    for item in resolved:
        if item in seen:
            raise RobustnessContractError(f"duplicate dataset {item}")
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def parse_int_tuple(spec: Optional[object], allowed: Sequence[int], *, name: str) -> Tuple[int, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return tuple(int(item) for item in allowed)
    parts = [part.strip() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(int(part) for part in parts)
    extra = [item for item in resolved if item not in set(allowed)]
    if extra:
        raise RobustnessContractError(f"{name} values {extra} are not in frozen set {list(allowed)}")
    seen = set()
    ordered: List[int] = []
    for item in resolved:
        if item in seen:
            raise RobustnessContractError(f"duplicate {name} {item}")
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not Path(path).is_file():
        raise RobustnessContractError(f"required CSV is missing: {path}")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise RobustnessContractError(f"required JSON is missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_v3_screening_primary_targets(project_root: Path) -> Tuple[CommonTarget, ...]:
    manifest = _read_json(v3_screening_manifest_path(project_root))
    datasets = manifest.get("datasets") or {}
    if set(datasets) != set(SCREENING_DATASETS):
        raise RobustnessContractError("V3 screening manifest must contain exactly D1-D6")
    targets: List[CommonTarget] = []
    for dataset in SCREENING_DATASETS:
        identity = str(datasets[dataset].get("primary_target") or "").strip()
        parsed = parse_target_identity(dataset, identity)
        targets.append(CommonTarget(dataset=dataset, target_identity=parsed["target_identity"]))
    observed = tuple((item.dataset, item.target_identity) for item in targets)
    if observed != EXPECTED_COMMON_TARGETS:
        raise RobustnessContractError(
            "V3 frozen primary targets drifted from the expected common set: "
            f"{observed} != {EXPECTED_COMMON_TARGETS}"
        )
    return tuple(targets)


def _lr_label_from_target_best_row(row: Mapping[str, Any]) -> str:
    raw = row.get("best_lr") or row.get("learning_rate_label") or row.get("learning_rate")
    if raw is None or str(raw).strip() == "":
        raise RobustnessContractError(f"target-best row missing LR: {dict(row)}")
    return format_lr(parse_lr(raw))


def load_target_best_lr_map(csv_path: Path) -> Dict[Tuple[str, str], str]:
    rows = _read_csv_rows(csv_path)
    mapping: Dict[Tuple[str, str], str] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip()
        identity = str(row.get("target_identity") or "").strip()
        if not dataset or not identity:
            continue
        if str(row.get("run_status", "OK")).upper() != "OK":
            continue
        mapping[(dataset, identity)] = _lr_label_from_target_best_row(row)
    return mapping


def reconstruct_target_best_lr_map(all_runs_csv: Path) -> Dict[Tuple[str, str], str]:
    """Rebuild target-best mapping from screening all-runs using frozen select_target_best_lr."""
    rows = _read_csv_rows(all_runs_csv)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "").strip()
        identity = str(row.get("target_identity") or "").strip()
        if not dataset or not identity:
            continue
        try:
            lr_label = format_lr(parse_lr(row.get("learning_rate_label") or row.get("learning_rate")))
        except (ScreeningContractError, TypeError, ValueError):
            continue
        payload = dict(row)
        payload["learning_rate"] = parse_lr(lr_label)
        payload["learning_rate_label"] = lr_label
        grouped.setdefault((dataset, identity), []).append(payload)
    mapping: Dict[Tuple[str, str], str] = {}
    for key, group in grouped.items():
        best = select_target_best_lr(group)
        if str(best.get("run_status", "OK")).upper() != "OK" or not best.get("best_lr"):
            continue
        mapping[key] = format_lr(parse_lr(best["best_lr"]))
    if not mapping:
        raise RobustnessContractError(f"could not reconstruct target-best LR map from {all_runs_csv}")
    return mapping


def load_arm_frozen_lr_map(
    arm: RobustnessArmSpec,
    common_targets: Sequence[CommonTarget],
    project_root: Path,
) -> Dict[Tuple[str, str], str]:
    best_path = target_best_csv_path(arm, project_root)
    runs_path = all_runs_csv_path(arm, project_root)
    if best_path.is_file():
        mapping = load_target_best_lr_map(best_path)
        if runs_path.is_file():
            reconstructed = reconstruct_target_best_lr_map(runs_path)
            for target in common_targets:
                left = mapping.get(target.key)
                right = reconstructed.get(target.key)
                if left is None or right is None:
                    continue
                if left != right:
                    raise RobustnessContractError(
                        f"{arm.label} target-best artifact disagrees with select_target_best_lr "
                        f"for {target.dataset} {target.target_identity}: {left} vs {right}"
                    )
    elif runs_path.is_file():
        mapping = reconstruct_target_best_lr_map(runs_path)
    else:
        raise RobustnessContractError(
            f"{arm.label} frozen LR source missing: {best_path} and {runs_path}"
        )
    resolved: Dict[Tuple[str, str], str] = {}
    missing: List[str] = []
    for target in common_targets:
        label = mapping.get(target.key)
        if label is None:
            missing.append(f"{target.dataset}:{target.target_identity}")
            continue
        resolved[target.key] = format_lr(parse_lr(label))
    if missing:
        raise RobustnessContractError(
            f"FAIL CLOSED: {arm.label} has no frozen selected LR for common targets {missing}. "
            "No dataset-primary / default / argmin fallback is allowed."
        )
    return resolved


def load_target_lr_contract_table(
    project_root: Path,
    *,
    common_targets: Optional[Sequence[CommonTarget]] = None,
) -> Dict[str, Any]:
    registry = robustness_arm_registry()
    targets = tuple(common_targets) if common_targets is not None else load_v3_screening_primary_targets(project_root)
    if tuple((item.dataset, item.target_identity) for item in targets) != EXPECTED_COMMON_TARGETS:
        raise RobustnessContractError("common target identity drift")
    arm_maps = {
        key: load_arm_frozen_lr_map(spec, targets, project_root)
        for key, spec in registry.items()
    }
    rows = []
    for target in targets:
        rows.append(
            {
                "dataset": target.dataset,
                "target_identity": target.target_identity,
                "v1_lr": arm_maps["v1"][target.key],
                "v2_lr": arm_maps["v2"][target.key],
                "v3_lr": arm_maps["v3"][target.key],
            }
        )
    return {
        "targets": targets,
        "arm_maps": arm_maps,
        "rows": rows,
        "COMMON_TARGET_COUNT": len(targets),
        "V1_LR_COUNT": len(arm_maps["v1"]),
        "V2_LR_COUNT": len(arm_maps["v2"]),
        "V3_LR_COUNT": len(arm_maps["v3"]),
        "TARGET_IDENTITY_MATCH": "YES",
        "LR_SELECTION_FROZEN": "YES",
    }


def build_run_identities(
    *,
    arms: Sequence[str],
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_table: Mapping[str, Any],
) -> Tuple[RunIdentity, ...]:
    registry = robustness_arm_registry()
    targets_by_dataset = {
        item.dataset: item for item in lr_table["targets"]
    }
    identities: List[RunIdentity] = []
    for arm_key in arms:
        spec = registry[arm_key]
        lr_map: Mapping[Tuple[str, str], str] = lr_table["arm_maps"][arm_key]
        for dataset in datasets:
            target = targets_by_dataset.get(dataset)
            if target is None:
                raise RobustnessContractError(f"no frozen common target for {dataset}")
            lr_label = lr_map.get(target.key)
            if lr_label is None:
                raise RobustnessContractError(
                    f"FAIL CLOSED: {spec.label} missing frozen LR for {dataset} {target.target_identity}"
                )
            lr_value = parse_lr(lr_label)
            for horizon in horizons:
                for seed in seeds:
                    identities.append(
                        RunIdentity(
                            arm=spec.key,
                            dataset=dataset,
                            target_identity=target.target_identity,
                            horizon=int(horizon),
                            seed=int(seed),
                            selected_lr=float(lr_value),
                            selected_lr_label=format_lr(lr_value),
                        )
                    )
    assert_unique_identities(identities)
    return tuple(identities)


def assert_unique_identities(identities: Sequence[RunIdentity]) -> None:
    tuples = [item.identity_tuple for item in identities]
    if len(tuples) != len(set(tuples)):
        raise RobustnessContractError("duplicate robustness cell identity")
    paired = {}
    for item in identities:
        paired.setdefault(item.paired_tuple, set()).add(item.arm)
    datasets = {item.dataset: item.target_identity for item in identities}
    # same dataset must not mix target identities across arms
    for item in identities:
        if datasets[item.dataset] != item.target_identity:
            raise RobustnessContractError(
                f"target identity drift on {item.dataset}: "
                f"{datasets[item.dataset]} vs {item.target_identity}"
            )


def expected_full_matrix_counts(
    arms: Sequence[str],
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
) -> Dict[str, int]:
    n_paired = len(datasets) * len(horizons) * len(seeds)
    n_runs = n_paired * len(arms)
    return {
        "paired_cells": n_paired,
        "runs_per_arm": len(datasets) * len(horizons) * len(seeds),
        "expected_runs": n_runs,
        "unique_runs": n_runs,
    }


def build_v3_frozen_robustness_manifest(project_root: Path) -> Dict[str, Any]:
    screening = _read_json(v3_screening_manifest_path(project_root))
    if screening.get("method") != V3_METHOD_DISPLAY:
        raise RobustnessContractError("V3 screening manifest method drifted")
    if screening.get("selection_frozen") is True:
        # Screening file must remain the unfrozen development artifact.
        raise RobustnessContractError("refusing to treat the screening manifest as the freeze artifact")
    spec = v3_arm_spec()
    targets = load_v3_screening_primary_targets(project_root)
    lr_map = load_arm_frozen_lr_map(spec, targets, project_root)
    datasets_payload: Dict[str, Any] = {}
    screening_datasets = screening.get("datasets") or {}
    for target in targets:
        selected = screening_datasets[target.dataset]
        lr_label = lr_map[target.key]
        screening_lr = format_lr(parse_lr(selected["learning_rate"]))
        if screening_lr != lr_label:
            raise RobustnessContractError(
                f"V3 primary LR mismatch for {target.dataset}: "
                f"manifest={screening_lr} target-best={lr_label}"
            )
        datasets_payload[target.dataset] = {
            "primary_target": target.target_identity,
            "learning_rate": lr_label,
            "rmse_original": selected.get("rmse_original"),
            "smape_original": selected.get("smape_original"),
            "verdict": selected.get("verdict"),
            "selection_role": "development_selected",
            "selection_frozen": True,
            "target_selection_stage": "development_screening",
        }
    recipe = dict(spec.training_recipe)
    revin = spec.revin_provenance()
    manifest = {
        "selection_frozen": True,
        "method": V3_METHOD_DISPLAY,
        "selection_stage": "development_screening",
        "common_target_policy": COMMON_TARGET_POLICY,
        "targets": 6,
        "screening_lr_grid": list(LR_GRID),
        "screening_lr_grid_labels": list(LR_GRID_LABELS),
        "screening_seed": SCREENING_SEED,
        "screening_horizon": SCREENING_HORIZON,
        "epochs": SCREENING_EPOCHS,
        "training_recipe": recipe,
        "RevIN": revin,
        "architecture": dict(spec.architecture),
        "architecture_digest": spec.architecture_digest,
        "selection_policy": {
            "metric": "rmse_original",
            "metric_space": ORIGINAL_SALES_SPACE,
            "relative_tie_threshold": RMSE_TIE_RELATIVE,
            "tie_break": "numerically_smallest_lr",
        },
        "metric_space": ORIGINAL_SALES_SPACE,
        "auto_start_robustness": False,
        "source_run_root": str(screening_run_root(spec, project_root)),
        "source_selection_manifest": str(v3_screening_manifest_path(project_root)),
        "source_target_best_csv": str(target_best_csv_path(spec, project_root)),
        "source_all_runs_csv": str(all_runs_csv_path(spec, project_root)),
        "source_screening_selection_frozen": False,
        "datasets": datasets_payload,
        "V3_SELECTION_CONTRACT_VERIFIED": True,
        "READY_TO_FREEZE_V3_AND_ENTER_ROBUSTNESS": "YES",
    }
    validate_v3_frozen_robustness_manifest(manifest, project_root=project_root)
    return manifest


def validate_v3_frozen_robustness_manifest(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> Dict[str, Any]:
    required = {
        "selection_frozen": True,
        "method": V3_METHOD_DISPLAY,
        "selection_stage": "development_screening",
        "common_target_policy": COMMON_TARGET_POLICY,
        "targets": 6,
        "screening_seed": SCREENING_SEED,
        "screening_horizon": SCREENING_HORIZON,
        "epochs": SCREENING_EPOCHS,
        "architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RobustnessContractError(
                f"frozen V3 manifest field {key}={manifest.get(key)!r} != {expected!r}"
            )
    if list(manifest.get("screening_lr_grid") or []) != list(LR_GRID):
        raise RobustnessContractError("frozen V3 LR grid drifted")
    recipe = dict(manifest.get("training_recipe") or {})
    if recipe != dict(v3_training_recipe()):
        raise RobustnessContractError(f"frozen V3 recipe drifted: {recipe}")
    revin = dict(manifest.get("RevIN") or {})
    if revin != v3_revin_config():
        raise RobustnessContractError(f"frozen V3 RevIN drifted: {revin}")
    policy = dict(manifest.get("selection_policy") or {})
    if policy.get("metric") != "rmse_original":
        raise RobustnessContractError("frozen selection metric must be rmse_original")
    if policy.get("metric_space") != ORIGINAL_SALES_SPACE:
        raise RobustnessContractError("frozen selection metric space drifted")
    if float(policy.get("relative_tie_threshold")) != RMSE_TIE_RELATIVE:
        raise RobustnessContractError("frozen tie threshold drifted")
    if policy.get("tie_break") != "numerically_smallest_lr":
        raise RobustnessContractError("frozen tie-break drifted")
    datasets = manifest.get("datasets") or {}
    if tuple(datasets) != SCREENING_DATASETS:
        raise RobustnessContractError("frozen V3 datasets must be D1-D6 in order")
    observed = []
    for dataset in SCREENING_DATASETS:
        row = datasets[dataset]
        identity = parse_target_identity(dataset, row["primary_target"])["target_identity"]
        observed.append((dataset, identity, format_lr(parse_lr(row["learning_rate"]))))
        if row.get("selection_frozen") is not True:
            raise RobustnessContractError(f"{dataset} is not frozen in the V3 robustness manifest")
    expected_identities = [(ds, tgt) for ds, tgt in EXPECTED_COMMON_TARGETS]
    if [(ds, tgt) for ds, tgt, _lr in observed] != expected_identities:
        raise RobustnessContractError(f"frozen common targets drifted: {observed}")
    if v3_architecture_digest() != manifest["architecture_digest"]:
        raise RobustnessContractError("frozen digest disagrees with live v3_architecture_digest()")
    source_manifest = Path(str(manifest.get("source_selection_manifest") or ""))
    if source_manifest.is_file():
        screening = _read_json(source_manifest)
        if screening.get("selection_frozen") is True:
            raise RobustnessContractError("screening selection manifest must stay unfrozen")
    return dict(manifest)


def write_v3_frozen_robustness_manifest(project_root: Path) -> Path:
    path = v3_frozen_manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v3_frozen_robustness_manifest(project_root)
    encoded = json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"
    path.write_text(encoded, encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_v3_frozen_robustness_manifest(loaded, project_root=project_root)
    if loaded.get("selection_frozen") is not True:
        raise RobustnessContractError("read-back freeze validation failed")
    return path


def load_v3_frozen_robustness_manifest(project_root: Path) -> Dict[str, Any]:
    path = v3_frozen_manifest_path(project_root)
    manifest = _read_json(path)
    return validate_v3_frozen_robustness_manifest(manifest, project_root=project_root)


def common_targets_from_frozen_manifest(manifest: Mapping[str, Any]) -> Tuple[CommonTarget, ...]:
    datasets = manifest.get("datasets") or {}
    targets = []
    for dataset in SCREENING_DATASETS:
        identity = parse_target_identity(dataset, datasets[dataset]["primary_target"])["target_identity"]
        targets.append(CommonTarget(dataset=dataset, target_identity=identity))
    observed = tuple((item.dataset, item.target_identity) for item in targets)
    if observed != EXPECTED_COMMON_TARGETS:
        raise RobustnessContractError("frozen common target set drifted")
    return tuple(targets)


def format_preflight(
    *,
    arms: Sequence[str],
    datasets: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lr_table: Mapping[str, Any],
    counts: Mapping[str, int],
) -> str:
    registry = robustness_arm_registry()
    arm_labels = ",".join(registry[key].label for key in arms)
    lines = [
        "============================================================",
        "MS-KATL FTO THREE-TRACK ROBUSTNESS PREFLIGHT",
        "============================================================",
        "",
        f"ARMS={arm_labels}",
        f"DATASETS={','.join(datasets)}",
        f"COMMON_TARGET_COUNT={lr_table['COMMON_TARGET_COUNT']}",
        f"HORIZONS={','.join(str(item) for item in horizons)}",
        f"SEEDS={','.join(str(item) for item in seeds)}",
        "",
        f"PAIRED_CELLS={counts['paired_cells']}",
        f"RUNS_PER_ARM={counts['runs_per_arm']}",
        f"EXPECTED_RUNS={counts['expected_runs']}",
        f"UNIQUE_RUNS={counts['unique_runs']}",
        "",
        f"TARGET_POLICY={COMMON_TARGET_POLICY}",
        f"LR_POLICY={LR_POLICY}",
        "ROBUSTNESS_RESELECTION=NO",
        "",
        "V1_CONFIG_FROZEN=YES",
        "V2_CONFIG_FROZEN=YES",
        "V3_CONFIG_FROZEN=YES",
        "",
        "FORMAL_METHODS_CHANGE=NO",
        f"SHARED_STAGE_A_EQUIVALENCE={assert_stage_a_is_arm_independent()}",
        "READY_FOR_FORMAL_RUN=YES",
        "============================================================",
        "",
        f"{'DATASET':<8}{'TARGET':<22}{'V1_LR':<10}{'V2_LR':<10}{'V3_LR':<10}",
        "------------------------------------------------------------",
    ]
    for row in lr_table["rows"]:
        if row["dataset"] not in set(datasets):
            continue
        lines.append(
            f"{row['dataset']:<8}{row['target_identity']:<22}"
            f"{row['v1_lr']:<10}{row['v2_lr']:<10}{row['v3_lr']:<10}"
        )
    lines.extend(
        [
            "------------------------------------------------------------",
            f"COMMON_TARGET_COUNT={lr_table['COMMON_TARGET_COUNT']}",
            f"V1_LR_COUNT={lr_table['V1_LR_COUNT']}",
            f"V2_LR_COUNT={lr_table['V2_LR_COUNT']}",
            f"V3_LR_COUNT={lr_table['V3_LR_COUNT']}",
            f"TARGET_IDENTITY_MATCH={lr_table['TARGET_IDENTITY_MATCH']}",
            f"LR_SELECTION_FROZEN={lr_table['LR_SELECTION_FROZEN']}",
            "============================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def relative_improvement(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """(old - new) / old. Positive = improvement."""
    if old is None or new is None:
        return None
    old_v = float(old)
    new_v = float(new)
    if not math.isfinite(old_v) or not math.isfinite(new_v):
        return None
    if old_v == 0.0:
        return 0.0 if new_v == 0.0 else None
    return (old_v - new_v) / old_v


def signed_delta(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None:
        return None
    return float(new) - float(old)


def win_tie_loss(old: float, new: float) -> str:
    if new < old:
        return "win"
    if new == old:
        return "tie"
    return "loss"


def pair_completed_runs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("status") or row.get("run_status") or "").upper() != "OK":
            continue
        key = (
            str(row["dataset"]),
            str(row["target_identity"]),
            int(row["horizon"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, {})[str(row["arm"]).lower()] = row
    paired: List[Dict[str, Any]] = []
    for (dataset, target, horizon, seed), arms in sorted(grouped.items()):
        if set(arms) != {"v1", "v2", "v3"}:
            continue
        v1 = arms["v1"]
        v2 = arms["v2"]
        v3 = arms["v3"]
        for item in (v1, v2, v3):
            if str(item.get("metric_space")) != ORIGINAL_SALES_SPACE:
                raise RobustnessContractError(
                    f"paired cell uses non-original metric space: {item.get('metric_space')}"
                )
        v1_rmse = float(v1["rmse_original"])
        v2_rmse = float(v2["rmse_original"])
        v3_rmse = float(v3["rmse_original"])
        v1_smape = float(v1["smape_original"])
        v2_smape = float(v2["smape_original"])
        v3_smape = float(v3["smape_original"])
        paired.append(
            {
                "dataset": dataset,
                "target": target,
                "horizon": horizon,
                "seed": seed,
                "V1_RMSE": v1_rmse,
                "V2_RMSE": v2_rmse,
                "V3_RMSE": v3_rmse,
                "V1_sMAPE": v1_smape,
                "V2_sMAPE": v2_smape,
                "V3_sMAPE": v3_smape,
                "dRMSE_V2_vs_V1": v2_rmse - v1_rmse,
                "dRMSE_V3_vs_V2": v3_rmse - v2_rmse,
                "dRMSE_V3_vs_V1": v3_rmse - v1_rmse,
                "relative_RMSE_improvement_V2_vs_V1": relative_improvement(v1_rmse, v2_rmse),
                "relative_RMSE_improvement_V3_vs_V2": relative_improvement(v2_rmse, v3_rmse),
                "relative_RMSE_improvement_V3_vs_V1": relative_improvement(v1_rmse, v3_rmse),
                "dSMAPE_V2_vs_V1": v2_smape - v1_smape,
                "dSMAPE_V3_vs_V2": v3_smape - v2_smape,
                "dSMAPE_V3_vs_V1": v3_smape - v1_smape,
                "relative_sMAPE_improvement_V2_vs_V1": relative_improvement(v1_smape, v2_smape),
                "relative_sMAPE_improvement_V3_vs_V2": relative_improvement(v2_smape, v3_smape),
                "relative_sMAPE_improvement_V3_vs_V1": relative_improvement(v1_smape, v3_smape),
            }
        )
    return paired


def _wtl_counts(rows: Sequence[Mapping[str, Any]], old_key: str, new_key: str) -> Dict[str, int]:
    wins = ties = losses = 0
    for row in rows:
        verdict = win_tie_loss(float(row[old_key]), float(row[new_key]))
        if verdict == "win":
            wins += 1
        elif verdict == "tie":
            ties += 1
        else:
            losses += 1
    return {"win": wins, "tie": ties, "loss": losses}


def _relative_stats(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Optional[float]]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {
            "mean_relative_improvement": None,
            "median_relative_improvement": None,
            "worst_case_regression": None,
            "best_case_improvement": None,
        }
    return {
        "mean_relative_improvement": float(sum(values) / len(values)),
        "median_relative_improvement": float(statistics.median(values)),
        "worst_case_regression": float(min(values)),
        "best_case_improvement": float(max(values)),
    }


def summarize_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "scope": scope,
        "dataset": dataset or "ALL",
        "paired_cells": len(rows),
        "V1_median_RMSE": float(statistics.median(float(r["V1_RMSE"]) for r in rows)) if rows else None,
        "V2_median_RMSE": float(statistics.median(float(r["V2_RMSE"]) for r in rows)) if rows else None,
        "V3_median_RMSE": float(statistics.median(float(r["V3_RMSE"]) for r in rows)) if rows else None,
        "V1_median_sMAPE": float(statistics.median(float(r["V1_sMAPE"]) for r in rows)) if rows else None,
        "V2_median_sMAPE": float(statistics.median(float(r["V2_sMAPE"]) for r in rows)) if rows else None,
        "V3_median_sMAPE": float(statistics.median(float(r["V3_sMAPE"]) for r in rows)) if rows else None,
    }
    comparisons = (
        ("V2_vs_V1", "V1_RMSE", "V2_RMSE", "relative_RMSE_improvement_V2_vs_V1",
         "V1_sMAPE", "V2_sMAPE", "relative_sMAPE_improvement_V2_vs_V1"),
        ("V3_vs_V2", "V2_RMSE", "V3_RMSE", "relative_RMSE_improvement_V3_vs_V2",
         "V2_sMAPE", "V3_sMAPE", "relative_sMAPE_improvement_V3_vs_V2"),
        ("V3_vs_V1", "V1_RMSE", "V3_RMSE", "relative_RMSE_improvement_V3_vs_V1",
         "V1_sMAPE", "V3_sMAPE", "relative_sMAPE_improvement_V3_vs_V1"),
    )
    for name, old_rmse, new_rmse, rel_rmse, old_smape, new_smape, rel_smape in comparisons:
        rmse_wtl = _wtl_counts(rows, old_rmse, new_rmse)
        smape_wtl = _wtl_counts(rows, old_smape, new_smape)
        payload[f"{name}_RMSE_win"] = rmse_wtl["win"]
        payload[f"{name}_RMSE_tie"] = rmse_wtl["tie"]
        payload[f"{name}_RMSE_loss"] = rmse_wtl["loss"]
        payload[f"{name}_sMAPE_win"] = smape_wtl["win"]
        payload[f"{name}_sMAPE_tie"] = smape_wtl["tie"]
        payload[f"{name}_sMAPE_loss"] = smape_wtl["loss"]
        for prefix, stats in (
            (f"{name}_RMSE", _relative_stats(rows, rel_rmse)),
            (f"{name}_sMAPE", _relative_stats(rows, rel_smape)),
        ):
            payload[f"{prefix}_mean_relative_improvement"] = stats["mean_relative_improvement"]
            payload[f"{prefix}_median_relative_improvement"] = stats["median_relative_improvement"]
            payload[f"{prefix}_worst_case_regression"] = stats["worst_case_regression"]
            payload[f"{prefix}_best_case_improvement"] = stats["best_case_improvement"]
    return payload


def summarize_by_dataset(paired_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    summaries = []
    for dataset in SCREENING_DATASETS:
        rows = by_dataset.get(dataset, [])
        if not rows:
            continue
        summaries.append(summarize_paired_rows(rows, scope="dataset", dataset=dataset))
    return summaries


def optional_paired_wilcoxon(paired_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "reason": "not_computed"}
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:  # pragma: no cover - environment-dependent
        result["reason"] = f"scipy_unavailable: {type(exc).__name__}: {exc}"
        return result

    def _test(old_key: str, new_key: str) -> Dict[str, Any]:
        old = [float(row[old_key]) for row in paired_rows]
        new = [float(row[new_key]) for row in paired_rows]
        diffs = [a - b for a, b in zip(old, new)]
        if not diffs:
            return {"status": "SKIP", "reason": "empty"}
        if all(value == 0.0 for value in diffs):
            return {"status": "SKIP", "reason": "zero_difference"}
        try:
            stat, pvalue = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        except Exception as exc:
            return {"status": "SKIP", "reason": f"{type(exc).__name__}: {exc}"}
        return {
            "status": "OK",
            "statistic": float(stat),
            "pvalue": float(pvalue),
            "n": len(diffs),
        }

    result["available"] = True
    result["reason"] = ""
    result["V2_vs_V1_RMSE"] = _test("V1_RMSE", "V2_RMSE")
    result["V3_vs_V2_RMSE"] = _test("V2_RMSE", "V3_RMSE")
    result["V3_vs_V1_RMSE"] = _test("V1_RMSE", "V3_RMSE")
    result["V2_vs_V1_sMAPE"] = _test("V1_sMAPE", "V2_sMAPE")
    result["V3_vs_V2_sMAPE"] = _test("V2_sMAPE", "V3_sMAPE")
    result["V3_vs_V1_sMAPE"] = _test("V1_sMAPE", "V3_sMAPE")
    return result


def cell_filename(identity: RunIdentity) -> str:
    return f"{identity.cell_id}.json"


def cell_path(cells_dir: Path, identity: RunIdentity) -> Path:
    return Path(cells_dir) / cell_filename(identity)


def jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingProxyType):
        return jsonable(dict(value))
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_cell_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(jsonable(dict(payload)), indent=2, default=str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
    return path


def classify_cell_artifact(
    path: Path,
    identity: RunIdentity,
    arm: RobustnessArmSpec,
) -> str:
    if not Path(path).is_file():
        return "MISSING"
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "MALFORMED"
    missing = [key for key in REQUIRED_CELL_FIELDS if key not in payload]
    if missing:
        return "MALFORMED"
    status = str(payload.get("status") or "").upper()
    if status == "FAILED":
        return "FAILED"
    if status != "OK":
        return "MALFORMED"
    try:
        if str(payload["arm"]).lower() != identity.arm:
            return "STALE"
        if str(payload["dataset"]) != identity.dataset:
            return "STALE"
        if str(payload["target_identity"]) != identity.target_identity:
            return "STALE"
        if int(payload["horizon"]) != int(identity.horizon):
            return "STALE"
        if int(payload["seed"]) != int(identity.seed):
            return "STALE"
        if format_lr(parse_lr(payload.get("learning_rate_label") or payload["learning_rate"])) != identity.selected_lr_label:
            return "STALE"
        if str(payload["metric_space"]) != ORIGINAL_SALES_SPACE:
            return "STALE"
        if str(payload["architecture_digest"]) != arm.architecture_digest:
            return "STALE"
        rmse = float(payload["rmse_original"])
        smape_value = float(payload["smape_original"])
        if not math.isfinite(rmse) or not math.isfinite(smape_value):
            return "MALFORMED"
    except (TypeError, ValueError, ScreeningContractError, KeyError):
        return "MALFORMED"
    return "OK"


def load_valid_ok_cells(
    cells_dir: Path,
    identities: Sequence[RunIdentity],
    registry: Mapping[str, RobustnessArmSpec],
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    completed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for identity in identities:
        path = cell_path(cells_dir, identity)
        status = classify_cell_artifact(path, identity, registry[identity.arm])
        if status != "OK":
            continue
        completed[identity.identity_tuple] = json.loads(path.read_text(encoding="utf-8"))
    return completed


def completion_guard(
    identities: Sequence[RunIdentity],
    cell_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected = len(identities)
    ok_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "OK"]
    failed_rows = [row for row in cell_rows if str(row.get("status") or "").upper() == "FAILED"]
    materialized = []
    for row in ok_rows:
        materialized.append(
            (
                str(row["arm"]).lower(),
                str(row["dataset"]),
                str(row["target_identity"]),
                int(row["horizon"]),
                int(row["seed"]),
                format_lr(parse_lr(row.get("learning_rate_label") or row["learning_rate"])),
            )
        )
    unique = set(materialized)
    expected_set = {item.identity_tuple for item in identities}
    missing = expected_set - unique
    extras = unique - expected_set
    duplicate = len(materialized) - len(unique)
    payload = {
        "EXPECTED_RUNS": expected,
        "OK_RUNS": len(unique),
        "FAILED_RUNS": len(failed_rows),
        "DUPLICATE_RUNS": int(duplicate),
        "MISSING_RUNS": len(missing),
        "EXTRA_RUNS": len(extras),
        "ROBUSTNESS_COMPLETE": (
            "YES"
            if expected == EXPECTED_RUNS
            and len(unique) == expected
            and not failed_rows
            and duplicate == 0
            and not missing
            and not extras
            else "NO"
        ),
    }
    return payload


def smoke_guard(cell_rows: Sequence[Mapping[str, Any]], paired_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    by_arm = {str(row.get("arm")).lower(): row for row in cell_rows if str(row.get("status")).upper() == "OK"}
    spaces = {str(row.get("metric_space")) for row in cell_rows}
    targets = {str(row.get("target_identity")) for row in cell_rows}
    return {
        "SMOKE_ARMS": str(len(by_arm)),
        "SMOKE_COMMON_TARGET": "YES" if targets == {"1/10"} else "NO",
        "SMOKE_V1_OK": "YES" if "v1" in by_arm else "NO",
        "SMOKE_V2_OK": "YES" if "v2" in by_arm else "NO",
        "SMOKE_V3_OK": "YES" if "v3" in by_arm else "NO",
        "SMOKE_PAIRED_ROW_COUNT": str(len(paired_rows)),
        "SMOKE_METRIC_SPACE": next(iter(spaces)) if len(spaces) == 1 else "MIXED",
    }
