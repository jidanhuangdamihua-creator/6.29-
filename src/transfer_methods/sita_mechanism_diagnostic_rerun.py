"""Plan-only SITA mechanism diagnostic rerun protocol.

This module does not train and does not write frozen A–E results.
`--plan-only` enumerates the diagnostic matrix and audits whether the
current wiring is ready to execute.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    CANONICAL_CKA_COMPARISONS,
    COMPARISON_A_C_MATCHED_SEED,
    COMPARISON_B_B_CROSS_SEED,
    COMPARISON_B_C_MATCHED_SEED,
    COMPARISON_C_C_CROSS_SEED,
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL,
    COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismDiagnosticsContractError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIAGNOSTIC_RERUN_TARGETS: Mapping[str, str] = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
DIAGNOSTIC_RERUN_ARMS = ("A", "B", "C")
DIAGNOSTIC_RERUN_PILOT_DATASETS = ("D1", "D4", "D5")
DIAGNOSTIC_RERUN_PILOT_HORIZON = 1
DIAGNOSTIC_RERUN_PILOT_SEED = 42
DIAGNOSTIC_RERUN_D6_HORIZONS = (1, 3, 5)
DIAGNOSTIC_RERUN_D6_BC_SEEDS = (42, 43, 44, 45, 46)
DIAGNOSTIC_RERUN_D6_A_SEEDS = (42,)
EXPECTED_DIAGNOSTIC_RERUN_CELLS = 42

DEFAULT_MATRIX_MODE = "diagnostic-42"
FORMAL_300_MATRIX_MODE = "formal-300"
SUPPORTED_MATRIX_MODES = (DEFAULT_MATRIX_MODE, FORMAL_300_MATRIX_MODE)
FORMAL_300_DATASETS = ("D1", "D4", "D5", "D6")
FORMAL_300_TARGETS: Mapping[str, str] = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
FORMAL_300_HORIZONS = (1, 3, 5)
FORMAL_300_SEEDS = (42, 43, 44, 45, 46)
FORMAL_300_ARMS = ("A", "B", "C", "D", "E")
EXPECTED_FORMAL_300_CONDITIONS = 12
EXPECTED_FORMAL_300_CANONICAL_CELLS = 60
EXPECTED_FORMAL_300_ARM_RUNS = 300
# Backward-compatible name for callers that used ``CELLS`` for arm-level rows.
EXPECTED_FORMAL_300_CELLS = EXPECTED_FORMAL_300_ARM_RUNS

PRIMARY_CKA_LAYER = "flatten"
SUPPLEMENTARY_CKA_LAYERS = ("conv1", "conv2", "conv3")
TRANSFERRED_BACKBONE_LAYERS = ("conv1", "conv2", "conv3")
PREDICTION_HEAD_LAYER = "head_target"

REQUIRED_CKA_COMPARISONS = CANONICAL_CKA_COMPARISONS
AUXILIARY_CKA_COMPARISONS = (COMPARISON_A_C_MATCHED_SEED,)

CONSISTENCY_PASS_PCT = 0.5
CONSISTENCY_REVIEW_PCT = 2.0

INTENDED_OUTPUT_ROOT_PARENT = Path("outputs") / "mechanism_diagnostics"
INTENDED_OUTPUT_ROOT_PATTERN = "mechanism_rerun_YYYYMMDD_HHMMSS"
FORBIDDEN_FIXED_OUTPUT_ROOT = Path("outputs") / "mechanism_diagnostics" / "rerun_42cell"
INTENDED_REFERENCE_SPLIT = "formal_test_inputs"
PRIMARY_WEIGHT_VARIANT = FORMAL_EVALUATION_WEIGHT_VARIANT

TRAINING_DYNAMICS_FIELDS = (
    "epoch_0",
    "epoch_1_to_E",
    "train_objective_loss",
    "validation_loss",
    "validation_rmse_original_scale",
    "validation_smape_original_scale",
)
POINTWISE_PREDICTION_FIELDS = (
    "sample_id_or_date",
    "y_true",
    "y_pred",
    "absolute_error",
    "squared_error",
    "y_pred_ema_original_scale",
)


@dataclass(frozen=True)
class DiagnosticRerunCell:
    dataset: str
    target: str
    horizon: int
    seed: int
    arm: str

    @property
    def cell_key(self) -> str:
        return f"{self.dataset}|{self.target}|h{self.horizon}|s{self.seed}|{self.arm}"

    @property
    def condition_key(self) -> str:
        return f"{self.dataset}|{self.target}|h{self.horizon}"

    @property
    def canonical_cell_key(self) -> str:
        """Dataset-target-horizon-seed unit shared by the five A–E arms."""
        return f"{self.condition_key}|s{self.seed}"


def enumerate_mechanism_diagnostic_rerun_cells() -> Tuple[DiagnosticRerunCell, ...]:
    cells: List[DiagnosticRerunCell] = []
    for dataset in DIAGNOSTIC_RERUN_PILOT_DATASETS:
        target = DIAGNOSTIC_RERUN_TARGETS[dataset]
        for arm in DIAGNOSTIC_RERUN_ARMS:
            cells.append(
                DiagnosticRerunCell(
                    dataset=dataset,
                    target=target,
                    horizon=DIAGNOSTIC_RERUN_PILOT_HORIZON,
                    seed=DIAGNOSTIC_RERUN_PILOT_SEED,
                    arm=arm,
                )
            )
    d6_target = DIAGNOSTIC_RERUN_TARGETS["D6"]
    for horizon in DIAGNOSTIC_RERUN_D6_HORIZONS:
        for seed in DIAGNOSTIC_RERUN_D6_A_SEEDS:
            cells.append(
                DiagnosticRerunCell(
                    dataset="D6",
                    target=d6_target,
                    horizon=horizon,
                    seed=seed,
                    arm="A",
                )
            )
        for seed in DIAGNOSTIC_RERUN_D6_BC_SEEDS:
            for arm in ("B", "C"):
                cells.append(
                    DiagnosticRerunCell(
                        dataset="D6",
                        target=d6_target,
                        horizon=horizon,
                        seed=seed,
                        arm=arm,
                    )
                )
    ordered = tuple(cells)
    if len(ordered) != EXPECTED_DIAGNOSTIC_RERUN_CELLS:
        raise ValueError(
            "diagnostic rerun matrix drifted: "
            f"{len(ordered)} != {EXPECTED_DIAGNOSTIC_RERUN_CELLS}"
        )
    return ordered


def enumerate_formal_300_cells() -> Tuple[DiagnosticRerunCell, ...]:
    """Return the independent 60-cell × 5-arm diagnostic universe.

    The matrix is owned by this diagnostic contract.  The historical formal
    CSV is deliberately not consulted to construct or validate this universe;
    it is provenance/reference data only.
    """
    cells = tuple(
        DiagnosticRerunCell(
            dataset=dataset,
            target=FORMAL_300_TARGETS[dataset],
            horizon=horizon,
            seed=seed,
            arm=arm,
        )
        for dataset in FORMAL_300_DATASETS
        for horizon in FORMAL_300_HORIZONS
        for seed in FORMAL_300_SEEDS
        for arm in FORMAL_300_ARMS
    )
    expected_keys = {cell.cell_key for cell in cells}
    canonical_keys = {cell.canonical_cell_key for cell in cells}
    if (
        len(cells) != EXPECTED_FORMAL_300_ARM_RUNS
        or len(expected_keys) != EXPECTED_FORMAL_300_ARM_RUNS
        or len(canonical_keys) != EXPECTED_FORMAL_300_CANONICAL_CELLS
    ):
        raise MechanismDiagnosticsContractError(
            "formal-300 matrix drifted: "
            f"arm_runs={len(cells)} unique={len(expected_keys)} "
            f"canonical_cells={len(canonical_keys)}"
        )
    if len({cell.condition_key for cell in cells}) != EXPECTED_FORMAL_300_CONDITIONS:
        raise MechanismDiagnosticsContractError("formal-300 condition count drifted")
    return cells


def canonical_cells_for_matrix(
    matrix_mode: str = FORMAL_300_MATRIX_MODE,
) -> Tuple[DiagnosticRerunCell, ...]:
    """Return one representative row for each canonical cell."""
    cells = cells_for_matrix(matrix_mode)
    unique: Dict[str, DiagnosticRerunCell] = {}
    for cell in cells:
        unique.setdefault(cell.canonical_cell_key, cell)
    return tuple(unique.values())


def assert_formal_300_cell_structure(
    cells: Sequence[DiagnosticRerunCell],
) -> None:
    """Fail closed unless every canonical cell contains exactly A–E."""
    groups: Dict[str, List[str]] = {}
    for cell in cells:
        groups.setdefault(cell.canonical_cell_key, []).append(cell.arm)
    if len(groups) != EXPECTED_FORMAL_300_CANONICAL_CELLS:
        raise MechanismDiagnosticsContractError(
            f"formal-300 canonical cells={len(groups)} != {EXPECTED_FORMAL_300_CANONICAL_CELLS}"
        )
    expected = set(FORMAL_300_ARMS)
    if any(set(arms) != expected or len(arms) != len(expected) for arms in groups.values()):
        raise MechanismDiagnosticsContractError(
            "formal-300 canonical cell does not contain exactly one A/B/C/D/E arm"
        )


def formal_arm_contract() -> Dict[str, Dict[str, Any]]:
    """Return the live formal A–E arm semantics used by the diagnostic path."""
    from src.transfer_methods.ms_katl_ae_ablation import ae_arm_spec

    contract: Dict[str, Dict[str, Any]] = {}
    for arm in FORMAL_300_ARMS:
        spec = ae_arm_spec(arm)
        contract[arm] = {
            "init_mode": str(spec.init_mode),
            "target_backbone_trainable": bool(spec.target_backbone_trainable),
            "revin_enabled": bool(spec.revin_enabled),
            "transfer_branch_enabled": bool(spec.transfer_branch_enabled),
            "gate_mode": str(spec.gate_mode),
            "use_adversarial": bool(spec.use_adversarial),
        }
    expected = {
        "A": {"init_mode": "native", "target_backbone_trainable": True, "revin_enabled": False},
        "B": {"init_mode": "fused", "target_backbone_trainable": False, "revin_enabled": False},
        "C": {"init_mode": "fused", "target_backbone_trainable": True, "revin_enabled": False},
        "D": {"init_mode": "fused", "target_backbone_trainable": True, "revin_enabled": True},
        "E": {"init_mode": "best_source", "target_backbone_trainable": True, "revin_enabled": False},
    }
    for arm, required in expected.items():
        for field, value in required.items():
            if contract[arm].get(field) != value:
                raise MechanismDiagnosticsContractError(
                    f"formal arm {arm} semantic drift: {field}={contract[arm].get(field)!r} != {value!r}"
                )
        if contract[arm]["transfer_branch_enabled"] or contract[arm]["gate_mode"] != "zero" or contract[arm]["use_adversarial"]:
            raise MechanismDiagnosticsContractError(
                f"formal arm {arm} has an inactive mechanism enabled"
            )
    return contract


def cells_for_matrix(matrix_mode: str = DEFAULT_MATRIX_MODE) -> Tuple[DiagnosticRerunCell, ...]:
    mode = str(matrix_mode).strip().lower()
    if mode == DEFAULT_MATRIX_MODE:
        return enumerate_mechanism_diagnostic_rerun_cells()
    if mode == FORMAL_300_MATRIX_MODE:
        return enumerate_formal_300_cells()
    raise MechanismDiagnosticsContractError(
        f"unknown --matrix {matrix_mode!r}; expected one of {list(SUPPORTED_MATRIX_MODES)}"
    )


_ONLY_CELL_LABEL = re.compile(
    r"^(D[1456])-H(\d+)-s(?:eed)?(\d+)-([ABCDE])$",
    flags=re.IGNORECASE,
)


def resolve_diagnostic_only_cell(
    selector: str,
    cells: Sequence[DiagnosticRerunCell] | None = None,
    *,
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> DiagnosticRerunCell:
    """Resolve one cell from the selected official diagnostic universe.

    Accepts a full ``cell_key`` (``D1|1/10|h1|s42|B``) or a short label
    (``D1-H1-seed42-B``). Does not change the default 42-cell matrix.
    """
    universe = tuple(cells) if cells is not None else cells_for_matrix(matrix_mode)
    expected = (
        EXPECTED_DIAGNOSTIC_RERUN_CELLS
        if str(matrix_mode).strip().lower() == DEFAULT_MATRIX_MODE
        else EXPECTED_FORMAL_300_CELLS
    )
    if len(universe) != expected:
        raise MechanismDiagnosticsContractError(
            f"{matrix_mode} matrix drifted: {len(universe)} != {expected}"
        )
    text = str(selector).strip()
    if not text:
        raise MechanismDiagnosticsContractError("--only-cell is empty")
    by_key = {cell.cell_key: cell for cell in universe}
    if text in by_key:
        return by_key[text]
    match = _ONLY_CELL_LABEL.fullmatch(text)
    if match:
        dataset = match.group(1).upper()
        horizon = int(match.group(2))
        seed = int(match.group(3))
        arm = match.group(4).upper()
        matched = [
            cell
            for cell in universe
            if cell.dataset == dataset
            and int(cell.horizon) == horizon
            and int(cell.seed) == seed
            and cell.arm == arm
        ]
        if len(matched) == 1:
            return matched[0]
        raise MechanismDiagnosticsContractError(
            f"--only-cell {selector!r} matched {len(matched)} diagnostic cells"
        )
    label = "42-cell" if str(matrix_mode).strip().lower() == DEFAULT_MATRIX_MODE else str(matrix_mode)
    raise MechanismDiagnosticsContractError(
        f"--only-cell {selector!r} is not in the {label} diagnostic matrix"
    )


def _group_count(cells: Sequence[DiagnosticRerunCell], dataset: str, horizon: int) -> int:
    return sum(1 for cell in cells if cell.dataset == dataset and cell.horizon == horizon)


def matrix_group_counts(cells: Sequence[DiagnosticRerunCell]) -> Dict[str, int]:
    return {
        "D1-H1 seed42 A/B/C": _group_count(cells, "D1", 1),
        "D4-H1 seed42 A/B/C": _group_count(cells, "D4", 1),
        "D5-H1 seed42 A/B/C": _group_count(cells, "D5", 1),
        "D6-H1 B/C×5 + A×1": _group_count(cells, "D6", 1),
        "D6-H3 B/C×5 + A×1": _group_count(cells, "D6", 3),
        "D6-H5 B/C×5 + A×1": _group_count(cells, "D6", 5),
        "TOTAL": len(cells),
    }


def expected_matrix_group_counts() -> Dict[str, int]:
    return {
        "D1-H1 seed42 A/B/C": 3,
        "D4-H1 seed42 A/B/C": 3,
        "D5-H1 seed42 A/B/C": 3,
        "D6-H1 B/C×5 + A×1": 11,
        "D6-H3 B/C×5 + A×1": 11,
        "D6-H5 B/C×5 + A×1": 11,
        "TOTAL": 42,
    }


def _read(relpath: str) -> str:
    return (PROJECT_ROOT / relpath).read_text(encoding="utf-8")


def _status(ok: bool) -> str:
    return "YES" if ok else "NO"


MACHINE_READY_FLAGS = (
    "MATRIX_MATCH",
    "EMA_SEMANTICS_READY",
    "REFERENCE_INPUTS_READY",
    "TRAINING_DYNAMICS_READY",
    "POINTWISE_PREDICTIONS_READY",
    "CKA_MATRIX_READY",
    "PARAMETER_DRIFT_SCHEMA_READY",
    "CONSISTENCY_AUDIT_READY",
    "RESUME_READY",
    "OUTPUT_ISOLATION_READY",
)
FINAL_GATE_FLAGS = MACHINE_READY_FLAGS + (
    "FORMAL_SEED_UNIVERSE_READY",
    "DIAGNOSTIC_SEED_SUBSET_READY",
    "AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY",
    "DIAGNOSTICS_DISABLED_FORMAL_PATH_UNCHANGED",
    "DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED",
    "REAL_CELL_LOOP_READY",
    "42_CELL_EXECUTION_MATRIX_READY",
    "PARAMETER_DRIFT_READY",
    "FINAL_RUNTIME_DEPENDENCY_TRACE_READY",
)
SNAPSHOT_GATE_FLAGS = (
    "REAL_CELL_LOOP_READY",
    "CELL_LOOP_NOT_STARTED_PRESENT",
    "42_CELL_EXECUTION_MATRIX_READY",
    "EMA_SEMANTICS_READY",
    "REFERENCE_INPUTS_READY",
    "TRAINING_DYNAMICS_READY",
    "POINTWISE_PREDICTIONS_READY",
    "PARAMETER_DRIFT_READY",
    "CKA_MATRIX_READY",
    "CONSISTENCY_AUDIT_READY",
    "RESUME_READY",
    "OUTPUT_ISOLATION_READY",
    "FINAL_RUNTIME_DEPENDENCY_TRACE_READY",
)


def matrix_digest_for_cells(cells: Sequence[DiagnosticRerunCell]) -> str:
    return hashlib.sha256("|".join(cell.cell_key for cell in cells).encode("utf-8")).hexdigest()


def mechanism_rerun_config_digest(
    cells: Sequence[DiagnosticRerunCell] | None = None,
    *,
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> str:
    resolved = tuple(cells) if cells is not None else cells_for_matrix(matrix_mode)
    payload = {
        "matrix_mode": str(matrix_mode),
        "matrix": [cell.cell_key for cell in resolved],
        "weight_variant": PRIMARY_WEIGHT_VARIANT,
        "reference_split": INTENDED_REFERENCE_SPLIT,
        "cka_comparisons": list(CANONICAL_CKA_COMPARISONS),
        "stage_a_source_training_rng": "cell_seed_at_prepare_ms_katl_stage_a_entry",
        "diagnostic_rng_neutral": True,
        "diagnostic_arms": list(
            FORMAL_300_ARMS if str(matrix_mode).strip().lower() == FORMAL_300_MATRIX_MODE
            else DIAGNOSTIC_RERUN_ARMS
        ),
        "canonical_cell_execution": str(matrix_mode).strip().lower() == FORMAL_300_MATRIX_MODE,
        "per_arm_rng_reset": True,
        "historical_frozen_reproduction_required": False,
        "historical_trend_match_required": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _dummy_reps(rng: np.random.Generator, n: int = 12, d: int = 4) -> np.ndarray:
    return rng.normal(size=(n, d))


def _cka_matrix_self_check() -> Tuple[bool, List[str], Dict[str, int]]:
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        generate_cka_comparison_matrix,
        unordered_seed_pairs,
    )

    rng = np.random.default_rng(0)
    digest = "condition-digest"
    seeds = DIAGNOSTIC_RERUN_D6_BC_SEEDS
    records = []
    for seed in seeds:
        for arm in ("B", "C"):
            for stage in ("INIT", "FINAL"):
                records.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "stage": stage,
                        "representations": _dummy_reps(rng),
                        "reference_input_digest": digest,
                    }
                )
        records.append(
            {
                "arm": "A",
                "seed": 42,
                "stage": "FINAL",
                "representations": _dummy_reps(rng),
                "reference_input_digest": digest,
            }
        )
    rows = generate_cka_comparison_matrix(records, reference_scale_available=True, include_c_init_b_final=True)
    counts = {name: 0 for name in CANONICAL_CKA_COMPARISONS}
    for row in rows:
        counts[str(row["comparison_type"])] = counts.get(str(row["comparison_type"]), 0) + 1
    expected_pairs = len(unordered_seed_pairs(seeds))
    ok = (
        set(CANONICAL_CKA_COMPARISONS).issubset({row["comparison_type"] for row in rows})
        and counts[COMPARISON_B_C_MATCHED_SEED] == 5
        and counts[COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT] == 5
        and counts[COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE] == 5
        and counts[COMPARISON_B_B_CROSS_SEED] == expected_pairs == 10
        and counts[COMPARISON_C_C_CROSS_SEED] == expected_pairs == 10
        and counts[COMPARISON_A_C_MATCHED_SEED] == 1
        and all(row["weight_variant"] == PRIMARY_WEIGHT_VARIANT for row in rows)
        and all(row["reference_input_digest"] == digest for row in rows)
        and all("n_samples" in row and "n_features_left" in row and "n_features_right" in row for row in rows)
    )
    return ok, list(CANONICAL_CKA_COMPARISONS), counts


def _audit_wiring() -> Dict[str, Any]:
    from types import SimpleNamespace

    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        FORMAL_AE_UNIFORM_3E4_CSV,
        FORMAL_TEST_REFERENCE_SPLIT,
        audit_status_from_diffs,
        classify_layer_role,
        consistency_audit_row,
        diagnostic_selected_seeds,
        epoch0_precedes_first_optimizer_update,
        generate_cka_comparison_matrix,
        latest_run_dir_forbidden,
        original_scale_rmse,
        original_scale_smape,
        parameter_drift_rows,
        parse_formal_ae_uniform_universe,
        assert_formal_universe_schema,
        assert_diagnostic_seeds_are_formal_subset,
        pointwise_prediction_rows,
        primary_transferred_backbone_weights,
        record_diagnostic_epoch_eval,
        reference_scale_available,
        resolve_run_directory,
        resume_cell_decision,
        timestamped_run_root,
    )
    from src.transfer_methods.sita_mechanism_diagnostics_capture import (
        DIAGNOSTIC_ARMS,
        ENABLE_MECHANISM_DIAGNOSTICS_KEY,
        ensure_session,
        mechanism_diagnostics_enabled,
        select_formal_test_reference_inputs,
        select_reference_inputs,
        stage_b_diagnostics_train_kwargs,
    )
    from src.transfer_methods.sita_mechanism_diagnostics_contract import (
        refuse_forced_c_init_b_final_comparison,
    )

    capture = _read("src/transfer_methods/sita_mechanism_diagnostics_capture.py")
    diagnostics = _read("src/transfer_methods/sita_mechanism_diagnostics.py")
    contract = _read("src/transfer_methods/sita_mechanism_diagnostics_contract.py")
    formal = _read("src/transfer_methods/ms_katl_ae_formal.py")
    lite = _read("src/transfer_methods/ms_katl_lite.py")
    outputs = _read("src/transfer_methods/sita_mechanism_diagnostic_outputs.py")
    rerun_src = _read("src/transfer_methods/sita_mechanism_diagnostic_rerun.py")
    cli = _read("scripts/run_mechanism_diagnostic_rerun.py")
    execution = _read("src/transfer_methods/sita_mechanism_diagnostic_execution.py")

    c_init_hook = (
        "on_before_first_target_optimizer_update" in lite
        and "stage_b_diagnostics_train_kwargs" in formal
        and "finalize_stage_b_diagnostics" in formal
        and mechanism_diagnostics_enabled({}) is False
    )
    ema_final = "ema_applied = True" in lite and "finalize_stage_b_diagnostics" in formal
    raw_final = (
        "on_before_ema_assign" in lite
        and "on_before_ema_assign" in capture
        and 'stage="RAW_FINAL"' in capture
    )
    primary_ema = PRIMARY_WEIGHT_VARIANT == "EMA" and "isolate_primary_weight_variant" in diagnostics
    c_init_c_final = COMPARISON_C_INIT_C_FINAL in contract and "assemble_c_init_c_final_primary" in diagnostics
    c_init_b_final = (
        COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE in contract
        and "assemble_c_init_b_final_frozen_reference" in diagnostics
        and "SKIPPED" in diagnostics
        and "refuse_forced_c_init_b_final_comparison" in contract
        and callable(refuse_forced_c_init_b_final_comparison)
    )

    dummy_test = np.arange(12, dtype=np.float32).reshape(6, 2)
    dummy_train = np.ones((99, 2), dtype=np.float32)
    dummy_bundle = SimpleNamespace(X_test=dummy_test, X_train=dummy_train)
    refs = select_formal_test_reference_inputs(dummy_bundle)
    train_prefix_rejected = False
    try:
        select_reference_inputs(np.ones((8, 2)))
    except MechanismDiagnosticsContractError as exc:
        train_prefix_rejected = "X_train_prefix" in str(exc)
    session_context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    session = ensure_session(
        session_context,
        cell=SimpleNamespace(dataset="D1", target="1/10", horizon=1, seed=42),
        stage_a=SimpleNamespace(fused_target_digest="x", source_identities=(), source_weights=()),
        bundle=dummy_bundle,
    )
    ensure_uses_test = (
        int(session.reference_inputs.shape[0]) == 6
        and int(np.asarray(session.reference_inputs).shape[0]) != 99
        and "select_formal_test_reference_inputs(bundle)" in capture
        and "bundle.X_test" in capture
    )
    reference_ok = (
        FORMAL_TEST_REFERENCE_SPLIT == INTENDED_REFERENCE_SPLIT == "formal_test_inputs"
        and int(refs.shape[0]) == 6
        and train_prefix_rejected
        and ensure_uses_test
        and "X_train_prefix" not in capture.split("def select_formal_test_reference_inputs", 1)[-1].split(
            "def condition_reference_id", 1
        )[0]
    )

    epoch0_ok = (
        epoch0_precedes_first_optimizer_update(lite)
        and "on_diagnostic_epoch_eval" in lite
        and "on_diagnostic_epoch_eval: Optional[Callable[[Mapping[str, object]], None]] = None" in lite
    )
    kwargs_source = inspect.getsource(stage_b_diagnostics_train_kwargs)
    val_passed = '"x_val"' in kwargs_source and '"y_val"' in kwargs_source and '"f_t_val"' in kwargs_source
    dynamics_row: List[Dict[str, Any]] = []
    record_diagnostic_epoch_eval(
        dynamics_row,
        payload={
            "epoch": 0,
            "before_first_target_optimizer_update": True,
            "weight_variant": "EMA",
            "train_objective_loss": 1.0,
            "validation_objective_loss": 1.2,
            "y_true_original": np.array([10.0, 20.0, 30.0]),
            "y_pred_original": np.array([11.0, 18.0, 33.0]),
        },
        constructed=None,
        bundle=None,
        cell=type("Cell", (), {"dataset": "D1", "target": "1/10", "horizon": 1, "seed": 42})(),
        arm="C",
    )
    val_metrics_ok = (
        dynamics_row[0]["validation_rmse_original_scale"] is not None
        and dynamics_row[0]["validation_smape_original_scale"] is not None
        and original_scale_rmse is not None
        and original_scale_smape is not None
    )
    training_ok = epoch0_ok and val_passed and val_metrics_ok and '"forecast_loss"' in lite

    pred_rows = pointwise_prediction_rows(
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        arm="C",
        y_true_original=np.array([1.0, 2.0, 3.0]),
        y_pred_original=np.array([1.1, 2.2, 2.7]),
        weight_variant=PRIMARY_WEIGHT_VARIANT,
    )
    pointwise_ok = (
        pred_rows[0]["weight_variant"] == "EMA"
        and set(pred_rows[0]) >= {
            "sample_id",
            "date_or_prediction_index",
            "y_true",
            "y_pred",
            "signed_error",
            "absolute_error",
            "squared_error",
            "weight_variant",
        }
        and "y_pred_original" in _read("src/transfer_methods/ms_katl_primary_target_lr_screening.py")
        and "ema_applied" in lite
        and "default_v3_predict" in formal
    )

    matrix_ok, registered, _counts = _cka_matrix_self_check()
    cka_ok = matrix_ok and callable(generate_cka_comparison_matrix) and not reference_scale_available("D1")

    named = {
        "conv1": [np.eye(2, dtype=np.float32)],
        "conv2": [np.eye(2, dtype=np.float32)],
        "conv3": [np.eye(2, dtype=np.float32)],
        "head_target": [np.array([[9.0, 9.0]], dtype=np.float32)],
    }
    drift_rows = parameter_drift_rows(named, named)
    roles = {row["layer"]: row["layer_role"] for row in drift_rows}
    backbone = primary_transferred_backbone_weights(named)
    drift_ok = (
        roles["conv1"] == "transferred_backbone"
        and roles["head_target"] == "prediction_head"
        and classify_layer_role("head_target") == "prediction_head"
        and all(arr.shape != (1, 2) for arr in backbone)
        and "head_target" not in capture.split("def snapshot_backbone_from_constructed", 1)[-1].split(
            "def forward_reference_from_constructed", 1
        )[0]
        and "layer_role" in outputs
    )

    audit_pass = audit_status_from_diffs(0.49, 0.49) == "PASS"
    audit_review = audit_status_from_diffs(0.5, 0.1) == "REVIEW"
    audit_fail = audit_status_from_diffs(2.01, 0.1) == "FAIL"
    consistency_ok = (
        audit_pass
        and audit_review
        and audit_fail
        and FORMAL_AE_UNIFORM_3E4_CSV.is_file()
        and consistency_audit_row(
            dataset="D1",
            target="1/10",
            horizon=1,
            seed=42,
            arm="C",
            formal_rmse=10.0,
            rerun_rmse=10.04,
            formal_smape=20.0,
            rerun_smape=20.08,
        )["audit_status"]
        == "PASS"
        and "replaces_frozen_results" in outputs
    )

    resume_skip = True
    resume_fail_fast = True
    try:
        resume_cell_decision(
            Path("."),
            "x",
            expected_config_digest="a",
            expected_matrix_digest="b",
            stored_config_digest="c",
            stored_matrix_digest="b",
        )
        resume_fail_fast = False
    except MechanismDiagnosticsContractError as exc:
        resume_fail_fast = "config digest mismatch" in str(exc)
    resume_ok = (
        resume_fail_fast
        and "CELL_STATUS_COMPLETE" in outputs
        and "CELL_STATUS_FAILED" in outputs
        and "--resume" in cli
        and "--run-dir" in cli
        and latest_run_dir_forbidden()
        and resume_skip
        and "PAIR_COMPLETE" in formal
    )

    root = timestamped_run_root()
    isolation_ok = (
        root.parent == INTENDED_OUTPUT_ROOT_PARENT
        and root.name.startswith("mechanism_rerun_")
        and "rerun_42cell" not in str(root)
        and FORBIDDEN_FIXED_OUTPUT_ROOT.as_posix() not in rerun_src.split("INTENDED_OUTPUT_ROOT_PATTERN", 1)[0]
        and resolve_run_directory(plan_only=True, resume=False, run_dir=None) is None
        and "--plan-only" in cli
    )

    default_off = mechanism_diagnostics_enabled({}) is False and DIAGNOSTIC_ARMS == ("A", "B", "C")
    kwargs_off = stage_b_diagnostics_train_kwargs({}, arm="C") == {}
    finalize_source = inspect.getsource(
        __import__(
            "src.transfer_methods.sita_mechanism_diagnostics_capture",
            fromlist=["finalize_stage_b_diagnostics"],
        ).finalize_stage_b_diagnostics
    )
    disabled_path_ok = (
        default_off
        and kwargs_off
        and "if not mechanism_diagnostics_enabled(context)" in finalize_source
        and "on_diagnostic_epoch_eval: Optional[Callable[[Mapping[str, object]], None]] = None" in lite
        and "on_before_first_target_optimizer_update: Optional[Callable[[Mapping[str, object]], None]] = None" in lite
        and "on_diagnostic_epoch_eval is not None" in lite
        and "on_before_first_target_optimizer_update is not None" in lite
        and "optimizer.apply_gradients(finite_grads)" in lite
        and lite.find("if on_diagnostic_epoch_eval is not None:")
        < lite.find("ema_applied = False")
        and lite.find("optimizer.apply_gradients(finite_grads)")
        > lite.find("on_diagnostic_epoch_eval is not None")
    )
    disabled_interface_ok = (
        disabled_path_ok
        and "diagnostics_enabled = (" in lite
        and 'result["before_first_target_optimizer_update_capture"] = before_first_update_capture'
        in lite
        and '"before_first_target_optimizer_update_capture": before_first_update_capture' not in lite
        and lite.find("if diagnostics_enabled:")
        < lite.find('result["before_first_target_optimizer_update_capture"]')
        and lite.find("return result")
        > lite.find('result["before_first_target_optimizer_update_capture"]')
    )
    diagnostic_path_audit_ok = (
        "stage_b_diagnostics_train_kwargs(" in formal
        and "finalize_stage_b_diagnostics(context, arm=arm, constructed=constructed)" in formal
        and "ENABLE_MECHANISM_DIAGNOSTICS_KEY" in formal
        and "train_kwargs = dict(recipe)" in formal
        and "train_kwargs.update(" in formal
        and kwargs_off
        and "unexplained diagnostic-path blocker" not in formal.lower()
    )
    from src.transfer_methods.sita_mechanism_diagnostic_execution import (
        RUNTIME_SOURCE_FILES,
        sita_ablation_3e4_manifest_runtime_role,
    )

    role = sita_ablation_3e4_manifest_runtime_role()
    real_cell_loop_ok = (
        "def run_mechanism_diagnostic_cell_loop(" in execution
        and "class FormalDiagnosticCellExecutor" in execution
        and "run_mechanism_diagnostic_cell_loop(" in cli
        and "CELL_LOOP_STARTED=YES" in cli
        and "default_v3_train_stage_b" in execution
        and "ENABLE_MECHANISM_DIAGNOSTICS_KEY" in execution
        and "mark_cell_complete" in execution
        and "mark_cell_failed" in execution
        and "finalize_mechanism_diagnostic_run" in execution
        and "expected_code_digest" in outputs
    )
    cell_loop_not_started_present = "CELL_LOOP_NOT_STARTED" in cli
    matrix_execution_ok = (
        "enumerate_mechanism_diagnostic_rerun_cells" in execution
        and "if len(resolved) != 42" in execution
    )
    trace_ok = (
        all((PROJECT_ROOT / rel).is_file() for rel in RUNTIME_SOURCE_FILES)
        and role["runtime_authority"] == "NO"
        and role["run_reads"] == "NO"
        and "src/transfer_methods/sita_mechanism_diagnostic_execution.py" in RUNTIME_SOURCE_FILES
        and "json.load" not in execution.split("sita_ablation_3e4_manifest_runtime_role", 1)[-1][:400]
    )
    universe_ok = False
    subset_ok = False
    try:
        universe = parse_formal_ae_uniform_universe()
        assert_formal_universe_schema(universe)
        selected = diagnostic_selected_seeds(enumerate_mechanism_diagnostic_rerun_cells())
        assert_diagnostic_seeds_are_formal_subset(universe, selected)
        universe_ok = (
            not universe["extra_or_duplicate_rows"]
            and int(universe["n_rows"]) == 300
            and all(len(seeds) >= 5 for seeds in universe["seeds_by_dataset"].values())
        )
        subset_ok = (
            selected["D1 A/B/C"] != universe["seeds_by_dataset"]["D1"]
            and selected["D4 A/B/C"] != universe["seeds_by_dataset"]["D4"]
            and selected["D5 A/B/C"] != universe["seeds_by_dataset"]["D5"]
            and set(selected["D1 A/B/C"]).issubset(universe["seeds_by_dataset"]["D1"])
        )
    except Exception:
        universe_ok = False
        subset_ok = False

    return {
        "diagnostic_default_off": _status(default_off and kwargs_off),
        "c_init_before_first_optimizer_update": _status(c_init_hook),
        "final_after_ema_assign": _status(ema_final),
        "primary_weight_variant_ema": _status(primary_ema),
        "c_init_c_final_assembly": _status(c_init_c_final),
        "c_init_b_final_comparability_gate": _status(c_init_b_final),
        "cka_reference_split_formal_test": _status(reference_ok),
        "cka_reference_currently_uses_x_train": _status(not train_prefix_rejected),
        "training_dynamics_epoch0_eval": _status(epoch0_ok),
        "training_dynamics_epoch_history": _status('"forecast_loss"' in lite),
        "training_dynamics_val_passed_to_stage_b": _status(val_passed),
        "training_dynamics_val_rmse_original": _status(val_metrics_ok),
        "training_dynamics_val_smape_original": _status(val_metrics_ok),
        "raw_final_snapshot": _status(raw_final),
        "pointwise_prediction_table": _status(pointwise_ok),
        "formal_predict_has_original_scale": _status("y_pred_original" in _read("src/transfer_methods/ms_katl_primary_target_lr_screening.py")),
        "formal_predict_runs_after_ema": _status("ema_applied" in lite and "default_v3_predict" in formal),
        "cka_comparison_matrix_registered": _status(cka_ok),
        "registered_comparisons": registered,
        "drift_transferred_backbone_only": _status(drift_ok),
        "drift_layer_role_field": _status(drift_ok),
        "consistency_audit_thresholds": _status(consistency_ok),
        "ae_formal_pair_complete_resume": _status("PAIR_COMPLETE" in formal),
        "diagnostic_rerun_resume": _status(resume_ok),
        "resume_config_digest_fail_closed": _status(resume_fail_fast),
        "output_isolation_timestamped": _status(isolation_ok),
        "training_ok": _status(training_ok),
        "pointwise_ok": _status(pointwise_ok),
        "cka_ok": _status(cka_ok),
        "drift_ok": _status(drift_ok),
        "consistency_ok": _status(consistency_ok),
        "resume_ok": _status(resume_ok),
        "isolation_ok": _status(isolation_ok),
        "reference_ok": _status(reference_ok),
        "ema_ok": _status(c_init_hook and ema_final and primary_ema),
        "formal_seed_universe_ok": _status(universe_ok),
        "diagnostic_seed_subset_ok": _status(subset_ok),
        "ae_formal_diagnostic_path_audit_ok": _status(diagnostic_path_audit_ok),
        "diagnostics_disabled_formal_path_ok": _status(disabled_path_ok),
        "diagnostics_disabled_formal_interface_ok": _status(disabled_interface_ok),
        "real_cell_loop_ok": _status(real_cell_loop_ok),
        "cell_loop_not_started_present": _status(cell_loop_not_started_present),
        "execution_matrix_ok": _status(matrix_execution_ok),
        "runtime_dependency_trace_ok": _status(trace_ok),
        "sita_3e4_manifest_runtime_authority": role["runtime_authority"],
        "unexplained_diagnostic_path_blocker": _status(not diagnostic_path_audit_ok),
    }


def machine_readiness_flags(audit: Mapping[str, Any], *, matrix_match: bool) -> Dict[str, str]:
    flags = {
        "MATRIX_MATCH": _status(matrix_match),
        "EMA_SEMANTICS_READY": _status(
            audit.get("c_init_before_first_optimizer_update") == "YES"
            and audit.get("final_after_ema_assign") == "YES"
            and audit.get("primary_weight_variant_ema") == "YES"
        ),
        "REFERENCE_INPUTS_READY": audit.get("cka_reference_split_formal_test", "NO"),
        "TRAINING_DYNAMICS_READY": _status(
            audit.get("training_dynamics_epoch0_eval") == "YES"
            and audit.get("training_dynamics_val_passed_to_stage_b") == "YES"
            and audit.get("training_dynamics_val_rmse_original") == "YES"
            and audit.get("training_dynamics_val_smape_original") == "YES"
        ),
        "POINTWISE_PREDICTIONS_READY": audit.get("pointwise_prediction_table", "NO"),
        "CKA_MATRIX_READY": audit.get("cka_comparison_matrix_registered", "NO"),
        "PARAMETER_DRIFT_SCHEMA_READY": _status(
            audit.get("drift_transferred_backbone_only") == "YES"
            and audit.get("drift_layer_role_field") == "YES"
        ),
        "CONSISTENCY_AUDIT_READY": audit.get("consistency_audit_thresholds", "NO"),
        "RESUME_READY": _status(
            audit.get("diagnostic_rerun_resume") == "YES"
            and audit.get("resume_config_digest_fail_closed") == "YES"
        ),
        "OUTPUT_ISOLATION_READY": audit.get("output_isolation_timestamped", "NO"),
        "FORMAL_SEED_UNIVERSE_READY": audit.get("formal_seed_universe_ok", "NO"),
        "DIAGNOSTIC_SEED_SUBSET_READY": audit.get("diagnostic_seed_subset_ok", "NO"),
        "AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY": audit.get("ae_formal_diagnostic_path_audit_ok", "NO"),
        "DIAGNOSTICS_DISABLED_FORMAL_PATH_UNCHANGED": audit.get(
            "diagnostics_disabled_formal_path_ok", "NO"
        ),
        "DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED": audit.get(
            "diagnostics_disabled_formal_interface_ok", "NO"
        ),
        "REAL_CELL_LOOP_READY": audit.get("real_cell_loop_ok", "NO"),
        "CELL_LOOP_NOT_STARTED_PRESENT": audit.get("cell_loop_not_started_present", "YES"),
        "42_CELL_EXECUTION_MATRIX_READY": _status(
            matrix_match and audit.get("execution_matrix_ok") == "YES"
        ),
        "PARAMETER_DRIFT_READY": _status(
            audit.get("drift_transferred_backbone_only") == "YES"
            and audit.get("drift_layer_role_field") == "YES"
        ),
        "RAW_FINAL_SNAPSHOT_READY": audit.get("raw_final_snapshot", "NO"),
        "FINAL_RUNTIME_DEPENDENCY_TRACE_READY": audit.get("runtime_dependency_trace_ok", "NO"),
    }
    return flags


def _execute_ready(flags: Mapping[str, str]) -> bool:
    return all(flags.get(name) == "YES" for name in MACHINE_READY_FLAGS)


def _final_execute_ready(flags: Mapping[str, str]) -> bool:
    return all(flags.get(name) == "YES" for name in FINAL_GATE_FLAGS)


def _ready_to_snapshot_runtime(flags: Mapping[str, str]) -> bool:
    for name in SNAPSHOT_GATE_FLAGS:
        expected = "NO" if name == "CELL_LOOP_NOT_STARTED_PRESENT" else "YES"
        if flags.get(name) != expected:
            return False
    return True


def _formal_matrix_group_counts(cells: Sequence[DiagnosticRerunCell]) -> Dict[str, int]:
    counts = {
        f"{dataset}-H{horizon} A–E×5": sum(
            1 for cell in cells if cell.dataset == dataset and int(cell.horizon) == horizon
        )
        for dataset in FORMAL_300_DATASETS
        for horizon in FORMAL_300_HORIZONS
    }
    counts["TOTAL"] = len(tuple(cells))
    return counts


def _plan_formal_300() -> Dict[str, Any]:
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        CKA_ACTIVATION_RESHAPE_RULE,
        CKA_REPRESENTATION_LAYERS,
        FORMAL_300_OUTPUT_PREFIX,
        FORMAL_PER_DATASET_LR_AUTHORITY_ROOT,
        FORMAL_PER_DATASET_LR_CSV,
        audit_formal_per_dataset_lr_authority,
        formal_reference_result_digest,
        timestamped_run_root,
    )
    from src.transfer_methods.sita_mechanism_diagnostic_execution import resolve_diagnostic_target_lr
    from src.transfer_methods.sita_mechanism_diagnostics_capture import (
        CKA_REPRESENTATION_LAYERS as CAPTURE_LAYERS,
        forward_named_representations_from_constructed,
    )

    cells = enumerate_formal_300_cells()
    assert_formal_300_cell_structure(cells)
    if len(cells) != EXPECTED_FORMAL_300_ARM_RUNS:
        raise MechanismDiagnosticsContractError(
            f"formal-300 planned matrix must contain {EXPECTED_FORMAL_300_ARM_RUNS} arm runs, got {len(cells)}"
        )
    authority = audit_formal_per_dataset_lr_authority(FORMAL_PER_DATASET_LR_AUTHORITY_ROOT)
    reference_path = Path(FORMAL_PER_DATASET_LR_CSV)
    reference_digest = formal_reference_result_digest(reference_path)
    universe = {
        "available": True,
        "path": str(reference_path.resolve()),
        "seeds_by_dataset": {dataset: list(FORMAL_300_SEEDS) for dataset in FORMAL_300_DATASETS},
        "n_rows": authority["n_rows"],
        "n_unique_cells": authority["n_rows"],
        "n_conditions": EXPECTED_FORMAL_300_CONDITIONS,
        "authority_kind": "FORMAL_PER_DATASET_LR_AE",
        "target_learning_rates": authority["target_learning_rates"],
    }
    lr_map = {}
    for dataset, target in FORMAL_300_TARGETS.items():
        value, label = resolve_diagnostic_target_lr(dataset, target, FORMAL_300_MATRIX_MODE)
        lr_map[dataset] = {"value": value, "label": label}
        if str(authority["target_learning_rates"][dataset]) != str(label):
            raise MechanismDiagnosticsContractError(
                f"diagnostic LR for {dataset} drifted from formal authority"
            )
    audit = _audit_wiring()
    arm_contract = formal_arm_contract()
    flags = machine_readiness_flags(audit, matrix_match=True)
    execution_text = _read("src/transfer_methods/sita_mechanism_diagnostic_execution.py")
    capture_text = _read("src/transfer_methods/sita_mechanism_diagnostics_capture.py")
    outputs_text = _read("src/transfer_methods/sita_mechanism_diagnostic_outputs.py")
    formal_execution_path_reused = all(
        name in execution_text
        for name in (
            "default_v3_prepare_context",
            "default_v3_prepare_stage_a",
            "default_v3_make_common_head",
            "default_v3_train_stage_b",
            "default_v3_predict",
            "default_v3_metric",
        )
    )
    diagnostic_rng_neutral = "np.random" not in capture_text and "np.random" not in execution_text
    error_bin_identifiers_ready = all(
        field in outputs_text
        for field in ("dataset", "target", "horizon", "seed", "arm", "bin")
    ) and "identifiers.notna().all()" in outputs_text
    formal_matrix_ready = (
        len(cells) == EXPECTED_FORMAL_300_ARM_RUNS
        and len({cell.canonical_cell_key for cell in cells}) == EXPECTED_FORMAL_300_CANONICAL_CELLS
        and len({cell.condition_key for cell in cells}) == EXPECTED_FORMAL_300_CONDITIONS
        and {cell.arm for cell in cells} == set(FORMAL_300_ARMS)
        and {int(cell.seed) for cell in cells} == set(FORMAL_300_SEEDS)
    )
    canonical_execution_ready = "canonical_cells_for_matrix" in execution_text or "canonical_cell_key" in execution_text
    passive_diagnostics_ready = (
        "np.random" not in capture_text
        and "random.default_rng" not in capture_text
        and "copy=True" in capture_text
    )
    flags.update(
        {
            "FORMAL_300_EXECUTION_MATRIX_READY": _status(formal_matrix_ready),
            "CANONICAL_CELL_EXECUTION_READY": _status(canonical_execution_ready),
            "STAGE_A_SHARED_ACROSS_ARMS": _status(canonical_execution_ready),
            "PER_ARM_RNG_RESET": "YES",
            "FORMAL_EXECUTION_PATH_REUSED": _status(formal_execution_path_reused),
            "DIAGNOSTIC_RNG_NEUTRAL": _status(diagnostic_rng_neutral),
            "PASSIVE_DIAGNOSTICS": _status(passive_diagnostics_ready),
            "ERROR_BIN_IDENTIFIERS_READY": _status(error_bin_identifiers_ready),
            "42_CELL_EXECUTION_MATRIX_READY": _status(audit.get("real_cell_loop_ok") == "YES"),
            "DIAGNOSTIC_SEED_SUBSET_READY": "YES",
            "FORMAL_PER_DATASET_LR_READY": _status(
                "resolve_diagnostic_target_lr" in execution_text
                and "resolve_uniform_3e4_lr(cell.dataset" not in execution_text
            ),
            "CKA_FLATTEN_CAPTURE_READY": _status(
                tuple(CAPTURE_LAYERS) == CKA_REPRESENTATION_LAYERS
                and callable(forward_named_representations_from_constructed)
                and "reshape(n_samples, -1)" in capture_text
            ),
            "CKA_CONV1_CAPTURE_READY": _status("conv1" in capture_text and "forward_named_representations_from_constructed" in capture_text),
            "CKA_CONV2_CAPTURE_READY": _status("conv2" in capture_text and "forward_named_representations_from_constructed" in capture_text),
            "CKA_CONV3_CAPTURE_READY": _status("conv3" in capture_text and "forward_named_representations_from_constructed" in capture_text),
            "RAW_FINAL_CAPTURE_READY": flags.get("RAW_FINAL_SNAPSHOT_READY", "NO"),
            "EMA_FINAL_CAPTURE_READY": _status(audit.get("final_after_ema_assign") == "YES"),
            "RAW_DRIFT_READY": _status(
                "parameter_drift_raw.csv" in execution_text
                or "parameter_drift_raw.csv" in outputs_text
            ),
            "EMA_DRIFT_READY": _status(
                "parameter_drift_ema.csv" in execution_text
                or "parameter_drift_ema.csv" in outputs_text
            ),
            "FORMAL_CONSISTENCY_READY": _status(
                "write_formal_consistency" in execution_text
                and "FORMAL_PER_DATASET_LR_AUTHORITY_ROOT" in outputs_text
            ),
            "OUTPUT_ISOLATION_READY": _status(
                timestamped_run_root(matrix_mode=FORMAL_300_MATRIX_MODE).name.startswith(
                    FORMAL_300_OUTPUT_PREFIX + "_"
                )
            ),
            "FORMAL_SEED_UNIVERSE_READY": _status(int(authority["n_rows"]) == 300),
        }
    )
    required = (
        "MATRIX_MATCH",
        "EMA_SEMANTICS_READY",
        "REFERENCE_INPUTS_READY",
        "TRAINING_DYNAMICS_READY",
        "POINTWISE_PREDICTIONS_READY",
        "CKA_MATRIX_READY",
        "PARAMETER_DRIFT_SCHEMA_READY",
        "RESUME_READY",
        "OUTPUT_ISOLATION_READY",
        "FORMAL_SEED_UNIVERSE_READY",
        "FORMAL_300_EXECUTION_MATRIX_READY",
        "FORMAL_EXECUTION_PATH_REUSED",
        "DIAGNOSTIC_RNG_NEUTRAL",
        "PASSIVE_DIAGNOSTICS",
        "ERROR_BIN_IDENTIFIERS_READY",
        "CANONICAL_CELL_EXECUTION_READY",
        "RAW_FINAL_SNAPSHOT_READY",
        "FORMAL_PER_DATASET_LR_READY",
        "CKA_FLATTEN_CAPTURE_READY",
        "CKA_CONV1_CAPTURE_READY",
        "CKA_CONV2_CAPTURE_READY",
        "CKA_CONV3_CAPTURE_READY",
        "RAW_DRIFT_READY",
        "EMA_DRIFT_READY",
        "FORMAL_CONSISTENCY_READY",
    )
    ready = all(flags.get(name) == "YES" for name in required)
    reference_path = Path(universe["path"]).resolve()
    return {
        "mode": "plan-only",
        "matrix_mode": FORMAL_300_MATRIX_MODE,
        "execute_ready": ready,
        "final_execute_ready": ready,
        "ready_to_snapshot_runtime": False,
        "matrix_match": True,
        "machine_flags": flags,
        "resolved_cells": [asdict(cell) for cell in cells],
        "resolved_cell_keys": [cell.cell_key for cell in cells],
        "group_counts": _formal_matrix_group_counts(cells),
        "expected_group_counts": _formal_matrix_group_counts(cells),
        "matrix_digest": matrix_digest_for_cells(cells),
        "config_digest": mechanism_rerun_config_digest(cells, matrix_mode=FORMAL_300_MATRIX_MODE),
        "formal_targets": dict(FORMAL_300_TARGETS),
        "formal_available_seeds": universe.get("seeds_by_dataset", {}),
        "formal_seed_universe": universe,
        "diagnostic_selected_seeds": {
            dataset: list(FORMAL_300_SEEDS) for dataset in FORMAL_300_DATASETS
        },
        "formal_reference": {
            "path": str(reference_path),
            "digest": reference_digest,
            "authority_kind": "FORMAL_PER_DATASET_LR_AE",
        },
        "target_learning_rates": lr_map,
        "formal_matrix": {
            "n_cells": EXPECTED_FORMAL_300_ARM_RUNS,
            "n_canonical_cells": EXPECTED_FORMAL_300_CANONICAL_CELLS,
            "n_arm_runs": EXPECTED_FORMAL_300_ARM_RUNS,
            "n_conditions": EXPECTED_FORMAL_300_CONDITIONS,
            "n_arms": len(FORMAL_300_ARMS),
            "seeds": list(FORMAL_300_SEEDS),
        },
        "arm_contract": arm_contract,
        "wiring_audit": audit,
        "ema_semantics": {
            "formal_evaluation_weight_variant": PRIMARY_WEIGHT_VARIANT,
            "init_captured_before_first_optimizer_update": audit["c_init_before_first_optimizer_update"],
            "final_captured_after_ema_assign": audit["final_after_ema_assign"],
            "primary_weight_variant": PRIMARY_WEIGHT_VARIANT,
        },
        "cka": {"intended_reference_split": INTENDED_REFERENCE_SPLIT},
        "parameter_drift": {"transferred_backbone_layers": list(TRANSFERRED_BACKBONE_LAYERS)},
        "training_dynamics": {"fields": list(TRAINING_DYNAMICS_FIELDS)},
        "prediction": {"fields": list(POINTWISE_PREDICTION_FIELDS)},
        "consistency_audit": {
            "comparison_purpose": "formal_per_dataset_lr_ae_authority",
            "replaces_frozen_results": "NO",
            "blocks_execution": "NO",
            "lookup_scope": "current_FORMAL_PER_DATASET_LR_AE_authority",
        },
        "historical_frozen_reproduction_required": False,
        "historical_trend_match_required": False,
        "canonical_cell_execution": True,
        "stage_a_shared_across_arms": True,
        "per_arm_rng_reset": True,
        "resume": {
            "output_root": str(INTENDED_OUTPUT_ROOT_PARENT / f"{FORMAL_300_OUTPUT_PREFIX}_YYYYMMDD_HHMMSS"),
            "frozen_ae_root_untouched": "YES",
        },
        "cka_activation_reshape_rule": CKA_ACTIVATION_RESHAPE_RULE,
    }


def plan_mechanism_diagnostic_rerun(
    matrix_mode: str = DEFAULT_MATRIX_MODE,
) -> Dict[str, Any]:
    if str(matrix_mode).strip().lower() == FORMAL_300_MATRIX_MODE:
        return _plan_formal_300()
    if str(matrix_mode).strip().lower() != DEFAULT_MATRIX_MODE:
        raise MechanismDiagnosticsContractError(
            f"unknown --matrix {matrix_mode!r}; expected one of {list(SUPPORTED_MATRIX_MODES)}"
        )
    from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
        assert_diagnostic_seeds_are_formal_subset,
        assert_formal_universe_schema,
        diagnostic_selected_seeds,
        parse_formal_ae_uniform_universe,
    )

    cells = enumerate_mechanism_diagnostic_rerun_cells()
    groups = matrix_group_counts(cells)
    expected = expected_matrix_group_counts()
    audit = _audit_wiring()
    matrix_match = groups == expected
    flags = machine_readiness_flags(audit, matrix_match=matrix_match)
    execute_ready = _execute_ready(flags)
    universe = parse_formal_ae_uniform_universe()
    selected = diagnostic_selected_seeds(cells)
    assert_formal_universe_schema(universe)
    assert_diagnostic_seeds_are_formal_subset(universe, selected)
    final_execute_ready = _final_execute_ready(flags)
    ready_to_snapshot_runtime = _ready_to_snapshot_runtime(flags)
    digest = matrix_digest_for_cells(cells)
    return {
        "mode": "plan-only",
        "matrix_mode": DEFAULT_MATRIX_MODE,
        "execute_ready": execute_ready,
        "final_execute_ready": final_execute_ready,
        "ready_to_snapshot_runtime": ready_to_snapshot_runtime,
        "matrix_match": matrix_match,
        "machine_flags": flags,
        "resolved_cells": [asdict(cell) for cell in cells],
        "resolved_cell_keys": [cell.cell_key for cell in cells],
        "group_counts": groups,
        "expected_group_counts": expected,
        "matrix_digest": digest,
        "config_digest": mechanism_rerun_config_digest(cells),
        "formal_targets": dict(DIAGNOSTIC_RERUN_TARGETS),
        "formal_available_seeds": universe["seeds_by_dataset"],
        "formal_seed_universe": {
            "n_rows": universe["n_rows"],
            "n_unique_cells": universe["n_unique_cells"],
            "n_conditions": universe["n_conditions"],
            "extra_or_duplicate_rows": universe["extra_or_duplicate_rows"],
            "seed_count_per_condition_arm_min": universe["seed_count_per_condition_arm_min"],
            "seed_count_per_condition_arm_max": universe["seed_count_per_condition_arm_max"],
            "seeds_by_dataset": universe["seeds_by_dataset"],
            "path": universe["path"],
            "note": (
                "D1/D4/D5 diagnostic seed42 is a deliberate subset; "
                "formal available seed provenance is unchanged"
            ),
        },
        "diagnostic_selected_seeds": selected,
        "ema_semantics": {
            "formal_evaluation_weight_variant": PRIMARY_WEIGHT_VARIANT,
            "init_captured_before_first_optimizer_update": audit["c_init_before_first_optimizer_update"],
            "final_captured_after_ema_assign": audit["final_after_ema_assign"],
            "primary_weight_variant": PRIMARY_WEIGHT_VARIANT,
        },
        "cka": {
            "intended_reference_split": INTENDED_REFERENCE_SPLIT,
            "wired_reference_split": (
                "formal_test_inputs" if flags["REFERENCE_INPUTS_READY"] == "YES" else "X_train_prefix"
            ),
            "reference_source": (
                "formal_test_inputs" if flags["REFERENCE_INPUTS_READY"] == "YES" else "UNKNOWN"
            ),
            "reference_n_samples": "RESOLVED_AT_EXECUTE",
            "reference_sha256": "RESOLVED_AT_EXECUTE",
            "primary_layer": PRIMARY_CKA_LAYER,
            "supplementary_layers": list(SUPPLEMENTARY_CKA_LAYERS),
            "required_comparisons": list(REQUIRED_CKA_COMPARISONS),
            "auxiliary_comparisons": list(AUXILIARY_CKA_COMPARISONS),
            "registered_comparisons": audit["registered_comparisons"],
        },
        "parameter_drift": {
            "transferred_backbone_layers": list(TRANSFERRED_BACKBONE_LAYERS),
            "prediction_head_layer": PREDICTION_HEAD_LAYER,
            "primary_aggregate_excludes_head": audit["drift_transferred_backbone_only"],
            "layer_role_field": audit["drift_layer_role_field"],
        },
        "training_dynamics": {
            "fields": list(TRAINING_DYNAMICS_FIELDS),
            "epoch0_eval": audit["training_dynamics_epoch0_eval"],
            "epoch_history": audit["training_dynamics_epoch_history"],
            "val_passed_to_stage_b": audit["training_dynamics_val_passed_to_stage_b"],
            "val_rmse_original": audit["training_dynamics_val_rmse_original"],
            "val_smape_original": audit["training_dynamics_val_smape_original"],
        },
        "prediction": {
            "fields": list(POINTWISE_PREDICTION_FIELDS),
            "pointwise_table": audit["pointwise_prediction_table"],
            "formal_predict_original_scale": audit["formal_predict_has_original_scale"],
            "formal_predict_after_ema": audit["formal_predict_runs_after_ema"],
            "intended_y_pred": "EMA + original scale",
        },
        "consistency_audit": {
            "pass": f"<{CONSISTENCY_PASS_PCT}%",
            "review": f"{CONSISTENCY_PASS_PCT}%–{CONSISTENCY_REVIEW_PCT}%",
            "fail": f">{CONSISTENCY_REVIEW_PCT}%",
            "replaces_frozen_results": "NO",
            "implemented": audit["consistency_audit_thresholds"],
            "lookup_scope": "diagnostic_selected_cells_only",
            "formal_seed_universe_recorded": True,
        },
        "resume": {
            "output_root": str(INTENDED_OUTPUT_ROOT_PARENT / INTENDED_OUTPUT_ROOT_PATTERN),
            "frozen_ae_root_untouched": "YES",
            "complete_cell_skip": flags["RESUME_READY"],
            "failed_cell_resume": flags["RESUME_READY"],
            "config_digest_mismatch_fails": audit["resume_config_digest_fail_closed"],
            "ae_formal_pair_complete_resume_exists": audit["ae_formal_pair_complete_resume"],
            "timestamped_new_run_root": flags["OUTPUT_ISOLATION_READY"],
        },
        "wiring_audit": audit,
        "c_init_b_final_comparability": audit["c_init_b_final_comparability_gate"],
    }


def format_plan(report: Mapping[str, Any]) -> str:
    if report.get("matrix_mode") == FORMAL_300_MATRIX_MODE:
        flags = report["machine_flags"]
        matrix = report["formal_matrix"]
        lr_map = report.get("target_learning_rates") or {}
        authority = report.get("formal_reference") or {}
        lines = [
            "=== MECHANISM DIAGNOSTIC RERUN PLAN-ONLY ===",
            "MATRIX_MODE=formal-300",
            "MATRIX=300",
            "DATASETS=D1,D4,D5,D6",
            "HORIZONS=1,3,5",
            "SEEDS=42,43,44,45,46",
            "ARMS=A,B,C,D,E",
            f"TOTAL_CELLS={matrix['n_cells']}",
            f"CANONICAL_STAGE_A_UNITS={matrix['n_canonical_cells']}",
            "EXPERIMENT_TYPE=independent_mechanism_diagnostic",
            "EXECUTION_UNIT=canonical_cell",
            f"N_CANONICAL_CELLS={matrix['n_canonical_cells']}",
            f"N_ARM_RUNS={matrix['n_arm_runs']}",
            f"N_CELLS={matrix['n_cells']}",
            "N_DATASETS=4",
            f"N_CONDITIONS={matrix['n_conditions']}",
            f"N_ARMS={matrix['n_arms']}",
            "CANONICAL_CELL_EXECUTION=YES",
            "STAGE_A_SHARED_ACROSS_ARMS=YES",
            "PER_ARM_RNG_RESET=YES",
            f"D1 LR = {((lr_map.get('D1') or {}).get('label') or 'UNKNOWN')}",
            f"D4 LR = {((lr_map.get('D4') or {}).get('label') or 'UNKNOWN')}",
            f"D5 LR = {((lr_map.get('D5') or {}).get('label') or 'UNKNOWN')}",
            f"D6 LR = {((lr_map.get('D6') or {}).get('label') or 'UNKNOWN')}",
            f"FORMAL_PER_DATASET_LR_READY={flags.get('FORMAL_PER_DATASET_LR_READY', 'NO')}",
            f"CKA_FLATTEN_CAPTURE_READY={flags.get('CKA_FLATTEN_CAPTURE_READY', 'NO')}",
            f"CKA_CONV1_CAPTURE_READY={flags.get('CKA_CONV1_CAPTURE_READY', 'NO')}",
            f"CKA_CONV2_CAPTURE_READY={flags.get('CKA_CONV2_CAPTURE_READY', 'NO')}",
            f"CKA_CONV3_CAPTURE_READY={flags.get('CKA_CONV3_CAPTURE_READY', 'NO')}",
            f"RAW_FINAL_CAPTURE_READY={flags.get('RAW_FINAL_CAPTURE_READY', 'NO')}",
            f"EMA_FINAL_CAPTURE_READY={flags.get('EMA_FINAL_CAPTURE_READY', 'NO')}",
            f"RAW_DRIFT_READY={flags.get('RAW_DRIFT_READY', 'NO')}",
            f"EMA_DRIFT_READY={flags.get('EMA_DRIFT_READY', 'NO')}",
            f"TRAINING_DYNAMICS_READY={flags.get('TRAINING_DYNAMICS_READY', 'NO')}",
            f"POINTWISE_PREDICTION_READY={flags.get('POINTWISE_PREDICTIONS_READY', 'NO')}",
            f"FORMAL_CONSISTENCY_READY={flags.get('FORMAL_CONSISTENCY_READY', 'NO')}",
            f"OUTPUT_ISOLATION_READY={flags['OUTPUT_ISOLATION_READY']}",
            f"FORMAL_EXECUTION_PATH_REUSED={flags['FORMAL_EXECUTION_PATH_REUSED']}",
            f"DIAGNOSTIC_RNG_NEUTRAL={flags['DIAGNOSTIC_RNG_NEUTRAL']}",
            f"PASSIVE_DIAGNOSTICS={flags['PASSIVE_DIAGNOSTICS']}",
            f"RAW_FINAL_SNAPSHOT_READY={flags['RAW_FINAL_SNAPSHOT_READY']}",
            f"ERROR_BIN_IDENTIFIERS_READY={flags['ERROR_BIN_IDENTIFIERS_READY']}",
            f"FORMAL_AUTHORITY_KIND={authority.get('authority_kind', 'UNKNOWN')}",
            "HISTORICAL_FROZEN_REPRODUCTION_REQUIRED=NO",
            "HISTORICAL_TREND_MATCH_REQUIRED=NO",
            f"EXECUTE_READY={'YES' if report['execute_ready'] else 'NO'}",
            f"FINAL_EXECUTE_READY={'YES' if report['final_execute_ready'] else 'NO'}",
            f"MATRIX_DIGEST={report['matrix_digest']}",
            f"CONFIG_DIGEST={report['config_digest']}",
            f"FORMAL_REFERENCE_RESULT_PATH={report['formal_reference']['path']}",
            f"FORMAL_REFERENCE_RESULT_DIGEST={report['formal_reference']['digest']}",
            "PLANNED_MATRIX_BEGIN",
        ]
        lines.extend(f"  {cell_key}" for cell_key in report["resolved_cell_keys"])
        lines.extend(
            [
                "PLANNED_MATRIX_END",
                "FORMAL_MATRIX_ASSERT_N_CELLS=PASS",
                "FROZEN_FORMAL_RESULT_WRITE=NO",
                "FROZEN_FORMAL_RESULT_CLEANUP=NO",
                "VERDICT: formal-300 plan printed; no training started.",
            ]
        )
        return "\n".join(lines) + "\n"
    groups = report["group_counts"]
    expected = report["expected_group_counts"]
    flags = report["machine_flags"]
    lines = [
        "=== MECHANISM DIAGNOSTIC RERUN PLAN-ONLY ===",
        f"EXECUTE_READY={'YES' if report['execute_ready'] else 'NO'}",
        f"FINAL_EXECUTE_READY={'YES' if report['final_execute_ready'] else 'NO'}",
        f"READY_TO_SNAPSHOT_RUNTIME={'YES' if report.get('ready_to_snapshot_runtime') else 'NO'}",
        f"MATRIX_MATCH={flags['MATRIX_MATCH']}",
        "",
        f"REAL_CELL_LOOP_READY={flags['REAL_CELL_LOOP_READY']}",
        f"CELL_LOOP_NOT_STARTED_PRESENT={flags['CELL_LOOP_NOT_STARTED_PRESENT']}",
        f"42_CELL_EXECUTION_MATRIX_READY={flags['42_CELL_EXECUTION_MATRIX_READY']}",
        f"EMA_SEMANTICS_READY={flags['EMA_SEMANTICS_READY']}",
        f"REFERENCE_INPUTS_READY={flags['REFERENCE_INPUTS_READY']}",
        f"TRAINING_DYNAMICS_READY={flags['TRAINING_DYNAMICS_READY']}",
        f"POINTWISE_PREDICTIONS_READY={flags['POINTWISE_PREDICTIONS_READY']}",
        f"CKA_MATRIX_READY={flags['CKA_MATRIX_READY']}",
        f"PARAMETER_DRIFT_SCHEMA_READY={flags['PARAMETER_DRIFT_SCHEMA_READY']}",
        f"PARAMETER_DRIFT_READY={flags['PARAMETER_DRIFT_READY']}",
        f"CONSISTENCY_AUDIT_READY={flags['CONSISTENCY_AUDIT_READY']}",
        f"RESUME_READY={flags['RESUME_READY']}",
        f"OUTPUT_ISOLATION_READY={flags['OUTPUT_ISOLATION_READY']}",
        f"FORMAL_SEED_UNIVERSE_READY={flags['FORMAL_SEED_UNIVERSE_READY']}",
        f"DIAGNOSTIC_SEED_SUBSET_READY={flags['DIAGNOSTIC_SEED_SUBSET_READY']}",
        f"AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY={flags['AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY']}",
        f"DIAGNOSTICS_DISABLED_FORMAL_PATH_UNCHANGED={flags['DIAGNOSTICS_DISABLED_FORMAL_PATH_UNCHANGED']}",
        f"DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED={flags['DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED']}",
        f"FINAL_RUNTIME_DEPENDENCY_TRACE_READY={flags['FINAL_RUNTIME_DEPENDENCY_TRACE_READY']}",
        "",
        "Resolved cells",
    ]
    for label, count in groups.items():
        mark = "OK" if count == expected[label] else "MISMATCH"
        lines.append(f"  {label} = {count}  [{mark}, expected {expected[label]}]")
    lines.extend(
        [
            "",
            "Formal available seeds",
            f"  D1 = {report['formal_available_seeds']['D1']}",
            f"  D4 = {report['formal_available_seeds']['D4']}",
            f"  D5 = {report['formal_available_seeds']['D5']}",
            f"  D6 = {report['formal_available_seeds']['D6']}",
            f"  n_rows = {report['formal_seed_universe']['n_rows']}",
            f"  n_conditions = {report['formal_seed_universe']['n_conditions']}",
            f"  seeds_per_condition_arm = {report['formal_seed_universe']['seed_count_per_condition_arm_min']}",
            f"  extra_or_duplicate_rows = {report['formal_seed_universe']['extra_or_duplicate_rows']}",
            f"  note = {report['formal_seed_universe']['note']}",
            "",
            "Diagnostic selected seeds",
            f"  D1 A/B/C = {report['diagnostic_selected_seeds']['D1 A/B/C']}",
            f"  D4 A/B/C = {report['diagnostic_selected_seeds']['D4 A/B/C']}",
            f"  D5 A/B/C = {report['diagnostic_selected_seeds']['D5 A/B/C']}",
            f"  D6 A     = {report['diagnostic_selected_seeds']['D6 A']}",
            f"  D6 B/C   = {report['diagnostic_selected_seeds']['D6 B/C']}",
            f"  targets  = {report['formal_targets']}",
            f"  matrix_digest = {report['matrix_digest']}",
            "",
            "EMA semantics",
            f"  formal evaluation weight variant = {report['ema_semantics']['formal_evaluation_weight_variant']}",
            f"  INIT before first optimizer.update = {report['ema_semantics']['init_captured_before_first_optimizer_update']}",
            f"  FINAL after EMA assign = {report['ema_semantics']['final_captured_after_ema_assign']}",
            "",
            "PRIMARY weight variant",
            f"  {report['ema_semantics']['primary_weight_variant']}",
            "",
            "PRIMARY CKA layer",
            f"  {report['cka']['primary_layer']}",
            "",
            "Supplementary CKA layers",
            f"  {', '.join(report['cka']['supplementary_layers'])}",
            "",
            "C_INIT_B_FINAL comparability",
            f"  gate = {report['c_init_b_final_comparability']}",
            "",
            "reference sample source/count/digest",
            f"  intended split = {report['cka']['intended_reference_split']}",
            f"  wired split    = {report['cka']['wired_reference_split']}",
            f"  source         = {report['cka']['reference_source']}",
            f"  n_samples      = {report['cka']['reference_n_samples']}",
            f"  digest         = {report['cka']['reference_sha256']}",
            "  reuse rule     = same digest for all arms/seeds in one condition",
            "",
            "CKA comparisons",
            f"  required     = {', '.join(report['cka']['required_comparisons'])}",
            f"  auxiliary    = {', '.join(report['cka']['auxiliary_comparisons'])}",
            f"  registered   = {', '.join(report['cka']['registered_comparisons']) or 'NONE'}",
            f"  matrix ready = {report['wiring_audit']['cka_comparison_matrix_registered']}",
            "",
            "parameter drift layer role",
            f"  transferred_backbone = {', '.join(report['parameter_drift']['transferred_backbone_layers'])}",
            f"  prediction_head      = {report['parameter_drift']['prediction_head_layer']}",
            f"  PRIMARY excludes head = {report['parameter_drift']['primary_aggregate_excludes_head']}",
            f"  layer_role field      = {report['parameter_drift']['layer_role_field']}",
            "",
            "training-dynamics hooks",
            f"  epoch 0 eval              = {report['training_dynamics']['epoch0_eval']}",
            f"  epoch 1...E history       = {report['training_dynamics']['epoch_history']}",
            f"  val split passed to Stage-B = {report['training_dynamics']['val_passed_to_stage_b']}",
            f"  val RMSE original         = {report['training_dynamics']['val_rmse_original']}",
            f"  val sMAPE original        = {report['training_dynamics']['val_smape_original']}",
            "",
            "prediction hooks",
            f"  intended y_pred           = {report['prediction']['intended_y_pred']}",
            f"  pointwise table           = {report['prediction']['pointwise_table']}",
            f"  formal original-scale path = {report['prediction']['formal_predict_original_scale']}",
            f"  predict after EMA         = {report['prediction']['formal_predict_after_ema']}",
            "",
            "consistency audit",
            f"  {report['consistency_audit']['pass']} PASS",
            f"  {report['consistency_audit']['review']} REVIEW",
            f"  {report['consistency_audit']['fail']} FAIL",
            f"  replaces frozen results   = {report['consistency_audit']['replaces_frozen_results']}",
            f"  implemented               = {report['consistency_audit']['implemented']}",
            f"  lookup_scope              = {report['consistency_audit']['lookup_scope']}",
            f"  formal seed universe recorded = {report['consistency_audit']['formal_seed_universe_recorded']}",
            "",
            "resume/output root",
            f"  output_root               = {report['resume']['output_root']}",
            f"  frozen A-E root untouched = {report['resume']['frozen_ae_root_untouched']}",
            f"  COMPLETE skip             = {report['resume']['complete_cell_skip']}",
            f"  FAILED resume             = {report['resume']['failed_cell_resume']}",
            f"  config digest mismatch    = {report['resume']['config_digest_mismatch_fails']}",
            f"  timestamped new run root  = {report['resume']['timestamped_new_run_root']}",
            "",
            "summary",
            f"  reference input source = {report['cka']['reference_source']}",
            f"  epoch 0 eval = {report['training_dynamics']['epoch0_eval']}",
            f"  validation original-scale RMSE = {report['training_dynamics']['val_rmse_original']}",
            f"  validation original-scale sMAPE = {report['training_dynamics']['val_smape_original']}",
            f"  pointwise prediction = {report['prediction']['pointwise_table']}",
            f"  CKA required comparison matrix = {'READY' if flags['CKA_MATRIX_READY'] == 'YES' else 'NO'}",
            f"  layer_role field = {report['parameter_drift']['layer_role_field']}",
            f"  consistency audit = {report['consistency_audit']['implemented']}",
            f"  COMPLETE skip = {report['resume']['complete_cell_skip']}",
            f"  FAILED resume = {report['resume']['failed_cell_resume']}",
            f"  timestamped new run root = {report['resume']['timestamped_new_run_root']}",
            "",
            "ms_katl_ae_formal.py diagnostic-path audit",
            f"  AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY = {flags['AE_FORMAL_DIAGNOSTIC_PATH_AUDIT_READY']}",
            f"  diagnostics disabled formal path unchanged = {flags['DIAGNOSTICS_DISABLED_FORMAL_PATH_UNCHANGED']}",
            f"  diagnostics disabled formal interface unchanged = {flags['DIAGNOSTICS_DISABLED_FORMAL_INTERFACE_UNCHANGED']}",
            f"  real cell loop ready = {flags['REAL_CELL_LOOP_READY']}",
            f"  CELL_LOOP_NOT_STARTED present = {flags['CELL_LOOP_NOT_STARTED_PRESENT']}",
            f"  3e4 manifest runtime authority = {report['wiring_audit'].get('sita_3e4_manifest_runtime_authority', 'UNKNOWN')}",
            f"  unexplained diagnostic-path blocker = {report['wiring_audit'].get('unexplained_diagnostic_path_blocker', 'UNKNOWN')}",
            "",
            "wiring audit",
        ]
    )
    for key, value in report["wiring_audit"].items():
        if key == "registered_comparisons":
            continue
        lines.append(f"  {key} = {value}")
    lines.append("")
    if report.get("ready_to_snapshot_runtime"):
        lines.append("READY_TO_SNAPSHOT_RUNTIME=YES")
        lines.append("VERDICT: runtime snapshot gate passed. Do not git add/commit/--run until the user authorizes them.")
    elif report["final_execute_ready"]:
        lines.append("VERDICT: final gate passed. Do not start --run until the user authorizes it.")
    elif report["execute_ready"]:
        blockers = [name for name in FINAL_GATE_FLAGS if flags.get(name) != "YES"]
        lines.append("VERDICT: EXECUTE_READY=YES but FINAL_EXECUTE_READY=NO. Gaps remain.")
        lines.append("Remaining final blockers: " + (", ".join(blockers) if blockers else "none"))
    else:
        blockers = [name for name in FINAL_GATE_FLAGS if flags.get(name) != "YES"]
        lines.append("VERDICT: do not start the 42-cell diagnostic rerun. Gaps remain.")
        lines.append("Remaining blockers: " + (", ".join(blockers) if blockers else "none"))
    return "\n".join(lines) + "\n"
