"""Contract tests for Arm A vs Arm C Stage-B LR sensitivity screening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    SCENARIO,
    V3_LR_MAP,
    ae_arm_spec,
    canonical_targets,
    load_ae_manifest,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_sita_ac_lr_sensitivity import (
    EXPECTED_ARMS,
    EXPECTED_DATASETS,
    EXPECTED_HORIZONS,
    EXPECTED_LEARNING_RATES,
    EXPECTED_SEEDS,
    EXPECTED_TOTAL_RUNS,
    FORMAL_ABLATION_RUNNER,
    FORMAL_CONFIG,
    RUNS_PER_DATASET,
    RUNS_PER_DATASET_HORIZON,
    SCREENING_ARMS,
    SCREENING_HORIZONS,
    SCREENING_LR_LABELS,
    SCREENING_SEEDS,
    UNLOCK_ENV,
    LRSensitivityContractError,
    LRSensitivityTrainingBlocked,
    assert_execute_unlocked,
    assert_formal_ae_identity_frozen,
    assert_output_isolated,
    default_output_dir,
    enumerate_lr_sensitivity_runs,
    frozen_target_mapping,
    run_lr_sensitivity_dry_run,
    select_lr_by_validation,
    summarize_fixed_lr_comparison,
    summarize_validation_selected,
    validate_lr_sensitivity_matrix,
)

WRAPPER = PROJECT_ROOT / "scripts" / "run_ms_katl_sita_ac_lr_sensitivity.py"
FORMAL_RUNNER = PROJECT_ROOT / "scripts" / "run_ms_katl_ae_ablation.py"
CONTRACT = PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_sita_ac_lr_sensitivity.py"


def test_wrapper_does_not_change_formal_ae_identity():
    flags = assert_formal_ae_identity_frozen(PROJECT_ROOT)
    assert flags["FORMAL_ABLATION_RUNNER"] == FORMAL_ABLATION_RUNNER == "scripts/run_ms_katl_ae_ablation.py"
    assert flags["FORMAL_CONFIG"] == FORMAL_CONFIG == "FUSED_TARGET_ONLY_V3"
    assert flags["parent_v3_architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert FORMAL_METHODS == (
        "No-TL",
        "SS-TL",
        "MSWA-TL",
        "MSSB-TL",
        "MSML-TL",
        "MSML-TL-RFE",
    )
    spec_a = ae_arm_spec("A")
    spec_c = ae_arm_spec("C")
    assert spec_a.name == AE_ARM_NAMES["A"] == "TARGET_NATIVE_INIT"
    assert spec_a.init_mode == "native"
    assert spec_a.target_backbone_trainable is True
    assert spec_a.revin_enabled is False
    assert spec_c.name == AE_ARM_NAMES["C"] == "FUSED_TRAINABLE"
    assert spec_c.init_mode == "fused"
    assert spec_c.target_backbone_trainable is True
    assert spec_c.revin_enabled is False
    assert tuple(AE_ARMS) == ("A", "B", "C", "D", "E")


def test_target_mapping_matches_frozen_ae_manifest():
    mapping = frozen_target_mapping(PROJECT_ROOT)
    manifest = load_ae_manifest(PROJECT_ROOT)
    assert mapping == canonical_targets()
    assert mapping == tuple(V3_LR_MAP.keys())
    assert mapping == tuple(
        (item["dataset"], item["target_identity"]) for item in manifest["targets"]
    )
    assert mapping == (
        ("D1", "1/10"),
        ("D2", "1/10"),
        ("D3", "10"),
        ("D4", "166/311"),
        ("D5", "48/1159414"),
        ("D6", "CA_1/FOODS_3_080"),
    )
    assert V3_LR_MAP[("D5", "48/1159414")] == "3e-4"
    assert V3_LR_MAP[("D4", "166/311")] == "3e-5"


def test_108_run_matrix_is_unique_and_complete():
    identities = enumerate_lr_sensitivity_runs(PROJECT_ROOT)
    report = validate_lr_sensitivity_matrix(identities, project_root=PROJECT_ROOT)
    assert report["DATASETS"] == EXPECTED_DATASETS == 6
    assert report["HORIZONS"] == EXPECTED_HORIZONS == 3
    assert report["SEEDS"] == EXPECTED_SEEDS == 1
    assert report["ARMS"] == EXPECTED_ARMS == 2
    assert report["LEARNING_RATES"] == EXPECTED_LEARNING_RATES == 3
    assert report["EXPECTED_TOTAL_RUNS"] == EXPECTED_TOTAL_RUNS == 108
    assert report["DRY_RUN_TOTAL_RUNS"] == 108
    assert report["UNIQUE_RUN_KEYS"] == 108
    assert report["UNIQUE_STAGE_A_KEYS"] == 18
    assert report["FORBIDDEN_ARMS_PRESENT"] == "NO"
    assert {item.arm for item in identities} == set(SCREENING_ARMS) == {"A", "C"}
    assert {item.horizon for item in identities} == set(SCREENING_HORIZONS) == {1, 3, 5}
    assert {item.seed for item in identities} == set(SCREENING_SEEDS) == {42}
    assert {item.learning_rate_label for item in identities} == set(SCREENING_LR_LABELS)
    assert {item.scenario for item in identities} == {SCENARIO} == {"without"}
    for dataset in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert report["RUNS_PER_DATASET"][dataset] == RUNS_PER_DATASET == 18
        for horizon in (1, 3, 5):
            assert report["RUNS_PER_DATASET_HORIZON"][f"{dataset}_h{horizon}"] == RUNS_PER_DATASET_HORIZON == 6


def test_output_isolation_refuses_formal_ae_roots(tmp_path):
    isolated = assert_output_isolated(default_output_dir(PROJECT_ROOT), project_root=PROJECT_ROOT)
    assert isolated.name == "sita_ablation_lr_sensitivity_ac_seed42"
    with pytest.raises(LRSensitivityContractError):
        assert_output_isolated(
            PROJECT_ROOT / "outputs" / "runs" / "x_MS_KATL_AE_CORE_ABLATION",
            project_root=PROJECT_ROOT,
        )
    with pytest.raises(LRSensitivityContractError):
        assert_output_isolated(PROJECT_ROOT / "outputs" / "runs" / "probe", project_root=PROJECT_ROOT)
    safe = assert_output_isolated(tmp_path / "sita_ablation_lr_sensitivity_ac_seed42", project_root=PROJECT_ROOT)
    assert safe.name == "sita_ablation_lr_sensitivity_ac_seed42"


def test_execute_remains_locked_without_unlock(monkeypatch):
    monkeypatch.delenv(UNLOCK_ENV, raising=False)
    with pytest.raises(LRSensitivityTrainingBlocked):
        assert_execute_unlocked({})
    assert_execute_unlocked({UNLOCK_ENV: "YES"})


def test_validation_lr_selection_ignores_test_rmse():
    rows = [
        {
            "run_status": "OK",
            "learning_rate_label": "3e-5",
            "validation_rmse_original": 2.0,
            "rmse_original": 0.1,
            "smape_original": 1.0,
        },
        {
            "run_status": "OK",
            "learning_rate_label": "1e-4",
            "validation_rmse_original": 1.0,
            "rmse_original": 9.0,
            "smape_original": 9.0,
        },
        {
            "run_status": "OK",
            "learning_rate_label": "3e-4",
            "validation_rmse_original": 1.5,
            "rmse_original": 0.2,
            "smape_original": 0.2,
        },
    ]
    chosen = select_lr_by_validation(rows)
    assert chosen["best_validation_lr"] == "1e-4"
    assert chosen["rmse_original"] == 9.0


def test_fixed_lr_comparison_uses_same_lr_for_a_and_c():
    rows = []
    for lr in SCREENING_LR_LABELS:
        rows.append(
            {
                "dataset": "D1",
                "target": "1/10",
                "horizon": 1,
                "arm": "A",
                "learning_rate_label": lr,
                "run_status": "OK",
                "rmse_original": 2.0,
                "smape_original": 4.0,
            }
        )
        rows.append(
            {
                "dataset": "D1",
                "target": "1/10",
                "horizon": 1,
                "arm": "C",
                "learning_rate_label": lr,
                "run_status": "OK",
                "rmse_original": 1.0,
                "smape_original": 2.0,
            }
        )
    summaries = summarize_fixed_lr_comparison(rows)
    assert [item["learning_rate"] for item in summaries] == list(SCREENING_LR_LABELS)
    for item in summaries:
        assert item["c_wins"] == 1
        assert item["a_wins"] == 0
        assert item["ties"] == 0
        assert item["mean_relative_rmse_improvement"] == pytest.approx(0.5)


def test_validation_selected_indexes_lr_labels_not_loop_variable():
    rows = []
    for lr, a_val, c_val, a_rmse, c_rmse in (
        ("3e-5", 3.0, 2.5, 4.0, 3.0),
        ("1e-4", 1.0, 2.0, 9.0, 2.0),
        ("3e-4", 2.0, 1.5, 5.0, 2.5),
    ):
        rows.append(
            {
                "dataset": "D1",
                "target": "1/10",
                "horizon": 1,
                "arm": "A",
                "learning_rate_label": lr,
                "run_status": "OK",
                "rmse_original": a_rmse,
                "smape_original": a_rmse,
                "validation_rmse_original": a_val,
                "formal_lr_label": "3e-5",
            }
        )
        rows.append(
            {
                "dataset": "D1",
                "target": "1/10",
                "horizon": 1,
                "arm": "C",
                "learning_rate_label": lr,
                "run_status": "OK",
                "rmse_original": c_rmse,
                "smape_original": c_rmse,
                "validation_rmse_original": c_val,
                "formal_lr_label": "3e-5",
            }
        )
    summary = summarize_validation_selected(rows)
    assert len(summary["condition_rows"]) == 1
    row = summary["condition_rows"][0]
    assert row["A_best_validation_lr"] == "1e-4"
    assert row["C_best_validation_lr"] == "3e-4"
    assert row["A_test_rmse"] == 9.0
    assert row["C_test_rmse"] == 2.5
    assert row["winner"] == "C"


def test_wrapper_script_and_formal_runner_are_separate():
    assert WRAPPER.is_file()
    assert FORMAL_RUNNER.is_file()
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    formal_text = FORMAL_RUNNER.read_text(encoding="utf-8")
    contract_text = CONTRACT.read_text(encoding="utf-8")
    assert "run_ae_formal_matrix" not in wrapper_text
    assert "--execute" in formal_text
    assert "AE_ARMS" in formal_text
    assert "ONLY_LEARNING_RATE_VARIED" in contract_text
    assert "TEST_BASED_LR_SELECTION=NO" in contract_text
    assert "sita_ablation_lr_sensitivity_ac_seed42" in wrapper_text


def test_dry_run_writes_108_manifest(tmp_path):
    out = tmp_path / "sita_ablation_lr_sensitivity_ac_seed42"
    result = run_lr_sensitivity_dry_run(
        output_dir=out,
        project_root=PROJECT_ROOT,
        resolve_real_cells=False,
    )
    assert result["DRY_RUN_TOTAL_RUNS"] == 108
    assert result["READY_FOR_LR_SCREENING"] == "YES"
    assert result["TRAINING_STARTED"] == "NO"
    snapshot = json.loads((out / "manifest_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["expected_runs"] == 108
    assert snapshot["arms"] == ["A", "C"]
    assert snapshot["learning_rates"] == ["3e-5", "1e-4", "3e-4"]
    assert snapshot["seeds"] == [42]
    planned = (out / "plans" / "planned_matrix.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(planned) == 109
    assert "TRAINING_STARTED=NO" in result["text"]
    assert "READY_FOR_LR_SCREENING=YES" in result["text"]
