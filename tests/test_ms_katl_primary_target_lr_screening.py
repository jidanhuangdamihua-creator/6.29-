"""Targeted tests for D1-D6 FUSED_TARGET_ONLY primary-target × LR screening."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import pytest

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.experiment import run_no_tl_experiment
from src.protocols.experiment_protocol import FORMAL_METHODS, formal_target_entity_keys
from src.transfer_methods import ms_katl_lite, ms_katl_mechanism_ablation, msml_tl
from src.transfer_methods.ms_katl_cross_target import D4_CROSS_TARGET_IDENTITIES
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_mechanism_ablation import (
    ABLATION_ARM_SPECS,
    REFERENCE_METRICS,
    assert_formal_methods_untouched,
)
from src.transfer_methods.ms_katl_metric_space_repair import MetricSpaceMismatchError
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    LR_GRID_LABELS,
    METHOD_DISPLAY,
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_HORIZON,
    SCREENING_SEED,
    architecture_digest,
    assert_lr_arms_share_stage_a,
    assert_runner_does_not_launch_robustness,
    assert_screening_does_not_change_formal_methods,
    attach_reference_deltas,
    build_frozen_manifest,
    canonical_candidate_targets,
    format_lr,
    fused_target_only_architecture,
    lookup_reference_row,
    parse_target_identity,
    plan_screening,
    print_screening_preflight,
    select_dataset_primary_target,
    select_target_best_lr,
)


def test_inventory_comes_from_canonical_protocol_not_csv_order():
    plan = plan_screening()
    assert plan["datasets"] == SCREENING_DATASETS
    assert plan["inventory"]["D1"] == formal_target_entity_keys(1) == ("1/10",)
    assert plan["inventory"]["D2"] == ("1/10",)
    assert plan["inventory"]["D3"] == ("10",)
    assert plan["inventory"]["D4"] == D4_CROSS_TARGET_IDENTITIES
    assert set(plan["inventory"]["D4"]) == set(formal_target_entity_keys(4))
    assert plan["inventory"]["D4"] != formal_target_entity_keys(4)
    assert plan["inventory"]["D5"] == formal_target_entity_keys(5)
    assert plan["inventory"]["D6"] == formal_target_entity_keys(6)
    assert plan["total_targets"] == 18
    assert plan["total_screening_runs"] == 54


def test_d4_identity_does_not_follow_protocol_tuple_order():
    protocol_order = formal_target_entity_keys(4)
    assert protocol_order[1] != "166/311"
    parsed = parse_target_identity("D4", "166_311")
    assert parsed["target_identity"] == "166/311"
    assert parsed["entity_key"] == "166_311"


def test_d6_identity_uses_canonical_slash_form():
    parsed = parse_target_identity("D6", "CA_1/FOODS_3_586")
    assert parsed["target_identity"] == "CA_1/FOODS_3_586"
    assert parsed["entity_key"] == "CA_1_FOODS_3_586"
    assert parse_target_identity("D6", "CA_1_FOODS_3_586")["target_identity"] == "CA_1/FOODS_3_586"


def test_unknown_target_fails_closed():
    with pytest.raises(Exception):
        parse_target_identity("D4", "999/999")
    with pytest.raises(Exception):
        parse_target_identity("D1", "2/10")


def test_lr_grid_horizon_seed_epochs_are_fixed():
    plan = plan_screening()
    assert plan["lr_grid"] == LR_GRID == (3e-5, 1e-4, 3e-4)
    assert plan["lr_grid_labels"] == LR_GRID_LABELS
    assert [format_lr(value) for value in LR_GRID] == ["3e-5", "1e-4", "3e-4"]
    assert plan["horizon"] == SCREENING_HORIZON == 1
    assert plan["seed"] == SCREENING_SEED == 42
    assert plan["epochs"] == SCREENING_EPOCHS == 50
    with pytest.raises(Exception):
        format_lr(5e-4)


def test_architecture_is_fused_target_only_on_every_dataset():
    contract = fused_target_only_architecture()
    spec = ABLATION_ARM_SPECS["fused_target_only"]
    assert spec.display == METHOD_DISPLAY
    assert contract["use_fused_target_init"] is True
    assert contract["target_backbone_trainable"] is True
    assert contract["transfer_branch_enabled"] is False
    assert contract["gate_mode"] == "zero"
    assert contract["lambda_adv"] == 0.0
    digests = {architecture_digest() for _ in SCREENING_DATASETS}
    assert len(digests) == 1


def test_preflight_lists_canonical_targets_and_run_count(capsys):
    text = print_screening_preflight()
    captured = capsys.readouterr().out
    assert "D1-D6 PRIMARY TARGET × LR SCREENING PREFLIGHT" in captured
    assert "TOTAL_TARGETS=18" in text
    assert "TOTAL_SCREENING_RUNS=54" in text
    assert "166/258" in text
    assert "166/311" in text
    assert "FORMAL_METHODS_CHANGE=NO" in text
    assert "ARCHITECTURE_CHANGE=NO" in text


def _arm_row(target: str, lr: float, rmse: float, **extra):
    row = {
        "dataset": "D4",
        "target_identity": target,
        "learning_rate": lr,
        "method": METHOD_DISPLAY,
        "rmse_original": rmse,
        "smape_original": rmse * 10,
        "rmse_normalized": rmse / 2,
        "metric_space": ORIGINAL_SALES_SPACE,
        "source_identities": [("a",), ("b",), ("c",)],
        "source_distances": [1.0, 2.0, 3.0],
        "source_model_digests": ["m1", "m2", "m3"],
        "source_scaler_digests": ["s1", "s2", "s3"],
        "fused_init_digest": "fused-same",
        "target_scaler_digest": "scaler-same",
        "target_split_digest": {"X_train": "x"},
        "architecture_digest": architecture_digest(),
        "run_status": "OK",
        "sales_min": 0.0,
        "sales_max": 10.0,
    }
    row.update(extra)
    return row


def test_three_lr_arms_must_share_stage_a_and_fused_init():
    rows = [
        _arm_row("166/258", 3e-5, 1.0),
        _arm_row("166/258", 1e-4, 0.9),
        _arm_row("166/258", 3e-4, 1.1),
    ]
    flags = assert_lr_arms_share_stage_a(rows)
    assert flags["LR_ARMS_FUSED_INIT_MATCH"] == "YES"
    broken = [dict(row) for row in rows]
    broken[2]["fused_init_digest"] = "different"
    with pytest.raises(Exception, match="fused-init"):
        assert_lr_arms_share_stage_a(broken)


def test_only_learning_rate_may_differ_between_lr_arms():
    rows = [
        _arm_row("166/258", 3e-5, 1.0),
        _arm_row("166/258", 1e-4, 0.9, target_scaler_digest="other"),
        _arm_row("166/258", 3e-4, 1.1),
    ]
    with pytest.raises(Exception, match="Stage-A"):
        assert_lr_arms_share_stage_a(rows)


def test_metric_space_must_be_original_for_selection():
    rows = [
        _arm_row("166/258", 3e-5, 1.0),
        _arm_row("166/258", 1e-4, 0.9, metric_space="normalized_minmax_space"),
        _arm_row("166/258", 3e-4, 1.1),
    ]
    with pytest.raises(MetricSpaceMismatchError):
        select_target_best_lr(rows)


def test_best_lr_uses_original_rmse_then_conservative_tie():
    rows = [
        _arm_row("166/258", 3e-5, 1.004),
        _arm_row("166/258", 1e-4, 1.000),
        _arm_row("166/258", 3e-4, 1.200),
    ]
    best = select_target_best_lr(rows)
    assert best["best_lr"] == "3e-5"
    far = [
        _arm_row("166/258", 3e-5, 1.05),
        _arm_row("166/258", 1e-4, 1.00),
        _arm_row("166/258", 3e-4, 1.20),
    ]
    assert select_target_best_lr(far)["best_lr"] == "1e-4"


def test_primary_target_gates_and_mixed_evidence():
    rows = [
        {
            **_arm_row("166/258", 1e-4, 0.80),
            "best_lr": "1e-4",
            "best_rmse_original": 0.80,
            "beats_no_tl": True,
            "beats_msml": False,
            "beats_bl4": False,
            "beats_best_bl": False,
            "relative_improvement_vs_best_bl": -0.20,
        },
        {
            **_arm_row("166/311", 3e-4, 0.70),
            "best_lr": "3e-4",
            "best_rmse_original": 0.70,
            "beats_no_tl": False,
            "beats_msml": False,
            "beats_bl4": False,
            "beats_best_bl": False,
            "relative_improvement_vs_best_bl": -0.10,
        },
    ]
    selected = select_dataset_primary_target(rows)
    assert selected["selected"]["target_identity"] == "166/258"
    assert selected["verdict"] == "PRIMARY_TARGET_MIXED_CANDIDATE"
    strong = [
        {
            **rows[0],
            "beats_msml": True,
            "beats_bl4": True,
            "beats_best_bl": True,
            "relative_improvement_vs_best_bl": 0.05,
        }
    ]
    assert select_dataset_primary_target(strong)["verdict"] == "PRIMARY_TARGET_STRONG_CANDIDATE"


def test_no_clear_primary_when_no_target_beats_no_tl():
    rows = [
        {
            **_arm_row("1/10", 1e-4, 2.0),
            "dataset": "D1",
            "best_lr": "1e-4",
            "best_rmse_original": 2.0,
            "beats_no_tl": False,
            "beats_msml": False,
            "beats_bl4": False,
        }
    ]
    selected = select_dataset_primary_target(rows)
    assert selected["verdict"] == "NO_CLEAR_PRIMARY_TARGET"
    assert selected["selected"]["target_identity"] == "1/10"


def test_reference_join_is_exact_identity_and_ignores_wrong_seed():
    rows = [
        {
            "dataset_id": "D4",
            "target_entity_key": "166_258",
            "scenario": "without",
            "method": "No-TL",
            "horizon": 1,
            "seed": 42,
            "rmse": 0.5,
            "original_scale_rmse": 0.5,
            "metric_space": ORIGINAL_SALES_SPACE,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166/258",
            "scenario": "without",
            "method": "No-TL",
            "horizon": 1,
            "seed": 43,
            "rmse": 9.9,
            "original_scale_rmse": 9.9,
            "metric_space": ORIGINAL_SALES_SPACE,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166/311",
            "scenario": "without",
            "method": "No-TL",
            "horizon": 1,
            "seed": 42,
            "rmse": 7.7,
            "original_scale_rmse": 7.7,
            "metric_space": ORIGINAL_SALES_SPACE,
        },
    ]
    found = lookup_reference_row(
        rows,
        dataset_id="D4",
        target_identity="166/258",
        method="No-TL",
        horizon=1,
        seed=42,
        mode="without",
    )
    assert found["status"] == "FOUND"
    assert found["rmse"] == 0.5


def test_normalized_reference_fails_closed():
    rows = [
        {
            "dataset_id": "D4",
            "target_entity_key": "166/258",
            "scenario": "without",
            "method": "No-TL",
            "horizon": 1,
            "seed": 42,
            "rmse": 0.3,
            "rmse_metric_space": "normalized_minmax_space",
            "metric_space": "normalized_minmax_space",
        }
    ]
    with pytest.raises(MetricSpaceMismatchError):
        lookup_reference_row(
            rows,
            dataset_id="D4",
            target_identity="166/258",
            method="No-TL",
        )


def test_hardcoded_reference_metrics_are_not_used_for_selection():
    source = inspect.getsource(
        sys.modules["src.transfer_methods.ms_katl_primary_target_lr_screening"]
    )
    assert "REFERENCE_METRICS" not in source
    assert REFERENCE_METRICS["FORMAL_NO_TL"]["usable_for_comparison"] is False
    row = attach_reference_deltas(
        _arm_row("166/258", 1e-4, 0.4),
        formal_refs={
            "no_tl": {"status": "FOUND", "rmse": 0.5, "method": "No-TL"},
            "msml": {"status": "MISSING", "rmse": None, "method": "MSML-TL"},
        },
        baseline_refs={
            "bl1": {"rmse": 0.6},
            "bl2": {"rmse": 0.7},
            "bl3": {"rmse": 0.8},
            "bl4": {"rmse": 0.9},
            "bl5": {"rmse": 1.0},
            "best": {"method": "BL1_HistoricalMean", "rmse": 0.6},
        },
    )
    assert row["no_tl_rmse"] == 0.5
    assert row["beats_no_tl"] is True
    assert row["msml_rmse"] is None
    assert row["delta_vs_no_tl"] == pytest.approx(-0.1)


def test_selected_target_and_lr_come_from_inventory_and_grid():
    plan = plan_screening()
    d4_targets = set(plan["inventory"]["D4"])
    rows = [
        {
            **_arm_row(identity, 1e-4, 0.5 + index * 0.1),
            "best_lr": "1e-4",
            "best_rmse_original": 0.5 + index * 0.1,
            "beats_no_tl": True,
            "beats_msml": True,
            "beats_bl4": False,
        }
        for index, identity in enumerate(D4_CROSS_TARGET_IDENTITIES)
    ]
    selected = select_dataset_primary_target(rows)["selected"]
    assert selected["target_identity"] in d4_targets
    assert selected["best_lr"] in LR_GRID_LABELS


def test_frozen_manifest_has_exactly_six_datasets():
    primary_rows = [
        {
            "dataset": dataset,
            "selected_primary_target": canonical_candidate_targets(dataset)[0],
            "selected_learning_rate": "1e-4",
            "rmse_original": 1.0,
            "verdict": "PRIMARY_TARGET_MIXED_CANDIDATE",
        }
        for dataset in SCREENING_DATASETS
    ]
    manifest = build_frozen_manifest(primary_rows)
    assert set(manifest["datasets"]) == set(SCREENING_DATASETS)
    assert len(manifest["datasets"]) == 6
    assert manifest["selection_frozen"] is True
    assert manifest["auto_start_robustness"] is False
    assert manifest["horizon_used_for_screening"] == 1
    assert manifest["seed_used_for_screening"] == 42
    assert manifest["epochs"] == 50
    assert manifest["lr_grid"] == list(LR_GRID)


def test_screening_does_not_modify_formal_methods_or_baselines():
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert FORMAL_METHODS == expected
    assert assert_screening_does_not_change_formal_methods() == expected
    assert METHOD_NAME not in FORMAL_METHODS
    assert METHOD_DISPLAY not in FORMAL_METHODS
    assert inspect.getsource(msml_tl).find("FUSED_TARGET_ONLY") == -1
    assert "fine_tune_learning_rate" not in inspect.getsource(run_no_tl_experiment)


def test_screening_does_not_change_ms_katl_architecture_defaults():
    source = inspect.getsource(ms_katl_lite)
    assert "transfer_branch_enabled: bool = True" in source
    spec = ABLATION_ARM_SPECS["fused_target_only"]
    assert spec.transfer_branch_enabled is False
    ablation_source = inspect.getsource(ms_katl_mechanism_ablation)
    assert "fused_target_only" in ablation_source


def test_runner_does_not_auto_start_robustness():
    runner = (
        PROJECT_ROOT / "scripts" / "run_ms_katl_primary_target_lr_screening.py"
    ).read_text(encoding="utf-8")
    assert_runner_does_not_launch_robustness(runner)
    assert "auto_start_robustness" in runner
    assert "H={1,3,5}" not in runner
    assert "seed={42,43,44}" not in runner


def test_formal_methods_untouched_helper_still_passes():
    assert_formal_methods_untouched()
