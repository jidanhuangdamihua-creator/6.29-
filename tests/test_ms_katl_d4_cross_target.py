"""Targeted tests for D4 MS-KATL-Lite A/B/C cross-target validation contracts."""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import pandas as pd
import pytest

from src.experiment.run_no_tl_experiment import run_no_tl_experiment
from src.protocols.experiment_protocol import FORMAL_METHODS, formal_target_entity_keys
from src.transfer_methods.ms_katl_cross_target import (
    D4_CROSS_TARGET_IDENTITIES,
    HISTORICAL_MECHANISM_RUN,
    PREFLIGHT_TEXT,
    aggregate_cross_target,
    assert_new_run_does_not_clobber_historical,
    assert_protected_artifacts_unchanged,
    build_target_comparison,
    decide_cross_target_verdict,
    load_baseline_references,
    load_formal_references,
    lookup_reference_row,
    parse_cross_target_arm_spec,
    parse_cross_target_targets,
    parse_d4_target_identity,
    plan_d4_cross_target,
    print_d4_cross_target_preflight,
)
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_mechanism_ablation import assert_formal_methods_untouched
from src.transfer_methods import msml_tl


def test_cross_target_plan_is_exactly_five_by_three():
    plan = plan_d4_cross_target()
    assert plan["n_targets"] == 5
    assert plan["arms_per_target"] == 3
    assert plan["total_arm_runs"] == 15
    assert plan["targets"] == D4_CROSS_TARGET_IDENTITIES
    assert list(plan["arms"]) == ["full_current", "fused_target_only", "raw_target_only"]
    assert [item["arm"] for item in plan["planned"] if item["target_identity"] == "166/311"] == list(plan["arms"])
    assert len({item["target_identity"] for item in plan["planned"]}) == 5


def test_cross_target_plan_does_not_use_protocol_row_order():
    protocol_order = formal_target_entity_keys(4)
    assert protocol_order != D4_CROSS_TARGET_IDENTITIES
    assert set(protocol_order) == set(D4_CROSS_TARGET_IDENTITIES)
    parsed = parse_cross_target_targets("166/433,166/258,166/432,166/311,166/313")
    assert parsed[0] == "166/433"
    assert parsed[1] == "166/258"


def test_target_identity_accepts_slash_and_underscore_but_canonicalizes():
    slash = parse_d4_target_identity("166/311")
    under = parse_d4_target_identity("166_311")
    assert slash["target_identity"] == under["target_identity"] == "166/311"
    assert slash["entity_key"] == "166_311"
    assert slash["store_id"] == "166"
    assert slash["product_id"] == "311"


def test_unknown_or_forbidden_cross_target_arms_fail_closed():
    with pytest.raises(ValueError, match="Unknown ablation_arm"):
        parse_cross_target_arm_spec("full_current,fused_target_only,mystery")
    with pytest.raises(ValueError, match="forbids D/E/F"):
        parse_cross_target_arm_spec("full_current,fused_target_only,raw_plus_transfer")
    with pytest.raises(ValueError, match="forbids D/E/F"):
        parse_cross_target_arm_spec("full_current,fused_target_only,fused_frozen_backbone")


def _bl_row(identity: str, method: str, rmse: float, seed: int = 42, horizon: int = 1) -> dict:
    return {
        "dataset_id": "D4",
        "target_entity_key": identity,
        "scenario": "without",
        "method": method,
        "horizon": horizon,
        "seed": seed,
        "rmse": rmse,
        "smape": rmse * 10,
    }


def test_baseline_join_is_identity_horizon_seed_mode_not_row_order(tmp_path):
    rows = [
        {
            "dataset_id": "D4",
            "target_entity_key": "166/433",
            "scenario": "without",
            "method": "BL4_LSTM",
            "horizon": 1,
            "seed": 42,
            "rmse": 9.99,
            "smape": 90.0,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166_258",
            "scenario": "without_information_sharing",
            "method": "BL4_LSTM",
            "horizon": 1,
            "seed": 42,
            "rmse": 1.07,
            "smape": 115.0,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166/258",
            "scenario": "without",
            "method": "BL4_LSTM",
            "horizon": 1,
            "seed": 43,
            "rmse": 3.33,
            "smape": 33.0,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166/258",
            "scenario": "with",
            "method": "BL4_LSTM",
            "horizon": 1,
            "seed": 42,
            "rmse": 4.44,
            "smape": 44.0,
        },
        {
            "dataset_id": "D4",
            "target_entity_key": "166/258",
            "scenario": "without",
            "method": "BL4_LSTM",
            "horizon": 2,
            "seed": 42,
            "rmse": 5.55,
            "smape": 55.0,
        },
    ]
    path = tmp_path / "baselines.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    found = load_baseline_references(path, target_identity="166/258", horizon=1, seed=42, mode="without")
    assert found["bl4"]["status"] == "FOUND"
    assert found["bl4"]["rmse"] == pytest.approx(1.07)
    missing = load_baseline_references(path, target_identity="166/311", horizon=1, seed=42, mode="without")
    assert missing["bl4"]["status"] == "MISSING"
    assert missing["bl4"]["rmse"] is None
    wrong_seed = lookup_reference_row(
        rows,
        target_identity="166/258",
        method="BL4_LSTM",
        horizon=1,
        seed=43,
        mode="without",
    )
    assert wrong_seed["rmse"] == pytest.approx(3.33)


def test_missing_formal_reference_is_marked_not_substituted(tmp_path):
    path = tmp_path / "formal.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "4",
                "target_entity_key": "166/258",
                "information_sharing": "without",
                "method": "No-TL",
                "horizon": 1,
                "seed": 42,
                "rmse": 0.68,
                "smape": 37.5,
            }
        ]
    ).to_csv(path, index=False)
    refs = load_formal_references(path, target_identity="166/311", horizon=1, seed=42, mode="without")
    assert refs["no_tl"]["status"] == "MISSING"
    assert refs["msml"]["status"] == "MISSING"
    assert refs["no_tl"]["rmse"] is None
    found = load_formal_references(path, target_identity="166/258", horizon=1, seed=42, mode="without")
    assert found["no_tl"]["status"] == "FOUND"
    assert found["no_tl"]["rmse"] == pytest.approx(0.68)
    assert found["msml"]["status"] == "MISSING"


def test_missing_csv_path_marks_all_missing(tmp_path):
    refs = load_formal_references(tmp_path / "nope.csv", target_identity="166/258")
    assert refs["no_tl"]["status"] == "MISSING"
    baselines = load_baseline_references(tmp_path / "nope.csv", target_identity="166/258")
    assert baselines["best"]["status"] == "MISSING"
    assert baselines["bl1"]["status"] == "MISSING"


def test_best_bl_is_min_rmse_by_identity_not_csv_order(tmp_path):
    path = tmp_path / "bl.csv"
    pd.DataFrame(
        [
            _bl_row("166/311", "BL1_HistoricalMean", 0.50),
            _bl_row("166/258", "BL1_HistoricalMean", 0.40),
            _bl_row("166/258", "BL2_MovingAverage", 0.22),
            _bl_row("166/258", "BL3_LightGBM", 0.90),
            _bl_row("166/258", "BL4_LSTM", 1.07),
            _bl_row("166/258", "BL5_SeasonalNaive", 0.30),
            _bl_row("166/311", "BL2_MovingAverage", 0.10),
        ]
    ).to_csv(path, index=False)
    refs = load_baseline_references(path, target_identity="166/258")
    assert refs["best"]["method"] == "BL2_MovingAverage"
    assert refs["best"]["rmse"] == pytest.approx(0.22)
    other = load_baseline_references(path, target_identity="166/311")
    assert other["best"]["method"] == "BL2_MovingAverage"
    assert other["best"]["rmse"] == pytest.approx(0.10)


def _arm_stub(name: str, rmse: float, *, fused: bool, digest: str, split: str = "split-a") -> dict:
    from src.transfer_methods.ms_katl_mechanism_ablation import ABLATION_ARM_SPECS

    spec = ABLATION_ARM_SPECS[name]
    return {
        "arm": name,
        "rmse": rmse,
        "smape": rmse * 10,
        "gate_mode": "learnable" if spec.transfer_branch_enabled else "disabled",
        "transfer_enabled": spec.transfer_branch_enabled,
        "source_identities": [("166", "184"), ("166", "530"), ("166", "229")],
        "source_distances": [0.1, 0.2, 0.3],
        "source_weight_digests": ["m1", "m2", "m3"],
        "source_scaler_digests": ["s1", "s2", "s3"],
        "fused_params_digest": "fused-params",
        "target_conv_init_digest": digest,
        "target_split_digest": {"X_train": split, "target_scaler": "scaler-a"},
        "gate_diagnostics": {"gate_mean_final": 0.12} if spec.transfer_branch_enabled else {"gate_mode": "disabled"},
        "transfer_diagnostics": {"transfer_ratio": 0.01, "correction_direction_accuracy": 0.5},
        "conv_drifts": {
            "conv1_relative_weight_drift": 0.03,
            "conv2_relative_weight_drift": 0.05,
            "conv3_relative_weight_drift": 0.08,
            "total_backbone_relative_weight_drift": 0.06,
        },
        "attention_diagnostics": {"attention_entropy_mean": 1.09, "mean_abs_alpha_minus_prior": 0.02},
        "y_true": [1.0],
        "y_pred": [1.1],
        "decomposition": {
            "target_prediction": [1.1],
            "transfer_raw": [0.0],
            "gate": [0.0],
            "transfer_gated": [0.0],
            "final_prediction": [1.1],
        },
        "fused_init": fused,
    }


def _yes_flags() -> dict:
    return {
        "STAGE_A_SHARED": "YES",
        "SOURCE_IDENTITIES_MATCH": "YES",
        "SOURCE_ORDER_MATCH": "YES",
        "SOURCE_MODEL_DIGESTS_MATCH": "YES",
        "SOURCE_SCALERS_MATCH": "YES",
        "TARGET_SPLIT_MATCH": "YES",
        "FUSED_PARAMS_DIGEST_MATCH": "YES",
        "FUSED_TARGET_INIT_MATCH": "YES",
        "RAW_TARGET_INIT_MATCH": "YES",
        "RAW_INIT_DISTINCT_FROM_FUSED": "YES",
    }


def test_build_target_comparison_computes_effects_and_win_flags():
    arms = {
        "full_current": _arm_stub("full_current", 0.41, fused=True, digest="fused"),
        "fused_target_only": _arm_stub("fused_target_only", 0.33, fused=True, digest="fused"),
        "raw_target_only": _arm_stub("raw_target_only", 0.49, fused=False, digest="raw"),
    }
    comparison = build_target_comparison(
        target_identity="166/258",
        arm_rows=arms,
        flags=_yes_flags(),
        formal_refs={
            "no_tl": {"status": "FOUND", "rmse": 0.68, "smape": 37.5, "method": "No-TL"},
            "msml": {"status": "FOUND", "rmse": 1.19, "smape": 141.4, "method": "MSML-TL"},
        },
        baseline_refs={
            "bl1": {"status": "FOUND", "rmse": 0.35, "smape": 22.0, "method": "BL1_HistoricalMean"},
            "bl2": {"status": "FOUND", "rmse": 0.34, "smape": 22.0, "method": "BL2_MovingAverage"},
            "bl3": {"status": "FOUND", "rmse": 0.72, "smape": 60.0, "method": "BL3_LightGBM"},
            "bl4": {"status": "FOUND", "rmse": 1.07, "smape": 115.0, "method": "BL4_LSTM"},
            "bl5": {"status": "FOUND", "rmse": 0.42, "smape": 28.0, "method": "BL5_SeasonalNaive"},
            "best": {"status": "FOUND", "rmse": 0.34, "smape": 22.0, "method": "BL2_MovingAverage"},
        },
        cell_meta={"dataset": "D4", "mode": "without", "horizon": 1, "seed": 42, "epochs": 50},
    )
    assert comparison["effects"]["INIT_EFFECT"] == pytest.approx(0.16)
    assert comparison["effects"]["TRANSFER_EFFECT"] == pytest.approx(-0.08)
    assert comparison["B_beats_A"] is True
    assert comparison["B_beats_C"] is True
    assert comparison["B_beats_NoTL"] is True
    assert comparison["B_beats_best_BL"] is True
    assert comparison["best_bl_method"] == "BL2_MovingAverage"


def test_pairing_failure_is_fail_closed():
    arms = {
        "full_current": _arm_stub("full_current", 0.41, fused=True, digest="fused"),
        "fused_target_only": _arm_stub("fused_target_only", 0.33, fused=True, digest="fused"),
        "raw_target_only": _arm_stub("raw_target_only", 0.49, fused=False, digest="raw"),
    }
    flags = _yes_flags()
    flags["STAGE_A_SHARED"] = "NO"
    with pytest.raises(ValueError, match="paired Stage-A contract failed"):
        build_target_comparison(
            target_identity="166/258",
            arm_rows=arms,
            flags=flags,
            formal_refs={"no_tl": {"status": "MISSING"}, "msml": {"status": "MISSING"}},
            baseline_refs={
                "bl1": {"status": "MISSING"},
                "bl2": {"status": "MISSING"},
                "bl3": {"status": "MISSING"},
                "bl4": {"status": "MISSING"},
                "bl5": {"status": "MISSING"},
                "best": {"status": "MISSING"},
            },
            cell_meta={},
        )


def test_c_loading_fused_weights_fails_closed():
    arms = {
        "full_current": _arm_stub("full_current", 0.41, fused=True, digest="fused"),
        "fused_target_only": _arm_stub("fused_target_only", 0.33, fused=True, digest="fused"),
        "raw_target_only": _arm_stub("raw_target_only", 0.49, fused=False, digest="fused"),
    }
    with pytest.raises(ValueError, match="C loaded fused weights"):
        build_target_comparison(
            target_identity="166/258",
            arm_rows=arms,
            flags=_yes_flags(),
            formal_refs={"no_tl": {"status": "MISSING"}, "msml": {"status": "MISSING"}},
            baseline_refs={
                "bl1": {"status": "MISSING"},
                "bl2": {"status": "MISSING"},
                "bl3": {"status": "MISSING"},
                "bl4": {"status": "MISSING"},
                "bl5": {"status": "MISSING"},
                "best": {"status": "MISSING"},
            },
            cell_meta={},
        )


def test_aggregates_and_verdict_do_not_change_architecture():
    comparisons = []
    for identity, b_rmse, a_rmse, c_rmse in (
        ("166/258", 0.33, 0.41, 0.49),
        ("166/311", 0.30, 0.40, 0.45),
        ("166/313", 0.32, 0.39, 0.44),
        ("166/432", 0.31, 0.38, 0.50),
        ("166/433", 0.35, 0.42, 0.47),
    ):
        comparisons.append(
            {
                "target_identity": identity,
                "A_rmse": a_rmse,
                "B_rmse": b_rmse,
                "C_rmse": c_rmse,
                "effects": {"INIT_EFFECT": c_rmse - b_rmse, "TRANSFER_EFFECT": b_rmse - a_rmse},
                "B_beats_A": b_rmse < a_rmse,
                "B_beats_C": b_rmse < c_rmse,
                "B_beats_NoTL": True,
                "B_beats_MSML": True,
                "B_beats_BL4": True,
                "B_beats_best_BL": True,
                "delta_rmse_vs_no_tl": b_rmse - 0.68,
                "delta_rmse_vs_msml": b_rmse - 1.19,
                "delta_rmse_vs_bl4": b_rmse - 1.07,
                "delta_rmse_vs_best_bl": b_rmse - 0.34,
                "relative_improvement_vs_no_tl": (0.68 - b_rmse) / 0.68,
                "relative_improvement_vs_msml": (1.19 - b_rmse) / 1.19,
                "relative_improvement_vs_bl4": (1.07 - b_rmse) / 1.07,
                "relative_improvement_vs_best_bl": (0.34 - b_rmse) / 0.34,
                "flags": {
                    "STAGE_A_SHARED": "YES",
                    "SOURCE_IDENTITIES_MATCH": "YES",
                    "SOURCE_MODEL_DIGESTS_MATCH": "YES",
                    "SOURCE_SCALERS_MATCH": "YES",
                    "TARGET_SPLIT_MATCH": "YES",
                    "FUSED_TARGET_INIT_MATCH": "YES",
                },
            }
        )
    aggregates = aggregate_cross_target(comparisons)
    assert aggregates["B_vs_A"]["label"] == "5/5"
    assert aggregates["B_vs_C"]["label"] == "5/5"
    assert decide_cross_target_verdict(aggregates, comparisons) == "FUSED_TARGET_ONLY_CROSS_TARGET_SUPPORTED"
    mixed = [dict(item) for item in comparisons]
    mixed[0]["B_beats_C"] = False
    mixed[1]["B_beats_C"] = False
    mixed[0]["C_rmse"] = mixed[0]["B_rmse"] - 0.05
    mixed[1]["C_rmse"] = mixed[1]["B_rmse"] - 0.05
    mixed[0]["effects"] = {
        "INIT_EFFECT": mixed[0]["C_rmse"] - mixed[0]["B_rmse"],
        "TRANSFER_EFFECT": mixed[0]["B_rmse"] - mixed[0]["A_rmse"],
    }
    mixed[1]["effects"] = {
        "INIT_EFFECT": mixed[1]["C_rmse"] - mixed[1]["B_rmse"],
        "TRANSFER_EFFECT": mixed[1]["B_rmse"] - mixed[1]["A_rmse"],
    }
    mixed_agg = aggregate_cross_target(mixed)
    assert mixed_agg["B_vs_C"]["wins"] == 3
    assert decide_cross_target_verdict(mixed_agg, mixed) == "FUSED_TARGET_ONLY_MIXED_CROSS_TARGET_EVIDENCE"


def test_historical_run_is_not_clobbered(tmp_path):
    historical = tmp_path / "outputs" / "runs" / HISTORICAL_MECHANISM_RUN
    historical.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite historical"):
        assert_new_run_does_not_clobber_historical(tmp_path, historical)
    new_dir = tmp_path / "outputs" / "runs" / "20990101_000000_MS_KATL_D4_H1_S42_CROSS_TARGET"
    new_dir.mkdir(parents=True)
    assert_new_run_does_not_clobber_historical(tmp_path, new_dir)


def test_protected_artifacts_detect_rewrite(tmp_path):
    csv_path = tmp_path / "file.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    before = {
        "baseline_csv": {
            "exists": True,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "mtime_ns": csv_path.stat().st_mtime_ns,
            "path": str(csv_path),
        }
    }
    csv_path.write_text("a,b\n1,3\n", encoding="utf-8")
    after = {
        "baseline_csv": {
            "exists": True,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "mtime_ns": csv_path.stat().st_mtime_ns,
            "path": str(csv_path),
        }
    }
    with pytest.raises(RuntimeError, match="protected artifact"):
        assert_protected_artifacts_unchanged(before, after)


def test_formal_methods_and_msml_no_tl_remain_untouched():
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert FORMAL_METHODS == expected
    assert assert_formal_methods_untouched() == expected
    assert METHOD_NAME not in FORMAL_METHODS
    assert inspect.getsourcefile(msml_tl.fuse_source_models_layerwise).endswith("msml_tl.py")
    assert inspect.getsourcefile(run_no_tl_experiment).endswith("run_no_tl_experiment.py")


def test_preflight_text_matches_contract(capsys):
    text = print_d4_cross_target_preflight()
    assert text == PREFLIGHT_TEXT
    assert "TARGETS=5" in text
    assert "TOTAL_ARM_RUNS=15" in text
    assert "A FULL_CURRENT" in text
    assert "B FUSED_TARGET_ONLY" in text
    assert "C RAW_TARGET_ONLY" in text
    assert "FORMAL_METHODS_RERUN=NO" in text
    assert "BASELINES_RERUN=NO" in text
    captured = capsys.readouterr()
    assert "D4 CROSS-TARGET PREFLIGHT" in captured.out
