"""Contract tests for MS-KATL FUSED_TARGET_ONLY three-track paired robustness."""

from __future__ import annotations

import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.experiment import run_no_tl_experiment
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods import msml_tl
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_COMMON_TARGETS,
    EXPECTED_RUNS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    PAIRED_CELLS,
    ROBUSTNESS_ARMS,
    ROBUSTNESS_HORIZONS,
    ROBUSTNESS_SEEDS,
    RUNS_PER_ARM,
    RobustnessContractError,
    assert_formal_methods_unchanged,
    assert_not_screening_output_dir,
    assert_stage_a_is_arm_independent,
    build_run_identities,
    build_v3_frozen_robustness_manifest,
    cell_path,
    classify_cell_artifact,
    common_targets_from_frozen_manifest,
    completion_guard,
    expected_full_matrix_counts,
    format_preflight,
    load_arm_frozen_lr_map,
    load_target_best_lr_map,
    load_target_lr_contract_table,
    load_valid_ok_cells,
    load_v3_frozen_robustness_manifest,
    pair_completed_runs,
    parse_arm_keys,
    parse_dataset_ids,
    parse_int_tuple,
    reconstruct_target_best_lr_map,
    relative_improvement,
    robustness_arm_registry,
    smoke_guard,
    summarize_paired_rows,
    v1_arm_spec,
    v2_arm_spec,
    v3_arm_spec,
    v3_frozen_manifest_path,
    validate_v3_frozen_robustness_manifest,
    write_cell_artifact,
    write_v3_frozen_robustness_manifest,
)
from src.transfer_methods.ms_katl_fto_v2_screening import V2_TRAINING_RECIPE, v2_training_recipe
from src.transfer_methods.ms_katl_fto_v3_screening import V3_REVIN_CONTRACT, v3_architecture_digest
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    FUSED_TARGET_ONLY_CONTRACT,
    LR_GRID,
    METHOD_DISPLAY,
    SCREENING_DATASETS,
    fused_target_only_architecture,
    select_dataset_primary_target,
    select_target_best_lr,
)
from scripts.run_ms_katl_fto_three_track_robustness import (
    EXPLICIT_SEED_ALLOWLIST,
    V3_COMPLETION_SEEDS,
    _resolve_matrix,
)

RUNNER = PROJECT_ROOT / "scripts" / "run_ms_katl_fto_three_track_robustness.py"
CONTRACT = PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_fto_three_track_robustness.py"


def _lr_table():
    return load_target_lr_contract_table(PROJECT_ROOT)


def _full_identities(lr_table=None):
    table = lr_table or _lr_table()
    return build_run_identities(
        arms=ROBUSTNESS_ARMS,
        datasets=SCREENING_DATASETS,
        horizons=ROBUSTNESS_HORIZONS,
        seeds=ROBUSTNESS_SEEDS,
        lr_table=table,
    )


def test_exactly_six_common_v3_primary_targets():
    table = _lr_table()
    observed = [(row["dataset"], row["target_identity"]) for row in table["rows"]]
    assert observed == list(EXPECTED_COMMON_TARGETS)
    assert table["COMMON_TARGET_COUNT"] == 6
    assert [item[0] for item in observed] == list(SCREENING_DATASETS)


def test_exactly_three_arms_horizons_seeds_and_162_identities():
    registry = robustness_arm_registry()
    assert tuple(registry) == ("v1", "v2", "v3")
    assert len(registry) == 3
    assert ROBUSTNESS_HORIZONS == (1, 3, 5)
    assert ROBUSTNESS_SEEDS == (42, 43, 44)
    identities = _full_identities()
    assert len(identities) == EXPECTED_RUNS == 162
    assert len({item.identity_tuple for item in identities}) == 162
    counts = expected_full_matrix_counts(
        ROBUSTNESS_ARMS, SCREENING_DATASETS, ROBUSTNESS_HORIZONS, ROBUSTNESS_SEEDS
    )
    assert counts["paired_cells"] == PAIRED_CELLS == 54
    assert counts["runs_per_arm"] == RUNS_PER_ARM == 54
    assert counts["expected_runs"] == 162


def test_all_arms_use_identical_target_identities():
    identities = _full_identities()
    by_arm = {}
    for item in identities:
        by_arm.setdefault(item.arm, set()).add((item.dataset, item.target_identity))
    assert by_arm["v1"] == by_arm["v2"] == by_arm["v3"] == set(EXPECTED_COMMON_TARGETS)


def test_v1_v2_v3_lr_mappings_are_frozen_from_screening_artifacts():
    table = _lr_table()
    expected = {
        ("D1", "1/10"): ("3e-5", "3e-4", "3e-5"),
        ("D2", "1/10"): ("3e-5", "3e-4", "3e-5"),
        ("D3", "10"): ("1e-4", "3e-4", "3e-5"),
        ("D4", "166/311"): ("3e-5", "3e-4", "3e-5"),
        ("D5", "48/1159414"): ("3e-5", "3e-4", "3e-4"),
        ("D6", "CA_1/FOODS_3_080"): ("3e-5", "3e-4", "3e-5"),
    }
    for row in table["rows"]:
        key = (row["dataset"], row["target_identity"])
        assert (row["v1_lr"], row["v2_lr"], row["v3_lr"]) == expected[key]
    registry = robustness_arm_registry()
    targets = table["targets"]
    for spec in registry.values():
        from_best = load_target_best_lr_map(
            PROJECT_ROOT
            / "outputs"
            / "runs"
            / spec.screening_run_dirname
            / "results"
            / spec.target_best_csv_name
        )
        from_all = reconstruct_target_best_lr_map(
            PROJECT_ROOT
            / "outputs"
            / "runs"
            / spec.screening_run_dirname
            / "results"
            / spec.all_runs_csv_name
        )
        live = load_arm_frozen_lr_map(spec, targets, PROJECT_ROOT)
        for target in targets:
            assert live[target.key] == from_best[target.key] == from_all[target.key]


def test_missing_target_lr_fails_closed(tmp_path, monkeypatch):
    spec = v1_arm_spec()
    table = _lr_table()
    empty = tmp_path / "empty.csv"
    empty.write_text("dataset,target_identity,best_lr,run_status\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.transfer_methods.ms_katl_fto_three_track_robustness.target_best_csv_path",
        lambda arm, root: empty if arm.key == "v1" else (
            PROJECT_ROOT / "outputs" / "runs" / arm.screening_run_dirname / "results" / arm.target_best_csv_name
        ),
    )
    monkeypatch.setattr(
        "src.transfer_methods.ms_katl_fto_three_track_robustness.all_runs_csv_path",
        lambda arm, root: tmp_path / "missing_all_runs.csv",
    )
    with pytest.raises(RobustnessContractError, match="FAIL CLOSED"):
        load_arm_frozen_lr_map(spec, table["targets"], PROJECT_ROOT)


def test_no_robustness_time_lr_or_target_selection():
    source = CONTRACT.read_text(encoding="utf-8") + "\n" + RUNNER.read_text(encoding="utf-8")
    assert "select_dataset_primary_target(" not in source
    assert source.count("select_target_best_lr(") == 1
    assert "reconstruct_target_best_lr_map" in source
    assert list(LR_GRID) == [3e-5, 1e-4, 3e-4]
    identities = _full_identities()
    lrs = {(item.dataset, item.arm, item.selected_lr_label) for item in identities}
    assert ("D3", "v1", "1e-4") in lrs
    assert ("D4", "v1", "3e-5") in lrs
    assert not any(item.selected_lr_label == "1e-3" for item in identities)


def test_original_metric_space_only_and_delta_direction():
    assert relative_improvement(10.0, 8.0) == pytest.approx(0.2)
    assert relative_improvement(10.0, 10.0) == pytest.approx(0.0)
    assert relative_improvement(10.0, 12.0) == pytest.approx(-0.2)
    rows = [
        {
            "status": "OK",
            "arm": arm,
            "dataset": "D1",
            "target_identity": "1/10",
            "horizon": 1,
            "seed": 42,
            "metric_space": ORIGINAL_SALES_SPACE,
            "rmse_original": rmse,
            "smape_original": smape,
        }
        for arm, rmse, smape in (("v1", 10.0, 20.0), ("v2", 8.0, 18.0), ("v3", 5.0, 10.0))
    ]
    paired = pair_completed_runs(rows)
    assert len(paired) == 1
    row = paired[0]
    assert row["relative_RMSE_improvement_V2_vs_V1"] == pytest.approx(0.2)
    assert row["relative_RMSE_improvement_V3_vs_V2"] == pytest.approx(0.375)
    assert row["relative_RMSE_improvement_V3_vs_V1"] == pytest.approx(0.5)
    summary = summarize_paired_rows(paired, scope="global")
    assert summary["V2_vs_V1_RMSE_win"] == 1
    assert summary["V3_vs_V2_RMSE_win"] == 1


def test_paired_merge_exactly_54_rows_after_synthetic_completion():
    identities = _full_identities()
    rows = []
    for item in identities:
        base = 10.0 + {"v1": 2.0, "v2": 1.0, "v3": 0.0}[item.arm]
        rows.append(
            {
                "status": "OK",
                "arm": item.arm,
                "dataset": item.dataset,
                "target_identity": item.target_identity,
                "horizon": item.horizon,
                "seed": item.seed,
                "learning_rate": item.selected_lr,
                "learning_rate_label": item.selected_lr_label,
                "metric_space": ORIGINAL_SALES_SPACE,
                "rmse_original": base,
                "smape_original": base + 5.0,
            }
        )
    paired = pair_completed_runs(rows)
    assert len(paired) == 54
    summary = summarize_paired_rows(paired, scope="global")
    assert summary["paired_cells"] == 54
    assert summary["V2_vs_V1_RMSE_win"] == 54
    guard = completion_guard(identities, rows)
    assert guard["EXPECTED_RUNS"] == 162
    assert guard["OK_RUNS"] == 162
    assert guard["FAILED_RUNS"] == 0
    assert guard["DUPLICATE_RUNS"] == 0
    assert guard["MISSING_RUNS"] == 0
    assert guard["ROBUSTNESS_COMPLETE"] == "YES"


def test_manifest_provenance_and_frozen_v3_readback():
    path = write_v3_frozen_robustness_manifest(PROJECT_ROOT)
    assert path == v3_frozen_manifest_path(PROJECT_ROOT)
    manifest = load_v3_frozen_robustness_manifest(PROJECT_ROOT)
    assert manifest["selection_frozen"] is True
    assert manifest["method"] == "FUSED_TARGET_ONLY_V3"
    assert manifest["targets"] == 6
    assert manifest["architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST == v3_architecture_digest()
    assert manifest["epochs"] == 50
    assert manifest["screening_seed"] == 42
    assert manifest["screening_horizon"] == 1
    assert list(manifest["screening_lr_grid"]) == list(LR_GRID)
    assert manifest["training_recipe"] == dict(v2_training_recipe()) == V2_TRAINING_RECIPE
    assert manifest["RevIN"]["enabled"] is True
    assert manifest["RevIN"]["affine"] is True
    assert manifest["RevIN"]["eps"] == pytest.approx(1e-5)
    assert manifest["RevIN"]["target_channel"] == 0
    assert "FTO_V3_SCREENING" in manifest["source_run_root"]
    screening = json.loads(Path(manifest["source_selection_manifest"]).read_text(encoding="utf-8"))
    assert screening["selection_frozen"] is False
    rebuilt = build_v3_frozen_robustness_manifest(PROJECT_ROOT)
    validate_v3_frozen_robustness_manifest(rebuilt, project_root=PROJECT_ROOT)
    assert common_targets_from_frozen_manifest(manifest)[3].target_identity == "166/311"
    assert manifest["datasets"]["D4"]["learning_rate"] == "3e-5"
    assert manifest["datasets"]["D5"]["learning_rate"] == "3e-4"


def test_v3_revin_v2_recipe_and_v1_config_exact():
    v1 = v1_arm_spec()
    v2 = v2_arm_spec()
    v3 = v3_arm_spec()
    assert dict(v1.architecture) == FUSED_TARGET_ONLY_CONTRACT == fused_target_only_architecture()
    assert v1.pass_training_recipe is False
    assert v1.run_ablation_kwargs() == {}
    assert v1.revin_enabled is False
    assert dict(v2.training_recipe) == {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    assert v2.revin_enabled is False
    assert "revin_enabled" not in v2.run_ablation_kwargs()
    assert dict(v3.training_recipe) == dict(v2.training_recipe)
    assert v3.revin_enabled is True
    kwargs = v3.run_ablation_kwargs()
    assert kwargs["revin_enabled"] is True
    assert kwargs["revin_affine"] is True
    assert kwargs["revin_eps"] == pytest.approx(1e-5)
    assert kwargs["revin_target_channel"] == 0
    assert dict(V3_REVIN_CONTRACT)["revin_enabled"] is True


def test_resume_does_not_duplicate_ok_cells(tmp_path):
    identities = _full_identities()[:3]
    registry = robustness_arm_registry()
    cells_dir = tmp_path / "cells"
    first = identities[0]
    payload = {
        "status": "OK",
        "arm": first.arm,
        "dataset": first.dataset,
        "target_identity": first.target_identity,
        "horizon": first.horizon,
        "seed": first.seed,
        "learning_rate": first.selected_lr,
        "learning_rate_label": first.selected_lr_label,
        "rmse_original": 1.23,
        "smape_original": 4.56,
        "metric_space": ORIGINAL_SALES_SPACE,
        "architecture_digest": registry[first.arm].architecture_digest,
    }
    write_cell_artifact(cell_path(cells_dir, first), payload)
    completed = load_valid_ok_cells(cells_dir, identities, registry)
    assert first.identity_tuple in completed
    pending = [item for item in identities if item.identity_tuple not in completed]
    assert first not in pending
    assert len(pending) == 2
    write_cell_artifact(cell_path(cells_dir, first), payload)
    completed_again = load_valid_ok_cells(cells_dir, identities, registry)
    assert len(completed_again) == 1


def test_malformed_and_stale_cells_rejected(tmp_path):
    identities = _full_identities()[:2]
    registry = robustness_arm_registry()
    cells_dir = tmp_path / "cells"
    ok_id, stale_id = identities
    bad_path = cell_path(cells_dir, ok_id)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not json", encoding="utf-8")
    assert classify_cell_artifact(bad_path, ok_id, registry[ok_id.arm]) == "MALFORMED"
    stale = {
        "status": "OK",
        "arm": stale_id.arm,
        "dataset": stale_id.dataset,
        "target_identity": stale_id.target_identity,
        "horizon": stale_id.horizon,
        "seed": stale_id.seed,
        "learning_rate": stale_id.selected_lr,
        "learning_rate_label": stale_id.selected_lr_label,
        "rmse_original": 1.0,
        "smape_original": 2.0,
        "metric_space": ORIGINAL_SALES_SPACE,
        "architecture_digest": "not-the-frozen-digest",
    }
    stale_path = cell_path(cells_dir, stale_id)
    write_cell_artifact(stale_path, stale)
    assert classify_cell_artifact(stale_path, stale_id, registry[stale_id.arm]) == "STALE"
    failed = dict(stale)
    failed["status"] = "FAILED"
    failed["architecture_digest"] = registry[stale_id.arm].architecture_digest
    write_cell_artifact(stale_path, failed)
    assert classify_cell_artifact(stale_path, stale_id, registry[stale_id.arm]) == "FAILED"
    completed = load_valid_ok_cells(cells_dir, identities, registry)
    assert completed == {}


def test_formal_methods_unchanged():
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert FORMAL_METHODS == expected
    assert assert_formal_methods_unchanged() == expected
    assert METHOD_NAME not in FORMAL_METHODS
    assert METHOD_DISPLAY not in FORMAL_METHODS
    assert inspect.getsource(msml_tl).find("THREE_TRACK_ROBUSTNESS") == -1
    assert "fine_tune_learning_rate" not in inspect.getsource(run_no_tl_experiment)
    protocol_src = (
        PROJECT_ROOT / "src" / "protocols" / "experiment_protocol.py"
    ).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert (
        'FORMAL_METHODS = (\n    "No-TL",\n    "SS-TL",\n    "MSWA-TL",\n'
        '    "MSSB-TL",\n    "MSML-TL",\n    "MSML-TL-RFE",\n)'
    ) in protocol_src


def test_preflight_banner_and_stage_a_reuse_contract():
    table = _lr_table()
    identities = _full_identities(table)
    counts = expected_full_matrix_counts(
        ROBUSTNESS_ARMS, SCREENING_DATASETS, ROBUSTNESS_HORIZONS, ROBUSTNESS_SEEDS
    )
    text = format_preflight(
        arms=ROBUSTNESS_ARMS,
        datasets=SCREENING_DATASETS,
        horizons=ROBUSTNESS_HORIZONS,
        seeds=ROBUSTNESS_SEEDS,
        lr_table=table,
        counts=counts,
    )
    assert "MS-KATL FTO THREE-TRACK ROBUSTNESS PREFLIGHT" in text
    assert "ARMS=V1,V2,V3" in text
    assert "DATASETS=D1,D2,D3,D4,D5,D6" in text
    assert "COMMON_TARGET_COUNT=6" in text
    assert "HORIZONS=1,3,5" in text
    assert "SEEDS=42,43,44" in text
    assert "PAIRED_CELLS=54" in text
    assert "RUNS_PER_ARM=54" in text
    assert "EXPECTED_RUNS=162" in text
    assert "UNIQUE_RUNS=162" in text
    assert "TARGET_POLICY=V3_FROZEN_PRIMARY_TARGETS" in text
    assert "LR_POLICY=ARM_SPECIFIC_FROZEN_TARGET_LR" in text
    assert "ROBUSTNESS_RESELECTION=NO" in text
    assert "V1_CONFIG_FROZEN=YES" in text
    assert "V2_CONFIG_FROZEN=YES" in text
    assert "V3_CONFIG_FROZEN=YES" in text
    assert "FORMAL_METHODS_CHANGE=NO" in text
    assert "READY_FOR_FORMAL_RUN=YES" in text
    assert "TARGET_IDENTITY_MATCH=YES" in text
    assert "LR_SELECTION_FROZEN=YES" in text
    assert "166/311" in text
    assert "SHARED_STAGE_A_EQUIVALENCE=PASS" in text
    assert assert_stage_a_is_arm_independent() == "PASS"
    assert len(identities) == 162


def test_refuses_to_write_into_screening_run_dirs():
    with pytest.raises(RobustnessContractError):
        assert_not_screening_output_dir(
            PROJECT_ROOT / "outputs" / "runs" / "x_MS_KATL_FTO_V3_SCREENING"
        )
    assert_not_screening_output_dir(
        PROJECT_ROOT / "outputs" / "runs" / "x_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
    )


def test_cli_parsers_reject_unknown_values():
    assert parse_arm_keys("v1,v2,v3") == ("v1", "v2", "v3")
    assert parse_dataset_ids("d1,d2") == ("D1", "D2")
    assert parse_int_tuple("1,5", ROBUSTNESS_HORIZONS, name="horizon") == (1, 5)
    with pytest.raises(RobustnessContractError):
        parse_arm_keys("v4")
    with pytest.raises(RobustnessContractError):
        parse_int_tuple("2", ROBUSTNESS_HORIZONS, name="horizon")
    with pytest.raises(RobustnessContractError):
        parse_int_tuple("45", ROBUSTNESS_SEEDS, name="seed")


def test_explicit_v3_seed45_46_completion_preserves_frozen_default():
    default = _resolve_matrix(
        Namespace(smoke=False, arms=None, datasets=None, horizons=None, seeds=None)
    )
    assert default["seeds"] == ROBUSTNESS_SEEDS == (42, 43, 44)
    assert V3_COMPLETION_SEEDS == (45, 46)
    assert EXPLICIT_SEED_ALLOWLIST == (42, 43, 44, 45, 46)

    completion = _resolve_matrix(
        Namespace(
            smoke=False,
            arms="v3",
            datasets="D1,D4,D5,D6",
            horizons="1,3,5",
            seeds="45,46",
        )
    )
    assert completion == {
        "arms": ("v3",),
        "datasets": ("D1", "D4", "D5", "D6"),
        "horizons": (1, 3, 5),
        "seeds": (45, 46),
        "smoke": False,
    }

    with pytest.raises(RobustnessContractError, match="only for the frozen V3 arm"):
        _resolve_matrix(
            Namespace(
                smoke=False,
                arms="v1,v2,v3",
                datasets="D1",
                horizons="1",
                seeds="45",
            )
        )


def test_smoke_guard_and_runner_does_not_reselect():
    rows = [
        {
            "status": "OK",
            "arm": arm,
            "dataset": "D1",
            "target_identity": "1/10",
            "horizon": 1,
            "seed": 42,
            "metric_space": ORIGINAL_SALES_SPACE,
            "rmse_original": 1.0,
            "smape_original": 2.0,
        }
        for arm in ("v1", "v2", "v3")
    ]
    flags = smoke_guard(rows, pair_completed_runs(rows))
    assert flags["SMOKE_ARMS"] == "3"
    assert flags["SMOKE_COMMON_TARGET"] == "YES"
    assert flags["SMOKE_V1_OK"] == "YES"
    assert flags["SMOKE_V2_OK"] == "YES"
    assert flags["SMOKE_V3_OK"] == "YES"
    assert flags["SMOKE_PAIRED_ROW_COUNT"] == "1"
    assert flags["SMOKE_METRIC_SPACE"] == ORIGINAL_SALES_SPACE
    runner = RUNNER.read_text(encoding="utf-8")
    assert "select_dataset_primary_target" not in runner
    assert "--epochs" not in runner
    assert "SCREENING_EPOCHS" in runner
    assert inspect.getsource(select_dataset_primary_target).count("def select_dataset_primary_target") == 1
    assert inspect.getsource(select_target_best_lr).count("relative") >= 1
