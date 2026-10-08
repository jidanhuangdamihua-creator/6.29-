"""Contract tests for minimal V3-with integration and the four-cell probe."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.ms_katl_fto_three_track_robustness import RunIdentity, cell_path
from src.transfer_methods.ms_katl_v3_frame_assembly import (
    _d46_config,
    assemble_v3_cell_frames,
    load_d13_pool,
    load_d46_pool,
    normalize_v3_mode,
    v3_knn_authority_path,
    v3_scenario_for_mode,
)
from src.transfer_methods.ms_katl_v3_with_probe import (
    EXPECTED_PROBE_CELLS,
    PROBE_TARGETS,
    build_probe_identities,
    build_probe_manifest,
    format_probe_preflight,
    paired_comparison_row,
    probe_cell_path,
    probe_run_root,
    summarize_comparisons,
)


def _identity(mode: str) -> RunIdentity:
    return RunIdentity(
        arm="v3",
        dataset="D4",
        target_identity="166/311",
        horizon=1,
        seed=42,
        selected_lr=3e-5,
        selected_lr_label="3e-5",
        mode=mode,
    )


def test_default_mode_is_without_and_keeps_without_authority_routing():
    assert normalize_v3_mode() == "without"
    assert v3_scenario_for_mode() == "without_information_sharing"
    assert inspect.signature(assemble_v3_cell_frames).parameters["mode"].default == "without"
    assert inspect.signature(load_d13_pool).parameters["mode"].default == "without"
    assert inspect.signature(load_d46_pool).parameters["mode"].default == "without"
    assert _d46_config(4, PROJECT_ROOT)["info_sharing"] == "without"


def test_explicit_with_routes_d13_lifecycle_and_d46_solidified_authority():
    assert v3_scenario_for_mode("with") == "with_information_sharing"
    for dataset_id in (4, 5, 6):
        config = _d46_config(dataset_id, PROJECT_ROOT, "with")
        authority = v3_knn_authority_path(f"D{dataset_id}", "with")
        assert config["info_sharing"] == "with"
        assert authority.name == "knn_with_info_sharing.json"
        assert authority.is_file()
        assert json.loads(authority.read_text(encoding="utf-8"))["info_sharing"] == "with"
    with pytest.raises(Exception, match="must be 'without' or 'with'"):
        normalize_v3_mode("unknown")


def test_mode_is_part_of_identity_duplicate_guard_and_output_path():
    without = _identity("without")
    with_identity = _identity("with")
    assert without.identity_tuple != with_identity.identity_tuple
    assert without.paired_tuple != with_identity.paired_tuple
    root = PROJECT_ROOT / "outputs" / "runs" / "synthetic" / "cells"
    assert cell_path(root, without) != cell_path(root, with_identity)
    assert "mode=without" in str(cell_path(root, without))
    assert "mode=with" in str(cell_path(root, with_identity))


def test_probe_plan_is_exactly_four_frozen_lr_with_cells(tmp_path):
    identities = build_probe_identities(PROJECT_ROOT)
    assert len(identities) == EXPECTED_PROBE_CELLS == 4
    assert tuple((item.dataset, item.target_identity) for item in identities) == PROBE_TARGETS
    assert {item.mode for item in identities} == {"with"}
    assert {item.horizon for item in identities} == {1}
    assert {item.seed for item in identities} == {42}
    assert [(item.dataset, item.selected_lr_label) for item in identities] == [
        ("D1", "3e-5"),
        ("D4", "3e-5"),
        ("D5", "3e-4"),
        ("D6", "3e-5"),
    ]
    assert len({item.identity_tuple for item in identities}) == 4
    assert len({probe_cell_path(tmp_path, item) for item in identities}) == 4


def test_manifest_and_preflight_record_real_mode_and_isolated_paths(tmp_path):
    identities = build_probe_identities(PROJECT_ROOT)
    run_root = probe_run_root(PROJECT_ROOT, tmp_path / "probe")
    manifest = build_probe_manifest(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        identities=identities,
    )
    assert manifest["mode"] == "with"
    assert manifest["source_mode"] == "with"
    assert manifest["fused_init_source_mode"] == "with"
    assert manifest["expected_probe_cells"] == 4
    assert all(cell["mode"] == "with" for cell in manifest["cells"])
    assert all("mode=with" in cell["output_path"] for cell in manifest["cells"])
    text = format_probe_preflight(
        project_root=PROJECT_ROOT, run_root=run_root, identities=identities
    )
    assert "ACTUAL_CELLS=4" in text
    assert "UNIQUE_CELLS=4" in text
    assert "HORIZONS=1" in text
    assert "SEEDS=42" in text
    assert "LR_SCREENING=NO" in text
    assert "FULL_MATRIX=NO" in text
    assert "OUTPUT_ROOT_COLLISION=NO" in text


def test_paired_comparison_uses_original_metrics_and_positive_gain_means_better():
    common = {
        "dataset": "D4",
        "target_identity": "166/311",
        "horizon": 1,
        "seed": 42,
        "learning_rate_label": "3e-5",
        "architecture_digest": "same-v3",
    }
    row = paired_comparison_row(
        {
            **common,
            "mode": "without",
            "rmse_original": 10.0,
            "smape_original": 20.0,
            "rmse_normalized": 0.1,
        },
        {
            **common,
            "mode": "with",
            "rmse_original": 8.0,
            "smape_original": 18.0,
            "rmse_normalized": 999.0,
        },
    )
    assert row["rmse_gain_percent"] == pytest.approx(20.0)
    assert row["smape_gain_percent"] == pytest.approx(10.0)
    summary = summarize_comparisons([row, row, row, {**row, "rmse_gain_percent": -1.0}])
    assert summary["WITH_RMSE_WINS"] == 3
    assert summary["RECOMMEND_NEXT"] == "12_CELL_HORIZON_PROBE"


def test_probe_runner_reuses_existing_v3_training_path_without_core_copy():
    runner = (PROJECT_ROOT / "scripts" / "run_ms_katl_v3_with_probe.py").read_text(
        encoding="utf-8"
    )
    assert "from scripts.run_ms_katl_fto_three_track_robustness import" in runner
    assert "_run_stage_a" in runner
    assert "_run_arm" in runner
    assert "run_ablation_arm(" not in runner
    assert "Conv1D(" not in runner
