"""Contract tests for D4–D6 all-candidate-target SITA enumeration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    COMMON_TARGET_POLICY,
    EXPECTED_COMMON_TARGETS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
)
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_sita_all_targets_d4_d6 import (
    ALL_TARGETS_HORIZONS,
    ALL_TARGETS_SEEDS,
    CELLS_PER_TARGET,
    D5_MAX_JOBS,
    DATASET_INHERITED_LR,
    EXPECTED_DATASET_CELLS,
    EXPECTED_TARGETS_PER_DATASET,
    EXPECTED_TOTAL_CELLS,
    MAX_JOBS,
    PAPER_SITA_MODE,
    REQUESTED_TARGET_SPECS,
    SITA_FORMAL_CONFIG,
    SITA_FORMAL_RUNNER,
    WORKERS,
    AllTargetsContractError,
    all_target_inventory,
    assert_output_does_not_clobber_frozen,
    assert_sita_identity_frozen,
    build_all_target_identities,
    default_output_dir,
    format_dry_run,
    validate_all_target_matrix,
)

WRAPPER = PROJECT_ROOT / "scripts" / "run_ms_katl_sita_all_targets_d4_d6.py"
THREE_TRACK = PROJECT_ROOT / "scripts" / "run_ms_katl_fto_three_track_robustness.py"
CONTRACT = PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_sita_all_targets_d4_d6.py"


def test_wrapper_does_not_change_formal_sita_identity():
    flags = assert_sita_identity_frozen(PROJECT_ROOT)
    assert flags["method"] == SITA_FORMAL_CONFIG == "FUSED_TARGET_ONLY_V3"
    assert flags["mode"] == PAPER_SITA_MODE == "without"
    assert flags["architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert FORMAL_METHODS == (
        "No-TL",
        "SS-TL",
        "MSWA-TL",
        "MSSB-TL",
        "MSML-TL",
        "MSML-TL-RFE",
    )
    assert COMMON_TARGET_POLICY == "V3_FROZEN_PRIMARY_TARGETS"
    assert EXPECTED_COMMON_TARGETS == (
        ("D1", "1/10"),
        ("D2", "1/10"),
        ("D3", "10"),
        ("D4", "166/311"),
        ("D5", "48/1159414"),
        ("D6", "CA_1/FOODS_3_080"),
    )


def test_three_track_runner_remains_primary_target_only():
    text = THREE_TRACK.read_text(encoding="utf-8")
    assert "--targets" not in text
    assert "common_targets_from_frozen_manifest" in text
    assert SITA_FORMAL_RUNNER == "scripts/run_ms_katl_fto_three_track_robustness.py"


def test_requested_targets_match_canonical_inventory_and_user_list():
    inventory = all_target_inventory()
    assert list(inventory) == ["D4", "D5", "D6"]
    assert inventory["D4"] == REQUESTED_TARGET_SPECS["D4"]
    assert inventory["D5"] == REQUESTED_TARGET_SPECS["D5"]
    assert inventory["D6"] == REQUESTED_TARGET_SPECS["D6"]
    assert all("/" in target for targets in inventory.values() for target in targets)
    assert "CA_1_FOODS_3_080" not in inventory["D6"]
    assert "1/10" not in {target for targets in inventory.values() for target in targets}


def test_225_cell_matrix_is_unique_and_complete():
    identities = build_all_target_identities(project_root=PROJECT_ROOT)
    report = validate_all_target_matrix(identities)
    assert report["D4_TARGETS"] == EXPECTED_TARGETS_PER_DATASET == 5
    assert report["D5_TARGETS"] == 5
    assert report["D6_TARGETS"] == 5
    assert report["HORIZONS"] == 3
    assert report["SEEDS"] == 5
    assert report["EXPECTED_D4_CELLS"] == EXPECTED_DATASET_CELLS == 75
    assert report["EXPECTED_D5_CELLS"] == 75
    assert report["EXPECTED_D6_CELLS"] == 75
    assert report["EXPECTED_TOTAL_CELLS"] == EXPECTED_TOTAL_CELLS == 225
    assert report["DRY_RUN_CELL_COUNT"] == 225
    assert report["DUPLICATE_CELLS"] == 0
    assert report["HAS_D1"] is False
    assert report["READY_FOR_FULL_RUN"] == "YES"
    assert ALL_TARGETS_HORIZONS == (1, 3, 5)
    assert ALL_TARGETS_SEEDS == (42, 43, 44, 45, 46)
    assert {item.mode for item in identities} == {PAPER_SITA_MODE}
    assert {item.arm for item in identities} == {"v3"}
    by_target = {}
    for item in identities:
        by_target.setdefault((item.dataset, item.target_identity), 0)
        by_target[(item.dataset, item.target_identity)] += 1
        assert item.selected_lr_label == DATASET_INHERITED_LR[item.dataset]
    assert len(by_target) == 15
    assert set(by_target.values()) == {CELLS_PER_TARGET}


def test_d1_and_extra_targets_are_rejected():
    with pytest.raises(AllTargetsContractError, match="D1"):
        build_all_target_identities(project_root=PROJECT_ROOT, datasets=("D1", "D4"))
    identities = build_all_target_identities(project_root=PROJECT_ROOT)
    mutated = list(identities) + [
        identities[0].__class__(
            arm="v3",
            dataset="D4",
            target_identity="166/999",
            horizon=1,
            seed=42,
            selected_lr=3e-5,
            selected_lr_label="3e-5",
            mode="without",
        )
    ]
    with pytest.raises(AllTargetsContractError):
        validate_all_target_matrix(mutated)


def test_output_dir_cannot_clobber_frozen_sita_results(tmp_path):
    frozen = (
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
    )
    with pytest.raises(AllTargetsContractError, match="frozen"):
        assert_output_does_not_clobber_frozen(frozen, PROJECT_ROOT)
    with pytest.raises(AllTargetsContractError, match="frozen"):
        assert_output_does_not_clobber_frozen(PROJECT_ROOT / "configs", PROJECT_ROOT)
    allowed = default_output_dir(PROJECT_ROOT)
    assert allowed.name == "sita_all_targets_d4_d6_5seed"
    assert_output_does_not_clobber_frozen(
        tmp_path / "sita_all_targets_d4_d6_5seed", PROJECT_ROOT
    )


def test_concurrency_stays_serial():
    assert MAX_JOBS == D5_MAX_JOBS == WORKERS == 1


def test_wrapper_reuses_three_track_numerical_path():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "_load_d46_pool" in text
    assert "_prepare_d46_target_cell" in text
    assert "_run_stage_a" in text
    assert "_run_arm" in text
    assert "classify_cell_artifact" in text
    assert "if args.dry_run:" in text
    assert "FUSED_TARGET_ONLY_V3_1" not in text
    assert "FUSED_TARGET_ONLY_V3_2" not in text
    assert "use_adversarial=True" not in text
    assert "transfer_branch_enabled=True" not in text
    contract = CONTRACT.read_text(encoding="utf-8")
    assert "V3_FROZEN_PRIMARY_TARGETS" in contract
    assert "DATASET_FROZEN_PRIMARY_LR" in contract


def test_dry_run_banner_prints_required_counts():
    identities = build_all_target_identities(project_root=PROJECT_ROOT)
    report = validate_all_target_matrix(identities)
    flags = assert_sita_identity_frozen(PROJECT_ROOT)
    text = format_dry_run(
        identities=identities,
        report=report,
        run_dir=default_output_dir(PROJECT_ROOT),
        identity_flags=flags,
    )
    assert "D4_TARGETS=5" in text
    assert "D5_TARGETS=5" in text
    assert "D6_TARGETS=5" in text
    assert "HORIZONS=3" in text
    assert "SEEDS=5" in text
    assert "EXPECTED_D4_CELLS=75" in text
    assert "EXPECTED_D5_CELLS=75" in text
    assert "EXPECTED_D6_CELLS=75" in text
    assert "EXPECTED_TOTAL_CELLS=225" in text
    assert "DRY_RUN_CELL_COUNT=225" in text
    assert "READY_FOR_FULL_RUN=YES" in text
    assert "SITA_MODEL_CHANGED=NO" in text
    assert "MAX_JOBS=1" in text
    assert "D5_MAX_JOBS=1" in text
