"""No-training contract tests for paper-candidate extra-target A–E slices."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    EXPECTED_RUNS,
    EXTRA_TARGET_CANONICAL_CELLS,
    EXTRA_TARGET_MATRIX_KIND,
    EXTRA_TARGET_RUNS,
    FROZEN_MATRIX_KIND,
    HORIZONS,
    SEEDS,
    V3_LR_MAP,
    AEContractError,
    resolve_cell_lr,
    resolve_extra_target_lr,
)
from src.transfer_methods.ms_katl_ae_execution import (
    enumerate_extra_target_cell_plans,
    enumerate_real_arm_plans,
    planned_extra_target_run_root,
)
from src.transfer_methods.ms_katl_ae_formal import (
    FormalCallLedger,
    make_mock_authorities,
    run_ae_formal_body,
)
from src.transfer_methods.ms_katl_sita_all_targets_d4_d6 import DATASET_INHERITED_LR


def test_extra_target_matrix_size_is_45():
    assert EXTRA_TARGET_CANONICAL_CELLS == len(HORIZONS) * len(SEEDS) == 9
    assert EXTRA_TARGET_RUNS == 9 * len(AE_ARMS) == 45
    assert EXPECTED_RUNS == 270
    assert FROZEN_MATRIX_KIND == "FROZEN_270"


def test_d4_166_258_inherits_sita_lr_and_does_not_mutate_frozen_map():
    lr_value, lr_label = resolve_extra_target_lr("D4", "166/258")
    assert lr_label == "3e-5"
    assert lr_label == DATASET_INHERITED_LR["D4"]
    assert lr_value == pytest.approx(3e-5)
    assert ("D4", "166/258") not in V3_LR_MAP
    assert V3_LR_MAP[("D4", "166/311")] == "3e-5"
    frozen_value, frozen_label = resolve_cell_lr("D4", "166/311")
    assert frozen_label == "3e-5"
    assert frozen_value == pytest.approx(3e-5)


def test_frozen_primary_target_is_rejected_as_extra_slice():
    with pytest.raises(AEContractError, match="frozen primary target"):
        resolve_extra_target_lr("D4", "166/311")


def test_enumerate_d4_166_258_is_9_by_45():
    cells = enumerate_extra_target_cell_plans("D4", "166/258", project_root=PROJECT_ROOT)
    assert len(cells) == 9
    assert {cell.target for cell in cells} == {"166/258"}
    assert {cell.dataset for cell in cells} == {"D4"}
    assert {int(cell.horizon) for cell in cells} == {1, 3, 5}
    assert {int(cell.seed) for cell in cells} == {42, 43, 44}
    assert {cell.selected_lr_label for cell in cells} == {"3e-5"}
    arms = enumerate_real_arm_plans(cells, expected_runs=45)
    assert len(arms) == 45
    assert {item.arm for item in arms} == set(AE_ARMS)
    with pytest.raises(AEContractError, match="real arm plans=45 != 270"):
        enumerate_real_arm_plans(cells)


def test_extra_target_run_root_keeps_ae_label():
    path = planned_extra_target_run_root("20260830_150000", "D4", "166/258")
    assert path.name == "20260830_150000_MS_KATL_AE_CORE_ABLATION_EXTRA_D4_166-258"
    assert "MS_KATL_AE_CORE_ABLATION" in path.name
    assert "EXTRA_D4_166-258" in path.name


def test_mock_extra_target_body_runs_45_and_not_270(tmp_path):
    cells = enumerate_extra_target_cell_plans("D4", "166/258", project_root=PROJECT_ROOT)
    ledger = FormalCallLedger()
    authorities = make_mock_authorities(ledger=ledger, publish_root=tmp_path)
    result = run_ae_formal_body(
        mock_execution_authorized=True,
        authorities=authorities,
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        cells=cells,
        matrix_kind=EXTRA_TARGET_MATRIX_KIND,
        ledger=ledger,
    )
    assert result["MATRIX_KIND"] == EXTRA_TARGET_MATRIX_KIND
    assert result["CANONICAL_CELLS"] == 9
    assert result["EXPECTED_RUNS"] == 45
    assert result["STAGE_B_CALLS"] == 45
    assert result["PAIR_COMPLETE"] == 9
    assert result["EXTRA_DATASET"] == "D4"
    assert result["EXTRA_TARGET"] == "166/258"
    assert result["TRAINING_STARTED"] == "NO"


def test_write_planned_matrix_csv_accepts_extra_target_45(tmp_path):
    from src.transfer_methods.ms_katl_ae_execution import write_planned_matrix_csv

    cells = enumerate_extra_target_cell_plans("D4", "166/258", project_root=PROJECT_ROOT)
    arms = enumerate_real_arm_plans(cells, expected_runs=45)
    path = write_planned_matrix_csv(tmp_path / "planned_matrix.csv", arms, expected_runs=45)
    assert path.is_file()
    with pytest.raises(AEContractError, match="planned matrix rows=45 != 270"):
        write_planned_matrix_csv(tmp_path / "frozen.csv", arms)
    cells = enumerate_extra_target_cell_plans("D4", "166/258", project_root=PROJECT_ROOT)
    ledger = FormalCallLedger()
    authorities = make_mock_authorities(ledger=ledger, publish_root=tmp_path)
    with pytest.raises(AEContractError, match="real arm plans=45 != 270"):
        run_ae_formal_body(
            mock_execution_authorized=True,
            authorities=authorities,
            output_root=tmp_path,
            project_root=PROJECT_ROOT,
            cells=cells,
            matrix_kind=FROZEN_MATRIX_KIND,
            ledger=ledger,
        )
