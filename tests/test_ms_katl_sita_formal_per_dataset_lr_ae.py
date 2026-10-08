"""Contract tests for the formal per-dataset-target-LR A–E slice."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULE_NAME = "src.transfer_methods.ms_katl_sita_formal_per_dataset_lr_ae"


def _module():
    module = importlib.import_module(MODULE_NAME)
    assert module is not None
    return module


def test_formal_per_dataset_lr_matrix_is_exactly_300_and_uses_corrected_targets():
    module = _module()
    identities = module.enumerate_formal_per_dataset_lr_identities()
    report = module.validate_formal_per_dataset_lr_matrix(identities)

    assert module.FORMAL_PER_DATASET_LR_MATRIX_KIND == "FORMAL_PER_DATASET_LR_AE"
    assert len(identities) == 300
    assert report["EXPECTED_TOTAL_RUNS"] == 300
    assert report["D1_LR"] == "3e-5"
    assert report["D4_LR"] == "3e-5"
    assert report["D5_LR"] == "3e-4"
    assert report["D6_LR"] == "3e-5"
    assert {item.target_identity for item in identities if item.dataset == "D4"} == {"166/258"}
    assert {item.selected_lr_label for item in identities if item.dataset == "D4"} == {"3e-5"}


def test_formal_output_isolated_and_old_uniform_root_is_not_a_formal_root(tmp_path):
    module = _module()
    root = module.assert_formal_per_dataset_lr_output_isolated(
        PROJECT_ROOT / "outputs" / "sita_ablation_formal_per_dataset_lr_ae_20260921",
        project_root=PROJECT_ROOT,
    )
    assert "formal_per_dataset_lr" in root.name
    with pytest.raises(module.FormalPerDatasetLRContractError, match="uniform"):
        module.assert_formal_per_dataset_lr_output_isolated(
            PROJECT_ROOT / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227",
            project_root=PROJECT_ROOT,
        )


def test_d5_reuse_audit_requires_exact_protocol_parity():
    module = _module()
    report = module.audit_existing_d5_reuse(
        PROJECT_ROOT / "outputs" / "sita_ablation_3e4_corrected_stage_a_rng_uniform_20260919_110227",
        project_root=PROJECT_ROOT,
    )
    assert report["status"] == "PASS"
    assert report["arm_runs"] == 75
    assert report["target_lr"] == "3e-4"
    assert report["stage_a_lr"] == "1e-4"
    assert report["source_selection"] == "PASS"
    assert report["current_corrected_stage_a_rng"] == "PASS"
    assert report["mismatches"] == []


def test_formal_manifest_declares_no_global_uniform_lr_or_sensitivity_grid():
    module = _module()
    payload = module.formal_per_dataset_lr_manifest_payload()
    assert payload["matrix_kind"] == "FORMAL_PER_DATASET_LR_AE"
    assert payload["target_learning_rates"] == {
        "D1": "3e-5",
        "D4": "3e-5",
        "D5": "3e-4",
        "D6": "3e-5",
    }
    assert "learning_rate_grid" not in payload
    assert payload["stage_a_learning_rate"] == "1e-4"
    assert payload["target_mapping"]["D4"] == "166/258"
