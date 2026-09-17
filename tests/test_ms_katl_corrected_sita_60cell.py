"""Mock/orchestration-only contracts for the corrected formal SITA runner."""

from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_sita_corrected_60cell as runner
from src.transfer_methods import ms_katl_corrected_sita_60cell as contract


def _digests() -> dict[str, str]:
    return {
        "matrix_digest": "matrix",
        "protocol_digest": "protocol",
        "critical_runtime_digest": "runtime",
        "working_tree_protocol_digest": "working",
    }


def _complete_authority(cell: contract.CellPlan) -> dict:
    return {
        "seed": cell.seed,
        "source_identities": [["source", 1]],
        "source_distances": [0.1],
        "source_weights": [1.0],
        "source_model_digests": ["source-digest"],
        "fused_init_digest": "fused-digest",
        "rmse_original": 1.0,
        "smape_original": 2.0,
        "prediction_artifact_path": "/tmp/mock-predictions.csv",
        "architecture_digest": "architecture",
        "training_recipe": {"use_ema": True},
        "ema_applied": True,
    }


def test_formal_matrix_is_exact_60_and_canonical() -> None:
    manifest = contract.load_manifest(contract.PROJECT_ROOT)
    cells = contract.enumerate_cells(manifest)
    assert len(cells) == 60
    assert len({cell.cell_key for cell in cells}) == 60
    assert [cell.dataset for cell in cells].count("D1") == 15
    assert [cell.dataset for cell in cells].count("D4") == 15
    assert [cell.dataset for cell in cells].count("D5") == 15
    assert [cell.dataset for cell in cells].count("D6") == 15
    assert [cell.horizon for cell in cells].count(1) == 20
    assert [cell.horizon for cell in cells].count(3) == 20
    assert [cell.horizon for cell in cells].count(5) == 20
    assert [cell.seed for cell in cells].count(42) == 12
    assert [cell.seed for cell in cells].count(46) == 12
    assert all(cell.seed == cell.matrix_row()["stage_a_seed"] for cell in cells)


def test_corrected_targets_and_target_lr_mapping_are_fail_closed() -> None:
    manifest = contract.load_manifest(contract.PROJECT_ROOT)
    contract.assert_formal_target_authority(manifest)
    assert manifest["targets"] == {
        "D1": "1/10",
        "D4": "166/258",
        "D5": "48/1159414",
        "D6": "CA_1/FOODS_3_080",
    }
    assert manifest["target_learning_rates"] == {
        "D1": "3e-5",
        "D4": "3e-5",
        "D5": "3e-4",
        "D6": "3e-5",
    }
    bad = dict(manifest)
    bad["targets"] = dict(manifest["targets"], D4="166/311")
    with pytest.raises(contract.CorrectedSITAContractError, match="166/258"):
        contract.assert_formal_target_authority(bad)


def test_sita_only_scope_and_no_reuse_or_training_authority_in_plan() -> None:
    source = inspect.getsource(contract.execute_authority_cell)
    assert contract.PAPER_METHOD_NAME == "SITA"
    assert contract.RUNTIME_METHOD_ID == "FUSED_TARGET_ONLY_V3"
    for forbidden in ("V3.1", "V3.2", "A-E", "diagnostic", "source_selection", "checkpoint", "load_weights"):
        assert forbidden not in source
    assert not any(name.endswith("_CACHE") for name in vars(contract))

    original = runner.execute_authority_cell
    runner.execute_authority_cell = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("TRAINING_CALLED")
    )
    try:
        assert runner.main([]) == 0
    finally:
        runner.execute_authority_cell = original


def test_corrected_execution_uses_src_frame_authority_not_three_track_launcher() -> None:
    source = inspect.getsource(contract.execute_authority_cell)
    assert "scripts.run_ms_katl_fto_three_track_robustness" not in source
    assert "assemble_v3_cell_frames" in source


def test_execute_authority_preserves_exact_cell_context_and_formal_lr(monkeypatch, tmp_path: Path) -> None:
    from src.transfer_methods import ms_katl_fto_three_track_robustness as v3_contract
    from src.transfer_methods import ms_katl_v3_frame_assembly as frame_authority
    from src.utils import d4_d6_runtime

    cell = contract.CellPlan("D4", "166/258", horizon=5, seed=45)
    captured: dict[str, object] = {}
    prepared = {"parsed": {"prediction_dirname": "target-166-258"}}

    def fake_assemble(dataset: str, target: str, *, horizon: int, seed: int, project_root: Path, mode: str) -> dict:
        captured["frame_call"] = (dataset, target, horizon, seed, project_root, mode)
        return prepared

    def fake_stage_a(**kwargs):
        captured["stage_a_call"] = kwargs
        return SimpleNamespace(stage_a_seed=cell.seed, source_weights=[]), object()

    def fake_spec():
        return SimpleNamespace(
            key="v3",
            method_display=contract.RUNTIME_METHOD_ID,
        )

    def fake_arm(**kwargs):
        captured["arm_call"] = kwargs
        return {"status": "OK", "seed": cell.seed}

    monkeypatch.setattr(frame_authority, "assemble_v3_cell_frames", fake_assemble)
    monkeypatch.setattr(v3_contract, "v3_arm_spec", fake_spec)
    monkeypatch.setattr(d4_d6_runtime, "load_default_metric_protocol", lambda _: {"name": "mock"})
    monkeypatch.setattr(contract, "_run_stage_a", fake_stage_a)
    monkeypatch.setattr(contract, "_run_arm", fake_arm)

    result = contract.execute_authority_cell(
        cell,
        attempt_path=tmp_path,
        project_root=contract.PROJECT_ROOT,
        manifest={"source_epochs": 50, "target_epochs": 50, "batch_size": 16},
    )

    assert captured["frame_call"] == (
        "D4",
        "166/258",
        5,
        45,
        contract.PROJECT_ROOT,
        "without",
    )
    stage_a_call = captured["stage_a_call"]
    assert stage_a_call["cell"] is prepared
    assert stage_a_call["horizon"] == 5
    assert stage_a_call["seed"] == 45
    assert stage_a_call["epochs"] == 50
    assert stage_a_call["batch_size"] == 16
    identity = captured["arm_call"]["identity"]
    assert identity.dataset == "D4"
    assert identity.target_identity == "166/258"
    assert identity.horizon == 5
    assert identity.seed == 45
    assert identity.selected_lr_label == "3e-5"
    assert identity.selected_lr == 3e-5
    assert result["stage_a_seed"] == 45


def test_corrected_runtime_critical_files_exclude_historical_launcher() -> None:
    assert "scripts/run_ms_katl_fto_three_track_robustness.py" not in contract.CRITICAL_RUNTIME_FILES


def test_new_root_plan_does_not_create_repo_output_root() -> None:
    requested = Path("outputs") / "_corrected_sita_unit_plan_only_root"
    resolved = contract.resolve_output_root(contract.PROJECT_ROOT, requested)
    assert resolved == (contract.PROJECT_ROOT / requested).resolve()
    assert not resolved.exists()


def test_output_isolation_rejects_historical_roots() -> None:
    with pytest.raises(contract.CorrectedSITAContractError):
        contract.assert_output_isolation(
            contract.PROJECT_ROOT / "outputs" / "sita_all_targets_d4_d6_5seed",
            contract.PROJECT_ROOT,
        )
    with pytest.raises(contract.CorrectedSITAContractError):
        contract.assert_output_isolation(contract.PROJECT_ROOT / "outputs", contract.PROJECT_ROOT)
    contract.assert_output_isolation(
        contract.PROJECT_ROOT / "outputs" / "sita_corrected_unit_new_root",
        contract.PROJECT_ROOT,
    )


def test_failed_attempt_is_preserved_and_resume_creates_next_attempt(tmp_path: Path) -> None:
    cell = contract.enumerate_cells()[0]
    first_action, first = contract.prepare_cell_attempt(
        tmp_path, cell, resume=False, digests=_digests()
    )
    assert first_action == "RUN"
    assert first is not None
    contract.finalize_attempt(first, status="FAILED", error=RuntimeError("mock failure"))
    second_action, second = contract.prepare_cell_attempt(
        tmp_path, cell, resume=True, digests=_digests()
    )
    assert second_action == "RUN"
    assert second is not None and second.name == "attempt_002"
    assert (first / "status.json").is_file()
    assert json.loads((first / "status.json").read_text())["status"] == "FAILED"


def test_interrupted_attempt_is_preserved_and_resume_creates_next_attempt(tmp_path: Path) -> None:
    cell = contract.enumerate_cells()[0]
    _, first = contract.prepare_cell_attempt(tmp_path, cell, resume=False, digests=_digests())
    assert first is not None
    status_path = first / "status.json"
    status = json.loads(status_path.read_text())
    status["pid"] = 99999999
    status_path.write_text(json.dumps(status), encoding="utf-8")
    _, second = contract.prepare_cell_attempt(tmp_path, cell, resume=True, digests=_digests())
    assert second is not None and second.name == "attempt_002"
    assert json.loads(status_path.read_text())["status"] == "INTERRUPTED"


def test_complete_attempt_is_skipped_without_overwrite(tmp_path: Path) -> None:
    cell = contract.enumerate_cells()[0]
    digests = _digests()
    _, attempt = contract.prepare_cell_attempt(tmp_path, cell, resume=False, digests=digests)
    assert attempt is not None
    result = contract.build_result_payload(
        cell=cell,
        attempt_path=attempt,
        authority_payload=_complete_authority(cell),
        digests=digests,
        branch="codex/test",
        head="head",
        start_time="start",
        end_time="end",
    )
    contract.finalize_attempt(attempt, status="COMPLETE", result=result)
    action, same = contract.prepare_cell_attempt(tmp_path, cell, resume=True, digests=digests)
    assert action == "SKIPPED_COMPLETE"
    assert same == attempt
    assert not (attempt.parent / "attempt_002").exists()


@pytest.mark.parametrize("mismatch", ["matrix_digest", "protocol_digest", "critical_runtime_digest"])
def test_resume_digest_mismatch_refuses(mismatch: str, tmp_path: Path) -> None:
    digests = _digests()
    payload = {
        **digests,
        "paper_method_name": "SITA",
        "runtime_method_id": "FUSED_TARGET_ONLY_V3",
        "run_root": str(tmp_path.resolve()),
    }
    payload[mismatch] = "different"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.CorrectedSITAContractError, match="RESUME_REFUSED_DIGEST_MISMATCH"):
        runner._validate_resume_manifest(tmp_path, digests, project_root=contract.PROJECT_ROOT)


def test_result_contract_rejects_other_runtime_method() -> None:
    manifest = contract.load_manifest(contract.PROJECT_ROOT)
    bad = dict(manifest, runtime_method_id="SITA_TCN")
    with pytest.raises(contract.CorrectedSITAContractError):
        contract.assert_formal_target_authority(bad)
