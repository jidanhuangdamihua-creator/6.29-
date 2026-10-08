"""Execution-loop dry tests. No TensorFlow and no 42-cell formal training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostic_execution import (
    FakeDiagnosticCellExecutor,
    FormalDiagnosticCellExecutor,
    RUNTIME_SOURCE_FILES,
    mechanism_rerun_code_digest,
    persist_synthetic_cell_artifacts,
    run_mechanism_diagnostic_cell_loop,
    sita_ablation_3e4_manifest_runtime_role,
)
from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
    CELL_STATUS_FAILED,
    filesystem_safe_id,
    prepare_run_root,
    required_cell_outputs_complete,
)
from src.transfer_methods.sita_mechanism_diagnostic_rerun import (
    enumerate_mechanism_diagnostic_rerun_cells,
    matrix_digest_for_cells,
    mechanism_rerun_config_digest,
    plan_mechanism_diagnostic_rerun,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import MechanismDiagnosticsContractError


def _prepare(tmp_path):
    cells = enumerate_mechanism_diagnostic_rerun_cells()
    cfg = mechanism_rerun_config_digest(cells)
    mat = matrix_digest_for_cells(cells)
    code = mechanism_rerun_code_digest()
    prepare_run_root(
        tmp_path,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
        resume=False,
    )
    return cells, cfg, mat, code


def test_cli_has_real_cell_loop_and_no_not_started_marker():
    cli = (PROJECT_ROOT / "scripts/run_mechanism_diagnostic_rerun.py").read_text(encoding="utf-8")
    execution = (PROJECT_ROOT / "src/transfer_methods/sita_mechanism_diagnostic_execution.py").read_text(
        encoding="utf-8"
    )
    assert "CELL_LOOP_NOT_STARTED" not in cli
    assert "run_mechanism_diagnostic_cell_loop(" in cli
    assert "--only-cell" in cli
    assert "allow_cell_subset=allow_subset" in cli
    assert "class FormalDiagnosticCellExecutor" in execution
    assert "default_v3_train_stage_b" in execution
    assert "default_v3_prepare_context" in execution
    assert "allow_cell_subset: bool = False" in execution


def test_sita_3e4_manifest_is_not_runtime_authority():
    role = sita_ablation_3e4_manifest_runtime_role()
    assert role["plan_only_reads"] == "NO"
    assert role["run_reads"] == "NO"
    assert role["matrix_parsed_from_manifest"] == "NO"
    assert role["consistency_audit_uses"] == "NO"
    assert role["run_manifest_records_digest"] == "NO"
    assert role["runtime_authority"] == "NO"
    files = [
        "scripts/run_mechanism_diagnostic_rerun.py",
        "src/transfer_methods/sita_mechanism_diagnostic_rerun.py",
        "src/transfer_methods/sita_mechanism_diagnostic_execution.py",
        "src/transfer_methods/sita_mechanism_diagnostic_outputs.py",
        "src/transfer_methods/sita_mechanism_diagnostics_capture.py",
    ]
    for rel in files:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        if "ms_katl_sita_ablation_3e4_manifest.json" not in text:
            continue
        assert "json.load(" not in text.split("ms_katl_sita_ablation_3e4_manifest.json")[0][-80:]
        assert ".read_text(" not in text.split("ms_katl_sita_ablation_3e4_manifest.json")[-1][:120]


def test_fake_executor_runs_all_42_and_finalizes(tmp_path):
    cells, cfg, mat, code = _prepare(tmp_path)
    executor = FakeDiagnosticCellExecutor()
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=executor,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    assert result["n_cells"] == 42
    assert result["n_ran"] == 42
    assert result["n_failed"] == 0
    assert result["n_skipped"] == 0
    assert result["finalized"] == "YES"
    assert result["cell_loop_started"] == "YES"
    assert len(executor.calls) == 42
    for cell in cells:
        assert required_cell_outputs_complete(tmp_path, cell.cell_key)
    summaries = tmp_path / "summaries"
    for name in (
        "training_dynamics_summary.csv",
        "error_by_sales_bin.csv",
        "layerwise_parameter_drift.csv",
        "parameter_drift_summary.csv",
        "cka_pairwise_matrix.csv",
        "cka_condition_summary.csv",
        "historical_frozen_comparison.csv",
        "cell_status.csv",
    ):
        assert (summaries / name).is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sita_ablation_3e4_manifest_runtime_authority"] == "NO"
    assert manifest["n_cells"] == 42
    assert manifest["historical_frozen_reproduction_required"] is False
    assert manifest["historical_trend_match_required"] is False
    assert "sita_ablation_3e4_d1_d4_d5_d6_5seed" not in str(tmp_path)


def test_complete_cells_are_skipped_and_failed_cells_resume(tmp_path):
    cells, cfg, mat, code = _prepare(tmp_path)
    fail_key = cells[0].cell_key
    first = FakeDiagnosticCellExecutor(fail_keys=(fail_key,))
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=first,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    assert result["n_failed"] == 1
    assert result["n_ran"] == 41
    assert result["finalized"] == "NO"
    status = json.loads((tmp_path / "status" / f"{filesystem_safe_id(cells[0].cell_key)}.json").read_text(encoding="utf-8"))
    assert status["status"] == CELL_STATUS_FAILED
    kept = tmp_path / "status" / f"{filesystem_safe_id(cells[1].cell_key)}.COMPLETE"
    assert kept.is_file()
    second = FakeDiagnosticCellExecutor()
    resumed = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=second,
        resume=True,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    assert resumed["n_skipped"] == 41
    assert resumed["n_ran"] == 1
    assert resumed["n_failed"] == 0
    assert resumed["finalized"] == "YES"
    assert fail_key in second.calls
    assert cells[1].cell_key not in second.calls
    third = FakeDiagnosticCellExecutor()
    skipped = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=third,
        resume=True,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    assert skipped["n_skipped"] == 42
    assert skipped["n_ran"] == 0
    assert third.calls == []


def test_exception_isolation_does_not_delete_complete_cells(tmp_path):
    cells, cfg, mat, code = _prepare(tmp_path)
    boom = cells[3].cell_key
    executor = FakeDiagnosticCellExecutor(raise_keys=(boom,))
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=cells,
        executor=executor,
        config_digest=cfg,
        matrix_digest=mat,
        code_digest=code,
    )
    assert result["n_failed"] == 1
    assert result["n_ran"] == 41
    assert result["finalized"] == "NO"
    assert required_cell_outputs_complete(tmp_path, cells[0].cell_key)
    failed_status = json.loads(
        (tmp_path / "status" / f"{filesystem_safe_id(boom)}.json").read_text(encoding="utf-8")
    )
    assert failed_status["status"] == CELL_STATUS_FAILED
    assert "traceback" in failed_status


def test_code_digest_mismatch_fail_closed(tmp_path):
    cells, cfg, mat, code = _prepare(tmp_path)
    persist_synthetic_cell_artifacts(tmp_path, cells[0])
    with pytest.raises(MechanismDiagnosticsContractError, match="code digest mismatch"):
        run_mechanism_diagnostic_cell_loop(
            tmp_path,
            cells=cells,
            executor=FakeDiagnosticCellExecutor(),
            resume=True,
            config_digest=cfg,
            matrix_digest=mat,
            code_digest="not-the-code-digest",
        )


def test_cell_loop_rejects_subset_unless_only_cell_flag(tmp_path):
    cells, cfg, mat, code = _prepare(tmp_path)
    with pytest.raises(MechanismDiagnosticsContractError, match="diagnostic matrix drifted"):
        run_mechanism_diagnostic_cell_loop(
            tmp_path,
            cells=cells[:1],
            executor=FakeDiagnosticCellExecutor(),
            config_digest=cfg,
            matrix_digest=mat,
            code_digest=code,
        )


def test_cell_loop_only_cell_smoke_subset_does_not_change_default_matrix(tmp_path):
    from src.transfer_methods.sita_mechanism_diagnostic_rerun import resolve_diagnostic_only_cell

    universe, cfg, mat, code = _prepare(tmp_path)
    smoke = resolve_diagnostic_only_cell("D1-H1-seed42-B", universe)
    assert smoke.dataset == "D1"
    assert smoke.horizon == 1
    assert smoke.seed == 42
    assert smoke.arm == "B"
    result = run_mechanism_diagnostic_cell_loop(
        tmp_path,
        cells=(smoke,),
        executor=FakeDiagnosticCellExecutor(),
        config_digest=mechanism_rerun_config_digest((smoke,)),
        matrix_digest=matrix_digest_for_cells((smoke,)),
        code_digest=code,
        allow_cell_subset=True,
    )
    assert result["n_cells"] == 1
    assert result["n_ran"] == 1
    assert result["n_failed"] == 0
    assert result["finalized"] == "YES"
    assert len(enumerate_mechanism_diagnostic_rerun_cells()) == 42
    assert len(universe) == 42


def test_plan_only_snapshot_gate_and_runtime_files():
    report = plan_mechanism_diagnostic_rerun()
    flags = report["machine_flags"]
    assert FormalDiagnosticCellExecutor is not FakeDiagnosticCellExecutor
    assert flags["REAL_CELL_LOOP_READY"] == "YES"
    assert flags["CELL_LOOP_NOT_STARTED_PRESENT"] == "NO"
    assert flags["42_CELL_EXECUTION_MATRIX_READY"] == "YES"
    assert flags["FINAL_RUNTIME_DEPENDENCY_TRACE_READY"] == "YES"
    assert report["ready_to_snapshot_runtime"] is True
    for rel in RUNTIME_SOURCE_FILES:
        assert (PROJECT_ROOT / rel).is_file()
    assert report["wiring_audit"]["sita_3e4_manifest_runtime_authority"] == "NO"
