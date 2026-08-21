"""Mock-driven tests for the formal A–E execution body. No real training."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    AE_FORMAL_EXECUTION_ENABLED,
    FORMAL_TRAINING_UNLOCK_ENV,
    PROTOCOL_VERSION,
    AEContractError,
    AETrainingBlocked,
    assert_formal_training_permitted,
    run_ae_formal_matrix,
)
from src.transfer_methods.ms_katl_ae_execution import (
    enumerate_real_arm_plans,
    enumerate_real_canonical_cell_plans,
    resolve_all_formal_stage_a_input_plans,
)
from src.transfer_methods.ms_katl_ae_formal import (
    AE_METRIC_AUTHORITY,
    AE_PREDICTION_AUTHORITY,
    AE_STAGE_A_AUTHORITY,
    AE_STAGE_B_AUTHORITY,
    MOCK_METRIC,
    MOCK_STATUS,
    FormalCallLedger,
    assert_no_old_v3_result_injection,
    make_mock_authorities,
    planned_matrix_digest,
    run_ae_formal_body,
    run_ae_formal_preflight,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_lite import (
    prepare_ms_katl_stage_a,
    predict_ms_katl_decomposed,
    train_ms_katl_head,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import compute_screening_metrics


@pytest.fixture(scope="module")
def cells():
    return enumerate_real_canonical_cell_plans(PROJECT_ROOT)


def _run_formal(tmp_path, cells, **kwargs):
    ledger = kwargs.pop("ledger", FormalCallLedger())
    fail_arm = kwargs.pop("fail_arm", None)
    live_contexts = kwargs.pop("live_contexts", None)
    authorities = make_mock_authorities(
        ledger=ledger,
        fail_arm=fail_arm,
        publish_root=tmp_path,
        live_contexts=live_contexts,
    )
    result = run_ae_formal_body(
        mock_execution_authorized=True,
        authorities=authorities,
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        cells=cells,
        ledger=ledger,
        **kwargs,
    )
    return result, ledger


def _write_resume_snapshot(root, cells, digest=None, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(exist_ok=True)
    (root / "plans" / "planned_matrix.csv").write_text("dataset,arm\n", encoding="utf-8")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_v3_architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
        "expected_runs": 270,
        "planned_matrix_digest": digest or planned_matrix_digest(enumerate_real_arm_plans(cells)),
    }
    payload.update(overrides)
    (root / "manifest_snapshot.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_formal_body_54_stage_a_calls(cells, tmp_path):
    result, ledger = _run_formal(tmp_path, cells)
    assert result["CANONICAL_CELLS"] == 54
    assert result["CONTEXT_PREP_CALLS"] == 54
    assert result["STAGE_A_CALLS"] == 54
    assert ledger.counts()["CONTEXT_PREP_CALLS"] == 54


def test_formal_body_270_stage_b_calls(cells, tmp_path):
    result, _ = _run_formal(tmp_path, cells)
    assert result["STAGE_B_CALLS"] == 270
    assert result["PREDICT_CALLS"] == 270
    assert result["METRIC_CALLS"] == 270
    assert result["BUNDLE_VALIDATIONS"] == 54
    assert result["PUBLISH_ATTEMPTS"] == 54


def test_each_arm_called_54_times(cells, tmp_path):
    result, _ = _run_formal(tmp_path, cells)
    for arm in AE_ARMS:
        assert result["%s_CALLS" % arm] == 54


def test_formal_body_uses_shared_stage_a(cells, tmp_path):
    result, ledger = _run_formal(tmp_path, cells)
    assert len(ledger.stage_a) == 54
    assert len(set(ledger.stage_a)) == 54
    by_cell = {}
    for key, arm in ledger.stage_b:
        by_cell.setdefault(key, []).append(arm)
    assert all(item == list(AE_ARMS) for item in by_cell.values())
    assert result["STAGE_A_CALLS"] != 270


def test_formal_body_e_uses_rank1(cells, tmp_path):
    result, _ = _run_formal(tmp_path, cells)
    for payload in result["identity_index"].values():
        assert payload["E"]["best"][0] == "S1"
        assert payload["E"]["init_mode"] == "best_source"
        assert payload["A"]["init_mode"] == "native"


def test_formal_bundle_digest_guards(cells, tmp_path):
    result, _ = _run_formal(tmp_path, cells)
    assert all(status == "PAIR_COMPLETE" for status in result["bundles"].values())
    for payload in result["identity_index"].values():
        assert payload["B"]["conv"] == payload["C"]["conv"] == payload["D"]["conv"]
        assert payload["A"]["head"] == payload["E"]["head"]
        assert payload["B"]["stage_a"] == payload["E"]["stage_a"]


def test_partial_arm_failure_marks_pair_incomplete(cells, tmp_path):
    result, _ = _run_formal(tmp_path, cells, fail_arm="E")
    assert all(status == "PAIR_INCOMPLETE" for status in result["bundles"].values())
    assert result["PUBLISH_ATTEMPTS"] == 54
    sample = json.loads(next((tmp_path / "cell_bundles").glob("*.json")).read_text(encoding="utf-8"))
    assert sample["pair_status"] == "PAIR_INCOMPLETE"
    assert sample["arms"]["E"]["status"] == "FAILED"
    assert sample["arms"]["A"]["rmse_original"] is None


def test_atomic_publish_only_after_five_arms(cells, tmp_path):
    result, ledger = _run_formal(tmp_path, cells)
    assert len(ledger.publish) == 54
    for key in ledger.publish:
        arms = [arm for cell, arm in ledger.stage_b if cell == key]
        assert arms == list(AE_ARMS)
    assert result["PUBLISH_ATTEMPTS"] == result["BUNDLE_VALIDATIONS"] == 54


def test_resume_skips_pair_complete(cells, tmp_path):
    first, _ = _run_formal(tmp_path, cells)
    digest = first["planned_matrix_digest"]
    _write_resume_snapshot(tmp_path, cells, digest=digest)
    ledger = FormalCallLedger()
    result, _ = _run_formal(tmp_path, cells, ledger=ledger, resume_root=tmp_path)
    assert result["STAGE_A_CALLS"] == 0
    assert result["STAGE_B_CALLS"] == 0
    assert all(status == "PAIR_COMPLETE" for status in first["bundles"].values())


def test_resume_rejects_manifest_mismatch(cells, tmp_path):
    _write_resume_snapshot(tmp_path, cells, parent_v3_architecture_digest="not-the-v3-digest")
    with pytest.raises(AEContractError, match="resume parent V3 digest mismatch"):
        _run_formal(tmp_path, cells, resume_root=tmp_path)


def test_arm_order_does_not_change_identity(cells, tmp_path):
    forward, _ = _run_formal(tmp_path / "fwd", cells, arm_order=AE_ARMS)
    reverse, _ = _run_formal(tmp_path / "rev", cells, arm_order=tuple(reversed(AE_ARMS)))
    assert forward["identity_index"] == reverse["identity_index"]
    assert set(forward["bundles"]) == set(reverse["bundles"])


def test_d_never_loads_old_v3_rmse():
    source = (PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_ae_formal.py").read_text(encoding="utf-8")
    assert assert_no_old_v3_result_injection(source) == "NO"
    assert AE_STAGE_A_AUTHORITY is prepare_ms_katl_stage_a
    assert AE_STAGE_B_AUTHORITY is train_ms_katl_head
    assert AE_PREDICTION_AUTHORITY is predict_ms_katl_decomposed
    assert AE_METRIC_AUTHORITY is compute_screening_metrics


def test_unlock_without_execute_does_not_train():
    assert AE_FORMAL_EXECUTION_ENABLED is True
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        assert_formal_training_permitted(
            execute=False, environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"}
        )
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_matrix(execute=False, environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"})
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_body(execute=False, environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"})


def test_execute_without_unlock_is_blocked():
    assert AE_FORMAL_EXECUTION_ENABLED is True
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        assert_formal_training_permitted(execute=True, environ={})
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_matrix(execute=True, environ={})
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_body(execute=True, environ={})


def test_formal_preflight_never_trains(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("TRAINING_FORBIDDEN_DURING_FORMAL_PREFLIGHT")

    import src.transfer_methods.ms_katl_lite as lite

    monkeypatch.setattr(lite, "prepare_ms_katl_stage_a", _boom)
    monkeypatch.setattr(lite, "train_ms_katl_head", _boom)
    result = run_ae_formal_preflight(project_root=PROJECT_ROOT, write_artifacts=False)
    assert result["FORMAL_PREFLIGHT"] == "PASS"
    assert result["TRAINING_STARTED"] == "NO"
    assert result["AE_FORMAL_EXECUTION_ENABLED"] == "YES"
    assert result["FORMAL_EXECUTION_CAPABILITY"] == "ENABLED"
    assert result["ACTUAL_EXECUTION_REQUESTED"] == "NO"
    assert result["D_OLD_RESULT_REUSE"] == "NO"
    assert result["REAL_DATA_FORMAL_PRECHECK"] == "PASS"
    assert result["LIVE_CONTEXT_CONSTRUCTION"] == "54/54 PASS"
    assert result["REAL_STAGE_A_KWARGS_OBJECTS"] == "54/54 PASS"
    assert result["STAGE_A_SIGNATURE_COMPATIBILITY"] == "54/54 PASS"
    assert result["REAL_SHARED_TARGET_BUNDLES"] == "54/54 PASS"
    assert result["REAL_STAGE_A_KWARGS_RESOLUTION"] == "54/54 PASS"
    assert "FORMAL_EXECUTION_BLOCKED_BECAUSE_ENABLE_FALSE" not in result["text"]
    assert "outputs/runs/<timestamp>_MS_KATL_AE_CORE_ABLATION" in result["text"]


def test_mock_metrics_are_not_real_results(cells, tmp_path):
    _run_formal(tmp_path, cells)
    for path in (tmp_path / "cell_bundles").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == MOCK_STATUS
        for arm in AE_ARMS:
            assert payload["arms"][arm]["rmse_original"] is None
            assert payload["arms"][arm]["metric_kind"] == MOCK_METRIC


def _load_ae_runner():
    path = PROJECT_ROOT / "scripts" / "run_ms_katl_ae_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ms_katl_ae_ablation_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_default_requires_explicit_mode_and_does_not_train(capsys):
    runner = _load_ae_runner()
    code = runner.main([])
    captured = capsys.readouterr().out
    assert code == 0
    assert "EXPLICIT_MODE_REQUIRED=YES" in captured
    assert "TRAINING_STARTED=NO" in captured
    assert "ACTUAL_EXECUTION_REQUESTED=NO" in captured


def test_cli_execute_without_unlock_is_blocked(capsys, monkeypatch):
    monkeypatch.delenv(FORMAL_TRAINING_UNLOCK_ENV, raising=False)
    runner = _load_ae_runner()
    code = runner.main(["--execute"])
    captured = capsys.readouterr().out
    assert code == 2
    assert "FORMAL_EXECUTION_BLOCKED" in captured
    assert "TRAINING_STARTED=NO" in captured


def test_cli_unlock_without_execute_does_not_train(capsys, monkeypatch):
    monkeypatch.setenv(FORMAL_TRAINING_UNLOCK_ENV, "YES")
    runner = _load_ae_runner()
    code = runner.main(["--preflight"])
    captured = capsys.readouterr().out
    assert code == 0
    assert "TRAINING_STARTED=NO" in captured
    assert "ACTUAL_EXECUTION_REQUESTED=NO" in captured


def test_real_stage_a_kwargs_resolution_is_54_of_54():
    cells = enumerate_real_canonical_cell_plans(project_root=PROJECT_ROOT)
    report = resolve_all_formal_stage_a_input_plans(cells, project_root=PROJECT_ROOT)
    assert report["REAL_DATA_FORMAL_PRECHECK"] == "PASS"
    assert report["REAL_STAGE_A_KWARGS_RESOLUTION"] == "54/54 PASS"
    assert report["resolved_count"] == 54
    assert report["failures"] == []
    assert len(report["plans"]) == 54
    for plan in report["plans"]:
        assert set(plan["stage_a_kwargs"]) >= {
            "source_df",
            "target_df",
            "feature_cols",
            "k",
            "horizon",
            "window_size",
            "learning_rate",
            "source_epochs",
            "batch_size",
            "tau_k",
            "group_cols",
        }
        assert Path(plan["stage_a_kwargs"]["source_df"]["path"]).is_file()
        assert Path(plan["stage_a_kwargs"]["target_df"]["path"]).is_file()
        assert plan["shared_target_bundle_inputs"]["target_identity"]
        assert len(plan["shared_target_bundle_inputs"]["knn_diagnostic_sources"]) == 3
