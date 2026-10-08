"""Real sealed D1-D6 live-context tests. No Stage-A / Stage-B training."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.transfer_methods.ms_katl_ae_ablation import AE_ARMS, EXPECTED_CANONICAL_CELLS, EXPECTED_RUNS
from src.transfer_methods.ms_katl_ae_execution import enumerate_real_canonical_cell_plans
from src.transfer_methods.ms_katl_ae_formal import FormalCallLedger, make_mock_authorities, run_ae_formal_body
from src.transfer_methods.ms_katl_ae_live_context import (
    CONTEXT_READY,
    construct_all_live_ae_contexts,
    representative_trace,
)


@pytest.fixture(scope="module")
def cells():
    return enumerate_real_canonical_cell_plans(PROJECT_ROOT)


@pytest.fixture(scope="module")
def live_report(cells):
    return construct_all_live_ae_contexts(cells, project_root=PROJECT_ROOT)


def test_live_context_construction_is_54_of_54(live_report):
    assert live_report["ok"] is True
    assert live_report["LIVE_CONTEXT_CONSTRUCTION"] == "54/54 PASS"
    assert live_report["REAL_STAGE_A_KWARGS_OBJECTS"] == "54/54 PASS"
    assert live_report["STAGE_A_SIGNATURE_COMPATIBILITY"] == "54/54 PASS"
    assert live_report["REAL_SHARED_TARGET_BUNDLES"] == "54/54 PASS"
    assert len(live_report["contexts"]) == EXPECTED_CANONICAL_CELLS
    assert live_report["failures"] == []
    for dataset in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert live_report["dataset_status"][dataset] == "PASS"


def test_live_contexts_have_real_frames_and_null_metrics(live_report):
    for ctx in live_report["contexts"]:
        assert ctx.status == CONTEXT_READY
        assert ctx.source_df is not None and not ctx.source_df.empty
        assert ctx.target_df is not None and not ctx.target_df.empty
        assert ctx.stage_a_key == ctx.target_bundle_key
        assert ctx.shared_target_bundle.fused_target_slot == "RESERVED_FOR_STAGE_A"
        assert ctx.rmse_original is None
        assert ctx.mae_original is None
        assert ctx.smape_original is None
        assert ctx.scenario == "without"
        assert ctx.source_pool_authority == "shared_protocol_selector"
        assert ctx.shared_target_bundle.target_test_excluded == "YES"
        assert ctx.stage_a_kwargs["seed"] == ctx.seed


def test_d4_representative_live_trace(live_report):
    matches = [
        ctx
        for ctx in live_report["contexts"]
        if ctx.dataset == "D4" and ctx.horizon == 1 and ctx.seed == 42
    ]
    assert len(matches) == 1
    trace = representative_trace(matches[0])
    assert "166/311" in trace["target_identity"]
    assert Path(trace["sealed_source_path"]).is_file()
    assert Path(trace["sealed_target_path"]).is_file()
    assert trace["STOP"] == "BEFORE prepare_ms_katl_stage_a"
    assert "prepare_ms_katl_stage_a" not in str(trace["stage_a_kwargs_keys"])
    assert "seed" in trace["stage_a_kwargs_keys"]
    assert matches[0].stage_a_kwargs["seed"] == matches[0].seed == 42


def test_real_context_mock_formal_body_topology(cells, live_report, tmp_path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("TRAINING_FORBIDDEN_IN_LIVE_CONTEXT_ACCEPTANCE")

    import src.transfer_methods.ms_katl_lite as lite
    import tensorflow as tf

    monkeypatch.setattr(lite, "prepare_ms_katl_stage_a", _boom)
    monkeypatch.setattr(lite, "train_ms_katl_head", _boom)
    monkeypatch.setattr(tf.keras.Model, "fit", _boom)

    live_map = {ctx.stage_a_key: ctx for ctx in live_report["contexts"]}
    assert len(live_map) == EXPECTED_CANONICAL_CELLS
    ledger = FormalCallLedger()
    authorities = make_mock_authorities(
        ledger=ledger,
        publish_root=tmp_path,
        live_contexts=live_map,
    )
    result = run_ae_formal_body(
        mock_execution_authorized=True,
        authorities=authorities,
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        cells=cells,
        ledger=ledger,
    )
    assert result["CONTEXT_PREP_CALLS"] == EXPECTED_CANONICAL_CELLS
    assert result["STAGE_A_CALLS"] == EXPECTED_CANONICAL_CELLS
    assert result["STAGE_B_CALLS"] == EXPECTED_RUNS
    for arm in AE_ARMS:
        assert result["%s_CALLS" % arm] == EXPECTED_CANONICAL_CELLS
