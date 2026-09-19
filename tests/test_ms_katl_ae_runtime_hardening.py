"""Synthetic, no-training tests for corrected A-E runtime hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.ms_katl_ae_formal import (
    AEContractError,
    FormalArmRuntime,
    build_formal_run_provenance,
    default_v3_publish,
    finalize_formal_cell_attempt,
    formal_inference_model_digest,
    prepare_formal_cell_attempt,
    serialize_formal_arm_runtime,
    validate_formal_resume_provenance,
)
from src.transfer_methods import ms_katl_ae_formal as formal_module
from src.transfer_methods.ms_katl_ae_ablation import ae_arm_spec
from src.transfer_methods.ms_katl_sita_ablation_3e4 import (
    enumerate_uniform_3e4_identities,
    select_uniform_3e4_cell,
    validate_uniform_3e4_matrix,
)


def _cell():
    return SimpleNamespace(
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        scenario="without",
        stage_a_key="D1|1/10|h1|s42|without",
        target_split_digest={"train": "train", "val": "val", "test": "test"},
        selected_lr=3e-4,
        selected_lr_label="3e-4",
    )


def _digests():
    return {
        "matrix_digest": "matrix",
        "protocol_digest": "protocol",
        "critical_runtime_digest": "runtime",
        "working_tree_protocol_digest": "working",
    }


def test_failed_attempt_is_preserved_and_next_attempt_is_allocated(tmp_path):
    first_action, first = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=False, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert first_action == "RUN"
    assert first is not None
    finalize_formal_cell_attempt(first, status="FAILED", error=RuntimeError("synthetic"))

    second_action, second = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert second_action == "RUN"
    assert second is not None and second.name == "attempt_002"
    assert json.loads((first / "status.json").read_text())["status"] == "FAILED"
    assert (first / "result.json").is_file()
    assert (first / "provenance.json").is_file()


def test_interrupted_attempt_is_preserved_and_complete_is_skipped(tmp_path):
    _, first = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=False, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert first is not None
    status_path = first / "status.json"
    status = json.loads(status_path.read_text())
    status["pid"] = 99999999
    status_path.write_text(json.dumps(status), encoding="utf-8")

    _, second = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert json.loads(status_path.read_text())["status"] == "INTERRUPTED"
    assert second is not None and second.name == "attempt_002"
    finalize_formal_cell_attempt(second, status="COMPLETE", result={"status": "COMPLETE"})
    action, skipped = prepare_formal_cell_attempt(
        tmp_path, _cell(), resume=True, digests=_digests(), branch="codex/youhua", head="head"
    )
    assert action == "SKIPPED_COMPLETE"
    assert skipped == second


@pytest.mark.parametrize("field", ["matrix_digest", "protocol_digest", "critical_runtime_digest"])
def test_resume_digest_mismatch_is_refused(tmp_path, field):
    root = tmp_path / "root"
    root.mkdir()
    payload = {**_digests(), "matrix_kind": "UNIFORM_3E4_PAPER_5SEED", "expected_runs": 300}
    (root / "manifest_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")
    changed = dict(payload, **{field: "different"})
    with pytest.raises(Exception, match="digest mismatch"):
        validate_formal_resume_provenance(root, changed)


def test_root_provenance_contains_required_hardening_fields():
    payload = build_formal_run_provenance(
        PROJECT_ROOT,
        matrix_digest="matrix",
        matrix_kind="UNIFORM_3E4_PAPER_5SEED",
        expected_runs=300,
        cells=[_cell()],
    )
    for key in (
        "paper_method_name",
        "ablation_protocol_name",
        "branch",
        "HEAD",
        "runtime_commit",
        "matrix_digest",
        "protocol_digest",
        "critical_runtime_digest",
        "working_tree_protocol_digest",
        "dataset_target_map",
        "stage_a_rng_contract",
        "historical_source_weight_reuse",
        "ARM_D_IS_FORMAL_MAIN_SITA",
        "ARM_D_REASON",
    ):
        assert key in payload


def test_single_cell_selector_uses_formal_target_mapping():
    selected = select_uniform_3e4_cell(dataset="D4", horizon=1, seed=42, arm="D")
    assert selected["target"] == "166/258"
    assert selected["arm"] == "D"
    assert selected["stage_a_key"] == "D4|166/258|h1|s42|without"

    with pytest.raises(Exception, match="D4 target must be 166/258"):
        select_uniform_3e4_cell(dataset="D4", horizon=1, seed=42, arm="D", target="166/311")


def _synthetic_formal_runtime(arm="D", *, prediction_digest="prediction", model_digest="model"):
    spec = ae_arm_spec(arm)
    return FormalArmRuntime(
        arm=arm,
        spec=spec,
        init_mode=spec.init_mode,
        source_init_consumed=spec.init_mode != "native",
        stage_a_shared_for_pairing=True,
        head_target_init_digest="head-init",
        target_initial_conv_digest=f"target-init-{arm}",
        fused_init_digest="fused-init",
        best_source_identity=("1", "7"),
        best_source_snapshot_digest="source-1",
        source_identities=[("1", "7"), ("1", "8"), ("1", "2")],
        source_model_digests=["source-1", "source-2", "source-3"],
        source_distances=[0.1, 0.2, 0.3],
        source_weights=[0.5, 0.3, 0.2],
        source_selection_digest="selection",
        stage_a_digest="stage-a",
        fused_params_digest="fused-params",
        selected_lr=3e-4,
        recipe_digest="recipe",
        target_split_digest="split",
        parent_method="FUSED_TARGET_ONLY_V3",
        parent_v3_architecture_digest="architecture",
        prediction_path_parameter_count=10,
        trainable_prediction_path_parameter_count=10,
        frozen_prediction_path_parameter_count=0,
        status="OK",
        rmse_original=1.25,
        smape_original=2.5,
        metric_kind="rmse_original",
        final_formal_model_digest=model_digest,
        formal_weight_variant="EMA",
        prediction_digest=prediction_digest,
        prediction_digest_space="original_sales",
        n_test=3,
    )


def test_real_stage_a_to_view_to_runtime_to_serializer_preserves_existing_digests(monkeypatch):
    artifacts = SimpleNamespace(
        source_identities=[("1", "7"), ("1", "8"), ("1", "2")],
        source_distances=[0.1, 0.2, 0.3],
        source_weights=[0.5, 0.3, 0.2],
        source_weight_digests=["source-1", "source-2", "source-3"],
        fused_params_digest="fused-params",
        effective_k=3,
        stage_a_seed=42,
    )
    stage_a_bridge = SimpleNamespace(
        source_selection_digest="selection_digest_x",
        stage_a_digest="stage_a_digest_x",
    )

    def fake_materialize(context, cell, *, stage_a_artifacts):
        assert stage_a_artifacts is artifacts
        context[formal_module.AE_STAGE_A_VIEW_KEY] = stage_a_bridge
        context[formal_module.AE_SHARED_BUNDLE_KEY] = SimpleNamespace(
            fused_target_digest="fused-target"
        )
        return "head", {}

    monkeypatch.setattr(formal_module, "AE_FORMAL_EXECUTION_ENABLED", True)
    monkeypatch.setattr(formal_module, "AE_STAGE_A_AUTHORITY", lambda **kwargs: artifacts)
    monkeypatch.setattr(formal_module, "materialize_formal_ae_arms", fake_materialize)

    context = {"stage_a_kwargs": {"seed": 42}}
    stage_a = formal_module.default_v3_prepare_stage_a(_cell(), context)
    assert stage_a.source_selection_digest == "selection_digest_x"
    assert stage_a.stage_a_digest == "stage_a_digest_x"

    runtime = _synthetic_formal_runtime("D")
    runtime.source_selection_digest = stage_a.source_selection_digest
    runtime.stage_a_digest = stage_a.stage_a_digest
    payload = formal_module.serialize_formal_arm_runtime(runtime)
    assert payload["source_selection_digest"] == "selection_digest_x"
    assert payload["stage_a_digest"] == "stage_a_digest_x"


def test_formal_evidence_schema_is_additive_deterministic_and_a_to_e_compatible():
    first = {
        arm: serialize_formal_arm_runtime(_synthetic_formal_runtime(arm))
        for arm in "ABCDE"
    }
    second = {
        arm: serialize_formal_arm_runtime(_synthetic_formal_runtime(arm))
        for arm in "ABCDE"
    }

    assert first == second
    for arm, payload in first.items():
        assert payload["arm"] == arm
        assert payload["source_identities"] == [["1", "7"], ["1", "8"], ["1", "2"]]
        assert payload["source_model_digests"] == ["source-1", "source-2", "source-3"]
        assert payload["source_selection_digest"] == "selection"
        assert payload["stage_a_digest"] == "stage-a"
        assert payload["fused_params_digest"] == "fused-params"
        assert payload["final_formal_model_digest"] == "model"
        assert payload["formal_weight_variant"] == "EMA"
        assert payload["prediction_digest"] == "prediction"
        assert payload["prediction_digest_space"] == "original_sales"
        assert payload["n_test"] == 3
        for existing in (
            "best_source_snapshot_digest",
            "fused_init_digest",
            "target_initial_conv_digest",
            "head_target_init_digest",
            "rmse_original",
            "smape_original",
        ):
            assert existing in payload

    changed = serialize_formal_arm_runtime(
        _synthetic_formal_runtime("D", prediction_digest="prediction-changed")
    )
    assert changed["prediction_digest"] != first["D"]["prediction_digest"]


def _complete_runtime_bundle():
    return {arm: _synthetic_formal_runtime(arm) for arm in "ABCDE"}


def _publish_complete_runtime_bundle(tmp_path, runtimes):
    root = tmp_path / "20260919_MS_KATL_AE_CORE_ABLATION"
    default_v3_publish(_cell(), runtimes, "PAIR_COMPLETE", root)
    return json.loads(next((root / "cell_bundles").glob("*.json")).read_text())


@pytest.mark.parametrize("arm", ["C", "D"])
def test_real_complete_c_d_full_evidence_publishes(tmp_path, arm):
    payload = _publish_complete_runtime_bundle(tmp_path, _complete_runtime_bundle())
    runtime = payload["arms"][arm]
    assert len(runtime["source_model_digests"]) == 3
    assert runtime["source_selection_digest"] == "selection"
    assert runtime["stage_a_digest"] == "stage-a"
    assert runtime["final_formal_model_digest"]
    assert runtime["prediction_digest"]
    assert runtime["n_test"] == 3


def test_synthetic_nonexecuted_missing_evidence_remains_json_safe():
    runtime = _synthetic_formal_runtime("C")
    runtime.source_model_digests = []
    runtime.source_selection_digest = None
    runtime.stage_a_digest = None
    runtime.final_formal_model_digest = None
    runtime.prediction_digest = None
    runtime.n_test = None
    payload = serialize_formal_arm_runtime(runtime)
    json.dumps(payload)
    assert payload["source_model_digests"] == []
    assert payload["source_selection_digest"] is None
    assert payload["stage_a_digest"] is None
    assert payload["final_formal_model_digest"] is None
    assert payload["prediction_digest"] is None
    assert payload["n_test"] is None


@pytest.mark.parametrize("field", ["source_model_digests"])
def test_real_complete_missing_source_digests_fails_closed(tmp_path, field):
    runtimes = _complete_runtime_bundle()
    runtimes["C"].source_model_digests = []
    with pytest.raises(AEContractError, match="source_model_digests"):
        _publish_complete_runtime_bundle(tmp_path, runtimes)


@pytest.mark.parametrize("count", [1, 2])
def test_real_complete_partial_source_digests_fail_closed(tmp_path, count):
    runtimes = _complete_runtime_bundle()
    runtimes["D"].source_model_digests = ["source-1"] * count
    with pytest.raises(AEContractError, match="source_model_digests"):
        _publish_complete_runtime_bundle(tmp_path, runtimes)


@pytest.mark.parametrize("field", ["source_selection_digest", "stage_a_digest"])
def test_real_complete_missing_stage_a_evidence_fails_closed(tmp_path, field):
    runtimes = _complete_runtime_bundle()
    setattr(runtimes["C"], field, None)
    with pytest.raises(AEContractError, match=field):
        _publish_complete_runtime_bundle(tmp_path, runtimes)


@pytest.mark.parametrize("field", ["final_formal_model_digest", "prediction_digest"])
def test_real_complete_missing_prediction_or_model_digest_fails_closed(tmp_path, field):
    runtimes = _complete_runtime_bundle()
    setattr(runtimes["C"], field, None)
    with pytest.raises(AEContractError, match=field):
        _publish_complete_runtime_bundle(tmp_path, runtimes)


@pytest.mark.parametrize("n_test", [None, 0, -1, "3", 1.5, True])
def test_real_complete_missing_or_invalid_n_test_fails_closed(tmp_path, n_test):
    runtimes = _complete_runtime_bundle()
    runtimes["D"].n_test = n_test
    with pytest.raises(AEContractError, match="n_test"):
        _publish_complete_runtime_bundle(tmp_path, runtimes)


def test_formal_model_digest_is_deterministic_changes_on_weight_change_and_does_not_mutate_input():
    class _Weights:
        def __init__(self, values):
            self.values = values

        def get_weights(self):
            return [value.copy() for value in self.values]

    target_weights = [np.asarray([1.0, 2.0], dtype=np.float32)]
    head_weights = [np.asarray([[3.0]], dtype=np.float32)]
    constructed = SimpleNamespace(
        target_cnn=_Weights(target_weights),
        head=_Weights(head_weights),
        revin=None,
    )
    before_target = target_weights[0].copy()
    before_head = head_weights[0].copy()

    first = formal_inference_model_digest(constructed)
    assert np.array_equal(target_weights[0], before_target)
    assert np.array_equal(head_weights[0], before_head)

    target_weights[0][0] = 99.0
    second = formal_inference_model_digest(constructed)
    assert second != first


def test_evidence_patch_preserves_uniform_300_run_and_60_stage_a_matrix():
    identities = enumerate_uniform_3e4_identities()
    report = validate_uniform_3e4_matrix(identities, project_root=PROJECT_ROOT)
    assert len(identities) == 300
    assert len({(item.dataset, item.target_identity, item.horizon, item.seed) for item in identities}) == 60
    assert report["EXPECTED_TOTAL_RUNS"] == 300
    assert report["UNIQUE_STAGE_A_KEYS"] == 60
