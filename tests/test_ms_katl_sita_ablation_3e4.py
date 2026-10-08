"""No-training contract tests for the uniform LR=3e-4 paper-target A–E slice."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    EXPECTED_RUNS,
    EXTRA_TARGET_RUNS,
    FROZEN_MATRIX_KIND,
    HORIZONS,
    SEEDS,
    UNIFORM_3E4_CANONICAL_CELLS,
    UNIFORM_3E4_DATASETS,
    UNIFORM_3E4_LEARNING_RATE_LABEL,
    UNIFORM_3E4_MATRIX_KIND,
    UNIFORM_3E4_RUNS,
    UNIFORM_3E4_RUNS_PER_DATASET,
    UNIFORM_3E4_RUNS_PER_DATASET_HORIZON,
    UNIFORM_3E4_SEEDS,
    UNIFORM_3E4_TARGETS,
    V3_LR_MAP,
    AEContractError,
    ae_arm_spec,
    canonical_targets,
    resolve_cell_lr,
    resolve_extra_target_lr,
    resolve_uniform_3e4_lr,
)
from src.transfer_methods.ms_katl_ae_execution import (
    RealCanonicalCellPlan,
    STAGE_A_KWARG_KEYS,
    assert_uniform_3e4_slice,
    enumerate_real_arm_plans,
    enumerate_uniform_3e4_cell_plans,
    plan_formal_stage_a_kwargs,
    stage_a_key,
)
from src.transfer_methods.ms_katl_ae_formal import (
    FormalCallLedger,
    default_formal_authorities,
    default_v3_prepare_context,
    default_v3_prepare_stage_a,
    make_mock_authorities,
    run_ae_formal_body,
)
from src.transfer_methods.ms_katl_ae_live_context import (
    build_stage_a_kwargs,
    validate_stage_a_kwargs_signature,
)
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest
from src.transfer_methods.ms_katl_sita_ablation_3e4 import (
    FORMAL_ABLATION_RUNNER,
    FORMAL_CONFIG,
    SLICE_LABEL,
    UNLOCK_ENV,
    Uniform3e4ContractError,
    assert_formal_ae_identity_frozen,
    assert_output_isolated,
    default_output_dir,
    enumerate_uniform_3e4_identities,
    prepare_uniform_3e4_execute_root,
    relocate_dry_run_snapshot,
    run_uniform_3e4_dry_run,
    run_uniform_3e4_execute,
    validate_uniform_3e4_matrix,
)


WRAPPER = PROJECT_ROOT / "scripts" / "run_ms_katl_sita_ablation_3e4.py"
FORMAL_RUNNER = PROJECT_ROOT / "scripts" / "run_ms_katl_ae_ablation.py"
FROZEN_MANIFEST = PROJECT_ROOT / "configs" / "ms_katl_ae_ablation_manifest.json"


def _fake_uniform_cells():
    cells = []
    for dataset in UNIFORM_3E4_DATASETS:
        target = UNIFORM_3E4_TARGETS[dataset]
        lr_value, lr_label = resolve_uniform_3e4_lr(dataset, target)
        for horizon in HORIZONS:
            for seed in UNIFORM_3E4_SEEDS:
                cells.append(
                    RealCanonicalCellPlan(
                        dataset=dataset,
                        target=target,
                        scenario="without",
                        horizon=int(horizon),
                        seed=int(seed),
                        selected_lr=float(lr_value),
                        selected_lr_label=lr_label,
                        dataset_source="FROZEN_V3_DATA_PROTOCOL",
                        target_identity=target,
                        entity_key=target.replace("/", "_"),
                        target_split_identity="temporal_split_by_ratio_or_dates",
                        target_split_digest="split",
                        source_pool_scope="shared_protocol",
                        source_selection_contract="shared_protocol_selector",
                        source_pool_authority="shared_protocol_selector",
                        knn_diagnostic_sources=("s1", "s2", "s3"),
                        source_ids_status="SOURCE_IDS_DEFERRED_TO_FORMAL_STAGE_A",
                        stage_a_key=stage_a_key(dataset, target, horizon, seed),
                    )
                )
    return cells


def test_frozen_270_identity_untouched():
    flags = assert_formal_ae_identity_frozen(PROJECT_ROOT)
    assert flags["FORMAL_ABLATION_RUNNER"] == FORMAL_ABLATION_RUNNER == "scripts/run_ms_katl_ae_ablation.py"
    assert flags["FORMAL_CONFIG"] == FORMAL_CONFIG == "FUSED_TARGET_ONLY_V3"
    assert SEEDS == (42, 43, 44)
    assert EXPECTED_RUNS == 270
    assert EXTRA_TARGET_RUNS == 45
    assert FROZEN_MATRIX_KIND == "FROZEN_270"
    assert V3_LR_MAP[("D4", "166/311")] == "3e-5"
    assert ("D4", "166/258") not in V3_LR_MAP
    assert ("D4", "166/311") in canonical_targets()
    assert ("D4", "166/258") not in canonical_targets()
    frozen_lr, frozen_label = resolve_cell_lr("D4", "166/311")
    assert frozen_label == "3e-5"
    assert frozen_lr == pytest.approx(3e-5)
    extra_lr, extra_label = resolve_extra_target_lr("D4", "166/258")
    assert extra_label == "3e-5"
    assert extra_lr == pytest.approx(3e-5)
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert payload["expected_runs"] == 270
    assert payload["seeds"] == [42, 43, 44]
    assert payload["v3_lr_map"]["D4"]["target_identity"] == "166/311"
    assert payload["v3_lr_map"]["D4"]["learning_rate"] == "3e-5"
    assert FORMAL_METHODS == (
        "No-TL",
        "SS-TL",
        "MSWA-TL",
        "MSSB-TL",
        "MSML-TL",
        "MSML-TL-RFE",
    )
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST
    spec = {arm: ae_arm_spec(arm) for arm in AE_ARMS}
    assert spec["A"].name == AE_ARM_NAMES["A"] == "TARGET_NATIVE_INIT"
    assert spec["A"].init_mode == "native" and spec["A"].target_backbone_trainable and not spec["A"].revin_enabled
    assert spec["B"].init_mode == "fused" and not spec["B"].target_backbone_trainable and not spec["B"].revin_enabled
    assert spec["C"].init_mode == "fused" and spec["C"].target_backbone_trainable and not spec["C"].revin_enabled
    assert spec["D"].init_mode == "fused" and spec["D"].target_backbone_trainable and spec["D"].revin_enabled
    assert spec["E"].init_mode == "best_source" and spec["E"].target_backbone_trainable and not spec["E"].revin_enabled


def test_uniform_3e4_matrix_is_exactly_300():
    identities = enumerate_uniform_3e4_identities()
    report = validate_uniform_3e4_matrix(identities, project_root=PROJECT_ROOT)
    assert UNIFORM_3E4_CANONICAL_CELLS == 60
    assert UNIFORM_3E4_RUNS == 300
    assert UNIFORM_3E4_RUNS_PER_DATASET == 75
    assert UNIFORM_3E4_RUNS_PER_DATASET_HORIZON == 25
    assert report["DRY_RUN_TOTAL_RUNS"] == 300
    assert report["EXPECTED_TOTAL_RUNS"] == 300
    assert report["NO_D2"] == "YES"
    assert report["NO_D3"] == "YES"
    assert report["ALL_LR_3E4"] == "YES"
    assert {item.dataset for item in identities} == {"D1", "D4", "D5", "D6"}
    assert {(item.dataset, item.target_identity) for item in identities} == {
        ("D1", "1/10"),
        ("D4", "166/258"),
        ("D5", "48/1159414"),
        ("D6", "CA_1/FOODS_3_080"),
    }
    assert {int(item.horizon) for item in identities} == {1, 3, 5}
    assert {int(item.seed) for item in identities} == {42, 43, 44, 45, 46}
    assert {item.arm for item in identities} == set(AE_ARMS)
    assert {item.selected_lr_label for item in identities} == {UNIFORM_3E4_LEARNING_RATE_LABEL}
    assert all(item.selected_lr == pytest.approx(3e-4) for item in identities)


def test_uniform_3e4_lr_overrides_only_stage_b_and_rejects_wrong_targets():
    value, label = resolve_uniform_3e4_lr("D1", "1/10")
    assert label == "3e-4"
    assert value == pytest.approx(3e-4)
    assert resolve_cell_lr("D1", "1/10")[1] == "3e-5"
    with pytest.raises(AEContractError, match="must be 166/258"):
        resolve_uniform_3e4_lr("D4", "166/311")
    with pytest.raises(AEContractError, match="forbids dataset"):
        resolve_uniform_3e4_lr("D2", "1/10")
    with pytest.raises(AEContractError, match="forbids dataset"):
        resolve_uniform_3e4_lr("D3", "10")


def test_output_isolated_from_frozen_ae_roots(tmp_path):
    allowed = assert_output_isolated(
        tmp_path / "sita_ablation_3e4_d1_d4_d5_d6_5seed",
        project_root=PROJECT_ROOT,
    )
    assert "sita_ablation_3e4" in allowed.name
    assert default_output_dir(PROJECT_ROOT).name == "sita_ablation_3e4_d1_d4_d5_d6_5seed"
    with pytest.raises(Uniform3e4ContractError, match="must include sita_ablation_3e4"):
        assert_output_isolated(tmp_path / "20260830_MS_KATL_AE_CORE_ABLATION", project_root=PROJECT_ROOT)
    with pytest.raises(Uniform3e4ContractError, match="protected path"):
        assert_output_isolated(
            tmp_path / "sita_ablation_3e4_MS_KATL_AE_CORE_ABLATION",
            project_root=PROJECT_ROOT,
        )
    with pytest.raises(Uniform3e4ContractError, match="outputs/runs"):
        assert_output_isolated(
            PROJECT_ROOT / "outputs" / "runs" / "sita_ablation_3e4_forbidden",
            project_root=PROJECT_ROOT,
        )


def test_dry_run_without_knn_is_300_and_does_not_train(tmp_path):
    out = tmp_path / "sita_ablation_3e4_dry"
    result = run_uniform_3e4_dry_run(
        output_dir=out,
        project_root=PROJECT_ROOT,
        resolve_real_cells=False,
    )
    assert result["DRY_RUN_TOTAL_RUNS"] == 300
    assert result["EXPECTED_TOTAL_RUNS"] == 300
    assert result["READY_FOR_FULL_RUN"] == "YES"
    assert result["TRAINING_STARTED"] == "NO"
    assert result["CODE_CHANGE_REQUIRED"] == "YES"
    text = result["text"]
    assert "D1_TARGET=1/10" in text
    assert "D4_TARGET=166/258" in text
    assert "D5_TARGET=48/1159414" in text
    assert "D6_TARGET=CA_1/FOODS_3_080" in text
    assert "DATASET_COUNT=4" in text
    assert "HORIZONS=1,3,5" in text
    assert "SEEDS=42,43,44,45,46" in text
    assert "ARMS=A,B,C,D,E" in text
    assert "LEARNING_RATE=3e-4" in text
    assert "EXPECTED_D1_RUNS=75" in text
    assert "EXPECTED_D4_RUNS=75" in text
    assert "EXPECTED_D5_RUNS=75" in text
    assert "EXPECTED_D6_RUNS=75" in text
    assert "EXPECTED_TOTAL_RUNS=300" in text
    assert "DRY_RUN_TOTAL_RUNS=300" in text
    assert "READY_FOR_FULL_RUN=YES" in text
    assert "SITA_MODEL_CHANGED=NO" in text
    assert "ABLATION_DEFINITION_CHANGED=NO" in text
    assert "TARGET_MAPPING_CHANGED=YES" in text
    assert "ONLY_GLOBAL_LR_CHANGED=YES" in text
    assert "FORMAL_RESULTS_OVERWRITTEN=NO" in text
    manifest = json.loads(Path(result["slice_manifest"]).read_text(encoding="utf-8"))
    assert manifest["expected_runs"] == 300
    assert manifest["paper_targets"]["D4"] == "166/258"
    assert manifest["learning_rate"] == "3e-4"
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert frozen["expected_runs"] == 270
    assert (out / "dry_run_manifest.json").is_file()
    assert not (out / "manifest_snapshot.json").is_file()


def test_mock_uniform_3e4_body_runs_300_and_not_270(tmp_path):
    cells = _fake_uniform_cells()
    assert len(cells) == 60
    arms = enumerate_real_arm_plans(cells, expected_runs=300)
    assert len(arms) == 300
    with pytest.raises(AEContractError, match="real arm plans=300 != 270"):
        enumerate_real_arm_plans(cells)
    ledger = FormalCallLedger()
    authorities = make_mock_authorities(ledger=ledger, publish_root=tmp_path)
    result = run_ae_formal_body(
        mock_execution_authorized=True,
        authorities=authorities,
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        cells=cells,
        matrix_kind=UNIFORM_3E4_MATRIX_KIND,
        ledger=ledger,
    )
    assert result["MATRIX_KIND"] == UNIFORM_3E4_MATRIX_KIND
    assert result["CANONICAL_CELLS"] == 60
    assert result["EXPECTED_RUNS"] == 300
    assert result["STAGE_B_CALLS"] == 300
    assert result["PAIR_COMPLETE"] == 60
    assert result["TRAINING_STARTED"] == "NO"
    with pytest.raises(AEContractError, match="formal body matrix drifted|real arm plans=300 != 270"):
        run_ae_formal_body(
            mock_execution_authorized=True,
            authorities=make_mock_authorities(ledger=FormalCallLedger(), publish_root=tmp_path),
            output_root=tmp_path / "frozen",
            project_root=PROJECT_ROOT,
            cells=cells,
            matrix_kind=FROZEN_MATRIX_KIND,
        )


def test_dry_run_snapshot_is_relocated_before_execute(tmp_path):
    out = tmp_path / "sita_ablation_3e4_resume"
    out.mkdir()
    leftover = {
        "label": SLICE_LABEL,
        "matrix_kind": UNIFORM_3E4_MATRIX_KIND,
        "expected_runs": 300,
        "planned_matrix_digest": "dry-run-schema-not-formal",
        "matrix_report": {"DRY_RUN_TOTAL_RUNS": 300},
    }
    (out / "manifest_snapshot.json").write_text(json.dumps(leftover), encoding="utf-8")
    assert relocate_dry_run_snapshot(out) is True
    assert not (out / "manifest_snapshot.json").is_file()
    moved = json.loads((out / "dry_run_manifest.json").read_text(encoding="utf-8"))
    assert moved["label"] == SLICE_LABEL
    run_dir, resume_root = prepare_uniform_3e4_execute_root(out, skip_existing=True, project_root=PROJECT_ROOT)
    assert run_dir == out.resolve()
    assert resume_root is None
    cells = _fake_uniform_cells()
    result = run_ae_formal_body(
        mock_execution_authorized=True,
        authorities=make_mock_authorities(ledger=FormalCallLedger(), publish_root=out),
        output_root=out,
        project_root=PROJECT_ROOT,
        cells=cells,
        matrix_kind=UNIFORM_3E4_MATRIX_KIND,
    )
    assert result["EXPECTED_RUNS"] == 300
    assert result["TRAINING_STARTED"] == "NO"


def test_wrapper_script_exists_and_unlock_is_formal_env():
    assert WRAPPER.is_file()
    assert FORMAL_RUNNER.is_file()
    assert UNLOCK_ENV == "AE_FORMAL_TRAINING_UNLOCK"
    text = WRAPPER.read_text(encoding="utf-8")
    assert "--dry-run" in text
    assert "--execute" in text
    assert "sita_ablation_3e4" in text


def test_uniform_3e4_stage_a_seed_propagates_for_60_cells(monkeypatch):
    """Paper 60-cell / 300-run Stage-A seed contract. No sealed frames, no training."""
    import src.transfer_methods.ms_katl_ae_formal as formal

    cells = _fake_uniform_cells()
    identities = enumerate_uniform_3e4_identities()
    arm_plans = enumerate_real_arm_plans(cells, expected_runs=UNIFORM_3E4_RUNS)
    assert_uniform_3e4_slice(cells, arm_plans)

    assert len(cells) == UNIFORM_3E4_CANONICAL_CELLS == 60
    assert len(arm_plans) == UNIFORM_3E4_RUNS == 300
    assert len(identities) == 300
    assert {cell.dataset for cell in cells} == set(UNIFORM_3E4_DATASETS) == {"D1", "D4", "D5", "D6"}
    assert {int(cell.horizon) for cell in cells} == set(HORIZONS) == {1, 3, 5}
    assert {int(cell.seed) for cell in cells} == set(UNIFORM_3E4_SEEDS) == {42, 43, 44, 45, 46}
    assert "D2" not in {cell.dataset for cell in cells}
    assert "D3" not in {cell.dataset for cell in cells}

    enumerate_src = inspect.getsource(enumerate_uniform_3e4_cell_plans)
    assert "for seed in UNIFORM_3E4_SEEDS" in enumerate_src
    assert "resolve_real_canonical_cell(" in enumerate_src
    execute_src = inspect.getsource(run_uniform_3e4_execute)
    assert "enumerate_uniform_3e4_cell_plans" in execute_src
    assert "run_ae_formal_body(" in execute_src
    assert "UNIFORM_3E4_MATRIX_KIND" in execute_src
    wrapper_src = WRAPPER.read_text(encoding="utf-8")
    assert "run_uniform_3e4_execute(" in wrapper_src
    authorities = default_formal_authorities()
    assert authorities.prepare_context is default_v3_prepare_context
    assert authorities.prepare_stage_a is default_v3_prepare_stage_a
    context_src = inspect.getsource(default_v3_prepare_context)
    assert "build_live_ae_cell_context" in context_src
    kwargs_src = inspect.getsource(build_stage_a_kwargs)
    assert '"seed": int(cell.seed)' in kwargs_src
    prepare_src = inspect.getsource(default_v3_prepare_stage_a)
    assert 'kwargs.get("seed") is None' in prepare_src
    assert 'int(kwargs["seed"]) != int(cell.seed)' in prepare_src
    assert "AE_STAGE_A_AUTHORITY(**kwargs)" in prepare_src
    plan_src = inspect.getsource(plan_formal_stage_a_kwargs)
    assert '"seed": int(cell.seed)' in plan_src
    assert "seed" in STAGE_A_KWARG_KEYS
    seed_param = inspect.signature(prepare_ms_katl_stage_a).parameters["seed"]
    assert seed_param.default is inspect.Parameter.empty
    assert seed_param.kind is inspect.Parameter.KEYWORD_ONLY

    monkeypatch.setattr(formal, "_require_enabled_real_path", lambda: None)
    for cell in cells:
        planned = {
            "source_df": object(),
            "target_df": object(),
            "feature_cols": ["sales"],
            "seed": int(cell.seed),
        }
        validate_stage_a_kwargs_signature(planned)
        assert planned["seed"] == cell.seed
        assert f"|s{int(cell.seed)}|" in cell.stage_a_key
        with pytest.raises(AEContractError, match="missing required prepare_ms_katl_stage_a args"):
            validate_stage_a_kwargs_signature(
                {
                    "source_df": object(),
                    "target_df": object(),
                    "feature_cols": ["sales"],
                }
            )
        with pytest.raises(AEContractError, match="missing stage_a seed"):
            default_v3_prepare_stage_a(
                cell,
                {"stage_a_kwargs": {"source_df": object(), "feature_cols": ["sales"]}},
            )
        with pytest.raises(AEContractError, match="!="):
            default_v3_prepare_stage_a(
                cell,
                {
                    "stage_a_kwargs": {
                        "source_df": object(),
                        "feature_cols": ["sales"],
                        "seed": int(cell.seed) + 1,
                    }
                },
            )
