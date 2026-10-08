"""Stage-A source-training RNG contract for SITA / FUSED_TARGET_ONLY_V3.

No full source CNN training and no formal 60-cell / 300-run execution.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.models.cnn_model import build_base_cnn
from src.transfer_methods.ms_katl_lite import (
    DEFAULT_LOSS_NAME,
    DEFAULT_LR_SCHEDULE,
    DEFAULT_USE_EMA,
    prepare_ms_katl_stage_a,
    reset_stage_a_source_training_rng,
    run_ms_katl_lite,
)
from src.transfer_methods.sita_mechanism_diagnostics_capture import (
    stage_b_diagnostics_train_kwargs,
)

STAGE_A_CALLERS = (
    PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_lite.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_fto_three_track_robustness.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v3_1_calibrated_robustness.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v3_2_seasonal_residual_robustness.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v3_screening.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v2_screening.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_primary_target_lr_screening.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_d4_cross_target.py",
    PROJECT_ROOT / "scripts" / "run_ms_katl_mechanism_ablation.py",
)
KWARGS_CALLERS = (
    PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_sita_source_selection_ablation.py",
    PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_sita_ac_lr_sensitivity.py",
)


def _fixed_sources():
    return {
        "sources": [
            {"source_key": ("e", "1"), "distance": 0.10, "weight": 0.50},
            {"source_key": ("e", "2"), "distance": 0.20, "weight": 0.30},
            {"source_key": ("e", "3"), "distance": 0.40, "weight": 0.20},
        ]
    }


def _tiny_frames():
    dates = pd.date_range("2020-01-01", periods=16, freq="D")
    rows = []
    for item in ("1", "2", "3", "t"):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "entity_id": "e",
                    "item_id": item,
                    "date": date,
                    "sales": float(i + 1),
                    "f0": float(i),
                }
            )
    frame = pd.DataFrame(rows)
    source_df = frame[frame["item_id"] != "t"].copy()
    target_df = frame[frame["item_id"] == "t"].copy()
    return source_df, target_df


def _install_init_only_mocks(monkeypatch, lite_module):
    rng_calls = {"reset": 0, "train": 0}

    def fake_select(self, *args, **kwargs):
        return _fixed_sources()

    def fake_train(
        source_sequence_df,
        feature_cols,
        horizon=1,
        window_size=10,
        learning_rate=0.001,
        source_epochs=3,
        batch_size=16,
        source_key=None,
        transformation_reuse_context=None,
    ):
        rng_calls["train"] += 1
        model = build_base_cnn((int(window_size), len(feature_cols)), learning_rate=learning_rate)
        return {
            "model": model,
            "input_shape": (int(window_size), len(feature_cols)),
            "num_samples": 4,
            "source_key": source_key,
        }

    def fake_scaler(*args, **kwargs):
        scaler = MinMaxScaler()
        scaler.fit([[0.0, 0.0], [1.0, 1.0]])
        tensor = np.zeros((4, 10, 2), dtype=np.float32)
        return scaler, tensor, ["sales", "f0"]

    original_reset = reset_stage_a_source_training_rng

    def counting_reset(seed):
        rng_calls["reset"] += 1
        return original_reset(seed)

    monkeypatch.setattr(lite_module.SourceSelector, "select_top_k_sources", fake_select)
    monkeypatch.setattr(lite_module, "train_source_cnn_for_msml", fake_train)
    monkeypatch.setattr(lite_module, "_source_scaler_and_train_tensor", fake_scaler)
    monkeypatch.setattr(lite_module, "reset_stage_a_source_training_rng", counting_reset)
    monkeypatch.setattr(lite_module, "bind_actual_cnn_source_frame", lambda *args, **kwargs: None)
    return rng_calls


def _run_mocked_stage_a(lite_module, seed: int):
    source_df, target_df = _tiny_frames()
    tf.keras.backend.clear_session()
    return lite_module.prepare_ms_katl_stage_a(
        source_df=source_df,
        target_df=target_df,
        feature_cols=["sales", "f0"],
        k=3,
        horizon=1,
        window_size=10,
        learning_rate=0.001,
        source_epochs=1,
        batch_size=16,
        seed=int(seed),
    )


def _prepare_call_has_seed(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name != "prepare_ms_katl_stage_a":
            continue
        found += 1
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        if "seed" not in keywords:
            return False
    return found > 0


def test_prepare_ms_katl_stage_a_missing_seed_fails_closed():
    source_df, target_df = _tiny_frames()
    with pytest.raises(TypeError, match="seed"):
        prepare_ms_katl_stage_a(
            source_df=source_df,
            target_df=target_df,
            feature_cols=["sales", "f0"],
        )
    with pytest.raises(ValueError, match="explicit seed"):
        prepare_ms_katl_stage_a(
            source_df=source_df,
            target_df=target_df,
            feature_cols=["sales", "f0"],
            seed=None,
        )
    from src.transfer_methods.ms_katl_ae_ablation import AEContractError
    from src.transfer_methods.ms_katl_ae_live_context import validate_stage_a_kwargs_signature

    with pytest.raises(AEContractError, match="seed"):
        validate_stage_a_kwargs_signature(
            {
                "source_df": object(),
                "target_df": object(),
                "feature_cols": ["sales"],
            }
        )


def test_same_seed_reproduces_source_and_fused_digests(monkeypatch):
    import src.transfer_methods.ms_katl_lite as lite

    calls = _install_init_only_mocks(monkeypatch, lite)
    first = _run_mocked_stage_a(lite, 42)
    second = _run_mocked_stage_a(lite, 42)
    assert first.source_weight_digests == second.source_weight_digests
    assert first.fused_params_digest == second.fused_params_digest
    assert first.stage_a_seed == second.stage_a_seed == 42
    assert calls["reset"] == 2
    assert calls["train"] == 6
    assert first.source_weight_digests[0] != first.source_weight_digests[1]
    assert first.source_weight_digests[1] != first.source_weight_digests[2]


def test_different_seeds_change_random_init_identity(monkeypatch):
    import src.transfer_methods.ms_katl_lite as lite

    _install_init_only_mocks(monkeypatch, lite)
    seed_42 = _run_mocked_stage_a(lite, 42)
    seed_43 = _run_mocked_stage_a(lite, 43)
    assert seed_42.source_weight_digests != seed_43.source_weight_digests
    assert seed_42.fused_params_digest != seed_43.fused_params_digest
    assert seed_42.stage_a_seed == 42
    assert seed_43.stage_a_seed == 43


def test_source_selection_identity_independent_of_rng_fix(monkeypatch):
    import src.transfer_methods.ms_katl_lite as lite

    _install_init_only_mocks(monkeypatch, lite)
    seed_42 = _run_mocked_stage_a(lite, 42)
    seed_43 = _run_mocked_stage_a(lite, 43)
    expected_ids = [("e", "1"), ("e", "2"), ("e", "3")]
    expected_distances = [0.10, 0.20, 0.40]
    for artifacts in (seed_42, seed_43):
        assert [tuple(item) for item in artifacts.source_identities] == expected_ids
        assert artifacts.source_distances == expected_distances
        np.testing.assert_allclose(artifacts.source_weights, [0.50, 0.30, 0.20])


def test_stage_a_resets_rng_once_before_source_cnn_loop():
    source = inspect.getsource(prepare_ms_katl_stage_a)
    assert source.count("reset_stage_a_source_training_rng") == 1
    assert source.find("reset_stage_a_source_training_rng") < source.find("for selected in selected_sources")
    assert source.find("for selected in selected_sources") < source.find("train_source_cnn_for_msml")
    loop_start = source.find("for selected in selected_sources")
    assert loop_start != -1
    assert "tf.random.set_seed" not in source[loop_start:]
    assert "set_random_seed" not in source[loop_start:]


def test_run_ms_katl_lite_propagates_cell_seed(monkeypatch):
    import src.transfer_methods.ms_katl_lite as lite

    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop_after_seed_capture")

    monkeypatch.setattr(lite, "prepare_ms_katl_stage_a", fake_prepare)
    source_df, target_df = _tiny_frames()
    with pytest.raises(RuntimeError, match="stop_after_seed_capture"):
        run_ms_katl_lite(
            source_df=source_df,
            target_df=target_df,
            feature_cols=["sales", "f0"],
            seed=46,
        )
    assert captured["seed"] == 46


def test_direct_prepare_callers_explicitly_pass_seed():
    for path in STAGE_A_CALLERS:
        assert _prepare_call_has_seed(path), f"{path} missing seed= in prepare_ms_katl_stage_a call"
    live_src = (PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_ae_live_context.py").read_text(
        encoding="utf-8"
    )
    assert '"seed": int(cell.seed)' in live_src
    for path in KWARGS_CALLERS:
        text = path.read_text(encoding="utf-8")
        assert "prepare_ms_katl_stage_a(**live.stage_a_kwargs)" in text
    runner_src = inspect.getsource(
        __import__("src.experiment.experiment_runner", fromlist=["run_ms_katl_lite_experiment"]).run_ms_katl_lite_experiment
    )
    assert "seed=int(random_state)" in runner_src


def test_diagnostics_disabled_training_protocol_unchanged_except_rng_boundary():
    sig = inspect.signature(prepare_ms_katl_stage_a)
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["seed"].default is inspect.Parameter.empty
    assert stage_b_diagnostics_train_kwargs({}, arm="C") == {}
    assert stage_b_diagnostics_train_kwargs(None, arm="C") == {}
    from src.transfer_methods.ms_katl_lite import train_ms_katl_head

    head_sig = inspect.signature(train_ms_katl_head)
    assert head_sig.parameters["loss_name"].default == DEFAULT_LOSS_NAME == "mse"
    assert head_sig.parameters["lr_schedule"].default == DEFAULT_LR_SCHEDULE == "constant"
    assert head_sig.parameters["use_ema"].default is DEFAULT_USE_EMA is False
    assert head_sig.parameters["on_before_first_target_optimizer_update"].default is None
    assert head_sig.parameters["on_diagnostic_epoch_eval"].default is None
    source = inspect.getsource(prepare_ms_katl_stage_a)
    assert "reset_stage_a_source_training_rng" in source
    assert "source_epochs" in source
    assert "batch_size" in source


def test_rng_repair_contains_no_private_trainability_api():
    import src.transfer_methods.ms_katl_lite as lite

    module_source = (PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_lite.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "_assign_variable_trainable",
        "_assign_layer_variable_trainable",
        "apply_cloned_cnn_trainability",
        "._trainable",
    )
    for token in forbidden:
        assert token not in module_source, f"private trainability API leaked: {token}"

    clone_source = inspect.getsource(lite.clone_cnn_from_snapshot)
    assert "apply_cloned_cnn_trainability" not in clone_source
    assert "._trainable" not in clone_source
