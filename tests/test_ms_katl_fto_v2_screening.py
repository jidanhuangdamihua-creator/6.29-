"""Tests for the FUSED_TARGET_ONLY_V2 training recipe (Huber + CosineDecay + EMA)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401  (must be imported before tensorflow)
import tensorflow as tf

from src.transfer_methods.ms_katl_fto_v2_screening import (
    V2_CANONICAL_BOUNDARY_LR,
    V2_METHOD_DISPLAY,
    V2_TRAINING_RECIPE,
    assert_v2_exploratory_arms_share_stage_a,
    assert_v2_lr_arms_share_stage_a,
    attach_boundary_deltas,
    attach_v1_deltas,
    extra_lr_recommendation,
    format_exploratory_lr,
    format_extra_lr_boundary_summary,
    load_v1_reference_runs,
    load_v2_reference_runs,
    parse_exploratory_lr,
    resolve_extra_lrs,
    v2_architecture,
    v2_architecture_digest,
    v2_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    MSKATLLite,
    compute_knn_prior,
    model_weight_digest,
    train_ms_katl_head,
)
from src.transfer_methods.ms_katl_mechanism_ablation import validate_training_recipe
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    FUSED_TARGET_ONLY_CONTRACT,
    LR_GRID,
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    ScreeningContractError,
    architecture_digest as v1_architecture_digest,
    format_lr,
)

D_MODEL = 6
N = 12
K = 2


def _tiny_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, D_MODEL)).astype(np.float32)
    f_t = [rng.normal(size=(N, D_MODEL)).astype(np.float32) for _ in range(K)]
    y = rng.normal(size=(N,)).astype(np.float32)
    y[0] += 25.0  # outlier so Huber and MSE genuinely diverge
    f_s_pools = [rng.normal(size=(16, D_MODEL)).astype(np.float32) for _ in range(K)]
    prior = compute_knn_prior([0.5, 1.0])
    return x, f_t, y, f_s_pools, prior


def _train(seed: int = 3, epochs: int = 3, learning_rate: float = 1e-2, **recipe):
    x, f_t, y, f_s_pools, prior = _tiny_data()
    tf.keras.utils.set_random_seed(seed)
    model = MSKATLLite(
        d_model=D_MODEL,
        num_sources=K,
        transfer_branch_enabled=False,
        gate_mode="zero",
    )
    summary = train_ms_katl_head(
        model,
        x_train=x,
        f_t_train=f_t,
        y_train=y,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        target_extractor=None,
        epochs=epochs,
        batch_size=4,
        learning_rate=learning_rate,
        seed=seed,
        use_adversarial=False,
        **recipe,
    )
    return model, summary


# -------------------------------------------------------------------
# Training recipe wiring
# -------------------------------------------------------------------

def test_default_recipe_reports_v1_behaviour():
    _model, summary = _train()
    recipe = summary["training_recipe"]
    assert recipe["loss_name"] == "mse"
    assert recipe["lr_schedule"] == "constant"
    assert recipe["use_ema"] is False
    assert recipe["ema_applied"] is False
    assert recipe["final_learning_rate"] == pytest.approx(recipe["initial_learning_rate"])


def test_huber_loss_changes_training_outcome():
    model_mse, _ = _train()
    model_huber, summary = _train(loss_name="huber", huber_delta=1.0)
    assert summary["training_recipe"]["loss_name"] == "huber"
    assert summary["training_recipe"]["huber_delta"] == pytest.approx(1.0)
    assert model_weight_digest(model_mse) != model_weight_digest(model_huber)


def test_cosine_schedule_decays_to_zero():
    model_const, _ = _train()
    model_cos, summary = _train(lr_schedule="cosine")
    recipe = summary["training_recipe"]
    assert recipe["lr_schedule"] == "cosine"
    assert recipe["final_learning_rate"] == pytest.approx(0.0, abs=1e-8)
    assert recipe["initial_learning_rate"] == pytest.approx(1e-2)
    assert model_weight_digest(model_const) != model_weight_digest(model_cos)


def test_ema_applied_and_changes_final_weights():
    model_raw, _ = _train()
    model_ema, summary = _train(use_ema=True, ema_momentum=0.99)
    recipe = summary["training_recipe"]
    assert recipe["use_ema"] is True
    assert recipe["ema_applied"] is True
    assert recipe["ema_momentum"] == pytest.approx(0.99)
    assert model_weight_digest(model_raw) != model_weight_digest(model_ema)
    weights = model_ema.get_weights()
    assert all(np.isfinite(w).all() for w in weights)


def test_full_v2_recipe_trains_and_predicts_finite():
    model, summary = _train(**V2_TRAINING_RECIPE)
    recipe = summary["training_recipe"]
    assert recipe["loss_name"] == "huber"
    assert recipe["lr_schedule"] == "cosine"
    assert recipe["ema_applied"] is True
    x, f_t, _y, _pools, prior = _tiny_data()
    from src.transfer_methods.ms_katl_lite import predict_ms_katl

    pred = predict_ms_katl(model, x, f_t, prior)
    assert np.isfinite(pred).all()


def test_invalid_recipe_values_fail_closed():
    with pytest.raises(ValueError):
        _train(loss_name="mae")
    with pytest.raises(ValueError):
        _train(lr_schedule="step")
    with pytest.raises(ValueError):
        _train(loss_name="huber", huber_delta=0.0)
    with pytest.raises(ValueError):
        _train(use_ema=True, ema_momentum=1.5)


def test_validate_training_recipe_rejects_unknown_keys():
    assert validate_training_recipe(None) == {}
    assert validate_training_recipe({}) == {}
    resolved = validate_training_recipe(V2_TRAINING_RECIPE)
    assert resolved == V2_TRAINING_RECIPE
    with pytest.raises(ValueError):
        validate_training_recipe({"loss_name": "huber", "warmup_steps": 10})


# -------------------------------------------------------------------
# V2 contract
# -------------------------------------------------------------------

def test_v2_architecture_identical_to_v1_but_digest_differs():
    arch = v2_architecture()
    assert arch == FUSED_TARGET_ONLY_CONTRACT
    assert v2_architecture_digest() != v1_architecture_digest()
    assert v2_training_recipe() == V2_TRAINING_RECIPE


def test_assert_v2_lr_arms_requires_v2_method():
    base = {
        "target_identity": "166/258",
        "source_identities": [("166", "1")],
        "source_distances": [0.1],
        "source_model_digests": ["a"],
        "source_scaler_digests": ["b"],
        "target_split_digest": {"X_train": "x"},
        "target_scaler_digest": "s",
        "fused_init_digest": "f",
        "architecture_digest": "d",
    }
    rows = [
        {**base, "learning_rate": lr, "method": V2_METHOD_DISPLAY}
        for lr in (3e-5, 1e-4, 3e-4)
    ]
    flags = assert_v2_lr_arms_share_stage_a(rows)
    assert flags["LR_ARMS_FUSED_INIT_MATCH"] == "YES"
    bad = [{**row, "method": V1_METHOD_DISPLAY} for row in rows]
    with pytest.raises(ScreeningContractError):
        assert_v2_lr_arms_share_stage_a(bad)


# -------------------------------------------------------------------
# V1 comparison helpers
# -------------------------------------------------------------------

def test_load_v1_reference_and_attach_deltas(tmp_path):
    csv_path = tmp_path / "screening_all_runs.csv"
    csv_path.write_text(
        "dataset,target_identity,learning_rate,method,run_status,rmse_original,smape_original\n"
        f"D4,166/258,0.0001,{V1_METHOD_DISPLAY},OK,0.4388,29.0230\n"
        f"D4,166/258,3e-05,{V1_METHOD_DISPLAY},OK,1.0982,122.9876\n"
        f"D4,166/258,0.0001,SOMETHING_ELSE,OK,9.9,9.9\n",
        encoding="utf-8",
    )
    index = load_v1_reference_runs(csv_path)
    assert len(index) == 2
    row = {
        "dataset": "D4",
        "target_identity": "166/258",
        "learning_rate": 1e-4,
        "rmse_original": 0.40,
        "smape_original": 25.0,
    }
    out = attach_v1_deltas(row, index)
    assert out["v1_status"] == "FOUND"
    assert out["v1_rmse_original"] == pytest.approx(0.4388)
    assert out["delta_rmse_vs_v1"] == pytest.approx(0.40 - 0.4388)
    assert out["v2_beats_v1_rmse"] is True

    missing = attach_v1_deltas({**row, "learning_rate": 3e-4}, index)
    assert missing["v1_status"] == "MISSING"
    assert missing["delta_rmse_vs_v1"] is None

    extra = attach_v1_deltas({**row, "learning_rate": 1e-3}, index)
    assert extra["v1_status"] == "MISSING"

    assert load_v1_reference_runs(tmp_path / "nope.csv") == {}


# -------------------------------------------------------------------
# Extra-LR exploration (does not mutate canonical LR_GRID)
# -------------------------------------------------------------------

def test_canonical_lr_grid_rejects_1e_3():
    assert LR_GRID == (3e-5, 1e-4, 3e-4)
    assert V2_CANONICAL_BOUNDARY_LR == pytest.approx(3e-4)
    with pytest.raises(ScreeningContractError):
        format_lr(1e-3)


def test_resolve_extra_lrs_outside_canonical_grid():
    assert resolve_extra_lrs(None) == ()
    assert resolve_extra_lrs("1e-3") == (1e-3,)
    assert resolve_extra_lrs("0.001") == (1e-3,)
    assert resolve_extra_lrs("1e-3,3e-3") == (1e-3, 3e-3)
    assert format_exploratory_lr(1e-3) == "1e-3"
    assert format_exploratory_lr(0.001) == "1e-3"
    assert parse_exploratory_lr("1e-03") == pytest.approx(1e-3)
    with pytest.raises(ScreeningContractError):
        parse_exploratory_lr("3e-4")
    with pytest.raises(ScreeningContractError):
        resolve_extra_lrs("1e-4")
    with pytest.raises(ScreeningContractError):
        resolve_extra_lrs("0")
    with pytest.raises(ScreeningContractError):
        resolve_extra_lrs("-1e-3")
    with pytest.raises(ScreeningContractError):
        resolve_extra_lrs("1e-3,1e-3")
    assert LR_GRID == (3e-5, 1e-4, 3e-4)


def test_exploratory_stage_a_assert_allows_single_extra_arm():
    base = {
        "target_identity": "166/258",
        "method": V2_METHOD_DISPLAY,
        "source_identities": [("166", "1")],
        "source_distances": [0.1],
        "source_model_digests": ["a"],
        "source_scaler_digests": ["b"],
        "target_split_digest": {"X_train": "x"},
        "target_scaler_digest": "s",
        "fused_init_digest": "f",
        "architecture_digest": "d",
        "learning_rate": 1e-3,
    }
    flags = assert_v2_exploratory_arms_share_stage_a([base])
    assert flags["LR_ARMS_FUSED_INIT_MATCH"] == "YES"
    with pytest.raises(ScreeningContractError):
        assert_v2_lr_arms_share_stage_a([base])


def test_attach_boundary_deltas_and_recommendation(tmp_path):
    csv_path = tmp_path / "v2_screening_all_runs.csv"
    csv_path.write_text(
        "dataset,target_identity,learning_rate,method,run_status,rmse_original,smape_original\n"
        f"D4,166/258,0.0003,{V2_METHOD_DISPLAY},OK,0.5000,40.0\n"
        f"D4,166/311,0.0003,{V2_METHOD_DISPLAY},OK,0.4200,35.0\n"
        f"D5,48/320682,0.0003,{V2_METHOD_DISPLAY},OK,1.0000,50.0\n",
        encoding="utf-8",
    )
    index = load_v2_reference_runs(csv_path)
    assert len(index) == 3

    better = attach_boundary_deltas(
        {
            "dataset": "D4",
            "target_identity": "166/258",
            "learning_rate": 1e-3,
            "learning_rate_label": "1e-3",
            "rmse_original": 0.40,
            "smape_original": 30.0,
        },
        index,
    )
    assert better["boundary_status"] == "FOUND"
    assert better["boundary_lr"] == "3e-4"
    assert better["boundary_verdict"] == "EXTRA_BETTER"
    assert better["delta_rmse_vs_boundary"] == pytest.approx(0.40 - 0.50)

    worse = attach_boundary_deltas(
        {
            "dataset": "D4",
            "target_identity": "166/311",
            "learning_rate": 1e-3,
            "learning_rate_label": "1e-3",
            "rmse_original": 0.80,
            "smape_original": 60.0,
        },
        index,
    )
    assert worse["boundary_verdict"] == "BOUNDARY_BETTER"

    missing = attach_boundary_deltas(
        {
            "dataset": "D1",
            "target_identity": "1/10",
            "learning_rate": 1e-3,
            "rmse_original": 0.1,
            "smape_original": 1.0,
        },
        index,
    )
    assert missing["boundary_verdict"] == "MISSING"

    rec = extra_lr_recommendation([better, worse, missing])
    assert rec["compared"] == 2
    assert rec["extra_better"] == 1
    assert rec["boundary_better"] == 1
    assert rec["recommendation"] == "MIXED_RESELECT"
    assert rec["canonical_lr_grid_unchanged"] is True
    assert rec["selection_frozen"] is False

    peak = extra_lr_recommendation([worse])
    assert peak["recommendation"] == "BOUNDARY_IS_PEAK"
    headroom = extra_lr_recommendation([better])
    assert headroom["recommendation"] == "GRID_HEADROOM_YES"

    text = format_extra_lr_boundary_summary([better, worse])
    assert "CANONICAL_LR_GRID_UNCHANGED = YES" in text
    assert "SELECTION_FROZEN = NO" in text
    assert "EXTRA_RMSE_WINS  = 1/2" in text
