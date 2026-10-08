"""Targeted tests for MS-KATL-Lite single-cell mechanism ablation contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401
import tensorflow as tf

from src.models.cnn_model import build_base_cnn
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_lite import (
    CONV_LAYER_NAMES,
    MSKATLLite,
    MSKATLStageAArtifacts,
    build_flatten_feature_extractor,
    clone_cnn_from_snapshot,
    compute_knn_prior,
    digest_weight_arrays,
    freeze_source_cnn,
    fused_params_digest,
    fused_target_init_snapshot,
    raw_target_init_snapshot,
    relative_weight_drift,
    scaler_digest,
    snapshot_named_layer_weights,
    train_ms_katl_head,
)
from src.transfer_methods.msml_tl import (
    fuse_source_models_layerwise,
    get_transferable_layer_names,
)
from src.transfer_methods.ms_katl_mechanism_ablation import (
    CROSS_TARGET_ARM_NAMES,
    MSKATLSharedBundle,
    assert_formal_methods_untouched,
    print_preflight,
    resolve_ablation_arm,
    resolve_cross_target_arms,
    run_ablation_arm,
    run_abc_mechanism_ablation,
    run_six_arm_mechanism_ablation,
    select_stage_b_trainable_variables,
    verify_shared_stage_a_contract,
)


def test_unknown_ablation_arm_fails_closed():
    with pytest.raises(ValueError, match="Unknown ablation_arm"):
        resolve_ablation_arm("not_an_arm")
    with pytest.raises(ValueError, match="Unknown ablation_arm"):
        resolve_ablation_arm("gate_off")


def test_full_current_alias_is_accepted_after_normalization():
    spec = resolve_ablation_arm("FULL_CURRENT")
    assert spec.arm == "full_current"
    assert spec.use_fused_target_init is True
    assert spec.transfer_branch_enabled is True
    assert spec.gate_mode == "learnable"


def test_formal_methods_untouched_by_mechanism_ablation():
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert FORMAL_METHODS == expected
    assert assert_formal_methods_untouched() == expected
    assert "MS-KATL-Lite" not in FORMAL_METHODS


def test_unknown_gate_mode_fails_closed():
    with pytest.raises(ValueError, match="Unknown gate_mode"):
        MSKATLLite(d_model=16, num_sources=2, gate_mode="soft")
    with pytest.raises(ValueError, match="requires gate_mode='zero'"):
        MSKATLLite(d_model=16, num_sources=2, transfer_branch_enabled=False, gate_mode="learnable")
    with pytest.raises(ValueError, match="requires transfer_branch_enabled=False"):
        MSKATLLite(d_model=16, num_sources=2, transfer_branch_enabled=True, gate_mode="zero")


def _tiny_inputs(batch=4, k=2, d_model=16, seed=0):
    rng = np.random.default_rng(seed)
    z_t = rng.normal(size=(batch, d_model)).astype(np.float32)
    f_t = [rng.normal(size=(batch, d_model)).astype(np.float32) for _ in range(k)]
    prior = compute_knn_prior([0.5, 1.2][:k])
    return z_t, f_t, prior


def test_bc_transfer_branch_disabled_equals_target_head():
    z_t, f_t, prior = _tiny_inputs()
    model = MSKATLLite(
        d_model=16, num_sources=2, d_adapter=8, d_k=8,
        transfer_branch_enabled=False, gate_mode="zero",
    )
    prior_b = tf.constant(np.repeat(prior.reshape(1, -1), 4, axis=0), dtype=tf.float32)
    y, alpha, g = model(
        tf.constant(z_t), [tf.constant(f) for f in f_t], prior_b, training=False
    )
    y_target = model.head_target(tf.constant(z_t))
    np.testing.assert_allclose(y.numpy(), y_target.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g.numpy(), np.zeros((4, 1), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(alpha.numpy(), np.zeros((4, 2), dtype=np.float32), atol=1e-6)


def test_e_gate_is_identically_one():
    z_t, f_t, prior = _tiny_inputs()
    model = MSKATLLite(
        d_model=16, num_sources=2, d_adapter=8, d_k=8,
        transfer_branch_enabled=True, gate_mode="one",
    )
    prior_b = tf.constant(np.repeat(prior.reshape(1, -1), 4, axis=0), dtype=tf.float32)
    decomposed = model.forward_decomposed(
        tf.constant(z_t), [tf.constant(f) for f in f_t], prior_b, training=False
    )
    np.testing.assert_allclose(decomposed["gate"].numpy(), np.ones((4, 1), dtype=np.float32), atol=1e-6)
    expected = decomposed["target_prediction"] + decomposed["transfer_raw"]
    np.testing.assert_allclose(decomposed["y_pred"].numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_decomposition_identity_holds():
    from src.transfer_methods.ms_katl_lite import predict_ms_katl_decomposed

    z_t, f_t, prior = _tiny_inputs()
    model = MSKATLLite(d_model=16, num_sources=2, d_adapter=8, d_k=8)
    out = predict_ms_katl_decomposed(model, z_t, f_t, prior)
    gap = np.max(np.abs(out["y_pred"] - out["target_prediction"] - out["transfer_gated"]))
    assert float(gap) < 1e-5


def _make_source_models(k=2, input_shape=(8, 3), seed=0):
    tf.random.set_seed(seed)
    models = []
    for _ in range(k):
        model = freeze_source_cnn(build_base_cnn(input_shape, learning_rate=1e-3))
        models.append(model)
    return models


def _synthetic_bundle(seed=0, n=8, k=2, window=8, n_features=3) -> MSKATLSharedBundle:
    rng = np.random.default_rng(seed)
    input_shape = (window, n_features)
    source_models = _make_source_models(k=k, input_shape=input_shape, seed=seed)
    layer_names = get_transferable_layer_names(source_models[0])
    weights = [0.6, 0.4][:k]
    fused = fuse_source_models_layerwise(source_models, weights, layer_names)
    scalers = []
    for _ in range(k):
        scaler = MinMaxScaler()
        scaler.fit(rng.normal(size=(20, n_features)))
        scalers.append(scaler)
    token = f"{seed}_{id(source_models[0])}"
    extractors = [
        build_flatten_feature_extractor(model, name=f"synth_src_{token}_{i}")
        for i, model in enumerate(source_models)
    ]
    X_train = rng.normal(size=(n, window, n_features)).astype(np.float32)
    y_train = rng.normal(size=(n,)).astype(np.float32)
    X_val = rng.normal(size=(n, window, n_features)).astype(np.float32)
    y_val = rng.normal(size=(n,)).astype(np.float32)
    X_test = rng.normal(size=(n, window, n_features)).astype(np.float32)
    y_test = rng.normal(size=(n,)).astype(np.float32)
    f_t = {
        "train": [ex(tf.constant(X_train), training=False).numpy() for ex in extractors],
        "val": [ex(tf.constant(X_val), training=False).numpy() for ex in extractors],
        "test": [ex(tf.constant(X_test), training=False).numpy() for ex in extractors],
    }
    f_s_pools = [ex(tf.constant(X_train), training=False).numpy() for ex in extractors]
    d_model = int(f_t["train"][0].shape[-1])
    from src.transfer_methods.ms_katl_lite import snapshot_model_weights

    snapshots = [snapshot_model_weights(m) for m in source_models]
    identities = [("src", str(i)) for i in range(k)]
    distances = [0.4, 1.1][:k]
    stage_a = MSKATLStageAArtifacts(
        selection_result={"sources": []},
        selected_sources=[],
        source_models=source_models,
        source_scalers=scalers,
        source_train_tensors=[X_train] * k,
        source_weights=weights,
        source_distances=distances,
        source_models_info=[
            {"source_key": identities[i], "distance": distances[i], "weight": weights[i], "num_samples": n}
            for i in range(k)
        ],
        source_identities=identities,
        knn_prior=compute_knn_prior(distances),
        input_shape=input_shape,
        fused_params=fused,
        layer_names=list(layer_names),
        source_weight_snapshots=snapshots,
        source_weight_digests=[digest_weight_arrays(s) for s in snapshots],
        source_scaler_digests=[scaler_digest(s) for s in scalers],
        fused_params_digest=fused_params_digest(fused),
        failed_sources=[],
        effective_k=k,
        tau_k=1.0,
        source_extractors=extractors,
    )
    fused_snapshot = fused_target_init_snapshot(input_shape, 1e-3, fused, seed=42)
    raw_snapshot = raw_target_init_snapshot(input_shape, 1e-3, seed=42)
    return MSKATLSharedBundle(
        stage_a=stage_a,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        f_t=f_t,
        f_s_pools=f_s_pools,
        target_scaler=scalers[0],
        target_feature_columns=["f0", "f1", "f2"],
        fused_target_snapshot=fused_snapshot,
        raw_target_snapshot=raw_snapshot,
        fused_target_digest=digest_weight_arrays(fused_snapshot),
        raw_target_digest=digest_weight_arrays(raw_snapshot),
        target_split_digest={
            "X_train": digest_weight_arrays([X_train]),
            "y_train": digest_weight_arrays([y_train]),
            "X_test": digest_weight_arrays([X_test]),
            "y_test": digest_weight_arrays([y_test]),
            "target_scaler": scaler_digest(scalers[0]),
        },
        d_model=d_model,
        learning_rate=1e-3,
        seed=42,
    )


def test_fused_and_raw_init_contracts_are_paired():
    bundle = _synthetic_bundle()
    fused_ids = []
    raw_ids = []
    for name in ("full_current", "fused_target_only", "fused_gate_one", "fused_frozen_backbone"):
        spec = resolve_ablation_arm(name)
        row = run_ablation_arm(spec, bundle, epochs=1, batch_size=8, gamma=0.5, use_adversarial=False)
        fused_ids.append(row["target_conv_init_digest"])
    for name in ("raw_target_only", "raw_plus_transfer"):
        spec = resolve_ablation_arm(name)
        row = run_ablation_arm(spec, bundle, epochs=1, batch_size=8, gamma=0.5, use_adversarial=False)
        raw_ids.append(row["target_conv_init_digest"])
    assert len(set(fused_ids)) == 1
    assert len(set(raw_ids)) == 1
    assert fused_ids[0] != raw_ids[0]


def test_frozen_backbone_conv_drift_is_zero():
    bundle = _synthetic_bundle(seed=1)
    spec = resolve_ablation_arm("fused_frozen_backbone")
    row = run_ablation_arm(spec, bundle, epochs=2, batch_size=8, gamma=0.5, use_adversarial=False)
    for name in CONV_LAYER_NAMES:
        assert row["conv_drifts"][f"{name}_relative_weight_drift"] == pytest.approx(0.0, abs=1e-12)


def test_lambda_adv_zero_excludes_adversarial_from_main_path():
    bundle = _synthetic_bundle(seed=2)
    spec = resolve_ablation_arm("full_current")
    target_cnn, extractor = clone_cnn_from_snapshot(
        bundle.stage_a.input_shape,
        bundle.learning_rate,
        bundle.fused_target_snapshot,
        target_backbone_trainable=True,
        name="adv_probe_extractor",
    )
    head = MSKATLLite(
        d_model=bundle.d_model,
        num_sources=bundle.stage_a.effective_k,
        transfer_branch_enabled=True,
        gate_mode="learnable",
        name="adv_probe_head",
    )
    probe_z = extractor(tf.constant(bundle.X_train[:1], dtype=tf.float32), training=False)
    _ = head(
        probe_z,
        [tf.constant(f[:1], dtype=tf.float32) for f in bundle.f_t["train"]],
        tf.constant(np.repeat(bundle.stage_a.knn_prior.reshape(1, -1), 1, axis=0), dtype=tf.float32),
        training=False,
    )
    trainable = select_stage_b_trainable_variables(head, extractor, spec)
    disc_ids = {id(v) for d in head.discriminators for v in d.trainable_variables}
    assert disc_ids.isdisjoint({id(v) for v in trainable})
    summary = train_ms_katl_head(
        head,
        x_train=bundle.X_train,
        f_t_train=bundle.f_t["train"],
        y_train=bundle.y_train,
        f_s_pools=bundle.f_s_pools,
        knn_prior=bundle.stage_a.knn_prior,
        target_extractor=extractor,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        seed=42,
        use_adversarial=False,
        trainable_variables=trainable,
    )
    assert all(v == 0.0 for v in summary["history"]["lambda"])
    assert all(v == 0.0 for v in summary["history"]["adv_loss"])


def test_six_arms_share_stage_a_identity_order_scaler_and_digests(tmp_path):
    bundle = _synthetic_bundle(seed=3)
    result = run_six_arm_mechanism_ablation(
        bundle,
        epochs=1,
        batch_size=8,
        gamma=0.5,
        lambda_adv=0.0,
        metric_protocol={
            "strict_paper_metrics": False,
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "normalized_minmax_space",
        },
        run_dir=tmp_path,
        cell_meta={
            "dataset": "D4",
            "target_identity": "166/258",
            "mode": "without",
            "horizon": 1,
            "seed": 42,
            "epochs": 1,
            "gamma": 0.5,
            "lambda_adv": 0.0,
        },
    )
    flags = result["flags"]
    assert flags["STAGE_A_SHARED"] == "YES"
    assert flags["SOURCE_IDENTITIES_MATCH"] == "YES"
    assert flags["SOURCE_ORDER_MATCH"] == "YES"
    assert flags["SOURCE_MODEL_DIGESTS_MATCH"] == "YES"
    assert flags["SOURCE_SCALERS_MATCH"] == "YES"
    assert flags["TARGET_SPLIT_MATCH"] == "YES"
    rows = result["arm_rows"]
    assert rows["fused_target_only"]["gate_mode"] == "disabled"
    assert rows["raw_target_only"]["gate_mode"] == "disabled"
    assert rows["fused_gate_one"]["gate_mode"] == "one"
    assert np.allclose(rows["fused_gate_one"]["decomposition"]["gate"], 1.0, atol=1e-6)
    assert rows["full_current"]["inference_gate_off"] is not None
    assert (tmp_path / "results" / "mechanism_ablation_results.csv").exists()
    assert (tmp_path / "predictions" / "full_current.csv").exists()
    verify_shared_stage_a_contract(list(rows.values()))


def test_lambda_adv_nonzero_fails_closed():
    bundle = _synthetic_bundle(seed=4)
    with pytest.raises(ValueError, match="lambda_adv=0"):
        run_six_arm_mechanism_ablation(bundle, epochs=1, lambda_adv=0.1)


def test_preflight_text_matches_fixed_cell_contract(capsys):
    text = print_preflight()
    assert "dataset=D4" in text
    assert "target=166/258" in text
    assert "mode=without" in text
    assert "horizon=1" in text
    assert "seed=42" in text
    assert "arms=6" in text
    assert "shared_stage_a=yes" in text
    captured = capsys.readouterr()
    assert "shared_stage_a=yes" in captured.out


def test_relative_weight_drift_zero_on_identical_snapshots():
    model = build_base_cnn((8, 3), learning_rate=1e-3)
    snap = snapshot_named_layer_weights(model, CONV_LAYER_NAMES)
    for name in CONV_LAYER_NAMES:
        assert relative_weight_drift(snap[name], snap[name]) == pytest.approx(0.0, abs=1e-12)


def test_cross_target_arms_reject_unknown_and_def_arms():
    with pytest.raises(ValueError, match="Unknown ablation_arm"):
        resolve_cross_target_arms(["full_current", "fused_target_only", "not_an_arm"])
    with pytest.raises(ValueError, match="forbids D/E/F"):
        resolve_cross_target_arms(["full_current", "fused_target_only", "fused_gate_one"])
    with pytest.raises(ValueError, match="exactly"):
        resolve_cross_target_arms(["full_current", "fused_target_only"])
    assert resolve_cross_target_arms(["FULL_CURRENT", "fused_target_only", "raw_target_only"]) == CROSS_TARGET_ARM_NAMES


def test_abc_arms_share_stage_a_and_keep_bc_contracts():
    bundle = _synthetic_bundle(seed=5)
    result = run_abc_mechanism_ablation(
        bundle,
        epochs=1,
        batch_size=8,
        gamma=0.5,
        lambda_adv=0.0,
        metric_protocol={
            "strict_paper_metrics": False,
            "current_metric_space": "normalized_minmax_space",
            "paper_metric_space": "normalized_minmax_space",
        },
        cell_meta={"dataset": "D4", "target_identity": "166/258", "mode": "without", "horizon": 1, "seed": 42},
    )
    rows = result["arm_rows"]
    assert set(rows) == set(CROSS_TARGET_ARM_NAMES)
    assert result["flags"]["STAGE_A_SHARED"] == "YES"
    assert result["flags"]["FUSED_TARGET_INIT_MATCH"] == "YES"
    assert result["flags"]["RAW_INIT_DISTINCT_FROM_FUSED"] == "YES"
    assert rows["full_current"]["target_conv_init_digest"] == rows["fused_target_only"]["target_conv_init_digest"]
    assert rows["raw_target_only"]["target_conv_init_digest"] != rows["fused_target_only"]["target_conv_init_digest"]
    assert rows["fused_target_only"]["gate_mode"] == "disabled"
    assert rows["raw_target_only"]["gate_mode"] == "disabled"
    assert rows["fused_target_only"]["transfer_enabled"] is False
    assert np.allclose(rows["fused_target_only"]["decomposition"]["transfer_gated"], 0.0, atol=1e-6)
    assert rows["fused_target_only"]["target_split_digest"] == rows["raw_target_only"]["target_split_digest"]
    verify_shared_stage_a_contract(list(rows.values()), expected_arms=CROSS_TARGET_ARM_NAMES)
    assert "INIT_EFFECT" in result["effects"]
    assert "TRANSFER_EFFECT" in result["effects"]
    assert "UNFREEZE_EFFECT" not in result["effects"]
