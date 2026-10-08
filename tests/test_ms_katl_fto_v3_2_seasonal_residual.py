"""Contract tests for MS-KATL FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.baselines.bl5_seasonal_naive import predict_bl5
from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.experiment import run_no_tl_experiment
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods import msml_tl
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    write_cell_artifact as write_json_cell,
)
from src.transfer_methods.ms_katl_fto_v3_2_seasonal_residual import (
    EXPECTED_COMMON_TARGETS,
    EXPECTED_RUNS,
    EXPECTED_V3_LR,
    LAG_DAYS,
    METHOD_ARM,
    METHOD_DISPLAY,
    PARENT_METHOD,
    REQUIRED_CELL_FIELDS,
    RESIDUALIZATION_SCOPE,
    SEASONAL_PRIOR_TYPE,
    V3_1_CALIBRATION_ENABLED,
    V3_SOURCE_FILES,
    V32ContractError,
    V32RunIdentity,
    assert_formal_methods_unchanged_v32,
    assert_mechanism_is_uniform,
    assert_no_baseline_result_consumption,
    assert_not_v3_parent_output_dir,
    assert_parent_fingerprints_unchanged,
    assert_residual_revin_semantics,
    assert_v3_1_not_inherited,
    assert_v3_parent_contract,
    build_entity_sales_lookup,
    build_v3_2_identities,
    classify_cell_artifact,
    denormalize_residual_numpy,
    expected_matrix_counts,
    fingerprint_parent_artifacts,
    format_preflight,
    frozen_v3_lr_map,
    inherited_v3_spec,
    load_v3_2_manifest,
    lookup_lag7_seasonal_prior,
    observable_history,
    parent_artifact_files,
    reconstruct_original_sales,
    residual_original,
    seasonal_prior_from_bl5_record,
    sequence_window_meta,
    v3_2_architecture,
    v3_2_revin_config,
    v3_2_training_recipe,
    v3_robustness_root,
    v3_screening_root,
    write_v3_2_manifest,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import METHOD_NAME, train_ms_katl_head
from src.transfer_methods.ms_katl_mechanism_ablation import run_ablation_arm
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    SCREENING_DATASETS,
)
from src.transfer_methods.revin import RevIN

CONTRACT = PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_fto_v3_2_seasonal_residual.py"
RUNNER = PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v3_2_seasonal_residual_robustness.py"
MANIFEST = PROJECT_ROOT / "configs" / "ms_katl_fto_v3_2_seasonal_residual_manifest.json"
PROTOCOL = PROJECT_ROOT / "src" / "protocols" / "experiment_protocol.py"


def _identity(**overrides) -> V32RunIdentity:
    payload = {
        "dataset": "D1",
        "target_identity": "1/10",
        "horizon": 1,
        "seed": 42,
        "selected_lr": 3e-5,
        "selected_lr_label": "3e-5",
    }
    payload.update(overrides)
    return V32RunIdentity(**payload)


def _ok_payload(identity: V32RunIdentity) -> dict:
    return {
        "status": "OK",
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "horizon": identity.horizon,
        "seed": identity.seed,
        "learning_rate": identity.selected_lr,
        "learning_rate_label": identity.selected_lr_label,
        "parent_method": PARENT_METHOD,
        "parent_architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
        "seasonal_prior_type": SEASONAL_PRIOR_TYPE,
        "seasonal_lag_days": LAG_DAYS,
        "train_residual_sample_count": 20,
        "validation_residual_sample_count": 8,
        "test_prediction_count": 12,
        "prior_maturity_guard": "PASS",
        "rmse_original": 1.5,
        "smape_original": 10.0,
        "parent_v3_rmse_reference": 1.6,
        "parent_v3_smape_reference": 11.0,
        "relative_improvement_vs_v3_rmse": 0.06,
        "relative_improvement_vs_v3_smape": 0.09,
        "metric_space": ORIGINAL_SALES_SPACE,
        "V3_1_CALIBRATION_ENABLED": "NO",
    }


def _history_record(horizon: int = 1):
    origin = pd.Timestamp("2026-01-14")
    target = origin + pd.Timedelta(days=horizon)
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    sales = np.arange(10.0, 210.0, 10.0)
    kept = [(d, s) for d, s in zip(dates, sales) if d <= origin]
    return SimpleNamespace(
        dataset_id="D1",
        target_key=("1", "10"),
        forecast_origin=origin.strftime("%Y-%m-%d"),
        label_date=target.strftime("%Y-%m-%d"),
        horizon=horizon,
        input_dates=tuple(d.strftime("%Y-%m-%d") for d, _ in kept),
        input_sales=tuple(float(s) for _, s in kept),
    )


def test_v3_existing_artifacts_unchanged():
    before = fingerprint_parent_artifacts(PROJECT_ROOT)
    after = fingerprint_parent_artifacts(PROJECT_ROOT)
    assert assert_parent_fingerprints_unchanged(before, after) == "YES"
    for path in parent_artifact_files(PROJECT_ROOT):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert METHOD_DISPLAY not in text


def test_v3_existing_manifest_unchanged():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    assert frozen["method"] == V3_METHOD_DISPLAY
    assert frozen["selection_frozen"] is True
    assert frozen["architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST
    v3_manifest = PROJECT_ROOT / "configs" / "ms_katl_fto_v3_robustness_manifest.json"
    payload = json.loads(v3_manifest.read_text(encoding="utf-8"))
    assert payload["method"] == "FUSED_TARGET_ONLY_V3"
    assert "V3_2" not in payload["method"]


def test_v3_1_not_inherited():
    assert PARENT_METHOD == "FUSED_TARGET_ONLY_V3"
    assert PARENT_METHOD != "FUSED_TARGET_ONLY_V3_1_CALIBRATED"
    assert V3_1_CALIBRATION_ENABLED == "NO"
    assert assert_v3_1_not_inherited() == "YES"
    runner = RUNNER.read_text(encoding="utf-8")
    assert "calibrate_v3_raw" not in runner
    assert "apply_bias_calibration" not in runner
    assert not any(
        line.strip().startswith("from src.transfer_methods.ms_katl_fto_v3_1_calibrated import")
        for line in runner.splitlines()
    )
    contract = CONTRACT.read_text(encoding="utf-8")
    assert not any(
        line.strip().startswith("from src.transfer_methods.ms_katl_fto_v3_1_calibrated import")
        for line in contract.splitlines()
    )


def test_v3_2_new_method_identity():
    if not MANIFEST.is_file():
        write_v3_2_manifest(PROJECT_ROOT)
    assert METHOD_DISPLAY == "FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL"
    assert METHOD_ARM == "fused_target_only_v3_2_seasonal_residual"
    assert PARENT_METHOD == "FUSED_TARGET_ONLY_V3"
    assert METHOD_DISPLAY != PARENT_METHOD
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["method"] == METHOD_DISPLAY
    assert manifest["parent_method"] == PARENT_METHOD


def test_exact_six_frozen_targets():
    assert tuple(EXPECTED_COMMON_TARGETS) == (
        ("D1", "1/10"),
        ("D2", "1/10"),
        ("D3", "10"),
        ("D4", "166/311"),
        ("D5", "48/1159414"),
        ("D6", "CA_1/FOODS_3_080"),
    )


def test_v3_2_inherits_exact_v3_lr_mapping():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    mapping = frozen_v3_lr_map(frozen)
    expected = {
        ("D1", "1/10"): "3e-5",
        ("D2", "1/10"): "3e-5",
        ("D3", "10"): "3e-5",
        ("D4", "166/311"): "3e-5",
        ("D5", "48/1159414"): "3e-4",
        ("D6", "CA_1/FOODS_3_080"): "3e-5",
    }
    assert mapping == expected
    assert dict(EXPECTED_V3_LR) == expected


def test_v3_2_inherits_v3_architecture():
    assert v3_2_architecture() == v3_architecture()
    assert inherited_v3_spec().architecture_digest == v3_architecture_digest()
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST


def test_v3_2_inherits_v3_recipe():
    assert v3_2_training_recipe() == v3_training_recipe()
    assert v3_2_training_recipe() == {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }


def test_v3_2_revin_config_and_residual_semantics():
    config = v3_2_revin_config()
    assert config["enabled"] is True
    assert config["affine"] is True
    assert config["eps"] == 1e-5
    assert config["target_channel"] == 0
    assert config["include_mean"] is False
    assert config["output_mode"] == "scale_only_residual"
    assert RESIDUALIZATION_SCOPE == "TARGET_STAGE_B_ONLY"
    assert assert_residual_revin_semantics() == "PASS"


def test_lag_exactly_seven_calendar_days():
    history = {
        pd.Timestamp("2026-01-01"): 10.0,
        pd.Timestamp("2026-01-08"): 80.0,
        pd.Timestamp("2026-01-14"): 140.0,
    }
    prior = lookup_lag7_seasonal_prior(
        target_date="2026-01-15",
        forecast_origin="2026-01-14",
        history=history,
        entity_key=("1", "10"),
        horizon=1,
    )
    assert prior["lag_timestamp"] == "2026-01-08"
    assert prior["lag_value"] == 80.0
    assert LAG_DAYS == 7


def test_same_entity_only():
    frame = pd.DataFrame(
        {
            "entity_id": ["1", "1", "2", "2"],
            "item_id": ["10", "10", "10", "10"],
            "date": pd.to_datetime(["2026-01-08", "2026-01-15", "2026-01-08", "2026-01-15"]),
            "sales": [11.0, 20.0, 999.0, 30.0],
        }
    )
    lookup = build_entity_sales_lookup(frame)
    history = observable_history(lookup[("1", "10")], "2026-01-14")
    prior = lookup_lag7_seasonal_prior(
        target_date="2026-01-15",
        forecast_origin="2026-01-14",
        history=history,
        entity_key=("1", "10"),
        horizon=1,
    )
    assert prior["lag_value"] == 11.0
    assert prior["lag_value"] != 999.0


def test_no_row_position_leakage():
    history = {
        pd.Timestamp("2026-01-09"): 999.0,
        pd.Timestamp("2026-01-01"): 1.0,
        pd.Timestamp("2026-01-08"): 80.0,
        pd.Timestamp("2026-01-14"): 140.0,
    }
    prior = lookup_lag7_seasonal_prior(
        target_date="2026-01-15",
        forecast_origin="2026-01-14",
        history=history,
        entity_key=("store", "item"),
        horizon=1,
    )
    assert prior["lag_value"] == 80.0
    assert list(history.values())[0] == 999.0


def test_missing_lag_fails_closed():
    history = {pd.Timestamp("2026-01-14"): 140.0}
    with pytest.raises(V32ContractError, match="SEASONAL_LAG7_HISTORY_MISSING"):
        lookup_lag7_seasonal_prior(
            target_date="2026-01-15",
            forecast_origin="2026-01-14",
            history=history,
            entity_key=("1", "10"),
            horizon=1,
        )


def test_lag_timestamp_le_forecast_origin():
    history = {pd.Timestamp("2026-01-20"): 1.0}
    with pytest.raises(V32ContractError, match="FUTURE_TARGET_ACCESS_FORBIDDEN"):
        lookup_lag7_seasonal_prior(
            target_date="2026-01-15",
            forecast_origin="2026-01-07",
            history=history,
            entity_key=("1", "10"),
        )


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_h1_h3_h5_maturity_pass(horizon: int):
    origin = pd.Timestamp("2026-01-14")
    target = origin + pd.Timedelta(days=horizon)
    lag = target - pd.Timedelta(days=7)
    assert lag <= origin
    history = {lag: 42.0, origin: 1.0}
    prior = lookup_lag7_seasonal_prior(
        target_date=target,
        forecast_origin=origin,
        history=history,
        entity_key=("1", "10"),
        horizon=horizon,
    )
    assert prior["prior_maturity_guard"] == "PASS"
    assert prior["lag_value"] == 42.0


def test_no_future_truth():
    origin = pd.Timestamp("2026-01-14")
    lag = pd.Timestamp("2026-01-08")
    history = {lag: 80.0, origin: 140.0, pd.Timestamp("2026-01-20"): 999999.0}
    visible = observable_history(history, origin)
    assert pd.Timestamp("2026-01-20") not in visible
    prior = lookup_lag7_seasonal_prior(
        target_date="2026-01-15",
        forecast_origin=origin,
        history=visible,
        entity_key=("1", "10"),
        horizon=1,
    )
    assert prior["lag_value"] == 80.0


def test_residual_and_reconstruction_formulas():
    y = np.array([10.0, 20.0, 30.0])
    prior = np.array([3.0, 5.0, 11.0])
    residual = residual_original(y, prior)
    np.testing.assert_allclose(residual, [7.0, 15.0, 19.0])
    sales_min, sales_max = 0.0, 50.0
    residual_minmax = residual / (sales_max - sales_min)
    rebuilt = reconstruct_original_sales(prior, residual_minmax, sales_min, sales_max)
    np.testing.assert_allclose(rebuilt, y)


def test_bl5_lag_contract_equivalence():
    for horizon in (1, 3, 5):
        record = _history_record(horizon=horizon)
        assert seasonal_prior_from_bl5_record(record) == predict_bl5(record)


def test_no_baseline_result_consumption():
    assert assert_no_baseline_result_consumption() == "YES"
    source = inspect.getsource(lookup_lag7_seasonal_prior) + inspect.getsource(
        seasonal_prior_from_bl5_record
    )
    assert "predict_bl5(" not in source
    assert "bl5_rmse" not in source


def test_no_lr_or_target_or_prior_reselection():
    flags = assert_mechanism_is_uniform()
    assert flags["LR_RESELECTION"] == "NO"
    assert flags["TARGET_RESELECTION"] == "NO"
    assert flags["PRIOR_RESELECTION"] == "NO"
    assert "lag=14" not in inspect.getsource(lookup_lag7_seasonal_prior)
    assert "MA7" not in inspect.getsource(lookup_lag7_seasonal_prior)


def test_same_mechanism_d1_d6_h1_h5_seeds():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    identities = build_v3_2_identities(
        datasets=SCREENING_DATASETS,
        horizons=(1, 3, 5),
        seeds=(42, 43, 44),
        lr_map=frozen_v3_lr_map(frozen),
    )
    assert all(item.arm == "v3_2" for item in identities)
    assert {item.dataset for item in identities} == set(SCREENING_DATASETS)
    assert {item.horizon for item in identities} == {1, 3, 5}
    assert {item.seed for item in identities} == {42, 43, 44}


def test_no_v3_1_calibration():
    assert V3_1_CALIBRATION_ENABLED == "NO"
    flags = assert_mechanism_is_uniform()
    assert flags["V3_1_CALIBRATION_ENABLED"] == "NO"
    payload = _ok_payload(_identity())
    assert payload["V3_1_CALIBRATION_ENABLED"] == "NO"


def test_original_metric_space():
    if not MANIFEST.is_file():
        write_v3_2_manifest(PROJECT_ROOT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["metric_space"] == ORIGINAL_SALES_SPACE
    payload = _ok_payload(_identity())
    assert payload["metric_space"] == ORIGINAL_SALES_SPACE


def test_expected_runs_and_unique_cells_are_54():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    identities = build_v3_2_identities(
        datasets=SCREENING_DATASETS,
        horizons=(1, 3, 5),
        seeds=(42, 43, 44),
        lr_map=frozen_v3_lr_map(frozen),
    )
    assert len(identities) == EXPECTED_RUNS == 54
    assert len({item.identity_tuple for item in identities}) == 54
    counts = expected_matrix_counts(SCREENING_DATASETS, (1, 3, 5), (42, 43, 44))
    assert counts["expected_runs"] == 54
    assert counts["unique_runs"] == 54


def test_resume_works(tmp_path: Path):
    identity = _identity()
    path = tmp_path / f"{identity.cell_id}.json"
    write_json_cell(path, _ok_payload(identity))
    from src.transfer_methods.ms_katl_fto_v3_2_seasonal_residual import load_valid_ok_cells

    completed = load_valid_ok_cells(tmp_path, [identity], EXPECTED_V3_ARCHITECTURE_DIGEST)
    assert identity.identity_tuple in completed
    other = _identity(seed=43)
    completed2 = load_valid_ok_cells(tmp_path, [identity, other], EXPECTED_V3_ARCHITECTURE_DIGEST)
    assert other.identity_tuple not in completed2


def test_malformed_cell_fails_closed(tmp_path: Path):
    identity = _identity()
    path = tmp_path / f"{identity.cell_id}.json"
    path.write_text("{not json", encoding="utf-8")
    assert classify_cell_artifact(path, identity, EXPECTED_V3_ARCHITECTURE_DIGEST) == "MALFORMED"
    path.write_text(json.dumps({"status": "OK", "dataset": "D1"}), encoding="utf-8")
    assert classify_cell_artifact(path, identity, EXPECTED_V3_ARCHITECTURE_DIGEST) == "MALFORMED"
    runner = RUNNER.read_text(encoding="utf-8")
    assert "malformed cell fails closed" in runner


def test_formal_methods_unchanged():
    expected = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert FORMAL_METHODS == expected
    assert assert_formal_methods_unchanged_v32() == expected
    assert METHOD_NAME not in FORMAL_METHODS
    assert METHOD_DISPLAY not in FORMAL_METHODS
    assert PARENT_METHOD not in FORMAL_METHODS
    assert V1_METHOD_DISPLAY not in FORMAL_METHODS
    protocol_src = PROTOCOL.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert (
        'FORMAL_METHODS = (\n    "No-TL",\n    "SS-TL",\n    "MSWA-TL",\n'
        '    "MSSB-TL",\n    "MSML-TL",\n    "MSML-TL-RFE",\n)'
    ) in protocol_src
    assert "FUSED_TARGET_ONLY_V3_2" not in protocol_src
    assert inspect.getsource(msml_tl).find("V3_2_SEASONAL") == -1
    assert "fine_tune_learning_rate" not in inspect.getsource(run_no_tl_experiment)


def test_screening_and_robustness_artifacts_unchanged():
    for rel in V3_SOURCE_FILES:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert "FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL" not in text
    screening = v3_screening_root(PROJECT_ROOT)
    robustness = v3_robustness_root(PROJECT_ROOT)
    for root in (screening, robustness):
        results = root / "results"
        if not results.is_dir():
            continue
        for path in results.glob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL" not in text


def test_v3_parent_numerical_revin_default_unchanged():
    x = np.zeros((2, 8, 3), dtype=np.float32)
    x[..., 0] = np.linspace(1.0, 8.0, 8, dtype=np.float32)
    x[1, :, 0] = np.linspace(4.0, 11.0, 8, dtype=np.float32)
    layer = RevIN(num_features=3, affine=False, target_channel=0)
    _, mean, stdev = layer.normalize(tf.constant(x))
    y_norm = tf.constant([[0.5], [-0.25]], dtype=tf.float32)
    default = layer.denormalize(y_norm, mean, stdev).numpy()
    explicit = layer.denormalize(y_norm, mean, stdev, include_mean=True).numpy()
    residual = layer.denormalize(y_norm, mean, stdev, include_mean=False).numpy()
    np.testing.assert_allclose(default, explicit, rtol=0.0, atol=0.0)
    mean_t = mean.numpy()[:, 0, 0].reshape(-1, 1)
    np.testing.assert_allclose(residual, default - mean_t, rtol=0.0, atol=1e-6)
    params = inspect.signature(train_ms_katl_head).parameters
    assert params["revin_include_mean"].default is True
    params_arm = inspect.signature(run_ablation_arm).parameters
    assert params_arm["revin_include_mean"].default is True
    residual_np = denormalize_residual_numpy(layer, y_norm.numpy(), mean.numpy(), stdev.numpy())
    np.testing.assert_allclose(residual_np.reshape(-1, 1), residual, rtol=0.0, atol=1e-6)


def test_refuse_v3_parent_output_dir(tmp_path: Path):
    screening = v3_screening_root(PROJECT_ROOT)
    with pytest.raises(V32ContractError):
        assert_not_v3_parent_output_dir(screening, PROJECT_ROOT)
    robustness = v3_robustness_root(PROJECT_ROOT)
    with pytest.raises(V32ContractError):
        assert_not_v3_parent_output_dir(robustness, PROJECT_ROOT)
    ok = tmp_path / "20260818_999999_MS_KATL_FTO_V3_2_SEASONAL_RESIDUAL_ROBUSTNESS"
    ok.mkdir()
    assert_not_v3_parent_output_dir(ok, PROJECT_ROOT)


def test_preflight_text_contains_required_flags():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    text = format_preflight(
        datasets=SCREENING_DATASETS,
        horizons=(1, 3, 5),
        seeds=(42, 43, 44),
        lr_map=frozen_v3_lr_map(frozen),
        counts={"expected_runs": 54, "unique_runs": 54},
    )
    assert "MS-KATL FTO V3.2 SEASONAL-RESIDUAL PREFLIGHT" in text
    assert "PARENT=FUSED_TARGET_ONLY_V3" in text
    assert "PRIOR=LAG7_SEASONAL" in text
    assert "RESIDUAL_REVIN_SEMANTICS=PASS" in text
    assert "V3_1_CALIBRATION_ENABLED=NO" in text
    assert "READY_FOR_SMOKE=YES" in text
    assert "FORMAL_METHODS_CHANGE=NO" in text


def test_required_cell_fields_and_prediction_columns():
    for field in REQUIRED_CELL_FIELDS:
        assert field in _ok_payload(_identity())
    runner = RUNNER.read_text(encoding="utf-8")
    for col in (
        "forecast_origin",
        "timestamp",
        "y_true",
        "seasonal_prior",
        "predicted_residual",
        "y_pred_v3_2",
    ):
        assert col in runner


def test_manifest_contract():
    if not MANIFEST.is_file():
        write_v3_2_manifest(PROJECT_ROOT)
    manifest = load_v3_2_manifest(PROJECT_ROOT)
    assert manifest["method"] == METHOD_DISPLAY
    assert manifest["parent_method"] == PARENT_METHOD
    assert manifest["seasonal_prior"]["lag_days"] == 7
    assert manifest["seasonal_prior"]["selection"] is False
    assert manifest["V3_1_calibration"] is False
    assert manifest["residual"]["definition"] == "y_t-y_t_minus_7"
    assert manifest["horizons"] == [1, 3, 5]
    assert manifest["seeds"] == [42, 43, 44]


def test_sequence_window_meta_is_date_aware():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 20,
            "item_id": ["10"] * 20,
            "date": dates,
            "sales": np.arange(20.0),
        }
    )
    meta = sequence_window_meta(frame, horizon=1, window_size=10, feature_columns=["sales"])
    assert meta[0]["window_end"] == "2026-01-10"
    assert meta[0]["label_date"] == "2026-01-11"
    origin = pd.Timestamp(meta[0]["window_end"])
    label = pd.Timestamp(meta[0]["label_date"])
    assert label == origin + pd.Timedelta(days=1)
