"""Contract tests for MS-KATL FUSED_TARGET_ONLY_V3_1_CALIBRATED."""

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

import numpy as np
import pandas as pd

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.experiment import run_no_tl_experiment
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods import msml_tl
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    write_cell_artifact as write_json_cell,
)
from src.transfer_methods.ms_katl_fto_v3_1_calibrated import (
    CALIBRATION_TYPE,
    EXPECTED_COMMON_TARGETS,
    EXPECTED_RUNS,
    EXPECTED_V3_LR,
    METHOD_ARM,
    METHOD_DISPLAY,
    PARENT_METHOD,
    REQUIRED_CELL_FIELDS,
    V3_SOURCE_FILES,
    V31ContractError,
    V31RunIdentity,
    apply_bias_calibration,
    assert_calibration_rule_is_uniform,
    assert_formal_methods_unchanged_v31,
    assert_not_v3_parent_output_dir,
    assert_parent_fingerprints_unchanged,
    assert_v3_parent_contract,
    build_v3_1_identities,
    classify_cell_artifact,
    compute_validation_median_bias,
    concat_train_val_context,
    count_calibration_samples,
    estimate_validation_median_bias,
    expected_matrix_counts,
    fingerprint_parent_artifacts,
    format_preflight,
    frozen_v3_lr_map,
    inherited_v3_spec,
    load_v3_1_manifest,
    load_valid_ok_cells,
    parent_artifact_files,
    sequence_window_meta,
    v3_1_architecture,
    v3_1_revin_config,
    v3_1_training_recipe,
    v3_robustness_root,
    v3_screening_root,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    SCREENING_DATASETS,
    format_lr,
)

CONTRACT = PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_fto_v3_1_calibrated.py"
RUNNER = PROJECT_ROOT / "scripts" / "run_ms_katl_fto_v3_1_calibrated_robustness.py"
MANIFEST = PROJECT_ROOT / "configs" / "ms_katl_fto_v3_1_calibrated_manifest.json"
PROTOCOL = PROJECT_ROOT / "src" / "protocols" / "experiment_protocol.py"


def _identity(**overrides) -> V31RunIdentity:
    payload = {
        "dataset": "D1",
        "target_identity": "1/10",
        "horizon": 1,
        "seed": 42,
        "selected_lr": 3e-5,
        "selected_lr_label": "3e-5",
    }
    payload.update(overrides)
    return V31RunIdentity(**payload)


def _ok_payload(identity: V31RunIdentity) -> dict:
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
        "calibration_type": CALIBRATION_TYPE,
        "calibration_sample_count": 12,
        "calibration_bias": 0.25,
        "rmse_v3_raw": 1.5,
        "smape_v3_raw": 10.0,
        "rmse_v3_1_calibrated": 1.2,
        "smape_v3_1_calibrated": 9.0,
        "rmse_improvement_vs_raw_v3": 0.2,
        "smape_improvement_vs_raw_v3": 0.1,
        "metric_space": ORIGINAL_SALES_SPACE,
    }


def _daily_frame(n: int, start: str, shift: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "entity_id": ["store"] * n,
            "item_id": ["item"] * n,
            "date": dates,
            "sales": np.arange(n, dtype=np.float64) + shift,
        }
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
    assert "V3_1" not in payload["method"]


def test_v3_1_new_method_identity():
    assert METHOD_DISPLAY == "FUSED_TARGET_ONLY_V3_1_CALIBRATED"
    assert METHOD_ARM == "fused_target_only_v3_1_calibrated"
    assert PARENT_METHOD == "FUSED_TARGET_ONLY_V3"
    assert METHOD_DISPLAY != PARENT_METHOD
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["method"] == METHOD_DISPLAY
    assert manifest["parent_method"] == PARENT_METHOD


def test_v3_1_inherits_exact_v3_lr_mapping():
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
    manifest = load_v3_1_manifest(PROJECT_ROOT)
    for dataset, target in EXPECTED_COMMON_TARGETS:
        assert format_lr(manifest["datasets"][dataset]["learning_rate"]) == expected[(dataset, target)]


def test_v3_1_inherits_v3_architecture():
    assert v3_1_architecture() == v3_architecture()
    assert inherited_v3_spec().architecture_digest == v3_architecture_digest()
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST


def test_v3_1_inherits_v3_recipe():
    assert v3_1_training_recipe() == v3_training_recipe()
    assert v3_1_training_recipe() == {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }


def test_v3_1_inherits_revin_config():
    config = v3_1_revin_config()
    assert config["enabled"] is True
    assert config["affine"] is True
    assert config["eps"] == 1e-5
    assert config["target_channel"] == 0
    arch = v3_1_architecture()
    assert arch["revin_enabled"] is True
    assert arch["revin_affine"] is True


def test_bias_formula_is_median_validation_residual():
    y_val = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    yhat = np.array([11.0, 18.0, 29.0, 50.0, 47.0])
    residual = y_val - yhat
    assert compute_validation_median_bias(y_val, yhat) == float(np.median(residual))
    assert compute_validation_median_bias(y_val, yhat) == 1.0


def test_additive_correction_is_yhat_plus_bias():
    yhat = np.array([1.0, 2.5, -3.0])
    got = apply_bias_calibration(yhat, 0.75)
    np.testing.assert_allclose(got, yhat + 0.75, rtol=0.0, atol=0.0)


def test_calibration_only_uses_validation_labels():
    train = _daily_frame(15, "2020-01-01")
    val = _daily_frame(15, "2020-01-16", shift=15)
    test = _daily_frame(15, "2020-01-31", shift=30)
    context = concat_train_val_context(train, val, raw_test=test)
    test_dates = set(pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"))
    context_dates = set(pd.to_datetime(context["date"]).dt.strftime("%Y-%m-%d"))
    assert not (context_dates & test_dates)
    n_cal = count_calibration_samples(
        train, val, horizon=5, window_size=10, feature_columns=["sales"], raw_test=test
    )
    val_only_meta = sequence_window_meta(
        val, horizon=5, window_size=10, feature_columns=["sales"]
    )
    assert n_cal == 15
    assert len(val_only_meta) == 1
    assert n_cal > len(val_only_meta)


def test_test_truth_forbidden_from_calibration():
    y_val = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.2, 1.8, 2.7])
    with pytest.raises(V31ContractError, match="forbidden calibration inputs"):
        estimate_validation_median_bias(
            y_val_original=y_val,
            yhat_val_original=yhat,
            y_test=np.array([9.0]),
        )
    with pytest.raises(V31ContractError):
        estimate_validation_median_bias(
            y_val_original=y_val,
            yhat_val_original=yhat,
            test_truth=np.array([9.0]),
        )


def test_baseline_metrics_forbidden_from_calibration():
    y_val = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.2, 1.8, 2.7])
    with pytest.raises(V31ContractError, match="forbidden calibration inputs"):
        estimate_validation_median_bias(
            y_val_original=y_val,
            yhat_val_original=yhat,
            baseline_metrics={"rmse": 1.0},
        )
    with pytest.raises(V31ContractError):
        estimate_validation_median_bias(
            y_val_original=y_val,
            yhat_val_original=yhat,
            baselines={"BL1": 1.0},
        )


def test_same_calibration_rule_d1_d6():
    y_val = np.array([4.0, 5.0, 9.0, 2.0])
    yhat = np.array([3.0, 6.0, 8.0, 4.0])
    bias = compute_validation_median_bias(y_val, yhat)
    for dataset, _target in EXPECTED_COMMON_TARGETS:
        assert compute_validation_median_bias(y_val, yhat) == bias, dataset
    flags = assert_calibration_rule_is_uniform()
    assert flags["CALIBRATION_UNIFORM_ACROSS_DATASETS"] == "YES"


def test_same_calibration_rule_h1_h3_h5():
    y_val = np.array([4.0, 5.0, 9.0])
    yhat = np.array([5.0, 5.0, 8.0])
    bias = compute_validation_median_bias(y_val, yhat)
    for horizon in (1, 3, 5):
        assert compute_validation_median_bias(y_val, yhat) == bias, horizon
    flags = assert_calibration_rule_is_uniform()
    assert flags["CALIBRATION_UNIFORM_ACROSS_HORIZONS"] == "YES"


def test_same_calibration_rule_seed_42_43_44():
    y_val = np.array([10.0, 12.0, 8.0])
    yhat = np.array([9.0, 11.0, 9.0])
    bias = compute_validation_median_bias(y_val, yhat)
    for seed in (42, 43, 44):
        assert compute_validation_median_bias(y_val, yhat) == bias, seed
    flags = assert_calibration_rule_is_uniform()
    assert flags["CALIBRATION_UNIFORM_ACROSS_SEEDS"] == "YES"


def test_no_dataset_horizon_or_seed_specific_switch():
    for fn in (
        compute_validation_median_bias,
        apply_bias_calibration,
        estimate_validation_median_bias,
    ):
        params = inspect.signature(fn).parameters
        assert "dataset" not in params
        assert "horizon" not in params
        assert "seed" not in params
    source = (
        inspect.getsource(compute_validation_median_bias)
        + inspect.getsource(apply_bias_calibration)
        + inspect.getsource(estimate_validation_median_bias)
    )
    assert "if dataset" not in source
    assert "if horizon" not in source
    assert "if seed" not in source
    flags = assert_calibration_rule_is_uniform()
    assert flags["TEST_TRUTH_USED_FOR_CALIBRATION"] == "NO"


def test_original_metric_space():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["metric_space"] == ORIGINAL_SALES_SPACE
    identity = _identity()
    payload = _ok_payload(identity)
    assert payload["metric_space"] == ORIGINAL_SALES_SPACE


def test_raw_v3_predictions_preserved_and_calibration_stored_independently():
    runner = RUNNER.read_text(encoding="utf-8")
    assert "y_pred_v3_raw" in runner
    assert "y_pred_v3_1_calibrated" in runner
    assert "validation_y_true" in runner
    assert "validation_y_pred" in runner
    assert "validation_residual" in runner
    assert "cal_root" in runner
    assert "predictions.csv" in runner
    contract = CONTRACT.read_text(encoding="utf-8")
    assert "y_pred_v3_raw" in contract
    assert "y_pred_v3_1_calibrated" in contract


def test_expected_runs_and_unique_cells_are_54():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    identities = build_v3_1_identities(
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
    assert len(EXPECTED_COMMON_TARGETS) == 6


def test_resume_works(tmp_path: Path):
    identity = _identity()
    path = tmp_path / f"{identity.cell_id}.json"
    write_json_cell(path, _ok_payload(identity))
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
    assert assert_formal_methods_unchanged_v31() == expected
    assert METHOD_NAME not in FORMAL_METHODS
    assert METHOD_DISPLAY not in FORMAL_METHODS
    assert PARENT_METHOD not in FORMAL_METHODS
    assert V1_METHOD_DISPLAY not in FORMAL_METHODS
    protocol_src = PROTOCOL.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert (
        'FORMAL_METHODS = (\n    "No-TL",\n    "SS-TL",\n    "MSWA-TL",\n'
        '    "MSSB-TL",\n    "MSML-TL",\n    "MSML-TL-RFE",\n)'
    ) in protocol_src
    assert "FUSED_TARGET_ONLY_V3_1" not in protocol_src
    assert inspect.getsource(msml_tl).find("V3_1_CALIBRATED") == -1
    assert "fine_tune_learning_rate" not in inspect.getsource(run_no_tl_experiment)


def test_screening_and_robustness_artifacts_unchanged():
    for rel in V3_SOURCE_FILES:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert "FUSED_TARGET_ONLY_V3_1_CALIBRATED" not in text
        assert "validation_median_bias" not in text
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
            assert "FUSED_TARGET_ONLY_V3_1_CALIBRATED" not in text


def test_preflight_banner_and_no_reselection():
    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    text = format_preflight(
        datasets=SCREENING_DATASETS,
        horizons=(1, 3, 5),
        seeds=(42, 43, 44),
        lr_map=frozen_v3_lr_map(frozen),
        counts={"expected_runs": 54, "unique_runs": 54},
    )
    assert "MS-KATL FTO V3.1 CALIBRATED ROBUSTNESS PREFLIGHT" in text
    assert "METHOD=FUSED_TARGET_ONLY_V3_1_CALIBRATED" in text
    assert "PARENT=FUSED_TARGET_ONLY_V3" in text
    assert "DATASETS=D1,D2,D3,D4,D5,D6" in text
    assert "TARGETS=6" in text
    assert "HORIZONS=1,3,5" in text
    assert "SEEDS=42,43,44" in text
    assert "EXPECTED_RUNS=54" in text
    assert "UNIQUE_RUNS=54" in text
    assert "CALIBRATION=VALIDATION_MEDIAN_BIAS" in text
    assert "CALIBRATION_SPACE=original_sales_space" in text
    assert "TEST_TRUTH_USED_FOR_CALIBRATION=NO" in text
    assert "DATASET_SPECIFIC_TUNING=NO" in text
    assert "HORIZON_SPECIFIC_TUNING=NO" in text
    assert "SEED_SPECIFIC_TUNING=NO" in text
    assert "LR_RESELECTION=NO" in text
    assert "TARGET_RESELECTION=NO" in text
    assert "V3_PARENT_UNCHANGED=YES" in text
    assert "V3_SCREENING_UNCHANGED=YES" in text
    assert "V3_ROBUSTNESS_UNCHANGED=YES" in text
    assert "FORMAL_METHODS_CHANGE=NO" in text
    assert "READY_FOR_SMOKE=YES" in text
    assert "3e-4" in text
    runner = RUNNER.read_text(encoding="utf-8")
    assert "select_dataset_primary_target" not in runner
    assert "select_target_best_lr" not in runner
    assert "best_seed" not in runner


def test_refuses_to_write_into_v3_parent_dirs():
    with pytest.raises(V31ContractError):
        assert_not_v3_parent_output_dir(v3_screening_root(PROJECT_ROOT), PROJECT_ROOT)
    with pytest.raises(V31ContractError):
        assert_not_v3_parent_output_dir(v3_robustness_root(PROJECT_ROOT), PROJECT_ROOT)
    with pytest.raises(V31ContractError):
        assert_not_v3_parent_output_dir(
            PROJECT_ROOT / "outputs" / "runs" / "x_MS_KATL_FTO_V3_SCREENING",
            PROJECT_ROOT,
        )


def test_required_cell_fields_cover_contract():
    required = set(REQUIRED_CELL_FIELDS)
    for key in (
        "rmse_v3_raw",
        "smape_v3_raw",
        "rmse_v3_1_calibrated",
        "smape_v3_1_calibrated",
        "calibration_bias",
        "calibration_sample_count",
        "parent_architecture_digest",
    ):
        assert key in required


def test_no_prediction_clipping_added_in_v31():
    source = inspect.getsource(apply_bias_calibration)
    assert "clip" not in source
    assert "maximum" not in source
    assert "max(" not in source
