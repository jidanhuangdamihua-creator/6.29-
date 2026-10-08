"""Targeted tests for MS-KATL D4 metric-space repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.evaluation.metrics import inverse_transform_sales
from src.transfer_methods.ms_katl_cross_target import (
    D4_CROSS_TARGET_IDENTITIES,
    load_baseline_references,
    load_formal_references,
)
from src.transfer_methods.ms_katl_mechanism_ablation import REFERENCE_METRICS
from src.transfer_methods.ms_katl_metric_space_repair import (
    ARM_FILES,
    SOURCE_RUN_NAME,
    FrozenSalesInverseScaler,
    MetricSpaceMismatchError,
    MetricSpaceRepairError,
    assert_historical_run_untouched,
    count_external_references,
    expected_prediction_files,
    original_rmse,
    recover_sales_minmax_from_identity,
    require_same_metric_space,
    snapshot_historical_run,
    source_run_dir,
)


SOURCE_RUN = source_run_dir(PROJECT_ROOT)
DIAGNOSTICS_PATH = SOURCE_RUN / "results" / "d4_h1_s42_cross_target_diagnostics.json"
REPAIR_RESULTS = (
    SOURCE_RUN / "metric_space_repair" / "results" / "d4_h1_s42_cross_target_original_results.csv"
)
REPAIR_SUMMARY = (
    SOURCE_RUN / "metric_space_repair" / "results" / "d4_h1_s42_cross_target_original_summary.csv"
)


def test_canonical_sales_inverse_transform_matches_known_identity():
    y_norm = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
    scaler = FrozenSalesInverseScaler(2.0, 12.0, n_features=3, sales_index=1)
    got = inverse_transform_sales(
        y_norm,
        sales_scaler=scaler,
        feature_columns=["stock_hour6_22_cnt", "sales", "discount"],
    )
    expected = y_norm * 10.0 + 2.0
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_canonical_inverse_clips_negatives_and_does_not_use_single_column_inverse():
    y_norm = np.array([-0.1, 0.0, 1.2], dtype=np.float64)
    scaler = FrozenSalesInverseScaler(0.0, 5.0, n_features=1, sales_index=0)
    got = inverse_transform_sales(
        y_norm,
        sales_scaler=scaler,
        feature_columns=["sales"],
    )
    assert got[0] == 0.0
    np.testing.assert_allclose(got[1:], np.array([0.0, 6.0]), rtol=0.0, atol=1e-12)


def test_metric_repair_path_never_calls_scaler_fit(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("scaler.fit/fit_transform must not be called")

    monkeypatch.setattr(MinMaxScaler, "fit", _boom)
    monkeypatch.setattr(MinMaxScaler, "fit_transform", _boom)
    y_orig = np.array([2.0, 4.0, 12.0], dtype=np.float64)
    y_norm = (y_orig - 2.0) / 10.0
    sales_min, sales_max = recover_sales_minmax_from_identity(y_norm, y_orig)
    scaler = FrozenSalesInverseScaler(sales_min, sales_max)
    with pytest.raises(RuntimeError, match="forbids scaler.fit"):
        scaler.fit(y_orig.reshape(-1, 1))
    with pytest.raises(RuntimeError, match="forbids scaler.fit_transform"):
        scaler.fit_transform(y_orig.reshape(-1, 1))
    restored = inverse_transform_sales(
        y_norm,
        sales_scaler=scaler,
        feature_columns=["sales"],
        apply_sales_value_policy=False,
    )
    np.testing.assert_allclose(restored, y_orig, rtol=0.0, atol=1e-12)
    assert sales_min == pytest.approx(2.0)
    assert sales_max == pytest.approx(12.0)


def test_metric_space_mismatch_fail_closed():
    with pytest.raises(MetricSpaceMismatchError, match="166/258") as exc:
        require_same_metric_space(
            target_identity="166/258",
            candidate_method="B_FUSED_TARGET_ONLY",
            candidate_metric_space="normalized",
            reference_method="BL2_MovingAverage",
            reference_metric_space=ORIGINAL_SALES_SPACE,
        )
    assert exc.value.candidate_method == "B_FUSED_TARGET_ONLY"
    assert exc.value.reference_method == "BL2_MovingAverage"
    assert exc.value.candidate_metric_space == "normalized"
    assert exc.value.reference_metric_space == ORIGINAL_SALES_SPACE


def test_166_258_original_rmse_does_not_beat_bl2():
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    b = diagnostics["targets"]["166/258"]["arms"]["fused_target_only"]
    pred = pd.read_csv(SOURCE_RUN / "predictions" / "166_258" / "fused_target_only.csv")
    y_true = pred["y_true"].to_numpy(dtype=np.float64)
    y_pred = pred["y_pred"].to_numpy(dtype=np.float64)
    rmse_norm = original_rmse(y_true, y_pred)
    assert rmse_norm == pytest.approx(0.3371, abs=5e-4)
    assert b["rmse"] == pytest.approx(0.3371, abs=5e-4)
    orig = float(b["metric_protocol"]["original_scale_rmse"])
    assert orig == pytest.approx(0.3708, abs=5e-4)
    bl2 = 0.3451690262214515
    assert orig > bl2
    scaler = FrozenSalesInverseScaler(0.0, orig / rmse_norm)
    y_true_o = inverse_transform_sales(
        y_true, sales_scaler=scaler, feature_columns=["sales"], apply_sales_value_policy=False
    )
    y_pred_o = inverse_transform_sales(
        y_pred, sales_scaler=scaler, feature_columns=["sales"], apply_sales_value_policy=False
    )
    repaired = original_rmse(y_true_o, y_pred_o)
    assert repaired == pytest.approx(0.3708, abs=5e-4)
    assert repaired != pytest.approx(0.3371, abs=1e-4)
    assert not (repaired < bl2)


def test_abc_same_target_scaler_digest():
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    for identity in D4_CROSS_TARGET_IDENTITIES:
        digests = [
            diagnostics["targets"][identity]["arms"][arm]["target_split_digest"]["target_scaler"]
            for arm in ARM_FILES
        ]
        assert len(set(digests)) == 1, identity
        splits = [
            diagnostics["targets"][identity]["arms"][arm]["target_split_digest"]
            for arm in ARM_FILES
        ]
        assert splits[0] == splits[1] == splits[2], identity


def test_abc_normalized_y_true_match():
    for identity in D4_CROSS_TARGET_IDENTITIES:
        dirname = identity.replace("/", "_")
        frames = [
            pd.read_csv(SOURCE_RUN / "predictions" / dirname / f"{arm}.csv")["y_true"].to_numpy(dtype=np.float64)
            for arm in ARM_FILES
        ]
        np.testing.assert_allclose(frames[0], frames[1], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(frames[0], frames[2], rtol=0.0, atol=1e-12)


@pytest.mark.skipif(not REPAIR_RESULTS.exists(), reason="repair outputs not written yet")
def test_abc_original_y_true_match_after_repair():
    for identity in D4_CROSS_TARGET_IDENTITIES:
        dirname = identity.replace("/", "_")
        frames = [
            pd.read_csv(
                SOURCE_RUN / "metric_space_repair" / "predictions_original" / dirname / f"{arm}.csv"
            )["y_true_original"].to_numpy(dtype=np.float64)
            for arm in ARM_FILES
        ]
        np.testing.assert_allclose(frames[0], frames[1], rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(frames[0], frames[2], rtol=0.0, atol=1e-10)


def test_external_reference_identity_join_is_35_of_35():
    counts = count_external_references(project_root=PROJECT_ROOT)
    assert counts["expected"] == 35
    assert counts["found"] == 35
    assert counts["missing"] == []
    assert counts["label"] == "35/35"
    for identity in D4_CROSS_TARGET_IDENTITIES:
        formal = load_formal_references(
            PROJECT_ROOT
            / "outputs/runs/20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8/d4_without/results/dataset4_without_results.csv",
            target_identity=identity,
        )
        baselines = load_baseline_references(
            PROJECT_ROOT / "outputs/baselines/d4_baselines_multiseed.csv",
            target_identity=identity,
        )
        assert formal["no_tl"]["status"] == "FOUND"
        assert formal["msml"]["status"] == "FOUND"
        for key in ("bl1", "bl2", "bl3", "bl4", "bl5"):
            assert baselines[key]["status"] == "FOUND"
        assert baselines["best"]["status"] == "FOUND"


def test_historical_run_snapshot_detects_overwrite(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "d4_h1_s42_cross_target_results.csv").write_text("old\n", encoding="utf-8")
    (results / "d4_h1_s42_cross_target_summary.csv").write_text("old\n", encoding="utf-8")
    (results / "d4_h1_s42_cross_target_diagnostics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    for identity in D4_CROSS_TARGET_IDENTITIES:
        d = tmp_path / "predictions" / identity.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        for arm in ARM_FILES:
            (d / f"{arm}.csv").write_text("y_true,y_pred\n0,1\n", encoding="utf-8")
    before = snapshot_historical_run(tmp_path)
    (tmp_path / "metric_space_repair" / "results").mkdir(parents=True)
    (tmp_path / "metric_space_repair" / "results" / "new.csv").write_text("new\n", encoding="utf-8")
    after = snapshot_historical_run(tmp_path)
    assert_historical_run_untouched(before, after)
    (results / "d4_h1_s42_cross_target_results.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(MetricSpaceRepairError, match="overwritten"):
        assert_historical_run_untouched(before, snapshot_historical_run(tmp_path))


def test_source_run_historical_files_still_present():
    assert SOURCE_RUN.name == SOURCE_RUN_NAME
    for path in expected_prediction_files(SOURCE_RUN):
        assert path.exists()
    assert (SOURCE_RUN / "results" / "d4_h1_s42_cross_target_results.csv").exists()
    assert (SOURCE_RUN / "results" / "d4_h1_s42_cross_target_summary.csv").exists()


def test_hardcoded_reference_metrics_are_not_usable_for_comparison():
    for payload in REFERENCE_METRICS.values():
        assert payload.get("usable_for_comparison") is False
    source = (
        PROJECT_ROOT / "src" / "transfer_methods" / "ms_katl_mechanism_ablation.py"
    ).read_text(encoding="utf-8")
    assert "DO NOT USE FOR FORMAL OR CROSS-METHOD COMPARISON" in source
    assert 'row["rmse"] - REFERENCE_METRICS["FORMAL_NO_TL"]["rmse"]' not in source


@pytest.mark.skipif(not REPAIR_SUMMARY.exists(), reason="repair outputs not written yet")
def test_repair_outputs_use_original_space_and_166_258_sanity():
    summary = pd.read_csv(REPAIR_SUMMARY)
    row = summary.loc[summary["target_identity"] == "166/258"].iloc[0]
    assert row["metric_space"] == ORIGINAL_SALES_SPACE
    assert float(row["B_rmse_normalized"]) == pytest.approx(0.3371, abs=5e-4)
    assert float(row["B_rmse_original"]) == pytest.approx(0.3708, abs=5e-4)
    assert float(row["bl2_rmse"]) == pytest.approx(0.3452, abs=5e-4)
    assert str(row["B_beats_best_BL"]).upper() in {"NO", "FALSE"}
    results = pd.read_csv(REPAIR_RESULTS)
    assert set(results["metric_space"].unique()) == {ORIGINAL_SALES_SPACE}
    assert len(results) == 15
