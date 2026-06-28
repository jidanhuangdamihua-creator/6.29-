from pathlib import Path

import numpy as np


def test_audit_script_imports():
    from scripts.audits import notl_vs_naive_horizon_audit as audit

    assert audit.DETAILS_CSV.name == "notl_vs_naive_horizon_details.csv"


def test_naive_persistence_uses_last_window_sales():
    from scripts.audits.notl_vs_naive_horizon_audit import naive_persistence_predict

    x = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )

    pred = naive_persistence_predict(x, ["sales", "promo"])

    assert pred.shape == (2, 1)
    np.testing.assert_allclose(pred.reshape(-1), np.array([3.0, 6.0], dtype=np.float32))


def test_comparison_columns_contract():
    from scripts.audits import notl_vs_naive_horizon_audit as audit

    required = {"dataset_id", "info_sharing", "horizon", "naive_rmse", "notl_rmse", "winner"}

    assert required.issubset(set(audit.COMPARISON_COLUMNS))


def test_horizon_and_info_sharing_coverage():
    from scripts.audits import notl_vs_naive_horizon_audit as audit

    assert audit.HORIZONS == [1, 2, 3, 4, 5]
    assert set(audit.INFO_SHARING_VALUES) == {True, False}


def test_outputs_are_under_audits_and_not_experiment_results():
    from scripts.audits import notl_vs_naive_horizon_audit as audit

    output_paths = [audit.DETAILS_CSV, audit.SUMMARY_CSV, audit.COMPARISON_CSV, audit.REPORT_MD]
    for path in output_paths:
        parts = Path(path).parts
        assert "outputs" in parts
        assert "audits" in parts
        assert "experiment_results" not in parts
        assert Path(path).parent == audit.OUT_DIR


def test_main_experiment_scripts_are_not_audit_outputs():
    from scripts.audits import notl_vs_naive_horizon_audit as audit

    output_paths = {audit.DETAILS_CSV, audit.SUMMARY_CSV, audit.COMPARISON_CSV, audit.REPORT_MD}

    assert not output_paths.intersection(set(audit.MAIN_EXPERIMENT_SCRIPTS))
