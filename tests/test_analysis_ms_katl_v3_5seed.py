from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "analysis_ms_katl_v3_5seed.py"
SPEC = importlib.util.spec_from_file_location("analysis_ms_katl_v3_5seed", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def baseline_rows() -> pd.DataFrame:
    rows = []
    for dataset in analysis.DATASETS:
        exact = analysis.TARGETS[dataset]
        targets = (exact,) if dataset == "D1" else (exact, f"{exact}_illegal_1", f"{exact}_illegal_2", f"{exact}_illegal_3", f"{exact}_illegal_4")
        for target in targets:
            for horizon in analysis.HORIZONS:
                for seed in analysis.SEEDS:
                    for method in analysis.BASELINES:
                        rows.append(
                            {
                                "dataset_id": dataset,
                                "target_entity_key": target,
                                "scenario": "without",
                                "method": method,
                                "horizon": horizon,
                                "seed": seed,
                                "rmse": 1.0,
                                "smape": 2.0,
                                "smape_normalized": 999.0,
                                "result_status": "confirmed_baseline",
                                "rmse_metric_space": "original_sales_space",
                                "smape_metric_space": "original_sales_space",
                            }
                        )
    return pd.DataFrame(rows)


def v3_ranges() -> pd.DataFrame:
    return pd.DataFrame(
        {"dataset": analysis.DATASETS, "sales_min": [0.0] * 4, "sales_max": [10.0] * 4}
    )


def write_baselines(repo: Path, frame: pd.DataFrame) -> None:
    path = repo / "outputs" / "baselines" / "d1_d6_bl1_bl5_baselines_multiseed.csv"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)


def v3_input() -> pd.DataFrame:
    rows = []
    for dataset in analysis.DATASETS:
        for horizon in analysis.HORIZONS:
            for seed in analysis.SEEDS:
                rows.append(
                    {
                        "status": "OK",
                        "arm": "v3",
                        "dataset": dataset,
                        "target_identity": analysis.TARGETS[dataset],
                        "mode": "without",
                        "horizon": horizon,
                        "seed": seed,
                        "method": analysis.V3_METHOD,
                        "rmse_original": 1.0,
                        "smape_original": 2.0,
                        "rmse_normalized": 999.0,
                        "smape_normalized": 999.0,
                        "metric_space": "original_sales_space",
                        "run_status": "OK",
                        "sales_min": 0.0,
                        "sales_max": 10.0,
                        "architecture_digest": "arch",
                        "training_recipe": "recipe",
                    }
                )
    return pd.DataFrame(rows)


def test_baseline_loader_filters_dataset_specific_exact_targets(tmp_path: Path) -> None:
    raw = baseline_rows()
    assert len(raw) == 1200
    write_baselines(tmp_path, raw)

    actual = analysis.load_baselines(tmp_path, v3_ranges())

    assert len(actual) == 300
    assert actual[analysis.METHOD_KEY].drop_duplicates().shape[0] == 300
    assert actual.groupby("dataset").size().to_dict() == {dataset: 75 for dataset in analysis.DATASETS}
    assert (actual["target"] == actual["dataset"].map(analysis.TARGETS)).all()
    assert (actual.groupby(["dataset", "horizon", "seed"])["method"].nunique() == 5).all()
    assert set(actual["smape"]) == {2.0}
    assert set(actual["rmse_metric_space"]) == {"original_sales_space"}
    assert set(actual["smape_metric_space"]) == {"original_sales_space"}


def test_v3_exact_60_row_contract_uses_original_metrics() -> None:
    actual = analysis.normalize_v3(v3_input(), analysis.SEEDS, "V3 test")

    assert len(actual) == 60
    assert actual[analysis.KEY].drop_duplicates().shape[0] == 60
    assert set(actual["rmse_original"]) == {1.0}
    assert set(actual["smape_original"]) == {2.0}


def test_v3_illegal_target_is_rejected() -> None:
    frame = v3_input()
    frame.loc[0, "target_identity"] = "illegal-target"

    with pytest.raises(analysis.FailClosed, match="target mismatch"):
        analysis.normalize_v3(frame, analysis.SEEDS, "V3 test")


def test_learning_paper_table_contains_original_rmse_and_smape() -> None:
    all_runs = []
    for method in analysis.METHODS:
        for dataset in analysis.DATASETS:
            for horizon in analysis.HORIZONS:
                for seed in analysis.SEEDS:
                    all_runs.append(
                        {
                            "dataset": dataset,
                            "target": analysis.TARGETS[dataset],
                            "horizon": horizon,
                            "seed": seed,
                            "method": method,
                            "rmse": 1.0,
                            "smape": 2.0,
                            "nrmse_range_percent": 10.0,
                        }
                    )
    table = analysis.learning_table(analysis.summary_stats(pd.DataFrame(all_runs)))

    assert len(table) == 12
    assert {"LightGBM RMSE", "LSTM RMSE", "V3 RMSE"}.issubset(table.columns)
    assert {"LightGBM sMAPE", "LSTM sMAPE", "V3 sMAPE"}.issubset(table.columns)
    assert "V3 vs best learning baseline RMSE gain %" in table
    assert "V3 vs best learning baseline sMAPE gain %" in table
