from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.statistical_tests import compare_methods_smape, run_friedman_test
from src.evaluation.metric_contract import METRIC_CONTRACT_VERSION, SMAPE_DEFINITION_ID
from src.protocols.experiment_protocol import FORMAL_SEEDS
from src.visualization.result_visualizer import filter_formally_comparable_results


def _row(dataset, method, smape, *, horizon=1, sharing="without", **overrides):
    row = {
        "dataset": dataset,
        "dataset_id": dataset,
        "target_entity_key": "target",
        "method": method,
        "horizon": horizon,
        "seed": 42,
        "information_sharing": sharing,
        "smape": smape,
        "strict_paper_metrics": True,
        "paper_metric_space_requested": "original_sales_space",
        "paper_metric_space_actual": "original_sales_space",
        "primary_metric_space_actual": "original_sales_space",
        "smape_metric_space": "original_sales_space",
        "inverse_transform_status": "not_required",
        "inverse_transform_applied": False,
        "paper_metric_computed_valid": True,
        "paper_metric_status": "valid",
        "paper_metric_error": "",
        "metric_sample_count": 2,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "smape_definition_id": SMAPE_DEFINITION_ID,
        "smape_unit": "percent",
        "smape_epsilon": 1e-8,
        "smape_range_min": 0.0,
        "smape_range_max": 200.0,
        "sales_value_policy": "clip_negative_to_zero_v1",
        "target_negative_count": 0,
        "metric_target_key": "target",
        "metric_horizon": horizon,
        "metric_date_start": "2024-01-02",
        "metric_date_end": "2024-01-03",
        "metric_index_digest": f"{dataset}-{method}-{horizon}-{sharing}",
        "accuracy": 1.0,
        "rmse": 1.0,
        "prediction_shape": "(2, 1)",
    }
    row.update(overrides)
    return row


def _seed_rows(dataset, method, smape, *, horizon=1, sharing="without"):
    return [
        _row(
            dataset,
            method,
            smape,
            horizon=horizon,
            sharing=sharing,
            seed=seed,
            metric_index_digest=f"{dataset}-{method}-{horizon}-{sharing}-{seed}",
        )
        for seed in FORMAL_SEEDS
    ]


def test_visualization_filter_accepts_not_required_and_excludes_numeric_non_strict():
    frame = pd.DataFrame(
        [
            _row("D1", "No-TL", 20.0),
            _row("D1", "SS-TL", 1.0, strict_paper_metrics=False),
        ]
    )

    filtered = filter_formally_comparable_results(frame)

    assert filtered["method"].tolist() == ["No-TL"]
    assert filtered.iloc[0]["inverse_transform_applied"] in {False, np.bool_(False)}


def test_statistics_uses_complete_case_dataset_blocks_and_holm_correction():
    rows = []
    values = {
        "D1": {"MSML-TL-RFE": 10.0, "No-TL": 14.0, "SS-TL": 12.0},
        "D2": {"MSML-TL-RFE": 11.0, "No-TL": 15.0, "SS-TL": 13.0},
        "D3": {"MSML-TL-RFE": 12.0, "No-TL": 16.0},
    }
    for dataset, methods in values.items():
        for method, smape in methods.items():
            rows.extend(_seed_rows(dataset, method, smape))

    result = compare_methods_smape(pd.DataFrame(rows), anchor="MSML-TL-RFE")

    compared = result[result["status"] == "ok"]
    assert set(compared["method_b"]) == {"No-TL", "SS-TL"}
    assert set(compared["n_datasets"]) == {2}
    assert compared["p_value_holm"].notna().all()
    assert (compared["p_value_holm"] >= compared["p_value"]).all()
    assert compared["effect_size_rank_biserial"].notna().all()


def test_statistics_never_combines_horizons_or_sharing_scenarios():
    rows = []
    for horizon in (1, 5):
        for sharing in ("without", "with"):
            for dataset in ("D1", "D2"):
                rows.extend(
                    _seed_rows(
                        dataset,
                        "MSML-TL-RFE",
                        10.0,
                        horizon=horizon,
                        sharing=sharing,
                    )
                )
                rows.extend(
                    _seed_rows(
                        dataset,
                        "No-TL",
                        12.0,
                        horizon=horizon,
                        sharing=sharing,
                    )
                )

    result = compare_methods_smape(pd.DataFrame(rows), anchor="MSML-TL-RFE")

    assert result.groupby(["horizon", "sharing_scenario"]).ngroups == 4
    assert set(result["n_datasets"]) == {2}


def test_friedman_records_every_horizon_scenario_stratum() -> None:
    rows = []
    for horizon in (1, 5):
        for sharing in ("without", "with"):
            methods = ("Method-A", "Method-B", "Method-C")
            if (horizon, sharing) == (5, "with"):
                methods = ("Method-A", "Method-B")
            for dataset_index, dataset in enumerate(("D1", "D2"), start=1):
                for method_index, method in enumerate(methods, start=1):
                    rows.extend(
                        _seed_rows(
                            dataset,
                            method,
                            float(dataset_index + method_index),
                            horizon=horizon,
                            sharing=sharing,
                        )
                    )

    result = run_friedman_test(pd.DataFrame(rows))

    assert isinstance(result, pd.DataFrame)
    assert result.groupby(["horizon", "sharing_scenario"]).ngroups == 4
    assert list(result.columns) == [
        "horizon",
        "sharing_scenario",
        "n_datasets",
        "n_methods",
        "statistic",
        "p_value",
        "status",
    ]
    insufficient = result[
        result["horizon"].eq(5) & result["sharing_scenario"].eq("with")
    ].iloc[0]
    assert insufficient["n_methods"] == 2
    assert insufficient["status"] == "insufficient_data"
