"""Formal original-sales-space sMAPE statistical comparisons."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from src.evaluation.metric_contract import build_formal_smape_aggregates
from src.protocols.experiment_protocol import FORMAL_SEEDS


def _dataset_macro(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    return build_formal_smape_aggregates(
        results_dataframe,
        expected_seeds=FORMAL_SEEDS,
    )["dataset_macro"]


def _holm_adjust(values: pd.Series) -> pd.Series:
    valid = values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def _rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    differences = differences[~np.isclose(differences, 0.0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    denominator = float(ranks.sum())
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / denominator)


def compare_methods_smape(
    results_dataframe: pd.DataFrame,
    *,
    anchor: str = "MSML-TL-RFE",
) -> pd.DataFrame:
    """Compare methods within fixed horizon/scenario using complete dataset blocks."""
    dataset_macro = _dataset_macro(results_dataframe)
    output: List[Dict[str, object]] = []
    if dataset_macro.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "sharing_scenario",
                "method_a",
                "method_b",
                "n_datasets",
                "statistic",
                "p_value",
                "p_value_holm",
                "effect_size_rank_biserial",
                "status",
            ]
        )

    for (horizon, scenario), group in dataset_macro.groupby(
        ["horizon", "sharing_scenario"], dropna=False
    ):
        pivot = group.pivot(index="dataset", columns="method", values="smape").sort_index()
        methods = sorted(str(method) for method in pivot.columns if str(method) != anchor)
        if anchor not in pivot.columns:
            for method in methods:
                output.append(
                    {
                        "horizon": horizon,
                        "sharing_scenario": scenario,
                        "method_a": anchor,
                        "method_b": method,
                        "n_datasets": 0,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "p_value_holm": np.nan,
                        "effect_size_rank_biserial": np.nan,
                        "status": "anchor_missing",
                    }
                )
            continue
        complete = pivot.dropna(axis=0, how="any")
        for method in methods:
            n_datasets = int(len(complete))
            base = {
                "horizon": horizon,
                "sharing_scenario": scenario,
                "method_a": anchor,
                "method_b": method,
                "n_datasets": n_datasets,
                "p_value_holm": np.nan,
            }
            if n_datasets < 2:
                output.append(
                    {
                        **base,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size_rank_biserial": np.nan,
                        "status": "descriptive_insufficient_datasets",
                    }
                )
                continue
            first = complete[anchor].to_numpy(dtype=float)
            second = complete[method].to_numpy(dtype=float)
            effect = _rank_biserial(first, second)
            if np.allclose(first, second):
                output.append(
                    {
                        **base,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size_rank_biserial": effect,
                        "status": "descriptive_identical",
                    }
                )
                continue
            statistic, p_value = wilcoxon(first, second)
            output.append(
                {
                    **base,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_size_rank_biserial": effect,
                    "status": "ok",
                }
            )

    result = pd.DataFrame(output)
    if not result.empty:
        for _, indices in result.groupby(["horizon", "sharing_scenario"]).groups.items():
            result.loc[indices, "p_value_holm"] = _holm_adjust(
                result.loc[indices, "p_value"]
            )
    return result


def run_pairwise_wilcoxon_tests(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    return compare_methods_smape(results_dataframe)


def run_friedman_test(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "sharing_scenario",
        "n_datasets",
        "n_methods",
        "statistic",
        "p_value",
        "status",
    ]
    dataset_macro = _dataset_macro(results_dataframe)
    if dataset_macro.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    for (horizon, scenario), group in dataset_macro.groupby(
        ["horizon", "sharing_scenario"],
        dropna=False,
        sort=True,
    ):
        pivot = group.pivot(
            index="dataset",
            columns="method",
            values="smape",
        ).dropna(how="any")
        n_datasets = int(pivot.shape[0])
        n_methods = int(pivot.shape[1])
        row: Dict[str, object] = {
            "horizon": horizon,
            "sharing_scenario": scenario,
            "n_datasets": n_datasets,
            "n_methods": n_methods,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "status": "insufficient_data",
        }
        if n_datasets >= 2 and n_methods >= 3:
            statistic, p_value = friedmanchisquare(
                *(pivot[column] for column in pivot.columns)
            )
            row.update(
                {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "status": "ok",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["horizon", "sharing_scenario"],
        kind="stable",
    ).reset_index(drop=True)


def compute_average_rank(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    dataset_macro = _dataset_macro(results_dataframe)
    if dataset_macro.empty:
        return pd.DataFrame(columns=["method", "horizon", "sharing_scenario", "average_rank"])
    rows = []
    for (horizon, scenario), group in dataset_macro.groupby(["horizon", "sharing_scenario"]):
        pivot = group.pivot(index="dataset", columns="method", values="smape")
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        for method, average_rank in ranks.mean(axis=0, skipna=True).items():
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "sharing_scenario": scenario,
                    "average_rank": float(average_rank),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "sharing_scenario", "average_rank"]
    ).reset_index(drop=True)
