"""Statistical significance tests for benchmark results."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


PIVOT_INDEX = "dataset"
PIVOT_COLUMNS = "method"
PIVOT_VALUES = "rmse"


def _to_method_dataset_rmse_table(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form results into dataset x method RMSE pivot table.

    Expected long-form columns: dataset, method, rmse.
    If duplicates exist, RMSE is aggregated by min to represent best run per
    method-dataset pair in matrix settings.
    """
    required = {"dataset", "method", "rmse"}
    missing = required.difference(results_dataframe.columns)
    if missing:
        raise ValueError(f"results_dataframe missing columns: {sorted(missing)}")

    df = results_dataframe.copy()
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna(subset=["dataset", "method", "rmse"])

    pivot = (
        df.groupby(["dataset", "method"], as_index=False)["rmse"]
        .min()
        .pivot(index=PIVOT_INDEX, columns=PIVOT_COLUMNS, values=PIVOT_VALUES)
        .sort_index()
    )
    return pivot


def run_friedman_test(results_dataframe: pd.DataFrame) -> Dict[str, float]:
    """Run Friedman test across methods using dataset-level RMSE blocks."""
    pivot = _to_method_dataset_rmse_table(results_dataframe)
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}

    samples = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
    statistic, p_value = friedmanchisquare(*samples)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def run_pairwise_wilcoxon_tests(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests for MSML-TL-RFE vs baselines.

    Comparisons:
    - No-TL
    - SS-TL
    - MSWA-TL
    - MSSB-TL
    - MSML-TL
    """
    pivot = _to_method_dataset_rmse_table(results_dataframe)

    anchor = "MSML-TL-RFE"
    baselines = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL"]

    rows: List[Dict[str, object]] = []
    if anchor not in pivot.columns:
        for b in baselines:
            rows.append(
                {
                    "method_a": anchor,
                    "method_b": b,
                    "n_datasets": 0,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "anchor_missing",
                }
            )
        return pd.DataFrame(rows)

    for b in baselines:
        if b not in pivot.columns:
            rows.append(
                {
                    "method_a": anchor,
                    "method_b": b,
                    "n_datasets": 0,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "baseline_missing",
                }
            )
            continue

        pair_df = pivot[[anchor, b]].dropna(how="any")
        n = int(len(pair_df))
        if n == 0:
            rows.append(
                {
                    "method_a": anchor,
                    "method_b": b,
                    "n_datasets": 0,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "no_overlap",
                }
            )
            continue

        if n < 2 or np.allclose(pair_df[anchor].values, pair_df[b].values):
            rows.append(
                {
                    "method_a": anchor,
                    "method_b": b,
                    "n_datasets": n,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "insufficient_or_identical",
                }
            )
            continue

        statistic, p_value = wilcoxon(pair_df[anchor].to_numpy(), pair_df[b].to_numpy())
        rows.append(
            {
                "method_a": anchor,
                "method_b": b,
                "n_datasets": n,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)


def compute_average_rank(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute average rank across datasets (lower RMSE gets better rank)."""
    pivot = _to_method_dataset_rmse_table(results_dataframe)

    if pivot.empty:
        return pd.DataFrame(columns=["method", "average_rank"])

    rank_rows: List[pd.Series] = []
    for _, row in pivot.iterrows():
        values = row.to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if not np.any(mask):
            continue

        ranked = np.full_like(values, np.nan, dtype=float)
        ranked[mask] = rankdata(values[mask], method="average")
        rank_rows.append(pd.Series(ranked, index=pivot.columns))

    if not rank_rows:
        return pd.DataFrame(columns=["method", "average_rank"])

    rank_df = pd.DataFrame(rank_rows)
    avg_rank = rank_df.mean(axis=0, skipna=True).sort_values(ascending=True)
    out = avg_rank.reset_index()
    out.columns = ["method", "average_rank"]
    return out
