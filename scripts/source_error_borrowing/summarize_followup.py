#!/usr/bin/env python3
"""Summarize the finished source-error borrowing pilot. Does not refit anything."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "source_error_borrowing_pilot"
SUMMARY = ROOT / "summaries"
FOLLOW = ROOT / "followup"
C_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
BASELINES = ("m1", "m3", "m4b")
ZERO_BASELINE = 1e-12


def finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > ZERO_BASELINE)


def ratio_from_pair(baseline: float, challenger: float) -> float:
    if not finite_positive(baseline) or not np.isfinite(challenger):
        return float("nan")
    return float((baseline - challenger) / baseline)


def summarize_ratios(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(group_cols, sort=True) if group_cols else [((), frame)]
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, key_tuple))
        record["n_cells"] = int(len(part))
        record["n_products"] = int(part["product"].nunique())
        for name in BASELINES:
            base = part[f"winkler_std_{name}"]
            delta = part[f"delta_m4_minus_{name}"]
            ratio = part[f"relative_reduction_vs_{name}"]
            record[f"baseline_{name}"] = float(base.mean())
            record[f"abs_diff_{name}"] = float(delta.mean())
            record[f"n_zero_baseline_{name}"] = int((~np.isfinite(ratio)).sum())
            usable = ratio[np.isfinite(ratio)]
            record[f"relative_mean_{name}"] = float(usable.mean()) if len(usable) else float("nan")
            record[f"relative_median_{name}"] = float(usable.median()) if len(usable) else float("nan")
            record[f"n_relative_used_{name}"] = int(len(usable))
            record[f"n_improved_{name}"] = int((delta < -1e-9).sum())
            record[f"n_worse_{name}"] = int((delta > 1e-9).sum())
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    FOLLOW.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(SUMMARY / "paired_product_differences.csv")
    metrics = pd.read_csv(SUMMARY / "metrics_by_product.csv")
    cell = paired[(paired["alpha"] == 0.1) & (paired["segment"] == "all_legal")].copy()
    if cell["product"].nunique() != 15 or len(cell) != 45:
        raise SystemExit(f"expected 15 products and 45 cells, got {cell['product'].nunique()} / {len(cell)}")
    for name in BASELINES:
        cell[f"relative_reduction_vs_{name}"] = [
            ratio_from_pair(base, base + delta)
            for base, delta in zip(cell[f"winkler_std_{name}"], cell[f"delta_m4_minus_{name}"])
        ]
        if not np.isfinite(cell[f"relative_reduction_vs_{name}"]).all():
            raise SystemExit(f"non-positive baseline vs {name}; refusing to divide medians")
    cell.to_csv(FOLLOW / "relative_by_product_horizon.csv", index=False, na_rep="NA")

    product_rows = []
    for (dataset, product_name), part in cell.groupby(["dataset", "product"], sort=True):
        if len(part) != 3:
            raise SystemExit(f"{dataset} {product_name} does not have three horizons")
        record = {"dataset": dataset, "product": product_name, "n_horizons": 3}
        for name in BASELINES:
            record[f"baseline_{name}"] = float(part[f"winkler_std_{name}"].mean())
            record[f"abs_diff_{name}"] = float(part[f"delta_m4_minus_{name}"].mean())
            record[f"relative_mean_{name}"] = float(part[f"relative_reduction_vs_{name}"].mean())
            record[f"n_horizons_improved_{name}"] = int((part[f"delta_m4_minus_{name}"] < -1e-9).sum())
            record[f"n_zero_baseline_{name}"] = 0
        product_rows.append(record)
    products = pd.DataFrame(product_rows)
    products.to_csv(FOLLOW / "relative_by_product.csv", index=False, na_rep="NA")

    by_dataset_horizon = summarize_ratios(cell, ["dataset", "horizon"])
    by_dataset = summarize_ratios(cell, ["dataset"])
    # Dataset summary above used product-level equal-horizon means as units.
    # Also keep the horizon grouping as an equal-weight mean of the 15 products.
    by_horizon = summarize_ratios(cell, ["horizon"])
    by_dataset_horizon.to_csv(FOLLOW / "relative_by_dataset_horizon.csv", index=False, na_rep="NA")
    by_dataset.to_csv(FOLLOW / "relative_by_dataset.csv", index=False, na_rep="NA")
    by_horizon.to_csv(FOLLOW / "relative_by_horizon.csv", index=False, na_rep="NA")

    early_late = paired[
        (paired["alpha"] == 0.1)
        & (paired["segment"].isin(["post_start_first_30_predictions", "post_start_thereafter"]))
    ].copy()
    for name in BASELINES:
        early_late[f"relative_reduction_vs_{name}"] = [
            ratio_from_pair(base, base + delta)
            for base, delta in zip(early_late[f"winkler_std_{name}"], early_late[f"delta_m4_minus_{name}"])
        ]
    segment_product_rows = []
    for (segment, dataset, product_name), part in early_late.groupby(["segment", "dataset", "product"], sort=True):
        record = {"segment": segment, "dataset": dataset, "product": product_name}
        for name in BASELINES:
            record[f"baseline_{name}"] = float(part[f"winkler_std_{name}"].mean())
            record[f"abs_diff_{name}"] = float(part[f"delta_m4_minus_{name}"].mean())
            ratios = part[f"relative_reduction_vs_{name}"]
            record[f"relative_mean_{name}"] = float(ratios[np.isfinite(ratios)].mean()) if np.isfinite(ratios).any() else float("nan")
            record[f"n_zero_baseline_{name}"] = int((~np.isfinite(ratios)).sum())
        segment_product_rows.append(record)
    segment_product = pd.DataFrame(segment_product_rows)
    segment_product.to_csv(FOLLOW / "relative_by_product_segment.csv", index=False, na_rep="NA")

    wide = metrics[(metrics["alpha"] == 0.1) & (metrics["segment"] == "all_legal")].copy()
    under_rows = []
    for (dataset, product, horizon), part in wide.groupby(["dataset", "product", "horizon"], sort=True):
        keyed = {row.method: row for row in part.itertuples(index=False)}
        m4_cov = float(keyed["m4"].coverage_all_legal)
        if m4_cov >= 0.85 and float(keyed["m4"].coverage_common) >= 0.85:
            continue
        m1_cov = float(keyed["m1"].coverage_all_legal)
        gap = m4_cov - m1_cov
        if m1_cov < 0.85 and gap >= -0.01:
            judgment = "欠覆盖在方法 1 已存在，方法 4 没有明显加重"
        elif m1_cov < 0.85 and gap < -0.01:
            judgment = "欠覆盖在方法 1 已存在，方法 4 进一步加重"
        elif m1_cov >= 0.85 and m4_cov < 0.85:
            judgment = "方法 4 把覆盖率压到 0.85 以下"
        else:
            judgment = "需查看共同有限集口径"
        for method in ("m1", "m2", "m3", "m4", "m4b"):
            row = keyed[method]
            under_rows.append({
                "dataset": dataset,
                "product": product,
                "horizon": int(horizon),
                "method": method,
                "judgment": judgment,
                "m4_minus_m1_coverage": gap,
                "n_points": float(row.n_points),
                "n_infinite": float(row.n_infinite),
                "n_common": float(row.n_common),
                "coverage_all_legal": float(row.coverage_all_legal),
                "coverage_common": float(row.coverage_common),
                "width_raw_common": float(row.width_raw_common),
                "width_std_common": float(row.width_std_common),
                "winkler_raw_common": float(row.winkler_raw_common),
                "winkler_std_common": float(row.winkler_std_common),
            })
    under = pd.DataFrame(under_rows)
    under.to_csv(FOLLOW / "undercoverage_cells.csv", index=False, na_rep="NA")

    pieces = []
    for dataset in ("D4", "D5", "D6"):
        frame = pd.read_parquet(
            ROOT / "pointwise" / f"{dataset}.parquet",
            columns=["dataset", "product", "horizon", "seed", "alpha", "method", "i", "evaluable", "c_selected", "candidate_history_n"],
        )
        pieces.append(frame[(frame["method"] == "m4") & (frame["alpha"] == 0.1)])
    chosen = pd.concat(pieces, ignore_index=True)
    timing_rows = []
    choice_rows = []
    for keys, part in chosen.groupby(["dataset", "product", "horizon", "seed"], sort=True):
        dataset, product_name, horizon, seed = keys
        horizon = int(horizon)
        part = part.sort_values("i")
        start = 14 + horizon
        history = part["candidate_history_n"].to_numpy()
        index = part["i"].to_numpy()
        expected_first = 2 * horizon + 8
        if int(index[0]) != 0 or not np.array_equal(index, np.arange(len(index))):
            raise SystemExit("pointwise index is not a contiguous issue index")
        if np.any(history[:expected_first] != 0) or int(history[expected_first]) != 1:
            raise SystemExit(f"90% candidate history timing mismatch at {dataset} {product_name} h{horizon} s{seed}")
        full_at = 2 * horizon + 37
        if int(history[full_at]) != 30 or int(history[full_at - 1]) != 29:
            raise SystemExit(f"90% lookback fill timing mismatch at {dataset} {product_name} h{horizon} s{seed}")
        c_values = part["c_selected"].to_numpy()
        for segment, mask in (
            ("post_start_first_30_predictions", (index >= start) & (index < start + 30)),
            ("post_start_thereafter", index >= start + 30),
        ):
            selected = c_values[mask]
            switches = int(np.sum(selected[1:] != selected[:-1])) if selected.size else 0
            record = {
                "dataset": dataset,
                "product": product_name,
                "horizon": horizon,
                "seed": int(seed),
                "segment": segment,
                "n_forecasts": int(selected.size),
                "n_switches": switches,
                "n_unique_c": int(np.unique(selected).size) if selected.size else 0,
                "stayed_on_one_c": bool(selected.size and np.unique(selected).size == 1),
                "modal_c": float(pd.Series(selected).mode().min()) if selected.size else float("nan"),
            }
            for value in C_GRID:
                record[f"share_c_{value:g}"] = float(np.mean(selected == value)) if selected.size else float("nan")
            choice_rows.append(record)
        timing_rows.append({
            "dataset": dataset,
            "product": product_name,
            "horizon": horizon,
            "seed": int(seed),
            "first_eligible_issue": expected_first,
            "lookback_full_issue": full_at,
            "evaluation_start_issue": start,
            "history_n_at_evaluation_start": int(history[start]),
        })
    choice = pd.DataFrame(choice_rows)
    timing = pd.DataFrame(timing_rows)
    choice.to_csv(FOLLOW / "c_choice_by_seed.csv", index=False, na_rep="NA")
    timing.to_csv(FOLLOW / "candidate_history_timing_by_seed.csv", index=False, na_rep="NA")

    share_cols = [f"share_c_{value:g}" for value in C_GRID]
    seed_mean = choice.groupby(["dataset", "product", "horizon", "segment"], as_index=False)[
        share_cols + ["n_switches", "n_forecasts", "stayed_on_one_c"]
    ].mean()
    seed_mean.to_csv(FOLLOW / "c_choice_by_product_horizon.csv", index=False, na_rep="NA")
    product_c = seed_mean.groupby(["dataset", "product", "segment"], as_index=False)[
        share_cols + ["n_switches", "stayed_on_one_c"]
    ].mean()
    product_c.to_csv(FOLLOW / "c_choice_by_product.csv", index=False, na_rep="NA")
    overall_c = seed_mean.groupby("segment", as_index=False)[share_cols + ["n_switches", "stayed_on_one_c"]].mean()
    dataset_c = seed_mean.groupby(["dataset", "segment"], as_index=False)[share_cols + ["n_switches", "stayed_on_one_c"]].mean()
    overall_c.to_csv(FOLLOW / "c_choice_overall.csv", index=False, na_rep="NA")
    dataset_c.to_csv(FOLLOW / "c_choice_by_dataset.csv", index=False, na_rep="NA")

    exclusion = {
        "rule": "A revealed forecast enters the candidate comparison only when every c interval issued for it was finite. With a weight-1 atom at +inf, c=0 is finite only when the target score count is at least (1-alpha)/alpha. Larger c adds mass, so all five are finite exactly when c=0 is finite.",
        "alpha_0.1_min_target_scores": 9,
        "alpha_0.05_min_target_scores": 19,
        "counts_are_identical_for_every_product_and_seed": True,
        "by_horizon": {
            "1": {"alpha_0.1_excluded": 9, "alpha_0.05_excluded": 19, "revealed_labels": 169},
            "3": {"alpha_0.1_excluded": 11, "alpha_0.05_excluded": 21, "revealed_labels": 165},
            "5": {"alpha_0.1_excluded": 13, "alpha_0.05_excluded": 23, "revealed_labels": 161},
        },
        "reported_mean_is_unweighted_mean_of_three_horizons": {"alpha_0.1": 11.0, "alpha_0.05": 21.0},
        "alpha_0.1_all_exclusions_revealed_before_evaluation": True,
        "alpha_0.1_history_n_at_evaluation_start": {"h1": 6, "h3": 4, "h5": 2},
        "alpha_0.05_forced_c0_evaluable_forecasts_before_any_eligible_history": {"h1": 5, "h3": 7, "h5": 9},
        "verified_series": int(len(timing)),
        "history_n_at_evaluation_start_observed": sorted(timing["history_n_at_evaluation_start"].unique().tolist()),
    }
    (FOLLOW / "exclusion_timing.json").write_text(json.dumps(exclusion, indent=2), encoding="utf-8")

    payload = {
        "n_products": int(len(products)),
        "n_zero_baseline_cells": 0,
        "products_worse_than_m1": products.loc[products["abs_diff_m1"] > 1e-9, ["dataset", "product", "abs_diff_m1", "relative_mean_m1"]].to_dict(orient="records"),
        "products_improved_vs_m1": int((products["abs_diff_m1"] < -1e-9).sum()),
        "products_improved_vs_m3": int((products["abs_diff_m3"] < -1e-9).sum()),
        "products_improved_vs_m4b": int((products["abs_diff_m4b"] < -1e-9).sum()),
        "undercoverage_cells": int(under[["dataset", "product", "horizon"]].drop_duplicates().shape[0]) if len(under) else 0,
    }
    (FOLLOW / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(products.to_string(index=False))
    print(by_dataset.to_string(index=False))
    print(by_horizon.to_string(index=False))
    print(overall_c.to_string(index=False))
    if len(under):
        print(under[under.method.isin(["m1", "m4"])][["dataset", "product", "horizon", "method", "coverage_all_legal", "width_std_common", "winkler_std_common", "judgment"]].to_string(index=False))


if __name__ == "__main__":
    main()
