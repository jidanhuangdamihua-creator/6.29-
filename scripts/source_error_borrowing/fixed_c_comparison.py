#!/usr/bin/env python3
"""Fixed-multiplier controls for the finished source-error borrowing pilot.

Replays the locked post-processing rule. Does not train, tune, or edit config.json.
Candidate quantiles are read from the same grid that method 4 indexes; they do not
depend on which candidate was selected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_and_run import (  # noqa: E402
    REPO,
    arrays_match,
    common_metrics,
    finite_bounds,
    load_config,
    log,
    run_panel,
    segment_masks,
)

OUT = REPO / "outputs" / "source_error_borrowing_pilot" / "fixed_c_controls"
OLD_METRICS = REPO / "outputs" / "source_error_borrowing_pilot" / "summaries" / "metrics_by_product.csv"
SCALE_PATH = REPO / "outputs" / "source_error_borrowing_pilot" / "audit" / "scale_and_zeros.csv"
COMPARE = ["m1", "m4", "m4b", "fixed_c_0.5", "fixed_c_1", "fixed_c_2", "fixed_c_4"]
FIXED = ["fixed_c_0.5", "fixed_c_1", "fixed_c_2", "fixed_c_4"]
SEGMENTS = ["all_legal", "post_start_first_30_predictions", "post_start_thereafter"]
FIVE = ["166_311", "166_433", "48_1159414", "48_364606", "CA_1_FOODS_3_586"]
ZERO = 1e-12


def ratio(baseline: float, challenger: float) -> float:
    if not (np.isfinite(baseline) and baseline > ZERO and np.isfinite(challenger)):
        return float("nan")
    return float((baseline - challenger) / baseline)


def prediction_path(config: dict, dataset: str, product: str, horizon: int, seed: int, learning_rate: str) -> Path:
    return (
        REPO / config["prediction_root"] / dataset / product
        / f"h{int(horizon)}_s{int(seed)}_{learning_rate}" / "predictions.csv"
    )


def main() -> None:
    config = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    alphas = np.array([
        float(config["quantile"]["primary_alpha"]),
        float(config["quantile"]["secondary_alpha"]),
    ])
    c_grid = np.asarray(config["candidate_control"]["c_grid"], dtype=np.float64)
    expected = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    if not np.array_equal(c_grid, expected):
        raise SystemExit(f"locked c grid changed: {c_grid}")
    fixed_index = {"fixed_c_0.5": 1, "fixed_c_1": 2, "fixed_c_2": 3, "fixed_c_4": 4}
    scales = pd.read_csv(SCALE_PATH).set_index(["dataset", "product"])
    old = pd.read_csv(OLD_METRICS)
    old = old[old["method"] == "m1"][
        ["dataset", "product", "horizon", "alpha", "segment", "n_points", "n_common"]
    ].rename(columns={"n_points": "old_n_points", "n_common": "old_n_common"})

    rows = []
    common_rows = []
    dependency_ok = True
    for dataset, products in config["products"].items():
        learning_rate = config["v3_primary_targets_actually_used"][dataset]["learning_rate"]
        for horizon in config["horizons"]:
            horizon = int(horizon)
            for seed in config["seeds"]:
                seed = int(seed)
                y = []
                yhat = []
                scale = []
                for product in products:
                    frame = pd.read_csv(prediction_path(config, dataset, product, horizon, seed, learning_rate))
                    y.append(frame["y_true_original"].to_numpy(dtype=np.float64))
                    yhat.append(frame["y_pred_original"].to_numpy(dtype=np.float64))
                    scale.append(float(scales.loc[(dataset, product), "scale"]))
                y = np.vstack(y)
                yhat = np.vstack(yhat)
                scale = np.asarray(scale, dtype=np.float64)
                result = run_panel(
                    y, yhat, scale, horizon,
                    alphas=alphas, c_grid=c_grid, n0=int(config["startup"]["n0"]),
                    delta=float(config["similarity"]["delta"]),
                    lookback=int(config["candidate_control"]["lookback"]),
                    min_scores=int(config["similarity"]["method3_min_scores_per_source"]),
                    min_sources=int(config["similarity"]["method3_min_sources_with_5_scores"]),
                )
                if result["c0_error"] > 1e-8 or not arrays_match(result["q"][0], result["q_c"][0]):
                    dependency_ok = False
                    raise SystemExit(f"c=0 candidate diverged from method 1 at {dataset} h{horizon} s{seed}")
                for alpha_i, _alpha in enumerate(alphas):
                    chosen = result["chosen_c"][alpha_i]
                    for product_i in range(len(products)):
                        index = np.array([
                            int(np.flatnonzero(np.isclose(c_grid, value))[0]) for value in chosen[product_i]
                        ])
                        picked = result["q_c"][index, alpha_i, product_i, np.arange(index.size)]
                        if not arrays_match(picked, result["q"][3, alpha_i, product_i]):
                            dependency_ok = False
                            raise SystemExit("method 4 quantile is not an index into the precomputed candidate grid")
                masks = segment_masks(y.shape[1], horizon)
                for product_i, product in enumerate(products):
                    for alpha_i, alpha in enumerate(alphas):
                        quantiles = {
                            "m1": result["q"][0, alpha_i, product_i],
                            "m4": result["q"][3, alpha_i, product_i],
                            "m4b": result["q"][4, alpha_i, product_i],
                        }
                        for name, index in fixed_index.items():
                            quantiles[name] = result["q_c"][index, alpha_i, product_i]
                        lows = []
                        highs = []
                        for name in COMPARE:
                            low, high = finite_bounds(yhat[product_i], float(scale[product_i]), quantiles[name])
                            lows.append(low)
                            highs.append(high)
                        for segment in SEGMENTS:
                            metrics = common_metrics(
                                y[product_i], lows, highs, float(scale[product_i]), float(alpha), masks[segment]
                            )
                            for method_i, method in enumerate(COMPARE):
                                nominal = 1.0 - float(alpha)
                                item = metrics[method_i]
                                rows.append({
                                    "dataset": dataset,
                                    "product": product,
                                    "horizon": horizon,
                                    "seed": seed,
                                    "alpha": float(alpha),
                                    "segment": segment,
                                    "method": method,
                                    "nominal_coverage": nominal,
                                    **item,
                                    "coverage_deviation_all": item["coverage_all_legal"] - nominal,
                                    "coverage_deviation_common": item["coverage_common"] - nominal,
                                })
                log(f"FIXED_C_DONE {dataset} h{horizon} s{seed}")

    seed_metrics = pd.DataFrame(rows)
    value_cols = [
        "n_points", "n_infinite", "coverage_all_legal", "coverage_deviation_all",
        "n_common", "coverage_common", "coverage_deviation_common",
        "width_std_common", "winkler_std_common",
    ]
    keys = ["dataset", "product", "horizon", "alpha", "segment", "method"]
    by_cell = seed_metrics.groupby(keys, as_index=False)[value_cols].mean()
    by_cell.to_csv(OUT / "metrics_by_product_horizon.csv", index=False, na_rep="NA")
    seed_metrics.to_csv(OUT / "metrics_by_seed.csv", index=False, na_rep="NA")

    check = by_cell.merge(old, on=["dataset", "product", "horizon", "alpha", "segment"], how="left")
    check["abs_n_common_gap"] = (check["n_common"] - check["old_n_common"]).abs()
    check["count_matches_previous_common_set"] = check["abs_n_common_gap"] <= 1e-6
    check["set_is_all_legal_points"] = np.isclose(check["n_common"], check["n_points"]) & (check["n_points"] > 0)
    check_summary = check.groupby(["alpha", "segment"], as_index=False).agg(
        n_cells=("n_common", "size"),
        n_match_old=("count_matches_previous_common_set", "sum"),
        n_all_legal=("set_is_all_legal_points", "sum"),
        max_abs_n_common_gap=("abs_n_common_gap", "max"),
    )
    check.to_csv(OUT / "common_set_by_cell.csv", index=False, na_rep="NA")
    check_summary.to_csv(OUT / "common_set_summary.csv", index=False, na_rep="NA")

    def add_ratios(frame: pd.DataFrame, unit_cols: list[str]) -> pd.DataFrame:
        pieces = []
        for key, part in frame.groupby(unit_cols + ["alpha", "segment"], sort=False):
            keyed = {row.method: row for row in part.itertuples(index=False)}
            base = keyed["m4"].winkler_std_common
            m1 = keyed["m1"].winkler_std_common
            record = dict(zip(unit_cols + ["alpha", "segment"], key if isinstance(key, tuple) else (key,)))
            for method in COMPARE:
                row = keyed[method]
                record[f"winkler_{method}"] = float(row.winkler_std_common)
                record[f"coverage_common_{method}"] = float(row.coverage_common)
                record[f"coverage_all_{method}"] = float(row.coverage_all_legal)
                record[f"coverage_deviation_common_{method}"] = float(row.coverage_deviation_common)
                record[f"width_{method}"] = float(row.width_std_common)
                record[f"n_points_{method}"] = float(row.n_points)
                record[f"n_infinite_{method}"] = float(row.n_infinite)
                record[f"n_common_{method}"] = float(row.n_common)
                record[f"abs_vs_m1_{method}"] = float(row.winkler_std_common - m1)
                record[f"relative_vs_m1_{method}"] = ratio(m1, row.winkler_std_common)
            for method in ["m1", "m4b", *FIXED]:
                other = keyed[method].winkler_std_common
                record[f"abs_m4_minus_{method}"] = float(base - other)
                record[f"relative_m4_vs_{method}"] = ratio(other, base)
            finite_fixed = [keyed[method].winkler_std_common for method in FIXED]
            best_index = int(np.nanargmin(finite_fixed))
            record["diagnostic_best_fixed_c"] = [0.5, 1.0, 2.0, 4.0][best_index]
            record["diagnostic_best_fixed_winkler"] = float(finite_fixed[best_index])
            record["diagnostic_uses_future_evaluation_labels"] = True
            pieces.append(record)
        return pd.DataFrame(pieces)

    paired_cell = add_ratios(by_cell, ["dataset", "product", "horizon"])
    paired_cell.to_csv(OUT / "paired_by_product_horizon.csv", index=False, na_rep="NA")

    product_rows = []
    for (dataset, product, alpha, segment), part in paired_cell.groupby(["dataset", "product", "alpha", "segment"]):
        if len(part) != 3:
            raise SystemExit(f"{dataset} {product} segment {segment} does not have 3 horizons")
        record = {"dataset": dataset, "product": product, "alpha": alpha, "segment": segment, "n_horizons": 3}
        for method in COMPARE:
            record[f"winkler_{method}"] = float(part[f"winkler_{method}"].mean())
            record[f"coverage_common_{method}"] = float(part[f"coverage_common_{method}"].mean())
            record[f"coverage_all_{method}"] = float(part[f"coverage_all_{method}"].mean())
            record[f"coverage_deviation_common_{method}"] = float(part[f"coverage_deviation_common_{method}"].mean())
            record[f"width_{method}"] = float(part[f"width_{method}"].mean())
            record[f"n_points_{method}"] = float(part[f"n_points_{method}"].mean())
            record[f"n_infinite_{method}"] = float(part[f"n_infinite_{method}"].mean())
            record[f"n_common_{method}"] = float(part[f"n_common_{method}"].mean())
            relatives = part[f"relative_vs_m1_{method}"]
            record[f"abs_vs_m1_{method}"] = float(part[f"abs_vs_m1_{method}"].mean())
            record[f"relative_vs_m1_{method}"] = float(relatives[np.isfinite(relatives)].mean()) if np.isfinite(relatives).any() else float("nan")
            record[f"n_zero_baseline_vs_m1_{method}"] = int((~np.isfinite(relatives)).sum())
        for method in ["m1", "m4b", *FIXED]:
            relatives = part[f"relative_m4_vs_{method}"]
            record[f"abs_m4_minus_{method}"] = float(part[f"abs_m4_minus_{method}"].mean())
            record[f"relative_m4_vs_{method}"] = float(relatives[np.isfinite(relatives)].mean()) if np.isfinite(relatives).any() else float("nan")
            record[f"n_zero_baseline_m4_vs_{method}"] = int((~np.isfinite(relatives)).sum())
        fixed_means = [record[f"winkler_{method}"] for method in FIXED]
        best_index = int(np.nanargmin(fixed_means))
        record["diagnostic_best_fixed_c"] = [0.5, 1.0, 2.0, 4.0][best_index]
        record["diagnostic_best_fixed_winkler"] = float(fixed_means[best_index])
        record["diagnostic_uses_future_evaluation_labels"] = True
        product_rows.append(record)
    products = pd.DataFrame(product_rows)
    products.to_csv(OUT / "paired_by_product.csv", index=False, na_rep="NA")

    def group_mean(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        rows_out = []
        for key, part in frame.groupby(group_cols + ["alpha", "segment"]):
            record = dict(zip(group_cols + ["alpha", "segment"], key if isinstance(key, tuple) else (key,)))
            record["n_products"] = int(part["product"].nunique()) if "product" in part.columns else int(len(part))
            record["n_rows"] = int(len(part))
            for method in COMPARE:
                record[f"winkler_{method}"] = float(part[f"winkler_{method}"].mean())
                record[f"coverage_common_{method}"] = float(part[f"coverage_common_{method}"].mean())
                record[f"coverage_deviation_common_{method}"] = float(part[f"coverage_deviation_common_{method}"].mean())
                record[f"width_{method}"] = float(part[f"width_{method}"].mean())
                record[f"n_infinite_{method}"] = float(part[f"n_infinite_{method}"].mean())
                record[f"n_common_{method}"] = float(part[f"n_common_{method}"].mean())
            for method in ["m1", "m4b", *FIXED]:
                relatives = part[f"relative_m4_vs_{method}"]
                record[f"abs_m4_minus_{method}"] = float(part[f"abs_m4_minus_{method}"].mean())
                record[f"relative_mean_m4_vs_{method}"] = float(np.nanmean(relatives))
                record[f"relative_median_m4_vs_{method}"] = float(np.nanmedian(relatives))
                record[f"n_m4_better_{method}"] = int((part[f"abs_m4_minus_{method}"] < -1e-9).sum())
                record[f"n_m4_worse_{method}"] = int((part[f"abs_m4_minus_{method}"] > 1e-9).sum())
                record[f"n_zero_baseline_{method}"] = int((~np.isfinite(relatives)).sum())
            rows_out.append(record)
        return pd.DataFrame(rows_out)

    by_dataset = group_mean(products, ["dataset"])
    by_horizon = group_mean(paired_cell, ["horizon"])
    by_dataset_horizon = group_mean(paired_cell, ["dataset", "horizon"])
    by_dataset.to_csv(OUT / "summary_by_dataset.csv", index=False, na_rep="NA")
    by_horizon.to_csv(OUT / "summary_by_horizon.csv", index=False, na_rep="NA")
    by_dataset_horizon.to_csv(OUT / "summary_by_dataset_horizon.csv", index=False, na_rep="NA")

    harm_rows = []
    primary_products = products[(products["alpha"] == 0.1) & (products["segment"] == "all_legal")]
    for method in ["m4", "m4b", *FIXED]:
        harmed = primary_products[primary_products[f"abs_vs_m1_{method}"] > 1e-9]
        if len(harmed):
            worst = harmed.loc[harmed[f"relative_vs_m1_{method}"].idxmin()]
            max_worsening = float(worst[f"relative_vs_m1_{method}"])
            worst_name = f"{worst['dataset']} {worst['product']}"
        else:
            max_worsening = float("nan")
            worst_name = ""
        harm_rows.append({
            "method": method,
            "n_harmed_products": int(len(harmed)),
            "n_products": int(len(primary_products)),
            "max_relative_worsening": max_worsening,
            "worst_product": worst_name,
        })
        for record in harmed.to_dict(orient="records"):
            harm_rows.append({
                "method": method,
                "dataset": record["dataset"],
                "product": record["product"],
                "abs_vs_m1": record[f"abs_vs_m1_{method}"],
                "relative_vs_m1": record[f"relative_vs_m1_{method}"],
                "coverage_common_method": record[f"coverage_common_{method}"],
                "coverage_deviation_common_method": record[f"coverage_deviation_common_{method}"],
                "coverage_common_m1": record["coverage_common_m1"],
                "coverage_deviation_common_m1": record["coverage_deviation_common_m1"],
                "width_method": record[f"width_{method}"],
                "width_m1": record["width_m1"],
                "winkler_method": record[f"winkler_{method}"],
                "winkler_m1": record["winkler_m1"],
                "n_common": record[f"n_common_{method}"],
                "n_infinite": record[f"n_infinite_{method}"],
            })
    pd.DataFrame(harm_rows).to_csv(OUT / "harm_vs_m1.csv", index=False, na_rep="NA")

    five = products[products["product"].isin(FIVE) & (products["alpha"] == 0.1)].copy()
    five.to_csv(OUT / "five_products_all_methods.csv", index=False, na_rep="NA")
    five_cells = paired_cell[paired_cell["product"].isin(FIVE) & (paired_cell["alpha"] == 0.1) & (paired_cell["segment"] == "all_legal")]
    five_cells.to_csv(OUT / "five_products_by_horizon.csv", index=False, na_rep="NA")

    oracle = primary_products[[
        "dataset", "product", "winkler_m1", "winkler_m4", "diagnostic_best_fixed_c",
        "diagnostic_best_fixed_winkler", "diagnostic_uses_future_evaluation_labels",
    ]].copy()
    oracle["note"] = "diagnostic reference only; the best fixed multiplier is chosen from this same evaluation window"
    oracle.to_csv(OUT / "diagnostic_oracle_best_fixed_c.csv", index=False, na_rep="NA")

    payload = {
        "candidate_intervals_saved_in_original_pointwise": False,
        "candidate_quantiles_independent_of_selected_c": dependency_ok,
        "reason": "Each candidate quantile is computed from the revealed residual pool and M_t=c*n_t before method 4 reads the grid. Method 4 stores the already computed entry for the selected c.",
        "common_set_summary": check_summary.to_dict(orient="records"),
        "replay_completed": True,
    }
    (OUT / "replay_notes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"FIXED_C_COMPLETE output={OUT}")
    print(check_summary.to_string(index=False))
    show = primary_products[["dataset", "product", "winkler_m1", "winkler_m4", "relative_m4_vs_m1", "relative_m4_vs_fixed_c_0.5", "relative_m4_vs_fixed_c_1", "relative_m4_vs_fixed_c_2", "relative_m4_vs_fixed_c_4", "relative_m4_vs_m4b"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
