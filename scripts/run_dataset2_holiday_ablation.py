"""Legacy script name kept for compatibility.

Current unified mapping treats the Rossmann holiday-feature ablation as Dataset3.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import pandas as pd

from experiment_runner import results_to_dataframe, run_all_experiments


def main() -> None:
    root = ROOT
    with (root / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    exp = cfg["single_experiment"]
    data_path = cfg["dataset_paths"]["Dataset3"]
    methods = ["No-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]

    settings = {
        "A": ["sales", "year", "month", "week", "day"],
        "B": ["sales", "year", "month", "week", "day", "holiday"],
    }

    rows = []
    for tag, feature_cols in settings.items():
        result = run_all_experiments(
            dataset_name="Dataset3",
            data_path=data_path,
            feature_cols=feature_cols,
            k=exp["k"],
            horizon=exp["horizon"],
            window_size=exp["window_size"],
            weight_mode=exp["weight_mode"],
            estimator_name=exp["estimator_name"],
            keep_ratio=exp["keep_ratio"],
            learning_rate=exp["learning_rate"],
            source_epochs=exp["source_epochs"],
            target_epochs=exp["target_epochs"],
            batch_size=exp["batch_size"],
            enabled_methods=methods,
            verbose_mode="summary",
            show_method_progress=False,
        )
        df = results_to_dataframe(result)
        for _, r in df.iterrows():
            rows.append(
                {
                    "feature_setting": tag,
                    "method": r["method"],
                    "rmse": float(r["rmse"]),
                    "accuracy": float(r["accuracy"]),
                }
            )

    out_df = pd.DataFrame(rows, columns=["feature_setting", "method", "rmse", "accuracy"])
    out_path = root / "outputs" / "ablation" / "dataset3_holiday_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    pivot = out_df.pivot(index="method", columns="feature_setting", values="rmse")
    if {"A", "B"}.issubset(set(pivot.columns)):
        pivot["delta_B_minus_A"] = pivot["B"] - pivot["A"]
        print(pivot[["A", "B", "delta_B_minus_A"]].sort_values("delta_B_minus_A"))
    print(f"saved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
