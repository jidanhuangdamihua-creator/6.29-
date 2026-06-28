"""Minimal training check for Dataset1/2/3.

Goal:
- Validate each dataset can finish at least one minimal training run and output RMSE.

Strategy:
- Prefer SS-TL.
- Fallback to MSML-TL if SS-TL fails for a dataset.

Output:
- outputs/pipeline_checks/all_dataset_min_train_check.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from dataset_registry import list_dataset_names

from experiment_runner import (
    prepare_base_data_for_experiments,
    run_msml_experiment,
    run_ss_tl_experiment,
)


DATASETS = list_dataset_names()
OUTPUT_DIR = ROOT / "outputs" / "pipeline_checks"
OUTPUT_CSV = OUTPUT_DIR / "all_dataset_min_train_check.csv"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shape_to_str(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, tuple):
        return str(value)
    if isinstance(value, list):
        return str(tuple(value))
    return str(value)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.nan)


def main() -> None:
    dataset_paths_cfg = _load_json(ROOT / "configs" / "dataset_paths.json")
    default_cfg = _load_json(ROOT / "configs" / "default_config.json")

    feature_cols = default_cfg.get("features", {}).get(
        "default_feature_cols",
        ["sales", "year", "month", "week", "day"],
    )
    window_size = int(default_cfg.get("single_experiment", {}).get("window_size", 10))
    learning_rate = float(default_cfg.get("single_experiment", {}).get("learning_rate", 0.001))

    horizon = 1
    k = 1
    source_epochs = 1
    target_epochs = 1
    batch_size = 16
    weight_mode = "inverse_distance"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for dataset_name in DATASETS:
        status = "failed"
        method = "N/A"
        rmse = float(np.nan)
        accuracy = float(np.nan)
        prediction_shape = "N/A"
        error_message = ""

        try:
            if dataset_name not in dataset_paths_cfg:
                raise KeyError(f"dataset_paths missing key: {dataset_name}")

            data_path = Path(dataset_paths_cfg[dataset_name])
            if not data_path.is_absolute():
                data_path = ROOT / data_path

            bundle = prepare_base_data_for_experiments(
                dataset_name=dataset_name,
                data_path=str(data_path),
                config=None,
            )
            source_df = bundle["source_df"]
            target_df = bundle["target_df"]

            try:
                result = run_ss_tl_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=feature_cols,
                    horizon=horizon,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                )
                method = "SS-TL"
                status = "success"
                rmse = _safe_float(result.get("rmse"))
                accuracy = _safe_float(result.get("accuracy"))
                prediction_shape = _shape_to_str(result.get("prediction_shape"))
            except Exception as ss_exc:
                try:
                    result = run_msml_experiment(
                        source_df=source_df,
                        target_df=target_df,
                        feature_cols=feature_cols,
                        k=k,
                        horizon=horizon,
                        window_size=window_size,
                        weight_mode=weight_mode,
                        learning_rate=learning_rate,
                        source_epochs=source_epochs,
                        target_epochs=target_epochs,
                        batch_size=batch_size,
                    )
                    method = "MSML-TL"
                    status = "success"
                    rmse = _safe_float(result.get("rmse"))
                    accuracy = _safe_float(result.get("accuracy"))
                    prediction_shape = _shape_to_str(result.get("prediction_shape"))
                    error_message = f"SS-TL failed, fallback used: {type(ss_exc).__name__}: {ss_exc}"
                except Exception as msml_exc:
                    status = "failed"
                    method = "SS-TL->MSML-TL"
                    error_message = (
                        f"SS-TL error: {type(ss_exc).__name__}: {ss_exc} | "
                        f"MSML-TL error: {type(msml_exc).__name__}: {msml_exc}"
                    )
        except Exception as exc:
            status = "failed"
            method = "N/A"
            error_message = f"{type(exc).__name__}: {exc}"

        print(f"\n[{dataset_name}] Minimal Train Check")
        print(f"  status: {status}")
        print(f"  method: {method}")
        print(f"  rmse: {rmse}")
        print(f"  accuracy: {accuracy}")
        print(f"  prediction_shape: {prediction_shape}")
        print(f"  error_message: {error_message}")

        records.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "status": status,
                "rmse": rmse,
                "accuracy": accuracy,
                "prediction_shape": prediction_shape,
                "error_message": error_message,
            }
        )

    summary_df = pd.DataFrame(
        records,
        columns=[
            "dataset_name",
            "method",
            "status",
            "rmse",
            "accuracy",
            "prediction_shape",
            "error_message",
        ],
    )
    summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n=== Minimal Training Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
