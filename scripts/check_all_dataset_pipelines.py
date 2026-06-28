"""Check full preprocessing pipelines for Dataset1/2/3.

This script verifies the following chain for each dataset:
1) load_dataset
2) extract_datetime_features
3) build_source_target_split
4) temporal_split_by_ratio_or_dates
5) normalize_features
6) build_tabular_sequence

Outputs a summary CSV to outputs/pipeline_checks/all_dataset_pipeline_check.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dataset_registry import list_dataset_names

from data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)


DATASETS = list_dataset_names()
OUTPUT_DIR = ROOT / "outputs" / "pipeline_checks"
OUTPUT_CSV = OUTPUT_DIR / "all_dataset_pipeline_check.csv"


def _shape_of(df: pd.DataFrame) -> str:
    return str(tuple(df.shape))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_data_path(dataset_paths: Dict[str, str], dataset_name: str) -> Path:
    if dataset_name not in dataset_paths:
        raise KeyError(f"dataset_paths missing key: {dataset_name}")
    p = Path(dataset_paths[dataset_name])
    return p if p.is_absolute() else (ROOT / p)


def _print_dataset_report(
    dataset_name: str,
    raw_shape: str,
    processed_shape: str,
    source_shape: str,
    target_shape: str,
    train_shape: str,
    val_shape: str,
    test_shape: str,
    x_shape: str,
    y_shape: str,
) -> None:
    print(f"\n[{dataset_name}] Pipeline check")
    print(f"  raw dataframe shape: {raw_shape}")
    print(f"  processed shape: {processed_shape}")
    print(f"  source shape: {source_shape}")
    print(f"  target shape: {target_shape}")
    print(f"  train/val/test shape: {train_shape} / {val_shape} / {test_shape}")
    print(f"  sequence X/y shape: {x_shape} / {y_shape}")


def main() -> None:
    dataset_paths_cfg = _load_json(ROOT / "configs" / "dataset_paths.json")
    default_cfg = _load_json(ROOT / "configs" / "default_config.json")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for dataset_name in DATASETS:
        raw_shape = "N/A"
        processed_shape = "N/A"
        source_shape = "N/A"
        target_shape = "N/A"
        train_shape = "N/A"
        val_shape = "N/A"
        test_shape = "N/A"
        train_x_shape = "N/A"
        train_y_shape = "N/A"

        try:
            data_path = _resolve_data_path(dataset_paths_cfg, dataset_name)

            raw_df = load_dataset(dataset_name=dataset_name, data_path=str(data_path))
            raw_shape = _shape_of(raw_df)

            processed_df = extract_datetime_features(raw_df)
            processed_shape = _shape_of(processed_df)

            source_df, target_df = build_source_target_split(processed_df, default_cfg)
            source_shape = _shape_of(source_df)
            target_shape = _shape_of(target_df)

            train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)
            train_shape = _shape_of(train_df)
            val_shape = _shape_of(val_df)
            test_shape = _shape_of(test_df)

            train_scaled, val_scaled, test_scaled, _, _ = normalize_features(train_df, val_df, test_df)

            window_size = int(default_cfg.get("single_experiment", {}).get("window_size", 10))
            horizon = int(default_cfg.get("single_experiment", {}).get("horizon", 1))

            x_train, y_train = build_tabular_sequence(
                train_scaled,
                horizon=horizon,
                window_size=window_size,
            )
            train_x_shape = str(tuple(x_train.shape))
            train_y_shape = str(tuple(y_train.shape))

            _print_dataset_report(
                dataset_name=dataset_name,
                raw_shape=raw_shape,
                processed_shape=processed_shape,
                source_shape=source_shape,
                target_shape=target_shape,
                train_shape=train_shape,
                val_shape=val_shape,
                test_shape=test_shape,
                x_shape=train_x_shape,
                y_shape=train_y_shape,
            )

            records.append(
                {
                    "dataset_name": dataset_name,
                    "status": "success",
                    "raw_shape": raw_shape,
                    "processed_shape": processed_shape,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                    "train_x_shape": train_x_shape,
                    "train_y_shape": train_y_shape,
                    "error_message": "",
                }
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"\n[{dataset_name}] Pipeline check")
            print(f"  error_message: {error_message}")

            records.append(
                {
                    "dataset_name": dataset_name,
                    "status": "failed",
                    "raw_shape": raw_shape,
                    "processed_shape": processed_shape,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                    "train_x_shape": train_x_shape,
                    "train_y_shape": train_y_shape,
                    "error_message": error_message,
                }
            )

    summary_df = pd.DataFrame(
        records,
        columns=[
            "dataset_name",
            "status",
            "raw_shape",
            "processed_shape",
            "source_shape",
            "target_shape",
            "train_x_shape",
            "train_y_shape",
            "error_message",
        ],
    )
    summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n=== Pipeline Check Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
