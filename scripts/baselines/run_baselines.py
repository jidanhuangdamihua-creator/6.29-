#!/usr/bin/env python3
"""Run BL1-BL4 target-only baselines for D1-D6."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, BASELINE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from baseline_data_loader import load_baseline_data
from baseline_metrics import compute_metrics
from bl1_historical_mean import predict_bl1
from bl2_moving_average import predict_bl2
from bl3_lightgbm import predict_bl3


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
OUTPUT_COLUMNS = ["dataset_id", "entity_key", "method", "smape", "rmse", "mae"]


def _result_row(dataset_id: str, entity_key: str, method: str, metrics: dict) -> dict:
    smape = float(metrics["smape"])
    if not 0.0 <= smape <= 200.0:
        print(
            f"WARNING: {dataset_id}/{entity_key}/{method} sMAPE={smape:.6f} "
            "is outside [0, 200]"
        )
    return {
        "dataset_id": dataset_id,
        "entity_key": entity_key,
        "method": method,
        "smape": smape,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
    }


def run_dataset(dataset_id: str) -> Path:
    """Run all four baselines for one dataset and write its result CSV."""
    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError(f"dataset_id must be d1 through d6, got {dataset_id!r}")

    entity_slices = load_baseline_data(normalized_id)
    rows = []
    for data in entity_slices:
        entity_key = str(data["entity_key"])
        truth = data["test_sales"]
        test_len = int(data["test_len"])

        predictions = [
            (
                "BL1_HistoricalMean",
                predict_bl1(data["observed_sales"], test_len),
            ),
            (
                "BL2_MovingAverage",
                predict_bl2(data["observed_sales"], test_len),
            ),
            (
                "BL3_LightGBM",
                predict_bl3(
                    data["feature_df"],
                    data["test_feature_df"],
                    random_state=42,
                ),
            ),
        ]
        from bl4_lstm import predict_bl4

        lstm_seed_42 = predict_bl4(
            data["train_sales"],
            data["val_sales"],
            test_len,
            seed=42,
        )
        predictions.append(("BL4_LSTM", lstm_seed_42))

        for method, prediction in predictions:
            rows.append(
                _result_row(
                    normalized_id,
                    entity_key,
                    method,
                    compute_metrics(truth, prediction),
                )
            )

        lstm_seed_43 = predict_bl4(
            data["train_sales"],
            data["val_sales"],
            test_len,
            seed=43,
        )
        seed_42_smape = compute_metrics(truth, lstm_seed_42)["smape"]
        seed_43_smape = compute_metrics(truth, lstm_seed_43)["smape"]
        difference = abs(float(seed_42_smape) - float(seed_43_smape))
        print(
            f"[BL4 variance] dataset={normalized_id} entity={entity_key} "
            f"seed42_smape={seed_42_smape:.6f} "
            f"seed43_smape={seed_43_smape:.6f} "
            f"absolute_difference={difference:.6f}"
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    expected_rows = len(entity_slices) * 4
    if len(output) != expected_rows:
        raise AssertionError(
            f"{normalized_id} expected {expected_rows} baseline rows, got {len(output)}"
        )
    numeric_metrics = output[["smape", "rmse", "mae"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_metrics).all() or (numeric_metrics < 0.0).any():
        raise AssertionError(f"{normalized_id} output metrics must be finite and non-negative")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{normalized_id}_baselines.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(output)} rows to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target-only BL1-BL4 baselines for D1-D6."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[f"d{number}" for number in range(1, 7)] + ["all"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_ids = (
        [f"d{number}" for number in range(1, 7)]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset_id in dataset_ids:
        run_dataset(dataset_id)


if __name__ == "__main__":
    main()
