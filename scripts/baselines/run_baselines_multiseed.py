#!/usr/bin/env python3
"""Run BL1-BL4 target-only baselines for D1-D6, with BL4 expanded across multiple seeds.

This is a standalone variant of run_baselines.py. It does not modify or overwrite
the original script or its output (`{dataset}_baselines.csv`); results are written
to a separate `{dataset}_baselines_multiseed.csv` file.

Differences from run_baselines.py:
  - BL4_LSTM is run once per seed (default: 42,43,44,45,46) instead of a single
    seed=42 result plus a seed=43 variance check.
  - Output rows include a `seed` column. BL1/BL2/BL3 rows use seed=42 as a fixed
    placeholder (these methods are deterministic / unaffected by the LSTM seed).
  - Output file name is suffixed with `_multiseed` to avoid touching the original
    baseline CSVs already integrated into the Excel dashboard.
"""

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
from bl4_lstm import predict_bl4


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
OUTPUT_COLUMNS = ["dataset_id", "entity_key", "method", "seed", "smape", "rmse", "mae"]
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
FIXED_METHOD_SEED_PLACEHOLDER = 42  # BL1-BL3 are deterministic; seed column is a placeholder


def _result_row(
    dataset_id: str, entity_key: str, method: str, seed: int, metrics: dict
) -> dict:
    smape = float(metrics["smape"])
    if not 0.0 <= smape <= 200.0:
        print(
            f"WARNING: {dataset_id}/{entity_key}/{method}/seed={seed} sMAPE={smape:.6f} "
            "is outside [0, 200]"
        )
    return {
        "dataset_id": dataset_id,
        "entity_key": entity_key,
        "method": method,
        "seed": seed,
        "smape": smape,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
    }


def run_dataset(dataset_id: str, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> Path:
    """Run BL1-BL3 once and BL4 across all `seeds` for one dataset; write result CSV."""
    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError(f"dataset_id must be d1 through d6, got {dataset_id!r}")

    entity_slices = load_baseline_data(normalized_id)
    rows = []
    for data in entity_slices:
        entity_key = str(data["entity_key"])
        truth = data["test_sales"]
        test_len = int(data["test_len"])

        # BL1-BL3: deterministic, run once per entity.
        fixed_predictions = [
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
        for method, prediction in fixed_predictions:
            rows.append(
                _result_row(
                    normalized_id,
                    entity_key,
                    method,
                    FIXED_METHOD_SEED_PLACEHOLDER,
                    compute_metrics(truth, prediction),
                )
            )

        # BL4: run once per seed.
        seed_smapes = []
        for seed in seeds:
            lstm_prediction = predict_bl4(
                data["train_sales"],
                data["val_sales"],
                test_len,
                seed=seed,
            )
            metrics = compute_metrics(truth, lstm_prediction)
            seed_smapes.append(float(metrics["smape"]))
            rows.append(
                _result_row(
                    normalized_id,
                    entity_key,
                    "BL4_LSTM",
                    seed,
                    metrics,
                )
            )

        if len(seed_smapes) > 1:
            smape_array = np.asarray(seed_smapes, dtype=float)
            print(
                f"[BL4 variance] dataset={normalized_id} entity={entity_key} "
                f"seeds={list(seeds)} "
                f"smapes={[f'{value:.6f}' for value in seed_smapes]} "
                f"mean={smape_array.mean():.6f} "
                f"std={smape_array.std(ddof=0):.6f} "
                f"min={smape_array.min():.6f} "
                f"max={smape_array.max():.6f}"
            )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    expected_rows = len(entity_slices) * (3 + len(seeds))
    if len(output) != expected_rows:
        raise AssertionError(
            f"{normalized_id} expected {expected_rows} baseline rows, got {len(output)}"
        )
    numeric_metrics = output[["smape", "rmse", "mae"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_metrics).all() or (numeric_metrics < 0.0).any():
        raise AssertionError(f"{normalized_id} output metrics must be finite and non-negative")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{normalized_id}_baselines_multiseed.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(output)} rows to {output_path}")
    return output_path


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--seeds must be a comma-separated list of integers, got {raw!r}"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run target-only BL1-BL3 baselines once and BL4 across multiple seeds "
            "for D1-D6."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[f"d{number}" for number in range(1, 7)] + ["all"],
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "Comma-separated list of seeds for BL4_LSTM, e.g. '42,43,44,45,46'. "
            f"Default: {','.join(str(s) for s in DEFAULT_SEEDS)}"
        ),
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
        run_dataset(dataset_id, seeds=args.seeds)


if __name__ == "__main__":
    main()
