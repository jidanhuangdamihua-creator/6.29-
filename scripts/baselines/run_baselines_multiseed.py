#!/usr/bin/env python3
"""Run BL1-BL4 on the shared five-horizon rolling-origin sample manifest."""

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
from bl1_historical_mean import predict_bl1
from bl2_moving_average import predict_bl2
from bl3_lightgbm import predict_bl3
from bl4_lstm import predict_bl4
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_SEEDS
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.rolling_origin import validate_feature_availability
from src.utils.result_validation import promote_complete_baseline_groups


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
DEFAULT_SEEDS = FORMAL_SEEDS
METHODS = ("BL1_HistoricalMean", "BL2_MovingAverage", "BL3_LightGBM", "BL4_LSTM")
STRICT_FORECAST_FEATURE_ALLOWLIST = {
    **{f"lag_{lag}": "known_at_origin" for lag in range(1, 8)},
    "day_of_week": "known_in_advance",
    "day_of_month": "known_in_advance",
    "month": "known_in_advance",
    "year": "known_in_advance",
}

def _predict_bl3_rolling(record, seed: int) -> float:
    history = [float(value) for value in record.input_sales]
    history_dates = [pd.Timestamp(value) for value in record.input_dates]
    if len(history) < 8:
        raise ValueError("BL3 requires at least eight legal historical sales values")

    feature_rows = []
    for index in range(7, len(history)):
        current_date = history_dates[index]
        row = {f"lag_{lag}": history[index - lag] for lag in range(1, 8)}
        row.update(
            {
                "day_of_week": current_date.dayofweek,
                "day_of_month": current_date.day,
                "month": current_date.month,
                "year": current_date.year,
                "sales": history[index],
            }
        )
        feature_rows.append(row)
    train_features = pd.DataFrame(feature_rows)
    model_columns = tuple(column for column in train_features.columns if column != "sales")
    validate_feature_availability(
        model_columns,
        allowlist=STRICT_FORECAST_FEATURE_ALLOWLIST,
    )

    for step in range(1, int(record.horizon) + 1):
        forecast_date = pd.Timestamp(record.forecast_origin) + pd.Timedelta(days=step)
        test_row = {f"lag_{lag}": history[-lag] for lag in range(1, 8)}
        test_row.update(
            {
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "month": forecast_date.month,
                "year": forecast_date.year,
            }
        )
        prediction = float(
            predict_bl3(
                train_features,
                pd.DataFrame([test_row]),
                random_state=int(seed),
            )[0]
        )
        history.append(prediction)
    return history[-1]


def _default_protocol_predictor(method: str, record, seed: int) -> float:
    observed = np.asarray(record.input_sales, dtype=float)
    if method == "BL1_HistoricalMean":
        return float(predict_bl1(observed, 1)[0])
    if method == "BL2_MovingAverage":
        return float(predict_bl2(observed, 1)[0])
    if method == "BL3_LightGBM":
        return _predict_bl3_rolling(record, seed)
    if method == "BL4_LSTM":
        if observed.size != 10:
            raise ValueError(f"BL4 requires exactly 10 manifest observations, got {observed.size}")
        prediction = predict_bl4(
            observed[:8],
            observed[8:],
            int(record.horizon),
            seed=int(seed),
        )
        return float(prediction[-1])
    raise ValueError(f"unsupported baseline method: {method!r}")


def evaluate_entity_protocol(data: dict, *, predictor=_default_protocol_predictor) -> pd.DataFrame:
    """Evaluate every method/seed/horizon on one immutable sample manifest."""
    manifest = data["sample_manifest"]
    rows = []
    for method in METHODS:
        for horizon in FORMAL_HORIZONS:
            samples = manifest.for_horizon(horizon)
            if not samples:
                raise AssertionError(f"manifest contains no samples for horizon={horizon}")
            truth = np.asarray([sample.label for sample in samples], dtype=float)
            for seed in FORMAL_SEEDS:
                set_protocol_seed(seed, include_frameworks=False)
                predictions = np.asarray(
                    [predictor(method, sample, seed) for sample in samples],
                    dtype=float,
                )
                metrics = compute_original_scale_metrics(truth, predictions)
                rows.append(
                    {
                        "dataset_id": str(data["dataset_id"]).upper(),
                        "target_entity_key": str(data["entity_key"]),
                        "scenario": "without",
                        "method": method,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        **metrics,
                        "rmse_metric_space": "original_sales_space",
                        "smape_metric_space": "original_sales_space",
                        "sample_count": int(len(samples)),
                        "sample_manifest_digest": manifest.digest,
                        "protocol_track": data["protocol_track"],
                        "protocol_version": data["protocol_version"],
                        "knn_observed_start": data["knn_observed_start"],
                        "knn_observed_end": data["knn_observed_end"],
                        "knn_representation": "not_applicable_target_only",
                        "target_test_excluded": True,
                        "source_future_excluded": True,
                        "candidate_pool_digest": "not_applicable_target_only",
                        "selection_result_digest": "not_applicable_target_only",
                        "result_status": "trial",
                    }
                )
    return pd.DataFrame(rows)


def run_dataset(dataset_id: str, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> Path:
    """Run strict rolling-origin BL1-BL4 and write protocol-auditable rows."""
    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError(f"dataset_id must be d1 through d6, got {dataset_id!r}")

    if tuple(seeds) != FORMAL_SEEDS:
        raise ValueError(f"formal baseline seeds must be exactly {FORMAL_SEEDS}")
    entity_slices = load_baseline_data(normalized_id)
    output = pd.concat(
        [evaluate_entity_protocol(data) for data in entity_slices],
        ignore_index=True,
    )
    output = promote_complete_baseline_groups(output)
    expected_rows = len(entity_slices) * len(METHODS) * len(FORMAL_HORIZONS) * len(FORMAL_SEEDS)
    if len(output) != expected_rows:
        raise AssertionError(
            f"{normalized_id} expected {expected_rows} baseline rows, got {len(output)}"
        )
    numeric_metrics = output[["smape", "rmse", "mae", "accuracy"]].to_numpy(dtype=float)
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
