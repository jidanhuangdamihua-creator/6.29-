#!/usr/bin/env python3
"""Run BL1-BL5 on the shared five-horizon rolling-origin sample manifest."""

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
from bl4_lstm import fit_bl4, predict_bl4
from bl5_seasonal_naive import predict_bl5
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_SEEDS, ProtocolViolation
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.rolling_origin import validate_feature_availability
from src.utils.result_validation import promote_complete_baseline_groups


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
DEFAULT_SEEDS = FORMAL_SEEDS
METHODS = (
    "BL1_HistoricalMean",
    "BL2_MovingAverage",
    "BL3_LightGBM",
    "BL4_LSTM",
    "BL5_SeasonalNaive",
)
STRICT_FORECAST_FEATURE_ALLOWLIST = {
    **{f"lag_{lag}": "known_at_origin" for lag in range(1, 8)},
    "day_of_week": "known_in_advance",
    "day_of_month": "known_in_advance",
    "month": "known_in_advance",
    "year": "known_in_advance",
}

def _bl3_direct_training_frame(data: dict, horizon: int) -> pd.DataFrame:
    """Build one caller-owned initial-train direct-H frame for BL3."""

    history = np.asarray(data["train_sales"], dtype=float)
    history_dates = pd.DatetimeIndex(pd.to_datetime(data["train_dates"], errors="raise"))
    resolved_horizon = int(horizon)
    feature_rows = []
    for origin_index in range(6, len(history) - resolved_horizon):
        label_index = origin_index + resolved_horizon
        label_date = history_dates[label_index]
        row = {
            f"lag_{lag}": float(history[origin_index - lag + 1])
            for lag in range(1, 8)
        }
        row.update(
            {
                "day_of_week": label_date.dayofweek,
                "day_of_month": label_date.day,
                "month": label_date.month,
                "year": label_date.year,
                "sales": float(history[label_index]),
            }
        )
        feature_rows.append(row)
    train_features = pd.DataFrame(feature_rows)
    if train_features.empty:
        raise ValueError(
            f"BL3 initial train cannot form direct-H samples for horizon={resolved_horizon}"
        )
    model_columns = tuple(column for column in train_features.columns if column != "sales")
    validate_feature_availability(
        model_columns,
        allowlist=STRICT_FORECAST_FEATURE_ALLOWLIST,
    )
    return train_features


def _predict_bl3_direct(
    data: dict,
    samples: tuple,
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Fit BL3 once on initial train and predict every matured origin directly."""

    train_features = _bl3_direct_training_frame(data, horizon)
    test_rows = []
    for record in samples:
        _validate_matured_record(record, lookback=int(data["lookback"]))
        history = np.asarray(record.input_sales, dtype=float)
        label_date = pd.Timestamp(record.label_date)
        test_row = {f"lag_{lag}": float(history[-lag]) for lag in range(1, 8)}
        test_row.update(
            {
                "day_of_week": label_date.dayofweek,
                "day_of_month": label_date.day,
                "month": label_date.month,
                "year": label_date.year,
            }
        )
        test_rows.append(test_row)
    return np.asarray(
        predict_bl3(
            train_features,
            pd.DataFrame(test_rows),
            random_state=int(seed),
        ),
        dtype=float,
    )


def _validate_matured_record(record, *, lookback: int) -> None:
    """Prove that one direct-H input contains no truth after its forecast origin."""

    dates = pd.DatetimeIndex(pd.to_datetime(record.input_dates, errors="raise")).normalize()
    origin = pd.Timestamp(record.forecast_origin).normalize()
    label = pd.Timestamp(record.label_date).normalize()
    resolved_horizon = int(record.horizon)
    if len(dates) != int(lookback) or len(record.input_sales) != int(lookback):
        raise ProtocolViolation(
            f"rolling-origin input must contain exactly {int(lookback)} target observations"
        )
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ProtocolViolation("rolling-origin target input dates must be unique and ordered")
    if dates.max() > origin:
        raise ProtocolViolation(
            f"future target truth is visible at origin={origin.date().isoformat()}"
        )
    if dates.max() != origin:
        raise ProtocolViolation("rolling-origin input must end at the forecast origin")
    expected_label = origin + pd.Timedelta(days=resolved_horizon)
    if label != expected_label:
        raise ProtocolViolation(
            f"direct-H label date mismatch: expected={expected_label.date()} actual={label.date()}"
        )


def _predict_bl4_direct(
    data: dict,
    samples: tuple,
    *,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Fit BL4 once on initial 15/15 roles and predict matured origins directly."""

    lookback = int(data["lookback"])
    fitted = fit_bl4(
        data["train_sales"],
        data["val_sales"],
        horizon=int(horizon),
        lookback=lookback,
        seed=int(seed),
    )
    predictions = []
    for record in samples:
        _validate_matured_record(record, lookback=lookback)
        predictions.append(predict_bl4(fitted, record.input_sales))
    return np.asarray(predictions, dtype=float)


def _default_protocol_predictor(method: str, record, seed: int) -> float:
    observed = np.asarray(record.input_sales, dtype=float)
    if method == "BL1_HistoricalMean":
        return float(predict_bl1(observed, 1)[0])
    if method == "BL2_MovingAverage":
        return float(predict_bl2(observed, 1)[0])
    if method == "BL3_LightGBM":
        raise RuntimeError("BL3 formal prediction requires one initial fit per horizon/seed")
    if method == "BL4_LSTM":
        raise RuntimeError("BL4 formal prediction requires one initial fit per horizon/seed")
    if method == "BL5_SeasonalNaive":
        return predict_bl5(record)
    raise ValueError(f"unsupported baseline method: {method!r}")


def evaluate_entity_protocol(
    data: dict,
    *,
    predictor=_default_protocol_predictor,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Evaluate every method/seed/horizon on one immutable sample manifest."""
    manifest = data["sample_manifest"]
    rows = []
    unknown_methods = tuple(method for method in methods if method not in METHODS)
    if unknown_methods:
        raise ValueError(f"unsupported baseline methods: {unknown_methods!r}")
    use_formal_default = predictor is _default_protocol_predictor
    for method in methods:
        for horizon in FORMAL_HORIZONS:
            samples = manifest.for_horizon(horizon)
            if not samples:
                raise AssertionError(f"manifest contains no samples for horizon={horizon}")
            for sample in samples:
                _validate_matured_record(sample, lookback=int(data["lookback"]))
            truth = np.asarray([sample.label for sample in samples], dtype=float)
            deterministic_predictions = None
            deterministic_metrics = None
            if method == "BL5_SeasonalNaive":
                deterministic_predictions = np.asarray(
                    [predictor(method, sample, FORMAL_SEEDS[0]) for sample in samples],
                    dtype=float,
                )
                deterministic_metrics = compute_original_scale_metrics(
                    truth,
                    deterministic_predictions,
                )
            for seed in FORMAL_SEEDS:
                if deterministic_predictions is None:
                    set_protocol_seed(seed, include_frameworks=False)
                    if use_formal_default and method == "BL3_LightGBM":
                        predictions = _predict_bl3_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                        )
                    elif use_formal_default and method == "BL4_LSTM":
                        predictions = _predict_bl4_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                        )
                    else:
                        predictions = np.asarray(
                            [predictor(method, sample, seed) for sample in samples],
                            dtype=float,
                        )
                    metrics = compute_original_scale_metrics(truth, predictions)
                else:
                    metrics = deterministic_metrics
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
    """Run strict rolling-origin BL1-BL5 and write protocol-auditable rows."""
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
            "Run target-only BL1-BL3/BL5 baselines once and BL4 across multiple seeds "
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
            "Comma-separated result seeds, e.g. '42,43,44,45,46'. "
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
