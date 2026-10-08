#!/usr/bin/env python3
"""Run BL1-BL5 on the shared five-horizon rolling-origin sample manifest."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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
from bl6_dlinear import fit_bl6, predict_bl6
from bl6_dlinear_revin import fit_bl6_revin, predict_bl6_revin
from bl7_tcn import fit_bl7, predict_bl7
from bl7_tcn_revin import fit_bl7_revin, predict_bl7_revin
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
    "BL6_DLinear",
    "BL7_TCN",
)
METHOD_METADATA = {
    "BL1_HistoricalMean": ("deterministic_baseline", "historical_mean", "deterministic"),
    "BL2_MovingAverage": ("deterministic_baseline", "moving_average", "deterministic"),
    "BL3_LightGBM": ("learning_baseline", "gradient_boosting", "stochastic"),
    "BL4_LSTM": ("learning_baseline", "recurrent", "stochastic"),
    "BL5_SeasonalNaive": ("deterministic_baseline", "seasonal_naive", "deterministic_repeat"),
    "BL6_DLinear": ("learning_baseline", "linear_time_series", "stochastic"),
    "BL7_TCN": ("learning_baseline", "temporal_convolution", "stochastic"),
}
ROBUSTNESS_METHODS = (
    "BL6_DLinear_RevIN",
    "BL7_TCN_RevIN",
)
ROBUSTNESS_METHOD_METADATA = {
    "BL6_DLinear_RevIN": ("robustness_baseline", "linear_time_series_revin", "stochastic"),
    "BL7_TCN_RevIN": ("robustness_baseline", "temporal_convolution_revin", "stochastic"),
}
SUPPORTED_METHODS = METHODS + ROBUSTNESS_METHODS
METHOD_CATALOG = {**METHOD_METADATA, **ROBUSTNESS_METHOD_METADATA}
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


def _predict_learning_direct(
    data: dict,
    samples: tuple,
    *,
    horizon: int,
    seed: int,
    fit_method,
    predict_method,
) -> np.ndarray:
    """Fit one target-entity/direct-H/seed model and predict the shared support."""

    lookback = int(data["lookback"])
    fitted = fit_method(
        data["train_sales"],
        data["val_sales"],
        horizon=int(horizon),
        lookback=lookback,
        seed=int(seed),
    )
    predictions = []
    for record in samples:
        _validate_matured_record(record, lookback=lookback)
        predictions.append(predict_method(fitted, record.input_sales))
    output = np.asarray(predictions, dtype=float).reshape(-1)
    if output.shape != (len(samples),) or not np.isfinite(output).all():
        raise ProtocolViolation(
            f"learning baseline prediction contract failed: expected={len(samples)}"
        )
    return output


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
    if method == "BL6_DLinear":
        raise RuntimeError("BL6 formal prediction requires one initial fit per horizon/seed")
    if method == "BL7_TCN":
        raise RuntimeError("BL7 formal prediction requires one initial fit per horizon/seed")
    if method == "BL6_DLinear_RevIN":
        raise RuntimeError("BL6 RevIN prediction requires one initial fit per horizon/seed")
    if method == "BL7_TCN_RevIN":
        raise RuntimeError("BL7 RevIN prediction requires one initial fit per horizon/seed")
    if method == "BL5_SeasonalNaive":
        return predict_bl5(record)
    raise ValueError(f"unsupported baseline method: {method!r}")


def evaluate_entity_protocol(
    data: dict,
    *,
    predictor=_default_protocol_predictor,
    methods: tuple[str, ...] = METHODS,
    horizons: tuple[int, ...] = FORMAL_HORIZONS,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    return_raw: bool = False,
    run_id: str = "",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every method/seed/horizon on one immutable sample manifest."""
    manifest = data["sample_manifest"]
    rows = []
    raw_rows = []
    unknown_methods = tuple(method for method in methods if method not in METHOD_CATALOG)
    if unknown_methods:
        raise ValueError(f"unsupported baseline methods: {unknown_methods!r}")
    selected_horizons = tuple(int(value) for value in horizons)
    selected_seeds = tuple(int(value) for value in seeds)
    if not selected_horizons or any(value not in FORMAL_HORIZONS for value in selected_horizons):
        raise ProtocolViolation(f"horizons must be a non-empty subset of {FORMAL_HORIZONS}")
    if not selected_seeds or any(value not in FORMAL_SEEDS for value in selected_seeds):
        raise ProtocolViolation(f"seeds must be a non-empty subset of {FORMAL_SEEDS}")
    use_formal_default = predictor is _default_protocol_predictor
    for method in methods:
        method_type, method_family, seed_semantics = METHOD_CATALOG[method]
        for horizon in selected_horizons:
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
            for seed in selected_seeds:
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
                    elif use_formal_default and method == "BL6_DLinear":
                        predictions = _predict_learning_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                            fit_method=fit_bl6,
                            predict_method=predict_bl6,
                        )
                    elif use_formal_default and method == "BL7_TCN":
                        predictions = _predict_learning_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                            fit_method=fit_bl7,
                            predict_method=predict_bl7,
                        )
                    elif use_formal_default and method == "BL6_DLinear_RevIN":
                        predictions = _predict_learning_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                            fit_method=fit_bl6_revin,
                            predict_method=predict_bl6_revin,
                        )
                    elif use_formal_default and method == "BL7_TCN_RevIN":
                        predictions = _predict_learning_direct(
                            data,
                            samples,
                            horizon=horizon,
                            seed=seed,
                            fit_method=fit_bl7_revin,
                            predict_method=predict_bl7_revin,
                        )
                    else:
                        predictions = np.asarray(
                            [predictor(method, sample, seed) for sample in samples],
                            dtype=float,
                        )
                    metrics = compute_original_scale_metrics(truth, predictions)
                else:
                    metrics = deterministic_metrics
                    predictions = deterministic_predictions
                predictions = np.asarray(predictions, dtype=float).reshape(-1)
                if predictions.shape != (len(samples),) or not np.isfinite(predictions).all():
                    raise ProtocolViolation(
                        f"{method} prediction support must contain {len(samples)} finite values"
                    )
                for sample, prediction in zip(samples, predictions):
                    raw_rows.append(
                        {
                            "run_id": str(run_id),
                            "dataset_id": str(data["dataset_id"]).upper(),
                            "target_entity_key": str(data["entity_key"]),
                            "scenario": "without",
                            "method": method,
                            "horizon": int(horizon),
                            "seed": int(seed),
                            "sample_key": sample.sample_key,
                            "forecast_origin": sample.forecast_origin,
                            "input_start": sample.input_start,
                            "input_end": sample.input_end,
                            "target_timestamp": sample.label_date,
                            "y_true": float(sample.label),
                            "y_pred": float(prediction),
                            "metric_space": "original_sales_space",
                        }
                    )
                rows.append(
                    {
                        "run_id": str(run_id),
                        "dataset_id": str(data["dataset_id"]).upper(),
                        "target_entity_key": str(data["entity_key"]),
                        "scenario": "without",
                        "method": method,
                        "method_type": method_type,
                        "method_family": method_family,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "seed_semantics": seed_semantics,
                        **metrics,
                        "metric_space": "original_sales_space",
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
    summary = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    _assert_no_duplicate_baseline_cells(summary)
    _assert_raw_support(summary, raw)
    _assert_manifest_alignment(
        raw,
        manifest=manifest,
        methods=tuple(methods),
        horizons=selected_horizons,
        seeds=selected_seeds,
    )
    return (summary, raw) if return_raw else summary


def _baseline_cell_key(row: pd.Series) -> list[str]:
    return [
        str(row.dataset_id),
        str(row.target_entity_key),
        str(row.scenario),
        str(row.method),
        str(int(row.horizon)),
        str(int(row.seed)),
    ]


def _assert_no_duplicate_baseline_cells(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ProtocolViolation("baseline summary rows may not be empty")
    keys = frame.apply(_baseline_cell_key, axis=1).map(tuple)
    if keys.duplicated().any():
        raise ProtocolViolation("duplicate baseline dataset×target×method×horizon×seed cell")
    metric_values = frame[["rmse", "mae", "smape", "accuracy"]].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all() or (metric_values < 0.0).any():
        raise ProtocolViolation("baseline metrics must be finite and non-negative")


def _assert_raw_support(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    required_raw = {
        "run_id", "dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed",
        "sample_key", "forecast_origin", "input_start", "input_end", "target_timestamp",
        "y_true", "y_pred", "metric_space",
    }
    missing = sorted(required_raw.difference(raw.columns))
    if missing:
        raise ProtocolViolation(f"raw prediction rows missing columns: {missing}")
    if raw.empty or not np.isfinite(raw[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ProtocolViolation("raw prediction rows must be non-empty and finite")
    key_columns = [
        "dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed", "sample_key"
    ]
    if raw.duplicated(key_columns).any():
        raise ProtocolViolation("duplicate raw prediction sample key")
    expected = summary.set_index(
        ["dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed"]
    )
    for identity, group in raw.groupby(
        ["dataset_id", "target_entity_key", "scenario", "method", "horizon", "seed"],
        sort=False,
    ):
        if identity not in expected.index:
            raise ProtocolViolation("raw prediction contains an unknown summary cell")
        if len(group) <= 0:
            raise ProtocolViolation("raw prediction cell is empty")
        expected_count = int(expected.loc[identity, "sample_count"])
        if len(group) != expected_count:
            raise ProtocolViolation(
                f"raw prediction support mismatch for cell={identity}: "
                f"expected={expected_count} actual={len(group)}"
            )
        if group["target_timestamp"].astype(str).duplicated().any():
            raise ProtocolViolation(f"raw prediction cell contains duplicate target timestamps: {identity}")


def _assert_manifest_alignment(
    raw: pd.DataFrame,
    *,
    manifest,
    methods: tuple[str, ...],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    """Require every method/seed to expose the exact ordered manifest support."""

    for method in methods:
        for horizon in horizons:
            expected_keys = {sample.sample_key for sample in manifest.for_horizon(horizon)}
            for seed in seeds:
                group = raw[
                    (raw["method"] == method)
                    & (raw["horizon"].astype(int) == int(horizon))
                    & (raw["seed"].astype(int) == int(seed))
                ]
                actual_keys = set(group["sample_key"].astype(str))
                if actual_keys != expected_keys:
                    raise ProtocolViolation(
                        f"{method} prediction support mismatch for horizon={horizon} seed={seed}"
                    )


def build_run_id(
    scope: str,
    dataset_id: str,
    target_entities: tuple[str, ...],
    methods: tuple[str, ...],
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
) -> str:
    """Create a unique, content-labelled baseline run identity."""

    content = "|".join(
        (
            str(dataset_id),
            ",".join(target_entities),
            ",".join(methods),
            ",".join(str(value) for value in horizons),
            ",".join(str(value) for value in seeds),
        )
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{str(scope).strip().lower()}_{str(dataset_id).strip().lower()}_{timestamp}_{digest}_{uuid4().hex[:6]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_baseline_run(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    output_dir: Path,
    run_id: str,
    smoke: bool = False,
) -> dict[str, str | int]:
    """Publish summary/raw/manifest artifacts, with isolated smoke protection."""

    destination = Path(output_dir).resolve()
    if smoke and destination.exists():
        raise FileExistsError(f"isolated smoke output already exists: {destination}")
    if smoke:
        destination.mkdir(parents=True, exist_ok=False)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty")
    _assert_no_duplicate_baseline_cells(summary)
    _assert_raw_support(summary, raw)

    summary_path = destination / "summary.csv"
    raw_path = destination / "predictions.csv"
    manifest_path = destination / "run_manifest.json"
    if smoke and any(path.exists() for path in (summary_path, raw_path, manifest_path)):
        raise FileExistsError(f"isolated smoke artifacts already exist under {destination}")

    raw_output = raw.copy()
    raw_output["run_id"] = str(run_id)
    raw_output.to_csv(raw_path, index=False, encoding="utf-8")
    raw_sha256 = _sha256_file(raw_path)
    summary_output = summary.copy()
    summary_output["run_id"] = str(run_id)
    summary_output["raw_prediction_file"] = str(raw_path)
    summary_output["source_file"] = str(raw_path)
    summary_output["source_sha256"] = raw_sha256
    summary_output.to_csv(summary_path, index=False, encoding="utf-8")
    summary_sha256 = _sha256_file(summary_path)
    manifest = {
        "run_id": str(run_id),
        "summary_path": str(summary_path),
        "raw_prediction_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "summary_sha256": summary_sha256,
        "raw_prediction_sha256": raw_sha256,
        "summary_rows": int(len(summary_output)),
        "raw_prediction_rows": int(len(raw_output)),
        "methods": sorted(summary_output["method"].astype(str).unique().tolist()),
        "horizons": sorted(summary_output["horizon"].astype(int).unique().tolist()),
        "seeds": sorted(summary_output["seed"].astype(int).unique().tolist()),
        "sample_manifest_digests": sorted(
            summary_output["sample_manifest_digest"].astype(str).unique().tolist()
        ),
        "git_head": _git_head(),
        "git_status_short": _git_status_short(),
        "protocol_authority": "scripts/baselines/run_baselines_multiseed.py",
        "promotion": "none" if smoke else "existing_baseline_promotion_contract",
        "smoke": bool(smoke),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_status_short() -> str:
    try:
        return subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_dataset(
    dataset_id: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    methods: tuple[str, ...] = METHODS,
    horizons: tuple[int, ...] = FORMAL_HORIZONS,
    output_dir: Path | None = None,
    run_id: str | None = None,
    smoke: bool = False,
) -> Path:
    """Run strict rolling-origin BL1-BL7 and write protocol-auditable rows."""
    normalized_id = str(dataset_id).strip().lower()
    if normalized_id not in {f"d{number}" for number in range(1, 7)}:
        raise ValueError(f"dataset_id must be d1 through d6, got {dataset_id!r}")

    selected_methods = tuple(methods)
    selected_horizons = tuple(int(value) for value in horizons)
    selected_seeds = tuple(int(value) for value in seeds)
    if not selected_methods or any(method not in SUPPORTED_METHODS for method in selected_methods):
        raise ValueError(f"methods must be a non-empty subset of {SUPPORTED_METHODS}")
    uses_robustness = any(method in ROBUSTNESS_METHODS for method in selected_methods)
    if uses_robustness:
        if output_dir is None:
            raise ValueError("RevIN robustness methods require an isolated output_dir")
        resolved_output = Path(output_dir).resolve()
        formal_output = OUTPUT_DIR.resolve()
        if resolved_output == formal_output or formal_output in resolved_output.parents:
            raise ValueError(
                "RevIN robustness methods must not write to the formal baseline output directory"
            )
    if not selected_horizons or any(horizon not in FORMAL_HORIZONS for horizon in selected_horizons):
        raise ValueError(f"horizons must be a non-empty subset of {FORMAL_HORIZONS}")
    if not selected_seeds or any(seed not in FORMAL_SEEDS for seed in selected_seeds):
        raise ValueError(f"seeds must be a non-empty subset of {FORMAL_SEEDS}")
    if smoke and output_dir is None:
        raise ValueError("smoke runs require an isolated output_dir")
    entity_slices = load_baseline_data(normalized_id)
    resolved_run_id = run_id or build_run_id(
        "baseline",
        normalized_id,
        tuple(str(data["entity_key"]) for data in entity_slices),
        selected_methods,
        selected_horizons,
        selected_seeds,
    )
    evaluated = [
        evaluate_entity_protocol(
            data,
            methods=selected_methods,
            horizons=selected_horizons,
            seeds=selected_seeds,
            return_raw=True,
            run_id=resolved_run_id,
        )
        for data in entity_slices
    ]
    output = pd.concat(
        [item[0] for item in evaluated],
        ignore_index=True,
    )
    raw = pd.concat([item[1] for item in evaluated], ignore_index=True)
    if not smoke and selected_methods == METHODS and selected_horizons == FORMAL_HORIZONS and selected_seeds == FORMAL_SEEDS:
        output = promote_complete_baseline_groups(output)
    expected_rows = len(entity_slices) * len(selected_methods) * len(selected_horizons) * len(selected_seeds)
    if len(output) != expected_rows:
        raise AssertionError(
            f"{normalized_id} expected {expected_rows} baseline rows, got {len(output)}"
        )
    numeric_metrics = output[["smape", "rmse", "mae", "accuracy"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_metrics).all() or (numeric_metrics < 0.0).any():
        raise AssertionError(f"{normalized_id} output metrics must be finite and non-negative")

    if smoke:
        manifest = publish_baseline_run(
            output,
            raw,
            output_dir=Path(output_dir),
            run_id=resolved_run_id,
            smoke=True,
        )
        output_path = Path(str(manifest["summary_path"]))
    else:
        destination = Path(output_dir) if output_dir is not None else OUTPUT_DIR
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / f"{normalized_id}_baselines_multiseed.csv"
        raw_path = destination / f"{normalized_id}_baseline_predictions.csv"
        output["raw_prediction_file"] = str(raw_path)
        raw["run_id"] = resolved_run_id
        raw.to_csv(raw_path, index=False, encoding="utf-8")
        output["source_file"] = str(raw_path)
        output["source_sha256"] = _sha256_file(raw_path)
        output.to_csv(output_path, index=False, encoding="utf-8")
        manifest_path = destination / f"{normalized_id}_baseline_run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_id": resolved_run_id,
                    "summary_path": str(output_path),
                    "raw_prediction_path": str(raw_path),
                    "summary_sha256": _sha256_file(output_path),
                    "raw_prediction_sha256": _sha256_file(raw_path),
                    "summary_rows": int(len(output)),
                    "raw_prediction_rows": int(len(raw)),
                    "methods": list(selected_methods),
                    "horizons": list(selected_horizons),
                    "seeds": list(selected_seeds),
                    "promotion": "existing_baseline_promotion_contract",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Saved {len(output)} rows to {output_path}")
    return output_path


def parse_selector(raw, *, allowed, name: str) -> tuple:
    """Parse an ordered, duplicate-free comma selector against an authority set."""

    values = raw if not isinstance(raw, str) else raw.split(",")
    parsed = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        candidate = int(text) if all(str(item).lstrip("-").isdigit() for item in allowed) else text
        if candidate not in allowed:
            raise argparse.ArgumentTypeError(
                f"--{name} contains unsupported value {candidate!r}; allowed={tuple(allowed)}"
            )
        if candidate not in parsed:
            parsed.append(candidate)
    if not parsed:
        raise argparse.ArgumentTypeError(f"--{name} must contain at least one value")
    return tuple(parsed)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return parse_selector(raw, allowed=FORMAL_SEEDS, name="seeds")


def _parse_horizons(raw: str) -> tuple[int, ...]:
    return parse_selector(raw, allowed=FORMAL_HORIZONS, name="horizons")


def _parse_methods(raw: str) -> tuple[str, ...]:
    return parse_selector(raw, allowed=SUPPORTED_METHODS, name="methods")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run target-only BL1-BL7 baselines on the shared direct-H rolling-origin "
            "manifest for D1-D6."
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
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=METHODS,
        help=(
            "Comma-separated baseline methods. Default remains the formal BL1-BL7 "
            "set. RevIN robustness methods BL6_DLinear_RevIN,BL7_TCN_RevIN are "
            "opt-in only and require --output-dir outside outputs/baselines."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=FORMAL_HORIZONS,
        help="Comma-separated direct horizons. Default: 1,2,3,4,5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; required for --smoke and isolated smoke artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identity; generated when omitted.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Publish only to a new isolated output directory without promotion.",
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
        run_dataset(
            dataset_id,
            seeds=args.seeds,
            methods=args.methods,
            horizons=args.horizons,
            output_dir=args.output_dir,
            run_id=args.run_id,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
