"""No-TL learning_rate x epochs x clipnorm x dropout x seed grid.

This script keeps the existing No-TL data preparation, temporal split, and CNN
backbone unchanged. It varies only Adam learning_rate, fit epochs, optimizer
clipnorm, dropout, and random seed, then writes per-seed detail rows,
seed-aggregate rows, and summary reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-codex"))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from environment import setup_logging, setup_reproducibility
from scripts.audits.cnn_lr_clipnorm_ablation import (
    _compute_metric,
    _load_config,
    _metric_protocol,
    _prepare_sequences,
)
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Model

from src.models.cnn_model import build_base_cnn
from src.utils.experiment_hyperparams import (
    CLIPNORM_LIST,
    DROPOUT_LIST,
    EPOCH_LIST,
    LR_LIST,
    FIXED_CLIPNORM,
    FIXED_DROPOUT,
    FIXED_LEARNING_RATE,
    fixed_hyperparams_slug,
    fixed_hyperparams_summary,
)
from src.utils.runtime_control import set_verbose_mode


OUT_DIR = ROOT / "outputs" / "no_tl_hyperparam_grid"
LOG_DIR = OUT_DIR / "logs"
_HP_SLUG = fixed_hyperparams_slug()
DETAIL_CSV = OUT_DIR / f"no_tl_lr_epoch_clipnorm_dropout_seed_results_{_HP_SLUG}.csv"
AGG_CSV = OUT_DIR / f"no_tl_lr_epoch_clipnorm_dropout_seed_agg_results_{_HP_SLUG}.csv"
BEST_BY_VAL_CSV = OUT_DIR / f"no_tl_best_by_validation_loss_{_HP_SLUG}.csv"
BEST_BY_TEST_RMSE_CSV = OUT_DIR / f"no_tl_best_by_test_rmse_{_HP_SLUG}.csv"
BEST_BY_NORMALIZED_RMSE_CSV = OUT_DIR / f"no_tl_best_by_normalized_rmse_{_HP_SLUG}.csv"
REPORT_MD = OUT_DIR / f"no_tl_hyperparam_summary_{_HP_SLUG}.md"

DATASETS = ["Dataset1", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset3": 3}
LEARNING_RATES = list(LR_LIST)
EPOCHS = list(EPOCH_LIST)
CLIPNORMS = list(CLIPNORM_LIST)
DROPOUTS = list(DROPOUT_LIST)
SEEDS = [42, 43, 44, 45, 46]
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_PATIENCE = 10
RESTORE_BEST_WEIGHTS = True
METHOD = "No-TL"
HORIZON = 1

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "learning_rate",
    "epochs",
    "clipnorm",
    "dropout",
    "seed",
    "optimizer_name",
    "early_stopping_enabled",
    "early_stopping_monitor",
    "early_stopping_patience",
    "restore_best_weights",
    "cnn_structure_changed",
    "batch_size_changed",
    "model_parameter_count",
    "original_model_parameter_count",
    "original_batch_size",
    "effective_batch_size",
    "train_windows",
    "val_windows",
    "test_windows",
    "train_loss",
    "validation_loss",
    "best_validation_loss",
    "actual_epochs_run",
    "early_stopping_triggered",
    "early_stopping_stopped_epoch",
    "best_validation_epoch",
    "test_mae",
    "test_rmse",
    "normalized_rmse",
    "normalized_mae",
    "original_scale_rmse",
    "original_scale_mae",
    "mape_exclude_zero",
    "training_time_seconds",
    "loss_anomaly",
    "gradient_explosion",
    "nan_detected",
    "overfitting_detected",
    "anomaly_notes",
    "metric_space",
    "prediction_shape",
    "status",
    "error_message",
    "log_file",
]

AGG_METRICS = [
    "train_loss",
    "validation_loss",
    "test_mae",
    "test_rmse",
    "normalized_rmse",
    "original_scale_rmse",
    "mape_exclude_zero",
    "early_stopping_stopped_epoch",
    "best_validation_epoch",
    "training_time_seconds",
]

AGG_COLUMNS = [
    "dataset_id",
    "dataset",
    "learning_rate",
    "epochs",
    "clipnorm",
    "dropout",
    "n_seeds",
    "successful_seeds",
    "failed_or_error_runs",
    "seed_list",
    "status",
]
for _metric in AGG_METRICS:
    AGG_COLUMNS.extend([f"{_metric}_mean", f"{_metric}_std"])
AGG_COLUMNS.extend(
    [
        "loss_anomaly_rate",
        "gradient_explosion_rate",
        "nan_rate",
        "overfitting_rate",
        "rank_by_normalized_rmse_mean",
        "rank_by_normalized_rmse_std",
        "stability_note",
    ]
)

BEST_COLUMNS = [
    "rank_metric",
    "rank",
    "dataset_id",
    "dataset",
    "learning_rate",
    "epochs",
    "clipnorm",
    "dropout",
    "seed",
    "train_loss",
    "validation_loss",
    "best_validation_epoch",
    "early_stopping_stopped_epoch",
    "test_rmse",
    "normalized_rmse",
    "original_scale_rmse",
    "test_mae",
    "mape_exclude_zero",
    "training_time_seconds",
    "loss_anomaly",
    "gradient_explosion",
    "nan_detected",
    "overfitting_detected",
    "status",
]


@dataclass(frozen=True)
class GridVariant:
    optimizer_name: str
    learning_rate: float
    epochs: int
    clipnorm: float | None
    dropout: float
    early_stopping_enabled: bool
    early_stopping_monitor: str
    early_stopping_patience: int
    restore_best_weights: bool
    cnn_structure_changed: bool
    batch_size_changed: bool
    original_batch_size: int
    effective_batch_size: int


def iter_grid_combinations(
    datasets: Sequence[str] | None = None,
    learning_rates: Sequence[float] | None = None,
    epochs: Sequence[int] | None = None,
    clipnorms: Sequence[float | None] | None = None,
    dropouts: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[Tuple[str, float, int, float | None, float, int]]:
    datasets = list(datasets or DATASETS)
    learning_rates = list(learning_rates or LEARNING_RATES)
    epochs = list(epochs or EPOCHS)
    clipnorms = list(clipnorms or CLIPNORMS)
    dropouts = list(dropouts or DROPOUTS)
    seeds = list(seeds or SEEDS)
    return [
        (dataset, float(learning_rate), int(epoch), clipnorm, float(dropout), int(seed))
        for dataset in datasets
        for learning_rate in learning_rates
        for epoch in epochs
        for clipnorm in clipnorms
        for dropout in dropouts
        for seed in seeds
    ]


def expected_detail_row_count(
    datasets: Sequence[str] | None = None,
    learning_rates: Sequence[float] | None = None,
    epochs: Sequence[int] | None = None,
    clipnorms: Sequence[float | None] | None = None,
    dropouts: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
) -> int:
    return len(iter_grid_combinations(datasets, learning_rates, epochs, clipnorms, dropouts, seeds))


def resolve_grid_variant(
    learning_rate: float,
    epochs: int,
    clipnorm: float | None,
    dropout: float,
    original_batch_size: int = 16,
) -> GridVariant:
    return GridVariant(
        optimizer_name="Adam",
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        clipnorm=None if clipnorm is None else float(clipnorm),
        dropout=float(dropout),
        early_stopping_enabled=True,
        early_stopping_monitor=EARLY_STOPPING_MONITOR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=RESTORE_BEST_WEIGHTS,
        cnn_structure_changed=False,
        batch_size_changed=False,
        original_batch_size=int(original_batch_size),
        effective_batch_size=int(original_batch_size),
    )


def build_grid_model(input_shape: tuple[int, int], learning_rate: float, clipnorm: float | None, dropout: float = FIXED_DROPOUT):
    """Build the fixed-dropout base CNN and keep clipnorm disabled when None."""
    import tensorflow as tf

    dropout_rate = float(dropout)
    if math.isclose(dropout_rate, 0.0, abs_tol=1e-12):
        model = build_base_cnn(input_shape=input_shape, learning_rate=float(learning_rate), dropout=dropout_rate, clipnorm=clipnorm)
    else:
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
        x = MaxPooling1D(pool_size=2, name="pool1")(x)
        x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
        x = MaxPooling1D(pool_size=2, name="pool2")(x)
        x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv3")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(rate=dropout_rate, name="dropout")(x)
        outputs = Dense(1, name="dense_out")(x)
        model = Model(inputs=inputs, outputs=outputs)
    optimizer_kwargs: Dict[str, Any] = {"learning_rate": float(learning_rate)}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = float(clipnorm)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _safe_last(values: Sequence[Any]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.nan
    return float(arr[-1])


def _safe_min(values: Sequence[Any]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.min(finite))


def analyze_training_history(history: Dict[str, Sequence[Any]], train_loss: float, validation_loss: float) -> Dict[str, Any]:
    losses = np.asarray(list(history.get("loss", [])) + list(history.get("val_loss", [])), dtype=np.float64)
    finite_losses = losses[np.isfinite(losses)]
    nan_detected = bool(losses.size > 0 and not np.all(np.isfinite(losses)))
    huge_loss = bool(finite_losses.size > 0 and np.max(np.abs(finite_losses)) > 1e6)
    growth = False
    train_losses = np.asarray(history.get("loss", []), dtype=np.float64)
    finite_train = train_losses[np.isfinite(train_losses)]
    if finite_train.size >= 2 and abs(float(finite_train[0])) > 1e-12:
        growth = bool(float(finite_train[-1]) / max(abs(float(finite_train[0])), 1e-12) > 100.0)
    gradient_explosion = bool(nan_detected or huge_loss or growth)
    loss_anomaly = bool(nan_detected or huge_loss or growth or (not np.isfinite(train_loss)) or (not np.isfinite(validation_loss)))

    best_val = _safe_min(history.get("val_loss", []))
    overfitting_detected = False
    if np.isfinite(train_loss) and np.isfinite(validation_loss):
        gap_rule = validation_loss > max(train_loss * 2.0, train_loss + 0.05)
        trend_rule = np.isfinite(best_val) and validation_loss > best_val * 1.2
        overfitting_detected = bool(gap_rule and (trend_rule or len(history.get("val_loss", [])) <= 2))

    notes: List[str] = []
    if nan_detected:
        notes.append("NaN or Inf detected in loss history.")
    if huge_loss:
        notes.append("Loss exceeded 1e6.")
    if growth:
        notes.append("Training loss grew more than 100x from first to last epoch.")
    if overfitting_detected:
        notes.append("Validation loss is substantially above train loss.")
    if not notes:
        notes.append("No loss instability flags triggered.")

    return {
        "loss_anomaly": loss_anomaly,
        "gradient_explosion": gradient_explosion,
        "nan_detected": nan_detected,
        "overfitting_detected": overfitting_detected,
        "anomaly_notes": " ".join(notes),
    }


def _inverse_sales(values: np.ndarray, prepared: Dict[str, Any]) -> np.ndarray | None:
    scaler = prepared.get("scaler")
    feature_columns = prepared.get("feature_columns")
    if scaler is None or feature_columns is None:
        return None
    columns = list(feature_columns)
    if "sales" not in columns:
        return None
    feature_index = columns.index("sales")
    if not hasattr(scaler, "scale_") or not hasattr(scaler, "min_"):
        return None
    scale = np.asarray(getattr(scaler, "scale_"), dtype=np.float64).reshape(-1)
    offset = np.asarray(getattr(scaler, "min_"), dtype=np.float64).reshape(-1)
    if feature_index >= len(scale) or feature_index >= len(offset) or np.isclose(scale[feature_index], 0.0):
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return (arr - float(offset[feature_index])) / float(scale[feature_index])


def _mape_exclude_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = ~np.isclose(y_true_arr, 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100.0)


def _clip_label(clipnorm: float | None) -> str:
    if isinstance(clipnorm, str):
        return clipnorm
    if clipnorm is None or pd.isna(clipnorm):
        return "None"
    return f"{float(clipnorm):g}"


def _dropout_label(dropout: float) -> str:
    return f"{float(dropout):g}"


def _lr_label(learning_rate: float) -> str:
    return f"{float(learning_rate):.0e}".replace("+0", "").replace("-0", "-")


def _log_path(
    output_dir: Path,
    dataset: str,
    learning_rate: float,
    epochs: int,
    clipnorm: float | None,
    dropout: float,
    seed: int,
) -> Path:
    return (
        output_dir
        / "logs"
        / (
            f"{dataset}_lr-{_lr_label(learning_rate)}_epochs-{int(epochs)}_clipnorm-{_clip_label(clipnorm)}"
            f"_dropout-{_dropout_label(dropout)}_seed-{int(seed)}.log"
        )
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _base_row(
    dataset: str,
    seed: int,
    meta: GridVariant,
    prepared: Dict[str, Any],
    original_parameter_count: int,
    log_file: Path,
) -> Dict[str, Any]:
    row = {
        "dataset_id": DATASET_ID.get(dataset, np.nan),
        "dataset": dataset,
        "method": METHOD,
        **asdict(meta),
        "seed": int(seed),
        "model_parameter_count": np.nan,
        "original_model_parameter_count": int(original_parameter_count),
        "train_windows": int(len(prepared.get("y_train", []))),
        "val_windows": int(len(prepared.get("y_val", []))),
        "test_windows": int(len(prepared.get("y_test", []))),
        "train_loss": np.nan,
        "validation_loss": np.nan,
        "best_validation_loss": np.nan,
        "actual_epochs_run": 0,
        "early_stopping_triggered": False,
        "early_stopping_stopped_epoch": np.nan,
        "best_validation_epoch": np.nan,
        "test_mae": np.nan,
        "test_rmse": np.nan,
        "normalized_rmse": np.nan,
        "normalized_mae": np.nan,
        "original_scale_rmse": np.nan,
        "original_scale_mae": np.nan,
        "mape_exclude_zero": np.nan,
        "training_time_seconds": 0.0,
        "loss_anomaly": False,
        "gradient_explosion": False,
        "nan_detected": False,
        "overfitting_detected": False,
        "anomaly_notes": "",
        "metric_space": "",
        "prediction_shape": "",
        "status": "OK",
        "error_message": "",
        "log_file": _display_path(log_file),
    }
    return {column: row.get(column, np.nan) for column in DETAIL_COLUMNS}


def _hard_condition_errors(meta: GridVariant, model_parameter_count: int, original_parameter_count: int) -> List[str]:
    errors: List[str] = []
    if meta.optimizer_name != "Adam":
        errors.append("optimizer_name must be Adam")
    if meta.cnn_structure_changed is not False:
        errors.append("cnn_structure_changed must be False")
    if meta.batch_size_changed is not False:
        errors.append("batch_size_changed must be False")
    if int(model_parameter_count) != int(original_parameter_count):
        errors.append("model_parameter_count must equal original_model_parameter_count")
    if int(meta.effective_batch_size) != int(meta.original_batch_size):
        errors.append("effective_batch_size must equal original_batch_size")
    return errors


def _write_run_log(log_file: Path, payload: Dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _run_one(
    prepared: Dict[str, Any],
    metric_protocol: Dict[str, Any],
    dataset: str,
    seed: int,
    learning_rate: float,
    epochs: int,
    clipnorm: float | None,
    dropout: float,
    original_batch_size: int,
    original_parameter_count: int,
    output_dir: Path,
) -> Dict[str, Any]:
    import tensorflow as tf

    meta = resolve_grid_variant(learning_rate, epochs, clipnorm, dropout, original_batch_size)
    log_file = _log_path(output_dir, dataset, learning_rate, epochs, clipnorm, dropout, seed)
    row = _base_row(dataset, seed, meta, prepared, original_parameter_count, log_file)
    start = time.perf_counter()
    payload: Dict[str, Any] = {
        "config": asdict(meta),
        "hyperparams_summary": fixed_hyperparams_summary(),
        "clipnorm_note": "clipnorm=None disables Adam gradient clipping.",
        "dataset": dataset,
        "seed": int(seed),
        "status": "OK",
    }

    if len(prepared["y_train"]) == 0 or len(prepared["y_test"]) == 0:
        row["status"] = "ERROR"
        row["error_message"] = "empty train/test windows"
        payload.update({"status": "ERROR", "error_message": row["error_message"]})
        _write_run_log(log_file, payload)
        return row

    setup_reproducibility(seed)
    tf.keras.backend.clear_session()
    setup_reproducibility(seed)

    try:
        model = build_grid_model(prepared["x_train"].shape[1:], learning_rate=learning_rate, clipnorm=clipnorm, dropout=dropout)
        model_parameter_count = int(model.count_params())
        row["model_parameter_count"] = model_parameter_count
        hard_errors = _hard_condition_errors(meta, model_parameter_count, original_parameter_count)
        if hard_errors:
            row["status"] = "FAIL"
            row["error_message"] = "; ".join(hard_errors)
            payload.update({"status": "FAIL", "error_message": row["error_message"]})
            _write_run_log(log_file, payload)
            return row

        callbacks = []
        early_stopping = None
        if len(prepared["y_val"]) > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=EARLY_STOPPING_MONITOR,
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=RESTORE_BEST_WEIGHTS,
            )
            callbacks.append(early_stopping)

        history_obj = model.fit(
            prepared["x_train"],
            prepared["y_train"],
            epochs=int(epochs),
            batch_size=int(meta.effective_batch_size),
            validation_data=(prepared["x_val"], prepared["y_val"]) if len(prepared["y_val"]) > 0 else None,
            callbacks=callbacks,
            verbose=0,
        )
        history = {key: [float(value) for value in values] for key, values in history_obj.history.items()}
        actual_epochs_run = int(len(history.get("loss", [])))
        row["train_loss"] = _safe_last(history.get("loss", []))
        row["validation_loss"] = _safe_last(history.get("val_loss", []))
        row["best_validation_loss"] = _safe_min(history.get("val_loss", []))
        row["actual_epochs_run"] = actual_epochs_run
        row["early_stopping_triggered"] = bool(early_stopping is not None and int(early_stopping.stopped_epoch) > 0)
        row["early_stopping_stopped_epoch"] = (
            int(early_stopping.stopped_epoch) + 1
            if early_stopping is not None and int(early_stopping.stopped_epoch) > 0
            else actual_epochs_run
        )
        val_losses = np.asarray(history.get("val_loss", []), dtype=np.float64)
        finite_val_mask = np.isfinite(val_losses)
        if val_losses.size > 0 and np.any(finite_val_mask):
            finite_indexes = np.where(finite_val_mask)[0]
            best_index = finite_indexes[int(np.argmin(val_losses[finite_val_mask]))]
            row["best_validation_epoch"] = int(best_index + 1)

        y_test_pred = model.predict(prepared["x_test"], verbose=0)
        test_metric = _compute_metric(prepared["y_test"], y_test_pred, metric_protocol, prepared)
        row["test_rmse"] = float(test_metric["rmse"])
        row["test_mae"] = float(test_metric["mae"])
        row["normalized_rmse"] = float(test_metric.get("normalized_rmse", test_metric["rmse"]))
        row["normalized_mae"] = float(test_metric.get("normalized_mae", test_metric["mae"]))
        row["original_scale_rmse"] = test_metric.get("original_scale_rmse")
        row["original_scale_mae"] = test_metric.get("original_scale_mae")
        row["metric_space"] = str(test_metric.get("metric_space", ""))
        row["prediction_shape"] = str(tuple(y_test_pred.shape))

        original_y_true = _inverse_sales(prepared["y_test"], prepared)
        original_y_pred = _inverse_sales(y_test_pred, prepared)
        if original_y_true is not None and original_y_pred is not None:
            row["mape_exclude_zero"] = _mape_exclude_zero(original_y_true, original_y_pred)
        else:
            row["mape_exclude_zero"] = _mape_exclude_zero(prepared["y_test"], y_test_pred)

        flags = analyze_training_history(history, train_loss=float(row["train_loss"]), validation_loss=float(row["validation_loss"]))
        for key, value in flags.items():
            row[key] = value

        row["training_time_seconds"] = float(time.perf_counter() - start)
        payload.update(
            {
                "status": row["status"],
                "history": history,
                "metrics": {
                    "train_loss": row["train_loss"],
                    "validation_loss": row["validation_loss"],
                    "best_validation_loss": row["best_validation_loss"],
                    "test_rmse": row["test_rmse"],
                    "test_mae": row["test_mae"],
                    "normalized_rmse": row["normalized_rmse"],
                    "original_scale_rmse": row["original_scale_rmse"],
                    "mape_exclude_zero": row["mape_exclude_zero"],
                },
                "early_stopping": {
                    "enabled": bool(len(prepared["y_val"]) > 0),
                    "monitor": EARLY_STOPPING_MONITOR,
                    "patience": EARLY_STOPPING_PATIENCE,
                    "restore_best_weights": RESTORE_BEST_WEIGHTS,
                    "triggered": row["early_stopping_triggered"],
                    "stopped_epoch": row["early_stopping_stopped_epoch"],
                    "best_validation_epoch": row["best_validation_epoch"],
                    "actual_epochs_run": row["actual_epochs_run"],
                },
                "flags": {key: row[key] for key in ("loss_anomaly", "gradient_explosion", "nan_detected", "overfitting_detected")},
                "training_time_seconds": row["training_time_seconds"],
            }
        )
    except Exception as exc:
        row["status"] = "ERROR"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["training_time_seconds"] = float(time.perf_counter() - start)
        payload.update({"status": "ERROR", "error_message": row["error_message"], "training_time_seconds": row["training_time_seconds"]})

    _write_run_log(log_file, payload)
    return row


def _ok_rows(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    for column in ("validation_loss", "test_rmse", "normalized_rmse"):
        ok = ok[pd.to_numeric(ok[column], errors="coerce").notna()]
    return ok


def build_seed_aggregate(details: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_id", "dataset", "learning_rate", "epochs", "clipnorm", "dropout"]
    if details.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)

    rows: List[Dict[str, Any]] = []
    grouped = details.groupby(group_cols, dropna=False, sort=True)
    for keys, group in grouped:
        key_dict = dict(zip(group_cols, keys))
        ok = group[group["status"].eq("OK")].copy()
        n_seeds = int(group["seed"].nunique()) if "seed" in group else 0
        successful_seeds = int(ok["seed"].nunique()) if "seed" in ok else 0
        failed_or_error_runs = int(group["status"].ne("OK").sum()) if "status" in group else 0
        row: Dict[str, Any] = {
            **key_dict,
            "n_seeds": n_seeds,
            "successful_seeds": successful_seeds,
            "failed_or_error_runs": failed_or_error_runs,
            "seed_list": ",".join(str(int(seed)) for seed in sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().unique())),
            "status": "OK" if failed_or_error_runs == 0 and successful_seeds == n_seeds else "PARTIAL",
            "loss_anomaly_rate": float(ok["loss_anomaly"].astype(bool).mean()) if not ok.empty and "loss_anomaly" in ok else np.nan,
            "gradient_explosion_rate": float(ok["gradient_explosion"].astype(bool).mean()) if not ok.empty and "gradient_explosion" in ok else np.nan,
            "nan_rate": float(ok["nan_detected"].astype(bool).mean()) if not ok.empty and "nan_detected" in ok else np.nan,
            "overfitting_rate": float(ok["overfitting_detected"].astype(bool).mean()) if not ok.empty and "overfitting_detected" in ok else np.nan,
            "stability_note": "",
        }
        if ok.empty:
            row["status"] = "ERROR"
            row["stability_note"] = "No successful seed rows."
        for metric in AGG_METRICS:
            values = pd.to_numeric(ok.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std()) if len(values) > 1 else np.nan
        rows.append(row)

    agg = pd.DataFrame(rows)
    for dataset, idx in agg.groupby("dataset").groups.items():
        dataset_rows = agg.loc[idx]
        agg.loc[idx, "rank_by_normalized_rmse_mean"] = dataset_rows["normalized_rmse_mean"].rank(method="min", ascending=True)
        agg.loc[idx, "rank_by_normalized_rmse_std"] = dataset_rows["normalized_rmse_std"].rank(method="min", ascending=True, na_option="bottom")
        std_values = pd.to_numeric(dataset_rows["normalized_rmse_std"], errors="coerce").dropna()
        if not std_values.empty:
            high_std_threshold = float(std_values.quantile(0.75))
            high_std_mask = (
                agg.index.isin(idx)
                & pd.to_numeric(agg["normalized_rmse_std"], errors="coerce").ge(high_std_threshold)
                & pd.to_numeric(agg["normalized_rmse_std"], errors="coerce").gt(0)
            )
            agg.loc[high_std_mask, "stability_note"] = "High normalized_rmse std within dataset grid."
    return agg.reindex(columns=AGG_COLUMNS)


def _rank_by(details: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = _ok_rows(details)
    rows: List[pd.DataFrame] = []
    for dataset, group in ok.groupby("dataset"):
        ranked = group.sort_values([metric, "validation_loss", "test_rmse", "learning_rate", "epochs", "dropout"], ascending=True).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked.insert(0, "rank_metric", metric)
        rows.append(ranked.reindex(columns=BEST_COLUMNS))
    if not rows:
        return pd.DataFrame(columns=BEST_COLUMNS)
    return pd.concat(rows, ignore_index=True).reindex(columns=BEST_COLUMNS)


def _write_dataset_best_files(details: pd.DataFrame, output_dir: Path) -> None:
    for dataset in DATASETS:
        dataset_rows: List[pd.DataFrame] = []
        subset = details[details["dataset"].eq(dataset)]
        for metric in ("validation_loss", "test_rmse", "normalized_rmse"):
            ranked = _rank_by(subset, metric)
            dataset_rows.append(ranked.head(10))
        out = pd.concat(dataset_rows, ignore_index=True) if dataset_rows else pd.DataFrame(columns=BEST_COLUMNS)
        out.to_csv(output_dir / f"no_tl_{dataset}_best_summary.csv", index=False)


def _learning_rate_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby("learning_rate", as_index=False)
        .agg(
            runs=("status", "size"),
            mean_validation_loss=("validation_loss", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            mean_original_scale_rmse=("original_scale_rmse", "mean"),
            loss_anomaly_rate=("loss_anomaly", "mean"),
            overfitting_rate=("overfitting_detected", "mean"),
        )
        .sort_values("mean_test_rmse")
    )


def _epoch_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby("epochs", as_index=False)
        .agg(
            runs=("status", "size"),
            mean_validation_loss=("validation_loss", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
        )
        .sort_values("epochs")
    )


def _clipnorm_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["clipnorm_label"] = ok["clipnorm"].apply(_clip_label)
    return (
        ok.groupby("clipnorm_label", as_index=False)
        .agg(
            runs=("status", "size"),
            mean_validation_loss=("validation_loss", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            loss_anomaly_rate=("loss_anomaly", "mean"),
            gradient_explosion_rate=("gradient_explosion", "mean"),
            nan_rate=("nan_detected", "mean"),
            overfitting_rate=("overfitting_detected", "mean"),
        )
        .sort_values("mean_test_rmse")
    )


def _top_n_per_dataset(ranked: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    return ranked.groupby("dataset", group_keys=False).head(n).reset_index(drop=True)


def _overall_setting_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["learning_rate", "epochs", "clipnorm", "dropout"], dropna=False, as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            runs=("status", "size"),
            mean_validation_loss=("validation_loss", "mean"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            mean_original_scale_rmse=("original_scale_rmse", "mean"),
            loss_anomaly_rate=("loss_anomaly", "mean"),
            overfitting_rate=("overfitting_detected", "mean"),
        )
        .sort_values(["mean_test_rmse", "mean_validation_loss", "learning_rate", "epochs"])
    )


def _format_value(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return " ".join(str(value).replace("|", "/").replace("\n", " ").split())


def _format_mean_std(mean_value: Any, std_value: Any, digits: int = 6) -> str:
    mean_text = _format_value(mean_value, digits=digits)
    std_text = _format_value(std_value, digits=digits)
    if not mean_text:
        return ""
    if not std_text:
        std_text = "NA"
    return f"{mean_text} ± {std_text}"


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 30) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _load_current_no_tl_baseline() -> Dict[str, float]:
    baseline_path = ROOT / "outputs" / "experiment_results" / "full_paper_results.csv"
    if not baseline_path.exists():
        return {}
    try:
        df = pd.read_csv(baseline_path)
    except Exception:
        return {}
    rows = df[df["method"].eq(METHOD)]
    baselines: Dict[str, float] = {}
    for dataset in DATASETS:
        dataset_rows = rows[rows["dataset"].eq(dataset)]
        if not dataset_rows.empty and "rmse" in dataset_rows:
            baselines[dataset] = float(pd.to_numeric(dataset_rows["rmse"], errors="coerce").dropna().iloc[0])
    return baselines


def _write_report(details: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path) -> None:
    best_by_val = _rank_by(details, "validation_loss")
    best_by_test = _rank_by(details, "test_rmse")
    best_by_norm = _rank_by(details, "normalized_rmse")
    lr_summary = _learning_rate_summary(details)
    epoch_summary = _epoch_summary(details)
    clip_summary = _clipnorm_summary(details)
    overall_settings = _overall_setting_summary(details)
    baselines = _load_current_no_tl_baseline()

    best_lines: List[str] = []
    best_mean_lines: List[str] = []
    stability_lines: List[str] = []
    gap_lines: List[str] = []
    next_step_lines: List[str] = []
    for dataset in DATASETS:
        best = best_by_test[best_by_test["dataset"].eq(dataset)].head(1)
        if best.empty:
            best_lines.append(f"- {dataset}: no successful run.")
            next_step_lines.append(f"- {dataset}: rerun failed combinations before changing model structure.")
            continue
        row = best.iloc[0]
        best_lines.append(
            f"- {dataset}: lr={float(row['learning_rate']):g}, epochs={int(row['epochs'])}, "
            f"clipnorm={_clip_label(row['clipnorm'])}, test RMSE={float(row['test_rmse']):.6f}, "
            f"validation loss={float(row['validation_loss']):.6f}."
        )
        baseline = baselines.get(dataset)
        if baseline is None:
            gap_lines.append(f"- {dataset}: no current paper-track No-TL baseline found for gap calculation.")
        else:
            gap = float(row["test_rmse"]) - baseline
            pct = gap / baseline * 100.0 if baseline else np.nan
            direction = "better" if gap < 0 else "worse"
            gap_lines.append(
                f"- {dataset}: best normalized RMSE {float(row['test_rmse']):.6f} vs current paper-track "
                f"No-TL baseline {baseline:.6f}; gap={gap:.6f} ({pct:.2f}%, {direction})."
            )
        if bool(row["loss_anomaly"]) or bool(row["nan_detected"]):
            next_step_lines.append(f"- {dataset}: stabilize training before model-structure changes.")
        else:
            next_step_lines.append(f"- {dataset}: prioritize TL/RFE reproduction next; do not change CNN structure based only on this grid.")

        dataset_agg = aggregate[aggregate["dataset"].eq(dataset) & aggregate["status"].isin(["OK", "PARTIAL"])].copy()
        dataset_agg = dataset_agg[pd.to_numeric(dataset_agg["normalized_rmse_mean"], errors="coerce").notna()]
        if dataset_agg.empty:
            best_mean_lines.append(f"- {dataset}: no aggregate rows with successful seed metrics.")
            stability_lines.append(f"- {dataset}: stability could not be assessed because aggregate metrics are missing.")
            continue
        best_mean = dataset_agg.sort_values(
            ["normalized_rmse_mean", "normalized_rmse_std", "validation_loss_mean", "learning_rate", "epochs", "dropout"],
            na_position="last",
        ).iloc[0]
        best_mean_lines.append(
            f"- {dataset}: lr={float(best_mean['learning_rate']):g}, epochs={int(best_mean['epochs'])}, "
            f"clipnorm={_clip_label(best_mean['clipnorm'])}, dropout={float(best_mean['dropout']):g}, "
            f"normalized_rmse = {_format_mean_std(best_mean['normalized_rmse_mean'], best_mean['normalized_rmse_std'])}, "
            f"test_rmse = {_format_mean_std(best_mean['test_rmse_mean'], best_mean['test_rmse_std'])}, "
            f"successful seeds={int(best_mean['successful_seeds'])}/{int(best_mean['n_seeds'])}."
        )
        best_std = best_mean.get("normalized_rmse_std", np.nan)
        median_std = pd.to_numeric(dataset_agg["normalized_rmse_std"], errors="coerce").median()
        if pd.isna(best_std):
            stability_lines.append(f"- {dataset}: best-mean setting needs at least 2 successful seeds to assess std.")
        else:
            stable_text = (
                "stable relative to this grid"
                if pd.isna(median_std) or float(best_std) <= float(median_std)
                else "less stable than the median std in this grid"
            )
            stability_lines.append(
                f"- {dataset}: best-mean setting is {stable_text}; normalized_rmse std={_format_value(best_std)}."
            )

    total_runs = len(details)
    ok_runs = int(details["status"].eq("OK").sum()) if "status" in details else 0
    expected_runs = expected_detail_row_count()
    status = "PASS" if total_runs == expected_runs and ok_runs == expected_runs else "PARTIAL"
    lr_best_text = ""
    if not lr_summary.empty:
        top_lr = lr_summary.iloc[0]
        lr_best_text = (
            f"Overall, lr={float(top_lr['learning_rate']):g} has the lowest mean test RMSE "
            f"({float(top_lr['mean_test_rmse']):.6f}) across successful runs."
        )
    epoch_best_text = ""
    if not epoch_summary.empty:
        top_epoch = epoch_summary.sort_values("mean_test_rmse").iloc[0]
        first_epoch = epoch_summary.sort_values("epochs").iloc[0]
        epoch_best_text = (
            f"Best mean test RMSE occurs at epochs={int(top_epoch['epochs'])}. "
            f"Compared with epochs={int(first_epoch['epochs'])}, longer training "
            f"{'improved' if float(top_epoch['mean_test_rmse']) < float(first_epoch['mean_test_rmse']) else 'did not clearly improve'} the aggregate score."
        )
    clip_text = ""
    if not clip_summary.empty:
        top_clip = clip_summary.iloc[0]
        clip_text = (
            f"Best mean test RMSE occurs at clipnorm={_clip_label(top_clip['clipnorm_label'])}; "
            f"NaN rate={float(top_clip['nan_rate']):.3f}, gradient-explosion flag rate={float(top_clip['gradient_explosion_rate']):.3f}."
        )
    overall_lines: List[str] = []
    if not overall_settings.empty:
        by_val = overall_settings.sort_values(["mean_validation_loss", "mean_test_rmse"]).iloc[0]
        by_test = overall_settings.sort_values(["mean_test_rmse", "mean_validation_loss"]).iloc[0]
        by_norm = overall_settings.sort_values(["mean_normalized_rmse", "mean_validation_loss"]).iloc[0]
        for label, row in (
            ("validation loss", by_val),
            ("test RMSE", by_test),
            ("normalized RMSE", by_norm),
        ):
            overall_lines.append(
                f"- By {label}: lr={float(row['learning_rate']):g}, epochs={int(row['epochs'])}, "
                f"clipnorm={_clip_label(row['clipnorm'])}, dropout={float(row['dropout']):g}, "
                f"mean validation loss={float(row['mean_validation_loss']):.6f}, "
                f"mean test RMSE={float(row['mean_test_rmse']):.6f}."
            )

    high_std_rows = pd.DataFrame()
    if not aggregate.empty and "normalized_rmse_std" in aggregate:
        high_std_rows = aggregate[pd.to_numeric(aggregate["normalized_rmse_std"], errors="coerce").notna()].copy()
        if not high_std_rows.empty:
            high_std_rows = high_std_rows.sort_values(["dataset", "normalized_rmse_std"], ascending=[True, False]).groupby("dataset", group_keys=False).head(5)

    recommended_rows = pd.DataFrame()
    if not aggregate.empty:
        candidate = aggregate[pd.to_numeric(aggregate["normalized_rmse_mean"], errors="coerce").notna()].copy()
        if not candidate.empty:
            recommended_rows = (
                candidate.sort_values(
                    ["dataset", "normalized_rmse_mean", "normalized_rmse_std", "test_rmse_mean", "learning_rate", "epochs", "dropout"],
                    na_position="last",
                )
                .groupby("dataset", group_keys=False)
                .head(1)
            )

    lines = [
        "# No-TL Hyperparameter Grid Summary",
        "",
        "## Run Count",
        "",
        f"- Expected grid size: {expected_runs} runs.",
        f"- Observed detail rows: {total_runs}.",
        f"- Successful rows: {ok_runs}.",
        f"- Status: {status}.",
        "",
        "## Best Setting By Dataset",
        "",
        "\n".join(best_lines),
        "",
        "## Best Average Result Across 5 Seeds",
        "",
        "\n".join(best_mean_lines),
        "",
        "## Seed Stability",
        "",
        "\n".join(stability_lines),
        "",
        "The final recommendation should prioritize mean performance first, then prefer smaller std when mean values are close. Report result values as mean ± std, for example `normalized_rmse = 0.400000 ± 0.020000`.",
        "",
        "## High-Std Combinations",
        "",
        _markdown_table(
            high_std_rows,
            [
                "dataset",
                "learning_rate",
                "epochs",
                "clipnorm",
                "dropout",
                "normalized_rmse_mean",
                "normalized_rmse_std",
                "test_rmse_mean",
                "test_rmse_std",
                "successful_seeds",
                "stability_note",
            ],
            max_rows=20,
        )
        if not high_std_rows.empty
        else "(empty)",
        "",
        "## Recommended Hyperparameters",
        "",
        _markdown_table(
            recommended_rows,
            [
                "dataset",
                "learning_rate",
                "epochs",
                "clipnorm",
                "dropout",
                "normalized_rmse_mean",
                "normalized_rmse_std",
                "test_rmse_mean",
                "test_rmse_std",
                "best_validation_epoch_mean",
                "training_time_seconds_mean",
            ],
            max_rows=10,
        )
        if not recommended_rows.empty
        else "(empty)",
        "",
        "## Overall Best Settings",
        "",
        "\n".join(overall_lines) if overall_lines else "(empty)",
        "",
        "## Best By Validation Loss",
        "",
        _markdown_table(_top_n_per_dataset(best_by_val), ["dataset", "rank", "learning_rate", "epochs", "clipnorm", "dropout", "validation_loss", "test_rmse"], max_rows=20),
        "",
        "## Best By Test RMSE",
        "",
        _markdown_table(_top_n_per_dataset(best_by_test), ["dataset", "rank", "learning_rate", "epochs", "clipnorm", "dropout", "test_rmse", "validation_loss"], max_rows=20),
        "",
        "## Best By Normalized RMSE",
        "",
        _markdown_table(_top_n_per_dataset(best_by_norm), ["dataset", "rank", "learning_rate", "epochs", "clipnorm", "dropout", "normalized_rmse", "validation_loss"], max_rows=20),
        "",
        "## Learning Rate Effect",
        "",
        lr_best_text,
        "",
        _markdown_table(lr_summary, lr_summary.columns, max_rows=20) if not lr_summary.empty else "(empty)",
        "",
        "## Epoch Effect",
        "",
        epoch_best_text,
        "",
        _markdown_table(epoch_summary, epoch_summary.columns, max_rows=20) if not epoch_summary.empty else "(empty)",
        "",
        "## Clipnorm Stability",
        "",
        clip_text,
        "",
        _markdown_table(clip_summary, clip_summary.columns, max_rows=20) if not clip_summary.empty else "(empty)",
        "",
        "## Gap To Current Paper-Track Baseline",
        "",
        "\n".join(gap_lines),
        "",
        "The repository documentation still treats normalized RMSE as the Table 7/8 candidate comparison space; original-scale RMSE is diagnostic.",
        "",
        "## Next Step",
        "",
        "\n".join(next_step_lines),
        "",
        "## Output Files",
        "",
        f"- `{DETAIL_CSV.relative_to(ROOT)}`",
        f"- `{AGG_CSV.relative_to(ROOT)}`",
        f"- `{BEST_BY_VAL_CSV.relative_to(ROOT)}`",
        f"- `{BEST_BY_TEST_RMSE_CSV.relative_to(ROOT)}`",
        f"- `{BEST_BY_NORMALIZED_RMSE_CSV.relative_to(ROOT)}`",
        f"- `outputs/no_tl_hyperparam_grid/no_tl_Dataset1_best_summary.csv`",
        f"- `outputs/no_tl_hyperparam_grid/no_tl_Dataset3_best_summary.csv`",
        f"- `outputs/no_tl_hyperparam_grid/logs/`",
    ]
    (output_dir / REPORT_MD.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(details: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = details.reindex(columns=DETAIL_COLUMNS)
    details.to_csv(output_dir / DETAIL_CSV.name, index=False)
    aggregate = build_seed_aggregate(details)
    aggregate.to_csv(output_dir / AGG_CSV.name, index=False)

    best_by_val = _rank_by(details, "validation_loss")
    best_by_test = _rank_by(details, "test_rmse")
    best_by_norm = _rank_by(details, "normalized_rmse")
    best_by_val.to_csv(output_dir / BEST_BY_VAL_CSV.name, index=False)
    best_by_test.to_csv(output_dir / BEST_BY_TEST_RMSE_CSV.name, index=False)
    best_by_norm.to_csv(output_dir / BEST_BY_NORMALIZED_RMSE_CSV.name, index=False)
    _write_dataset_best_files(details, output_dir)
    _write_report(details, aggregate, output_dir)


def _parse_float_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_clipnorm_list(raw: str) -> List[float | None]:
    values: List[float | None] = []
    for part in str(raw).split(","):
        normalized = part.strip()
        if not normalized:
            continue
        if normalized.lower() in {"none", "null", "nan"}:
            values.append(None)
        else:
            values.append(float(normalized))
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--learning-rates", default=",".join(f"{value:g}" for value in LEARNING_RATES))
    parser.add_argument("--epochs", default=",".join(str(value) for value in EPOCHS))
    parser.add_argument("--clipnorms", default=",".join("None" if value is None else f"{value:g}" for value in CLIPNORMS))
    parser.add_argument("--dropouts", default=",".join(f"{value:g}" for value in DROPOUTS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of grid rows for smoke tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(log_level="WARNING", log_file=None)
    set_verbose_mode("summary")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    datasets = _parse_str_list(args.datasets)
    learning_rates = _parse_float_list(args.learning_rates)
    epochs = _parse_int_list(args.epochs)
    clipnorms = _parse_clipnorm_list(args.clipnorms)
    dropouts = _parse_float_list(args.dropouts)
    seeds = _parse_int_list(args.seeds)
    combinations = iter_grid_combinations(datasets, learning_rates, epochs, clipnorms, dropouts, seeds)
    if args.limit and args.limit > 0:
        combinations = combinations[: int(args.limit)]

    config = _load_config()
    exp = config.get("single_experiment", {})
    original_batch_size = int(exp.get("batch_size", config.get("batch_size", 16)))
    metric_protocol = _metric_protocol(config)

    prepared_by_dataset: Dict[str, Dict[str, Any]] = {}
    original_params_by_dataset: Dict[str, int] = {}
    for dataset in datasets:
        prepared = _prepare_sequences(dataset, config)
        prepared_by_dataset[dataset] = prepared
        original_params_by_dataset[dataset] = int(
            build_grid_model(
                prepared["x_train"].shape[1:],
                learning_rate=FIXED_LEARNING_RATE,
                clipnorm=FIXED_CLIPNORM,
                dropout=FIXED_DROPOUT,
            ).count_params()
        )

    rows: List[Dict[str, Any]] = []
    for index, (dataset, learning_rate, epoch, clipnorm, dropout, seed) in enumerate(combinations, start=1):
        print(
            f"[{index}/{len(combinations)}] dataset={dataset} lr={learning_rate:g} "
            f"epochs={epoch} clipnorm={_clip_label(clipnorm)} dropout={dropout:g} seed={seed}",
            flush=True,
        )
        print(f"[hyperparams] {fixed_hyperparams_summary()}; clipnorm=None disables gradient clipping.", flush=True)
        rows.append(
            _run_one(
                prepared=prepared_by_dataset[dataset],
                metric_protocol=metric_protocol,
                dataset=dataset,
                seed=seed,
                learning_rate=learning_rate,
                epochs=epoch,
                clipnorm=clipnorm,
                dropout=dropout,
                original_batch_size=original_batch_size,
                original_parameter_count=original_params_by_dataset[dataset],
                output_dir=output_dir,
            )
        )

    details = pd.DataFrame(rows).reindex(columns=DETAIL_COLUMNS)
    write_outputs(details, output_dir)
    print(f"Wrote {output_dir / DETAIL_CSV.name}")
    print(f"Wrote {output_dir / REPORT_MD.name}")


if __name__ == "__main__":
    main()
