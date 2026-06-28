"""Audit No-TL target training loss curves without changing main experiments.

The script mirrors the current No-TL data/model/fit path, captures the Keras
History object, and writes audit-only artifacts under outputs/audits/.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from environment import setup_logging, setup_reproducibility
from src.models.cnn_model import resolve_cnn_ablation_training_config
from src.models.no_tl_model import build_no_tl_cnn_model


OUT_DIR = ROOT / "outputs" / "audits"
LOSS_CURVE_CSV = OUT_DIR / "notl_training_loss_curve_audit.csv"
SUMMARY_CSV = OUT_DIR / "notl_training_loss_curve_summary.csv"
REPORT_MD = OUT_DIR / "notl_training_loss_curve_audit.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}

LOSS_CURVE_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "random_seed",
    "target_epochs",
    "early_stopping_enabled",
    "epoch",
    "train_loss",
    "val_loss",
    "train_rmse",
    "val_rmse",
    "train_mae",
    "val_mae",
    "batch_size",
    "effective_batch_size",
    "learning_rate",
    "horizon",
    "window_size",
    "train_windows",
    "val_windows",
    "test_windows",
    "status",
    "error_message",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "random_seed",
    "target_epochs",
    "epochs_observed",
    "early_stopping_enabled",
    "early_stopping_monitor",
    "early_stopping_patience",
    "early_stopping_mode",
    "restore_best_weights",
    "last_10_epoch_count",
    "last_10_first_train_loss",
    "last_10_last_train_loss",
    "last_10_mean_train_loss",
    "last_10_mean_slope",
    "last_10_relative_slope",
    "last_10_std",
    "last_10_std_over_mean",
    "last_3_consecutive_increases",
    "last_10_trend",
    "interpretation",
    "train_windows",
    "val_windows",
    "test_windows",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class SplitBundle:
    dataset: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_scaled: pd.DataFrame
    val_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    feature_columns: List[str]


def _load_config() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
    # Match current main paper-track entrypoints. This only mutates the local
    # in-memory copy used by this audit script.
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    cfg["paper_reproduction"]["strict_paper_split"] = True
    cfg["paper_reproduction"]["paper_strict_split"] = True
    cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    return cfg


def _resolve_experiment_seed(cfg: Dict[str, Any]) -> int:
    for path in (("experiment", "seed"), ("single_experiment", "seed"), ("seed",)):
        current: Any = cfg
        for part in path:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return int(current)

    config_yaml = ROOT / "config.yaml"
    if config_yaml.exists():
        for line in config_yaml.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*seed\s*:\s*([0-9]+)\s*(?:#.*)?$", line)
            if match:
                return int(match.group(1))
    return 42


def _prepare_dataset(dataset: str, cfg: Dict[str, Any]) -> SplitBundle:
    data_path = ROOT / str(cfg["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, _, feature_columns = normalize_features(train_df, val_df, test_df)
    return SplitBundle(
        dataset=dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        feature_columns=feature_columns,
    )


def _build_sequences(bundle: SplitBundle, horizon: int, window_size: int) -> Dict[str, np.ndarray]:
    x_train, y_train = build_tabular_sequence(bundle.train_scaled, horizon=horizon, window_size=window_size)
    x_val, y_val = build_tabular_sequence(bundle.val_scaled, horizon=horizon, window_size=window_size)
    x_test, y_test = build_tabular_sequence(bundle.test_scaled, horizon=horizon, window_size=window_size)
    return {
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
    }


def _history_value(history: Dict[str, Sequence[float]], names: Sequence[str], index: int) -> float:
    for name in names:
        values = history.get(name)
        if values is not None and index < len(values):
            return float(values[index])
    return float("nan")


def _empty_summary_row(
    dataset: str,
    seed: int,
    target_epochs: int,
    train_windows: int,
    val_windows: int,
    test_windows: int,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset_id": DATASET_ID[dataset],
            "dataset": dataset,
            "method": "No-TL",
            "random_seed": int(seed),
            "target_epochs": int(target_epochs),
            "epochs_observed": 0,
            "early_stopping_enabled": False,
            "early_stopping_monitor": "",
            "early_stopping_patience": "",
            "early_stopping_mode": "",
            "restore_best_weights": "",
            "last_10_epoch_count": 0,
            "last_10_trend": "not_available",
            "interpretation": "No train_loss history was available for this dataset.",
            "train_windows": int(train_windows),
            "val_windows": int(val_windows),
            "test_windows": int(test_windows),
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def classify_last_10_train_loss(losses: Sequence[float]) -> Dict[str, Any]:
    clean = np.asarray([float(v) for v in losses if pd.notna(v)], dtype=float)
    if clean.size == 0:
        return {
            "last_10_epoch_count": 0,
            "last_10_trend": "not_available",
            "interpretation": "No train_loss values available.",
        }

    tail = clean[-10:]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) >= 2 else 0.0
    mean = float(np.mean(tail))
    std = float(np.std(tail, ddof=0))
    denom = max(abs(mean), 1e-12)
    relative_slope = slope / denom
    std_over_mean = std / denom
    diffs = np.diff(tail)
    last_3_increases = bool(len(diffs) >= 3 and np.all(diffs[-3:] > 0))
    net_change = float(tail[-1] - tail[0]) if len(tail) >= 2 else 0.0

    if last_3_increases and net_change > 0.005 * denom:
        trend = "possible_divergence"
        interpretation = "Last epochs increase consecutively; this suggests possible divergence or instability."
    elif relative_slope < -0.005:
        trend = "still_decreasing"
        if len(tail) < 10:
            interpretation = (
                "Observed train_loss is still decreasing, but fewer than 10 epochs were run; "
                "current epoch count may be too small and true last-10 behavior cannot be assessed."
            )
        else:
            interpretation = "Last-stage train_loss is still decreasing; current epoch count may be too small."
    elif len(tail) >= 4 and std_over_mean >= 0.10:
        trend = "oscillating"
        interpretation = "Last-stage train_loss variance is high relative to its mean; this suggests oscillation."
    elif abs(relative_slope) <= 0.005:
        trend = "plateau"
        interpretation = "Last-stage train_loss is approximately flat; more epochs may have limited benefit."
    else:
        trend = "increasing"
        interpretation = "Last-stage train_loss is increasing; this suggests possible over-training or instability."

    return {
        "last_10_epoch_count": int(len(tail)),
        "last_10_first_train_loss": float(tail[0]),
        "last_10_last_train_loss": float(tail[-1]),
        "last_10_mean_train_loss": mean,
        "last_10_mean_slope": slope,
        "last_10_relative_slope": relative_slope,
        "last_10_std": std,
        "last_10_std_over_mean": std_over_mean,
        "last_3_consecutive_increases": last_3_increases,
        "last_10_trend": trend,
        "interpretation": interpretation,
    }


def _run_dataset(dataset: str, cfg: Dict[str, Any], seed: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    exp = cfg["single_experiment"]
    horizon = int(exp["horizon"])
    window_size = int(exp["window_size"])
    target_epochs = int(exp["target_epochs"])
    batch_size = int(exp["batch_size"])
    learning_rate = float(exp.get("learning_rate", 1e-4))
    variant = str(exp.get("cnn_ablation_variant", "original"))

    bundle = _prepare_dataset(dataset, cfg)
    sequences = _build_sequences(bundle, horizon=horizon, window_size=window_size)
    train_windows = int(len(sequences["y_train"]))
    val_windows = int(len(sequences["y_val"]))
    test_windows = int(len(sequences["y_test"]))

    if train_windows == 0 or test_windows == 0:
        return [], _empty_summary_row(
            dataset,
            seed,
            target_epochs,
            train_windows,
            val_windows,
            test_windows,
            "SKIPPED",
            "No-TL target train/test windows are empty.",
        )

    import tensorflow as tf

    setup_reproducibility(seed)
    tf.keras.backend.clear_session()
    setup_reproducibility(seed)

    training = resolve_cnn_ablation_training_config(
        cnn_ablation_variant=variant,
        original_batch_size=batch_size,
        original_learning_rate=learning_rate,
    )
    model = build_no_tl_cnn_model(
        input_shape=sequences["x_train"].shape[1:],
        learning_rate=learning_rate,
        cnn_ablation_variant=variant,
    )
    fit_kwargs: Dict[str, Any] = {
        "epochs": target_epochs,
        "batch_size": training.effective_batch_size,
        "verbose": 0,
    }
    if val_windows > 0:
        fit_kwargs["validation_data"] = (sequences["x_val"], sequences["y_val"])

    start = time.perf_counter()
    try:
        history_obj = model.fit(sequences["x_train"], sequences["y_train"], **fit_kwargs)
        history = dict(history_obj.history)
        status = "OK"
        error_message = ""
    except Exception as exc:
        return [], _empty_summary_row(
            dataset,
            seed,
            target_epochs,
            train_windows,
            val_windows,
            test_windows,
            "ERROR",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        run_time_seconds = time.perf_counter() - start

    epochs_observed = len(history.get("loss", []))
    rows: List[Dict[str, Any]] = []
    for idx in range(epochs_observed):
        rows.append(
            {
                "dataset_id": DATASET_ID[dataset],
                "dataset": dataset,
                "method": "No-TL",
                "random_seed": int(seed),
                "target_epochs": int(target_epochs),
                "early_stopping_enabled": False,
                "epoch": idx + 1,
                "train_loss": _history_value(history, ["loss"], idx),
                "val_loss": _history_value(history, ["val_loss"], idx),
                "train_rmse": _history_value(history, ["rmse", "root_mean_squared_error"], idx),
                "val_rmse": _history_value(history, ["val_rmse", "val_root_mean_squared_error"], idx),
                "train_mae": _history_value(history, ["mae", "mean_absolute_error"], idx),
                "val_mae": _history_value(history, ["val_mae", "val_mean_absolute_error"], idx),
                "batch_size": int(batch_size),
                "effective_batch_size": int(training.effective_batch_size),
                "learning_rate": float(training.learning_rate),
                "horizon": int(horizon),
                "window_size": int(window_size),
                "train_windows": train_windows,
                "val_windows": val_windows,
                "test_windows": test_windows,
                "status": status,
                "error_message": error_message,
            }
        )

    trend = classify_last_10_train_loss(history.get("loss", []))
    summary = _empty_summary_row(
        dataset,
        seed,
        target_epochs,
        train_windows,
        val_windows,
        test_windows,
        status,
        error_message,
    )
    summary.update(trend)
    summary.update(
        {
            "epochs_observed": int(epochs_observed),
            "run_time_seconds": float(run_time_seconds),
            "early_stopping_enabled": False,
            "early_stopping_monitor": "",
            "early_stopping_patience": "",
            "early_stopping_mode": "",
            "restore_best_weights": "",
        }
    )
    return rows, summary


def _line_ref(path: Path, pattern: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return f"{path.relative_to(ROOT)}: missing"
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return f"{path.relative_to(ROOT)}:{idx}: `{line.strip()}`"
    return f"{path.relative_to(ROOT)}: pattern not found: `{pattern}`"


def _markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return ""
    rows = df.loc[:, list(columns)].copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(fmt(row[column]) for column in columns) + " |"
        for _, row in rows.iterrows()
    ]
    return "\n".join([header, separator] + body)


def _write_report(loss_df: pd.DataFrame, summary_df: pd.DataFrame, cfg: Dict[str, Any], seed: int) -> None:
    exp = cfg["single_experiment"]
    target_epochs = int(exp["target_epochs"])
    target_epochs_pattern = '"target_epochs": 2'
    main_target_epochs_pattern = 'target_epochs=exp_cfg["target_epochs"]'
    full_target_epochs_pattern = 'target_epochs=int(exp_cfg["target_epochs"]),'
    notl_fit_kwargs_pattern = 'fit_kwargs = {"epochs": target_epochs'
    lines = [
        "# No-TL Training Loss Curve Audit",
        "",
        "Scope: No-TL only; Dataset1, Dataset2, Dataset3; audit-only outputs under `outputs/audits/`.",
        "",
        "## Current No-TL Epoch Setting",
        "",
        f"- Actual main JSON `single_experiment.target_epochs`: `{target_epochs}`.",
        f"- Random seed used by this audit: `{seed}`.",
        "- No-TL final `model.fit(..., epochs=?)` value in current main entrypoints is `target_epochs`, therefore `2` for the checked JSON configuration.",
        "",
        "Evidence:",
        f"- {_line_ref(ROOT / 'configs' / 'default_config.json', target_epochs_pattern)}",
        f"- {_line_ref(ROOT / 'scripts' / 'run_main_experiment.py', main_target_epochs_pattern)}",
        f"- {_line_ref(ROOT / 'scripts' / 'run_full_paper_experiments.py', full_target_epochs_pattern)}",
        f"- {_line_ref(ROOT / 'src' / 'experiment' / 'run_no_tl_experiment.py', notl_fit_kwargs_pattern)}",
        f"- {_line_ref(ROOT / 'src' / 'experiment' / 'run_no_tl_experiment.py', 'model.fit(x_train, y_train, **fit_kwargs)')}",
        "",
        "## Early Stopping",
        "",
        "- No-TL has no early stopping in the current runner. It does not pass `callbacks` to `model.fit`.",
        "- `config.yaml` defines a generic early-stopping block, but the current No-TL runner and JSON experiment entrypoints do not consume it.",
        "",
        "Evidence:",
        f"- {_line_ref(ROOT / 'src' / 'experiment' / 'run_no_tl_experiment.py', notl_fit_kwargs_pattern)}",
        f"- {_line_ref(ROOT / 'src' / 'experiment' / 'run_no_tl_experiment.py', 'model.fit(x_train, y_train, **fit_kwargs)')}",
        f"- {_line_ref(ROOT / 'config.yaml', 'early_stopping:')}",
        f"- {_line_ref(ROOT / 'config.yaml', 'restore_best_weights: true')}",
        "",
        "## Trend Rules",
        "",
        "- `last_10_mean_slope / abs(last_10_mean) < -0.005`: still decreasing.",
        "- absolute relative slope `<= 0.005`: plateau.",
        "- `last_10_std / abs(last_10_mean) >= 0.10`: oscillating.",
        "- last three epoch-to-epoch differences all positive and net increase above 0.5% of mean: possible divergence.",
        "- With fewer than 10 epochs, the same rules are applied to all observed epochs and `last_10_epoch_count` records the actual count.",
        "",
        "## Dataset Summary",
        "",
    ]

    if summary_df.empty:
        lines.append("No summary rows were generated.")
    else:
        display_cols = [
            "dataset",
            "target_epochs",
            "epochs_observed",
            "last_10_epoch_count",
            "last_10_first_train_loss",
            "last_10_last_train_loss",
            "last_10_mean_slope",
            "last_10_std_over_mean",
            "last_10_trend",
            "interpretation",
        ]
        lines.append(_markdown_table(summary_df, display_cols))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Loss curve CSV: `{LOSS_CURVE_CSV.relative_to(ROOT)}`",
            f"- Summary CSV: `{SUMMARY_CSV.relative_to(ROOT)}`",
            "",
            "This audit does not modify training logic, model structure, KNN, RFE, split, RMSE calculation, or main result CSVs.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging(log_level="WARNING", log_file=None)
    cfg = _load_config()
    seed = _resolve_experiment_seed(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        rows, summary = _run_dataset(dataset, cfg, seed)
        all_rows.extend(rows)
        summary_rows.append(summary)

    loss_df = pd.DataFrame(all_rows, columns=LOSS_CURVE_COLUMNS)
    summary_df = pd.DataFrame(summary_rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = np.nan
    summary_df = summary_df[SUMMARY_COLUMNS + [c for c in summary_df.columns if c not in SUMMARY_COLUMNS]]

    loss_df.to_csv(LOSS_CURVE_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    _write_report(loss_df, summary_df, cfg, seed)

    print(f"Wrote {LOSS_CURVE_CSV.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
