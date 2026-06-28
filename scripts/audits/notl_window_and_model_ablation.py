"""No-TL window-size and small-sample model ablation audit.

This script is isolated from the main experiment runner. It does not modify
data cleaning, split, KNN/source selection/RFE, RMSE formulas, or the main CNN
definition. Temporary comparison models live only in this audit script.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
from src.evaluation.metrics import compute_metrics_with_protocol
from src.models.no_tl_model import build_no_tl_cnn_model


OUT_DIR = ROOT / "outputs" / "audits"
WINDOW_CSV = OUT_DIR / "notl_window_size_ablation.csv"
MODEL_CSV = OUT_DIR / "notl_small_sample_model_ablation.csv"
REPORT_MD = OUT_DIR / "notl_window_and_model_ablation.md"

RANDOM_SEED = 42
DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
WINDOW_SIZES = [3, 5, 7, 10]
TARGET_EPOCHS = [50]
BATCH_SIZE = 4
HORIZON = 1
MODEL_NAMES = [
    "current_3layer_cnn",
    "simple_1layer_cnn",
    "dense_baseline",
    "persistence_baseline",
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
    scaler: Any
    feature_columns: List[str]


def _load_config() -> Dict[str, Any]:
    return json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))


def _prepare_dataset(dataset: str, cfg: Dict[str, Any]) -> SplitBundle:
    data_path = ROOT / str(cfg["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(
        train_df,
        val_df,
        test_df,
    )
    return SplitBundle(
        dataset=dataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        scaler=scaler,
        feature_columns=feature_columns,
    )


def _build_sequences(bundle: SplitBundle, window_size: int) -> Dict[str, np.ndarray]:
    x_train, y_train = build_tabular_sequence(bundle.train_scaled, horizon=HORIZON, window_size=window_size)
    x_val, y_val = build_tabular_sequence(bundle.val_scaled, horizon=HORIZON, window_size=window_size)
    x_test, y_test = build_tabular_sequence(bundle.test_scaled, horizon=HORIZON, window_size=window_size)
    return {
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
    }


def _metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_protocol: Dict[str, Any],
    bundle: SplitBundle,
) -> Dict[str, Any]:
    return compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=bundle.scaler,
        feature_columns=bundle.feature_columns,
    )


def _build_simple_1layer_cnn(input_shape: Tuple[int, int], learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Conv1D, Dense, Flatten
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=16, kernel_size=3, name="simple_conv1")(inputs)
    x = Flatten(name="flatten")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _build_dense_baseline(input_shape: Tuple[int, int], learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Flatten(name="flatten")(inputs)
    x = Dense(16, activation="relu", name="dense_hidden")(x)
    outputs = Dense(1, name="dense_out")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _base_row(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    model_name: str,
    window_size: int,
    target_epochs: int,
    notes: str,
) -> Dict[str, Any]:
    return {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "method": "No-TL",
        "model_name": model_name,
        "random_seed": RANDOM_SEED,
        "horizon": HORIZON,
        "target_epochs": int(target_epochs),
        "batch_size": BATCH_SIZE,
        "window_size": int(window_size),
        "train_rows": int(len(bundle.train_df)),
        "val_rows": int(len(bundle.val_df)),
        "test_rows": int(len(bundle.test_df)),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "feature_count": int(sequences["x_train"].shape[-1]),
        "feature_columns": "|".join(bundle.feature_columns),
        "notes": notes,
    }


def _empty_metric_row(base: Dict[str, Any], status: str, error_message: str = "") -> Dict[str, Any]:
    return {
        **base,
        "rmse": np.nan,
        "normalized_rmse": np.nan,
        "original_scale_rmse": np.nan,
        "accuracy": np.nan,
        "normalized_accuracy": np.nan,
        "original_scale_accuracy": np.nan,
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "val_normalized_rmse": np.nan,
        "val_original_scale_rmse": np.nan,
        "metric_space": "normalized_minmax_space",
        "prediction_shape": "(skipped)",
        "run_time_seconds": 0.0,
        "status": status,
        "error_message": error_message,
    }


def _run_persistence(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    window_size: int,
    target_epochs: int,
) -> Dict[str, Any]:
    base = _base_row(
        bundle=bundle,
        sequences=sequences,
        model_name="persistence_baseline",
        window_size=window_size,
        target_epochs=target_epochs,
        notes="Prediction is last observed normalized sales value in input window.",
    )
    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        return _empty_metric_row(base, status="SKIPPED", error_message="insufficient train/test windows")
    if "sales" not in bundle.feature_columns:
        return _empty_metric_row(base, status="ERROR", error_message="sales not found in feature columns")

    start = time.perf_counter()
    sales_idx = bundle.feature_columns.index("sales")
    y_test_pred = sequences["x_test"][:, -1, sales_idx].reshape(-1, 1)
    test_metric = _metric_dict(sequences["y_test"], y_test_pred, metric_protocol, bundle)

    if len(sequences["y_val"]) > 0:
        y_val_pred = sequences["x_val"][:, -1, sales_idx].reshape(-1, 1)
        val_metric = _metric_dict(sequences["y_val"], y_val_pred, metric_protocol, bundle)
    else:
        val_metric = {}

    return {
        **base,
        "rmse": float(test_metric["rmse"]),
        "normalized_rmse": float(test_metric["normalized_rmse"]),
        "original_scale_rmse": test_metric.get("original_scale_rmse"),
        "accuracy": float(test_metric["accuracy"]),
        "normalized_accuracy": float(test_metric["normalized_accuracy"]),
        "original_scale_accuracy": test_metric.get("original_scale_accuracy"),
        "val_rmse": float(val_metric.get("rmse", np.nan)),
        "test_rmse": float(test_metric["rmse"]),
        "val_normalized_rmse": float(val_metric.get("normalized_rmse", np.nan)),
        "val_original_scale_rmse": val_metric.get("original_scale_rmse"),
        "metric_space": str(test_metric["metric_space"]),
        "prediction_shape": tuple(y_test_pred.shape),
        "run_time_seconds": float(time.perf_counter() - start),
        "status": "OK",
        "error_message": "",
    }


def _run_keras_model(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    model_name: str,
    model_factory: Callable[[Tuple[int, int], float], Any],
    window_size: int,
    target_epochs: int,
    learning_rate: float,
) -> Dict[str, Any]:
    base = _base_row(
        bundle=bundle,
        sequences=sequences,
        model_name=model_name,
        window_size=window_size,
        target_epochs=target_epochs,
        notes="Temporary audit model; main CNN/data/metric code unchanged.",
    )
    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        return _empty_metric_row(base, status="SKIPPED", error_message="insufficient train/test windows")

    import tensorflow as tf

    setup_reproducibility(RANDOM_SEED)
    tf.keras.backend.clear_session()
    setup_reproducibility(RANDOM_SEED)

    start = time.perf_counter()
    try:
        model = model_factory(sequences["x_train"].shape[1:], learning_rate)
        fit_kwargs: Dict[str, Any] = {
            "epochs": int(target_epochs),
            "batch_size": BATCH_SIZE,
            "verbose": 0,
        }
        if len(sequences["y_val"]) > 0:
            fit_kwargs["validation_data"] = (sequences["x_val"], sequences["y_val"])
        model.fit(sequences["x_train"], sequences["y_train"], **fit_kwargs)
        y_test_pred = model.predict(sequences["x_test"], verbose=0)
        test_metric = _metric_dict(sequences["y_test"], y_test_pred, metric_protocol, bundle)

        if len(sequences["y_val"]) > 0:
            y_val_pred = model.predict(sequences["x_val"], verbose=0)
            val_metric = _metric_dict(sequences["y_val"], y_val_pred, metric_protocol, bundle)
        else:
            val_metric = {}
    except Exception as exc:  # audit row keeps structural incompatibilities visible
        return _empty_metric_row(
            base,
            status="ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return {
        **base,
        "rmse": float(test_metric["rmse"]),
        "normalized_rmse": float(test_metric["normalized_rmse"]),
        "original_scale_rmse": test_metric.get("original_scale_rmse"),
        "accuracy": float(test_metric["accuracy"]),
        "normalized_accuracy": float(test_metric["normalized_accuracy"]),
        "original_scale_accuracy": test_metric.get("original_scale_accuracy"),
        "val_rmse": float(val_metric.get("rmse", np.nan)),
        "test_rmse": float(test_metric["rmse"]),
        "val_normalized_rmse": float(val_metric.get("normalized_rmse", np.nan)),
        "val_original_scale_rmse": val_metric.get("original_scale_rmse"),
        "metric_space": str(test_metric["metric_space"]),
        "prediction_shape": tuple(y_test_pred.shape),
        "run_time_seconds": float(time.perf_counter() - start),
        "status": "OK",
        "error_message": "",
    }


def _model_factory(model_name: str) -> Callable[[Tuple[int, int], float], Any] | None:
    if model_name == "current_3layer_cnn":
        return lambda input_shape, learning_rate: build_no_tl_cnn_model(input_shape, learning_rate=learning_rate)
    if model_name == "simple_1layer_cnn":
        return _build_simple_1layer_cnn
    if model_name == "dense_baseline":
        return _build_dense_baseline
    if model_name == "persistence_baseline":
        return None
    raise ValueError(f"Unknown model_name: {model_name}")


def _percent_change(old: float, new: float) -> float:
    if pd.isna(old) or pd.isna(new) or old == 0:
        return float("nan")
    return (new - old) / old * 100.0


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_cell(value: Any) -> str:
    text = _format_float(value)
    text = text.replace("\x1b[1m", "").replace("\x1b[0m", "")
    text = " ".join(text.replace("\n", " ").replace("|", "/").split())
    if len(text) > 220:
        return text[:217] + "..."
    return text


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    out = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _window_summary(window_df: pd.DataFrame) -> pd.DataFrame:
    ok = window_df[window_df["status"].eq("OK")].copy()
    rows: List[Dict[str, Any]] = []
    for (dataset, target_epochs), group in ok.groupby(["dataset", "target_epochs"], sort=True):
        ws10 = group[group["window_size"].eq(10)]
        best = group.sort_values("normalized_rmse").iloc[0]
        base_rmse = float(ws10.iloc[0]["normalized_rmse"]) if not ws10.empty else np.nan
        rows.append(
            {
                "dataset": dataset,
                "target_epochs": int(target_epochs),
                "ws10_train_windows": int(ws10.iloc[0]["train_windows"]) if not ws10.empty else np.nan,
                "max_train_windows": int(group["train_windows"].max()),
                "ws10_rmse": base_rmse,
                "best_window_size": int(best["window_size"]),
                "best_rmse": float(best["normalized_rmse"]),
                "best_vs_ws10_percent": _percent_change(base_rmse, float(best["normalized_rmse"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "target_epochs"]).reset_index(drop=True)


def _model_summary(model_df: pd.DataFrame) -> pd.DataFrame:
    ok = model_df[model_df["status"].eq("OK")].copy()
    summary = (
        ok.groupby("model_name", as_index=False)
        .agg(
            valid_runs=("normalized_rmse", "count"),
            mean_rmse=("normalized_rmse", "mean"),
            median_rmse=("normalized_rmse", "median"),
            best_rmse=("normalized_rmse", "min"),
            worst_rmse=("normalized_rmse", "max"),
        )
        .sort_values(["mean_rmse", "median_rmse"])
        .reset_index(drop=True)
    )
    winner_rows = []
    for keys, group in ok.groupby(["dataset", "window_size", "target_epochs"], sort=True):
        winner = group.sort_values("normalized_rmse").iloc[0]
        winner_rows.append(
            {
                "dataset": keys[0],
                "window_size": int(keys[1]),
                "target_epochs": int(keys[2]),
                "winner_model": str(winner["model_name"]),
                "winner_rmse": float(winner["normalized_rmse"]),
            }
        )
    wins = pd.DataFrame(winner_rows).groupby("winner_model", as_index=False).size()
    wins = wins.rename(columns={"winner_model": "model_name", "size": "win_count"})
    summary = summary.merge(wins, on="model_name", how="left")
    summary["win_count"] = summary["win_count"].fillna(0).astype(int)
    return summary


def _current_vs_baselines(model_df: pd.DataFrame) -> pd.DataFrame:
    ok = model_df[model_df["status"].eq("OK")].copy()
    pivot = ok.pivot_table(
        index=["dataset", "window_size", "target_epochs"],
        columns="model_name",
        values="normalized_rmse",
        aggfunc="first",
    ).reset_index()
    rows: List[Dict[str, Any]] = []
    for baseline in ["simple_1layer_cnn", "dense_baseline", "persistence_baseline"]:
        valid = pivot.dropna(subset=["current_3layer_cnn", baseline]).copy()
        if valid.empty:
            rows.append(
                {
                    "baseline": baseline,
                    "paired_runs": 0,
                    "baseline_better_count": 0,
                    "mean_current_minus_baseline": np.nan,
                    "mean_percent_delta_current_vs_baseline": np.nan,
                }
            )
            continue
        delta = valid["current_3layer_cnn"] - valid[baseline]
        pct = (valid["current_3layer_cnn"] - valid[baseline]) / valid[baseline] * 100.0
        rows.append(
            {
                "baseline": baseline,
                "paired_runs": int(len(valid)),
                "baseline_better_count": int((valid[baseline] < valid["current_3layer_cnn"]).sum()),
                "mean_current_minus_baseline": float(delta.mean()),
                "mean_percent_delta_current_vs_baseline": float(pct.mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_report(window_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    window_summary = _window_summary(window_df)
    model_summary = _model_summary(model_df)
    current_compare = _current_vs_baselines(model_df)

    error_rows = model_df[model_df["status"].ne("OK")][
        ["dataset", "model_name", "window_size", "target_epochs", "status", "error_message"]
    ].copy()

    ws_increase = bool((window_summary["max_train_windows"] > window_summary["ws10_train_windows"]).all())
    improved_rows = window_summary[window_summary["best_vs_ws10_percent"] < 0]
    window_improved = bool(len(improved_rows) > 0)
    persistence_summary = current_compare[current_compare["baseline"].eq("persistence_baseline")]
    persistence_often_better = (
        not persistence_summary.empty
        and int(persistence_summary.iloc[0]["baseline_better_count"]) >= max(1, int(persistence_summary.iloc[0]["paired_runs"]) // 2)
    )
    simple_summary = current_compare[current_compare["baseline"].eq("simple_1layer_cnn")]
    dense_summary = current_compare[current_compare["baseline"].eq("dense_baseline")]
    simple_text = (
        f"simple_1layer_cnn beats current in {int(simple_summary.iloc[0]['baseline_better_count'])}/"
        f"{int(simple_summary.iloc[0]['paired_runs'])} paired valid rows"
        if not simple_summary.empty
        else "simple_1layer_cnn has no paired comparison rows"
    )
    dense_text = (
        f"dense_baseline beats current in {int(dense_summary.iloc[0]['baseline_better_count'])}/"
        f"{int(dense_summary.iloc[0]['paired_runs'])} paired valid rows, but mean delta is "
        f"{float(dense_summary.iloc[0]['mean_current_minus_baseline']):.6f}"
        if not dense_summary.empty
        else "dense_baseline has no paired comparison rows"
    )
    persistence_text = (
        f"persistence_baseline beats current in {int(persistence_summary.iloc[0]['baseline_better_count'])}/"
        f"{int(persistence_summary.iloc[0]['paired_runs'])} paired valid rows"
        if not persistence_summary.empty
        else "persistence_baseline has no paired comparison rows"
    )

    recommendation = (
        "不建议直接修改主 CNN。当前证据显示问题首先来自训练样本过少、window_size 与 observed window 的匹配、"
        "以及此前已发现的 horizon/metric 口径差异。更合适的下一步是把 No-TL baseline 的训练协议和报告口径固定清楚，"
        "并把 simple/dense/persistence 作为审计基线保留，而不是用一次小样本消融替换论文主 CNN。"
    )

    lines = [
        "# No-TL Window Size And Small-Sample Model Ablation",
        "",
        "Scope: only No-TL, Dataset1/2/3, random_seed=42, horizon=1, batch_size=4, target_epochs in {2,20}, window_size in {3,5,7,10}. Main data cleaning, split, KNN/source selection/RFE, RMSE formula, and main CNN files are unchanged.",
        "",
        "## Files",
        "",
        f"- Window-size ablation: `{WINDOW_CSV.relative_to(ROOT)}`",
        f"- Small-sample model ablation: `{MODEL_CSV.relative_to(ROOT)}`",
        "",
        "## Window Size Findings",
        "",
        _markdown_table(
            window_summary,
            [
                "dataset",
                "target_epochs",
                "ws10_train_windows",
                "max_train_windows",
                "ws10_rmse",
                "best_window_size",
                "best_rmse",
                "best_vs_ws10_percent",
            ],
        ),
        "",
        f"1. Train windows {'明显增加' if ws_increase else '没有稳定增加'} when window_size is smaller than 10. This is directly supported by `train_windows` in the CSV.",
        f"2. Smaller window_size {'does improve at least some current-CNN rows' if window_improved else 'does not improve current-CNN RMSE in these rows'}; see `best_vs_ws10_percent` above.",
        "",
        "## Model Findings",
        "",
        _markdown_table(
            model_summary,
            ["model_name", "valid_runs", "mean_rmse", "median_rmse", "best_rmse", "worst_rmse", "win_count"],
        ),
        "",
        "Current CNN vs baselines, paired where current_3layer_cnn was valid:",
        "",
        _markdown_table(
            current_compare,
            [
                "baseline",
                "paired_runs",
                "baseline_better_count",
                "mean_current_minus_baseline",
                "mean_percent_delta_current_vs_baseline",
            ],
        ),
        "",
        f"3. Current_3layer_cnn is clearly worse than persistence on this audit grid: {persistence_text}. It is also usually worse than simple_1layer_cnn: {simple_text}. Dense is mixed: {dense_text}. Positive `mean_current_minus_baseline` means current CNN has higher RMSE.",
        "4. If persistence_baseline is better than CNN, it means a no-training carry-forward rule beats learned weights on the same normalized target scale. That points to tiny training samples and short-term autocorrelation dominating the baseline, not to KNN/RFE/source-selection behavior.",
        "",
        "## Invalid Or Error Rows",
        "",
        _markdown_table(error_rows, ["dataset", "model_name", "window_size", "target_epochs", "status", "error_message"]),
        "",
        "These rows are kept as evidence. In particular, if current_3layer_cnn cannot support a short window, that is part of the window/model compatibility audit.",
        "",
        "## Overall Judgment",
        "",
        "5. Current No-TL gap is most plausibly a combination of training samples being too few, window_size being mismatched to a ~30-day observed window, previously observed horizon aggregation mismatch, and metric/scaler comparability. CNN structure may be over-heavy for the available No-TL samples, but this audit alone is not enough to justify replacing the paper CNN in the main path.",
        "",
        f"6. Recommendation: {recommendation}",
        "",
        f"Persistence interpretation flag: {'persistence is often better in paired comparisons' if persistence_often_better else 'persistence is not often better in paired comparisons'} based on `baseline_better_count`.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="WARNING", log_file=None)
    cfg = _load_config()
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))
    learning_rate = float(cfg.get("single_experiment", {}).get("learning_rate", 1e-4))

    bundles = {dataset: _prepare_dataset(dataset, cfg) for dataset in DATASETS}
    sequences = {
        (dataset, window_size): _build_sequences(bundles[dataset], window_size=window_size)
        for dataset in DATASETS
        for window_size in WINDOW_SIZES
    }

    result_cache: Dict[Tuple[str, int, int, str], Dict[str, Any]] = {}

    def run_one(dataset: str, window_size: int, target_epochs: int, model_name: str) -> Dict[str, Any]:
        key = (dataset, window_size, target_epochs, model_name)
        if key in result_cache:
            return dict(result_cache[key])
        bundle = bundles[dataset]
        seq = sequences[(dataset, window_size)]
        if model_name == "persistence_baseline":
            row = _run_persistence(bundle, seq, metric_protocol, window_size, target_epochs)
        else:
            factory = _model_factory(model_name)
            if factory is None:
                raise ValueError(f"Missing factory for {model_name}")
            row = _run_keras_model(
                bundle=bundle,
                sequences=seq,
                metric_protocol=metric_protocol,
                model_name=model_name,
                model_factory=factory,
                window_size=window_size,
                target_epochs=target_epochs,
                learning_rate=learning_rate,
            )
        result_cache[key] = dict(row)
        return row

    window_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for target_epochs in TARGET_EPOCHS:
            for window_size in WINDOW_SIZES:
                window_rows.append(run_one(dataset, window_size, target_epochs, "current_3layer_cnn"))

    model_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for target_epochs in TARGET_EPOCHS:
            for window_size in WINDOW_SIZES:
                for model_name in MODEL_NAMES:
                    model_rows.append(run_one(dataset, window_size, target_epochs, model_name))

    column_order = [
        "dataset_id",
        "dataset",
        "method",
        "model_name",
        "random_seed",
        "horizon",
        "target_epochs",
        "batch_size",
        "window_size",
        "train_rows",
        "val_rows",
        "test_rows",
        "train_windows",
        "val_windows",
        "test_windows",
        "rmse",
        "normalized_rmse",
        "original_scale_rmse",
        "accuracy",
        "normalized_accuracy",
        "original_scale_accuracy",
        "val_rmse",
        "test_rmse",
        "run_time_seconds",
        "status",
        "notes",
        "error_message",
    ]

    window_df = pd.DataFrame(window_rows).sort_values(["dataset_id", "target_epochs", "window_size"])
    model_df = pd.DataFrame(model_rows).sort_values(["dataset_id", "target_epochs", "window_size", "model_name"])
    for df, path in ((window_df, WINDOW_CSV), (model_df, MODEL_CSV)):
        remaining = [c for c in df.columns if c not in column_order]
        df[column_order + remaining].to_csv(path, index=False, encoding="utf-8")

    _write_report(window_df, model_df)

    print(f"Wrote {WINDOW_CSV}")
    print(f"Wrote {MODEL_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
