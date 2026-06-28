"""No-TL CNN epoch/batch/horizon ablation audit.

This script is intentionally isolated from the main experiment runner. It only
uses existing No-TL data-preparation, model-construction, and metric helpers.
Outputs are written under outputs/audits/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
RESULT_CSV = OUT_DIR / "notl_epoch_batch_horizon_ablation.csv"
REPORT_MD = OUT_DIR / "notl_epoch_batch_horizon_ablation.md"
HORIZON_SUMMARY_CSV = OUT_DIR / "notl_horizon_1_5_summary.csv"
WINDOW_COUNT_CSV = OUT_DIR / "notl_training_window_count_check.csv"

RANDOM_SEED = 42
DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
EPOCH_VALUES = [50]
BATCH_VALUES = [4, 8, 16]
HORIZON_VALUES = [1, 2, 3, 4, 5]


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


def _run_single_notl(
    bundle: SplitBundle,
    sequences: Dict[str, np.ndarray],
    metric_protocol: Dict[str, Any],
    horizon: int,
    target_epochs: int,
    batch_size: int,
    learning_rate: float,
    window_size: int,
    notes: str,
) -> Dict[str, Any]:
    base_row: Dict[str, Any] = {
        "dataset_id": DATASET_ID[bundle.dataset],
        "dataset": bundle.dataset,
        "method": "No-TL",
        "random_seed": RANDOM_SEED,
        "horizon": int(horizon),
        "target_epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "train_rows": int(len(bundle.train_df)),
        "val_rows": int(len(bundle.val_df)),
        "test_rows": int(len(bundle.test_df)),
        "train_windows": int(len(sequences["y_train"])),
        "val_windows": int(len(sequences["y_val"])),
        "test_windows": int(len(sequences["y_test"])),
        "feature_columns": "|".join(bundle.feature_columns),
        "window_size": int(window_size),
    }
    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        skipped_notes = (
            f"{notes}; skipped: insufficient windows "
            f"(train_windows={len(sequences['y_train'])}, test_windows={len(sequences['y_test'])})"
        )
        return {
            **base_row,
            "rmse": np.nan,
            "normalized_rmse": np.nan,
            "original_scale_rmse": np.nan,
            "accuracy": np.nan,
            "normalized_accuracy": np.nan,
            "original_scale_accuracy": np.nan,
            "val_rmse": np.nan,
            "test_rmse": np.nan,
            "run_time_seconds": 0.0,
            "notes": skipped_notes,
            "val_normalized_rmse": np.nan,
            "val_original_scale_rmse": np.nan,
            "metric_space": "normalized_minmax_space",
            "prediction_shape": "(skipped)",
        }

    setup_reproducibility(RANDOM_SEED)

    import tensorflow as tf

    tf.keras.backend.clear_session()
    setup_reproducibility(RANDOM_SEED)

    start = time.perf_counter()
    model = build_no_tl_cnn_model(
        input_shape=sequences["x_train"].shape[1:],
        learning_rate=learning_rate,
    )
    fit_kwargs: Dict[str, Any] = {
        "epochs": int(target_epochs),
        "batch_size": int(batch_size),
        "verbose": 0,
    }
    if len(sequences["y_val"]) > 0:
        fit_kwargs["validation_data"] = (sequences["x_val"], sequences["y_val"])

    model.fit(sequences["x_train"], sequences["y_train"], **fit_kwargs)
    y_test_pred = model.predict(sequences["x_test"], verbose=0)
    test_metric = _metric_dict(
        sequences["y_test"],
        y_test_pred,
        metric_protocol=metric_protocol,
        bundle=bundle,
    )

    val_metric: Dict[str, Any]
    if len(sequences["y_val"]) > 0:
        y_val_pred = model.predict(sequences["x_val"], verbose=0)
        val_metric = _metric_dict(
            sequences["y_val"],
            y_val_pred,
            metric_protocol=metric_protocol,
            bundle=bundle,
        )
    else:
        val_metric = {}

    elapsed = time.perf_counter() - start

    return {
        **base_row,
        "rmse": float(test_metric["rmse"]),
        "normalized_rmse": float(test_metric["normalized_rmse"]),
        "original_scale_rmse": test_metric.get("original_scale_rmse"),
        "accuracy": float(test_metric["accuracy"]),
        "normalized_accuracy": float(test_metric["normalized_accuracy"]),
        "original_scale_accuracy": test_metric.get("original_scale_accuracy"),
        "val_rmse": float(val_metric.get("rmse", np.nan)),
        "test_rmse": float(test_metric["rmse"]),
        "run_time_seconds": float(elapsed),
        "notes": notes,
        "val_normalized_rmse": float(val_metric.get("normalized_rmse", np.nan)),
        "val_original_scale_rmse": val_metric.get("original_scale_rmse"),
        "metric_space": str(test_metric["metric_space"]),
        "prediction_shape": tuple(y_test_pred.shape),
    }


def _percent_change(old: float, new: float) -> float:
    if old == 0 or np.isnan(old) or np.isnan(new):
        return float("nan")
    return (new - old) / old * 100.0


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    out = df[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: _format_float(x))
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _choose_stable_batch(results_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    batch_df = results_df[results_df["experiment_group"].eq("batch_ablation")].copy()
    summary = (
        batch_df.groupby("batch_size", as_index=False)
        .agg(
            mean_rmse=("normalized_rmse", "mean"),
            std_rmse=("normalized_rmse", "std"),
            max_rmse=("normalized_rmse", "max"),
        )
        .sort_values(["mean_rmse", "std_rmse", "batch_size"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return int(summary.iloc[0]["batch_size"]), summary


def _build_horizon_summary(results_df: pd.DataFrame, stable_batch_size: int) -> pd.DataFrame:
    horizon_df = results_df[results_df["experiment_group"].eq("horizon_ablation")].copy()
    rows: List[Dict[str, Any]] = []
    for dataset, group in horizon_df.groupby("dataset", sort=True):
        h1 = group[group["horizon"].eq(1)]
        valid = group.dropna(subset=["normalized_rmse"]).copy()
        valid_horizons = "|".join(str(int(v)) for v in valid["horizon"].tolist())
        invalid_horizons = "|".join(str(int(v)) for v in group.loc[group["normalized_rmse"].isna(), "horizon"].tolist())
        h1_rmse = float(h1.iloc[0]["normalized_rmse"]) if not h1.empty and pd.notna(h1.iloc[0]["normalized_rmse"]) else np.nan
        mean_rmse = float(valid["normalized_rmse"].mean()) if not valid.empty else np.nan
        rows.append(
            {
                "dataset_id": DATASET_ID[dataset],
                "dataset": dataset,
                "method": "No-TL",
                "random_seed": RANDOM_SEED,
                "target_epochs": 50,
                "batch_size": int(stable_batch_size),
                "horizon_values_requested": "1|2|3|4|5",
                "valid_horizon_values": valid_horizons,
                "invalid_horizon_values": invalid_horizons,
                "horizon_1_rmse": h1_rmse,
                "horizon_1_5_mean_rmse": mean_rmse,
                "horizon_1_5_std_rmse": float(valid["normalized_rmse"].std(ddof=1)) if len(valid) > 1 else np.nan,
                "horizon_1_5_mean_original_scale_rmse": float(valid["original_scale_rmse"].mean()) if not valid.empty else np.nan,
                "horizon_1_5_mean_accuracy": float(valid["normalized_accuracy"].mean()) if not valid.empty else np.nan,
                "horizon_1_vs_mean_percent_delta": _percent_change(
                    h1_rmse,
                    mean_rmse,
                ),
                "notes": (
                    "Mean is arithmetic mean of normalized_rmse over valid requested horizons. "
                    "Invalid horizons have no train windows under the unchanged split/window_size."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def _build_window_count_check(
    bundles: Dict[str, SplitBundle],
    sequence_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = bundles[dataset]
        for horizon in HORIZON_VALUES:
            seq = sequence_cache[(dataset, horizon)]
            rows.append(
                {
                    "dataset_id": DATASET_ID[dataset],
                    "dataset": dataset,
                    "method": "No-TL",
                    "random_seed": RANDOM_SEED,
                    "horizon": horizon,
                    "window_size": 10,
                    "train_rows": int(len(bundle.train_df)),
                    "val_rows": int(len(bundle.val_df)),
                    "test_rows": int(len(bundle.test_df)),
                    "train_windows": int(len(seq["y_train"])),
                    "val_windows": int(len(seq["y_val"])),
                    "test_windows": int(len(seq["y_test"])),
                    "feature_count": int(seq["x_train"].shape[-1]),
                "feature_columns": "|".join(bundle.feature_columns),
                    "notes": (
                        "Window counts are from unchanged build_tabular_sequence()."
                        if len(seq["y_train"]) > 0
                        else "No train windows under unchanged split/window_size/horizon; training is invalid."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_id", "horizon"]).reset_index(drop=True)


def _write_report(
    results_df: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    batch_summary: pd.DataFrame,
    stable_batch_size: int,
) -> None:
    epoch_df = results_df[results_df["experiment_group"].eq("epoch_ablation")].copy()
    epoch_summary_rows: List[Dict[str, Any]] = []
    for dataset, group in epoch_df.groupby("dataset", sort=True):
        e2 = group[group["target_epochs"].eq(2)].iloc[0]
        best_after = group[group["target_epochs"].gt(2)].sort_values("normalized_rmse").iloc[0]
        epoch_summary_rows.append(
            {
                "dataset": dataset,
                "epoch2_rmse": float(e2["normalized_rmse"]),
                "best_epoch_after_2": int(best_after["target_epochs"]),
                "best_after_2_rmse": float(best_after["normalized_rmse"]),
                "percent_change": _percent_change(float(e2["normalized_rmse"]), float(best_after["normalized_rmse"])),
            }
        )
    epoch_summary = pd.DataFrame(epoch_summary_rows)

    batch_spread_rows: List[Dict[str, Any]] = []
    for dataset, group in results_df[results_df["experiment_group"].eq("batch_ablation")].groupby("dataset", sort=True):
        min_rmse = float(group["normalized_rmse"].min())
        max_rmse = float(group["normalized_rmse"].max())
        batch_spread_rows.append(
            {
                "dataset": dataset,
                "best_batch_size": int(group.sort_values("normalized_rmse").iloc[0]["batch_size"]),
                "min_rmse": min_rmse,
                "max_rmse": max_rmse,
                "spread_percent": _percent_change(min_rmse, max_rmse),
            }
        )
    batch_spread = pd.DataFrame(batch_spread_rows)

    epoch_clear = bool((epoch_summary["percent_change"] <= -10.0).any())
    batch_clear = bool((batch_spread["spread_percent"] >= 10.0).any())
    horizon_clear = bool((horizon_summary["horizon_1_vs_mean_percent_delta"].abs() >= 10.0).any())

    conclusion = (
        "当前证据更支持训练协议与 horizon 口径是 No-TL 差距的主要来源，而不是 CNN 层级结构本身。"
        "本实验保持了同一 `build_no_tl_cnn_model()` 结构；若 RMSE 随 epochs/batch/horizon 明显变化，"
        "结构不变下的协议因素已经足以解释相当一部分差异。"
    )
    if not epoch_clear and not batch_clear and not horizon_clear:
        conclusion = (
            "本轮最小消融没有显示 epochs、batch 或 horizon 口径带来大幅变化；在这种情况下，"
            "No-TL 差距需要继续审计其他训练协议细节或数据/metric 口径，而不能仅归因于这三项。"
        )

    lines = [
        "# No-TL CNN Epoch/Batch/Horizon Ablation",
        "",
        "Scope: only No-TL, Dataset1/2/3, random_seed=42. The script does not modify KNN, source selection, RFE, data cleaning, split, RMSE formula, or CNN structure.",
        "",
        "## Files",
        "",
        f"- Results: `{RESULT_CSV.relative_to(ROOT)}`",
        f"- Horizon summary: `{HORIZON_SUMMARY_CSV.relative_to(ROOT)}`",
        f"- Window count check: `{WINDOW_COUNT_CSV.relative_to(ROOT)}`",
        "",
        "## Protocol",
        "",
        "- CNN factory: `src.models.no_tl_model.build_no_tl_cnn_model()`.",
        "- Data path: `load_dataset -> extract_datetime_features -> build_source_target_split -> temporal_split_by_ratio_or_dates -> normalize_features -> build_tabular_sequence`.",
        "- Metric path: `compute_metrics_with_protocol()`; primary `rmse` is normalized-space RMSE.",
        f"- Stable batch size chosen for horizon 1-5: `{stable_batch_size}`. Selection rule: lowest cross-dataset mean normalized RMSE in batch ablation; ties by lower std.",
        "",
        "## Epoch Ablation",
        "",
        _markdown_table(epoch_summary, ["dataset", "epoch2_rmse", "best_epoch_after_2", "best_after_2_rmse", "percent_change"]),
        "",
        f"Judgment: epoch increase is {'明显改善 at least one dataset' if epoch_clear else 'not clearly improving by the 10% threshold'} (threshold: >=10% RMSE reduction from epoch=2).",
        "",
        "## Batch Size Ablation",
        "",
        _markdown_table(batch_spread, ["dataset", "best_batch_size", "min_rmse", "max_rmse", "spread_percent"]),
        "",
        "Cross-dataset batch summary:",
        "",
        _markdown_table(batch_summary, ["batch_size", "mean_rmse", "std_rmse", "max_rmse"]),
        "",
        f"Judgment: batch size effect is {'明显' if batch_clear else 'not clearly large'} by a 10% within-dataset spread threshold.",
        "",
        "## Horizon 1-5 Aggregation",
        "",
        _markdown_table(
            horizon_summary,
            [
                "dataset",
                "valid_horizon_values",
                "invalid_horizon_values",
                "horizon_1_rmse",
                "horizon_1_5_mean_rmse",
                "horizon_1_5_std_rmse",
                "horizon_1_vs_mean_percent_delta",
            ],
        ),
        "",
        f"Judgment: horizon=1 vs requested horizon 1-5 average difference is {'明显' if horizon_clear else 'not clearly large'} by a 10% absolute delta threshold. Invalid horizons are excluded from the arithmetic mean and listed explicitly.",
        "",
        "## Overall Judgment",
        "",
        conclusion,
        "",
        "## Notes",
        "",
        "- Because the target observed window is only about 30 days and window_size=10, No-TL train windows remain very small across horizons.",
        "- Neural network determinism can still vary slightly across TensorFlow/hardware backends, but every run resets random_seed=42 before model construction.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level="WARNING", log_file=None)
    cfg = _load_config()
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))
    exp_cfg = cfg["single_experiment"]
    learning_rate = float(exp_cfg.get("learning_rate", 1e-4))
    window_size = int(exp_cfg.get("window_size", 10))

    bundles = {dataset: _prepare_dataset(dataset, cfg) for dataset in DATASETS}
    sequence_cache = {
        (dataset, horizon): _build_sequences(bundles[dataset], horizon=horizon, window_size=window_size)
        for dataset in DATASETS
        for horizon in HORIZON_VALUES
    }

    raw_cache: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    output_rows: List[Dict[str, Any]] = []

    def run_or_get(dataset: str, horizon: int, target_epochs: int, batch_size: int, notes: str) -> Dict[str, Any]:
        key = (dataset, horizon, target_epochs, batch_size)
        if key not in raw_cache:
            raw_cache[key] = _run_single_notl(
                bundle=bundles[dataset],
                sequences=sequence_cache[(dataset, horizon)],
                metric_protocol=metric_protocol,
                horizon=horizon,
                target_epochs=target_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                window_size=window_size,
                notes=notes,
            )
        row = dict(raw_cache[key])
        row["notes"] = notes
        return row

    for dataset in DATASETS:
        for epochs in EPOCH_VALUES:
            row = run_or_get(
                dataset=dataset,
                horizon=1,
                target_epochs=epochs,
                batch_size=16,
                notes="epoch ablation: batch_size=16, horizon=1",
            )
            row["experiment_group"] = "epoch_ablation"
            output_rows.append(row)

        for batch_size in BATCH_VALUES:
            row = run_or_get(
                dataset=dataset,
                horizon=1,
                target_epochs=50,
                batch_size=batch_size,
                notes="batch size ablation: target_epochs=50, horizon=1",
            )
            row["experiment_group"] = "batch_ablation"
            output_rows.append(row)

    first_pass_df = pd.DataFrame(output_rows)
    stable_batch_size, batch_summary = _choose_stable_batch(first_pass_df)

    for dataset in DATASETS:
        for horizon in HORIZON_VALUES:
            row = run_or_get(
                dataset=dataset,
                horizon=horizon,
                target_epochs=50,
                batch_size=stable_batch_size,
                notes=f"horizon ablation: target_epochs=50, stable_batch_size={stable_batch_size}",
            )
            row["experiment_group"] = "horizon_ablation"
            output_rows.append(row)

    results_df = pd.DataFrame(output_rows).sort_values(
        ["dataset_id", "experiment_group", "horizon", "target_epochs", "batch_size"]
    )
    required_first = [
        "dataset_id",
        "dataset",
        "method",
        "random_seed",
        "horizon",
        "target_epochs",
        "batch_size",
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
        "notes",
    ]
    remaining = [c for c in results_df.columns if c not in required_first]
    results_df = results_df[required_first + remaining]
    results_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")

    window_counts = _build_window_count_check(bundles, sequence_cache)
    window_counts.to_csv(WINDOW_COUNT_CSV, index=False, encoding="utf-8")

    horizon_summary = _build_horizon_summary(results_df, stable_batch_size=stable_batch_size)
    horizon_summary.to_csv(HORIZON_SUMMARY_CSV, index=False, encoding="utf-8")

    _write_report(
        results_df=results_df,
        horizon_summary=horizon_summary,
        batch_summary=batch_summary,
        stable_batch_size=stable_batch_size,
    )

    print(f"Wrote {RESULT_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {HORIZON_SUMMARY_CSV}")
    print(f"Wrote {WINDOW_COUNT_CSV}")


if __name__ == "__main__":
    main()
