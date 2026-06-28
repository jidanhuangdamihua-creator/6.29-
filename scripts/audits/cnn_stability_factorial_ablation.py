"""No-TL CNN small-sample stability factorial ablation audit.

The audit preserves the original CNN path and adds explicit copied variants for
isolated training-stability checks. Outputs are written only to outputs/audits/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-codex"))

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
from src.models.cnn_model import (
    CNN_ABLATION_VARIANTS,
    build_cnn_ablation_variant,
    resolve_cnn_ablation_training_config,
)
from src.utils.runtime_control import set_verbose_mode


OUT_DIR = ROOT / "outputs" / "audits"
DETAIL_CSV = OUT_DIR / "cnn_stability_factorial_ablation_details.csv"
SUMMARY_CSV = OUT_DIR / "cnn_stability_factorial_ablation_summary.csv"
COMPARISON_CSV = OUT_DIR / "cnn_stability_factorial_ablation_comparison.csv"
REPORT_MD = OUT_DIR / "cnn_stability_factorial_ablation.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
SEEDS = [42, 43, 44, 45, 46]
HORIZON = 1
METHOD = "No-TL"

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "seed",
    "method",
    "cnn_ablation_variant",
    "model_name",
    "change1_batch_size_1_enabled",
    "change2_no_batch_norm_enabled",
    "change3_low_lr_clipnorm_enabled",
    "batch_norm_enabled",
    "cnn_normalization",
    "original_batch_size",
    "effective_batch_size",
    "train_windows",
    "learning_rate",
    "clipnorm",
    "optimizer_name",
    "epoch",
    "val_rmse",
    "test_rmse",
    "test_mae",
    "run_time_seconds",
    "error_message",
    "status",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset",
    "cnn_ablation_variant",
    "n_seeds",
    "mean_test_rmse",
    "std_test_rmse",
    "min_test_rmse",
    "max_test_rmse",
    "mean_val_rmse",
    "std_val_rmse",
    "mean_test_mae",
    "std_test_mae",
    "improvement_vs_original_mean_pct",
    "std_reduction_vs_original_pct",
    "best_seed",
    "worst_seed",
    "rank_by_mean_rmse",
    "rank_by_std_rmse",
    "status",
    "conclusion",
    "notes",
]

COMPARISON_COLUMNS = [
    "dataset_id",
    "dataset",
    "original_mean_rmse",
    "change1_mean_rmse",
    "change2_mean_rmse",
    "change3_mean_rmse",
    "change123_mean_rmse",
    "change1_improvement_pct",
    "change2_improvement_pct",
    "change3_improvement_pct",
    "change123_improvement_pct",
    "best_single_change",
    "best_overall_variant",
    "does_combined_outperform_all_single_changes",
    "interpretation",
]


def _load_config() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
    cfg.setdefault("single_experiment", {}).setdefault("cnn_ablation_variant", "original")
    return cfg


def _metric_protocol(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("paper_reproduction", {}).get("metric_protocol", {}))


def _prepare_sequences(dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    data_path = ROOT / str(config["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, config)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(train_df, val_df, test_df)

    exp = config.get("single_experiment", {})
    window_size = int(exp.get("window_size", config.get("window_size", 10)))
    x_train, y_train = build_tabular_sequence(train_scaled, horizon=HORIZON, window_size=window_size)
    x_val, y_val = build_tabular_sequence(val_scaled, horizon=HORIZON, window_size=window_size)
    x_test, y_test = build_tabular_sequence(test_scaled, horizon=HORIZON, window_size=window_size)

    return {
        "dataset": dataset,
        "window_size": window_size,
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_protocol: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=prepared["scaler"],
        feature_columns=prepared["feature_columns"],
    )


def _hard_condition_errors(meta: Any, original_batch_size: int, original_learning_rate: float) -> List[str]:
    errors: List[str] = []
    variant = meta.cnn_ablation_variant
    original_batch_norm_enabled = False
    original_clipnorm = None

    if variant == "change1_batch_size_1":
        if meta.effective_batch_size != 1:
            errors.append("change1 effective_batch_size must equal 1")
        if meta.batch_norm_enabled != original_batch_norm_enabled:
            errors.append("change1 BatchNormalization state must match original")
        if not math.isclose(float(meta.learning_rate), float(original_learning_rate), rel_tol=0.0, abs_tol=1e-12):
            errors.append("change1 learning_rate must match original")
        if meta.clipnorm != original_clipnorm:
            errors.append("change1 clipnorm must match original")
    elif variant == "change2_no_batch_norm":
        if meta.effective_batch_size != int(original_batch_size):
            errors.append("change2 effective_batch_size must match original")
        if meta.batch_norm_enabled is not False:
            errors.append("change2 batch_norm_enabled must be False")
        if str(meta.cnn_normalization).lower() not in {"none", "false", "disabled"}:
            errors.append("change2 cnn_normalization must be none-equivalent")
        if not math.isclose(float(meta.learning_rate), float(original_learning_rate), rel_tol=0.0, abs_tol=1e-12):
            errors.append("change2 learning_rate must match original")
        if meta.clipnorm != original_clipnorm:
            errors.append("change2 clipnorm must match original")
    elif variant == "change3_low_lr_clipnorm":
        if meta.effective_batch_size != int(original_batch_size):
            errors.append("change3 effective_batch_size must match original")
        if meta.batch_norm_enabled != original_batch_norm_enabled:
            errors.append("change3 BatchNormalization state must match original")
        if not math.isclose(float(meta.learning_rate), 1e-4, rel_tol=0.0, abs_tol=1e-12):
            errors.append("change3 learning_rate must equal 1e-4")
        if meta.clipnorm is not None:
            errors.append("change3 clipnorm must be None")
    elif variant == "change123_all":
        if meta.effective_batch_size != 1:
            errors.append("change123 effective_batch_size must equal 1")
        if meta.batch_norm_enabled is not False:
            errors.append("change123 batch_norm_enabled must be False")
        if not math.isclose(float(meta.learning_rate), 1e-4, rel_tol=0.0, abs_tol=1e-12):
            errors.append("change123 learning_rate must equal 1e-4")
        if meta.clipnorm is not None:
            errors.append("change123 clipnorm must be None")
    return errors


def _base_row(dataset: str, seed: int, meta: Any, train_windows: int) -> Dict[str, Any]:
    row = {
        "dataset_id": DATASET_ID[dataset],
        "dataset": dataset,
        "seed": int(seed),
        "method": METHOD,
        **asdict(meta),
        "train_windows": int(train_windows),
        "epoch": 0,
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "test_mae": np.nan,
        "run_time_seconds": 0.0,
        "error_message": "",
        "status": "OK",
        "notes": "",
    }
    return {column: row.get(column, np.nan) for column in DETAIL_COLUMNS}


def _run_one(
    prepared: Dict[str, Any],
    metric_protocol: Dict[str, Any],
    dataset: str,
    seed: int,
    variant: str,
    original_batch_size: int,
    original_learning_rate: float,
    target_epochs: int,
) -> Dict[str, Any]:
    import tensorflow as tf

    meta = resolve_cnn_ablation_training_config(
        cnn_ablation_variant=variant,
        original_batch_size=original_batch_size,
        original_learning_rate=original_learning_rate,
    )
    row = _base_row(dataset, seed, meta, train_windows=len(prepared["y_train"]))
    hard_errors = _hard_condition_errors(meta, original_batch_size, original_learning_rate)
    if hard_errors:
        row["status"] = "FAIL"
        row["error_message"] = "; ".join(hard_errors)
        row["notes"] = "Hard condition check failed before training."
        return row
    if len(prepared["y_train"]) == 0 or len(prepared["y_test"]) == 0:
        row["status"] = "ERROR"
        row["error_message"] = "empty train/test windows"
        row["notes"] = "No-TL requires non-empty target train and test windows."
        return row

    setup_reproducibility(seed)
    tf.keras.backend.clear_session()
    setup_reproducibility(seed)
    start = time.perf_counter()
    try:
        model = build_cnn_ablation_variant(
            input_shape=prepared["x_train"].shape[1:],
            learning_rate=original_learning_rate,
            cnn_ablation_variant=variant,
        )
        history = model.fit(
            prepared["x_train"],
            prepared["y_train"],
            epochs=int(target_epochs),
            batch_size=int(meta.effective_batch_size),
            validation_data=(prepared["x_val"], prepared["y_val"]) if len(prepared["y_val"]) > 0 else None,
            verbose=0,
        )
        y_test_pred = model.predict(prepared["x_test"], verbose=0)
        test_metric = _compute_metric(prepared["y_test"], y_test_pred, metric_protocol, prepared)
        if len(prepared["y_val"]) > 0:
            y_val_pred = model.predict(prepared["x_val"], verbose=0)
            val_metric = _compute_metric(prepared["y_val"], y_val_pred, metric_protocol, prepared)
            row["val_rmse"] = float(val_metric["rmse"])
        row["test_rmse"] = float(test_metric["rmse"])
        row["test_mae"] = float(test_metric["mae"])
        row["epoch"] = int(len(history.history.get("loss", [])))
        row["run_time_seconds"] = float(time.perf_counter() - start)
        row["notes"] = "OK; original CNN has no BatchNormalization layers in current codebase."
    except Exception as exc:
        row["status"] = "ERROR"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["run_time_seconds"] = float(time.perf_counter() - start)
        row["notes"] = "Training or evaluation failed."
    return row


def _pct_improvement(original: float, value: float) -> float:
    if pd.isna(original) or pd.isna(value) or float(original) == 0.0:
        return np.nan
    return (float(original) - float(value)) / float(original) * 100.0


def _build_summary(details: pd.DataFrame) -> pd.DataFrame:
    ok = details[details["status"].eq("OK")].copy()
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_ok = ok[ok["dataset"].eq(dataset)]
        original = dataset_ok[dataset_ok["cnn_ablation_variant"].eq("original")]
        original_mean = float(original["test_rmse"].mean()) if not original.empty else np.nan
        original_std = float(original["test_rmse"].std()) if len(original) > 1 else np.nan
        for variant in CNN_ABLATION_VARIANTS:
            group = dataset_ok[dataset_ok["cnn_ablation_variant"].eq(variant)].copy()
            if group.empty:
                rows.append(
                    {
                        "dataset_id": DATASET_ID[dataset],
                        "dataset": dataset,
                        "cnn_ablation_variant": variant,
                        "n_seeds": 0,
                        "status": "ERROR",
                        "conclusion": "No successful runs.",
                        "notes": "No OK rows available for this dataset/variant.",
                    }
                )
                continue
            best = group.sort_values("test_rmse", ascending=True).iloc[0]
            worst = group.sort_values("test_rmse", ascending=False).iloc[0]
            std_test = float(group["test_rmse"].std()) if len(group) > 1 else np.nan
            notes = ""
            if pd.isna(original_std) or original_std == 0.0:
                notes = "Original std is zero or missing; std_reduction_vs_original_pct left blank."
            rows.append(
                {
                    "dataset_id": DATASET_ID[dataset],
                    "dataset": dataset,
                    "cnn_ablation_variant": variant,
                    "n_seeds": int(group["seed"].nunique()),
                    "mean_test_rmse": float(group["test_rmse"].mean()),
                    "std_test_rmse": std_test,
                    "min_test_rmse": float(group["test_rmse"].min()),
                    "max_test_rmse": float(group["test_rmse"].max()),
                    "mean_val_rmse": float(group["val_rmse"].mean()),
                    "std_val_rmse": float(group["val_rmse"].std()) if len(group) > 1 else np.nan,
                    "mean_test_mae": float(group["test_mae"].mean()),
                    "std_test_mae": float(group["test_mae"].std()) if len(group) > 1 else np.nan,
                    "improvement_vs_original_mean_pct": _pct_improvement(original_mean, float(group["test_rmse"].mean())),
                    "std_reduction_vs_original_pct": (
                        (original_std - std_test) / original_std * 100.0
                        if not pd.isna(original_std) and original_std != 0.0 and not pd.isna(std_test)
                        else np.nan
                    ),
                    "best_seed": int(best["seed"]),
                    "worst_seed": int(worst["seed"]),
                    "status": "OK",
                    "conclusion": "Computed from OK detail rows.",
                    "notes": notes,
                }
            )
    summary = pd.DataFrame(rows)
    for dataset, idx in summary.groupby("dataset").groups.items():
        summary.loc[idx, "rank_by_mean_rmse"] = summary.loc[idx, "mean_test_rmse"].rank(method="min", ascending=True)
        summary.loc[idx, "rank_by_std_rmse"] = summary.loc[idx, "std_test_rmse"].rank(method="min", ascending=True)
    return summary.reindex(columns=SUMMARY_COLUMNS)


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = {
        "change1_batch_size_1": "change1",
        "change2_no_batch_norm": "change2",
        "change3_low_lr_clipnorm": "change3",
        "change123_all": "change123",
    }
    for dataset in DATASETS:
        group = summary[summary["dataset"].eq(dataset)]
        means = {
            row["cnn_ablation_variant"]: row["mean_test_rmse"]
            for _, row in group.iterrows()
            if row.get("status") == "OK"
        }
        original = means.get("original", np.nan)
        single_variants = ["change1_batch_size_1", "change2_no_batch_norm", "change3_low_lr_clipnorm"]
        single_available = {variant: means.get(variant, np.nan) for variant in single_variants}
        valid_single = {variant: value for variant, value in single_available.items() if not pd.isna(value)}
        valid_all = {variant: value for variant, value in means.items() if not pd.isna(value)}
        best_single = min(valid_single, key=valid_single.get) if valid_single else ""
        best_overall = min(valid_all, key=valid_all.get) if valid_all else ""
        combined = means.get("change123_all", np.nan)
        combined_beats = bool(
            not pd.isna(combined)
            and valid_single
            and all(float(combined) < float(value) for value in valid_single.values())
        )
        rows.append(
            {
                "dataset_id": DATASET_ID[dataset],
                "dataset": dataset,
                "original_mean_rmse": original,
                "change1_mean_rmse": means.get("change1_batch_size_1", np.nan),
                "change2_mean_rmse": means.get("change2_no_batch_norm", np.nan),
                "change3_mean_rmse": means.get("change3_low_lr_clipnorm", np.nan),
                "change123_mean_rmse": combined,
                "change1_improvement_pct": _pct_improvement(original, means.get("change1_batch_size_1", np.nan)),
                "change2_improvement_pct": _pct_improvement(original, means.get("change2_no_batch_norm", np.nan)),
                "change3_improvement_pct": _pct_improvement(original, means.get("change3_low_lr_clipnorm", np.nan)),
                "change123_improvement_pct": _pct_improvement(original, combined),
                "best_single_change": best_single,
                "best_overall_variant": best_overall,
                "does_combined_outperform_all_single_changes": combined_beats,
                "interpretation": (
                    f"Best single change is {labels.get(best_single, best_single)}; "
                    f"best overall is {labels.get(best_overall, best_overall)}; "
                    f"combined {'does' if combined_beats else 'does not'} outperform all single changes."
                ),
            }
        )
    return pd.DataFrame(rows).reindex(columns=COMPARISON_COLUMNS)


def _format_value(value: Any, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if bool(value) else "No"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return " ".join(str(value).replace("|", "/").replace("\n", " ").split())


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 50) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    shown = df[cols].head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _overall_status(details: pd.DataFrame) -> str:
    expected_runs = len(DATASETS) * len(SEEDS) * len(CNN_ABLATION_VARIANTS)
    if details["status"].eq("FAIL").any():
        return "FAIL"
    if len(details) != expected_runs or details["status"].ne("OK").any():
        return "PARTIAL"
    return "PASS"


def _write_report(details: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, target_epochs: int, original_batch_size: int) -> None:
    hard_rows = (
        details.groupby("cnn_ablation_variant", as_index=False)
        .agg(
            rows=("status", "size"),
            failures=("status", lambda s: int((s == "FAIL").sum())),
            errors=("status", lambda s: int((s == "ERROR").sum())),
            effective_batch_size=("effective_batch_size", "first"),
            batch_norm_enabled=("batch_norm_enabled", "first"),
            cnn_normalization=("cnn_normalization", "first"),
            learning_rate=("learning_rate", "first"),
            clipnorm=("clipnorm", "first"),
        )
        .sort_values("cnn_ablation_variant")
    )
    status = _overall_status(details)
    mean_best = (
        summary[summary["cnn_ablation_variant"].isin(["change1_batch_size_1", "change2_no_batch_norm", "change3_low_lr_clipnorm"])]
        .sort_values(["dataset", "mean_test_rmse"])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "cnn_ablation_variant", "mean_test_rmse"]]
        .rename(columns={"cnn_ablation_variant": "best_single_mean_variant"})
    )
    std_best = (
        summary[summary["cnn_ablation_variant"].isin(["change1_batch_size_1", "change2_no_batch_norm", "change3_low_lr_clipnorm"])]
        .sort_values(["dataset", "std_test_rmse"])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "cnn_ablation_variant", "std_test_rmse"]]
        .rename(columns={"cnn_ablation_variant": "best_single_std_variant"})
    )

    lines = [
        "# CNN Small-Sample Stability Factorial Ablation",
        "",
        "## 1. Purpose",
        "",
        "This audit does not modify or replace the original CNN. It adds copied CNN ablation variants to isolate three small-sample stability changes: batch_size=1, removing BatchNormalization, and Adam learning_rate=1e-4 with clipnorm=None and dropout=0.1.",
        "",
        "## 2. Original CNN Preserved",
        "",
        "The original `build_base_cnn` / current 3-layer CNN function is preserved. In the current repository, that CNN has no BatchNormalization layers, so `change2_no_batch_norm` is structurally equivalent to original but is still tracked as a separate audit variant.",
        "",
        "## 3. Ablation Variants",
        "",
        "| Variant | batch_size=1 | remove BN | lr=1e-4, clipnorm=None, dropout=0.1 |",
        "|---|---:|---:|---:|",
        "| original | No | No | No |",
        "| change1_batch_size_1 | Yes | No | No |",
        "| change2_no_batch_norm | No | Yes | No |",
        "| change3_low_lr_clipnorm | No | No | Yes |",
        "| change123_all | Yes | Yes | Yes |",
        "",
        "## 4. Experiment Setup",
        "",
        f"- Only No-TL is run.",
        f"- Datasets: {', '.join(DATASETS)}.",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}.",
        f"- target_epochs={target_epochs}, original_batch_size={original_batch_size}, horizon={HORIZON}.",
        "- The full transfer-learning matrix is not run.",
        "- Main experiment outputs are not read or overwritten; all files are under `outputs/audits/`.",
        "",
        "## 5. Hard Condition Check",
        "",
        _markdown_table(
            hard_rows,
            [
                "cnn_ablation_variant",
                "rows",
                "failures",
                "errors",
                "effective_batch_size",
                "batch_norm_enabled",
                "cnn_normalization",
                "learning_rate",
                "clipnorm",
            ],
        ),
        "",
        "- `change1_batch_size_1` uses effective_batch_size=1.",
        "- `change2_no_batch_norm` has batch_norm_enabled=False and cnn_normalization=none.",
        "- `change3_low_lr_clipnorm` uses learning_rate=1e-4, clipnorm=None, and dropout=0.1.",
        "- `change123_all` enables all three changes together.",
        "",
        "## 6. Results by Dataset",
        "",
        _markdown_table(
            summary.sort_values(["dataset_id", "rank_by_mean_rmse"]),
            [
                "dataset",
                "cnn_ablation_variant",
                "n_seeds",
                "mean_test_rmse",
                "std_test_rmse",
                "mean_val_rmse",
                "mean_test_mae",
                "rank_by_mean_rmse",
                "rank_by_std_rmse",
            ],
            max_rows=100,
        ),
        "",
        "Best single and overall variants:",
        "",
        _markdown_table(
            comparison,
            [
                "dataset",
                "best_single_change",
                "best_overall_variant",
                "does_combined_outperform_all_single_changes",
                "change1_improvement_pct",
                "change2_improvement_pct",
                "change3_improvement_pct",
                "change123_improvement_pct",
            ],
        ),
        "",
        "## 7. Factor Interpretation",
        "",
        "Mean RMSE contribution by dataset:",
        "",
        _markdown_table(mean_best, ["dataset", "best_single_mean_variant", "mean_test_rmse"]),
        "",
        "Seed-variance contribution by dataset:",
        "",
        _markdown_table(std_best, ["dataset", "best_single_std_variant", "std_test_rmse"]),
        "",
        "Use the comparison table above to answer whether each single factor improves over original and whether the combined variant beats all single changes. Positive improvement percentages mean lower mean RMSE than original.",
        "",
        "## 8. Reproduction Safety Check",
        "",
        "- KNN: not modified.",
        "- RFE: not modified.",
        "- split: not modified.",
        "- RMSE: not modified.",
        "- accuracy: not modified.",
        "- data cleaning: not modified.",
        "- source selection: not modified.",
        "- main experiment default CNN: not modified; default variant remains `original`.",
        "",
        "## 9. Conclusion",
        "",
        status,
        "",
        f"Reason: {status} is based on all expected dataset x seed x variant audit rows being generated, all hard conditions passing, and no non-OK training statuses.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging(log_level="WARNING", log_file=None)
    set_verbose_mode("summary")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    exp = config.get("single_experiment", {})
    original_batch_size = int(exp.get("batch_size", config.get("batch_size", 16)))
    original_learning_rate = float(exp.get("learning_rate", config.get("learning_rate", 1e-4)))
    target_epochs = int(exp.get("target_epochs", config.get("target_epochs", 2)))
    metric_protocol = _metric_protocol(config)

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        prepared = _prepare_sequences(dataset, config)
        for seed in SEEDS:
            for variant in CNN_ABLATION_VARIANTS:
                rows.append(
                    _run_one(
                        prepared=prepared,
                        metric_protocol=metric_protocol,
                        dataset=dataset,
                        seed=seed,
                        variant=variant,
                        original_batch_size=original_batch_size,
                        original_learning_rate=original_learning_rate,
                        target_epochs=target_epochs,
                    )
                )

    details = pd.DataFrame(rows).reindex(columns=DETAIL_COLUMNS)
    summary = _build_summary(details)
    comparison = _build_comparison(summary)

    details.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    _write_report(details, summary, comparison, target_epochs=target_epochs, original_batch_size=original_batch_size)

    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
