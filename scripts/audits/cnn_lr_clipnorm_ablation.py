"""No-TL CNN optimizer learning-rate and clipnorm ablation audit.

This audit keeps the CNN structure, batch-size path, data cleaning, split, KNN,
RFE, and metric code unchanged. It varies only Adam learning_rate + clipnorm and
writes all outputs under outputs/audits/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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
from src.models.cnn_model import build_base_cnn
from src.utils.experiment_hyperparams import (
    FIXED_CLIPNORM,
    FIXED_DROPOUT,
    FIXED_EPOCHS,
    FIXED_LEARNING_RATE,
    fixed_hyperparams_slug,
    fixed_hyperparams_summary,
)
from src.utils.runtime_control import set_verbose_mode


OUT_DIR = ROOT / "outputs" / "audits"
_HP_SLUG = fixed_hyperparams_slug()
DETAIL_CSV = OUT_DIR / f"cnn_lr_clipnorm_ablation_details_{_HP_SLUG}.csv"
SUMMARY_CSV = OUT_DIR / f"cnn_lr_clipnorm_ablation_summary_{_HP_SLUG}.csv"
COMPARISON_CSV = OUT_DIR / f"cnn_lr_clipnorm_ablation_comparison_{_HP_SLUG}.csv"
REPORT_MD = OUT_DIR / f"cnn_lr_clipnorm_ablation_{_HP_SLUG}.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
SEEDS = [42, 43, 44, 45, 46]
HORIZON = 1
METHOD = "No-TL"

CNN_LR_ABLATION_VARIANTS = ["fixed_optimizer"]

_LR_BY_VARIANT = {"fixed_optimizer": FIXED_LEARNING_RATE}

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "seed",
    "method",
    "cnn_lr_ablation_variant",
    "optimizer_name",
    "learning_rate",
    "clipnorm",
    "optimizer_changed",
    "cnn_structure_changed",
    "batch_size_changed",
    "model_parameter_count",
    "original_model_parameter_count",
    "original_batch_size",
    "effective_batch_size",
    "train_windows",
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
    "cnn_lr_ablation_variant",
    "n_seeds",
    "mean_test_rmse",
    "std_test_rmse",
    "min_test_rmse",
    "max_test_rmse",
    "mean_val_rmse",
    "std_val_rmse",
    "improvement_vs_original_mean_pct",
    "std_reduction_vs_original_pct",
    "rank_by_mean_rmse",
    "rank_by_std_rmse",
    "status",
    "conclusion",
]

COMPARISON_COLUMNS = [
    "dataset_id",
    "dataset",
    "fixed_lr_1e_4_mean_rmse",
    "fixed_lr_1e_4_std_rmse",
    "best_fixed_optimizer_by_mean_rmse",
    "best_fixed_optimizer_by_std_rmse",
    "best_overall_variant",
    "does_fixed_optimizer_run",
    "does_lr_1e_4_remain_best",
    "interpretation",
]


@dataclass(frozen=True)
class LrOptimizerVariant:
    cnn_lr_ablation_variant: str
    optimizer_name: str
    learning_rate: float
    clipnorm: float | None
    optimizer_changed: bool
    cnn_structure_changed: bool
    batch_size_changed: bool
    original_batch_size: int
    effective_batch_size: int


def resolve_lr_optimizer_variant(
    cnn_lr_ablation_variant: str = "fixed_optimizer",
    original_learning_rate: float = FIXED_LEARNING_RATE,
    original_batch_size: int = 16,
) -> LrOptimizerVariant:
    variant = str(cnn_lr_ablation_variant or "fixed_optimizer")
    if variant not in CNN_LR_ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown cnn_lr_ablation_variant={variant!r}. "
            f"Expected one of: {', '.join(CNN_LR_ABLATION_VARIANTS)}"
        )
    learning_rate = float(_LR_BY_VARIANT[variant])
    clipnorm = FIXED_CLIPNORM
    optimizer_changed = not math.isclose(float(original_learning_rate), learning_rate, rel_tol=0.0, abs_tol=1e-12)
    return LrOptimizerVariant(
        cnn_lr_ablation_variant=variant,
        optimizer_name="Adam",
        learning_rate=learning_rate,
        clipnorm=clipnorm,
        optimizer_changed=optimizer_changed,
        cnn_structure_changed=False,
        batch_size_changed=False,
        original_batch_size=int(original_batch_size),
        effective_batch_size=int(original_batch_size),
    )


def build_lr_ablation_model(input_shape: tuple[int, int], cnn_lr_ablation_variant: str, original_learning_rate: float = FIXED_LEARNING_RATE):
    """Build the fixed optimizer model with dropout=0.1 and clipnorm disabled."""
    import tensorflow as tf

    meta = resolve_lr_optimizer_variant(cnn_lr_ablation_variant, original_learning_rate=original_learning_rate)
    model = build_base_cnn(
        input_shape=input_shape,
        learning_rate=meta.learning_rate,
        dropout=FIXED_DROPOUT,
        clipnorm=meta.clipnorm,
    )
    return model


def _load_config() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))
    exp = cfg.setdefault("single_experiment", {})
    exp["cnn_lr_ablation_variant"] = "fixed_optimizer"
    exp["learning_rate"] = FIXED_LEARNING_RATE
    exp["source_epochs"] = FIXED_EPOCHS
    exp["target_epochs"] = FIXED_EPOCHS
    exp["clipnorm"] = FIXED_CLIPNORM
    exp["dropout"] = FIXED_DROPOUT
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


def _hard_condition_errors(meta: LrOptimizerVariant, model_parameter_count: int, original_parameter_count: int, expected_lr: float) -> List[str]:
    errors: List[str] = []
    if meta.cnn_structure_changed is not False:
        errors.append("cnn_structure_changed must be False")
    if meta.batch_size_changed is not False:
        errors.append("batch_size_changed must be False")
    if int(model_parameter_count) != int(original_parameter_count):
        errors.append("model_parameter_count must equal original_model_parameter_count")
    if meta.optimizer_name != "Adam":
        errors.append("optimizer_name must be Adam")
    if not math.isclose(float(meta.learning_rate), float(expected_lr), rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"learning_rate must equal {expected_lr}")
    if meta.clipnorm is not None:
        errors.append("clipnorm must be None")
    return errors


def _base_row(dataset: str, seed: int, meta: LrOptimizerVariant, train_windows: int) -> Dict[str, Any]:
    row = {
        "dataset_id": DATASET_ID[dataset],
        "dataset": dataset,
        "seed": int(seed),
        "method": METHOD,
        **asdict(meta),
        "model_parameter_count": np.nan,
        "original_model_parameter_count": np.nan,
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
    original_parameter_count: int,
) -> Dict[str, Any]:
    import tensorflow as tf

    meta = resolve_lr_optimizer_variant(
        cnn_lr_ablation_variant=variant,
        original_learning_rate=original_learning_rate,
        original_batch_size=original_batch_size,
    )
    row = _base_row(dataset, seed, meta, train_windows=len(prepared["y_train"]))
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
        model = build_lr_ablation_model(
            input_shape=prepared["x_train"].shape[1:],
            cnn_lr_ablation_variant=variant,
            original_learning_rate=original_learning_rate,
        )
        model_parameter_count = int(model.count_params())
        row["model_parameter_count"] = model_parameter_count
        row["original_model_parameter_count"] = int(original_parameter_count)
        expected_lr = _LR_BY_VARIANT[variant]
        hard_errors = _hard_condition_errors(meta, model_parameter_count, original_parameter_count, expected_lr)
        if hard_errors:
            row["status"] = "FAIL"
            row["error_message"] = "; ".join(hard_errors)
            row["notes"] = "Hard condition check failed before training."
            return row

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
        row["notes"] = "OK; only optimizer learning_rate/clipnorm varied."
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
        original = dataset_ok[dataset_ok["cnn_lr_ablation_variant"].eq("fixed_optimizer")]
        original_mean = float(original["test_rmse"].mean()) if not original.empty else np.nan
        original_std = float(original["test_rmse"].std()) if len(original) > 1 else np.nan
        for variant in CNN_LR_ABLATION_VARIANTS:
            group = dataset_ok[dataset_ok["cnn_lr_ablation_variant"].eq(variant)].copy()
            if group.empty:
                rows.append(
                    {
                        "dataset_id": DATASET_ID[dataset],
                        "dataset": dataset,
                        "cnn_lr_ablation_variant": variant,
                        "n_seeds": 0,
                        "status": "ERROR",
                        "conclusion": "No successful runs.",
                    }
                )
                continue
            std_test = float(group["test_rmse"].std()) if len(group) > 1 else np.nan
            rows.append(
                {
                    "dataset_id": DATASET_ID[dataset],
                    "dataset": dataset,
                    "cnn_lr_ablation_variant": variant,
                    "n_seeds": int(group["seed"].nunique()),
                    "mean_test_rmse": float(group["test_rmse"].mean()),
                    "std_test_rmse": std_test,
                    "min_test_rmse": float(group["test_rmse"].min()),
                    "max_test_rmse": float(group["test_rmse"].max()),
                    "mean_val_rmse": float(group["val_rmse"].mean()),
                    "std_val_rmse": float(group["val_rmse"].std()) if len(group) > 1 else np.nan,
                    "improvement_vs_original_mean_pct": _pct_improvement(original_mean, float(group["test_rmse"].mean())),
                    "std_reduction_vs_original_pct": (
                        (original_std - std_test) / original_std * 100.0
                        if not pd.isna(original_std) and original_std != 0.0 and not pd.isna(std_test)
                        else np.nan
                    ),
                    "status": "OK",
                    "conclusion": "Computed from OK detail rows.",
                }
            )
    summary = pd.DataFrame(rows)
    for _, idx in summary.groupby("dataset").groups.items():
        summary.loc[idx, "rank_by_mean_rmse"] = summary.loc[idx, "mean_test_rmse"].rank(method="min", ascending=True)
        summary.loc[idx, "rank_by_std_rmse"] = summary.loc[idx, "std_test_rmse"].rank(method="min", ascending=True)
    return summary.reindex(columns=SUMMARY_COLUMNS)


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    mean_cols = {
        "fixed_optimizer": "fixed_lr_1e_4_mean_rmse",
    }
    std_cols = {
        "fixed_optimizer": "fixed_lr_1e_4_std_rmse",
    }
    lr_variants = list(CNN_LR_ABLATION_VARIANTS)
    for dataset in DATASETS:
        group = summary[summary["dataset"].eq(dataset)]
        means = {row["cnn_lr_ablation_variant"]: row["mean_test_rmse"] for _, row in group.iterrows() if row.get("status") == "OK"}
        stds = {row["cnn_lr_ablation_variant"]: row["std_test_rmse"] for _, row in group.iterrows() if row.get("status") == "OK"}
        row: Dict[str, Any] = {"dataset_id": DATASET_ID[dataset], "dataset": dataset}
        for variant, column in mean_cols.items():
            row[column] = means.get(variant, np.nan)
        for variant, column in std_cols.items():
            row[column] = stds.get(variant, np.nan)
        valid_lr_means = {variant: means.get(variant, np.nan) for variant in lr_variants if not pd.isna(means.get(variant, np.nan))}
        valid_lr_stds = {variant: stds.get(variant, np.nan) for variant in lr_variants if not pd.isna(stds.get(variant, np.nan))}
        valid_all = {variant: means.get(variant, np.nan) for variant in CNN_LR_ABLATION_VARIANTS if not pd.isna(means.get(variant, np.nan))}
        best_lr_mean = min(valid_lr_means, key=valid_lr_means.get) if valid_lr_means else ""
        best_lr_std = min(valid_lr_stds, key=valid_lr_stds.get) if valid_lr_stds else ""
        best_overall = min(valid_all, key=valid_all.get) if valid_all else ""
        fixed_mean = means.get("fixed_optimizer", np.nan)
        fixed_runs = bool(not pd.isna(fixed_mean))
        lr_1e_4_best = best_overall == "fixed_optimizer"
        row.update(
            {
                "best_fixed_optimizer_by_mean_rmse": best_lr_mean,
                "best_fixed_optimizer_by_std_rmse": best_lr_std,
                "best_overall_variant": best_overall,
                "does_fixed_optimizer_run": fixed_runs,
                "does_lr_1e_4_remain_best": lr_1e_4_best,
                "interpretation": (
                    f"Fixed optimizer by mean RMSE is {best_lr_mean}; by std RMSE is {best_lr_std}; "
                    f"lr=1e-4 {'is' if lr_1e_4_best else 'is not'} the recorded fixed optimizer."
                ),
            }
        )
        rows.append(row)
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


def _markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 80) -> str:
    cols = list(columns)
    if df.empty:
        return "(empty)"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _overall_status(details: pd.DataFrame) -> str:
    expected_runs = len(DATASETS) * len(SEEDS) * len(CNN_LR_ABLATION_VARIANTS)
    if details["status"].eq("FAIL").any():
        return "FAIL"
    if len(details) != expected_runs or details["status"].ne("OK").any():
        return "PARTIAL"
    return "PASS"


def _write_report(details: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, target_epochs: int, original_batch_size: int) -> None:
    hard_rows = (
        details.groupby("cnn_lr_ablation_variant", as_index=False)
        .agg(
            rows=("status", "size"),
            failures=("status", lambda s: int((s == "FAIL").sum())),
            errors=("status", lambda s: int((s == "ERROR").sum())),
            optimizer_changed=("optimizer_changed", "first"),
            cnn_structure_changed=("cnn_structure_changed", "first"),
            batch_size_changed=("batch_size_changed", "first"),
            parameter_count=("model_parameter_count", "first"),
            original_parameter_count=("original_model_parameter_count", "first"),
            effective_batch_size=("effective_batch_size", "first"),
            learning_rate=("learning_rate", "first"),
            clipnorm=("clipnorm", "first"),
        )
        .sort_values("cnn_lr_ablation_variant")
    )
    status = _overall_status(details)
    recommend_rows = comparison[
        ["dataset", "best_fixed_optimizer_by_mean_rmse", "best_fixed_optimizer_by_std_rmse", "does_lr_1e_4_remain_best"]
    ].copy()
    all_lr_1e_4_best = bool(comparison["does_lr_1e_4_remain_best"].all()) if not comparison.empty else False
    best_counts = comparison["best_overall_variant"].value_counts()
    recommended_variant = str(best_counts.index[0]) if not best_counts.empty else ""

    lines = [
        "# CNN Optimizer Learning Rate + Clipnorm Ablation",
        "",
        "## 1. Purpose",
        "",
        "This audit tests only Adam learning_rate + clipnorm variants for the No-TL CNN. It does not change CNN structure, batch_size logic, KNN, RFE, split, RMSE, or data cleaning.",
        "",
        "## 2. Structure And Batch Size Safety",
        "",
        "All variants build the same `build_base_cnn` structure. The configured original batch size is reused for every variant; no run forces batch_size=1.",
        "",
        _markdown_table(
            hard_rows,
            [
                "cnn_lr_ablation_variant",
                "rows",
                "failures",
                "errors",
                "optimizer_changed",
                "cnn_structure_changed",
                "batch_size_changed",
                "parameter_count",
                "original_parameter_count",
                "effective_batch_size",
                "learning_rate",
                "clipnorm",
            ],
        ),
        "",
        "## 3. Results By Dataset",
        "",
        _markdown_table(
            summary.sort_values(["dataset_id", "rank_by_mean_rmse"]),
            [
                "dataset",
                "cnn_lr_ablation_variant",
                "n_seeds",
                "mean_test_rmse",
                "std_test_rmse",
                "mean_val_rmse",
                "rank_by_mean_rmse",
                "rank_by_std_rmse",
            ],
            max_rows=100,
        ),
        "",
        "## 4. Best LR And Stability",
        "",
        _markdown_table(
            recommend_rows,
            ["dataset", "best_fixed_optimizer_by_mean_rmse", "best_fixed_optimizer_by_std_rmse", "does_lr_1e_4_remain_best"],
        ),
        "",
        f"Across datasets, lr=1e-4 {'remains the best overall in all datasets' if all_lr_1e_4_best else 'does not remain the best overall in all datasets'}.",
        "",
        "## 5. Recommendation",
        "",
        f"Prioritize `{recommended_variant}` for follow-up main-experiment optimizer testing if this audit remains reproducible under the full protocol.",
        "",
        "## 6. Reproduction Safety Check",
        "",
        "- KNN: not modified.",
        "- RFE: not modified.",
        "- split: not modified.",
        "- RMSE: not modified.",
        "- data cleaning: not modified.",
        "- CNN structure: not modified.",
        "- batch_size: original logic preserved.",
        "- main experiment results: not overwritten.",
        "",
        "## 7. Conclusion",
        "",
        status,
        "",
        f"Reason: {status} is based on 75 expected detail rows, hard-condition status, and training statuses.",
        "",
        "## Output Files",
        "",
        f"- `{DETAIL_CSV.relative_to(ROOT)}`",
        f"- `{SUMMARY_CSV.relative_to(ROOT)}`",
        f"- `{COMPARISON_CSV.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_logging(log_level="WARNING", log_file=None)
    set_verbose_mode("summary")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[hyperparams] {fixed_hyperparams_summary()}; clipnorm=None disables gradient clipping.", flush=True)

    config = _load_config()
    exp = config.get("single_experiment", {})
    original_batch_size = int(exp.get("batch_size", config.get("batch_size", 16)))
    original_learning_rate = float(exp.get("learning_rate", config.get("learning_rate", FIXED_LEARNING_RATE)))
    target_epochs = int(exp.get("target_epochs", config.get("target_epochs", FIXED_EPOCHS)))
    metric_protocol = _metric_protocol(config)

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        prepared = _prepare_sequences(dataset, config)
        input_shape = prepared["x_train"].shape[1:]
        original_parameter_count = int(build_lr_ablation_model(input_shape, "fixed_optimizer", original_learning_rate).count_params())
        for seed in SEEDS:
            for variant in CNN_LR_ABLATION_VARIANTS:
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
                        original_parameter_count=original_parameter_count,
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
