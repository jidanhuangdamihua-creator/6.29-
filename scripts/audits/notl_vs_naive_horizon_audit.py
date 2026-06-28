"""Audit CNN No-TL against a naive persistence baseline across horizons.

This is an audit-only entry point. It does not modify the main experiment
pipeline and writes only under outputs/audits/.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

OUT_DIR = ROOT / "outputs" / "audits"
DETAILS_CSV = OUT_DIR / "notl_vs_naive_horizon_details.csv"
SUMMARY_CSV = OUT_DIR / "notl_vs_naive_horizon_summary.csv"
COMPARISON_CSV = OUT_DIR / "notl_vs_naive_horizon_comparison.csv"
REPORT_MD = OUT_DIR / "notl_vs_naive_horizon_audit.md"

MAIN_EXPERIMENT_SCRIPTS = [
    ROOT / "scripts" / "run_main_experiment.py",
    ROOT / "scripts" / "run_full_paper_experiments.py",
    ROOT / "src" / "experiment" / "run_no_tl_experiment.py",
]

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
HORIZONS = [1, 2, 3, 4, 5]
INFO_SHARING_VALUES = [True, False]
RANDOM_SEED = 42

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "info_sharing",
    "horizon",
    "method",
    "model_name",
    "rmse",
    "normalized_rmse",
    "original_scale_rmse",
    "metric_space",
    "window_size",
    "target_window_days",
    "n_test_samples",
    "random_seed",
    "batch_size",
    "target_epochs",
    "test_rmse",
    "test_mae",
    "prediction_shape",
    "status",
    "error_message",
    "run_time_seconds",
    "notes",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset",
    "info_sharing",
    "horizon",
    "naive_rmse_mean",
    "notl_rmse_mean",
    "naive_rmse_std",
    "notl_rmse_std",
    "naive_better_than_notl",
    "rmse_diff_notl_minus_naive",
    "rmse_pct_diff_vs_naive",
    "best_method",
    "n_runs",
]

COMPARISON_COLUMNS = [
    "dataset_id",
    "dataset",
    "info_sharing",
    "horizon",
    "naive_rmse",
    "notl_rmse",
    "winner",
    "notl_minus_naive",
    "notl_vs_naive_pct",
    "interpretation",
]


def _load_config() -> dict[str, Any]:
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _config_for_dataset(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    local = copy.deepcopy(cfg)
    local["dataset_name"] = dataset
    local.setdefault("single_experiment", {})["dataset_name"] = dataset
    return local


def _shape(value: Any) -> str:
    return str(tuple(np.asarray(value).shape))


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any, digits: int = 6) -> str:
    value = _safe_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def _target_window_days(target_df: pd.DataFrame, cfg: dict[str, Any]) -> int:
    attr_value = target_df.attrs.get("target_window_expected_days")
    if attr_value is not None:
        return int(attr_value)
    protocol = cfg.get("paper_reproduction", {}).get("paper_split_protocol", {})
    observed = int(protocol.get("target_observed_window_days", 30))
    forecast = int(protocol.get("target_forecast_window_days", 180))
    return observed + forecast


def naive_persistence_predict(
    x_window: np.ndarray,
    feature_columns: list[str] | tuple[str, ...],
    sales_feature: str = "sales",
) -> np.ndarray:
    """Predict y(t+h) with the last observed sales value y(t)."""
    columns = list(feature_columns)
    if sales_feature not in columns:
        raise ValueError(f"{sales_feature!r} is required in feature_columns for naive persistence.")
    sales_idx = columns.index(sales_feature)
    x_arr = np.asarray(x_window, dtype=np.float32)
    if x_arr.ndim != 3:
        raise ValueError(f"x_window must be 3D (samples, window, features), got shape={x_arr.shape}.")
    return x_arr[:, -1, sales_idx].reshape(-1, 1)


def _prepare_target_bundle(dataset: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from data_preprocessing import (
        build_source_target_split,
        extract_datetime_features,
        load_dataset,
        normalize_features,
        temporal_split_by_ratio_or_dates,
    )

    local_cfg = _config_for_dataset(cfg, dataset)
    data_path = ROOT / str(local_cfg["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, local_cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())
    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(
        train_df,
        val_df,
        test_df,
    )
    return {
        "dataset": dataset,
        "cfg": local_cfg,
        "target_df": target_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "target_window_days": _target_window_days(target_df, local_cfg),
    }


def _build_sequences(bundle: dict[str, Any], horizon: int, window_size: int) -> dict[str, Any]:
    from data_preprocessing import build_tabular_sequence, to_cnn_tensor

    x_train, y_train = build_tabular_sequence(bundle["train_scaled"], horizon=horizon, window_size=window_size)
    x_val, y_val = build_tabular_sequence(bundle["val_scaled"], horizon=horizon, window_size=window_size)
    x_test, y_test = build_tabular_sequence(bundle["test_scaled"], horizon=horizon, window_size=window_size)
    return {
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
    }


def _metric_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))


def _compute_metrics(bundle: dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from src.evaluation.metrics import compute_metrics_with_protocol

    return compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=_metric_protocol(bundle["cfg"]),
        sales_scaler=bundle["scaler"],
        feature_columns=bundle["feature_columns"],
    )


def _base_row(
    bundle: dict[str, Any],
    horizon: int,
    method: str,
    model_name: str,
    window_size: int,
    batch_size: int,
    target_epochs: int,
) -> dict[str, Any]:
    dataset = str(bundle["dataset"])
    return {
        "dataset_id": DATASET_ID[dataset],
        "dataset": dataset,
        "info_sharing": "",
        "horizon": int(horizon),
        "method": method,
        "model_name": model_name,
        "rmse": math.nan,
        "normalized_rmse": math.nan,
        "original_scale_rmse": math.nan,
        "metric_space": "",
        "window_size": int(window_size),
        "target_window_days": int(bundle["target_window_days"]),
        "n_test_samples": 0,
        "random_seed": RANDOM_SEED,
        "batch_size": int(batch_size),
        "target_epochs": int(target_epochs),
        "test_rmse": math.nan,
        "test_mae": math.nan,
        "prediction_shape": "",
        "status": "PENDING",
        "error_message": "",
        "run_time_seconds": 0.0,
        "notes": "",
    }


def _apply_metrics_to_row(row: dict[str, Any], metrics: dict[str, Any], y_pred: np.ndarray, elapsed: float) -> dict[str, Any]:
    row.update(
        {
            "rmse": float(metrics["rmse"]),
            "normalized_rmse": float(metrics["normalized_rmse"]),
            "original_scale_rmse": metrics.get("original_scale_rmse"),
            "metric_space": str(metrics["metric_space"]),
            "test_rmse": float(metrics["rmse"]),
            "test_mae": float(metrics["mae"]),
            "prediction_shape": _shape(y_pred),
            "status": "OK",
            "run_time_seconds": float(elapsed),
        }
    )
    return row


def run_dataset_horizon_pair(bundle: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    """Run CNN No-TL and Naive for one dataset/horizon before scenario cloning."""
    from src.models.cnn_model import resolve_cnn_ablation_training_config
    from src.models.no_tl_model import build_no_tl_cnn_model

    cfg = bundle["cfg"]
    exp_cfg = cfg["single_experiment"]
    window_size = int(exp_cfg["window_size"])
    target_epochs = int(exp_cfg["target_epochs"])
    batch_size = int(exp_cfg["batch_size"])
    learning_rate = float(exp_cfg.get("learning_rate", 1e-4))
    cnn_variant = str(exp_cfg.get("cnn_ablation_variant", "original"))
    training_cfg = resolve_cnn_ablation_training_config(
        cnn_ablation_variant=cnn_variant,
        original_batch_size=batch_size,
        original_learning_rate=learning_rate,
    )

    notl_row = _base_row(
        bundle,
        horizon,
        method="CNN No-TL",
        model_name=training_cfg.model_name,
        window_size=window_size,
        batch_size=training_cfg.effective_batch_size,
        target_epochs=target_epochs,
    )
    naive_row = _base_row(
        bundle,
        horizon,
        method="Naive persistence baseline",
        model_name="naive_persistence",
        window_size=window_size,
        batch_size=training_cfg.effective_batch_size,
        target_epochs=50,
    )

    try:
        seq = _build_sequences(bundle, horizon=horizon, window_size=window_size)
        for row in (notl_row, naive_row):
            row["n_test_samples"] = int(len(seq["y_test"]))
        if len(seq["y_train"]) == 0 or len(seq["y_test"]) == 0:
            message = (
                f"Insufficient windows: train={len(seq['y_train'])}, "
                f"test={len(seq['y_test'])}."
            )
            for row in (notl_row, naive_row):
                row["status"] = "SKIPPED"
                row["error_message"] = message
                row["notes"] = "No rows were dropped from outputs; skipped row is retained as evidence."
            return [notl_row, naive_row]
    except Exception as exc:
        for row in (notl_row, naive_row):
            row["status"] = "FAILED"
            row["error_message"] = f"sequence_build: {exc}"
        return [notl_row, naive_row]

    try:
        start = time.perf_counter()
        y_naive = naive_persistence_predict(seq["x_test"], bundle["feature_columns"])
        naive_metrics = _compute_metrics(bundle, seq["y_test"], y_naive)
        _apply_metrics_to_row(naive_row, naive_metrics, y_naive, time.perf_counter() - start)
        naive_row["notes"] = (
            "Naive baseline has no training process; y_pred(t+h)=last observed normalized sales y(t)."
        )
    except Exception as exc:
        naive_row["status"] = "FAILED"
        naive_row["error_message"] = f"naive_predict_or_metric: {exc}"

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED + DATASET_ID[bundle["dataset"]] * 100 + int(horizon))
        start = time.perf_counter()
        model = build_no_tl_cnn_model(
            input_shape=seq["x_train"].shape[1:],
            learning_rate=learning_rate,
            cnn_ablation_variant=cnn_variant,
        )
        fit_kwargs: dict[str, Any] = {
            "epochs": target_epochs,
            "batch_size": training_cfg.effective_batch_size,
            "verbose": 0,
        }
        if len(seq["y_val"]) > 0:
            fit_kwargs["validation_data"] = (seq["x_val"], seq["y_val"])
        model.fit(seq["x_train"], seq["y_train"], **fit_kwargs)
        y_pred = model.predict(seq["x_test"], verbose=0)
        metrics = _compute_metrics(bundle, seq["y_test"], y_pred)
        _apply_metrics_to_row(notl_row, metrics, y_pred, time.perf_counter() - start)
        notl_row["notes"] = (
            "Target-only CNN No-TL; no KNN, no RFE, no source pretraining, no TL model. "
            "The same No-TL result is reported for both information-sharing labels because No-TL does not use a source pool."
        )
    except Exception as exc:
        notl_row["status"] = "FAILED"
        notl_row["error_message"] = f"notl_train_or_metric: {exc}"

    return [notl_row, naive_row]


def _clone_for_scenarios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for info_sharing in INFO_SHARING_VALUES:
        for row in rows:
            out = dict(row)
            out["info_sharing"] = bool(info_sharing)
            if row["method"] == "Naive persistence baseline":
                out["notes"] = str(out["notes"]) + " Same test windows as CNN No-TL."
            cloned.append(out)
    return cloned


def build_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = details_df[details_df["status"].eq("OK")].copy()
    for dataset in DATASETS:
        for info_sharing in INFO_SHARING_VALUES:
            for horizon in HORIZONS:
                group = ok[
                    ok["dataset"].eq(dataset)
                    & ok["info_sharing"].eq(bool(info_sharing))
                    & ok["horizon"].eq(int(horizon))
                ]
                naive = group[group["method"].eq("Naive persistence baseline")]["normalized_rmse"].astype(float)
                notl = group[group["method"].eq("CNN No-TL")]["normalized_rmse"].astype(float)
                naive_mean = float(naive.mean()) if len(naive) else math.nan
                notl_mean = float(notl.mean()) if len(notl) else math.nan
                diff = notl_mean - naive_mean if not (math.isnan(naive_mean) or math.isnan(notl_mean)) else math.nan
                pct = diff / naive_mean * 100.0 if not math.isnan(diff) and not math.isclose(naive_mean, 0.0) else math.nan
                if math.isnan(diff):
                    best = "N/A"
                    naive_better = ""
                elif abs(diff) < 1e-6:
                    best = "Tie"
                    naive_better = False
                elif diff > 0:
                    best = "Naive persistence baseline"
                    naive_better = True
                else:
                    best = "CNN No-TL"
                    naive_better = False
                rows.append(
                    {
                        "dataset_id": DATASET_ID[dataset],
                        "dataset": dataset,
                        "info_sharing": bool(info_sharing),
                        "horizon": int(horizon),
                        "naive_rmse_mean": naive_mean,
                        "notl_rmse_mean": notl_mean,
                        "naive_rmse_std": float(naive.std(ddof=0)) if len(naive) else math.nan,
                        "notl_rmse_std": float(notl.std(ddof=0)) if len(notl) else math.nan,
                        "naive_better_than_notl": naive_better,
                        "rmse_diff_notl_minus_naive": diff,
                        "rmse_pct_diff_vs_naive": pct,
                        "best_method": best,
                        "n_runs": int(len(naive) + len(notl)),
                    }
                )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        naive_rmse = _safe_float(row["naive_rmse_mean"])
        notl_rmse = _safe_float(row["notl_rmse_mean"])
        diff = notl_rmse - naive_rmse if not (math.isnan(naive_rmse) or math.isnan(notl_rmse)) else math.nan
        pct = diff / naive_rmse * 100.0 if not math.isnan(diff) and not math.isclose(naive_rmse, 0.0) else math.nan
        if math.isnan(diff):
            winner = "N/A"
            interpretation = "N/A"
        elif abs(diff) < 1e-6:
            winner = "Tie"
            interpretation = "Tie"
        elif diff < 0:
            winner = "CNN No-TL"
            interpretation = "CNN No-TL better than Naive"
        else:
            winner = "Naive persistence baseline"
            interpretation = "Naive better than CNN No-TL"
        rows.append(
            {
                "dataset_id": int(row["dataset_id"]),
                "dataset": row["dataset"],
                "info_sharing": bool(row["info_sharing"]),
                "horizon": int(row["horizon"]),
                "naive_rmse": naive_rmse,
                "notl_rmse": notl_rmse,
                "winner": winner,
                "notl_minus_naive": diff,
                "notl_vs_naive_pct": pct,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    work = df[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in work.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _winner_counts(df: pd.DataFrame, horizons: list[int]) -> str:
    sub = df[df["horizon"].isin(horizons)].copy()
    if sub.empty:
        return "No valid comparison rows."
    counts = sub["winner"].value_counts().to_dict()
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def _dataset_conclusion(df: pd.DataFrame, dataset: str) -> str:
    sub = df[df["dataset"].eq(dataset)].copy()
    if sub.empty:
        return f"- {dataset}: no comparison rows."
    all_naive = int((sub["winner"] == "Naive persistence baseline").sum())
    all_notl = int((sub["winner"] == "CNN No-TL").sum())
    ties = int((sub["winner"] == "Tie").sum())
    return f"- {dataset}: CNN No-TL wins {all_notl}/10, Naive wins {all_naive}/10, ties {ties}/10."


def _pattern_flags(comparison_df: pd.DataFrame) -> tuple[bool, bool]:
    naive_only_h1_any = False
    naive_all_horizons_any = False
    for dataset in DATASETS:
        for info_sharing in INFO_SHARING_VALUES:
            sub = comparison_df[
                comparison_df["dataset"].eq(dataset)
                & comparison_df["info_sharing"].eq(bool(info_sharing))
            ].sort_values("horizon")
            if len(sub) != len(HORIZONS):
                continue
            winners = {int(r["horizon"]): str(r["winner"]) for _, r in sub.iterrows()}
            if winners.get(1) == "Naive persistence baseline" and all(
                winners.get(h) == "CNN No-TL" for h in [2, 3, 4, 5]
            ):
                naive_only_h1_any = True
            if all(winners.get(h) == "Naive persistence baseline" for h in HORIZONS):
                naive_all_horizons_any = True
    return naive_only_h1_any, naive_all_horizons_any


def build_report(details_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    table_df = comparison_df.rename(
        columns={
            "dataset": "Dataset",
            "info_sharing": "Info Sharing",
            "horizon": "Horizon",
            "naive_rmse": "Naive RMSE",
            "notl_rmse": "CNN No-TL RMSE",
            "winner": "Winner",
            "notl_minus_naive": "Difference",
            "notl_vs_naive_pct": "% Difference",
        }
    )
    h1_counts = _winner_counts(comparison_df, [1])
    h2_5_counts = _winner_counts(comparison_df, [2, 3, 4, 5])
    naive_only_h1_any, naive_all_horizons_any = _pattern_flags(comparison_df)
    notl_wins = int((comparison_df["winner"] == "CNN No-TL").sum())
    naive_wins = int((comparison_df["winner"] == "Naive persistence baseline").sum())

    if naive_all_horizons_any or naive_wins > notl_wins:
        explanation = (
            "Naive is better in many horizon comparisons. This indicates a structural risk for CNN No-TL "
            "under the small target-domain setting, and motivates checking CNN structure, learning rate, "
            "epochs, normalization, and training sample construction."
        )
    elif naive_only_h1_any:
        explanation = (
            "Naive only wins in a one-step pattern for at least one dataset/scenario. This supports the view "
            "that CNN No-TL keeps multi-step value and that one-step persistence advantage should not by itself reject CNN."
        )
    else:
        explanation = (
            "CNN No-TL wins most comparisons. The earlier naive single-step result is best interpreted as a short-term "
            "persistence advantage rather than broad evidence against CNN No-TL."
        )

    lines = [
        "# CNN No-TL vs Naive Persistence Horizon Audit",
        "",
        "## 1. Experiment Purpose",
        "",
        "This audit checks whether CNN No-TL only loses to Naive persistence at horizon=1, or whether the issue appears across horizons 1-5.",
        "",
        "## 2. Experiment Design",
        "",
        "- Dataset1 / Dataset2 / Dataset3",
        "- info_sharing=True / False",
        "- horizon=1,2,3,4,5",
        "- CNN No-TL",
        "- Naive persistence baseline",
        "",
        "## 3. Naive baseline Definition",
        "",
        "`y_pred(t+h) = y(t)`",
        "",
        "The baseline has no training process. It copies the last observed `sales` value in the input window and uses it as the prediction for the requested horizon.",
        "",
        "## 4. Fixed Conditions",
        "",
        "This audit does not modify data cleaning, split, scaler, RMSE formula, CNN structure, batch_size, KNN, RFE, or the main experiment CSV files. It writes only under `outputs/audits/`.",
        "",
        "Because No-TL does not use a source pool, the same target-only CNN and Naive results are reported under both information-sharing labels.",
        "",
        "## 5. Main Result Table",
        "",
        _markdown_table(
            table_df,
            [
                "Dataset",
                "Info Sharing",
                "Horizon",
                "Naive RMSE",
                "CNN No-TL RMSE",
                "Winner",
                "Difference",
                "% Difference",
            ],
        ),
        "",
        "## 6. Horizon Conclusions",
        "",
        f"- horizon=1 winner counts: {h1_counts}.",
        f"- horizon=2~5 winner counts: {h2_5_counts}.",
        f"- Naive only short-term advantage observed: {'Yes' if naive_only_h1_any else 'No'}.",
        f"- Naive wins all horizons for at least one dataset/scenario: {'Yes' if naive_all_horizons_any else 'No'}.",
        "",
        "## 7. Dataset Conclusions",
        "",
        _dataset_conclusion(comparison_df, "Dataset1"),
        _dataset_conclusion(comparison_df, "Dataset2"),
        _dataset_conclusion(comparison_df, "Dataset3"),
        "",
        "## 8. Interpretation for Current CNN No-TL",
        "",
        explanation,
        "",
        "## Output Files",
        "",
        f"- Details CSV: `{DETAILS_CSV.relative_to(ROOT)}`",
        f"- Summary CSV: `{SUMMARY_CSV.relative_to(ROOT)}`",
        f"- Comparison CSV: `{COMPARISON_CSV.relative_to(ROOT)}`",
        f"- Report MD: `{REPORT_MD.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = _load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        print(f"[audit] preparing dataset={dataset}")
        bundle = _prepare_target_bundle(dataset, cfg)
        for horizon in HORIZONS:
            print(f"[audit] dataset={dataset} horizon={horizon}")
            pair_rows = run_dataset_horizon_pair(bundle, horizon)
            detail_rows.extend(_clone_for_scenarios(pair_rows))

    details_df = pd.DataFrame(detail_rows)
    for col in DETAIL_COLUMNS:
        if col not in details_df.columns:
            details_df[col] = ""
    details_df = details_df[DETAIL_COLUMNS]

    summary_df = build_summary(details_df)
    comparison_df = build_comparison(summary_df)

    details_df.to_csv(DETAILS_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8")
    REPORT_MD.write_text(build_report(details_df, comparison_df), encoding="utf-8")

    print(f"[audit_saved] details={DETAILS_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] summary={SUMMARY_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] comparison={COMPARISON_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] report={REPORT_MD.relative_to(ROOT)}")
    return details_df, summary_df, comparison_df


if __name__ == "__main__":
    run_audit()
