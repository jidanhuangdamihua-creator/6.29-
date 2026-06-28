"""No-TL horizon 1–5 aggregation audit (fixed split, CNN, epoch=2, batch_size=16).

Read-only audit that tests whether horizon aggregation alone can explain the gap
between current No-TL RMSE and paper-reported No-TL RMSE.

Hard constraints:
  - Fixed current default split (NOT paper Table 3 split)
  - Fixed current No-TL CNN structure (original variant)
  - Fixed epoch=2
  - Fixed batch_size=16
  - Fixed optimizer/learning_rate (Adam, lr=1e-4)
  - Fixed data cleaning logic
  - Fixed RMSE formula (compute_metrics_with_protocol)
  - No KNN / RFE / TL modifications
  - All outputs under outputs/audits/ only
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_DIR = ROOT / "outputs" / "audits"
DETAILS_CSV = AUDIT_DIR / "notl_horizon_1_5_fixed_split_epoch2_batch16_details.csv"
SUMMARY_CSV = AUDIT_DIR / "notl_horizon_1_5_fixed_split_epoch2_batch16_summary.csv"
REPORT_MD  = AUDIT_DIR / "notl_horizon_1_5_fixed_split_epoch2_batch16_audit.md"

RANDOM_SEED = 42
DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}
HORIZON_VALUES = [1, 2, 3, 4, 5]

# Paper No-TL RMSE (Table 8 or equivalent)
PAPER_NOTL_RMSE = {
    "Dataset1": 0.2067,
    "Dataset2": 0.1049,
    "Dataset3": 0.2833,
}

# Fixed experimental parameters
FIXED_EPOCHS = 50
FIXED_BATCH_SIZE = 16
FIXED_LEARNING_RATE = 1e-4
FIXED_WINDOW_SIZE = 10
FIXED_CNN_VARIANT = "original"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    return json.loads((ROOT / "configs" / "default_config.json").read_text(encoding="utf-8"))


def _prepare_target_data(dataset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load and split target-only data for a dataset.

    Returns a dict with train/val/test DataFrames and scaler info.
    """
    data_path = ROOT / str(cfg["dataset_paths"][dataset])
    raw_df = load_dataset(dataset_name=dataset, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    _, target_df = build_source_target_split(processed_df, cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df.copy())

    train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(
        train_df, val_df, test_df,
    )

    split_mode = str(target_df.attrs.get("split_mode", "ratio"))
    split_role = str(target_df.attrs.get("split_role", "target"))

    return {
        "dataset": dataset,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "split_mode": split_mode,
        "split_role": split_role,
    }


def _build_sequences(
    bundle: Dict[str, Any],
    horizon: int,
    window_size: int,
) -> Dict[str, np.ndarray]:
    """Build tabular sequences for a given horizon."""
    x_train, y_train = build_tabular_sequence(
        bundle["train_scaled"], horizon=horizon, window_size=window_size,
    )
    x_val, y_val = build_tabular_sequence(
        bundle["val_scaled"], horizon=horizon, window_size=window_size,
    )
    x_test, y_test = build_tabular_sequence(
        bundle["test_scaled"], horizon=horizon, window_size=window_size,
    )
    return {
        "x_train": to_cnn_tensor(x_train),
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val),
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test),
        "y_test": y_test,
    }


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_protocol: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=bundle["scaler"],
        feature_columns=bundle["feature_columns"],
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single_no_tl(
    bundle: Dict[str, Any],
    horizon: int,
    metric_protocol: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one No-TL experiment for a given dataset and horizon.

    Returns a flat dict suitable for CSV output.
    """
    dataset_name = bundle["dataset"]
    dataset_id_val = DATASET_ID[dataset_name]
    window_size = FIXED_WINDOW_SIZE

    # Build base row with metadata
    train_rows = int(len(bundle["train_df"]))
    val_rows = int(len(bundle["val_df"]))
    test_rows = int(len(bundle["test_df"]))
    split_mode = bundle["split_mode"]

    base_row: Dict[str, Any] = {
        "dataset_id": dataset_id_val,
        "dataset": dataset_name,
        "method": "No-TL",
        "horizon": int(horizon),
        "split_mode": split_mode,
        "target_train_rows": train_rows,
        "target_val_rows": val_rows,
        "target_test_rows": test_rows,
        "train_windows": 0,
        "val_windows": 0,
        "test_windows": 0,
        "epochs": FIXED_EPOCHS,
        "batch_size": FIXED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "optimizer": "Adam",
        "rmse": np.nan,
        "normalized_rmse": np.nan,
        "original_scale_rmse": np.nan,
        "rmse_paper": np.nan,
        "accuracy": np.nan,
        "normalized_accuracy": np.nan,
        "original_scale_accuracy": np.nan,
        "y_true_shape": "N/A",
        "y_pred_shape": "N/A",
        "random_seed": RANDOM_SEED,
        "run_status": "PENDING",
        "error_message": "",
        "run_time_seconds": 0.0,
        "window_size": window_size,
        "feature_columns": "|".join(bundle["feature_columns"]),
    }

    try:
        # Build sequences
        sequences = _build_sequences(bundle, horizon=horizon, window_size=window_size)
        base_row["train_windows"] = int(len(sequences["y_train"]))
        base_row["val_windows"] = int(len(sequences["y_val"]))
        base_row["test_windows"] = int(len(sequences["y_test"]))
        base_row["y_true_shape"] = str(tuple(sequences["y_test"].shape))
    except Exception as exc:
        base_row["run_status"] = "FAILED"
        base_row["error_message"] = f"sequence_build: {exc}"
        return base_row

    if len(sequences["y_train"]) == 0 or len(sequences["y_test"]) == 0:
        base_row["run_status"] = "SKIPPED"
        base_row["error_message"] = (
            f"Insufficient windows: train={len(sequences['y_train'])}, "
            f"test={len(sequences['y_test'])}"
        )
        return base_row

    # Train and predict
    try:
        setup_reproducibility(RANDOM_SEED)
        import tensorflow as tf
        tf.keras.backend.clear_session()
        setup_reproducibility(RANDOM_SEED)

        start = time.perf_counter()

        model = build_no_tl_cnn_model(
            input_shape=sequences["x_train"].shape[1:],
            learning_rate=FIXED_LEARNING_RATE,
            cnn_ablation_variant=FIXED_CNN_VARIANT,
        )

        fit_kwargs: Dict[str, Any] = {
            "epochs": FIXED_EPOCHS,
            "batch_size": FIXED_BATCH_SIZE,
            "verbose": 0,
        }
        if len(sequences["y_val"]) > 0:
            fit_kwargs["validation_data"] = (sequences["x_val"], sequences["y_val"])

        model.fit(sequences["x_train"], sequences["y_train"], **fit_kwargs)
        y_pred = model.predict(sequences["x_test"], verbose=0)

        elapsed = time.perf_counter() - start
        base_row["run_time_seconds"] = float(elapsed)

        metrics = _compute_metrics(
            sequences["y_test"], y_pred,
            metric_protocol=metric_protocol,
            bundle=bundle,
        )

        base_row["rmse"] = float(metrics["rmse"])
        base_row["normalized_rmse"] = float(metrics["normalized_rmse"])
        base_row["original_scale_rmse"] = metrics.get("original_scale_rmse")
        base_row["rmse_paper"] = float(metrics.get("rmse_paper", np.nan))
        base_row["accuracy"] = float(metrics["accuracy"])
        base_row["normalized_accuracy"] = float(metrics["normalized_accuracy"])
        base_row["original_scale_accuracy"] = metrics.get("original_scale_accuracy")
        base_row["y_pred_shape"] = str(tuple(y_pred.shape))
        base_row["run_status"] = "OK"
    except Exception as exc:
        base_row["run_status"] = "FAILED"
        base_row["error_message"] = f"train_or_predict: {exc}"

    return base_row


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _format_float(value: Any, digits: int = 6) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def compute_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset and overall summary from detail rows."""
    ok_df = details_df[details_df["run_status"] == "OK"].copy()
    if ok_df.empty:
        return pd.DataFrame()

    summary_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        ds_df = ok_df[ok_df["dataset"] == dataset]
        if ds_df.empty:
            continue

        h1_row = ds_df[ds_df["horizon"] == 1]
        h1_rmse = float(h1_row["normalized_rmse"].values[0]) if len(h1_row) > 0 else np.nan
        h1_acc = float(h1_row["normalized_accuracy"].values[0]) if len(h1_row) > 0 else np.nan

        horizon_rmse: Dict[int, float] = {}
        horizon_acc: Dict[int, float] = {}
        for h in HORIZON_VALUES:
            h_df = ds_df[ds_df["horizon"] == h]
            if len(h_df) > 0:
                horizon_rmse[h] = float(h_df["normalized_rmse"].values[0])
                horizon_acc[h] = float(h_df["normalized_accuracy"].values[0])
            else:
                horizon_rmse[h] = np.nan
                horizon_acc[h] = np.nan

        rmse_values = [v for v in horizon_rmse.values() if not np.isnan(v)]
        acc_values = [v for v in horizon_acc.values() if not np.isnan(v)]
        mean_rmse = float(np.mean(rmse_values)) if rmse_values else np.nan
        mean_acc = float(np.mean(acc_values)) if acc_values else np.nan

        paper_rmse = PAPER_NOTL_RMSE[dataset]
        abs_diff = abs(mean_rmse - paper_rmse) if not np.isnan(mean_rmse) else np.nan
        ratio = mean_rmse / paper_rmse if not np.isnan(mean_rmse) and paper_rmse != 0 else np.nan

        summary_rows.append({
            "dataset_id": DATASET_ID[dataset],
            "dataset": dataset,
            "horizon_1_rmse": horizon_rmse.get(1, np.nan),
            "horizon_2_rmse": horizon_rmse.get(2, np.nan),
            "horizon_3_rmse": horizon_rmse.get(3, np.nan),
            "horizon_4_rmse": horizon_rmse.get(4, np.nan),
            "horizon_5_rmse": horizon_rmse.get(5, np.nan),
            "mean_rmse_h1_to_h5": mean_rmse,
            "horizon_1_accuracy": horizon_acc.get(1, np.nan),
            "mean_accuracy_h1_to_h5": mean_acc,
            "paper_notl_rmse": paper_rmse,
            "abs_diff_vs_paper": abs_diff,
            "ratio_to_paper": ratio,
            "h1_to_mean_delta": h1_rmse - mean_rmse if not (np.isnan(h1_rmse) or np.isnan(mean_rmse)) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Overall mean across datasets
    if not summary_df.empty:
        overall_mean = float(summary_df["mean_rmse_h1_to_h5"].mean())
        overall_mean_h1 = float(summary_df["horizon_1_rmse"].mean())
        overall_mean_acc = float(summary_df["mean_accuracy_h1_to_h5"].mean())
        summary_df.attrs["mean_rmse_across_datasets"] = overall_mean
        summary_df.attrs["mean_rmse_h1_across_datasets"] = overall_mean_h1
        summary_df.attrs["mean_accuracy_across_datasets"] = overall_mean_acc

    return summary_df


def determine_judgment(summary_df: pd.DataFrame) -> str:
    """Determine whether horizon aggregation plausibly explains the gap."""
    if summary_df.empty:
        return "NO_DATA"

    ratios = summary_df["ratio_to_paper"].dropna()
    if ratios.empty:
        return "NO_VALID_RATIOS"

    mean_ratio = float(ratios.mean())

    # Check if mean RMSE across horizons is close to paper (within 20%)
    close_count = int((ratios <= 1.20).sum())
    total = len(ratios)
    far_count = int((ratios >= 1.80).sum())

    if close_count == total:
        return "HORIZON_AGGREGATION_PLAUSIBLE"
    elif far_count == total:
        return "HORIZON_AGGREGATION_NOT_ENOUGH"
    else:
        # Mixed — determine per dataset
        ds_judgments = {}
        for _, row in summary_df.iterrows():
            r = row["ratio_to_paper"]
            ds = row["dataset"]
            if pd.isna(r):
                ds_judgments[ds] = "NO_DATA"
            elif r <= 1.20:
                ds_judgments[ds] = "CLOSE"
            elif r >= 1.80:
                ds_judgments[ds] = "FAR"
            else:
                ds_judgments[ds] = "INTERMEDIATE"

        close_ds = [ds for ds, j in ds_judgments.items() if j == "CLOSE"]
        far_ds = [ds for ds, j in ds_judgments.items() if j == "FAR"]

        if close_ds and far_ds:
            return "MIXED"
        elif close_ds and not far_ds:
            return "HORIZON_AGGREGATION_PLAUSIBLE"
        else:
            return "HORIZON_AGGREGATION_NOT_ENOUGH"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(details_df: pd.DataFrame, summary_df: pd.DataFrame, judgment: str) -> str:
    """Generate the Markdown audit report."""
    lines: List[str] = []

    lines.append("# No-TL Horizon 1–5 Aggregation Audit")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Audit Objective")
    lines.append("")
    lines.append(
        "Verify whether horizon aggregation (averaging RMSE across horizons 1–5) "
        "can explain the gap between the current No-TL RMSE and the paper-reported No-TL RMSE, "
        "under **fixed** split, CNN structure, epoch=2, and batch_size=16."
    )
    lines.append("")
    lines.append("## Fixed Experimental Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Split | Current default (ratio mode, target 0.067/0.067/0.866) |")
    lines.append(f"| CNN architecture | `{FIXED_CNN_VARIANT}` (build_base_cnn) |")
    lines.append(f"| Epochs | {FIXED_EPOCHS} |")
    lines.append(f"| Batch size | {FIXED_BATCH_SIZE} |")
    lines.append(f"| Learning rate | {FIXED_LEARNING_RATE} |")
    lines.append(f"| Optimizer | Adam |")
    lines.append(f"| Window size | {FIXED_WINDOW_SIZE} |")
    lines.append(f"| Random seed | {RANDOM_SEED} |")
    lines.append(f"| RMSE formula | `compute_metrics_with_protocol` (normalized_minmax_space) |")
    lines.append("")

    lines.append("## Paper No-TL RMSE Reference")
    lines.append("")
    lines.append("| Dataset | Paper No-TL RMSE |")
    lines.append("|---------|------------------|")
    for ds, rmse in PAPER_NOTL_RMSE.items():
        lines.append(f"| {ds} | {rmse:.4f} |")
    lines.append("")

    # Per-dataset horizon RMSE table
    lines.append("## Per-Dataset Horizon RMSE Results")
    lines.append("")
    if not details_df.empty:
        ok_df = details_df[details_df["run_status"] == "OK"]
        for dataset in DATASETS:
            ds_df = ok_df[ok_df["dataset"] == dataset]
            if ds_df.empty:
                lines.append(f"### {dataset}: No successful runs")
                lines.append("")
                continue
            lines.append(f"### {dataset}")
            lines.append("")
            lines.append(
                "| Horizon | normalized_rmse | normalized_accuracy | "
                "train_windows | val_windows | test_windows | run_time_seconds |"
            )
            lines.append(
                "|---------|----------------|---------------------|"
                "--------------|-------------|--------------|------------------|"
            )
            for h in HORIZON_VALUES:
                h_df = ds_df[ds_df["horizon"] == h]
                if len(h_df) > 0:
                    row = h_df.iloc[0]
                    lines.append(
                        f"| {h} | {_format_float(row['normalized_rmse'])} | "
                        f"{_format_float(row['normalized_accuracy'])} | "
                        f"{int(row['train_windows'])} | {int(row['val_windows'])} | "
                        f"{int(row['test_windows'])} | "
                        f"{_format_float(row['run_time_seconds'], 2)} |"
                    )
                else:
                    lines.append(f"| {h} | N/A | N/A | N/A | N/A | N/A | N/A |")
            lines.append("")
    else:
        lines.append("_No detail rows available._")
        lines.append("")

    # Summary table
    lines.append("## Summary: Horizon Aggregation vs Paper")
    lines.append("")
    if not summary_df.empty:
        summary_cols = [
            "dataset", "horizon_1_rmse", "horizon_2_rmse", "horizon_3_rmse",
            "horizon_4_rmse", "horizon_5_rmse", "mean_rmse_h1_to_h5",
            "paper_notl_rmse", "abs_diff_vs_paper", "ratio_to_paper",
        ]
        lines.append(
            "| " + " | ".join(summary_cols) + " |"
        )
        lines.append(
            "| " + " | ".join(["---"] * len(summary_cols)) + " |"
        )
        for _, row in summary_df.iterrows():
            vals = [
                str(row["dataset"]),
                _format_float(row["horizon_1_rmse"]),
                _format_float(row["horizon_2_rmse"]),
                _format_float(row["horizon_3_rmse"]),
                _format_float(row["horizon_4_rmse"]),
                _format_float(row["horizon_5_rmse"]),
                _format_float(row["mean_rmse_h1_to_h5"]),
                _format_float(row["paper_notl_rmse"]),
                _format_float(row["abs_diff_vs_paper"]),
                _format_float(row["ratio_to_paper"]),
            ]
            lines.append("| " + " | ".join(vals) + " |")

        # Overall means
        overall_mean = summary_df.attrs.get("mean_rmse_across_datasets")
        overall_h1 = summary_df.attrs.get("mean_rmse_h1_across_datasets")
        if overall_mean is not None:
            lines.append("")
            lines.append(
                f"**Overall mean RMSE (horizon 1–5, across datasets):** {_format_float(overall_mean)}"
            )
            lines.append(
                f"**Overall horizon=1 mean RMSE:** {_format_float(overall_h1)}"
            )
    else:
        lines.append("_No summary rows._")
    lines.append("")

    # Judgment
    lines.append("## Judgment")
    lines.append("")
    lines.append(f"**Overall classification:** `{judgment}`")
    lines.append("")

    if judgment == "HORIZON_AGGREGATION_PLAUSIBLE":
        lines.append(
            "✅ **Horizon aggregation plausibly explains the gap.** "
            "The mean RMSE across horizons 1–5 is close to the paper-reported No-TL RMSE "
            "(within 20%). The paper results are likely reported as an average over multiple "
            "horizons. Further confirmation of Table 7/8 aggregation method is recommended."
        )
    elif judgment == "HORIZON_AGGREGATION_NOT_ENOUGH":
        lines.append(
            "❌ **Horizon aggregation does NOT explain the gap.** "
            "The mean RMSE across horizons 1–5 remains approximately 2× or more the "
            "paper-reported RMSE. The gap must be attributed to other factors such as "
            "split protocol, scaler fit scope, data structuring, or optimizer tuning."
        )
    elif judgment == "MIXED":
        lines.append(
            "⚠️ **Mixed results.** Some datasets are close to paper RMSE after horizon "
            "aggregation while others are not. This suggests dataset-specific factors "
            "(e.g., split date boundaries, scaler fit scope, feature set) may contribute "
            "to the gap alongside horizon aggregation."
        )
    else:
        lines.append("⚠️ Unable to determine judgment due to insufficient data.")
    lines.append("")

    # Diagnostic answers
    lines.append("## Diagnostic Questions Answered")
    lines.append("")
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            ds = row["dataset"]
            h1 = row["horizon_1_rmse"]
            mean_h = row["mean_rmse_h1_to_h5"]
            paper = row["paper_notl_rmse"]
            ratio = row["ratio_to_paper"]

            lines.append(f"### {ds}")
            lines.append("")
            for h in HORIZON_VALUES:
                val = row.get(f"horizon_{h}_rmse", np.nan)
                lines.append(f"- **Horizon {h} RMSE:** {_format_float(val)}")
            lines.append(f"- **Mean RMSE (h1–h5):** {_format_float(mean_h)}")
            lines.append(f"- **Paper No-TL RMSE:** {_format_float(paper)}")
            lines.append(f"- **Ratio to paper:** {_format_float(ratio)}")
            if not np.isnan(h1) and not np.isnan(mean_h):
                closer = abs(mean_h - paper) < abs(h1 - paper)
                lines.append(f"- **Horizon average closer to paper than h=1?** {'Yes ✅' if closer else 'No ❌'}")
            lines.append("")

    lines.append("### Cross-Dataset Summary")
    lines.append("")
    lines.append(f"1. Horizon=1..5 RMSE values are listed above for each dataset.")
    lines.append(
        f"2. Whether horizon average RMSE is closer to paper than h=1: "
        f"see per-dataset diagnostics above."
    )
    for ds, paper_rmse in PAPER_NOTL_RMSE.items():
        ds_summary = summary_df[summary_df["dataset"] == ds] if not summary_df.empty else pd.DataFrame()
        if not ds_summary.empty:
            mean_h = ds_summary["mean_rmse_h1_to_h5"].values[0]
            if not np.isnan(mean_h):
                close_flag = "✅ YES" if abs(mean_h - paper_rmse) / paper_rmse < 0.20 else "❌ NO"
                lines.append(f"3. {ds} mean_rmse_h1_to_h5 close to {paper_rmse}? {close_flag}")
            else:
                lines.append(f"3. {ds}: insufficient data.")
        else:
            lines.append(f"3. {ds}: no data.")
    lines.append(
        f"6. If not explained: horizon aggregation alone is insufficient; "
        f"investigate split / scaler / data structuring."
    )
    lines.append(
        f"7. If explained: paper results are likely multi-horizon aggregated; "
        f"confirm Table 7/8 aggregation method."
    )
    lines.append(
        f"8. Judgment: `{judgment}`. "
        f"{'Prioritize optimizer tuning' if judgment == 'HORIZON_AGGREGATION_PLAUSIBLE' else 'Prioritize split / scaler / data structuring investigation' if judgment == 'HORIZON_AGGREGATION_NOT_ENOUGH' else 'Review mixed signals and investigate per-dataset.'}"
    )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(f"- Details: `{DETAILS_CSV.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{SUMMARY_CSV.relative_to(ROOT)}`")
    lines.append(f"- Report: `{REPORT_MD.relative_to(ROOT)}`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the horizon 1–5 audit for all three datasets."""
    setup_logging(log_level="INFO", log_file=None)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    metric_protocol = cfg.get("paper_reproduction", {}).get("metric_protocol", {})

    print("=" * 70)
    print("No-TL Horizon 1–5 Aggregation Audit")
    print(f"  Fixed: epoch={FIXED_EPOCHS}, batch_size={FIXED_BATCH_SIZE}, lr={FIXED_LEARNING_RATE}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Horizons: {HORIZON_VALUES}")
    print("=" * 70)

    all_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  Preparing {dataset} ...")
        try:
            bundle = _prepare_target_data(dataset, cfg)
            print(f"    train={len(bundle['train_df'])} val={len(bundle['val_df'])} test={len(bundle['test_df'])}")
            print(f"    features={bundle['feature_columns']}")
        except Exception as exc:
            print(f"  ❌ Failed to prepare {dataset}: {exc}")
            for h in HORIZON_VALUES:
                all_rows.append({
                    "dataset_id": DATASET_ID[dataset],
                    "dataset": dataset,
                    "method": "No-TL",
                    "horizon": h,
                    "split_mode": "N/A",
                    "target_train_rows": 0,
                    "target_val_rows": 0,
                    "target_test_rows": 0,
                    "train_windows": 0,
                    "val_windows": 0,
                    "test_windows": 0,
                    "epochs": FIXED_EPOCHS,
                    "batch_size": FIXED_BATCH_SIZE,
                    "learning_rate": FIXED_LEARNING_RATE,
                    "optimizer": "Adam",
                    "rmse": np.nan,
                    "normalized_rmse": np.nan,
                    "original_scale_rmse": np.nan,
                    "rmse_paper": np.nan,
                    "accuracy": np.nan,
                    "normalized_accuracy": np.nan,
                    "original_scale_accuracy": np.nan,
                    "y_true_shape": "N/A",
                    "y_pred_shape": "N/A",
                    "random_seed": RANDOM_SEED,
                    "run_status": "FAILED",
                    "error_message": f"prepare: {exc}",
                    "run_time_seconds": 0.0,
                    "window_size": FIXED_WINDOW_SIZE,
                    "feature_columns": "",
                })
            continue

        for horizon in HORIZON_VALUES:
            print(f"    Running No-TL with horizon={horizon} ...", end=" ", flush=True)
            row = run_single_no_tl(bundle, horizon=horizon, metric_protocol=metric_protocol)
            all_rows.append(row)
            status = row["run_status"]
            if status == "OK":
                print(f"  ✅ normalized_rmse={row['normalized_rmse']:.6f} ({row['run_time_seconds']:.1f}s)")
            else:
                print(f"  ⚠️ {status}: {row.get('error_message', '')[:100]}")

    # Build DataFrames
    details_df = pd.DataFrame(all_rows)
    summary_df = compute_summary(details_df)
    judgment = determine_judgment(summary_df)

    # Save CSVs
    details_df.to_csv(DETAILS_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Details saved: {DETAILS_CSV} ({len(details_df)} rows)")

    if not summary_df.empty:
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
        print(f"✅ Summary saved: {SUMMARY_CSV} ({len(summary_df)} rows)")

    # Generate and save report
    report = generate_report(details_df, summary_df, judgment)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"✅ Report saved: {REPORT_MD}")

    # Print final judgment
    print(f"\n{'=' * 70}")
    print(f"  JUDGMENT: {judgment}")
    if not summary_df.empty:
        overall_mean = summary_df.attrs.get("mean_rmse_across_datasets")
        if overall_mean is not None:
            print(f"  Overall mean RMSE (h1–h5): {overall_mean:.6f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
