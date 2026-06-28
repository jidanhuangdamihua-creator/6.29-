"""Paper-aligned split and horizon reproduction audit for No-TL only.

This script is intentionally audit-only. It does not modify the main experiment
pipeline and writes only:
  outputs/audits/notl_paper_aligned_split_horizon_reproduction.csv
  outputs/audits/notl_paper_aligned_split_horizon_reproduction.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

OUT_DIR = ROOT / "outputs" / "audits"
RESULT_CSV = OUT_DIR / "notl_paper_aligned_split_horizon_reproduction.csv"
REPORT_MD = OUT_DIR / "notl_paper_aligned_split_horizon_reproduction.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
HORIZONS = [1, 2, 3, 4, 5]
SEED = 42

PAPER_TABLE3_SPLITS = {
    "Dataset1": {"train": 15, "val": 15, "test": 185},
    "Dataset2": {"train": 14, "val": 15, "test": 179},
    "Dataset3": {"train": 16, "val": 15, "test": 181},
}
PAPER_DATASET_RMSE = {
    "Dataset1": 0.2067,
    "Dataset2": 0.1049,
    "Dataset3": 0.2833,
}
PAPER_TABLE8_MEAN_RMSE = 0.1983
PAPER_NOTL_ACCURACY = 4.83

RESULT_COLUMNS = [
    "row_type",
    "dataset",
    "horizon",
    "status",
    "normalized_rmse",
    "accuracy",
    "paper_dataset_rmse",
    "paper_table8_mean_rmse",
    "paper_accuracy",
    "abs_delta_vs_paper_dataset_rmse",
    "abs_delta_vs_table8_mean_rmse",
    "abs_delta_vs_paper_accuracy",
    "train_rows",
    "val_rows",
    "test_rows",
    "paper_train_rows",
    "paper_val_rows",
    "paper_test_rows",
    "train_rows_match_paper_table3",
    "val_rows_match_paper_table3",
    "test_rows_match_paper_table3",
    "all_rows_match_paper_table3",
    "train_windows",
    "val_windows",
    "test_windows",
    "x_train_shape",
    "x_val_shape",
    "x_test_shape",
    "y_true_shape",
    "y_pred_shape",
    "window_size",
    "target_epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "cnn_structure",
    "target_only",
    "knn_invoked",
    "rfe_invoked",
    "source_pretrained_model_loaded",
    "source_rows_used_for_training",
    "split_method",
    "target_start_date",
    "target_end_date",
    "train_start_date",
    "train_end_date",
    "val_start_date",
    "val_end_date",
    "test_start_date",
    "test_end_date",
    "feature_columns",
    "scaler_fit_scope",
    "current_main_single_horizon_rmse",
    "current_main_single_horizon_abs_delta_vs_paper",
    "audit_horizon1_rmse",
    "audit_horizon1_abs_delta_vs_paper",
    "horizon_mean_rmse",
    "horizon_mean_accuracy",
    "horizon_mean_abs_delta_vs_paper",
    "horizon_mean_closer_than_current_single_horizon",
    "horizon_mean_closer_than_audit_horizon1",
    "three_dataset_mean_rmse",
    "three_dataset_mean_accuracy",
    "three_dataset_mean_rmse_close_to_0_1983",
    "three_dataset_mean_accuracy_close_to_4_83",
    "valid_horizons",
    "invalid_horizons",
    "notes",
    "csv_evidence",
]


def _load_config() -> dict[str, Any]:
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _date_min(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d")


def _date_max(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")


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


def _base_row(dataset: str, cfg: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    paper = PAPER_TABLE3_SPLITS[dataset]
    train_df = split["train_df"]
    val_df = split["val_df"]
    test_df = split["test_df"]
    return {
        "dataset": dataset,
        "paper_dataset_rmse": PAPER_DATASET_RMSE[dataset],
        "paper_table8_mean_rmse": PAPER_TABLE8_MEAN_RMSE,
        "paper_accuracy": PAPER_NOTL_ACCURACY,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "paper_train_rows": paper["train"],
        "paper_val_rows": paper["val"],
        "paper_test_rows": paper["test"],
        "train_rows_match_paper_table3": len(train_df) == paper["train"],
        "val_rows_match_paper_table3": len(val_df) == paper["val"],
        "test_rows_match_paper_table3": len(test_df) == paper["test"],
        "all_rows_match_paper_table3": (
            len(train_df) == paper["train"]
            and len(val_df) == paper["val"]
            and len(test_df) == paper["test"]
        ),
        "window_size": int(cfg["single_experiment"]["window_size"]),
        "target_epochs": int(cfg["single_experiment"]["target_epochs"]),
        "batch_size": int(cfg["single_experiment"]["batch_size"]),
        "learning_rate": float(cfg["single_experiment"]["learning_rate"]),
        "optimizer": "Adam",
        "cnn_structure": "current build_no_tl_cnn_model -> build_base_cnn; Conv1D(32)-Pool-Conv1D(64)-Pool-Conv1D(128)-Flatten-Dense(1)",
        "target_only": True,
        "knn_invoked": False,
        "rfe_invoked": False,
        "source_pretrained_model_loaded": False,
        "source_rows_used_for_training": 0,
        "split_method": "audit-local paper Table 3 row-count split on selected target domain tail rows",
        "target_start_date": _date_min(split["target_df"]),
        "target_end_date": _date_max(split["target_df"]),
        "train_start_date": _date_min(train_df),
        "train_end_date": _date_max(train_df),
        "val_start_date": _date_min(val_df),
        "val_end_date": _date_max(val_df),
        "test_start_date": _date_min(test_df),
        "test_end_date": _date_max(test_df),
        "scaler_fit_scope": "target train+val only; target test transformed, not fit",
        "csv_evidence": "this row",
    }


def build_paper_aligned_target_split(dataset: str, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    from data_preprocessing import extract_datetime_features, load_dataset

    data_path = ROOT / cfg["dataset_paths"][dataset]
    raw = load_dataset(dataset, str(data_path))
    processed = extract_datetime_features(raw)
    target_df = select_target_domain(processed, dataset)

    paper = PAPER_TABLE3_SPLITS[dataset]
    total_rows = paper["train"] + paper["val"] + paper["test"]
    ordered = target_df.sort_values(["date", "entity_id", "item_id"]).tail(total_rows).reset_index(drop=True)
    if len(ordered) < total_rows:
        raise ValueError(f"{dataset} target has {len(ordered)} rows, cannot build Table 3 split requiring {total_rows}.")

    train_end = paper["train"]
    val_end = train_end + paper["val"]
    return {
        "target_df": ordered.copy(),
        "train_df": ordered.iloc[:train_end].copy(),
        "val_df": ordered.iloc[train_end:val_end].copy(),
        "test_df": ordered.iloc[val_end:].copy(),
    }


def select_target_domain(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    ordered = df.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
    if dataset == "Dataset1":
        mask = (pd.to_numeric(ordered["entity_id"], errors="coerce") == 1) & (
            pd.to_numeric(ordered["item_id"], errors="coerce") == 10
        )
    elif dataset == "Dataset2":
        mask = ordered["entity_id"].astype(str).eq("B1") & (
            pd.to_numeric(ordered["item_id"], errors="coerce") == 10
        )
    elif dataset == "Dataset3":
        mask = pd.to_numeric(ordered["item_id"], errors="coerce") == 10
    else:
        raise ValueError(f"Unsupported dataset={dataset!r}")
    target_df = ordered[mask].copy().reset_index(drop=True)
    if target_df.empty:
        raise ValueError(f"No target rows found for {dataset}.")
    return target_df


def _build_sequences(split: dict[str, pd.DataFrame], horizon: int, window_size: int) -> dict[str, Any]:
    from data_preprocessing import build_tabular_sequence, normalize_features, to_cnn_tensor

    train_s, val_s, test_s, scaler, feature_columns = normalize_features(
        split["train_df"],
        split["val_df"],
        split["test_df"],
    )
    x_train, y_train = build_tabular_sequence(train_s, horizon=horizon, window_size=window_size)
    x_val, y_val = build_tabular_sequence(val_s, horizon=horizon, window_size=window_size)
    x_test, y_test = build_tabular_sequence(test_s, horizon=horizon, window_size=window_size)
    return {
        "train_scaled": train_s,
        "val_scaled": val_s,
        "test_scaled": test_s,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "x_train": to_cnn_tensor(x_train) if len(y_train) else x_train,
        "y_train": y_train,
        "x_val": to_cnn_tensor(x_val) if len(y_val) else x_val,
        "y_val": y_val,
        "x_test": to_cnn_tensor(x_test) if len(y_test) else x_test,
        "y_test": y_test,
    }


def run_notl_horizon(dataset: str, cfg: dict[str, Any], split: dict[str, pd.DataFrame], horizon: int) -> dict[str, Any]:
    import tensorflow as tf

    from src.evaluation.metrics import compute_metrics_with_protocol
    from src.models.no_tl_model import build_no_tl_cnn_model
    from src.utils.runtime_control import keras_verbose

    window_size = int(cfg["single_experiment"]["window_size"])
    target_epochs = int(cfg["single_experiment"]["target_epochs"])
    batch_size = int(cfg["single_experiment"]["batch_size"])
    learning_rate = float(cfg["single_experiment"]["learning_rate"])
    seq = _build_sequences(split, horizon=horizon, window_size=window_size)

    row = _base_row(dataset, cfg, split)
    row.update(
        {
            "row_type": "horizon_result",
            "horizon": horizon,
            "train_windows": int(len(seq["y_train"])),
            "val_windows": int(len(seq["y_val"])),
            "test_windows": int(len(seq["y_test"])),
            "x_train_shape": _shape(seq["x_train"]),
            "x_val_shape": _shape(seq["x_val"]),
            "x_test_shape": _shape(seq["x_test"]),
            "y_true_shape": _shape(seq["y_test"]),
            "y_pred_shape": "",
            "feature_columns": "|".join(seq["feature_columns"]),
        }
    )

    if len(seq["y_train"]) == 0 or len(seq["y_test"]) == 0:
        row.update(
            {
                "status": "no_train_windows" if len(seq["y_train"]) == 0 else "no_test_windows",
                "notes": "No-TL training skipped because current split-local sliding-window construction produced empty windows.",
            }
        )
        return row

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + DATASETS.index(dataset) * 100 + horizon)
    model = build_no_tl_cnn_model(input_shape=seq["x_train"].shape[1:], learning_rate=learning_rate)
    fit_kwargs: dict[str, Any] = {"epochs": target_epochs, "batch_size": batch_size, "verbose": keras_verbose()}
    if len(seq["y_val"]) > 0:
        fit_kwargs["validation_data"] = (seq["x_val"], seq["y_val"])
    model.fit(seq["x_train"], seq["y_train"], **fit_kwargs)

    y_pred = model.predict(seq["x_test"], verbose=0)
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))
    metric_protocol["current_accuracy_definition"] = "1/RMSE"
    metrics = compute_metrics_with_protocol(
        y_true=seq["y_test"],
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=seq["scaler"],
        feature_columns=seq["feature_columns"],
    )
    rmse = float(metrics["normalized_rmse"])
    accuracy = float(1.0 / rmse) if rmse != 0 else math.inf
    row.update(
        {
            "status": "ok",
            "normalized_rmse": rmse,
            "accuracy": accuracy,
            "abs_delta_vs_paper_dataset_rmse": abs(rmse - PAPER_DATASET_RMSE[dataset]),
            "abs_delta_vs_table8_mean_rmse": abs(rmse - PAPER_TABLE8_MEAN_RMSE),
            "abs_delta_vs_paper_accuracy": abs(accuracy - PAPER_NOTL_ACCURACY),
            "y_pred_shape": _shape(y_pred),
            "notes": "No-TL trained on target train windows only; validation uses target val windows when non-empty.",
        }
    )
    return row


def load_current_main_single_horizon() -> dict[str, float]:
    path = ROOT / "outputs" / "experiment_results" / "paper_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    no_tl = df[
        df["method"].eq("No-TL")
        & df["information_sharing"].eq("without_information_sharing")
    ].copy()
    return {str(r["dataset"]): float(r["normalized_rmse"]) for _, r in no_tl.iterrows()}


def build_split_rows(cfg: dict[str, Any], splits: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, split in splits.items():
        row = _base_row(dataset, cfg, split)
        row.update(
            {
                "row_type": "split_summary",
                "horizon": "",
                "status": "ok" if row["all_rows_match_paper_table3"] else "row_count_mismatch",
                "notes": "Paper Table 3 row-count split constructed inside audit script.",
            }
        )
        rows.append(row)
    return rows


def build_dataset_summary_rows(detail: pd.DataFrame, current_main: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        group = detail[(detail["row_type"].eq("horizon_result")) & (detail["dataset"].eq(dataset))].copy()
        valid = group[group["status"].eq("ok")].copy()
        split_row = group.iloc[0].to_dict() if not group.empty else {}
        mean_rmse = float(valid["normalized_rmse"].mean()) if not valid.empty else math.nan
        mean_accuracy = float(valid["accuracy"].mean()) if not valid.empty else math.nan
        paper_rmse = PAPER_DATASET_RMSE[dataset]
        current_rmse = current_main.get(dataset, math.nan)
        audit_h1 = valid.loc[valid["horizon"].eq(1), "normalized_rmse"]
        audit_h1_rmse = float(audit_h1.iloc[0]) if not audit_h1.empty else math.nan
        current_delta = abs(current_rmse - paper_rmse) if not math.isnan(current_rmse) else math.nan
        h1_delta = abs(audit_h1_rmse - paper_rmse) if not math.isnan(audit_h1_rmse) else math.nan
        mean_delta = abs(mean_rmse - paper_rmse) if not math.isnan(mean_rmse) else math.nan
        row = {col: "" for col in RESULT_COLUMNS}
        for col in RESULT_COLUMNS:
            if col in split_row:
                row[col] = split_row[col]
        row.update(
            {
                "row_type": "dataset_summary",
                "dataset": dataset,
                "horizon": "1|2|3|4|5",
                "status": "ok" if not valid.empty else "no_valid_horizons",
                "current_main_single_horizon_rmse": current_rmse,
                "current_main_single_horizon_abs_delta_vs_paper": current_delta,
                "audit_horizon1_rmse": audit_h1_rmse,
                "audit_horizon1_abs_delta_vs_paper": h1_delta,
                "horizon_mean_rmse": mean_rmse,
                "horizon_mean_accuracy": mean_accuracy,
                "horizon_mean_abs_delta_vs_paper": mean_delta,
                "horizon_mean_closer_than_current_single_horizon": mean_delta < current_delta if not math.isnan(current_delta) and not math.isnan(mean_delta) else "",
                "horizon_mean_closer_than_audit_horizon1": mean_delta < h1_delta if not math.isnan(h1_delta) and not math.isnan(mean_delta) else "",
                "valid_horizons": "|".join(str(int(v)) for v in valid["horizon"].tolist()),
                "invalid_horizons": "|".join(str(int(v)) for v in group.loc[~group["status"].eq("ok"), "horizon"].tolist()),
                "notes": "Dataset arithmetic mean over valid horizon RMSE rows; invalid horizons are excluded and listed.",
                "csv_evidence": "horizon_result rows for this dataset",
            }
        )
        rows.append(row)
    return rows


def build_overall_summary_row(dataset_summaries: pd.DataFrame) -> dict[str, Any]:
    valid = dataset_summaries[dataset_summaries["row_type"].eq("dataset_summary")].copy()
    mean_rmse = float(valid["horizon_mean_rmse"].mean()) if not valid.empty else math.nan
    mean_accuracy = float(valid["horizon_mean_accuracy"].mean()) if not valid.empty else math.nan
    row = {col: "" for col in RESULT_COLUMNS}
    row.update(
        {
            "row_type": "overall_summary",
            "dataset": "ALL",
            "horizon": "dataset horizon means",
            "status": "ok" if not valid.empty else "no_dataset_summaries",
            "paper_table8_mean_rmse": PAPER_TABLE8_MEAN_RMSE,
            "paper_accuracy": PAPER_NOTL_ACCURACY,
            "three_dataset_mean_rmse": mean_rmse,
            "three_dataset_mean_accuracy": mean_accuracy,
            "abs_delta_vs_table8_mean_rmse": abs(mean_rmse - PAPER_TABLE8_MEAN_RMSE) if not math.isnan(mean_rmse) else math.nan,
            "abs_delta_vs_paper_accuracy": abs(mean_accuracy - PAPER_NOTL_ACCURACY) if not math.isnan(mean_accuracy) else math.nan,
            "three_dataset_mean_rmse_close_to_0_1983": abs(mean_rmse - PAPER_TABLE8_MEAN_RMSE) <= 0.05 if not math.isnan(mean_rmse) else "",
            "three_dataset_mean_accuracy_close_to_4_83": abs(mean_accuracy - PAPER_NOTL_ACCURACY) <= 0.5 if not math.isnan(mean_accuracy) else "",
            "notes": "Arithmetic mean of the three dataset horizon_mean_rmse and horizon_mean_accuracy values.",
            "csv_evidence": "dataset_summary rows",
        }
    )
    return row


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    work = df[[c for c in cols if c in df.columns]].copy()
    if work.empty:
        return "_No rows._"
    out = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in work.columns) + " |")
    return "\n".join(out)


def build_report(df: pd.DataFrame) -> str:
    split = df[df["row_type"].eq("split_summary")].copy()
    horizons = df[df["row_type"].eq("horizon_result")].copy()
    dataset_summary = df[df["row_type"].eq("dataset_summary")].copy()
    overall = df[df["row_type"].eq("overall_summary")].copy()
    overall_row = overall.iloc[0] if not overall.empty else {}

    all_test_match = bool(split["test_rows_match_paper_table3"].all()) if not split.empty else False
    current_closer_count = int(dataset_summary["horizon_mean_closer_than_current_single_horizon"].eq(True).sum()) if not dataset_summary.empty else 0
    h1_closer_count = int(dataset_summary["horizon_mean_closer_than_audit_horizon1"].eq(True).sum()) if not dataset_summary.empty else 0
    invalid = horizons[~horizons["status"].eq("ok")].copy()

    lines = [
        "# No-TL Paper-Aligned Split + Horizon Reproduction Audit",
        "",
        "Scope: No-TL only. The audit keeps the existing CNN structure, optimizer defaults, data cleaning functions, sliding-window function, and RMSE metric helper. It does not call KNN, RFE, transfer learning, source pretraining, or frozen source layers. Existing main result CSV files are not overwritten; outputs are written only under `outputs/audits/`.",
        "",
        f"- Output CSV evidence: `{RESULT_CSV.relative_to(ROOT)}`.",
        f"- Seed: `{SEED}` before each model fit.",
        "",
        "## Required Answers",
        "",
        f"1. Paper-aligned split test rows equal Table 3: {'YES' if all_test_match else 'NO'}. Evidence: CSV rows `row_type=split_summary`, columns `train_rows/val_rows/test_rows` and `*_match_paper_table3`.",
        "",
        "2. Horizon=1..5 RMSE values are listed below. Evidence: CSV rows `row_type=horizon_result`.",
        "",
        markdown_table(
            horizons,
            [
                "dataset",
                "horizon",
                "status",
                "normalized_rmse",
                "accuracy",
                "train_windows",
                "val_windows",
                "test_windows",
                "y_true_shape",
                "y_pred_shape",
            ],
        ),
        "",
        f"3. Horizon mean closer than current single-horizon main result: {current_closer_count}/3 datasets. Horizon mean closer than this audit's horizon=1 result: {h1_closer_count}/3 datasets. Evidence: CSV rows `row_type=dataset_summary`.",
        "",
        markdown_table(
            dataset_summary,
            [
                "dataset",
                "paper_dataset_rmse",
                "current_main_single_horizon_rmse",
                "current_main_single_horizon_abs_delta_vs_paper",
                "audit_horizon1_rmse",
                "audit_horizon1_abs_delta_vs_paper",
                "horizon_mean_rmse",
                "horizon_mean_abs_delta_vs_paper",
                "horizon_mean_closer_than_current_single_horizon",
                "horizon_mean_closer_than_audit_horizon1",
                "valid_horizons",
                "invalid_horizons",
            ],
        ),
        "",
        f"4. Three-dataset mean RMSE close to 0.1983: {overall_row.get('three_dataset_mean_rmse_close_to_0_1983', '')}. Observed mean RMSE={_fmt(overall_row.get('three_dataset_mean_rmse', math.nan))}, delta={_fmt(overall_row.get('abs_delta_vs_table8_mean_rmse', math.nan))}. Evidence: CSV row `row_type=overall_summary`.",
        "",
        f"5. accuracy=1/RMSE close to 4.83: {overall_row.get('three_dataset_mean_accuracy_close_to_4_83', '')}. Observed mean accuracy={_fmt(overall_row.get('three_dataset_mean_accuracy', math.nan))}, delta={_fmt(overall_row.get('abs_delta_vs_paper_accuracy', math.nan))}. Evidence: CSV row `row_type=overall_summary`.",
        "",
        "6. If the gap remains large, the remaining largest suspects are still: CNN training hyperparameters; MinMax scaler fit scope; data structuring / lag window; unpublished paper RMSE aggregation details; data version or target item/store mismatch. Evidence in this audit especially highlights data structuring / lag window when any requested horizon has `status != ok`.",
        "",
        "## Split Evidence",
        "",
        markdown_table(
            split,
            [
                "dataset",
                "train_rows",
                "val_rows",
                "test_rows",
                "paper_train_rows",
                "paper_val_rows",
                "paper_test_rows",
                "all_rows_match_paper_table3",
                "train_start_date",
                "train_end_date",
                "val_start_date",
                "val_end_date",
                "test_start_date",
                "test_end_date",
            ],
        ),
    ]
    if not invalid.empty:
        lines.extend(
            [
                "",
                "## Invalid Horizons",
                "",
                "These rows are retained as evidence instead of being dropped.",
                "",
                markdown_table(invalid, ["dataset", "horizon", "status", "train_windows", "test_windows", "notes"]),
            ]
        )
    return "\n".join(lines) + "\n"


def run_audit() -> pd.DataFrame:
    cfg = _load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_main = load_current_main_single_horizon()
    splits = {dataset: build_paper_aligned_target_split(dataset, cfg) for dataset in DATASETS}

    rows: list[dict[str, Any]] = []
    rows.extend(build_split_rows(cfg, splits))
    for dataset in DATASETS:
        for horizon in HORIZONS:
            print(f"[audit] dataset={dataset} horizon={horizon}")
            rows.append(run_notl_horizon(dataset, cfg, splits[dataset], horizon))

    detail = pd.DataFrame(rows)
    dataset_summary_rows = build_dataset_summary_rows(detail, current_main)
    detail = pd.concat([detail, pd.DataFrame(dataset_summary_rows)], ignore_index=True)
    overall_row = build_overall_summary_row(detail)
    detail = pd.concat([detail, pd.DataFrame([overall_row])], ignore_index=True)

    for col in RESULT_COLUMNS:
        if col not in detail.columns:
            detail[col] = ""
    detail = detail[RESULT_COLUMNS]
    detail.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    REPORT_MD.write_text(build_report(detail), encoding="utf-8")
    print(f"[audit_saved] csv={RESULT_CSV.relative_to(ROOT)}")
    print(f"[audit_saved] report={REPORT_MD.relative_to(ROOT)}")
    return detail


if __name__ == "__main__":
    run_audit()
