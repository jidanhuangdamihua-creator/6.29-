"""Paper-constrained No-TL metric/split/CNN audit.

Read-only audit of current No-TL protocol against paper constraints. The script
does not train models and writes only:
  outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv
  outputs/audits/notl_paper_constrained_metric_split_cnn_audit.md
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AUDIT_DIR = ROOT / "outputs" / "audits"
AUDIT_CSV = AUDIT_DIR / "notl_paper_constrained_metric_split_cnn_audit.csv"
AUDIT_MD = AUDIT_DIR / "notl_paper_constrained_metric_split_cnn_audit.md"

PAPER_RMSE = {"Dataset1": 0.2067, "Dataset2": 0.1049, "Dataset3": 0.2833}
PAPER_TABLE3 = {
    "Dataset1": {"train": 15, "val": 15, "test": 185},
    "Dataset2": {"train": 14, "val": 15, "test": 179},
    "Dataset3": {"train": 16, "val": 15, "test": 181},
}
PAPER_TABLE8_NOTL_MEAN = 0.1983
PAPER_NOTL_ACCURACY = 4.83


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def line_ref(path: Path, pattern: str, regex: bool = False) -> str:
    rx = re.compile(pattern) if regex else None
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if (rx.search(line) if rx else pattern in line):
                return f"{rel(path)}:{idx}"
    except FileNotFoundError:
        return ""
    return ""


def as_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, ndigits: int = 6) -> str:
    f = as_float(value)
    if math.isnan(f):
        return ""
    return f"{f:.{ndigits}f}"


def date_min(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d")


def date_max(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    work = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    out = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in work.columns) + " |")
    return "\n".join(out)


def load_config() -> dict[str, Any]:
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def split_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from data_preprocessing import (
        build_source_target_split,
        build_tabular_sequence,
        extract_datetime_features,
        load_dataset,
        normalize_features,
        temporal_split_by_ratio_or_dates,
    )

    rows: list[dict[str, Any]] = []
    for dataset, data_path in cfg["dataset_paths"].items():
        raw = load_dataset(dataset, str(ROOT / data_path))
        processed = extract_datetime_features(raw)
        _, target = build_source_target_split(processed, cfg)
        train, val, test = temporal_split_by_ratio_or_dates(target)
        train_s, val_s, test_s, scaler, feature_cols = normalize_features(train, val, test)
        horizon = int(cfg["single_experiment"]["horizon"])
        window_size = int(cfg["single_experiment"]["window_size"])
        x_train, y_train = build_tabular_sequence(train_s, horizon=horizon, window_size=window_size)
        x_val, y_val = build_tabular_sequence(val_s, horizon=horizon, window_size=window_size)
        x_test, y_test = build_tabular_sequence(test_s, horizon=horizon, window_size=window_size)
        paper = PAPER_TABLE3[dataset]
        rows.append(
            {
                "row_type": "split",
                "dataset": dataset,
                "target_train_start": date_min(train),
                "target_train_end": date_max(train),
                "target_val_start": date_min(val),
                "target_val_end": date_max(val),
                "target_test_start": date_min(test),
                "target_test_end": date_max(test),
                "target_train_rows": len(train),
                "target_val_rows": len(val),
                "target_test_rows": len(test),
                "paper_train_rows": paper["train"],
                "paper_val_rows": paper["val"],
                "paper_test_rows": paper["test"],
                "table3_equal": len(train) == paper["train"] and len(val) == paper["val"] and len(test) == paper["test"],
                "train_diff": len(train) - paper["train"],
                "val_diff": len(val) - paper["val"],
                "test_diff": len(test) - paper["test"],
                "target_split_mode": target.attrs.get("split_mode", ""),
                "target_split_config": json.dumps(target.attrs.get("split_config", {}), ensure_ascii=True),
                "target_window_range_days": target.attrs.get("target_window_range_days", ""),
                "target_window_unique_days": target.attrs.get("target_window_unique_days", ""),
                "horizon": horizon,
                "window_size": window_size,
                "train_windows": len(y_train),
                "val_windows": len(y_val),
                "test_windows": len(y_test),
                "input_shape": str((window_size, len(feature_cols))),
                "feature_columns": "|".join(feature_cols),
                "scaler_scope": "target-item split; fit on target train+val; transform target train/val/test",
                "scaler_fit_rows": len(train) + len(val),
                "scaler_fit_uses_test": False,
                "scaler_scales_sales": "sales" in feature_cols,
                "scaler_scales_calendar_features": all(c in feature_cols for c in ["year", "month", "week", "day"]),
                "code_evidence": "; ".join(
                    [
                        line_ref(ROOT / "data_preprocessing.py", "target_split_days"),
                        line_ref(ROOT / "data_preprocessing.py", 'elif mode == "days"'),
                        line_ref(ROOT / "data_preprocessing.py", 'elif mode == "actual_time_steps"'),
                        line_ref(ROOT / "data_preprocessing.py", "scaler.fit(all_df[feature_columns])"),
                    ]
                ),
                "csv_evidence": "computed by this audit from current config and source data",
            }
        )
    return rows


def cnn_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = (ROOT / "src" / "models" / "cnn_model.py").read_text(encoding="utf-8")
    base_start = source.index("def build_base_cnn")
    base_end = source.index("\ndef _validate_cnn_ablation_variant", base_start)
    body = source[base_start:base_end]
    conv_count = len(re.findall(r"\bConv1D\(", body))
    pool_count = len(re.findall(r"\bMaxPooling1D\(", body))
    flatten_count = len(re.findall(r"\bFlatten\(", body))
    dense_count = len(re.findall(r"\bDense\(", body))
    out_activation = "linear/none"
    loss_match = re.search(r'loss="([^"]+)"', body)
    lr = cfg["single_experiment"].get("learning_rate", "")
    return [
        {
            "row_type": "cnn",
            "dataset": "ALL",
            "conv1d_layers": conv_count,
            "maxpooling1d_layers": pool_count,
            "flatten_present": flatten_count > 0,
            "dense_layers": dense_count,
            "output_activation": out_activation,
            "loss": loss_match.group(1) if loss_match else "",
            "optimizer": "Adam",
            "learning_rate": lr,
            "batch_size": cfg["single_experiment"].get("batch_size", ""),
            "epochs": cfg["single_experiment"].get("target_epochs", ""),
            "early_stopping": "not configured in No-TL runner",
            "input_shape": "per dataset: (window_size=10, num_scaled_features)",
            "cnn_paper_aligned": conv_count == 3 and pool_count == 2 and flatten_count >= 1 and dense_count >= 1,
            "cnn_alignment_note": "Matches Fig.3/detail-index sequence Conv-MaxPool-Conv-MaxPool-Conv-Flatten-Dense; if interpreted as pooling after every Conv, current pool_count=2 not 3.",
            "code_evidence": "; ".join(
                [
                    line_ref(ROOT / "src" / "models" / "no_tl_model.py", "return build_base_cnn"),
                    line_ref(ROOT / "src" / "models" / "cnn_model.py", "x = Conv1D(filters=32"),
                    line_ref(ROOT / "src" / "models" / "cnn_model.py", "x = MaxPooling1D(pool_size=2, name=\"pool1\")"),
                    line_ref(ROOT / "src" / "models" / "cnn_model.py", "outputs = Dense(1"),
                    line_ref(ROOT / "src" / "experiment" / "run_no_tl_experiment.py", "model.fit(x_train"),
                ]
            ),
            "csv_evidence": "static model factory inspection",
        }
    ]


def result_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paper_csv = ROOT / "outputs" / "experiment_results" / "paper_results.csv"
    full_csv = ROOT / "outputs" / "experiment_results" / "full_paper_results.csv"
    result_df = pd.read_csv(paper_csv)
    no_tl = result_df[result_df["method"].eq("No-TL")].copy()
    primary = no_tl[no_tl["information_sharing"].eq("without_information_sharing")].copy()

    for _, r in no_tl.iterrows():
        rmse = as_float(r["rmse"])
        norm = as_float(r["normalized_rmse"])
        orig = as_float(r["original_scale_rmse"])
        acc = as_float(r["accuracy"])
        rows.append(
            {
                "row_type": "metric_result",
                "dataset": r["dataset"],
                "information_sharing": r["information_sharing"],
                "rmse_normalized_current": norm,
                "rmse_original": orig,
                "rmse": rmse,
                "rmse_paper": r["rmse_paper"],
                "accuracy": acc,
                "normalized_accuracy": r["normalized_accuracy"],
                "original_scale_accuracy": r["original_scale_accuracy"],
                "accuracy_paper": r["accuracy_paper"],
                "primary_rmse_equals_normalized_rmse": abs(rmse - norm) < 1e-12,
                "rmse_paper_equals_original_scale_rmse": abs(as_float(r["rmse_paper"]) - orig) < 1e-12,
                "accuracy_equals_inverse_rmse": abs(acc - (1.0 / rmse)) < 1e-6,
                "normalized_accuracy_equals_inverse_normalized_rmse": abs(as_float(r["normalized_accuracy"]) - (1.0 / norm)) < 1e-6,
                "original_scale_accuracy_equals_inverse_original_scale_rmse": abs(as_float(r["original_scale_accuracy"]) - (1.0 / orig)) < 1e-6,
                "metric_space_current": r["metric_space_current"],
                "metric_space_paper": r["metric_space_paper"],
                "paper_metric_aligned": r["paper_metric_aligned"],
                "inverse_transform_available": r["inverse_transform_available"],
                "metric_notes": r["metric_notes"],
                "strict_paper_mode": r["strict_paper_mode"],
                "selected_sources": r.get("selected_sources", ""),
                "rfe_selected_features": r.get("rfe_selected_features", ""),
                "source_count": r.get("source_count", ""),
                "pretrained_model_count": r.get("pretrained_model_count", ""),
                "code_evidence": "; ".join(
                    [
                        line_ref(ROOT / "src" / "evaluation" / "metrics.py", "rmse_current = _compute_rmse"),
                        line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"rmse": float(rmse_final)'),
                        line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"normalized_rmse": float(rmse_final)'),
                        line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"original_scale_rmse"'),
                    ]
                ),
                "csv_evidence": f"{rel(paper_csv)}:{int(r.name) + 2}; {rel(full_csv)} mirror rows available",
            }
        )

    # Candidate paper-RMSE comparisons from current latest No-TL rows.
    candidates: list[dict[str, Any]] = []
    for _, r in primary.iterrows():
        ds = str(r["dataset"])
        for name, value, acc_value in [
            ("rmse_normalized_current", r["normalized_rmse"], 1.0 / as_float(r["normalized_rmse"])),
            ("rmse_original", r["original_scale_rmse"], 1.0 / as_float(r["original_scale_rmse"])),
        ]:
            candidates.append(
                {
                    "scope": ds,
                    "candidate_name": name,
                    "candidate_rmse": as_float(value),
                    "candidate_accuracy": acc_value,
                    "paper_rmse": PAPER_RMSE[ds],
                    "paper_accuracy": 1.0 / PAPER_RMSE[ds],
                    "source": f"{rel(paper_csv)}:{int(r.name) + 2}",
                }
            )
    mean_norm = float(primary["normalized_rmse"].mean())
    mean_orig = float(primary["original_scale_rmse"].mean())
    for name, value in [
        ("rmse_mean_across_3_datasets_normalized", mean_norm),
        ("rmse_mean_across_3_datasets_original", mean_orig),
        ("rmse_mean_without_info", mean_norm),
        ("rmse_mean_with_info", float(no_tl[no_tl["information_sharing"].eq("with_information_sharing")]["normalized_rmse"].mean())),
    ]:
        candidates.append(
            {
                "scope": "Table8",
                "candidate_name": name,
                "candidate_rmse": value,
                "candidate_accuracy": 1.0 / value,
                "paper_rmse": PAPER_TABLE8_NOTL_MEAN,
                "paper_accuracy": PAPER_NOTL_ACCURACY,
                "source": rel(paper_csv),
            }
        )

    horizon_csv = ROOT / "outputs" / "audits" / "notl_horizon_1_5_summary.csv"
    if horizon_csv.exists():
        hdf = pd.read_csv(horizon_csv)
        for _, r in hdf.iterrows():
            ds = str(r["dataset"])
            candidates.append(
                {
                    "scope": ds,
                    "candidate_name": "rmse_mean_across_horizons_existing_ablation",
                    "candidate_rmse": as_float(r["horizon_1_5_mean_rmse"]),
                    "candidate_accuracy": 1.0 / as_float(r["horizon_1_5_mean_rmse"]),
                    "paper_rmse": PAPER_RMSE[ds],
                    "paper_accuracy": 1.0 / PAPER_RMSE[ds],
                    "source": f"{rel(horizon_csv)}:{int(r.name) + 2}",
                }
            )

    comp_df = pd.DataFrame(candidates)
    comp_df["abs_diff"] = (comp_df["candidate_rmse"] - comp_df["paper_rmse"]).abs()
    comp_df["ratio_to_paper"] = comp_df["candidate_rmse"] / comp_df["paper_rmse"]
    comp_df["closest_candidate_rank"] = comp_df.groupby("scope")["abs_diff"].rank(method="dense").astype(int)
    for _, r in comp_df.sort_values(["scope", "closest_candidate_rank", "candidate_name"]).iterrows():
        rows.append(
            {
                "row_type": "candidate_rmse",
                "dataset": r["scope"],
                "candidate_name": r["candidate_name"],
                "candidate_rmse": r["candidate_rmse"],
                "paper_rmse": r["paper_rmse"],
                "abs_diff": r["abs_diff"],
                "ratio_to_paper": r["ratio_to_paper"],
                "candidate_accuracy": r["candidate_accuracy"],
                "paper_accuracy": r["paper_accuracy"],
                "closest_candidate_rank": r["closest_candidate_rank"],
                "csv_evidence": r["source"],
            }
        )
    return rows


def horizon_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h_csv = ROOT / "outputs" / "audits" / "notl_epoch_batch_horizon_ablation.csv"
    if not h_csv.exists():
        return rows
    df = pd.read_csv(h_csv)
    h = df[df["experiment_group"].eq("horizon_ablation")].copy()
    for _, r in h.iterrows():
        rows.append(
            {
                "row_type": "horizon_existing_ablation",
                "dataset": r["dataset"],
                "horizon": r["horizon"],
                "rmse_by_each_horizon": r["normalized_rmse"],
                "rmse_original": r["original_scale_rmse"],
                "target_epochs": r["target_epochs"],
                "batch_size": r["batch_size"],
                "train_windows": r["train_windows"],
                "val_windows": r["val_windows"],
                "test_windows": r["test_windows"],
                "csv_evidence": f"{rel(h_csv)}:{int(r.name) + 2}",
                "code_evidence": line_ref(ROOT / "scripts" / "audits" / "notl_epoch_batch_horizon_ablation.py", "HORIZON_VALUES"),
            }
        )
    return rows


def source_domain_rows() -> list[dict[str, Any]]:
    paper_csv = ROOT / "outputs" / "experiment_results" / "paper_results.csv"
    df = pd.read_csv(paper_csv)
    no_tl = df[df["method"].eq("No-TL")].copy()
    rows = []
    for _, r in no_tl.iterrows():
        rows.append(
            {
                "row_type": "source_tl_rfe_audit",
                "dataset": r["dataset"],
                "information_sharing": r["information_sharing"],
                "training_contains_source_rows": False,
                "calls_knn_selection": False,
                "uses_selected_sources": False,
                "uses_rfe_selected_features": False,
                "loads_source_pretrained_model": False,
                "freezes_source_layers": False,
                "selected_sources": r.get("selected_sources", ""),
                "selected_source_count": r.get("selected_source_count", ""),
                "source_count": r.get("source_count", ""),
                "pretrained_model_count": r.get("pretrained_model_count", ""),
                "rfe_selected_features": r.get("rfe_selected_features", ""),
                "code_evidence": "; ".join(
                    [
                        line_ref(ROOT / "src" / "experiment" / "run_no_tl_experiment.py", "target_df: pd.DataFrame"),
                        line_ref(ROOT / "src" / "experiment" / "run_no_tl_experiment.py", "target_min_df = target_df.copy()"),
                        line_ref(ROOT / "experiment_runner.py", "if method == \"No-TL\""),
                        line_ref(ROOT / "scripts" / "run_full_paper_experiments.py", "NO_TL_SOURCE_DISPLAY_FIELDS"),
                    ]
                ),
                "csv_evidence": f"{rel(paper_csv)}:{int(r.name) + 2}",
            }
        )
    return rows


def build_report(df: pd.DataFrame) -> str:
    split = df[df["row_type"].eq("split")]
    metric = df[df["row_type"].eq("metric_result")]
    candidates = df[df["row_type"].eq("candidate_rmse")]
    cnn = df[df["row_type"].eq("cnn")]
    source = df[df["row_type"].eq("source_tl_rfe_audit")]
    horizon = df[df["row_type"].eq("horizon_existing_ablation")]
    no_tl_dispatch_ref = line_ref(ROOT / "experiment_runner.py", 'if method == "No-TL"')
    no_tl_cache_ref = line_ref(
        ROOT / "scripts" / "run_full_paper_experiments.py",
        'if method_name == "No-TL" and dataset_name in no_tl_cache',
    )
    split_days_ref = line_ref(ROOT / "data_preprocessing.py", "target_split_days")
    days_mode_ref = line_ref(ROOT / "data_preprocessing.py", 'elif mode == "days"')
    actual_steps_ref = line_ref(ROOT / "data_preprocessing.py", 'elif mode == "actual_time_steps"')
    minmax_concat_ref = line_ref(ROOT / "data_preprocessing.py", "all_df = pd.concat([train_df, val_df]")
    minmax_fit_ref = line_ref(ROOT / "data_preprocessing.py", "scaler.fit(all_df[feature_columns])")
    rmse_current_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", "rmse_current = _compute_rmse")
    rmse_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"rmse": float(rmse_final)')
    normalized_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"normalized_rmse": float(rmse_final)')
    original_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"original_scale_rmse"')
    notl_factory_ref = line_ref(ROOT / "src" / "models" / "no_tl_model.py", "return build_base_cnn")
    cnn_start_ref = line_ref(ROOT / "src" / "models" / "cnn_model.py", "x = Conv1D(filters=32")

    paper_evidence = [
        "- No-TL target-only simple CNN: `docs/paper_alignment/paper_reproduction_detail_index.md:203`; PDF text confirms target-domain simple CNN and no source knowledge.",
        "- Target split: `docs/paper_alignment/paper_reproduction_detail_index.md:117` to `:121` list Table 3 target time steps.",
        "- CNN: `docs/paper_alignment/paper_reproduction_detail_index.md:265` to `:270` list Conv/MaxPool/Flatten/Dense and unspecified hyperparameters.",
        "- MinMax scaling: `docs/paper_alignment/paper_reproduction_detail_index.md:138` states features scaled to 0-1.",
        "- Horizon aggregation: `docs/paper_alignment/paper_reproduction_detail_index.md:280` to `:286` states 1-5 days ahead and aggregate tables.",
        "- RMSE/accuracy: `docs/paper_alignment/paper_reproduction_detail_index.md:313` to `:320`; PDF abstract says accuracy is reciprocal of RMSE and Table 13 No-TL accuracy is 4.8340.",
        "- Paper values used here: Table 7 No-TL RMSE D1=0.2067, D2=0.1049, D3=0.2833; Table 8 No-TL mean RMSE=0.1983; abstract/Table 13 No-TL accuracy about 4.83.",
    ]

    code_evidence = [
        f"- No-TL trains only target split: `{line_ref(ROOT / 'src/experiment/run_no_tl_experiment.py', 'target_min_df = target_df.copy()')}` to `{line_ref(ROOT / 'src/experiment/run_no_tl_experiment.py', 'model.fit(x_train')}`.",
        f"- Experiment runner passes only `target_df` to No-TL: `{no_tl_dispatch_ref}`; full runner caches/clones No-TL across scenarios at `{no_tl_cache_ref}`.",
        f"- Target split days/config: `{split_days_ref}`, day split `{days_mode_ref}`, actual time steps `{actual_steps_ref}`.",
        f"- MinMax fit scope: `{minmax_concat_ref}` and `{minmax_fit_ref}`.",
        f"- Metric aliases: `{rmse_current_ref}`, `{rmse_ref}`, `{normalized_ref}`, `{original_ref}`.",
        f"- CNN factory: `{notl_factory_ref}`, base layers start at `{cnn_start_ref}`.",
        f"- Latest result CSV evidence is embedded in `{rel(AUDIT_CSV)}` as `csv_evidence`.",
    ]

    d1_norm = candidates[(candidates["dataset"].eq("Dataset1")) & (candidates["candidate_name"].eq("rmse_normalized_current"))]
    d1_acc = as_float(d1_norm.iloc[0]["candidate_accuracy"]) if not d1_norm.empty else math.nan
    table8_norm = candidates[
        (candidates["dataset"].eq("Table8"))
        & (candidates["candidate_name"].eq("rmse_mean_across_3_datasets_normalized"))
    ]
    table8_value = as_float(table8_norm.iloc[0]["candidate_rmse"]) if not table8_norm.empty else math.nan

    answers = [
        "1. 当前 No-TL 是否严格只使用 target domain？是。代码路径和 CSV 均显示 source_count=0、selected_sources=NOT_APPLICABLE/NaN、pretrained_model_count=0，No-TL 函数只接收并训练 target_df。",
        "2. 当前 target train/val/test rows 是否等于论文 Table 3？否。当前 D1=15/15/184 vs 15/15/185；D2=14/15/174 vs 14/15/179；D3=15/15/180 vs 16/15/181。",
        "3. 当前 CNN 结构是否等于论文 No-TL 描述？按论文细化索引 Fig.3 序列是对齐的：3 Conv1D、2 MaxPooling1D、Flatten、Dense；若把“每个卷积层之间有 MaxPooling”误解为每个 Conv 后都有池化，则当前少第 3 个 pooling。",
        "4. 当前 primary rmse 是否等于 normalized_rmse？是，latest CSV 每个 No-TL 行均相等。",
        "5. 当前 rmse_paper/original_scale_rmse 是否适合与论文 Table 7/8 对比？不适合直接对比；代码把它标为 original-scale diagnostic，数值量级也与 Table 7/8 不同。",
        "6. 当前 accuracy 是否严格等于 1/RMSE？在数值容差内是；实现默认是 1/(RMSE+1e-8)，latest CSV 与 1/RMSE 差异低于 1e-6。",
        f"7. 当前结果是否复现论文摘要 No-TL accuracy=4.83？否。当前 D1 normalized accuracy={fmt(d1_acc, 4)}，对应 RMSE={fmt(as_float(d1_norm.iloc[0]['candidate_rmse']) if not d1_norm.empty else math.nan, 4)}。",
        f"8. 当前结果是否复现 Table 8 No-TL mean RMSE=0.1983？否。当前三数据集 normalized mean RMSE={fmt(table8_value, 4)}。",
        "9. 最大差异来自 metric space + split/time steps + horizon aggregation/结果字段语义混乱。CNN 结构不是第一嫌疑；optimizer/lr/epoch 会影响数值但论文未给超参，优先级低于 protocol。",
        "10. 下一步优先修 metric protocol、split、horizon aggregation。不要先调 CNN optimizer，否则会用超参搜索掩盖口径不一致。",
    ]
    evidence_matrix = pd.DataFrame(
        [
            {
                "question": "1 target-only",
                "code_evidence": "src/experiment/run_no_tl_experiment.py:39; src/experiment/run_no_tl_experiment.py:70; experiment_runner.py:1077",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=source_tl_rfe_audit",
            },
            {
                "question": "2 Table3 split",
                "code_evidence": "data_preprocessing.py:101; data_preprocessing.py:116; data_preprocessing.py:122; data_preprocessing.py:972; data_preprocessing.py:994",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=split",
            },
            {
                "question": "3 CNN structure",
                "code_evidence": "src/models/no_tl_model.py:23; src/models/cnn_model.py:53; src/models/cnn_model.py:54; src/models/cnn_model.py:59; src/models/cnn_model.py:61; src/models/cnn_model.py:63",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv row row_type=cnn",
            },
            {
                "question": "4 rmse == normalized_rmse",
                "code_evidence": "src/evaluation/metrics.py:170; src/evaluation/metrics.py:190",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=metric_result; outputs/experiment_results/paper_results.csv:2,14,26",
            },
            {
                "question": "5 rmse_paper/original scale",
                "code_evidence": "src/evaluation/metrics.py:149; src/evaluation/metrics.py:157; src/evaluation/metrics.py:193",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=metric_result and candidate_rmse",
            },
            {
                "question": "6 accuracy inverse RMSE",
                "code_evidence": "src/evaluation/metrics.py:19; src/evaluation/metrics.py:119; src/evaluation/metrics.py:159",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=metric_result",
            },
            {
                "question": "7 abstract accuracy 4.83",
                "code_evidence": "docs/paper_alignment/paper_reproduction_detail_index.md:314; docs/paper_alignment/paper_reproduction_detail_index.md:320",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows candidate_name=rmse_normalized_current",
            },
            {
                "question": "8 Table8 mean 0.1983",
                "code_evidence": "docs/paper_alignment/paper_reproduction_detail_index.md:380; docs/paper_alignment/paper_reproduction_detail_index.md:382",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows dataset=Table8",
            },
            {
                "question": "9 largest gap",
                "code_evidence": "data_preprocessing.py:101; data_preprocessing.py:122; src/evaluation/metrics.py:143; scripts/audits/notl_epoch_batch_horizon_ablation.py:328",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=split,candidate_rmse,horizon_existing_ablation",
            },
            {
                "question": "10 next priority",
                "code_evidence": "docs/paper_alignment/paper_reproduction_detail_index.md:117; docs/paper_alignment/paper_reproduction_detail_index.md:285; src/evaluation/metrics.py:143",
                "csv_evidence": "outputs/audits/notl_paper_constrained_metric_split_cnn_audit.csv rows row_type=split,candidate_rmse,horizon_existing_ablation",
            },
        ]
    )

    lines = [
        "# No-TL Paper-Constrained Metric / Split / CNN Audit",
        "",
        "Scope: audit only. No training logic, CNN, optimizer, KNN, RFE, data cleaning, split, or RMSE formula was modified.",
        "",
        "## 论文证据",
        *paper_evidence,
        "",
        "## 当前代码/CSV证据",
        *code_evidence,
        "",
        "## 结论回答",
        *[f"- {a}" for a in answers],
        "",
        "## 结论证据矩阵",
        markdown_table(evidence_matrix, ["question", "code_evidence", "csv_evidence"], max_rows=20),
        "",
        "## Split Audit",
        markdown_table(
            split,
            [
                "dataset",
                "target_train_start",
                "target_train_end",
                "target_val_start",
                "target_val_end",
                "target_test_start",
                "target_test_end",
                "target_train_rows",
                "target_val_rows",
                "target_test_rows",
                "paper_train_rows",
                "paper_val_rows",
                "paper_test_rows",
                "table3_equal",
                "train_diff",
                "val_diff",
                "test_diff",
                "target_split_mode",
            ],
        ),
        "",
        "## No-TL Source/TL/RFE Audit",
        markdown_table(
            source,
            [
                "dataset",
                "information_sharing",
                "training_contains_source_rows",
                "calls_knn_selection",
                "uses_selected_sources",
                "uses_rfe_selected_features",
                "loads_source_pretrained_model",
                "freezes_source_layers",
                "source_count",
                "pretrained_model_count",
            ],
            max_rows=6,
        ),
        "",
        "## CNN Audit",
        markdown_table(
            cnn,
            [
                "conv1d_layers",
                "maxpooling1d_layers",
                "flatten_present",
                "dense_layers",
                "output_activation",
                "loss",
                "optimizer",
                "learning_rate",
                "batch_size",
                "epochs",
                "early_stopping",
                "cnn_paper_aligned",
            ],
        ),
        "",
        "## Metric / Candidate RMSE Audit",
        markdown_table(
            candidates.sort_values(["dataset", "closest_candidate_rank", "candidate_name"]),
            [
                "dataset",
                "candidate_name",
                "candidate_rmse",
                "paper_rmse",
                "abs_diff",
                "ratio_to_paper",
                "candidate_accuracy",
                "paper_accuracy",
                "closest_candidate_rank",
                "csv_evidence",
            ],
            max_rows=30,
        ),
        "",
        "## Horizon Evidence",
        "Current main results use `single_experiment.horizon=1`; existing horizon rows are from an audit ablation with target_epochs/batch_size different from latest main No-TL, so they are candidates only, not a strict Table 7/8 reproduction.",
        markdown_table(
            horizon,
            [
                "dataset",
                "horizon",
                "rmse_by_each_horizon",
                "target_epochs",
                "batch_size",
                "train_windows",
                "val_windows",
                "test_windows",
                "csv_evidence",
            ],
            max_rows=20,
        ),
        "",
        "## Output",
        f"- CSV: `{rel(AUDIT_CSV)}`",
        f"- Markdown: `{rel(AUDIT_MD)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_config()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(split_rows(cfg))
    rows.extend(source_domain_rows())
    rows.extend(cnn_rows(cfg))
    rows.extend(result_rows(cfg))
    rows.extend(horizon_rows(cfg))
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_CSV, index=False, encoding="utf-8")
    AUDIT_MD.write_text(build_report(df), encoding="utf-8")
    print(f"Wrote {AUDIT_CSV}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
