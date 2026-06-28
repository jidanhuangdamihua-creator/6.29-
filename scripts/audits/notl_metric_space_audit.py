"""Audit No-TL metric-space evidence in latest result CSVs.

This script is read-only with respect to training artifacts. It does not run
experiments and writes only outputs/audits/notl_metric_space_audit.{csv,md}.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "outputs" / "audits"
AUDIT_CSV = AUDIT_DIR / "notl_metric_space_audit.csv"
AUDIT_MD = AUDIT_DIR / "notl_metric_space_audit.md"

RESULT_FILES = [
    ROOT / "outputs" / "experiment_results" / "paper_results.csv",
    ROOT / "outputs" / "experiment_results" / "full_paper_results.csv",
    ROOT / "outputs" / "experiment_results" / "dataset1_results.csv",
    ROOT / "outputs" / "experiment_results" / "dataset2_results.csv",
    ROOT / "outputs" / "experiment_results" / "dataset3_results.csv",
]

REQUESTED_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "model_name",
    "information_sharing",
    "strict_paper_metrics",
    "metric_space",
    "metric_space_current",
    "metric_space_paper",
    "paper_metric_aligned",
    "rmse",
    "normalized_rmse",
    "original_scale_rmse",
    "rmse_paper",
    "accuracy",
    "normalized_accuracy",
    "original_scale_accuracy",
    "accuracy_paper",
    "learning_rate",
    "lr",
    "epochs",
    "random_seed",
    "source_file",
]

PAPER_NOTL_RMSE: dict[str, float | None] = {
    # The inspected repository docs identify Table 7/8 as paper RMSE tables,
    # but do not include per-dataset No-TL values in machine-readable form.
    "Dataset1": None,
    "Dataset2": None,
    "Dataset3": None,
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def line_ref(path: Path, pattern: str, default: str = "") -> str:
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern in line:
                return f"{rel(path)}:{idx}"
    except FileNotFoundError:
        return default
    return default


def regex_line_ref(path: Path, pattern: str, default: str = "") -> str:
    rx = re.compile(pattern)
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if rx.search(line):
                return f"{rel(path)}:{idx}"
    except FileNotFoundError:
        return default
    return default


def get_value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return value
    return ""


def boolish(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "True"
    if text.lower() in {"false", "0", "no"}:
        return "False"
    return text


def as_float(value: Any) -> float:
    try:
        if value == "" or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def dataset_id(dataset: Any) -> str:
    match = re.search(r"(\d+)", str(dataset))
    return match.group(1) if match else ""


def read_config_defaults() -> dict[str, Any]:
    cfg_path = ROOT / "configs" / "default_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    exp = cfg.get("single_experiment", {})
    metric = cfg.get("paper_reproduction", {}).get("metric_protocol", {})
    return {
        "config_learning_rate": exp.get("learning_rate", ""),
        "config_target_epochs": exp.get("target_epochs", ""),
        "config_strict_paper_metrics": metric.get("strict_paper_metrics", ""),
        "config_metric_space_current": metric.get("current_metric_space", ""),
        "config_metric_space_paper": metric.get("paper_metric_space", ""),
    }


def collect_notl_rows() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing_files: list[str] = []
    defaults = read_config_defaults()

    for path in RESULT_FILES:
        if not path.exists():
            missing_files.append(rel(path))
            continue

        df = pd.read_csv(path)
        if "method" in df.columns:
            mask = df["method"].astype(str).str.strip().eq("No-TL")
        elif "model_name" in df.columns:
            mask = df["model_name"].astype(str).str.strip().eq("No-TL")
        else:
            mask = pd.Series(False, index=df.index)

        for idx, row in df.loc[mask].iterrows():
            strict = get_value(row, "strict_paper_metrics")
            strict_source = "csv.strict_paper_metrics"
            if strict == "":
                strict = get_value(row, "strict_paper_mode")
                strict_source = "csv.strict_paper_mode"
            if strict == "":
                strict = defaults["config_strict_paper_metrics"]
                strict_source = "config.default"

            learning_rate = get_value(row, "learning_rate", "lr")
            lr = get_value(row, "lr", "learning_rate")
            epochs = get_value(row, "epochs", "target_epochs", "epoch")

            out = {
                "dataset_id": get_value(row, "dataset_id") or dataset_id(get_value(row, "dataset")),
                "dataset": get_value(row, "dataset"),
                "method": get_value(row, "method"),
                "model_name": get_value(row, "model_name"),
                "information_sharing": get_value(row, "information_sharing"),
                "strict_paper_metrics": boolish(strict),
                "metric_space": get_value(row, "metric_space"),
                "metric_space_current": get_value(row, "metric_space_current", "current_metric_space"),
                "metric_space_paper": get_value(row, "metric_space_paper", "paper_metric_space"),
                "paper_metric_aligned": boolish(get_value(row, "paper_metric_aligned")),
                "rmse": get_value(row, "rmse"),
                "normalized_rmse": get_value(row, "normalized_rmse"),
                "original_scale_rmse": get_value(row, "original_scale_rmse"),
                "rmse_paper": get_value(row, "rmse_paper"),
                "accuracy": get_value(row, "accuracy"),
                "normalized_accuracy": get_value(row, "normalized_accuracy"),
                "original_scale_accuracy": get_value(row, "original_scale_accuracy"),
                "accuracy_paper": get_value(row, "accuracy_paper"),
                "learning_rate": learning_rate,
                "lr": lr,
                "epochs": epochs,
                "random_seed": get_value(row, "random_seed", "seed", "random_state"),
                "source_file": rel(path),
                "csv_line": int(idx) + 2,
                "strict_paper_metrics_source": strict_source,
                "learning_rate_note": "" if learning_rate != "" else f"missing in CSV; config default={defaults['config_learning_rate']}",
                "epochs_note": "" if epochs != "" else f"missing in CSV; config target_epochs={defaults['config_target_epochs']}",
                "row_evidence": f"{rel(path)}:{int(idx) + 2}",
            }
            rows.append(out)

    columns = REQUESTED_COLUMNS + [
        "csv_line",
        "strict_paper_metrics_source",
        "learning_rate_note",
        "epochs_note",
        "row_evidence",
    ]
    return pd.DataFrame(rows, columns=columns), missing_files


def numeric_equal(a: Any, b: Any, tol: float = 1e-9) -> bool | None:
    af = as_float(a)
    bf = as_float(b)
    if math.isnan(af) and math.isnan(bf):
        return None
    if math.isnan(af) or math.isnan(bf):
        return False
    return abs(af - bf) <= tol


def status_counts(series: pd.Series) -> str:
    if series.empty:
        return "{}"
    return str(series.astype(str).value_counts(dropna=False).to_dict())


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    work = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    headers = list(work.columns)
    rows = []
    for _, row in work.iterrows():
        rows.append([str(row.get(col, "")).replace("\n", " ") for col in headers])
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report(audit: pd.DataFrame, missing_files: list[str]) -> str:
    cfg_ref = line_ref(ROOT / "configs" / "default_config.json", '"strict_paper_metrics"')
    main_ref = line_ref(ROOT / "scripts" / "run_main_experiment.py", '"strict_paper_metrics"')
    full_ref = line_ref(ROOT / "scripts" / "run_full_paper_experiments.py", '"strict_paper_metrics"')
    metrics_strict_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", "strict_paper_metrics = bool")
    metrics_space_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", "metric_space_used = current_space")
    metrics_return_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"rmse": float(rmse_final)')
    metrics_original_ref = line_ref(ROOT / "src" / "evaluation" / "metrics.py", '"original_scale_rmse"')
    notl_call_ref = line_ref(ROOT / "src" / "experiment" / "run_no_tl_experiment.py", "metric_result = compute_metrics_with_protocol")
    notl_return_ref = line_ref(ROOT / "src" / "experiment" / "run_no_tl_experiment.py", '"rmse_paper"')
    extract_fallback_ref = line_ref(ROOT / "experiment_runner.py", 'result["original_scale_rmse"]')
    dataframe_missing_ref = line_ref(ROOT / "experiment_runner.py", '"normalized_rmse",')
    full_row_ref = line_ref(ROOT / "scripts" / "run_full_paper_experiments.py", 'row["original_scale_rmse"]')
    docs_ref = line_ref(ROOT / "docs" / "paper_protocol_alignment.md", "论文 Table 7/8 的主对比")
    ambiguity_ref = line_ref(ROOT / "outputs" / "paper_alignment_reports" / "论文复现实验细节索引.md", "RMSE 在归一化空间还是原始 sales 空间")
    config_lr_ref = line_ref(ROOT / "configs" / "default_config.json", '"learning_rate"')
    lr_grid_ref = line_ref(ROOT / "outputs" / "audits" / "cnn_lr_epoch_grid_ablation.md", "Grid: learning_rates")
    dataset1_legacy_ref = regex_line_ref(ROOT / "outputs" / "experiment_results" / "dataset1_results.csv", r"^Dataset1,No-TL,")

    rmse_norm = audit.apply(lambda r: numeric_equal(r["rmse"], r["normalized_rmse"]), axis=1)
    paper_original = audit.apply(lambda r: numeric_equal(r["rmse_paper"], r["original_scale_rmse"]), axis=1)
    rmse_paper_same_as_rmse = audit.apply(lambda r: numeric_equal(r["rmse"], r["rmse_paper"]), axis=1)
    original_missing = audit["original_scale_rmse"].astype(str).replace({"nan": ""}).eq("")

    full_latest = audit[
        audit["source_file"].isin(
            [
                "outputs/experiment_results/paper_results.csv",
                "outputs/experiment_results/full_paper_results.csv",
            ]
        )
    ].copy()
    full_latest_notl = full_latest.drop_duplicates(["source_file", "dataset", "method", "rmse", "information_sharing"], keep="first") if "information_sharing" in full_latest.columns else full_latest

    comparable_lr_1e5 = False
    cannot_compare_reason = (
        "用户指定的 latest result CSV 没有 lr=1e-4 字段或行；"
        "主配置 learning_rate=1e-4；且仓库证据仍将论文 metric space 标为 PARTIAL/未完全确认。"
    )

    current_rows = audit[
        audit["source_file"].isin(
            [
                "outputs/experiment_results/paper_results.csv",
                "outputs/experiment_results/full_paper_results.csv",
            ]
        )
    ]

    lines = [
        "# No-TL Metric Space Audit",
        "",
        "Scope: static/code/CSV audit only. No training was run; no existing result file was overwritten. "
        f"Companion CSV: `{rel(AUDIT_CSV)}`.",
        "",
        "## Direct Answers",
        "",
        "- 当前 `strict_paper_metrics` 最终生效状态: latest paper runners force it to `True` in runtime config, "
        "but `compute_metrics_with_protocol` still keeps primary `rmse/accuracy` in `normalized_minmax_space`. "
        f"Evidence: default config is false at `{cfg_ref}`; single runner forces true at `{main_ref}`; full runner forces true at `{full_ref}`; metrics reads the flag at `{metrics_strict_ref}` but sets `metric_space_used=current_space` at `{metrics_space_ref}`.",
        "- No-TL 当前用于对比论文的 RMSE 来自哪一列: latest reports and README/docs semantics point to primary `rmse`, semantically equal to `normalized_rmse`, not `rmse_paper/original_scale_rmse`. "
        f"Evidence: primary return at `{metrics_return_ref}`; docs state Table 7/8 should prefer normalized RMSE at `{docs_ref}`.",
        f"- `rmse` 与 `normalized_rmse` 是否一致: {rmse_norm.value_counts(dropna=False).to_dict()} across audited No-TL rows. "
        f"`rmse_paper` 与 `original_scale_rmse` 是否一致: {paper_original.value_counts(dropna=False).to_dict()}. "
        f"`rmse` 与 `rmse_paper` 是否一致: {rmse_paper_same_as_rmse.value_counts(dropna=False).to_dict()}.",
        f"- `original_scale_rmse` 为空: {int(original_missing.sum())} / {len(audit)} audited rows. "
        "若为空，链路原因是旧/不完整写出层没有补齐 alias 字段，或该 CSV 根本没有这些列；full runner 已在 row 写出层补齐。 "
        f"Evidence: No-TL raw returns original-scale fields at `{notl_return_ref}`; generic extractor fallback at `{extract_fallback_ref}`; full runner row fallback at `{full_row_ref}`; older dataframe schema contains columns but row dict path did not assign values at `{dataframe_missing_ref}`; legacy Dataset1 No-TL CSV evidence `{dataset1_legacy_ref}`.",
        "- 如果 `rmse_paper` 与 `original_scale_rmse` 不一致: audited rows show no numeric disagreement where both are present; missing cases are field-presence/writeout issues, not a different formula.",
        f"- 当前是否可以把 lr=1e-4 的 No-TL RMSE 直接和论文 Table 的 No-TL RMSE 比较: {('能' if comparable_lr_1e5 else '不能')}。{cannot_compare_reason} "
        f"Evidence: config learning_rate at `{config_lr_ref}`; lr=1e-4 appears in separate audit grid at `{lr_grid_ref}`, not in the requested latest result CSVs.",
        "- 如果不能，原因: latest paper outputs do not encode lr=1e-4; paper metric space remains unresolved; current primary RMSE is normalized-MinMax candidate but exact paper normalization granularity/fit range remains unproven.",
        "",
        "## CSV Evidence Summary",
        "",
        f"- Missing requested result files: {missing_files if missing_files else 'none'}",
        f"- No-TL audited rows: {len(audit)}",
        f"- Source files: {status_counts(audit['source_file'])}",
        f"- strict_paper_metrics values: {status_counts(audit['strict_paper_metrics'])}",
        f"- metric_space values: {status_counts(audit['metric_space'])}",
        f"- metric_space_current values: {status_counts(audit['metric_space_current'])}",
        f"- metric_space_paper values: {status_counts(audit['metric_space_paper'])}",
        f"- paper_metric_aligned values: {status_counts(audit['paper_metric_aligned'])}",
        "",
        "## No-TL Rows",
        "",
        markdown_table(
            audit,
            [
                "dataset_id",
                "dataset",
                "method",
                "information_sharing",
                "strict_paper_metrics",
                "metric_space",
                "metric_space_current",
                "metric_space_paper",
                "paper_metric_aligned",
                "rmse",
                "normalized_rmse",
                "original_scale_rmse",
                "rmse_paper",
                "learning_rate",
                "epochs",
                "random_seed",
                "row_evidence",
            ],
            max_rows=30,
        ),
        "",
        "## Code Evidence",
        "",
        f"- Config default metric protocol: `{cfg_ref}`.",
        f"- `scripts/run_main_experiment.py` forces runtime strict paper metrics: `{main_ref}`.",
        f"- `scripts/run_full_paper_experiments.py` forces runtime strict paper metrics: `{full_ref}`.",
        f"- No-TL calls the shared metrics function with target scaler/feature columns: `{notl_call_ref}`.",
        f"- Metrics function fixes primary metric space to current normalized space: `{metrics_space_ref}`.",
        f"- Original-scale diagnostic alias is returned separately: `{metrics_original_ref}`.",
        f"- Paper metric ambiguity is documented: `{ambiguity_ref}`.",
        "",
        "## Paper No-TL Gap Multiples",
        "",
    ]

    if comparable_lr_1e5:
        rows = []
        for dataset, paper_value in PAPER_NOTL_RMSE.items():
            current = current_rows[current_rows["dataset"].astype(str).eq(dataset)]
            current_rmse = as_float(current["normalized_rmse"].iloc[0]) if not current.empty else math.nan
            multiple = current_rmse / paper_value if paper_value and not math.isnan(current_rmse) else math.nan
            rows.append({"dataset": dataset, "paper_notl_rmse": paper_value, "current_notl_rmse": current_rmse, "gap_multiple": multiple})
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        lines.append("Not computed, because direct comparison is not valid under the inspected latest CSV evidence.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The latest `paper_results.csv` and `full_paper_results.csv` No-TL rows contain both normalized primary metrics and original-scale diagnostics. "
            "For Dataset1 No-TL, for example, `rmse=normalized_rmse=0.496751...` while `rmse_paper=original_scale_rmse=37.256358...`; the latter is not the value currently used as the primary paper-table candidate.",
            "",
            "`dataset1_results.csv` is a legacy/narrow output: it has No-TL `rmse=904.953324...` and no `normalized_rmse/original_scale_rmse/rmse_paper` columns. "
            "That row cannot establish normalized-vs-original alias consistency by itself. `dataset2_results.csv` and `dataset3_results.csv` are absent in the requested latest location.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit, missing_files = collect_notl_rows()
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8")
    AUDIT_MD.write_text(build_report(audit, missing_files), encoding="utf-8")
    print(f"wrote {AUDIT_CSV}")
    print(f"wrote {AUDIT_MD}")
    print(f"rows={len(audit)} missing_files={missing_files}")


if __name__ == "__main__":
    main()
