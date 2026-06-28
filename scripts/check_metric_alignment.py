"""Check whether each experiment row is using paper metric protocol under current configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs" / "experiment_results" / "full_paper_results.csv"
DEFAULT_REPORT_DIR = ROOT / "outputs" / "paper_alignment_reports"
DEFAULT_REPORT_CSV = DEFAULT_REPORT_DIR / "metric_alignment_check_report.csv"
DEFAULT_REPORT_JSON = DEFAULT_REPORT_DIR / "metric_alignment_check_report.json"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def main() -> None:
    if not DEFAULT_RESULTS.exists():
        raise FileNotFoundError(f"results csv not found: {DEFAULT_RESULTS}")

    df = pd.read_csv(DEFAULT_RESULTS)
    if df.empty:
        raise ValueError("results csv is empty")

    required_base = {"dataset", "method", "rmse", "accuracy"}
    missing_base = sorted(required_base.difference(df.columns))
    if missing_base:
        raise ValueError(f"missing required columns: {missing_base}")

    if "metric_space_current" not in df.columns:
        df["metric_space_current"] = df.get("metric_space", "normalized_minmax_space")
    if "metric_space_paper" not in df.columns:
        df["metric_space_paper"] = df.get("paper_metric_space", "original_sales_space")
    if "paper_metric_aligned" not in df.columns:
        df["paper_metric_aligned"] = False
    if "inverse_transform_applied" not in df.columns:
        df["inverse_transform_applied"] = False
    if "metric_notes" not in df.columns:
        df["metric_notes"] = ""

    status_rows = []
    for _, row in df.iterrows():
        paper_metric_aligned = _to_bool(row.get("paper_metric_aligned", False))
        inverse_transform_applied = _to_bool(row.get("inverse_transform_applied", False))
        metric_space_current = str(row.get("metric_space_current", ""))
        metric_space_paper = str(row.get("metric_space_paper", ""))

        paper_original_metric = paper_metric_aligned and (
            metric_space_current == metric_space_paper or inverse_transform_applied
        )

        status_rows.append(
            {
                "dataset": row.get("dataset", "N/A"),
                "method": row.get("method", "N/A"),
                "rmse": row.get("rmse"),
                "accuracy": row.get("accuracy"),
                "metric_space_current": metric_space_current,
                "metric_space_paper": metric_space_paper,
                "paper_metric_aligned": paper_metric_aligned,
                "inverse_transform_applied": inverse_transform_applied,
                "is_paper_original_metric": bool(paper_original_metric),
                "metric_notes": str(row.get("metric_notes", "")),
            }
        )

    out_df = pd.DataFrame(status_rows)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(DEFAULT_REPORT_CSV, index=False, encoding="utf-8")

    summary = {
        "input_csv": str(DEFAULT_RESULTS),
        "report_csv": str(DEFAULT_REPORT_CSV),
        "total_rows": int(len(out_df)),
        "paper_original_metric_rows": int(out_df["is_paper_original_metric"].sum()),
        "not_paper_original_metric_rows": int((~out_df["is_paper_original_metric"]).sum()),
    }
    DEFAULT_REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Metric alignment check completed")
    print(f"Input: {DEFAULT_RESULTS}")
    print(f"Report CSV: {DEFAULT_REPORT_CSV}")
    print(f"Report JSON: {DEFAULT_REPORT_JSON}")
    print(f"Rows: {summary['total_rows']} paper_original_metric={summary['paper_original_metric_rows']}")


if __name__ == "__main__":
    main()
