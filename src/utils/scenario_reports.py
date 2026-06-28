from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


SCENARIOS = [
    "without_information_sharing",
    "with_information_sharing",
]
REQUIRED_REPORT_COLUMNS = {"dataset", "method", "scenario", "rmse", "accuracy"}
SMAPE_MISSING_MESSAGE = "No sMAPE column found. Please rerun experiments after metric update."


def normalize_results_for_scenario_reports(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a report-ready copy with canonical scenario/metric columns."""
    df = results_df.copy()
    if "scenario" not in df.columns and "information_sharing" in df.columns:
        df["scenario"] = df["information_sharing"]

    missing = REQUIRED_REPORT_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"results_df missing columns for scenario reports: {sorted(missing)}")
    if "smape" not in df.columns and "original_scale_smape" not in df.columns:
        print(SMAPE_MISSING_MESSAGE)
        return pd.DataFrame()
    if "smape" not in df.columns:
        df["smape"] = pd.to_numeric(df["original_scale_smape"], errors="coerce")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip().eq("")].copy()

    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["smape"] = pd.to_numeric(df["smape"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df.dropna(subset=["dataset", "method", "scenario", "smape", "rmse", "accuracy"]).copy()


def generate_scenario_separated_reports(results_df: pd.DataFrame, output_dir: Path | str) -> Dict[str, Any]:
    """Save scenario-separated RMSE, accuracy, improvement, and best-method reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_df = normalize_results_for_scenario_reports(results_df)
    if report_df.empty:
        print(SMAPE_MISSING_MESSAGE)
        return {
            "all_long_path": None,
            "duplicate_rows_path": None,
            "smape_tables": {},
            "rmse_tables": {},
            "accuracy_tables": {},
            "improvement_tables": {},
            "best_method_tables": {},
        }

    all_long_path = out_dir / "all_results_long_format.csv"
    report_df.to_csv(all_long_path, index=False, encoding="utf-8")

    duplicate_counts = (
        report_df.groupby(["dataset", "method", "scenario"])
        .size()
        .reset_index(name="count")
    )
    duplicate_rows = duplicate_counts[duplicate_counts["count"] > 1]
    duplicate_rows_path: Optional[Path] = None
    if not duplicate_rows.empty:
        duplicate_rows_path = out_dir / "duplicate_dataset_method_scenario_rows.csv"
        duplicate_rows.to_csv(duplicate_rows_path, index=False, encoding="utf-8")
        print(
            "[WARNING] Duplicate dataset-method-scenario rows detected. "
            "See duplicate_dataset_method_scenario_rows.csv"
        )

    smape_tables: Dict[str, pd.DataFrame] = {}
    rmse_tables: Dict[str, pd.DataFrame] = {}
    accuracy_tables: Dict[str, pd.DataFrame] = {}
    improvement_tables: Dict[str, pd.DataFrame] = {}
    best_method_tables: Dict[str, pd.DataFrame] = {}

    for scenario in SCENARIOS:
        scenario_df = report_df[report_df["scenario"] == scenario].copy()

        if scenario_df.empty:
            print(f"[WARNING] No rows found for scenario={scenario}")
            continue

        smape_table = scenario_df.pivot_table(
            index="dataset",
            columns="method",
            values="smape",
            aggfunc="first",
        )
        smape_table.to_csv(out_dir / f"smape_comparison_{scenario}.csv", encoding="utf-8")
        smape_tables[scenario] = smape_table

        rmse_table = scenario_df.pivot_table(
            index="dataset",
            columns="method",
            values="rmse",
            aggfunc="first",
        )
        rmse_table.to_csv(out_dir / f"rmse_comparison_{scenario}.csv", encoding="utf-8")
        rmse_tables[scenario] = rmse_table

        accuracy_table = scenario_df.pivot_table(
            index="dataset",
            columns="method",
            values="accuracy",
            aggfunc="first",
        )
        accuracy_table.to_csv(out_dir / f"accuracy_comparison_{scenario}.csv", encoding="utf-8")
        accuracy_tables[scenario] = accuracy_table

        _save_metric_bar_chart(
            smape_table,
            out_dir / f"smape_comparison_{scenario}.png",
            title=f"sMAPE comparison: {scenario}",
            ylabel="sMAPE",
        )
        _save_metric_bar_chart(
            accuracy_table,
            out_dir / f"accuracy_comparison_{scenario}.png",
            title=f"Accuracy comparison (normalized_minmax_space): {scenario}",
            ylabel="normalized_minmax_space Accuracy (1/RMSE)",
        )

        improvement_records = []
        for dataset, group in scenario_df.groupby("dataset"):
            notl_rows = group[group["method"] == "No-TL"]
            if notl_rows.empty:
                print(f"[WARNING] No No-TL baseline for dataset={dataset}, scenario={scenario}")
                continue

            baseline_smape = float(notl_rows.iloc[0]["smape"])

            for _, row in group.iterrows():
                smape = float(row["smape"])
                improvement = None if baseline_smape == 0 else (baseline_smape - smape) / baseline_smape * 100.0
                improvement_records.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "method": row["method"],
                        "baseline_method": "No-TL",
                        "baseline_smape": baseline_smape,
                        "smape": smape,
                        "smape_improvement_percent": improvement,
                    }
                )

        improvement_df = pd.DataFrame(improvement_records)
        improvement_df.to_csv(
            out_dir / f"improvement_vs_notl_{scenario}.csv",
            index=False,
            encoding="utf-8",
        )
        improvement_tables[scenario] = improvement_df

        best_rows = []
        for _, group in scenario_df.groupby("dataset"):
            best_rows.append(group.sort_values("smape", ascending=True).iloc[0])

        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(out_dir / f"best_methods_{scenario}.csv", index=False, encoding="utf-8")
        best_method_tables[scenario] = best_df

    return {
        "all_long_path": all_long_path,
        "duplicate_rows_path": duplicate_rows_path,
        "smape_tables": smape_tables,
        "rmse_tables": rmse_tables,
        "accuracy_tables": accuracy_tables,
        "improvement_tables": improvement_tables,
        "best_method_tables": best_method_tables,
    }


def _save_metric_bar_chart(table: pd.DataFrame, output_path: Path, title: str, ylabel: str) -> None:
    if table.empty:
        return

    import matplotlib.pyplot as plt

    ax = table.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("dataset")
    ax.set_ylabel(ylabel)
    ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
