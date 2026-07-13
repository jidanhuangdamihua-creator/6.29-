"""Run statistical significance analysis for experiment results."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.statistical_tests import (
    compute_average_rank,
    run_friedman_test,
    run_pairwise_wilcoxon_tests,
)
from src.visualization.result_visualizer import (
    plot_method_average_rank_bar_chart,
    save_wilcoxon_significance_table,
)


OUTPUT_DIR = ROOT / "outputs" / "statistical_reports"


def _resolve_input_csv() -> Path:
    exp_dir = ROOT / "outputs" / "experiment_results"
    preferred = exp_dir / "paper_results.csv"
    if preferred.exists():
        return preferred

    legacy = exp_dir / "full_paper_results.csv"
    if legacy.exists():
        return legacy

    all_csv = sorted(exp_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_csv:
        raise FileNotFoundError("No CSV found under outputs/experiment_results")
    return all_csv[0]


def _filter_success_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "error" in out.columns:
        out = out[out["error"].fillna("").astype(str).str.strip().eq("")]
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("success")]
    return out


def main() -> None:
    input_csv = _resolve_input_csv()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = _filter_success_rows(df)

    friedman = run_friedman_test(df)
    wilcoxon_df = run_pairwise_wilcoxon_tests(df)
    avg_rank_df = compute_average_rank(df)

    friedman_df = friedman

    friedman_path = OUTPUT_DIR / "friedman_test_results.csv"
    wilcoxon_path = OUTPUT_DIR / "wilcoxon_pairwise_results.csv"
    avg_rank_path = OUTPUT_DIR / "method_average_ranks.csv"

    friedman_df.to_csv(friedman_path, index=False, encoding="utf-8")
    wilcoxon_df.to_csv(wilcoxon_path, index=False, encoding="utf-8")
    avg_rank_df.to_csv(avg_rank_path, index=False, encoding="utf-8")

    # Additional artifacts requested in result visualizer update.
    rank_png = OUTPUT_DIR / "method_average_rank_bar.png"
    wilcoxon_table = OUTPUT_DIR / "wilcoxon_significance_table.csv"
    plot_method_average_rank_bar_chart(avg_rank_df, str(rank_png))
    save_wilcoxon_significance_table(wilcoxon_df, str(wilcoxon_table))

    print("Statistical analysis completed.")
    print("Input:")
    print(str(input_csv))
    print("Outputs:")
    print(str(friedman_path))
    print(str(wilcoxon_path))
    print(str(avg_rank_path))
    print(str(rank_png))
    print(str(wilcoxon_table))


if __name__ == "__main__":
    main()
