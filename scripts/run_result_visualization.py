"""Generate formatted benchmark tables and plots from full paper results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.result_visualizer import run_result_visualization


def main() -> None:
    exp_dir = ROOT / "outputs" / "experiment_results"
    csv_path = exp_dir / "paper_results.csv"
    if not csv_path.exists():
        csv_path = exp_dir / "full_paper_results.csv"
    report = run_result_visualization(
        csv_path=str(csv_path),
        output_dir=str(ROOT / "outputs" / "results_reports"),
        method_order=["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"],
    )

    print("Result visualization completed.")
    print("Formatted table:")
    print(report["formatted_table_path"])
    print("sMAPE plot:")
    print(report.get("smape_plot_path", report["rmse_plot_path"]))
    print("Accuracy plot:")
    print(report["accuracy_plot_path"])


if __name__ == "__main__":
    main()
