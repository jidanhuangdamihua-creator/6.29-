"""结果表与图表生成入口脚本。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result_visualizer import run_result_visualization


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    csv_path = ROOT / "outputs" / "experiment_results" / "dataset1_results.csv"
    output_dir = ROOT / cfg["outputs"]["results_reports_dir"]

    result = run_result_visualization(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
    )

    print("Results Report Generated")
    print("Formatted Table Path:")
    print(result["formatted_table_path"])
    print("RMSE Plot Path:")
    print(result["rmse_plot_path"])
    print("Accuracy Plot Path:")
    print(result["accuracy_plot_path"])


if __name__ == "__main__":
    main()
