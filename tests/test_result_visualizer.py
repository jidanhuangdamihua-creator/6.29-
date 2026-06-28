"""
模块11最小测试：论文结果表与图表生成

目标：
1. 读取模块10输出 CSV
2. 生成格式化结果表与图表
3. 打印摘要与表头预览
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result_visualizer import run_result_visualization


def main() -> None:
    result = run_result_visualization(
        csv_path="outputs/experiment_results/dataset1_results.csv",
        output_dir="outputs/results_reports",
    )

    print("Result Visualization Completed Successfully")
    print()
    print("Formatted Table Path:")
    print(result["formatted_table_path"])
    print()
    print("RMSE Plot Path:")
    print(result["rmse_plot_path"])
    print()
    print("Accuracy Plot Path:")
    print(result["accuracy_plot_path"])
    print()
    print("Summary:")
    print(result["summary"])
    print()
    print("Formatted Results Table:")
    print(result["results_df"].head().to_string(index=False))


if __name__ == "__main__":
    main()
