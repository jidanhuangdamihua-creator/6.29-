"""
模块10最小测试：论文实验总运行器

目标：
1. 统一跑完整套方法
2. 打印对比结果表
3. 保存 CSV 到 outputs/experiment_results/dataset1_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.environment import setup_logging
from src.experiment.experiment_runner import (
    print_results_table,
    results_to_dataframe,
    run_all_experiments,
    save_results_to_csv,
)


def main() -> None:
    setup_logging(log_level="INFO", log_file=None)

    experiment_results = run_all_experiments(
        dataset_name="Dataset1",
        data_path="demand-forecasting-kernels-only (1)/train.csv",
        feature_cols=["sales", "year", "month", "week", "day"],
        k=3,
        horizon=1,
        window_size=10,
        weight_mode="inverse_distance",
        estimator_name="random_forest",
        keep_ratio=0.5,
        source_epochs=2,
        target_epochs=2,
        batch_size=16,
    )

    results_df = results_to_dataframe(experiment_results)

    print("Experiment Runner Completed Successfully")
    print()
    print_results_table(results_df)

    output_csv = "outputs/experiment_results/dataset1_results.csv"
    save_results_to_csv(results_df, output_csv)

    print()
    print("Saved CSV:")
    print(output_csv)


if __name__ == "__main__":
    main()
