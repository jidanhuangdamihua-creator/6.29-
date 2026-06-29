"""
模块12最小测试：完整实验矩阵运行器

目标：
1. 先用 smoke test 矩阵验证矩阵运行器
2. 输出 master_results.csv 与 matrix_snapshot.json
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.experiment_matrix_runner import build_smoke_test_matrix, run_experiment_matrix


def main() -> None:
    data_path_map = {
        "Dataset1": "demand-forecasting-kernels-only (1)/train.csv",
    }

    smoke_matrix = build_smoke_test_matrix()
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    result = run_experiment_matrix(
        data_path_map=data_path_map,
        config=cfg,
        feature_cols=["sales", "year", "month", "week", "day"],
        dataset_names=smoke_matrix["dataset_names"],
        horizons=smoke_matrix["horizons"],
        source_counts=smoke_matrix["source_counts"],
        weight_modes=smoke_matrix["weight_modes"],
        keep_ratios=smoke_matrix["keep_ratios"],
        enabled_methods_options=smoke_matrix["enabled_methods_options"],
        source_epochs=2,
        target_epochs=2,
        batch_size=16,
    )

    print("Experiment Matrix Runner Completed Successfully")
    print()
    print("Number of Experiments:")
    print(result["num_experiments"])
    print()
    print("Master Results Path:")
    print(result["master_results_path"])
    print()
    print("Snapshot Path:")
    print(result["snapshot_path"])
    print()
    print("Master Results Preview:")
    print(result["master_df"].head().to_string(index=False))


if __name__ == "__main__":
    main()
