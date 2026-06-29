"""
模块12完整矩阵启动脚本。

职责：
1. 使用 build_full_matrix_default() 生成完整矩阵配置
2. 调用 run_experiment_matrix(...) 执行完整矩阵
3. 打印 master_results.csv 与 matrix_snapshot.json 路径
"""

from __future__ import annotations

import json
from pathlib import Path

import tf_compat  # must be imported before tensorflow/keras

from dataset_registry import get_dataset_path_map
from src.experiment.experiment_matrix_runner import build_full_matrix_default, run_experiment_matrix


ROOT = Path(__file__).resolve().parent


def main() -> None:
    full_matrix = build_full_matrix_default()
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_path_map = get_dataset_path_map()

    result = run_experiment_matrix(
        data_path_map=data_path_map,
        config=cfg,
        feature_cols=["sales", "year", "month", "week", "day"],
        dataset_names=full_matrix["dataset_names"],
        horizons=full_matrix["horizons"],
        source_counts=full_matrix["source_counts"],
        weight_modes=full_matrix["weight_modes"],
        keep_ratios=full_matrix["keep_ratios"],
        enabled_methods_options=full_matrix["enabled_methods_options"],
    )

    print("Full Experiment Matrix Runner Completed")
    print()
    print("Master Results Path:")
    print(result["master_results_path"])
    print()
    print("Snapshot Path:")
    print(result["snapshot_path"])


if __name__ == "__main__":
    main()
