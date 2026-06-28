"""Smoke test entry script for minimal end-to-end matrix execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

from experiment_matrix_runner import build_smoke_test_matrix, run_experiment_matrix


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    smoke = build_smoke_test_matrix()

    result = run_experiment_matrix(
        data_path_map=cfg["dataset_paths"],
        config=cfg,
        feature_cols=cfg["features"]["default_feature_cols"],
        dataset_names=smoke["dataset_names"],
        horizons=smoke["horizons"],
        source_counts=smoke["source_counts"],
        weight_modes=smoke["weight_modes"],
        keep_ratios=smoke["keep_ratios"],
        enabled_methods_options=smoke["enabled_methods_options"],
        learning_rate=cfg["single_experiment"].get("learning_rate", 0.001),
        source_epochs=cfg["single_experiment"]["source_epochs"],
        target_epochs=cfg["single_experiment"]["target_epochs"],
        batch_size=cfg["single_experiment"]["batch_size"],
    )

    print("Smoke Test Completed")
    print("Master Results Path:")
    print(result["master_results_path"])
    print("Snapshot Path:")
    print(result["snapshot_path"])


if __name__ == "__main__":
    main()
