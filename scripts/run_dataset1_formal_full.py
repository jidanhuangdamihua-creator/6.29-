"""Dataset1 formal full experiment (all methods, 50 epochs, unified hyperparams)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

from src.utils.environment import setup_logging
from src.experiment.experiment_runner import results_to_dataframe, run_all_experiments, save_results_to_csv
from paper_reproduction_protocol import get_results_output_paths, load_paper_protocol
from src.source_selection.source_selector import KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE
from src.utils.experiment_hyperparams import FIXED_CLIPNORM, FIXED_DROPOUT, FIXED_LEARNING_RATE
from src.utils.runtime_control import set_verbose_mode

DATA_PATH = ROOT / "数据集/原始数据/(Dataset 1/train.csv"
DATASET_NAME = "Dataset1"
ENABLED_METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]
SOURCE_EPOCHS = 50
TARGET_EPOCHS = 50
LEARNING_RATE = FIXED_LEARNING_RATE


def main() -> None:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol = load_paper_protocol(cfg)
    output_paths = get_results_output_paths(ROOT, protocol)
    output_paths["results_dir"].mkdir(parents=True, exist_ok=True)
    output_csv = output_paths["results_dir"] / "dataset1_results.csv"
    log_file = output_paths["results_dir"] / "dataset1_formal_run.log"

    print("=" * 72)
    print("DATASET1 FORMAL FULL EXPERIMENT - PRE-RUN CHECK")
    print("=" * 72)
    print("Command:")
    print(f"  cd {ROOT}")
    print(f"  python3 scripts/run_dataset1_formal_full.py")
    print()
    print(f"Dataset path: {DATA_PATH}")
    print(f"Dataset exists: {DATA_PATH.exists()}")
    print(f"Output directory: {output_paths['results_dir']}")
    print(f"Output CSV: {output_csv}")
    print(f"Log file: {log_file}")
    print("Resolved hyperparameters:")
    print(f"  learning_rate = {LEARNING_RATE}")
    print(f"  source_epochs = {SOURCE_EPOCHS}")
    print(f"  target_epochs = {TARGET_EPOCHS}")
    print(f"  clipnorm = {FIXED_CLIPNORM}")
    print(f"  dropout = {FIXED_DROPOUT}")
    print(
        "  knn_representation = "
        f"{KNN_REPRESENTATION_PAPER_OBSERVED_SEQUENCE} (default; not passed explicitly)"
    )
    print(f"  enabled_methods = {ENABLED_METHODS}")
    print(f"  verbose_mode = summary")
    print(f"  show_method_progress = True")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Official Dataset1 file not found: {DATA_PATH}")

    set_verbose_mode("summary")
    setup_logging(log_level="WARNING", log_file=str(log_file))

    exp_cfg = cfg["single_experiment"]
    try:
        result = run_all_experiments(
            dataset_name=DATASET_NAME,
            data_path=str(DATA_PATH),
            config=cfg,
            feature_cols=cfg["features"]["default_feature_cols"],
            k=int(exp_cfg["k"]),
            number_of_sources=int(exp_cfg["k"]),
            horizon=int(exp_cfg["horizon"]),
            window_size=int(exp_cfg["window_size"]),
            weight_mode=str(exp_cfg["weight_mode"]),
            estimator_name=str(exp_cfg["estimator_name"]),
            keep_ratio=float(exp_cfg["keep_ratio"]),
            include_sales_in_knn=True,
            learning_rate=LEARNING_RATE,
            source_epochs=SOURCE_EPOCHS,
            target_epochs=TARGET_EPOCHS,
            batch_size=int(exp_cfg["batch_size"]),
            enabled_methods=ENABLED_METHODS,
            verbose_mode="summary",
            show_method_progress=True,
        )
    except Exception:
        print("\nEXPERIMENT FAILED - FULL TRACEBACK:")
        traceback.print_exc()
        raise

    df = results_to_dataframe(result)
    save_results_to_csv(df, str(output_csv))

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 72)
    summary_cols = ["method", "smape", "original_scale_smape", "rmse", "accuracy", "mae", "mape", "metric_space_used"]
    present = [c for c in summary_cols if c in df.columns]
    if "smape" in df.columns:
        print(df[present].sort_values("smape").to_string(index=False))
    else:
        print("No sMAPE column found. Please rerun experiments after metric update.")
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
