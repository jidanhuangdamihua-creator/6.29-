import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SOURCE_HISTORY_DAYS
from src.constants import RESULT_SCHEMA_COLUMNS

import tf_compat  # must be imported before tensorflow/keras

import pandas as pd

from src.utils.run_utils import create_run_dir
from src.utils.entity_experiment import run_single_entity_experiment
from src.utils.parquet_data_loader import (
    load_parquet_source_target,
    read_dataset_windows,
    attach_window_attrs,
    load_knn_results,
)
from src.data_processing.data_preprocessing import infer_source_selection_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_COLUMNS = [
    "dataset_id",
    "scenario",
    "target_entity_key",
    "source_identifier",
    "selected_sources",
]

config = {
    "use_parquet": True,
    "dataset_id": 6,
    "dataset_name": "Dataset6",
    "parquet_dir": "数据集/固化数据",
    "knn_json_dir": "outputs/knn_selection/Dataset6",
    "info_sharing": "without",
    "source_history_days": SOURCE_HISTORY_DAYS,
    "entity_col": "entity_id",
    "enabled_methods": [
        "No-TL",
        "SS-TL",
        "MSWA-TL",
        "MSSB-TL",
        "MSML-TL",
        "MSML-TL-RFE",
    ],
    "output_filename": "dataset6_results.csv",
    "horizon": 1,
    "window_size": 10,
    "learning_rate": 0.0001,
    "source_epochs": 50,
    "target_epochs": 50,
    "batch_size": 16,
    "random_state": 42,
    "source_count": 3,
    "smoke": False,
    "smoke_target_limit": 1,
    "smoke_source_limit": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dataset6 fixed-parquet experiment.")
    parser.add_argument("--info-sharing", choices=["without", "with"], default=config["info_sharing"])
    parser.add_argument("--smoke", action="store_true", help="Run tiny target/source limits with the same window logic.")
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override source/target epochs for lightweight checks.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run directory.")
    return parser.parse_args()


def _reference_result_columns() -> list[str]:
    return RESULT_SCHEMA_COLUMNS


def _align_results_to_reference_schema(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    reference_columns = _reference_result_columns()
    for column in reference_columns + TRACE_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = None
    return aligned[reference_columns + TRACE_COLUMNS]


def main() -> None:
    args = _parse_args()
    config["info_sharing"] = str(args.info_sharing)
    config["smoke"] = bool(args.smoke)
    if args.target_limit is not None:
        config["smoke_target_limit"] = int(args.target_limit)
    if args.source_limit is not None:
        config["smoke_source_limit"] = int(args.source_limit)
    if args.epochs is not None:
        config["source_epochs"] = int(args.epochs)
        config["target_epochs"] = int(args.epochs)
    config["output_filename"] = f"dataset{config['dataset_id']}_{config['info_sharing']}_results.csv"

    root = PROJECT_ROOT
    label = f"D{config['dataset_id']}_{config['source_history_days']}d_{config['info_sharing']}"
    if args.output_dir is None:
        run_dir = create_run_dir(root, label=label)
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
    )
    source_df, target_df = load_parquet_source_target(
        dataset_id=config["dataset_id"],
        parquet_dir=config["parquet_dir"],
        windows=windows,
        source_history_days=config["source_history_days"],
    )
    source_df = attach_window_attrs(source_df, windows, role="source")
    target_df = attach_window_attrs(target_df, windows, role="target")

    feature_info = infer_source_selection_feature_columns(source_df, target_df)
    feature_cols = feature_info["selected_features"]

    knn_data = load_knn_results(config["knn_json_dir"], config["info_sharing"])
    target_entity_keys = list(knn_data["results"].keys())
    if config["smoke"]:
        target_entity_keys = target_entity_keys[: int(config["smoke_target_limit"])]
        selected_source_entities = {
            str(item.get("source_entity"))
            for values in knn_data["results"].values()
            for item in values[: int(config["smoke_source_limit"])]
            if isinstance(item, dict) and item.get("source_entity") is not None
        }
        if selected_source_entities:
            source_df = source_df[source_df[config["entity_col"]].astype(str).isin(selected_source_entities)].copy()

    all_rows = []
    for entity_key in target_entity_keys:
        target_entity_df = target_df[target_df[config["entity_col"]] == entity_key].copy()
        if target_entity_df.empty:
            print(f"WARNING: entity {entity_key} not found in target_df, skipping")
            continue

        rows = run_single_entity_experiment(
            entity_key=entity_key,
            source_df=source_df,
            target_entity_df=target_entity_df,
            feature_cols=feature_cols,
            config=config,
            enabled_methods=config["enabled_methods"],
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = _align_results_to_reference_schema(df)
    out_path = results_dir / config["output_filename"]
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Results saved to {out_path}")
    print(df[["target_entity_key", "method", "smape", "rmse"]].to_string())


if __name__ == "__main__":
    main()
