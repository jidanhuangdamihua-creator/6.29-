import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SOURCE_HISTORY_DAYS
from src.constants import RESULT_SCHEMA_COLUMNS
from src.constants import SOLIDIFIED_KNN_ROOT
from src.utils.result_schema import TRACE_COLUMNS, align_d4_d6_result_records

import tf_compat  # must be imported before tensorflow/keras

import pandas as pd

from src.utils.run_utils import create_run_dir
from src.utils.entity_experiment import run_single_entity_experiment
from src.utils.parquet_data_loader import (
    load_parquet_source_target,
    read_dataset_windows,
    load_knn_results,
)
from src.utils.d4_d6_runtime import apply_runtime_source_domain_policy, load_default_metric_protocol
from src.utils.finite_diagnostics import validate_feature_frame_finite
from src.utils.knn_feature_loader import resolve_knn_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config = {
    "use_parquet": True,
    "dataset_id": 4,
    "dataset_name": "Dataset4",
    "parquet_dir": "数据集/固化数据",
    "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / "Dataset4"),
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
    "output_filename": "dataset4_results.csv",
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
    parser = argparse.ArgumentParser(description="Run Dataset4 fixed-parquet experiment.")
    parser.add_argument("--info-sharing", choices=["without", "with"], default=config["info_sharing"])
    parser.add_argument("--smoke", action="store_true", help="Run tiny target/source limits with the same window logic.")
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override source/target epochs for lightweight checks.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run directory.")
    parser.add_argument("--repair-source-numeric-na", action="store_true")
    return parser.parse_args()


def _reference_result_columns() -> list[str]:
    return RESULT_SCHEMA_COLUMNS


def _align_results_to_reference_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and len(df.columns) > 0:
        template = align_d4_d6_result_records([{column: "" for column in df.columns}])
        return template.iloc[0:0].copy()
    return align_d4_d6_result_records(df.to_dict(orient="records"))


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
    config["repair_source_numeric_na"] = bool(args.repair_source_numeric_na)
    config["metric_protocol"] = load_default_metric_protocol(PROJECT_ROOT)
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

    knn_data = load_knn_results(config["knn_json_dir"], config["info_sharing"])
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise ValueError(f"D4 KNN payload requires two group_cols, got: {group_cols}")
    config["group_cols"] = list(group_cols)

    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
        info_sharing=config["info_sharing"],
        knn_payload=knn_data,
    )
    source_df, target_df = load_parquet_source_target(
        dataset_id=config["dataset_id"],
        parquet_dir=config["parquet_dir"],
        windows=windows,
        source_history_days=config["source_history_days"],
    )
    feature_info = resolve_knn_feature_columns(
        dataset_id=config["dataset_id"],
        information_sharing=config["info_sharing"],
        knn_root=SOLIDIFIED_KNN_ROOT,
        source_df=source_df,
        target_df=target_df,
        knn_payload=knn_data,
    )
    feature_cols = list(feature_info["selected_features"])
    if feature_info["feature_consistency_status"] != "aligned":
        print(
            "[FEATURE WARNING] solidified JSON features differ from runtime inferred features "
            f"json_only={feature_info.get('json_only_features', [])} "
            f"runtime_only={feature_info.get('runtime_only_features', [])} "
            f"using={feature_info.get('feature_source')}"
        )
    if config["repair_source_numeric_na"]:
        source_df, repair_diag = validate_feature_frame_finite(
            source_df,
            feature_cols,
            context="source_repair_numeric_na",
            dataset_id=config["dataset_id"],
            role="source",
            stage="source_repair_numeric_na",
            allow_fill=True,
        )
        config["source_numeric_na_repaired"] = bool(repair_diag.get("source_numeric_na_repaired", False))
        config["repaired_columns"] = repair_diag.get("repaired_columns", {})
    config.update(
        {
            "feature_source": feature_info.get("feature_source", ""),
            "knn_feature_mode": feature_info.get("knn_feature_mode", ""),
            "source_selection_feature_cols": list(feature_cols),
            "model_feature_cols": list(feature_cols),
            "feature_consistency_status": feature_info.get("feature_consistency_status", ""),
            "json_only_features": feature_info.get("json_only_features", []),
            "runtime_only_features": feature_info.get("runtime_only_features", []),
            "source_numeric_na_repaired": bool(config.get("source_numeric_na_repaired", False)),
            "repaired_columns": config.get("repaired_columns", {}),
        }
    )

    source_df = apply_runtime_source_domain_policy(source_df, knn_data, config)
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
