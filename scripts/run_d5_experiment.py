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

from src.utils.run_utils import create_run_dir, reserve_new_output_dir
from src.utils.entity_experiment import run_single_entity_experiment
from src.utils.parquet_data_loader import (
    ParquetSourceTargetLoad,
    expected_target_dates_from_windows,
    load_parquet_source_target_with_diagnostics,
    read_dataset_windows,
    load_knn_results,
)
from src.utils.d5_calendar_reconstruction import load_d5_authorities
from src.utils.d4_d6_runtime import apply_runtime_source_domain_policy, load_default_metric_protocol
from src.utils.finite_diagnostics import validate_feature_frame_finite
from src.utils.knn_feature_loader import resolve_knn_feature_columns
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.formal_input_paths import (
    require_explicit_formal_paths,
    resolve_formal_dataset_paths,
)
from src.utils.run_artifacts import publish_formal_cell_output_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config = {
    "use_parquet": True,
    "dataset_id": 5,
    "dataset_name": "Dataset5",
    "raw_dir": "数据集/原始数据/Dataset 5Favorita",
    "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / "Dataset5"),
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
    "output_filename": "dataset5_results.csv",
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
    parser = argparse.ArgumentParser(description="Run Dataset5 fixed-parquet experiment.")
    parser.add_argument("--info-sharing", choices=["without", "with"], default=config["info_sharing"])
    parser.add_argument("--smoke", action="store_true", help="Run tiny target/source limits with the same window logic.")
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--target-keys", nargs="+", default=None)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override source/target epochs for lightweight checks.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run directory.")
    parser.add_argument("--repair-source-numeric-na", action="store_true")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46], default=42)
    parser.add_argument("--formal-source-path", type=Path, default=None)
    parser.add_argument("--formal-target-path", type=Path, default=None)
    return parser.parse_args()


def _reference_result_columns() -> list[str]:
    return RESULT_SCHEMA_COLUMNS


def _align_results_to_reference_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty and len(df.columns) > 0:
        template = align_d4_d6_result_records([{column: "" for column in df.columns}])
        return template.iloc[0:0].copy()
    return align_d4_d6_result_records(df.to_dict(orient="records"))


def load_d5_runtime_inputs(
    *,
    raw_dir: Path,
    source_path: Path,
    target_path: Path,
    windows: dict[str, object],
    source_history_days: int,
) -> ParquetSourceTargetLoad:
    """Load every D5 authority once, then reconstruct from fixed window dates."""
    authorities = load_d5_authorities(Path(raw_dir), use_holidays=True)
    expected_dates = expected_target_dates_from_windows(windows)
    return load_parquet_source_target_with_diagnostics(
        dataset_id=5,
        source_path=source_path,
        target_path=target_path,
        windows=windows,
        source_history_days=source_history_days,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )


def main() -> None:
    args = _parse_args()
    if (args.formal_source_path is None) != (args.formal_target_path is None):
        raise SystemExit("FORMAL_INPUT_RESOLVER_PARITY_MISMATCH dataset=5")
    if args.formal_source_path is None:
        formal_paths = resolve_formal_dataset_paths(5, repository_root=PROJECT_ROOT)
    else:
        formal_paths = require_explicit_formal_paths(
            5,
            source_path=args.formal_source_path,
            target_path=args.formal_target_path,
            repository_root=PROJECT_ROOT,
        )
    config["formal_source_path"] = str(formal_paths.source_path)
    config["formal_target_path"] = str(formal_paths.target_path)
    config["info_sharing"] = str(args.info_sharing)
    config["smoke"] = bool(args.smoke)
    if args.target_limit is not None:
        config["smoke_target_limit"] = int(args.target_limit)
    if args.target_keys is not None:
        config["target_keys"] = [str(key) for key in args.target_keys]
    if args.source_limit is not None:
        config["smoke_source_limit"] = int(args.source_limit)
    if args.epochs is not None:
        config["source_epochs"] = int(args.epochs)
        config["target_epochs"] = int(args.epochs)
    config["repair_source_numeric_na"] = bool(args.repair_source_numeric_na)
    config["horizon"] = int(args.horizon)
    config["random_state"] = int(args.seed)
    set_protocol_seed(int(args.seed), include_frameworks=True)
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
        reserve_new_output_dir(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    knn_data = load_knn_results(config["knn_json_dir"], config["info_sharing"])
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise ValueError(f"D5 KNN payload requires two group_cols, got: {group_cols}")
    config["group_cols"] = list(group_cols)

    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
        info_sharing=config["info_sharing"],
        knn_payload=knn_data,
    )
    runtime_inputs = load_d5_runtime_inputs(
        raw_dir=root / config["raw_dir"],
        source_path=formal_paths.source_path,
        target_path=formal_paths.target_path,
        windows=windows,
        source_history_days=config["source_history_days"],
    )
    source_df = runtime_inputs.source_df
    target_df = runtime_inputs.target_df
    if runtime_inputs.calendar_reconstruction is None:
        raise AssertionError("D5 runtime loader did not return reconstruction diagnostics")
    config["d5_calendar_reconstruction"] = runtime_inputs.calendar_reconstruction.to_dict()
    config["d5_source_history_validation_path"] = str(
        source_df.attrs.get("source_history_validation_path", "runtime_reconstruction")
    )
    config["d5_precomputed_source_history_active"] = (
        config["d5_source_history_validation_path"] == "precomputed_static_file"
    )
    print(
        "[D5 SOURCE HISTORY] "
        f"validation_path={config['d5_source_history_validation_path']} "
        f"precomputed_active={str(config['d5_precomputed_source_history_active']).lower()} "
        f"rows={len(source_df)}"
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
    if config.get("target_keys"):
        requested_target_keys = [str(key) for key in config["target_keys"]]
        missing_target_keys = [key for key in requested_target_keys if key not in knn_data["results"]]
        if missing_target_keys:
            raise ValueError(f"--target-keys not found in D5 KNN results: {missing_target_keys}")
        target_entity_keys = requested_target_keys
    elif config["smoke"]:
        target_entity_keys = target_entity_keys[: int(config["smoke_target_limit"])]

    if config["smoke"] or args.source_limit is not None:
        selected_source_entities = {
            str(item.get("source_entity"))
            for key in target_entity_keys
            for item in knn_data["results"].get(key, [])[: int(config["smoke_source_limit"])]
            if isinstance(item, dict) and item.get("source_entity") is not None
        }
        if not selected_source_entities and config["smoke"]:
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
    if config["smoke"]:
        df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        canonical_targets = tuple(
            dict.fromkeys(df["target_entity_key"].astype(str))
        )
        publish_formal_cell_output_frame(
            df,
            stable_path=out_path,
            dataset_id=config["dataset_id"],
            mode=config["info_sharing"],
            targets=canonical_targets,
            horizon=config["horizon"],
            seed=config["random_state"],
            project_root=PROJECT_ROOT,
        )
    print(f"Results saved to {out_path}")
    print(df[["target_entity_key", "method", "smape", "rmse"]].to_string())


if __name__ == "__main__":
    main()
