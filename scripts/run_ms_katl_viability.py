"""MS-KATL-Lite viability 入口（extended 轨道）。

范围：1 个数据集（D4）× 1 个 target × 1 个 horizon × 1 个 seed，
对比 {MSML-TL, MS-KATL-Lite}。结果标记为 extended（smoke track），
不进入正式验收产物。

用法：
    python scripts/run_ms_katl_viability.py                # H=1, seed=42, 50 epochs
    python scripts/run_ms_katl_viability.py --epochs 3     # 轻量联调
    python scripts/run_ms_katl_viability.py --gamma 0.0    # gamma 扫描
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SOURCE_HISTORY_DAYS, SOLIDIFIED_KNN_ROOT

import tf_compat  # noqa: F401  (must be imported before tensorflow/keras)

import pandas as pd

from src.utils.run_utils import create_run_dir, reserve_new_output_dir
from src.utils.entity_experiment import run_single_entity_experiment
from src.utils.parquet_data_loader import (
    load_parquet_source_target,
    read_dataset_windows,
    load_knn_results,
)
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    load_default_metric_protocol,
    validate_runtime_target_domain,
)
from src.utils.knn_feature_loader import resolve_knn_feature_columns
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.transformation_identity import validate_runtime_feature_contract
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.formal_input_paths import resolve_formal_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config = {
    "use_parquet": True,
    "dataset_id": 4,
    "dataset_name": "Dataset4",
    "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / "Dataset4"),
    "info_sharing": "without",
    "source_history_days": SOURCE_HISTORY_DAYS,
    "entity_col": "entity_id",
    "enabled_methods": ["MSML-TL", "MS-KATL-Lite"],
    "output_filename": "ms_katl_viability_results.csv",
    "horizon": 1,
    "window_size": 10,
    "learning_rate": 0.0001,
    "source_epochs": 50,
    "target_epochs": 50,
    "batch_size": 16,
    "random_state": 42,
    "source_count": 3,
    # viability 结果走 extended/smoke 轨道，不进入正式验收
    "smoke": True,
    "extended_methods_enabled": True,
    "smoke_target_limit": 1,
    "smoke_source_limit": 3,
    "ms_katl_gamma": 0.5,
    "ms_katl_tau_k": 1.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-KATL-Lite viability run (extended track).")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46], default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override source/target epochs for lightweight checks.")
    parser.add_argument("--gamma", type=float, default=0.5, help="KNN prior strength (choose from {0, 0.5, 1.0}).")
    parser.add_argument("--tau-k", type=float, default=1.0)
    parser.add_argument("--target-limit", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formal_paths = resolve_formal_dataset_paths(4, repository_root=PROJECT_ROOT)
    config["formal_source_path"] = str(formal_paths.source_path)
    config["formal_target_path"] = str(formal_paths.target_path)
    config["horizon"] = int(args.horizon)
    config["random_state"] = int(args.seed)
    config["smoke_target_limit"] = int(args.target_limit)
    config["ms_katl_gamma"] = float(args.gamma)
    config["ms_katl_tau_k"] = float(args.tau_k)
    if args.epochs is not None:
        config["source_epochs"] = int(args.epochs)
        config["target_epochs"] = int(args.epochs)
    set_protocol_seed(int(args.seed), include_frameworks=True)
    config["metric_protocol"] = load_default_metric_protocol(PROJECT_ROOT)

    label = f"MS_KATL_viability_D4_h{config['horizon']}_s{config['random_state']}_g{config['ms_katl_gamma']}"
    if args.output_dir is None:
        run_dir = create_run_dir(PROJECT_ROOT, label=label)
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        reserve_new_output_dir(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
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
        source_path=formal_paths.source_path,
        target_path=formal_paths.target_path,
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
    validate_runtime_feature_contract(
        "D4",
        model_feature_cols=feature_cols,
        knn_feature_cols=get_experiment_protocol("D4").knn_feature_columns,
    )
    config.update(
        {
            "feature_source": feature_info.get("feature_source", ""),
            "knn_feature_mode": feature_info.get("knn_feature_mode", ""),
            "source_selection_feature_cols": list(feature_cols),
            "model_feature_cols": list(feature_cols),
            "feature_consistency_status": feature_info.get("feature_consistency_status", ""),
            "json_only_features": feature_info.get("json_only_features", []),
            "runtime_only_features": feature_info.get("runtime_only_features", []),
        }
    )

    source_df = apply_runtime_source_domain_policy(source_df, knn_data, config)
    target_entity_keys = list(knn_data["results"].keys())
    validate_runtime_target_domain(target_df, target_entity_keys, knn_data, config)
    target_entity_keys = target_entity_keys[: int(config["smoke_target_limit"])]
    selected_source_entities = {
        str(item.get("source_entity"))
        for values in knn_data["results"].values()
        for item in values[: int(config["smoke_source_limit"])]
        if isinstance(item, dict) and item.get("source_entity") is not None
    }
    if selected_source_entities:
        source_df = source_df[
            source_df[config["entity_col"]].astype(str).isin(selected_source_entities)
        ].copy()

    observed_start = target_df.attrs.get(
        "knn_observed_start",
        target_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_df["date"], errors="raise").min(),
        ),
    )
    prepared_pool = prepare_daily_sequence_pool(
        source_df,
        group_cols=tuple(str(column) for column in config["group_cols"]),
        observed_start=observed_start,
        feature_cols=get_experiment_protocol(4).knn_feature_columns,
        capture_source_observed_frame_digest=True,
    )

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
            prepared_pool=prepared_pool,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = results_dir / config["output_filename"]
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Results saved to {out_path}")
    show_cols = [c for c in ("target_entity_key", "method", "smape", "rmse", "error") if c in df.columns]
    print(df[show_cols].to_string())


if __name__ == "__main__":
    main()
