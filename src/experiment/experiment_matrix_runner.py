"""
模块12：完整实验矩阵运行器

职责：
1. 在模块10统一运行器基础上批量运行实验组合
2. 保存单组实验结果与总表
3. 保存实验矩阵配置快照
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from dataset_registry import list_dataset_names

try:
    from .experiment_runner import results_to_dataframe, run_all_experiments
except ImportError:
    from src.experiment.experiment_runner import results_to_dataframe, run_all_experiments

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None

from src.utils.console_reporter import print_global_progress
from src.utils.progress_tracker import ExperimentProgressTracker
from src.utils.runtime_control import apply_logging_level, log_level_name, set_verbose_mode


LOGGER_NAME = "experiment"


def _get_logger() -> logging.Logger:
    """获取统一日志器；若尚未初始化则使用默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level=log_level_name(), log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    apply_logging_level()
    return logger


def _sanitize_token(value: Any) -> str:
    """将任意值转换为文件名友好的安全 token。"""
    text = str(value).strip().lower()
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"-", "_"}:
            safe.append("_")
        else:
            safe.append("_")
    token = "".join(safe)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "na"


def _compress_weight_mode(weight_mode: str) -> str:
    """将 weight_mode 压缩为较短标识。"""
    mapping = {
        "inverse_distance": "inverse",
        "raw_distance": "raw",
    }
    return mapping.get(str(weight_mode), _sanitize_token(weight_mode))


def _compress_enabled_methods(enabled_methods: Sequence[str]) -> str:
    """将启用方法组合压缩为短标识。"""
    methods = list(enabled_methods)
    full = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]
    smoke = ["No-TL", "SS-TL", "MSML-TL", "MSML-TL-RFE"]

    if methods == full:
        return "allmethods"
    if methods == smoke:
        return "smoke"

    tokens = [_sanitize_token(m).replace("_tl", "") for m in methods]
    merged = "_".join(tokens)
    return merged[:60] if merged else "methods"


def build_experiment_matrix(
    dataset_names: Sequence[str],
    horizons: Sequence[int],
    source_counts: Sequence[int],
    weight_modes: Sequence[str],
    keep_ratios: Sequence[float],
    enabled_methods_options: Sequence[Sequence[str]],
) -> List[Dict[str, Any]]:
    """
    根据多维参数构建稳定顺序的实验组合矩阵。

    Args:
        dataset_names: 数据集名称列表。
        horizons: 预测步长列表。
        source_counts: source 数量（k）列表。
        weight_modes: 权重模式列表。
        keep_ratios: RFE 保留比例列表。
        enabled_methods_options: 方法组合列表。

    Returns:
        实验配置字典列表，每个元素对应一组完整参数。
    """
    matrix: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        for horizon in horizons:
            for k in source_counts:
                for weight_mode in weight_modes:
                    for keep_ratio in keep_ratios:
                        for enabled_methods in enabled_methods_options:
                            matrix.append(
                                {
                                    "dataset_name": str(dataset_name),
                                    "horizon": int(horizon),
                                    "k": int(k),
                                    "weight_mode": str(weight_mode),
                                    "keep_ratio": float(keep_ratio),
                                    "enabled_methods": list(enabled_methods),
                                }
                            )
    return matrix


def make_experiment_id(experiment_config: Dict[str, Any]) -> str:
    """
    为单个实验配置生成稳定、可读、文件名友好的实验 ID。

    示例：dataset1_h1_k3_inverse_rfe50_allmethods
    """
    dataset_token = _sanitize_token(experiment_config.get("dataset_name", "dataset"))
    horizon_token = f"h{int(experiment_config.get('horizon', 1))}"
    k_token = f"k{int(experiment_config.get('k', 1))}"
    weight_token = _compress_weight_mode(str(experiment_config.get("weight_mode", "inverse_distance")))

    keep_ratio = float(experiment_config.get("keep_ratio", 0.5))
    rfe_percent = int(round(keep_ratio * 100))
    rfe_token = f"rfe{rfe_percent}"

    methods = experiment_config.get("enabled_methods", [])
    methods_token = _compress_enabled_methods(list(methods) if isinstance(methods, Iterable) else [])

    return "_".join([dataset_token, horizon_token, k_token, weight_token, rfe_token, methods_token])


def run_single_experiment_config(
    experiment_config: Dict[str, Any],
    data_path_map: Dict[str, str],
    feature_cols: Sequence[str],
    learning_rate: float = 0.001,
    source_epochs: int = 2,
    target_epochs: int = 2,
    batch_size: int = 16,
    output_dir: str = "outputs/matrix_runs",
    verbose_mode: str = "summary",
) -> Dict[str, Any]:
    """
    运行单组实验配置，保存单次结果 CSV，并返回运行信息。

    Args:
        experiment_config: 单组实验配置。
        data_path_map: 数据集到路径的映射。
        feature_cols: 统一特征列。
        learning_rate: 学习率。
        source_epochs: source 训练轮数。
        target_epochs: target 训练轮数。
        batch_size: 批大小。
        output_dir: 单次结果根目录。

    Returns:
        {
          "experiment_id": str,
          "experiment_config": dict,
          "results_df": pd.DataFrame,
          "result_csv_path": str,
        }
    """
    logger = _get_logger()

    dataset_name = str(experiment_config["dataset_name"])
    if dataset_name not in data_path_map:
        raise ValueError(f"dataset_name '{dataset_name}' missing in data_path_map")

    experiment_id = make_experiment_id(experiment_config)
    logger.info("[run_single_experiment_config] Start. experiment_id=%s", experiment_id)

    experiment_results = run_all_experiments(
        dataset_name=dataset_name,
        data_path=data_path_map[dataset_name],
        feature_cols=list(feature_cols),
        k=int(experiment_config["k"]),
        horizon=int(experiment_config["horizon"]),
        window_size=10,
        weight_mode=str(experiment_config["weight_mode"]),
        estimator_name="random_forest",
        keep_ratio=float(experiment_config["keep_ratio"]),
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
        enabled_methods=list(experiment_config["enabled_methods"]),
        verbose_mode=verbose_mode,
        show_method_progress=False,
    )

    results_df = results_to_dataframe(experiment_results).copy()
    results_df["experiment_id"] = experiment_id
    results_df["dataset_name"] = dataset_name
    results_df["horizon"] = int(experiment_config["horizon"])
    results_df["k"] = int(experiment_config["k"])
    results_df["weight_mode"] = str(experiment_config["weight_mode"])
    results_df["keep_ratio"] = float(experiment_config["keep_ratio"])
    results_df["enabled_methods"] = "|".join(list(experiment_config["enabled_methods"]))
    results_df["prediction_shape"] = results_df["prediction_shape"].astype(str)

    config_cols = [
        "experiment_id",
        "dataset_name",
        "horizon",
        "k",
        "weight_mode",
        "keep_ratio",
        "enabled_methods",
    ]
    other_cols = [c for c in results_df.columns if c not in config_cols]
    results_df = results_df[config_cols + other_cols]

    experiment_dir = Path(output_dir) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    result_csv_path = experiment_dir / "results.csv"
    results_df.to_csv(result_csv_path, index=False, encoding="utf-8")

    logger.info(
        "[run_single_experiment_config] Finished. experiment_id=%s rows=%d csv=%s",
        experiment_id,
        len(results_df),
        result_csv_path,
    )

    return {
        "experiment_id": experiment_id,
        "experiment_config": dict(experiment_config),
        "results_df": results_df,
        "result_csv_path": str(result_csv_path),
    }


def concat_experiment_results(results_df_list: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    合并多组实验结果为总表。

    Args:
        results_df_list: 多个单组实验结果 DataFrame。

    Returns:
        合并后的 DataFrame；若无有效输入则返回空表。
    """
    valid = [df for df in results_df_list if isinstance(df, pd.DataFrame) and not df.empty]
    if not valid:
        return pd.DataFrame()

    master_df = pd.concat(valid, axis=0, ignore_index=True, sort=False)

    config_cols = [
        "experiment_id",
        "dataset_name",
        "horizon",
        "k",
        "weight_mode",
        "keep_ratio",
        "enabled_methods",
    ]
    front = [c for c in config_cols if c in master_df.columns]
    rest = [c for c in master_df.columns if c not in front]
    return master_df[front + rest]


def save_master_results(master_df: pd.DataFrame, output_path: str) -> None:
    """
    保存实验总表 CSV。

    Args:
        master_df: 总结果 DataFrame。
        output_path: 输出路径。
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_path, index=False, encoding="utf-8")


def save_experiment_matrix_snapshot(matrix_configs: Sequence[Dict[str, Any]], output_path: str) -> None:
    """
    保存实验矩阵配置快照 JSON。

    Args:
        matrix_configs: 实验矩阵配置列表。
        output_path: 快照 JSON 路径。
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_experiments": len(matrix_configs),
        "matrix_configs": list(matrix_configs),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_experiment_matrix(
    data_path_map: Dict[str, str],
    feature_cols: Sequence[str],
    dataset_names: Sequence[str],
    horizons: Sequence[int],
    source_counts: Sequence[int],
    weight_modes: Sequence[str],
    keep_ratios: Sequence[float],
    enabled_methods_options: Sequence[Sequence[str]],
    learning_rate: float = 0.001,
    source_epochs: int = 2,
    target_epochs: int = 2,
    batch_size: int = 16,
    output_dir: str = "outputs/matrix_runs",
    master_csv_path: str = "outputs/matrix_runs/master_results.csv",
    snapshot_path: str = "outputs/matrix_runs/matrix_snapshot.json",
    verbose_mode: str = "summary",
) -> Dict[str, Any]:
    """
    运行完整实验矩阵并输出总表和配置快照。

    失败策略：任一组合失败即抛错停止。
    """
    set_verbose_mode(verbose_mode)
    logger = _get_logger()

    matrix_configs = build_experiment_matrix(
        dataset_names=dataset_names,
        horizons=horizons,
        source_counts=source_counts,
        weight_modes=weight_modes,
        keep_ratios=keep_ratios,
        enabled_methods_options=enabled_methods_options,
    )

    total = len(matrix_configs)
    tracker = ExperimentProgressTracker(total_runs=max(1, total))
    logger.info("[run_experiment_matrix] Start. total_experiments=%d", total)

    all_results: List[pd.DataFrame] = []

    for idx, cfg in enumerate(matrix_configs, start=1):
        exp_id = make_experiment_id(cfg)
        dataset_name = str(cfg.get("dataset_name", "N/A"))
        method_name = "|".join(list(cfg.get("enabled_methods", [])))
        snapshot = tracker.update(current_dataset=dataset_name, current_method=method_name)
        if str(verbose_mode).lower() == "summary":
            print_global_progress(
                current=idx,
                total=total,
                dataset_name=dataset_name,
                method_name=method_name,
                eta_seconds=snapshot.eta_seconds,
            )
        message = f"Running experiment {idx}/{total}: {exp_id}"
        logger.info("[run_experiment_matrix] %s", message)
        if str(verbose_mode).lower() == "full":
            print(message)

        one = run_single_experiment_config(
            experiment_config=cfg,
            data_path_map=data_path_map,
            feature_cols=feature_cols,
            learning_rate=learning_rate,
            source_epochs=source_epochs,
            target_epochs=target_epochs,
            batch_size=batch_size,
            output_dir=output_dir,
            verbose_mode=verbose_mode,
        )
        all_results.append(one["results_df"])
        tracker.mark_completed()

    master_df = concat_experiment_results(all_results)
    save_master_results(master_df, master_csv_path)
    save_experiment_matrix_snapshot(matrix_configs, snapshot_path)

    logger.info(
        "[run_experiment_matrix] Finished. total=%d master_rows=%d",
        total,
        len(master_df),
    )

    return {
        "num_experiments": total,
        "master_results_path": str(master_csv_path),
        "snapshot_path": str(snapshot_path),
        "master_df": master_df,
    }


def build_smoke_test_matrix() -> Dict[str, Any]:
    """
    构建最小可运行 smoke test 矩阵配置。

    Returns:
        包含 dataset_names/horizons/source_counts/weight_modes/keep_ratios/enabled_methods_options 的字典。
    """
    return {
        "dataset_names": ["Dataset1"],
        "horizons": [1],
        "source_counts": [3],
        "weight_modes": ["inverse_distance"],
        "keep_ratios": [0.5],
        "enabled_methods_options": [["No-TL", "SS-TL", "MSML-TL", "MSML-TL-RFE"]],
    }


def build_full_matrix_default() -> Dict[str, Any]:
    """
    构建完整实验矩阵默认配置（仅生成配置，不保证当前环境可直接执行）。

    Returns:
        默认完整矩阵参数字典。
    """
    return {
        "dataset_names": list_dataset_names(),
        "horizons": [1, 5],
        "source_counts": [1, 3],
        "weight_modes": ["inverse_distance", "raw_distance"],
        "keep_ratios": [0.5, 0.8],
        "enabled_methods_options": [["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]],
    }
