"""
数据预处理流水线测试脚本。

该脚本仅验证模块2的数据处理流程，不涉及模型和训练。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dataset_registry import get_default_dataset_path, normalize_dataset_name
from config import Config
from environment import setup_logging

from data_preprocessing import (
    build_tabular_sequence,
    build_source_target_split,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)


def _resolve_dataset_csv(config: Config) -> tuple[str, str]:
    """根据当前配置解析数据集名称和CSV路径。"""
    canonical_name = normalize_dataset_name(config.dataset.name)
    ds_path = str(config.dataset.path).strip()

    if ds_path:
        candidate = Path(ds_path)
        if candidate.is_dir():
            if canonical_name == "Dataset1":
                return canonical_name, str(candidate / "train.csv")
            if canonical_name == "Dataset3":
                return canonical_name, str(candidate / "train ross.csv")
        if candidate.is_file():
            return canonical_name, str(candidate)

    return canonical_name, get_default_dataset_path(canonical_name)


def _first_window_alignment_info(
    split_df: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> tuple[pd.Timestamp, pd.Timestamp, float, float]:
    """在组内恢复第一个有效窗口与标签位置，用于偏移校验。"""
    ordered = split_df.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)
    for _, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) >= window_size + horizon:
            last_idx = window_size - 1
            label_idx = last_idx + horizon
            return (
                g.loc[last_idx, "date"],
                g.loc[label_idx, "date"],
                float(g.loc[last_idx, "sales"]),
                float(g.loc[label_idx, "sales"]),
            )
    raise ValueError("No valid group found for alignment check under this horizon.")


def main() -> None:
    """运行完整数据处理流程并打印关键统计信息。"""
    setup_logging(log_level="INFO", log_file=None)
    config = Config(config_file="config.yaml", supply_chain_file="supply_chain.yaml", verbose=True)
    window_size = 10
    horizon = 1

    dataset_name, csv_path = _resolve_dataset_csv(config)
    df = load_dataset(dataset_name=dataset_name, data_path=csv_path)
    df = extract_datetime_features(df)

    source_df, target_df = build_source_target_split(df, config)

    # 按目标域进行拆分与归一化，更贴近后续新产品预测流程
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)
    train_df, val_df, test_df, _, feature_columns = normalize_features(train_df, val_df, test_df)

    # 此处使用 temporal_split_by_ratio_or_dates 返回的真实训练集变量。
    train_split_for_checks = train_df

    train_x, train_y = build_tabular_sequence(train_split_for_checks, horizon=horizon, window_size=window_size)
    val_x, val_y = build_tabular_sequence(val_df, horizon=horizon, window_size=window_size)
    test_x, test_y = build_tabular_sequence(test_df, horizon=horizon, window_size=window_size)

    # horizon 更大时，可用 end_idx 范围变窄，因此样本数通常比 horizon=1 更少。
    horizon_5 = 5
    train_x_h5, train_y_h5 = build_tabular_sequence(train_split_for_checks, horizon=horizon_5, window_size=window_size)

    train_x = to_cnn_tensor(train_x)
    val_x = to_cnn_tensor(val_x)
    test_x = to_cnn_tensor(test_x)
    train_x_h5 = to_cnn_tensor(train_x_h5)

    num_features = train_x.shape[2] if train_x.ndim == 3 else len(feature_columns)

    print("Data Loaded Successfully")
    print(f"Source Samples: {len(source_df)}")
    print(f"Target Samples: {len(target_df)}")
    print(f"window_size: {window_size}")
    print(f"horizon: {horizon}")
    print(f"num_features: {num_features}")
    print(f"Train X Shape: {train_x.shape}")
    print(f"Train y Shape: {train_y.shape}")
    print(f"Validation X Shape: {val_x.shape}")
    print(f"Validation y Shape: {val_y.shape}")
    print(f"Test X Shape: {test_x.shape}")
    print(f"Test y Shape: {test_y.shape}")
    print(f"Horizon=5 Train X Shape: {train_x_h5.shape}")
    print(f"Horizon=5 Train y Shape: {train_y_h5.shape}")

    alignment_horizon = horizon_5 if len(train_y_h5) > 0 else horizon
    last_date, label_date, last_sales, expected_label_sales = _first_window_alignment_info(
        train_split_for_checks,
        window_size=window_size,
        horizon=alignment_horizon,
    )
    actual_y_value = float(train_y_h5[0]) if alignment_horizon == horizon_5 else float(train_y[0])
    gap_days = int((label_date - last_date).days)
    print("Window-Label Alignment Check")
    print(f"Last date in first window: {last_date.strftime('%Y-%m-%d')}")
    print(f"Label date: {label_date.strftime('%Y-%m-%d')}")
    print(f"Horizon: {alignment_horizon}")
    print(f"Date gap (days): {gap_days}")
    print(f"Last sales in first window: {last_sales}")
    print(f"Expected label sales: {expected_label_sales}")
    print(f"Actual y value: {actual_y_value}")

    # 日期偏移与标签值必须和组内 t+horizon 位置一致。
    assert gap_days == alignment_horizon, f"Date gap mismatch: expected={alignment_horizon}, got={gap_days}"
    assert abs(actual_y_value - expected_label_sales) < 1e-6, "Label value mismatch."


if __name__ == "__main__":
    main()
