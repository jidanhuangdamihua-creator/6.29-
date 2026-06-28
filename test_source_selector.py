"""
Test script for Module 5: Similar Source Selection

最小可运行测试：验证 top-k source、distance、weight 输出。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config
from environment import setup_logging
from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from source_selector import SourceSelector


def _choose_target_short_history(target_df: pd.DataFrame, history_length: int = 30) -> pd.DataFrame:
    """从 target 中选取一个序列的短期历史片段。"""
    ordered = target_df.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)
    first_key = ordered[["entity_id", "item_id"]].drop_duplicates().iloc[0]

    seq_df = ordered[
        (ordered["entity_id"] == first_key["entity_id"]) &
        (ordered["item_id"] == first_key["item_id"])
    ].copy()

    if len(seq_df) > history_length:
        seq_df = seq_df.tail(history_length).copy()

    return seq_df


def _print_results(title: str, results: list[dict]) -> None:
    """打印 top-k 结果与权重和。"""
    print(f"Top-K Sources ({title}):")
    for i, row in enumerate(results, start=1):
        print(
            f"{i}. source_key={row['source_key']} "
            f"distance={row['distance']:.4f} "
            f"weight={row['weight']:.4f}"
        )
    weight_sum = sum(float(r["weight"]) for r in results)
    print(f"Weight Sum: {weight_sum:.4f}")


def main() -> None:
    """运行 Source Selector 最小测试。"""
    setup_logging(log_level="INFO", log_file=None)

    # 1) 使用模块2数据加载逻辑读取 Dataset1
    config = Config(config_file="config.yaml", supply_chain_file="supply_chain.yaml", verbose=True)
    csv_path = str(Path(str(config.dataset.path)) / "train.csv")

    df = load_dataset(dataset_name="Dataset1", data_path=csv_path)
    df = extract_datetime_features(df)

    # 2) 获取 target_df / source_df
    source_df, target_df = build_source_target_split(df, config)

    # 3) 使用 target 的短期历史片段作为输入
    target_short_df = _choose_target_short_history(target_df, history_length=30)

    # 4) 设置特征列（避免将分组键作为相似性特征）
    candidate_cols = ["sales", "year", "month", "week", "day"]
    feature_cols = [c for c in candidate_cols if c in source_df.columns and c in target_short_df.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns found for source selection.")

    selector = SourceSelector()

    # 5) inverse_distance
    inverse_results = selector.select_top_k_sources(
        target_df=target_short_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=3,
        weight_mode="inverse_distance",
    )

    print("Source Selection Completed")
    print(f"Meta: {inverse_results['meta']}")
    print(f"Meta weight_mode: {inverse_results['meta']['weight_mode']}")
    print(f"Meta target_signature_dim: {inverse_results['meta']['target_signature_dim']}")
    print(f"Meta feature_cols: {inverse_results['meta']['feature_cols']}")
    _print_results("inverse_distance", inverse_results["sources"])
    print()

    # 6) raw_distance
    raw_results = selector.select_top_k_sources(
        target_df=target_short_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=3,
        weight_mode="raw_distance",
    )

    print(f"Meta: {raw_results['meta']}")
    print(f"Meta weight_mode: {raw_results['meta']['weight_mode']}")
    print(f"Meta target_signature_dim: {raw_results['meta']['target_signature_dim']}")
    print(f"Meta feature_cols: {raw_results['meta']['feature_cols']}")
    _print_results("raw_distance", raw_results["sources"])


if __name__ == "__main__":
    main()
