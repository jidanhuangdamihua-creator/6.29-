"""
只读审计脚本：验证 Dataset1/Dataset2 KNN source selection 未被意外破坏。
不修改任何文件。
"""
import sys
import logging
import numpy as np
import pandas as pd

from data_preprocessing import (
    load_dataset,
    extract_datetime_features,
    build_source_target_split,
    KNN_FEATURE_MODE_PAPER_NO_IDS,
    KNN_FEATURE_MODE_SALES_ONLY,
    KNN_FEATURE_MODE_ALL_NUMERIC,
)
from source_selector import SourceSelector

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

FEATURE_MODES = [
    (KNN_FEATURE_MODE_PAPER_NO_IDS, "paper_no_ids"),
    (KNN_FEATURE_MODE_SALES_ONLY, "sales_only"),
    (KNN_FEATURE_MODE_ALL_NUMERIC, "all_numeric"),
]


def test_dataset(ds_name, path, expected_target_item=10):
    print(f"\n{'='*80}")
    print(f"  Dataset: {ds_name} — KNN Source Selection 审计")
    print(f"{'='*80}")

    raw_df = load_dataset(ds_name, path)
    processed = extract_datetime_features(raw_df)
    cfg = {
        "dataset_name": ds_name,
        "paper_reproduction": {
            "strict_paper_mode": True,
            "paper_strict_mode": True,
            "paper_split_protocol": {
                "target_observed_window_days": 30,
                "target_forecast_window_days": 180,
            },
        },
    }
    source_df, target_df = build_source_target_split(processed, cfg)

    print(f"  Source rows: {len(source_df)}, unique items: {sorted(source_df['item_id'].unique())[:15]}...")
    print(f"  Target rows: {len(target_df)}, target item: {sorted(target_df['item_id'].unique())}")
    print(f"  Target split_mode: {target_df.attrs.get('split_mode')}")
    print(f"  Target split_config: {target_df.attrs.get('split_config')}")

    # Source columns (numeric)
    src_num = [c for c in source_df.columns if pd.api.types.is_numeric_dtype(source_df[c])]
    tgt_num = [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])]
    print(f"  Source numeric cols: {src_num}")
    print(f"  Target numeric cols: {tgt_num}")

    selector = SourceSelector()
    for mode_name, mode_label in FEATURE_MODES:
        try:
            result = selector.select_top_k_sources(
                target_df=target_df,
                source_df=source_df,
                feature_cols=[],
                k=3,
                group_cols=("entity_id", "item_id"),
                weight_mode="inverse_distance",
                debug_mode=False,
                include_sales_in_knn=True,
                knn_representation="paper_observed_sequence",
                knn_feature_mode=mode_name,
            )
            meta = result.get("meta", {})
            features = meta.get("feature_cols", [])
            top3 = []
            for s in result.get("sources", [])[:3]:
                key = s["source_key"]
                sn = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else str(key)
                top3.append(sn)
            print(f"  [{mode_label:<15s}] features=({len(features)}) {features}")
            print(f"  [{mode_label:<15s}] top-3: {top3}")
        except Exception as e:
            print(f"  [{mode_label:<15s}] ❌ ERROR: {e}")

    print()


def main():
    print("Dataset1/Dataset2 KNN Source Selection — Read-Only Audit")
    print("检查 KNN feature mode 变更是否影响非 Dataset3 数据集")

    test_dataset("Dataset1", "数据集/Dataset1-Challenge.csv")
    test_dataset("Dataset2", "数据集/Dataset2-pasta.csv")


if __name__ == "__main__":
    main()
