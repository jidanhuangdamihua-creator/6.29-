"""
只读审计：Dataset1 No Information Sharing KNN 根因分析
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
)
from source_selector import SourceSelector

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


def main():
    print("=" * 90)
    print("  Dataset1 No Information Sharing — KNN 根因审计")
    print("=" * 90)

    # ── Step 1: 加载数据 ────────────────────────────────────
    raw_df = load_dataset("Dataset1", "数据集/Dataset1-Challenge.csv")
    processed = extract_datetime_features(raw_df)
    cfg = {
        "dataset_name": "Dataset1",
        "paper_reproduction": {
            "strict_paper_mode": True,
            "paper_strict_mode": True,
            "paper_split_protocol": {
                "target_observed_window_days": 30,
                "target_forecast_window_days": 180,
            },
        },
    }
    source_df_full, target_df = build_source_target_split(processed, cfg)

    # ── Step 2: 检查 build_source_target_split 产出的 source pool ──
    src_entities = sorted(source_df_full["entity_id"].unique())
    src_items = sorted(source_df_full["item_id"].unique())
    print(f"\n📂 build_source_target_split 源池:")
    print(f"   entities: {src_entities}")
    print(f"   items:    {src_items}")
    print(f"   rows:     {len(source_df_full)}")

    # 按 entity 分组统计
    for ent in src_entities:
        ent_df = source_df_full[source_df_full["entity_id"] == ent]
        ent_items = sorted(ent_df["item_id"].unique())
        print(f"   entity={ent}: {len(ent_df)} rows, items={ent_items}")

    # ── Step 3: 检查 without_information_sharing 应有的过滤 ──
    strict_protocol = {
        "Dataset1": {
            "target_entity_id": 1,
            "target_item_id": 10,
            "allowed_entities": [1, 2, 3],
            "source_item_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "without_information_sharing_scope": "same_store",
        }
    }
    ds_cfg = strict_protocol["Dataset1"]
    target_entity = ds_cfg["target_entity_id"]
    print(f"\n🔍 without_information_sharing 约束:")
    print(f"   scope:          {ds_cfg['without_information_sharing_scope']}")
    print(f"   target_entity:  {target_entity}")
    print(f"   expected source: Store {target_entity}, Items 1-9")

    # 正确过滤后的 source pool
    source_df_filtered = source_df_full[
        source_df_full["entity_id"].astype(int) == target_entity
    ].copy()
    print(f"\n📂 过滤后源池 (only entity={target_entity}):")
    print(f"   entities: {sorted(source_df_filtered['entity_id'].unique())}")
    print(f"   items:    {sorted(source_df_filtered['item_id'].unique())}")
    print(f"   rows:     {len(source_df_filtered)}")

    # ── Step 4: 检查 target ──────────────────────────────────
    tgt_entities = sorted(target_df["entity_id"].unique())
    tgt_items = sorted(target_df["item_id"].unique())
    print(f"\n🎯 Target:")
    print(f"   entity: {tgt_entities}")
    print(f"   item:   {tgt_items}")
    print(f"   rows:   {len(target_df)}")
    print(f"   split_mode: {target_df.attrs.get('split_mode')}")
    split_cfg = target_df.attrs.get("split_config", {})
    print(f"   split_config: train_days={split_cfg.get('train_days')}, "
          f"val_days={split_cfg.get('val_days')}, test_days={split_cfg.get('test_days')}")

    # ── Step 5: 用错误源池（未过滤）跑 KNN ──────────────────
    print(f"\n{'=' * 90}")
    print("  ❌ 错误源池 KNN (包含 Store 2,3 的 items) — 之前 smoke test")
    print(f"{'=' * 90}")
    selector = SourceSelector()
    r_wrong = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df_full,  # ← 未按 entity 过滤
        feature_cols=[],
        k=9,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        knn_representation="paper_observed_sequence",
        knn_feature_mode="paper_available_features_no_ids",
    )
    print(f"   features: {r_wrong['meta']['feature_cols']}")
    print(f"   observed_window_rows: {r_wrong['meta'].get('observed_window_rows')}")
    print(f"   observed_window_dates: {r_wrong['meta'].get('observed_window_unique_dates')}")
    print(f"   {'Rank':<6}{'Source':<30}{'Distance':<16}{'Weight':<16}")
    print(f"   {'-' * 68}")
    for s in r_wrong["sources"]:
        key = s["source_key"]
        store = key[0] if isinstance(key, (list, tuple)) else "?"
        item = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else "?"
        print(f"   {s['source_rank']:<6}Store {str(store):>3} Item {str(item):>3}   "
              f"{s['distance']:<16.4f}{s['weight']:<16.6f}")

    # ── Step 6: 用正确源池（仅 Store 1）跑 KNN ──────────────
    print(f"\n{'=' * 90}")
    print("  ✅ 正确源池 KNN (only Store 1 Items 1-9) — paper aligned")
    print(f"{'=' * 90}")
    r_correct = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df_filtered,  # ← 已按 entity=1 过滤
        feature_cols=[],
        k=9,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        knn_representation="paper_observed_sequence",
        knn_feature_mode="paper_available_features_no_ids",
    )
    print(f"   features: {r_correct['meta']['feature_cols']}")
    print(f"   observed_window_rows: {r_correct['meta'].get('observed_window_rows')}")
    print(f"   observed_window_dates: {r_correct['meta'].get('observed_window_unique_dates')}")
    print(f"   {'Rank':<6}{'Source':<30}{'Distance':<16}{'Weight':<16}")
    print(f"   {'-' * 68}")
    for s in r_correct["sources"]:
        key = s["source_key"]
        store = key[0] if isinstance(key, (list, tuple)) else "?"
        item = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else "?"
        print(f"   {s['source_rank']:<6}Store {str(store):>3} Item {str(item):>3}   "
              f"{s['distance']:<16.4f}{s['weight']:<16.6f}")

    # ── Step 7: 论文 Table 5 对照 ────────────────────────────
    print(f"\n{'=' * 90}")
    print("  📖 对照论文 Table 5")
    print(f"{'=' * 90}")
    paper_expected = [(1, 7), (1, 8), (1, 2)]  # Store 1 Item 7, Store 1 Item 8, Store 1 Item 2

    wrong_top3 = []
    for s in r_wrong["sources"][:3]:
        key = s["source_key"]
        wrong_top3.append((key[0], key[1]))
    correct_top3 = []
    for s in r_correct["sources"][:3]:
        key = s["source_key"]
        correct_top3.append((key[0], key[1]))

    print(f"   论文期望:       {paper_expected}")
    print(f"   错误源池 Top-3:  {wrong_top3}  ← smoke test 结果")
    print(f"   正确源池 Top-3:  {correct_top3}")
    print(f"   匹配: {'✅' if correct_top3 == paper_expected else '❌'}")

    # ── Step 8: 根因总结 ────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  📋 根因分析")
    print(f"{'=' * 90}")
    print(f"""
  1. build_source_target_split 在 strict paper mode 下:
     - 正确设置 target = Store 1 Item 10 ✅
     - 正确设置 source_items = [1..9] ✅
     - 但 source pool 包含 Store 1/2/3 的所有 items 1-9 ❌
       (narrowed 过滤到 entities [1,2,3]，但 source_df 未进一步按 entity_id 过滤)

  2. without_information_sharing 的 same-store 过滤:
     - 由 run_full_paper_experiments.py 的 _apply_source_pool_filter 实现
     - 该函数在 build_source_target_split 之后执行
     - smoke test 直接调用 build_source_target_split，跳过了此过滤

  3. 因此 smoke test 的源池包含了 Store 2 和 Store 3 的 items，
     这些 items 的销售模式与 Store 1 Target 不同，
     导致 KNN 选出了跨 store 的 items (如 Store 1 Item 6, Store 2 Item 9 等)。

  4. KNN feature mode 变更对此无影响:
     - Dataset1 无 store_id/region_id 字段
     - 旧的 _SOURCE_SELECTION_EXCLUDE_EXACT 排除 entity_id, item_id, date
     - 新的 _SOURCE_SELECTION_EXCLUDE_EXACT 增加了 store_id/region_id/entity_id_code/brand_code
     - 但 Dataset1 不含这些列，故实际排除集完全相同

  5. KNN features (5): sales, year, month, week, day — 修正前后一致。
""")


if __name__ == "__main__":
    main()
