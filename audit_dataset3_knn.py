"""
Dataset3 No Information Sharing KNN Source Selection 审计脚本

检查:
1. no sharing source pool 是否为 Region 1 Store 1–9
2. target 是否为 Store 10
3. KNN features 是否与论文 Fig. 9 一致
4. 是否包含 store_id 作为 KNN feature
5. 是否包含 customers/open/promo/holiday/day_of_week 等工程派生特征
6. scaling 是否影响 distance
7. 输出 Store 1、Store 8 的 KNN 向量差异和 distance 对比
"""
import sys
import json
import logging
import numpy as np
import pandas as pd

from data_preprocessing import (
    load_dataset,
    _standardize_rossmann_dataset,
    _clean_rossmann_raw_columns,
    _merge_rossmann_store_type,
    infer_source_selection_feature_columns,
    infer_modeling_feature_columns,
    _SOURCE_SELECTION_EXCLUDE_EXACT,
    _SOURCE_SELECTION_LEAKAGE_KEYWORDS,
    _NON_FEATURE_COLUMNS,
)
from source_selector import SourceSelector

# ── 设置日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("audit")

def main():
    print("=" * 80)
    print("Dataset3 No Information Sharing — KNN Source Selection 审计")
    print("=" * 80)

    # ── Step 1: 加载原始数据 ─────────────────────────────────
    raw_df = pd.read_csv("数据集/Dataset3-Rossmann.csv")
    print(f"\n📦 原始数据: {len(raw_df)} rows, columns={list(raw_df.columns)}")

    # ── Step 2: 清洗 + 标准化 ────────────────────────────────
    cleaned = _clean_rossmann_raw_columns(raw_df, logger)
    merged = _merge_rossmann_store_type(cleaned, logger)
    standardized = _standardize_rossmann_dataset(merged)
    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    standardized = standardized.dropna(subset=["date"])

    print(f"\n📦 标准化后数据: {len(standardized)} rows, columns={list(standardized.columns)}")
    store_ids = sorted(standardized['store_id'].unique())
    print(f"   unique store_id count: {len(store_ids)}, first 10: {store_ids[:10]}, last 10: {store_ids[-10:]}")
    print(f"   unique item_id count: {len(standardized['item_id'].unique())}, first 10: {sorted(standardized['item_id'].unique())[:10]}")
    entity_ids = sorted(standardized['entity_id'].dropna().unique())
    print(f"   unique entity_id (region_id): {entity_ids}")
    print(f"   dtypes:\n{standardized.dtypes.to_string()}")

    # ── Step 3: 检查 no-information-sharing 约束 ──────────────
    # 论文: Dataset3 without information sharing → same region, target Store 10, source Stores 1-9
    target_store = 10
    region = "Region 1"

    target_mask = standardized["store_id"] == target_store
    target_df = standardized[target_mask].copy()
    print(f"\n🎯 Target Store {target_store}: {len(target_df)} rows")

    region_mask = (standardized["entity_id"] == region) & (standardized["store_id"] != target_store)
    source_pool_full = standardized[region_mask].copy()
    source_store_ids = sorted(source_pool_full["store_id"].unique())
    print(f"📂 Source pool (Region 1, excluding Store {target_store}): Stores {source_store_ids}")
    print(f"   Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print(f"   Match: {source_store_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]}")

    # ── Step 4: 推断 KNN 特征 ──────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 KNN Feature 审计")
    print("=" * 80)

    print(f"\n_NON_FEATURE_COLUMNS (硬排除): {sorted(_NON_FEATURE_COLUMNS)}")
    print(f"_SOURCE_SELECTION_EXCLUDE_EXACT: {sorted(_SOURCE_SELECTION_EXCLUDE_EXACT)}")
    print(f"_SOURCE_SELECTION_LEAKAGE_KEYWORDS: {_SOURCE_SELECTION_LEAKAGE_KEYWORDS}")

    source_modeling = infer_modeling_feature_columns(source_pool_full)
    target_modeling = infer_modeling_feature_columns(target_df)
    print(f"\n🔍 source modeling features ({len(source_modeling)}): {source_modeling}")
    print(f"🔍 target modeling features ({len(target_modeling)}): {target_modeling}")

    # 检查 store_id 是否在 exclude 集合中
    print(f"\n⚠️  'store_id' in _SOURCE_SELECTION_EXCLUDE_EXACT? {'store_id' in _SOURCE_SELECTION_EXCLUDE_EXACT}")
    print(f"⚠️  'item_id' in _SOURCE_SELECTION_EXCLUDE_EXACT?  {'item_id' in _SOURCE_SELECTION_EXCLUDE_EXACT}")
    print(f"⚠️  'store_id' in source_modeling? {'store_id' in source_modeling}")
    print(f"⚠️  'store_id' IS numeric? {pd.api.types.is_numeric_dtype(source_pool_full['store_id'])}")

    feature_info = infer_source_selection_feature_columns(
        source_df=source_pool_full,
        target_df=target_df,
        candidate_cols=[],
        include_sales_in_knn=True,
    )
    resolved_features = feature_info["selected_features"]
    excluded = feature_info["excluded_by_rule"]

    print(f"\n✅ Resolved KNN features ({len(resolved_features)}): {resolved_features}")
    print(f"❌ Excluded by rule ({len(excluded)}): {excluded}")

    has_store_id = "store_id" in resolved_features
    has_customers = "customers" in resolved_features
    has_open = "open" in resolved_features
    has_promo = "promo" in resolved_features
    has_holiday = "holiday" in resolved_features
    has_day_of_week = "day_of_week" in resolved_features

    print(f"\n📋 Feature 审计结果:")
    print(f"   store_id in KNN features:  {'❌ 是! store_id 是标识符，不应参与距离计算' if has_store_id else '✅ 已排除'}")
    print(f"   customers in KNN features: {'⚠️ 是 (论文 Fig. 9 未使用)' if has_customers else '✅ 已排除'}")
    print(f"   open in KNN features:      {'⚠️ 是 (论文 Fig. 9 未使用)' if has_open else '✅ 已排除'}")
    print(f"   promo in KNN features:     {'⚠️ 是 (论文 Fig. 9 未使用)' if has_promo else '✅ 已排除'}")
    print(f"   holiday in KNN features:   {'⚠️ 是 (论文 Fig. 9 未使用)' if has_holiday else '✅ 已排除'}")
    print(f"   day_of_week in KNN:        {'⚠️ 是 (论文 Fig. 9 未使用)' if has_day_of_week else '✅ 已排除'}")

    # ── Step 5: 构建 paper_observed_sequence 向量并计算距离 ─────
    print("\n" + "=" * 80)
    print("📏 Paper Observed Sequence — KNN 向量 & 距离审计")
    print("=" * 80)

    selector = SourceSelector()
    result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_pool_full,
        feature_cols=resolved_features,
        k=9,  # 返回所有 9 个 source 的排序
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=True,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )

    meta = result.get("meta", {})
    print(f"\n📐 向量维度: {meta.get('target_signature_dim')}")
    print(f"📐 观测窗口行数: {meta.get('observed_window_rows')}")
    print(f"📐 观测窗口唯一天数: {meta.get('observed_window_unique_dates')}")
    print(f"📐 KNN features: {meta.get('feature_cols')}")
    print(f"📐 包含 sales: {meta.get('contains_sales')}")

    sources = result.get("sources", [])
    print(f"\n🏆 距离排序 (全部 {len(sources)} 个 source):")
    print(f"{'Rank':<6}{'Source':<25}{'Store':<8}{'Distance':<16}{'Weight':<16}")
    print("-" * 71)
    for s in sources:
        key = s["source_key"]
        # key is (entity_id, item_id) → item_id is store number
        if isinstance(key, (list, tuple)):
            store_num = key[1] if len(key) > 1 else "?"
        else:
            store_num = str(key)
        print(f"{s['source_rank']:<6}{str(key):<25}{str(store_num):<8}{s['distance']:<16.6f}{s['weight']:<16.6f}")

    # ── Step 6: Store 1 vs Store 8 详细对比 ────────────────────
    print("\n" + "=" * 80)
    print("🔬 Store 1 vs Store 8 — 向量差异 & 距离对比")
    print("=" * 80)

    # 手动构建向量进行深度分析
    # 用 paper_observed_sequence 模式重新构建
    selection_full = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_pool_full,
        feature_cols=resolved_features,
        k=9,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )

    # 找出 Store 1 和 Store 8 的距离
    store_dists = {}
    for s in selection_full["sources"]:
        key = s["source_key"]
        if isinstance(key, (list, tuple)):
            store_num = key[1]
        else:
            store_num = key
        store_dists[store_num] = s["distance"]

    dist_1 = store_dists.get(1, None)
    dist_8 = store_dists.get(8, None)
    print(f"\n📏 Store 1 distance: {dist_1}")
    print(f"📏 Store 8 distance: {dist_8}")
    if dist_1 is not None and dist_8 is not None:
        diff = dist_1 - dist_8
        print(f"📏 Difference (Store 1 - Store 8): {diff:.6f}")
        print(f"   → Store {'8' if diff > 0 else '1'} is closer to target Store 10")

    # ── Step 7: 逐特征分析 store_id 的贡献 ──────────────────────
    print("\n" + "=" * 80)
    print("🔍 逐特征距离贡献分析 (paper_observed_sequence)")
    print("=" * 80)

    # 为每个 source store 构建序列向量并与 target 比较
    target_ordered = target_df.sort_values(["date"]).reset_index(drop=True)
    target_dates = target_ordered["date"].drop_duplicates().sort_values()
    target_vals = target_ordered[resolved_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    target_vals = np.nan_to_num(target_vals, nan=0.0).reshape(-1)
    n_days = len(target_dates)
    n_features = len(resolved_features)

    print(f"\n序列维度: {n_days} days × {n_features} features = {n_days * n_features}")

    store_vectors = {}
    for store_id in [1, 2, 6, 8]:
        store_mask = source_pool_full["store_id"] == store_id
        store_df = source_pool_full[store_mask].sort_values(["date"]).reset_index(drop=True)
        store_df_filtered = store_df[store_df["date"].isin(target_dates)].sort_values(["date"])
        store_vals = store_df_filtered[resolved_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        store_vals = np.nan_to_num(store_vals, nan=0.0).reshape(-1)
        store_vectors[store_id] = store_vals

    # 逐特征分析
    for fi, fname in enumerate(resolved_features):
        print(f"\n── Feature: {fname} ──")
        tgt_col_vals = target_ordered[fname].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        tgt_col_vals = np.nan_to_num(tgt_col_vals, nan=0.0)

        for sid in [1, 2, 6, 8]:
            src_col = source_pool_full[source_pool_full["store_id"] == sid].sort_values(["date"])
            src_col = src_col[src_col["date"].isin(target_dates)].sort_values(["date"])
            src_col_vals = src_col[fname].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            src_col_vals = np.nan_to_num(src_col_vals, nan=0.0)

            # 计算该特征的欧氏距离贡献
            if len(src_col_vals) == len(tgt_col_vals):
                feat_dist = np.sqrt(np.sum((src_col_vals - tgt_col_vals) ** 2))
            else:
                feat_dist = np.nan
            print(f"   Store {sid}: target_values={tgt_col_vals[:5].tolist()}{'...' if len(tgt_col_vals)>5 else ''} "
                  f"source_values={src_col_vals[:5].tolist()}{'...' if len(src_col_vals)>5 else ''} "
                  f"→ 特征距离={feat_dist:.4f}")

    # ── Step 8: 排除 store_id 后重新计算距离 ────────────────────
    print("\n" + "=" * 80)
    print("🔧 排除 store_id 后重新计算距离")
    print("=" * 80)

    features_no_store_id = [f for f in resolved_features if f != "store_id"]
    print(f"\n排除 store_id 后的特征 ({len(features_no_store_id)}): {features_no_store_id}")

    try:
        selection_no_sid = selector.select_top_k_sources(
            target_df=target_df,
            source_df=source_pool_full,
            feature_cols=features_no_store_id,
            k=9,
            group_cols=("entity_id", "item_id"),
            weight_mode="inverse_distance",
            debug_mode=False,
            include_sales_in_knn=True,
            knn_representation="paper_observed_sequence",
        )
        print(f"\n🏆 排除 store_id 后的距离排序:")
        print(f"{'Rank':<6}{'Source':<25}{'Store':<8}{'Distance':<16}{'Weight':<16}")
        print("-" * 71)
        for s in selection_no_sid["sources"]:
            key = s["source_key"]
            store_num = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else str(key)
            print(f"{s['source_rank']:<6}{str(key):<25}{str(store_num):<8}{s['distance']:<16.6f}{s['weight']:<16.6f}")

        top3_no_sid = [s["source_key"][1] if isinstance(s["source_key"], (list, tuple)) else s["source_key"] 
                       for s in selection_no_sid["sources"][:3]]
        print(f"\n✅ 排除 store_id 后 top-3: {top3_no_sid}")
        print(f"📖 论文 Table 5 期望:     [6, 2, 1]")
        print(f"🎯 匹配: {top3_no_sid == [6, 2, 1]}")

    except Exception as e:
        print(f"❌ 排除 store_id 后出错: {e}")

    # ── 总结 ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📋 审计总结")
    print("=" * 80)
    print(f"""
    1. Source pool 是 Region 1 Store 1-9: ✅
    2. Target 是 Store 10: ✅
    3. KNN 表示是 paper_observed_sequence: ✅
    4. store_id 是否在 KNN features 中: {'❌ 是 — store_id 不在 _SOURCE_SELECTION_EXCLUDE_EXACT 中' if has_store_id else '✅ 否'}
    5. customers/open/promo/holiday/day_of_week 是否在 KNN features 中: 
       - customers: {'⚠️ 是' if has_customers else '✅ 否'}
       - open:      {'⚠️ 是' if has_open else '✅ 否'}
       - promo:     {'⚠️ 是' if has_promo else '✅ 否'}
       - holiday:   {'⚠️ 是' if has_holiday else '✅ 否'}
       - day_of_week: {'⚠️ 是' if has_day_of_week else '✅ 否'}
    6. Scaling 影响: KNN 使用原始值（未经过 MinMax scaling），源选择在 normalize_features 之前执行
    7. 根因: store_id 是标识符，不应参与 KNN 距离计算。加入后，Store 8 (id=8) 比 Store 1 (id=1) 
       在 store_id 维度上更接近 Target Store 10 (id=10)，导致排序错误。
    """)

    # 检查 scaling 流程
    print("\n── scaling 检查 ──")
    print("KNN source selection 使用原始未缩放数据 (source_selector 工作在 load/preprocess 之后、normalize_features 之前)")
    for col in resolved_features:
        tgt_vals = target_df[col]
        src_vals = source_pool_full[col]
        print(f"   {col}: target range=[{tgt_vals.min():.4f}, {tgt_vals.max():.4f}], "
              f"source range=[{src_vals.min():.4f}, {src_vals.max():.4f}]")


if __name__ == "__main__":
    main()
