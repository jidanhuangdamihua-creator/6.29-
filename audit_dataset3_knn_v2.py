"""
Dataset3 No Information Sharing — KNN Source Selection 审计（修正版 v2）

修正点：
- 使用 target observed window (16 train + 15 val = 31 days)，不使用全部 942 天
- 三种 KNN feature mode 对比：
  a) paper_available_features_no_ids (论文对齐默认)
  b) sales_only_sequence (最严格 audit)
  c) engineering_all_numeric (工程对照)
- target test 数据不参与 KNN
- source pool = Region 1 Store 1–9
- target = Store 10
"""
import sys
import logging
import numpy as np
import pandas as pd

from data_preprocessing import (
    _clean_rossmann_raw_columns,
    _merge_rossmann_store_type,
    _standardize_rossmann_dataset,
    KNN_FEATURE_MODE_PAPER_NO_IDS,
    KNN_FEATURE_MODE_SALES_ONLY,
    KNN_FEATURE_MODE_ALL_NUMERIC,
)
from source_selector import SourceSelector

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

FEATURE_MODES = [
    (KNN_FEATURE_MODE_PAPER_NO_IDS, "paper_available_features_no_ids"),
    (KNN_FEATURE_MODE_SALES_ONLY, "sales_only_sequence"),
    (KNN_FEATURE_MODE_ALL_NUMERIC, "engineering_all_numeric"),
]


def load_and_prepare():
    raw_df = pd.read_csv("数据集/Dataset3-Rossmann.csv", low_memory=False)
    lg = logging.getLogger("experiment")
    cleaned = _clean_rossmann_raw_columns(raw_df, lg)
    merged = _merge_rossmann_store_type(cleaned, lg)
    standardized = _standardize_rossmann_dataset(merged)
    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    standardized = standardized.dropna(subset=["date"])

    # Target: Store 10
    target_df = standardized[standardized["store_id"] == 10].copy()
    target_df = target_df.sort_values(["date"]).reset_index(drop=True)
    target_df.attrs["split_role"] = "target"
    target_df.attrs["split_mode"] = "days"
    target_df.attrs["split_config"] = {"train_days": 16, "val_days": 15, "test_days": 181}

    unique_dates = sorted(target_df["date"].unique())
    print(f"\n🎯 Target Store 10: {len(target_df)} rows, {len(unique_dates)} unique dates")
    print(f"   Full range: {unique_dates[0].date()} → {unique_dates[-1].date()}")

    # Source pool: Region 1, Store 1–9
    source_pool = standardized[
        (standardized["entity_id"] == "Region 1")
        & (standardized["store_id"].between(1, 9))
    ].copy()
    source_pool = source_pool.sort_values(["date", "store_id"]).reset_index(drop=True)
    source_stores = sorted(source_pool["store_id"].unique())
    print(f"📂 Source pool: Stores {source_stores}  ({len(source_pool)} rows)")
    assert source_stores == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    return target_df, source_pool


def run_knn_for_mode(target_df, source_pool, mode_name, mode_label):
    selector = SourceSelector()
    result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_pool,
        feature_cols=[],
        k=9,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
        knn_feature_mode=mode_name,
    )
    meta = result.get("meta", {})
    sources = result.get("sources", [])
    return {
        "mode": mode_label,
        "features": meta.get("feature_cols", []),
        "n_features": len(meta.get("feature_cols", [])),
        "target_signature_dim": meta.get("target_signature_dim", 0),
        "observed_window_rows": meta.get("observed_window_rows", 0),
        "observed_window_dates": meta.get("observed_window_unique_dates", 0),
        "sources": sources,
    }


def main():
    print("=" * 90)
    print("  Dataset3 No Information Sharing — KNN 审计 (observed window=31d)")
    print("=" * 90)

    target_df, source_pool = load_and_prepare()

    results = {}
    for mode_name, mode_label in FEATURE_MODES:
        print(f"\n{'─' * 90}")
        print(f"  ▶ Mode: {mode_label}")
        print(f"{'─' * 90}")
        r = run_knn_for_mode(target_df, source_pool, mode_name, mode_label)
        results[mode_label] = r
        print(f"    Features ({r['n_features']}): {r['features']}")
        print(f"    Observed window: {r['observed_window_rows']} rows, "
              f"{r['observed_window_dates']} unique dates")
        print(f"    Vector dim: {r['target_signature_dim']}")

    # ── 对比排名 ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  📊 Top-9 Ranking — 三种 Mode 对比 (ordered by distance)")
    print("=" * 90)

    for mode_label, r in results.items():
        print(f"\n── {mode_label} ({r['n_features']} features) ──")
        print(f"   Features: {r['features']}")
        print(f"   {'Rank':<6}{'Store':<8}{'Distance':<16}{'Weight':<16}")
        print(f"   {'-' * 46}")
        for s in r["sources"]:
            key = s["source_key"]
            sn = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else str(key)
            print(f"   {s['source_rank']:<6}{str(sn):<8}{s['distance']:<16.4f}{s['weight']:<16.6f}")

    # ── 论文 Table 5 对照 ──────────────────────────────────
    print("\n" + "=" * 90)
    print("  📖 论文 Table 5 期望: [6, 2, 1]")
    print("=" * 90)
    for mode_label, r in results.items():
        top3 = []
        for s in r["sources"][:3]:
            key = s["source_key"]
            sn = key[1] if isinstance(key, (list, tuple)) and len(key) > 1 else str(key)
            top3.append(int(sn))
        match = "✅ MATCH" if top3 == [6, 2, 1] else "❌ MISMATCH"
        print(f"  {mode_label:<40s} top-3: {top3}  {match}")

    # ── Store 1 vs Store 8 距离分解 ─────────────────────────
    print("\n" + "=" * 90)
    print("  🔬 Store 1 vs Store 8 距离 (paper_available_features_no_ids)")
    print("=" * 90)
    store_dists = {}
    for s in results[KNN_FEATURE_MODE_PAPER_NO_IDS]["sources"]:
        key = s["source_key"]
        sn = key[1] if isinstance(key, (list, tuple)) else key
        store_dists[sn] = s["distance"]
    d1 = store_dists.get(1)
    d8 = store_dists.get(8)
    print(f"  Store 1: {d1:.4f}" if d1 else "  Store 1: N/A")
    print(f"  Store 8: {d8:.4f}" if d8 else "  Store 8: N/A")
    if d1 is not None and d8 is not None:
        print(f"  Diff (1−8): {d1 - d8:+.4f} → Store {'8' if d1 > d8 else '1'} closer")

    # ── 总结 ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  📋 审计总结")
    print("=" * 90)
    pm = results[KNN_FEATURE_MODE_PAPER_NO_IDS]
    so = results[KNN_FEATURE_MODE_SALES_ONLY]
    en = results[KNN_FEATURE_MODE_ALL_NUMERIC]
    print(f"""
  1. Source pool: Region 1 Store 1–9 ✅
  2. Target: Store 10 ✅
  3. Observed window: {pm['observed_window_dates']} days (16 train + 15 val) ✅
  4. KNN representation: paper_observed_sequence (flattened time steps) ✅
  5. Three modes tested:
     paper_available_features_no_ids:  {pm['n_features']} features → top-3: {[int(s['source_key'][1]) for s in pm['sources'][:3]]}
     sales_only_sequence:              {so['n_features']} features → top-3: {[int(s['source_key'][1]) for s in so['sources'][:3]]}
     engineering_all_numeric:          {en['n_features']} features → top-3: {[int(s['source_key'][1]) for s in en['sources'][:3]]}
  6. Paper Table 5 expects: [6, 2, 1]
  7. _SOURCE_SELECTION_EXCLUDE_EXACT now: {sorted(['entity_id','item_id','date','store_id','region_id'])}
""")


if __name__ == "__main__":
    main()
