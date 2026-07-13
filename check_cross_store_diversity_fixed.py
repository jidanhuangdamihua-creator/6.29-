#!/usr/bin/env python3
"""
FIXED: Check cross-store diversity for WITH scenario
Correctly uses (store_id, product_id) as entity key
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 120)
print("D4 CROSS-STORE DIVERSITY ANALYSIS (FIXED - Using store_id + product_id as entity)")
print("=" * 120)

# Load data
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

target_store = 166
target_first_cat = 15
target_second_cat = 20

# Get target entities (as composite keys)
target_entities = set(
    target_df[['store_id', 'product_id']].drop_duplicates().itertuples(index=False, name=None)
)

print(f"\nTarget store: {target_store}")
print(f"Target first_category: {target_first_cat}")
print(f"Target second_category: {target_second_cat}")
print(f"Target entities (store, product) pairs: {len(target_entities)}")

# D4 observation window
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

print(f"Observation Window: {obs_start.date()} to {obs_end.date()} ({len(required_dates)} days)")

source_df['date'] = pd.to_datetime(source_df['date'])

def check_30day_completeness(entity_df, required_dates):
    """Check if an entity (store, product) has complete 30-day observation"""
    entity_dates = pd.to_datetime(entity_df['date'].unique()).normalize()
    required_set = set(required_dates.normalize())
    entity_set = set(entity_dates)
    missing = required_set - entity_set
    return len(missing) == 0

def analyze_cross_store_diversity(category_level, category_value, scenario_name):
    """Analyze store distribution of cross-store candidates (FIXED)"""
    print(f"\n{'='*120}")
    print(f"{scenario_name}: {category_level}={category_value}")
    print(f"{'='*120}")
    
    # Get cross-store candidates
    if category_level == 'second_category_id':
        candidates_df = source_df[
            (source_df['store_id'] != target_store) &
            (source_df['second_category_id'] == category_value)
        ]
    else:  # first_category_id
        candidates_df = source_df[
            (source_df['store_id'] != target_store) &
            (source_df['first_category_id'] == category_value)
        ]
    
    # Get unique entities (store, product) pairs
    candidate_entities = candidates_df[['store_id', 'product_id']].drop_duplicates()
    total_entities = len(candidate_entities)
    
    print(f"\nTotal candidate entities (store, product) pairs: {total_entities}")
    
    # Exclude target entities
    candidate_tuples = set(candidate_entities.itertuples(index=False, name=None))
    candidate_tuples_filtered = candidate_tuples - target_entities
    
    print(f"After excluding target entities: {len(candidate_tuples_filtered)}")
    
    # Check 30-day completeness for each entity
    valid_entities = []
    
    for store_id, product_id in candidate_tuples_filtered:
        entity_df = candidates_df[
            (candidates_df['store_id'] == store_id) &
            (candidates_df['product_id'] == product_id)
        ]
        if check_30day_completeness(entity_df, required_dates):
            valid_entities.append((store_id, product_id))
    
    valid_count = len(valid_entities)
    print(f"Valid candidate entities (30-day complete): {valid_count}")
    
    if valid_count == 0:
        print("⚠️  No valid candidates - cannot analyze diversity")
        return None
    
    # Convert to DataFrame for analysis
    valid_df = pd.DataFrame(valid_entities, columns=['store_id', 'product_id'])
    
    # Products per store (CORRECTED)
    entities_per_store = valid_df.groupby('store_id').size().sort_values(ascending=False)
    unique_stores = len(entities_per_store)
    
    print(f"\n--- STORE DISTRIBUTION (CORRECTED) ---")
    print(f"Unique stores contributing: {unique_stores}")
    print(f"\nTop stores by entity (store, product) count:")
    for i, (store_id, count) in enumerate(entities_per_store.head(10).items(), 1):
        pct = count / valid_count * 100
        products = valid_df[valid_df['store_id'] == store_id]['product_id'].unique()
        print(f"  {i:2d}. store_id={store_id:3d}: {count:3d} entities ({pct:5.1f}%) - {len(products)} unique products")
    
    # Sanity check
    total_check = entities_per_store.sum()
    print(f"\nSanity check: sum of entities per store = {total_check} (should equal {valid_count})")
    assert total_check == valid_count, "FATAL: Entity count mismatch!"
    
    # Concentration metrics (CORRECTED)
    print(f"\n--- CONCENTRATION ANALYSIS ---")
    
    top1_pct = entities_per_store.iloc[0] / valid_count * 100 if len(entities_per_store) > 0 else 0
    top3_pct = entities_per_store.head(3).sum() / valid_count * 100 if len(entities_per_store) >= 3 else 100
    top5_pct = entities_per_store.head(5).sum() / valid_count * 100 if len(entities_per_store) >= 5 else 100
    
    print(f"Top-1 store concentration: {top1_pct:.1f}%")
    print(f"Top-3 stores concentration: {top3_pct:.1f}%")
    print(f"Top-5 stores concentration: {top5_pct:.1f}%")
    
    # Sanity checks
    assert top1_pct <= 100, f"FATAL: Top-1 concentration {top1_pct}% > 100%"
    assert top3_pct <= 100, f"FATAL: Top-3 concentration {top3_pct}% > 100%"
    assert top5_pct <= 100, f"FATAL: Top-5 concentration {top5_pct}% > 100%"
    
    # Gini coefficient
    counts = entities_per_store.values
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts_sorted)) / (n * np.sum(counts_sorted)) - (n + 1) / n
    
    print(f"\nGini coefficient: {gini:.3f} (0=equality, 1=inequality)")
    
    # Shannon entropy
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n) if n > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"Shannon entropy: {entropy:.2f} (normalized: {normalized_entropy:.3f}, 0=concentrated, 1=diverse)")
    
    # Diversity assessment
    print(f"\n--- DIVERSITY ASSESSMENT ---")
    
    if top1_pct > 50:
        print(f"⚠️  HIGH CONCENTRATION: Top store dominates ({top1_pct:.1f}%)")
        print(f"   'Cross-store' may be misleading - heavily dependent on store {entities_per_store.index[0]}")
    elif top3_pct > 80:
        print(f"⚠️  MODERATE CONCENTRATION: Top-3 stores account for {top3_pct:.1f}%")
        print(f"   Cross-store diversity is limited")
    else:
        print(f"✅ GOOD DIVERSITY: Top-3 stores = {top3_pct:.1f}%, spread across {unique_stores} stores")
    
    # Product diversity analysis
    print(f"\n--- PRODUCT-LEVEL ANALYSIS ---")
    unique_products = valid_df['product_id'].nunique()
    print(f"Unique product_id values across all stores: {unique_products}")
    print(f"Average stores per product: {valid_count / unique_products:.2f}")
    
    # Check if products are replicated across stores
    products_multi_store = valid_df.groupby('product_id')['store_id'].nunique()
    replicated = products_multi_store[products_multi_store > 1]
    
    if len(replicated) > 0:
        print(f"\nProducts appearing in multiple stores: {len(replicated)}/{unique_products}")
        print(f"Examples (product_id: num_stores):")
        for pid, count in replicated.head(5).items():
            print(f"  product_id={pid}: {count} stores")
    else:
        print(f"\n✓ Each product appears in only one store (no cross-store replication)")
    
    return {
        'total_entities': total_entities,
        'valid_entities': valid_count,
        'unique_stores': unique_stores,
        'unique_products': unique_products,
        'entities_per_store': entities_per_store,
        'top1_pct': top1_pct,
        'top3_pct': top3_pct,
        'top5_pct': top5_pct,
        'gini': gini,
        'entropy_normalized': normalized_entropy,
        'valid_entity_list': valid_entities,
    }

# ============================================================================
# Analyze both scenarios
# ============================================================================

result_second = analyze_cross_store_diversity('second_category_id', target_second_cat, 
                                               "WITH + second_category")

result_first = analyze_cross_store_diversity('first_category_id', target_first_cat,
                                              "WITH + first_category")

# ============================================================================
# COMPARISON
# ============================================================================

if result_second and result_first:
    print("\n" + "=" * 120)
    print("COMPARISON: second_category vs first_category")
    print("=" * 120)
    
    print(f"\nValid candidate entities:")
    print(f"  second_category: {result_second['valid_entities']}")
    print(f"  first_category:  {result_first['valid_entities']}")
    print(f"  Increase: +{result_first['valid_entities'] - result_second['valid_entities']}")
    
    print(f"\nStore diversity:")
    print(f"  second_category: {result_second['unique_stores']} stores")
    print(f"  first_category:  {result_first['unique_stores']} stores")
    
    print(f"\nUnique products:")
    print(f"  second_category: {result_second['unique_products']} products")
    print(f"  first_category:  {result_first['unique_products']} products")
    
    print(f"\nTop-3 concentration:")
    print(f"  second_category: {result_second['top3_pct']:.1f}%")
    print(f"  first_category:  {result_first['top3_pct']:.1f}%")
    
    # New entities analysis
    second_entities = set(result_second['valid_entity_list'])
    first_entities = set(result_first['valid_entity_list'])
    new_entities = first_entities - second_entities
    
    print(f"\n--- NEW ENTITIES (first_category only) ---")
    print(f"New entities: {len(new_entities)}")
    print(f"Percentage of first_category total: {len(new_entities) / result_first['valid_entities'] * 100:.1f}%")
    
    # Sanity check
    assert len(new_entities) / result_first['valid_entities'] <= 1.0, "FATAL: New entities percentage > 100%"
    
    # Analyze new entities by store
    new_df = pd.DataFrame(list(new_entities), columns=['store_id', 'product_id'])
    new_stores = new_df['store_id'].nunique()
    new_products = new_df['product_id'].nunique()
    
    print(f"New entities distributed across: {new_stores} stores")
    print(f"Involving: {new_products} unique products")

print("\n" + "=" * 120)
print("ANALYSIS COMPLETE (VERIFIED)")
print("=" * 120)
