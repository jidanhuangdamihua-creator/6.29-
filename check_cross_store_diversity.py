#!/usr/bin/env python3
"""
Check cross-store diversity for WITH scenario
Analyze store distribution to validate "cross-store" diversity claim
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 120)
print("D4 CROSS-STORE DIVERSITY ANALYSIS (WITH Scenario)")
print("=" * 120)

# Load data
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

target_store = 166
target_first_cat = 15
target_second_cat = 20

# D4 observation window
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

source_df['date'] = pd.to_datetime(source_df['date'])

def check_30day_completeness(product_df, required_dates):
    """Check if a product has complete 30-day observation"""
    product_dates = pd.to_datetime(product_df['date'].unique()).normalize()
    required_set = set(required_dates.normalize())
    product_set = set(product_dates)
    missing = required_set - product_set
    return len(missing) == 0

def analyze_cross_store_diversity(category_level, category_value, scenario_name):
    """Analyze store distribution of cross-store candidates"""
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
    
    # Total unique products (before 30-day check)
    total_products = candidates_df['product_id'].nunique()
    print(f"\nTotal candidate products (all stores): {total_products}")
    
    # Check 30-day completeness for each product
    valid_products = []
    valid_stores = []
    
    for product_id in candidates_df['product_id'].unique():
        product_df = candidates_df[candidates_df['product_id'] == product_id]
        if check_30day_completeness(product_df, required_dates):
            valid_products.append(product_id)
            # Get store(s) for this product
            stores = product_df['store_id'].unique()
            valid_stores.extend(stores)
    
    valid_count = len(valid_products)
    print(f"Valid candidate products (30-day complete): {valid_count}")
    
    if valid_count == 0:
        print("⚠️  No valid candidates - cannot analyze diversity")
        return None
    
    # Analyze store distribution (for valid candidates only)
    valid_candidates_df = candidates_df[candidates_df['product_id'].isin(valid_products)]
    
    # Products per store
    products_per_store = valid_candidates_df.groupby('store_id')['product_id'].nunique().sort_values(ascending=False)
    unique_stores = len(products_per_store)
    
    print(f"\n--- STORE DISTRIBUTION ---")
    print(f"Unique stores contributing: {unique_stores}")
    print(f"\nTop stores by product count:")
    for i, (store_id, count) in enumerate(products_per_store.head(10).items(), 1):
        pct = count / valid_count * 100
        print(f"  {i:2d}. store_id={store_id:3d}: {count:3d} products ({pct:5.1f}%)")
    
    # Concentration metrics
    print(f"\n--- CONCENTRATION ANALYSIS ---")
    
    # Top-N concentration
    top1_pct = products_per_store.iloc[0] / valid_count * 100 if len(products_per_store) > 0 else 0
    top3_pct = products_per_store.head(3).sum() / valid_count * 100 if len(products_per_store) >= 3 else 100
    top5_pct = products_per_store.head(5).sum() / valid_count * 100 if len(products_per_store) >= 5 else 100
    
    print(f"Top-1 store concentration: {top1_pct:.1f}%")
    print(f"Top-3 stores concentration: {top3_pct:.1f}%")
    print(f"Top-5 stores concentration: {top5_pct:.1f}%")
    
    # Gini coefficient (inequality measure)
    counts = products_per_store.values
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts_sorted)) / (n * np.sum(counts_sorted)) - (n + 1) / n
    
    print(f"\nGini coefficient: {gini:.3f}")
    print(f"  (0 = perfect equality, 1 = maximum inequality)")
    
    # Shannon entropy (diversity measure)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n) if n > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"\nShannon entropy: {entropy:.2f} (normalized: {normalized_entropy:.3f})")
    print(f"  (0 = concentrated, 1 = perfectly diverse)")
    
    # Diversity assessment
    print(f"\n--- DIVERSITY ASSESSMENT ---")
    
    if top1_pct > 50:
        print(f"⚠️  HIGH CONCENTRATION: Top store dominates ({top1_pct:.1f}%)")
        print(f"   'Cross-store' may be misleading - heavily dependent on store {products_per_store.index[0]}")
    elif top3_pct > 80:
        print(f"⚠️  MODERATE CONCENTRATION: Top-3 stores account for {top3_pct:.1f}%")
        print(f"   Cross-store diversity is limited")
    else:
        print(f"✅ GOOD DIVERSITY: Top-3 stores = {top3_pct:.1f}%, spread across {unique_stores} stores")
    
    return {
        'total_products': total_products,
        'valid_products': valid_count,
        'unique_stores': unique_stores,
        'products_per_store': products_per_store,
        'top1_pct': top1_pct,
        'top3_pct': top3_pct,
        'top5_pct': top5_pct,
        'gini': gini,
        'entropy_normalized': normalized_entropy,
    }

# ============================================================================
# Analyze both scenarios
# ============================================================================

print("\n" + "=" * 120)
print("SCENARIO 1: WITH + second_category=20")
print("=" * 120)
result_second = analyze_cross_store_diversity('second_category_id', target_second_cat, 
                                                "WITH + second_category")

print("\n" + "=" * 120)
print("SCENARIO 2: WITH + first_category=15")
print("=" * 120)
result_first = analyze_cross_store_diversity('first_category_id', target_first_cat,
                                              "WITH + first_category")

# ============================================================================
# COMPARISON
# ============================================================================

if result_second and result_first:
    print("\n" + "=" * 120)
    print("COMPARISON: second_category vs first_category")
    print("=" * 120)
    
    print(f"\nValid candidate products:")
    print(f"  second_category: {result_second['valid_products']}")
    print(f"  first_category:  {result_first['valid_products']}")
    print(f"  Increase: +{result_first['valid_products'] - result_second['valid_products']}")
    
    print(f"\nStore diversity:")
    print(f"  second_category: {result_second['unique_stores']} stores")
    print(f"  first_category:  {result_first['unique_stores']} stores")
    
    print(f"\nTop-3 concentration:")
    print(f"  second_category: {result_second['top3_pct']:.1f}%")
    print(f"  first_category:  {result_first['top3_pct']:.1f}%")
    print(f"  Change: {result_first['top3_pct'] - result_second['top3_pct']:+.1f}%")
    
    print(f"\nDiversity metrics:")
    print(f"  Gini (lower=better):    second={result_second['gini']:.3f}, first={result_first['gini']:.3f}")
    print(f"  Entropy (higher=better): second={result_second['entropy_normalized']:.3f}, first={result_first['entropy_normalized']:.3f}")
    
    # New candidates analysis
    print("\n" + "=" * 120)
    print("NEW CANDIDATES ANALYSIS (first_category only)")
    print("=" * 120)
    
    # Get products unique to first_category
    second_products = set(result_second['products_per_store'].index) if result_second else set()
    first_products = set(result_first['products_per_store'].index) if result_first else set()
    
    new_stores = first_products - second_products
    
    if new_stores:
        print(f"\nNew stores contributing (not in second_category): {len(new_stores)}")
        
        # Analyze new stores
        new_store_products = result_first['products_per_store'][list(new_stores)]
        new_product_count = new_store_products.sum()
        
        print(f"Products from new stores: {new_product_count}")
        print(f"Percentage of total first_category candidates: {new_product_count / result_first['valid_products'] * 100:.1f}%")
        
        print(f"\nTop new stores:")
        for i, (store_id, count) in enumerate(new_store_products.sort_values(ascending=False).head(5).items(), 1):
            print(f"  {i}. store_id={store_id}: {count} products")
    else:
        print("\n⚠️  NO NEW STORES: All first_category candidates come from same stores as second_category")
        print("   → Store diversity does not increase, only product count per store increases")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 120)
print("RECOMMENDATIONS FOR PAPER")
print("=" * 120)

if result_first:
    if result_first['top1_pct'] > 50:
        print("\n⚠️  HIGH CONCENTRATION WARNING:")
        print(f"   Top store contributes {result_first['top1_pct']:.1f}% of cross-store candidates")
        print("\n   PAPER MITIGATION:")
        print("   1. Acknowledge in methodology: 'Cross-store candidates are predominantly from Store X'")
        print("   2. Report store distribution statistics")
        print("   3. Discuss potential impact on generalizability")
        print("\n   ALTERNATIVE:")
        print("   - Consider filtering/balancing sources by store to increase diversity")
        
    elif result_first['top3_pct'] > 75:
        print("\n⚠️  MODERATE CONCENTRATION:")
        print(f"   Top-3 stores contribute {result_first['top3_pct']:.1f}% of candidates")
        print("\n   PAPER MENTION:")
        print("   - Report store distribution in results/appendix")
        print("   - Note as limitation if diversity is below expectations")
    
    else:
        print("\n✅ ACCEPTABLE DIVERSITY:")
        print(f"   Candidates spread across {result_first['unique_stores']} stores")
        print(f"   Top-3 concentration: {result_first['top3_pct']:.1f}%")
        print("\n   PAPER STRENGTH:")
        print("   - Can confidently claim 'diverse cross-store sources'")
        print("   - Report metrics to demonstrate multi-store representation")

print("\n" + "=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)
